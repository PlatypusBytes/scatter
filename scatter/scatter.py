"""Entry point of the scatter finite element model."""
import os
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
from solvers.base_solver import BaseSolverABC, TimeIntegrationType
from solvers.newmark_solver import NewmarkExplicit

from scatter import export_results, force_external, mesher, random_fields, system_matrix, validator
from scatter.rose_utils import RoseUtils


def scatter(mesh_file: str,
            outfile_folder: str,
            materials: Dict[str, Dict[str, float]],
            boundaries: mesher.BoundaryConditions,
            inp_settings: Dict[str, Any],
            loading: Dict[str, Any],
            time_step: float = 0.1,
            solver: BaseSolverABC = NewmarkExplicit(),
            random_props: Union[bool, Dict[str, Dict[str, Any]]] = False) -> export_results.Write:
    r"""Run a dynamic finite element calculation, with a consistent mass matrix.

    The mesh is generated with gmsh (https://gmsh.info/), and the coordinate system is the one of gmsh:

         ^  _
       y |  /| z
         | /
         -----> x

    Args:
        mesh_file: gmsh mesh file.
        outfile_folder: folder where the results are written.
        materials: material properties ("Young" [Pa], "poisson" [-] and "density" [kg/m^3]) per material name.
        boundaries: boundary conditions, see `mesher.ReadMesh.read_bc`.
        inp_settings: numerical settings.
        loading: loading conditions.
        time_step: time step of the analysis [s].
        solver: numerical solver.
        random_props: random field properties per material name, or False for no random fields.

    Returns:
        Results of the calculation.

    Raises:
        TypeError: if random_props is neither False nor a dictionary.
        ValueError: if the time integration type of the solver is not supported.
    """
    with open(os.path.join(os.path.dirname(__file__), '../docs/static/message.txt'), "r") as f:
        print(f.read())

    validator.ValidateLoad.validate(loading)

    # read gmsh mesh, boundary conditions, equation numbers and connectivities
    model = mesher.ReadMesh(mesh_file)
    model.read_gmsh()
    model.read_bc(boundaries)
    model.mapping()
    model.connectivities()
    if loading["type"] == "rose":
        loading["model"] = RoseUtils.pre_process_rose_model(loading["model"])
        model.rose_connectivities(loading["model"])

    model.get_mesh_edges()

    if random_props:
        if not isinstance(random_props, dict):
            raise TypeError("random_props must be False or the random field properties per material name.")
        print("Generating random field")
        rf = random_fields.RF(random_props, materials, outfile_folder, model.element_type)
        # generate the random fields and update the model and materials in place
        rf.generate_random_fields(model, model.dimension, angles=0.0)
        rf.dump()

    print("Generating global matrices scatter")
    matrix = system_matrix.GenerateMatrix(model.number_eq, inp_settings['int_order'])
    matrix.generate_stiffness_and_mass(model, materials)
    matrix.absorbing_boundaries(model, materials, inp_settings["absorbing_BC"], inp_settings["absorbing_BC_stiff"])

    # couple the scatter and rose stiffness, mass and damping matrices
    if loading["type"] == "rose":
        matrix.add_rose_stiffness(model, loading["model"])
        matrix.add_rose_mass(model, loading["model"])
        matrix.add_rose_damping(model, loading["model"])

    matrix.damping_Rayleigh(inp_settings["damping"])

    time = np.linspace(0, loading["time"], int(np.ceil(loading["time"] / time_step) + 1))

    solver.state.output_interval = inp_settings["output_interval"]
    solver.initialise(model.number_eq, time)

    print("Setting load")
    F = force_external.Force()

    # TODO that moving load at plane, only works at hexa8 elements
    top_surface_elements: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
    if loading["type"] == "moving_at_plane":
        top_surface_elements = model.get_top_surface()

    F.initialise_load(loading, time, model, solver, top_surface_elements=top_surface_elements)
    solver.force.update_rhs_at_time_step_func = F.update_load_at_t

    print("solver started")
    if solver.type is TimeIntegrationType.DYNAMIC:
        solver.calculate(matrix.M, matrix.C, matrix.K, F.force_vector, 0, len(F.time) - 1)
    elif solver.type is TimeIntegrationType.STATIC:
        solver.calculate(matrix.K, F.force_vector, 0, len(F.time) - 1)
    else:
        raise ValueError("Time integration type not supported")

    results = export_results.Write(outfile_folder, model, materials, solver)
    results.pickle(write=inp_settings["pickle"], nodes=inp_settings["pickle_nodes"])
    results.vtk(write=inp_settings["VTK"], binary=inp_settings["VTK_binary"])

    print("\n\n\n\x1B[3m" + "  Never tell me the odds. " + "\x1B[0m")
    print("\x1B[3m" + "--- Han Solo" + "\x1B[0m")
    return results
