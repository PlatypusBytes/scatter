import os
import sys
import numpy as np
from scatter import mesher
from scatter import system_matrix
from scatter import force_external
from scatter import random_fields
from scatter import export_results
from scatter import utils
from scatter import validator
from scatter.rose_utils import RoseUtils
from solvers import newmark_solver, static_solver, central_difference_solver, bathe_solver
from solvers.utils import LumpingMethod

def scatter(mesh_file: str, outfile_folder: str, materials: dict, boundaries: dict,
            inp_settings: dict, loading: dict, time_step: float = 0.1, random_props: bool = False,
            type_analysis="dynamic_implicit", solver=newmark_solver.NewmarkExplicit, gnn=False) -> export_results.Write:
    r"""
    3D finite element code.
                                                            ^  _
                                                          y |  /| z
    Mesh is generated with gmsh https://gmsh.info/          | /
    The coordinate system is the same as defined in gmsh    -----> x

    Consistent Mass matrix

    Parameters
    ----------
    :param mesh_file: mesh file
    :param outfile_folder: location of the output folder
    :param materials: dictionary with material properties
    :param boundaries: dictionary with boundary conditions
    :param inp_settings: dictionary with numerical settings
    :param loading: dictionary with loading conditions
    :param time_step: time step for the analysis (optional: default 0.1 s)
    :param random_props: bool with random fields analysis
    :param type_analysis: 'dynamic' or 'static' (default 'dynamic')
    :param gnn: bool for exporting results for GNN analysis (default False)
    """

    # print message
    print(open(os.path.join(os.path.dirname(__file__), '../docs/static/message.txt'), "r").read())

    # validate loading
    validator.ValidateLoad.validate(loading)

    # read gmsh mesh & create structure
    model = mesher.ReadMesh(mesh_file)
    # read gmsh file: file_name, dimension, nb_nodes_elem, materials, nodes, elem, element type
    model.read_gmsh()
    # define boundary conditions
    model.read_bc(boundaries)
    # mapping of equation numbers and BC
    model.mapping()
    # connectivities
    model.connectivities()
    # add rose connectivities
    if loading["type"] == "rose":
        # pre process rose
        loading["model"] = RoseUtils.pre_process_rose_model(loading["model"])
        # add rose connectivities
        model.rose_connectivities(loading["model"])

    # mesh edges
    model.get_mesh_edges()

    if random_props:
        print("Generating random field")
        rf = random_fields.RF(random_props, materials, outfile_folder, model.element_type)
        # find material index in materials dict of material which should have a random field
        material_idx = [material[1] for material in model.materials if material[2] == random_props["material"]][0]
        # find all elements find should be part of the random field
        elements = model.elem[model.materials_index == material_idx]
        # generate random field
        rf.generate_gstools_rf(model.nodes, elements, model.dimension, angles=0.0)
        rf.dump()
        # add all random field materials to materials dict
        rf.update_material_list(materials, model, material_idx)
        materials.update(rf.new_material)

    # generate matrix internal
    print("Generating global matrices scatter")
    matrix = system_matrix.GenerateMatrix(model.number_eq, inp_settings['int_order'])
    matrix.generate_stiffness_and_mass(model, materials)
    matrix.absorbing_boundaries(model, materials, inp_settings["absorbing_BC"], inp_settings["absorbing_BC_stiff"])

    # add connect scatter and rose stiffness and mass matrices and absorbing boundaries
    if loading["type"] == "rose":
        matrix.add_rose_stiffness(model, loading["model"])
        matrix.add_rose_mass(model, loading["model"])
        matrix.add_rose_damping(model, loading["model"])

    # generate C matrix with Rayleigh damping
    matrix.damping_Rayleigh(inp_settings["damping"])

    # definition of time
    time = np.linspace(0, loading["time"], int(np.ceil(loading["time"] / time_step) + 1))

    # matrix.C *= 0
    # initialise solver
    if type_analysis == "dynamic_explicit":
        if solver == central_difference_solver.CentralDifferenceSolver:
            lumped = False
            method = LumpingMethod.RowSum
            # matrix.C = matrix.M * 0.12161003820347586
            numerical = central_difference_solver.CentralDifferenceSolver(lumped, method)
        elif solver == bathe_solver.BatheSolver:
            lumped = False
            method = LumpingMethod.RowSum
            numerical = bathe_solver.BatheSolver(lumped, method)
        else:
            numerical = newmark_solver.NewmarkExplicit()
    elif type_analysis == "dynamic_implicit":
        numerical = newmark_solver.NewmarkImplicitForce()
    elif type_analysis == "static":
        numerical = static_solver.StaticSolver()
    elif type_analysis == "harmonic_response":
        from scatter.harmonic_response import HarmonicResponse
        numerical = HarmonicResponse()
    else:
        sys.exit(f"Error: {type_analysis} not supported")

    if "output_interval" in inp_settings.keys():
        output_interval = inp_settings["output_interval"]
    else:
        output_interval = 1
    numerical.output_interval = output_interval
    numerical.initialise(model.number_eq, time)

    # generate matrix external
    print("Setting load")
    F = force_external.Force()

    # TODO that moving load at plane, only works at hexa8 elements
    if loading["type"] == "moving_at_plane":
        top_surface_elements = model.get_top_surface()
    else:
        top_surface_elements = []

    F.initialise_load(loading, time, model, numerical, top_surface_elements=top_surface_elements)
    numerical.update_rhs_at_time_step_func = F.update_load_at_t

    print("solver started")
    # start solver
    if type_analysis.startswith("dynamic"):

        force = np.array([F.update_load_at_t(i) for i, t in enumerate(F.time)])
        import pickle
        with open("data.pickle", "wb") as fo:
            pickle.dump([matrix.M, matrix.C, matrix.K, force.T, 0, model.number_eq, F.time], fo)
        sys.exit()
        numerical.update(0)
        numerical.calculate(matrix.M, matrix.C, matrix.K, F.force_vector, 0, len(F.time) - 1)
    elif type_analysis == "static":
        numerical.calculate(matrix.K, F.force_vector, 0, len(F.time) -1)
    elif type_analysis == "harmonic_response":
        omega, results = numerical.calculate(matrix.M, matrix.C, matrix.K, F.force_vector, inp_settings["damping"])
        # get the index
        import matplotlib.pyplot as plt


        # find index:
        node_nb = [int(node) for node in loading["node"]][0]
        idx_hr = model.eq_nb_dof[node_nb - 1][1]
        res = results[:, idx_hr]
        real_part = np.real(res)
        imag_part = np.imag(res)
        # amp = np.abs(real_part + imag_part)
        # plt.plot(omega, amp)
        # plt.grid()
        # plt.show()
        # make the plot
        import json
        with open(os.path.join(outfile_folder, "harmonic_loading.json"), "w") as fo:
            json.dump({"omega": omega.tolist(), "real_part": real_part.tolist(), "imag_part": imag_part.tolist()}, fo, indent=2)

    # export results
    results = export_results.Write(outfile_folder, model, materials, numerical)
    # export results to pickle
    results.pickle(write=inp_settings["pickle"], nodes=inp_settings["pickle_nodes"])
    # export results to VTK
    results.vtk(write=inp_settings["VTK"], output_interval=1)

    # generate inputs for GNN
    if gnn:
        utils.generate_gnn_files(model, matrix, F, numerical, os.path.join(outfile_folder, "GNN"))

    # print end statement
    print("\n\n\n\x1B[3m" + "  Never tell me the odds. " + "\x1B[0m")
    print("\x1B[3m" + "--- Han Solo" + "\x1B[0m")
    return results
