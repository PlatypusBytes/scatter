import os
import sys
import numpy as np
from scatter import mesher
from scatter import system_matrix
from scatter import force_external
from scatter import random_fields
from scatter import export_results
from scatter.rose_utils import RoseUtils
from solvers import newmark_solver, static_solver


def scatter(mesh_file: str, outfile_folder: str, materials: dict, boundaries: dict,
            inp_settings: dict, loading: dict, time_step: float = 0.1, random_props: bool = False,
            type_analysis="dynamic") -> export_results.Write:
    r"""
    3D finite element code.
                                                          y ^
    Mesh is generated with gmsh https://gmsh.info/          | /z
    The coordinate system is the same as defined in gmsh    --> x

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
    """

    # print message
    print(open(os.path.join(os.path.dirname(__file__), '../docs/static/message.txt'), "r").read())

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

    matrix.absorbing_boundaries(model, materials, inp_settings["absorbing_BC"])

    # add connect scatter and rose stiffness and mass matrices and absorbing boundaries
    if loading["type"] == "rose":
        matrix.add_rose_stiffness(model, loading["model"])
        matrix.add_rose_mass(model, loading["model"])
        matrix.reshape_absorbing_boundaries_with_rose()

    # generate C matrix with Rayleigh damping
    matrix.damping_Rayleigh(inp_settings["damping"])

    # connect scatter and rose damping matrices
    if loading["type"] == "rose":
        matrix.add_rose_damping(model, loading["model"])

    # definition of time
    time = np.linspace(0, loading["time"], int(np.ceil(loading["time"] / time_step) + 1))

    # initialise solver
    if type_analysis == "dynamic":
        numerical = newmark_solver.NewmarkExplicit()
    elif type_analysis == "static":
        numerical = static_solver.StaticSolver()
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
    top_surface_elements = model.get_top_surface()
    F.initialise_load(model.number_eq, model.eq_nb_dof, model.nodes, loading, time,top_surface_elements=top_surface_elements)
    numerical.update_rhs_at_time_step_func = F.update_load_at_t

    # if loading["type"] == "pulse":
    #     F.pulse_load(model.number_eq, model.eq_nb_dof, model.nodes, loading, time)
    # elif loading["type"] == "heaviside":
    #     F.heaviside_load(model.number_eq, model.eq_nb_dof, model.nodes, loading, time)
    # elif loading["type"] == "moving":
    #     F.moving_load(model.number_eq, model.eq_nb_dof, model.nodes, loading, time, model.nodes)
    # elif loading["type"] == "moving_at_plane":
    #     top_surface_elements = model.get_top_surface()
    #     F.moving_load_at_plane(model.number_eq, model.eq_nb_dof, loading, loading["start_coord"], time, top_surface_elements,
    #                            model.nodes)
    # elif loading["type"] == "rose":
    #     F.add_rose_load(model, loading["model"])
    #     RoseUtils.set_rose_loading(model, loading["model"], numerical)
    #     print("Rose model is connected")
    # else:
    #     sys.exit(f'Error: Load type {loading["type"]} not supported')

    print("solver started")
    # start solver
    if type_analysis == "dynamic":
        numerical.absorbing_boundary = matrix.absorbing_bc
        numerical.update(0)
        numerical.calculate(matrix.M, matrix.C, matrix.K, F.force_vector, 0, len(F.time)-1)
    elif type_analysis == "static":
        numerical.calculate(matrix.K, F.force_vector, 0,len(F.time)-1)

    # export results
    results = export_results.Write(outfile_folder, model, materials, numerical)
    # export results to pickle
    results.pickle(write=inp_settings["pickle"], nodes=inp_settings["pickle_nodes"])
    # export results to VTK
    results.vtk(write=inp_settings["VTK"], output_interval=1)

    # print end statement
    print("\n\n\n\x1B[3m" + "  Never tell me the odds. " + "\x1B[0m")
    print("\x1B[3m" + "--- Han Solo" + "\x1B[0m")
    return results
