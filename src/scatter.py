import sys
from src import mesher
from src import system_matrix
from src import force_external
from src import solver
from src import random_fields
from src import export_results


def scatter(mesh_file: str, outfile_folder: str, materials: dict, boundaries: dict,
            inp_settings: dict, loading: dict, time_step: float = 0.1, random_props: bool = False) -> None:
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
    """

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

    if random_props:
        # model.remap_elements()
        rf = random_fields.RF(random_props, materials, outfile_folder)

        rf.generate_gstools_rf(model.nodes, model.elem, model.dimension, angles=0.0, model_name='Gaussian')
        # rf.generate(model.nodes, model.elem)
        rf.dump()
        materials = rf.new_material
        model.materials = rf.new_model_material
        model.materials_index = rf.new_material_index

    # generate matrix internal
    # M, C, K
    matrix = system_matrix.GenerateMatrix(model.number_eq, inp_settings['int_order'])
    matrix.stiffness(model, materials)
    matrix.mass(model, materials)
    matrix.damping_Rayleigh(inp_settings["damping"])
    matrix.absorbing_boundaries(model, materials, inp_settings["absorbing_BC"])

    # generate matrix external
    F = force_external.Force()
    if loading["type"] == "pulse":
        F.pulse_load(model.number_eq, model.eq_nb_dof, loading, loading["node"], time_step)
    elif loading["type"] == "heaviside":
        F.heaviside_load(model.number_eq, model.eq_nb_dof, loading, loading["node"], time_step)
    elif loading["type"] == "moving":
        F.moving_load(model.number_eq, model.eq_nb_dof, loading, loading["node"], time_step, model.nodes)
    else:
        sys.exit(f'Error: Load type {loading["type"]} not supported')

    print("solver started")
    # solver
    numerical = solver.Solver(model.number_eq)
    # res.static(matrix.K, F.force, time_step, loading["time"])
    numerical.newmark(inp_settings, matrix.M, matrix.C, matrix.K, F.force, matrix.absorbing_bc, time_step, loading["time"])

    # export results
    results = export_results.Write(outfile_folder, model, materials, numerical)
    # export results to pickle
    results.pickle()
    # export results to VTK
    results.vtk()

    # print
    print("Analysis done")
    return
