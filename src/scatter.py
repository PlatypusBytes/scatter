def scatter(mesh_file, materials, boundaries, inp_settings, loading, time_step=0.1):
    r"""
    3D finite element code.

    Mesh is generated with gmsh (add the link)
    the coordinate system is the same as defined in gmsh

    Consistent mass matrix
    """
    import mesher
    import gen_matrix
    import force_external
    import solver

    # read gmsh mesh
    # create structure
    model = mesher.ReadMesh(mesh_file)
    # read gmesh file: file_name, dimension, nb_nodes_elem, materials, nodes, elem
    model.read_gmsh()
    # define boundary conditions
    model.read_bc(boundaries)
    # mapping of equation numbers and BC
    model.mapping()
    # connectivities
    model.connectivities()

    # generate matrix internal
    # M, C, K
    matrix = gen_matrix.GenerateMatrix(model.number_eq, inp_settings['int_order'])
    matrix.stiffness(model, materials)
    matrix.mass(model, materials)
    matrix.damping_Rayleigh(inp_settings["damping"])

    # generate matrix external
    F = force_external.Force()
    if loading["type"] == "pulse":
        F.pulse_load(model.number_eq, model.eq_nb_dof, loading, loading["node"])
    elif loading["type"] == "heaviside":
        F.heaviside_load(model.number_eq, model.eq_nb_dof, loading, loading["node"], time_step)

    print("solver started")
    # solver
    res = solver.Solver(model.number_eq)
    # res.static(inp_settings, matrix.K, F.force, time_step, loading["time"])
    res.newmark(inp_settings, matrix.M, matrix.C, matrix.K, F.force, time_step, loading["time"])

    # post processing
    # do something with paraview
    import pickle
    data = {"displacement": res.u,
            "time": res.time}
    with open("./res.pickle", "wb") as f:
        pickle.dump(data, f)
    print("Analysis done")
    return


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.05, 10, 0.05]}
    # boundary conditions
    # BC = {"bottom": ["010", [[0, 0, 0], [10, 0, 0], [0, 0, -10], [10, 0, -10]]],
    #       "left": ["100", [[0, 0, 0], [0, 10, 0], [0, 0, -10], [0, 10, -10]]],
    #       "right": ["100", [[10, 0, 0], [10, 0, -10], [10, 10, 0], [10, 10, -10]]],
    #       "front": ["001", [[0, 0, -10], [10, 0, -10], [10, 10, -10], [0, 10, -10]]],
    #       "back": ["001", [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]],
    #       }
    BC = {"bottom": ["111", [[0, 0, 0], [10, 0, 0], [0, 0, -10], [10, 0, -10]]],
          }

    # material dictionary: rho, E, v
    mat = {"top": [1500, 30e5, 0.2],
           "bottom": [1800, 20e4, 0.15]}
    load = {"force": [10, 10, 0],
            "node": 2,
            "time": 0.001,
            "type": "heaviside"}  # pulse or heaviside

    # run scatter
    scatter(r"./../gmsh_test/test_1E.msh", mat, BC, sett, load, time_step=1e-5)
