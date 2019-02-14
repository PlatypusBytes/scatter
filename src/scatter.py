def scatter(mesh_file, outfile_folder, materials, boundaries, inp_settings, loading, time_step=0.1):
    r"""
    3D finite element code.
                                                         y ^
    Mesh is generated with gmsh (add the link)             | /z
    the coordinate system is the same as defined in gmsh    --> x

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
    res = solver.Solver(model.number_eq, outfile_folder)
    # res.static(inp_settings, matrix.K, F.force, time_step, loading["time"])
    res.newmark(inp_settings, matrix.M, matrix.C, matrix.K, F.force, time_step, loading["time"])
    # save data
    res.save_data()

    # print
    print("Analysis done")
    return


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.001, 50, 0.001]}
    # boundary conditions
    BC = {"bottom": ["010", [[0, 0, 0], [1, 0, 0], [0, 0, -1], [1, 0, -1]]],
          "left": ["100", [[0, 0, 0], [0, 0, -1], [0, 0, 10], [0, 10, -1]]],
          "right": ["100", [[1, 0, 0], [1, 0, -1], [1, 10, 0], [1, 10, -1]]],
          "front": ["001", [[0, 0, 0], [1, 0, 0], [0, 10, 0], [1, 10, 0]]],
          "back": ["001", [[0, 0, -1], [1, 0, -1], [0, 10, -1], [1, 10, -1]]],
          }

    # material dictionary: rho, E, v
    mat = {"solid": [1500, 30e6, 0.2],
           "bottom": [1800, 20e4, 0.15]}
    load = {"force": [0, -1000, 0],
            "node": [3, 4, 7, 8],
            "time": 1,
            "type": "heaviside"}  # pulse or heaviside

    # run scatter
    scatter(r"./../gmsh_test/column.msh", "../results", mat, BC, sett, load, time_step=0.5e-3)
