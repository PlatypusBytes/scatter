def scatter(mesh_file, materials, boundaries, inp_settings, loading):
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
    model = mesher.ReadMesh(mesh_file)
    model.read_gmsh()
    model.read_bc(boundaries)

    # generate matrix internal
    # M, C, K
    matrix = gen_matrix.GenerateMatrix(model.NEQ, inp_settings['int_order'])
    matrix.stiffness(model, materials)
    matrix.mass(model, materials)
    matrix.damping_Rayleigh(inp_settings["damping"])

    # generate matrix external
    F = force_external.Force()
    if loading["type"] == "pulse":
        F.pulse_load(model.NEQ, model.ID, loading, loading["node"])
    elif loading["type"] == "heaviside":
        F.heaviside_load(model.NEQ, model.ID, loading, loading["node"])

    print("solver started")
    # solver
    res = solver.Solver(model.NEQ)
    res.newmark(inp_settings, matrix.M, matrix.C, matrix.K, F.force, 0.1, 1)

    # post processing
    # do something with paraview

    print("Analysis done")
    return


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 3,
            "damping": [1, 0.05, 10, 0.05]}
    # boundary conditions
    BC = {"bottom": ["010", [[0, 0, 0], [10, 0, 0], [0, 0, 3], [10, 0, 3]]],
          "left": ["100", [[0, 0, 0], [0, 6, 0], [0, 0, 3], [0, 6, 3]]],
          "right": ["100", [[10, 0, 0], [10, 0, 3], [10, 6, 0], [10, 6, 3]]],
          "front": ["001", [[0, 0, 3], [10, 0, 3], [10, 6, 3], [0, 6, 3]]],
          "back": ["001", [[0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0]]],
          }
    # material dictionary: rho, E, v
    mat = {"top": [1500, 30e5, 0.2],
           "bottom": [1800, 20e2, 0.15]}
    load = {"force": [0, 1000, 0],
            "node": 2821,
            "time": 1,
            "type": "heaviside"}  # pulse or heaviside

    # run scatter
    scatter(r"./../gmsh_test/test.msh", mat, BC, sett, load)
