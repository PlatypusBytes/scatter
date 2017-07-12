def scatter(mesh_file, materials, boundaries, inp_settings):
    r"""
    3D finite element code.

    Mesh is generated with gmsh (add the link)
    the coordinate system is the same as defined in gmsh

    """

    import mesher
    import gen_matrix
    
    # read gmsh mesh
    model = mesher.ReadMesh(mesh_file)
    model.read_gmsh()
    model.read_bc(boundaries)

    # generate matrix internal
    # M, C, K
    matrix = gen_matrix.GenerateMatrix(model.NEQ, inp_settings['int_order'])
    matrix.stiffness(model, materials)
    matrix.mass(model)
    matrix.damping(inp_settings)
    
    # generate matrix external
    # F = 
    # solver

    # post processing
    # do something with paraview

    print("Analysis done")
    return

if __name__ == "__main__":
    # computational settings
    sett = {"alpha": 0.25,
            "beta": 0.5,
            "int_order": 2}
    # boundary conditions
    BC = {"bottom": ["010", [[0, 0, 0], [10, 0, 0], [0, 0, 3], [10, 0, 3]]],
          "left": ["100", [[0, 0, 0], [0, 6, 0], [0, 0, 3], [0, 6, 3]]],
          "right": ["100", [[10, 0, 0], [10, 0, 3], [10, 6, 0], [10, 6, 3]]],
          "front": ["001", [[0, 0, 3], [10, 0, 3], [10, 6, 3], [0, 6, 3]]],
          "back": ["001", [[0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0]]],
          }
    # material dictionary: E, v
    mat = {'top': [30e5, 0.2],
           'bottom': [20e2, 0.15]}
    # run scatter
    scatter(r"./../gmsh_test/test.msh", mat, BC, sett)
