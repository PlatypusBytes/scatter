def scatter(mesh_file, boundaries, inp_settings):
    r"""
    3D finite element code.

    Mesh is generated with gmsh (add the link)
    the coordinate system is the same as defined in gmsh

    """

    import mesher
    import gen_matrix
    
    # read gmesh mesh
    model = mesher.ReadMesh(mesh_file)
    model.read_gmsh()
    model.read_bc(boundaries)

    # here read the file
    
    #model.mesh_square(50, 30, 60, 10, 10, 10)
    # ToDo: think about BC
    
    
    # generate matrix internal
    # M, C, K
    matrix = gen_matrix.Generate_Matrix()
    matrix.stiffness(model)
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
            "beta": 0.5}
    # boundary conditions
    BC = {"bottom": ["010", [[0, 0, 0], [10, 0, 0], [0, 0, 3], [10, 0, 3]]],
          "left": ["100", [[0, 0, 0], [0, 6, 0], [0, 0, 3], [0, 6, 3]]],
          "right": ["100", [[10, 0, 0], [10, 0, 3], [10, 6, 0], [10, 6, 3]]],
          "front": ["001", [[0, 0, 3], [10, 0, 3], [10, 6, 3], [0, 6, 3]]],
          "back": ["001", [[0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0]]],
          }
    # material dictionary: E, v
    material = {'mat 1': [30e5, 0.2],
                'mat 1': [20e2, 0.15]}
    # run scatter
    scatter(r"./../gmsh_test/test_linear_elem.msh", BC, sett)
