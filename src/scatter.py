def scatter(mesh_file, outfile_folder, materials, boundaries, inp_settings, loading, time_step=0.1, random_props=False):
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
    import random_fields

    # read gmsh mesh
    # create structure
    model = mesher.ReadMesh(mesh_file, outfile_folder)
    # read gmesh file: file_name, dimension, nb_nodes_elem, materials, nodes, elem
    model.read_gmsh()
    # define boundary conditions
    model.read_bc(boundaries)
    # mapping of equation numbers and BC
    model.mapping()
    # connectivities
    model.connectivities()
    if random_props:
        # model.remap_elements()
        rf = random_fields.RF(random_props, outfile_folder)
        rf.generate(model.nodes, model.elem)
        rf.dump()
        materials = rf.new_material
        model.materials = rf.new_model_material
        model.elem = rf.new_elements

    # generate matrix internal
    # M, C, K
    matrix = gen_matrix.GenerateMatrix(model.number_eq, inp_settings['int_order'])
    matrix.stiffness(model, materials)
    matrix.mass(model, materials)
    matrix.damping_Rayleigh(inp_settings["damping"])

    # generate matrix external
    F = force_external.Force()
    if loading["type"] == "pulse":
        F.pulse_load(model.number_eq, model.eq_nb_dof, loading, loading["node"], time_step)
    elif loading["type"] == "heaviside":
        F.heaviside_load(model.number_eq, model.eq_nb_dof, loading, loading["node"], time_step)

    print("solver started")
    # solver
    res = solver.Solver(model.number_eq)
    # res.static(inp_settings, matrix.K, F.force, time_step, loading["time"])
    res.newmark(inp_settings, matrix.M, matrix.C, matrix.K, F.force, time_step, loading["time"])

    # remap the data to output structure
    model.remap_results(res.time, res.u, res.v, res.a)

    # print
    print("Analysis done")
    return


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.01, 30, 0.01]}

    x = 1
    y = 10
    z = -1
    BC = {"bottom": ["010", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
          "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
          "right": ["100", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
          "front": ["001", [[0, 0, 0], [z, 0, 0], [0, y, 0], [x, y, 0]]],
          "back": ["001", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
          }

    # material dictionary: rho, E, v
    mat = {"solid": [1500, 30e6, 0.2],
           "bottom": [1800, 20e4, 0.15]}
    load = {"force": [0, -1000, 0],
            "node": [3, 4, 7, 8],
            "time": 1,
            "type": "heaviside"}  # pulse or heaviside

    # Random field properties
    RF_props = {"number_realisations": 1,
                "element_size": 1,
                "theta": 5,
                "seed_number": -26021981,
                "material": mat["solid"],
                "index_material": 1,
                "std_value": 1e5,
                "aniso_x": 1,
                "aniso_y": 1,
                }

    # run scatter
    scatter(r"../mesh/column.msh", "../results", mat, BC, sett, load, time_step=0.5e-3)
    scatter(r"../mesh/column.msh", "../results_RF", mat, BC, sett, load, time_step=0.5e-3, random_props=RF_props)
