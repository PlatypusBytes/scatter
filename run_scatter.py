from scatter.scatter import scatter, Solver


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.01, 30, 0.01],
            "absorbing_BC": [1, 1],
            "absorbing_BC_stiff": 1e3,
            "pickle": True,
            "pickle_nodes": "all",
            "VTK": False,
            "VTK_binary": True,
            }

    x = 1
    y = 10
    z = -1
    BC = {"bottom": ["020", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
          "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
          "right": ["100", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
          "front": ["001", [[0, 0, 0], [z, 0, 0], [0, y, 0], [x, y, 0]]],
          "back": ["001", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
          }

    # material dictionary: rho, E, v
    mat = {"solid": {"density": 1500,
                     "Young": 30e6,
                     "poisson": 0.2},
           "bottom": {"density": 1500.0,
                      "Young": 30e6,
                      "poisson": 0.2}}

    load = {"force": [0, -1000, 0],
            "node": [3, 4, 7, 8],
            "time": 0.25,
            "type": "heaviside",  # pulse or heaviside or moving
            "speed": 80}  # only for moving

    # Random field properties
    RF_props = {"number_realisations": 1,
                "element_size": 1,
                "theta": 5,
                "seed_number": -26021981,
                "material": "solid",
                "key_material": "Young",
                "std_value": 1e6,
                "aniso_x": 1,
                "aniso_y": 1,
                "aniso_z": 1,
                }

    # run scatter
    scatter(r"./mesh/column.msh", "./results_abs_newmark", mat, BC, sett, load, time_step=0.05e-4, solver=Solver.NEWMARK_EXPLICIT)
    scatter(r"./mesh/column.msh", "./results_abs_bathe", mat, BC, sett, load, time_step=0.05e-4, solver=Solver.BATHE)
