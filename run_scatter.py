from src.scatter import scatter


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.01, 30, 0.01],
            "absorbing_BC": [1, 1],
            "pickle": True,
            "pickle_nodes": "all",
            "VTK": True,
            }

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
    mat = {"solid": {"density": 1500,
                     "Young": 30e6,
                     "poisson": 0.2},
           "bottom": {"density": 1200.0,
                      "Young": 300e6,
                      "poisson": 0.25}}

    load = {"force": [0, -1000, 0],
            "node": [3, 4, 7, 8],
            "time": 1,
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
    scatter(r"./mesh/column.msh", "./results", mat, BC, sett, load, time_step=0.5e-3)
    # scatter(r"./mesh/column.msh", "./results_static", mat, BC, sett, load, time_step=0.5e-3, type_analysis="static")
    # scatter(r"./mesh/column.msh", "./results_RF", mat, BC, sett, load, time_step=0.5e-3, random_props=RF_props)
