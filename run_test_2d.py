import sys
# sys.path.append(r"C:\Users\zuada\software_dev\scatter")
from src.scatter import scatter


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.05, 20, 0.05],
            "absorbing_BC": [1, 1],
            "pickle": True,
            "pickle_nodes": "all",
            "VTK": True}

    # boundary conditions
    x = 1
    y = 10
    #z = -20
    BC = {"bottom": ["01", [[0, 0,0], [x, 0, 0]]],
          "left": ["10", [[0, 0,0], [0, y, 0]]],
          "right": ["10", [[x, 0,0], [x, y, 0]]],

          }

    # material dictionary: rho, E, v
    mat = {"solid": {"density": 0.002000,
                     "Young": 30e6,
                     "poisson": 0.2},
           "bottom": {"density": 0.002500,
                      "Young": 300e6,
                      "poisson": 0.25}}
    load = {"force": [0, -1e6, 0],
            "node": [3,4],
            "time": 1.0,
            "type": "heaviside"}

    # Random field properties
    RF_props = {"number_realisations": 1,
                "element_size": 1,
                "theta": 1,
                "seed_number": -26021981,
                "material": "solid",
                "key_material": "Young",
                "std_value": 3e6,
                "aniso_x": 2,
                "aniso_y": 5,
                }

    # run scatter
    # scatter(r"./mesh/brick.msh", "./results_mn_mean", mat, BC, sett, load, time_step=5e-3, random_props=RF_props)
    scatter(r"./mesh/column2.msh", "./results_tmp", mat, BC, sett, load, time_step=5e-3)
    # scatter(r"./mesh/brick.msh", "./results_mn_mean", mat, BC, sett, load, time_step=5e-3)

    # for i in range(1):
    #     RF_props["seed_number"] = i
    #     scatter(r"./mesh/brick.msh", "./results_mv_RF_1/run_" + str(i), mat, BC, sett, load,
    #             time_step=1.5e-3, random_props=RF_props)
#
