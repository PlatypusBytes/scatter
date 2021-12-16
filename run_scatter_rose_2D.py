import sys
# sys.path.append(r"C:\Users\zuada\software_dev\scatter")
from src.scatter import scatter

import create_rose
from src.rose_utils import RoseUtils


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.005, 20, 0.005],
            "absorbing_BC": [1, 1],
            "pickle": True,
            "pickle_nodes": "all",
            "VTK": True,
            "output_interval": 100}

    # boundary conditions
    x = 120
    y = 1.8
    #z = -20
    BC = {"bottom": ["01", [[0, 0, 0], [x, 0, 0]]],
          "left": ["10", [[0, 0, 0], [0, y, 0]]],
          "right": ["10", [[x, 0,0], [x, y, 0]]],

          }

    # material dictionary: rho, E, v
    mat = {"solid": {"density": 0.002000,
                     "Young": 30e8,
                     "poisson": 0.2},
           "bottom": {"density": 0.002500,
                      "Young": 300e8,
                      "poisson": 0.25}}

    rose_data = create_rose.create_input_dict()

    # set time integration,
    # note that each timestep is equal. This includes the initialisation stage in Rose
    time_step = 1e-4
    loading_time = rose_data["time_integration"]["tot_ini_time"] + rose_data["time_integration"]["tot_calc_time"]
    rose_data["time_integration"]["n_t_ini"] = round(rose_data["time_integration"]["tot_ini_time"] / time_step)
    rose_data["time_integration"]["n_t_calc"] = round(rose_data["time_integration"]["tot_calc_time"] / time_step)

    coupled_model = RoseUtils.assign_data_to_coupled_model(rose_data)

    load = {"model": coupled_model,
            "type": "rose",
            "time": loading_time}

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
    scatter(r"./mesh/box2d.msh", "./results_rose_2d", mat, BC, sett, load, time_step=time_step)

#
