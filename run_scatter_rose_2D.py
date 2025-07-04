from scatter.scatter import scatter
from scatter.rose_utils import RoseUtils
from rose.pre_process.default_trains import TrainType
from scatter import create_rose


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.005, 20, 0.005],
            "absorbing_BC": [1, 1],
            "absorbing_BC_stiff": 1e3,
            "pickle": True,
            "pickle_nodes": "all",
            "VTK": True,
            "output_interval": 100}

    # boundary conditions
    x = 90

    y_top = 0.5
    y_bot = -3

    BC = {"bottom": ["11", [[0, y_bot, 0], [x, y_bot, 0]]],
          "left": ["10", [[0, y_bot, 0], [0, y_top, 0]]],
          "right": ["10", [[x, y_bot, 0], [x, y_top, 0]]],
          }

    mat = {"embankment": {"density": 2000,
                          "Young": 100e6,
                          "poisson": 0.2},
           "soil1": {"density": 1700,
                     "Young": 500e5,
                     "poisson": 0.2},
           "soil2": {"density": 2000,
                     "Young": 200e5,
                     "poisson": 0.2},
           }

    rose_data = create_rose.create_input_dict(100, 0.1, 1.2,
                                              15,
                                              r"./mesh/rose_2D_side.msh",
                                              TrainType.DOUBLEDEKKER)


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
                "material": "soil1",
                "key_material": "Young",
                "std_value": 3e6,
                "aniso_x": 10,
                "aniso_z": 5,
                "model_name": "Gaussian"
                }

    # run scatter
    scatter(r"./mesh/rose_2D_side.msh", "./results_rose_2D_side", mat, BC, sett, load, time_step=time_step, random_props=RF_props)
