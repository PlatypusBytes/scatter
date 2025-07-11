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
            "VTK_binary": True,
            "output_interval": 10}

    # boundary conditions
    x = 10
    y_min, y_max = -5, 0.5
    z = -78

    BC = {"bottom": ["020", [[0, y_min, 0], [x, y_min, 0], [0, y_min, z], [x, y_min, z]]],
          "left": ["100", [[0, y_min, 0], [0, y_min, z], [0, y_max, 0], [0, y_max, z]]],
          "right": ["200", [[x, y_min, 0], [x, y_min, z], [x, y_max, 0], [x, y_max, z]]],
          "front": ["001", [[0, y_min, 0], [x, y_min, 0], [0, y_max, 0], [x, y_max, 0]]],
          "back": ["001", [[0, y_min, z], [x, y_min, z], [0, y_max, z], [x, y_max, z]]],
          }

    # material dictionary: rho, E, v
    mat = {"embankment": {"density": 2000,
                          "Young": 100e6,
                          "poisson": 0.2},
           "soil1": {"density": 1700,
                     "Young": 40e6,
                     "poisson": 0.2},
           "soil2": {"density": 2000,
                     "Young": 10e6,
                     "poisson": 0.2},
           }

    rose_data = create_rose.create_input_dict(100, 0.01, 1.2,
                                              15,
                                              r"./mesh/embankment_rose.msh",
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
                "material": "soil2",
                "key_material": "Young",
                "std_value": 1e6,
                "aniso_x": 10,
                "aniso_z": 10,
                "model_name": "Gaussian"
                }

    # run scatter
    scatter(r"./mesh/embankment_rose.msh", "./results_rose_embankment_3d_rf3", mat, BC, sett, load, time_step=time_step, random_props=False)
