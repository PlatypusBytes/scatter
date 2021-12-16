
from rose.pre_process.default_trains import TrainType, set_train


def geometry(nb_sleeper, fact=1):
    """
    Sets track geometry parameters

    :param nb_sleeper:
    :param fact:
    :return:
    """
    # Set geometry parameters
    geometry = {}
    geometry["n_segments"] = len(nb_sleeper)
    geometry["n_sleepers"] = [int(n / fact) for n in nb_sleeper]  # number of sleepers per segment
    geometry["sleeper_distance"] = 0.6 * fact  # distance between sleepers, equal for each segment
    geometry["depth_soil"] = [1.]  # depth of the soil [m] per segment

    return geometry


def materials():
    """
    Sets track material parameters
    :return:
    """
    material = {}
    # set parameters of the rail
    material["young_mod_beam"] = 210e9  # young modulus rail
    material["poisson_beam"] = 0.0  # poison ration rail
    material["inertia_beam"] = 2.24E-05  # inertia of the rail
    material["rho"] = 7860.  # density of the rail
    material["rail_area"] = 69.6e-2  # area of the rail
    material["shear_factor_rail"] = 0.  # Timoshenko shear factor

    # Rayleigh damping system
    material["damping_ratio"] = 0.02  # damping
    material["omega_one"] = 6.283  # first radial_frequency
    material["omega_two"] = 125.66  # second radial_frequency

    # set parameters rail pad
    material["mass_rail_pad"] = 5.  # mass of the rail pad [kg]
    material["stiffness_rail_pad"] = 750e6  # stiffness of the rail pad [N/m2]
    material["damping_rail_pad"] = 750e3  # damping of the rail pad [N/m2/s]

    # set parameters sleeper
    material["mass_sleeper"] = 140.  # [kg]

    # set up contact parameters
    material["hertzian_contact_coef"] = 9.1e-7  # Hertzian contact coefficient
    material["hertzian_power"] = 3 / 2  # Hertzian power

    return material


def time_integration():
    """
    Sets time integration data

    :return:
    """
    time = {}

    # set time parameters in two stages
    time["tot_ini_time"] = 0.4  # total initalisation time  [s]
    time["n_t_ini"] = None  # number of time steps initialisation time  [-]

    time["tot_calc_time"] = 1.2  # total time during calculation phase   [s]
    time["n_t_calc"] = None  # number of time steps during calculation phase [-]

    return time


def create_input_dict():
    import numpy as np


    # set time integration and track information
    time_data = time_integration()
    track_materials = materials()
    track_geometry = geometry([201],fact=1)

    track_info = {"geometry": track_geometry,
                  "materials": track_materials}

    # get default trains
    train_velocity = 100 / 3.6
    train_type = "sprinter"
    train_model = set_train(np.nan, np.nan, 15, TrainType.SPRINTER)
    train_dict = {"velocity": train_velocity,
                  "type": train_type,
                  "model": train_model}

    #set soil data
    soil_dict={"stiffness": 200e6,
               "damping": 20e6}

    input_dict = {"traffic_data": train_dict,
                  "track_info": track_info,
                  "soil_data": soil_dict,
                  "time_integration": time_data}

    return input_dict


if __name__ == '__main__':
    create_input_dict()
