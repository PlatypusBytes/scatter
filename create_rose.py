import numpy as np
# rose packages
from rose.pre_process.default_trains import set_train
from src.mesher import ReadMesh


def geometry(nb_sleeper, fact=1):
    """
    Sets track geometry parameters

    :param nb_sleeper: number of sleepers
    :param fact: scale factor between sleepers. Default 1
    :return: dictionary with geometry
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

    :return: Dictionary with track materials
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


def time_integration(t_ini, t_calc):
    """
    Sets time integration data

    :param t_ini: Time for initialisation
    :param t_calc: Time for calculation
    :return: time dictionary
    """
    time = {"tot_ini_time": t_ini,
            "n_t_ini": None,
            "tot_calc_time": t_calc,
            "n_t_calc": None}
    return time


def create_input_dict(speed, initial_time, travelling_time,
                      stiffness, damping,
                      start_coordinate, mesh, train_type):
    """
    Creates ROSE input dictionary

    :param speed: travelling speed [km/h]
    :param initial_time: Initialisation time
    :param travelling_time: Running time
    :param stiffness: Soil stiffness
    :param damping: Soil damping
    :param start_coordinate: starting y coordinate of the train
    :param mesh: file for the mesh
    :param train_type: Train type
    :return: ROSE input dictionary
    """
    # set time integration and track information
    time_data = time_integration(initial_time, travelling_time)
    track_materials = materials()

    mesh = ReadMesh(mesh)
    mesh.read_gmsh()

    track_geometry = geometry([len(mesh.rose_nodes)], fact=1)

    track_info = {"geometry": track_geometry,
                  "materials": track_materials}

    # get default trains
    train_velocity = speed / 3.6
    train_model = set_train(np.nan, np.nan, start_coordinate, train_type)
    train_dict = {"velocity": train_velocity,
                  "type": train_type.name,
                  "model": train_model}

    # set soil data
    soil_dict = {"stiffness": stiffness,
                 "damping": damping}

    input_dict = {"traffic_data": train_dict,
                  "track_info": track_info,
                  "soil_data": soil_dict,
                  "time_integration": time_data}

    return input_dict
