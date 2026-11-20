"""Default input of the rose train-track model."""
from typing import Any, Dict, List, Optional

import numpy as np
from rose.pre_process.default_trains import TrainType, set_train

from scatter.mesher import ReadMesh


def geometry(nb_sleeper: List[int], fact: int = 1) -> Dict[str, Any]:
    """Set the track geometry.

    Args:
        nb_sleeper: number of sleepers of each segment.
        fact: scale factor on the distance between sleepers.

    Returns:
        Track geometry.
    """
    return {
        "n_segments": len(nb_sleeper),
        # number of sleepers per segment
        "n_sleepers": [int(n / fact) for n in nb_sleeper],
        # distance between sleepers, equal for each segment [m]
        "sleeper_distance": 0.6 * fact,
        # depth of the soil per segment [m]
        "depth_soil": [1.],
    }


def materials() -> Dict[str, float]:
    """Set the track materials.

    Returns:
        Track material properties.
    """
    material = {}
    # rail: Young modulus [Pa], Poisson ratio [-], inertia [m^4], density [kg/m^3], area [m^2] and Timoshenko shear
    # factor [-]
    material["young_mod_beam"] = 210e9
    material["poisson_beam"] = 0.0
    material["inertia_beam"] = 2.24E-05
    material["rho"] = 7860.
    material["rail_area"] = 69.6e-2
    material["shear_factor_rail"] = 0.

    # Rayleigh damping: damping ratio [-] and first and second radial frequency [rad/s]
    material["damping_ratio"] = 0.02
    material["omega_one"] = 6.283
    material["omega_two"] = 125.66

    # rail pad: mass [kg], stiffness [N/m] and damping [N s/m]
    material["mass_rail_pad"] = 5.
    material["stiffness_rail_pad"] = 750e6
    material["damping_rail_pad"] = 750e3

    # sleeper mass [kg]
    material["mass_sleeper"] = 140.

    # Hertzian contact coefficient and power
    material["hertzian_contact_coef"] = 9.1e-7
    material["hertzian_power"] = 3 / 2

    return material


def time_integration(t_ini: float, t_calc: float) -> Dict[str, Optional[float]]:
    """Set the time integration data.

    Args:
        t_ini: duration of the initialisation [s].
        t_calc: duration of the calculation [s].

    Returns:
        Time integration data; the numbers of time steps are not set yet.
    """
    return {"tot_ini_time": t_ini, "n_t_ini": None, "tot_calc_time": t_calc, "n_t_calc": None}


def create_input_dict(speed: float,
                      initial_time: float,
                      travelling_time: float,
                      start_coordinate: float,
                      mesh: str,
                      train_type: TrainType,
                      nb_carts: int = 1) -> Dict[str, Any]:
    """Create the rose input dictionary.

    Args:
        speed: travelling speed [km/h].
        initial_time: duration of the initialisation [s].
        travelling_time: duration of the calculation [s].
        start_coordinate: starting y coordinate of the train [m].
        mesh: gmsh mesh file, with the rose connection elements.
        train_type: train type.
        nb_carts: number of train carts.

    Returns:
        rose input dictionary.
    """
    time_data = time_integration(initial_time, travelling_time)
    track_materials = materials()

    model = ReadMesh(mesh)
    model.read_gmsh()
    track_geometry = geometry([len(model.rose_nodes)], fact=1)

    track_info = {"geometry": track_geometry, "materials": track_materials}

    # default train
    train_model = set_train(np.nan, np.nan, start_coordinate, train_type, nb_carts=nb_carts)
    train_dict = {"velocity": speed / 3.6, "type": train_type.name, "model": train_model}

    return {"traffic_data": train_dict, "track_info": track_info, "time_integration": time_data}
