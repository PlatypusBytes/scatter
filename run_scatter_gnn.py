from pathlib import Path
import numpy as np
import create_geo_file
from scatter.scatter import scatter
from solvers.newmark_solver import NewmarkExplicitGPU
from solvers.preconditioners import JacobiPreconditionerGPU
from solvers.linear_equations_solvers import CGSolverGPU


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.01, 80, 0.01],
            "absorbing_BC": [1, 1],
            "absorbing_BC_stiff": 1e3,
            "pickle": True,
            "pickle_nodes": "4",
            "VTK": False,
            "VTK_binary": True,
            "output_interval": 1,
            "write_GNN": True,
            }


    for i in range(50):
        np.random.seed(i)
        element_size = np.round(np.random.uniform(0.2, 1), 2)
        model_size = int(np.random.uniform(5, 20))
        model_depth = int(np.random.uniform(5, 20))
        aniso_z = int(np.random.uniform(1, 10))

        x = model_size
        y = model_depth
        z = model_size

        BC = {"bottom": ["010", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
            "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
            "right": ["200", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
            "front": ["001", [[0, 0, 0], [z, 0, 0], [0, y, 0], [x, y, 0]]],
            "back": ["002", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
            }

        create_geo_file.main(x, y, z, element_size, Path(f"GNN/run_{i}"), name=f"brick")

        # material dictionary: rho, E, v
        mat = {"solid": {"density": 1500,
                        "Young": 30e6,
                        "poisson": 0.2},
                }

        load = {"force": [0, -1e6, 0],
                "node": [4],
                "time": 0.4,
                "type": "heaviside",  # pulse or heaviside or moving
                "speed": 80}  # only for moving

        # Random field properties
        RF_props = {"number_realisations": 1,
                    "element_size": element_size,
                    "theta": 2,
                    "seed_number": i,
                    "material": "solid",
                    "key_material": "Young",
                    "std_value": 50e6,
                    "aniso_x": 1,
                    "aniso_z": aniso_z,
                    "model_name": "Gaussian",
                    }

        # run scatter
        scatter(Path(f"GNN/run_{i}") / "brick.msh", Path(f"GNN/run_{i}"), mat, BC, sett, load, time_step=5e-4,
                solver=NewmarkExplicitGPU(linear_solver=CGSolverGPU(),
                                        preconditioner=JacobiPreconditionerGPU()), random_props=RF_props)
