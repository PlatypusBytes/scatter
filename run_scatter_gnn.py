import os
import subprocess
import numpy as np
from scatter.scatter import scatter


if __name__ == "__main__":
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.01, 30, 0.01],
            "absorbing_BC": [1, 1],
            "absorbing_BC_stiff": 1e3,
            "pickle": True,
            "pickle_nodes": "4",
            "VTK": False,
            }

    for i in range(50):

        # define element size:
        np.random.seed(i)
        element_size = np.round(np.random.uniform(0.2, 1), 1)

        x = element_size
        y = 10
        z = -element_size

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
            "bottom": {"density": 1500.0,
                        "Young": 30e6,
                        "poisson": 0.2}}

        load = {"force": [0, -1000, 0],
                "node": [3, 4, 7, 8],
                "time": 0.025,
                "type": "heaviside",  # pulse or heaviside or moving
                "speed": 80}  # only for moving

        # Random field properties
        RF_props = {"number_realisations": 1,
                    "element_size": element_size,
                    "theta": 2,
                    "seed_number": -1,
                    "material": "solid",
                    "key_material": "Young",
                    "std_value": 50e6,
                    "aniso_x": 1,
                    "aniso_z": 1,
                    "model_name": "Gaussian",
                    }

        # edit geo file
        mesh_file = "./mesh/column.geo"

        RF_props["seed_number"] = i

        with open(mesh_file, "r") as file:
            lines = file.read().splitlines()
        lines[1] = f"boxdim = {element_size};"
        # output_folder
        output_folder = f"./results_gnn/run{i}"
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "column.geo"), "w") as file:
            file.write("\n".join(lines))

        # generate mesh
        subprocess.run(["/opt/gmsh-4.13.1-Linux64/bin/./gmsh", os.path.join(output_folder, "column.geo"), "-format", "msh2", "-3", "-o",  os.path.join(output_folder, "column.msh")])
        # run scatter
        scatter(os.path.join(output_folder, "column.msh"), output_folder, mat, BC, sett, load, time_step=1e-4, type_analysis="dynamic_explicit", random_props=RF_props, gnn=True)
