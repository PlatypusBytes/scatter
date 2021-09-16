
import os
from pathlib import Path
# import pytest
import unittest
from src.scatter import scatter

import pickle
import numpy as np

class TestBenchmarkSet(unittest.TestCase):

    def setUp(self):
        pass

    def test_moving_load_plain(self):
        """
        Regression test for moving load at plane. Note that only one node is checked
        """

        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.00, 30, 0.00],
                # "damping": [1, 0.01, 30, 0.01],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "VTK": False}

        x = 10
        y = 10
        z = -10
        BC = {"bottom": ["010", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
              "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
              "right": ["100", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
              "front": ["001", [[0, 0, 0], [z, 0, 0], [0, y, 0], [x, y, 0]]],
              "back": ["001", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
              }

        # material dictionary: rho, E, v
        mat = {"solid": {"density": 1500,
                         "Young": 10e6,
                         "poisson": 0.2},
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}

        load = {"force": [0, -1000, 0],
                "start_coord": [0.5, -0.5],
                "time": 1,
                "type": "moving_at_plane",
                "direction": [0.5, -1],
                "speed": 10}  # pulse or heaviside

        # Random field properties
        RF_props = {"number_realisations": 1,
                    "element_size": 1,
                    "theta": 1,
                    "seed_number": -26021981,
                    "material": "solid",
                    "key_material": "Young",
                    "std_value": 7.5e6,
                    "aniso_x": 1,
                    "aniso_y": 5,
                    }

        # run scatter
        input_file = r"test_data/cube.msh"
        output_dir = "./results_RF/cube_res"
        scatter(input_file, output_dir, mat, BC, sett, load, time_step=0.5e-2, random_props=RF_props)

        # open results and delete file
        with open(Path(output_dir,"data.pickle"), "rb") as f:
            res_data = pickle.load(f)
        Path(output_dir, "data.pickle").unlink()

        # open results to be asserted with
        with open("./test_data/moving_load_plane_res.pickle", "rb") as f:
            assert_data = pickle.load(f)

        np.testing.assert_array_almost_equal(res_data["displacement"]["500"]["x"],assert_data["displacement"]["500"]["x"])
        np.testing.assert_array_almost_equal(res_data["displacement"]["500"]["y"],
                                             assert_data["displacement"]["500"]["y"])
        np.testing.assert_array_almost_equal(res_data["displacement"]["500"]["z"],
                                             assert_data["displacement"]["500"]["z"])

