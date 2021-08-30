import os
import numpy as np
import unittest
import shutil
import pickle
from src.scatter import scatter


dec_places = 5

def read_pickle(file):
    # read pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


class Test1DWavePropagation(unittest.TestCase):
    def setUp(self):

        self.root = "integration_tests"
        file = os.path.join(self.root, r"./results_mean/data.pickle")
        self.mean_data = read_pickle(file)
        file = os.path.join(self.root, r"./results_mean_high/data.pickle")
        self.mean_data_high = read_pickle(file)
        file = os.path.join(self.root, r"./results_rf/data.pickle")
        self.random_data = read_pickle(file)

        self.fold_results = []
        return

    def test_1(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "VTK": False}

        x = 0.1
        y = 20
        z = -0.1
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
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}
        load = {"force": [0, -1000, 0],
                "node": [3, 4, 7, 8],
                "time": 0.5,
                "type": "pulse"}  # pulse or heaviside

        # run scatter
        self.fold_results = os.path.join(self.root, "./_results_mean")
        scatter(os.path.join(self.root, r"./mesh/column.msh"), self.fold_results, mat, BC, sett, load, time_step=0.5e-3)

        # compare results
        data = read_pickle(os.path.join(self.fold_results, "data.pickle"))

        self.assert_dict_almost_equal(data, self.mean_data)
        return

    def test_2(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "VTK": False}

        x = 0.1
        y = 20
        z = -0.1
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
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}
        load = {"force": [0, -1000, 0],
                "node": [3, 4, 7, 8],
                "time": 0.5,
                "type": "pulse"}  # pulse or heaviside

        # Random field properties
        RF_props = {"number_realisations": 1,
                    "element_size": 1,
                    "theta": 5,
                    "seed_number": -26021981,
                    "material": "solid",
                    "key_material": "Young",
                    "std_value": 1e6,
                    "aniso_x": 1,
                    "aniso_y": 1,
                    }

        # run scatter
        self.fold_results = os.path.join(self.root, "./_results_rf")
        scatter(os.path.join(self.root, r"./mesh/column.msh"),  self.fold_results, mat, BC, sett, load, time_step=0.5e-3,  random_props=RF_props)

        # compare results
        data = read_pickle(os.path.join(self.fold_results, "data.pickle"))

        self.assert_dict_almost_equal(data, self.random_data)

        return

    def assert_dict_almost_equal(self, expected, actual):

        for key in expected:
            if isinstance(expected[key], dict):
                self.assert_dict_almost_equal(expected[key], actual[key])
            else:
                if isinstance(expected[key], np.ndarray):
                    np.testing.assert_almost_equal(expected[key], actual[key], decimal=dec_places)
                elif isinstance(expected[key], (int, float)):
                    self.assertAlmostEqual(expected[key], actual[key], places=dec_places)
                elif all(isinstance(n, str) for n in expected[key]):
                    # if elements are string
                    self.assertAlmostEqual(expected[key], actual[key])
                
        return

    def test_3(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "VTK": False}

        x = 0.1
        y = 20
        z = -0.1
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
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}
        load = {"force": [0, -1000, 0],
                "node": [3, 4, 7, 8],
                "time": 0.5,
                "type": "pulse"}  # pulse or heaviside

        # run scatter
        self.fold_results = os.path.join(self.root, "./_results_mean_high")
        scatter(os.path.join(self.root, r"./mesh/column_high_order.msh"), self.fold_results, mat, BC, sett, load, time_step=0.5e-3)

        # compare results
        data = read_pickle(os.path.join(self.fold_results, "data.pickle"))

        self.assert_dict_almost_equal(data, self.mean_data_high)
        return

    def tearDown(self):
        shutil.rmtree(self.fold_results)
        return


if __name__ == "__main__":
    unittest.main()
