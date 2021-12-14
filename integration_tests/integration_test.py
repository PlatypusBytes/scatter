import os
import numpy as np
import unittest
import shutil
import pickle
from pathlib import Path
from src.scatter import scatter


dec_places = 5


def is_str_float(n: str) -> bool:
    try:
        float(n)
        return True
    except ValueError:
        return False


def read_pickle(file):
    # read pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


def assert_dict_almost_equal(expected, actual):

    for key in expected:
        if isinstance(expected[key], dict):
            assert_dict_almost_equal(expected[key], actual[key])
        else:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_almost_equal(expected[key], actual[key], decimal=dec_places)
            elif isinstance(expected[key], (int, float)):
                unittest.TestCase.assertAlmostEqual(expected[key], actual[key], places=dec_places)
            elif all(isinstance(n, str) for n in expected[key]):
                # if elements are string
                unittest.TestCase.assertAlmostEqual(expected[key], actual[key])

    return


class Test1DWavePropagation_3D(unittest.TestCase):
    def setUp(self):

        self.root = "integration_tests"
        file = os.path.join(self.root, r"./results_mean/data.pickle")
        self.mean_data = read_pickle(file)
        self.mean_data_vtk = os.path.join(self.root, r"./results_mean/VTK")
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
                "pickle_nodes": "all",
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

        assert_dict_almost_equal(data, self.mean_data)
        return

    def test_2(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
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

        assert_dict_almost_equal(data, self.random_data)

        return

    def test_3(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
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

        assert_dict_almost_equal(data, self.mean_data_high)
        return

    def test_vtk(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
                "VTK": True}

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

        # compare VTK
        vtk_files = os.listdir(os.path.join(self.fold_results, "VTK"))
        for vtk in vtk_files:
            with open(os.path.join(self.fold_results, "VTK", vtk)) as fi:
                computed = fi.read().splitlines()
            with open(os.path.join(self.mean_data_vtk, vtk)) as fi:
                correct = fi.read().splitlines()

            for i in range(len(correct)):
                if all([is_str_float(n) for n in correct[i].split()]):
                    for n in zip(correct[i].split(), computed[i].split()):
                        self.assertAlmostEqual(float(n[0]), float(n[1]), dec_places)
                else:
                    self.assertTrue(correct[i], computed[i])
        return

    def tearDown(self):
        shutil.rmtree(self.fold_results)
        return


class Test1DWavePropagation_2D(unittest.TestCase):
    def setUp(self):

        self.root = "integration_tests"
        file = os.path.join(self.root, r"./results_mean_2d/data.pickle")
        self.mean_data = read_pickle(file)
        self.mean_data_vtk = os.path.join(self.root, r"./results_mean_2d/VTK")
        file = os.path.join(self.root, r"./results_rf_2d/data.pickle")
        self.random_data = read_pickle(file)

        self.fold_results = []
        return

    def test_1(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.005, 20, 0.005],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
                "VTK": False}

        # boundary conditions
        x = 1
        y = 10
        BC = {"bottom": ["01", [[0, 0, 0], [x, 0, 0]]],
              "left": ["10", [[0, 0, 0], [0, y, 0]]],
              "right": ["10", [[x, 0, 0], [x, y, 0]]],
              }

        # material dictionary: rho, E, v
        mat = {"solid": {"density": 1500,
                         "Young": 30e6,
                         "poisson": 0.2},
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}
        load = {"force": [0, -1e6, 0],
                "node": [3, 4, 25],
                "time": 1.0,
                "type": "heaviside"}

        # run scatter
        self.fold_results = os.path.join(self.root, "/_results_mean_2d")
        scatter(os.path.join(self.root, r"./mesh/column_2D.msh"), self.fold_results, mat, BC, sett, load, time_step=5e-3)

        # compare results
        data = read_pickle(os.path.join(self.fold_results, "data.pickle"))

        assert_dict_almost_equal(data, self.mean_data)
        return

    def test_2(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.005, 20, 0.005],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
                "VTK": False}

        # boundary conditions
        x = 1
        y = 10
        BC = {"bottom": ["01", [[0, 0, 0], [x, 0, 0]]],
              "left": ["10", [[0, 0, 0], [0, y, 0]]],
              "right": ["10", [[x, 0, 0], [x, y, 0]]],
              }

        # material dictionary: rho, E, v
        mat = {"solid": {"density": 1500,
                         "Young": 30e6,
                         "poisson": 0.2},
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}
        load = {"force": [0, -1e6, 0],
                "node": [3, 4, 25],
                "time": 1.0,
                "type": "heaviside"}

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
        self.fold_results = os.path.join(self.root, "./_results_rf_2d")
        scatter(os.path.join(self.root, r"./mesh/column_2D.msh"), self.fold_results, mat, BC, sett, load, time_step=5e-3, random_props=RF_props)

        # compare results
        data = read_pickle(os.path.join(self.fold_results, "data.pickle"))

        assert_dict_almost_equal(data, self.random_data)
        return

    def test_vtk(self):
        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.005, 20, 0.005],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
                "VTK": True}

        # boundary conditions
        x = 1
        y = 10
        BC = {"bottom": ["01", [[0, 0, 0], [x, 0, 0]]],
              "left": ["10", [[0, 0, 0], [0, y, 0]]],
              "right": ["10", [[x, 0, 0], [x, y, 0]]],
              }

        # material dictionary: rho, E, v
        mat = {"solid": {"density": 1500,
                         "Young": 30e6,
                         "poisson": 0.2},
               "bottom": {"density": 1200,
                          "Young": 300e6,
                          "poisson": 0.25}}
        load = {"force": [0, -1e6, 0],
                "node": [3, 4, 25],
                "time": 1.0,
                "type": "heaviside"}

        # run scatter
        self.fold_results = os.path.join(self.root, "./_results_mean_2d")
        scatter(os.path.join(self.root, r"./mesh/column_2D.msh"), self.fold_results, mat, BC, sett, load, time_step=5e-3)

        # compare VTK
        vtk_files = os.listdir(os.path.join(self.fold_results, "VTK"))
        for vtk in vtk_files:
            with open(os.path.join(self.fold_results, "VTK", vtk)) as fi:
                computed = fi.read().splitlines()
            with open(os.path.join(self.mean_data_vtk, vtk)) as fi:
                correct = fi.read().splitlines()

            for i in range(len(correct)):
                if all([is_str_float(n) for n in correct[i].split()]):
                    for n in zip(correct[i].split(), computed[i].split()):
                        self.assertAlmostEqual(float(n[0]), float(n[1]), dec_places)
                else:
                    self.assertTrue(correct[i], computed[i])
        return

    def tearDown(self):
        shutil.rmtree(self.fold_results)
        return


class TestBenchmarkSet(unittest.TestCase):

    def setUp(self):
        pass

    def test_moving_load_plain(self):
        """
        Regression test for moving load at plane.
        """

        # computational settings
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.00, 30, 0.00],
                # "damping": [1, 0.01, 30, 0.01],
                "absorbing_BC": [1, 1],
                "pickle": True,
                "pickle_nodes": "all",
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
        input_file = r"integration_tests/mesh/cube.msh"
        output_dir = "integration_tests/results_RF/cube_res"
        scatter(input_file, output_dir, mat, BC, sett, load, time_step=0.5e-2, random_props=RF_props)

        # open results and delete file
        with open(Path(output_dir, "data.pickle"), "rb") as f:
            res_data = pickle.load(f)
        Path(output_dir, "data.pickle").unlink()

        # open results to be asserted with
        with open("integration_tests/test_data/moving_load_plane_res.pickle", "rb") as f:
            assert_data = pickle.load(f)

        assert_dict_almost_equal(res_data, assert_data)


if __name__ == "__main__":
    unittest.main()
