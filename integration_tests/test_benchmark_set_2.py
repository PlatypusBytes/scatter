# this file contains benchmark tests where the analytical/semi-analytical solution is known, however the exact solution
# is not feasible to obtain numerically. Numerical solutions are visually checked and recursively asserted.

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pytest
from scatter.scatter import scatter
from analytical_solutions import analytical_wave_prop

CHECK_RESULT = False


class TestBenchmarkSet2:

    @pytest.mark.parametrize("element_type,loading_nodes,n_dim",[("tri3", [3, 4, 25], 2),
                                                                 ("tri6", [3, 4, 47, 48, 49], 2),
                                                                 ("tetra4", [3, 4, 7, 8, 29, 69, 91, 92, 132], 3),
                                                                 ("tetra10", [3, 4, 7, 8, 51, 52, 53, 135, 136, 137,
                                                                              183, 184, 185, 186, 187, 188, 432, 433,
                                                                              434, 435, 436, 437, 438, 439, 440], 3)
                                                                 ])
    def test_heaviside_load_1d_column(self, element_type, loading_nodes, n_dim):
        """
        Tests heaviside loading on a 1d column with multiple types of elements

        :param element_type: type of element
        :param loading_nodes: list of all node numbers of the top boundary
        :param n_dim: number of dimensions (either 2d or 3d)

        """
        sett = {"gamma": 0.5,
                "beta": 0.25,
                "int_order": 2,
                "damping": [1, 0.001, 30, 0.001],
                "absorbing_BC": [1, 1],
                "absorbing_BC_stiff": 1e3,
                "pickle": True,
                "pickle_nodes": [3],
                "VTK": False,
                "output_interval": 10
                }

        x = 1
        y = 10
        z = -1
        if n_dim == 3:
            BC = {"bottom": ["010", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
                  "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
                  "right": ["100", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
                  "front": ["001", [[0, 0, 0], [x, 0, 0], [0, y, 0], [x, y, 0]]],
                  "back": ["001", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
                  }
        elif n_dim == 2:
            BC = {"bottom": ["01", [[0, 0, 0], [x, 0, 0]]],
                  "left": ["10", [[0, 0, 0], [0, y, 0]]],
                  "right": ["10", [[x, 0, 0], [x, y, 0]]]
                 }
        else:
            raise Exception(f"n_dim {n_dim} is not supported")

        # material dictionary: rho, E, v
        mat = {"solid": {"density": 1500,
                         "Young": 30e6,
                         "poisson": 0.0}}

        load = {"force": [0, 1000, 0],
                "node": loading_nodes,
                "time": 1,
                "type": "heaviside"}  # pulse or heaviside or moving

        # distribute load on nodes such that pressure == vertical force
        load["force"][1] = load["force"][1] / len(load["node"])

        # run scatter
        input_file = f"integration_tests/mesh/column_{n_dim}D_{element_type}.msh"
        output_dir = f"integration_tests/results_{element_type}"
        expected_res_file = f"integration_tests/test_data/column_{n_dim}D_{element_type}.pickle"
        res = scatter(input_file, output_dir, mat, BC, sett, load, time_step=5e-4)

        # get results
        calculated_disp = res.dis[:,0]
        calculated_vel = res.vel[:,0]

        # load expected results
        with open(expected_res_file,"rb") as f:
            expected_res = pickle.load(f)
            expected_disp = expected_res["displacement"]["3"]["y"]
            expected_vel = expected_res["velocity"]["3"]["y"]

        # assert displacement and velocity on top node
        np.testing.assert_array_almost_equal(calculated_disp, expected_disp[0::10])
        np.testing.assert_array_almost_equal(calculated_vel, expected_vel[0::10])

        # compare velocity with analytical solution if CHECK_RESULTS is true
        if CHECK_RESULT:

            res_analitic = analytical_wave_prop.OneDimWavePropagation(nb_terms=50)
            res_analitic.properties(mat["solid"]["density"], mat["solid"]["Young"],load["force"][1] * len(load["node"]),
                                    y, 20, time=res.time)
            res_analitic.solution()

            plt.plot(res.time, res.vel[:, 0])
            plt.plot(res_analitic.time, res_analitic.v[19, :])
            plt.show()


