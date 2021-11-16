# unit test for gen_matrix
# unit_test based on Bathe
import unittest
from src import solver
import numpy as np
from scipy.sparse import lil_matrix


class TestSolver(unittest.TestCase):
    def setUp(self):
        # newmark settings
        self.settings = {'beta': 0.25,
                         'gamma': 0.5,
                         }

        # example from bathe
        M = [[2., 0], [0, 1.]]
        K = [[6., -2.], [-2., 4.]]
        C = [[0, 0], [0, 0]]
        F = np.zeros((2, 12))
        F_abs = np.zeros((2, 2))
        F[1, :] = 10
        self.M = lil_matrix(M)
        self.K = lil_matrix(K)
        self.C = lil_matrix(C)
        self.F = lil_matrix(F)
        self.F_abs = lil_matrix(F_abs)

        self.u0 = np.zeros(2)
        self.v0 = np.zeros(2)

        self.t_step = 0.28
        self.time = np.linspace(0, 11 * self.t_step, 12)

        self.number_eq = 2
        return

    def test_const(self):
        # check the constants
        aux = solver.const(self.settings["beta"], self.settings["gamma"], self.t_step)
        # assert if it is true
        self.assertEqual(aux[0], 1 / (self.settings["beta"] * self.t_step ** 2))
        self.assertEqual(aux[1], 1 / (self.settings["beta"] * self.t_step))
        self.assertEqual(aux[2], (1 / (2 * self.settings["beta"]) - 1))
        self.assertEqual(aux[3], self.settings["gamma"] / (self.settings["beta"] * self.t_step))
        self.assertEqual(aux[4], (self.settings["gamma"] / self.settings["beta"]) - 1)
        self.assertEqual(aux[5], self.t_step / 2 * (self.settings["gamma"] / self.settings["beta"] - 2))
        return

    def test_a_init(self):
        force = np.array([float(i) for i in self.F.getcol(0).todense()])
        # check computation of the acceleration
        acc = solver.init(self.M, self.C, self.K, force, self.u0, self.v0)

        # assert if true
        np.testing.assert_array_equal(acc, np.array([0, 10]))
        return

    def test_solver_newmark(self):

        res = solver.Solver(self.number_eq)
        res.newmark(self.settings, self.M, self.C, self.K, self.F, self.F_abs, self.t_step, self.time)
        np.testing.assert_array_almost_equal(np.round(res.u, 2), np.round(np.array([[0.00673, 0.364],
                                                                           [0.0505, 1.35],
                                                                           [0.189, 2.68],
                                                                           [0.485, 4.00],
                                                                           [0.961, 4.95],
                                                                           [1.58, 5.34],
                                                                           [2.23, 5.13],
                                                                           [2.76, 4.48],
                                                                           [3.00, 3.64],
                                                                           [2.85, 2.90],
                                                                           [2.28, 2.44],
                                                                           [1.40, 2.31],
                                                                           ]), 2))
        return

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
