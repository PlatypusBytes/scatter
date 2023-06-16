# unit test for gen_matrix
# add the scatter folder to the path to search for files
import unittest
from scatter import system_matrix
import numpy as np
from scipy.sparse import lil_matrix


class TestDamp(unittest.TestCase):
    def setUp(self):
        pass

    def test_damp1(self):
        f1 = 1
        d1 = 0.01
        f2 = 30
        d2 = 0.01
        settings = {"damping": [f1, d1, f2, d2]
                    }

        matrix = system_matrix.GenerateMatrix(2, 2)
        matrix.K = lil_matrix(np.zeros((2, 2)))
        matrix.M = lil_matrix(np.zeros((2, 2)))
        matrix.K[0, 0] = 0.75
        matrix.K[1, 1] = 0.75
        matrix.M[0, 0] = 0.53
        matrix.M[1, 1] = 0.35

        matrix.damping_Rayleigh(settings["damping"])

        # analytical solution
        a = (4 * d2 * f1**2 * f2 - 4 * d1 * f1 * f2**2) * np.pi / (f1**2 - f2**2)
        b = (d1 * f1 - d2 * f2) / ((f1**2 - f2**2) * np.pi)

        np.testing.assert_array_almost_equal(matrix.C.todense(), matrix.M.todense() * a + matrix.K.todense() * b)

        return

    def test_damp2(self):
        f1 = 1
        d1 = 0.01
        f2 = 30
        d2 = 0.01
        settings = {"damping": [f1, d1, f2, d2]
                    }

        matrix = system_matrix.GenerateMatrix(1, 2)
        matrix.K = lil_matrix(np.zeros((1, 1)))
        matrix.M = lil_matrix(np.zeros((1, 1)))
        matrix.K[0, 0] = 0.5
        matrix.M[0, 0] = 0.5

        matrix.damping_Rayleigh(settings["damping"])

        # analytical solution
        a = (4 * d2 * f1**2 * f2 - 4 * d1 * f1 * f2**2) * np.pi / (f1**2 - f2**2)
        b = (d1 * f1 - d2 * f2) / ((f1**2 - f2**2) * np.pi)

        np.testing.assert_array_almost_equal(matrix.C.todense(), matrix.M.todense() * a + matrix.K.todense() * b)
        return

    def mass_matrix(self):

        return

    def stiff_matrix(self):

        return

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
