# unit test for gen_matrix

import sys
# add the src folder to the path to search for files
sys.path.append('../src/')
import unittest
import gen_matrix
import numpy as np


class TestDamp(unittest.TestCase):
    def setUp(self):
        pass

    def test_damp1(self):
        settings = {'alpha': 1.,
                    'beta': 1.,
                    }

        matrix = gen_matrix.Generate_Matrix()
        matrix.K = 1
        matrix.M = 1

        matrix.damping_Rayleigh(settings)

        self.assertAlmostEqual(matrix.C, 2.)

    def test_damp2(self):
        settings = {'alpha': 1.,
                    'beta': 1.,
                    }

        matrix = gen_matrix.Generate_Matrix()
        matrix.K = .5
        matrix.M = .5

        matrix.damping_Rayleigh(settings)

        self.assertAlmostEqual(matrix.C, 1.)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
