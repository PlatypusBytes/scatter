# unit test for utils

import sys
# add the src folder to the path to search for files
sys.path.append('../src/')
import unittest
import utils
import numpy as np


class TestUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_area_1(self):
        c = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]])
        a = utils.area_polygon(c)

        self.assertEqual(a, 4.)

        return

    def test_area_2(self):
        c = np.array([[0, 0, 0], [2, 2, 0], [0, 2, 0], [2, 0, 0]])
        a = utils.area_polygon(c)

        self.assertEqual(a, 4.)

        return

    def test_area_3(self):
        c = np.array([[0, 0, 0], [2, 0, 2], [0, 0, 2], [2, 0, 0]])
        a = utils.area_polygon(c)

        self.assertEqual(a, 4.)

        return

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
