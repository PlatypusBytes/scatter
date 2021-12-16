import numpy as np


class HexEight:
    r"""
    8-node brick element.

    Node numbering follows:

           v
    3----------2
    |\     ^   |\
    | \    |   | \
    |  \   |   |  \
    |   7------+---6
    |   |  +-- |-- | -> u
    0---+---\--1   |
     \  |    \  \  |
      \ |     \  \ |
       \|      w  \|
        4----------5
    """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((8, 1))
        self.dN = np.zeros((8, 3))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = np.array([[0, 1, 2, 3], [0, 1, 4, 5], [4, 5, 6, 7], [2, 3, 6, 7], [0, 3, 4, 7], [1, 2, 5, 6]])

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 8

    @property
    def n_boundary_nodes(self):
        return 4

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xyz: list):
        r"""
        Shape functions Volume 8 node element.

        Parameters
        ----------
        :param xyz: list with node coordinate
        """
        u = xyz[0]
        v = xyz[1]
        w = xyz[2]

        # shape functions
        self.N[0] = 1. / 8. * (1 - u) * (1 - v) * (1 - w)
        self.N[1] = 1. / 8. * (1 + u) * (1 - v) * (1 - w)
        self.N[2] = 1. / 8. * (1 + u) * (1 + v) * (1 - w)
        self.N[3] = 1. / 8. * (1 - u) * (1 + v) * (1 - w)
        self.N[4] = 1. / 8. * (1 - u) * (1 - v) * (1 + w)
        self.N[5] = 1. / 8. * (1 + u) * (1 - v) * (1 + w)
        self.N[6] = 1. / 8. * (1 + u) * (1 + v) * (1 + w)
        self.N[7] = 1. / 8. * (1 - u) * (1 + v) * (1 + w)

        # derivative in u
        self.dN[0, 0] = -((1 - v) * (1 - w)) / 8
        self.dN[1, 0] = ((1 - v) * (1 - w)) / 8
        self.dN[2, 0] = ((v + 1) * (1 - w)) / 8
        self.dN[3, 0] = -((v + 1) * (1 - w)) / 8
        self.dN[4, 0] = -((1 - v) * (w + 1)) / 8
        self.dN[5, 0] = ((1 - v) * (w + 1)) / 8
        self.dN[6, 0] = ((v + 1) * (w + 1)) / 8
        self.dN[7, 0] = -((v + 1) * (w + 1)) / 8

        # derivative in v
        self.dN[0, 1] = -((1 - u) * (1 - w)) / 8
        self.dN[1, 1] = -((u + 1) * (1 - w)) / 8
        self.dN[2, 1] = ((u + 1) * (1 - w)) / 8
        self.dN[3, 1] = ((1 - u) * (1 - w)) / 8
        self.dN[4, 1] = -((1 - u) * (w + 1)) / 8
        self.dN[5, 1] = -((u + 1) * (w + 1)) / 8
        self.dN[6, 1] = ((u + 1) * (w + 1)) / 8
        self.dN[7, 1] = ((1 - u) * (w + 1)) / 8

        # derivative in w
        self.dN[0, 2] = -((1 - u) * (1 - v)) / 8
        self.dN[1, 2] = -((u + 1) * (1 - v)) / 8
        self.dN[2, 2] = -((u + 1) * (v + 1)) / 8
        self.dN[3, 2] = -((1 - u) * (v + 1)) / 8
        self.dN[4, 2] = ((1 - u) * (1 - v)) / 8
        self.dN[5, 2] = ((u + 1) * (1 - v)) / 8
        self.dN[6, 2] = ((u + 1) * (v + 1)) / 8
        self.dN[7, 2] = ((1 - u) * (v + 1)) / 8

        return


class HexTwenty:
    r"""
    20-node brick element.

    Node numbering follows:

           v
    3----13----2
    |\     ^   |\
    | 15   |   | 14
    9  \   |   11 \
    |   7----19+---6
    |   |  +-- |-- | -> u
    0---+-8-\--1   |
     \  17   \  \  18
     10 |     \  12|
       \|      w  \|
        4----16----5
    """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((20, 1))
        self.dN = np.zeros((20, 3))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []  # ToDo

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 20

    @property
    def n_boundary_nodes(self):
        return 8

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xyz: list):
        r"""
        Shape functions Volume 8 node element.

        Parameters
        ----------
        :param xyz: list with node coordinate
        """
        u = xyz[0]
        v = xyz[1]
        w = xyz[2]

        # shape functions
        self.N[0] = 1. / 8. * (1 - u) * (1 - v) * (1 - w) * (-u - v - w - 2)
        self.N[1] = 1. / 8. * (1 + u) * (1 - v) * (1 - w) * (u - v - w - 2)
        self.N[2] = 1. / 8. * (1 + u) * (1 + v) * (1 - w) * (u + v - w - 2)
        self.N[3] = 1. / 8. * (1 - u) * (1 + v) * (1 - w) * (-u + v - w - 2)
        self.N[4] = 1. / 8. * (1 - u) * (1 - v) * (1 + w) * (-u - v + w - 2)
        self.N[5] = 1. / 8. * (1 + u) * (1 - v) * (1 + w) * (u - v + w - 2)
        self.N[6] = 1. / 8. * (1 + u) * (1 + v) * (1 + w) * (u + v + w - 2)
        self.N[7] = 1. / 8. * (1 - u) * (1 + v) * (1 + w) * (-u + v + w - 2)
        self.N[8] = 1. / 4. * (1 - u ** 2) * (1 - v) * (1 - w)
        self.N[9] = 1. / 4. * (1 - u) * (1 - v ** 2) * (1 - w)
        self.N[10] = 1. / 4. * (1 - u) * (1 - v) * (1 - w ** 2)
        self.N[11] = 1. / 4. * (1 + u) * (1 - v ** 2) * (1 - w)
        self.N[12] = 1. / 4. * (1 + u) * (1 - v) * (1 - w ** 2)
        self.N[13] = 1. / 4. * (1 - u ** 2) * (1 + v) * (1 - w)
        self.N[14] = 1. / 4. * (1 + u) * (1 + v) * (1 - w ** 2)
        self.N[15] = 1. / 4. * (1 - u) * (1 + v) * (1 - w ** 2)
        self.N[16] = 1. / 4. * (1 - u ** 2) * (1 - v) * (1 + w)
        self.N[17] = 1. / 4. * (1 - u) * (1 - v ** 2) * (1 + w)
        self.N[18] = 1. / 4. * (1 + u) * (1 - v ** 2) * (1 + w)
        self.N[19] = 1. / 4. * (1 - u ** 2) * (1 + v) * (1 + w)

        # derivative in u
        self.dN[0, 0] = -((1 - v) * (1 - w) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
        self.dN[1, 0] = ((1 - v) * (1 - w) * (-w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (1 - w)) / 8
        self.dN[2, 0] = ((v + 1) * (1 - w) * (-w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (1 - w)) / 8
        self.dN[3, 0] = -((v + 1) * (1 - w) * (-w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (1 - w)) / 8
        self.dN[4, 0] = -((1 - v) * (w + 1) * (w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (w + 1)) / 8
        self.dN[5, 0] = ((1 - v) * (w + 1) * (w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (w + 1)) / 8
        self.dN[6, 0] = ((v + 1) * (w + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
        self.dN[7, 0] = -((v + 1) * (w + 1) * (w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (w + 1)) / 8
        self.dN[8, 0] = -(u * (1 - v) * (1 - w)) / 2
        self.dN[9, 0] = -((1 - v ** 2) * (1 - w)) / 4
        self.dN[10, 0] = -((1 - v) * (1 - w ** 2)) / 4
        self.dN[11, 0] = ((1 - v ** 2) * (1 - w)) / 4
        self.dN[12, 0] = ((1 - v) * (1 - w ** 2)) / 4
        self.dN[13, 0] = -(u * (v + 1) * (1 - w)) / 2
        self.dN[14, 0] = ((v + 1) * (1 - w ** 2)) / 4
        self.dN[15, 0] = -((v + 1) * (1 - w ** 2)) / 4
        self.dN[16, 0] = -(u * (1 - v) * (w + 1)) / 2
        self.dN[17, 0] = -((1 - v ** 2) * (w + 1)) / 4
        self.dN[18, 0] = ((1 - v ** 2) * (w + 1)) / 4
        self.dN[19, 0] = -(u * (v + 1) * (w + 1)) / 2

        # derivative in v
        self.dN[0, 1] = -((1 - u) * (1 - w) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
        self.dN[1, 1] = -((u + 1) * (1 - w) * (-w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (1 - w)) / 8
        self.dN[2, 1] = ((u + 1) * (1 - w) * (-w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (1 - w)) / 8
        self.dN[3, 1] = ((1 - u) * (1 - w) * (-w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (1 - w)) / 8
        self.dN[4, 1] = -((1 - u) * (w + 1) * (w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (w + 1)) / 8
        self.dN[5, 1] = -((u + 1) * (w + 1) * (w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (w + 1)) / 8
        self.dN[6, 1] = ((u + 1) * (w + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
        self.dN[7, 1] = ((1 - u) * (w + 1) * (w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (w + 1)) / 8
        self.dN[8, 1] = -((1 - u ** 2) * (1 - w)) / 4
        self.dN[9, 1] = -((1 - u) * v * (1 - w)) / 2
        self.dN[10, 1] = -((1 - u) * (1 - w ** 2)) / 4
        self.dN[11, 1] = -((u + 1) * v * (1 - w)) / 2
        self.dN[12, 1] = -((u + 1) * (1 - w ** 2)) / 4
        self.dN[13, 1] = ((1 - u ** 2) * (1 - w)) / 4
        self.dN[14, 1] = ((u + 1) * (1 - w ** 2)) / 4
        self.dN[15, 1] = ((1 - u) * (1 - w ** 2)) / 4
        self.dN[16, 1] = -((1 - u ** 2) * (w + 1)) / 4
        self.dN[17, 1] = -((1 - u) * v * (w + 1)) / 2
        self.dN[18, 1] = -((u + 1) * v * (w + 1)) / 2
        self.dN[19, 1] = ((1 - u ** 2) * (w + 1)) / 4

        # derivative in w
        self.dN[0, 2] = -((1 - u) * (1 - v) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
        self.dN[1, 2] = -((u + 1) * (1 - v) * (-w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (1 - w)) / 8
        self.dN[2, 2] = -((u + 1) * (v + 1) * (-w + v + u - 2)) / 8 - ((u + 1) * (v + 1) * (1 - w)) / 8
        self.dN[3, 2] = -((1 - u) * (v + 1) * (-w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (1 - w)) / 8
        self.dN[4, 2] = ((1 - u) * (1 - v) * (w - v - u - 2)) / 8 + ((1 - u) * (1 - v) * (w + 1)) / 8
        self.dN[5, 2] = ((u + 1) * (1 - v) * (w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (w + 1)) / 8
        self.dN[6, 2] = ((u + 1) * (v + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
        self.dN[7, 2] = ((1 - u) * (v + 1) * (w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (w + 1)) / 8
        self.dN[8, 2] = -((1 - u ** 2) * (1 - v)) / 4
        self.dN[9, 2] = -((1 - u) * (1 - v ** 2)) / 4
        self.dN[10, 2] = -((1 - u) * (1 - v) * w) / 2
        self.dN[11, 2] = -((u + 1) * (1 - v ** 2)) / 4
        self.dN[12, 2] = -((u + 1) * (1 - v) * w) / 2
        self.dN[13, 2] = -((1 - u ** 2) * (v + 1)) / 4
        self.dN[14, 2] = -((u + 1) * (v + 1) * w) / 2
        self.dN[15, 2] = -((1 - u) * (v + 1) * w) / 2
        self.dN[16, 2] = ((1 - u ** 2) * (1 - v)) / 4
        self.dN[17, 2] = ((1 - u) * (1 - v ** 2)) / 4
        self.dN[18, 2] = ((u + 1) * (1 - v ** 2)) / 4
        self.dN[19, 2] = ((1 - u ** 2) * (v + 1)) / 4
        return


class QuadFour:
    r"""
    4-node quad element.

    Node numbering follows:

          v
          ^
          |
    3-----------2
    |     |     |
    |     |     |
    |     +---- | --> u
    |           |
    |           |
    0-----------1
    """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((4, 1))
        self.dN = np.zeros((4, 2))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 4

    @property
    def n_boundary_nodes(self):
        return 2

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xy: list):
        r"""
        Shape functions 4 node quadrilateral element.

        Parameters
        ----------
        :param xy: list with node coordinate
        :return: Shape function and derivative of shape functions
        """

        u = xy[0]
        v = xy[1]

        # shape functions
        self.N[0] = 1. / 4. * (1 - u) * (1 - v)
        self.N[1] = 1. / 4. * (1 + u) * (1 - v)
        self.N[2] = 1. / 4. * (1 + u) * (1 + v)
        self.N[3] = 1. / 4. * (1 - u) * (1 + v)

        # derivative in u
        self.dN[0, 0] = -(1 - v) / 4
        self.dN[1, 0] = (1 - v) / 4
        self.dN[2, 0] = (v + 1) / 4
        self.dN[3, 0] = -(v + 1) / 4

        # derivative in v
        self.dN[0, 1] = -(1 - u) / 4
        self.dN[1, 1] = -(u + 1) / 4
        self.dN[2, 1] = (u + 1) / 4
        self.dN[3, 1] = (1 - u) / 4
        return


class QuadEight:
    r"""
    8-node quad element.

    Node numbering follows:

          v
          ^
          |
    3-----6-----2
    |     |     |
    |     |     |
    7     +---- 5  --> u
    |           |
    |           |
    0-----4-----1
    """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((8, 1))
        self.dN = np.zeros((8, 2))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 8

    @property
    def n_boundary_nodes(self):
        return 4

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xy: list):
        r"""
        Shape functions 8 node quadrilateral element.

        Parameters
        ----------
        :param xy: list with node coordinate
        :return: Shape function and derivative of shape functions
        """

        u = xy[0]
        v = xy[1]

        # shape functions
        self.N[0] = 1. / 4. * (1 - u) * (1 - v)
        self.N[1] = 1. / 4. * (1 + u) * (1 - v)
        self.N[2] = 1. / 4. * (1 + u) * (1 + v)
        self.N[3] = 1. / 4. * (1 - u) * (1 + v)
        self.N[4] = 1. / 2. * (1 - u ** 2) * (1 - v)
        self.N[5] = 1. / 2. * (1 + u) * (1 - v ** 2)
        self.N[6] = 1. / 2. * (1 - u ** 2) * (1 + v)
        self.N[7] = 1. / 2. * (1 - u) * (1 - v ** 2)

        # derivative in u
        self.dN[0, 0] = -(1 - v) / 4
        self.dN[1, 0] = (1 - v) / 4
        self.dN[2, 0] = (v + 1) / 4
        self.dN[3, 0] = -(v + 1) / 4
        self.dN[4, 0] = -u * (1 - v)
        self.dN[5, 0] = (1 - v ** 2) / 2
        self.dN[6, 0] = -u * (1 + v)
        self.dN[7, 0] = -(1 - v ** 2) / 2

        # derivative in v
        self.dN[0, 1] = -(1 - u) / 4
        self.dN[1, 1] = -(u + 1) / 4
        self.dN[2, 1] = (u + 1) / 4
        self.dN[3, 1] = (1 - u) / 4
        self.dN[4, 1] = -(1 - u ** 2) / 2
        self.dN[5, 1] = -v * (1 + u)
        self.dN[6, 1] = (1 - u ** 2) / 2
        self.dN[7, 1] = -v * (1 - u)
        return

class TriThree:
    r"""
    3-node triangular element.

    Node numbering follows:


    v
    ^
    |
    2
    |`\
    |  `\
    |    `\
    |      `\
    |        `\
    0----------1 --> u

    """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((3, 1))
        self.dN = np.zeros((3, 2))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []

    @property
    def surfaces(self):
        return self.__surfaces

    # @property
    # def max_element_connections(self):
    #     return 4
    #
    # @property
    # def n_boundary_nodes(self):
    #     return 2

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xy: list):
        r"""
        Shape functions 4 node quadrilateral element.

        Parameters
        ----------
        :param xy: list with node coordinate
        :return: Shape function and derivative of shape functions
        """

        u = xy[0]
        v = xy[1]

        # shape functions
        self.N[0] = (1 - u - v)
        self.N[1] = u
        self.N[2] = v

        # derivative in u
        self.dN[0, 0] = 0
        self.dN[1, 0] = 1
        self.dN[2, 0] = 0

        # derivative in v
        self.dN[0, 1] = 0
        self.dN[1, 1] = 0
        self.dN[2, 1] = 1


class TriSix:
    r"""
       3-node triangular element.

       Node numbering follows:


       v
       ^
       |
       2
       |`\
       |  `\
       5    `4
       |      `\
       |        `\
       0-----3----1 --> u

       """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((6, 1))
        self.dN = np.zeros((6, 2))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 4

    @property
    def n_boundary_nodes(self):
        return 2

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xy: list):
        r"""
        Shape functions 4 node quadrilateral element.

        Parameters
        ----------
        :param xy: list with node coordinate
        :return: Shape function and derivative of shape functions
        """

        u = xy[0]
        v = xy[1]

        # shape functions
        self.N[0] = (2*(1-u-v)-1)*(1-u-v)
        self.N[1] = (2*u-1) * u
        self.N[2] = (2*v-1) * v
        self.N[3] = 4*(1-u-v) * u
        self.N[4] = 4 * u * v
        self.N[5] = 4*(1-u-v) * v

        # derivative in u
        self.dN[0, 0] = 1 -4*(-v-u+1)
        self.dN[1, 0] = 4 * u-1
        self.dN[2, 0] = 0
        self.dN[3, 0] = 4*(-v-u+1)-4*u
        self.dN[4, 0] = 0
        self.dN[5, 0] = 0

        # derivative in v
        self.dN[0, 1] = 1 - 4 * (-v-u+1)
        self.dN[1, 1] = 0
        self.dN[2, 1] = 4 * v-1
        self.dN[3, 1] = 0
        self.dN[3, 1] = 0
        self.dN[3, 1] = 4*(-v-u+1)-4*v



class TetraFour:
    r"""
       4-node tetrahedron element.

       Node numbering follows:


                           v
                         .
                       ,/
                      /
                   2
                 ,/|`\
               ,/  |  `\
             ,/    '.   `\
           ,/       |     `\
         ,/         |       `\
        0-----------'.--------1 --> u
         `\.         |      ,/
            `\.      |    ,/
               `\.   '. ,/
                  `\. |/
                     `3
                        `\.
                           ` w

       """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((4, 1))
        self.dN = np.zeros((4, 2))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 4

    @property
    def n_boundary_nodes(self):
        return 2

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xy: list):
        r"""
        Shape functions 4 node quadrilateral element.

        Parameters
        ----------
        :param xy: list with node coordinate
        :return: Shape function and derivative of shape functions
        """

        u = xy[0]
        v = xy[1]

        # shape functions
        self.N[0] = 1. / 4. * (1 - u) * (1 - v)
        self.N[1] = 1. / 4. * (1 + u) * (1 - v)
        self.N[2] = 1. / 4. * (1 + u) * (1 + v)
        self.N[3] = 1. / 4. * (1 - u) * (1 + v)

        # derivative in u
        self.dN[0, 0] = -(1 - v) / 4
        self.dN[1, 0] = (1 - v) / 4
        self.dN[2, 0] = (v + 1) / 4
        self.dN[3, 0] = -(v + 1) / 4

        # derivative in v
        self.dN[0, 1] = -(1 - u) / 4
        self.dN[1, 1] = -(u + 1) / 4
        self.dN[2, 1] = (u + 1) / 4
        self.dN[3, 1] = (1 - u) / 4
        return



class TetraTen:
    r"""
       4-node tetrahedron element.

       Node numbering follows:


                           v
                         .
                       ,/
                      /
                   2
                 ,/|`\
               ,/  |  `\
             ,6    '.   `5
           ,/       8     `\
         ,/         |       `\
        0-----------'.--------1 --> u
         `\.         |      ,/
            `\.      |    ,9
               `7.   '. ,/
                  `\. |/
                     `3
                        `\.
                           ` w

   """

    def __init__(self):
        self.__surfaces = None
        self.N = np.zeros((4, 1))
        self.dN = np.zeros((4, 2))
        return

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = []

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 4

    @property
    def n_boundary_nodes(self):
        return 2

    @property
    def is_quadratic(self):
        return False

    def shape_functions(self, xy: list):
        r"""
        Shape functions 4 node quadrilateral element.

        Parameters
        ----------
        :param xy: list with node coordinate
        :return: Shape function and derivative of shape functions
        """

        u = xy[0]
        v = xy[1]

        # shape functions
        self.N[0] = 1. / 4. * (1 - u) * (1 - v)
        self.N[1] = 1. / 4. * (1 + u) * (1 - v)
        self.N[2] = 1. / 4. * (1 + u) * (1 + v)
        self.N[3] = 1. / 4. * (1 - u) * (1 + v)

        # derivative in u
        self.dN[0, 0] = -(1 - v) / 4
        self.dN[1, 0] = (1 - v) / 4
        self.dN[2, 0] = (v + 1) / 4
        self.dN[3, 0] = -(v + 1) / 4

        # derivative in v
        self.dN[0, 1] = -(1 - u) / 4
        self.dN[1, 1] = -(u + 1) / 4
        self.dN[2, 1] = (u + 1) / 4
        self.dN[3, 1] = (1 - u) / 4
        return