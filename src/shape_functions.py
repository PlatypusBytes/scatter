import numpy as np
import sys


class ShapeFunctionVolume:
    def __init__(self, elem_type: str = "hexa8", order: int = 2) -> None:
        """
        Shape function and numerical integration functions for volume elements.
        Supported 8 and 20 node volume elements.

        Parameters
        ----------
        :param elem_type: element type (optional: default "linear")
        :param order: order of Gaussian integration  (optional: default 2)
        """
        # element type
        self.type = elem_type
        self.n = order  # number of Gauss points
        self.N = []  # shape functions
        self.dN = []  # derivative of the shape functions
        self.W = []  # weight of the Gauss integration
        self.d_jacob = []  # determinant of Jacobian
        self.dN_global = []  # derivative shape function in global coordinates
        self.B_matrix = []  # strain displacement matrix
        self.N_matrix = []  # displacement interpolation matrix

        return

    def generate(self, coordinates: list) -> None:
        r"""
        Generate shape functions, Jacobian, *B* and *N* matrix for solid element

        Parameters
        ----------
        :param coordinates: list of coordinates of nodes in one element
        """

        # natural coordinates and integration weights: Gauss integration
        coords, weights = Gauss_weights(self.n)

        for i1 in range(self.n):
            for i2 in range(self.n):
                for i3 in range(self.n):
                    X = [coords[i1], coords[i2], coords[i3]]
                    Wt = np.prod([weights[i1], weights[i2], weights[i3]])

                    # call shape functions depending on the type of element
                    if self.type == 'hexa8':
                        N, dN = shape_hexa8(X)
                    elif self.type == 'hexa20':
                        N, dN = shape_hexa20(X)

                    # add to self
                    self.N.append(N)
                    self.dN.append(dN)
                    self.W.append(Wt)

        # compute jacobian
        self.jacob(coordinates)
        # compute B matrix
        self.matrix_B()
        # compute N matrix
        self.matrix_N()

        return

    def jacob(self, xyz: list) -> None:
        """
        Computes the Jacobian and the derivative of the Jacobian

        Parameters
        ----------
        :param xyz: list of coordinates of nodes in one element
        """

        for deriv in self.dN:
            jcb = np.transpose(deriv).dot(xyz)
            self.d_jacob.append(np.linalg.det(jcb))
            self.dN_global.append(np.dot(deriv, np.linalg.inv(np.transpose(jcb))))

        return

    def matrix_B(self) -> None:
        """
        Computes B matrix (strain-displacement matrix): the derivative of the shape functions
        """

        for dnx in self.dN_global:
            B = np.zeros((6, dnx.shape[0] * dnx.shape[1]))

            for i in range(int(dnx.shape[0])):
                idx = i * 3
                B[0, idx + 0] = dnx[i, 0]
                B[1, idx + 1] = dnx[i, 1]
                B[2, idx + 2] = dnx[i, 2]
                B[3, idx + 0] = dnx[i, 1]
                B[3, idx + 1] = dnx[i, 0]
                B[4, idx + 1] = dnx[i, 2]
                B[4, idx + 2] = dnx[i, 1]
                B[5, idx + 0] = dnx[i, 2]
                B[5, idx + 2] = dnx[i, 0]

            self.B_matrix.append(B)

        return

    def matrix_N(self):
        """
        Computes N matrix: matrix with the shape functions
        """

        for nx in self.N:
            N = np.zeros((self.dN_global[0][0].shape[0], nx.shape[0] * self.dN_global[0][0].shape[0]))

            for i in range(int(nx.shape[0])):
                idx = i * 3
                N[0, idx] = nx[i]
                N[1, idx + 1] = nx[i]
                N[2, idx + 2] = nx[i]

            self.N_matrix.append(N)

        return

    def compute_stiffness(self, D: np.ndarray) -> np.ndarray:
        """
        Compute the stiffness matrix over one element

        Parameters
        ----------
        :param D: Material matrix
        :return: Stiffness matrix
        """
        Ke = np.zeros((self.B_matrix[0].shape[1], self.B_matrix[0].shape[1]))
        for i, b in enumerate(self.B_matrix):
            Ke += np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]

        return Ke

    def compute_mass(self, rho: float) -> np.ndarray:
        """
        Compute the mass matrix over one element

        Parameters
        ----------
        :param rho: Material density
        :return: Mass matrix
        """
        Me = np.zeros((self.N_matrix[0].shape[1], self.N_matrix[0].shape[1]))
        for i, N in enumerate(self.N_matrix):
            Me += np.dot(np.dot(np.transpose(N), rho), N) * self.d_jacob[i] * self.W[i]

        return Me


class ShapeFunctionSurface:
    def __init__(self, elem_type: str = "quad4", order: int = 2) -> None:
        """
        Shape function and numerical integration functions for surface elements.
        Supported 4 and 8 node quad elements.

        Parameters
        ----------
        :param elem_type: element type (optional: default "linear")
        :param order: order of Gaussian integration  (optional: default 2)
        """
        # element type
        self.type = elem_type
        self.n = order  # number of Gauss points
        self.N = []  # shape functions
        self.dN = []  # derivative of the shape functions
        self.W = []  # weight of the Gauss integration
        self.d_jacob = []  # determinant of Jacobian
        self.dN_global = []  # derivative shape function in global coordinates
        self.B_matrix = []  # strain displacement matrix
        self.N_matrix = []  # displacement interpolation matrix

        return

    def generate(self, coordinates):
        r"""
        Generate shape functions, Jacobian, *B* and *N* matrix for plane element

        Parameters
        ----------
        :param coordinates: list of coordinates of nodes in one element
        """

        # natural coordinates and integration weights: Gauss integration
        coords, weights = Gauss_weights(self.n)

        for i1 in range(self.n):
            for i2 in range(self.n):
                X = [coords[i1], coords[i2]]
                Wt = np.prod([weights[i1], weights[i2]])

                # call shape functions depending on the type of element
                if self.type == 'quad4':
                    N, dN = shape_quad4(X)
                elif self.type == 'quad8':
                    N, dN = shape_quad8(X)

                # add to self
                self.N.append(N)
                self.dN.append(dN)
                self.W.append(Wt)

        # compute jacobian
        self.jacob(coordinates)
        # compute B matrix
        self.matrix_B()
        # compute N matrix
        self.matrix_N()

        return

    def jacob(self, xy):
        """
        Computes the Jacobian and the derivative of the Jacobian

        Parameters
        ----------
        :param xy: list of coordinates of nodes in one element
        """

        for deriv in self.dN:
            jcb = np.transpose(deriv).dot(xy)
            self.d_jacob.append(np.linalg.det(jcb))
            self.dN_global.append(np.dot(deriv, np.linalg.inv(np.transpose(jcb))))

        return

    def matrix_B(self):
        """
        Computes B matrix (strain-displacement matrix): the derivative of the shape functions
        """

        for dn in self.dN_global:
            B = np.zeros((3, dn.shape[0] * dn.shape[1]))

            for i in range(int(dn.shape[0])):
                idx = i * 2
                B[0, idx + 0] = dn[i, 0]
                B[1, idx + 1] = dn[i, 1]
                B[2, idx + 0] = dn[i, 1]
                B[2, idx + 1] = dn[i, 0]

            self.B_matrix.append(B)

        return

    def matrix_N(self):
        """
        Computes N matrix: matrix with the shape functions
        """

        for nx in self.N:
            N = np.zeros((self.dN_global[0][0].shape[0], nx.shape[0] * self.dN_global[0][0].shape[0]))

            for i in range(int(nx.shape[0])):
                idx = i * 2
                N[0, idx] = nx[i]
                N[1, idx + 1] = nx[i]

            self.N_matrix.append(N)

        return

    def compute_stiffness(self, D: np.ndarray) -> np.ndarray:
        """
        Compute the stiffness matrix over one element

        Parameters
        ----------
        :param D: Material matrix
        :return: Stiffness matrix
        """
        Ke = np.zeros((self.B_matrix[0].shape[1], self.B_matrix[0].shape[1]))
        for i, b in enumerate(self.B_matrix):
            Ke += np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]

        return Ke

    def compute_mass(self, rho: float) -> np.ndarray:
        """
        Compute the mass matrix over one element

        Parameters
        ----------
        :param rho: Material density
        :return: Mass matrix
        """
        Me = np.zeros((self.N_matrix[0].shape[1], self.N_matrix[0].shape[1]))
        for i, N in enumerate(self.N_matrix):
            Me += np.dot(np.dot(np.transpose(N), rho), N) * self.d_jacob[i] * self.W[i]

        return Me

    def compute_abs_bound(self) -> np.ndarray:
        """
        Compute the absorbing boundary unitary force matrix over one element.
        Following Lysmer and Kuhlemyer.

        Parameters
        ----------
        :return: Force vector
        """

        f_abs = np.zeros((self.N_matrix[0].shape[1], self.N_matrix[0].shape[1]))
        for i, N in enumerate(self.N_matrix):
            f_abs += np.dot(np.dot(np.transpose(N), 1.), N) * self.d_jacob[i] * self.W[i]

        return f_abs


def Gauss_weights(n: int) -> [np.array, np.array]:
    r"""
    Coordinates and weights in Gauss–Legendre quadrilateral integration formulae

    Parameters
    ----------
    :param n: number of Gauss points
    :return:  points, weights
    """

    if n == 1:
        x = [0.]
        w = [2.]
    elif n == 2:
        x = [-np.sqrt(1. / 3.), np.sqrt(1. / 3.)]
        w = [1., 1.]
    elif n == 3:
        x = [-np.sqrt(3. / 5.), 0, np.sqrt(3. / 5.)]
        w = [5. / 9., 8. / 9., 5. / 9.]
    else:
        sys.exit("ERROR: integration order not supported")

    return np.array(x), np.array(w)


def shape_quad4(xy):
    r"""
    Shape functions 4 node quadrilateral element.
    Node numbering follow:
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

    Parameters
    ----------
    :param xy: list with node coordinate
    :return: Shape function and derivative of shape functions
    """

    N = np.zeros((4, 1))
    dN = np.zeros((4, 2))

    u = xy[0]
    v = xy[1]

    # shape functions
    N[0] = 1. / 4. * (1 - u) * (1 - v)
    N[1] = 1. / 4. * (1 + u) * (1 - v)
    N[2] = 1. / 4. * (1 + u) * (1 + v)
    N[3] = 1. / 4. * (1 - u) * (1 + v)

    # derivative in u
    dN[0, 0] = -(1 - v) / 4
    dN[1, 0] = (1 - v) / 4
    dN[2, 0] = (v + 1) / 4
    dN[3, 0] = -(v + 1) / 4

    # derivative in v
    dN[0, 1] = -(1 - u) / 4
    dN[1, 1] = -(u + 1) / 4
    dN[2, 1] = (u + 1) / 4
    dN[3, 1] = (1 - u) / 4

    return N, dN


def shape_quad8(xy):
    r"""
    Shape functions 8 node quadrilateral element.
    Node numbering follow:
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

    Parameters
    ----------
    :param xy: list with node coordinate
    :return: Shape function and derivative of shape functions
    """

    N = np.zeros((8, 1))
    dN = np.zeros((8, 2))

    u = xy[0]
    v = xy[1]

    # shape functions
    N[0] = 1. / 4. * (1 - u) * (1 - v)
    N[1] = 1. / 4. * (1 + u) * (1 - v)
    N[2] = 1. / 4. * (1 + u) * (1 + v)
    N[3] = 1. / 4. * (1 - u) * (1 + v)
    N[4] = 1. / 2. * (1 - u ** 2) * (1 - v)
    N[5] = 1. / 2. * (1 + u) * (1 - v ** 2)
    N[6] = 1. / 2. * (1 - u ** 2) * (1 + v)
    N[7] = 1. / 2. * (1 - u) * (1 - v ** 2)

    # derivative in u
    dN[0, 0] = -(1 - v) / 4
    dN[1, 0] = (1 - v) / 4
    dN[2, 0] = (v + 1) / 4
    dN[3, 0] = -(v + 1) / 4
    dN[4, 0] = -u * (1 - v)
    dN[5, 0] = (1 - v ** 2) / 2
    dN[6, 0] = -u * (1 + v)
    dN[7, 0] = -(1 - v ** 2) / 2

    # derivative in v
    dN[0, 1] = -(1 - u) / 4
    dN[1, 1] = -(u + 1) / 4
    dN[2, 1] = (u + 1) / 4
    dN[3, 1] = (1 - u) / 4
    dN[4, 1] = -(1 - u ** 2) / 2
    dN[5, 1] = -v * (1 + u)
    dN[6, 1] = (1 - u ** 2) / 2
    dN[7, 1] = -v * (1 - u)

    return N, dN


def shape_hexa8(xyz: list) -> [np.ndarray, np.ndarray]:
    r"""
    Shape functions Volume 8 node element.
    Node numbering follow:
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

    Parameters
    ----------
    :param xyz: list with node coordinate
    :return: Shape function and derivative of shape functions
    """

    N = np.zeros((8, 1))
    dN = np.zeros((8, 3))

    u = xyz[0]
    v = xyz[1]
    w = xyz[2]

    # shape functions
    N[0] = 1. / 8. * (1 - u) * (1 - v) * (1 - w)
    N[1] = 1. / 8. * (1 + u) * (1 - v) * (1 - w)
    N[2] = 1. / 8. * (1 + u) * (1 + v) * (1 - w)
    N[3] = 1. / 8. * (1 - u) * (1 + v) * (1 - w)
    N[4] = 1. / 8. * (1 - u) * (1 - v) * (1 + w)
    N[5] = 1. / 8. * (1 + u) * (1 - v) * (1 + w)
    N[6] = 1. / 8. * (1 + u) * (1 + v) * (1 + w)
    N[7] = 1. / 8. * (1 - u) * (1 + v) * (1 + w)

    # derivative in u
    dN[0, 0] = -((1 - v) * (1 - w)) / 8
    dN[1, 0] = ((1 - v) * (1 - w)) / 8
    dN[2, 0] = ((v + 1) * (1 - w)) / 8
    dN[3, 0] = -((v + 1) * (1 - w)) / 8
    dN[4, 0] = -((1 - v) * (w + 1)) / 8
    dN[5, 0] = ((1 - v) * (w + 1)) / 8
    dN[6, 0] = ((v + 1) * (w + 1)) / 8
    dN[7, 0] = -((v + 1) * (w + 1)) / 8

    # derivative in v
    dN[0, 1] = -((1 - u) * (1 - w)) / 8
    dN[1, 1] = -((u + 1) * (1 - w)) / 8
    dN[2, 1] = ((u + 1) * (1 - w)) / 8
    dN[3, 1] = ((1 - u) * (1 - w)) / 8
    dN[4, 1] = -((1 - u) * (w + 1)) / 8
    dN[5, 1] = -((u + 1) * (w + 1)) / 8
    dN[6, 1] = ((u + 1) * (w + 1)) / 8
    dN[7, 1] = ((1 - u) * (w + 1)) / 8

    # derivative in w
    dN[0, 2] = -((1 - u) * (1 - v)) / 8
    dN[1, 2] = -((u + 1) * (1 - v)) / 8
    dN[2, 2] = -((u + 1) * (v + 1)) / 8
    dN[3, 2] = -((1 - u) * (v + 1)) / 8
    dN[4, 2] = ((1 - u) * (1 - v)) / 8
    dN[5, 2] = ((u + 1) * (1 - v)) / 8
    dN[6, 2] = ((u + 1) * (v + 1)) / 8
    dN[7, 2] = ((1 - u) * (v + 1)) / 8

    return N, dN


def shape_hexa20(xyz):
    r"""
    Shape functions Volume 20 node element.
    Node numbering follow:
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

    Parameters
    ----------
    :param xyz: list with node coordinate
    :return: Shape function and derivative of shape functions
    """

    N = np.zeros((20, 1))
    dN = np.zeros((20, 3))

    u = xyz[0]
    v = xyz[1]
    w = xyz[2]

    # shape functions
    N[0] = 1. / 8. * (1 - u) * (1 - v) * (1 - w) * (-u - v - w - 2)
    N[1] = 1. / 8. * (1 + u) * (1 - v) * (1 - w) * (u - v - w - 2)
    N[2] = 1. / 8. * (1 + u) * (1 + v) * (1 - w) * (u + v - w - 2)
    N[3] = 1. / 8. * (1 - u) * (1 + v) * (1 - w) * (-u + v - w - 2)
    N[4] = 1. / 8. * (1 - u) * (1 - v) * (1 + w) * (-u - v + w - 2)
    N[5] = 1. / 8. * (1 + u) * (1 - v) * (1 + w) * (u - v + w - 2)
    N[6] = 1. / 8. * (1 + u) * (1 + v) * (1 + w) * (u + v + w - 2)
    N[7] = 1. / 8. * (1 - u) * (1 + v) * (1 + w) * (-u + v + w - 2)
    N[8] = 1. / 4. * (1 - u ** 2) * (1 - v) * (1 - w)
    N[9] = 1. / 4. * (1 - u) * (1 - v ** 2) * (1 - w)
    N[10] = 1. / 4. * (1 - u) * (1 - v) * (1 - w ** 2)
    N[11] = 1. / 4. * (1 + u) * (1 - v ** 2) * (1 - w)
    N[12] = 1. / 4. * (1 + u) * (1 - v) * (1 - w ** 2)
    N[13] = 1. / 4. * (1 - u ** 2) * (1 + v) * (1 - w)
    N[14] = 1. / 4. * (1 + u) * (1 + v) * (1 - w ** 2)
    N[15] = 1. / 4. * (1 - u) * (1 + v) * (1 - w ** 2)
    N[16] = 1. / 4. * (1 - u ** 2) * (1 - v) * (1 + w)
    N[17] = 1. / 4. * (1 - u) * (1 - v ** 2) * (1 + w)
    N[18] = 1. / 4. * (1 + u) * (1 - v ** 2) * (1 + w)
    N[19] = 1. / 4. * (1 - u ** 2) * (1 + v) * (1 + w)

    # derivative in u
    dN[0, 0] = -((1 - v) * (1 - w) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
    dN[1, 0] = ((1 - v) * (1 - w) * (-w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (1 - w)) / 8
    dN[2, 0] = ((v + 1) * (1 - w) * (-w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (1 - w)) / 8
    dN[3, 0] = -((v + 1) * (1 - w) * (-w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (1 - w)) / 8
    dN[4, 0] = -((1 - v) * (w + 1) * (w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (w + 1)) / 8
    dN[5, 0] = ((1 - v) * (w + 1) * (w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (w + 1)) / 8
    dN[6, 0] = ((v + 1) * (w + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
    dN[7, 0] = -((v + 1) * (w + 1) * (w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (w + 1)) / 8
    dN[8, 0] = -(u * (1 - v) * (1 - w)) / 2
    dN[9, 0] = -((1 - v ** 2) * (1 - w)) / 4
    dN[10, 0] = -((1 - v) * (1 - w ** 2)) / 4
    dN[11, 0] = ((1 - v ** 2) * (1 - w)) / 4
    dN[12, 0] = ((1 - v) * (1 - w ** 2)) / 4
    dN[13, 0] = -(u * (v + 1) * (1 - w)) / 2
    dN[14, 0] = ((v + 1) * (1 - w ** 2)) / 4
    dN[15, 0] = -((v + 1) * (1 - w ** 2)) / 4
    dN[16, 0] = -(u * (1 - v) * (w + 1)) / 2
    dN[17, 0] = -((1 - v ** 2) * (w + 1)) / 4
    dN[18, 0] = ((1 - v ** 2) * (w + 1)) / 4
    dN[19, 0] = -(u * (v + 1) * (w + 1)) / 2

    # derivative in v
    dN[0, 1] = -((1 - u) * (1 - w) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
    dN[1, 1] = -((u + 1) * (1 - w) * (-w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (1 - w)) / 8
    dN[2, 1] = ((u + 1) * (1 - w) * (-w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (1 - w)) / 8
    dN[3, 1] = ((1 - u) * (1 - w) * (-w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (1 - w)) / 8
    dN[4, 1] = -((1 - u) * (w + 1) * (w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (w + 1)) / 8
    dN[5, 1] = -((u + 1) * (w + 1) * (w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (w + 1)) / 8
    dN[6, 1] = ((u + 1) * (w + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
    dN[7, 1] = ((1 - u) * (w + 1) * (w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (w + 1)) / 8
    dN[8, 1] = -((1 - u ** 2) * (1 - w)) / 4
    dN[9, 1] = -((1 - u) * v * (1 - w)) / 2
    dN[10, 1] = -((1 - u) * (1 - w ** 2)) / 4
    dN[11, 1] = -((u + 1) * v * (1 - w)) / 2
    dN[12, 1] = -((u + 1) * (1 - w ** 2)) / 4
    dN[13, 1] = ((1 - u ** 2) * (1 - w)) / 4
    dN[14, 1] = ((u + 1) * (1 - w ** 2)) / 4
    dN[15, 1] = ((1 - u) * (1 - w ** 2)) / 4
    dN[16, 1] = -((1 - u ** 2) * (w + 1)) / 4
    dN[17, 1] = -((1 - u) * v * (w + 1)) / 2
    dN[18, 1] = -((u + 1) * v * (w + 1)) / 2
    dN[19, 1] = ((1 - u ** 2) * (w + 1)) / 4

    # derivative in w
    dN[0, 2] = -((1 - u) * (1 - v) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
    dN[1, 2] = -((u + 1) * (1 - v) * (-w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (1 - w)) / 8
    dN[2, 2] = -((u + 1) * (v + 1) * (-w + v + u - 2)) / 8 - ((u + 1) * (v + 1) * (1 - w)) / 8
    dN[3, 2] = -((1 - u) * (v + 1) * (-w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (1 - w)) / 8
    dN[4, 2] = ((1 - u) * (1 - v) * (w - v - u - 2)) / 8 + ((1 - u) * (1 - v) * (w + 1)) / 8
    dN[5, 2] = ((u + 1) * (1 - v) * (w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (w + 1)) / 8
    dN[6, 2] = ((u + 1) * (v + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
    dN[7, 2] = ((1 - u) * (v + 1) * (w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (w + 1)) / 8
    dN[8, 2] = -((1 - u ** 2) * (1 - v)) / 4
    dN[9, 2] = -((1 - u) * (1 - v ** 2)) / 4
    dN[10, 2] = -((1 - u) * (1 - v) * w) / 2
    dN[11, 2] = -((u + 1) * (1 - v ** 2)) / 4
    dN[12, 2] = -((u + 1) * (1 - v) * w) / 2
    dN[13, 2] = -((1 - u ** 2) * (v + 1)) / 4
    dN[14, 2] = -((u + 1) * (v + 1) * w) / 2
    dN[15, 2] = -((1 - u) * (v + 1) * w) / 2
    dN[16, 2] = ((1 - u ** 2) * (1 - v)) / 4
    dN[17, 2] = ((1 - u) * (1 - v ** 2)) / 4
    dN[18, 2] = ((u + 1) * (1 - v ** 2)) / 4
    dN[19, 2] = ((1 - u ** 2) * (v + 1)) / 4

    return N, dN
