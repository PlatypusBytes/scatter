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
        self.H_matrix = []  # displacement interpolation matrix

        return

    def generate(self) -> None:
        r"""
        Generate shape functions for solid
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

        return

    def jacob(self, xyz):

        for deriv in self.dN:
            jcb = np.transpose(deriv).dot(xyz)
            self.d_jacob.append(np.linalg.det(jcb))
            self.dN_global.append(np.dot(deriv, np.linalg.inv(np.transpose(jcb))))

        return

    def matrix_B(self):

        for dnx in self.dN_global:
            B = np.zeros((6, dnx.shape[0] * dnx.shape[1]))

            for i in range(int(dnx.shape[0])):
                N = i * 3
                B[0, N + 0] = dnx[i, 0]  # E_xx
                B[1, N + 1] = dnx[i, 1]  # E_yy
                B[2, N + 2] = dnx[i, 2]  # E_zz
                B[3, N + 0] = dnx[i, 1]  # 2 * E_xy
                B[3, N + 1] = dnx[i, 0]
                B[4, N + 1] = dnx[i, 2]  # 2 * E_yz
                B[4, N + 2] = dnx[i, 1]
                B[5, N + 0] = dnx[i, 2]  # 2 * E_xz
                B[5, N + 2] = dnx[i, 0]

            self.B_matrix.append(B)

        return

    def int_H(self):

        for nx in self.N:
            H = np.zeros((self.dN_global[0][0].shape[0], nx.shape[0] * self.dN_global[0][0].shape[0]))

            for i in range(int(nx.shape[0])):
                N = i * 3
                H[0, N] = nx[i]
                H[1, N + 1] = nx[i]
                H[2, N + 2] = nx[i]

            self.H_matrix.append(H)

        return

    def compute_stiffness(self, D):

        Ke = np.zeros((self.B_matrix[0].shape[1], self.B_matrix[0].shape[1]))
        for i, b in enumerate(self.B_matrix):
            Ke += np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]

        return Ke

    def compute_mass(self, rho):

        Me = np.zeros((self.H_matrix[0].shape[1], self.H_matrix[0].shape[1]))
        for i, H in enumerate(self.H_matrix):
            Me += np.dot(np.dot(np.transpose(H), rho), H) * self.d_jacob[i] * self.W[i]

        return Me


class ShapeFunctionSurface:
    def __init__(self, elem_type, order):

        if elem_type == "boundary":
            self.type = "boundary"

        # order Gauss integration
        self.n = order
        # shape functions
        self.N = []
        # derivative of the shape functions
        self.dN = []
        # weight of the Gauss integration
        self.W = []
        # determinant Jacobian
        self.d_jacob = []
        # derivative shape function in global coordinates
        self.DNX = []
        # strain displacement matrix
        self.B = []
        # displacement interpolation matrix
        self.H = []

        return

    def generate(self):
        """" generate shape functions for plane elements """

        # natural coordinates Gauss integration points
        coords, weights = Gauss_weights(self.n)

        for i1 in range(self.n):
            for i2 in range(self.n):
                X = [coords[i1], coords[i2]]
                Wt = np.prod([weights[i1], weights[i2]])

                N, dN = shape4(X)
                # add to self
                self.N.append(N)
                self.dN.append(dN)
                self.W.append(Wt)

        return

    def jacob(self, xyz):

        for deriv in self.dN:
            jcb = np.transpose(deriv).dot(xyz)
            self.d_jacob.append(np.linalg.det(jcb))
            self.DNX.append(np.dot(deriv, np.linalg.inv(np.transpose(jcb))))

        return

    def matrix_B(self):

        for dnx in self.DNX:
            B = np.zeros((6, dnx.shape[0] * dnx.shape[1]))

            for i in range(int(dnx.shape[0])):
                N = i * 3
                B[0, N + 0] = dnx[i, 0]  # E_xx
                B[1, N + 1] = dnx[i, 1]  # E_yy
                B[2, N + 2] = dnx[i, 2]  # E_zz
                B[3, N + 0] = dnx[i, 1]  # 2 * E_xy
                B[3, N + 1] = dnx[i, 0]
                B[4, N + 1] = dnx[i, 2]  # 2 * E_yz
                B[4, N + 2] = dnx[i, 1]
                B[5, N + 0] = dnx[i, 2]  # 2 * E_xz
                B[5, N + 2] = dnx[i, 0]

            self.B.append(B)

        return

    def int_H(self):

        for nx in self.N:
            H = np.zeros((self.DNX[0][0].shape[0], nx.shape[0] * self.DNX[0][0].shape[0]))

            for i in range(int(nx.shape[0])):
                N = i * 2
                H[0, N] = nx[i]
                H[1, N + 1] = nx[i]

            self.H.append(H)

        return

    def compute_stiffness(self, D):

        Ke = np.zeros((self.B[0].shape[1], self.B[0].shape[1]))
        for i, b in enumerate(self.B):
            Ke += np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]

        return Ke

    def compute_mass(self, rho):

        Me = np.zeros((self.H[0].shape[1], self.H[0].shape[1]))
        for i, H in enumerate(self.H):
            Me += np.dot(np.dot(np.transpose(H), rho), H) * self.d_jacob[i] * self.W[i]

        return Me

    def compute_abs_bound(self, a, rho, vp):

        f_abs = np.zeros((self.H[0].shape[1], self.H[0].shape[1]))

        # absorbing boundaries are applied at the surface. The integral is made over the surface and not over the volume
        # therefore the shape functions, jacobian and weights are different

        for i, H in enumerate(self.H):
            # integration over the surface not over the area!!
            f_abs += np.dot(np.dot(np.transpose(H), a * rho * vp), H) * self.d_jacob[i] * self.W[i]

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


def shape4(xy):
    """shape functions 4 node plane element"""

    N = np.zeros((4, 1))
    dN = np.zeros((4, 2))

    x1 = xy[0]
    x2 = xy[1]

    # shape functions
    N[0] = 1. / 4. * (1 - x1) * (1 - x2)
    N[1] = 1. / 4. * (1 + x1) * (1 - x2)
    N[2] = 1. / 4. * (1 + x1) * (1 + x2)
    N[3] = 1. / 4. * (1 - x1) * (1 + x2)

    # derivative in x1
    dN[0, 0] = -(1 - x2) / 4
    dN[1, 0] = (1 - x2) / 4
    dN[2, 0] = (x2 + 1) / 4
    dN[3, 0] = -(x2 + 1) / 4

    # derivative in x2
    dN[0, 1] = -(1 - x1) / 4
    dN[1, 1] = -(x1 + 1) / 4
    dN[2, 1] = (x1 + 1) / 4
    dN[3, 1] = (1 - x1) / 4

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
