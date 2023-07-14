import numpy as np
import math
import sys
# import scatter packages
from scatter.element_types import HexEight, HexTwenty, TetraFour, TetraTen, QuadFour, QuadEight, TriThree, TriSix


class VolumeElement:
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

    def hex_integration(self):
        coords, weights = Gauss_weights(self.n, type='quad')
        for i1 in range(self.n):
            for i2 in range(self.n):
                for i3 in range(self.n):
                    X = [coords[i1], coords[i2], coords[i3]]
                    Wt = np.prod([weights[i1], weights[i2], weights[i3]])

                    # call shape functions depending on the type of element
                    if self.type == 'hexa8':
                        hex = HexEight()
                        hex.shape_functions(X)
                    elif self.type == 'hexa20':
                        hex = HexTwenty()
                        hex.shape_functions(X)

                    # shape functions
                    N = hex.N
                    dN = hex.dN

                    # add to self
                    self.N.append(N)
                    self.dN.append(dN)
                    self.W.append(Wt)

    def tetra_integration(self):
        coords, weights = Gauss_weights(self.n, type='tetra')
        n_points = len(weights)

        for i1 in range(n_points):

            X = coords[:, i1]
            Wt = weights[i1]
            # call shape functions depending on the type of element
            if self.type == 'tetra4':
                tetra = TetraFour()
                tetra.shape_functions(X)
            elif self.type == 'tetra10':
                tetra = TetraTen()
                tetra.shape_functions(X)

            # shape functions
            N = tetra.N
            dN = tetra.dN

            # add to self
            self.N.append(N)
            self.dN.append(dN)
            self.W.append(Wt)


    def generate(self, coordinates: list) -> None:
        r"""
        Generate shape functions, Jacobian, *B* and *N* matrix for solid element

        Parameters
        ----------
        :param coordinates: list of coordinates of nodes in one element
        """

        # natural coordinates and integration weights: Gauss integration

        if self.type =='hexa8' or self.type =='hexa20':
            self.hex_integration()

        elif self.type == 'tetra4' or self.type == 'tetra10':
            self.tetra_integration()

        # compute jacobian
        self.jacob(coordinates)
        # compute B matrix
        self.matrix_B()
        # compute N matrix
        self.matrix_N()

    def jacob(self, xyz: list) -> None:
        """
        Computes the Jacobian and the derivative of the Jacobian

        Parameters
        ----------
        :param xyz: list of coordinates of nodes in one element
        """

        # transpose dN
        dN_array = np.transpose(self.dN, (0, 2, 1))

        # calculate Jacobians
        jcbs = np.transpose(np.einsum('ijk,kl', dN_array, xyz), (0, 2, 1))

        # invert Jacobians
        inv_jcbs = np.linalg.inv(jcbs)

        # calculate determinant Jacobians
        self.d_jacob = np.linalg.det(jcbs)

        # generate global dN array, by doing a dotproduct of the dN with the inverse Jacobian at each point
        self.dN_global = np.einsum("ijk, ikl -> ijl", np.transpose(dN_array, (0, 2, 1)), inv_jcbs)

        # Above procedure is equivalent to procedure as shown below, however, the above procedure is more efficient
        #
        #
        # for deriv in self.dN:
        #     jcb = np.transpose(deriv).dot(xyz)
        #     self.d_jacob.append(np.linalg.det(jcb))
        #     self.dN_global.append(np.dot(deriv, np.linalg.inv(np.transpose(jcb))))

    def matrix_B(self) -> None:
        """
        Computes B matrix (strain-displacement matrix): the derivative of the shape functions
        """

        self.B_matrix = np.zeros((self.dN_global.shape[0],6,self.dN_global.shape[1] * self.dN_global.shape[2]))

        for i in range(self.dN_global.shape[1]):
            idx = i * 3
            self.B_matrix[:, 0, idx + 0] = self.dN_global[:,i, 0]
            self.B_matrix[:, 1, idx + 1] = self.dN_global[:,i, 1]
            self.B_matrix[:, 2, idx + 2] = self.dN_global[:,i, 2]
            self.B_matrix[:, 3, idx + 0] = self.dN_global[:,i, 1]
            self.B_matrix[:, 3, idx + 1] = self.dN_global[:,i, 0]
            self.B_matrix[:, 4, idx + 1] = self.dN_global[:,i, 2]
            self.B_matrix[:, 4, idx + 2] = self.dN_global[:,i, 1]
            self.B_matrix[:, 5, idx + 0] = self.dN_global[:,i, 2]
            self.B_matrix[:, 5, idx + 2] = self.dN_global[:,i, 0]

        return

    def matrix_N(self):
        """
        Computes N matrix: matrix with the shape functions
        """
        ndof = 3

        for nx in self.N:
            N = np.zeros((self.dN_global.shape[2], nx.shape[0] * self.dN_global.shape[2]))

            N[0, 0::ndof] = nx
            N[1, 1::ndof] = nx
            N[2, 2::ndof] = nx

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

        # B_transpose dot D
        BtD = np.einsum("ikj,kl-> ijl", self.B_matrix, D)

        #Ke = BtDB * det_jacobian * weight
        Ke = np.einsum("ijk,ikl, i-> jl", BtD, self.B_matrix, self.d_jacob * self.W)


        # above einstein summation is equivalent to the code as stated below

        # Ke = np.zeros((self.B_matrix[0].shape[1], self.B_matrix[0].shape[1]))
        # for i in range(len(self.B_matrix)):
        #     #Ke += np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]
        #     Ke += BtDB[i,:,:] * self.d_jacob[i] * self.W[i]

        return Ke

    def compute_mass(self, rho: float) -> np.ndarray:
        """
        Compute the mass matrix over one element

        Parameters
        ----------
        :param rho: Material density
        :return: Mass matrix
        """

        np_N_matrix = np.array(self.N_matrix)
        Me = np.einsum("ikj, ikl, i -> jl", np_N_matrix, np_N_matrix, self.d_jacob * self.W) * rho

        # above einstein summation is equivalent to the code below

        # Me = np.zeros((self.N_matrix[0].shape[1], self.N_matrix[0].shape[1]))
        # for i, N in enumerate(self.N_matrix):
        #     Me += np.dot(np.dot(np.transpose(N), rho), N) * self.d_jacob[i] * self.W[i]

        return Me

    def compute_abs_bound(self) -> np.ndarray:
        """
        Compute the absorbing boundary unitary force matrix over one element.
        Following Lysmer and Kuhlemyer.

        Parameters
        ----------
        :return: Absorbing force
        """

        f_abs = np.zeros((self.N_matrix[0].shape[1], self.N_matrix[0].shape[1]))
        for i, N in enumerate(self.N_matrix):
            f_abs = f_abs + np.dot(np.dot(np.transpose(N), 1), N) * self.d_jacob[i] * self.W[i]

        return f_abs


class SurfaceElement:
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

    def tri_integration(self):
        coords, weights = Gauss_weights(self.n, type='tri')
        n_points = len(weights)

        for i1 in range(n_points):
            X = coords[:, i1]
            Wt = weights[i1]

            # call shape functions depending on the type of element
            if self.type == 'tri3':
                tri = TriThree()
                tri.shape_functions(X)
            elif self.type == 'tri6':
                tri = TriSix()
                tri.shape_functions(X)

            # shape functions
            N = tri.N
            dN = tri.dN

            # add to self
            self.N.append(N)
            self.dN.append(dN)
            self.W.append(Wt)

    def generate(self, coordinates):
        r"""
        Generate shape functions, Jacobian, *B* and *N* matrix for plane element

        Parameters
        ----------
        :param coordinates: list of coordinates of nodes in one element
        """

        # natural coordinates and integration weights: Gauss integration

        if self.type == 'quad4' or self.type == 'quad8':
            coords, weights = Gauss_weights(self.n, type='quad')
            for i1 in range(self.n):
                for i2 in range(self.n):
                    X = [coords[i1], coords[i2]]
                    Wt = np.prod([weights[i1], weights[i2]])

                    # call shape functions depending on the type of element
                    if self.type == 'quad4':
                        quad = QuadFour()
                        quad.shape_functions(X)
                    elif self.type == 'quad8':
                        quad = QuadEight()
                        quad.shape_functions(X)

                    # shape functions
                    N = quad.N
                    dN = quad.dN

                    # add to self
                    self.N.append(N)
                    self.dN.append(dN)
                    self.W.append(Wt)

        elif self.type == 'tri3' or self.type == 'tri6':
            self.tri_integration()

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
            Ke = Ke + np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]

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
            Me = Me + np.dot(np.dot(np.transpose(N), rho), N) * self.d_jacob[i] * self.W[i]

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
            f_abs = f_abs + np.dot(np.dot(np.transpose(N), 1), N) * self.d_jacob[i] * self.W[i]

        return f_abs


def Gauss_weights(n: int, type: str) -> [np.array, np.array]:
    r"""
    Coordinates and weights in Gauss–Legendre quadrilateral integration formulae

    Parameters
    ----------
    :param n: number of Gauss points
    :return:  points, weights
    """

    if type == "quad":

        if n == 1:
            x = [0.]
            w = [2.]
        elif n == 2:
            x = [-math.sqrt(1. / 3.), math.sqrt(1. / 3.)]
            w = [1., 1.]
        elif n == 3:
            x = [-math.sqrt(3. / 5.), 0, math.sqrt(3. / 5.)]
            w = [5. / 9., 8. / 9., 5. / 9.]
        else:
            sys.exit("ERROR: integration order not supported")
    elif type == "tri":
        if n == 1:
            x = [[1/3], [1/3]]
            w = [1/2]
        elif n == 2:
            x = [[1/6, 2/3, 1/6], [1/6,1/6,2/3]]
            w = [1/6, 1/6, 1/6]
        elif n == 3:
            x = [[1/3, 1/5, 3/5, 1/5], [1/3, 1/5, 1/5, 3/5]]
            w = [-27/96, 25/96, 25/96, 25/96]
        else:
            sys.exit("ERROR: integration order not supported")
    elif type == "tetra":
        if n == 1:
            x = [[1 / 4],
                 [1 / 4],
                 [1 / 4]]
            w = [1 / 6]
        elif n == 2:

            constant_1 = 1 / 4 - 1/20 * math.sqrt(5)
            constant_2 = 1 / 4 + 3 / 20 * math.sqrt(5)

            x = [[constant_1, constant_1, constant_1,
                  constant_2],
                 [constant_1, constant_1, constant_2,
                  constant_1],
                 [constant_1, constant_2, constant_1,
                  constant_1]]

            weight_constant = 1/24
            w = [weight_constant, weight_constant, weight_constant, weight_constant]
        else:
            sys.exit(f"ERROR: integration order not supported for type {type}")

    else:
        sys.exit("ERROR: integration type not supported")

    return np.array(x), np.array(w)
