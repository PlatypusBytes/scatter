class ShapeFunction:
    def __init__(self, elem_type, order):

        if elem_type == 5:
            self.type = 'linear'
        elif elem_type == 17:
            self.type = 'quad'

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
        """" generate shape functions for solid """
        import numpy as np

        # natural coordinates Gauss integration points
        coords, weights = Gauss_weights(self.n)

        for i1 in range(self.n):
            for i2 in range(self.n):
                for i3 in range(self.n):
                    X = [coords[i1], coords[i2], coords[i3]]
                    Wt = np.prod([weights[i1], weights[i2], weights[i3]])
                    # N, dN = shape20(X)

                    if self.type == 'linear':
                        N, dN = shape8(X)
                    elif self.type == 'quad':
                        N, dN = shape20(X)

                    # add to self
                    self.N.append(N)
                    self.dN.append(dN)
                    self.W.append(Wt)

        return

    def jacob(self, xyz):
        import numpy as np

        for deriv in self.dN:
            jcb = np.transpose(deriv).dot(xyz)
            self.d_jacob.append(np.linalg.det(jcb))
            self.DNX.append(np.dot(deriv, np.linalg.inv(np.transpose(jcb))))

        return

    def matrix_B(self):
        import numpy as np

        for dnx in self.DNX:
            B = np.zeros((6, dnx.shape[0] * dnx.shape[1]))

            for i in range(int(dnx.shape[0])):
                idx = i * 3
                B[0, idx + 0] = dnx[i, 0]  # E_xx
                B[1, idx + 1] = dnx[i, 1]  # E_yy
                B[2, idx + 2] = dnx[i, 2]  # E_zz
                B[3, idx + 0] = dnx[i, 1]  # 2 * E_xy
                B[3, idx + 1] = dnx[i, 0]
                B[4, idx + 1] = dnx[i, 2]  # 2 * E_yz
                B[4, idx + 2] = dnx[i, 1]
                B[5, idx + 0] = dnx[i, 2]  # 2 * E_xz
                B[5, idx + 2] = dnx[i, 0]

            self.B.append(B)

        return

    def int_H(self):
        import numpy as np

        for N in self.N:
            H = np.zeros((3, self.dN[0].shape[0] * self.dN[0].shape[1]))

            for i in range(int(N.shape[0])):
                idx = i * 3
                H[0, idx + 0] = N[i]
                H[1, idx + 1] = N[i]
                H[2, idx + 2] = N[i]

            self.H.append(H)

        return

    def compute_stiffness(self, D):
        import numpy as np

        Ke = np.zeros((self.B[0].shape[1], self.B[0].shape[1]))
        for i, b in enumerate(self.B):
            Ke += np.dot(np.dot(np.transpose(b), D), b) * self.d_jacob[i] * self.W[i]

        return Ke

    def compute_mass(self, rho):
        import numpy as np

        Me = np.zeros((self.H[0].shape[1], self.H[0].shape[1]))
        for i, H in enumerate(self.H):
            Me += np.dot(np.dot(np.transpose(H), rho), H) * self.d_jacob[i] * self.W[i]

        return Me


def Gauss_weights(n):
    """Gauss quadrature

        points & weights
    """
    import numpy as np
    import sys

    if n == 1:
        x = 0.
        w = 2.
    elif n == 2:
        x = [-np.sqrt(1. / 3.), np.sqrt(1. / 3.)]
        w = [1., 1.]
    elif n == 3:
        x = [-np.sqrt(3. / 5.), 0, np.sqrt(3. / 5.)]
        w = [5. / 9., 8. / 9., 5. / 9.]
    else:
        sys.exit("ERROR: integration order not supported")

    return np.array(x), np.array(w)


def shape8(xyz):
    """shape functions 8 node element"""
    import numpy as np

    N = np.zeros((8, 1))
    dN = np.zeros((8, 3))

    # natural coordinates
    Xi = [[-1, -1, -1],
          [1, -1, -1],
          [1, 1, -1],
          [-1, 1, -1],
          [-1, -1, 1],
          [1, 1, 1],
          [-1, 1, 1],
          ]
    Xi = np.array(Xi)

    for i in range(len(Xi)):
        N[i] = 1. / 8. * (1. + Xi[i, 0] * xyz[0]) * (1. + Xi[i, 1] * xyz[1]) * (1. + Xi[i, 2] * xyz[2])
        dN[i, 0] = 1. / 8. * Xi[i, 0] * (1. + Xi[i, 1] * xyz[1]) * (1. + Xi[i, 2] * xyz[2])
        dN[i, 1] = 1. / 8. * (1. + Xi[i, 0] * xyz[0]) * Xi[i, 1] * (1. + Xi[i, 2] * xyz[2])
        dN[i, 2] = 1. / 8. * (1. + Xi[i, 0] * xyz[0]) * (1. + Xi[i, 1] * xyz[1]) * Xi[i, 2]

    return N, dN


def shape20(xyz):
    """shape functions 20 node element"""
    import numpy as np

    N = np.zeros((20, 1))
    dN = np.zeros((20, 3))

    N[1] = 1. / 4. * (1 - xyz[0]**2) * (1 - xyz[1]) * (1 - xyz[2])
    N[3] = 1. / 4. * (1 + xyz[0]) * (1 - xyz[1]**2) * (1 - xyz[2])
    N[5] = 1. / 4. * (1 - xyz[0]**2) * (1 + xyz[1]) * (1 - xyz[2])
    N[7] = 1. / 4. * (1 - xyz[0]) * (1 - xyz[1]**2) * (1 - xyz[2])

    N[8] = 0.25 * (1 - xyz[0]) * (1 - xyz[1]) * (1 - xyz[2]**2)
    N[9] = 0.25 * (1 + xyz[0]) * (1 - xyz[1]) * (1 - xyz[2]**2)
    N[10] = 0.25 * (1 + xyz[0]) * (1 + xyz[1]) * (1 - xyz[2]**2)
    N[11] = 0.25 * (1 - xyz[0]) * (1 + xyz[1]) * (1 - xyz[2]**2)

    N[13] = 0.25 * (1 - xyz[0]**2) * (1 - xyz[1]) * (1 + xyz[2])
    N[15] = 0.25 * (1 + xyz[0]) * (1 - xyz[1]**2) * (1 + xyz[2])
    N[17] = 0.25 * (1 - xyz[0]**2) * (1 + xyz[1]) * (1 + xyz[2])
    N[19] = 0.25 * (1 - xyz[0]) * (1 - xyz[1]**2) * (1 + xyz[2])

    N[0] = 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 - xyz[2]) * (-xyz[0] - xyz[1] - xyz[2] - 2)
    N[2] = 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 - xyz[2]) * (xyz[0] - xyz[1] - xyz[2] - 2)
    N[4] = 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 - xyz[2]) * (xyz[0] + xyz[1] - xyz[2] - 2)
    N[6] = 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 - xyz[2]) * (-xyz[0] + xyz[1] - xyz[2] - 2)

    N[12] = 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 + xyz[2]) * (-xyz[0] - xyz[1] + xyz[2] - 2)
    N[14] = 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 + xyz[2]) * (xyz[0] - xyz[1] + xyz[2] - 2)
    N[16] = 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 + xyz[2]) * (xyz[0] + xyz[1] + xyz[2] - 2)
    N[18] = 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 + xyz[2]) * (-xyz[0] + xyz[1] + xyz[2] - 2)

    # derivative in x1
    dN[1, 0] = -0.5 * xyz[0] * (1 - xyz[1]) * (1 - xyz[2])
    dN[3, 0] = 0.25 * (1 - xyz[1]**2) * (1 - xyz[2])
    dN[5, 0] = -0.5 * xyz[0] * (1 + xyz[1]) * (1 - xyz[2])
    dN[7, 0] = -0.25 * (1 - xyz[1]**2) * (1 - xyz[2])

    dN[8, 0] = -0.25 * (1 - xyz[1]) * (1 - xyz[2]**2)
    dN[9, 0] = 0.25 * (1 - xyz[1]) * (1 - xyz[2]**2)
    dN[10, 0] = 0.25 * (1 + xyz[1]) * (1 - xyz[2]**2)
    dN[11, 0] = -0.25 * (1 + xyz[1]) * (1 - xyz[2]**2)

    dN[13, 0] = -0.5 * xyz[0] * (1 - xyz[1]) * (1 + xyz[2])
    dN[15, 0] = 0.25 * (1 - xyz[1]**2) * (1 + xyz[2])
    dN[17, 0] = -0.5 * xyz[0] * (1 + xyz[1]) * (1 + xyz[2])
    dN[19, 0] = -0.25 * (1 - xyz[1]**2) * (1 + xyz[2])

    dN[0, 0] = -0.125 * (1 - xyz[1]) * (1 - xyz[2]) * (-xyz[0] - xyz[1] - xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 - xyz[2])
    dN[2, 0] = 0.125 * (1 - xyz[1]) * (1 - xyz[2]) * (xyz[0] - xyz[1] - xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 - xyz[2])
    dN[4, 0] = 0.125 * (1 + xyz[1]) * (1 - xyz[2]) * (xyz[0] + xyz[1] - xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 - xyz[2])
    dN[6, 0] = -0.125 * (1 + xyz[1]) * (1 - xyz[2]) * (-xyz[0] + xyz[1] - xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 - xyz[2])

    dN[12, 0] = -0.125 * (1 - xyz[1]) * (1 + xyz[2]) * (-xyz[0] - xyz[1] + xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 + xyz[2])
    dN[14, 0] = 0.125 * (1 - xyz[1]) * (1 + xyz[2]) * (xyz[0] - xyz[1] + xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 + xyz[2])
    dN[16, 0] = 0.125 * (1 + xyz[1]) * (1 + xyz[2]) * (xyz[0] + xyz[1] + xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 + xyz[2])
    dN[18, 0] = -0.125 * (1 + xyz[1]) * (1 + xyz[2]) * (-xyz[0] + xyz[1] + xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 + xyz[2])

    # derivative in x2
    dN[1, 1] = -0.25 * (1 - xyz[0]**2) * (1 - xyz[2])
    dN[3, 1] = -0.5 * (1 + xyz[0]) * xyz[1] * (1 - xyz[2])
    dN[5, 1] = 0.25 * (1 - xyz[0]**2) * (1 - xyz[2])
    dN[7, 1] = -0.5 * (1 - xyz[0]) * xyz[1] * (1 - xyz[2])

    dN[8, 1] = -0.25 * (1 - xyz[0]) * (1 - xyz[2]**2)
    dN[9, 1] = -0.25 * (1 + xyz[0]) * (1 - xyz[2]**2)
    dN[10, 1] = 0.25 * (1 + xyz[0]) * (1 - xyz[2]**2)
    dN[11, 1] = 0.25 * (1 - xyz[0]) * (1 - xyz[2]**2)

    dN[13, 1] = -0.25 * (1 - xyz[0]**2) * (1 + xyz[2])
    dN[15, 1] = -0.5 * (1 + xyz[0]) * xyz[1] * (1 + xyz[2])
    dN[17, 1] = 0.25 * (1 - xyz[0]**2) * (1 + xyz[2])
    dN[19, 1] = -0.5 * (1 - xyz[0]) * xyz[1] * (1 + xyz[2])

    dN[0, 1] = -0.125 * (1 - xyz[0]) * (1 - xyz[2]) * (-xyz[0] - xyz[1] - xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 - xyz[2])
    dN[2, 1] = -0.125 * (1 + xyz[0]) * (1 - xyz[2]) * (xyz[0] - xyz[1] - xyz[2] - 2) - 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 - xyz[2])
    dN[4, 1] = 0.125 * (1 + xyz[0]) * (1 - xyz[2]) * (xyz[0] + xyz[1] - xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 - xyz[2])
    dN[6, 1] = 0.125 * (1 - xyz[0]) * (1 - xyz[2]) * (-xyz[0] + xyz[1] - xyz[2] - 2) + 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 - xyz[2])

    dN[12, 1] = -0.125 * (1 - xyz[0]) * (1 + xyz[2]) * (-xyz[0] - xyz[1] + xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 + xyz[2])
    dN[14, 1] = -0.125 * (1 + xyz[0]) * (1 + xyz[2]) * (xyz[0] - xyz[1] + xyz[2] - 2) - 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 + xyz[2])
    dN[16, 1] = 0.125 * (1 + xyz[0]) * (1 + xyz[2]) * (xyz[0] + xyz[1] + xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 + xyz[2])
    dN[18, 1] = 0.125 * (1 - xyz[0]) * (1 + xyz[2]) * (-xyz[0] + xyz[1] + xyz[2] - 2) + 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 + xyz[2])

    # derivative in x3
    dN[1,2] = -0.25 * (1 - xyz[0]**2) * (1 - xyz[1])
    dN[3, 2] = -0.25 * (1 + xyz[0]) * (1 - xyz[1]**2)
    dN[5, 2] = -0.25 * (1 - xyz[0]**2) * (1 + xyz[1])
    dN[7, 2] = -0.25 * (1 - xyz[0]) * (1 - xyz[1]**2)

    dN[8, 2] = -0.5 * (1 - xyz[0]) * (1 - xyz[1]) * xyz[2]
    dN[9, 2] = -0.5 * (1 + xyz[0]) * (1 - xyz[1]) * xyz[2]
    dN[10, 2] = -0.5 * (1 + xyz[0]) * (1 + xyz[1]) * xyz[2]
    dN[11, 2] = -0.5 * (1 - xyz[0]) * (1 + xyz[1]) * xyz[2]

    dN[13, 2] = 0.25 * (1 - xyz[0]**2) * (1 - xyz[1])
    dN[15, 2] = 0.25 * (1 + xyz[0]) * (1 - xyz[1]**2)
    dN[17, 2] = 0.25 * (1 - xyz[0]**2) * (1 + xyz[1])
    dN[19, 2] = 0.25 * (1 - xyz[0]) * (1 - xyz[1]**2)

    dN[0, 2] = -0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (-xyz[0] - xyz[1] - xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 - xyz[2])
    dN[2, 2] = -0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (xyz[0] - xyz[1] - xyz[2] - 2) - 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 - xyz[2])
    dN[4, 2] = -0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (xyz[0] + xyz[1] - xyz[2] - 2) - 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 - xyz[2])
    dN[6, 2] = -0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (-xyz[0] + xyz[1] - xyz[2] - 2) - 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 - xyz[2])
    
    dN[12, 2] = 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (-xyz[0] - xyz[1] + xyz[2] - 2) + 0.125 * (1 - xyz[0]) * (1 - xyz[1]) * (1 + xyz[2])
    dN[14, 2] = 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (xyz[0] - xyz[1] + xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 - xyz[1]) * (1 + xyz[2])
    dN[16, 2] = 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (xyz[0] + xyz[1] + xyz[2] - 2) + 0.125 * (1 + xyz[0]) * (1 + xyz[1]) * (1 + xyz[2])
    dN[18, 2] = 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (-xyz[0] + xyz[1] + xyz[2] - 2) + 0.125 * (1 - xyz[0]) * (1 + xyz[1]) * (1 + xyz[2])

    return N, dN
