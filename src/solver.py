def const(beta, gamma, h):
    r"""
    Constants for the Newmark-Beta solver.

    :param beta: Parameter :math:`\beta` that weights the contribution of the initial and final acceleration to the
        change of displacement.
    :type beta: float
    :param gamma: Parameter :math:`\gamma` that weights the contribution of the initial and final acceleration to the
        change of velocity.
    :type gamma: float
    :param h: Time step.
    :type h: float

    :return a1: Parameter :math:`\alpha_1`.
    :return a2: Parameter :math:`\alpha_2`.
    :return a3: Parameter :math:`\alpha_3`.
    :return a4: Parameter :math:`\alpha_4`.
    :return a5: Parameter :math:`\alpha_5`.
    :return a6: Parameter :math:`\alpha_6`.

    :raises ValueError:
    :raises TypeError:
    """

    a1 = 1 / (beta * h ** 2)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / (beta * h)
    a5 = (gamma / beta) - 1
    a6 = (gamma / beta - 2) * h / 2

    return a1, a2, a3, a4, a5, a6


def init(m_global, c_global, k_global, force_ini, u, v):
    r"""
    Calculation of the initial conditions - acceleration for the first time-step.

    :param m_global: Global mass matrix.
    :type m_global: np.ndarray
    :param c_global: Global damping matrix.
    :type c_global: np.ndarray
    :param k_global: Global stiffness matrix beam.
    :type k_global: np.ndarray
    :param force_ini: Initial force.
    :type force_ini: np.ndarray
    :param u: Initial conditions - displacement.
    :type u: np.ndarray
    :param v: Initial conditions - velocity.
    :type v: np.ndarray

    :return a: Initial acceleration.

    :raises ValueError:
    :raises TypeError:
    """
    from scipy.linalg import inv

    k_part = k_global.dot(u)
    c_part = c_global.dot(v)

    a = inv(m_global.todense()).dot(force_ini - k_part - c_part)

    return a


class Solver:
    def __init__(self, NEQ):
        import numpy as np

        self.u0 = np.zeros((NEQ))
        self.v0 = np.zeros((NEQ))

        self.u = []
        return

    def newmark(self, settings, M, C, K, F, t_step, t_total):
        import numpy as np
        from scipy.sparse.linalg import spsolve

        # constants for the Newmark
        a1, a2, a3, a4, a5, a6 = const(settings["beta"], settings["gamma"], t_step)

        # initial conditions
        u = self.u0
        v = self.v0

        a = init(M, C, K, F[:, 0], u, v)

        K_till = K + C.dot(a4) + M.dot(a1)

        time = np.linspace(0, t_total, np.ceil(t_total / t_step))

        for t in range(len(time)):

            m_part = u.dot(a1) + v.dot(a2) + a.dot(a3)
            c_part = u.dot(a4) + v.dot(a5) + a.dot(a6)
            m_part = M.dot(m_part)
            c_part = C.dot(c_part)

            # external force
            force_ext = F[:, t] + m_part + c_part
            # solve
            uu = spsolve(K_till, force_ext)

            # velocity calculated through Newmark relation
            vv = a.dot(a6) - v.dot(a5) + (uu - u).dot(a4)
            # acceleration calculated through equilibrium equation (to preserve equilibrium at each time step)
            aa = init(M, C, K, force_ext, uu, vv)

            self.u.append(uu)

            u == uu
            a == aa
            v == vv

        return