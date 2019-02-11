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
    from scipy.sparse.linalg import inv

    k_part = k_global.dot(u)
    c_part = c_global.dot(v)

    a = inv(m_global.tocsc()).dot(force_ini - c_part - k_part)

    return a


class Solver:
    def __init__(self, number_equations):
        import numpy as np

        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

        self.u = []
        self.time = []
        return

    def newmark(self, settings, M, C, K, F, t_step, t_total):
        import numpy as np
        from scipy.sparse.linalg import spsolve

        # constants for the Newmark
        a1, a2, a3, a4, a5, a6 = const(settings["beta"], settings["gamma"], t_step)

        # initial conditions
        u = self.u0
        v = self.v0
        F_ini = np.array([float(i) for i in F.getcol(0).todense()])

        a = init(M, C, K, F_ini, u, v)

        K_till = K + C.dot(a4) + M.dot(a1)

        self.time = np.linspace(0, t_total, int(np.ceil(t_total / t_step)))

        for t in range(len(self.time)):
            m_part = u.dot(a1) + v.dot(a2) + a.dot(a3)
            c_part = u.dot(a4) + v.dot(a5) + a.dot(a6)
            m_part = M.dot(m_part)
            c_part = C.dot(c_part)

            # external force
            force = np.array([float(i) for i in F.getcol(t).todense()])
            force_ext = force + m_part + c_part
            # solve
            uu = spsolve(K_till, force_ext)
            # velocity calculated through Newmark relation
            vv = (uu - u).dot(a4) - v.dot(a5) - a.dot(a6)
            # acceleration calculated through equilibrium equation (to preserve equilibrium at each time step)
            # aa = init(M, C, K, force_ext, uu, vv)
            aa = (uu - u).dot(a1) - v.dot(a2) - a.dot(a3)

            self.u.append(uu)

            u = uu
            a = aa
            v = vv

        self.u = np.array(self.u)
        return

    def static(self, settings, K, F, t_step, t_total):
        import numpy as np
        from scipy.sparse.linalg import spsolve

        # initial conditions
        u = self.u0
        self.time = np.linspace(0, t_total, np.ceil(t_total / t_step))

        for t in range(len(self.time)):
            # external force
            force = np.array([float(i) for i in F.getcol(t).todense()])

            # solve
            uu = spsolve(K.tocsr(), force)

            self.u.append(uu)

        self.u = np.array(self.u)

        return
