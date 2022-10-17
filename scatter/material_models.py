import sys
import numpy as np


def stiffness_elasticity(E: float, poisson: float, dimension: int) -> np.ndarray:
    r"""
    Stiffness matrix for isotropic elastic material

    $\stress = \frac{1}{E} \times D \times \vareplison$

    Parameters
    ----------
    :param E: Young modulus
    :param poisson: Poisson ratio
    :return: Stiffness matrix
    """

    if dimension ==3:
        D = np.zeros((6, 6))

        D[:3, :3] = [[1. - poisson, poisson, poisson],
                     [poisson, 1. - poisson, poisson],
                     [poisson, poisson, 1. - poisson]]

        D[3:, 3:] = [[(1. - 2. * poisson) / 2, 0, 0],
                     [0, (1. - 2. * poisson) / 2, 0],
                     [0, 0, (1. - 2. * poisson) / 2]]

        D =D * (E / ((1. + poisson) * (1. - 2. * poisson)))

    elif dimension == 2:
        D = np.zeros((3, 3))

        D[:2, :2] = [[1. - poisson, poisson],
                     [poisson, 1. - poisson]]

        D[2, 2] = (1. - 2. * poisson) / 2

        D = D * E / ((1. + poisson) * (1. - 2. * poisson))
    else:
        sys.exit(f"ERROR dimension: {dimension} is not supported")

    return D
