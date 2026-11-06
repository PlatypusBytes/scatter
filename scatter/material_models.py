import sys
import numpy as np
import numpy.typing as npt


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


def stiffness_elasticity_batch(E: npt.NDArray[np.float64], poisson: npt.NDArray[np.float64],
                               dimension: int) -> npt.NDArray[np.float64]:
    """Compute the isotropic elastic material matrix of each element.

    Vectorised counterpart of `stiffness_elasticity`. In 2D it assumes plane strain.

    Args:
        E: Young modulus of each element [Pa].
        poisson: Poisson ratio of each element [-].
        dimension: dimension of the model.

    Returns:
        Material matrices, with shape (element, strain, strain) [Pa].

    Raises:
        ValueError: if the dimension is not 2 or 3.
    """
    if dimension == 3:
        nb_normal, nb_strains = 3, 6
    elif dimension == 2:
        nb_normal, nb_strains = 2, 3
    else:
        raise ValueError(f"Dimension {dimension} is not supported.")

    factor = E / ((1. + poisson) * (1. - 2. * poisson))
    normal = np.arange(nb_normal)
    shear = np.arange(nb_normal, nb_strains)

    D = np.zeros((len(E), nb_strains, nb_strains))
    D[:, :nb_normal, :nb_normal] = (factor * poisson)[:, None, None]
    D[:, normal, normal] = (factor * (1. - poisson))[:, None]
    D[:, shear, shear] = (factor * (1. - 2. * poisson) / 2)[:, None]
    return D
