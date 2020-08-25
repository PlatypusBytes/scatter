def stiffness_elasticity(E, poisson):
    """
    Stifness matrix for isotropic elastic material

    $\stress = \frac{1}{E} \times D \times \vareplison$
    """
    import numpy as np

    D = np.zeros((6, 6))

    D[:3, :3] = [[1. - poisson, poisson, poisson],
                 [poisson, 1. - poisson, poisson],
                 [poisson, poisson, 1. - poisson]]

    D[3:, 3:] = [[(1. - 2. * poisson) / 2, 0, 0],
                 [0, (1. - 2. * poisson) / 2, 0],
                 [0, 0, (1. - 2. * poisson) / 2]]

    D *= E / ((1. + poisson) * (1. - 2. * poisson))

    return D
