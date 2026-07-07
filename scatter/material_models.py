import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np


# string identifiers for the supported material formulations
DRY = "dry"
BIOT = "biot"


@dataclass
class MaterialProperties:
    r"""
    Parsed material properties for a single material.

    For a ``dry`` (single phase) material only the solid skeleton properties are
    used. For a ``biot`` (saturated, two-phase) material the additional fluid and
    coupling coefficients required by the *u-w* formulation are computed as well.

    The *u-w* (solid displacement :math:`\mathbf{u}` and relative/Darcy fluid
    displacement :math:`\mathbf{w}`) formulation leads to the block system

    .. math::

        \begin{bmatrix} M_{uu} & M_{uw} \\ M_{uw}^T & M_{ww} \end{bmatrix}
        \begin{Bmatrix} \ddot{u} \\ \ddot{w} \end{Bmatrix} +
        \begin{bmatrix} C_{uu} & 0 \\ 0 & C_{ww} \end{bmatrix}
        \begin{Bmatrix} \dot{u} \\ \dot{w} \end{Bmatrix} +
        \begin{bmatrix} K_{uu} & K_{uw} \\ K_{uw}^T & K_{ww} \end{bmatrix}
        \begin{Bmatrix} u \\ w \end{Bmatrix} =
        \begin{Bmatrix} f_u \\ f_w \end{Bmatrix}

    with the element coefficients stored in this dataclass.
    """
    formulation: str  # "dry" or "biot"
    rho: float  # bulk/mixture density used for the solid inertia and absorbing BC
    E: float  # (drained) Young modulus of the skeleton
    poisson: float  # (drained) Poisson ratio of the skeleton
    D: np.ndarray  # drained elastic constitutive matrix

    # Biot (u-w) coefficients - None for a dry material
    alpha: Optional[float] = None  # Biot-Willis coefficient
    biot_modulus: Optional[float] = None  # Biot modulus M
    mass_coupling: Optional[float] = None  # density for M_uw block (= rho_fluid)
    mass_fluid: Optional[float] = None  # density for M_ww block (= rho_fluid / n * tortuosity)
    drag: Optional[float] = None  # drag coefficient for C_ww block (= rho_fluid * g / k)

    @property
    def is_biot(self) -> bool:
        """True if the material uses the Biot (u-w) formulation."""
        return self.formulation == BIOT


def get_material_properties(material: dict, dimension: int) -> MaterialProperties:
    r"""
    Parse a raw material dictionary into a :class:`MaterialProperties` object.

    A material is treated as ``dry`` (current single phase behaviour) unless it
    defines ``"formulation": "biot"``. The ``formulation`` key is optional and
    defaults to ``dry`` for backward compatibility.

    Dry material keys: ``density``, ``Young``, ``poisson``.

    Biot material keys: ``porosity``, ``solid_density``, ``fluid_density``,
    ``Young``, ``poisson``, ``solid_bulk_modulus``, ``fluid_bulk_modulus``,
    ``permeability`` (Darcy hydraulic conductivity [m/s]). Optional Biot keys:
    ``biot_coefficient`` (defaults to :math:`1 - K_b/K_s`), ``biot_modulus``
    (defaults to :math:`(n/K_f + (\alpha-n)/K_s)^{-1}`), ``tortuosity``
    (defaults to 1), ``gravity`` (defaults to 9.81) and ``drag`` (defaults to
    :math:`\rho_f g / k`).

    Parameters
    ----------
    :param material: raw material dictionary
    :param dimension: dimension of the problem (2 or 3)
    :return: parsed :class:`MaterialProperties`
    """

    formulation = str(material.get("formulation", DRY)).lower()

    # drained skeleton elastic properties (shared by both formulations)
    E = material["Young"]
    poisson = material["poisson"]
    D = stiffness_elasticity(E, poisson, dimension)

    if formulation == DRY:
        return MaterialProperties(formulation=DRY, rho=material["density"], E=E, poisson=poisson, D=D)

    if formulation == BIOT:
        n = material["porosity"]
        rho_s = material["solid_density"]
        rho_f = material["fluid_density"]
        K_s = material["solid_bulk_modulus"]
        K_f = material["fluid_bulk_modulus"]

        # bulk (mixture) density
        rho_bulk = (1.0 - n) * rho_s + n * rho_f

        # drained bulk modulus of the skeleton
        K_b = E / (3.0 * (1.0 - 2.0 * poisson))

        # Biot-Willis coefficient (default computed from the skeleton/grain stiffness)
        alpha = material.get("biot_coefficient", 1.0 - K_b / K_s)

        # Biot modulus M (default computed from the constituents)
        biot_modulus = material.get("biot_modulus", 1.0 / (n / K_f + (alpha - n) / K_s))

        # apparent fluid density including tortuosity (added mass)
        tortuosity = material.get("tortuosity", 1.0)
        mass_fluid = rho_f / n * tortuosity

        # viscous drag coefficient of the generalised Darcy law
        # default from the Darcy hydraulic conductivity k [m/s]: b = rho_f * g / k
        gravity = material.get("gravity", 9.81)
        drag = material.get("drag", rho_f * gravity / material["permeability"])

        return MaterialProperties(formulation=BIOT, rho=rho_bulk, E=E, poisson=poisson, D=D,
                                  alpha=alpha, biot_modulus=biot_modulus, mass_coupling=rho_f,
                                  mass_fluid=mass_fluid, drag=drag)

    sys.exit(f"ERROR: material formulation '{formulation}' is not supported. Use '{DRY}' or '{BIOT}'.")


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
