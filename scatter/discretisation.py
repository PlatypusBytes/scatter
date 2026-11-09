"""Shape functions and Gauss integration of single elements and batches of elements."""
import itertools
import math
import sys
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

from scatter.element_types import create_element

VOLUME_ELEMENTS = ("hexa8", "hexa20", "tetra4", "tetra10")
PLANE_ELEMENTS = ("quad4", "quad8", "tri3", "tri6")
BOUNDARY_ELEMENTS = ("line2", "line3", "quad4", "quad8", "tri3", "tri6")


def Gauss_weights(n: int, integration_type: str) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the coordinates and weights of the Gauss integration rule.

    Args:
        n: order of the integration: the number of points per direction for quad, the rule index for tri and tetra.
        integration_type: quad (1D Gauss-Legendre rule), tri or tetra.

    Returns:
        Natural coordinates of the points, with shape (point,) for quad and (direction, point) for tri and tetra
        [-], and the weights of the points [-].

    Raises:
        SystemExit: if the order is not supported for a quad rule.
        SystemExit: if the order is not supported for a tri rule.
        SystemExit: if the order is not supported for a tetra rule.
        SystemExit: if the integration type is not supported.
    """
    x: Union[List[float], List[List[float]]]
    w: List[float]
    if integration_type == "quad":
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
    elif integration_type == "tri":
        if n == 1:
            x = [[1 / 3], [1 / 3]]
            w = [1 / 2]
        elif n == 2:
            x = [[1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]]
            w = [1 / 6, 1 / 6, 1 / 6]
        elif n == 3:
            x = [[1 / 3, 1 / 5, 3 / 5, 1 / 5], [1 / 3, 1 / 5, 1 / 5, 3 / 5]]
            w = [-27 / 96, 25 / 96, 25 / 96, 25 / 96]
        else:
            sys.exit("ERROR: integration order not supported")
    elif integration_type == "tetra":
        if n == 1:
            x = [[1 / 4], [1 / 4], [1 / 4]]
            w = [1 / 6]
        elif n == 2:
            constant_1 = 1 / 4 - 1 / 20 * math.sqrt(5)
            constant_2 = 1 / 4 + 3 / 20 * math.sqrt(5)
            x = [[constant_1, constant_1, constant_1, constant_2], [constant_1, constant_1, constant_2, constant_1],
                 [constant_1, constant_2, constant_1, constant_1]]
            w = [1 / 24, 1 / 24, 1 / 24, 1 / 24]
        else:
            sys.exit(f"ERROR: integration order not supported for type {integration_type}")
    else:
        sys.exit("ERROR: integration type not supported")

    return np.array(x, dtype=float), np.array(w, dtype=float)


def gauss_points(integration_type: str, order: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get the Gauss points and weights of a reference element.

    Args:
        integration_type: line, quad, hexa, tri or tetra.
        order: order of the Gauss integration.

    Returns:
        Natural coordinates of the points, with shape (point, direction) [-], and the weights of the points [-].
    """
    # line, quad and hexa rules are tensor products of the 1D rule, with the first direction varying slowest
    nb_directions = {"line": 1, "quad": 2, "hexa": 3}
    if integration_type in nb_directions:
        coordinates, weights = Gauss_weights(order, "quad")
        indices = list(itertools.product(range(len(weights)), repeat=nb_directions[integration_type]))
        points = np.array([[coordinates[i] for i in index] for index in indices])
        point_weights = np.array([np.prod([weights[i] for i in index]) for index in indices])
        return points, point_weights

    coordinates, weights = Gauss_weights(order, integration_type)
    return coordinates.T, weights


def reference_integration(
        elem_type: str, order: int) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Evaluate the shape functions of a reference element at its Gauss points.

    Args:
        elem_type: element type, see `element_types.create_element`.
        order: order of the Gauss integration.

    Returns:
        Shape functions, with shape (point, node), their derivatives in natural coordinates, with shape (point, node,
        direction), and the Gauss weights.
    """
    element = create_element(elem_type)
    points, weights = gauss_points(element.integration_type, order)

    N = []
    dN = []
    for point in points:
        element.shape_functions(point)
        # the element overwrites its arrays at every point
        N.append(element.N.copy())
        dN.append(element.dN.copy())
    return np.array(N), np.array(dN), weights


def integrate_scalar_mass(N: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Integrate N^T N over each element, for a single degree of freedom per node.

    Args:
        N: shape functions at the Gauss points, with shape (point, node).
        weights: Gauss weight times Jacobian determinant (times density) of each point of each element, with shape
            (element, point).

    Returns:
        Scalar mass matrices, with shape (element, node, node).
    """
    return np.einsum("gi,gj,eg->eij", N, N, weights)


class VolumeElement:
    """Shape functions and numerical integration of a single 3D element.

    Reference implementation; the assembly uses the vectorised `ElementBatch`.
    """

    def __init__(self, elem_type: str = "hexa8", order: int = 2) -> None:
        """Evaluate the shape functions at the Gauss points of the reference element.

        Args:
            elem_type: element type: hexa8, hexa20, tetra4 or tetra10.
            order: order of the Gauss integration.

        Raises:
            ValueError: if the element type is not a volume element.
        """
        if elem_type not in VOLUME_ELEMENTS:
            raise ValueError(f"Element type {elem_type} is not a volume element.")
        self.type = elem_type
        self.n = order
        self.N, self.dN, self.W = reference_integration(elem_type, order)
        self.d_jacob = np.empty(0)  # determinant of the Jacobian per Gauss point
        self.dN_global = np.empty((0, 0, 0))  # derivative of the shape functions in global coordinates
        self.B_matrix = np.empty((0, 0, 0))  # strain displacement matrix per Gauss point
        self.N_matrix = np.empty((0, 0, 0))  # displacement interpolation matrix per Gauss point

    def generate(self, coordinates: npt.NDArray[np.float64]) -> None:
        """Compute the Jacobian, the B matrix and the N matrix of the element.

        Args:
            coordinates: coordinates of the nodes of the element [m].
        """
        self.jacob(coordinates)
        self.matrix_B()
        self.matrix_N()

    def jacob(self, xyz: npt.NDArray[np.float64]) -> None:
        """Compute the determinant of the Jacobian and the global derivatives of the shape functions.

        Args:
            xyz: coordinates of the nodes of the element [m].
        """
        jacobian = np.einsum("gnd,nk->gdk", self.dN, xyz)
        self.d_jacob = np.linalg.det(jacobian)
        self.dN_global = np.einsum("gnd,gkd->gnk", self.dN, np.linalg.inv(jacobian))

    def matrix_B(self) -> None:
        """Compute the B matrix (strain-displacement matrix) from the global shape function derivatives."""
        nb_points, nb_nodes, _ = self.dN_global.shape
        self.B_matrix = np.zeros((nb_points, 6, nb_nodes * 3))
        for i in range(nb_nodes):
            idx = i * 3
            self.B_matrix[:, 0, idx + 0] = self.dN_global[:, i, 0]
            self.B_matrix[:, 1, idx + 1] = self.dN_global[:, i, 1]
            self.B_matrix[:, 2, idx + 2] = self.dN_global[:, i, 2]
            self.B_matrix[:, 3, idx + 0] = self.dN_global[:, i, 1]
            self.B_matrix[:, 3, idx + 1] = self.dN_global[:, i, 0]
            self.B_matrix[:, 4, idx + 1] = self.dN_global[:, i, 2]
            self.B_matrix[:, 4, idx + 2] = self.dN_global[:, i, 1]
            self.B_matrix[:, 5, idx + 0] = self.dN_global[:, i, 2]
            self.B_matrix[:, 5, idx + 2] = self.dN_global[:, i, 0]

    def matrix_N(self) -> None:
        """Compute the N matrix (displacement interpolation matrix) per Gauss point."""
        nb_points, nb_nodes = self.N.shape
        self.N_matrix = np.zeros((nb_points, 3, nb_nodes * 3))
        for direction in range(3):
            self.N_matrix[:, direction, direction::3] = self.N

    def compute_stiffness(self, D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the stiffness matrix of the element.

        Args:
            D: material matrix [Pa].

        Returns:
            Stiffness matrix [N/m].
        """
        BtD = np.einsum("ikj,kl->ijl", self.B_matrix, D)
        return np.einsum("ijk,ikl,i->jl", BtD, self.B_matrix, self.d_jacob * self.W)

    def compute_mass(self, rho: float) -> npt.NDArray[np.float64]:
        """Compute the consistent mass matrix of the element.

        Args:
            rho: density of the element [kg/m^3].

        Returns:
            Mass matrix [kg].
        """
        return np.einsum("ikj,ikl,i->jl", self.N_matrix, self.N_matrix, self.d_jacob * self.W) * rho


class SurfaceElement:
    """Shape functions and numerical integration of a single 2D plane strain element.

    Reference implementation; the assembly uses the vectorised `ElementBatch`.
    """

    def __init__(self, elem_type: str = "quad4", order: int = 2) -> None:
        """Evaluate the shape functions at the Gauss points of the reference element.

        Args:
            elem_type: element type: quad4, quad8, tri3 or tri6.
            order: order of the Gauss integration.

        Raises:
            ValueError: if the element type is not a plane element.
        """
        if elem_type not in PLANE_ELEMENTS:
            raise ValueError(f"Element type {elem_type} is not a plane element.")
        self.type = elem_type
        self.n = order
        self.N, self.dN, self.W = reference_integration(elem_type, order)
        self.d_jacob = np.empty(0)  # determinant of the Jacobian per Gauss point
        self.dN_global = np.empty((0, 0, 0))  # derivative of the shape functions in global coordinates
        self.B_matrix = np.empty((0, 0, 0))  # strain displacement matrix per Gauss point
        self.N_matrix = np.empty((0, 0, 0))  # displacement interpolation matrix per Gauss point

    def generate(self, coordinates: npt.NDArray[np.float64]) -> None:
        """Compute the Jacobian, the B matrix and the N matrix of the element.

        Args:
            coordinates: coordinates of the nodes of the element; only x and y are used [m].
        """
        self.jacob(coordinates[:, :2])
        self.matrix_B()
        self.matrix_N()

    def jacob(self, xy: npt.NDArray[np.float64]) -> None:
        """Compute the determinant of the Jacobian and the global derivatives of the shape functions.

        Args:
            xy: x and y coordinates of the nodes of the element [m].
        """
        jacobian = np.einsum("gnd,nk->gdk", self.dN, xy)
        self.d_jacob = np.linalg.det(jacobian)
        self.dN_global = np.einsum("gnd,gkd->gnk", self.dN, np.linalg.inv(jacobian))

    def matrix_B(self) -> None:
        """Compute the B matrix (strain-displacement matrix) from the global shape function derivatives."""
        nb_points, nb_nodes, _ = self.dN_global.shape
        self.B_matrix = np.zeros((nb_points, 3, nb_nodes * 2))
        self.B_matrix[:, 0, 0::2] = self.dN_global[:, :, 0]
        self.B_matrix[:, 1, 1::2] = self.dN_global[:, :, 1]
        self.B_matrix[:, 2, 0::2] = self.dN_global[:, :, 1]
        self.B_matrix[:, 2, 1::2] = self.dN_global[:, :, 0]

    def matrix_N(self) -> None:
        """Compute the N matrix (displacement interpolation matrix) per Gauss point."""
        nb_points, nb_nodes = self.N.shape
        self.N_matrix = np.zeros((nb_points, 2, nb_nodes * 2))
        for direction in range(2):
            self.N_matrix[:, direction, direction::2] = self.N

    def compute_stiffness(self, D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the stiffness matrix of the element.

        Args:
            D: material matrix [Pa].

        Returns:
            Stiffness matrix [N/m].
        """
        BtD = np.einsum("ikj,kl->ijl", self.B_matrix, D)
        return np.einsum("ijk,ikl,i->jl", BtD, self.B_matrix, self.d_jacob * self.W)

    def compute_mass(self, rho: float) -> npt.NDArray[np.float64]:
        """Compute the consistent mass matrix of the element.

        Args:
            rho: density of the element [kg/m^3].

        Returns:
            Mass matrix [kg].
        """
        return np.einsum("ikj,ikl,i->jl", self.N_matrix, self.N_matrix, self.d_jacob * self.W) * rho


class ElementBatch:
    """Shape functions and numerical integration for a batch of elements of the same type.

    Vectorised counterpart of VolumeElement and SurfaceElement: the shape functions at the Gauss points are evaluated
    once on the reference element, and the Jacobian, B matrix, stiffness and mass are computed for all elements of the
    batch at once.
    """

    def __init__(self, elem_type: str, order: int = 2) -> None:
        """Evaluate the shape functions at the Gauss points of the reference element.

        Args:
            elem_type: element type, one of hexa8, hexa20, tetra4, tetra10, quad4, quad8, tri3 or tri6.
            order: order of the Gauss integration.

        Raises:
            ValueError: if the element type is not supported.
        """
        if elem_type in VOLUME_ELEMENTS:
            self.dimension = 3
        elif elem_type in PLANE_ELEMENTS:
            self.dimension = 2
        else:
            raise ValueError(f"Element type {elem_type} is not supported.")

        self.type = elem_type
        # shape functions (point, node), local derivatives (point, node, direction) and Gauss weights (point)
        self.N, self.dN, self.W = reference_integration(elem_type, order)
        self.d_jacob = np.empty((0, len(self.W)))  # determinant of Jacobian (element, point)
        self.dN_global = np.empty((0, ) + self.dN.shape)  # global derivatives (element, point, node, direction)
        self.B_matrix = np.empty((0, len(self.W), self.nb_strains, self.nb_dofs))  # strain displacement matrix

    @property
    def nb_dofs(self) -> int:
        """Number of degrees of freedom of one element."""
        return int(self.N.shape[1] * self.dimension)

    @property
    def nb_strains(self) -> int:
        """Number of strain components."""
        return 6 if self.dimension == 3 else 3

    @property
    def memory_per_element(self) -> int:
        """Estimate of the memory needed to compute and assemble the matrices of one element [bytes]."""
        # B and D*B per Gauss point, plus Ke, Me and the row, column, mask and value arrays used in the assembly
        return 8 * (2 * len(self.W) * self.nb_strains * self.nb_dofs + 8 * self.nb_dofs**2)

    def generate(self, coordinates: npt.NDArray[np.float64]) -> None:
        """Compute the Jacobian, the global shape function derivatives and the B matrix of all elements.

        Args:
            coordinates: coordinates of the nodes of each element, with shape (element, node, direction) [m]. Only the
                first `dimension` directions are used.
        """
        jacobian = np.einsum("gnd,enk->egdk", self.dN, coordinates[..., :self.dimension])
        self.d_jacob = np.linalg.det(jacobian)
        self.dN_global = np.einsum("gnd,egkd->egnk", self.dN, np.linalg.inv(jacobian))
        self.__matrix_B()

    def __matrix_B(self) -> None:
        """Compute the B matrix (strain-displacement matrix) from the global shape function derivatives."""
        dN = self.dN_global
        dim = self.dimension
        self.B_matrix = np.zeros(dN.shape[:2] + (self.nb_strains, self.nb_dofs))
        if dim == 3:
            self.B_matrix[..., 0, 0::dim] = dN[..., 0]
            self.B_matrix[..., 1, 1::dim] = dN[..., 1]
            self.B_matrix[..., 2, 2::dim] = dN[..., 2]
            self.B_matrix[..., 3, 0::dim] = dN[..., 1]
            self.B_matrix[..., 3, 1::dim] = dN[..., 0]
            self.B_matrix[..., 4, 1::dim] = dN[..., 2]
            self.B_matrix[..., 4, 2::dim] = dN[..., 1]
            self.B_matrix[..., 5, 0::dim] = dN[..., 2]
            self.B_matrix[..., 5, 2::dim] = dN[..., 0]
        else:
            self.B_matrix[..., 0, 0::dim] = dN[..., 0]
            self.B_matrix[..., 1, 1::dim] = dN[..., 1]
            self.B_matrix[..., 2, 0::dim] = dN[..., 1]
            self.B_matrix[..., 2, 1::dim] = dN[..., 0]

    def compute_stiffness(self, D: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the stiffness matrix of each element.

        Args:
            D: material matrix of each element, with shape (element, strain, strain) [Pa].

        Returns:
            Stiffness matrices, with shape (element, dof, dof) [N/m].
        """
        DB = np.einsum("eij,egjk->egik", D, self.B_matrix)
        return np.einsum("egji,egjk,eg->eik", self.B_matrix, DB, self.d_jacob * self.W, optimize=True)

    def compute_scalar_mass(self, rho: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the integral of rho * N^T N over each element, for a single degree of freedom per node.

        Args:
            rho: density of each element [kg/m^3].

        Returns:
            Scalar mass matrices, with shape (element, node, node) [kg].
        """
        return integrate_scalar_mass(self.N, self.d_jacob * self.W * rho[:, None])

    def compute_mass(self, rho: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the consistent mass matrix of each element.

        Args:
            rho: density of each element [kg/m^3].

        Returns:
            Mass matrices, with shape (element, dof, dof) [kg].
        """
        scalar_mass = self.compute_scalar_mass(rho)
        # the mass only couples equal directions: expand the scalar matrix to every direction
        mass = np.zeros((scalar_mass.shape[0], self.nb_dofs, self.nb_dofs))
        for direction in range(self.dimension):
            mass[:, direction::self.dimension, direction::self.dimension] = scalar_mass
        return mass


class BoundaryElementBatch:
    """Numerical integration over a batch of boundary elements: surfaces in 3D or lines in 2D.

    The Jacobian determinant is the measure of the element in the global space, sqrt(det(J^T J)), so the boundary
    can have any orientation.
    """

    def __init__(self, elem_type: str, order: int = 2) -> None:
        """Evaluate the shape functions at the Gauss points of the reference element.

        Args:
            elem_type: element type, one of line2, line3, quad4, quad8, tri3 or tri6.
            order: order of the Gauss integration.

        Raises:
            ValueError: if the element type is not a boundary element.
        """
        if elem_type not in BOUNDARY_ELEMENTS:
            raise ValueError(f"Element type {elem_type} is not a boundary element.")
        self.type = elem_type
        self.N, self.dN, self.W = reference_integration(elem_type, order)
        self.d_jacob = np.empty((0, len(self.W)))  # area (3D) or length (2D) per unit natural area (element, point)

    def generate(self, coordinates: npt.NDArray[np.float64]) -> None:
        """Compute the measure of each boundary element at the Gauss points.

        Args:
            coordinates: coordinates of the nodes of each element, in gmsh order, with shape (element, node,
                direction) [m].
        """
        tangents = np.einsum("gnd,enk->egkd", self.dN, coordinates)
        metric = np.einsum("egki,egkj->egij", tangents, tangents)
        self.d_jacob = np.sqrt(np.linalg.det(metric))

    def compute_scalar_mass(self, rho: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute the integral of rho * N^T N over each boundary element.

        Args:
            rho: value per unit area (3D) or length (2D) of each element.

        Returns:
            Scalar mass matrices, with shape (element, node, node).
        """
        return integrate_scalar_mass(self.N, self.d_jacob * self.W * rho[:, None])
