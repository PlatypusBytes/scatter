"""Shape functions of the finite elements, following the gmsh node numbering."""
from typing import Callable, Dict, Optional, Protocol

import numpy as np
import numpy.typing as npt


class Element(Protocol):
    """Interface shared by all element types."""

    N: npt.NDArray[np.float64]
    dN: npt.NDArray[np.float64]

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes; in gmsh they come before the midside nodes."""
        ...

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration: line, quad, hexa, tri or tetra."""
        ...

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates of the point [-].
        """
        ...


class LineTwo:
    """2-node line element.

    Node numbering follows:

    0-----------1 --> u
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.N = np.zeros(2)
        self.dN = np.zeros((2, 1))

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 2

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "line"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinate u of the point [-].
        """
        u = natural_coordinates[0]

        self.N[0] = (1 - u) / 2
        self.N[1] = (1 + u) / 2

        self.dN[0, 0] = -1 / 2
        self.dN[1, 0] = 1 / 2


class LineThree:
    """3-node line element.

    Node numbering follows:

    0-----2-----1 --> u
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.N = np.zeros(3)
        self.dN = np.zeros((3, 1))

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 2

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "line"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinate u of the point [-].
        """
        u = natural_coordinates[0]

        self.N[0] = u * (u - 1) / 2
        self.N[1] = u * (u + 1) / 2
        self.N[2] = 1 - u**2

        self.dN[0, 0] = u - 1 / 2
        self.dN[1, 0] = u + 1 / 2
        self.dN[2, 0] = -2 * u


class HexEight:
    r"""8-node brick element.

    Node numbering follows:

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
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 4), dtype=np.int64)
        self.N = np.zeros(8)
        self.dN = np.zeros((8, 3))

    def get_surfaces(self) -> None:
        """Set the local node indices of each surface of the element."""
        self.__surfaces = np.array([[0, 1, 2, 3], [0, 1, 4, 5], [4, 5, 6, 7], [2, 3, 6, 7], [0, 3, 4, 7], [1, 2, 5, 6]])

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each surface, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> int:
        """Number of elements connected to an interior node of a structured mesh."""
        return 8

    @property
    def n_boundary_nodes(self) -> int:
        """Number of nodes of one surface."""
        return 4

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return False

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 8

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "hexa"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u, v and w of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]
        w = natural_coordinates[2]

        # shape functions
        self.N[0] = 1. / 8. * (1 - u) * (1 - v) * (1 - w)
        self.N[1] = 1. / 8. * (1 + u) * (1 - v) * (1 - w)
        self.N[2] = 1. / 8. * (1 + u) * (1 + v) * (1 - w)
        self.N[3] = 1. / 8. * (1 - u) * (1 + v) * (1 - w)
        self.N[4] = 1. / 8. * (1 - u) * (1 - v) * (1 + w)
        self.N[5] = 1. / 8. * (1 + u) * (1 - v) * (1 + w)
        self.N[6] = 1. / 8. * (1 + u) * (1 + v) * (1 + w)
        self.N[7] = 1. / 8. * (1 - u) * (1 + v) * (1 + w)

        # derivative in u
        self.dN[0, 0] = -((1 - v) * (1 - w)) / 8
        self.dN[1, 0] = ((1 - v) * (1 - w)) / 8
        self.dN[2, 0] = ((v + 1) * (1 - w)) / 8
        self.dN[3, 0] = -((v + 1) * (1 - w)) / 8
        self.dN[4, 0] = -((1 - v) * (w + 1)) / 8
        self.dN[5, 0] = ((1 - v) * (w + 1)) / 8
        self.dN[6, 0] = ((v + 1) * (w + 1)) / 8
        self.dN[7, 0] = -((v + 1) * (w + 1)) / 8

        # derivative in v
        self.dN[0, 1] = -((1 - u) * (1 - w)) / 8
        self.dN[1, 1] = -((u + 1) * (1 - w)) / 8
        self.dN[2, 1] = ((u + 1) * (1 - w)) / 8
        self.dN[3, 1] = ((1 - u) * (1 - w)) / 8
        self.dN[4, 1] = -((1 - u) * (w + 1)) / 8
        self.dN[5, 1] = -((u + 1) * (w + 1)) / 8
        self.dN[6, 1] = ((u + 1) * (w + 1)) / 8
        self.dN[7, 1] = ((1 - u) * (w + 1)) / 8

        # derivative in w
        self.dN[0, 2] = -((1 - u) * (1 - v)) / 8
        self.dN[1, 2] = -((u + 1) * (1 - v)) / 8
        self.dN[2, 2] = -((u + 1) * (v + 1)) / 8
        self.dN[3, 2] = -((1 - u) * (v + 1)) / 8
        self.dN[4, 2] = ((1 - u) * (1 - v)) / 8
        self.dN[5, 2] = ((u + 1) * (1 - v)) / 8
        self.dN[6, 2] = ((u + 1) * (v + 1)) / 8
        self.dN[7, 2] = ((1 - u) * (v + 1)) / 8


class HexTwenty:
    r"""20-node brick element.

    Node numbering follows:

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
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 8), dtype=np.int64)
        self.N = np.zeros(20)
        self.dN = np.zeros((20, 3))

    def get_surfaces(self) -> None:
        """Set the local node indices of each surface of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 8), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each surface, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> int:
        """Number of elements connected to an interior corner node of a structured mesh."""
        return 20

    @property
    def n_boundary_nodes(self) -> int:
        """Number of nodes of one surface."""
        return 8

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return True

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 8

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "hexa"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u, v and w of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]
        w = natural_coordinates[2]

        # shape functions
        self.N[0] = 1. / 8. * (1 - u) * (1 - v) * (1 - w) * (-u - v - w - 2)
        self.N[1] = 1. / 8. * (1 + u) * (1 - v) * (1 - w) * (u - v - w - 2)
        self.N[2] = 1. / 8. * (1 + u) * (1 + v) * (1 - w) * (u + v - w - 2)
        self.N[3] = 1. / 8. * (1 - u) * (1 + v) * (1 - w) * (-u + v - w - 2)
        self.N[4] = 1. / 8. * (1 - u) * (1 - v) * (1 + w) * (-u - v + w - 2)
        self.N[5] = 1. / 8. * (1 + u) * (1 - v) * (1 + w) * (u - v + w - 2)
        self.N[6] = 1. / 8. * (1 + u) * (1 + v) * (1 + w) * (u + v + w - 2)
        self.N[7] = 1. / 8. * (1 - u) * (1 + v) * (1 + w) * (-u + v + w - 2)
        self.N[8] = 1. / 4. * (1 - u**2) * (1 - v) * (1 - w)
        self.N[9] = 1. / 4. * (1 - u) * (1 - v**2) * (1 - w)
        self.N[10] = 1. / 4. * (1 - u) * (1 - v) * (1 - w**2)
        self.N[11] = 1. / 4. * (1 + u) * (1 - v**2) * (1 - w)
        self.N[12] = 1. / 4. * (1 + u) * (1 - v) * (1 - w**2)
        self.N[13] = 1. / 4. * (1 - u**2) * (1 + v) * (1 - w)
        self.N[14] = 1. / 4. * (1 + u) * (1 + v) * (1 - w**2)
        self.N[15] = 1. / 4. * (1 - u) * (1 + v) * (1 - w**2)
        self.N[16] = 1. / 4. * (1 - u**2) * (1 - v) * (1 + w)
        self.N[17] = 1. / 4. * (1 - u) * (1 - v**2) * (1 + w)
        self.N[18] = 1. / 4. * (1 + u) * (1 - v**2) * (1 + w)
        self.N[19] = 1. / 4. * (1 - u**2) * (1 + v) * (1 + w)

        # derivative in u
        self.dN[0, 0] = -((1 - v) * (1 - w) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
        self.dN[1, 0] = ((1 - v) * (1 - w) * (-w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (1 - w)) / 8
        self.dN[2, 0] = ((v + 1) * (1 - w) * (-w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (1 - w)) / 8
        self.dN[3, 0] = -((v + 1) * (1 - w) * (-w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (1 - w)) / 8
        self.dN[4, 0] = -((1 - v) * (w + 1) * (w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (w + 1)) / 8
        self.dN[5, 0] = ((1 - v) * (w + 1) * (w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (w + 1)) / 8
        self.dN[6, 0] = ((v + 1) * (w + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
        self.dN[7, 0] = -((v + 1) * (w + 1) * (w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (w + 1)) / 8
        self.dN[8, 0] = -(u * (1 - v) * (1 - w)) / 2
        self.dN[9, 0] = -((1 - v**2) * (1 - w)) / 4
        self.dN[10, 0] = -((1 - v) * (1 - w**2)) / 4
        self.dN[11, 0] = ((1 - v**2) * (1 - w)) / 4
        self.dN[12, 0] = ((1 - v) * (1 - w**2)) / 4
        self.dN[13, 0] = -(u * (v + 1) * (1 - w)) / 2
        self.dN[14, 0] = ((v + 1) * (1 - w**2)) / 4
        self.dN[15, 0] = -((v + 1) * (1 - w**2)) / 4
        self.dN[16, 0] = -(u * (1 - v) * (w + 1)) / 2
        self.dN[17, 0] = -((1 - v**2) * (w + 1)) / 4
        self.dN[18, 0] = ((1 - v**2) * (w + 1)) / 4
        self.dN[19, 0] = -(u * (v + 1) * (w + 1)) / 2

        # derivative in v
        self.dN[0, 1] = -((1 - u) * (1 - w) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
        self.dN[1, 1] = -((u + 1) * (1 - w) * (-w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (1 - w)) / 8
        self.dN[2, 1] = ((u + 1) * (1 - w) * (-w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (1 - w)) / 8
        self.dN[3, 1] = ((1 - u) * (1 - w) * (-w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (1 - w)) / 8
        self.dN[4, 1] = -((1 - u) * (w + 1) * (w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (w + 1)) / 8
        self.dN[5, 1] = -((u + 1) * (w + 1) * (w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (w + 1)) / 8
        self.dN[6, 1] = ((u + 1) * (w + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
        self.dN[7, 1] = ((1 - u) * (w + 1) * (w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (w + 1)) / 8
        self.dN[8, 1] = -((1 - u**2) * (1 - w)) / 4
        self.dN[9, 1] = -((1 - u) * v * (1 - w)) / 2
        self.dN[10, 1] = -((1 - u) * (1 - w**2)) / 4
        self.dN[11, 1] = -((u + 1) * v * (1 - w)) / 2
        self.dN[12, 1] = -((u + 1) * (1 - w**2)) / 4
        self.dN[13, 1] = ((1 - u**2) * (1 - w)) / 4
        self.dN[14, 1] = ((u + 1) * (1 - w**2)) / 4
        self.dN[15, 1] = ((1 - u) * (1 - w**2)) / 4
        self.dN[16, 1] = -((1 - u**2) * (w + 1)) / 4
        self.dN[17, 1] = -((1 - u) * v * (w + 1)) / 2
        self.dN[18, 1] = -((u + 1) * v * (w + 1)) / 2
        self.dN[19, 1] = ((1 - u**2) * (w + 1)) / 4

        # derivative in w
        self.dN[0, 2] = -((1 - u) * (1 - v) * (-w - v - u - 2)) / 8 - ((1 - u) * (1 - v) * (1 - w)) / 8
        self.dN[1, 2] = -((u + 1) * (1 - v) * (-w - v + u - 2)) / 8 - ((u + 1) * (1 - v) * (1 - w)) / 8
        self.dN[2, 2] = -((u + 1) * (v + 1) * (-w + v + u - 2)) / 8 - ((u + 1) * (v + 1) * (1 - w)) / 8
        self.dN[3, 2] = -((1 - u) * (v + 1) * (-w + v - u - 2)) / 8 - ((1 - u) * (v + 1) * (1 - w)) / 8
        self.dN[4, 2] = ((1 - u) * (1 - v) * (w - v - u - 2)) / 8 + ((1 - u) * (1 - v) * (w + 1)) / 8
        self.dN[5, 2] = ((u + 1) * (1 - v) * (w - v + u - 2)) / 8 + ((u + 1) * (1 - v) * (w + 1)) / 8
        self.dN[6, 2] = ((u + 1) * (v + 1) * (w + v + u - 2)) / 8 + ((u + 1) * (v + 1) * (w + 1)) / 8
        self.dN[7, 2] = ((1 - u) * (v + 1) * (w + v - u - 2)) / 8 + ((1 - u) * (v + 1) * (w + 1)) / 8
        self.dN[8, 2] = -((1 - u**2) * (1 - v)) / 4
        self.dN[9, 2] = -((1 - u) * (1 - v**2)) / 4
        self.dN[10, 2] = -((1 - u) * (1 - v) * w) / 2
        self.dN[11, 2] = -((u + 1) * (1 - v**2)) / 4
        self.dN[12, 2] = -((u + 1) * (1 - v) * w) / 2
        self.dN[13, 2] = -((1 - u**2) * (v + 1)) / 4
        self.dN[14, 2] = -((u + 1) * (v + 1) * w) / 2
        self.dN[15, 2] = -((1 - u) * (v + 1) * w) / 2
        self.dN[16, 2] = ((1 - u**2) * (1 - v)) / 4
        self.dN[17, 2] = ((1 - u) * (1 - v**2)) / 4
        self.dN[18, 2] = ((u + 1) * (1 - v**2)) / 4
        self.dN[19, 2] = ((1 - u**2) * (v + 1)) / 4


class QuadFour:
    """4-node quad element.

    Node numbering follows:

          v
          ^
          |
    3-----------2
    |     |     |
    |     |     |
    |     +---- | --> u
    |           |
    |           |
    0-----------1
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 2), dtype=np.int64)
        self.N = np.zeros(4)
        self.dN = np.zeros((4, 2))

    def get_surfaces(self) -> None:
        """Set the local node indices of each edge of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 2), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each edge, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> int:
        """Number of elements connected to an interior node of a structured mesh."""
        return 4

    @property
    def n_boundary_nodes(self) -> int:
        """Number of nodes of one edge."""
        return 2

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return False

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 4

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "quad"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u and v of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]

        # shape functions
        self.N[0] = 1 / 4. * (1 - u) * (1 - v)
        self.N[1] = 1 / 4. * (1 + u) * (1 - v)
        self.N[2] = 1 / 4. * (1 + u) * (1 + v)
        self.N[3] = 1 / 4. * (1 - u) * (1 + v)

        # derivative in u
        self.dN[0, 0] = -(1 - v) / 4
        self.dN[1, 0] = (1 - v) / 4
        self.dN[2, 0] = (v + 1) / 4
        self.dN[3, 0] = -(v + 1) / 4

        # derivative in v
        self.dN[0, 1] = -(1 - u) / 4
        self.dN[1, 1] = -(u + 1) / 4
        self.dN[2, 1] = (u + 1) / 4
        self.dN[3, 1] = (1 - u) / 4


class QuadEight:
    """8-node quad element.

    Node numbering follows:

          v
          ^
          |
    3-----6-----2
    |     |     |
    |     |     |
    7     +---- 5  --> u
    |           |
    |           |
    0-----4-----1
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 3), dtype=np.int64)
        self.N = np.zeros(8)
        self.dN = np.zeros((8, 2))

    def get_surfaces(self) -> None:
        """Set the local node indices of each edge of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 3), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each edge, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> int:
        """Number of elements connected to an interior corner node of a structured mesh."""
        return 8

    @property
    def n_boundary_nodes(self) -> int:
        """Number of nodes of one edge."""
        return 4

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return True

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 4

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "quad"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u and v of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]

        # shape functions
        self.N[0] = 1. / 4. * (1 - u) * (1 - v) * (-u - v - 1)
        self.N[1] = 1. / 4. * (1 + u) * (1 - v) * (u - v - 1)
        self.N[2] = 1. / 4. * (1 + u) * (1 + v) * (u + v - 1)
        self.N[3] = 1. / 4. * (1 - u) * (1 + v) * (-u + v - 1)
        self.N[4] = 1. / 2. * (1 - u**2) * (1 - v)
        self.N[5] = 1. / 2. * (1 + u) * (1 - v**2)
        self.N[6] = 1. / 2. * (1 - u**2) * (1 + v)
        self.N[7] = 1. / 2. * (1 - u) * (1 - v**2)

        # derivative in u
        self.dN[0, 0] = (1 - v) * (2 * u + v) / 4
        self.dN[1, 0] = (1 - v) * (2 * u - v) / 4
        self.dN[2, 0] = (1 + v) * (2 * u + v) / 4
        self.dN[3, 0] = (1 + v) * (2 * u - v) / 4
        self.dN[4, 0] = -u * (1 - v)
        self.dN[5, 0] = (1 - v**2) / 2
        self.dN[6, 0] = -u * (1 + v)
        self.dN[7, 0] = -(1 - v**2) / 2

        # derivative in v
        self.dN[0, 1] = (1 - u) * (u + 2 * v) / 4
        self.dN[1, 1] = (1 + u) * (-u + 2 * v) / 4
        self.dN[2, 1] = (1 + u) * (u + 2 * v) / 4
        self.dN[3, 1] = (1 - u) * (-u + 2 * v) / 4
        self.dN[4, 1] = -(1 - u**2) / 2
        self.dN[5, 1] = -v * (1 + u)
        self.dN[6, 1] = (1 - u**2) / 2
        self.dN[7, 1] = -v * (1 - u)


class TriThree:
    r"""3-node triangular element.

    Node numbering follows:

    v
    ^
    |
    2
    |`\
    |  `\
    |    `\
    |      `\
    |        `\
    0----------1 --> u
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 2), dtype=np.int64)
        self.N = np.zeros(3)
        self.dN = np.zeros((3, 2))

    def get_surfaces(self) -> None:
        """Set the local node indices of each edge of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 2), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each edge, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> Optional[int]:
        """Number of elements connected to an interior node; not defined for an unstructured mesh."""
        return None

    @property
    def n_boundary_nodes(self) -> Optional[int]:
        """Number of nodes of one edge; not defined yet."""
        return None

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return False

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 3

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "tri"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u and v of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]

        # shape functions
        self.N[0] = (1 - u - v)
        self.N[1] = u
        self.N[2] = v

        # derivative in u
        self.dN[0, 0] = -1
        self.dN[1, 0] = 1
        self.dN[2, 0] = 0

        # derivative in v
        self.dN[0, 1] = -1
        self.dN[1, 1] = 0
        self.dN[2, 1] = 1


class TriSix:
    r"""6-node triangular element.

    Node numbering follows:

    v
    ^
    |
    2
    |`\
    |  `\
    5    `4
    |      `\
    |        `\
    0-----3----1 --> u
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 3), dtype=np.int64)
        self.N = np.zeros(6)
        self.dN = np.zeros((6, 2))

    def get_surfaces(self) -> None:
        """Set the local node indices of each edge of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 3), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each edge, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> Optional[int]:
        """Number of elements connected to an interior node; not defined for an unstructured mesh."""
        return None

    @property
    def n_boundary_nodes(self) -> Optional[int]:
        """Number of nodes of one edge; not defined yet."""
        return None

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return True

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 3

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "tri"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u and v of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]

        # shape functions
        self.N[0] = (2 * (1 - u - v) - 1) * (1 - u - v)
        self.N[1] = (2 * u - 1) * u
        self.N[2] = (2 * v - 1) * v
        self.N[3] = 4 * (1 - u - v) * u
        self.N[4] = 4 * u * v
        self.N[5] = 4 * (1 - u - v) * v

        # derivative in u
        self.dN[0, 0] = 1 - 4 * (-v - u + 1)
        self.dN[1, 0] = 4 * u - 1
        self.dN[2, 0] = 0
        self.dN[3, 0] = 4 * (-v - u + 1) - 4 * u
        self.dN[4, 0] = 4 * v
        self.dN[5, 0] = -4 * v

        # derivative in v
        self.dN[0, 1] = 1 - 4 * (-v - u + 1)
        self.dN[1, 1] = 0
        self.dN[2, 1] = 4 * v - 1
        self.dN[3, 1] = -4 * u
        self.dN[4, 1] = 4 * u
        self.dN[5, 1] = 4 * (-v - u + 1) - 4 * v


class TetraFour:
    r"""4-node tetrahedron element.

    Node numbering follows:

                       v
                     .
                   ,/
                  /
               2
             ,/|`\
           ,/  |  `\
         ,/    '.   `\
       ,/       |     `\
     ,/         |       `\
    0-----------'.--------1 --> u
     `\.         |      ,/
        `\.      |    ,/
           `\.   '. ,/
              `\. |/
                 `3
                    `\.
                       ` w
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 3), dtype=np.int64)
        self.N = np.zeros(4)
        self.dN = np.zeros((4, 3))

    def get_surfaces(self) -> None:
        """Set the local node indices of each surface of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 3), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each surface, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> Optional[int]:
        """Number of elements connected to an interior node; not defined for an unstructured mesh."""
        return None

    @property
    def n_boundary_nodes(self) -> Optional[int]:
        """Number of nodes of one surface; not defined yet."""
        return None

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return False

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 4

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "tetra"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u, v and w of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]
        w = natural_coordinates[2]

        # shape functions
        self.N[0] = (1 - u - v - w)
        self.N[1] = u
        self.N[2] = v
        self.N[3] = w

        # derivative in u
        self.dN[0, 0] = -1
        self.dN[1, 0] = 1
        self.dN[2, 0] = 0
        self.dN[3, 0] = 0

        # derivative in v
        self.dN[0, 1] = -1
        self.dN[1, 1] = 0
        self.dN[2, 1] = 1
        self.dN[3, 1] = 0

        # derivative in w
        self.dN[0, 2] = -1
        self.dN[1, 2] = 0
        self.dN[2, 2] = 0
        self.dN[3, 2] = 1


class TetraTen:
    r"""10-node tetrahedron element.

    Node numbering follows:

                       v
                     .
                   ,/
                  /
               2
             ,/|`\
           ,/  |  `\
         ,6    '.   `5
       ,/       8     `\
     ,/         |       `\
    0--------4--'.--------1 --> u
     `\.         |      ,/
        `\.      |    ,9
           `7.   '. ,/
              `\. |/
                 `3
                    `\.
                       ` w
    """

    def __init__(self) -> None:
        """Allocate the shape functions and their derivatives."""
        self.__surfaces = np.empty((0, 6), dtype=np.int64)
        self.N = np.zeros(10)
        self.dN = np.zeros((10, 3))

    def get_surfaces(self) -> None:
        """Set the local node indices of each surface of the element (not implemented yet)."""
        self.__surfaces = np.empty((0, 6), dtype=np.int64)

    @property
    def surfaces(self) -> npt.NDArray[np.int64]:
        """Local node indices of each surface, after get_surfaces has been called."""
        return self.__surfaces

    @property
    def max_element_connections(self) -> Optional[int]:
        """Number of elements connected to an interior node; not defined for an unstructured mesh."""
        return None

    @property
    def n_boundary_nodes(self) -> Optional[int]:
        """Number of nodes of one surface; not defined yet."""
        return None

    @property
    def is_quadratic(self) -> bool:
        """Whether the element has midside nodes."""
        return True

    @property
    def n_corner_nodes(self) -> int:
        """Number of corner nodes."""
        return 4

    @property
    def integration_type(self) -> str:
        """Type of Gauss integration."""
        return "tetra"

    def shape_functions(self, natural_coordinates: npt.NDArray[np.float64]) -> None:
        """Evaluate the shape functions and their derivatives at one point.

        Args:
            natural_coordinates: natural coordinates u, v and w of the point [-].
        """
        u = natural_coordinates[0]
        v = natural_coordinates[1]
        w = natural_coordinates[2]
        x = (1 - u - v - w)

        # shape functions
        self.N[0] = (2 * x - 1) * x
        self.N[1] = (2 * u - 1) * u
        self.N[2] = (2 * v - 1) * v
        self.N[3] = (2 * w - 1) * w
        self.N[4] = 4 * u * x
        self.N[5] = 4 * u * v
        self.N[6] = 4 * v * x
        self.N[7] = 4 * w * x
        self.N[8] = 4 * v * w
        self.N[9] = 4 * u * w

        # derivative in u
        self.dN[0, 0] = 4 * u + 4 * v + 4 * w - 3
        self.dN[1, 0] = 4 * u - 1
        self.dN[2, 0] = 0
        self.dN[3, 0] = 0
        self.dN[4, 0] = 4 * (-w - v - u + 1) - 4 * u
        self.dN[5, 0] = 4 * v
        self.dN[6, 0] = -4 * v
        self.dN[7, 0] = -4 * w
        self.dN[8, 0] = 0
        self.dN[9, 0] = 4 * w

        # derivative in v
        self.dN[0, 1] = 4 * u + 4 * v + 4 * w - 3
        self.dN[1, 1] = 0
        self.dN[2, 1] = 4 * v - 1
        self.dN[3, 1] = 0
        self.dN[4, 1] = -4 * u
        self.dN[5, 1] = 4 * u
        self.dN[6, 1] = 4 * (-w - v - u + 1) - 4 * v
        self.dN[7, 1] = -4 * w
        self.dN[8, 1] = 4 * w
        self.dN[9, 1] = 0

        # derivative in w
        self.dN[0, 2] = 4 * u + 4 * v + 4 * w - 3
        self.dN[1, 2] = 0
        self.dN[2, 2] = 0
        self.dN[3, 2] = 4 * w - 1
        self.dN[4, 2] = -4 * u
        self.dN[5, 2] = 0
        self.dN[6, 2] = -4 * v
        self.dN[7, 2] = 4 * (-w - v - u + 1) - 4 * w
        self.dN[8, 2] = 4 * v
        self.dN[9, 2] = 4 * u


def create_element(elem_type: str) -> Element:
    """Create an element from its scatter name.

    Args:
        elem_type: element type: line2, line3, quad4, quad8, tri3, tri6, hexa8, hexa20, tetra4 or tetra10.

    Returns:
        Element with its shape functions not yet evaluated.

    Raises:
        ValueError: if the element type is not supported.
    """
    element_classes: Dict[str, Callable[[], Element]] = {
        "line2": LineTwo,
        "line3": LineThree,
        "quad4": QuadFour,
        "quad8": QuadEight,
        "tri3": TriThree,
        "tri6": TriSix,
        "hexa8": HexEight,
        "hexa20": HexTwenty,
        "tetra4": TetraFour,
        "tetra10": TetraTen,
    }
    if elem_type not in element_classes:
        raise ValueError(f"Element type {elem_type} is not supported.")
    return element_classes[elem_type]()
