"""Read gmsh meshes and number the equations of the model."""
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import numpy.typing as npt

from scatter import utils
from scatter.element_types import HexEight
from scatter.rose_utils import RoseUtils

# per boundary: the condition of each dof as a string, and the points that define the boundary plane or line [m]
BoundaryConditions = Dict[str, Sequence[Any]]


class ReadMesh:
    """Mesh geometry, boundary conditions and equation numbering of a model."""

    def __init__(self, file_name: str) -> None:
        """Check the mesh file and initialise the mesh structure.

        Args:
            file_name: gmsh mesh file (.msh, format 2.2).

        Raises:
            SystemExit: if the mesh file does not exist.
            SystemExit: if the mesh file does not have the .msh extension.
        """
        if not os.path.isfile(file_name):
            sys.exit("ERROR: Mesh file does not exit.")
        self.file_name = file_name

        if os.path.splitext(self.file_name)[-1] != ".msh":
            sys.exit("ERROR: Mesh file is not a valid file")

        # node number and x, y, z coordinates of each node
        self.nodes: npt.NDArray[np.float64] = np.empty((0, 4))
        # node numbers of each element
        self.elem: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        # node numbers of each rose connection element
        self.rose_elem: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        # node numbers of the rose connection nodes
        self.rose_nodes: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        # node numbers of each boundary surface
        self.boundary_elem: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        # number of nodes per element
        self.nb_nodes_elem = 0
        # physical names: dimension, material index and material name
        self.materials: List[List[Any]] = []
        # boundary condition type of each dof of each node: 0 free, 1 fixed, 2 absorbing
        self.BC: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        # whether the boundary of each dof of each node is perpendicular to that dof
        self.BC_dir: npt.NDArray[np.int64] = np.empty((0, 0), dtype=np.int64)
        # number of equations
        self.number_eq = 0
        # boundary condition name of each dof of each node: Normal, Fixed or Absorb
        self.type_BC: npt.NDArray[np.str_] = np.empty((0, 0), dtype="<U6")
        # equation number of each dof of each node, NaN for fixed dofs
        self.eq_nb_dof: npt.NDArray[np.float64] = np.empty((0, 0))
        # equation number of each dof of each element, NaN for fixed dofs
        self.eq_nb_elem: npt.NDArray[np.float64] = np.empty((0, 0))
        # scatter equation numbers that are connected to the rose model
        self.eq_nb_dof_rose_nodes: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        # rose equation numbers that are connected to the scatter model
        self.rose_eq_nb: List[int] = []
        # boundary condition name of each dof of each element
        self.type_BC_elem: npt.NDArray[np.str_] = np.empty((0, 0), dtype="<U6")
        # element type of the mesh
        self.element_type = ""
        # element type of the boundary of the mesh, used for the absorbing boundaries
        self.lower_element_type = ""
        # number of nodes of the boundary elements
        self.nb_nodes_lower_elem = 0
        # material index of each element
        self.materials_index: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        # dimension of the problem
        self.dimension = 3

    def read_gmsh(self) -> None:
        r"""Read the gmsh mesh file.

        The file is organised as:
            $MeshFormat
            version-number file-type data-size
            $EndMeshFormat
            $PhysicalNames
            number-of-names
            physical-dimension physical-number "physical-name"
            …
            $EndPhysicalNames
            $Nodes
            number-of-nodes
            node-number x-coord y-coord z-coord
            …
            $EndNodes
            $Elements
            number-of-elements
            elm-number elm-type number-of-tags < tag > … node-number-list
            …
            $EndElements

        The supported gmsh element types are 2 (tri3), 9 (tri6), 3 (quad4), 5 (hexa8), 17 (hexa20), 4 (tetra4) and
        11 (tetra10); the node numbering follows `element_types`. Line elements of the physical group "rose" connect
        the model to rose.

        Raises:
            SystemExit: if the mesh has an element type that is not supported.
        """
        with open(self.file_name, 'r') as f:
            data = f.readlines()

        # read Nodes
        node_entries, _ = utils.search_idx(data, r"$Nodes", r"$EndNodes")
        nodes = np.array(node_entries, dtype=float)

        # read PhysicalNames, with the material index as integer
        names, _ = utils.search_idx(data, r"$PhysicalNames", r"$EndPhysicalNames")
        for name in names:
            name[1] = int(name[1])

        # read Elements
        element_entries, _ = utils.search_idx(data, r"$Elements", r"$EndElements")
        elem = [[int(i) for i in entry] for entry in element_entries]

        # retrieve rose elements
        rose_idx = None
        for name in names:
            if name[2] == "rose":
                rose_idx = name[1]
                rose_elements = np.array([el for el in elem if el[3] == rose_idx])
                self.rose_elem = rose_elements[:, 5:]
                break

        if rose_idx is not None:
            rose_nodes = np.array([el[5:] for el in elem if len(el) == len(rose_elements[0]) and el in rose_elements])
            self.rose_nodes = np.unique(rose_nodes)

            # get all geometry elements which are not part of ROSE
            geometry_elem = np.array([el for el in elem if el[3] != rose_idx])
        else:
            geometry_elem = np.array(elem)

        element_type = set(geometry_elem[:, 1])
        if not all(x in [2, 9, 3, 5, 17, 4, 11] for x in element_type):
            sys.exit("ERROR: Element type not supported")

        # element type, its boundary element type for the absorbing boundaries, and the dimension
        if all(x == 2 for x in element_type):
            self.element_type = 'tri3'
            self.lower_element_type = 'line2'
            self.nb_nodes_lower_elem = 2
            self.dimension = 2
        elif all(x == 9 for x in element_type):
            self.element_type = 'tri6'
            self.lower_element_type = 'line3'
            self.nb_nodes_lower_elem = 3
            self.dimension = 2
        elif all(x == 3 for x in element_type):
            self.element_type = 'quad4'
            self.lower_element_type = 'line2'
            self.nb_nodes_lower_elem = 2
            self.dimension = 2
        elif all(x == 5 for x in element_type):
            self.element_type = 'hexa8'
            self.lower_element_type = 'quad4'
            self.nb_nodes_lower_elem = 4
            self.dimension = 3
        elif all(x == 17 for x in element_type):
            self.element_type = 'hexa20'
            self.lower_element_type = 'quad8'
            self.nb_nodes_lower_elem = 8
            self.dimension = 3
        elif all(x == 4 for x in element_type):
            self.element_type = 'tetra4'
            self.lower_element_type = 'tri3'
            self.nb_nodes_lower_elem = 3
            self.dimension = 3
        elif all(x == 11 for x in element_type):
            self.element_type = 'tetra10'
            self.lower_element_type = "tri6"
            self.nb_nodes_lower_elem = 6
            self.dimension = 3

        self.nodes = nodes
        self.elem = geometry_elem[:, 5:]
        self.materials_index = geometry_elem[:, 3]
        self.materials = names
        self.nb_nodes_elem = len(self.elem[0])

    def read_bc(self, bc: BoundaryConditions) -> None:
        """Determine the boundary conditions of all nodes.

        In 3D each boundary is a plane through the first three (non-collinear) points; in 2D it is the line segment
        between the first two points.

        Args:
            bc: per boundary name, the condition of each dof ("0" free, "1" fixed, "2" absorbing, e.g. "020") and
                the points that define the boundary [m].

        Raises:
            SystemExit: if the dimension of the model is not 2 or 3.
        """
        self.BC = np.zeros((len(self.nodes), self.dimension), dtype=int)
        self.BC_dir = np.zeros((len(self.nodes), self.dimension), dtype=int)

        for boundary in bc.values():
            typ, points = boundary[0], boundary[1]
            if self.dimension == 3:
                # nodes within the plane
                plane, direction = utils.define_plane(points[0], points[1], points[2])
                residual = (self.nodes[:, 1] * plane[0] + self.nodes[:, 2] * plane[1] + self.nodes[:, 3] * plane[2] -
                            plane[3])
            elif self.dimension == 2:
                # nodes on the line segment
                vector = np.array(points[1]) - np.array(points[0])
                normal = np.array([-vector[1], vector[0]])
                direction = np.abs(normal) / np.linalg.norm(normal)
                residual = (utils.calculate_distance(points[0], self.nodes[:, 1:]) +
                            utils.calculate_distance(points[1], self.nodes[:, 1:]) -
                            utils.calculate_distance(points[0], points[1]))
            else:
                sys.exit(f"ERROR: dimension: {self.dimension}, is  not supported")

            indices = np.where(np.isclose(residual, 0., atol=1.e-5))[0]

            # the strictest boundary condition wins
            for idx in indices:
                for j, val in enumerate(typ):
                    self.BC[idx, j] = max(self.BC[idx, j], int(val))
                    self.BC_dir[idx, j] = max(self.BC_dir[idx, j], abs(int(direction[j])))

    def mapping(self) -> None:
        """Define the equation number and the type of boundary condition of each dof of each node.

        Raises:
            SystemExit: if a boundary condition is not 0, 1 or 2.
        """
        self.eq_nb_dof = np.zeros((len(self.nodes), self.dimension))
        self.type_BC = np.full((len(self.nodes), self.dimension), "Normal")

        equation_nb = 0
        for i in range(len(self.BC)):
            for j in range(len(self.BC[i])):
                if self.BC[i][j] == 0:
                    self.eq_nb_dof[i, j] = equation_nb
                    self.type_BC[i, j] = "Normal"
                    equation_nb += 1
                elif self.BC[i][j] == 1:
                    self.eq_nb_dof[i, j] = np.nan
                    self.type_BC[i, j] = "Fixed"
                elif self.BC[i][j] == 2:
                    self.eq_nb_dof[i, j] = equation_nb
                    self.type_BC[i, j] = "Absorb"
                    equation_nb += 1
                else:
                    sys.exit("Error in the boundary condition definition. \n"
                             f"{self.BC[i][j]} is not a valid boundary condition.")
        self.number_eq = equation_nb

    def connectivities(self) -> None:
        """Define the equation number and the type of boundary condition of each dof of each element."""
        node_rows = self.get_node_rows(self.elem)
        self.eq_nb_elem = self.eq_nb_dof[node_rows].reshape(self.elem.shape[0], -1)
        self.type_BC_elem = self.type_BC[node_rows].reshape(self.elem.shape[0], -1)

    def get_node_rows(self, node_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Find the rows of the node list that hold the given node numbers.

        Args:
            node_ids: gmsh node numbers, of any shape.

        Returns:
            Row index in the node list of each node number, with the same shape as `node_ids`.

        Raises:
            ValueError: if a node number is not in the node list.
        """
        node_numbers = self.nodes[:, 0].astype(np.int64)
        ids = np.asarray(node_ids).astype(np.int64)

        # node numbers can have gaps, so map them through a lookup table instead of assuming row = number - 1
        lookup = np.full(max(int(np.max(node_numbers)), int(np.max(ids, initial=0))) + 1, -1, dtype=np.int64)
        lookup[node_numbers] = np.arange(len(node_numbers))
        rows = lookup[ids]
        if np.any(rows < 0) or np.any(ids < 0):
            missing = np.unique(ids[(rows < 0) | (ids < 0)])
            raise ValueError(f"Nodes {missing.tolist()} are not in the node list.")
        return rows

    def get_mesh_edges(self) -> None:
        """Find the boundary surfaces of the mesh; only implemented for hexa8 elements."""
        if self.element_type != 'hexa8':
            return
        element_type = HexEight()
        element_type.get_surfaces()

        # a node is on the boundary if it is connected to fewer elements than an interior node
        node_rows = self.get_node_rows(self.elem)
        nb_attached_elements = np.bincount(node_rows.ravel(), minlength=self.nodes.shape[0])
        is_node_boundary_node = nb_attached_elements < element_type.max_element_connections

        # boundary elements are the elements connected to at least one boundary node
        boundary_elements = np.where(is_node_boundary_node[node_rows].any(axis=1))[0]

        boundary_surfaces = []
        for i in boundary_elements:
            is_node_boundary = is_node_boundary_node[node_rows[i]]

            # if more than one boundary surface is present at boundary element, find all surfaces
            if sum(is_node_boundary) > element_type.n_boundary_nodes:
                for surface in element_type.surfaces:
                    if all(is_node_boundary[surface]):
                        boundary_surfaces.append(self.elem[i][surface])
            else:
                boundary_surfaces.append(self.elem[i][is_node_boundary])

        self.boundary_elem = np.unique(np.array(boundary_surfaces), axis=0)

    def get_top_surface(self) -> npt.NDArray[np.int64]:
        """Get the top surface elements from the boundary elements, with y as the vertical coordinate.

        Only works if the side and bottom boundaries are horizontal and vertical.

        Returns:
            Node numbers of each top surface element.
        """
        epsilon = 1e-10

        # centroids of all boundary elements
        elem_coordinates = self.nodes[:, 1:][self.boundary_elem - 1, :]
        centroids = np.array([utils.calculate_centroid(elem) for elem in elem_coordinates])

        min_x, max_x = min(centroids[:, 0]), max(centroids[:, 0])
        min_y = min(centroids[:, 1])
        min_z, max_z = min(centroids[:, 2]), max(centroids[:, 2])

        # boundary element is a top surface element if centroid coordinates lay within the xyz limits
        is_surface_elem = ((centroids[:, 0] > min_x + epsilon) & (centroids[:, 0] < max_x - epsilon) &
                           (centroids[:, 1] > min_y + epsilon) & (centroids[:, 2] > min_z + epsilon) &
                           (centroids[:, 2] < max_z - epsilon))

        return self.boundary_elem[is_surface_elem]

    def rose_connectivities(self, rose_model: Any) -> None:
        """Connect scatter to a rose model, through the vertical dof of the rose connection nodes.

        Args:
            rose_model: rose coupled train track model.

        Raises:
            SystemExit: if the number of rose and scatter connection dofs differ.
        """
        vertical_index = 1
        node_rows = self.get_node_rows(self.rose_elem)

        # unique coordinates of the rose nodes, in order of appearance
        node_coords = self.nodes[node_rows, 1:].reshape(-1, self.nodes.shape[1] - 1)
        indices = np.unique(node_coords, axis=0, return_index=True)[1]
        node_coords = node_coords[np.sort(indices)]

        # sort on x-coordinate, then z-coordinate
        sorted_coords_indices = np.lexsort((node_coords[:, 0], node_coords[:, 2]))

        # unique equation numbers, in order of appearance
        stack = self.eq_nb_dof[node_rows, vertical_index].ravel()
        unique_indices = np.unique(stack, return_index=True)[1]
        sorted_eq_nb_dof_rose = stack[np.sort(unique_indices)].astype(int)

        self.eq_nb_dof_rose_nodes = sorted_eq_nb_dof_rose[sorted_coords_indices]
        self.rose_eq_nb = RoseUtils.get_bottom_boundary(rose_model)

        if len(self.eq_nb_dof_rose_nodes) != len(self.rose_eq_nb):
            sys.exit(f"ERROR: rose connectivities, {len(self.rose_eq_nb)}, does not have the same size as scatter "
                     f"connectivities, {len(self.eq_nb_dof_rose_nodes)}")
