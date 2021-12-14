# import packages
import os
import sys
import numpy as np
# import scatter packages
from src import utils
from src.element_types import HexEight


class ReadMesh:
    def __init__(self, file_name: str) -> None:
        """
        Reads the mesh and creates the mesh geometry structure.

        Parameters
        ----------
        :param file_name: filename of gmsh file
        """
        # check if file name exists
        if os.path.isfile(file_name):
            self.file_name = file_name
        else:
            # if does not exist exit with error message
            sys.exit("ERROR: Mesh file does not exit.")

        # check if the file has the correct extension
        if os.path.splitext(self.file_name)[-1] != ".msh":
            sys.exit("ERROR: Mesh file is not a valid file")

        # define variables
        self.nodes = []  # node list
        self.elem = []  # element list
        self.boundary_elem = [] # boundary element list
        self.nb_nodes_elem = []  # number of nodes
        self.materials = []  # materials
        self.BC = []  # Boundary conditions for each node
        self.BC_dir = []  # Perpendicular direction of the BC for each node
        self.number_eq = []  # number of equations
        self.type_BC = []  # type of BC for each dof in node list
        self.type_BC_dir = []  # perpendicular direction of BC for each dof in node list
        self.eq_nb_dof = []  # number of equation for each dof in node list
        self.eq_nb_elem = []  # list containing equation number for the dof's per element
        self.eq_nb_dof_rose_nodes = []  # array containing all equation numbers which are connected to the rose model
        self.rose_eq_nb = []  # array containing all equation numbers of rose which are connected to the scatter model
        self.type_BC_elem = []  # list containing type of BC for the dof's per element
        self.element_type = []  # element type
        self.materials_index = []  # list containing material index for each element
        self.dimension = 3  # Dimension of the problem
        return

    def read_gmsh(self) -> None:
        r"""
        Read gmsh mesh

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

        The element type that are accepted are 5 (8 node volume element) and 17 (20 node volume element).
        There is also the need to use quadrilateral elements for absorbing boundary conditions
        (4 node quad element for 8 node volume element and 8 node quad element for 20 node volume element).
        The node numbering is as follow:

        8 node volume:
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

        20 node volume:
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

        4 node quad:
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

        8 node quad:
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

        # read the file
        with open(self.file_name, 'r') as f:
            data = f.readlines()

        # read Nodes
        nodes, nb_nodes = utils.search_idx(data, r"$Nodes", r"$EndNodes")
        nodes = np.array(nodes)

        # read PhysicalNames
        names, nb_names = utils.search_idx(data, r"$PhysicalNames", r"$EndPhysicalNames")

        # read Elements
        elem, _ = utils.search_idx(data, r"$Elements", r"$EndElements")
        elem = [list(map(int, i)) for i in elem]

        # retrieve rose elements
        rose_elements = []
        for name in names:
            if name[2] == "rose":
                rose_idx = name[1]
                rose_elements = [el for el in elem if el[3] == rose_idx]
                break

        # get all geometry elements which are not part of ROSE
        geometry_elem = [el for el in elem if el not in rose_elements]

        elem = np.array(geometry_elem)
        rose_elem = np.array(rose_elements)
        # nb_elem = elem.shape[0]

        # check if element type is 5 or 17
        element_type = set(elem[:, 1])
        if not all(x in [3, 5, 17] for x in element_type):
            sys.exit("ERROR: Element type not supported")

        # add element type to self
        if all(x == 3 for x in element_type):
            self.element_type = 'quad4'
            self.dimension = 2
        if all(x == 5 for x in element_type):
            self.element_type = 'hexa8'
            self.dimension = 3
        elif all(x == 17 for x in element_type):
            self.element_type = 'hexa20'
            self.dimension = 3

        # add variables to self
        self.nodes = nodes
        self.elem = elem[:, 5:]
        self.rose_elem = rose_elem[:,5:]
        self.materials_index = elem[:, 3]
        self.materials = names
        self.nb_nodes_elem = len(self.elem[0])

        return

    def read_bc(self, bc: dict) -> None:
        r"""
        Determines boundary conditions for all nodes.
        Assumes that the three coordinates are non-collinear.

        Parameters
        ----------
        :param bc: Dictionary with boundary conditions
        """

        # variables generation
        self.BC = np.zeros((len(self.nodes), self.dimension), dtype=int)
        self.BC_dir = np.zeros((len(self.nodes), self.dimension), dtype=int)

        # for each boundary plane
        for boundary in bc:
            type = bc[boundary][0]
            nodes = bc[boundary][1]

            # find all the nodes that are within this plane
            if self.dimension == 3:
                # assuming that the three points are non-collinear
                plane, direction = utils.define_plane(nodes[0], nodes[1], nodes[2])

                residual = self.nodes[:, 1] * plane[0] + self.nodes[:, 2] * plane[1] + self.nodes[:, 3] * plane[2] - plane[3]

            # find all the nodes that are on a line
            elif self.dimension == 2:
                vector = np.array(nodes[1]) - np.array(nodes[0])
                direction = np.array([-vector[1], vector[0]])
                residual = (utils.calculate_distance(nodes[0], self.nodes[:, 1:]) + utils.calculate_distance(nodes[1], self.nodes[:, 1:]) - utils.calculate_distance(nodes[0], nodes[1]))

            else:
                sys.exit(f"ERROR: dimension: {self.dimension}, is  not supported")

            # find indexes
            indices = np.where(residual == 0.)[0]

            # assign BC type and perpendicular direction
            for idx in indices:
                for j, val in enumerate(type):
                    # chooses the maximum type of BC
                    self.BC[idx, j] = max(self.BC[idx, j], int(val))
                    self.BC_dir[idx, j] = max(self.BC_dir[idx, j], int(direction[j]))
        return

    def mapping(self) -> None:
        r"""
        Define equation numbers and type of boundary condition for each *dof* in the node list
        """

        # initialise variables
        self.eq_nb_dof = np.zeros((len(self.nodes), self.dimension))
        self.type_BC = np.full((len(self.nodes), self.dimension), "Normal")
        self.type_BC_dir = np.zeros((len(self.nodes), self.dimension))

        # equation number:
        equation_nb = 0

        # loop in all all the nodes
        for i in range(len(self.BC)):
            # loop in all the dof of a node
            for j in range(len(self.BC[i])):
                # if BC = 0: there is no BC
                if self.BC[i][j] == 0:
                    self.eq_nb_dof[i, j] = equation_nb
                    self.type_BC[i, j] = "Normal"
                    equation_nb += int(1)
                # if it is a fixed boundary
                elif self.BC[i][j] == 1:
                    self.eq_nb_dof[i, j] = np.nan
                    self.type_BC[i, j] = "Fixed"
                # if it is an absorbing boundary
                elif self.BC[i][j] == 2:
                    self.eq_nb_dof[i, j] = equation_nb
                    self.type_BC[i, j] = "Absorb"
                    equation_nb += int(1)
                else:
                    sys.exit("Error in the boundary condition definition. \n"
                             f"{self.BC[i][j]} is not a valid boundary condition.")

                # add perpendicular direction
                self.type_BC_dir[i, j] = self.BC_dir[i][j]
        self.number_eq = equation_nb
        return

    def connectivities(self) -> None:
        r"""
        Define equation numbers and type of boundary condition for each *dof* in the element list
        """

        # initialise variables
        self.eq_nb_elem = np.zeros((self.elem.shape[0], self.nb_nodes_elem * self.dimension))
        self.type_BC_elem = np.full((self.elem.shape[0], self.nb_nodes_elem * self.dimension), "Normal")

        # loop element
        for i in range(self.elem.shape[0]):
            idx_nodes = [np.where(self.nodes[:, 0] == j)[0][0] for j in self.elem[i]]
            self.eq_nb_elem[i, :] = self.eq_nb_dof[idx_nodes].flatten()
            self.type_BC_elem[i, :] = self.type_BC[idx_nodes].flatten()
        return

    def get_mesh_edges(self):
        """
        Get boundary elements of the mesh, currently only boundaries of hex8 elements can be found

                    #todo add other element types
        """

        # initialise element type
        if self.element_type == 'hexa8':
            element_type = HexEight()
            element_type.get_surfaces()
        else:
            return

        # Find all nodes of the mesh which lay on the edge, also find elements which contain the corresponding nodes
        boundary_elements = []
        is_node_boundary_node = np.empty(self.nodes.shape[0], dtype=bool) * False
        for idx, node in enumerate(self.nodes):


            # find the elements which are connected to the current node
            attached_elements = np.where(self.elem == int(node[0]))[0]

            # node is boundary node if it is connected to less than the maximum of element connections
            #todo methodology does not work for quadratic elements
            if not element_type.is_quadratic:
                is_boundary_node = len(attached_elements) < element_type.max_element_connections
            else:
                return
            is_node_boundary_node[idx] = is_boundary_node

            # if node is on boundary, add connected elements to list
            if is_boundary_node:
                boundary_elements.append(attached_elements)

        # get the boundary surfaces of all the boundary elements
        boundary_surfaces = []
        # loop element connectivities of boundary nodes
        for elements in boundary_elements:
            # loop over boundary elements
            for i in elements:

                # get boundary nodes corresponding to boundary element
                is_node_boundary = np.empty(len(self.elem[i]),dtype=bool)*False
                for idx, j in enumerate(self.elem[i]):
                    if j in self.nodes[is_node_boundary_node, 0].astype(int):
                        is_node_boundary[idx] = True

                # if more than one boundary surface is present at boundary element, find all surfaces
                if sum(is_node_boundary)>element_type.n_boundary_nodes:

                    # find all boundary surfaces by comparing boundary nodes to the local node numbering
                    for surface in element_type.surfaces:
                        if all(is_node_boundary[surface]):
                            boundary_surface = self.elem[i][surface]
                            boundary_surfaces.append(boundary_surface)

                # else add boundary surface to boundary surface list
                else:
                    boundary_surface = self.elem[i][is_node_boundary]
                    boundary_surfaces.append(boundary_surface)

        # get unique boundary elements and add to boundary elem array
        self.boundary_elem = np.unique(np.array(boundary_surfaces), axis=0)

    def get_top_surface(self):
        """
        Gets the top surface elements from the boundary elements, note that the y-coordinate is considered to represent
        the vertical coordinate.

        Method only works if side and bottom boundaries are horizontal and vertical
        """
        epsilon = 1e-10

        # get all nodal coordinates
        nodal_coords = self.nodes[:,1:]

        # get all coordinates of all the boundary elements
        elem_coordinates = nodal_coords[self.boundary_elem - 1, :]

        # calculate centroids of all boundary elements
        centroids = np.array([utils.calculate_centroid(elem) for elem in elem_coordinates])

        # get x,y,z limits of all centroids
        min_x, max_x = min(centroids[:, 0]), max(centroids[:, 0])
        min_y = min(centroids[:, 1])
        min_z, max_z = min(centroids[:, 2]), max(centroids[:, 2])

        # boundary element is a top surface element if centroid coordinates lay within the xyz limits
        is_surface_elem = (centroids[:, 0] > min_x + epsilon) * (centroids[:, 0] < max_x - epsilon) * \
                          (centroids[:, 1] > min_y + epsilon) * (centroids[:, 2] > min_z + epsilon) * \
                          (centroids[:, 2] < max_z - epsilon)

        # get top surface elems
        top_surface_elems = self.boundary_elem[is_surface_elem]

        return top_surface_elems

    def rose_connectivities(self, rose_model):
        """
        Connects scatter to a rose model. Rose is connected to the vertical degree of freedom in the scatter elements.
        """

        vertical_index = 1
        eq_nb_dof_rose = []
        node_coords = []
        for elem in self.rose_elem:
            idx_nodes = [np.where(self.nodes[:, 0] == j)[0][0] for j in elem]
            # eq_nb_dof_rose_nodes = self.eq_nb_dof[elem][:,vertical_index]
            eq_nb_dof_rose.append(self.eq_nb_dof[idx_nodes][:, vertical_index])
            node_coords.append(self.nodes[idx_nodes][:,1:])

        node_coords = np.array(node_coords)

        node_coords = node_coords.reshape((node_coords.shape[0]*node_coords.shape[1], node_coords.shape[2]))
        # np.array(node_coords).reshape((40, 3))
        # np.unique(np.array(node_coords).reshape((40, 3)),axis=1)
        indices = np.unique(node_coords, axis=0, return_index=True)[1]
        node_coords = np.array([node_coords[index] for index in sorted(indices)])

        # sort on x-coordinate, then z-coordinate
        sorted_coords_indices = np.lexsort((node_coords[:, 0], node_coords[:, 2]))

        # indexes = np.unique(a, return_index=True)[1]
        np.unique(node_coords, return_index=True)
        # add all unqique equation number to one array
        #todo sort based on coordinate
        self.eq_nb_dof_rose_nodes = np.unique(eq_nb_dof_rose).astype(int)[sorted_coords_indices]
        self.rose_eq_nb = RoseUtils.get_bottom_boundary(rose_model)

        #check if rose and scatter connectivities have the same size
        if len(self.eq_nb_dof_rose_nodes) != len(self.rose_eq_nb):
            sys.exit(f"ERROR: rose connectivities, {len(self.rose_eq_nb)}, does not have the same size as scatter connectivities, {len(self.eq_nb_dof_rose_nodes)}")
