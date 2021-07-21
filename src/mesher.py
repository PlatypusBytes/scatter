# import packages
import os
import sys
import numpy as np
# import scatter packages
from src import utils

class HexEight:
    """
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
    """
    def __init__(self):
        self.__surfaces = None

    def get_surfaces(self):
        """
        Get node index arrays for each surface of the element
        """
        self.__surfaces = np.array([[0,1,2,3],[0,1,4,5],[4,5,6,7],[2,3,6,7],[0,3,4,7],[1,2,5,6]])

    @property
    def surfaces(self):
        return self.__surfaces

    @property
    def max_element_connections(self):
        return 8

    @property
    def n_boundary_nodes(self):
        return 4

    @property
    def is_quadratic(self):
        return False

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
        elem, nb_elem = utils.search_idx(data, r"$Elements", r"$EndElements")
        elem = [list(map(int, i)) for i in elem]
        elem = np.array(elem)

        # check if element type is 5 or 17
        element_type = set(elem[:, 1])
        if not all(x in [5, 17] for x in element_type):
            sys.exit("ERROR: Element type not supported")

        # add element type to self
        if all(x == 5 for x in element_type):
            self.element_type = 'hexa8'
        elif all(x == 17 for x in element_type):
            self.element_type = 'hexa20'

        # add variables to self
        self.nodes = nodes
        self.elem = elem[:, 5:]
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
            # assuming that the three points are non-collinear
            plane, direction = utils.define_plane(nodes[0], nodes[1], nodes[2])

            residual = self.nodes[:, 1] * plane[0] + self.nodes[:, 2] * plane[1] + self.nodes[:, 3] * plane[2] - plane[
                3]

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
