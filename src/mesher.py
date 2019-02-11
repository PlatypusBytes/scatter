class ReadMesh:

    def __init__(self, file_name):
        import os
        import sys

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
        self.nb_nodes_elem = []  # number of nodes
        self.materials = []  # materials
        self.BC = []  # BC list
        self.number_eq = []  # number of equations
        self.eq_nb_dof = []  # list containing equation number for the dof's per node
        self.eq_nb_elem = []  # list containing equation number for the dof's per element
        self.dimension = 3  # Dimension of the problem

        return
        
    def read_gmsh(self):
        """" read gmsh mesh

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

             The element type that are accepted are 5 (8 node brick element) and 17 (20 node brick element)
             """

        # import packages
        import numpy as np
        import sys
        import utils

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

        # add variables to self
        self.nodes = nodes
        self.elem = elem
        self.materials = names
        self.nb_nodes_elem = len(elem[0][5:])

        return

    def read_bc(self, bc):
        """ determines boundary conditions for all nodes

             Assumes that the three coordinates are non-collinear.
        """

        # import packages
        import numpy as np
        import utils

        # variables generation
        self.BC = np.zeros((len(self.nodes), self.dimension), dtype=int)

        # for each boundary plane
        for boundary in bc:
            dof = bc[boundary][0]
            nodes = bc[boundary][1]

            # find all the nodes that are within this plane
            # assuming that the three points are non-collinear
            plane = utils.define_plane(nodes[0], nodes[1], nodes[2])

            residual = self.nodes[:, 1] * plane[0] + self.nodes[:, 2] * plane[1] + self.nodes[:, 3] * plane[2] - plane[3]

            # find indexes
            indices = np.where(residual == 0.)[0]

            for idx in indices:
                for j, val in enumerate(dof):
                    self.BC[idx, j] += int(val)
        return

    def mapping(self):
        r"""
        define equation numbers for each dof in each node

        """
        import numpy as np

        # initialise variables
        self.eq_nb_dof = np.zeros((len(self.nodes), self.dimension))

        # equation number:
        equation_nb = 0

        # loop in all all the nodes
        for i in range(len(self.BC)):
            # loop in all the dof of a node
            for j in range(len(self.BC[i])):
                # if BC = 0: it is an equation
                if self.BC[i][j] == 0:
                    self.eq_nb_dof[i, j] = equation_nb
                    equation_nb += int(1)
                # else it is a boundary
                else:
                    self.eq_nb_dof[i, j] = np.nan

        self.number_eq = equation_nb
        return

    def connectivities(self):
        r"""
        define equation numbers for each dof in each element

        """
        import numpy as np

        # initialise variables
        self.eq_nb_elem = np.zeros((len(self.elem), self.nb_nodes_elem * self.dimension))

        # loop element
        for i in range(len(self.elem)):
            self.eq_nb_elem[i, :] = self.eq_nb_dof[self.elem[i][5:] - 1].flatten()
        return
