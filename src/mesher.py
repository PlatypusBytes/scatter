def define_plane(p1, p2, p3):
    import numpy as np

    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return [a, b, c, d]


def search_idx(data, string1, string2):

    # search string1
    idx = [i for i, val in enumerate(data) if val.startswith(string1)][0]
    nb = int(data[idx + 1])

    # search string2
    idx_end_nodes = [i for i, val in enumerate(data) if val.startswith(string2)][0]

    res = []
    for i in range(idx + 2, idx_end_nodes):
        aux = []
        for j in data[i].split():
            try:
                aux.append(float(j))
            except ValueError:
                aux.append(str(j.replace('"', '')))
        res.append(aux)

    return res, nb


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
        # self.nb_elem = []  # number of elements
        # self.elem_type = []  # element type
        self.materials = []  # materials
        self.BC = []  # BC list
        self.NEQ = 0  # number of equations
        self.ID = []  # list containing equation number for the dof's of each node
        self.LM = []  # list containing equation number for the dof's of each element
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

        # read the file
        with open(self.file_name, 'r') as f:
            data = f.readlines()

        # read Nodes
        nodes, nb_nodes = search_idx(data, r"$Nodes", r"$EndNodes")
        nodes = np.array(nodes)

        # read PhysicalNames
        names, nb_names = search_idx(data, r"$PhysicalNames", r"$EndPhysicalNames")

        # read Elements
        elem, nb_elem = search_idx(data, r"$Elements", r"$EndElements")
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
        """ read boundary conditions

             Assumes that the three coordinates are non-collinear.
        """

        # import packages
        import numpy as np

        # variables generation
        self.BC = np.zeros((len(self.nodes), self.dimension))
        self.ID = np.zeros((len(self.nodes), self.dimension))

        # for each boundary plane
        for boundary in bc:
            dof = bc[boundary][0]
            nodes = bc[boundary][1]

            # find all the nodes that are within this plane
            # assuming that the three points are non-collinear
            plane = define_plane(nodes[0], nodes[1], nodes[2])

            residual = self.nodes[:, 1] * plane[0] + self.nodes[:, 2] * plane[1] + self.nodes[:, 3] * plane[2] - plane[3]

            # find indexes
            indices = np.where(residual == 0.)[0]

            for idx in indices:
                self.BC[idx] = [j for j in dof]

        # NEQ and ID
        for i in range(len(self.BC)):
            for j in range(len(self.BC[i])):
                if self.BC[i][j] == 0:
                    self.NEQ += int(1)
                    self.ID[i, j] = self.NEQ
                else:
                    self.ID[i, j] = np.nan

        # LM
        for i in range(len(self.elem)):
            self.LM.append(self.ID[self.elem[i][5:] - 1].flatten())
        return
