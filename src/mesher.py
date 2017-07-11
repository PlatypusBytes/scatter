class ReadMesh:

    def __init__(self, file_name):
        import os
        import sys

        # check if file name exists
        if os.path.isfile(file_name):
            self.file_name = file_name
        else:
            # if does not exist exit with error message
            sys.exit("Mesh file does not exit.")

        # check if the file has the correct extension
        if os.path.splitext(self.file_name)[-1] != ".msh":
            sys.exit("Mesh file is not a valid file")

        # define variables
        self.nodes = []
        self.elem = []
        self.nb_nodes = []
        self.nb_elem = []

        return
        
    def read_gmsh(self):
        import numpy as np

        # read the file
        with open(self.file_name, 'r') as f:
            data = f.readlines()

        # search $Nodes
        idx_nodes = [i for i, val in enumerate(data) if val.startswith(r"$Nodes")][0]
        nb_nodes = int(data[idx_nodes + 1])

        # search $EndNodes
        idx_end_nodes = [i for i, val in enumerate(data) if val.startswith(r"$EndNodes")][0]

        nodes = []
        for i in range(idx_nodes + 2, idx_end_nodes):
            nodes.append([float(j) for j in data[i].split()])

        nodes = np.array(nodes)
        nodes[:, 0] = nodes[:, 0].astype(int)

        # search $Elements
        idx_elem = [i for i, val in enumerate(data) if val.startswith(r"$Elements")][0]
        nb_elem = int(data[idx_elem + 1])

        # search $EndElements
        idx_end_elemn = [i for i, val in enumerate(data) if val.startswith(r"$EndElements")][0]

        elemn = []
        for i in range(idx_elem + 2, idx_end_elemn):
            elemn.append([int(j) for j in data[i].split()])

        elem = np.array(elemn)

        # add variables to self
        self.nodes = nodes
        self.elem = elem
        self.nb_nodes = nb_nodes
        self.nb_elem = nb_elem

        return

    def read_bc(self, bc):

        for boundary in bc:
            dof = bc[boundary][0]
            nodes = bc[boundary][1]
            # ToDo get material from mesh



        return