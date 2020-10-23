# import packages
import os
import sys
import pickle
import numpy as np
# import scatter packages
from src import utils


class ReadMesh:
    def __init__(self, file_name: str, output_folder: str) -> None:
        """
        Reads the mesh and creates the mesh geometry structure.

        Parameters
        ----------
        :param file_name: filename of gmsh file
        :param output_folder: location to save output results
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
        self.nb_nodes_elem = []  # number of nodes
        self.materials = []  # materials
        self.BC = []  # Boundary conditions for each node
        self.number_eq = []  # number of equations
        self.type_BC = []  # type of BC for each dof in node list
        self.eq_nb_dof = []  # number of equation for each dof in node list
        self.eq_nb_elem = []  # list containing equation number for the dof's per element
        self.type_BC_elem = []  # list containing type of BC for the dof's per element
        self.element_type = []  # element type
        self.materials_index = []  # list containing material index for each element
        self.dimension = 3  # Dimension of the problem

        # check if output folder exists. if not creates
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        self.output_folder = output_folder

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

        The element type that are accepted are 5 (8 node brick element) and 17 (20 node brick element)
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
            self.element_type = 'linear'
        elif all(x == 17 for x in element_type):
            self.element_type = 'quad'

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

    def mapping(self) -> None:
        r"""
        Define equation numbers and type of boundary condition for each *dof* in the node list
        """

        # initialise variables
        self.eq_nb_dof = np.zeros((len(self.nodes), self.dimension))
        self.type_BC = np.full((len(self.nodes), self.dimension), "Normal")

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
            idx_nodes = [np.where(self.nodes[:, 0]==j)[0][0] for j in self.elem[i]]
            self.eq_nb_elem[i, :] = self.eq_nb_dof[idx_nodes].flatten()
            self.type_BC_elem[i, :] = self.type_BC[idx_nodes].flatten()
        return

    def remap_results(self, time, dis, vel, acc):
        # dict with results
        data = {}
        data.update({"time": time,
                     "nodes": self.nodes[:, 0],
                     "position": self.nodes[:, 1:],
                     "displacement": {},
                     "velocity": {},
                     "acceleration": {},
                     })

        for i in range(len(self.nodes)):
            dof_x = self.eq_nb_dof[i][0]
            dof_y = self.eq_nb_dof[i][1]
            dof_z = self.eq_nb_dof[i][2]

            # x direction
            if np.isnan(dof_x):
                ux = np.ones(len(time)) * np.nan
                vx = np.ones(len(time)) * np.nan
                ax = np.ones(len(time)) * np.nan
            else:
                ux = dis[:, int(dof_x)]
                vx = vel[:, int(dof_x)]
                ax = acc[:, int(dof_x)]

            # y direction
            if np.isnan(dof_y):
                uy = np.ones(len(time)) * np.nan
                vy = np.ones(len(time)) * np.nan
                ay = np.ones(len(time)) * np.nan
            else:
                uy = dis[:, int(dof_y)]
                vy = vel[:, int(dof_y)]
                ay = acc[:, int(dof_y)]

            # z direction
            if np.isnan(dof_z):
                uz = np.ones(len(time)) * np.nan
                vz = np.ones(len(time)) * np.nan
                az = np.ones(len(time)) * np.nan
            else:
                uz = dis[:, int(dof_z)]
                vz = vel[:, int(dof_z)]
                az = acc[:, int(dof_z)]

            # update dic
            data["displacement"].update({str(i + 1): {"x": ux,
                                                      "y": uy,
                                                      "z": uz
                                                      }
                                         })
            data["velocity"].update({str(i + 1): {"x": vx,
                                                  "y": vy,
                                                  "z": vz
                                                  }
                                     })
            data["acceleration"].update({str(i + 1): {"x": ax,
                                                      "y": ay,
                                                      "z": az
                                                      }
                                         })

        # dump data
        with open(os.path.join(self.output_folder, "data.pickle"), "wb") as f:
            pickle.dump(data, f)
        return
