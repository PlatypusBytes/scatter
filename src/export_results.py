# import packages
import os
import pickle
import numpy as np
from collections import defaultdict
# import VTK writer
from vtk_tools import VTK_writer


class Write:
    def __init__(self, output_folder: str, model: object, materials: dict, numerical: object) -> None:
        """
        Writes the output

        Parameters
        ----------
        :param model: Object with mesh
        :param materials: Dictionary with materials
        :param numerical: Object with numerical results
        :param output_folder: location to save output results
        """

        # check if output folder exists. if not creates
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # link between gmsh and VTK no index
        self.element_type = model.element_type
        if model.element_type == "hexa8":
            self.idx_vtk = [0, 1, 2, 3, 4, 5, 6, 7]
        elif model.element_type == "hexa20":
            self.idx_vtk = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13, 14, 15, 16, 17, 18, 19]
        elif model.element_type == "quad4":
            self.idx_vtk = [0, 1, 2, 3]
        elif model.element_type == "tri3":
            self.idx_vtk = [0, 1, 2]
        elif model.element_type == "tri6":
            self.idx_vtk = [0, 1, 2, 3, 4, 5]

        # output folder
        self.output_folder = output_folder

        # variables
        self.nodes = model.nodes[:, 0].astype(int)
        self.eq_nb_dof = model.eq_nb_dof
        self.coordinates = model.nodes[:, 1:]
        self.elements = model.elem[:, self.idx_vtk] - 1
        self.time = numerical.time
        self.dis = numerical.u
        self.vel = numerical.v
        self.acc = numerical.a
        self.mat = model.materials
        self.mat_idx = model.materials_index
        self.materials = materials
        self.bc = model.BC
        self.n_dim = model.dimension

        self.data = {}

        # parse the data
        self.parse_data()
        return

    def parse_data(self) -> None:
        """
        Parses the data into a dictionary
        """

        # dict with results
        self.data.update({"time": self.time,
                          "nodes": list(map(int, self.nodes)),
                          "position": self.coordinates,
                          "displacement": defaultdict(dict),
                          "velocity": defaultdict(dict),
                          "acceleration": defaultdict(dict),
                          })

        if self.n_dim == 3:
            iterator_xyz = [0, 1, 2]
            label_xyz = ["x", "y", "z"]
        elif self.n_dim == 2:
            iterator_xyz = [0, 1]
            label_xyz = ["x", "y"]

        for i in range(len(self.nodes)):
            for idx in iterator_xyz:
                dof = self.eq_nb_dof[i][idx]
                if np.isnan(dof):
                    u = v = a = np.zeros(len(self.time))
                else:
                    u = self.dis[:, int(dof)]
                    v = self.vel[:, int(dof)]
                    a = self.acc[:, int(dof)]

                # update dic
                self.data["displacement"][str(int(self.nodes[i]))][label_xyz[idx]] = u
                self.data["velocity"][str(int(self.nodes[i]))][label_xyz[idx]] = v
                self.data["acceleration"][str(int(self.nodes[i]))][label_xyz[idx]] = a
        return

    def pickle(self, name="data", write=True, nodes="all") -> None:
        """
        Writes pickle file in binary

        :param name: (optional, default data) name of the pickle file
        :param write: (optional, default True) checks if file needs to be written
        :param nodes: (optional, default 'all') nodes to be written in pickle file
        """
        if not write:
            return

        # if list of nodes exists -> dump results only for nodes
        if nodes != "all":
            idx = [self.data["nodes"].index(i) for i in nodes]

            data = {"time": self.data["time"],
                    "nodes": nodes,
                    "position": [self.data["position"][i] for i in idx],
                    "displacement": defaultdict(dict),
                    "velocity": defaultdict(dict),
                    "acceleration": defaultdict(dict),
                    }

            for n in nodes:
                data["displacement"].update({str(n): self.data["displacement"][str(n)]})
                data["velocity"].update({str(n): self.data["velocity"][str(n)]})
                data["acceleration"].update({str(n): self.data["acceleration"][str(n)]})

            # dump data
            with open(os.path.join(self.output_folder, f"{name}.pickle"), "wb") as f:
                pickle.dump(data, f)

        else:
            # dump data
            with open(os.path.join(self.output_folder, f"{name}.pickle"), "wb") as f:
                pickle.dump(self.data, f)
        return

    def vtk(self, name="data", write=True, output_interval=1) -> None:
        """
        Writes VTK file

        :param name: (optional, default data) basename of the VTK file
        :param write: (optional, default True) checks if file needs to be written
        :param output_interval: (optional, default 1) interval in timesteps which are written to vtk
        """
        if not write:
            return

        nb_nodes = len(self.nodes)
        nb_elements = len(self.elements)

        # find material properties
        list_props = list(set([tuple(i.keys()) for i in self.materials.values()]))[0]

        # define materials
        material = np.zeros(nb_elements)
        material_prop = np.zeros((nb_elements, len(list_props)))
        for n in range(nb_elements):
            # material index
            material[n] = self.mat_idx[n]
            # find material name
            material_name = [i[2] for i in self.mat if i[1] == material[n]][0]
            #  material property
            for j, m in enumerate(list_props):
                material_prop[n, j] = self.materials[material_name][m]

        # make sure dimension are correct for writing to VTK
        bc = np.zeros((self.bc.shape[0], 3))
        if self.n_dim == 2:
            bc[:, :2] = self.bc
        elif self.n_dim == 3:
            bc = self.bc

        # for each output time writes a VTK file
        for output_t in range(int(len(self.time)/output_interval)):
            # calculate actual time step
            t = int(output_t*output_interval)

            # define displacement and velocity
            displacement = np.zeros((nb_nodes, 3))
            velocity = np.zeros((nb_nodes, 3))
            for i in range(nb_nodes):
                if self.n_dim == 3:
                    displacement[i, :] = np.array([self.data["displacement"][str(int(self.nodes[i]))]["x"][t],
                                                   self.data["displacement"][str(int(self.nodes[i]))]["y"][t],
                                                   self.data["displacement"][str(int(self.nodes[i]))]["z"][t]])
                    velocity[i, :] = np.array([self.data["velocity"][str(int(self.nodes[i]))]["x"][t],
                                               self.data["velocity"][str(int(self.nodes[i]))]["y"][t],
                                               self.data["velocity"][str(int(self.nodes[i]))]["z"][t]])
                elif self.n_dim == 2:
                    displacement[i, :2] = np.array([self.data["displacement"][str(int(self.nodes[i]))]["x"][t],
                                                   self.data["displacement"][str(int(self.nodes[i]))]["y"][t]])
                    velocity[i, :2] = np.array([self.data["velocity"][str(int(self.nodes[i]))]["x"][t],
                                               self.data["velocity"][str(int(self.nodes[i]))]["y"][t]])

            # write VTK at time t
            vtk = VTK_writer.Write(os.path.join(self.output_folder, "VTK"), file_name=f"{name}_{output_t}")
            vtk.add_mesh(self.coordinates, self.elements, self.element_type)
            vtk.add_vector("displacement", displacement)
            vtk.add_vector("velocity", velocity, header=False)
            vtk.add_vector("boundary_conditions", bc, header=False)
            vtk.add_scalar("material_index", material)
            for j, m in enumerate(list_props):
                vtk.add_scalar(f"material_prop_{m}", material_prop[:, j], header=False)
            vtk.save()
        return
