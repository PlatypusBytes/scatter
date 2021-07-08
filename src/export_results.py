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
        if model.element_type == "hexa8":
            self.idx_vtk = [3, 2, 6, 7, 0, 1, 5, 4]
        elif model.element_type == "hexa20":
            self.idx_vtk = [3, 2, 6, 7, 0, 1, 5, 4, 10, 19, 11, 18, 9, 14, 13, 15, 8, 16, 17, 12]

        # output folder
        self.output_folder = output_folder

        # variables
        self.nodes = model.nodes[:, 0]
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
                          "nodes": self.nodes,
                          "position": self.coordinates,
                          "displacement": defaultdict(dict),
                          "velocity": defaultdict(dict),
                          "acceleration": defaultdict(dict),
                          })

        iterator_xyz = [0, 1, 2]
        label_xyz = ["x", "y", "z"]
        for i in range(len(self.nodes)):
            for idx in iterator_xyz:
                dof = self.eq_nb_dof[i][idx]
                if np.isnan(dof):
                    u = v = a = np.ones(len(self.time)) * 0
                else:
                    u = self.dis[:, int(dof)]
                    v = self.vel[:, int(dof)]
                    a = self.acc[:, int(dof)]

                # update dic
                self.data["displacement"][str(int(self.nodes[i]))][label_xyz[idx]] = u
                self.data["velocity"][str(int(self.nodes[i]))][label_xyz[idx]] = v
                self.data["acceleration"][str(int(self.nodes[i]))][label_xyz[idx]] = a
        return

    def pickle(self, name="data", write=True) -> None:
        """
        Writes pickle file in binary

        :param name: (optional, default data) name of the pickle file
        :param write: (optional, default True) checks if file needs to be written
        """
        if not write:
            return

        # dump data
        with open(os.path.join(self.output_folder, f"{name}.pickle"), "wb") as f:
            pickle.dump(self.data, f)
        return

    def vtk(self, name="data", write=True) -> None:
        """
        Writes VTK file

        :param name: (optional, default data) basename of the VTK file
        :param write: (optional, default True) checks if file needs to be written
        """
        if not write:
            return

        nb_nodes = len(self.nodes)
        nb_elements = len(self.elements)

        # find material properties
        list_props = list(set([tuple(i.keys()) for i in self.materials.values()]))[0]

        # for each time writes a VTK file
        for t in range(len(self.time)):
            displacement = np.zeros((nb_nodes, 3))
            velocity = np.zeros((nb_nodes, 3))
            material = np.zeros(nb_elements)
            material_prop = np.zeros((nb_elements, len(list_props)))
            for i in range(nb_nodes):
                displacement[i, :] = np.array([self.data["displacement"][str(int(self.nodes[i]))]["x"][t],
                                               self.data["displacement"][str(int(self.nodes[i]))]["y"][t],
                                               self.data["displacement"][str(int(self.nodes[i]))]["z"][t]])
                velocity[i, :] = np.array([self.data["velocity"][str(int(self.nodes[i]))]["x"][t],
                                           self.data["velocity"][str(int(self.nodes[i]))]["y"][t],
                                           self.data["velocity"][str(int(self.nodes[i]))]["z"][t]])

            for n in range(nb_elements):
                # material index
                material[n] = self.mat_idx[n]
                # find material name
                material_name = [i[2] for i in self.mat if i[1] == material[n]][0]
                #  material property
                for j, m in enumerate(list_props):
                    material_prop[n, j] = self.materials[material_name][m]

            # write VTK
            vtk = VTK_writer.Write(os.path.join(self.output_folder, "VTK"), file_name=f"{name}_{t}")
            vtk.add_mesh(self.coordinates, self.elements)
            vtk.add_vector("displacement", displacement)
            vtk.add_vector("velocity", velocity, header=False)
            vtk.add_scalar("material_index", material)
            for j, m in enumerate(list_props):
                vtk.add_scalar(f"material_prop_{m}", material_prop[:, j], header=False)
            vtk.save()
        return
