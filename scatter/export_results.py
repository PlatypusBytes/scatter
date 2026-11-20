"""Export of the results to pickle and VTK files."""
import os
import pickle
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import numpy.typing as npt
from vtk_tools import VTK_writer

from scatter.mesher import ReadMesh

# order of the gmsh nodes in the VTK cell of each element type
VTK_NODE_ORDER = {
    "hexa8": list(range(8)),
    "hexa20": list(range(20)),
    "quad4": [0, 1, 2, 3],
    "tri3": [0, 1, 2],
    "tri6": [0, 1, 2, 3, 4, 5],
    "tetra4": [0, 1, 2, 3],
    "tetra10": [0, 1, 2, 3, 4, 5, 6, 7, 9, 8],
}


class Write:
    """Writes the results of a calculation."""

    def __init__(self, output_folder: str, model: ReadMesh, materials: Dict[str, Dict[str, float]],
                 numerical: Any) -> None:
        """Collect the results of a calculation, and create the output folder if it does not exist.

        Args:
            output_folder: folder where the results are written.
            model: scatter model.
            materials: material properties per material name.
            numerical: numerical solver, with the results in its state.
        """
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        self.element_type = model.element_type
        self.idx_vtk = VTK_NODE_ORDER[model.element_type]
        self.output_folder = output_folder

        self.nodes = model.nodes[:, 0].astype(int)
        self.eq_nb_dof = model.eq_nb_dof
        self.coordinates = model.nodes[:, 1:]
        self.elements = model.elem[:, self.idx_vtk] - 1
        self.time = numerical.state.output_time
        self.time_idx = numerical.state.output_time_indices
        self.dis = numerical.state.u
        self.vel = numerical.state.v
        self.acc = numerical.state.a
        self.mat = model.materials
        self.mat_idx = model.materials_index
        self.materials = materials
        self.bc = model.BC
        self.n_dim = model.dimension

        self.data: Dict[str, Any] = {}
        self.parse_data()

    def parse_data(self) -> None:
        """Parse the results into a dictionary per node and direction."""
        self.data.update({
            "time": self.time,
            "nodes": list(map(int, self.nodes)),
            "position": self.coordinates,
            "displacement": defaultdict(dict),
            "velocity": defaultdict(dict),
            "acceleration": defaultdict(dict),
        })

        label_xyz = ["x", "y", "z"][:self.n_dim]
        for i in range(len(self.nodes)):
            for idx, label in enumerate(label_xyz):
                dof = self.eq_nb_dof[i][idx]
                if np.isnan(dof):
                    u = v = a = np.zeros(len(self.time))
                else:
                    u = self.dis[:, int(dof)]
                    v = self.vel[:, int(dof)]
                    a = self.acc[:, int(dof)]

                self.data["displacement"][str(int(self.nodes[i]))][label] = u
                self.data["velocity"][str(int(self.nodes[i]))][label] = v
                self.data["acceleration"][str(int(self.nodes[i]))][label] = a

    def pickle(self, name: str = "data", write: bool = True, nodes: Union[str, Sequence[str]] = "all") -> None:
        """Write the results to a binary pickle file.

        Args:
            name: name of the pickle file, without extension.
            write: whether the file is written.
            nodes: node numbers to write, or "all".
        """
        if not write:
            return

        if nodes == "all":
            data = self.data
        else:
            idx = [self.data["nodes"].index(int(i)) for i in nodes]
            data = {
                "time": self.data["time"],
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

        with open(os.path.join(self.output_folder, f"{name}.pickle"), "wb") as f:
            pickle.dump(data, f)

    def vtk(self, name: str = "data", binary: bool = True, write: bool = True) -> None:
        """Write a VTK file at each output time.

        Args:
            name: base name of the VTK files.
            binary: whether the VTK files are written in binary format.
            write: whether the files are written.
        """
        if not write:
            return

        nb_nodes = len(self.nodes)
        nb_elements = len(self.elements)
        list_props = list(set([tuple(i.keys()) for i in self.materials.values()]))[0]

        # material index and properties of each element; the first material with an index wins
        material_names: Dict[Any, str] = {}
        for material_entry in self.mat:
            material_names.setdefault(material_entry[1], material_entry[2])
        material = np.zeros(nb_elements)
        material_prop = np.zeros((nb_elements, len(list_props)))
        for n in range(nb_elements):
            material[n] = self.mat_idx[n]
            for j, m in enumerate(list_props):
                material_prop[n, j] = self.materials[material_names[material[n]]][m]

        # VTK needs three components
        bc: npt.NDArray[Any]
        if self.n_dim == 3:
            bc = self.bc
        else:
            bc = np.zeros((self.bc.shape[0], 3))
            bc[:, :self.n_dim] = self.bc

        labels: List[str] = ["x", "y", "z"][:self.n_dim]
        for t, output_t in enumerate(self.time_idx):
            displacement = np.zeros((nb_nodes, 3))
            velocity = np.zeros((nb_nodes, 3))
            for i in range(nb_nodes):
                node = str(int(self.nodes[i]))
                displacement[i, :self.n_dim] = [self.data["displacement"][node][label][t] for label in labels]
                velocity[i, :self.n_dim] = [self.data["velocity"][node][label][t] for label in labels]

            vtk = VTK_writer.Write(os.path.join(self.output_folder, "VTK"),
                                   file_name=f"{name}_{output_t}",
                                   write_binary=binary)
            vtk.add_mesh(self.coordinates, self.elements, self.element_type)
            vtk.add_vector("displacement", displacement)
            vtk.add_vector("velocity", velocity, header=False)
            vtk.add_vector("boundary_conditions", bc, header=False)
            vtk.add_scalar("material_index", material)
            for j, m in enumerate(list_props):
                vtk.add_scalar(f"material_prop_{m}", material_prop[:, j], header=False)
            vtk.save()
