# import packages
import os
import pickle
import numpy as np
from collections import defaultdict
import h5py
from scipy import sparse
# import VTK writer
from vtk_tools import VTK_writer


# element edge topology in gmsh node ordering. Element types not listed fall back to an
# all-to-all connection within the element.
ELEMENT_EDGES = {
    "tri3": [(0, 1), (1, 2), (2, 0)],
    "tri6": [(0, 3), (3, 1), (1, 4), (4, 2), (2, 5), (5, 0)],
    "quad4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "tetra4": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "tetra10": [(0, 4), (4, 1), (1, 5), (5, 2), (2, 6), (6, 0),
                (0, 7), (7, 3), (2, 8), (8, 3), (1, 9), (9, 3)],
    "hexa8": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7)],
}


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
        elif model.element_type == "tetra4":
            self.idx_vtk = [0, 1, 2, 3]
        elif model.element_type == "tetra10":
            self.idx_vtk = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]

        # output folder
        self.output_folder = output_folder

        # variables
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
        self.solver = numerical

        self.data = {}

        # parse the data
        self.parse_data()


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
            idx = [self.data["nodes"].index(int(i)) for i in nodes]

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


    def vtk(self, name="data", binary=True, write=True) -> None:
        """
        Writes VTK file at the pre-difined output interval

        :param name: (optional, default data) basename of the VTK file
        :param write: (optional, default True) checks if file needs to be written
        :param binary: (optional, default True) writes VTK in binary format
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
        for t, output_t in enumerate(self.time_idx):
            # calculate actual time step
            # t = int(output_t*output_interval)

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
            vtk = VTK_writer.Write(os.path.join(self.output_folder, "VTK"), file_name=f"{name}_{output_t}", write_binary=binary)
            vtk.add_mesh(self.coordinates, self.elements, self.element_type)
            vtk.add_vector("displacement", displacement)
            vtk.add_vector("velocity", velocity, header=False)
            vtk.add_vector("boundary_conditions", bc, header=False)
            vtk.add_scalar("material_index", material)
            for j, m in enumerate(list_props):
                vtk.add_scalar(f"material_prop_{m}", material_prop[:, j], header=False)
            vtk.save()



    def generate_gnn_files_(self,  model: object, matrix: object, F: object, write: bool):
        """
        Generate files for training Graph Neural Networks


        Node features:
        - coordinates
        - BC
        - u, v, a (at time t)
        - Delta force (for time t + 1)
        - matrix properties (M, C, K) (diagonal terms)
        - force (x, y, z)

        Edge features:
        - distance
        - matrix properties (M, C, K) (non-diagonal terms)

        Target:
        - Delta u (at time t + 1)

        Graph feature:
        - time
        - time-step

        Parameters
        ----------
        :param model: model object
        :param matrix: matrix object
        :param F: force object
        :param write: boolean flag to write files
        """

        if not write:
            return

        # ToDo: only works for hexa8 elements
        if model.element_type != "hexa8":
            raise ValueError("Only hexa8 elements currently supported")

        os.makedirs(self.output_folder, exist_ok=True)
        output_file = os.path.join(self.output_folder, "data.hdf5")

        with h5py.File(output_file, "w") as f:
            mesh_group = f.create_group("mesh")

            mesh_group.create_dataset("nodes_id", data=model.nodes[:, 0], dtype='i', compression="gzip")
            mesh_group.create_dataset("coordinates", data=model.nodes[:, 1:], dtype='f', compression="gzip")
            mesh_group.create_dataset("BC", data=model.BC, dtype='f', compression="gzip")
            mesh_group.create_dataset("eq_nb_dof", data=model.eq_nb_dof, dtype='f', compression="gzip")
            mesh_group.create_dataset("elements", data=model.elem[:, self.idx_vtk] - 1, dtype="i", compression="gzip")

            mat_group = f.create_group("matrices")

            for name, mat in [("stiffness", matrix.K), ("mass", matrix.M), ("damping", matrix.C)]:
                grp = mat_group.create_group(name)
                mat = sparse.csr_matrix(mat)
                grp.create_dataset("data", data=mat.data, compression="gzip")
                grp.create_dataset("indices", data=mat.indices, compression="gzip")
                grp.create_dataset("indptr", data=mat.indptr, compression="gzip")
                grp.attrs["shape"] = mat.shape

            time_group = f.create_group("time_results")
            time_group.create_dataset("time", data=self.time, compression="gzip")
            time_group.create_dataset("displacement", data=self.dis, compression="gzip")
            time_group.create_dataset("velocity", data=self.vel, compression="gzip")
            time_group.create_dataset("acceleration", data=self.acc, compression="gzip")

            force_group = f.create_group("force_results")
            force = []
            for ti, _ in enumerate(self.time):
                force.append(F.update_load_at_t(ti))
            force_group.create_dataset("force", data=force, compression="gzip")


    def _element_edges(self, elements: np.ndarray) -> np.ndarray:
        """
        Undirected node pairs of every element, repeated once per element.

        :param elements: (n_elem, n_nodes_elem) 0-based element connectivity
        :return: (n_elem * n_edges_elem, 2) node pairs, aligned with a repeated element index
        """
        topology = ELEMENT_EDGES.get(self.element_type)
        if topology is None:
            # unknown topology -> connect every node pair inside the element
            n = elements.shape[1]
            topology = [(i, j) for i in range(n) for j in range(i + 1, n)]

        local = np.asarray(topology, dtype=np.int64)
        return elements[:, local].reshape(-1, 2)

    def _element_material_properties(self, model: object) -> np.ndarray:
        """
        Young modulus, Poisson ratio and density of every element.

        Uses a lookup table indexed by material id so the cost is independent of the number
        of materials (a random field creates one material per element).

        :param model: model object
        :return: (n_elem, 3) array with [Young, poisson, density]
        """
        mat_ids = np.asarray(model.materials_index, dtype=np.int64)
        table = np.full((int(mat_ids.max()) + 1, 3), np.nan)
        for _, mat_id, name in model.materials:
            props = self.materials[name]
            table[int(mat_id)] = [props["Young"], props["poisson"], props["density"]]

        return table[mat_ids]

    def generate_gnn_files(self, model: object, matrix: object, F: object, write: bool = True,
                           file_name: str = "data.hdf5") -> None:
        """
        Generate one HDF5 file with everything needed to train a MeshGraphNet-like surrogate
        of a single Newmark step.

        The graph is static (mesh + material), the state is a time series. One training
        sample is one time index, so the file stores the two parts separately instead of
        duplicating the graph for every step.

        Layout
        ------
        attrs           element_type, n_dim, n_nodes, n_elem, n_eq, dt, beta, gamma,
                        rayleigh_alpha, rayleigh_beta
        mesh/           node_ids, coordinates, elements, bc, dof_index, dof_mask
        graph/          edge_index (2, n_edges), edge_attr
        node/           material (E, nu, rho), wave_speed (cp, cs), m_diag, c_diag, k_diag
        element/        materials_index, material (E, nu, rho)
        matrices/       mass, damping, stiffness as CSR (data, indices, indptr)
        time/           time
        dynamic/        displacement, velocity, acceleration, force, all (n_time, n_eq)

        The dynamic arrays are stored in equation ordering; `mesh/dof_index` maps them onto
        the (n_nodes, n_dim) nodal layout (-1 marks a prescribed dof).

        :param model: model object
        :param matrix: object with the assembled M, C and K matrices
        :param F: force object (kept for backwards compatibility, forces are read from the solver state)
        :param write: boolean flag to write the file
        :param file_name: (optional, default data.hdf5) name of the output file
        """

        if not write:
            return

        # the Newmark relations only link consecutive solver steps, so subsampled output
        # cannot be used as training pairs
        output_interval = getattr(self.solver.state, "output_interval", 1)
        if output_interval != 1:
            raise ValueError("GNN export requires output_interval == 1, "
                             f"got {output_interval}. Newmark consistency between consecutive "
                             "samples is lost when the output is subsampled.")

        dt = np.diff(self.time)
        if not np.allclose(dt, dt[0]):
            raise ValueError("GNN export requires a constant time step")
        dt = float(dt[0])

        os.makedirs(self.output_folder, exist_ok=True)
        output_file = os.path.join(self.output_folder, file_name)

        n_dim = self.n_dim
        n_eq = int(model.number_eq)
        node_ids = model.nodes[:, 0].astype(np.int64)
        coordinates = model.nodes[:, 1:1 + n_dim].astype(np.float64)
        n_nodes = coordinates.shape[0]

        # gmsh node id -> row index in the node array (ids are not guaranteed to be 0..N-1)
        id_to_index = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
        id_to_index[node_ids] = np.arange(n_nodes)
        elements = id_to_index[model.elem.astype(np.int64)]
        if np.any(elements < 0):
            raise ValueError("Element connectivity refers to unknown node ids")

        # -------------------------------------------------------------- element properties
        elem_props = self._element_material_properties(model)      # (n_elem, 3): E, nu, rho

        # -------------------------------------------------------------- edges
        pairs = self._element_edges(elements)
        elem_of_pair = np.repeat(np.arange(elements.shape[0]), pairs.shape[0] // elements.shape[0])

        # unique undirected edges; an edge shared by several elements appears once
        undirected = np.sort(pairs, axis=1)
        undirected, inverse = np.unique(undirected, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

        # average the material of all elements sharing the edge (an edge on a material
        # interface is not owned by a single element)
        counts = np.bincount(inverse, minlength=undirected.shape[0]).astype(np.float64)
        edge_props = np.stack(
            [np.bincount(inverse, weights=elem_props[elem_of_pair, k],
                         minlength=undirected.shape[0]) / counts for k in range(3)],
            axis=1,
        )

        # both directions, so that messages travel symmetrically
        edge_index = np.concatenate([undirected, undirected[:, ::-1]], axis=0).T
        edge_props = np.concatenate([edge_props, edge_props], axis=0)

        rel_pos = coordinates[edge_index[1]] - coordinates[edge_index[0]]
        length = np.linalg.norm(rel_pos, axis=1, keepdims=True)

        E_e, nu_e, rho_e = edge_props[:, 0:1], edge_props[:, 1:2], edge_props[:, 2:3]
        cp_e = self._p_wave_velocity(E_e, nu_e, rho_e)

        edge_attr = np.concatenate(
            [rel_pos,                    # direction, keeps the model translation invariant
             length,
             np.log10(E_e),
             nu_e,
             np.log10(rho_e),
             cp_e * dt / np.clip(length, 1e-12, None)],   # local Courant number
            axis=1,
        ).astype(np.float32)

        # -------------------------------------------------------------- nodal properties
        # volume weighting is not available here, so the arithmetic mean of the attached
        # elements is used
        nodes_of_elem = elements.reshape(-1)
        elem_of_node = np.repeat(np.arange(elements.shape[0]), elements.shape[1])
        node_counts = np.bincount(nodes_of_elem, minlength=n_nodes).astype(np.float64)
        node_counts = np.clip(node_counts, 1.0, None)
        node_props = np.stack(
            [np.bincount(nodes_of_elem, weights=elem_props[elem_of_node, k],
                         minlength=n_nodes) / node_counts for k in range(3)],
            axis=1,
        )
        cp = self._p_wave_velocity(node_props[:, 0:1], node_props[:, 1:2], node_props[:, 2:3])
        cs = np.sqrt(node_props[:, 0:1] / (2 * (1 + node_props[:, 1:2])) / node_props[:, 2:3])

        # -------------------------------------------------------------- dof bookkeeping
        eq_nb_dof = np.asarray(model.eq_nb_dof, dtype=np.float64)[:, :n_dim]
        dof_index = np.where(np.isnan(eq_nb_dof), -1, np.nan_to_num(eq_nb_dof)).astype(np.int32)
        dof_mask = (dof_index >= 0).astype(np.int8)

        # diagonals of the operators, scattered onto the nodal layout
        diagonals = {}
        for name, mat in (("m_diag", matrix.M), ("c_diag", matrix.C), ("k_diag", matrix.K)):
            diag = sparse.csr_matrix(mat).diagonal()[:n_eq]
            nodal = np.zeros((n_nodes, n_dim))
            nodal[dof_mask.astype(bool)] = diag[dof_index[dof_mask.astype(bool)]]
            diagonals[name] = nodal

        # -------------------------------------------------------------- dynamic state
        # extra dofs of a coupled rose model are dropped, they are not mesh nodes
        u = np.asarray(self.dis)[:, :n_eq]
        v = np.asarray(self.vel)[:, :n_eq]
        a = np.asarray(self.acc)[:, :n_eq]
        force = np.asarray(self.solver.state.F_out)[:, :n_eq]

        # -------------------------------------------------------------- write
        with h5py.File(output_file, "w") as f:
            f.attrs["element_type"] = self.element_type
            f.attrs["n_dim"] = n_dim
            f.attrs["n_nodes"] = n_nodes
            f.attrs["n_elem"] = elements.shape[0]
            f.attrs["n_eq"] = n_eq
            f.attrs["n_time"] = len(self.time)
            f.attrs["dt"] = dt
            f.attrs["beta"] = float(getattr(self.solver, "beta", np.nan))
            f.attrs["gamma"] = float(getattr(self.solver, "gamma", np.nan))
            f.attrs["rayleigh_alpha"] = float(getattr(matrix, "alpha", np.nan))
            f.attrs["rayleigh_beta"] = float(getattr(matrix, "beta", np.nan))

            mesh = f.create_group("mesh")
            mesh.create_dataset("node_ids", data=node_ids, dtype="i8", compression="gzip")
            mesh.create_dataset("coordinates", data=coordinates, dtype="f4", compression="gzip")
            mesh.create_dataset("elements", data=elements, dtype="i4", compression="gzip")
            mesh.create_dataset("bc", data=self.bc[:, :n_dim], dtype="i1", compression="gzip")
            mesh.create_dataset("dof_index", data=dof_index, dtype="i4", compression="gzip")
            mesh.create_dataset("dof_mask", data=dof_mask, dtype="i1", compression="gzip")

            graph = f.create_group("graph")
            graph.create_dataset("edge_index", data=edge_index, dtype="i4", compression="gzip")
            attr_dataset = graph.create_dataset("edge_attr", data=edge_attr, dtype="f4",
                                                compression="gzip")
            attr_dataset.attrs["names"] = [f"rel_pos_{i}" for i in range(n_dim)] + \
                                          ["length", "log10_Young", "poisson",
                                           "log10_density", "courant"]

            node = f.create_group("node")
            node.create_dataset("material", data=node_props, dtype="f4", compression="gzip")
            node.create_dataset("wave_speed", data=np.concatenate([cp, cs], axis=1),
                                dtype="f4", compression="gzip")
            for name, values in diagonals.items():
                node.create_dataset(name, data=values, dtype="f4", compression="gzip")

            element = f.create_group("element")
            element.create_dataset("materials_index", data=model.materials_index,
                                   dtype="i4", compression="gzip")
            element.create_dataset("material", data=elem_props, dtype="f4", compression="gzip")

            matrices = f.create_group("matrices")
            for name, mat in (("mass", matrix.M), ("damping", matrix.C), ("stiffness", matrix.K)):
                csr = sparse.csr_matrix(mat)[:n_eq, :n_eq]
                grp = matrices.create_group(name)
                grp.create_dataset("data", data=csr.data, dtype="f8", compression="gzip")
                grp.create_dataset("indices", data=csr.indices, dtype="i4", compression="gzip")
                grp.create_dataset("indptr", data=csr.indptr, dtype="i4", compression="gzip")
                grp.attrs["shape"] = csr.shape

            time_group = f.create_group("time")
            time_group.create_dataset("time", data=self.time, dtype="f8", compression="gzip")

            # float32 halves the file size and stays far above the noise floor of the solver
            chunk = (min(64, u.shape[0]), n_eq)
            dynamic = f.create_group("dynamic")
            for name, values in (("displacement", u), ("velocity", v),
                                 ("acceleration", a), ("force", force)):
                dynamic.create_dataset(name, data=values, dtype="f4",
                                       chunks=chunk, compression="gzip")

    @staticmethod
    def _p_wave_velocity(young: np.ndarray, poisson: np.ndarray, density: np.ndarray) -> np.ndarray:
        """
        Compressional wave velocity of a linear elastic material
        """
        constrained_modulus = young * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
        return np.sqrt(constrained_modulus / density)
