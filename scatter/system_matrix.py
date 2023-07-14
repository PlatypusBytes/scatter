from collections import defaultdict
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix

# import scatter packages
from scatter import discretisation, material_models
from scatter.utils import clockwise_sort_2D_elements


class GenerateMatrix:
    r"""
    Define and assembles global matrices for the system of equations.
    It uses sparse matrices.
    """

    def __init__(self, nb_equations: int, order: int) -> None:
        """
        Initialise global matrices

        Parameters
        ----------
        :param nb_equations: number of equations in the system
        :param order: order of Gaussian numerical integration
        """
        # generation of variable
        self.M = lil_matrix((nb_equations, nb_equations))
        self.C = lil_matrix((nb_equations, nb_equations))
        self.K = lil_matrix((nb_equations, nb_equations))

        # order of the Gauss integration
        self.order = order

        return

    def generate_stiffness_and_mass(self, data: classmethod, material:dict) -> None:
        r"""
        Global stiffness and mass matrix generation.

        Generates and assembles the global stiffness and mass matrix for the structure.

        Parameters
        ----------
        :param data: mesh and geometry data class
        :param material: material dictionary
        """

        # initialise stiffness and mass dictionary
        k_dict = defaultdict(float)
        mass_dict = defaultdict(float)

        # create dictionaries of materials and nodes for quicker look-up
        dict_materials = dict(np.array(data.materials)[:, 1:])
        dict_nodes = dict([(node[0], node[1:]) for node in data.nodes])
        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # call shape functions
            if data.dimension == 3:
                shape_fct = discretisation.VolumeElement(data.element_type, self.order)
            if data.dimension == 2:
                shape_fct = discretisation.SurfaceElement(data.element_type, self.order)

            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = dict_materials[str(mat_idx)]

            # solid elastic properties
            E = material[name_material]["Young"]
            v = material[name_material]["poisson"]
            rho = material[name_material]["density"]

            # element stiffness matrix
            D = material_models.stiffness_elasticity(E, v, data.dimension)

            # coordinates for all the nodes in one element
            xyz = np.array([dict_nodes[node] for node in elem])

            # generate shape functions B and H matrix
            shape_fct.generate(xyz)

            # compute stiffness
            Ke = shape_fct.compute_stiffness(D)

            # compute mass
            Me = shape_fct.compute_mass(rho)

            # assemble Stiffness and mass matrix
            # equation number where the stiff and mass matrix exists
            i1 = data.eq_nb_elem[idx][~np.isnan(data.eq_nb_elem[idx])].astype(int)
            id1 = np.argsort(i1)
            i1 = np.sort(i1)
            # index where stiffness and mass matrix exists
            i2 = np.where(~np.isnan(data.eq_nb_elem[idx]))[0]
            i2 = i2[id1]

            # assign to the global stiffness matrix dictionary
            for i, j in zip(i1, i2):
                for k, l in zip(i1, i2):

                    # add value to dictionary
                    k_dict[i, k] += Ke[j, l]
                    mass_dict[i, k] += Me[j, l]

        # create arrays from dictionary keys and values

        # stiffness data
        keys_k = np.array(list(k_dict.keys())).astype("uint32")
        row_k = keys_k[:,0]
        col_k = keys_k[:,1]
        values_k = list(k_dict.values())

        # mass data
        keys_m = np.array(list(mass_dict.keys())).astype("uint32")
        row_m = keys_m[:,0]
        col_m = keys_m[:,1]
        values_m = list(mass_dict.values())

        # create sparse lil matrices
        self.K = coo_matrix((values_k, (row_k, col_k))).tolil()
        self.M = coo_matrix((values_m, (row_m, col_m))).tolil()

    def add_rose_stiffness(self,data, rose_model):
        """
        Adds rose stiffness matrix to scatter global stiffness matrix

        Generates and assembles the global stiffness matrix for the structure.

        Parameters
        ----------
        :param data:  mesh and geometry data class
        :param rose_model: rose coupled train track system

        """


        combined_k = self.add_rose_matrix(self.K, rose_model.global_stiffness_matrix, data.eq_nb_dof_rose_nodes,
                                          data.rose_eq_nb)

        self.K = combined_k
        rose_model.global_stiffness_matrix = combined_k
        rose_model.track.global_stiffness_matrix = combined_k[:data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes),
                                                   :data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes)]

    def add_rose_mass(self, data, rose_model):
        r"""
        Adds rose mass matrix to scatter global mass matrix

        Generates and assembles the global mass matrix for the structure.

        Parameters
        ----------
        :param data:  mesh and geometry data class
        :param rose_model: rose coupled train track system
        """

        combined_M = self.add_rose_matrix(self.M, rose_model.global_mass_matrix, data.eq_nb_dof_rose_nodes,
                                          data.rose_eq_nb)

        # update global mass matrix of rose
        self.M = combined_M
        rose_model.global_mass_matrix = combined_M
        rose_model.track.global_mass_matrix = combined_M[:data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes),
                                                   :data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes)]

    def damping_Rayleigh(self, damp):
        r"""
        Global Rayleigh damping matrix

        Generates and assembles the Rayleigh damping matrix for the structure.

        Parameters
        ----------
        :param damp: settings for damping.
        :type damp: dict.

        :return C: Global damping matrix.
        """

        # damping and frequencies
        f1 = damp[0]
        d1 = damp[1]
        f2 = damp[2]
        d2 = damp[3]

        if f1 == f2:
            raise SystemExit('Frequencies for the Rayleigh damping are the same.')

        # damping matrix
        damp_mat = 1 / 2 * np.array([[1 / (2 * np.pi * f1), 2 * np.pi * f1],
                                     [1 / (2 * np.pi * f2), 2 * np.pi * f2]])

        damp_qsi = np.array([d1, d2])

        # solution
        coefs = np.linalg.solve(damp_mat, damp_qsi)

        self.C = self.C + (self.M.tocsr() * coefs[0] + self.K.tocsr() * coefs[1]).tolil()

    def add_rose_damping(self, data, rose_model):

        combined_C = self.add_rose_matrix(self.C, rose_model.global_damping_matrix, data.eq_nb_dof_rose_nodes, data.rose_eq_nb)

        # update global damping matrix of rose
        self.C = combined_C
        rose_model.global_damping_matrix = combined_C
        rose_model.track.global_damping_matrix = combined_C[:data.number_eq + rose_model.track.total_n_dof - len(
            data.eq_nb_dof_rose_nodes),
                                              :data.number_eq + rose_model.track.total_n_dof - len(
                                                  data.eq_nb_dof_rose_nodes)]

    def add_rose_matrix(self, scatter_matrix, rose_matrix, contact_dofs_scatter, contact_dofs_rose):
        """
        Add rose matrix to scatter matrix

        Parameters
        ----------
        :param scatter_matrix: scatter matrix
        :param rose_matrix: rose matrix
        :param scatter_data: scatter data
        :return:
        """

        # rose_matrix
        combined_matrix = lil_matrix((scatter_matrix.shape[0] + rose_matrix.shape[0] - len(contact_dofs_scatter), scatter_matrix.shape[1]
                                 + rose_matrix.shape[1] - len(contact_dofs_scatter)))

        # mask rows and columns of rose damping matrix which coincide with scatter mass matrix
        mask = np.ones(rose_matrix.shape[0], bool)
        mask[np.array(contact_dofs_rose)] = False
        masked_rose = rose_matrix.toarray()[:, mask]
        masked_rose = masked_rose[mask, :]

        # add scatter damping matrix to combined damping matrix

        # first transform data to coo matrices for efficient memory usage
        combined_matrix = combined_matrix.tocoo()
        coo_scatter = scatter_matrix.tocoo()
        reshaped_coo = coo_matrix(((coo_scatter.data), (coo_scatter.row, coo_scatter.col)), shape=combined_matrix.shape)
        combined_matrix = combined_matrix + reshaped_coo
        combined_matrix = combined_matrix.tolil()

        # add non-diagonal of rose connectivities
        combined_matrix[scatter_matrix.shape[0]:, contact_dofs_scatter] = rose_matrix.toarray()[mask, :][:, contact_dofs_rose]
        combined_matrix[contact_dofs_scatter, scatter_matrix.shape[1]:] = rose_matrix.toarray()[:, mask][contact_dofs_rose, :]

        # add rose damping matrix
        combined_matrix[scatter_matrix.shape[0]:, scatter_matrix.shape[1]:] = masked_rose

        # add diagonal of rose connectivities
        combined_matrix[contact_dofs_scatter, contact_dofs_scatter] = \
            combined_matrix[contact_dofs_scatter, contact_dofs_scatter] + rose_matrix[contact_dofs_rose, contact_dofs_rose]

        return combined_matrix

    # def absorbing_boundaries__(self, data: classmethod, material: dict, parameters: list) -> None:
    #     """
    #     Compute absorbing boundary force

    #     Parameters
    #     ----------
    #     :param data:  mesh and geometry data class
    #     :param material: material dictionary
    #     :param parameters: absorbing boundary parameters
    #     :return:
    #     """

    #     absorbing_bc = lil_matrix(self.C.shape)
    #     absorbing_bc_stiff = lil_matrix(self.C.shape)

    #     # create dictionaries of materials and nodes for quicker look-up
    #     dict_materials = dict(np.array(data.materials)[:, 1:])
    #     dict_nodes = dict([(node[0], node[1:]) for node in data.nodes])
    #     # compute material matrix for isotropic elasticity
    #     for idx, elem in enumerate(data.elem):
    #         # call shape functions
    #         if data.dimension == 3:
    #             shape_fct = discretisation.VolumeElement(data.element_type, self.order)
    #         if data.dimension == 2:
    #             # TODO: implement
    #             exit("Absorbing boundaries not implemented for 2D yet")
    #             # shape_fct = discretisation.SurfaceElement(data.element_type, self.order)

    #         # material index
    #         mat_idx = data.materials_index[idx]
    #         # find material name
    #         name_material = dict_materials[str(mat_idx)]

    #         # solid elastic properties
    #         rho = material[name_material]["density"]
    #         E = material[name_material]["Young"]
    #         v = material[name_material]["poisson"]

    #         # computation of velocities
    #         Ec = E * (1 - v) / ((1 + v) * (1 - 2 * v))
    #         G = E / (2 * (1 + v))
    #         vp = np.sqrt(Ec / rho)
    #         vs = np.sqrt(G / rho)

    #         # check if abs boundary exist
    #         xyz_ = []
    #         for node in elem:
    #             # get global coordinates of the node
    #             idx_node = np.where(data.nodes[:, 0] == node)[0][0]
    #             if "Absorb" in data.type_BC[idx_node]:
    #                 xyz_.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

    #         # if there are no absorbing boundaries goes to next element
    #         if len(xyz_) == 0:
    #             continue

    #         # coordinates for all the nodes in one element
    #         xyz = np.array([dict_nodes[node] for node in elem])

    #         # generate shape functions B and H matrix
    #         shape_fct.generate(xyz)

    #         # compute unitary absorbing boundary force
    #         abs_bound = shape_fct.compute_abs_bound()

    #         # assemble absorbing boundary matrix
    #         # equation number where the absorbing matrix exists
    #         i1 = data.eq_nb_elem[idx][(~np.isnan(data.eq_nb_elem[idx])) & (data.type_BC_elem[idx] == "Absorb")]
    #         id1 = np.argsort(i1)
    #         i1 = np.sort(i1)
    #         i2 = np.where((~np.isnan(data.eq_nb_elem[idx])) & (data.type_BC_elem[idx] == "Absorb"))[0]
    #         i2 = i2[id1]

    #         # assign the absorbing boundary coefficients: vp for perpendicular vs otherwise
    #         fct = np.ones(len(i1)) * parameters[1] * rho * vs
    #         fct2 = np.ones(len(i1)) * G
    #         for i, val in enumerate(i1):
    #             j = np.where(data.eq_nb_dof == val)
    #             direct = data.BC_dir[j[0], j[1]]
    #             if direct == int(1):
    #                 fct[i] = abs(parameters[0] * rho * vp)
    #                 fct2[i] = Ec

    #         # assign to the global absorbing boundary force
    #         absorbing_bc[i1.reshape(len(i1), 1), i1] = absorbing_bc[i1.reshape(len(i1), 1), i1] + \
    #                                                     abs_bound[i2.reshape(len(i2), 1), i2] * fct * 0.2 *0.2

    #     self.C = self.C + absorbing_bc
    #     # self.K = self.K + absorbing_bc_stiff / 10000

    def absorbing_boundaries(self, data: classmethod, material: dict, parameters_viscous: list, parameters_stiff: float) -> None:
        """
        Compute absorbing boundary force

        Parameters
        ----------
        :param data:  mesh and geometry data class
        :param material: material dictionary
        :param parameters: viscous absorbing boundary parameters
        :param parameters_stiff: stiffness absorbing boundary parameters
        """

        absorbing_bc = lil_matrix(self.C.shape)
        absorbing_bc_stiff = lil_matrix(self.C.shape)

        # create dictionaries of materials and nodes for quicker look-up
        dict_materials = dict(np.array(data.materials)[:, 1:])
        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):
            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = dict_materials[str(mat_idx)]

            # solid elastic properties
            rho = material[name_material]["density"]
            E = material[name_material]["Young"]
            v = material[name_material]["poisson"]

            # computation of velocities
            Ec = E * (1 - v) / ((1 + v) * (1 - 2 * v))
            G = E / (2 * (1 + v))
            vp = np.sqrt(Ec / rho)
            vs = np.sqrt(G / rho)

            # check if abs boundary exist
            index_nodes = []
            xyz = []  # coordinates of the nodes in the element
            xyz_abs = []  # coordinates of the nodes in the element where abs boundary exists
            eq_nb = []  # equation number of the nodes in the element where abs boundary exists
            # bc_dir = []  # perpendicular direction of the nodes in the element where abs boundary exists
            bc_type = []
            for node in elem:
                # get global coordinates of the node
                idx_node = np.where(data.nodes[:, 0] == node)[0][0]
                index_nodes.append(idx_node)
                xyz.append(data.nodes[idx_node, 1:])
                if "Absorb" in data.type_BC[idx_node]:
                    bc_type.append(data.type_BC[idx_node])
                    xyz_abs.append(data.nodes[data.nodes[:, 0] == node, 1:][0])
                    eq_nb.append(data.eq_nb_dof[idx_node])
                    # bc_dir.append(data.BC_dir[idx_node])

            # if there are no absorbing boundaries goes to next element
            if not bc_type:
                continue

            # find indices of surfaces that have absorbing boundaries
            idx_surfaces = [i for i in range(data.dimension) if len(np.where(np.array(bc_type)[:, i] == "Absorb")[0]) == data.nb_nodes_lower_elem]

            # apply abs boundary to each surface
            for idx_bc in idx_surfaces:
                # get the coordinates of the surface without the common coordinate
                idx_col = np.where(np.array(bc_type)[:, idx_bc] == "Absorb")[0]
                xy_abs = np.delete(np.array(xyz_abs)[idx_col, :], idx_bc, axis=1)

                if data.dimension == 3:
                    shape_fct_surf = discretisation.SurfaceElement(data.lower_element_type, self.order)
                if data.dimension == 2:
                    # TODO: implement
                    exit("Absorbing boundaries not implemented for 2D yet")
                    # shape_fct = discretisation.SurfaceElement(data.lower_element_type, self.order)

                # sort the nodes in clockwise order following gmsh order
                xy_abs = clockwise_sort_2D_elements(xy_abs)

                # generate shape functions B and H matrix
                shape_fct_surf.generate(xy_abs)

                # compute unitary absorbing boundary force
                abs_bound = shape_fct_surf.compute_abs_bound()

                # extend abs_bound to 3D
                abs_extended = np.copy(abs_bound)

                # increase the columns
                new_column = np.zeros(abs_bound.shape[0])
                for i in range(data.nb_nodes_lower_elem):
                    abs_extended = np.insert(abs_extended, 2 + (i * data.dimension), new_column, axis=1)

                # increase the rows
                new_row = np.zeros(abs_extended.shape[1])
                for i in range(data.nb_nodes_lower_elem):
                    new_row = np.copy(abs_extended[2 + (i * data.dimension)-1, :])
                    new_row = np.concatenate(([new_row[-1]], new_row[:-1]))
                    abs_extended = np.insert(abs_extended, 2 + (i * data.dimension), new_row, axis=0)

                # assemble absorbing boundary matrix
                # equation number where the absorbing matrix exists
                aux = np.array(eq_nb)[idx_col, idx_bc]
                i1 = np.sort(aux).astype(int)
                i2 = np.linspace(idx_bc, idx_bc + (data.nb_nodes_lower_elem - 1) * (data.dimension), data.nb_nodes_lower_elem, dtype=int)

                # assign the absorbing boundary coefficients: vp for perpendicular vs otherwise
                fct = np.ones(len(i1)) * parameters_viscous[1] * rho * vs
                fct2 = np.ones(len(i1)) * G
                for i, val in enumerate(i1):
                    j = np.where(data.eq_nb_dof == val)
                    direct = data.BC_dir[j[0], j[1]]
                    if direct == 1:
                        fct[i] = parameters_viscous[0] * rho * vp
                        fct2[i] = Ec

                # # assign to the global damping matrix dictionary
                absorbing_bc[i1.reshape(len(i1), 1), i1] = absorbing_bc[i1.reshape(len(i1), 1), i1] +\
                                                              abs_extended[i2.reshape(len(i2), 1), i2] * fct

                absorbing_bc_stiff[i1.reshape(len(i1), 1), i1] += np.abs(abs_extended[i2.reshape(len(i2), 1), i2]) * fct2

        self.C = self.C + absorbing_bc
        self.K = self.K + absorbing_bc_stiff / parameters_stiff
