import numpy as np
from scipy.sparse import lil_matrix

# import scatter packages
from src import shape_functions
from src import material_models


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
        self.absorbing_bc = lil_matrix((nb_equations, nb_equations))

        # order of the Gauss integration
        self.order = order

        return
    
    def stiffness(self, data: classmethod, material: dict) -> None:
        r"""
        Global stiffness matrix generation.

        Generates and assembles the global stiffness matrix for the structure.

        Parameters
        ----------
        :param data: mesh and geometry data class
        :param material: material dictionary
        """

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # call shape functions
            shape_fct = shape_functions.ShapeFunctionVolume(data.element_type, self.order)

            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = [i[2] for i in data.materials if i[1] == mat_idx][0]

            # solid elastic properties
            E = material[name_material]["Young"]
            v = material[name_material]["poisson"]

            # element stiffness matrix
            D = material_models.stiffness_elasticity(E, v)

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # generate shape functions B and H matrix
            shape_fct.generate(xyz)

            # compute stiffness
            Ke = shape_fct.compute_stiffness(D)

            # assemble Stiffness matrix
            # equation number where the stiff matrix exists
            i1 = data.eq_nb_elem[idx][~np.isnan(data.eq_nb_elem[idx])]
            # index where mass matrix exists
            i2 = np.where(~np.isnan(data.eq_nb_elem[idx]))[0]

            # assign to the global stiffness matrix
            self.K[i1.reshape(len(i1), 1), i1] += Ke[i2.reshape(len(i2), 1), i2]

        return

    def mass(self, data: classmethod, material: dict) -> None:
        r"""
        Global mass matrix generation.

        Generates and assembles the global mass matrix for the structure.

        Parameters
        ----------
        :param data:  mesh and geometry data class
        :param material: material dictionary
        """

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # call shape functions
            shape_fct = shape_functions.ShapeFunctionVolume(data.element_type, self.order)

            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = [i[2] for i in data.materials if i[1] == mat_idx][0]

            # solid elastic properties
            rho = material[name_material]["density"]

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # generate shape functions B and H matrix
            shape_fct.generate(xyz)

            # compute mass
            Me = shape_fct.compute_mass(rho)

            # assemble Mass matrix
            # equation number where the mass matrix exists
            i1 = data.eq_nb_elem[idx][~np.isnan(data.eq_nb_elem[idx])]
            # index where mass matrix exists
            i2 = np.where(~np.isnan(data.eq_nb_elem[idx]))[0]

            # assign to the global mass matrix
            self.M[i1.reshape(len(i1), 1), i1] += Me[i2.reshape(len(i2), 1), i2]

        return

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

        self.C = (self.M.tocsr().dot(coefs[0]) + self.K.tocsr().dot(coefs[1])).tolil()
        
        return

    def absorbing_boundaries(self, data: classmethod, material: dict, parameters: list) -> None:
        """
        Compute absorbing boundary force

        Parameters
        ----------
        :param data:  mesh and geometry data class
        :param material: material dictionary
        :param parameters: absorbing boundary parameters
        :return:
        """

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # call shape functions
            shape_fct = shape_functions.ShapeFunctionVolume(data.element_type, self.order)

            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = [i[2] for i in data.materials if i[1] == mat_idx][0]

            # solid elastic properties
            rho = material[name_material]["density"]
            E = material[name_material]["Young"]
            v = material[name_material]["poisson"]

            # computation of velocities
            Ec = E / (3 * (1 - 2 * v))
            G = E / (2 * (1 + v))
            vp = np.sqrt(Ec / rho)
            vs = np.sqrt(G / rho)

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # generate shape functions B and H matrix
            shape_fct.generate(xyz)

            # compute mass
            Me = shape_fct.compute_mass(1)

            # assemble Mass matrix
            # equation number where the mass matrix exists
            i1 = data.eq_nb_elem[idx][(~np.isnan(data.eq_nb_elem[idx])) & (data.type_BC_elem[idx] == "Absorb")]
            # index where mass matrix exists
            i2 = np.where((~np.isnan(data.eq_nb_elem[idx])) & (data.type_BC_elem[idx] == "Absorb"))[0]

            # find common axis in nodes of the element
            xyz_ = []
            id_dof = []
            for node in elem:
                # get global coordinates of the node
                idx_node = np.where(data.nodes[:, 0] == node)[0][0]
                if "Absorb" in data.type_BC[idx_node]:
                    # index where absorb is
                    id_dof.append([node, data.type_BC[idx_node]])
                    xyz_.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            if len(xyz_) == 0:
                continue

            xyz_ = np.array(xyz_)
            # index that it is common: direction of the compression
            idx_xy = np.where((xyz_ == xyz_[0, :]).all(0))[0][0]

            fct = np.ones(len(i2)) * parameters[1] * rho * vs

            for i, val in enumerate(i1):
                # find column where it is
                idx = np.where(data.eq_nb_dof == val)[1][0]
                # if the column is the same as the common plane: apply vp
                if idx == idx_xy:
                    fct[i] = parameters[0] * rho * vp

            # assign to the global mass matrix
            self.absorbing_bc[i1.reshape(len(i1), 1), i1] += np.dot(Me[i2.reshape(len(i2), 1), i2], fct)

        return
