import numpy as np
from scipy.sparse import lil_matrix

# import scatter packages
from src import discretisation
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
            if data.dimension == 3:
                shape_fct = discretisation.VolumeElement(data.element_type, self.order)
            if data.dimension == 2:
                shape_fct = discretisation.SurfaceElement(data.element_type, self.order)

            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = [i[2] for i in data.materials if i[1] == mat_idx][0]

            # solid elastic properties
            E = material[name_material]["Young"]
            v = material[name_material]["poisson"]

            # element stiffness matrix
            D = material_models.stiffness_elasticity(E, v, data.dimension)

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # generate shape functions B and H matrix
            shape_fct.generate(np.array(xyz))

            # compute stiffness
            Ke = shape_fct.compute_stiffness(D)

            # assemble Stiffness matrix
            # equation number where the stiff matrix exists
            i1 = data.eq_nb_elem[idx][~np.isnan(data.eq_nb_elem[idx])]
            # index where stiffness matrix exists
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
            if data.dimension == 3:
                shape_fct = discretisation.VolumeElement(data.element_type, self.order)
            if data.dimension == 2:
                shape_fct = discretisation.SurfaceElement(data.element_type, self.order)

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
            shape_fct.generate(np.array(xyz))

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
        # ToDo: improve this with surface elements. this is not very well done at the moment

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):
            # call shape functions
            if data.dimension == 3:
                shape_fct = discretisation.VolumeElement(data.element_type, self.order)
            if data.dimension == 2:
                # ToDo: implement
                shape_fct = discretisation.SurfaceElement(data.element_type, self.order)

            # material index
            mat_idx = data.materials_index[idx]
            # find material name
            name_material = [i[2] for i in data.materials if i[1] == mat_idx][0]

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
            xyz_ = []
            for node in elem:
                # get global coordinates of the node
                idx_node = np.where(data.nodes[:, 0] == node)[0][0]
                if "Absorb" in data.type_BC[idx_node]:
                    xyz_.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # if there are no absorbing boundaries goes to next element
            if len(xyz_) == 0:
                continue
            else:
                if data.dimension == 2:
                    exit("ERROR: BC not supported in 2D")

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # generate shape functions B and H matrix
            shape_fct.generate(xyz)

            # compute unitary absorbing boundary force
            abs_bound = shape_fct.compute_abs_bound()

            # assemble absorbing boundary matrix
            # equation number where the absorbing matrix exists
            i1 = data.eq_nb_elem[idx][(~np.isnan(data.eq_nb_elem[idx])) & (data.type_BC_elem[idx] == "Absorb")]
            i2 = np.where((~np.isnan(data.eq_nb_elem[idx])) & (data.type_BC_elem[idx] == "Absorb"))[0]

            # assign the absorbing boundary coefficients: vp for perpendicular vs otherwise
            fct = np.ones(len(i1)) * parameters[1] * rho * vs
            for i, val in enumerate(i1):
                j = np.where(data.eq_nb_dof == val)
                direct = data.type_BC_dir[j[0], j[1]]
                if direct == int(1):
                    fct[i] = parameters[0] * rho * vp

            # assign to the global absorbing boundary force
            self.absorbing_bc[i1.reshape(len(i1), 1), i1] += abs_bound[i2.reshape(len(i2), 1), i2] * fct

        return
