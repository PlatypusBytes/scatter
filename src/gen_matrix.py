class GenerateMatrix:
    def __init__(self, nb_equations, order):
        # import packages
        import numpy as np
        from scipy.sparse import lil_matrix

        # generation of variable
        self.M = lil_matrix((nb_equations, nb_equations))
        self.C = lil_matrix((nb_equations, nb_equations))
        self.K = lil_matrix((nb_equations, nb_equations))

        # order of the Gauss integration
        self.order = order

        return
    
    def stiffness(self, data, material):
        r"""
        Global stiffness matrix generation.

        Generates and assembles the global stiffness matrix for the structure.

        :param data: data.
        :type data: class.
        :param material: list of materials.
        :type data: dict.

        :return K: Global stiffness matrix.
        """
    
        # import packages
        import shape_functions
        import material_models
        import numpy as np

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # element type
            elem_type = elem[1]

            # call shape functions
            shape_fct = shape_functions.ShapeFunction(elem_type, self.order)

            # material index
            mat_idx = elem[3]

            # find material index
            for i in data.materials:
                if i[1] == mat_idx:
                    key = i[2]

            # solid elastic properties
            E = material[key][1]
            v = material[key][2]

            # element stiffness matrix
            D = material_models.stiffness_elasticity(E, v)

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem[5:]:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # call shape function
            shape_fct.generate()
            # Jacobian
            shape_fct.jacob(xyz)
            # matrix B strain-displacement
            shape_fct.matrix_B()
            # compute stiffness
            Ke = shape_fct.compute_stiffness(D)

            # assemble Stiffness matrix
            # equation number where the mass matrix exists
            i1 = data.eq_nb_elem[idx][~np.isnan(data.eq_nb_elem[idx])]
            # index where mass matrix exists
            i2 = np.where(~np.isnan(data.eq_nb_elem[idx]))[0]

            # assign to the global stiffness matrix
            self.K[i1.reshape(len(i1), 1), i1] += Ke[i2.reshape(len(i2), 1), i2]

        return

    def mass(self, data, material):
        r"""
        Global mass matrix generation.

        Generates and assembles the global mass matrix for the structure.

        :param data: data.
        :type data: class.
        :param material: list of materials.
        :type data: dict.

        :return M: Global mass matrix.
        """

        # import packages
        import shape_functions
        import numpy as np

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # element type
            elem_type = elem[1]

            # call shape functions
            shape_fct = shape_functions.ShapeFunction(elem_type, self.order)

            # material index
            mat_idx = elem[3]

            # find material index
            for i in data.materials:
                if i[1] == mat_idx:
                    key = i[2]

            # solid elastic properties
            rho = material[key][0]

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem[5:]:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # call shape function
            shape_fct.generate()
            # Jacobian
            shape_fct.jacob(xyz)

            # displacement interpolation matrix
            shape_fct.int_H()

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
        Rayleigh damping generation.

        Generates and assembles the Rayleigh damping matrix for the structure.

        :param data: data.
        :type data: class.

        :return C: Global stiffness matrix.
        """

        # import packages
        import numpy as np
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

        self.C = self.M.tocsr().dot(coefs[0]) + self.K.tocsr().dot(coefs[1])
        
        return
