def stiffness_elasticity(E, poisson):
    """ Stifness matrix for isotropic elastic material

        $\stress = \frac{1}{E} \times D \times \vareplison$
        """

    import numpy as np

    D = np.zeros((6, 6))

    D[:3, :3] = [[1. - poisson, poisson, poisson],
                 [poisson, 1. - poisson, poisson],
                 [poisson, poisson, 1. - poisson]]

    D[3:, 3:] = [[1. - 2. * poisson, poisson, poisson],
                 [poisson, 1. - 2. * poisson, poisson],
                 [poisson, poisson, 1. - 2. * poisson]]

    D *= E / ((1. + poisson) * (1. - 2. * poisson))

    return D


class GenerateMatrix:
    def __init__(self, nb_equations, order):
        # import packages
        import numpy as np
        from scipy.sparse import lil_matrix

        # generation of variable
        self.M = lil_matrix(np.zeros((nb_equations, nb_equations)))
        self.C = lil_matrix(np.zeros((nb_equations, nb_equations)))
        self.K = lil_matrix(np.zeros((nb_equations, nb_equations)))

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
        import numpy as np

        # compute material matrix for isotropic elasticity
        for idx, elem in enumerate(data.elem):

            # element type
            elem_type = elem[1]

            # call shape functions
            N = shape_functions.ShapeFunction(elem_type, self.order)

            # material index
            mat_idx = elem[2]

            # find material index
            for i in data.materials:
                if i[1] == mat_idx:
                    key = i[2]

            # solid elastic properties
            E = material[key][1]
            v = material[key][2]

            # element stiffness matrix
            D = stiffness_elasticity(E, v)

            # coordinates for all the nodes in one element
            xyz = []
            for node in elem[5:]:
                # get global coordinates of the node
                xyz.append(data.nodes[data.nodes[:, 0] == node, 1:][0])

            # call shape function
            N.generate()
            # Jacobian
            N.jacob(xyz)
            # matrix B strain-displacement
            N.matrix_B()
            # compute stiffness
            Ke = N.compute_stiffness(D)
            # assemble
            i1 = data.LM[idx][~np.isnan(data.LM[idx])]
            i2 = np.where(~np.isnan(data.LM[idx]))[0]
            self.K[(i1 - 1).reshape(len(i1), 1), i1 - 1] += Ke[i2, i2]

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
            N = shape_functions.ShapeFunction(elem_type, self.order)

            # material index
            mat_idx = elem[2]

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
            N.generate()
            # Jacobian
            N.jacob(xyz)

            # displacement interpolation matrix
            N.int_H()

            # compute mass
            Me = N.compute_mass(rho)

            # assemble
            i1 = data.LM[idx][~np.isnan(data.LM[idx])]
            i2 = np.where(~np.isnan(data.LM[idx]))[0]
            self.M[(i1 - 1).reshape(len(i1), 1), i1 - 1] += Me[i2, i2]

        return

    def damping_Rayleigh(self, settings):
        r"""
        Rayleigh damping generation.

        Generates and assembles the Rayleigh damping matrix for the structure.

        :param data: data.
        :type data: class.

        :return C: Global stiffness matrix.
        """
        
        self.C = settings['alpha'] * self.M + settings['beta'] * self.K
        
        return