class Generate_Matrix:
    def __init__(self):
        return
    
    def stiffness(self, data):
        r"""
        Global stiffness generation.

        Generates and assembles the global stiffness matrix for the structure.

        :param data: data.
        :type data: class.

        :return K: Global stiffness matrix.
        """
    
        # import packages
        import numpy as np
        from scipy.sparse import lil_matrix

        # generation of variable
        self.K = lil_matrix(np.zeros((data.coord.shape[0] * 6, data.coord.shape[0] * 6)))
    
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