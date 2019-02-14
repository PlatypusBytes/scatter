class Force:
    def __init__(self):
        self.force = []
        return

    # ToDo the force is not applied on the correct placed due to the np.nan: fix this
    def pulse_load(self, nb_equations, ID, load_set, node):
        import numpy as np
        from scipy.sparse import lil_matrix

        # time
        time = load_set["time"]
        time = np.linspace(0, float(time), 100)
        # generation of variable
        self.force = lil_matrix(np.zeros((nb_equations, len(time))))

        factor = load_set["force"]

        for i, eq in enumerate(ID[node - 1]):
            if ~np.isnan(eq):
                self.force[int(eq), 1] = 1.0 * float(factor[i])

        return

    def heaviside_load(self, nb_equations, eq_nb_dof, load_set, node, time_step):
        import numpy as np
        from scipy.sparse import lil_matrix

        # time
        time = load_set["time"]
        time = np.linspace(0, time, np.ceil(time / time_step))
        # generation of variable
        self.force = lil_matrix(np.zeros((nb_equations, len(time))))

        factor = load_set["force"]

        # for each node with load
        for n in node:
            for i, eq in enumerate(eq_nb_dof[n - 1]):
                if ~np.isnan(eq):
                    self.force[int(eq), 1:] = float(factor[i])

        return
