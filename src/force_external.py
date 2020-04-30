import sys

class Force:
    def __init__(self):
        self.force = []
        return

    def pulse_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, steps=5):
        import numpy as np
        from scipy.sparse import lil_matrix

        time = load_set["time"]
        time = np.linspace(0, time, np.ceil(time / time_step))
        # generation of variable
        self.force = lil_matrix((nb_equations, len(time)))

        factor = load_set["force"]

        # check that length of computation is bigger than the number of steps
        if len(time) <= steps:
            sys.exit("Error: Number of loading steps smaller than " + str(steps))

        # for each node with load
        for n in node:
            for i, eq in enumerate(eq_nb_dof[n - 1]):
                if ~np.isnan(eq):
                    # pulse in steps
                    for k in range(steps):
                        self.force[int(eq), k] = float(factor[i]) * np.append(np.linspace(0, 1, int((steps - 1)/2), endpoint=False),
                                                                              np.linspace(1, 0, int((steps + 1)/2)))[k]

        return

    def heaviside_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, steps=5):
        import numpy as np
        from scipy.sparse import lil_matrix

        # time
        time = load_set["time"]
        time = np.linspace(0, time, int(np.ceil(time / time_step)))
        # generation of variable
        self.force = lil_matrix(np.zeros((nb_equations, len(time))))

        factor = load_set["force"]

        # check that length of computation is bigger than the number of steps
        if len(time) <= steps:
            sys.exit("Error: Number of loading steps smaller than " + str(steps))

        # for each node with load
        for n in node:
            for i, eq in enumerate(eq_nb_dof[n - 1]):
                if ~np.isnan(eq):
                    # smooth over steps
                    for k in range(steps):
                        self.force[int(eq), k] = float(factor[i]) * np.linspace(0, 1, steps)[k]
                    self.force[int(eq), steps:] = float(factor[i])

        return
