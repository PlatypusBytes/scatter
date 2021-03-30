import sys
import numpy as np
from scipy.sparse import lil_matrix


class Force:
    def __init__(self):
        self.force = []
        return

    def pulse_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, steps=5):

        time = load_set["time"]
        time = np.linspace(0, time, int(np.ceil(time / time_step)))
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

    def moving_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, nodes_coord, steps=5):

        # time
        time = load_set["time"]
        time = np.linspace(0, time, int(np.ceil(time / time_step)))
        # generation of variable
        self.force = lil_matrix(np.zeros((nb_equations, len(time))))

        factor = load_set["force"]

        # speed
        speed = load_set["speed"]

        # index node
        idx = np.where(nodes_coord[:, 0] == node)[0][0]
        # find nodes along  with same x and y (load moves along z-axis)
        idx_list = np.where((nodes_coord[:, 1] == nodes_coord[idx, 1]) & (nodes_coord[:, 2] == nodes_coord[idx, 2]))[0]

        # find distances
        dist = []
        for i in idx_list:
            dist.append(np.sqrt((nodes_coord[i, 3] - nodes_coord[idx, 3])**2))

        idx_list = idx_list[np.argsort(np.array(dist))]
        dist = np.sort(np.array(dist))

        # for each time in the analysis
        for t in range(len(time)):

            # if the load as reached the end of the model returns
            if speed * time[t] >= np.max(dist):
                return

            # find location of the load
            id = np.where(dist <= speed * time[t])[0][-1]
            node = [int(nodes_coord[idx_list[id], 0]), int(nodes_coord[idx_list[id + 1], 0])]

            # compute local shape functions
            x = speed * time[t] + nodes_coord[idx_list[id], 3]
            l = dist[id + 1] - dist[id]

            shp = np.array([1 - x / l,
                            x * l])

            # for each node with load
            for j, n in enumerate(node):
                for i, eq in enumerate(eq_nb_dof[n - 1]):
                    print(eq)
                    if ~np.isnan(eq):
                        self.force[int(eq), t] = float(factor[i]) * shp[j]

        return
