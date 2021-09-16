import rose
import numpy as np

from scipy.sparse import lil_matrix

class RoseUtils():

    def __init__(self):

        self.train = None

    def add_train_to_matrices(self,M, C, K, F, mesher):
        new_M = lil_matrix(self.train.total_n_dof + M.size[0], self.train.total_n_dof + M.size[0])
        new_C = lil_matrix(self.train.total_n_dof + C.size[0], self.train.total_n_dof + C.size[0])
        new_K = lil_matrix(self.train.total_n_dof + K.size[0], self.train.total_n_dof + K.size[0])
        new_F = lil_matrix(self.train.total_n_dof + F.size[0], F.size[1])

        # add scatter system to new matrices
        new_M[:M.size[0], :M.size[0]] = M
        new_C[:C.size[0], :C.size[0]] = C
        new_K[:K.size[0], :K.size[0]] = K
        new_F[:F.size[0], :] = F

        # add train system to new matrices
        new_M[M.size[0]:, M.size[0]:] = self.train.global_mass_matrix
        new_C[C.size[0]:, C.size[0]:] = self.train.global_damping_matrix
        new_K[K.size[0]:, K.size[0]:] = self.train.global_stiffness_matrix
        new_F[K.size[0]:, :] = self.train.global_force_vector

        # update number of equations
        new_number_eq = mesher.numer_eq + self.train.total_n_dof

        return new_M, new_C, new_K, new_F, new_number_eq


#    moving_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, nodes_coord, steps=50)
    def create_train_load(self):
        """
              Moving load along z-axis

              :param nb_equations: total number of equations
              :param eq_nb_dof: number of equation for each dof in node list
              :param load_set: loading settings
              :param node: node where the load starts
              :param time_step: time step
              :param nodes_coord: list of coordinates
              :param steps: (optional: default = 50) number of steps to initialise load
              """
        # time

        time = self.train.time


        for wheel in self.train.wheels:


        # time = load_set["time"]
        # time = np.linspace(0, time, int(np.ceil(time / time_step)))
        # generation of variable
        self.force = lil_matrix(np.zeros((nb_equations, len(time))))
        # load factor
        factor = load_set["force"]

        # index node
        idx = np.where(nodes_coord[:, 0] == node)[0][0]
        # find nodes along  with same x and y (load moves along z-axis)
        idx_list = np.where((nodes_coord[:, 1] == nodes_coord[idx, 1]) & (nodes_coord[:, 2] == nodes_coord[idx, 2]))[0]

        # find distances
        dist = []
        for i in idx_list:
            dist.append(np.sqrt((nodes_coord[i, 3] - nodes_coord[idx, 3]) ** 2))

        idx_list = idx_list[np.argsort(np.array(dist))]
        dist = np.sort(np.array(dist))

        # for each time in the analysis
        for t in range(len(time)):

            # load not moving while steps
            if t < steps:
                speed = 0
                fct = np.linspace(0, 1, steps)[t]
            else:
                speed = load_set["speed"]
                fct = 1

            # if the load as reached the end of the model returns
            if speed * (time[t] - time[steps - 1]) >= np.max(dist):
                return

            # find location of the load
            id = np.where(dist <= speed * (time[t] - time[steps - 1]))[0][-1]
            node = [int(nodes_coord[idx_list[id], 0]), int(nodes_coord[idx_list[id + 1], 0])]

            # compute local shape functions
            x = speed * (time[t] - time[steps - 1]) + nodes_coord[idx_list[id], 3]
            l = dist[id + 1] - dist[id]

            shp = np.array([1 - x / l,
                            x * l])

            # for each node with load
            for j, n in enumerate(node):
                for i, eq in enumerate(eq_nb_dof[n - 1]):
                    if ~np.isnan(eq):
                        self.force[int(eq), t] = float(factor[i]) * shp[j] * fct

        return



