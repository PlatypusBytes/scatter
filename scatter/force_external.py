import sys
import numpy as np
import scipy.spatial.kdtree
from scipy.sparse import lil_matrix

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from scatter.rose_utils import RoseUtils


class Force:
    def __init__(self):
        self.force_vector = []

        self.factor = None
        self.contact_nodes = None
        self.step_factors = None

    def initialise_load(self,  load_set, time, model, solver, **kwargs):
        """
       initialises load,
       #todo allow different elements than hexa8 for moving load at plane

       :param load_set: loading settings
       :param time: time array
       :param model: scatter model
       :param solver: numerical solver
       :param kwargs: key word arguments
       """
        self.nb_equations = model.number_eq # total number of equations
        self.factor = load_set.get("force") # force value
        self.contact_nodes = load_set.get("node") # nodes numbers where force is located
        self.eq_nb_dof = model.eq_nb_dof  # number of equation for each dof in node list
        self.model_nodes = model.nodes # all model nodes
        self.time = time # time array
        self.steps = load_set["ini_steps"] #  number of steps to apply the load following a triangular form
        self.loading_type = load_set["type"] # loading type
        self.solver = solver

        if self.loading_type == "pulse":
            self.initialise_pulse_load()
        elif self.loading_type == "heaviside":
            self.initialise_heaviside_load()
        elif self.loading_type == "moving_at_plane":
            self.initialise_moving_load_at_plane(kwargs["top_surface_elements"], load_set["speed"],
                                                 load_set["direction"], load_set["start_coord"])
        elif self.loading_type == "moving":
            self.initialise_moving_load(load_set["speed"])
        elif self.loading_type == "rose":
            self.initialise_rose_load(model, load_set["model"], self.solver)
        else:
            sys.exit(f'Error: Load type {load_set["type"]} not supported')


    def update_load_at_t(self, t, **kwargs):
        """
       Updates load at timestep t,

       :param t: time index
       :param kwargs: key word arguments, which is required for numerical solver
       """

        if self.loading_type == "pulse":
            self.update_pulse_load(t)
        elif self.loading_type == "heaviside":
            self.update_heaviside_load(t)
        elif self.loading_type == "moving_at_plane":
            self.update_moving_load_at_plane(t)
        elif self.loading_type == "moving":
            self.update_moving_load(t)
        elif self.loading_type == "rose":
            self.update_rose_load(t)

        return self.force_vector

    def initialise_pulse_load(self):

        # check that length of computation is bigger than the number of steps
        if len(self.time) <= self.steps:
            sys.exit("Error: Number of loading steps smaller than " + str(self.steps))

        self.step_factors = np.append(np.linspace(0, 1, int((self.steps - 1) / 2), endpoint=False),
                                                               np.linspace(1, 0, int((self.steps + 1) / 2)))

        self.update_pulse_load(0)

    def initialise_heaviside_load(self):

        # check that length of computation is bigger than the number of steps
        if len(self.time) <= self.steps:
            sys.exit("Error: Number of loading steps smaller than " + str(self.steps))

        initialise_steps = np.linspace(0, 1, self.steps)
        self.step_factors = np.ones(len(self.time))
        self.step_factors[:self.steps] = initialise_steps

        self.update_heaviside_load(0)

    def initialise_moving_load(self, load_speed):
        """
        Moving load along z-axis

        :param nb_equations: total number of equations
        :param eq_nb_dof: number of equation for each dof in node list
        :param nodes_list: list of nodes
        :param load_set: loading settings
        :param time: list of time
        :param nodes_coord: list of coordinates
        :param steps: (optional: default = 50) number of steps to initialise load
        """

        # check that length of computation is bigger than the number of steps
        if len(self.time) <= self.steps:
            sys.exit("Error: Number of loading steps smaller than " + str(self.steps))

        # set loading factor
        initialise_steps = np.linspace(0, 1, self.steps)
        self.step_factors = np.ones(len(self.time))
        self.step_factors[:self.steps] = initialise_steps

        # generation of variable

        # index node
        idx = np.where(self.model_nodes[:, 0] == self.contact_nodes)[0][0]
        # find nodes along  with same x and y (load moves along z-axis)
        self.idx_list = \
        np.where((self.model_nodes[:, 1] == self.model_nodes[idx, 1]) &
                 (self.model_nodes[:, 2] == self.model_nodes[idx, 2]))[0]

        # find distances, z-distances only ??
        dist = []
        for i in self.idx_list:
            dist.append(np.sqrt((self.model_nodes[i, 3] - self.model_nodes[idx, 3]) ** 2))

        self.idx_list = self.idx_list[np.argsort(np.array(dist))]
        dist = np.sort(np.array(dist))

        self.node_distances = dist

        # define speed at each time step
        speed_array = np.ones(len(self.time)) * load_speed
        speed_array[:self.steps] = 0

        # calculate load distances at each time step
        self.load_distances = speed_array * (self.time - self.time[self.steps])


        self.update_moving_load(0)
        pass

    def initialise_moving_load_at_plane(self, xz_plane_elements, load_speed, load_direction, start_coord):

        # get coordinates of each element on the xz plane
        elem_coordinates = self.model_nodes[xz_plane_elements - 1, 1:]

        # add elem x,z coordinates to shapely polygons, get convex hull such that coordinate sorting is correct
        polygons = np.array([Polygon(elem).convex_hull for elem in elem_coordinates[:, :, [0, 2]]])

        # set loading factor
        initialise_steps = np.linspace(0, 1, self.steps)
        self.step_factors = np.ones(len(self.time))
        self.step_factors[:self.steps] = initialise_steps

        # calculate traveled distance of point load at each time step
        dt = np.diff(self.time[self.steps:])
        distance = np.zeros(len(self.time))
        distance[self.steps:] = np.append(0, np.cumsum(load_speed * dt))

        # calculate angle of the direction of the moving point load
        if np.isclose(load_direction[0], 0):
            if load_direction[1] > 0:
                angle_direction = 0.5 * np.pi
            else:
                angle_direction = -0.5 * np.pi
        else:
            angle_direction = np.arctan(load_direction[1] / load_direction[0])

        # calculate position of the point load at each time step
        x_coordinates = np.cos(angle_direction) * distance + start_coord[0]
        z_coordinates = np.sin(angle_direction) * distance + start_coord[1]
        self.position = np.array([x_coordinates, z_coordinates])

        # define kdtree of centroids of all surface polygons, to speed up search of active elements
        centroids = np.array([np.array(polygon.centroid.xy)[:, 0] for polygon in polygons])
        tree = scipy.spatial.kdtree.KDTree(centroids)

        # todo, make nbr_nearest_neightbours more general
        nbr_nearest_neighbours = 10

        # find elements which are in contact with the moving load at each time step
        self.active_elements = []
        for coord in self.position.T:

            # get nearest centroids of polygons to speed up search
            nearest_centroids_indices = tree.query(coord, nbr_nearest_neighbours)[1]
            near_polygons = polygons[nearest_centroids_indices]

            # find elements which contain the point load, only elements with near polygons are checked
            found_element = False
            for idx, polygon in enumerate(near_polygons):
                if polygon.contains(Point(coord)):
                    element_idx = nearest_centroids_indices[idx]
                    self.active_elements.append(xz_plane_elements[element_idx])
                    found_element = True
                    break

            # if no polygon is found, add none to active elements
            if not found_element:
                self.active_elements.append(None)

        self.update_moving_load_at_plane(0)


    def initialise_rose_load(self,scatter_model, rose_model, solver):

        """
        Adds rose load to global force matrix

        :param scatter_model: model generated with scatter
        :param rose_model: rose coupled train track model
        """

        self.ndof_scatter = scatter_model.number_eq
        self.contact_nodes = scatter_model.eq_nb_dof_rose_nodes
        self.rose_model = rose_model
        self.ndof_rose_scatter = self.ndof_scatter + rose_model.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)
        self.ndof_track = self.ndof_scatter  + rose_model.track.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)


        # initialise force vector
        self.force_vector = np.zeros(self.ndof_rose_scatter)

        # get rose global force vector
        rose_F = rose_model.global_force_vector
        self.rose_mask = np.ones(rose_F.shape[0], bool)

        # mask rose degrees of freedom which correspond to scatter model
        self.rose_mask[np.array(scatter_model.rose_eq_nb)] = False
        masked_rose = rose_F[self.rose_mask]

        # add masked rose force matrix to global force matrix
        self.force_vector[scatter_model.number_eq:] = masked_rose

        # return global force matrix to rose model
        self.rose_model.global_force_vector = self.force_vector
        self.rose_model.track.global_force_vector = self.force_vector[:self.ndof_track]

        RoseUtils.set_rose_loading(scatter_model, self.rose_model, solver)

    def update_pulse_load(self, t):
        """

        :param time: time index
        """

        self.force_vector = np.zeros(self.nb_equations)

        if t < self.steps-1:
            # for each node with load
            for n in self.contact_nodes:
                # index of node
                idx = list(self.model_nodes[:, 0].astype(int)).index(n)
                # for dof which have equation number
                for i, eq in enumerate(self.eq_nb_dof[idx]):
                    if ~np.isnan(eq):
                        self.force_vector[int(eq)] = float(self.factor[i]) * self.step_factors[t]

    def update_heaviside_load(self, t):

        self.force_vector = np.zeros(self.nb_equations)

        # for each node with load
        for n in self.contact_nodes:
            # index of node
            idx = list(self.model_nodes[:, 0].astype(int)).index(n)
            # for dof which have equation number
            for i, eq in enumerate(self.eq_nb_dof[idx]):
                if ~np.isnan(eq):
                    self.force_vector[int(eq)] = float(self.factor[i]) * self.step_factors[t]

    def update_moving_load_at_plane(self,t):
        self.force_vector = np.zeros(self.nb_equations)

        # get x,z coords of active elements at time t
        xz_coords_active_el = self.model_nodes[self.active_elements[t] - 1][:, [1, 3]]

        # calculate distance of point load to the nodes of the active element
        distances_to_nodes = xz_coords_active_el - self.position[:, t]
        x_dist = abs(distances_to_nodes[:, 0])
        z_dist = abs(distances_to_nodes[:, 1])

        # determine interpolation weights in x and z direction separately
        if any(x_dist < 1e-10):
            x_weights = (x_dist < 1e-10) * 1
        else:
            x_weights = 1 / x_dist

        if any(z_dist < 1e-10):
            z_weights = (z_dist < 1e-10) * 1
        else:
            z_weights = 1 / z_dist

        # calculate combined interpolation weights
        weights = x_weights * z_weights
        weights /= weights.sum()

        # calculate force on nodes of the active element

        point_load = np.array(self.factor) * self.step_factors[t]

        nodal_force = weights[:, None].dot(point_load[:][None, :])

        # get active degrees of freedom of the active element
        active_dof_el = self.eq_nb_dof[self.active_elements[t] - 1]

        # add force at active element at time t to global force matrix
        valid_dofs = ~np.isnan(active_dof_el)
        self.force_vector[active_dof_el[valid_dofs].astype(int)] = nodal_force[valid_dofs]

    def update_moving_load(self, t):
        self.force_vector = np.zeros(self.nb_equations)

        # if the load as reached the end of the model returns
        if self.load_distances[t] >= np.max(self.node_distances):
            return

        # find location of the load
        id_n = np.where(self.node_distances <= self.load_distances[t])[0][-1]
        nodes = [int(self.model_nodes[self.idx_list[id_n], 0]), int(self.model_nodes[self.idx_list[id_n + 1], 0])]

        # compute local shape functions
        x = self.load_distances[t] - self.model_nodes[self.idx_list[id_n], 3] + self.model_nodes[self.idx_list[0], 3]
        l = self.node_distances[id_n + 1] - self.node_distances[id_n]

        shp = np.array([1 - x / l,
                        x * l])

        # for each node with load
        for j, n in enumerate(nodes):
            # index of node
            idx = list(self.model_nodes[:, 0].astype(int)).index(n)
            # for dof which have equation number
            for i, eq in enumerate(self.eq_nb_dof[idx]):
                if ~np.isnan(eq):
                    self.force_vector[int(eq)] = float(self.factor[i]) * shp[j] * self.step_factors[t]

    def update_rose_load(self, t):
        self.force_vector = np.zeros(self.ndof_rose_scatter)
        self.force_vector = self.rose_model.update_time_step_rhs(t)

    def add_rose_load(self, scatter_model, rose_model):
        """
        Adds rose load to global force matrix

        :param scatter_model: model generated with scatter
        :param rose_model: rose coupled train track model
        """

        # initialise force matrix
        ndof = scatter_model.number_eq + rose_model.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)
        self.force = lil_matrix((ndof, len(rose_model.time)))

        # get rose global force vector
        rose_F = rose_model.global_force_vector
        mask = np.ones(rose_F.shape[0], bool)

        # mask rose degrees of freedom which correspond to scatter model
        mask[np.array(scatter_model.rose_eq_nb)] = False
        masked_rose = rose_F.toarray()[mask,:]

        # add masked rose force matrix to global force matrix
        self.force[scatter_model.number_eq:, :] = masked_rose

        # return global force matrix to rose model
        rose_model.global_force_vector = self.force
        rose_model.track.global_force_vector = self.force[:scatter_model.number_eq + rose_model.track.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes), :]


