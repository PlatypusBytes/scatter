import sys
import numpy as np
import scipy.spatial.kdtree
from scipy.sparse import lil_matrix

from shapely.geometry.polygon import Polygon
from shapely.geometry import Point


class Force:
    def __init__(self):
        self.force = []
        return

    def pulse_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, steps=5):
        """
        Pulse load on nodes

        :param nb_equations: total number of equations
        :param eq_nb_dof: number of equation for each dof in node list
        :param load_set: loading settings
        :param node: list of nodes where load is applied
        :param time_step: time step
        :param steps: (optional: default = 5) number of steps to apply the load following a triangular form
        """

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
        """
        Heaviside load on nodes

        :param nb_equations: total number of equations
        :param eq_nb_dof: number of equation for each dof in node list
        :param load_set: loading settings
        :param node: list of nodes where load is applied
        :param time_step: time step
        :param steps: (optional: default = 5) number of steps to initialise load
        """
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

    def moving_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, nodes_coord, steps=50):
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
        time = load_set["time"]
        time = np.linspace(0, time, int(np.ceil(time / time_step)))
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
            dist.append(np.sqrt((nodes_coord[i, 3] - nodes_coord[idx, 3])**2))

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

    def moving_load_at_plane(self, nb_equations, eq_nb_dof, load_set, coordinate, time_step, xz_plane_elements,
                             nodes_coord, steps=50):
        """
        Moving load at the xz plane.

        #todo load is distributed on element nodes based on inverse distance interpolation, check if this is correct for each case

        :param nb_equations: total number of equations
        :param eq_nb_dof: number of equation for each dof in node list
        :param load_set: loading settings
        :param coordinate: xz coordinate where the load starts
        :param time_step: time step
        :param xz_plane_elements: elements at xz_plane_elements
        :param nodes_coord: all nodal coordinates
        :param steps: (optional: default = 50) number of steps to initialise load
        """

        # time
        time = load_set["time"]
        time = np.linspace(0, time, int(np.ceil(time / time_step)))

        # initialise force matrix
        self.force = lil_matrix(np.zeros((nb_equations, len(time))))
        # load factor
        factor = load_set["force"]

        # get coordinates of each element on the xz plane
        elem_coordinates = nodes_coord[xz_plane_elements - 1, 1:]

        # add elem x,z coordinates to shapely polygons, get convex hull such that coordinate sorting is correct
        polygons = np.array([Polygon(elem).convex_hull for elem in elem_coordinates[:,:,[0,2]]])

        # set loading factor
        fct = np.ones(len(time))
        fct[:steps] = np.linspace(0, 1, steps)

        # calculate point load at each time step
        point_load = np.array([fct*load for load in factor])

        # calculate traveled distance of point load at each time step
        dt = np.diff(time[steps:])
        distance = np.zeros(len(time))
        distance[steps:] = np.append(0,np.cumsum(load_set["speed"] * dt))

        # calculate angle of the direction of the moving point load
        if np.isclose(load_set["direction"][0],0):
            if load_set["direction"][1] > 0:
                angle_direction = 0.5*np.pi
            else:
                angle_direction = -0.5*np.pi
        else:
            angle_direction = np.arctan(load_set["direction"][1]/load_set["direction"][0])

        # calculate position of the point load at each time step
        x_coordinates = np.cos(angle_direction) * distance + coordinate[0]
        z_coordinates = np.sin(angle_direction) * distance + coordinate[1]
        position = np.array([x_coordinates,z_coordinates])

        # define kdtree of centroids of all surface polygons, to speed up search of active elements
        centroids = np.array([np.array(polygon.centroid.xy)[:,0] for polygon in polygons])
        tree = scipy.spatial.kdtree.KDTree(centroids)

        # todo, make nbr_nearest_neightbours more general
        nbr_nearest_neighbours = 10

        # find elements which are in contact with the moving load at each time step
        active_elements = []
        for coord in position.T:

            # get nearest centroids of polygons to speed up search
            nearest_centroids_indices = tree.query(coord, nbr_nearest_neighbours)[1]
            near_polygons = polygons[nearest_centroids_indices]

            # find elements which contain the point load, only elements with near polygons are checked
            found_element = False
            for idx, polygon in enumerate(near_polygons):
                if polygon.contains(Point(coord)):
                    element_idx = nearest_centroids_indices[idx]
                    active_elements.append(xz_plane_elements[element_idx])
                    found_element = True
                    break

            # if no polygon is found, add none to active elements
            if not found_element:
                active_elements.append(None)

        # loop over each time step
        for t in range(len(time)):

            # get x,z coords of active elements at time t
            xz_coords_active_el = nodes_coord[active_elements[t]-1][:, [1, 3]]

            # calculate distance of point load to the nodes of the active element
            distances_to_nodes = xz_coords_active_el - position[:,t]
            x_dist = distances_to_nodes[:,0]
            z_dist = distances_to_nodes[:,1]

            # determine interpolation weights in x and z direction separately
            if any(x_dist < 1e-10):
                x_weights = (x_dist < 1e-10) * 1
            else:
                x_weights = 1 / abs(x_dist)

            if any(z_dist < 1e-10):
                z_weights = (z_dist < 1e-10) * 1
            else:
                z_weights = 1 / abs(z_dist)

            # calculate combined interpolation weights
            weights = x_weights * z_weights
            weights /= weights.sum()

            # calculate force on nodes of the active element
            nodal_force = weights[:,None].dot(point_load[:,t][None,:])

            # get active degrees of freedom of the active element
            active_dof_el = eq_nb_dof[active_elements[t]-1]

            # add force at active element at time t to global force matrix
            valid_dofs = ~np.isnan(active_dof_el)
            self.force[active_dof_el[valid_dofs].astype(int), t] = nodal_force[valid_dofs]

