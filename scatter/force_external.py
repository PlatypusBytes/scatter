"""External loads: pulse, heaviside, moving loads and the rose train-track model."""
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from scatter.mesher import ReadMesh
from scatter.rose_utils import RoseUtils


class Force:
    """External force vector at each time step."""

    def __init__(self) -> None:
        """Initialise an empty load."""
        self.force_vector: npt.NDArray[np.float64] = np.empty(0)
        self.factor: Sequence[float] = []  # force per direction [N]
        self.contact_nodes: Union[Sequence[int], npt.NDArray[np.int64]] = []  # node numbers where the force acts
        self.step_factors: npt.NDArray[np.float64] = np.empty(0)  # load factor per time step [-]

        self.nb_equations = 0
        self.eq_nb_dof: npt.NDArray[np.float64] = np.empty((0, 0))
        self.model_nodes: npt.NDArray[np.float64] = np.empty((0, 4))
        self.time: npt.NDArray[np.float64] = np.empty(0)
        self.steps = 0
        self.loading_type = ""
        self.solver: Any = None

        # moving load along z
        self.idx_list: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self.node_distances: npt.NDArray[np.float64] = np.empty(0)
        self.load_distances: npt.NDArray[np.float64] = np.empty(0)

        # moving load on a plane
        self.position: npt.NDArray[np.float64] = np.empty((2, 0))
        self.active_elements: List[Optional[npt.NDArray[np.int64]]] = []

        # rose load
        self.rose_model: Any = None
        self.ndof_scatter = 0
        self.ndof_rose_scatter = 0
        self.ndof_track = 0
        self.rose_mask: npt.NDArray[np.bool_] = np.empty(0, dtype=bool)
        self.force: Any = None

    def initialise_load(self, load_set: Dict[str, Any], time: npt.NDArray[np.float64], model: ReadMesh, solver: Any,
                        **kwargs: Any) -> None:
        """Initialise the load.

        Args:
            load_set: loading settings.
            time: time of each step [s].
            model: scatter model.
            solver: numerical solver.
            **kwargs: "top_surface_elements", the node numbers of the top surface elements, for a moving load on a
                plane (only for hexa8 elements).

        Raises:
            SystemExit: if the load type is not supported.
        """
        self.nb_equations = model.number_eq
        self.factor = load_set.get("force", [])
        self.contact_nodes = load_set.get("node", [])
        self.eq_nb_dof = model.eq_nb_dof
        self.model_nodes = model.nodes
        self.time = time
        # number of steps to apply the load following a triangular form
        self.steps = load_set["ini_steps"]
        self.loading_type = load_set["type"]
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

    def update_load_at_t(self, t: int, **kwargs: Any) -> npt.NDArray[np.float64]:
        """Update the load at a time step.

        Args:
            t: time index.
            **kwargs: keyword arguments passed by the numerical solver, not used.

        Returns:
            Force vector at the time step [N].
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

    def initialise_pulse_load(self) -> None:
        """Initialise a triangular pulse load over `steps` time steps.

        Raises:
            SystemExit: if the number of time steps is not larger than the number of loading steps.
        """
        if len(self.time) <= self.steps:
            sys.exit("Error: Number of loading steps smaller than " + str(self.steps))

        self.step_factors = np.append(np.linspace(0, 1, int((self.steps - 1) / 2), endpoint=False),
                                      np.linspace(1, 0, int((self.steps + 1) / 2)))

        self.update_pulse_load(0)

    def initialise_heaviside_load(self) -> None:
        """Initialise a heaviside load that ramps up linearly over `steps` time steps.

        Raises:
            SystemExit: if the number of time steps is not larger than the number of loading steps.
        """
        if len(self.time) <= self.steps:
            sys.exit("Error: Number of loading steps smaller than " + str(self.steps))

        self.step_factors = np.ones(len(self.time))
        self.step_factors[:self.steps] = np.linspace(0, 1, self.steps)

        self.update_heaviside_load(0)

    def initialise_moving_load(self, load_speed: float) -> None:
        """Initialise a point load that moves along the z-axis, through the nodes with the x and y of the load node.

        Args:
            load_speed: speed of the load [m/s].

        Raises:
            SystemExit: if the number of time steps is not larger than the number of loading steps.
        """
        if len(self.time) <= self.steps:
            sys.exit("Error: Number of loading steps smaller than " + str(self.steps))

        self.step_factors = np.ones(len(self.time))
        self.step_factors[:self.steps] = np.linspace(0, 1, self.steps)

        # nodes with the same x and y as the load node, sorted by their z-distance to it
        idx = np.where(self.model_nodes[:, 0] == self.contact_nodes)[0][0]
        self.idx_list = np.where((self.model_nodes[:, 1] == self.model_nodes[idx, 1])
                                 & (self.model_nodes[:, 2] == self.model_nodes[idx, 2]))[0]
        dist = np.abs(self.model_nodes[self.idx_list, 3] - self.model_nodes[idx, 3])
        self.idx_list = self.idx_list[np.argsort(dist)]
        self.node_distances = np.sort(dist)

        # the load starts moving after the initialisation steps
        speed_array = np.ones(len(self.time)) * load_speed
        speed_array[:self.steps] = 0
        self.load_distances = speed_array * (self.time - self.time[self.steps])

        self.update_moving_load(0)

    def initialise_moving_load_at_plane(self, xz_plane_elements: npt.NDArray[np.int64], load_speed: float,
                                        load_direction: Sequence[float], start_coord: Sequence[float]) -> None:
        """Initialise a point load that moves over the top surface, in the xz plane.

        Args:
            xz_plane_elements: node numbers of each top surface element.
            load_speed: speed of the load [m/s].
            load_direction: x and z components of the direction of the load.
            start_coord: x and z coordinates of the start of the load [m].
        """
        # x and z coordinates of each element, as polygons; the convex hull sorts the coordinates
        elem_coordinates = self.model_nodes[xz_plane_elements - 1, 1:]
        polygons = np.array([Polygon(elem).convex_hull for elem in elem_coordinates[:, :, [0, 2]]])

        self.step_factors = np.ones(len(self.time))
        self.step_factors[:self.steps] = np.linspace(0, 1, self.steps)

        # travelled distance of the point load at each time step
        dt = np.diff(self.time[self.steps:])
        distance = np.zeros(len(self.time))
        distance[self.steps:] = np.append(0, np.cumsum(load_speed * dt))

        # angle of the direction of the moving point load
        if np.isclose(load_direction[0], 0):
            angle_direction = 0.5 * np.pi if load_direction[1] > 0 else -0.5 * np.pi
        else:
            angle_direction = np.arctan(load_direction[1] / load_direction[0])

        # position of the point load at each time step
        x_coordinates = np.cos(angle_direction) * distance + start_coord[0]
        z_coordinates = np.sin(angle_direction) * distance + start_coord[1]
        self.position = np.array([x_coordinates, z_coordinates])

        # kd-tree of the centroids of all surface polygons, to speed up the search of active elements
        centroids = np.array([np.array(polygon.centroid.xy)[:, 0] for polygon in polygons])
        tree = KDTree(centroids)

        # todo, make nbr_nearest_neightbours more general
        nbr_nearest_neighbours = 10

        # element that contains the moving load at each time step, None if there is none
        self.active_elements = []
        for coord in self.position.T:
            nearest_centroids_indices = tree.query(coord, nbr_nearest_neighbours)[1]
            near_polygons = polygons[nearest_centroids_indices]

            active_element = None
            for idx, polygon in enumerate(near_polygons):
                if polygon.contains(Point(coord)):
                    active_element = xz_plane_elements[nearest_centroids_indices[idx]]
                    break
            self.active_elements.append(active_element)

        self.update_moving_load_at_plane(0)

    def initialise_rose_load(self, scatter_model: ReadMesh, rose_model: Any, solver: Any) -> None:
        """Add the rose load to the global force vector and set up the coupled rose model.

        Args:
            scatter_model: scatter model.
            rose_model: rose coupled train track model.
            solver: numerical solver.
        """
        self.ndof_scatter = scatter_model.number_eq
        self.contact_nodes = scatter_model.eq_nb_dof_rose_nodes
        self.rose_model = rose_model
        self.ndof_rose_scatter = self.ndof_scatter + rose_model.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)
        self.ndof_track = self.ndof_scatter + rose_model.track.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)

        self.force_vector = np.zeros(self.ndof_rose_scatter)

        # rose dofs that coincide with scatter dofs are not added again
        rose_F = rose_model.global_force_vector
        self.rose_mask = np.ones(rose_F.shape[0], bool)
        self.rose_mask[np.array(scatter_model.rose_eq_nb)] = False
        self.force_vector[scatter_model.number_eq:] = rose_F[self.rose_mask]

        # return global force vector to rose model
        self.rose_model.global_force_vector = self.force_vector
        self.rose_model.track.global_force_vector = self.force_vector[:self.ndof_track]

        RoseUtils.set_rose_loading(scatter_model, self.rose_model, solver)

    def __set_nodal_force(self, node: int, shape_factor: float, step_factor: float = 1.) -> None:
        """Set the force on every free dof of a node.

        Args:
            node: node number.
            shape_factor: share of the load on the node [-].
            step_factor: load factor of the time step [-].
        """
        idx = list(self.model_nodes[:, 0].astype(int)).index(node)
        for i, eq in enumerate(self.eq_nb_dof[idx]):
            if not np.isnan(eq):
                self.force_vector[int(eq)] = float(self.factor[i]) * shape_factor * step_factor

    def update_pulse_load(self, t: int) -> None:
        """Update the pulse load at a time step.

        Args:
            t: time index.
        """
        self.force_vector = np.zeros(self.nb_equations)
        if t < self.steps - 1:
            for n in self.contact_nodes:
                self.__set_nodal_force(int(n), self.step_factors[t])

    def update_heaviside_load(self, t: int) -> None:
        """Update the heaviside load at a time step.

        Args:
            t: time index.
        """
        self.force_vector = np.zeros(self.nb_equations)
        for n in self.contact_nodes:
            self.__set_nodal_force(int(n), self.step_factors[t])

    def update_moving_load_at_plane(self, t: int) -> None:
        """Update the moving load on the top surface at a time step, distributed over the nodes of its element.

        Args:
            t: time index.
        """
        self.force_vector = np.zeros(self.nb_equations)
        active_element = self.active_elements[t]
        if active_element is None:
            # the load is outside the top surface
            return

        # distance of the point load to the nodes of the active element
        xz_coords_active_el = self.model_nodes[active_element - 1][:, [1, 3]]
        distances_to_nodes = xz_coords_active_el - self.position[:, t]
        x_dist = abs(distances_to_nodes[:, 0])
        z_dist = abs(distances_to_nodes[:, 1])

        # inverse distance interpolation weights in x and z direction separately
        if any(x_dist < 1e-10):
            x_weights = (x_dist < 1e-10) * 1.
        else:
            x_weights = 1 / x_dist
        if any(z_dist < 1e-10):
            z_weights = (z_dist < 1e-10) * 1.
        else:
            z_weights = 1 / z_dist
        weights = x_weights * z_weights
        weights /= weights.sum()

        # force on the nodes of the active element
        point_load = np.array(self.factor) * self.step_factors[t]
        nodal_force = weights[:, None].dot(point_load[None, :])

        active_dof_el = self.eq_nb_dof[active_element - 1]
        valid_dofs = ~np.isnan(active_dof_el)
        self.force_vector[active_dof_el[valid_dofs].astype(int)] = nodal_force[valid_dofs]

    def update_moving_load(self, t: int) -> None:
        """Update the load moving along the z-axis at a time step, distributed over the two nearest nodes.

        Args:
            t: time index.
        """
        self.force_vector = np.zeros(self.nb_equations)

        # the load has reached the end of the model
        if self.load_distances[t] >= np.max(self.node_distances):
            return

        # nodes before and after the load
        id_n = np.where(self.node_distances <= self.load_distances[t])[0][-1]
        nodes = [int(self.model_nodes[self.idx_list[id_n], 0]), int(self.model_nodes[self.idx_list[id_n + 1], 0])]

        # linear shape functions between the two nodes
        x = self.load_distances[t] - self.model_nodes[self.idx_list[id_n], 3] + self.model_nodes[self.idx_list[0], 3]
        length = self.node_distances[id_n + 1] - self.node_distances[id_n]
        shp = np.array([1 - x / length, x / length])

        for j, n in enumerate(nodes):
            self.__set_nodal_force(n, shp[j], self.step_factors[t])

    def update_rose_load(self, t: int) -> None:
        """Update the rose load at a time step.

        Args:
            t: time index.
        """
        self.force_vector = self.rose_model.update_time_step_rhs(t)

    def add_rose_load(self, scatter_model: ReadMesh, rose_model: Any) -> None:
        """Add the rose load to a global force matrix over all time steps.

        Args:
            scatter_model: scatter model.
            rose_model: rose coupled train track model.
        """
        ndof = scatter_model.number_eq + rose_model.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)
        self.force = lil_matrix((ndof, len(rose_model.time)))

        # rose dofs that coincide with scatter dofs are not added again
        rose_F = rose_model.global_force_vector
        mask = np.ones(rose_F.shape[0], bool)
        mask[np.array(scatter_model.rose_eq_nb)] = False
        self.force[scatter_model.number_eq:, :] = rose_F.toarray()[mask, :]

        # return global force matrix to rose model
        nb_track = scatter_model.number_eq + rose_model.track.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)
        rose_model.global_force_vector = self.force
        rose_model.track.global_force_vector = self.force[:nb_track, :]
