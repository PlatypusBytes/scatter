"""Coupling between scatter and the rose train-track model."""
import itertools
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
from rose.model.geometry import Element, Mesh, Node
from rose.model.global_system import GlobalSystem
from rose.model.model_part import ConstraintModelPart, Material, Section
from rose.model.track import Rail, RailPad, Sleeper
from rose.model.train_track_interaction import CoupledTrainTrack
from rose.pre_process.mesh_utils import combine_horizontal_tracks
from solvers.base_solver import BaseSolverABC

if TYPE_CHECKING:
    # mesher imports this module, so ReadMesh is only imported for type checking
    from scatter.mesher import ReadMesh


class RoseUtils:
    """Helpers to set up rose and connect it to scatter."""

    @staticmethod
    def pre_process_rose_model(rose_model: CoupledTrainTrack) -> CoupledTrainTrack:
        """Validate and initialise the rose model, and combine the train and track matrices.

        Args:
            rose_model: rose coupled train track model.

        Returns:
            The pre-processed rose model.
        """
        rose_model.validate_input()
        rose_model.initialise()
        rose_model.combine_global_matrices()

        return rose_model

    @staticmethod
    def set_rose_loading(scatter_model: "ReadMesh", rose_model: CoupledTrainTrack, solver: BaseSolverABC) -> None:
        """Set the rose loading.

        Renumbers the dofs of the coupled system, initialises the solver, and calculates the initial static
        displacement of the track and the train and the initial contact deformation.

        Args:
            scatter_model: scatter model.
            rose_model: rose coupled train track model.
            solver: numerical solver.
        """
        # recalculate number of degrees of freedom with rose and scatter model combined
        RoseUtils.recalculate_ndof(scatter_model, rose_model)

        # reinitialise solver with correct ndof
        solver.initialise(scatter_model.number_eq, solver.state.time)

        # initial displacement of the track system and the train
        rose_model.calculate_initial_displacement_track()
        disp_at_wheels = rose_model.get_disp_track_at_wheels(0, rose_model.track.solver.u[0, :])
        rose_model.calculate_initial_displacement_train(disp_at_wheels)

        # calculate initial Hertzian contact deformation
        rose_model.calculate_static_contact_deformation()

        solver.force.update_rhs_at_non_linear_iteration_func = rose_model.update_non_linear_iteration

        # add track displacement and velocity to global system
        solver.u[0, :rose_model.track.total_n_dof] = rose_model.track.solver.u[0, :]
        solver.v[0, :rose_model.track.total_n_dof] = rose_model.track.solver.v[0, :]

        # add train displacement and velocity to global system
        solver.u[0, rose_model.track.total_n_dof:] = rose_model.train.solver.u[0, :]
        solver.v[0, rose_model.track.total_n_dof:] = rose_model.train.solver.v[0, :]

    @staticmethod
    def create_horizontal_rose_track(n_sleepers: int, sleeper_distance: float) -> Tuple[Dict[str, Any], Mesh]:
        """Create the mesh of a horizontal track, with the top of the track at y = 0 and sleepers of 1 m thick.

        Args:
            n_sleepers: number of sleepers [-].
            sleeper_distance: distance between sleepers [m].

        Returns:
            The rail, rail pad and sleeper model parts, and the mesh.
        """
        track_level = 0.0
        sleeper_thickness = 1.0
        n_rail_per_sleeper = 1

        mesh = Mesh()

        # add track nodes and elements to mesh
        nodes_track = [
            Node(i * sleeper_distance / n_rail_per_sleeper, track_level, 0.0)
            for i in range(n_sleepers * n_rail_per_sleeper)
        ]
        mesh.add_unique_nodes_to_mesh(nodes_track)
        elements_track = [
            Element([nodes_track[i], nodes_track[i + 1]]) for i in range(n_sleepers * n_rail_per_sleeper - 1)
        ]
        mesh.add_unique_elements_to_mesh(elements_track)

        # add railpad nodes and elements to mesh
        points_rail_pad_per_sleeper = [[
            nodes_track[i * n_rail_per_sleeper],
            Node(i * sleeper_distance, -sleeper_thickness, 0.0)
        ] for i in range(n_sleepers)]
        points_rail_pad = list(itertools.chain.from_iterable(points_rail_pad_per_sleeper))
        mesh.add_unique_nodes_to_mesh(points_rail_pad)

        elements_rail_pad = [Element([points_rail_pad[i * 2], points_rail_pad[i * 2 + 1]]) for i in range(n_sleepers)]
        mesh.add_unique_elements_to_mesh(elements_rail_pad)

        # add sleeper nodes to mesh
        nodes_sleeper = [points_rail_pad[i * 2 + 1] for i in range(n_sleepers)]
        mesh.add_unique_nodes_to_mesh(nodes_sleeper)

        # reorder node and element indices
        mesh.reorder_element_ids()
        mesh.reorder_node_ids()

        rail_model_part = Rail()
        rail_model_part.elements = elements_track
        rail_model_part.nodes = nodes_track
        rail_model_part.length_rail = sleeper_distance

        rail_pad_model_part = RailPad()
        rail_pad_model_part.elements = elements_rail_pad
        rail_pad_model_part.nodes = points_rail_pad

        sleeper_model_part = Sleeper()
        sleeper_model_part.nodes = nodes_sleeper

        model_parts = {
            "rail": rail_model_part,
            "rail_pad": rail_pad_model_part,
            "sleeper": sleeper_model_part,
            "soil": []
        }
        return model_parts, mesh

    @staticmethod
    def assign_data_to_coupled_model(rose_data: Dict[str, Any],
                                     solver: BaseSolverABC,
                                     seed: int = 14) -> CoupledTrainTrack:
        """Assign rose data from a dictionary to a rose coupled model.

        Args:
            rose_data: rose traffic data, track info and time integration data.
            solver: solver of the coupled train track model.
            seed: seed of the irregularities.

        Returns:
            Coupled train track model.
        """
        all_element_model_parts = []
        all_meshes = []
        for idx in range(rose_data["track_info"]["geometry"]["n_segments"]):
            # set geometry of one segment
            element_model_parts, mesh = RoseUtils.create_horizontal_rose_track(
                rose_data["track_info"]["geometry"]["n_sleepers"][idx],
                rose_data["track_info"]["geometry"]["sleeper_distance"])
            all_element_model_parts.append(element_model_parts)
            all_meshes.append(mesh)

        # Setup global mesh and combine model parts of all segments
        rail_model_part, sleeper_model_part, rail_pad_model_part, _, all_mesh = \
            combine_horizontal_tracks(all_element_model_parts, all_meshes)

        # time steps of the initialisation and the calculation
        time_integration = rose_data["time_integration"]
        initialisation_time = np.linspace(0, time_integration["tot_ini_time"], time_integration["n_t_ini"] + 1)
        calculation_time = np.linspace(initialisation_time[-1],
                                       initialisation_time[-1] + time_integration["tot_calc_time"],
                                       time_integration["n_t_calc"] + 1)
        time = np.concatenate((initialisation_time, calculation_time[1:]))

        # set elements
        track_materials = rose_data["track_info"]["materials"]
        material = Material()
        material.youngs_modulus = track_materials["young_mod_beam"]
        material.poisson_ratio = track_materials["poisson_beam"]
        material.density = track_materials["rho"]

        section = Section()
        section.area = track_materials["rail_area"]
        section.sec_moment_of_inertia = track_materials["inertia_beam"]
        section.shear_factor = track_materials["shear_factor_rail"]

        rail_model_part.section = section
        rail_model_part.material = material

        rail_pad_model_part.mass = track_materials["mass_rail_pad"]
        rail_pad_model_part.stiffness = track_materials["stiffness_rail_pad"]
        rail_pad_model_part.damping = track_materials["damping_rail_pad"]

        sleeper_model_part.mass = track_materials["mass_sleeper"]

        # the train does not move in the initialisation phase
        velocities = np.ones(len(time)) * rose_data["traffic_data"]["velocity"]
        velocities[0:len(initialisation_time)] = 0

        # constraint rotation at the side boundaries
        side_boundaries = ConstraintModelPart(x_disp_dof=False, y_disp_dof=True, z_rot_dof=True)
        side_boundaries.nodes = [rail_model_part.nodes[0], rail_model_part.nodes[-1]]

        # populate global system
        track = GlobalSystem()
        track.mesh = all_mesh
        track.time = time
        track.model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, side_boundaries]

        # set up train
        train = rose_data["traffic_data"]["model"]
        train.time = time
        train.velocities = velocities
        train.use_irregularities = True
        train.irregularity_parameters = {"Av": 0.00002095, "seed": seed}

        # setup coupled train track system
        coupled_model = CoupledTrainTrack()
        coupled_model.train = train
        coupled_model.track = track
        coupled_model.rail = rail_model_part
        coupled_model.time = time
        coupled_model.initialisation_time = initialisation_time
        coupled_model.hertzian_contact_coef = track_materials["hertzian_contact_coef"]
        coupled_model.hertzian_power = track_materials["hertzian_power"]
        coupled_model.solver = solver
        # rayleigh damping is added via scatter
        coupled_model.is_rayleigh_damping = False

        return coupled_model

    @staticmethod
    def get_bottom_boundary(rose_model: CoupledTrainTrack) -> List[int]:
        """Get the vertical dof numbers of the bottom nodes of the coupled train track model.

        The bottom nodes are the nodes with the minimum y-coordinate.

        Args:
            rose_model: coupled train track model.

        Returns:
            Vertical dof numbers of the bottom nodes.
        """
        min_y_coord = min(node.coordinates[1] for node in rose_model.track.mesh.nodes)
        bottom_nodes = [node for node in rose_model.track.mesh.nodes if np.isclose(node.coordinates[1], min_y_coord)]
        return [node.index_dof[1] for node in bottom_nodes]

    @staticmethod
    def recalculate_ndof(scatter_model: "ReadMesh", rose_model: CoupledTrainTrack) -> None:
        """Renumber the dofs of the scatter and rose models into one coupled system.

        Args:
            scatter_model: scatter model; its number of equations becomes the one of the coupled system.
            rose_model: rose coupled train track model.
        """
        nb_contact = len(scatter_model.eq_nb_dof_rose_nodes)

        # recalculate rose track numbers of degree of freedom, this includes scatter + rose track
        rose_model.track.total_n_dof = scatter_model.number_eq + rose_model.track.total_n_dof - nb_contact
        rose_model.total_n_dof = scatter_model.number_eq + rose_model.total_n_dof - nb_contact

        # this works because rose contact nodes are placed after the rail nodes in the mesh
        rose_model.track_global_indices = rose_model.track_global_indices + scatter_model.number_eq

        # recalculate rose train numbers of degree of freedom
        rose_model.train.contact_dofs = list(
            np.array(rose_model.train.contact_dofs) + scatter_model.number_eq - nb_contact)

        # reinitialise rose solvers with new number of degree of freedom
        rose_model.track.solver.initialise(rose_model.track.total_n_dof, [0, 1])

        # recalculate indices of degree of freedom on rose track nodes
        contact_nodes_found = 0
        for node in rose_model.track.mesh.nodes:
            for i, dof in enumerate(node.index_dof):
                if dof is None:
                    continue
                if dof in scatter_model.rose_eq_nb:
                    node.index_dof[i] = scatter_model.eq_nb_dof_rose_nodes[scatter_model.rose_eq_nb.index(dof)]
                    contact_nodes_found += 1
                else:
                    # this works because rose contact nodes are placed after the rail nodes in the mesh
                    node.index_dof[i] = dof + scatter_model.number_eq - contact_nodes_found
                # todo connect rose-scatter nodes, currently visualisation of the results on the rose nodes will not
                # work properly on the bottom elements of the rose model

        # recalculate indices of degree of freedom on rose train nodes
        for node in rose_model.train.mesh.nodes:
            for i, dof in enumerate(node.index_dof):
                if dof is not None:
                    node.index_dof[i] = dof + scatter_model.number_eq - contact_nodes_found

        # add dof indices to model part
        for model_part in rose_model.track.model_parts:
            model_part.update_dof_indices()

        scatter_model.number_eq = rose_model.total_n_dof
