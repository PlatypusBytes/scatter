from rose.model.model_part import Material, Section
from rose.model.train_track_interaction import *
from solvers.newmark_solver import NewmarkSolver
import numpy as np


class RoseUtils:
    @staticmethod
    def pre_process_rose_model(rose_model: CoupledTrainTrack):
        """
        Pre processes rose model, i.e., validate, initialise and combine train an track matrices
        """

        # pre process rose
        rose_model.validate_input()
        rose_model.initialise()
        rose_model.combine_global_matrices()

        return rose_model

    @staticmethod
    def set_rose_loading(scatter_model, rose_model: CoupledTrainTrack, solver):
        """
        Sets rose loading. Initialises solver; calculates initial static displacement due to train load; calculates
        initial contact deformation

        :param scatter_model: scatter model object
        :param rose_model: rose model object
        :param solver: solver
        """

        # recalculate number of degrees of freedom with rose and scatter model combined
        RoseUtils.recalculate_ndof(scatter_model, rose_model)

        # reinitialise solver with correct ndof
        solver.initialise(scatter_model.number_eq, solver.time)

        # calculate initial displacement of the track system
        rose_model.calculate_initial_displacement_track()

        # calculate initial displacement of the train
        disp_at_wheels = rose_model.get_disp_track_at_wheels(0, rose_model.track.solver.u[0, :])
        rose_model.calculate_initial_displacement_train(disp_at_wheels)

        # calculate initial Hertzian contact deformation
        rose_model.calculate_static_contact_deformation()

        # self.solver.update_rhs_at_time_step_func = self.update_time_step_rhs
        solver.update_rhs_at_non_linear_iteration_func = rose_model.update_non_linear_iteration
        # solver.load_func = rose_model.update_non_linear_iteration

        # add track displacement and velocity to global system
        solver.u[0, :rose_model.track.total_n_dof] = rose_model.track.solver.u[0, :]
        solver.v[0, :rose_model.track.total_n_dof] = rose_model.track.solver.v[0, :]

        # add train displacement and velocity to global system
        solver.u[0, rose_model.track.total_n_dof:] = rose_model.train.solver.u[0, :]
        solver.v[0, rose_model.track.total_n_dof:] = rose_model.train.solver.v[0, :]

    @staticmethod
    def create_horizontal_rose_track(n_sleepers, sleeper_distance):
        """
        Creates mesh of an horizontal track. Where the top of the track lies at z=0; the sleeper thickness is 1.0m.

        :param n_sleepers: number of sleepers [-]
        :param sleeper_distance: distance between sleepers [m]
        :return: Dictionary with: rail model part, rail pad model part, sleeper model part. Mesh

        """
        # define constants
        track_level = 0.0
        sleeper_thickness = 1.0

        n_rail_per_sleeper = 1

        # initialise mesh
        mesh = Mesh()

        # add track nodes and elements to mesh
        nodes_track = [Node(i * sleeper_distance/n_rail_per_sleeper, track_level, 0.0) for i in range(n_sleepers * n_rail_per_sleeper)]
        mesh.add_unique_nodes_to_mesh(nodes_track)
        elements_track = [
            Element([nodes_track[i], nodes_track[i + 1]]) for i in range(n_sleepers * n_rail_per_sleeper - 1)
        ]
        mesh.add_unique_elements_to_mesh(elements_track)

        # add railpad nodes and elements to mesh
        points_rail_pad = [
            [nodes_track[i*n_rail_per_sleeper], Node(i * sleeper_distance, -sleeper_thickness, 0.0)]
            for i in range(n_sleepers)
        ]
        points_rail_pad = list(itertools.chain.from_iterable(points_rail_pad))
        mesh.add_unique_nodes_to_mesh(points_rail_pad)

        elements_rail_pad = [
            Element([points_rail_pad[i * 2], points_rail_pad[i * 2 + 1]])
            for i in range(n_sleepers)
        ]
        mesh.add_unique_elements_to_mesh(elements_rail_pad)

        # add sleeper nodes to mesh
        nodes_sleeper = [points_rail_pad[i * 2 + 1] for i in range(n_sleepers)]
        mesh.add_unique_nodes_to_mesh(nodes_sleeper)

        # reorder node and element indices
        mesh.reorder_element_ids()
        mesh.reorder_node_ids()

        # initialise rail model part
        rail_model_part = Rail()
        rail_model_part.elements = elements_track
        rail_model_part.nodes = nodes_track
        rail_model_part.length_rail = sleeper_distance

        # initialise railpad model part
        rail_pad_model_part = RailPad()
        rail_pad_model_part.elements = elements_rail_pad
        rail_pad_model_part.nodes = points_rail_pad

        # initialise sleeper model part
        sleeper_model_part = Sleeper()
        sleeper_model_part.nodes = nodes_sleeper

        # return dictionary of model parts present in horizontal track
        return (
            {
                "rail": rail_model_part,
                "rail_pad": rail_pad_model_part,
                "sleeper": sleeper_model_part,
                "soil": []
            },
            mesh,
        )

    @staticmethod
    def assign_data_to_coupled_model(rose_data: dict, seed=14) -> CoupledTrainTrack:
        """
        assigns rose data from dictionary to rose coupled model

        :param rose_data: Dictionary containing rose train_info, track_info, time_int and soil data
        :param seed: seed for irregularities (default = 14)
        :return: Coupled train track model
        """
        # choose solver
        solver = NewmarkSolver()

        all_element_model_parts = []
        all_meshes = []
        # loop over number of segments
        for idx in range(rose_data["track_info"]["geometry"]["n_segments"]):
            # set geometry of one segment
            element_model_parts, mesh = RoseUtils.create_horizontal_rose_track(
                rose_data["track_info"]["geometry"]["n_sleepers"][idx],
                rose_data["track_info"]["geometry"]["sleeper_distance"])

            # add segment model parts and mesh to list
            all_element_model_parts.append(element_model_parts)
            all_meshes.append(mesh)

        # Setup global mesh and combine model parts of all segments
        rail_model_part, sleeper_model_part, rail_pad_model_part, _, all_mesh = \
            combine_horizontal_tracks(all_element_model_parts, all_meshes, 0.6)

        # set initialisation time
        initialisation_time = np.linspace(0, rose_data["time_integration"]["tot_ini_time"], rose_data["time_integration"]["n_t_ini"]+1)
        # set calculation time
        calculation_time = np.linspace(initialisation_time[-1], initialisation_time[-1] + rose_data["time_integration"]["tot_calc_time"],
                                       rose_data["time_integration"]["n_t_calc"] + 1)
        # Combine all time steps in an array
        time = np.concatenate((initialisation_time, calculation_time[1:]))

        # set elements
        material = Material()
        material.youngs_modulus = rose_data["track_info"]["materials"]["young_mod_beam"]
        material.poisson_ratio = rose_data["track_info"]["materials"]["poisson_beam"]
        material.density = rose_data["track_info"]["materials"]["rho"]

        section = Section()
        section.area = rose_data["track_info"]["materials"]["rail_area"]
        section.sec_moment_of_inertia = rose_data["track_info"]["materials"]["inertia_beam"]
        section.shear_factor = rose_data["track_info"]["materials"]["shear_factor_rail"]

        rail_model_part.section = section
        rail_model_part.material = material

        rail_pad_model_part.mass = rose_data["track_info"]["materials"]["mass_rail_pad"]
        rail_pad_model_part.stiffness = rose_data["track_info"]["materials"]["stiffness_rail_pad"]
        rail_pad_model_part.damping = rose_data["track_info"]["materials"]["damping_rail_pad"]

        sleeper_model_part.mass = rose_data["track_info"]["materials"]["mass_sleeper"]

        # set velocity of train
        velocities = np.ones(len(time)) * rose_data["traffic_data"]["velocity"]

        # prevent train from moving in initialisation phase
        velocities[0:len(initialisation_time)] = 0

        # constraint rotation at the side boundaries
        side_boundaries = ConstraintModelPart(x_disp_dof=False, y_disp_dof=True, z_rot_dof=True)
        side_boundaries.nodes = [rail_model_part.nodes[0], rail_model_part.nodes[-1]]

        # populate global system
        track = GlobalSystem()
        track.mesh = all_mesh
        track.time = time

        # collect all model parts track
        model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, side_boundaries]

        track.model_parts = model_parts

        # set up train
        train = rose_data["traffic_data"]["model"]
        train.time = time
        train.velocities = velocities
        train.use_irregularities = True
        train.irregularity_parameters = {"Av": 0.00002095,
                                         "seed": seed}


        # setup coupled train track system
        coupled_model = CoupledTrainTrack()

        coupled_model.train = train
        coupled_model.track = track
        coupled_model.rail = rail_model_part
        coupled_model.time = time
        coupled_model.initialisation_time = initialisation_time

        coupled_model.hertzian_contact_coef = rose_data["track_info"]["materials"]["hertzian_contact_coef"]
        coupled_model.hertzian_power = rose_data["track_info"]["materials"]["hertzian_power"]

        coupled_model.solver = solver

        coupled_model.is_rayleigh_damping = False # rayleigh is added via scatter

        return coupled_model

    @staticmethod
    def get_bottom_boundary(rose_model: CoupledTrainTrack) -> List:
        """
        get all vertical degrees of freedom numbers of all  the bottom nodes of the coupled train track model.

        Note that for this procedure, it is required that the bottom nodes of the coupled train track model have an
        equal y-coordinate.

        :param rose_model: coupled train track model
        :return: list of bottom vertical degree of freedom numbers of the coupled train track model
        """

        # get minimum y coordinate of coupled train track mdoel
        y_coords = [node.coordinates[1] for node in rose_model.track.mesh.nodes]
        min_y_coord = min(y_coords)

        # find all nodes which have an equal y coordinate as the min y-coord
        bottom_nodes = [node for node in rose_model.track.mesh.nodes if np.isclose(node.coordinates[1], min_y_coord)]

        # get all vertical degrees of freedom of the bottom nodes
        bottom_dofs = [node.index_dof[1] for node in bottom_nodes]
        return bottom_dofs


    @staticmethod
    def recalculate_ndof(scatter_model: 'ReadMesh', rose_model: CoupledTrainTrack):
        """
        Recalculates all number of degrees of freedom of the scatter and rose model.

        :param scatter_model: scatter model
        :param rose_model: rose coupled train track model
        """

        # recalculate rose track numbers of degree of freedom, this includes scatter + rose track
        rose_model.track.total_n_dof = scatter_model.number_eq + rose_model.track.total_n_dof - len(scatter_model.eq_nb_dof_rose_nodes)
        rose_model.total_n_dof = scatter_model.number_eq +rose_model.total_n_dof -len(scatter_model.eq_nb_dof_rose_nodes)

        # todo check
        # rose_model.track_global_indices = rose_model.track_global_indices + scatter_model.number_eq - len(scatter_model.eq_nb_dof_rose_nodes)

        # this works because rose contact nodes are placed after the rail nodes in the mesh
        rose_model.track_global_indices = rose_model.track_global_indices + scatter_model.number_eq

        # recalculate rose train numbers of degree of freedom
        rose_model.train.contact_dofs = list(np.array(rose_model.train.contact_dofs) + scatter_model.number_eq -len(scatter_model.eq_nb_dof_rose_nodes))

        # reinitialise rose solvers with new number of degree of freedom
        # rose_model.solver.initialise(rose_model.total_n_dof, rose_model.time)
        rose_model.track.solver.initialise(rose_model.track.total_n_dof, [0, 1])

        # recalculate indices of degree of freedom on rose track nodes

        contact_nodes_found = 0
        for node in rose_model.track.mesh.nodes:
            for i, dof in enumerate(node.index_dof):
                if dof is not None:
                    if node.index_dof[i] in scatter_model.rose_eq_nb:
                        node.index_dof[i] = scatter_model.eq_nb_dof_rose_nodes[scatter_model.rose_eq_nb.index(dof)]
                        contact_nodes_found += 1
                    else:
                        # this works because rose contact nodes are placed after the rail nodes in the mesh
                        node.index_dof[i] = node.index_dof[i] + scatter_model.number_eq - contact_nodes_found
                    #todo connect rose-scatter nodes, currently visualisation of the results on the rose nodes will not
                    # work properly on the bottom elements of the rose model

        # recalculate indices of degree of freedom on rose train nodes
        for node in rose_model.train.mesh.nodes:
            for i, dof in enumerate(node.index_dof):
                if dof is not None:
                    node.index_dof[i] = node.index_dof[i] + scatter_model.number_eq - contact_nodes_found

        # add dof indices to model part
        for model_part in rose_model.track.model_parts:
            model_part.update_dof_indices()

        scatter_model.number_eq = rose_model.total_n_dof
