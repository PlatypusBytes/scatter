import rose

from rose.model.model_part import Material, Section
from rose.model.train_model import *
from rose.model.train_track_interaction import *
from solvers.newmark_solver import NewmarkSolver

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

#
# #    moving_load(self, nb_equations, eq_nb_dof, load_set, node, time_step, nodes_coord, steps=50)
#     def create_train_load(self):
#         """
#               Moving load along z-axis
#
#               :param nb_equations: total number of equations
#               :param eq_nb_dof: number of equation for each dof in node list
#               :param load_set: loading settings
#               :param node: node where the load starts
#               :param time_step: time step
#               :param nodes_coord: list of coordinates
#               :param steps: (optional: default = 50) number of steps to initialise load
#               """
#         # time
#
#         time = self.train.time
#
#
#         for wheel in self.train.wheels:
#
#
#         # time = load_set["time"]
#         # time = np.linspace(0, time, int(np.ceil(time / time_step)))
#         # generation of variable
#         self.force = lil_matrix(np.zeros((nb_equations, len(time))))
#         # load factor
#         factor = load_set["force"]
#
#         # index node
#         idx = np.where(nodes_coord[:, 0] == node)[0][0]
#         # find nodes along  with same x and y (load moves along z-axis)
#         idx_list = np.where((nodes_coord[:, 1] == nodes_coord[idx, 1]) & (nodes_coord[:, 2] == nodes_coord[idx, 2]))[0]
#
#         # find distances
#         dist = []
#         for i in idx_list:
#             dist.append(np.sqrt((nodes_coord[i, 3] - nodes_coord[idx, 3]) ** 2))
#
#         idx_list = idx_list[np.argsort(np.array(dist))]
#         dist = np.sort(np.array(dist))
#
#         # for each time in the analysis
#         for t in range(len(time)):
#
#             # load not moving while steps
#             if t < steps:
#                 speed = 0
#                 fct = np.linspace(0, 1, steps)[t]
#             else:
#                 speed = load_set["speed"]
#                 fct = 1
#
#             # if the load as reached the end of the model returns
#             if speed * (time[t] - time[steps - 1]) >= np.max(dist):
#                 return
#
#             # find location of the load
#             id = np.where(dist <= speed * (time[t] - time[steps - 1]))[0][-1]
#             node = [int(nodes_coord[idx_list[id], 0]), int(nodes_coord[idx_list[id + 1], 0])]
#
#             # compute local shape functions
#             x = speed * (time[t] - time[steps - 1]) + nodes_coord[idx_list[id], 3]
#             l = dist[id + 1] - dist[id]
#
#             shp = np.array([1 - x / l,
#                             x * l])
#
#             # for each node with load
#             for j, n in enumerate(node):
#                 for i, eq in enumerate(eq_nb_dof[n - 1]):
#                     if ~np.isnan(eq):
#                         self.force[int(eq), t] = float(factor[i]) * shp[j] * fct
#
#         return

    @staticmethod
    def assign_data_to_coupled_model(rose_data): #, train_info, track_info, time_int, soil):
        # choose solver
        # solver = solver_c.NewmarkSolver()
        solver = NewmarkSolver()
        # solver = solver_c.ZhaiSolver()

        all_element_model_parts = []
        all_meshes = []
        # loop over number of segments
        for idx in range(rose_data["track_info"]["geometry"]["n_segments"]):
            # set geometry of one segment
            element_model_parts, mesh = create_horizontal_track(rose_data["track_info"]["geometry"]["n_sleepers"][idx],
                                                                rose_data["track_info"]["geometry"]["sleeper_distance"],
                                                                rose_data["track_info"]["geometry"]["depth_soil"][idx])
            # add segment model parts and mesh to list
            all_element_model_parts.append(element_model_parts)
            all_meshes.append(mesh)

        # Setup global mesh and combine model parts of all segments
        rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, all_mesh = \
            combine_horizontal_tracks(all_element_model_parts, all_meshes)

        # Fixate the bottom boundary
        # bottom_boundaries = [add_no_displacement_boundary_to_bottom(soil_model_part)["bottom_boundary"] for
        #                      soil_model_part
        #                      in soil_model_parts]

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

        for idx, soil_model_part in enumerate(soil_model_parts):
            soil_model_part.stiffness = rose_data["soil_data"]["stiffness"]
            soil_model_part.damping = rose_data["soil_data"]["damping"]
            soil_model_part.mass = 1000

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
        # model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, side_boundaries] \
        #               + soil_model_parts + bottom_boundaries
        model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, side_boundaries] \
                      + soil_model_parts

        track.model_parts = model_parts

        # set up train
        train = rose_data["traffic_data"]["model"]
        train.time = time
        train.velocities = velocities

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

        coupled_model.is_rayleigh_damping = True
        coupled_model.damping_ratio = rose_data["track_info"]["materials"]["damping_ratio"]
        coupled_model.radial_frequency_one = rose_data["track_info"]["materials"]["omega_one"]
        coupled_model.radial_frequency_two = rose_data["track_info"]["materials"]["omega_two"]

        return coupled_model

    @staticmethod
    def get_bottom_boundary(rose_model):

        y_coords = [node.coordinates[1] for node in rose_model.track.mesh.nodes]
        min_y_coord = min(y_coords)

        bottom_nodes = [node for node in rose_model.track.mesh.nodes if np.isclose(node.coordinates[1], min_y_coord)]

        bottom_dofs = [node.index_dof[1] for node in bottom_nodes]
        return bottom_dofs


    @staticmethod
    def recalculate_ndof(data, rose_model):
        rose_model.track.total_n_dof = data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes)
        rose_model.total_n_dof = data.number_eq +rose_model.total_n_dof -len(data.eq_nb_dof_rose_nodes)

        rose_model.track_global_indices += data.number_eq

        #todo check if rose requires a list
        rose_model.train.contact_dofs = list(np.array(rose_model.train.contact_dofs) + data.number_eq -len(data.eq_nb_dof_rose_nodes))

        rose_model.solver.initialise(rose_model.total_n_dof, rose_model.time)
        rose_model.track.solver.initialise(rose_model.track.total_n_dof, rose_model.time)

        for node in rose_model.track.mesh.nodes:
            for i, dof in enumerate(node.index_dof):
                if dof is not None:
                    node.index_dof[i] += data.number_eq -len(data.eq_nb_dof_rose_nodes)
                    #todo connect rose-scatter nodes

        for node in rose_model.train.mesh.nodes:
            for i, dof in enumerate(node.index_dof):
                if dof is not None:
                    node.index_dof[i] += data.number_eq - len(data.eq_nb_dof_rose_nodes)

        data.number_eq = rose_model.total_n_dof
