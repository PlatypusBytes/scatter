"""Assembly of the global stiffness, mass and damping matrices."""
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix, spmatrix

from scatter import discretisation, material_models, utils
from scatter.element_types import create_element
from scatter.globals import ASSEMBLY_MEMORY_BUDGET
from scatter.mesher import ReadMesh

# material properties ("Young" [Pa], "poisson" [-] and "density" [kg/m^3]) per material name
Materials = Dict[str, Dict[str, float]]


class SparseAccumulator:
    """Sums a sequence of sparse matrices.

    The matrices are added pairwise, like a binary counter, so the cost grows with nnz * log(n) instead of nnz * n when
    adding n chunks into one growing matrix.
    """

    def __init__(self, shape: Tuple[int, int]) -> None:
        """Initialise an empty sum.

        Args:
            shape: shape of the matrices.
        """
        self.__shape = shape
        self.__partial_sums: List[Tuple[int, csr_matrix]] = []

    def add(self, matrix: csr_matrix) -> None:
        """Add a matrix to the sum.

        Args:
            matrix: matrix to add.
        """
        level = 0
        while self.__partial_sums and self.__partial_sums[-1][0] == level:
            matrix = matrix + self.__partial_sums.pop()[1]
            level += 1
        self.__partial_sums.append((level, matrix))

    def total(self) -> csr_matrix:
        """Return the sum of all matrices added so far."""
        total = csr_matrix(self.__shape)
        for _, matrix in self.__partial_sums:
            total = total + matrix
        return total


class GenerateMatrix:
    """Global sparse matrices of the system of equations."""

    def __init__(self, nb_equations: int, order: int) -> None:
        """Initialise empty global matrices.

        Args:
            nb_equations: number of equations in the system.
            order: order of the Gauss integration.
        """
        self.M = csr_matrix((nb_equations, nb_equations))
        self.C = csr_matrix((nb_equations, nb_equations))
        self.K = csr_matrix((nb_equations, nb_equations))
        self.order = order

    def generate_stiffness_and_mass(self, data: ReadMesh, material: Materials) -> None:
        """Generate and assemble the global stiffness and consistent mass matrices.

        The element matrices are computed for chunks of elements at once, with the chunk size set by
        `ASSEMBLY_MEMORY_BUDGET`.

        Args:
            data: mesh and geometry data.
            material: material properties per material name.
        """
        element = discretisation.ElementBatch(data.element_type, self.order)
        young, poisson, density = self.__element_properties(data, material)
        node_rows = data.get_node_rows(data.elem)
        coordinates = data.nodes[:, 1:]
        shape = (data.number_eq, data.number_eq)

        stiffness = SparseAccumulator(shape)
        mass = SparseAccumulator(shape)
        chunk_size = max(1, ASSEMBLY_MEMORY_BUDGET // element.memory_per_element)
        for start in range(0, len(data.elem), chunk_size):
            chunk = slice(start, start + chunk_size)

            # generate shape functions, Jacobian and B matrix of all elements in the chunk
            element.generate(coordinates[node_rows[chunk]])

            D = material_models.stiffness_elasticity_batch(young[chunk], poisson[chunk], data.dimension)
            stiffness.add(self.__assemble(data.eq_nb_elem[chunk], element.compute_stiffness(D), shape))
            mass.add(self.__assemble(data.eq_nb_elem[chunk], element.compute_mass(density[chunk]), shape))

        self.K = stiffness.total()
        self.M = mass.total()

    @staticmethod
    def __element_properties(
            data: ReadMesh,
            material: Materials) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Collect the elastic properties of each element.

        Args:
            data: mesh and geometry data.
            material: material properties per material name.

        Returns:
            Young modulus [Pa], Poisson ratio [-] and density [kg/m^3] of each element.
        """
        material_names = dict(np.array(data.materials)[:, 1:])

        # look up each material once: with random fields there can be one material per element
        material_index, element_material = np.unique(np.asarray(data.materials_index, dtype=int), return_inverse=True)
        properties = np.array([[material[material_names[str(idx)]][key] for key in ("Young", "poisson", "density")]
                               for idx in material_index],
                              dtype=float)

        young, poisson, density = properties[element_material.ravel()].T
        return young, poisson, density

    @staticmethod
    def __assemble(eq_nb: npt.NDArray[np.float64], values: npt.NDArray[np.float64], shape: Tuple[int,
                                                                                                 int]) -> csr_matrix:
        """Assemble element matrices into a global sparse matrix.

        Args:
            eq_nb: equation number of each degree of freedom of each element, NaN where the dof is fixed, with shape
                (element, dof).
            values: element matrices, with shape (element, dof, dof).
            shape: shape of the global matrix.

        Returns:
            Global matrix, where contributions to the same entry are summed.
        """
        nb_dofs = eq_nb.shape[1]
        rows = np.repeat(eq_nb, nb_dofs, axis=1).ravel()
        cols = np.tile(eq_nb, (1, nb_dofs)).ravel()
        is_free = ~(np.isnan(rows) | np.isnan(cols))

        return coo_matrix(
            (values.reshape(-1)[is_free], (rows[is_free].astype(np.int64), cols[is_free].astype(np.int64))),
            shape=shape).tocsr()

    def add_rose_stiffness(self, data: ReadMesh, rose_model: Any) -> None:
        """Couple the rose stiffness matrix to the scatter stiffness matrix.

        Args:
            data: mesh and geometry data.
            rose_model: rose coupled train track model; its global and track stiffness matrices are updated.
        """
        combined_k = self.add_rose_matrix(self.K, rose_model.global_stiffness_matrix, data.eq_nb_dof_rose_nodes,
                                          data.rose_eq_nb)
        nb_track = data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes)

        self.K = combined_k
        rose_model.global_stiffness_matrix = combined_k
        rose_model.track.global_stiffness_matrix = combined_k[:nb_track, :nb_track]

    def add_rose_mass(self, data: ReadMesh, rose_model: Any) -> None:
        """Couple the rose mass matrix to the scatter mass matrix.

        Args:
            data: mesh and geometry data.
            rose_model: rose coupled train track model; its global and track mass matrices are updated.
        """
        combined_m = self.add_rose_matrix(self.M, rose_model.global_mass_matrix, data.eq_nb_dof_rose_nodes,
                                          data.rose_eq_nb)
        nb_track = data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes)

        self.M = combined_m
        rose_model.global_mass_matrix = combined_m
        rose_model.track.global_mass_matrix = combined_m[:nb_track, :nb_track]

    def add_rose_damping(self, data: ReadMesh, rose_model: Any) -> None:
        """Couple the rose damping matrix to the scatter damping matrix.

        Args:
            data: mesh and geometry data.
            rose_model: rose coupled train track model; its global and track damping matrices are updated.
        """
        combined_c = self.add_rose_matrix(self.C, rose_model.global_damping_matrix, data.eq_nb_dof_rose_nodes,
                                          data.rose_eq_nb)
        nb_track = data.number_eq + rose_model.track.total_n_dof - len(data.eq_nb_dof_rose_nodes)

        self.C = combined_c
        rose_model.global_damping_matrix = combined_c
        rose_model.track.global_damping_matrix = combined_c[:nb_track, :nb_track]

    def damping_Rayleigh(self, damp: Sequence[float]) -> None:
        """Add Rayleigh damping, C = a * M + b * K, to the damping matrix.

        Args:
            damp: first frequency [Hz], damping ratio at the first frequency [-], second frequency [Hz] and damping
                ratio at the second frequency [-].

        Raises:
            SystemExit: if the two frequencies are equal.
        """
        f1, d1, f2, d2 = damp[0], damp[1], damp[2], damp[3]
        if f1 == f2:
            raise SystemExit('Frequencies for the Rayleigh damping are the same.')

        damp_mat = 1 / 2 * np.array([[1 / (2 * np.pi * f1), 2 * np.pi * f1], [1 / (2 * np.pi * f2), 2 * np.pi * f2]])
        coefs = np.linalg.solve(damp_mat, np.array([d1, d2]))

        self.C = csr_matrix(self.C + self.M.tocsr() * coefs[0] + self.K.tocsr() * coefs[1])

    def add_rose_matrix(self, scatter_matrix: spmatrix, rose_matrix: Union[spmatrix, npt.NDArray[np.float64]],
                        contact_dofs_scatter: Union[Sequence[int], npt.NDArray[np.int64]],
                        contact_dofs_rose: Union[Sequence[int], npt.NDArray[np.int64]]) -> csr_matrix:
        """Combine a scatter matrix and a rose matrix into one coupled matrix.

        The rose degrees of freedom in contact with scatter are merged into the scatter degrees of freedom; the other
        rose degrees of freedom are appended after the scatter ones, in their original order. Of the couplings between
        contact degrees of freedom, only the diagonal is added to the scatter matrix.

        Args:
            scatter_matrix: scatter matrix.
            rose_matrix: rose matrix.
            contact_dofs_scatter: scatter equation numbers of the contact degrees of freedom.
            contact_dofs_rose: rose equation numbers of the contact degrees of freedom, in the same order as
                `contact_dofs_scatter`.

        Returns:
            Coupled matrix.
        """
        nb_scatter = scatter_matrix.shape[0]
        nb_rose = rose_matrix.shape[0]
        dofs_scatter = np.asarray(contact_dofs_scatter, dtype=np.int64)
        dofs_rose = np.asarray(contact_dofs_rose, dtype=np.int64)

        is_contact = np.zeros(nb_rose, dtype=bool)
        is_contact[dofs_rose] = True

        # coupled equation number of each rose dof
        rose_to_coupled = np.empty(nb_rose, dtype=np.int64)
        rose_to_coupled[~is_contact] = nb_scatter + np.arange(np.count_nonzero(~is_contact))
        rose_to_coupled[dofs_rose] = dofs_scatter

        rose = coo_matrix(rose_matrix)
        rose.sum_duplicates()
        keep = ~(is_contact[rose.row] & is_contact[rose.col] & (rose.row != rose.col))

        scatter = coo_matrix(scatter_matrix)
        rows = np.concatenate((scatter.row, rose_to_coupled[rose.row[keep]]))
        cols = np.concatenate((scatter.col, rose_to_coupled[rose.col[keep]]))
        values = np.concatenate((scatter.data, rose.data[keep]))

        nb_coupled = nb_scatter + nb_rose - len(dofs_scatter)
        return coo_matrix((values, (rows, cols)), shape=(nb_coupled, nb_coupled)).tocsr()

    def absorbing_boundaries(self, data: ReadMesh, material: Materials, parameters_viscous: Sequence[float],
                             parameters_stiff: float) -> None:
        """Add the absorbing boundaries to the damping and stiffness matrices, following Lysmer and Kuhlemeyer.

        An element has an absorbing boundary in a direction when exactly `nb_nodes_lower_elem` of its nodes are
        absorbing in that direction: a surface in 3D, a line in 2D. The dashpots and springs act on the dof in that
        direction, with the P-wave terms where the boundary is perpendicular to the dof and the S-wave terms otherwise.

        Args:
            data: mesh and geometry data.
            material: material properties per material name.
            parameters_viscous: factors on the dashpots perpendicular and tangential to the boundary [-].
            parameters_stiff: divisor of the spring stiffness of the absorbing boundary [m].
        """
        young, poisson, density = self.__element_properties(data, material)
        constrained_modulus = young * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
        shear_modulus = young / (2 * (1 + poisson))
        vp = np.sqrt(constrained_modulus / density)
        vs = np.sqrt(shear_modulus / density)

        # perpendicular direction flag of each equation
        is_free = ~np.isnan(data.eq_nb_dof)
        bc_dir_eq = np.zeros(self.C.shape[0], dtype=int)
        bc_dir_eq[data.eq_nb_dof[is_free].astype(int)] = data.BC_dir[is_free]

        node_rows = data.get_node_rows(data.elem)
        coordinates = data.nodes[:, 1:1 + data.dimension]
        is_absorbing = data.type_BC == "Absorb"
        nb_corner_nodes = create_element(data.element_type).n_corner_nodes
        boundary = discretisation.BoundaryElementBatch(data.lower_element_type, self.order)
        shape = self.C.shape
        absorbing_damping = SparseAccumulator(shape)
        absorbing_stiffness = SparseAccumulator(shape)

        for direction in range(data.dimension):
            is_absorbing_elem = is_absorbing[node_rows, direction]
            elements = np.where(is_absorbing_elem.sum(axis=1) == data.nb_nodes_lower_elem)[0]
            if len(elements) == 0:
                continue

            # boundary nodes in gmsh order of the boundary element; corners come first in the element connectivity
            local_nodes = np.nonzero(is_absorbing_elem[elements])[1].reshape(len(elements), -1)
            boundary_rows = np.take_along_axis(node_rows[elements], local_nodes, axis=1)
            gmsh_order = utils.order_face_nodes(coordinates[boundary_rows], local_nodes < nb_corner_nodes)
            boundary_rows = np.take_along_axis(boundary_rows, gmsh_order, axis=1)

            # unitary absorbing boundary matrix, with the equation numbers in the same node order
            boundary.generate(coordinates[boundary_rows])
            abs_bound = boundary.compute_scalar_mass(np.ones(len(elements)))
            eq_nb = data.eq_nb_dof[boundary_rows, direction]

            # absorbing boundary coefficients: vp for perpendicular, vs otherwise
            is_perpendicular = bc_dir_eq[eq_nb.astype(int)] == 1
            damping_factor = np.where(is_perpendicular, parameters_viscous[0] * (density * vp)[elements, None],
                                      parameters_viscous[1] * (density * vs)[elements, None])
            stiffness_factor = np.where(is_perpendicular, constrained_modulus[elements, None], shear_modulus[elements,
                                                                                                             None])

            absorbing_damping.add(self.__assemble(eq_nb, abs_bound * damping_factor[:, None, :], shape))
            absorbing_stiffness.add(self.__assemble(eq_nb, np.abs(abs_bound) * stiffness_factor[:, None, :], shape))

        self.C = csr_matrix(self.C + absorbing_damping.total())
        self.K = csr_matrix(self.K + absorbing_stiffness.total() / parameters_stiff)
