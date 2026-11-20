"""Tests of the element routines, the assembly of the global matrices and the absorbing boundaries."""
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse import random as sparse_random

from scatter import discretisation, element_types, material_models, mesher, random_fields, system_matrix, utils
from scatter.scatter import scatter

MESH_FOLDER = os.path.join(os.path.dirname(__file__), "..", "integration_tests", "mesh")

MESHES = {
    "hexa8": "cube.msh",
    "hexa20": "column_high_order.msh",
    "tetra4": "column_3D_tetra4.msh",
    "tetra10": "column_3D_tetra10.msh",
    "quad4": "column_2D.msh",
    "tri3": "column_2D_tri3.msh",
    "tri6": "column_2D_tri6.msh",
}

MATERIALS = {
    "soft": {
        "density": 1800.,
        "Young": 20e6,
        "poisson": 0.25
    },
    "stiff": {
        "density": 2100.,
        "Young": 80e6,
        "poisson": 0.3
    },
}


def read_mesh(element_type: str, boundary: str = "fixed", two_materials: bool = True) -> mesher.ReadMesh:
    """Read a test mesh and number its equations.

    Args:
        element_type: element type of the mesh, a key of MESHES.
        boundary: "fixed" to fix the bottom and the sides, "absorbing" to make the bottom and the sides absorbing in
            their perpendicular direction, "absorbing_bottom" or "absorbing_bottom_all" to make only the bottom
            absorbing in its perpendicular direction or in all directions, "free" for no boundary conditions.
        two_materials: assign the materials "soft" and "stiff" to alternating elements, instead of only "soft".

    Returns:
        Mesh with equation numbers and connectivities.
    """
    model = mesher.ReadMesh(os.path.join(MESH_FOLDER, MESHES[element_type]))
    model.read_gmsh()

    (x0, y0, z0), (x1, y1, z1) = model.nodes[:, 1:].min(axis=0), model.nodes[:, 1:].max(axis=0)
    bc: Dict[str, List[Any]]
    if boundary == "free":
        bc = {}
    elif boundary.startswith("absorbing_bottom"):
        everywhere = boundary == "absorbing_bottom_all"
        if model.dimension == 2:
            bc = {"bottom": ["22" if everywhere else "02", [[x0, y0, 0], [x1, y0, 0]]]}
        else:
            bc = {"bottom": ["222" if everywhere else "020", [[x0, y0, z0], [x1, y0, z0], [x0, y0, z1]]]}
    elif model.dimension == 2:
        bc = {"bottom": ["01", [[x0, y0, 0], [x1, y0, 0]]], "left": ["10", [[x0, y0, 0], [x0, y1, 0]]]}
    else:
        side = "2" if boundary == "absorbing" else "1"
        bc = {
            "bottom": [f"0{side}0", [[x0, y0, z0], [x1, y0, z0], [x0, y0, z1]]],
            "left": [f"{side}00", [[x0, y0, z0], [x0, y0, z1], [x0, y1, z0]]],
            "right": [f"{side}00", [[x1, y0, z0], [x1, y0, z1], [x1, y1, z0]]],
            "front": [f"00{side}", [[x0, y0, z0], [x1, y0, z0], [x0, y1, z0]]],
            "back": [f"00{side}", [[x0, y0, z1], [x1, y0, z1], [x0, y1, z1]]],
        }
    model.read_bc(bc)
    model.mapping()
    model.connectivities()

    model.materials = [[model.dimension, 1, "soft"], [model.dimension, 2, "stiff"]]
    model.materials_index = np.ones(len(model.elem), dtype=int)
    if two_materials:
        model.materials_index[1::2] = 2
    return model


def element_properties(model: mesher.ReadMesh, index: int) -> Tuple[npt.NDArray[np.float64], float, float, float]:
    """Return the node coordinates, Young modulus, Poisson ratio and density of one element of a test mesh."""
    name = "soft" if model.materials_index[index] == 1 else "stiff"
    rows = model.get_node_rows(model.elem[index])
    return (model.nodes[rows, 1:], MATERIALS[name]["Young"], MATERIALS[name]["poisson"], MATERIALS[name]["density"])


def single_element(model: mesher.ReadMesh, xyz: npt.NDArray[np.float64]) -> Any:
    """Compute one element with the per-element implementation, which serves as reference."""
    element: Any
    if model.dimension == 3:
        element = discretisation.VolumeElement(model.element_type, 2)
    else:
        element = discretisation.SurfaceElement(model.element_type, 2)
    element.generate(xyz)
    return element


def reference_matrices(model: mesher.ReadMesh) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Assemble dense global stiffness and mass matrices element by element."""
    K = np.zeros((model.number_eq, model.number_eq))
    M = np.zeros((model.number_eq, model.number_eq))
    for index in range(len(model.elem)):
        xyz, young, poisson, density = element_properties(model, index)
        element = single_element(model, xyz)
        Ke = element.compute_stiffness(material_models.stiffness_elasticity(young, poisson, model.dimension))
        Me = element.compute_mass(density)

        is_free = ~np.isnan(model.eq_nb_elem[index])
        eq_nb = model.eq_nb_elem[index][is_free].astype(int)
        K[np.ix_(eq_nb, eq_nb)] += Ke[np.ix_(is_free, is_free)]
        M[np.ix_(eq_nb, eq_nb)] += Me[np.ix_(is_free, is_free)]
    return K, M


def bottom_extent(model: mesher.ReadMesh) -> Tuple[float, float, float]:
    """Return the start and end x of the bottom [m], and its size in z (1 in 2D) [m]."""
    (x0, _, z0), (x1, _, z1) = model.nodes[:, 1:].min(axis=0), model.nodes[:, 1:].max(axis=0)
    return x0, x1, (z1 - z0) if model.dimension == 3 else 1.


def wave_properties(material: Dict[str, float]) -> Tuple[float, float, float, float]:
    """Return the constrained modulus [Pa], shear modulus [Pa], P-wave speed [m/s] and S-wave speed [m/s]."""
    young, poisson, rho = material["Young"], material["poisson"], material["density"]
    constrained_modulus = young * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
    shear_modulus = young / (2 * (1 + poisson))
    return constrained_modulus, shear_modulus, np.sqrt(constrained_modulus / rho), np.sqrt(shear_modulus / rho)


def assert_gmsh_order(coordinates: npt.NDArray[np.float64], nb_corners: int) -> None:
    """Check that the corners of a face are cyclic and that midside node k lies between corners k and k + 1."""
    corners = coordinates[:nb_corners]
    if nb_corners == 4:
        # consecutive corners of a rectangle are not opposite: the diagonals are the longest distances
        edges = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        diagonals = np.linalg.norm(corners[:2] - corners[2:], axis=1)
        assert np.all(edges[:, None] < diagonals[None, :])
    if len(coordinates) == nb_corners:
        return
    nb_edges = nb_corners if nb_corners > 2 else 1
    middles = ((corners + np.roll(corners, -1, axis=0)) / 2)[:nb_edges]
    np.testing.assert_allclose(coordinates[nb_corners:], middles, atol=1e-12)


class TestElementTypes:
    """Shape functions of the element types."""

    @pytest.mark.parametrize(
        "elem_type", ["line2", "line3", "quad4", "quad8", "tri3", "tri6", "hexa8", "hexa20", "tetra4", "tetra10"])
    def test_partition_of_unity(self, elem_type: str):
        """The shape functions sum to one and their derivatives to zero at the Gauss points."""
        N, dN, _ = discretisation.reference_integration(elem_type, 2)
        np.testing.assert_allclose(N.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose(dN.sum(axis=1), 0., atol=1e-12)

    @pytest.mark.parametrize(
        "elem_type", ["line2", "line3", "quad4", "quad8", "tri3", "tri6", "hexa8", "hexa20", "tetra4", "tetra10"])
    def test_derivatives_match_finite_differences(self, elem_type: str):
        """The derivatives of the shape functions equal central finite differences of the shape functions."""
        element = element_types.create_element(elem_type)
        point = np.array([0.21, 0.13, 0.17])[:element.dN.shape[1]]
        element.shape_functions(point)
        derivatives = element.dN.copy()

        step = 1e-6
        for direction in range(len(point)):
            shift = np.eye(len(point))[direction] * step
            element.shape_functions(point + shift)
            forward = element.N.copy()
            element.shape_functions(point - shift)
            backward = element.N.copy()
            np.testing.assert_allclose(derivatives[:, direction], (forward - backward) / (2 * step), atol=1e-8)

    @pytest.mark.parametrize("node_coordinates",
                             [[[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]]])
    def test_quad8_shape_functions_at_nodes(self, node_coordinates: List[List[float]]):
        """Each quad8 shape function is one at its own node and zero at the other nodes, in gmsh node order."""
        element = element_types.create_element("quad8")
        for i, uv in enumerate(node_coordinates):
            element.shape_functions(np.array(uv, dtype=float))
            np.testing.assert_allclose(element.N, np.eye(8)[i], atol=1e-12)

    @pytest.mark.parametrize("elem_type, node_coordinates", [("line2", [-1., 1.]), ("line3", [-1., 1., 0.])])
    def test_line_shape_functions_at_nodes(self, elem_type: str, node_coordinates: List[float]):
        """Each line shape function is one at its own node and zero at the other nodes, in gmsh node order."""
        element = element_types.create_element(elem_type)
        for i, u in enumerate(node_coordinates):
            element.shape_functions(np.array([u]))
            np.testing.assert_allclose(element.N, np.eye(len(node_coordinates))[i], atol=1e-12)

    def test_unsupported_element(self):
        """An unknown element type raises a ValueError."""
        with pytest.raises(ValueError, match=re.escape("Element type pyramid5 is not supported.")):
            element_types.create_element("pyramid5")


class TestElementBatch:
    """Vectorised element matrices."""

    @pytest.mark.parametrize("element_type", list(MESHES))
    def test_matches_single_element(self, element_type: str):
        """Batched stiffness and mass matrices equal the per-element VolumeElement / SurfaceElement result."""
        model = read_mesh(element_type)
        indices = np.arange(min(6, len(model.elem)))
        properties = [element_properties(model, i) for i in indices]

        batch = discretisation.ElementBatch(model.element_type, 2)
        batch.generate(np.array([p[0] for p in properties]))
        Ke = batch.compute_stiffness(
            material_models.stiffness_elasticity_batch(np.array([p[1] for p in properties]),
                                                       np.array([p[2] for p in properties]), model.dimension))
        Me = batch.compute_mass(np.array([p[3] for p in properties]))

        for i, (xyz, young, poisson, density) in enumerate(properties):
            element = single_element(model, xyz)
            D = material_models.stiffness_elasticity(young, poisson, model.dimension)
            scale_K = np.abs(Ke[i]).max()
            scale_M = np.abs(Me[i]).max()
            np.testing.assert_allclose(Ke[i] / scale_K, element.compute_stiffness(D) / scale_K, rtol=0, atol=1e-12)
            np.testing.assert_allclose(Me[i] / scale_M, element.compute_mass(density) / scale_M, rtol=0, atol=1e-12)

    def test_unsupported_element(self):
        """A boundary-only element type raises a ValueError."""
        with pytest.raises(ValueError, match=re.escape("Element type line2 is not supported.")):
            discretisation.ElementBatch("line2", 2)

    @pytest.mark.parametrize("element_class, elem_type, message", [
        (discretisation.VolumeElement, "quad4", "Element type quad4 is not a volume element."),
        (discretisation.SurfaceElement, "hexa8", "Element type hexa8 is not a plane element."),
    ])
    def test_single_element_wrong_type(self, element_class: Any, elem_type: str, message: str):
        """The per-element classes reject element types of the other dimension."""
        with pytest.raises(ValueError, match=re.escape(message)):
            element_class(elem_type, 2)


class TestBoundaryElementBatch:
    """Integration over boundary elements."""

    @pytest.mark.parametrize("elem_type, coordinates", [
        ("quad4", [[0, 0, 0], [2, 0, 0], [2, 3, 3], [0, 3, 3]]),
        ("quad8", [[0, 0, 0], [2, 0, 0], [2, 3, 3], [0, 3, 3], [1, 0, 0], [2, 1.5, 1.5], [1, 3, 3], [0, 1.5, 1.5]]),
        ("tri3", [[0, 0, 0], [2, 0, 0], [0, 3, 3]]),
        ("tri6", [[0, 0, 0], [2, 0, 0], [0, 3, 3], [1, 0, 0], [1, 1.5, 1.5], [0, 1.5, 1.5]]),
    ])
    def test_area_of_inclined_surface(self, elem_type: str, coordinates: List[List[float]]):
        """The integral of N^T N over an inclined surface sums to its area."""
        boundary = discretisation.BoundaryElementBatch(elem_type, 2)
        boundary.generate(np.array([coordinates], dtype=float))
        area = 2 * np.sqrt(3**2 + 3**2) * (1 if elem_type.startswith("quad") else 0.5)
        np.testing.assert_allclose(boundary.compute_scalar_mass(np.ones(1)).sum(), area)

    @pytest.mark.parametrize("elem_type, coordinates", [("line2", [[1, 1], [4, 5]]),
                                                        ("line3", [[1, 1], [4, 5], [2.5, 3]])])
    def test_length_of_inclined_line(self, elem_type: str, coordinates: List[List[float]]):
        """The integral of N^T N over an inclined line sums to its length."""
        boundary = discretisation.BoundaryElementBatch(elem_type, 2)
        boundary.generate(np.array([coordinates], dtype=float))
        np.testing.assert_allclose(boundary.compute_scalar_mass(np.ones(1)).sum(), 5.)

    def test_unsupported_element(self):
        """A volume element type raises a ValueError."""
        with pytest.raises(ValueError, match=re.escape("Element type hexa8 is not a boundary element.")):
            discretisation.BoundaryElementBatch("hexa8", 2)


class TestOrderFaceNodes:
    """gmsh node order of boundary faces."""

    @pytest.mark.parametrize("coordinates, nb_corners", [
        ([[0, 0, 0], [2, 0, 0], [0, 0, 1], [2, 0, 1]], 4),
        ([[0, 0, 0], [2, 0, 1], [2, 0, 0], [0, 0, 1], [1, 0, 1], [0, 0, 0.5], [2, 0, 0.5], [1, 0, 0]], 4),
        ([[0, 0, 0], [0, 2, 1], [2, 0, 0], [1, 0, 0], [0, 1, 0.5], [1, 1, 0.5]], 3),
        ([[0, 0], [3, 0], [1.5, 0]], 2),
        ([[3, 0], [0, 0]], 2),
    ])
    def test_gmsh_order(self, coordinates: List[List[float]], nb_corners: int):
        """Corners in connectivity order (Z-ordered quads included) come out cyclic, with matching midside nodes."""
        xyz = np.array([coordinates], dtype=float)
        is_corner = np.arange(len(coordinates))[None, :] < nb_corners
        order = utils.order_face_nodes(xyz, is_corner)
        assert sorted(order[0].tolist()) == list(range(len(coordinates)))
        assert_gmsh_order(xyz[0, order[0]], nb_corners)

    def test_unequal_number_of_corners(self):
        """Faces with different numbers of corner nodes raise a ValueError."""
        is_corner = np.array([[True, True, True, False], [True, True, False, False]])
        with pytest.raises(ValueError, match=re.escape("All faces must have the same number of corner nodes.")):
            utils.order_face_nodes(np.zeros((2, 4, 3)), is_corner)

    def test_wrong_number_of_midside_nodes(self):
        """A triangle with two midside nodes raises a ValueError."""
        xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0.5, 0.5, 0]]], dtype=float)
        is_corner = np.array([[True, True, True, False, False]])
        with pytest.raises(ValueError,
                           match=re.escape("A face with 3 corner nodes must have 0 or 3 midside nodes, "
                                           "not 2.")):
            utils.order_face_nodes(xyz, is_corner)

    def test_midside_nodes_not_on_edges(self):
        """Midside nodes that do not match the edges one to one raise a ValueError."""
        xyz = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0.5, 0.01, 0], [0.5, 0.02, 0]]], dtype=float)
        is_corner = np.array([[True, True, True, False, False, False]])
        with pytest.raises(ValueError, match=re.escape("The midside nodes of a face cannot be matched to its edges.")):
            utils.order_face_nodes(xyz, is_corner)


class TestStiffnessElasticityBatch:
    """Vectorised elastic material matrices."""

    @pytest.mark.parametrize("dimension", [2, 3])
    def test_matches_single_material(self, dimension: int):
        """Each material matrix equals the one from stiffness_elasticity."""
        young = np.array([10e6, 50e6, 200e9])
        poisson = np.array([0.1, 0.3, 0.45])
        D = material_models.stiffness_elasticity_batch(young, poisson, dimension)
        for i in range(len(young)):
            np.testing.assert_allclose(D[i], material_models.stiffness_elasticity(young[i], poisson[i], dimension))

    def test_unsupported_dimension(self):
        """A dimension other than 2 or 3 raises a ValueError."""
        with pytest.raises(ValueError, match=re.escape("Dimension 1 is not supported.")):
            material_models.stiffness_elasticity_batch(np.array([1e6]), np.array([0.2]), 1)


class TestGenerateStiffnessAndMass:
    """Assembly of the global stiffness and mass matrices."""

    @pytest.mark.parametrize("element_type", ["hexa20", "tetra4", "tetra10", "quad4", "tri6"])
    def test_matches_element_by_element_assembly(self, element_type: str):
        """Global matrices with fixed dofs and two materials equal an element-by-element dense assembly."""
        model = read_mesh(element_type)
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.generate_stiffness_and_mass(model, MATERIALS)

        K_ref, M_ref = reference_matrices(model)
        assert matrix.K.shape == (model.number_eq, model.number_eq)
        np.testing.assert_allclose(matrix.K.toarray() / np.abs(K_ref).max(), K_ref / np.abs(K_ref).max(), atol=1e-12)
        np.testing.assert_allclose(matrix.M.toarray() / np.abs(M_ref).max(), M_ref / np.abs(M_ref).max(), atol=1e-12)

    def test_total_mass_and_rigid_body_motion(self):
        """Without boundary conditions, the mass sums to rho * volume per direction and translations cost no force."""
        model = read_mesh("hexa8", boundary="free", two_materials=False)
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.generate_stiffness_and_mass(model, MATERIALS)

        volume = np.prod(model.nodes[:, 1:].max(axis=0) - model.nodes[:, 1:].min(axis=0))
        np.testing.assert_allclose(matrix.M.sum(), 3 * MATERIALS["soft"]["density"] * volume)
        for direction in range(3):
            translation = np.zeros(model.number_eq)
            translation[model.eq_nb_dof[:, direction].astype(int)] = 1.
            force = matrix.K @ translation
            np.testing.assert_allclose(force / np.abs(matrix.K).max(), 0., atol=1e-10)

    def test_chunking_gives_same_result(self, monkeypatch: pytest.MonkeyPatch):
        """Assembling one element per chunk gives the same matrices as a single chunk."""
        model = read_mesh("tetra4")
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.generate_stiffness_and_mass(model, MATERIALS)

        monkeypatch.setattr(system_matrix, "ASSEMBLY_MEMORY_BUDGET", 1)
        chunked = system_matrix.GenerateMatrix(model.number_eq, 2)
        chunked.generate_stiffness_and_mass(model, MATERIALS)

        np.testing.assert_allclose(chunked.K.toarray(), matrix.K.toarray(), rtol=1e-12, atol=1e-6)
        np.testing.assert_allclose(chunked.M.toarray(), matrix.M.toarray(), rtol=1e-12, atol=1e-12)

    def test_matrices_are_csr(self):
        """The assembled matrices are CSR, so no LIL conversion is needed downstream."""
        model = read_mesh("tetra4")
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.generate_stiffness_and_mass(model, MATERIALS)
        matrix.damping_Rayleigh([1, 0.02, 60, 0.02])
        assert all(isinstance(m, csr_matrix) for m in (matrix.K, matrix.M, matrix.C))

    def test_equal_rayleigh_frequencies(self):
        """Equal Rayleigh frequencies stop the calculation."""
        matrix = system_matrix.GenerateMatrix(2, 2)
        with pytest.raises(SystemExit, match=re.escape("Frequencies for the Rayleigh damping are the same.")):
            matrix.damping_Rayleigh([1, 0.02, 1, 0.02])


class TestAbsorbingBoundaries:
    """Absorbing boundaries in 2D and 3D."""

    @pytest.mark.parametrize("element_type", list(MESHES))
    def test_dashpots_integrate_over_bottom(self, element_type: str):
        """The bottom dashpots sum to rho * vp * area, and u^T C u to rho * vp * integral of x^2 for u = x.

        The second check depends on which node pairs are coupled, so it fails when the node order of the boundary
        elements does not match the order of their equation numbers.
        """
        model = read_mesh(element_type, boundary="absorbing_bottom", two_materials=False)
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.absorbing_boundaries(model, MATERIALS, [1, 1], 1e3)

        x0, x1, depth = bottom_extent(model)
        _, _, vp, _ = wave_properties(MATERIALS["soft"])
        impedance = MATERIALS["soft"]["density"] * vp

        is_bottom = np.isclose(model.nodes[:, 2], model.nodes[:, 2].min())
        u = np.zeros(model.number_eq)
        u[model.eq_nb_dof[is_bottom, 1].astype(int)] = model.nodes[is_bottom, 1]

        np.testing.assert_allclose(matrix.C.sum(), impedance * (x1 - x0) * depth)
        np.testing.assert_allclose(u @ (matrix.C @ u), impedance * depth * (x1**3 - x0**3) / 3)
        np.testing.assert_allclose(matrix.C.toarray(), matrix.C.toarray().T, atol=1e-9)

    @pytest.mark.parametrize("element_type", ["hexa8", "tetra4", "quad4", "tri3"])
    def test_springs_sum_over_bottom(self, element_type: str):
        """For linear boundary elements, the bottom springs sum to Ec * area / parameters_stiff."""
        model = read_mesh(element_type, boundary="absorbing_bottom", two_materials=False)
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.absorbing_boundaries(model, MATERIALS, [1, 1], 1e3)

        x0, x1, depth = bottom_extent(model)
        constrained_modulus, _, _, _ = wave_properties(MATERIALS["soft"])
        np.testing.assert_allclose(matrix.K.sum(), constrained_modulus * (x1 - x0) * depth / 1e3)

    @pytest.mark.parametrize("element_type", ["hexa8", "tri6"])
    def test_tangential_dashpots(self, element_type: str):
        """Dofs along the boundary get dashpots with the S-wave speed, scaled by the second viscous parameter."""
        model = read_mesh(element_type, boundary="absorbing_bottom_all", two_materials=False)
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.absorbing_boundaries(model, MATERIALS, [1, 0.5], 1e3)

        x0, x1, depth = bottom_extent(model)
        _, _, _, vs = wave_properties(MATERIALS["soft"])
        x_dofs = model.eq_nb_dof[~np.isnan(model.eq_nb_dof[:, 0]), 0].astype(int)
        C_x = matrix.C.toarray()[np.ix_(x_dofs, x_dofs)]
        np.testing.assert_allclose(C_x.sum(), 0.5 * MATERIALS["soft"]["density"] * vs * (x1 - x0) * depth)

    def test_totals_on_cube(self):
        """On a box with five absorbing faces, the dashpots sum to rho * vp * area of the faces."""
        model = read_mesh("hexa8", boundary="absorbing", two_materials=False)
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.absorbing_boundaries(model, MATERIALS, [1, 1], 1e3)

        size_x, size_y, size_z = model.nodes[:, 1:].max(axis=0) - model.nodes[:, 1:].min(axis=0)
        area = size_x * size_z + 2 * size_y * size_z + 2 * size_x * size_y
        _, _, vp, _ = wave_properties(MATERIALS["soft"])
        np.testing.assert_allclose(matrix.C.sum(), MATERIALS["soft"]["density"] * vp * area)

    def test_viscous_parameters_scale_perpendicular_dashpots(self):
        """The first viscous parameter scales the perpendicular dashpots linearly."""
        model = read_mesh("tetra4", boundary="absorbing")
        single = system_matrix.GenerateMatrix(model.number_eq, 2)
        single.absorbing_boundaries(model, MATERIALS, [1, 1], 1e3)
        double = system_matrix.GenerateMatrix(model.number_eq, 2)
        double.absorbing_boundaries(model, MATERIALS, [2, 1], 1e3)

        assert single.C.nnz > 0
        np.testing.assert_allclose(double.C.toarray(), 2 * single.C.toarray())

    def test_no_absorbing_boundary(self):
        """Without absorbing boundary conditions the matrices are left unchanged."""
        model = read_mesh("tri3")
        matrix = system_matrix.GenerateMatrix(model.number_eq, 2)
        matrix.generate_stiffness_and_mass(model, MATERIALS)
        K = matrix.K.copy()
        matrix.absorbing_boundaries(model, MATERIALS, [1, 1], 1e3)

        assert matrix.C.nnz == 0
        np.testing.assert_allclose(matrix.K.toarray(), K.toarray())


class TestAddRoseMatrix:
    """Coupling of scatter and rose matrices."""

    def test_coupled_matrix(self):
        """Contact dofs merge into scatter, other rose dofs are appended and contact couplings keep their diagonal."""
        scatter_matrix = lil_matrix(np.array([[1., 2., 0.], [2., 3., 4.], [0., 4., 5.]]))
        rose = np.array([[10., 11., 12.], [11., 13., 14.], [12., 14., 15.]])

        # rose dof 1 sits on scatter dof 2 and rose dof 2 on scatter dof 0; rose dof 0 becomes coupled dof 3
        coupled = system_matrix.GenerateMatrix(1, 2).add_rose_matrix(scatter_matrix, rose, [2, 0], [1, 2])

        # the coupling between the two contact dofs (rose 1-2, value 14) is not transferred
        expected = np.array([[1. + 15., 2., 0., 12.], [2., 3., 4., 0.], [0., 4., 5. + 13., 11.], [12., 0., 11., 10.]])
        assert isinstance(coupled, csr_matrix)
        np.testing.assert_allclose(coupled.toarray(), expected)


class TestSparseAccumulator:
    """Pairwise summation of sparse matrices."""

    def test_sum(self):
        """The accumulated total equals the plain sum of the matrices."""
        matrices = [csr_matrix(sparse_random(30, 30, 0.1, random_state=seed)) for seed in range(11)]
        accumulator = system_matrix.SparseAccumulator((30, 30))
        for matrix in matrices:
            accumulator.add(matrix)
        np.testing.assert_allclose(accumulator.total().toarray(), sum(m.toarray() for m in matrices))

    def test_empty(self):
        """Without matrices the total is an empty matrix of the given shape."""
        total = system_matrix.SparseAccumulator((4, 5)).total()
        assert total.shape == (4, 5)
        assert total.nnz == 0


class TestMesh:
    """Node lookup, connectivities and boundaries of the mesh."""

    def test_node_rows_with_gaps(self):
        """Node numbers are mapped to rows even when the numbering has gaps and is unordered."""
        model = read_mesh("tetra4")
        model.nodes = np.array([[7, 0., 0., 0.], [3, 1., 0., 0.], [12, 0., 1., 0.]])
        np.testing.assert_array_equal(model.get_node_rows([[12, 7], [3, 3]]), [[2, 0], [1, 1]])

    @pytest.mark.parametrize("node_ids, missing", [([3, 5], [5]), ([3, 40], [40]), ([-1, 3], [-1])])
    def test_node_rows_missing(self, node_ids: List[int], missing: List[int]):
        """A node number that is not in the node list raises a ValueError."""
        model = read_mesh("tetra4")
        model.nodes = np.array([[7, 0., 0., 0.], [3, 1., 0., 0.], [12, 0., 1., 0.]])
        with pytest.raises(ValueError, match=re.escape(f"Nodes {missing} are not in the node list.")):
            model.get_node_rows(node_ids)

    @pytest.mark.parametrize("element_type", ["hexa8", "tetra10", "tri6"])
    def test_connectivities(self, element_type: str):
        """Equation numbers and BC types per element follow the node numbers of the element."""
        model = read_mesh(element_type)
        for index in range(0, len(model.elem), max(1, len(model.elem) // 25)):
            rows = [np.where(model.nodes[:, 0] == node)[0][0] for node in model.elem[index]]
            np.testing.assert_array_equal(model.eq_nb_elem[index], model.eq_nb_dof[rows].ravel())
            np.testing.assert_array_equal(model.type_BC_elem[index], model.type_BC[rows].ravel())

    def test_mesh_edges_of_cube(self):
        """The boundary of a 10x10x10 hexa8 cube consists of 600 quadrilateral faces."""
        model = read_mesh("hexa8")
        model.get_mesh_edges()
        assert model.boundary_elem.shape == (600, 4)

    @pytest.mark.parametrize("element_type, lower_element_type, nb_nodes", [("quad4", "line2", 2), ("tri3", "line2", 2),
                                                                            ("tri6", "line3", 3)])
    def test_2d_boundary_elements(self, element_type: str, lower_element_type: str, nb_nodes: int):
        """2D meshes have line boundary elements for the absorbing boundaries."""
        model = read_mesh(element_type)
        assert model.lower_element_type == lower_element_type
        assert model.nb_nodes_lower_elem == nb_nodes

    def test_2d_boundary_direction_is_normalised(self):
        """In 2D the perpendicular-direction flag is 0 or 1, also when the boundary line is longer than 1 m."""
        model = mesher.ReadMesh(os.path.join(MESH_FOLDER, MESHES["tri3"]))
        model.read_gmsh()
        model.read_bc({"bottom": ["02", [[0, 0, 0], [10, 0, 0]]]})

        is_bottom = np.isclose(model.nodes[:, 2], 0.)
        np.testing.assert_array_equal(model.BC_dir[is_bottom], [[0, 1]] * int(is_bottom.sum()))
        assert set(np.unique(model.BC_dir)) <= {0, 1}


class TestInputErrors:
    """Errors on invalid input."""

    def test_random_props_must_be_a_dictionary(self, tmp_path: Any):
        """random_props=True, without random field properties, raises a TypeError."""
        settings = {"int_order": 2, "absorbing_BC": [1, 1], "absorbing_BC_stiff": 1e3}
        load = {"type": "heaviside", "force": [0, -1000, 0], "node": [3], "time": 0.1}
        with pytest.raises(TypeError,
                           match=re.escape("random_props must be False or the random field properties "
                                           "per material name.")):
            scatter(os.path.join(MESH_FOLDER, MESHES["tetra4"]),
                    str(tmp_path),
                    MATERIALS, {},
                    settings,
                    load,
                    random_props=True)

    def test_unsupported_covariance_model(self, tmp_path: Any):
        """An unknown random field covariance model raises a ValueError."""
        model = read_mesh("tetra4")
        generator = random_fields.RF({}, MATERIALS, str(tmp_path), "tetra4")
        props = {"seed_number": 1, "aniso_x": 1, "aniso_z": 1, "theta": 1, "std_value": 1e6, "model_name": "Cubic"}
        with pytest.raises(ValueError, match=re.escape('model name: "Cubic" is not supported')):
            generator.generate_gstools_rf(model.nodes, model.elem, 3, props, 20e6)
