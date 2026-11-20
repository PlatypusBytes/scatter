"""Geometric helpers and gmsh file parsing."""
from typing import List, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt


def calculate_distance(p1: npt.ArrayLike, p2: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Calculate the distance between two points, or between each point of two arrays of points.

    Args:
        p1: one point, or an array of points with shape (point, direction) [m].
        p2: one point, or an array of points with shape (point, direction) [m].

    Returns:
        Distance between the points [m].
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    if p1.size <= 3:
        p1 = p1[None, :]
    if p2.size <= 3:
        p2 = p2[None, :]
    return np.linalg.norm(p1 - p2, axis=1)


def calculate_centroid(coordinates: npt.NDArray[np.float64]) -> Tuple[float, float, float]:
    """Calculate the centroid of a set of 3D points.

    Args:
        coordinates: coordinates of the points, with shape (point, direction) [m].

    Returns:
        x, y and z coordinates of the centroid [m].
    """
    length = coordinates.shape[0]
    sum_x = np.sum(coordinates[:, 0])
    sum_y = np.sum(coordinates[:, 1])
    sum_z = np.sum(coordinates[:, 2])
    return float(sum_x / length), float(sum_y / length), float(sum_z / length)


def define_plane(p1: Sequence[float], p2: Sequence[float],
                 p3: Sequence[float]) -> Tuple[List[float], npt.NDArray[np.float64]]:
    """Define the plane that contains three non-collinear points.

    Args:
        p1: coordinates of the first point [m].
        p2: coordinates of the second point [m].
        p3: coordinates of the third point [m].

    Returns:
        Coefficients a, b, c and d of the plane a * x + b * y + c * z = d, and the absolute value of its unit normal.
    """
    point_1 = np.array(p1)
    point_2 = np.array(p2)
    point_3 = np.array(p3)

    # these two vectors are in the plane, so their cross product is normal to it
    cp = np.cross(point_3 - point_1, point_2 - point_1)
    a, b, c = cp
    d = np.dot(cp, point_3)

    return [float(a), float(b), float(c), float(d)], np.abs(cp / np.linalg.norm(cp))


def search_idx(data: List[str], string1: str, string2: str) -> Tuple[List[List[Union[float, str]]], int]:
    """Parse the lines of a gmsh section, between the line that starts with string1 and the one with string2.

    Args:
        data: lines of the gmsh file.
        string1: start of the opening line of the section.
        string2: start of the closing line of the section.

    Returns:
        Entries of each line of the section, as floats where possible and as strings otherwise, and the number of
        entries given on the line after the opening line.
    """
    idx = [i for i, val in enumerate(data) if val.startswith(string1)][0]
    nb = int(data[idx + 1])
    idx_end = [i for i, val in enumerate(data) if val.startswith(string2)][0]

    res = []
    for i in range(idx + 2, idx_end):
        aux: List[Union[float, str]] = []
        for j in data[i].split():
            try:
                aux.append(float(j))
            except ValueError:
                aux.append(str(j.replace('"', '')))
        res.append(aux)

    return res, nb


def area_polygon(coords: npt.ArrayLike) -> float:
    """Compute the area of a planar 3D polygon that is perpendicular to one of the axes.

    The points are sorted around their centroid before the area is computed.

    Args:
        coords: coordinates of the corner points of the polygon, with shape (point, direction) [m].

    Returns:
        Area of the polygon [m^2].
    """
    xyz = np.array(coords)
    # index of the axis along which all points have the same coordinate
    try:
        idx_xy = np.where((xyz == xyz[0, :]).all(0))[0][0]
    except IndexError:
        # ToDo: improve this assumption. related to the other todo.
        idx_xy = 1

    xy = np.delete(xyz, idx_xy, axis=1)
    centroid = np.mean(xy, axis=0)

    # sort the points by their angle around the centroid
    angle = np.arctan2(xy[:, 1] - centroid[1], xy[:, 0] - centroid[0])
    sorted_coords = xyz[np.argsort(angle)]

    # unit normal of the plane
    normal = np.cross(sorted_coords[1] - sorted_coords[0], sorted_coords[2] - sorted_coords[0])
    normal = normal / np.linalg.norm(normal)

    area = 0.
    for i in range(len(sorted_coords) - 1):
        area += float(np.dot(normal, np.cross(sorted_coords[i], sorted_coords[i + 1])))

    return area / 2


def order_face_nodes(coordinates: npt.NDArray[np.float64], is_corner: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
    """Find the gmsh node order of boundary faces: corners in cyclic order, then the midside node of each edge.

    Edge k of a face runs from corner k to corner k + 1. A line has one edge, from corner 0 to corner 1.

    Args:
        coordinates: coordinates of the nodes of each face, in any order, with shape (face, node, direction) [m].
        is_corner: whether each node is a corner node, with shape (face, node).

    Returns:
        Position of each node in `coordinates` in gmsh order, with shape (face, node).

    Raises:
        ValueError: if the faces do not all have the same number of corner nodes.
        ValueError: if the number of midside nodes does not match the number of edges.
        ValueError: if the midside nodes cannot be matched one to one with the edges.
    """
    nb_faces, nb_nodes = is_corner.shape
    nb_corners = int(is_corner[0].sum()) if nb_faces > 0 else 0
    if np.any(is_corner.sum(axis=1) != nb_corners):
        raise ValueError("All faces must have the same number of corner nodes.")

    face = np.arange(nb_faces)[:, None]
    corners = np.nonzero(is_corner)[1].reshape(nb_faces, nb_corners)
    midsides = np.nonzero(~is_corner)[1].reshape(nb_faces, nb_nodes - nb_corners)

    if nb_corners > 2:
        # sort the corners by their angle around the centroid, in the best-fit plane of the face
        centred = coordinates[face, corners] - coordinates[face, corners].mean(axis=1, keepdims=True)
        plane_directions = np.linalg.svd(centred)[2][:, :2, :]
        in_plane = np.einsum("fcd,fkd->fck", centred, plane_directions)
        angle = np.arctan2(in_plane[..., 1], in_plane[..., 0])
        corners = np.take_along_axis(corners, np.argsort(angle, axis=1), axis=1)
        nb_edges = nb_corners
    else:
        nb_edges = 1

    nb_midsides = nb_nodes - nb_corners
    if nb_midsides == 0:
        return corners
    if nb_midsides != nb_edges:
        raise ValueError(f"A face with {nb_corners} corner nodes must have 0 or {nb_edges} midside nodes, "
                         f"not {nb_midsides}.")

    # the midside node of an edge is the one closest to the middle of its corners
    corner_xyz = coordinates[face, corners]
    edge_middle = ((corner_xyz + np.roll(corner_xyz, -1, axis=1)) / 2)[:, :nb_edges]
    distance = np.linalg.norm(edge_middle[:, :, None, :] - coordinates[face, midsides][:, None, :, :], axis=-1)
    nearest = np.argmin(distance, axis=2)
    if np.any(np.sort(nearest, axis=1) != np.arange(nb_midsides)):
        raise ValueError("The midside nodes of a face cannot be matched to its edges.")

    return np.concatenate((corners, np.take_along_axis(midsides, nearest, axis=1)), axis=1)
