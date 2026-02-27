import os
from typing import Tuple, Union
import numpy as np
import h5py
from scipy import sparse


def calculate_distance(p1: list, p2: list) -> float:
    r"""
    Calculates the distance between two points or array of points
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    if p1.size <= 3:
        p1 = p1[None, :]
    if p2.size <= 3:
        p2 = p2[None, :]
    dist = np.linalg.norm(p1-p2, axis=1)
    return dist


def calculate_centroid(coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate centroid of numpy array
    :param coordinates: numpy array of coordinates in 3D
    :return: centroid
    """
    length = coordinates.shape[0]
    sum_x = np.sum(coordinates[:, 0])
    sum_y = np.sum(coordinates[:, 1])
    sum_z = np.sum(coordinates[:, 2])
    return sum_x / length, sum_y / length, sum_z / length


def define_plane(p1: list, p2: list, p3: list) -> Union[list, np.ndarray]:
    r"""
    Finds all the nodes that are within the plane containing the points p1, p2 and p3.
    Assumes that the three points are non-collinear

    Parameters
    ----------
    :param p1: coordinate point p1
    :param p2: coordinate point p2
    :param p3: coordinate point p3
    :return: 4 nodes that are in the plane; normal vector
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    return [a, b, c, d], np.abs(cp / np.linalg.norm(cp))


def search_idx(data: list, string1: str, string2: str) -> Union[list, int]:
    """
    Search data for the text in between string1 and string2

    Parameters
    ----------
    :param data: list with text
    :param string1: initial string
    :param string2: final string
    :return: text in between strings, indexes
    """
    # search string1
    idx = [i for i, val in enumerate(data) if val.startswith(string1)][0]
    nb = int(data[idx + 1])

    # search string2
    idx_end_nodes = [i for i, val in enumerate(data) if val.startswith(string2)][0]

    res = []
    for i in range(idx + 2, idx_end_nodes):
        aux = []
        for j in data[i].split():
            try:
                aux.append(float(j))
            except ValueError:
                aux.append(str(j.replace('"', '')))
        res.append(aux)

    return res, nb


def area_polygon(coords: list) -> float:
    """
    Compute area of 3D planar polygon with one common axis (transforms it into a 2D)
    It sorts the points clockwise

    Parameters
    ----------
    :param coords: list with coordinates
    :return: area
    """

    # find common axis in coords
    xyz = np.array(coords)
    # index that it is common
    try:
        idx_xy = np.where((xyz == xyz[0, :]).all(0))[0][0]
    except IndexError:
        # ToDo: improve this assumption. related to the other todo.
        idx_xy = 1

    xy = [np.delete(i, idx_xy) for i in xyz]

    # determine centroid
    centroid = np.mean(xy, axis=0)

    # compute angle between all points and centroid using origin
    angle = []
    for i in range(len(xy)):
        angle.append(np.arctan2(xy[i][1] - centroid[1], xy[i][0] - centroid[0]))

    # reorganise coordinates clock-wise
    coords = np.array(coords)[np.argsort(angle)]

    # These two vectors are in the plane
    vec1 = coords[1] - coords[0]
    vec2 = coords[2] - coords[0]

    # the cross product is a vector normal to the plane
    n = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))

    # compute area
    area = 0
    for i in range(len(coords) - 1):
        area += np.dot(n, np.cross(coords[i], coords[i + 1]))

    return area / 2


def clockwise_sort_2D_elements(points: np.ndarray) -> np.ndarray:
    """
    Sorts a list of 2D coordinates clockwise, following the gmsh node numbering convention

    Parameters
    ----------
    :param points: list of 2D coordinates
    :return: clockwise sorted list of 2D coordinates following gmsh node numbering convention
    """
    # find the reference point (the point with the lowest y-coordinate)
    ref_point = min(points, key=lambda p: p[1])

    # define a custom key function that computes the angle of each point with respect to the reference point
    def angle_key(point):
        x, y = point[0] - ref_point[0], point[1] - ref_point[1]
        return np.arctan2(y, x)

    # sort the points by angle
    sorted_points = sorted(points, key=angle_key)

    # after points being sorted, find corners of the element

    corner = [sorted_points[0]]
    basic_idx = 0
    for i in range(len(sorted_points) - 1):
        if not are_collinear(sorted_points[basic_idx:i + 2]):
            corner.append(sorted_points[i])
            basic_idx = i

    set1 = set(map(tuple, sorted_points))
    set2 = set(map(tuple, corner))
    middle = list(map(list, set1.symmetric_difference(set2)))
    corner.extend(middle)

    return np.array(corner)


def are_collinear(coords: np.ndarray) -> bool:
    """
    Check if a list of coordinates are collinear

    Parameters
    ----------
    :param coords: list of coordinates
    :return: True if collinear, False otherwise
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    for i in range(2, len(coords)):
        x1, y1 = coords[i-1]
        x2, y2 = coords[i]
        new_slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        if new_slope != slope:
            return False
    return True


def generate_gnn_files(model: object, matrix: object, F: object, results, output_folder: str):
    """
    Generate files for training Graph Neural Networks


    Node features:
    - coordinates
    - BC
    - u, v, a (at time t)
    - Delta force (for time t + 1)
    - matrix properties (M, C, K) (diagonal terms)
    - force (x, y, z)

    Edge features:
    - distance
    - matrix properties (M, C, K) (non-diagonal terms)

    Target:
    - Delta u (at time t + 1)

    Graph feature:
    - time
    - time-step

    Parameters
    ----------
    :param model: model object
    :param materials: dictionary with material properties
    :param matrix: matrix object
    :param inp_settings: dictionary with numerical settings
    :param loading: dictionary with loading conditions
    :param output_folder: location of the output folder
    """

    # ToDo: only works for hexa8 elements
    if model.element_type != "hexa8":
        raise ValueError("Only hexa8 elements currently supported")

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "data.hdf5")

    with h5py.File(output_file, "w") as f:
        mesh_group = f.create_group("mesh")

        mesh_group.create_dataset("nodes_id", data=model.nodes[:, 0], dtype='i', compression="gzip")
        mesh_group.create_dataset("coordinates", data=model.nodes[:, 1:], dtype='f', compression="gzip")
        mesh_group.create_dataset("BC", data=model.BC, dtype='f', compression="gzip")
        mesh_group.create_dataset("eq_nb_dof", data=model.eq_nb_dof, dtype='f', compression="gzip")

        mat_group = f.create_group("matrices")

        for name, mat in [("stiffness", matrix.K), ("mass", matrix.M), ("damping", matrix.C)]:
            grp = mat_group.create_group(name)
            mat = sparse.csr_matrix(mat)
            grp.create_dataset("data", data=mat.data, compression="gzip")
            grp.create_dataset("indices", data=mat.indices, compression="gzip")
            grp.create_dataset("indptr", data=mat.indptr, compression="gzip")
            grp.attrs["shape"] = mat.shape

        time_group = f.create_group("time_results")
        time_group.create_dataset("time", data=results.time, compression="gzip")
        time_group.create_dataset("displacement", data=results.u, compression="gzip")
        time_group.create_dataset("velocity", data=results.v, compression="gzip")
        time_group.create_dataset("acceleration", data=results.a, compression="gzip")

        force_group = f.create_group("force_results")
        force = []
        for ti, _ in enumerate(results.time):
            force.append(F.update_load_at_t(ti))
        force_group.create_dataset("force", data=force, compression="gzip")
