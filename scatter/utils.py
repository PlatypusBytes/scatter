import os
import pickle
from typing import Tuple, Union
import numpy as np


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
    Generate files for Graph Neural Networks


    Node features:
    - coordinates
    - BC
    - matrix properties (M, C, K) (diagonal terms)
    - time
    - force

    Edge features:
    - distance
    - matrix properties (M, C, K) (non-diagonal terms)



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
        raise ValueError("Only hexa8 elements are supported for GNNs")

    os.makedirs(output_folder, exist_ok=True)

    # connectivity list only valid for hex8 ToDo: add this as a property of element types
    connectivities_nodes = {0: [1, 3, 4],
                            1: [0, 2, 5],
                            2: [1, 3, 6],
                            3: [0, 2, 7],
                            4: [0, 5, 7],
                            5: [1, 4, 6],
                            6: [2, 5, 7],
                            7: [3, 4, 6]
                            }

    data = {}
    data["nb_nodes"] = len(model.nodes)
    data["nodes"] = model.nodes[:, 0]
    data["nodes_coordinates"] = model.nodes[:, 1:]
    data["nb_elements"] = len(model.elem)
    data["element_type"] = model.element_type
    data["BC"] = model.BC
    data["connectivities"] = []

    data["node_features"] = []
    data["edge_features"] = []
    data["results"] = []

    # find connectivities for each node
    for n in data["nodes"]:
        aux = []
        for i, k in enumerate(model.elem):
            idx = np.where(k == n)[0]
            if len(idx) > 0:
                aux.extend(k[connectivities_nodes[int(idx)]])
        data["connectivities"].append(sorted(set(aux)))


    # generate the node and edge features (for the mesh and matrices)
    for idx_n, _ in enumerate(data["nodes"]):

        idx_dofs = [i.astype(int) for i in model.eq_nb_dof[idx_n] if not np.isnan(i)]

        if len(idx_dofs) == 0:
            continue

        for idx_dof in idx_dofs:
            # node features: coordinates, BC, matrix
            node_features = [model.nodes[idx_dof, 1:],
                             model.BC[idx_dof]]
            node_features = np.array(node_features).flatten().tolist()

            node_features.append(matrix.M[idx_dof.astype(int), idx_dof.astype(int)])
            node_features.append(matrix.C[idx_dof.astype(int), idx_dof.astype(int)])
            node_features.append(matrix.K[idx_dof.astype(int), idx_dof.astype(int)])

            data["node_features"].append(node_features)

        # find connectivities for each node
        node_connect = data["connectivities"][idx_n]
        idx_connect = [model.nodes[:,0].tolist().index(c) for c in node_connect]
        idx_connections = []
        for i in idx_connect:
            for j in model.eq_nb_dof[i]:
                if not np.isnan(j):
                    idx_connections.append(j.astype(int))

        # distance, matrix
        aux_edge_features = []
        for i in idx_connections:
            distance = np.linalg.norm(model.nodes[idx_n, 1:] - model.nodes[i, 1:])
            edge_features = [distance]

            # for each node dof
            for j in idx_dofs:
                edge_features.append(matrix.M[i, j.astype(int)])
                edge_features.append(matrix.C[i, j.astype(int)])
                edge_features.append(matrix.K[i, j.astype(int)])
            aux_edge_features.append(edge_features)
        data["edge_features"].append(aux_edge_features)


    # update the node features with the force
    for t in range(0, len(F.time), results.output_interval):
        force = F.update_load_at_t(t)

        if t == 0:
            for i, f in enumerate(force):
                data["node_features"][i].append(F.time[t])
                data["node_features"][i].append(f)
                data["results"].append([results.u[t, :], results.v[t, :], results.a[t, :]])
        else:
            for i, f in enumerate(force):
                data["node_features"][i][-1] = F.time[t]
                data["node_features"][i][-1] = f
                data["results"][i][-1] = [results.u[t, :], results.v[t, :], results.a[t, :]]

        # save the data for each time step
        with open(f"{output_folder}/data_{t}.pickle", "wb") as f:
            pickle.dump({"node_features": data["node_features"],
                         "edge_features": data["edge_features"],
                         "results": data["results"]},
                         f)

