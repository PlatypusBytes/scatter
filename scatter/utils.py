import os
from typing import Tuple, Union
import numpy as np
import h5py


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
    - matrix properties (M, C, K) (diagonal terms)
    - force (x, y, z)

    Edge features:
    - distance
    - matrix properties (M, C, K) (non-diagonal terms)


    k_till K_till_inv

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
    h5_path = os.path.join(output_folder, "data.h5")

    # Mesh & DOF structure
    nodes_id = model.nodes[:, 0].astype(int)
    coords = model.nodes[:, 1:]
    n_nodes, dim = coords.shape

    dof_map = model.eq_nb_dof.copy()
    dof_mask = ~np.isnan(dof_map)
    dof_map = np.where(dof_mask, dof_map, -1).astype(int)
    dof_per_node = dof_map.shape[1]

    # BC type per DOF (0=free, 1=fixed, 2=absorbing)
    bc_type = np.zeros_like(dof_map)
    for i in range(n_nodes):
        for d in range(dof_per_node):
            if not dof_mask[i, d]:
                bc_type[i, d] = 1
            else:
                bc_type[i, d] = model.BC[dof_map[i, d]]

    # Connectivity → graph edges
    elem_nodes = model.elem.astype(int)
    node_id_to_idx = {nid: i for i, nid in enumerate(nodes_id)}
    elem_idx = np.vectorize(node_id_to_idx.get)(elem_nodes)

    # connectivity list only valid for hex8 ToDo: add this as a property of element types
    connectivities_idx_nodes = {0: [1, 3, 4],
                                1: [0, 2, 5],
                                2: [1, 3, 6],
                                3: [0, 2, 7],
                                4: [0, 5, 7],
                                5: [1, 4, 6],
                                6: [2, 5, 7],
                                7: [3, 4, 6]
                                }

    edges = set()
    for elem in elem_idx:
        for li, ni in enumerate(elem):
            for lj in connectivities_idx_nodes[li]:
                nj = elem[lj]
                edges.add((ni, nj))
                edges.add((nj, ni))

    edge_index = np.array(list(edges), dtype=int).T
    src, dst = edge_index

    dx = coords[dst] - coords[src]
    dist = np.linalg.norm(dx, axis=1, keepdims=True)
    dir_vec = dx / np.maximum(dist, 1e-12)

    # Node features (diagonal terms)
    Mii = np.zeros((n_nodes, dof_per_node))
    Cii = np.zeros_like(Mii)
    Kii = np.zeros_like(Mii)

    for i in range(n_nodes):
        for d in range(dof_per_node):
            gdof = dof_map[i, d]
            if gdof >= 0:
                Mii[i, d] = matrix.M[gdof, gdof]
                Cii[i, d] = matrix.C[gdof, gdof]
                Kii[i, d] = matrix.K[gdof, gdof]

    # Edge coupling features (aggregated)
    def coupling_norm(mat, i, j):
        vals = []
        for di in range(dof_per_node):
            for dj in range(dof_per_node):
                gi = dof_map[i, di]
                gj = dof_map[j, dj]
                if gi >= 0 and gj >= 0:
                    vals.append(mat[gi, gj])
        if not vals:
            return 0.0
        return float(np.linalg.norm(vals))

    Kij = np.array([coupling_norm(matrix.K, i, j) for i, j in zip(src, dst)])[:, None]
    Mij = np.array([coupling_norm(matrix.M, i, j) for i, j in zip(src, dst)])[:, None]
    Cij = np.array([coupling_norm(matrix.C, i, j) for i, j in zip(src, dst)])[:, None]

    # Time series (node-wise)
    time = np.array(F.time)
    out_idx = np.arange(0, len(time), results.output_interval)

    def reshape_series(arr):
        out = np.zeros((len(out_idx), n_nodes, dof_per_node))
        for t_i, t in enumerate(out_idx):
            for i in range(n_nodes):
                for d in range(dof_per_node):
                    gdof = dof_map[i, d]
                    if gdof >= 0:
                        out[t_i, i, d] = arr[t, gdof]
        return out

    u = reshape_series(results.u)
    v = reshape_series(results.v)
    a = reshape_series(results.a)

    force = np.zeros_like(u)
    for ti, t in enumerate(out_idx):
        f_glob = F.update_load_at_t(time[t])
        for i in range(n_nodes):
            for d in range(dof_per_node):
                gdof = dof_map[i, d]
                if gdof >= 0:
                    force[ti, i, d] = f_glob[gdof]

    # Sparse matrices
    def save_sparse(group, name, mat):
        rows, cols = mat.nonzero()
        grp = group.create_group(name)
        grp.create_dataset("row", data=rows)
        grp.create_dataset("col", data=cols)
        grp.create_dataset("val", data=mat[rows, cols])
        grp.create_dataset("shape", data=mat.shape)

    # Write HDF5
    with h5py.File(h5_path, "w") as h5:
        mesh = h5.create_group("mesh")
        mesh.create_dataset("coords", data=coords)
        mesh.create_dataset("dof_map", data=dof_map)
        mesh.create_dataset("dof_mask", data=dof_mask)
        mesh.create_dataset("bc_type", data=bc_type)
        mesh.create_dataset("edge_index", data=edge_index)

        static = h5.create_group("static")
        static.create_dataset("Mii", data=Mii)
        static.create_dataset("Cii", data=Cii)
        static.create_dataset("Kii", data=Kii)
        static.create_dataset("edge_dist", data=dist)
        static.create_dataset("edge_dir", data=dir_vec)
        static.create_dataset("Kij", data=Kij)
        static.create_dataset("Mij", data=Mij)
        static.create_dataset("Cij", data=Cij)

        series = h5.create_group("series")
        series.create_dataset("time", data=time[out_idx])
        series.create_dataset("u", data=u)
        series.create_dataset("v", data=v)
        series.create_dataset("a", data=a)
        series.create_dataset("force", data=force)

        meta = h5.create_group("meta")
        meta.create_dataset("dt", data=results.dt)
        meta.create_dataset("beta", data=results.beta)
        meta.create_dataset("gamma", data=results.gamma)

        mats = h5.create_group("matrices")
        save_sparse(mats, "M", matrix.M)
        save_sparse(mats, "K", matrix.K)
        save_sparse(mats, "C", matrix.C)
