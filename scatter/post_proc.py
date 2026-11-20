"""Plots and movies of the results."""
import os
import pickle
import shutil
from typing import Any, List, Union

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LightSource


def read_pickle(file: str) -> Any:
    """Read a pickle file.

    Args:
        file: pickle file.

    Returns:
        Content of the pickle file.
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def plot_surface(X: npt.NDArray[np.float64], Z: npt.NDArray[np.float64], coords: List[List[float]],
                 data_plane: List[npt.NDArray[np.float64]], time: npt.NDArray[np.float64], t: int, label: str,
                 folder: str) -> None:
    """Plot the result on a horizontal plane at one time step, and save it as png.

    Args:
        X: x coordinates of the grid [m].
        Z: z coordinates of the grid [m].
        coords: coordinates of the nodes on the plane [m].
        data_plane: result of each node on the plane, at each time step.
        time: time of each step [s].
        t: time index.
        label: name of the result.
        folder: folder where the png is saved.
    """
    Y = np.zeros(X.shape)
    fig = plt.figure(1, figsize=(6, 5))
    ax: Any = fig.add_subplot(projection='3d')

    for i, c in enumerate(coords):
        idx = np.where((X == c[0]) & (Z == c[2]))
        Y[idx[0][0], idx[1][0]] = data_plane[i][t]

    # light source from 90 degrees azimuth and 45 degrees altitude
    light = LightSource(90, 45)
    illuminated_surface = light.shade(Y, cmap=plt.get_cmap("gray"))
    ax.plot_surface(X, Z, Y, rstride=1, cstride=1, linewidth=0, facecolors=illuminated_surface)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.set_zlim((-0.1, 0.1))
    ax.text2D(0.80, 0.95, "time = " + str(round(time[t], len(str(len(time))))) + " s", transform=ax.transAxes)
    ax.set_xlabel("Distance X [m]")
    ax.set_ylabel("Distance Z [m]")
    ax.set_zlabel(r"Vertical velocity [m/s]")
    ax.view_init(35, -135)
    plt.savefig(os.path.join(folder, str(t).zfill(len(str(len(time)))) + ".png"))
    plt.close()


def make_movie(data_location: str,
               y_ref: float,
               elem_size: float,
               dimension_x: float,
               dimension_z: float,
               data_val: str,
               t_max: Union[bool, float] = False,
               output_file: str = "movie.gif",
               temp_folder: str = "./tmp") -> None:
    """Make a movie of a result on a horizontal plane; only works for box meshes with square elements.

    Args:
        data_location: pickle file with the results.
        y_ref: y coordinate of the plane [m].
        elem_size: element size [m].
        dimension_x: size of the box in x [m].
        dimension_z: size of the box in z [m].
        data_val: name of the result: displacement, velocity or acceleration.
        t_max: last time of the movie [s], or False for the end of the calculation.
        output_file: gif file of the movie.
        temp_folder: folder for the frames, removed afterwards.
    """
    data = read_pickle(data_location)
    if not t_max:
        t_max = np.max(data["time"])

    # grid of the plane
    x = np.linspace(0, dimension_x, abs(int(dimension_x / elem_size)) + 1)
    z = np.linspace(0, dimension_z, abs(int(dimension_z / elem_size)) + 1)

    # nodes and coordinates on the plane
    nodes = []
    coords = []
    for i, val in enumerate(data["position"]):
        if val[1] == y_ref:
            nodes.append(str(i + 1))
            coords.append([round(j, 2) for j in val])

    data_plane = [data[data_val][i]["y"] for i in nodes]

    if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder)

    X, Z = np.meshgrid(x, z)
    for t in range(0, int(np.where(data["time"] >= t_max)[0][0])):
        plot_surface(X, Z, coords, data_plane, data["time"], t, data_val, temp_folder)

    # frames are named by their zero-padded time index, so sorting puts them in time order; 30 frames per second
    filenames = sorted(os.listdir(temp_folder))
    # bound before the with statement: imageio's stubs type the result of __enter__ as a writer without append_data
    writer = imageio.get_writer(output_file, mode='I', duration=1000 / 30)
    with writer:
        for filename in filenames:
            writer.append_data(imageio.imread(os.path.join(temp_folder, filename)))

    shutil.rmtree(temp_folder)
