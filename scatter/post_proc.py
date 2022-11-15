import pickle
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import imageio


def read_pickle(file):
    # read pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def plot_surface(X, Z, coords, data_plane, time, t, label, folder):
    # create Y variable
    Y = np.zeros(X.shape)
    # make plot
    fig = plt.figure(1, figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')

    for i, c in enumerate(coords):
        idx = np.where((X == c[0]) & (Z == c[2]))

        Y[idx[0][0], idx[1][0]] = data_plane[i][t]

    # Create a light source object for light from
    light = LightSource(90, 45)
    illuminated_surface = light.shade(Y, cmap=plt.cm.gray)
    ax.plot_surface(X, Z, Y, rstride=1, cstride=1, linewidth=0,
                    facecolors=illuminated_surface)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Customize the z axis.
    # ax.set_zlim((np.min(data_plane), np.max(data_plane)))
    ax.set_zlim((-0.08, 0.04))
    ax.set_zlim((-0.1, 0.1))
    ax.text2D(0.80, 0.95, "time = " + str(round(time[t], len(str(len(time))))) + " s", transform=ax.transAxes)
    ax.set_xlabel("Distance X [m]")
    ax.set_ylabel("Distance Z [m]")
    ax.set_zlabel(r"Vertical velocity [m/s]")
    # ax.set_zlabel(f"{label}")
    ax.view_init(35, -135)
    # ax.set_zticks([])
    plt.savefig(os.path.join(folder, str(t).zfill(len(str(len(time)))) + ".png"))
    # plt.savefig(os.path.join(folder, str(t).zfill(len(str(len(time)))) + ".pdf"))
    plt.close()
    return


def make_movie(data_location, y_ref, elem_size, dimension_x, dimension_z, data_val, t_max=False,
               output_file="movie.gif", temp_folder="./tmp"):
    # only works for square bricks with dimension and elem_size

    # read pickle file from FEM
    data = read_pickle(data_location)

    # if t_max not defined
    if not t_max:
        t_max = np.max(data["time"])

    # define mesh model
    x = np.linspace(0, dimension_x, abs(int(dimension_x / elem_size)) + 1)
    z = np.linspace(0, dimension_z, abs(int(dimension_z / elem_size)) + 1)

    # get coordinates and nodes on the y-plane
    nodes = []
    coords = []
    for i, val in enumerate(data["position"]):
        if val[1] == y_ref:
            nodes.append(str(i + 1))
            coords.append([round(j, 2) for j in val])

    # read data_plane
    data_plane = [data[data_val][i]["y"] for i in nodes]

    # create tmp folder
    if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder)

    # mesh grid
    X, Z = np.meshgrid(x, z)

    # create multiple plots
    for t in range(0, int(np.where(data["time"] >= t_max)[0][0])):
        plot_surface(X, Z, coords, data_plane, data["time"], t, data_val, temp_folder)

    # make de video
    filenames = os.listdir(temp_folder)
    with imageio.get_writer(output_file, mode='I', fps=30) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(temp_folder, filename))
            writer.append_data(image)
    writer.close()

    shutil.rmtree(temp_folder)

    return


if __name__ == "__main__":
    make_movie(r"./data.pickle", 10, 0.5, 20, "velocity", t_max=False,
               output_file=r"./output.gif", temp_folder=r"./tmp")
