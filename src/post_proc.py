import pickle
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
import imageio


# tol = 0.05

def read_pickle(file):
    # read pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def plot_surface(X, Z, coords, data_plane, time, t, folder):
    # create Y variable
    Y = np.zeros(X.shape)
    # make plot
    fig = plt.figure(1, figsize=(6, 5))
    ax = fig.gca(projection='3d')

    # for i, c in enumerate(coords):
    #     idx = np.where((np.abs(X - c[0]) < tol) & (np.abs(Z == c[2]) < tol))

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
    ax.set_zlim((-0.00015, 0.00))
    # ax.set_zlim((-0.6, 0.1))
    ax.text2D(0.80, 0.95, "time = " + str(time) + " s", transform=ax.transAxes)
    ax.set_xlabel("Distance X [m]")
    ax.set_ylabel("Distance Z [m]")
    # ax.set_zlabel(r"Vertical displacement $\times 10 ^{-6}$ [m]")
    ax.set_zlabel(r"Vertical velocity [m/s]")
    ax.view_init(35, -135)
    # ax.view_init(90, 90)
    # ax.set_zticks([])
    plt.savefig(os.path.join(folder, str(t).zfill(3) + ".png"))
    # plt.savefig(os.path.join(folder, str(t).zfill(3) + ".pdf"))
    plt.close()
    return


def make_movie(data_location, y_ref, elem_size, dimension, t_max, data_val,
               output_file="movie.gif", temp_folder="./tmp"):
    # only works for square bricks with dimension and elem_size

    # read pickle file from FEM
    data = read_pickle(data_location)

    # define mesh model
    x = np.linspace(0, dimension, int(dimension / elem_size) + 1)
    z = np.linspace(0, -dimension, int(dimension / elem_size) + 1)

    # get coordinates and nodes on the y-plane
    nodes = []
    coords = []
    for i, val in enumerate(data["position"]):
        # if np.abs(val[1] - y_ref) < tol:
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
        plot_surface(X, Z, coords, data_plane, round(data["time"][t], 3), t, temp_folder)

    # make de video
    filenames = os.listdir(temp_folder)
    with imageio.get_writer(output_file, mode='I', fps=15) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(temp_folder, filename))
            writer.append_data(image)
    writer.close()

    # shutil.rmtree(temp_folder)

    return


if __name__ == "__main__":
    # make_movie(r"D:\brick\results_mean\data.pickle", 10, 1, 20, 0.5, output_file=r"D:\brick\mean.gif", temp_folder=r"D:\tmp")
    # make_movie(r"D:\brick\results_RF_1\run_0\data.pickle", 10, 1, 20, 0.5, output_file=r"D:\brick\rf.gif", temp_folder=r"D:\tmp")
    # make_movie(r"D:\brick\results_mn_mean\data.pickle", 10, 1, 20, 1, "velocity", output_file=r"D:\brick\mean_mv.gif", temp_folder=r"D:\tmp")
    # make_movie(r"D:\brick\results_mn_mean_stat\data.pickle", 10, 1, 20, 1, "velocity", output_file=r"D:\brick\mean_mv_stat.gif", temp_folder=r"D:\tmp")

    shutil.rmtree("../results_brick/tmp")
    make_movie(r"../results_brick/data.pickle", 5, 1, 10, 0.25, "velocity", output_file=r"../results_brick/moving.gif", temp_folder=r"../results_brick/tmp")
