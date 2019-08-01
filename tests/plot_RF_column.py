import matplotlib.pylab as plt
import sys
sys.path.append(r"..\src")
import random_field_example as rf
import numpy as np
import matplotlib.pyplot as plt

n = 1  # number of realisations in one set
max_lvl = 8  # number of levels of subdivision (2**max_lvl) is size.
cellsize = 0.1  # Cell size
theta = 10
xcells = 1  # Number of cells x dir
ycells = 200  # Number of cells x dir
zcells = 1  # Number of cells x dir
seed = 12057
mean = 30000000.0
sd = 6000000.0
lognormal = True
fieldfromcentre = False
# anisox = 1
# anisoy = 1

# everytime you call this you need a new seed number.
fields = rf.rand3d(n, max_lvl, cellsize, theta, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre)
# make plot
fig = plt.figure(1, figsize=(3, 10))
plt.axes().set_position([0.15, 0.1, 0.8, 0.8])
# ax = fig.gca(projection='3d')

plt.ylabel("Distance Y [m]", fontsize=14)
plt.axes().get_xaxis().set_visible(False)
plt.imshow(fields[0][:, :, 0].transpose(), extent=[0, 1, 0, 20], vmin=1e7, vmax=5e7)
plt.colorbar()
# plt.savefig("./images/RF_column.pdf")
# plt.savefig("./images/RF_column.png")
# plt.close()
plt.tight_layout()
plt.show()


# everytime you call this you need a new seed number.
fields2 = rf.rand3d(n, max_lvl, cellsize, 2, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre)

# everytime you call this you need a new seed number.
fields10 = rf.rand3d(n, max_lvl, cellsize, 10, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre)

# everytime you call this you need a new seed number.
fields20 = rf.rand3d(n, max_lvl, cellsize, 20, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre)

plt.plot(np.linspace(0, 20, len(fields2[0][0,:,0])), fields2[0][0,:,0], label="2")
plt.plot(np.linspace(0, 20, len(fields2[0][0,:,0])), fields10[0][0,:,0], label="10", marker="^")
plt.plot(np.linspace(0, 20, len(fields2[0][0,:,0])), fields20[0][0,:,0], label="20", marker='x')
plt.legend()
plt.show()