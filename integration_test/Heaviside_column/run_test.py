import sys
sys.path.append("../../src")
import scatter

# computational settings
sett = {"gamma": 0.5,
        "beta": 0.25,
        "int_order": 2,
        "damping": [1, 0.001, 50, 0.001]}

# boundary conditions
x = 1
y = 10
z = -1
BC = {"bottom": ["010", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
      "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
      "right": ["100", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
      "front": ["001", [[0, 0, 0], [z, 0, 0], [0, y, 0], [x, y, 0]]],
      "back": ["001", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
      }

# material dictionary: rho, E, v
mat = {"solid": [1500, 30e6, 0.2],
       "bottom": [1800, 20e4, 0.15]}
load = {"force": [0, -1000, 0],
        "node": [3, 4, 7, 8],
        "time": 1,
        "type": "heaviside"}  # pulse or heaviside

# run scatter
scatter.scatter(r"./column.msh", "./results", mat, BC, sett, load, time_step=0.5e-3)
