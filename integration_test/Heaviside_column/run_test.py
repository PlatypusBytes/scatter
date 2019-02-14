import sys
sys.path.append("../../src")
import scatter

# computational settings
sett = {"gamma": 0.5,
        "beta": 0.25,
        "int_order": 2,
        "damping": [1, 0.001, 50, 0.001]}
# boundary conditions
BC = {"bottom": ["010", [[0, 0, 0], [1, 0, 0], [0, 0, -1], [1, 0, -1]]],
      "left": ["100", [[0, 0, 0], [0, 0, -1], [0, 10, 0], [0, 10, -1]]],
      "right": ["100", [[1, 0, 0], [1, 0, -1], [1, 10, 0], [1, 10, -1]]],
      "front": ["001", [[0, 0, 0], [1, 0, 0], [0, 10, 0], [1, 10, 0]]],
      "back": ["001", [[0, 0, -1], [1, 0, -1], [0, 10, -1], [1, 10, -1]]],
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
