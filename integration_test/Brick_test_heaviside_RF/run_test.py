import sys
sys.path.append("../../src")
import scatter

# computational settings
sett = {"gamma": 0.5,
        "beta": 0.25,
        "int_order": 2,
        "damping": [1, 0.01, 30, 0.01]}

# boundary conditions
x = 15
y = 10
z = -15
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
        "node": [4],
        "time": 0.15,
        "type": "heaviside"}  # pulse or heaviside

# random field properties
RF_props = {"number_realisations": 1,
            "element_size": 0.5,
            "theta": 2,
            "seed_number": -26021981,
            "material": mat["solid"],
            "index_material": 1,
            "std_value": 0.5,
            }

# run scatter
scatter.scatter(r"./brick.msh", "./results", mat, BC, sett, load, time_step=1e-3, random_props=RF_props)