"""
Generates the scatter runs that train the GNN in GNN/.

Every run samples its own geometry, mesh size, mean soil properties, random field of the
Young modulus and time step, seeded by the run index. The runs are independent, so on a cluster
they can be split over an array job:

    python run_scatter_gnn.py --first 0 --count 50 --output GNN
"""
import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import create_geo_file
from scatter import mesher, random_fields
from scatter.scatter import scatter
from solvers.newmark_solver import NewmarkExplicit
from solvers.preconditioners import JacobiPreconditioner
from solvers.linear_equations_solvers import CGSolver

# ranges of the mean soil properties, they cover the materials of the other run scripts
DENSITY_RANGE = (1400, 2200)  # [kg/m^3]
YOUNG_RANGE = (5e6, 5e8)  # [Pa], sampled log-uniform
POISSON_RANGE = (0.15, 0.45)
# coefficient of variation of the random field of the Young modulus, sampled log-uniform
YOUNG_COV_RANGE = (0.05, 1.7)
# time step as a fraction of the critical time step of the run
TIME_STEP_FRACTION_RANGE = (0.1, 0.9)

# a larger mesh adds little for a local operator, but every training sample gets heavier
MAX_NODES = 300_000
# consecutive steps of a small time step are nearly identical, and the solver keeps all of them in
# memory (~70 bytes per step and equation)
MAX_STEPS = 5_000


def element_dimensions(x: float, y: float, z: float, element_size: float) -> Tuple[float, float, float]:
    """Element dimensions of the structured brick mesh written by create_geo_file.

    gmsh divides every side into a whole number of elements, so the elements are slightly smaller
    than the target size.

    Args:
        x: size of the brick along x [m].
        y: size of the brick along y [m].
        z: size of the brick along z [m].
        element_size: target element size [m].

    Returns:
        Element dimensions along x, y and z [m].
    """
    return tuple(length / max(1, math.ceil(length / element_size)) for length in (x, y, z))


def n_mesh_nodes(x: float, y: float, z: float, element_size: float) -> int:
    """Number of nodes of the structured brick mesh written by create_geo_file.

    Args:
        x: size of the brick along x [m].
        y: size of the brick along y [m].
        z: size of the brick along z [m].
        element_size: target element size [m].

    Returns:
        Number of nodes of the mesh.
    """
    return math.prod(round(length / size) + 1
                     for length, size in zip((x, y, z), element_dimensions(x, y, z, element_size)))


def critical_time_step(mesh_file: Path, mat: Dict, RF_props: Dict, element_size: float) -> float:
    """Critical time step of a run: smallest element dimension over the largest P-wave speed.

    The random field is generated here exactly as scatter generates it (same mesh, same seed), so
    the bound uses the Young modulus of the stiffest element of the actual model.

    Args:
        mesh_file: gmsh mesh of the run.
        mat: material dictionary of the run.
        RF_props: random field properties of the run.
        element_size: target element size of the mesh [m].

    Returns:
        Critical time step [s].
    """
    model = mesher.ReadMesh(mesh_file)
    model.read_gmsh()
    rf = random_fields.RF(RF_props, mat, str(mesh_file.parent), model.element_type)
    # same element selection as scatter, the field is evaluated at the element centroids
    material_idx = [m[1] for m in model.materials if m[2] == RF_props["material"]][0]
    rf.generate_gstools_rf(model.nodes, model.elem[model.materials_index == material_idx], model.dimension,
                           angles=0.0)

    props = mat[RF_props["material"]]
    nu = props["poisson"]
    constrained_modulus = np.max(rf.fields[0]) * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    p_wave_speed = np.sqrt(constrained_modulus / props["density"])
    x, y, z = (float(np.ptp(model.nodes[:, k])) for k in (1, 2, 3))
    return min(element_dimensions(x, y, z, element_size)) / p_wave_speed


def time_step(fraction: float, dt_critical: float, duration: float) -> float:
    """Time step of a run: a fraction of the critical time step, with at most MAX_STEPS steps.

    Args:
        fraction: sampled fraction of the critical time step.
        dt_critical: critical time step of the run [s].
        duration: duration of the analysis [s].

    Returns:
        Time step [s], at most the upper end of TIME_STEP_FRACTION_RANGE times dt_critical.
    """
    # a run that would exceed MAX_STEPS gets a larger fraction, but never above the range
    fraction = max(fraction, duration / (MAX_STEPS * dt_critical))
    return min(fraction, TIME_STEP_FRACTION_RANGE[1]) * dt_critical


def sample_run(i: int) -> Dict[str, float]:
    """Draw the parameters of run i.

    The draws only depend on i, so any run can be regenerated on its own.

    Args:
        i: index of the run.

    Returns:
        Geometry [m], element size [m], mean density [kg/m^3], mean Young modulus [Pa],
        Poisson ratio, random field parameters and time step fraction of the run.
    """
    rng = np.random.RandomState(i)
    # same draws, in the same order, as the first version of this script
    element_size = float(np.round(rng.uniform(0.2, 1), 2))
    model_size = int(rng.uniform(5, 20))
    model_depth = int(rng.uniform(5, 20))
    aniso_z = int(rng.uniform(1, 10))
    theta = float(rng.uniform(1, 10))

    young = float(np.exp(rng.uniform(*np.log(YOUNG_RANGE))))
    density = float(rng.uniform(*DENSITY_RANGE))
    poisson = float(rng.uniform(*POISSON_RANGE))
    young_cov = float(np.exp(rng.uniform(*np.log(YOUNG_COV_RANGE))))
    time_step_fraction = float(rng.uniform(*TIME_STEP_FRACTION_RANGE))

    x, y, z = model_size, model_depth, model_size
    while n_mesh_nodes(x, y, z, element_size) > MAX_NODES:
        element_size = round(element_size + 0.01, 2)

    return {"x": x, "y": y, "z": z, "element_size": element_size, "aniso_z": aniso_z, "theta": theta,
            "density": density, "young": young, "poisson": poisson, "young_cov": young_cov,
            "time_step_fraction": time_step_fraction}


def material(p: Dict[str, float]) -> Dict:
    """Material dictionary of a run.

    Args:
        p: parameters of the run, from sample_run.

    Returns:
        Material dictionary: rho, E, v.
    """
    return {"solid": {"density": p["density"],
                      "Young": p["young"],
                      "poisson": p["poisson"]},
            }


def random_field(i: int, p: Dict[str, float]) -> Dict:
    """Random field properties of a run.

    Args:
        i: index of the run, seeds the field.
        p: parameters of the run, from sample_run.

    Returns:
        Random field properties of the Young modulus.
    """
    return {"number_realisations": 1,
            "element_size": p["element_size"],
            "theta": p["theta"],
            "seed_number": i,
            "material": "solid",
            "key_material": "Young",
            # relative to the mean, an absolute value would give a different contrast in every run
            "std_value": p["young_cov"] * p["young"],
            "aniso_x": 1,
            "aniso_z": p["aniso_z"],
            "model_name": "Gaussian",
            }


def run(i: int, output: Path) -> None:
    """Run scatter for run i and write output/run_<i>/data.hdf5.

    Args:
        i: index of the run.
        output: folder that receives the run folders.
    """
    # computational settings
    sett = {"gamma": 0.5,
            "beta": 0.25,
            "int_order": 2,
            "damping": [1, 0.01, 80, 0.01],
            "absorbing_BC": [1, 1],
            "absorbing_BC_stiff": 1e3,
            "pickle": True,
            "pickle_nodes": "4",
            "VTK": False,
            "VTK_binary": True,
            "output_interval": 1,
            "write_GNN": True,
            }

    p = sample_run(i)
    print(f"run {i}: {p}, {n_mesh_nodes(p['x'], p['y'], p['z'], p['element_size'])} nodes")
    folder = output / f"run_{i}"
    x, y, z = p["x"], p["y"], p["z"]

    BC = {"bottom": ["010", [[0, 0, 0], [x, 0, 0], [0, 0, z], [x, 0, z]]],
          "left": ["100", [[0, 0, 0], [0, 0, z], [0, y, 0], [0, y, z]]],
          "right": ["200", [[x, 0, 0], [x, 0, z], [x, y, 0], [x, y, z]]],
          "front": ["001", [[0, 0, 0], [x, 0, 0], [0, y, 0], [x, y, 0]]],
          "back": ["002", [[0, 0, z], [x, 0, z], [0, y, z], [x, y, z]]],
          }

    create_geo_file.main(x, y, z, p["element_size"], folder, name="brick")

    mat = material(p)
    RF_props = random_field(i, p)

    load = {"force": [0, -1e6, 0],
            "node": [4],
            "time": 0.4,
            "type": "heaviside",  # pulse or heaviside or moving
            "speed": 80}  # only for moving

    dt_critical = critical_time_step(folder / "brick.msh", mat, RF_props, p["element_size"])
    # scatter rounds the number of steps up, so the step it uses is never larger than this one
    dt = time_step(p["time_step_fraction"], dt_critical, load["time"])
    print(f"run {i}: critical time step {dt_critical:.3e} s, time step {dt:.3e} s, "
          f"{math.ceil(load['time'] / dt)} steps")

    # run scatter
    scatter(folder / "brick.msh", folder, mat, BC, sett, load, time_step=dt,
            solver=NewmarkExplicit(linear_solver=CGSolver(),
                                   preconditioner=JacobiPreconditioner()), random_props=RF_props)


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        Index of the first run, number of runs and output folder.
    """
    parser = argparse.ArgumentParser(description="Generate the scatter runs for the GNN training")
    parser.add_argument("--first", type=int, default=0, help="index of the first run")
    parser.add_argument("--count", type=int, default=50, help="number of runs")
    parser.add_argument("--output", type=Path, default=Path("GNN"), help="folder that receives run_<i>/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for i in range(args.first, args.first + args.count):
        run(i, args.output)
