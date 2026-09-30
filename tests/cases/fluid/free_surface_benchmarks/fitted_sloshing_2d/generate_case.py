#!/usr/bin/env python3
"""Write one case of the 2D fitted-ALE sloshing benchmark (tracker M5, decision D5).

The first antisymmetric standing wave in a rectangular tank with free-slip
walls, released from rest with a small cosine displacement of the free
surface, gravity only (zero surface tension).  Unlike linear_sloshing_2d the
free surface is a boundary of the mesh (fitted ALE): the liquid mesh moves
with a coupled harmonic mesh displacement, the free-surface normal kinematics
are enforced on the mesh, and the mesh slides along the walls.  The physical
setup, the reference solution (the exact linear viscous dispersion relation
with free-slip walls) and the metrics are those of linear_sloshing_2d, whose
reference code is imported from the sibling directory.

The case is written for the new OOP solver: solver.xml, the liquid Triangle3
mesh (mapped to the initial surface shape) with the initial fields, the four
boundary face files, and case.json with every parameter verify.py needs.  See
README.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_linear_sloshing():
    path = HERE.parent / "linear_sloshing_2d" / "generate_case.py"
    spec = importlib.util.spec_from_file_location("linear_sloshing_2d_generate_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LS = _load_linear_sloshing()

# ---------------------------------------------------------------------------
# Protocol constants, fixed for every case (principle P1).  The physical
# constants are those of linear_sloshing_2d; README.md gives the source or the
# reason for each numerical one.
# ---------------------------------------------------------------------------
LEVELS = (16, 32, 64)                       # cells per tank length (L/h)
DT_STUDY_LEVEL = 32                         # mesh of the separate time-step study
DT_STUDY_STEPS_PER_PERIOD = (64, 128, 256, 512)
SPATIAL_STUDY_STEPS_PER_PERIOD = 512        # fixed small time step of the spatial study (D10)
DENSITY = LS.DENSITY
GRAVITY = LS.GRAVITY
KINEMATIC_VISCOSITY = LS.KINEMATIC_VISCOSITY
TANK_LENGTH = LS.TANK_LENGTH
MEAN_DEPTH = LS.MEAN_DEPTH                  # same H0 as linear_sloshing_2d, same reference
AMPLITUDE = LS.AMPLITUDE
MODE = LS.MODE
PERIODS = LS.PERIODS
SNAPSHOTS_PER_PERIOD = LS.SNAPSHOTS_PER_PERIOD
PROBE_X = LS.PROBE_X
EXTERNAL_PRESSURE = 0.0
# Mesh motion: harmonic extension of the mesh velocity (Harmonic_quantity =
# velocity).  kappa only scales the mesh equation (the harmonic extension
# does not depend on it); it is fixed at 1.
MESH_MOTION_KAPPA = 1.0
# Nitsche constant of the mesh-side normal kinematic row (penalty
# gamma_N / h_n on the normal mesh-velocity mismatch), fixed once from the P1
# trace inverse inequality ||d_n v||_F^2 <= (2/h_n) ||grad v||_T^2
# (h_n = 2|T|/|F|): the row is coercive for gamma_N > 2 kappa; 10 leaves a
# factor 5 (README.md).
KINEMATIC_NITSCHE_GAMMA = 10.0
KINEMATIC_ENFORCEMENT = ("MeshNitsche", "Nitsche", "Penalty")
TANGENTIAL_MESH_POLICY = "Free"
WALLS = ("wall_left", "wall_right", "wall_bottom")
FREE_SURFACE = "free_surface"
# Free slip for the fluid and sliding for the mesh: strong zero wall-normal
# component only (Effective_direction selects it).
EFFECTIVE_DIRECTION = {"wall_left": "1 0", "wall_right": "1 0", "wall_bottom": "0 1"}
WALL_MESH_MOTION = ("slip", "pinned")
SURFACE_MESH_MOTION = ("free", "vertical")      # diagnostic: vertical spines fix d_x on the surface


# ---------------------------------------------------------------------------
# Reference and schedule (shared with linear_sloshing_2d)
# ---------------------------------------------------------------------------
def reference(depth: float = MEAN_DEPTH) -> dict:
    return LS.reference(depth=depth)


def time_schedule(steps_per_period: int, periods: float = PERIODS,
                  depth: float = MEAN_DEPTH) -> dict:
    ref = reference(depth)
    cadence = max(1, steps_per_period // SNAPSHOTS_PER_PERIOD)
    steps = int(round(periods * steps_per_period))
    steps -= steps % cadence
    return {"dt": ref["period_inviscid"] / steps_per_period, "steps": steps,
            "steps_per_period": steps_per_period, "output_cadence": cadence}


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def rows_for_level(level: int, depth: float = MEAN_DEPTH) -> int:
    """Cell rows over the depth: L/(2h) (8, 16, 32), so that successive
    levels halve both cell dimensions; the row height H0/rows is 1.016 h."""
    del depth
    return max(2, level // 2)


def surface_elevation(x: np.ndarray, k: float) -> np.ndarray:
    return AMPLITUDE * np.cos(k * x)


def liquid_triangle_mesh(level: int, k: float, depth: float = MEAN_DEPTH):
    """Liquid region under the initial surface, split into right triangles.

    A structured grid on [0, L] x [0, H0] is mapped vertically onto
    0 <= y <= H0 + A cos(k x) (y -> y (H0 + eta(x)) / H0), so the top row lies
    on the initial surface.  The diagonal alternates with (i + j) parity; with
    an even number of cells per row the pattern is mirror-symmetric about
    x = L/2, like the mode.
    """
    nx = int(round(TANK_LENGTH * level))
    ny = rows_for_level(level, depth)
    xs = np.linspace(0.0, TANK_LENGTH, nx + 1)
    ss = np.linspace(0.0, 1.0, ny + 1)
    xx, ssg = np.meshgrid(xs, ss)                   # row j = y index
    yy = ssg * (depth + surface_elevation(xx, k))
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    cells = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            cells += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    cells = np.asarray(cells, dtype=np.int64)

    def parent(i: int, j: int, side: str) -> int:
        even = (i + j) % 2 == 0
        second = {"bottom": False, "top": True, "left": even, "right": not even}[side]
        return 2 * (j * nx + i) + int(second)

    faces = {
        "wall_bottom": ([vid(i, 0) for i in range(nx + 1)],
                        [parent(i, 0, "bottom") for i in range(nx)]),
        FREE_SURFACE: ([vid(i, ny) for i in range(nx + 1)],
                       [parent(i, ny - 1, "top") for i in range(nx)]),
        "wall_left": ([vid(0, j) for j in range(ny + 1)],
                      [parent(0, j, "left") for j in range(ny)]),
        "wall_right": ([vid(nx, j) for j in range(ny + 1)],
                       [parent(nx - 1, j, "right") for j in range(ny)]),
    }
    return points, cells, faces, (nx, ny)


def initial_pressure(points: np.ndarray, k: float, depth: float = MEAN_DEPTH) -> np.ndarray:
    """Linear standing-wave pressure at rest (zero on the surface to second order in A)."""
    x, y = points[:, 0], points[:, 1]
    return (EXTERNAL_PRESSURE + DENSITY * GRAVITY * (depth - y)
            + DENSITY * GRAVITY * AMPLITUDE * np.cosh(k * y) / math.cosh(k * depth)
            * np.cos(k * x))


def polygon_area(points: np.ndarray, cells: np.ndarray) -> float:
    p = points[cells][:, :, :2]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return float(0.5 * np.sum(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])))


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def linear_solver_block() -> str:
    return LS.linear_solver_block()


def free_surface_block(enforcement: str, nitsche_gamma: float = KINEMATIC_NITSCHE_GAMMA) -> str:
    if enforcement == "Penalty":
        kinematic = """      <Kinematic_enforcement>Penalty</Kinematic_enforcement>
      <Kinematic_penalty>1.0e4</Kinematic_penalty>"""
    else:
        kinematic = f"""      <Kinematic_enforcement>{enforcement}</Kinematic_enforcement>
      <Kinematic_nitsche_gamma>{nitsche_gamma:.17g}</Kinematic_nitsche_gamma>"""
    return f"""    <Add_BC name="{FREE_SURFACE}">
      <Type>Free_surface</Type>
      <Implementation>FittedALE</Implementation>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>0.0</Surface_tension>
      <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
      <Tangential_mesh_policy>{TANGENTIAL_MESH_POLICY}</Tangential_mesh_policy>
{kinematic}
    </Add_BC>"""


def solver_xml(schedule: dict, steps: int, cadence: int, *, enforcement: str = "MeshNitsche",
               wall_mesh_motion: str = "slip", nitsche_gamma: float = KINEMATIC_NITSCHE_GAMMA,
               surface_mesh_motion: str = "free") -> str:
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS + (FREE_SURFACE,))
    fluid_bcs = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>{d}</Effective_direction>
    </Add_BC>""" for w, d in EFFECTIVE_DIRECTION.items())
    if wall_mesh_motion == "slip":
        mesh_bcs = fluid_bcs
        if surface_mesh_motion == "vertical":
            mesh_bcs += f"""
    <Add_BC name="{FREE_SURFACE}">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>1 0</Effective_direction>
    </Add_BC>"""
    else:
        mesh_bcs = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>""" for w in WALLS)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- fitted_sloshing_2d benchmark; generated by generate_case.py -->
<svMultiPhysicsFile version="0.1">
  <GeneralSimulationParameters>
    <Use_new_OOP_solver>true</Use_new_OOP_solver>
    <Continue_previous_simulation>false</Continue_previous_simulation>
    <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
    <Number_of_time_steps>{steps}</Number_of_time_steps>
    <Time_step_size>{schedule['dt']:.17g}</Time_step_size>
    <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
    <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
    <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
    <Combine_time_series>true</Combine_time_series>
    <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
    <Increment_in_saving_VTK_files>{cadence}</Increment_in_saving_VTK_files>
    <Start_saving_after_time_step>{cadence}</Start_saving_after_time_step>
    <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
    <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
    <Verbose>1</Verbose>
    <Warning>0</Warning>
    <Debug>0</Debug>
  </GeneralSimulationParameters>

  <Add_mesh name="tank">
    <Mesh_file_path>mesh/mesh-complete.mesh.vtu</Mesh_file_path>
{faces}
  </Add_mesh>

  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>12</Max_iterations>
    <Tolerance>1.0e-6</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
    <Enable_ALE>true</Enable_ALE>
    <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
    <Mesh_velocity_field>mesh_velocity</Mesh_velocity_field>
    <Mesh_displacement_field>mesh_displacement</Mesh_displacement_field>
    <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
    <Moving_mesh_tangent_path>SymbolicRequired</Moving_mesh_tangent_path>
    <Density>{DENSITY:.17g}</Density>
    <Force_x>0.0</Force_x>
    <Force_y>{-GRAVITY:.17g}</Force_y>
    <Force_z>0.0</Force_z>
    <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
    <Viscosity model="Constant">
      <Value>{DENSITY * KINEMATIC_VISCOSITY:.17g}</Value>
    </Viscosity>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
      <Mesh_displacement>true</Mesh_displacement>
      <Mesh_velocity>true</Mesh_velocity>
    </Output>
{linear_solver_block()}
{fluid_bcs}
{free_surface_block(enforcement, nitsche_gamma)}
  </Add_equation>

  <Add_equation type="mesh_motion">
    <Coupled>true</Coupled>
    <Model>Harmonic</Model>
    <Field_name>mesh_displacement</Field_name>
    <Operator_tag>equations</Operator_tag>
    <Kappa>{MESH_MOTION_KAPPA:.17g}</Kappa>
    <Harmonic_quantity>velocity</Harmonic_quantity>
    <Moving_mesh_tangent_path>SymbolicRequired</Moving_mesh_tangent_path>
    <Module_options>jit=true; jit_specialization=true</Module_options>
{mesh_bcs}
  </Add_equation>
</svMultiPhysicsFile>
"""


# ---------------------------------------------------------------------------
def protocol_role(level: int, steps_per_period: int) -> str | None:
    if steps_per_period == SPATIAL_STUDY_STEPS_PER_PERIOD and level in LEVELS:
        return "spatial"
    if level == DT_STUDY_LEVEL and steps_per_period in DT_STUDY_STEPS_PER_PERIOD:
        return "time"
    return None


def generate(level: int, output_dir: Path, *, steps_per_period: int = SPATIAL_STUDY_STEPS_PER_PERIOD,
             periods: float = PERIODS, enforcement: str = "MeshNitsche",
             wall_mesh_motion: str = "slip", max_steps: int | None = None,
             nitsche_gamma: float = KINEMATIC_NITSCHE_GAMMA, surface_mesh_motion: str = "free",
             force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if not periods > 0.0:
        raise ValueError("--periods must be positive")
    if steps_per_period < SNAPSHOTS_PER_PERIOD:
        raise ValueError(f"--steps-per-period must be at least {SNAPSHOTS_PER_PERIOD}")
    if enforcement not in KINEMATIC_ENFORCEMENT:
        raise ValueError(f"--kinematic-enforcement must be one of {KINEMATIC_ENFORCEMENT}")
    if wall_mesh_motion not in WALL_MESH_MOTION:
        raise ValueError(f"--wall-mesh-motion must be one of {WALL_MESH_MOTION}")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    ref = reference()
    schedule = time_schedule(steps_per_period, periods)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, (nx, ny) = liquid_triangle_mesh(level, ref["wavenumber"])
    n_points = points.shape[0]
    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    zeros3 = np.zeros((n_points, 3))
    LS.write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
                 {"GlobalNodeID": ("Int64", np.arange(n_points)),
                  "Velocity": ("Float64", zeros3),
                  "Pressure": ("Float64", initial_pressure(points, ref["wavenumber"])),
                  "mesh_displacement": ("Float64", zeros3),
                  "mesh_velocity": ("Float64", zeros3)},
                 {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for name in WALLS + (FREE_SURFACE,):
        node_ids, parents = faces[name]
        LS.write_face_vtp(mesh_dir / "mesh-surfaces" / f"{name}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(
        solver_xml(schedule, steps, cadence, enforcement=enforcement,
                   wall_mesh_motion=wall_mesh_motion, nitsche_gamma=nitsche_gamma,
                   surface_mesh_motion=surface_mesh_motion),
        encoding="utf-8")

    role = protocol_role(level, steps_per_period)
    protocol = (role is not None and enforcement == "MeshNitsche"
                and wall_mesh_motion == "slip" and periods == PERIODS
                and nitsche_gamma == KINEMATIC_NITSCHE_GAMMA
                and surface_mesh_motion == "free")
    case = {
        "benchmark": "fitted_sloshing_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/fitted_sloshing_2d/generate_case.py",
        "level_cells_per_length": level,
        "h": TANK_LENGTH / level,
        "rows": ny,
        "row_height": MEAN_DEPTH / ny,
        "tank_length": TANK_LENGTH,
        "cells_per_side": [nx, ny],
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "free_surface_nodes": [int(v) for v in faces[FREE_SURFACE][0]],
        "density": DENSITY,
        "gravity": GRAVITY,
        "kinematic_viscosity": KINEMATIC_VISCOSITY,
        "mean_depth": MEAN_DEPTH,
        "amplitude": AMPLITUDE,
        "mode": MODE,
        **ref,
        "probe_x": PROBE_X,
        "initial_liquid_area_sampled": polygon_area(points, cells),
        "velocity_field": "Velocity",
        "displacement_field": "mesh_displacement",
        "kinematic_enforcement": enforcement,
        "kinematic_nitsche_gamma": nitsche_gamma,
        "mesh_motion_kappa": MESH_MOTION_KAPPA,
        "tangential_mesh_policy": TANGENTIAL_MESH_POLICY,
        "wall_mesh_motion": wall_mesh_motion,
        "surface_mesh_motion": surface_mesh_motion,
        "periods": periods,
        "dt": schedule["dt"],
        "steps_per_period": schedule["steps_per_period"],
        "study": role,
        "protocol_run": bool(protocol),
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "wall_bc": ("fluid free slip (strong zero normal velocity); mesh "
                    + ("slides along the walls (strong zero normal displacement)"
                       if wall_mesh_motion == "slip" else "pinned (zero displacement)")),
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS,
                        help="cells per tank length L/h")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps-per-period", type=int, default=SPATIAL_STUDY_STEPS_PER_PERIOD,
                        help=f"time steps per inviscid period (spatial study "
                             f"{SPATIAL_STUDY_STEPS_PER_PERIOD}; time-step study "
                             f"{DT_STUDY_STEPS_PER_PERIOD} at L/h = {DT_STUDY_LEVEL})")
    parser.add_argument("--periods", type=float, default=PERIODS,
                        help="run length in inviscid periods (protocol value 4)")
    parser.add_argument("--kinematic-enforcement", choices=KINEMATIC_ENFORCEMENT,
                        default="MeshNitsche",
                        help="diagnostic runs only (protocol value MeshNitsche)")
    parser.add_argument("--wall-mesh-motion", choices=WALL_MESH_MOTION, default="slip",
                        help="diagnostic runs only (protocol value slip)")
    parser.add_argument("--kinematic-nitsche-gamma", type=float, default=KINEMATIC_NITSCHE_GAMMA,
                        help="diagnostic sensitivity runs only (protocol value 10, fixed by P1)")
    parser.add_argument("--surface-mesh-motion", choices=SURFACE_MESH_MOTION, default="free",
                        help="diagnostic runs only (protocol value free)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; verify.py rejects "
                             "such runs for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)
    case = generate(args.level, args.output_dir, steps_per_period=args.steps_per_period,
                    periods=args.periods, enforcement=args.kinematic_enforcement,
                    wall_mesh_motion=args.wall_mesh_motion, max_steps=args.max_steps,
                    nitsche_gamma=args.kinematic_nitsche_gamma,
                    surface_mesh_motion=args.surface_mesh_motion, force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("level_cells_per_length", "rows", "n_vertices", "n_triangles",
                "omega_inviscid", "omega_reference", "damping_rate_reference", "dt",
                "steps_per_period", "steps", "output_cadence", "end_time", "study",
                "protocol_run", "truncated"):
        print(f"  {key} = {case[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
