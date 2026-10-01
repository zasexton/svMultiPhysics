#!/usr/bin/env python3
"""Write one case of the 2D fitted-ALE capillary-wave benchmark (tracker M3/M5, decision D5).

The physical case of capillary_wave_2d: a small-amplitude standing capillary
wave on a deep one-phase liquid with zero gravity, released from rest with
the surface elevation y0 + a0 cos(kx), half a wavelength between two
free-slip mirror walls, La = 3000, rho = gamma = 1.  Unlike capillary_wave_2d
the free surface is a boundary of the mesh (fitted ALE): the liquid mesh
moves with a coupled harmonic mesh-velocity extension, the free-surface
normal kinematics are enforced on the mesh (MeshNitsche), the mesh slides
along the walls, and surface tension is the fitted Laplace-Beltrami form
(SurfaceStress).  The physical constants, the time schedule and Prosperetti's
reference are imported from capillary_wave_2d, so both benchmarks describe
one case.

The case is written for the new OOP solver: solver.xml, the liquid Triangle3
mesh (mapped to the initial surface) with the initial fields, the four
boundary face files, and case.json with every parameter verify.py needs.
See README.md.
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


def _load(directory: str, name: str):
    path = HERE.parent / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CW = _load("capillary_wave_2d", "generate_case")       # physical case, schedule, reference
LS = _load("linear_sloshing_2d", "generate_case")      # VTU/VTP writers, direct solver block
reference = CW.reference                               # prosperetti_reference module

# ---------------------------------------------------------------------------
# Protocol constants, fixed for every case (principle P1).  The physical ones
# are those of capillary_wave_2d; README.md gives the source or the reason
# for each numerical one.
# ---------------------------------------------------------------------------
LEVELS = CW.LEVELS                          # lambda / h = 16, 32, 64
DT_STUDY_LEVEL = 32                         # mesh of the separate time-step study (D10)
DT_DIVISORS = CW.DT_DIVISORS                # dt, dt/2, dt/4 of the shared protocol step
DENSITY = CW.DENSITY
SURFACE_TENSION = CW.SURFACE_TENSION
WAVELENGTH = CW.WAVELENGTH
WAVENUMBER = CW.WAVENUMBER
AMPLITUDE = CW.AMPLITUDE_OVER_WAVELENGTH * CW.WAVELENGTH
LAPLACE_NUMBER = CW.DEFAULT_LAPLACE_NUMBER
WIDTH = CW.BOX_WIDTH                        # walls at the crest (x = 0) and the trough (x = lambda/2)
MEAN_LEVEL = CW.MEAN_LEVEL                  # same liquid depth as capillary_wave_2d
PERIODS = CW.DEFAULT_PERIODS
SNAPSHOTS = CW.DEFAULT_SNAPSHOTS
EXTERNAL_PRESSURE = CW.EXTERNAL_PRESSURE
MESH_MOTION_KAPPA = 1.0                     # only scales the harmonic mesh equation
KINEMATIC_NITSCHE_GAMMA = 10.0              # as fitted_sloshing_2d (P1 trace inverse inequality)
TANGENTIAL_MESH_POLICY = "Free"
WALLS = ("wall_left", "wall_right", "wall_bottom")
FREE_SURFACE = "free_surface"
# Free slip for the fluid and sliding for the mesh: strong zero wall-normal
# component only, the same input in both equations.
EFFECTIVE_DIRECTION = {"wall_left": "1 0", "wall_right": "1 0", "wall_bottom": "0 1"}


# ---------------------------------------------------------------------------
# Time schedule
# ---------------------------------------------------------------------------
def time_schedule(level: int, dt_divisor: int = 1,
                  dt_over_capillary_limit: float | None = None) -> dict:
    """The capillary_wave_2d schedule (one step shared by all levels, D10).

    dt_over_capillary_limit (diagnostic only) replaces the protocol step by
    the largest step at most that multiple of this level's one-sided
    capillary limit sqrt(rho h^3/(4 pi gamma)) that divides the run into the
    protocol number of output intervals.
    """
    schedule = CW.time_schedule(level, LAPLACE_NUMBER, PERIODS, SNAPSHOTS, dt_divisor)
    limit = CW.DT_SAFETY * CW.capillary_dt_limit(schedule["h"])     # one-sided limit of this level
    schedule["dt_capillary_limit_one_sided"] = limit
    if dt_over_capillary_limit is not None:
        if not dt_over_capillary_limit > 0.0:
            raise ValueError("--dt-over-capillary-limit must be positive")
        cadence = max(1, math.ceil(schedule["end_time"] /
                                   (SNAPSHOTS * dt_over_capillary_limit * limit)))
        schedule.update(output_cadence=cadence, steps=SNAPSHOTS * cadence,
                        dt=schedule["end_time"] / (SNAPSHOTS * cadence))
    schedule["dt_over_capillary_limit"] = schedule["dt"] / limit
    return schedule


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def surface_elevation(x: np.ndarray, amplitude: float = AMPLITUDE) -> np.ndarray:
    return MEAN_LEVEL + amplitude * np.cos(WAVENUMBER * x)


def liquid_triangle_mesh(level: int, amplitude: float = AMPLITUDE):
    """Liquid region under the initial surface, split into right triangles.

    A structured grid of level/2 x level cells on [0, lambda/2] x [0, 1] is
    mapped onto 0 <= y <= y0 + a0 cos(kx) (y -> s (y0 + eta(x))), so the top
    row lies on the initial surface; the row height is y0/level = 1.014 h
    with h = lambda/level, and successive levels halve both cell sizes.  The
    diagonal alternates with (i + j) parity; with an even number of columns
    the pattern is mirror-symmetric about x = lambda/4, the node line of the
    mode, so crest and trough see the same mesh (as in capillary_wave_2d).
    """
    nx, ny = level // 2, level
    xs = np.linspace(0.0, WIDTH, nx + 1)
    ss = np.linspace(0.0, 1.0, ny + 1)
    xx, ssg = np.meshgrid(xs, ss)                   # row j = y index
    yy = ssg * surface_elevation(xx, amplitude)
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


def initial_pressure(points: np.ndarray, amplitude: float = AMPLITUDE) -> np.ndarray:
    """Linear pressure of the released state (capillary_wave_2d): harmonic,
    zero normal derivative at the bottom, gamma * kappa at the mean level;
    linear in the amplitude (p_ext = 0)."""
    return CW.initial_pressure(points) * (amplitude / AMPLITUDE)


def polygon_area(points: np.ndarray, cells: np.ndarray) -> float:
    p = points[cells][:, :, :2]
    e1, e2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return float(0.5 * np.sum(np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])))


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def solver_xml(schedule: dict, steps: int, cadence: int) -> str:
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS + (FREE_SURFACE,))
    wall_bcs = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>{d}</Effective_direction>
    </Add_BC>""" for w, d in EFFECTIVE_DIRECTION.items())
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- fitted_capillary_wave_2d benchmark; generated by generate_case.py -->
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

  <Add_mesh name="liquid">
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
    <Force_y>0.0</Force_y>
    <Force_z>0.0</Force_z>
    <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
    <Viscosity model="Constant">
      <Value>{schedule['viscosity']:.17g}</Value>
    </Viscosity>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
      <Mesh_displacement>true</Mesh_displacement>
      <Mesh_velocity>true</Mesh_velocity>
    </Output>
{LS.linear_solver_block()}
{wall_bcs}
    <Add_BC name="{FREE_SURFACE}">
      <Type>Free_surface</Type>
      <Implementation>FittedALE</Implementation>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>{SURFACE_TENSION:.17g}</Surface_tension>
      <Surface_tension_form>SurfaceStress</Surface_tension_form>
      <Allow_fitted_surface_stress>true</Allow_fitted_surface_stress>
      <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
      <Tangential_mesh_policy>{TANGENTIAL_MESH_POLICY}</Tangential_mesh_policy>
      <Kinematic_enforcement>MeshNitsche</Kinematic_enforcement>
      <Kinematic_nitsche_gamma>{KINEMATIC_NITSCHE_GAMMA:.17g}</Kinematic_nitsche_gamma>
    </Add_BC>
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
{wall_bcs}
  </Add_equation>
</svMultiPhysicsFile>
"""


# ---------------------------------------------------------------------------
def study_role(level: int, dt_divisor: int, dt_over_capillary_limit: float | None,
               amplitude: float = AMPLITUDE) -> str | None:
    """'spatial' (shared step, every level), 'time' (refined step at the
    time-study level) or None (diagnostic).  The spatial run at the
    time-study level is also the coarsest point of the time-step study."""
    if dt_over_capillary_limit is not None or amplitude != AMPLITUDE:
        return None
    if dt_divisor == 1 and level in LEVELS:
        return "spatial"
    if level == DT_STUDY_LEVEL and dt_divisor in DT_DIVISORS:
        return "time"
    return None


def generate(level: int, output_dir: Path, *, dt_divisor: int = 1,
             dt_over_capillary_limit: float | None = None,
             amplitude_over_wavelength: float | None = None,
             max_steps: int | None = None, force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    if dt_over_capillary_limit is not None and dt_divisor != 1:
        raise ValueError("--dt-over-capillary-limit replaces the protocol step; use --dt-divisor 1")
    amplitude = AMPLITUDE
    if amplitude_over_wavelength is not None:
        if not 0.0 < amplitude_over_wavelength <= 0.05:
            raise ValueError("--amplitude-over-wavelength must lie in (0, 0.05]")
        amplitude = amplitude_over_wavelength * WAVELENGTH
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule(level, dt_divisor, dt_over_capillary_limit)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, (nx, ny) = liquid_triangle_mesh(level, amplitude)
    n_points = points.shape[0]
    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    zeros3 = np.zeros((n_points, 3))
    LS.write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
                 {"GlobalNodeID": ("Int64", np.arange(n_points)),
                  "Velocity": ("Float64", zeros3),
                  "Pressure": ("Float64", initial_pressure(points, amplitude)),
                  "mesh_displacement": ("Float64", zeros3),
                  "mesh_velocity": ("Float64", zeros3)},
                 {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for name in WALLS + (FREE_SURFACE,):
        node_ids, parents = faces[name]
        LS.write_face_vtp(mesh_dir / "mesh-surfaces" / f"{name}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(solver_xml(schedule, steps, cadence), encoding="utf-8")

    role = study_role(level, dt_divisor, dt_over_capillary_limit, amplitude)
    h = schedule["h"]
    case = {
        "benchmark": "fitted_capillary_wave_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/fitted_capillary_wave_2d/generate_case.py",
        "level_lambda_over_h": level,
        "h": h,
        "row_height": MEAN_LEVEL / ny,
        "cells_per_side": [nx, ny],
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "free_surface_nodes": [int(v) for v in faces[FREE_SURFACE][0]],
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "laplace_number": LAPLACE_NUMBER,
        "viscosity": schedule["viscosity"],
        "kinematic_viscosity": schedule["kinematic_viscosity"],
        "wavelength": WAVELENGTH,
        "wavenumber": WAVENUMBER,
        "initial_amplitude": amplitude,
        "mean_level": MEAN_LEVEL,
        "width": WIDTH,
        "external_pressure": EXTERNAL_PRESSURE,
        "omega0": schedule["omega0"],
        "inviscid_period": schedule["inviscid_period"],
        "epsilon": schedule["epsilon"],
        "normal_mode_omega": schedule["normal_mode_omega"],
        "normal_mode_damping_rate": schedule["normal_mode_damping_rate"],
        "boundary_layer_thickness_over_h": schedule["boundary_layer_thickness"] / h,
        "initial_liquid_area_sampled": polygon_area(points, cells),
        "velocity_field": "Velocity",
        "displacement_field": "mesh_displacement",
        "kinematic_enforcement": "MeshNitsche",
        "kinematic_nitsche_gamma": KINEMATIC_NITSCHE_GAMMA,
        "mesh_motion_kappa": MESH_MOTION_KAPPA,
        "tangential_mesh_policy": TANGENTIAL_MESH_POLICY,
        "surface_tension_form": "SurfaceStress",
        "wall_bc": "fluid free slip (strong zero normal velocity); mesh slides along the walls",
        "periods": PERIODS,
        "end_time_protocol": schedule["end_time"],
        "dt_divisor": dt_divisor,
        "dt": schedule["dt"],
        "dt_capillary_limit_one_sided": schedule["dt_capillary_limit_one_sided"],
        "dt_over_capillary_limit": schedule["dt_over_capillary_limit"],
        "dt_over_capillary_limit_requested": dt_over_capillary_limit,
        "study": role,
        "protocol_run": role is not None,
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="lambda/h")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dt-divisor", type=int, default=1, choices=DT_DIVISORS,
                        help="divide the shared protocol step (time-step study at "
                             f"lambda/h = {DT_STUDY_LEVEL}, D10; protocol value 1)")
    parser.add_argument("--dt-over-capillary-limit", type=float, default=None,
                        help="diagnostic runs only: a step of about this multiple of the "
                             "level's capillary limit (never gated)")
    parser.add_argument("--amplitude-over-wavelength", type=float, default=None,
                        help="diagnostic runs only: initial amplitude a0/lambda (protocol value "
                             f"{CW.AMPLITUDE_OVER_WAVELENGTH}; never gated)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; verify.py rejects "
                             "such runs for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)
    try:
        case = generate(args.level, args.output_dir, dt_divisor=args.dt_divisor,
                        dt_over_capillary_limit=args.dt_over_capillary_limit,
                        amplitude_over_wavelength=args.amplitude_over_wavelength,
                        max_steps=args.max_steps, force=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output_dir}")
    for key in ("level_lambda_over_h", "n_vertices", "n_triangles", "omega0", "normal_mode_omega",
                "normal_mode_damping_rate", "dt", "dt_over_capillary_limit", "steps",
                "output_cadence", "end_time", "study", "protocol_run", "truncated"):
        print(f"  {key} = {case[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
