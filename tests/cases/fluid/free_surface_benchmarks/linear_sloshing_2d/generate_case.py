#!/usr/bin/env python3
"""Write one resolution level of the 2D linear-sloshing benchmark (tracker M1).

The first antisymmetric standing wave in a rectangular tank with free-slip
walls, released from rest with a small cosine surface displacement.  Gravity
is the only restoring force (zero surface tension).  The unfitted level-set
solution is compared with linear theory: the frequency and damping rate of
the viscous standing wave, and liquid-volume conservation.

The case is written for the new OOP solver: solver.xml, an affine Triangle3
background mesh with the initial fields, the four wall face files, and
case.json with every parameter that verify.py needs, including the reference
frequency and damping rate.  See README.md.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants, fixed for every level (principle P1).  README.md gives
# the source or the reason for each one.  Units: rho = g = 1, tank length 1.
# ---------------------------------------------------------------------------
LEVELS = (16, 32, 64)                       # cells per tank length (L/h)
DENSITY = 1.0
GRAVITY = 1.0                               # body force (0, -g, 0)
KINEMATIC_VISCOSITY = 5.0e-4                # nu; mu = rho * nu
TANK_LENGTH = 1.0                           # L
TANK_HEIGHT = 0.625                         # background mesh height
# Mean depth H0 = 0.5 + 1/128.  With A = 0.005 < 1/128 the surface stays
# inside (0.5, 0.515625), which contains no vertex row at any level, so the
# interface never passes through a vertex (see README.md).
MEAN_DEPTH = 0.5 + 1.0 / 128.0
AMPLITUDE = 0.005
MODE = 1                                    # k = MODE * pi / L
PERIODS = 4                                 # run length in inviscid periods
STEPS_PER_PERIOD_PER_LEVEL = 2              # dt = T0 / (2 L/h): 32, 64, 128 steps per period
SNAPSHOTS_PER_PERIOD = 32
PROBE_X = 0.0                               # wave gauge on the left wall (antinode)
EXTERNAL_PRESSURE = 0.0
LEVEL_SET_FIELD = "phi"
INTERFACE_DOMAIN_ID = "linear_sloshing_surface"
WALLS = ("wall_left", "wall_right", "wall_bottom", "wall_top")
# Free slip: strong zero velocity on the wall-normal component only.  The top
# wall is dry and carries no condition.
EFFECTIVE_DIRECTION = {"wall_left": "1 0", "wall_right": "1 0", "wall_bottom": "0 1"}


# ---------------------------------------------------------------------------
# Linear theory
# ---------------------------------------------------------------------------
def wavenumber(length: float = TANK_LENGTH, mode: int = MODE) -> float:
    return mode * math.pi / length


def inviscid_frequency(g: float, k: float, depth: float) -> float:
    """omega0^2 = g k tanh(k H0) (Faltinsen and Timokha 2009)."""
    return math.sqrt(g * k * math.tanh(k * depth))


def viscous_dispersion(s: complex, g: float, k: float, depth: float, nu: float) -> complex:
    """Linear viscous standing wave with free-slip walls and bottom.

    Modes Phi = A cosh(k y) cos(k x) e^{st} and Psi = B sinh(m y) sin(k x) e^{st},
    m^2 = k^2 + s/nu, satisfy the free-slip conditions on x = 0, L and y = 0
    exactly.  The kinematic, tangential-stress and normal-stress conditions at
    y = H0 leave D(s) = 0 with
        D = g k (S - beta) + s^2 C + 2 nu k^2 s C - 2 nu k m beta s coth(m H0),
        S = sinh(k H0), C = cosh(k H0), beta = 2 k^2 S / (m^2 + k^2).
    For H0 -> infinity it reduces to Lamb's (1932, art. 349) deep-water relation
    (s + 2 nu k^2)^2 + g k = 4 nu^2 k^3 m.
    """
    m = cmath.sqrt(k * k + s / nu)
    if m.real < 0.0:
        m = -m
    big_s, big_c = math.sinh(k * depth), math.cosh(k * depth)
    beta = 2.0 * k * k * big_s / (m * m + k * k)
    e = cmath.exp(-2.0 * m * depth)
    coth = (1.0 + e) / (1.0 - e)
    return (g * k * (big_s - beta) + s * s * big_c + 2.0 * nu * k * k * s * big_c
            - 2.0 * nu * k * m * beta * s * coth)


def viscous_root(g: float, k: float, depth: float, nu: float) -> complex:
    """Root s = -gamma + i omega_d of the dispersion relation near i omega0 - 2 nu k^2."""
    omega0 = inviscid_frequency(g, k, depth)
    s = complex(-2.0 * nu * k * k, omega0)
    for _ in range(100):
        f = viscous_dispersion(s, g, k, depth, nu)
        d = 1e-7 * abs(s)
        fp = (viscous_dispersion(s + d, g, k, depth, nu)
              - viscous_dispersion(s - d, g, k, depth, nu)) / (2.0 * d)
        step = f / fp
        s -= step
        if abs(step) < 1e-15 * abs(s):
            break
    else:                                   # pragma: no cover - never observed
        raise RuntimeError("dispersion relation root did not converge")
    return s


def reference(g: float = GRAVITY, depth: float = MEAN_DEPTH, nu: float = KINEMATIC_VISCOSITY,
              length: float = TANK_LENGTH, mode: int = MODE) -> dict:
    k = wavenumber(length, mode)
    omega0 = inviscid_frequency(g, k, depth)
    s = viscous_root(g, k, depth, nu)
    return {
        "wavenumber": k,
        "omega_inviscid": omega0,
        "period_inviscid": 2.0 * math.pi / omega0,
        "omega_reference": s.imag,
        "damping_rate_reference": -s.real,
        "damping_rate_lamb": 2.0 * nu * k * k,
        "viscous_parameter_nu_k2_over_omega": nu * k * k / omega0,
        "surface_boundary_layer_thickness": math.sqrt(2.0 * nu / omega0),
    }


def time_schedule(level: int, periods: float = PERIODS,
                  steps_per_period: int | None = None) -> dict:
    ref = reference()
    if steps_per_period is None:
        steps_per_period = STEPS_PER_PERIOD_PER_LEVEL * level
    cadence = max(1, steps_per_period // SNAPSHOTS_PER_PERIOD)
    steps = int(round(periods * steps_per_period))
    steps -= steps % cadence
    return {"dt": ref["period_inviscid"] / steps_per_period, "steps": steps,
            "steps_per_period": steps_per_period, "output_cadence": cadence}


def initial_fields(points: np.ndarray, k: float) -> dict:
    """Linear standing wave at t = 0: eta = A cos(k x), u = 0, and the linear pressure."""
    x, y = points[:, 0], points[:, 1]
    phi = y - MEAN_DEPTH - AMPLITUDE * np.cos(k * x)
    pressure = (EXTERNAL_PRESSURE + DENSITY * GRAVITY * (MEAN_DEPTH - y)
                + DENSITY * GRAVITY * AMPLITUDE * np.cosh(k * y) / math.cosh(k * MEAN_DEPTH)
                * np.cos(k * x))
    return {"phi": phi, "pressure": pressure}


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def structured_triangle_mesh(level: int):
    """Tank [0, L] x [0, H_tank] split into right triangles.

    The diagonal alternates with (i + j) parity; with an even number of cells
    per row the pattern is mirror-symmetric about x = L/2, like the mode.
    """
    nx = int(round(TANK_LENGTH * level))
    ny = int(round(TANK_HEIGHT * level / TANK_LENGTH))
    if not math.isclose(ny * TANK_LENGTH / level, TANK_HEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("tank height must be an integer number of cells")
    xs = np.linspace(0.0, TANK_LENGTH, nx + 1)
    ys = np.linspace(0.0, TANK_HEIGHT, ny + 1)
    xx, yy = np.meshgrid(xs, ys)                    # row j = y index
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
        "wall_top": ([vid(i, ny) for i in range(nx + 1)],
                     [parent(i, ny - 1, "top") for i in range(nx)]),
        "wall_left": ([vid(0, j) for j in range(ny + 1)],
                      [parent(0, j, "left") for j in range(ny)]),
        "wall_right": ([vid(nx, j) for j in range(ny + 1)],
                       [parent(nx - 1, j, "right") for j in range(ny)]),
    }
    return points, cells, faces, (nx, ny)


def _ascii(values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    return "\n".join(" ".join(fmt.format(v) for v in flat[s:s + 6])
                     for s in range(0, flat.size, 6))


def write_vtu(path: Path, points, cells, point_data: dict, cell_data: dict) -> None:
    n_cells = cells.shape[0]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<UnstructuredGrid>",
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">', "<PointData>"]
    for name, (vtk_type, values) in point_data.items():
        values = np.asarray(values)
        ncomp = 1 if values.ndim == 1 else values.shape[1]
        fmt = "{:d}" if vtk_type.startswith("Int") else "{:.17g}"
        out += [f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">',
                _ascii(values, fmt), "</DataArray>"]
    out += ["</PointData>", "<CellData>"]
    for name, (vtk_type, values) in cell_data.items():
        out += [f'<DataArray type="{vtk_type}" Name="{name}" format="ascii">',
                _ascii(values, "{:d}"), "</DataArray>"]
    out += ["</CellData>", "<Points>",
            '<DataArray type="Float64" NumberOfComponents="3" format="ascii">',
            _ascii(points, "{:.17g}"), "</DataArray>", "</Points>", "<Cells>",
            '<DataArray type="Int64" Name="connectivity" format="ascii">', _ascii(cells, "{:d}"),
            "</DataArray>", '<DataArray type="Int64" Name="offsets" format="ascii">',
            _ascii(3 * np.arange(1, n_cells + 1), "{:d}"), "</DataArray>",
            '<DataArray type="UInt8" Name="types" format="ascii">',
            _ascii(np.full(n_cells, 5), "{:d}"), "</DataArray>",           # VTK_TRIANGLE
            "</Cells>", "</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points, node_ids, parent_cells) -> None:
    node_ids = np.asarray(node_ids, dtype=np.int64)
    n_lines = node_ids.size - 1
    local = np.column_stack([np.arange(n_lines), np.arange(1, n_lines + 1)])
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           "<PolyData>",
           f'<Piece NumberOfPoints="{node_ids.size}" NumberOfVerts="0" NumberOfLines="{n_lines}" '
           'NumberOfStrips="0" NumberOfPolys="0">',
           "<PointData>", '<DataArray type="Int64" Name="GlobalNodeID" format="ascii">',
           _ascii(node_ids, "{:d}"), "</DataArray>", "</PointData>", "<CellData>",
           '<DataArray type="Int64" Name="GlobalElementID" format="ascii">',
           _ascii(parent_cells, "{:d}"), "</DataArray>", "</CellData>", "<Points>",
           '<DataArray type="Float64" NumberOfComponents="3" format="ascii">',
           _ascii(points[node_ids], "{:.17g}"), "</DataArray>", "</Points>", "<Lines>",
           '<DataArray type="Int64" Name="connectivity" format="ascii">', _ascii(local, "{:d}"),
           "</DataArray>", '<DataArray type="Int64" Name="offsets" format="ascii">',
           _ascii(2 * np.arange(1, n_lines + 1), "{:d}"), "</DataArray>", "</Lines>",
           "</Piece>", "</PolyData>", "</VTKFile>"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def linear_solver_block() -> str:
    return """    <LS type="Direct">
      <Linear_algebra type="eigen">
        <Preconditioner>none</Preconditioner>
      </Linear_algebra>
      <Max_iterations>100</Max_iterations>
      <Krylov_space_dimension>50</Krylov_space_dimension>
      <Tolerance>1.0e-8</Tolerance>
      <Absolute_tolerance>1.0e-10</Absolute_tolerance>
    </LS>"""


def solver_xml(schedule: dict, steps: int, cadence: int) -> str:
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS)
    bcs = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
      <Effective_direction>{d}</Effective_direction>
    </Add_BC>""" for w, d in EFFECTIVE_DIRECTION.items())
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- linear_sloshing_2d benchmark; generated by generate_case.py -->
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

  <Add_equation type="level_set">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>4</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
    <Operator_tag>equations</Operator_tag>
    <Level_set_source>prescribed_data</Level_set_source>
    <Velocity_source>coupled_field</Velocity_source>
    <Velocity_field_name>Velocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Enable_SUPG>true</Enable_SUPG>
    <SUPG_tau_scale>0.5</SUPG_tau_scale>
    <SUPG_transient_scale>2.0</SUPG_transient_scale>
    <Enable_reinitialization>false</Enable_reinitialization>
    <Enable_volume_correction>false</Enable_volume_correction>
    <Output type="Spatial">
      <Level_set>true</Level_set>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{linear_solver_block()}
  </Add_equation>

  <Add_equation type="fluid">
    <Coupled>true</Coupled>
    <Min_iterations>1</Min_iterations>
    <Max_iterations>8</Max_iterations>
    <Tolerance>1.0e-4</Tolerance>
    <Module_options>jit=true; jit_specialization=true</Module_options>
    <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
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
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{linear_solver_block()}
{bcs}
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
      <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
      <Level_set_isovalue>0.0</Level_set_isovalue>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>0.0</Surface_tension>
      <Enable_velocity_extension>false</Enable_velocity_extension>
      <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
      <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
      <Use_cut_metadata_scale>false</Use_cut_metadata_scale>
      <Small_cut_aggregation>true</Small_cut_aggregation>
    </Add_BC>
  </Add_equation>
</svMultiPhysicsFile>
"""


# ---------------------------------------------------------------------------
def generate(level: int, output_dir: Path, *, periods: float = PERIODS,
             steps_per_period: int | None = None, max_steps: int | None = None,
             force: bool = False) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if not periods > 0.0:
        raise ValueError("--periods must be positive")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    ref = reference()
    if steps_per_period is not None and steps_per_period < SNAPSHOTS_PER_PERIOD:
        raise ValueError(f"--steps-per-period must be at least {SNAPSHOTS_PER_PERIOD}")
    schedule = time_schedule(level, periods, steps_per_period)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, (nx, ny) = structured_triangle_mesh(level)
    h = TANK_LENGTH / level
    fields = initial_fields(points, ref["wavenumber"])
    rows = np.arange(ny + 1) * h
    band = (MEAN_DEPTH - AMPLITUDE, MEAN_DEPTH + AMPLITUDE)
    gap = float(np.min(np.minimum(np.abs(rows - band[0]), np.abs(rows - band[1]))))
    if np.any((rows >= band[0]) & (rows <= band[1])):
        gap = 0.0
    n_points = points.shape[0]

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(n_points)),
               LEVEL_SET_FIELD: ("Float64", fields["phi"]),
               "Velocity": ("Float64", np.zeros((n_points, 3))),
               # Linear pressure on every vertex, including the signed
               # continuation on dry vertices of cut cells.
               "Pressure": ("Float64", fields["pressure"])},
              {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in WALLS:
        node_ids, parents = faces[wall]
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(solver_xml(schedule, steps, cadence), encoding="utf-8")

    case = {
        "benchmark": "linear_sloshing_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/linear_sloshing_2d/generate_case.py",
        "level_cells_per_length": level,
        "h": h,
        "tank_length": TANK_LENGTH,
        "tank_height": TANK_HEIGHT,
        "cells_per_side": [nx, ny],
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "density": DENSITY,
        "gravity": GRAVITY,
        "kinematic_viscosity": KINEMATIC_VISCOSITY,
        "mean_depth": MEAN_DEPTH,
        "amplitude": AMPLITUDE,
        "mode": MODE,
        **ref,
        "probe_x": PROBE_X,
        "interface_band_vertex_gap_over_h": gap / h,
        "level_set_field": LEVEL_SET_FIELD,
        "velocity_field": "Velocity",
        "pressure_field": "Pressure",
        "liquid_side": "phi<0",
        "periods": periods,
        "dt": schedule["dt"],
        "steps_per_period": schedule["steps_per_period"],
        "protocol_time_step": schedule["steps_per_period"] == STEPS_PER_PERIOD_PER_LEVEL * level,
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "wall_bc": "free slip: strong zero normal velocity (Effective_direction), top wall dry",
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
    parser.add_argument("--periods", type=float, default=PERIODS,
                        help="run length in inviscid periods (protocol value 4)")
    parser.add_argument("--steps-per-period", type=int, default=None,
                        help="diagnostic time-step studies only (protocol value 2 L/h); "
                             "verify.py reports such runs but does not gate them")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; verify.py rejects "
                             "such runs for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)
    case = generate(args.level, args.output_dir, periods=args.periods,
                    steps_per_period=args.steps_per_period, max_steps=args.max_steps,
                    force=args.force)
    print(f"wrote {args.output_dir}")
    for key in ("level_cells_per_length", "n_vertices", "n_triangles", "mean_depth",
                "omega_inviscid", "omega_reference", "damping_rate_reference",
                "damping_rate_lamb", "dt", "steps_per_period", "steps", "output_cadence",
                "end_time", "interface_band_vertex_gap_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
