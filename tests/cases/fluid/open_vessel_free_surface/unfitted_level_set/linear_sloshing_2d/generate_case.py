#!/usr/bin/env python3
"""Generate the 2D linear standing-wave sloshing free-surface test."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent
MESH_SUBDIR = Path("mesh/background")

DEFAULT_NX = 32
DEFAULT_NY = 32
DEFAULT_L = 1.0
DEFAULT_H0 = 0.5
DEFAULT_H_TANK = 0.75
DEFAULT_DENSITY = 998.2
DEFAULT_VISCOSITY = 1.0e-5
DEFAULT_GRAVITY = 9.81
DEFAULT_AMPLITUDE = 0.005
DEFAULT_MODE_N = 1
DEFAULT_TIME_STEPS = 240
DEFAULT_OUTPUT_CADENCE = 1


def node_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i


def sloshing_parameters(*, length: float, depth: float, gravity: float, mode_n: int) -> dict[str, float]:
    k = mode_n * math.pi / length
    omega = math.sqrt(gravity * k * math.tanh(k * depth))
    period = 2.0 * math.pi / omega
    return {"k": k, "omega": omega, "period": period}


def eta(x: np.ndarray | float, t: float, *, amplitude: float, k: float, omega: float) -> np.ndarray | float:
    return amplitude * np.cos(k * x) * math.cos(omega * t)


def height(x: np.ndarray | float, t: float, *, depth: float, amplitude: float, k: float, omega: float) -> np.ndarray | float:
    return depth + eta(x, t, amplitude=amplitude, k=k, omega=omega)


def phi_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    depth: float,
    amplitude: float,
    k: float,
    omega: float,
) -> np.ndarray:
    return y - height(x, t, depth=depth, amplitude=amplitude, k=k, omega=omega)


def velocity_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    depth: float,
    amplitude: float,
    k: float,
    omega: float,
) -> np.ndarray:
    denom = math.sinh(k * depth)
    sinwt = math.sin(omega * t)
    u = amplitude * omega * np.cosh(k * y) / denom * np.sin(k * x) * sinwt
    v = -amplitude * omega * np.sinh(k * y) / denom * np.cos(k * x) * sinwt
    out = np.zeros((np.size(x), 3), dtype=float)
    out[:, 0] = np.asarray(u, dtype=float).reshape(-1)
    out[:, 1] = np.asarray(v, dtype=float).reshape(-1)
    return out


def pressure_exact(
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    *,
    density: float,
    gravity: float,
    depth: float,
    amplitude: float,
    k: float,
    omega: float,
) -> np.ndarray:
    denom = math.sinh(k * depth)
    dynamic = density * (amplitude * omega * omega / k) * np.cosh(k * y) / denom * np.cos(k * x) * math.cos(omega * t)
    hydrostatic = density * gravity * (depth - y)
    return hydrostatic + dynamic


def structured_quad_mesh(args: argparse.Namespace, *, k: float, omega: float) -> pv.UnstructuredGrid:
    x = np.linspace(0.0, args.length, args.nx + 1)
    y = np.linspace(0.0, args.tank_height, args.ny + 1)
    points = np.array([[xi, yi, 0.0] for yi in y for xi in x], dtype=float)

    cells = []
    for j in range(args.ny):
        for i in range(args.nx):
            cells.extend(
                [
                    4,
                    node_id(i, j, args.nx),
                    node_id(i + 1, j, args.nx),
                    node_id(i + 1, j + 1, args.nx),
                    node_id(i, j + 1, args.nx),
                ]
            )
    grid = pv.UnstructuredGrid(
        np.array(cells, dtype=np.int64),
        np.full(args.nx * args.ny, int(pv.CellType.QUAD), dtype=np.uint8),
        points,
    )

    px = points[:, 0]
    py = points[:, 1]
    grid.point_data["GlobalNodeID"] = np.arange(points.shape[0], dtype=np.int32)
    grid.point_data["phi"] = phi_exact(
        px,
        py,
        0.0,
        depth=args.depth,
        amplitude=args.amplitude,
        k=k,
        omega=omega,
    )
    grid.point_data["Velocity"] = velocity_exact(
        px,
        py,
        0.0,
        depth=args.depth,
        amplitude=args.amplitude,
        k=k,
        omega=omega,
    )
    grid.point_data["Pressure"] = pressure_exact(
        px,
        py,
        0.0,
        density=args.density,
        gravity=args.gravity,
        depth=args.depth,
        amplitude=args.amplitude,
        k=k,
        omega=omega,
    )
    grid.cell_data["GlobalElementID"] = np.arange(args.nx * args.ny, dtype=np.int32)
    return grid


def line_polydata(grid: pv.UnstructuredGrid, edges: list[tuple[int, int]], parent_cells: list[int]) -> pv.PolyData:
    used = sorted({node for edge in edges for node in edge})
    local = {global_id: local_id for local_id, global_id in enumerate(used)}
    lines = np.array([[2, local[a], local[b]] for a, b in edges], dtype=np.int64).ravel()
    poly = pv.PolyData(grid.points[np.array(used, dtype=np.int64)], lines=lines)
    poly.point_data["GlobalNodeID"] = np.array(used, dtype=np.int32)
    poly.cell_data["GlobalElementID"] = np.array(parent_cells, dtype=np.int32)
    return poly


def boundary_node_ids(nx: int, ny: int) -> dict[str, list[int]]:
    return {
        "wall_left": [node_id(0, j, nx) for j in range(ny + 1)],
        "wall_right": [node_id(nx, j, nx) for j in range(ny + 1)],
        "wall_bottom": [node_id(i, 0, nx) for i in range(nx + 1)],
    }


def write_boundary_surfaces(grid: pv.UnstructuredGrid, nx: int, ny: int, surface_dir: Path) -> None:
    surface_dir.mkdir(parents=True, exist_ok=True)
    specs: dict[str, tuple[list[tuple[int, int]], list[int]]] = {
        "wall_left": (
            [(node_id(0, j, nx), node_id(0, j + 1, nx)) for j in range(ny)],
            [j * nx for j in range(ny)],
        ),
        "wall_right": (
            [(node_id(nx, j, nx), node_id(nx, j + 1, nx)) for j in range(ny)],
            [j * nx + (nx - 1) for j in range(ny)],
        ),
        "wall_bottom": (
            [(node_id(i, 0, nx), node_id(i + 1, 0, nx)) for i in range(nx)],
            list(range(nx)),
        ),
        "wall_top": (
            [(node_id(i, ny, nx), node_id(i + 1, ny, nx)) for i in range(nx)],
            [(ny - 1) * nx + i for i in range(nx)],
        ),
    }
    for name, (edges, parent_cells) in specs.items():
        line_polydata(grid, edges, parent_cells).save(surface_dir / f"{name}.vtp", binary=False)


def time_samples_for_period(period: float, time_step: float) -> list[float]:
    count = max(2, int(math.ceil(period / time_step)) + 1)
    times = [i * period / (count - 1) for i in range(count)]
    times[0] = 0.0
    times[-1] = period
    return times


def write_velocity_bc_files(args: argparse.Namespace, grid: pv.UnstructuredGrid, *, k: float, omega: float, period: float) -> None:
    bc_dir = CASE_DIR / "bc"
    if bc_dir.exists():
        shutil.rmtree(bc_dir)
    bc_dir.mkdir(parents=True)
    times = time_samples_for_period(period, args.time_step)
    for name, ids in boundary_node_ids(args.nx, args.ny).items():
        path = bc_dir / f"{name}_velocity.dat"
        with path.open("w") as output:
            output.write(f"2 {len(times)} {len(ids)}\n")
            for t in times:
                output.write(f"{t:.12e}\n")
            for gid in ids:
                x, y, _ = grid.points[gid]
                values = velocity_exact(
                    np.array([x]),
                    np.array([y]),
                    0.0,
                    depth=args.depth,
                    amplitude=args.amplitude,
                    k=k,
                    omega=omega,
                )
                output.write(f"{gid + 1}\n")
                for t in times:
                    values = velocity_exact(
                        np.array([x]),
                        np.array([y]),
                        t,
                        depth=args.depth,
                        amplitude=args.amplitude,
                        k=k,
                        omega=omega,
                    )
                    output.write(f"{values[0, 0]:.18e} {values[0, 1]:.18e}\n")


def write_solver_xml(args: argparse.Namespace, *, final_time: float) -> None:
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{args.time_steps}</Number_of_time_steps>
  <Time_step_size>{args.time_step:.12g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>

  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Combine_time_series>true</Combine_time_series>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>{args.output_cadence}</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>{args.output_cadence}</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{args.time_steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="tank">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
  <Add_face name="wall_left"><Face_file_path>mesh/background/mesh-surfaces/wall_left.vtp</Face_file_path></Add_face>
  <Add_face name="wall_right"><Face_file_path>mesh/background/mesh-surfaces/wall_right.vtp</Face_file_path></Add_face>
  <Add_face name="wall_bottom"><Face_file_path>mesh/background/mesh-surfaces/wall_bottom.vtp</Face_file_path></Add_face>
  <Add_face name="wall_top"><Face_file_path>mesh/background/mesh-surfaces/wall_top.vtp</Face_file_path></Add_face>
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>1.0e-5</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>coupled_field</Velocity_source>
  <Velocity_field_name>Velocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Use_wet_extension_advection_velocity>false</Use_wet_extension_advection_velocity>
  <Enable_SUPG>false</Enable_SUPG>
  <Interface_kinematic_marker>1030234</Interface_kinematic_marker>
  <Interface_kinematic_weight_scale>1.0</Interface_kinematic_weight_scale>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <Enable_reinitialization>false</Enable_reinitialization>
  <Reinitialization_method>projection</Reinitialization_method>
  <Reinitialization_cadence_steps>10</Reinitialization_cadence_steps>
  <Reinitialization_max_iterations>4</Reinitialization_max_iterations>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type="Volume_integral"><Volume>true</Volume></Output>
  <LS type="Direct">
    <Linear_algebra type="eigen"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-6</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>20</Max_iterations>
  <Tolerance>5.0e-3</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>{args.density:.12g}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>{-args.gravity:.12g}</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_field_name>Pressure</Hydrostatic_pressure_field_name>
  <Viscosity model="Constant"><Value>{args.viscosity:.12g}</Value></Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>
  <Output type="Volume_integral"><Volume>true</Volume></Output>
  <LS type="Direct">
    <Linear_algebra type="eigen"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-5</Tolerance>
    <Absolute_tolerance>1.0e-8</Absolute_tolerance>
    <NS_GM_max_iterations>150</NS_GM_max_iterations>
    <NS_GM_tolerance>1.0e-5</NS_GM_tolerance>
    <NS_CG_max_iterations>150</NS_CG_max_iterations>
    <NS_CG_tolerance>1.0e-5</NS_CG_tolerance>
    <NS_min_outer_iterations>1</NS_min_outer_iterations>
    <NS_Schur_preconditioner>blockdiag-l</NS_Schur_preconditioner>
    <NS_Momentum_approximation>ilu-k</NS_Momentum_approximation>
    <NS_Use_coupled_outer_FGMRES>true</NS_Use_coupled_outer_FGMRES>
  </LS>

  <Add_BC name="wall_left">
    <Type>Dir</Type>
    <Time_dependence>General</Time_dependence>
    <Temporal_and_spatial_values_file_path>bc/wall_left_velocity.dat</Temporal_and_spatial_values_file_path>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Time_dependence>General</Time_dependence>
    <Temporal_and_spatial_values_file_path>bc/wall_right_velocity.dat</Temporal_and_spatial_values_file_path>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Dir</Type>
    <Time_dependence>General</Time_dependence>
    <Temporal_and_spatial_values_file_path>bc/wall_bottom_velocity.dat</Temporal_and_spatial_values_file_path>
  </Add_BC>

  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>linear_sloshing_surface</Generated_interface_domain_id>
    <Interface_marker>1030234</Interface_marker>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Enable_velocity_extension>true</Enable_velocity_extension>
    <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_velocity_gradient_penalty>1.0</Cut_cell_velocity_gradient_penalty>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
"""
    (CASE_DIR / "solver.xml").write_text(xml)


def write_expected(args: argparse.Namespace, *, k: float, omega: float, period: float, final_time: float) -> None:
    interface_pressure_abs = max(20.0, 0.5 * args.density * args.gravity * args.amplitude)
    tolerances = {
        "interface_mean_abs": 1.0e-2,
        "interface_amplitude_relative": 0.25,
        "interface_l2_height_abs": 1.0e-2,
        "interface_max_height_abs": 2.0e-2,
        "area_relative": 5.0e-2,
        "velocity_relative_l2": 0.35,
        "velocity_max_abs": 0.08,
        "pressure_relative_rms": 0.35,
        "pressure_rms_after_offset_relative": 0.25,
        "interface_pressure_abs": interface_pressure_abs,
    }
    expected = {
        "case": "linear_sloshing_2d",
        "purpose": "Small-amplitude linear standing-wave free-surface regression test for the OOP unfitted level-set path.",
        "mesh": {
            "nx": args.nx,
            "ny": args.ny,
            "length": args.length,
            "mean_depth": args.depth,
            "tank_height": args.tank_height,
            "element_size": max(args.length / args.nx, args.tank_height / args.ny),
        },
        "fluid": {
            "density": args.density,
            "dynamic_viscosity": args.viscosity,
            "kinematic_viscosity": args.viscosity / args.density,
            "gravity": args.gravity,
            "body_force": [0.0, -args.gravity, 0.0],
            "force_sign_convention": "OOP residual uses rho*(... - f) + grad(p) - div(stress), so hydrostatic balance is grad(p)=rho*f.",
        },
        "run": {
            "time_step": args.time_step,
            "time_steps": args.time_steps,
            "final_time": final_time,
            "output_cadence": args.output_cadence,
            "combine_time_series": True,
        },
        "analytic_solution": {
            "L": args.length,
            "H0": args.depth,
            "H_tank": args.tank_height,
            "amplitude": args.amplitude,
            "mode_n": args.mode_n,
            "k": k,
            "omega": omega,
            "period": period,
            "eta": "A*cos(k*x)*cos(omega*t)",
            "level_set": "phi=y-(H0+A*cos(k*x)*cos(omega*t))",
            "velocity_potential": "-(A*omega/k)*(cosh(k*y)/sinh(k*H0))*cos(k*x)*sin(omega*t)",
            "pressure": "rho*g*(H0-y)+rho*(A*omega^2/k)*(cosh(k*y)/sinh(k*H0))*cos(k*x)*cos(omega*t)",
            "expected_area": args.length * args.depth,
        },
        "boundary_conditions": {
            "free_surface": {
                "implementation": "UnfittedLevelSet",
                "active_domain": "LevelSetNegative",
                "external_pressure": 0.0,
                "surface_tension": 0.0,
            },
            "solid_walls": {
                "realization": "exact_time_space_dirichlet_velocity",
                "note": "The linear theory assumes impermeable slip walls. The current OOP XML path supports exact time/space Dirichlet data, so the wall velocity components are prescribed from the analytic potential-flow solution.",
            },
        },
        "verification": {
            "checked_fields": ["phi", "Velocity", "Pressure"],
            "wet_region": "phi_exact < -2*h_mesh and finite result values",
            "suggested_tolerances": tolerances,
        },
        "suggested_tolerances": tolerances,
        "assumptions": [
            "linearized free-surface kinematics and dynamics",
            "inviscid potential-flow analytic solution",
            "small amplitude",
            "low positive viscosity is used because the OOP constant-viscosity parser requires mu > 0",
        ],
    }
    (CASE_DIR / "expected_results.json").write_text(json.dumps(expected, indent=2) + "\n")


def write_benchmark(args: argparse.Namespace, *, omega: float, period: float, final_time: float) -> None:
    benchmark = {
        "name": "linear_sloshing_2d",
        "type": "fluid_free_surface_unfitted_level_set",
        "solver": "new_oop",
        "description": "Small-amplitude 2D standing sloshing wave with exact wall velocity data.",
        "mesh": {"nx": args.nx, "ny": args.ny},
        "run": {"dt": args.time_step, "steps": args.time_steps, "final_time": final_time},
        "targets": {
            "omega": omega,
            "period": period,
            "final_phase": "cos(omega*t) should be approximately -1 for the default half-period run.",
            "most_important_metrics": [
                "interface_cos_coeff",
                "area_relative_error",
                "pressure_rms_error_after_constant_offset_removal",
                "interface_pressure_rms",
            ],
        },
    }
    (CASE_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")


def write_readme(args: argparse.Namespace, *, omega: float, period: float, final_time: float) -> None:
    readme = f"""# Linear Sloshing 2D

This is a small-amplitude standing-wave free-surface regression test for the
new OOP incompressible Navier-Stokes solver with an unfitted level-set active
domain. Negative `phi` denotes liquid.

The analytic reference is the linearized inviscid potential-flow solution in a
rectangular tank. It is exact for impermeable slip walls and zero surface
tension, not for viscous no-slip Navier-Stokes. The solver XML uses exact
time/space Dirichlet velocity data on the left, right, and bottom walls because
that is the currently supported OOP fallback for the slip-wall analytic data.

Default parameters:

- `L = {args.length}`
- `H0 = {args.depth}`
- `H_tank = {args.tank_height}`
- `A = {args.amplitude}`
- `k = {args.mode_n}*pi/L = {args.mode_n * math.pi / args.length:.12g}`
- `omega = {omega:.12g}`
- `period = {period:.12g}`
- `final_time = {final_time:.12g}`

The free surface is

```text
h(x,t) = H0 + A*cos(k*x)*cos(omega*t)
phi(x,y,t) = y - h(x,t)
```

The pressure reference is zero at the free surface to linear order:

```text
p = rho*g*(H0-y)
  + rho*(A*omega^2/k)*(cosh(k*y)/sinh(k*H0))*cos(k*x)*cos(omega*t)
```

## Generate

```bash
python3 generate_case.py
```

Useful smoke-test override:

```bash
python3 generate_case.py --nx 16 --ny 16 --time-steps 120
```

## Run

```bash
/path/to/svmultiphysics solver.xml
```

The XML saves VTK output at the requested cadence and combines the time series
into `result.pvd`.

## Verify

```bash
python3 verify_expected_results.py
```

To check the generated initial condition without running the solver:

```bash
python3 verify_expected_results.py mesh/background/mesh-complete.mesh.vtu --time 0
```

The verifier reconstructs the `phi=0` crossings, deduplicates them, fits the
standing-wave mode, clips the active `phi<=0` liquid area, compares velocity
and pressure in wet nodes, and checks pressure interpolated onto the free
surface. The default tolerances are smoke/regression tolerances for a coarse
mesh, not an accuracy benchmark.

Key metrics:

- `interface_mean`, `interface_cos_coeff`, `interface_sin_coeff`: modal fit of
  the reconstructed free surface.
- `interface_l2_height_error`, `interface_max_height_error`: geometric error
  against the analytic free-surface height.
- `relative_area_error`: active-liquid volume conservation for the zero-mean
  standing wave.
- `velocity_relative_l2_error`: wet-region velocity error against the
  potential-flow field.
- `pressure_relative_rms_error`: absolute-gauge pressure error.
- `pressure_relative_rms_error_after_constant_offset_removal`: pressure-plane
  error after removing one gauge offset.
- `interface_pressure_rms`, `interface_pressure_max_abs`: direct pressure check
  on the reconstructed free surface.
"""
    (CASE_DIR / "README.md").write_text(readme)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--length", type=float, default=DEFAULT_L)
    parser.add_argument("--depth", type=float, default=DEFAULT_H0)
    parser.add_argument("--tank-height", type=float, default=DEFAULT_H_TANK)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--viscosity", type=float, default=DEFAULT_VISCOSITY)
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY)
    parser.add_argument("--amplitude", type=float, default=DEFAULT_AMPLITUDE)
    parser.add_argument("--mode-n", type=int, default=DEFAULT_MODE_N)
    parser.add_argument("--time-step", type=float, default=None)
    parser.add_argument("--time-steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument("--output-cadence", type=int, default=DEFAULT_OUTPUT_CADENCE)
    args = parser.parse_args()

    if args.nx < 2 or args.ny < 2:
        raise ValueError("--nx and --ny must be at least 2")
    if args.depth <= 0.0 or args.tank_height <= args.depth + args.amplitude:
        raise ValueError("--tank-height must leave dry space above the free surface")
    if args.viscosity <= 0.0:
        raise ValueError("the OOP constant-viscosity model requires --viscosity > 0")
    return args


def main() -> None:
    args = parse_args()
    params = sloshing_parameters(length=args.length, depth=args.depth, gravity=args.gravity, mode_n=args.mode_n)
    k = params["k"]
    omega = params["omega"]
    period = params["period"]
    target_final_time = 0.5 * period
    if args.time_step is None:
        args.time_step = target_final_time / args.time_steps
        final_time = target_final_time
    else:
        final_time = args.time_step * args.time_steps

    mesh_dir = CASE_DIR / MESH_SUBDIR
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True)

    grid = structured_quad_mesh(args, k=k, omega=omega)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu", binary=False)
    write_boundary_surfaces(grid, args.nx, args.ny, mesh_dir / "mesh-surfaces")
    write_velocity_bc_files(args, grid, k=k, omega=omega, period=period)
    write_solver_xml(args, final_time=final_time)
    write_expected(args, k=k, omega=omega, period=period, final_time=final_time)
    write_benchmark(args, omega=omega, period=period, final_time=final_time)
    write_readme(args, omega=omega, period=period, final_time=final_time)

    print(f"Generated {CASE_DIR}")
    print(f"  mesh: {mesh_dir / 'mesh-complete.mesh.vtu'}")
    print(f"  dt: {args.time_step:.12g}")
    print(f"  final_time: {final_time:.12g}")
    print(f"  period: {period:.12g}")


if __name__ == "__main__":
    main()
