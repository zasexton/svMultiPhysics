#!/usr/bin/env python3
"""Write one resolution level of the oscillating 2D drop benchmark (tracker M3).

A two-dimensional liquid drop of equilibrium radius R, fully enclosed by its
free surface, with zero gravity and exterior pressure p_ext = 0, is released
from rest with the area-preserving mode-2 shape r = R0 (1 + eps cos(2 theta)),
R0 = R / sqrt(1 + eps^2 / 2).  The history of the cos(2 theta) shape mode is
compared with the exact linear viscous solution (drop_reference.py) by
verify.py: Lamb's frequency and the viscous damping rate.

Time step (protocol fixed on 2026-10-07 before the first protocol run, as
D13/D19 for the capillary wave): the lagged normal-increment capillary term
(Surface_tension_semi_implicit = NormalIncrement) is on, and every level uses
100 steps per inviscid period 2 pi / omega0, one step shared by all levels;
--dt-divisor 2 gives the 200-steps-per-period check.  --transport selects the
level-set advection velocity (decision D9).

The background mesh and its wall faces are those of static_drop_2d (square
box [0, 3R]^2); the drop centre is offset by (sqrt(3), pi)/100 R from the box
centre.  The case is written for the new OOP solver: solver.xml, an affine
Triangle3 mesh with the initial fields, the four wall face files, and
case.json with every parameter that verify.py needs.  See README.md.
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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import drop_reference as reference  # noqa: E402


def _load(directory: str, name: str):
    path = HERE.parent / directory / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{directory}_{name}_for_oscillating_drop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SD = _load("static_drop_2d", "generate_case")       # box mesh, wall faces, VTU/VTP writers
CW = _load("capillary_wave_2d", "generate_case")    # level-set transport and linear-solver blocks

# ---------------------------------------------------------------------------
# Protocol constants.  They are fixed for every level and every capillary form
# (principle P1); README.md gives the source or the derivation of each one.
# ---------------------------------------------------------------------------
LEVELS = (8, 16, 32)                        # R/h
CAPILLARY_FORMS = ("surface_stress", "kag_consistent", "kag_lumped")
DT_DIVISORS = (1, 2, 4)                     # time-step study (D10, D19)
STEPS_PER_PERIOD = 100                      # inviscid period 2 pi / omega0 in 100 steps
SEMI_IMPLICIT_OPTIONS = ("None", "NormalIncrement")
DEFAULT_SEMI_IMPLICIT = "NormalIncrement"   # lagged normal-increment term (D13)
TRANSPORTS = CW.TRANSPORTS                  # level-set advection velocity (D9)
DEFAULT_TRANSPORT = "pde_extension"         # harmonic PDE extension, monolithic (D9, D15)
KINEMATIC_RECONCILIATION = True             # Enable_kinematic_reconciliation (D14)
DENSITY = SD.DENSITY                        # rho = 1
SURFACE_TENSION = SD.SURFACE_TENSION        # gamma = 1
RADIUS = SD.RADIUS                          # R = 1, equilibrium radius (length unit)
BOX_SIDE = SD.BOX_SIDE                      # square box [0, 3R]^2
# Drop centre: (sqrt(3), pi)/100 R from the box centre.  Irrational, so the
# sampled surface cannot pass exactly through a vertex of the nested dyadic
# grids.  Among 15 simple offsets of this kind it keeps the largest vertex
# clearance of the released shape, min |phi|/h = 0.018 / 0.0082 / 0.011 at
# R/h = 8 / 16 / 32 (README.md); the static-drop offset (pi, e)/100 R would
# leave 1.3e-6 at R/h = 32.
CENTRE_OFFSET = (math.sqrt(3.0) / 100.0 * RADIUS, math.pi / 100.0 * RADIUS)
MODE = 2                                    # shape mode n
AMPLITUDE_OVER_RADIUS = 0.01                # eps; shape r = R0 (1 + eps cos(2 theta))
DEFAULT_LAPLACE_NUMBER = 800.0              # La = rho gamma D / mu^2, D = 2R: mu = 0.05
EXTERNAL_PRESSURE = 0.0
DEFAULT_PERIODS = 4.0                       # run length in inviscid periods
DEFAULT_SNAPSHOTS = 100                     # VTU outputs per run
DT_SAFETY = SD.DT_SAFETY                    # one-sided capillary limit, reported only
MIN_PHI_OVER_H_WARNING = 1.0e-6             # "vertex touch" warning threshold
LEVEL_SET_FIELD = "phi"
CURVATURE_FIELD = "kappa"
INTERFACE_DOMAIN_ID = "oscillating_drop_surface"
WALLS = SD.WALLS                            # dry no-slip box walls


def viscosity_from_laplace(laplace: float) -> float:
    """mu from La = rho*gamma*D/mu^2 with D = 2R (as static_drop_2d)."""
    return math.sqrt(DENSITY * SURFACE_TENSION * 2.0 * RADIUS / laplace)


def area_preserving_radius(eps: float = AMPLITUDE_OVER_RADIUS) -> float:
    """R0 with area of r < R0 (1 + eps cos(n theta)) equal to pi R^2."""
    return RADIUS / math.sqrt(1.0 + 0.5 * eps * eps)


def drop_centre() -> np.ndarray:
    return np.array([0.5 * BOX_SIDE + CENTRE_OFFSET[0], 0.5 * BOX_SIDE + CENTRE_OFFSET[1]])


def physical_parameters(laplace: float) -> dict:
    mu = viscosity_from_laplace(laplace)
    nu = mu / DENSITY
    params = dict(mode=MODE, kinematic_viscosity=nu, surface_tension=SURFACE_TENSION,
                  density=DENSITY, radius=RADIUS)
    mode = reference.normal_mode(**params)
    weak = reference.weak_viscosity_mode(**params)
    omega0 = mode["omega0"]
    return {
        "viscosity": mu,
        "kinematic_viscosity": nu,
        "ohnesorge_number": mu / math.sqrt(DENSITY * SURFACE_TENSION * RADIUS),
        "omega0": omega0,
        "inviscid_period": 2.0 * math.pi / omega0,
        "epsilon": weak["epsilon"],
        "weak_damping_rate": mode["weak_damping_rate"],
        "weak_viscosity_omega": weak["omega"],
        "weak_viscosity_damping_rate": weak["beta"],
        "normal_mode_omega": mode["omega"],
        "normal_mode_damping_rate": mode["beta"],
        "boundary_layer_thickness": math.sqrt(2.0 * nu / omega0),
        "viscous_time": DENSITY * RADIUS ** 2 / mu,
    }


def time_schedule(level: int, laplace: float, periods: float, snapshots: int,
                  dt_divisor: int = 1) -> dict:
    """Time step shared by all levels, refined by dt_divisor.

    dt = T0 / STEPS_PER_PERIOD with T0 = 2 pi / omega0; the run of `periods`
    periods must be a whole number of such steps, and the output cadence is
    the largest divisor of the step count that still gives at least
    `snapshots` outputs.  A divisor d refines the step exactly to dt/d (d
    times the steps and the cadence), so the output times are unchanged.
    """
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    h = RADIUS / level
    phys = physical_parameters(laplace)
    end_time = periods * phys["inviscid_period"]
    base_steps = round(periods * STEPS_PER_PERIOD)
    if base_steps < 1 or not math.isclose(base_steps, periods * STEPS_PER_PERIOD, abs_tol=1e-9):
        raise ValueError(f"--periods times {STEPS_PER_PERIOD} steps per period must be a whole "
                         "number of steps")
    base_cadence = max([c for c in range(1, base_steps + 1)
                        if base_steps % c == 0 and base_steps // c >= min(snapshots, base_steps)])
    steps = base_steps * dt_divisor
    dt = end_time / steps
    capillary_limit = SD.capillary_dt_limit(h)
    return {
        "h": h,
        "end_time": end_time,
        "dt_capillary_limit": capillary_limit,
        "dt": dt,
        "dt_over_capillary_limit": dt / (DT_SAFETY * capillary_limit),
        "steps_per_period": steps / periods,
        "steps": steps,
        "output_cadence": base_cadence * dt_divisor,
        **phys,
    }


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
def polar(points: np.ndarray):
    c = drop_centre()
    dx, dy = points[:, 0] - c[0], points[:, 1] - c[1]
    return np.hypot(dx, dy), np.arctan2(dy, dx), dx, dy


def surface_radius(theta, eps: float = AMPLITUDE_OVER_RADIUS):
    """rho(theta) = R0 (1 + eps cos(n theta)), the released shape."""
    return area_preserving_radius(eps) * (1.0 + eps * np.cos(MODE * theta))


def initial_level_set(points: np.ndarray, eps: float = AMPLITUDE_OVER_RADIUS) -> np.ndarray:
    """phi = r - R0 - R0 eps cos(n theta) (r / rho(theta))^2; liquid where phi < 0.

    The zero set is exactly r = rho(theta).  The perturbation is written with
    the factor (r / rho)^2 = (x^2 - y^2 terms) so that phi is continuous with a
    continuous gradient at the drop centre (r - rho(theta) alone would jump
    there); near the surface |grad phi| = 1 + O(2 eps).
    """
    r, theta, _, _ = polar(points)
    r0 = area_preserving_radius(eps)
    rho = surface_radius(theta, eps)
    return r - r0 - r0 * eps * np.cos(MODE * theta) * (r / rho) ** 2


def initial_pressure(points: np.ndarray, eps: float = AMPLITUDE_OVER_RADIUS) -> np.ndarray:
    """Linear pressure of the released state (u = 0, t = 0+).

    Harmonic, and equal to p_ext + gamma kappa = gamma/R + gamma (n^2 - 1)
    a0 cos(n theta) / R^2 on r = R, with a0 = R0 eps:
    p = gamma/R + gamma (n^2 - 1) a0 (r/R)^n cos(n theta) / R^2, a polynomial
    continued smoothly through the dry vertices that cut cells need.
    """
    r, theta, _, _ = polar(points)
    a0 = area_preserving_radius(eps) * eps
    return (EXTERNAL_PRESSURE + SURFACE_TENSION / RADIUS
            + SURFACE_TENSION * (MODE ** 2 - 1) * a0 / RADIUS ** 2
            * (r / RADIUS) ** MODE * np.cos(MODE * theta))


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def solver_xml(form: str, schedule: dict, steps: int, cadence: int,
               transport: str = DEFAULT_TRANSPORT,
               kinematic_reconciliation: bool = KINEMATIC_RECONCILIATION,
               semi_implicit: str = DEFAULT_SEMI_IMPLICIT) -> str:
    if semi_implicit not in SEMI_IMPLICIT_OPTIONS:
        raise ValueError(f"semi_implicit must be one of {SEMI_IMPLICIT_OPTIONS}")
    if form not in CAPILLARY_FORMS:
        raise ValueError(f"capillary form must be one of {CAPILLARY_FORMS}")
    semi_implicit_bc = ("" if semi_implicit == "None" else
                        f"\n      <Surface_tension_semi_implicit>{semi_implicit}"
                        "</Surface_tension_semi_implicit>")
    velocity = CW.level_set_velocity_block(transport)
    reconciliation = ("\n    <Enable_kinematic_reconciliation>true</Enable_kinematic_reconciliation>"
                      if kinematic_reconciliation else "")
    kag = form in ("kag_consistent", "kag_lumped")
    curvature_projection = ""
    if kag:
        mass = ("\n    <Curvature_projection_kinematic_area_gradient_mass>Lumped"
                "</Curvature_projection_kinematic_area_gradient_mass>"
                if form == "kag_lumped" else "")
        curvature_projection = f"""
    <Enable_curvature_projection>true</Enable_curvature_projection>
    <Curvature_field_name>{CURVATURE_FIELD}</Curvature_field_name>
    <Curvature_projection_recovery_mode>KinematicAreaGradient</Curvature_projection_recovery_mode>
    <Curvature_projection_kinematic_area_gradient_filter_coefficient>0.0</Curvature_projection_kinematic_area_gradient_filter_coefficient>{mass}
    <Curvature_projection_cadence_steps>1</Curvature_projection_cadence_steps>"""
    tension_form = "KinematicAreaGradientTraction" if kag else "SurfaceStress"
    curvature_bc = (f"\n      <Curvature_field>{CURVATURE_FIELD}</Curvature_field>" if kag else "")
    walls = "\n".join(f"""    <Add_BC name="{w}">
      <Type>Dir</Type>
      <Value>0.0</Value>
    </Add_BC>""" for w in WALLS)
    faces = "\n".join(
        f'    <Add_face name="{w}"><Face_file_path>mesh/mesh-surfaces/{w}.vtp</Face_file_path></Add_face>'
        for w in WALLS)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- oscillating_drop_2d benchmark, capillary form {form}, transport {transport}; generated by generate_case.py -->
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

  <Add_mesh name="box">
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
{velocity}
    <Enable_SUPG>true</Enable_SUPG>
    <SUPG_tau_scale>0.5</SUPG_tau_scale>
    <SUPG_transient_scale>2.0</SUPG_transient_scale>
    <Enable_reinitialization>false</Enable_reinitialization>
    <Enable_volume_correction>false</Enable_volume_correction>{reconciliation}{curvature_projection}
    <Output type="Spatial">
      <Level_set>true</Level_set>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{CW.fsils_gmres_block()}
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
    <Force_y>0.0</Force_y>
    <Force_z>0.0</Force_z>
    <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
    <Viscosity model="Constant">
      <Value>{schedule['viscosity']:.17g}</Value>
    </Viscosity>
    <Output type="Spatial">
      <Velocity>true</Velocity>
      <Pressure>true</Pressure>
    </Output>
    <Output type="Volume_integral">
      <Volume>true</Volume>
    </Output>
{CW.fsils_gmres_block()}
{walls}
    <Add_BC name="free_surface">
      <Type>Free_surface</Type>
      <Implementation>UnfittedLevelSet</Implementation>
      <Level_set_field_name>{LEVEL_SET_FIELD}</Level_set_field_name>
      <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
      <Level_set_isovalue>0.0</Level_set_isovalue>
      <Active_domain>LevelSetNegative</Active_domain>
      <Active_domain_method>CutVolume</Active_domain_method>
      <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
      <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
      <Interface_quadrature_order>2</Interface_quadrature_order>
      <External_pressure>{EXTERNAL_PRESSURE:.17g}</External_pressure>
      <Surface_tension>{SURFACE_TENSION:.17g}</Surface_tension>
      <Surface_tension_form>{tension_form}</Surface_tension_form>{semi_implicit_bc}{curvature_bc}
      <Use_level_set_curvature>false</Use_level_set_curvature>
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
def generate(level: int, form: str, output_dir: Path, *,
             laplace: float = DEFAULT_LAPLACE_NUMBER,
             periods: float = DEFAULT_PERIODS,
             snapshots: int = DEFAULT_SNAPSHOTS,
             dt_divisor: int = 1, transport: str = DEFAULT_TRANSPORT,
             kinematic_reconciliation: bool = KINEMATIC_RECONCILIATION,
             max_steps: int | None = None, force: bool = False,
             semi_implicit: str = DEFAULT_SEMI_IMPLICIT) -> dict:
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if form not in CAPILLARY_FORMS:
        raise ValueError(f"--capillary-form must be one of {CAPILLARY_FORMS}")
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    CW.level_set_velocity_block(transport)              # validates the transport choice
    if semi_implicit not in SEMI_IMPLICIT_OPTIONS:
        raise ValueError(f"--surface-tension-semi-implicit must be one of {SEMI_IMPLICIT_OPTIONS}")
    if not (math.isfinite(laplace) and laplace > 0.0):
        raise ValueError("--laplace-number must be positive and finite")
    if not (periods > 0.0 and snapshots >= 4):
        raise ValueError("need periods > 0 and at least 4 snapshots")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    schedule = time_schedule(level, laplace, periods, snapshots, dt_divisor)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, n_cells_side = SD.structured_triangle_mesh(level)
    phi = initial_level_set(points)
    h = schedule["h"]
    eps = AMPLITUDE_OVER_RADIUS
    r0 = area_preserving_radius(eps)
    centre = drop_centre()
    min_phi_over_h = float(np.min(np.abs(phi)) / h)
    wall_gap_over_h = float((0.5 * BOX_SIDE - r0 * (1.0 + eps)
                             - max(abs(CENTRE_OFFSET[0]), abs(CENTRE_OFFSET[1]))) / h)

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    SD.write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
                 {"GlobalNodeID": ("Int64", np.arange(n_points)),
                  LEVEL_SET_FIELD: ("Float64", phi),
                  "Velocity": ("Float64", np.zeros((n_points, 3))),
                  "Pressure": ("Float64", initial_pressure(points))},
                 {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in WALLS:
        node_ids, parents = faces[wall]
        SD.write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(solver_xml(form, schedule, steps, cadence, transport,
                                                      kinematic_reconciliation, semi_implicit),
                                           encoding="utf-8")

    case = {
        "benchmark": "oscillating_drop_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/oscillating_drop_2d/generate_case.py",
        "level_R_over_h": level,
        "capillary_form": form,
        "transport": transport,
        "kinematic_reconciliation": bool(kinematic_reconciliation),
        "surface_tension_semi_implicit": semi_implicit,
        "sign_definite_patch_bounds": False,
        "laplace_number": laplace,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": schedule["viscosity"],
        "kinematic_viscosity": schedule["kinematic_viscosity"],
        "ohnesorge_number": schedule["ohnesorge_number"],
        "radius": RADIUS,
        "mode": MODE,
        "amplitude_over_radius": eps,
        "area_preserving_radius": r0,
        "initial_amplitude": r0 * eps,
        "centre": centre.tolist(),
        "centre_offset": list(CENTRE_OFFSET),
        "box": [0.0, BOX_SIDE, 0.0, BOX_SIDE],
        "cells_per_side": n_cells_side,
        "h": h,
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "external_pressure": EXTERNAL_PRESSURE,
        "level_set_field": LEVEL_SET_FIELD,
        "interface_domain_id": INTERFACE_DOMAIN_ID,
        "liquid_side": "phi<0",
        "omega0": schedule["omega0"],
        "inviscid_period": schedule["inviscid_period"],
        "epsilon": schedule["epsilon"],
        "weak_damping_rate": schedule["weak_damping_rate"],
        "weak_viscosity_omega": schedule["weak_viscosity_omega"],
        "weak_viscosity_damping_rate": schedule["weak_viscosity_damping_rate"],
        "normal_mode_omega": schedule["normal_mode_omega"],
        "normal_mode_damping_rate": schedule["normal_mode_damping_rate"],
        "boundary_layer_thickness": schedule["boundary_layer_thickness"],
        "boundary_layer_thickness_over_h": schedule["boundary_layer_thickness"] / h,
        "viscous_time": schedule["viscous_time"],
        "periods": periods,
        "end_time_protocol": schedule["end_time"],
        "dt_capillary_limit": schedule["dt_capillary_limit"],
        "dt_safety_factor": DT_SAFETY,
        "dt_rule": "steps-per-period",
        "time_step_rule": (f"shared by all levels: {STEPS_PER_PERIOD} steps per inviscid period "
                           "(protocol of 2026-10-07, as D13/D19)"),
        "dt_divisor": dt_divisor,
        "steps_per_period": schedule["steps_per_period"],
        "dt_over_capillary_limit": schedule["dt_over_capillary_limit"],
        "dt": schedule["dt"],
        "steps_protocol": schedule["steps"],
        "steps": steps,
        "end_time": steps * schedule["dt"],
        "output_cadence": cadence,
        "truncated": truncated,
        "min_abs_phi_over_h": min_phi_over_h,
        "wall_gap_over_h": wall_gap_over_h,
        "result_prefix": "result",
    }
    (output_dir / "case.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    return case


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", type=int, required=True, choices=LEVELS, help="R/h")
    parser.add_argument("--capillary-form", default="surface_stress", choices=CAPILLARY_FORMS)
    parser.add_argument("--transport", default=DEFAULT_TRANSPORT, choices=TRANSPORTS,
                        help="level-set advection velocity (decision D9; default "
                             f"{DEFAULT_TRANSPORT}, the harmonic PDE extension)")
    parser.add_argument("--laplace-number", type=float, default=DEFAULT_LAPLACE_NUMBER,
                        help="La = rho*gamma*D/mu^2, D = 2R (protocol value 800, mu = 0.05)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--periods", type=float, default=DEFAULT_PERIODS,
                        help="run length in inviscid periods 2*pi/omega0 (protocol value 4)")
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS,
                        help="number of VTU outputs over the run (protocol value 100)")
    parser.add_argument("--dt-divisor", type=int, default=1, choices=DT_DIVISORS,
                        help="divide the shared time step by this factor with unchanged output "
                             f"times (2: the {2 * STEPS_PER_PERIOD}-steps-per-period check)")
    parser.add_argument("--surface-tension-semi-implicit", choices=SEMI_IMPLICIT_OPTIONS,
                        default=DEFAULT_SEMI_IMPLICIT,
                        help="Surface_tension_semi_implicit of the free surface (protocol value "
                             f"{DEFAULT_SEMI_IMPLICIT}, D13)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; the case is "
                             "marked truncated and verify.py rejects it for acceptance")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    parser.add_argument("--kinematic-reconciliation", choices=("on", "off"),
                        default="on" if KINEMATIC_RECONCILIATION else "off",
                        help="accepted-step kinematic reconciliation of the level set "
                             "(protocol: on, D14)")
    args = parser.parse_args(argv)

    try:
        case = generate(args.level, args.capillary_form, args.output_dir,
                        laplace=args.laplace_number, periods=args.periods,
                        snapshots=args.snapshots, dt_divisor=args.dt_divisor,
                        transport=args.transport,
                        kinematic_reconciliation=args.kinematic_reconciliation == "on",
                        max_steps=args.max_steps, force=args.force,
                        semi_implicit=args.surface_tension_semi_implicit)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output_dir}")
    for key in ("level_R_over_h", "capillary_form", "transport", "laplace_number", "viscosity",
                "epsilon", "omega0", "normal_mode_omega", "normal_mode_damping_rate",
                "end_time", "dt", "dt_capillary_limit", "dt_over_capillary_limit",
                "dt_divisor", "steps_per_period", "surface_tension_semi_implicit", "steps",
                "output_cadence", "n_vertices", "n_triangles", "min_abs_phi_over_h",
                "wall_gap_over_h", "boundary_layer_thickness_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    if case["min_abs_phi_over_h"] < MIN_PHI_OVER_H_WARNING:
        print(f"WARNING: min |phi|/h = {case['min_abs_phi_over_h']:.3e} < "
              f"{MIN_PHI_OVER_H_WARNING:g}: the sampled surface nearly touches a vertex",
              file=sys.stderr)
    if case["wall_gap_over_h"] < 2.0:
        print("WARNING: the drop is less than two cells from the box wall", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
