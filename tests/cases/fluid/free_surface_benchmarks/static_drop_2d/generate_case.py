#!/usr/bin/env python3
"""Write one resolution level of the 2D static-drop benchmark (tracker M2).

A circular liquid drop of radius R, fully enclosed by its free surface, sits
in a square box with zero gravity and exterior pressure p_ext = 0.  The
sampled analytic state (phi = |x - c| - R, u = 0, p = gamma/R) is released
and the flow relaxes for a fixed number of viscous times (decision D3).

Time step (decision D13, 2026-10-05): the lagged normal-increment capillary
term (Surface_tension_semi_implicit = NormalIncrement) is on, and every level
uses one fixed physical step, 0.02 at La = 12 and 0.01 at La = 120.  Other
Laplace numbers keep the earlier capillary-limit rule (a multiple of dt_B).
--dt-divisor 2 halves the step for the time-step check.  The earlier protocol
is reproduced by --dt-multiple <m> --surface-tension-semi-implicit None (m = 2
at La = 12, m = 1 at La = 120).

The case is written for the new OOP solver: solver.xml, an affine Triangle3
background mesh with the initial fields, the four wall face files, and
case.json with every parameter that verify.py needs.  See README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Protocol constants.  They are fixed for every level and every capillary form
# (principle P1); README.md gives the source or the derivation of each one.
# ---------------------------------------------------------------------------
LEVELS = (8, 16, 32, 64)                    # R/h
CAPILLARY_FORMS = ("surface_stress", "kag_consistent", "kag_lumped")
DENSITY = 1.0                               # rho
SURFACE_TENSION = 1.0                       # gamma
RADIUS = 1.0                                # R (length unit)
BOX_SIDE = 3.0 * RADIUS                     # square box [0, 3R]^2
# Fixed, non-grid-aligned centre: (pi, e)/100 * R from the box centre.  The
# offsets are transcendental, so the circle cannot pass exactly through a
# vertex of any of the nested dyadic grids h = R/level.
CENTRE_OFFSET = (math.pi / 100.0 * RADIUS, math.e / 100.0 * RADIUS)
EXTERNAL_PRESSURE = 0.0
DEFAULT_VISCOUS_TIMES = 5.0                 # run length in units rho*R^2/mu
DEFAULT_SNAPSHOTS = 100                     # VTU outputs per run
# Capillary time-step limit (Brackbill, Kothe & Zemach 1992) evaluated with
# the one-sided density sum of a free surface (rho_liquid + rho_void = rho):
#   dt <= sqrt(rho h^3 / (4 pi gamma)) = SAFETY * sqrt(rho h^3 / (2 pi gamma)).
DT_SAFETY = 1.0 / math.sqrt(2.0)
# Protocol time step per Laplace number (decision D13, 2026-10-05): one fixed
# physical step for every level, possible because the lagged normal-increment
# capillary term removes the capillary limit.  Validation: design note
# Documentation/free_surface_semi_implicit_surface_tension_design.md, section 9.
PROTOCOL_DT = {12.0: 0.02, 120.0: 0.01}
# Earlier rule, kept for the Laplace numbers without a fixed step and for
# --dt-multiple: a multiple of the capillary limit dt_B per Laplace number.  The
# step-0 measurement (tracker, 2026-09-30, jobs 46075447 and 46076505) found
# that the outer geometry loop accepts 2 dt_B at La = 12 and dt_B at La = 120
# with its default 12-pass cap.  Other Laplace numbers use dt_B.
DT_MULTIPLE = {12.0: 2.0, 120.0: 1.0}
# Time-step check (D13): the protocol step divided by 1 or 2; output times and
# end time are unchanged.
DT_DIVISORS = (1, 2)
# Level-set advection velocity: the fluid velocity itself (coupled_field) or
# the PDE extension of it into the dry region (pde_harmonic, pde_normal).
LEVEL_SET_VELOCITY = ("coupled_field",
                      "pde_harmonic_monolithic", "pde_harmonic_prescribed",
                      "pde_normal_monolithic", "pde_normal_prescribed")
DEFAULT_LEVEL_SET_VELOCITY = "pde_harmonic_monolithic"   # decision D9
# Accepted-step kinematic reconciliation of the transported level set
# (Enable_kinematic_reconciliation; FE/LevelSet/LevelSetKinematicReconciliation.h):
# each step's change of the sharp liquid area equals the interface flux of the
# transport velocity.  Local and parameter-free; "off" reproduces earlier decks.
KINEMATIC_RECONCILIATION = True
# Semi-implicit capillary term (Surface_tension_semi_implicit, decision D13);
# the protocol value is NormalIncrement since 2026-10-05 (None before).
SEMI_IMPLICIT_OPTIONS = ("None", "NormalIncrement")
DEFAULT_SEMI_IMPLICIT = "NormalIncrement"
MIN_PHI_OVER_H_WARNING = 1.0e-6             # "vertex touch" warning threshold
LEVEL_SET_FIELD = "phi"
CURVATURE_FIELD = "kappa"
INTERFACE_DOMAIN_ID = "static_drop_surface"
WALLS = ("wall_left", "wall_right", "wall_bottom", "wall_top")


TIME_STEP_RULES = {
    "protocol_fixed": "fixed physical step for every level, PROTOCOL_DT[La] (D13, 2026-10-05)",
    "protocol_capillary_limit": "no fixed D13 step at this La: multiple of the capillary limit "
                                "dt_B, rounded down to the output intervals (earlier rule)",
    "fixed": "fixed step given by --dt",
    "multiple_override": "multiple of the capillary limit dt_B given by --dt-multiple, rounded "
                         "down to the output intervals (earlier protocol)",
}


def viscosity_from_laplace(laplace: float) -> float:
    """mu from La = rho*gamma*D/mu^2 with D = 2R."""
    return math.sqrt(DENSITY * SURFACE_TENSION * 2.0 * RADIUS / laplace)


def capillary_dt_limit(h: float) -> float:
    """Tracker form sqrt(rho h^3 / (2 pi gamma)), before the safety factor."""
    return math.sqrt(DENSITY * h ** 3 / (2.0 * math.pi * SURFACE_TENSION))


def dt_multiple(laplace: float) -> float:
    return DT_MULTIPLE.get(float(laplace), 1.0)


def dt_rule(laplace: float, multiple: float | None = None,
            fixed_dt: float | None = None) -> str:
    """Name of the rule that sets the base step (recorded in case.json)."""
    if fixed_dt is not None:
        return "fixed"
    if multiple is not None:
        return "multiple_override"
    return "protocol_fixed" if float(laplace) in PROTOCOL_DT else "protocol_capillary_limit"


def time_schedule(level: int, laplace: float, viscous_times: float,
                  snapshots: int, *, multiple: float | None = None,
                  fixed_dt: float | None = None, dt_divisor: int = 1) -> dict:
    """Protocol schedule, or a time-step study.

    Protocol (D13): the fixed step PROTOCOL_DT[La] at every level; Laplace
    numbers without one use the capillary-limit rule below.  multiple
    overrides that with a multiple of dt_B, rounded down to 100 output
    intervals as in the earlier protocol.  A fixed step (protocol or fixed_dt)
    is kept exactly: the cadence is the nearest whole number of steps per
    output and the run ends at the first output at or after 5 viscous times,
    so runs with different fixed steps share their output times when the steps
    nest.  dt_divisor then refines the base step exactly to dt/d (d times the
    steps and the cadence), so the output times are unchanged.
    """
    if multiple is not None and fixed_dt is not None:
        raise ValueError("give at most one of --dt-multiple and --dt")
    if dt_divisor not in DT_DIVISORS:
        raise ValueError(f"--dt-divisor must be one of {DT_DIVISORS}")
    if multiple is None and fixed_dt is None:
        fixed_dt = PROTOCOL_DT.get(float(laplace))
    schedule = _base_schedule(level, laplace, viscous_times, snapshots, multiple, fixed_dt)
    schedule["dt_base"] = schedule["dt"]
    schedule["dt_divisor"] = dt_divisor
    if dt_divisor != 1:
        schedule["dt"] = schedule["dt"] / dt_divisor
        schedule["steps"] *= dt_divisor
        schedule["output_cadence"] *= dt_divisor
        schedule["dt_multiple_of_capillary_limit"] /= dt_divisor
    return schedule


def _base_schedule(level: int, laplace: float, viscous_times: float, snapshots: int,
                   multiple: float | None, fixed_dt: float | None) -> dict:
    h = RADIUS / level
    mu = viscosity_from_laplace(laplace)
    viscous_time = DENSITY * RADIUS ** 2 / mu
    end_time = viscous_times * viscous_time
    m = dt_multiple(laplace) if multiple is None else multiple
    if not (math.isfinite(m) and m > 0.0):
        raise ValueError("--dt-multiple must be positive")
    if fixed_dt is not None:
        if not (math.isfinite(fixed_dt) and fixed_dt > 0.0):
            raise ValueError("--dt must be positive")
        cadence = max(1, round(end_time / (snapshots * fixed_dt)))
        steps = cadence * math.ceil(end_time / (cadence * fixed_dt) - 1.0e-9)
        return {
            "h": h,
            "viscosity": mu,
            "viscous_time": viscous_time,
            "capillary_time": math.sqrt(DENSITY * RADIUS ** 3 / SURFACE_TENSION),
            "end_time": steps * fixed_dt,
            "dt_capillary_limit": capillary_dt_limit(h),
            "dt_max": fixed_dt,
            "dt_multiple_of_capillary_limit": fixed_dt / (DT_SAFETY * capillary_dt_limit(h)),
            "dt": fixed_dt,
            "steps": steps,
            "output_cadence": cadence,
        }
    dt_max = m * DT_SAFETY * capillary_dt_limit(h)
    cadence = max(1, math.ceil(end_time / (snapshots * dt_max)))
    steps = snapshots * cadence
    return {
        "h": h,
        "viscosity": mu,
        "viscous_time": viscous_time,
        "capillary_time": math.sqrt(DENSITY * RADIUS ** 3 / SURFACE_TENSION),
        "end_time": end_time,
        "dt_capillary_limit": capillary_dt_limit(h),
        "dt_max": dt_max,
        "dt_multiple_of_capillary_limit": m,
        "dt": end_time / steps,
        "steps": steps,
        "output_cadence": cadence,
    }


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------
def structured_triangle_mesh(level: int):
    """Square [0, BOX_SIDE]^2 split into right triangles.

    The diagonal alternates with (i + j) parity, so the pattern is symmetric
    under reflection about the grid lines and has no preferred diagonal.
    """
    n = int(round(BOX_SIDE * level / RADIUS))
    if not math.isclose(n * RADIUS / level, BOX_SIDE, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("box side must be an integer number of cells")
    coords = np.linspace(0.0, BOX_SIDE, n + 1)
    xx, yy = np.meshgrid(coords, coords)            # row j = y index
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])

    def vid(i: int, j: int) -> int:
        return j * (n + 1) + i

    cells = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            if (i + j) % 2 == 0:
                cells += [(a, b, c), (a, c, d)]
            else:
                cells += [(a, b, d), (b, c, d)]
    cells = np.asarray(cells, dtype=np.int64)

    # Parent triangle of each wall segment: the first triangle of cell (i, j)
    # holds the bottom edge; the second holds the top edge; the left and
    # right edges switch with the diagonal parity.
    def parent(i: int, j: int, side: str) -> int:
        even = (i + j) % 2 == 0
        second = {"bottom": False, "top": True, "left": even, "right": not even}[side]
        return 2 * (j * n + i) + int(second)

    faces = {
        "wall_bottom": ([vid(i, 0) for i in range(n + 1)],
                        [parent(i, 0, "bottom") for i in range(n)]),
        "wall_top": ([vid(i, n) for i in range(n + 1)],
                     [parent(i, n - 1, "top") for i in range(n)]),
        "wall_left": ([vid(0, j) for j in range(n + 1)],
                      [parent(0, j, "left") for j in range(n)]),
        "wall_right": ([vid(n, j) for j in range(n + 1)],
                       [parent(n - 1, j, "right") for j in range(n)]),
    }
    return points, cells, faces, n


def _ascii(values, fmt: str) -> str:
    flat = np.asarray(values).ravel()
    lines = []
    for start in range(0, flat.size, 6):
        lines.append(" ".join(fmt.format(v) for v in flat[start:start + 6]))
    return "\n".join(lines)


def write_vtu(path: Path, points, cells, point_data: dict, cell_data: dict) -> None:
    n_cells = cells.shape[0]
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<UnstructuredGrid>',
           f'<Piece NumberOfPoints="{points.shape[0]}" NumberOfCells="{n_cells}">',
           '<PointData>']
    for name, (vtk_type, values) in point_data.items():
        values = np.asarray(values)
        ncomp = 1 if values.ndim == 1 else values.shape[1]
        fmt = "{:d}" if vtk_type.startswith("Int") else "{:.17g}"
        out.append(f'<DataArray type="{vtk_type}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">')
        out.append(_ascii(values, fmt))
        out.append('</DataArray>')
    out.append('</PointData>')
    out.append('<CellData>')
    for name, (vtk_type, values) in cell_data.items():
        out.append(f'<DataArray type="{vtk_type}" Name="{name}" format="ascii">')
        out.append(_ascii(values, "{:d}"))
        out.append('</DataArray>')
    out.append('</CellData>')
    out.append('<Points>')
    out.append('<DataArray type="Float64" NumberOfComponents="3" format="ascii">')
    out.append(_ascii(points, "{:.17g}"))
    out.append('</DataArray>')
    out.append('</Points>')
    out.append('<Cells>')
    out.append('<DataArray type="Int64" Name="connectivity" format="ascii">')
    out.append(_ascii(cells, "{:d}"))
    out.append('</DataArray>')
    out.append('<DataArray type="Int64" Name="offsets" format="ascii">')
    out.append(_ascii(3 * np.arange(1, n_cells + 1), "{:d}"))
    out.append('</DataArray>')
    out.append('<DataArray type="UInt8" Name="types" format="ascii">')
    out.append(_ascii(np.full(n_cells, 5), "{:d}"))           # VTK_TRIANGLE
    out.append('</DataArray>')
    out.append('</Cells>')
    out.append('</Piece>')
    out.append('</UnstructuredGrid>')
    out.append('</VTKFile>')
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_face_vtp(path: Path, points, node_ids, parent_cells) -> None:
    node_ids = np.asarray(node_ids, dtype=np.int64)
    n_lines = node_ids.size - 1
    local = np.column_stack([np.arange(n_lines), np.arange(1, n_lines + 1)])
    out = ['<?xml version="1.0"?>',
           '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
           '<PolyData>',
           f'<Piece NumberOfPoints="{node_ids.size}" NumberOfVerts="0" NumberOfLines="{n_lines}" '
           'NumberOfStrips="0" NumberOfPolys="0">',
           '<PointData>',
           '<DataArray type="Int64" Name="GlobalNodeID" format="ascii">',
           _ascii(node_ids, "{:d}"),
           '</DataArray>',
           '</PointData>',
           '<CellData>',
           '<DataArray type="Int64" Name="GlobalElementID" format="ascii">',
           _ascii(parent_cells, "{:d}"),
           '</DataArray>',
           '</CellData>',
           '<Points>',
           '<DataArray type="Float64" NumberOfComponents="3" format="ascii">',
           _ascii(points[node_ids], "{:.17g}"),
           '</DataArray>',
           '</Points>',
           '<Lines>',
           '<DataArray type="Int64" Name="connectivity" format="ascii">',
           _ascii(local, "{:d}"),
           '</DataArray>',
           '<DataArray type="Int64" Name="offsets" format="ascii">',
           _ascii(2 * np.arange(1, n_lines + 1), "{:d}"),
           '</DataArray>',
           '</Lines>',
           '</Piece>',
           '</PolyData>',
           '</VTKFile>']
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Solver input
# ---------------------------------------------------------------------------
def fsils_gmres_block() -> str:
    return """    <LS type="GMRES">
      <Linear_algebra type="fsils">
        <Preconditioner>rcs</Preconditioner>
      </Linear_algebra>
      <Max_iterations>100</Max_iterations>
      <Krylov_space_dimension>50</Krylov_space_dimension>
      <Tolerance>1.0e-8</Tolerance>
      <Absolute_tolerance>1.0e-10</Absolute_tolerance>
    </LS>"""


def level_set_velocity_block(mode: str) -> str:
    if mode == "coupled_field":
        return """    <Velocity_source>coupled_field</Velocity_source>
    <Velocity_field_name>Velocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>"""
    method, coupling = mode.rsplit("_", 1)
    return f"""    <Velocity_source>prescribed_data</Velocity_source>
    <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
    <Auto_register_velocity_field>true</Auto_register_velocity_field>
    <Source_velocity_field_name>Velocity</Source_velocity_field_name>
    <Advection_velocity_extension_method>{method}</Advection_velocity_extension_method>
    <Advection_velocity_extension_coupling>{coupling}</Advection_velocity_extension_coupling>"""


def solver_xml(form: str, schedule: dict, steps: int, cadence: int,
               level_set_velocity: str = DEFAULT_LEVEL_SET_VELOCITY,
               kinematic_reconciliation: bool = KINEMATIC_RECONCILIATION,
               semi_implicit: str = DEFAULT_SEMI_IMPLICIT) -> str:
    if semi_implicit not in SEMI_IMPLICIT_OPTIONS:
        raise ValueError(f"semi_implicit must be one of {SEMI_IMPLICIT_OPTIONS}")
    semi_implicit_bc = ("" if semi_implicit == "None" else
                        f"\n      <Surface_tension_semi_implicit>{semi_implicit}"
                        "</Surface_tension_semi_implicit>")
    kag = form in ("kag_consistent", "kag_lumped")
    reconciliation = ("\n    <Enable_kinematic_reconciliation>true</Enable_kinematic_reconciliation>"
                      if kinematic_reconciliation else "")
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
<!-- static_drop_2d benchmark, capillary form {form}; generated by generate_case.py -->
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
{level_set_velocity_block(level_set_velocity)}
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
{fsils_gmres_block()}
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
{fsils_gmres_block()}
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
def generate(level: int, form: str, laplace: float, output_dir: Path, *,
             viscous_times: float = DEFAULT_VISCOUS_TIMES,
             snapshots: int = DEFAULT_SNAPSHOTS,
             level_set_velocity: str = DEFAULT_LEVEL_SET_VELOCITY,
             kinematic_reconciliation: bool = KINEMATIC_RECONCILIATION,
             max_steps: int | None = None, force: bool = False,
             dt_multiple_override: float | None = None,
             fixed_dt: float | None = None,
             semi_implicit: str = DEFAULT_SEMI_IMPLICIT,
             dt_divisor: int = 1) -> dict:
    if level_set_velocity not in LEVEL_SET_VELOCITY:
        raise ValueError(f"--level-set-velocity must be one of {LEVEL_SET_VELOCITY}")
    if level not in LEVELS:
        raise ValueError(f"--level must be one of {LEVELS}")
    if form not in CAPILLARY_FORMS:
        raise ValueError(f"--capillary-form must be one of {CAPILLARY_FORMS}")
    if not (math.isfinite(laplace) and laplace > 0.0):
        raise ValueError("--laplace-number must be positive and finite")
    if not (viscous_times > 0.0 and snapshots >= 4):
        raise ValueError("need viscous_times > 0 and at least 4 snapshots")
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty (use --force)")

    if semi_implicit not in SEMI_IMPLICIT_OPTIONS:
        raise ValueError(f"--surface-tension-semi-implicit must be one of {SEMI_IMPLICIT_OPTIONS}")
    schedule = time_schedule(level, laplace, viscous_times, snapshots,
                             multiple=dt_multiple_override, fixed_dt=fixed_dt,
                             dt_divisor=dt_divisor)
    steps, cadence, truncated = schedule["steps"], schedule["output_cadence"], False
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps must be positive")
        if max_steps < steps:
            steps, cadence, truncated = max_steps, 1, True

    points, cells, faces, n_cells_side = structured_triangle_mesh(level)
    centre = np.array([0.5 * BOX_SIDE + CENTRE_OFFSET[0], 0.5 * BOX_SIDE + CENTRE_OFFSET[1]])
    phi = np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) - RADIUS
    h = schedule["h"]
    min_phi_over_h = float(np.min(np.abs(phi)) / h)
    wall_gap_over_h = float((0.5 * BOX_SIDE - RADIUS - max(abs(CENTRE_OFFSET[0]),
                                                           abs(CENTRE_OFFSET[1]))) / h)
    laplace_pressure = SURFACE_TENSION / RADIUS

    mesh_dir = output_dir / "mesh"
    (mesh_dir / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    n_points = points.shape[0]
    write_vtu(mesh_dir / "mesh-complete.mesh.vtu", points, cells,
              {"GlobalNodeID": ("Int64", np.arange(n_points)),
               LEVEL_SET_FIELD: ("Float64", phi),
               "Velocity": ("Float64", np.zeros((n_points, 3))),
               # Sampled analytic state: the constant Laplace pressure on the
               # whole background support (inactive DOFs are pinned by the
               # solver).  A phi-sign mask would add an O(dp/h) gradient in
               # every retained cut cell.
               "Pressure": ("Float64", np.full(n_points, EXTERNAL_PRESSURE + laplace_pressure))},
              {"GlobalElementID": ("Int64", np.arange(cells.shape[0]))})
    for wall in WALLS:
        node_ids, parents = faces[wall]
        write_face_vtp(mesh_dir / "mesh-surfaces" / f"{wall}.vtp", points, node_ids, parents)
    (output_dir / "solver.xml").write_text(solver_xml(form, schedule, steps, cadence,
                                                      level_set_velocity,
                                                      kinematic_reconciliation,
                                                      semi_implicit),
                                           encoding="utf-8")

    case = {
        "benchmark": "static_drop_2d",
        "generator": "tests/cases/fluid/free_surface_benchmarks/static_drop_2d/generate_case.py",
        "level_R_over_h": level,
        "capillary_form": form,
        "laplace_number": laplace,
        "density": DENSITY,
        "surface_tension": SURFACE_TENSION,
        "viscosity": schedule["viscosity"],
        "radius": RADIUS,
        "centre": centre.tolist(),
        "centre_offset": list(CENTRE_OFFSET),
        "box": [0.0, BOX_SIDE, 0.0, BOX_SIDE],
        "cells_per_side": n_cells_side,
        "h": h,
        "n_vertices": int(n_points),
        "n_triangles": int(cells.shape[0]),
        "external_pressure": EXTERNAL_PRESSURE,
        "laplace_pressure_nominal": laplace_pressure,
        "level_set_field": LEVEL_SET_FIELD,
        "velocity_field": "Velocity",
        "pressure_field": "Pressure",
        "liquid_side": "phi<0",
        "viscous_time": schedule["viscous_time"],
        "capillary_time": schedule["capillary_time"],
        "viscous_times": viscous_times,
        "end_time_protocol": schedule["end_time"],
        "dt_capillary_limit": schedule["dt_capillary_limit"],
        "dt_safety_factor": DT_SAFETY,
        "dt_multiple_of_capillary_limit": schedule["dt_multiple_of_capillary_limit"],
        "level_set_velocity": level_set_velocity,
        "kinematic_reconciliation": bool(kinematic_reconciliation),
        "surface_tension_semi_implicit": semi_implicit,
        "dt_rule": dt_rule(laplace, dt_multiple_override, fixed_dt),
        "time_step_rule": TIME_STEP_RULES[dt_rule(laplace, dt_multiple_override, fixed_dt)],
        "dt_base": schedule["dt_base"],
        "dt_divisor": dt_divisor,
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
    parser.add_argument("--capillary-form", required=True, choices=CAPILLARY_FORMS)
    parser.add_argument("--laplace-number", type=float, required=True,
                        help="La = rho*gamma*D/mu^2, D = 2R")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--viscous-times", type=float, default=DEFAULT_VISCOUS_TIMES,
                        help="run length in units rho*R^2/mu (protocol value 5)")
    parser.add_argument("--snapshots", type=int, default=DEFAULT_SNAPSHOTS,
                        help="number of VTU outputs over the run (protocol value 100)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke runs only: stop after this many steps; the case is "
                             "marked truncated and verify.py rejects it for acceptance")
    parser.add_argument("--level-set-velocity", choices=LEVEL_SET_VELOCITY,
                        default=DEFAULT_LEVEL_SET_VELOCITY,
                        help=f"level-set advection velocity (protocol value "
                             f"{DEFAULT_LEVEL_SET_VELOCITY})")
    parser.add_argument("--kinematic-reconciliation", choices=("on", "off"),
                        default="on" if KINEMATIC_RECONCILIATION else "off",
                        help="accepted-step kinematic reconciliation of the level set "
                             "(protocol: on; off reproduces the earlier decks)")
    parser.add_argument("--dt-multiple", type=float, default=None,
                        help="multiple of dt_B instead of the protocol step, rounded down to the "
                             "output intervals; with --surface-tension-semi-implicit None this "
                             "reproduces the earlier protocol (m = 2 at La = 12, 1 at La = 120)")
    parser.add_argument("--dt", type=float, default=None,
                        help="time-step study: this exact step; the run ends at the first output "
                             "at or after the protocol end time")
    parser.add_argument("--dt-divisor", type=int, choices=DT_DIVISORS, default=1,
                        help="divide the step by this factor with unchanged output times "
                             "(time-step check of D13; protocol runs use 1 and 2)")
    parser.add_argument("--surface-tension-semi-implicit", choices=SEMI_IMPLICIT_OPTIONS,
                        default=DEFAULT_SEMI_IMPLICIT,
                        help="Surface_tension_semi_implicit of the free surface (protocol value "
                             f"{DEFAULT_SEMI_IMPLICIT}, D13; None reproduces the earlier decks)")
    parser.add_argument("--force", action="store_true", help="allow a non-empty output dir")
    args = parser.parse_args(argv)

    case = generate(args.level, args.capillary_form, args.laplace_number, args.output_dir,
                    viscous_times=args.viscous_times, snapshots=args.snapshots,
                    level_set_velocity=args.level_set_velocity,
                    kinematic_reconciliation=args.kinematic_reconciliation == "on",
                    max_steps=args.max_steps, force=args.force,
                    dt_multiple_override=args.dt_multiple, fixed_dt=args.dt,
                    semi_implicit=args.surface_tension_semi_implicit,
                    dt_divisor=args.dt_divisor)
    print(f"wrote {args.output_dir}")
    for key in ("level_R_over_h", "capillary_form", "laplace_number", "viscosity",
                "viscous_time", "end_time", "dt_rule", "dt", "dt_divisor", "dt_capillary_limit",
                "dt_multiple_of_capillary_limit", "level_set_velocity",
                "kinematic_reconciliation", "surface_tension_semi_implicit", "steps",
                "output_cadence", "n_vertices", "n_triangles", "min_abs_phi_over_h",
                "wall_gap_over_h", "truncated"):
        print(f"  {key} = {case[key]}")
    if case["min_abs_phi_over_h"] < MIN_PHI_OVER_H_WARNING:
        print(f"WARNING: min |phi|/h = {case['min_abs_phi_over_h']:.3e} < "
              f"{MIN_PHI_OVER_H_WARNING:g}: the sampled circle nearly touches a vertex",
              file=sys.stderr)
    if case["wall_gap_over_h"] < 2.0:
        print("WARNING: drop is less than two cells from the box wall", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
