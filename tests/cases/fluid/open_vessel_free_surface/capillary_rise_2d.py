"""Generate and measure the resolved-slip two-dimensional capillary-rise case.

The case is the Omega = 1, resolved Navier-slip half channel of Gruending et
al. (2020); the frozen comparison contract is
tests/cases/fluid/free_surface_wp5_capillary_rise_comparison_v1.json.

Two deck profiles are written (``write_case(..., profile=...)``):

* ``"d4"`` (default, tracker M4): the unfitted wetting configuration of the
  sessile-drop benchmark.  PrescribedAngle contact line (variational Young
  term, angle-preserving scale-only wall maintenance, no repair to the target
  angle, decision D4), Navier slip on the wetted wall and a strong normal-only
  wall velocity, SurfaceStress capillarity with small-cut aggregation, the
  harmonic PDE velocity extension for the level-set transport (D15), the
  accepted-step kinematic reconciliation (D14) and the sign-definite patch
  bounds (D21), no reinitialization, and the generalized-alpha / FSILS GMRES
  solver with the benchmark tolerances (1e-4).
* ``"legacy"``: the 2026-08-30 deck (PrescribedContactAngle, wet-extension
  transport, projection reinitialization every step, discontinuity capturing,
  corner-linearized cut geometry, tolerances 1e-6), reproduced byte for byte;
  run_test05_velocity_growth_smoke.py keeps writing it.

Run as a script, the module writes one comparison level (``--half-gap-cells``
10, 20 or 40) with the time step of the frozen contact-motion limit of that
level; see ``protocol_schedule``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


HALF_GAP_M = 0.005
DOMAIN_HEIGHT_M = 0.04
NOMINAL_INITIAL_HEIGHT_M = 0.01
LIQUID_DENSITY_KG_PER_M3 = 83.1
LIQUID_VISCOSITY_PA_S = 0.01
GRAVITY_M_PER_S2 = 4.17
SURFACE_TENSION_N_PER_M = 0.04
CONTACT_ANGLE_DEGREES = 30.0
SLIP_LENGTH_M = 0.001

# Deck profiles (module docstring).  "legacy" reproduces the 2026-08-30 decks.
PROFILES = ("d4", "legacy")
DEFAULT_PROFILE = "d4"
INTERFACE_DOMAIN_ID = "capillary_rise_surface"
COMPARISON_CONTRACT = (
    Path(__file__).resolve().parents[1]
    / "free_surface_wp5_capillary_rise_comparison_v1.json"
)
# Comparison levels: half-gap cells -> frozen maximum contact motion in cells
# per step (comparison contract, candidate_requirements; slip-to-cell ratios
# 2, 4, 8 with the 1 mm physical slip length).
PROTOCOL_LEVELS = (10, 20, 40)
# Largest apex speed of the frozen reference centre (115 mm/s at t = 0.059 s,
# envelope-v1).  The contact point can move faster than the apex while the
# meniscus reshapes, so the step is sized for CONTACT_SPEED_SAFETY times it.
REFERENCE_MAXIMUM_APEX_SPEED_M_PER_S = 0.115
CONTACT_SPEED_SAFETY = 1.5
# History sampling: one output per millisecond, the step of the comparison
# grid, over its whole support [0, 0.69 s].
OUTPUT_INTERVAL_S = 0.001
PROTOCOL_END_TIME_S = 0.69


def initial_geometry() -> dict[str, float]:
    """Return the volume-corrected circular initial meniscus geometry."""
    angle = math.radians(CONTACT_ANGLE_DEGREES)
    radius = HALF_GAP_M / math.cos(angle)
    root = math.sqrt(radius * radius - HALF_GAP_M * HALF_GAP_M)
    arc_integral = (
        radius * HALF_GAP_M -
        0.5 * (
            HALF_GAP_M * root +
            radius * radius * math.asin(HALF_GAP_M / radius)
        )
    )
    mean_sag = arc_integral / HALF_GAP_M
    apex_height = NOMINAL_INITIAL_HEIGHT_M - mean_sag
    contact_height = (
        apex_height + radius -
        math.sqrt(radius * radius - HALF_GAP_M * HALF_GAP_M)
    )
    return {
        "circle_radius_m": radius,
        "mean_meniscus_sag_m": mean_sag,
        "apex_height_m": apex_height,
        "wall_contact_height_m": contact_height,
    }


def initial_interface_height(x: np.ndarray) -> np.ndarray:
    geometry = initial_geometry()
    radius = geometry["circle_radius_m"]
    radicand = np.maximum(radius * radius - np.square(x), 0.0)
    return geometry["apex_height_m"] + radius - np.sqrt(radicand)


def initial_closed_inlet_pressure_offset_pa() -> float:
    """Return the circular-meniscus preload before the inlet is opened."""
    radius = initial_geometry()["circle_radius_m"]
    capillary_jump = -SURFACE_TENSION_N_PER_M / radius
    hydrostatic_head = (
        LIQUID_DENSITY_KG_PER_M3 * GRAVITY_M_PER_S2 *
        NOMINAL_INITIAL_HEIGHT_M
    )
    return capillary_jump + hydrostatic_head


def discrete_initial_column_heights(
        half_gap_cells: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return a target-angle P1 meniscus with the exact nominal liquid area."""
    xs = np.linspace(0.0, HALF_GAP_M, half_gap_cells + 1)
    heights = initial_interface_height(xs)
    geometry = initial_geometry()
    wall_slope = 1.0 / math.tan(math.radians(CONTACT_ANGLE_DEGREES))
    heights[-2:] = (
        geometry["wall_contact_height_m"] +
        wall_slope * (xs[-2:] - HALF_GAP_M)
    )
    target_area = HALF_GAP_M * NOMINAL_INITIAL_HEIGHT_M
    unshifted_area = float(np.trapz(heights, xs))
    vertical_shift = (target_area - unshifted_area) / HALF_GAP_M
    heights += vertical_shift
    discrete_area = float(np.trapz(heights, xs))
    return xs, heights, {
        "apex_height_m": float(heights[0]),
        "wall_contact_height_m": float(heights[-1]),
        "target_liquid_area_m2": target_area,
        "trapezoidal_liquid_area_m2": discrete_area,
        "target_angle_tangent_band_columns": 1,
        "volume_preserving_vertical_shift_m": vertical_shift,
    }


def _write_boundary(path: Path,
                    points: np.ndarray,
                    node_ids: list[int],
                    first_cell_id: int) -> None:
    lines: list[int] = []
    for index in range(len(node_ids) - 1):
        lines.extend((2, index, index + 1))
    boundary = pv.PolyData()
    boundary.points = points[np.asarray(node_ids, dtype=np.int64)]
    boundary.lines = np.asarray(lines, dtype=np.int64)
    boundary.point_data["GlobalNodeID"] = np.asarray(
        node_ids, dtype=np.int64)
    boundary.cell_data["GlobalElementID"] = np.arange(
        first_cell_id,
        first_cell_id + len(node_ids) - 1,
        dtype=np.int64,
    )
    boundary.save(path)


def write_mesh(case_dir: Path, half_gap_cells: int) -> dict[str, Any]:
    """Write an isotropic triangular half-channel and its four boundaries."""
    if (not isinstance(half_gap_cells, int) or
            isinstance(half_gap_cells, bool) or half_gap_cells < 2):
        raise ValueError("capillary-rise half-gap cells must be an integer at least 2")
    nx = half_gap_cells
    ny = 8 * half_gap_cells
    dx = HALF_GAP_M / float(nx)
    xs, column_heights, discrete_geometry = (
        discrete_initial_column_heights(nx))
    ys = np.linspace(0.0, DOMAIN_HEIGHT_M, ny + 1)
    points = np.asarray(
        [[x, y, 0.0] for y in ys for x in xs], dtype=float)

    cells: list[int] = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            if (i + j) % 2 == 0:
                cells.extend((
                    3, lower_left, lower_right, upper_right,
                    3, lower_left, upper_right, upper_left,
                ))
            else:
                cells.extend((
                    3, lower_left, lower_right, upper_left,
                    3, lower_right, upper_right, upper_left,
                ))

    cell_count = 2 * nx * ny
    grid = pv.UnstructuredGrid(
        np.asarray(cells, dtype=np.int64),
        np.full(cell_count, pv.CellType.TRIANGLE, dtype=np.uint8),
        points,
    )
    y = points[:, 1]
    interface_height = np.tile(column_heights, ny + 1)
    phi = y - interface_height
    pressure = (
        initial_closed_inlet_pressure_offset_pa() -
        LIQUID_DENSITY_KG_PER_M3 * GRAVITY_M_PER_S2 * y
    )
    grid.point_data["GlobalNodeID"] = np.arange(
        points.shape[0], dtype=np.int64)
    grid.point_data["phi"] = phi
    grid.point_data["Pressure"] = pressure
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(
        cell_count, dtype=np.int64)

    mesh_dir = case_dir / "mesh/background"
    surface_dir = mesh_dir / "mesh-surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu")

    left = [j * (nx + 1) for j in range(ny + 1)]
    right = [j * (nx + 1) + nx for j in range(ny + 1)]
    bottom = list(range(nx + 1))
    top = [ny * (nx + 1) + i for i in range(nx + 1)]
    offset = cell_count
    _write_boundary(surface_dir / "wall_left.vtp", points, left, offset)
    offset += ny
    _write_boundary(surface_dir / "wall_right.vtp", points, right, offset)
    offset += ny
    _write_boundary(surface_dir / "wall_bottom.vtp", points, bottom, offset)
    offset += nx
    _write_boundary(surface_dir / "wall_top.vtp", points, top, offset)

    return {
        "half_gap_cells": nx,
        "height_cells": ny,
        "dx_m": dx,
        "dy_m": DOMAIN_HEIGHT_M / float(ny),
        "slip_length_to_dx": SLIP_LENGTH_M / dx,
        "vertex_count": int(points.shape[0]),
        "triangle_count": cell_count,
        "discrete_initial_geometry": discrete_geometry,
    }


def _legacy_solver_xml(steps: int, time_step_size: float) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{steps}</Number_of_time_steps>
  <Time_step_size>{time_step_size:.16g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>1</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="capillary_channel">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
  <Add_face name="wall_left">
    <Face_file_path>mesh/background/mesh-surfaces/wall_left.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_right">
    <Face_file_path>mesh/background/mesh-surfaces/wall_right.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_bottom">
    <Face_file_path>mesh/background/mesh-surfaces/wall_bottom.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_top">
    <Face_file_path>mesh/background/mesh-surfaces/wall_top.vtp</Face_file_path>
  </Add_face>
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>1.0e-6</Tolerance>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>prescribed_data</Velocity_source>
  <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
  <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  <Wet_extension_advection_velocity_method>wall_compatible_normal</Wet_extension_advection_velocity_method>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <SUPG_transient_scale>2.0</SUPG_transient_scale>
  <Enable_discontinuity_capturing>true</Enable_discontinuity_capturing>
  <Discontinuity_capturing_scale>0.1</Discontinuity_capturing_scale>
  <Discontinuity_capturing_gradient_epsilon>1.0e-12</Discontinuity_capturing_gradient_epsilon>
  <Discontinuity_capturing_max_courant>0.5</Discontinuity_capturing_max_courant>
  <Enable_bound_preserving_limiter>false</Enable_bound_preserving_limiter>
  <Enable_reinitialization>true</Enable_reinitialization>
  <Reinitialization_method>projection</Reinitialization_method>
  <Reinitialization_cadence_steps>1</Reinitialization_cadence_steps>
  <Reinitialization_max_iterations>4</Reinitialization_max_iterations>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>rcs</Preconditioner>
    </Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
  <Add_BC name="wall_bottom">
    <Type>LevelSetInflow</Type>
    <Value>-0.01</Value>
    <Penalty_scale>2.0</Penalty_scale>
  </Add_BC>
  <Add_BC name="wall_top">
    <Type>LevelSetOutflow</Type>
  </Add_BC>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>12</Max_iterations>
  <Tolerance>1.0e-6</Tolerance>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>{LIQUID_DENSITY_KG_PER_M3:.16g}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>-{GRAVITY_M_PER_S2:.16g}</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
  <Viscosity model="Constant">
    <Value>{LIQUID_VISCOSITY_PA_S:.16g}</Value>
  </Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>rcs</Preconditioner>
    </Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
  <Add_BC name="wall_left">
    <Type>Dir</Type>
    <Value>0.0</Value>
    <Effective_direction>1 0</Effective_direction>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Value>0.0</Value>
    <Effective_direction>1 0</Effective_direction>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Neu</Type>
    <Time_dependence>Steady</Time_dependence>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>capillary_rise_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
    <Allow_corner_linearized_cut_geometry>true</Allow_corner_linearized_cut_geometry>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>{SURFACE_TENSION_N_PER_M:.16g}</Surface_tension>
    <Surface_tension_form>SurfaceStress</Surface_tension_form>
    <Contact_line_model>PrescribedContactAngle</Contact_line_model>
    <Contact_line_wall_face>wall_right</Contact_line_wall_face>
    <Contact_line_wall_normal>1 0 0</Contact_line_wall_normal>
    <Contact_angle_degrees>{CONTACT_ANGLE_DEGREES:.16g}</Contact_angle_degrees>
    <Wall_slip_model>Navier</Wall_slip_model>
    <Wall_slip_length>{SLIP_LENGTH_M:.16g}</Wall_slip_length>
    <Active_domain_smoothing_width>0.0</Active_domain_smoothing_width>
    <Enable_velocity_extension>false</Enable_velocity_extension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
"""


def _d4_solver_xml(steps: int,
                   time_step_size: float,
                   output_cadence: int) -> str:
    """Deck of the "d4" profile (module docstring); mirrors sessile_drop_2d."""
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<!-- capillary-rise benchmark (Gruending et al. 2020, Omega = 1), profile d4; generated by capillary_rise_2d.py -->
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{steps}</Number_of_time_steps>
  <Time_step_size>{time_step_size:.17g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>{output_cadence}</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>{output_cadence}</Start_saving_after_time_step>
  <Increment_in_saving_restart_files>{steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>
  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="capillary_channel">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
  <Add_face name="wall_left">
    <Face_file_path>mesh/background/mesh-surfaces/wall_left.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_right">
    <Face_file_path>mesh/background/mesh-surfaces/wall_right.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_bottom">
    <Face_file_path>mesh/background/mesh-surfaces/wall_bottom.vtp</Face_file_path>
  </Add_face>
  <Add_face name="wall_top">
    <Face_file_path>mesh/background/mesh-surfaces/wall_top.vtp</Face_file_path>
  </Add_face>
</Add_mesh>

<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>prescribed_data</Velocity_source>
  <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  <Advection_velocity_extension_method>pde_harmonic</Advection_velocity_extension_method>
  <Advection_velocity_extension_coupling>monolithic</Advection_velocity_extension_coupling>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <SUPG_transient_scale>2.0</SUPG_transient_scale>
  <Enable_reinitialization>false</Enable_reinitialization>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Enable_kinematic_reconciliation>true</Enable_kinematic_reconciliation>
  <Enable_sign_definite_patch_bounds>true</Enable_sign_definite_patch_bounds>
  <Output type="Spatial">
    <Level_set>true</Level_set>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>rcs</Preconditioner>
    </Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
  <Add_BC name="wall_bottom">
    <Type>LevelSetInflow</Type>
    <Value>-0.01</Value>
    <Penalty_scale>2.0</Penalty_scale>
  </Add_BC>
  <Add_BC name="wall_top">
    <Type>LevelSetOutflow</Type>
  </Add_BC>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>8</Max_iterations>
  <Tolerance>1.0e-4</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>{LIQUID_DENSITY_KG_PER_M3:.17g}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>-{GRAVITY_M_PER_S2:.17g}</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
  <Viscosity model="Constant">
    <Value>{LIQUID_VISCOSITY_PA_S:.17g}</Value>
  </Viscosity>
  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
  </Output>
  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>
  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>rcs</Preconditioner>
    </Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
  <Add_BC name="wall_left">
    <Type>Dir</Type>
    <Value>0.0</Value>
    <Effective_direction>1 0</Effective_direction>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Value>0.0</Value>
    <Effective_direction>1 0</Effective_direction>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Neu</Type>
    <Time_dependence>Steady</Time_dependence>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>{INTERFACE_DOMAIN_ID}</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Active_domain_smoothing_width>0.0</Active_domain_smoothing_width>
    <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
    <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>
    <Interface_quadrature_order>2</Interface_quadrature_order>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>{SURFACE_TENSION_N_PER_M:.17g}</Surface_tension>
    <Surface_tension_form>SurfaceStress</Surface_tension_form>
    <Use_level_set_curvature>false</Use_level_set_curvature>
    <Contact_line_model>PrescribedAngle</Contact_line_model>
    <Contact_line_wall_face>wall_right</Contact_line_wall_face>
    <Contact_line_wall_normal>1 0 0</Contact_line_wall_normal>
    <Contact_angle_degrees>{CONTACT_ANGLE_DEGREES:.17g}</Contact_angle_degrees>
    <Wall_slip_model>Navier</Wall_slip_model>
    <Wall_slip_length>{SLIP_LENGTH_M:.17g}</Wall_slip_length>
    <Enable_velocity_extension>false</Enable_velocity_extension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
    <Use_cut_metadata_scale>false</Use_cut_metadata_scale>
    <Small_cut_aggregation>true</Small_cut_aggregation>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
"""


def _solver_xml(steps: int,
                time_step_size: float,
                profile: str = DEFAULT_PROFILE,
                output_cadence: int = 1) -> str:
    if profile == "legacy":
        if output_cadence != 1:
            raise ValueError("the legacy capillary-rise deck saves every step")
        return _legacy_solver_xml(steps, time_step_size)
    if profile == "d4":
        return _d4_solver_xml(steps, time_step_size, output_cadence)
    raise ValueError(f"capillary-rise profile must be one of {PROFILES}")


def protocol_schedule(half_gap_cells: int,
                      motion_limit_cells_per_step: float | None = None) -> dict[str, Any]:
    """Time step of one comparison level.

    The step is the output interval (1 ms) divided by the smallest integer m
    for which CONTACT_SPEED_SAFETY times the largest reference apex speed
    moves the contact point at most the frozen limit of the level (cells per
    step).  Outputs fall on the 1 ms comparison grid; the run covers
    [0, 0.69 s].
    """
    if (not isinstance(half_gap_cells, int) or isinstance(half_gap_cells, bool)
            or half_gap_cells < 2):
        raise ValueError("capillary-rise half-gap cells must be an integer at least 2")
    if motion_limit_cells_per_step is None:
        motion_limit_cells_per_step = frozen_contact_motion_limits()[half_gap_cells]
    if not (math.isfinite(motion_limit_cells_per_step)
            and motion_limit_cells_per_step > 0.0):
        raise ValueError("contact-motion limit must be positive and finite")
    dx = HALF_GAP_M / float(half_gap_cells)
    design_speed = CONTACT_SPEED_SAFETY * REFERENCE_MAXIMUM_APEX_SPEED_M_PER_S
    substeps = max(1, math.ceil(
        OUTPUT_INTERVAL_S * design_speed / (motion_limit_cells_per_step * dx)
        - 1.0e-12))
    outputs = int(round(PROTOCOL_END_TIME_S / OUTPUT_INTERVAL_S))
    dt = OUTPUT_INTERVAL_S / substeps
    return {
        "half_gap_cells": half_gap_cells,
        "dx_m": dx,
        "maximum_contact_motion_cells_per_step": motion_limit_cells_per_step,
        "design_contact_speed_m_per_s": design_speed,
        "design_contact_motion_cells_per_step": design_speed * dt / dx,
        "time_step_size_s": dt,
        "output_cadence_steps": substeps,
        "output_interval_s": OUTPUT_INTERVAL_S,
        "number_of_time_steps": outputs * substeps,
        "end_time_s": outputs * substeps * dt,
    }


def frozen_contact_motion_limits() -> dict[int, float]:
    """Half-gap cells -> maximum contact motion per step (comparison contract)."""
    contract = json.loads(COMPARISON_CONTRACT.read_text(encoding="utf-8"))
    requirements = contract["candidate_requirements"]
    limits = requirements["maximum_contact_motion_cells_per_step"]
    ratios = requirements["slip_length_to_mesh_ratios"]
    if len(limits) != len(PROTOCOL_LEVELS) or len(ratios) != len(PROTOCOL_LEVELS):
        raise ValueError("comparison contract levels do not match PROTOCOL_LEVELS")
    out: dict[int, float] = {}
    for cells, ratio, limit in zip(PROTOCOL_LEVELS, ratios, limits):
        if not math.isclose(SLIP_LENGTH_M / (HALF_GAP_M / cells), float(ratio)):
            raise ValueError("comparison contract slip ratios do not match the levels")
        out[cells] = float(limit)
    return out


def write_case(case_dir: Path,
               steps: int,
               time_step_size: float,
               half_gap_cells: int,
               profile: str = DEFAULT_PROFILE,
               output_cadence: int = 1) -> dict[str, Any]:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
        raise ValueError("capillary-rise step count must be positive")
    if not math.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("capillary-rise time-step size must be positive and finite")
    if profile not in PROFILES:
        raise ValueError(f"capillary-rise profile must be one of {PROFILES}")
    if (not isinstance(output_cadence, int) or isinstance(output_cadence, bool)
            or output_cadence <= 0):
        raise ValueError("capillary-rise output cadence must be a positive integer")
    if profile == "legacy" and output_cadence != 1:
        raise ValueError("the legacy capillary-rise deck saves every step")
    case_dir.mkdir(parents=True)
    mesh = write_mesh(case_dir, half_gap_cells)
    geometry = initial_geometry()
    benchmark = {
        "schema_version": 1,
        "benchmark": "resolved-slip transient capillary rise",
        "reference_id": "gruending_2020_capillary_rise_omega1_resolved_slip",
        "reference_registry_path": (
            "tests/cases/fluid/free_surface_wp5_capillary_rise_reference.json"
        ),
        "comparison_contract_path": (
            "tests/cases/fluid/free_surface_wp5_capillary_rise_comparison_v1.json"
        ),
        "spatial_dimension": 2,
        "active_domain": "LevelSetNegative",
        "dimensions_m": {
            "channel_half_gap": HALF_GAP_M,
            "domain_height": DOMAIN_HEIGHT_M,
            "profile_window_x_min": 0.5 * HALF_GAP_M,
        },
        "density": LIQUID_DENSITY_KG_PER_M3,
        "viscosity": LIQUID_VISCOSITY_PA_S,
        "gravity_m_per_s2": GRAVITY_M_PER_S2,
        "surface_tension": SURFACE_TENSION_N_PER_M,
        "capillary_rise": {
            "observable": "symmetry-plane interface apex height",
            "observable_units": "m",
            "equilibrium_contact_angle_degrees": CONTACT_ANGLE_DEGREES,
            "wall_face": "wall_right",
            "wall_coordinate_m": HALF_GAP_M,
            "wall_normal": [1.0, 0.0, 0.0],
            "wall_tangent": [0.0, 1.0, 0.0],
            "slip_length_m": SLIP_LENGTH_M,
            "nominal_initial_height_m": NOMINAL_INITIAL_HEIGHT_M,
            "initial_pressure_model": (
                "closed_inlet_circular_capillary_hydrostatic_preload"
            ),
            "initial_pressure_offset_pa": (
                initial_closed_inlet_pressure_offset_pa()
            ),
            "volume_correction_enabled": False,
            "bottom_fluid_boundary": "zero-pressure open inlet",
            "top_level_set_boundary": "outflow",
            "initial_geometry": geometry,
            "discrete_initial_geometry": mesh[
                "discrete_initial_geometry"],
        },
        "mesh_resolution": mesh,
        "time_step_size_s": time_step_size,
        "number_of_time_steps": steps,
    }
    if profile != "legacy":
        # The legacy benchmark.json stays byte-identical to the 2026-08-30 one.
        benchmark["deck_profile"] = profile
        benchmark["output_cadence_steps"] = output_cadence
        benchmark["d4_configuration"] = {
            "contact_line_model": "PrescribedAngle",
            "wall_maintenance": "angle_preserving_scale_only",
            "repair_to_target_angle": False,
            "young_wall_energy": True,
            "wall_slip": "Navier",
            "wall_velocity": "strong_normal_only",
            "capillary_form": "SurfaceStress",
            "small_cut_aggregation": True,
            "level_set_transport": "pde_harmonic_monolithic",
            "kinematic_reconciliation": True,
            "sign_definite_patch_bounds": True,
            "reinitialization": False,
            "time_integration": "GeneralizedAlpha",
            "nonlinear_tolerance": 1.0e-4,
        }
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (case_dir / "solver.xml").write_text(
        _solver_xml(steps, time_step_size, profile, output_cadence),
        encoding="utf-8")
    return benchmark


def _point_values(dataset: pv.DataSet, name: str) -> np.ndarray:
    if name not in dataset.point_data:
        raise ValueError(f"capillary-rise state lacks point field {name!r}")
    values = np.asarray(dataset.point_data[name], dtype=float)
    if values.shape[0] != dataset.n_points:
        raise ValueError(f"capillary-rise point field {name!r} has invalid size")
    return values


def _boundary_profile(dataset: pv.DataSet,
                      x_coordinate: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(dataset.points, dtype=float)
    phi = _point_values(dataset, "phi").reshape(-1)
    scale = max(HALF_GAP_M, DOMAIN_HEIGHT_M, 1.0)
    mask = np.abs(points[:, 0] - x_coordinate) <= 1.0e-12 * scale
    indices = np.flatnonzero(mask)
    if indices.size < 2:
        raise ValueError(
            f"capillary-rise boundary x={x_coordinate:.16g} has fewer than two vertices")
    order = np.argsort(points[indices, 1], kind="stable")
    indices = indices[order]
    return indices, phi[indices]


def _single_boundary_root(dataset: pv.DataSet,
                          x_coordinate: float) -> dict[str, Any]:
    indices, phi = _boundary_profile(dataset, x_coordinate)
    points = np.asarray(dataset.points, dtype=float)
    velocity = _point_values(dataset, "Velocity")
    if velocity.ndim == 1:
        velocity = velocity.reshape((-1, 1))
    roots: list[dict[str, Any]] = []
    tolerance = 1.0e-13 * max(1.0, float(np.max(np.abs(phi))))
    for position in range(indices.size - 1):
        i0 = int(indices[position])
        i1 = int(indices[position + 1])
        p0 = float(phi[position])
        p1 = float(phi[position + 1])
        if abs(p0) <= tolerance and abs(p1) <= tolerance:
            raise ValueError("capillary-rise boundary contains a zero-valued edge")
        if abs(p0) <= tolerance:
            fraction = 0.0
        elif abs(p1) <= tolerance:
            fraction = 1.0
        elif p0 * p1 < 0.0:
            fraction = -p0 / (p1 - p0)
        else:
            continue
        height = (
            float(points[i0, 1]) +
            fraction * float(points[i1, 1] - points[i0, 1])
        )
        sample_velocity = velocity[i0] + fraction * (velocity[i1] - velocity[i0])
        if roots and abs(height - roots[-1]["height_m"]) <= 1.0e-13:
            continue
        roots.append({
            "height_m": height,
            "edge_point_ids": [i0, i1],
            "edge_fraction": float(fraction),
            "velocity": [float(value) for value in sample_velocity],
        })
    if len(roots) != 1:
        raise ValueError(
            "capillary-rise boundary must contain exactly one interface root; "
            f"found {len(roots)} at x={x_coordinate:.16g}")
    return roots[0]


def _contact_fragment_sample(dataset: pv.DataSet,
                             wall_root: dict[str, Any]) -> dict[str, Any]:
    phi = _point_values(dataset, "phi").reshape(-1)
    points = np.asarray(dataset.points, dtype=float)
    edge = set(int(value) for value in wall_root["edge_point_ids"])
    candidates: list[dict[str, Any]] = []
    for cell_id in range(dataset.n_cells):
        cell = dataset.get_cell(cell_id)
        point_ids = [int(value) for value in cell.point_ids]
        if len(point_ids) != 3 or not edge.issubset(point_ids):
            continue
        xy = points[np.asarray(point_ids, dtype=np.int64), :2]
        matrix = np.column_stack((xy, np.ones(3, dtype=float)))
        coefficients = np.linalg.solve(
            matrix, phi[np.asarray(point_ids, dtype=np.int64)])
        gradient = coefficients[:2]
        norm = float(np.linalg.norm(gradient))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("capillary-rise contact triangle has a degenerate level-set gradient")
        normal = gradient / norm
        dynamic_cosine = -float(normal[0])
        dynamic_cosine = min(1.0, max(-1.0, dynamic_cosine))
        candidates.append({
            "cell_id": cell_id,
            "outward_liquid_normal": [float(normal[0]), float(normal[1]), 0.0],
            "contact_angle_degrees": math.degrees(math.acos(dynamic_cosine)),
            "dynamic_cosine": dynamic_cosine,
        })
    if len(candidates) != 1:
        raise ValueError(
            "capillary-rise wall root must belong to exactly one triangular "
            f"contact fragment; found {len(candidates)}")
    return candidates[0]


def _integral_square_linear(value0: float,
                            value1: float,
                            lower: float,
                            upper: float) -> float:
    delta = value1 - value0
    return (
        value0 * value0 * (upper - lower) +
        value0 * delta * (upper * upper - lower * lower) +
        delta * delta * (upper ** 3 - lower ** 3) / 3.0
    )


def _sharp_wall_slip_metrics(dataset: pv.DataSet) -> dict[str, float]:
    indices, phi = _boundary_profile(dataset, HALF_GAP_M)
    points = np.asarray(dataset.points, dtype=float)
    velocity = _point_values(dataset, "Velocity")
    if velocity.ndim == 1:
        velocity = velocity.reshape((-1, 1))
    wetted_length = 0.0
    velocity_square_integral = 0.0
    for position in range(indices.size - 1):
        i0 = int(indices[position])
        i1 = int(indices[position + 1])
        p0 = float(phi[position])
        p1 = float(phi[position + 1])
        length = float(abs(points[i1, 1] - points[i0, 1]))
        if p0 <= 0.0 and p1 <= 0.0:
            lower, upper = 0.0, 1.0
        elif p0 > 0.0 and p1 > 0.0:
            continue
        else:
            root = -p0 / (p1 - p0)
            lower, upper = ((0.0, root) if p0 <= 0.0 else (root, 1.0))
        tangential0 = float(velocity[i0, 1])
        tangential1 = float(velocity[i1, 1])
        wetted_length += length * (upper - lower)
        velocity_square_integral += length * _integral_square_linear(
            tangential0, tangential1, lower, upper)
    return {
        "sharp_wetted_wall_length_m": wetted_length,
        "sharp_wall_tangential_velocity_square_integral_m3_per_s2": (
            velocity_square_integral
        ),
        "sharp_wall_slip_dissipation_w_per_m": (
            LIQUID_VISCOSITY_PA_S / SLIP_LENGTH_M * velocity_square_integral
        ),
    }


def _triangle_area(dataset: pv.DataSet, cell_id: int) -> float:
    cell = dataset.get_cell(cell_id)
    ids = np.asarray(cell.point_ids, dtype=np.int64)
    if ids.size != 3:
        return 0.0
    xyz = np.asarray(dataset.points, dtype=float)[ids]
    return 0.5 * float(np.linalg.norm(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])))


def _liquid_area(dataset: pv.DataSet) -> tuple[float | None, str | None]:
    if "WetVolumeMeasure" in dataset.cell_data:
        values = np.asarray(
            dataset.cell_data["WetVolumeMeasure"], dtype=float).reshape(-1)
        if values.size == dataset.n_cells and np.isfinite(values).all():
            return float(np.sum(values)), "WetVolumeMeasure"
    if "WetVolumeFraction" in dataset.cell_data:
        fractions = np.asarray(
            dataset.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
        if fractions.size == dataset.n_cells and np.isfinite(fractions).all():
            areas = np.asarray(
                [_triangle_area(dataset, cell_id)
                 for cell_id in range(dataset.n_cells)],
                dtype=float,
            )
            return float(np.sum(fractions * areas)), "WetVolumeFraction"
    return None, None


def state_metrics(dataset: pv.DataSet,
                  benchmark: dict[str, Any]) -> dict[str, Any]:
    """Measure the apex, physical-wall contact state, pressure, and slip work."""
    capillary = benchmark.get("capillary_rise")
    if not isinstance(capillary, dict):
        return {"available": False, "error": "missing capillary-rise metadata"}
    try:
        apex = _single_boundary_root(dataset, 0.0)
        wall = _single_boundary_root(dataset, HALF_GAP_M)
        contact = _contact_fragment_sample(dataset, wall)
        slip = _sharp_wall_slip_metrics(dataset)
        phi = _point_values(dataset, "phi").reshape(-1)
        velocity = _point_values(dataset, "Velocity")
        if velocity.ndim == 1:
            velocity = velocity.reshape((-1, 1))
        speed = np.linalg.norm(velocity, axis=1)
        mesh = benchmark.get("mesh_resolution", {})
        dx = float(mesh.get("dx_m", HALF_GAP_M))
        liquid = phi <= 0.0
        strict_liquid = phi < -0.5 * dx
        if not np.any(liquid):
            raise ValueError("capillary-rise state has no liquid-side vertices")

        state: dict[str, Any] = {
            "available": True,
            "apex_height_m": float(apex["height_m"]),
            "apex_height_mm": 1000.0 * float(apex["height_m"]),
            "wall_contact_height_m": float(wall["height_m"]),
            "wall_contact_height_mm": 1000.0 * float(wall["height_m"]),
            "wall_contact_fluid_speed_m_per_s": float(wall["velocity"][1]),
            "wall_contact_fluid_speed_source": (
                "same_state_P1_wall_edge_interpolation_at_phi_zero"
            ),
            "contact_angle_degrees": float(contact["contact_angle_degrees"]),
            "contact_angle_error_degrees": (
                float(contact["contact_angle_degrees"]) - CONTACT_ANGLE_DEGREES
            ),
            "contact_dynamic_cosine": float(contact["dynamic_cosine"]),
            "contact_fragment_cell_id": int(contact["cell_id"]),
            "contact_outward_liquid_normal": contact["outward_liquid_normal"],
            "contact_geometry_source": (
                "same_state_LinearCorner_triangle_gradient_at_wall_root"
            ),
            "max_active_liquid_speed_m_per_s": float(np.max(speed[liquid])),
            "mean_active_liquid_speed_m_per_s": float(np.mean(speed[liquid])),
            **slip,
        }
        if np.any(strict_liquid):
            state["max_strict_liquid_speed_m_per_s"] = float(
                np.max(speed[strict_liquid]))
        area, source = _liquid_area(dataset)
        if area is not None:
            state["liquid_area_m2"] = area
            state["liquid_area_source"] = source

        if "Pressure" in dataset.point_data:
            pressure = _point_values(dataset, "Pressure").reshape(-1)
            finite = np.isfinite(pressure) & np.isfinite(phi)
            liquid_pressure = finite & strict_liquid
            gas_pressure = finite & (phi > 0.5 * dx)
            if np.any(liquid_pressure):
                state["liquid_pressure_median_pa"] = float(
                    np.median(pressure[liquid_pressure]))
            if np.any(gas_pressure):
                state["gas_pressure_median_pa"] = float(
                    np.median(pressure[gas_pressure]))

        try:
            interface = dataset.contour(isosurfaces=[0.0], scalars="phi")
            sized = interface.compute_cell_sizes(
                length=True, area=False, volume=False)
            lengths = np.asarray(
                sized.cell_data.get("Length", []), dtype=float).reshape(-1)
            if lengths.size:
                state["interface_length_m"] = float(np.sum(lengths))
            state["interface_point_count"] = int(interface.n_points)
            state["interface_segment_count"] = int(interface.n_cells)
        except Exception as exc:
            state["interface_measure_error"] = str(exc)
        return state
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write one capillary-rise comparison level (module docstring).")
    parser.add_argument("--half-gap-cells", type=int, required=True,
                        help="cells across the half gap (protocol levels 10, 20, 40)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="truncate the protocol run (smoke tests)")
    parser.add_argument("--motion-limit", type=float, default=None,
                        help="contact motion per step in cells (default: frozen limit of the level)")
    args = parser.parse_args(arguments)
    if args.profile == "legacy":
        parser.error("the protocol schedule writes the d4 profile; the legacy deck "
                     "is written by run_test05_velocity_growth_smoke.py")
    schedule = protocol_schedule(args.half_gap_cells, args.motion_limit)
    steps = schedule["number_of_time_steps"]
    truncated = args.max_steps is not None and args.max_steps < steps
    if truncated:
        steps = args.max_steps
    benchmark = write_case(args.output_dir, steps, schedule["time_step_size_s"],
                           args.half_gap_cells, profile=args.profile,
                           output_cadence=schedule["output_cadence_steps"])
    benchmark["protocol_schedule"] = schedule
    benchmark["truncated"] = truncated
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case": str(args.output_dir), "steps": steps,
                      "dt": schedule["time_step_size_s"],
                      "cadence": schedule["output_cadence_steps"],
                      "truncated": truncated}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
