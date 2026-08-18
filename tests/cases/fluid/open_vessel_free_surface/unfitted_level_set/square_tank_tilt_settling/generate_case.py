#!/usr/bin/env python3
"""Generate the 2D tilted-square free-surface level-set test case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent
MESH_SUBDIR = Path("mesh/background")

# Keep the default case intentionally small: the unfitted free-surface
# BlockSchur path is the behavior under test, and odd counts keep the initial
# horizontal interface inside cut cells rather than aligned to element edges.
DEFAULT_NX = 9
DEFAULT_NY = 9
DEFAULT_TILT_DEGREES = 10.0
DEFAULT_FILL_HEIGHT = 0.5
DEFAULT_DENSITY = 998.2
DEFAULT_VISCOSITY = 2.0e-2
DEFAULT_GRAVITY = 9.81
DEFAULT_TIME_STEP = 1.0e-3
DEFAULT_TIME_STEPS = 1000
DEFAULT_INITIAL_STATE = "settling"
DEFAULT_VERIFICATION_PROFILE = "auto"
DEFAULT_USE_CUT_METADATA_SCALE = False
DEFAULT_CUT_CELL_VELOCITY_GRADIENT_PENALTY = 1.0
DEFAULT_CUT_CELL_PRESSURE_GRADIENT_PENALTY = 1.0
DEFAULT_CUT_CELL_METADATA_SCALE_CAP = None


def xml_bool(value: bool) -> str:
    return "true" if value else "false"


def node_id(i: int, j: int, nx: int) -> int:
    return j * (nx + 1) + i


def qnode_id(i: int, j: int, nx: int) -> int:
    return j * (2 * nx + 1) + i


def structured_quad_mesh(
    nx: int,
    ny: int,
    *,
    element_order: int,
    fill_height: float,
    density: float,
    gravity: float,
    tilt_degrees: float,
    body_force: tuple[float, float, float],
    reference_point: tuple[float, float, float],
    initial_state: str,
) -> pv.UnstructuredGrid:
    if element_order == 1:
        x = np.linspace(0.0, 1.0, nx + 1)
        y = np.linspace(0.0, 1.0, ny + 1)
    elif element_order == 2:
        x = np.linspace(0.0, 1.0, 2 * nx + 1)
        y = np.linspace(0.0, 1.0, 2 * ny + 1)
    else:
        raise ValueError("element_order must be 1 or 2")
    points = np.array([[xi, yi, 0.0] for yi in y for xi in x], dtype=float)

    cells = []
    if element_order == 1:
        for j in range(ny):
            for i in range(nx):
                cells.extend(
                    [
                        4,
                        node_id(i, j, nx),
                        node_id(i + 1, j, nx),
                        node_id(i + 1, j + 1, nx),
                        node_id(i, j + 1, nx),
                    ]
                )
        cell_types = np.full(nx * ny, int(pv.CellType.QUAD), dtype=np.uint8)
    else:
        for j in range(ny):
            for i in range(nx):
                ii = 2 * i
                jj = 2 * j
                cells.extend(
                    [
                        9,
                        qnode_id(ii, jj, nx),
                        qnode_id(ii + 2, jj, nx),
                        qnode_id(ii + 2, jj + 2, nx),
                        qnode_id(ii, jj + 2, nx),
                        qnode_id(ii + 1, jj, nx),
                        qnode_id(ii + 2, jj + 1, nx),
                        qnode_id(ii + 1, jj + 2, nx),
                        qnode_id(ii, jj + 1, nx),
                        qnode_id(ii + 1, jj + 1, nx),
                    ]
                )
        cell_types = np.full(nx * ny, int(pv.CellType.BIQUADRATIC_QUAD), dtype=np.uint8)
    grid = pv.UnstructuredGrid(np.array(cells, dtype=np.int64), cell_types, points)

    if initial_state == "equilibrium":
        surface = final_surface_parameters(tilt_degrees, fill_height)
        initial_phi = points[:, 1] - (surface["intercept"] + surface["slope"] * points[:, 0])
        initial_pressure = np.array(
            [
                expected_pressure(
                    tuple(point),
                    density=density,
                    body_force=body_force,
                    reference_point=reference_point,
                )
                for point in points
            ],
            dtype=float,
        )
    elif initial_state == "settling":
        initial_phi = points[:, 1] - fill_height
        # Pressure is consumed only on the retained CutVolume support.  Its
        # signed continuation on dry cut-cell vertices makes the P1 trace
        # exactly zero at the initial horizontal free surface.
        initial_pressure = density * gravity * (fill_height - points[:, 1])
    else:
        raise ValueError(f"unsupported initial_state: {initial_state}")

    grid.point_data["GlobalNodeID"] = np.arange(points.shape[0], dtype=np.int32)
    grid.point_data["phi"] = initial_phi
    grid.point_data["Velocity"] = np.zeros((points.shape[0], 3), dtype=float)
    grid.point_data["Pressure"] = initial_pressure
    grid.cell_data["GlobalElementID"] = np.arange(nx * ny, dtype=np.int32)
    return grid


def line_polydata(grid: pv.UnstructuredGrid, edges: list[tuple[int, ...]], parent_cells: list[int]) -> pv.PolyData:
    used = sorted({node for edge in edges for node in edge})
    local = {global_id: local_id for local_id, global_id in enumerate(used)}
    lines = np.array(
        [[len(edge), *[local[node] for node in edge]] for edge in edges],
        dtype=np.int64,
    ).ravel()
    poly = pv.PolyData(grid.points[np.array(used, dtype=np.int64)], lines=lines)
    poly.point_data["GlobalNodeID"] = np.array(used, dtype=np.int32)
    poly.cell_data["GlobalElementID"] = np.array(parent_cells, dtype=np.int32)
    return poly


def write_boundary_surfaces(grid: pv.UnstructuredGrid, nx: int, ny: int, surface_dir: Path) -> None:
    surface_dir.mkdir(parents=True, exist_ok=True)
    if grid.celltypes[0] == int(pv.CellType.BIQUADRATIC_QUAD):
        def split_quadratic_edges(
            edges: list[tuple[int, int, int]],
            parents: list[int],
        ) -> tuple[list[tuple[int, ...]], list[int]]:
            split_edges: list[tuple[int, ...]] = []
            split_parents: list[int] = []
            for (a, m, b), parent in zip(edges, parents):
                split_edges.extend([(a, m), (m, b)])
                split_parents.extend([parent, parent])
            return split_edges, split_parents

        left_parents = [j * nx for j in range(ny)]
        right_parents = [j * nx + (nx - 1) for j in range(ny)]
        bottom_parents = list(range(nx))
        top_parents = [(ny - 1) * nx + i for i in range(nx)]
        specs: dict[str, tuple[list[tuple[int, ...]], list[int]]] = {
            "wall_left": split_quadratic_edges(
                [
                    (qnode_id(0, 2 * j, nx), qnode_id(0, 2 * j + 1, nx), qnode_id(0, 2 * j + 2, nx))
                    for j in range(ny)
                ],
                left_parents,
            ),
            "wall_right": split_quadratic_edges(
                [
                    (
                        qnode_id(2 * nx, 2 * j, nx),
                        qnode_id(2 * nx, 2 * j + 1, nx),
                        qnode_id(2 * nx, 2 * j + 2, nx),
                    )
                    for j in range(ny)
                ],
                right_parents,
            ),
            "wall_bottom": split_quadratic_edges(
                [
                    (qnode_id(2 * i, 0, nx), qnode_id(2 * i + 1, 0, nx), qnode_id(2 * i + 2, 0, nx))
                    for i in range(nx)
                ],
                bottom_parents,
            ),
            "wall_top": split_quadratic_edges(
                [
                    (
                        qnode_id(2 * i, 2 * ny, nx),
                        qnode_id(2 * i + 1, 2 * ny, nx),
                        qnode_id(2 * i + 2, 2 * ny, nx),
                    )
                    for i in range(nx)
                ],
                top_parents,
            ),
        }
    else:
        specs = {
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


def final_surface_parameters(tilt_degrees: float, fill_height: float) -> dict[str, float]:
    angle = math.radians(tilt_degrees)
    slope = math.tan(angle)
    intercept = fill_height - 0.5 * slope
    return {
        "slope": slope,
        "intercept": intercept,
        "left_height": intercept,
        "right_height": intercept + slope,
    }


def expected_pressure(
    point: tuple[float, float, float],
    *,
    density: float,
    body_force: tuple[float, float, float],
    reference_point: tuple[float, float, float],
) -> float:
    dx = np.array(point, dtype=float) - np.array(reference_point, dtype=float)
    return float(density * np.dot(np.array(body_force, dtype=float), dx))


def write_probe_points(
    path: Path,
    *,
    density: float,
    body_force: tuple[float, float, float],
    reference_point: tuple[float, float, float],
) -> list[dict[str, object]]:
    points = [
        ("left_bottom", (0.25, 0.25, 0.0)),
        ("center_bottom", (0.50, 0.25, 0.0)),
        ("right_bottom", (0.75, 0.25, 0.0)),
        ("center_wet", (0.50, 0.40, 0.0)),
    ]
    rows = []
    with path.open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["name", "x", "y", "z", "expected_final_pressure"])
        for name, point in points:
            pressure = expected_pressure(
                point,
                density=density,
                body_force=body_force,
                reference_point=reference_point,
            )
            writer.writerow([name, *point, f"{pressure:.12g}"])
            rows.append(
                {
                    "name": name,
                    "coordinates": list(point),
                    "expected_final_pressure": pressure,
                }
            )
    return rows


def write_solver_xml(
    path: Path,
    *,
    element_order: int,
    fluid_taylor_hood: bool,
    density: float,
    viscosity: float,
    body_force: tuple[float, float, float],
    reference_point: tuple[float, float, float],
    time_step: float,
    time_steps: int,
    use_cut_metadata_scale: bool,
    cut_cell_metadata_scale_cap: float | None,
    cut_cell_velocity_gradient_penalty: float,
    cut_cell_pressure_gradient_penalty: float,
) -> None:
    fx, fy, fz = body_force
    rx, ry, rz = reference_point
    fluid_order_xml = f"  <Element_order>{element_order}</Element_order>\n" if element_order != 1 else ""
    taylor_hood_xml = "  <Use_taylor_hood_type_basis>true</Use_taylor_hood_type_basis>\n" if fluid_taylor_hood else ""
    corner_linearized_xml = (
        "    <Allow_corner_linearized_geometry>true</Allow_corner_linearized_geometry>\n"
        "    <Geometry_tangent_policy>RefreshedFrozenQuadrature</Geometry_tangent_policy>\n"
        if element_order != 1
        else ""
    )
    metadata_scale_cap_xml = (
        f"    <Cut_cell_metadata_scale_cap>{cut_cell_metadata_scale_cap:.12g}</Cut_cell_metadata_scale_cap>\n"
        if cut_cell_metadata_scale_cap is not None
        else ""
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>2</Number_of_spatial_dimensions>
  <Number_of_time_steps>{time_steps}</Number_of_time_steps>
  <Time_step_size>{time_step:.8g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>

  <Save_results_to_VTK_format>true</Save_results_to_VTK_format>
  <Combine_time_series>true</Combine_time_series>
  <Name_prefix_of_saved_VTK_files>result</Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files>1</Increment_in_saving_VTK_files>
  <Start_saving_after_time_step>1</Start_saving_after_time_step>

  <Increment_in_saving_restart_files>{time_steps}</Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format>0</Convert_BIN_to_VTK_format>

  <Verbose>1</Verbose>
  <Warning>0</Warning>
  <Debug>0</Debug>
</GeneralSimulationParameters>

<Add_mesh name="tank">
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
  <Tolerance>1.0e-5</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>

  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>prescribed_data</Velocity_source>
  <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
  <Advection_velocity_from_field>Velocity</Advection_velocity_from_field>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <Enable_reinitialization>true</Enable_reinitialization>
  <Reinitialization_method>projection</Reinitialization_method>
  <Reinitialization_cadence_steps>10</Reinitialization_cadence_steps>
  <Reinitialization_max_iterations>4</Reinitialization_max_iterations>
  <Enable_volume_correction>true</Enable_volume_correction>
  <Volume_correction_use_initial_volume>true</Volume_correction_use_initial_volume>
  <Volume_correction_cadence_steps>10</Volume_correction_cadence_steps>
  <Volume_correction_tolerance>1.0e-5</Volume_correction_tolerance>
  <Volume_correction_max_iterations>50</Volume_correction_max_iterations>

  <Output type="Spatial">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>

  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>

  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>80</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>1.0e-6</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
{fluid_order_xml.rstrip()}
{taylor_hood_xml.rstrip()}
  <Min_iterations>1</Min_iterations>
  <Max_iterations>20</Max_iterations>
  <Tolerance>5.0e-3</Tolerance>
  <Module_options>jit=true; jit_specialization=true</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>

  <Density>{density:.12g}</Density>
  <Force_x>{fx:.12g}</Force_x>
  <Force_y>{fy:.12g}</Force_y>
  <Force_z>{fz:.12g}</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>{rx:.12g} {ry:.12g} {rz:.12g}</Hydrostatic_pressure_reference_point>
  <Hydrostatic_pressure_field_name>Pressure</Hydrostatic_pressure_field_name>
  <Viscosity model="Constant">
    <Value>{viscosity:.12g}</Value>
  </Viscosity>

  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>

  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>

  <LS type="Direct">
    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>80</Krylov_space_dimension>
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
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="wall_right">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>
  <Add_BC name="wall_bottom">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>

  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>open_vessel_surface</Generated_interface_domain_id>
{corner_linearized_xml.rstrip()}
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Enable_velocity_extension>true</Enable_velocity_extension>
    <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>0.0</Surface_tension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>{xml_bool(use_cut_metadata_scale)}</Use_cut_metadata_scale>
{metadata_scale_cap_xml}    <Cut_cell_velocity_gradient_penalty>{cut_cell_velocity_gradient_penalty:.12g}</Cut_cell_velocity_gradient_penalty>
    <Cut_cell_pressure_gradient_penalty>{cut_cell_pressure_gradient_penalty:.12g}</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
"""
    path.write_text(xml)


def write_expected_files(
    case_dir: Path,
    *,
    nx: int,
    ny: int,
    element_order: int,
    fluid_taylor_hood: bool,
    tilt_degrees: float,
    fill_height: float,
    density: float,
    viscosity: float,
    gravity: float,
    body_force: tuple[float, float, float],
    reference_point: tuple[float, float, float],
    time_step: float,
    time_steps: int,
    initial_state: str,
    verification_profile: str,
    use_cut_metadata_scale: bool,
    cut_cell_metadata_scale_cap: float | None,
    cut_cell_velocity_gradient_penalty: float,
    cut_cell_pressure_gradient_penalty: float,
) -> None:
    surface = final_surface_parameters(tilt_degrees, fill_height)
    slope = surface["slope"]
    expected_area = fill_height
    expected_centroid = {
        "x": 0.5 + slope / (12.0 * fill_height),
        "y": (fill_height * fill_height + slope * slope / 12.0) / (2.0 * fill_height),
        "z": 0.0,
    }
    probes = write_probe_points(
        case_dir / "expected_probe_points.csv",
        density=density,
        body_force=body_force,
        reference_point=reference_point,
    )
    if verification_profile == "auto":
        verification_profile = "equilibrium" if initial_state == "equilibrium" else "settling"
    if initial_state == "equilibrium" and verification_profile != "equilibrium":
        raise ValueError("equilibrium initial state requires the equilibrium verification profile")
    if verification_profile == "equilibrium" and initial_state != "equilibrium":
        raise ValueError("equilibrium verification profile requires --initial-state equilibrium")
    final_equilibrium_required = verification_profile == "equilibrium"
    pressure_gradient_required = verification_profile in {
        "settling",
        "equilibrium",
        "transient_pressure",
        "transient_pressure_interior",
        "transient_pressure_core",
    }
    interface_pressure_scope = {
        "transient_pressure_interior": "interior",
        "transient_pressure_core": "core",
    }.get(verification_profile, "all")
    interface_pressure_boundary_guard_required = interface_pressure_scope in {"interior", "core"}
    interface_pressure_near_boundary_guard_required = interface_pressure_scope == "core"

    stabilization = {
        "enable_cut_cell_stabilization": True,
        "use_cut_metadata_scale": use_cut_metadata_scale,
        "cut_cell_velocity_gradient_penalty": cut_cell_velocity_gradient_penalty,
        "cut_cell_pressure_gradient_penalty": cut_cell_pressure_gradient_penalty,
    }
    if cut_cell_metadata_scale_cap is not None:
        stabilization["cut_cell_metadata_scale_cap"] = cut_cell_metadata_scale_cap

    expected = {
        "case": "square_tank_tilt_settling",
        "purpose": (
            "A simple 2D unfitted level-set free-surface settling problem for the "
            "OOP incompressible Navier-Stokes setup."
        ),
        "mesh": {
            "type": "structured quadrilateral background mesh",
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 0.0]},
            "nx": nx,
            "ny": ny,
            "element_order": element_order,
            "cell_count": nx * ny,
            "point_count": (nx + 1) * (ny + 1) if element_order == 1 else (2 * nx + 1) * (2 * ny + 1),
        },
        "discretization": {
            "mesh_element_order": element_order,
            "level_set_element_order": 1,
            "fluid_taylor_hood": fluid_taylor_hood,
        },
        "stabilization": stabilization,
        "fluid": {
            "density": density,
            "dynamic_viscosity": viscosity,
            "gravity_magnitude": gravity,
            "body_force": list(body_force),
        },
        "initial_condition": {
            "mode": initial_state,
            "level_set": (
                "phi(x,y) = y - 0.5"
                if initial_state == "settling"
                else "phi(x,y) = phi_eq(x,y)"
            ),
            "fluid_region": "phi <= 0",
            "area": expected_area,
            "free_surface_line": (
                {
                    "equation": "y = fill_height",
                    "slope": 0.0,
                    "intercept": fill_height,
                }
                if initial_state == "settling"
                else {
                    "equation": "y = intercept + slope*x",
                    "slope": surface["slope"],
                    "intercept": surface["intercept"],
                }
            ),
            "velocity": [0.0, 0.0, 0.0],
            "pressure": (
                "p_ext(x,y) = rho*g*(0.5 - y); negative dry-side values are a P1 cut-support continuation, not gas pressure"
                if initial_state == "settling"
                else "p(x,y,z) = p_eq(x,y,z)"
            ),
        },
        "tilt": {
            "angle_degrees": tilt_degrees,
            "angle_radians": math.radians(tilt_degrees),
            "interpretation": (
                "The square vessel is expressed in the tilted frame; gravity is "
                "resolved into the mesh coordinates."
            ),
        },
        "analytic_equilibrium": {
            "reference_pressure": 0.0,
            "external_pressure": 0.0,
            "reference_point": list(reference_point),
            "free_surface_line": {
                "equation": "y = intercept + slope*x",
                "slope": surface["slope"],
                "intercept": surface["intercept"],
                "left_height": surface["left_height"],
                "right_height": surface["right_height"],
            },
            "level_set": "phi_eq(x,y) = y - (intercept + slope*x)",
            "fluid_region": "phi_eq <= 0",
            "area": expected_area,
            "centroid": expected_centroid,
            "pressure": (
                "p_eq(x,y,z) = rho * dot(body_force, "
                "([x,y,z] - reference_point)) in the active fluid"
            ),
            "velocity": [0.0, 0.0, 0.0],
            "surface_tension": 0.0,
        },
        "verification": {
            "script": "verify_expected_results.py",
            "profile": verification_profile,
            "final_equilibrium_required": final_equilibrium_required,
            "interface_slope_progress_required": verification_profile == "settling",
            "interface_intercept_required": verification_profile in {"settling", "equilibrium"},
            "pressure_gradient_required": pressure_gradient_required,
            "interface_pressure_check_scope": interface_pressure_scope,
            "interface_pressure_boundary_guard_required": interface_pressure_boundary_guard_required,
            "interface_pressure_near_boundary_guard_required": interface_pressure_near_boundary_guard_required,
            "probe_points": probes,
            "suggested_tolerances": {
                "interface_slope_abs": 3.0e-2,
                "interface_slope_progress_min": 5.0e-1,
                "interface_intercept_abs": 2.0e-2,
                "interface_line_rms_residual_abs": 4.0e-2,
                "interface_line_max_abs_residual_abs": 8.0e-2,
                "area_abs": 3.0e-2,
                "centroid_abs": 3.0e-2,
                "velocity_max": 5.0e-1,
                "pressure_rms_relative": 3.0e-1,
                "pressure_gradient_abs": 5.0e2,
                "pressure_gradient_relative": 5.0e-2,
                "interface_pressure_rms_abs": 2.5e2,
                "interface_pressure_max_abs": 6.5e2,
                "interface_pressure_boundary_max_abs_guard": 1.0e3,
                "interface_pressure_near_boundary_max_abs_guard": 1.0e3,
                "probe_pressure_abs": 9.0e2,
                "probe_pressure_relative": 6.0e-1,
            },
            "wet_pressure_margin": 5.0e-2,
        },
        "run": {
            "time_step": time_step,
            "time_steps": time_steps,
            "final_time": time_step * time_steps,
            "save_every_step": True,
            "combine_time_series": True,
        },
    }

    with (case_dir / "expected_results.json").open("w") as output:
        json.dump(expected, output, indent=2, sort_keys=True)
        output.write("\n")

    benchmark = {
        "benchmark": "2D tilted square-tank settling equilibrium",
        "representation": "unfitted_level_set",
        "mesh_tools": ["PyVista", "structured quadrilateral grid"],
        "source": "analytic hydrostatic free-surface equilibrium in a tilted frame",
        "expected_results": "expected_results.json",
        "dimensions_m": {
            "tank_length": 1.0,
            "tank_height": 1.0,
            "initial_fill_height": fill_height,
        },
        "notes": [
            "Negative level-set values denote the active fluid.",
            "Velocity and pressure are interpreted physically on the active wet side; OOP VTK output keeps background values unmasked so interface diagnostics can interpolate across phi=0.",
            "The initial surface is horizontal; the body force is tilted at t=0.",
            "At equilibrium, the free surface is perpendicular to the tilted body force.",
            "The final free-surface line passes through the square center, preserving area 0.5.",
            "The default settling mode is a finite-time smoke/regression run; strict final-equilibrium slope checks are reserved for --initial-state equilibrium.",
            "Use --initial-state equilibrium --time-steps 5 to generate a static-equilibrium companion mode.",
            "Use --verification-profile early_transient for short refined settling smoke runs that should not gate final slope/intercept or hydrostatic pressure-gradient convergence.",
            "Use --verification-profile transient_pressure for staged refined transient targets that should gate pressure-gradient recovery before final-equilibrium intercept/slope closure.",
            "Use --verification-profile transient_pressure_interior for staged contact-line-aware targets that gate interior free-surface pressure while keeping wall-contact pressure as bounded diagnostics.",
            "Use --verification-profile transient_pressure_core for staged targets that gate the core free-surface pressure trace while retaining bounded wall-contact and one-cell near-wall guards.",
            "Use --use-cut-metadata-scale, --cut-cell-metadata-scale-cap, and --cut-cell-*-gradient-penalty only for explicit stabilization diagnostics; the default generated smoke keeps metadata scaling disabled and unbounded.",
            "Use --element-order 2 --fluid-taylor-hood to generate an opt-in P2/P1 refined diagnostic; the default checked-in smoke remains linear/equal-order.",
        ],
    }
    with (case_dir / "benchmark.json").open("w") as output:
        json.dump(benchmark, output, indent=2, sort_keys=True)
        output.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=DEFAULT_NX)
    parser.add_argument("--ny", type=int, default=DEFAULT_NY)
    parser.add_argument("--tilt-degrees", type=float, default=DEFAULT_TILT_DEGREES)
    parser.add_argument("--fill-height", type=float, default=DEFAULT_FILL_HEIGHT)
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--viscosity", type=float, default=DEFAULT_VISCOSITY)
    parser.add_argument("--gravity", type=float, default=DEFAULT_GRAVITY)
    parser.add_argument("--time-step", type=float, default=DEFAULT_TIME_STEP)
    parser.add_argument("--time-steps", type=int, default=DEFAULT_TIME_STEPS)
    parser.add_argument(
        "--element-order",
        type=int,
        choices=(1, 2),
        default=1,
        help="background mesh polynomial order; default keeps the linear checked-in smoke",
    )
    parser.add_argument(
        "--fluid-taylor-hood",
        action="store_true",
        help="use a P2/P1 Taylor-Hood fluid pair; requires --element-order 2",
    )
    parser.add_argument(
        "--initial-state",
        choices=("settling", "equilibrium"),
        default=DEFAULT_INITIAL_STATE,
        help="settling starts from a horizontal surface; equilibrium starts from phi_eq and p_eq",
    )
    parser.add_argument(
        "--verification-profile",
        choices=(
            "auto",
            "settling",
            "equilibrium",
            "early_transient",
            "transient_pressure",
            "transient_pressure_interior",
            "transient_pressure_core",
        ),
        default=DEFAULT_VERIFICATION_PROFILE,
        help="verification contract to write into expected_results.json",
    )
    parser.add_argument(
        "--use-cut-metadata-scale",
        action="store_true",
        default=DEFAULT_USE_CUT_METADATA_SCALE,
        help="enable cut-metadata stabilization scaling for explicit diagnostics",
    )
    parser.add_argument(
        "--cut-cell-metadata-scale-cap",
        type=float,
        default=DEFAULT_CUT_CELL_METADATA_SCALE_CAP,
        help="optional finite cap >= 1 applied to cut-metadata stabilization scaling",
    )
    parser.add_argument(
        "--cut-cell-velocity-gradient-penalty",
        type=float,
        default=DEFAULT_CUT_CELL_VELOCITY_GRADIENT_PENALTY,
        help="velocity ghost-penalty coefficient for cut-cell stabilization",
    )
    parser.add_argument(
        "--cut-cell-pressure-gradient-penalty",
        type=float,
        default=DEFAULT_CUT_CELL_PRESSURE_GRADIENT_PENALTY,
        help="pressure ghost-penalty coefficient for cut-cell stabilization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nx < 2 or args.ny < 2:
        raise ValueError("nx and ny must be at least 2")
    if not (0.0 < args.fill_height < 1.0):
        raise ValueError("fill height must lie inside the unit square")
    if args.fluid_taylor_hood and args.element_order < 2:
        raise ValueError("--fluid-taylor-hood requires --element-order 2")
    if (
        args.cut_cell_metadata_scale_cap is not None
        and (
            not math.isfinite(args.cut_cell_metadata_scale_cap)
            or args.cut_cell_metadata_scale_cap < 1.0
        )
    ):
        raise ValueError("--cut-cell-metadata-scale-cap must be finite and at least 1")
    if args.cut_cell_velocity_gradient_penalty < 0.0:
        raise ValueError("--cut-cell-velocity-gradient-penalty must be nonnegative")
    if args.cut_cell_pressure_gradient_penalty < 0.0:
        raise ValueError("--cut-cell-pressure-gradient-penalty must be nonnegative")

    mesh_dir = CASE_DIR / MESH_SUBDIR
    if mesh_dir.exists():
        shutil.rmtree(mesh_dir)
    mesh_dir.mkdir(parents=True)

    angle = math.radians(args.tilt_degrees)
    body_force = (
        args.gravity * math.sin(angle),
        -args.gravity * math.cos(angle),
        0.0,
    )
    reference_point = (0.5, args.fill_height, 0.0)

    grid = structured_quad_mesh(
        args.nx,
        args.ny,
        element_order=args.element_order,
        fill_height=args.fill_height,
        density=args.density,
        gravity=args.gravity,
        tilt_degrees=args.tilt_degrees,
        body_force=body_force,
        reference_point=reference_point,
        initial_state=args.initial_state,
    )
    grid.save(mesh_dir / "mesh-complete.mesh.vtu", binary=False)
    write_boundary_surfaces(grid, args.nx, args.ny, mesh_dir / "mesh-surfaces")

    write_solver_xml(
        CASE_DIR / "solver.xml",
        element_order=args.element_order,
        fluid_taylor_hood=args.fluid_taylor_hood,
        density=args.density,
        viscosity=args.viscosity,
        body_force=body_force,
        reference_point=reference_point,
        time_step=args.time_step,
        time_steps=args.time_steps,
        use_cut_metadata_scale=args.use_cut_metadata_scale,
        cut_cell_metadata_scale_cap=args.cut_cell_metadata_scale_cap,
        cut_cell_velocity_gradient_penalty=args.cut_cell_velocity_gradient_penalty,
        cut_cell_pressure_gradient_penalty=args.cut_cell_pressure_gradient_penalty,
    )
    write_expected_files(
        CASE_DIR,
        nx=args.nx,
        ny=args.ny,
        element_order=args.element_order,
        fluid_taylor_hood=args.fluid_taylor_hood,
        tilt_degrees=args.tilt_degrees,
        fill_height=args.fill_height,
        density=args.density,
        viscosity=args.viscosity,
        gravity=args.gravity,
        body_force=body_force,
        reference_point=reference_point,
        time_step=args.time_step,
        time_steps=args.time_steps,
        initial_state=args.initial_state,
        verification_profile=args.verification_profile,
        use_cut_metadata_scale=args.use_cut_metadata_scale,
        cut_cell_metadata_scale_cap=args.cut_cell_metadata_scale_cap,
        cut_cell_velocity_gradient_penalty=args.cut_cell_velocity_gradient_penalty,
        cut_cell_pressure_gradient_penalty=args.cut_cell_pressure_gradient_penalty,
    )

    try:
        display_dir = CASE_DIR.relative_to(Path.cwd())
    except ValueError:
        display_dir = CASE_DIR
    print(f"generated {display_dir}: {grid.n_points} points, {grid.n_cells} quads")


if __name__ == "__main__":
    main()
