"""Generate and measure simplex-P1 static capillary cases in three dimensions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyvista as pv


BOX_WALLS = (
    "wall_left",
    "wall_right",
    "wall_bottom",
    "wall_top",
    "wall_front",
    "wall_back",
)


def axis_wall_frame(wall_face: str) -> dict[str, Any]:
    """Return the outward frame of a unit-cube wall."""
    frames = {
        "wall_left": (0, 0.0, (-1.0, 0.0, 0.0), (1, 2)),
        "wall_right": (0, 1.0, (1.0, 0.0, 0.0), (1, 2)),
        "wall_bottom": (1, 0.0, (0.0, -1.0, 0.0), (0, 2)),
        "wall_top": (1, 1.0, (0.0, 1.0, 0.0), (0, 2)),
        "wall_front": (2, 0.0, (0.0, 0.0, -1.0), (0, 1)),
        "wall_back": (2, 1.0, (0.0, 0.0, 1.0), (0, 1)),
    }
    try:
        axis, coordinate, normal, tangent_axes = frames[wall_face]
    except KeyError as exc:
        raise ValueError(
            "spatial sessile wall must be a named unit-cube wall") from exc
    effective = [0, 0, 0]
    effective[axis] = 1
    return {
        "wall_face": wall_face,
        "wall_axis": axis,
        "wall_coordinate": coordinate,
        "wall_normal": normal,
        "wall_tangent_axes": tangent_axes,
        "effective_direction": " ".join(str(value) for value in effective),
    }


def spherical_cap_geometry(contact_angle_degrees: float,
                           radius: float) -> dict[str, float]:
    """Return the analytic geometry of a spherical cap."""
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("spherical-cap radius must be positive and finite")
    angle = math.radians(contact_angle_degrees)
    if not math.isfinite(angle) or not (0.0 < angle < math.pi):
        raise ValueError(
            "spherical-cap contact angle must lie strictly in (0, 180)")
    cosine = math.cos(angle)
    sine = math.sin(angle)
    height = radius * (1.0 - cosine)
    base_radius = radius * sine
    volume = math.pi * height * height * (radius - height / 3.0)
    return {
        "center_inward_distance": -radius * cosine,
        "height": height,
        "base_radius": base_radius,
        "volume": volume,
        "liquid_gas_area": 2.0 * math.pi * radius * height,
        "wetted_wall_area": math.pi * base_radius * base_radius,
        "contact_line_measure": 2.0 * math.pi * base_radius,
    }


def _node_id(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i


def _oriented_tetra(points: np.ndarray,
                     node_ids: tuple[int, int, int, int]
                     ) -> tuple[int, int, int, int]:
    a, b, c, d = node_ids
    signed_six_volume = float(np.dot(
        np.cross(points[b] - points[a], points[c] - points[a]),
        points[d] - points[a],
    ))
    if signed_six_volume < 0.0:
        return (a, b, d, c)
    if signed_six_volume == 0.0:
        raise ValueError("generated tetrahedron is degenerate")
    return node_ids


def _tetrahedra(nx: int, ny: int, nz: int,
                 points: np.ndarray) -> list[tuple[int, int, int, int]]:
    tetrahedra: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n000 = _node_id(i, j, k, nx, ny)
                n100 = _node_id(i + 1, j, k, nx, ny)
                n010 = _node_id(i, j + 1, k, nx, ny)
                n110 = _node_id(i + 1, j + 1, k, nx, ny)
                n001 = _node_id(i, j, k + 1, nx, ny)
                n101 = _node_id(i + 1, j, k + 1, nx, ny)
                n011 = _node_id(i, j + 1, k + 1, nx, ny)
                n111 = _node_id(i + 1, j + 1, k + 1, nx, ny)
                cube = (
                    (n000, n001, n011, n111),
                    (n000, n011, n010, n111),
                    (n000, n010, n110, n111),
                    (n000, n110, n100, n111),
                    (n000, n100, n101, n111),
                    (n000, n101, n001, n111),
                )
                tetrahedra.extend(
                    _oriented_tetra(points, tetrahedron)
                    for tetrahedron in cube
                )
    return tetrahedra


def _boundary_faces(
        tetrahedra: list[tuple[int, int, int, int]],
        points: np.ndarray,
) -> dict[str, list[tuple[int, int, int]]]:
    face_indices = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    counts: dict[tuple[int, int, int], tuple[int, tuple[int, int, int]]] = {}
    for tetrahedron in tetrahedra:
        for local_face in face_indices:
            face = tuple(tetrahedron[index] for index in local_face)
            key = tuple(sorted(face))
            count, stored = counts.get(key, (0, face))
            counts[key] = (count + 1, stored)

    result = {wall: [] for wall in BOX_WALLS}
    tolerance = 1.0e-12
    for key, (count, face) in counts.items():
        if count != 1:
            continue
        coordinates = points[np.asarray(key, dtype=np.int64)]
        for wall in BOX_WALLS:
            frame = axis_wall_frame(wall)
            axis = int(frame["wall_axis"])
            coordinate = float(frame["wall_coordinate"])
            if np.all(np.abs(coordinates[:, axis] - coordinate) <= tolerance):
                result[wall].append(face)
                break
    if any(not result[wall] for wall in BOX_WALLS):
        raise RuntimeError("generated tetrahedral box has an empty wall")
    return result


def write_tetrahedral_box(
        case_dir: Path,
        nx: int,
        ny: int,
        nz: int,
        level_set: Callable[[np.ndarray], np.ndarray],
        pressure: float,
) -> pv.UnstructuredGrid:
    """Write a conforming unit-cube Tetra4 mesh and its six Triangle3 walls."""
    if min(nx, ny, nz) < 2:
        raise ValueError("spatial synthetic mesh resolution must be at least 2")
    if not math.isfinite(pressure):
        raise ValueError("initial pressure must be finite")
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    zs = np.linspace(0.0, 1.0, nz + 1)
    points = np.asarray(
        [[x, y, z] for z in zs for y in ys for x in xs], dtype=float)
    tetrahedra = _tetrahedra(nx, ny, nz, points)
    cells = np.asarray(
        [[4, *tetrahedron] for tetrahedron in tetrahedra],
        dtype=np.int64,
    ).reshape(-1)
    cell_types = np.full(
        len(tetrahedra), int(pv.CellType.TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    phi = np.asarray(level_set(points), dtype=float).reshape(-1)
    if phi.shape != (grid.n_points,) or not np.isfinite(phi).all():
        raise ValueError("generated spatial level set is invalid")
    grid.point_data["GlobalNodeID"] = np.arange(
        grid.n_points, dtype=np.int64)
    grid.point_data["phi"] = phi
    grid.point_data["Pressure"] = np.full(grid.n_points, pressure)
    grid.point_data["Velocity"] = np.zeros((grid.n_points, 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(
        grid.n_cells, dtype=np.int64)

    mesh_dir = case_dir / "mesh/background"
    surface_dir = mesh_dir / "mesh-surfaces"
    surface_dir.mkdir(parents=True, exist_ok=True)
    grid.save(mesh_dir / "mesh-complete.mesh.vtu")
    for wall, faces in _boundary_faces(tetrahedra, points).items():
        used = sorted({point_id for face in faces for point_id in face})
        local = {point_id: index for index, point_id in enumerate(used)}
        surface_cells = np.asarray(
            [[3, *(local[point_id] for point_id in face)] for face in faces],
            dtype=np.int64,
        ).reshape(-1)
        surface = pv.UnstructuredGrid(
            surface_cells,
            np.full(len(faces), int(pv.CellType.TRIANGLE), dtype=np.uint8),
            points[np.asarray(used, dtype=np.int64)],
        )
        surface.point_data["GlobalNodeID"] = np.asarray(used, dtype=np.int64)
        surface.cell_data["GlobalElementID"] = np.arange(
            len(faces), dtype=np.int64)
        surface.save(surface_dir / f"{wall}.vtu")
    return grid


def _write_solver_xml(case_dir: Path,
                      steps: int,
                      time_step_size: float,
                      surface_tension: float,
                      contact: dict[str, Any] | None) -> None:
    if steps < 1 or not math.isfinite(time_step_size) or time_step_size <= 0.0:
        raise ValueError("spatial capillary time controls are invalid")
    face_blocks = "\n".join(
        f"""  <Add_face name=\"{wall}\">
    <Face_file_path>mesh/background/mesh-surfaces/{wall}.vtu</Face_file_path>
  </Add_face>"""
        for wall in BOX_WALLS
    )
    wall_blocks = []
    for wall in BOX_WALLS:
        direction = None
        if contact is not None and wall == contact["wall"]:
            direction = axis_wall_frame(wall)["effective_direction"]
        direction_line = (
            "" if direction is None else
            f"\n    <Effective_direction>{direction}</Effective_direction>")
        wall_blocks.append(f"""  <Add_BC name=\"{wall}\">
    <Type>Dir</Type>
    <Value>0.0</Value>{direction_line}
  </Add_BC>""")
    contact_block = ""
    if contact is not None:
        normal = " ".join(
            f"{float(value):.1f}" for value in contact["wall_normal"])
        contact_block = f"""
    <Contact_line_model>PrescribedContactAngle</Contact_line_model>
    <Contact_line_wall_face>{contact["wall"]}</Contact_line_wall_face>
    <Contact_line_wall_normal>{normal}</Contact_line_wall_normal>
    <Contact_angle_degrees>{float(contact['equilibrium_contact_angle_degrees']):.16g}</Contact_angle_degrees>
    <Active_domain_smoothing_width>0.0</Active_domain_smoothing_width>"""

    solver = f"""<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<svMultiPhysicsFile version=\"0.1\">
<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>3</Number_of_spatial_dimensions>
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
<Add_mesh name=\"tank\">
  <Mesh_file_path>mesh/background/mesh-complete.mesh.vtu</Mesh_file_path>
{face_blocks}
</Add_mesh>
<Add_equation type=\"level_set\">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>2</Max_iterations>
  <Tolerance>1.0e-6</Tolerance>
  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
  <Velocity_source>coupled_field</Velocity_source>
  <Velocity_field_name>Velocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <SUPG_transient_scale>2.0</SUPG_transient_scale>
  <Enable_discontinuity_capturing>true</Enable_discontinuity_capturing>
  <Discontinuity_capturing_scale>0.1</Discontinuity_capturing_scale>
  <Discontinuity_capturing_gradient_epsilon>1.0e-12</Discontinuity_capturing_gradient_epsilon>
  <Discontinuity_capturing_max_courant>0.5</Discontinuity_capturing_max_courant>
  <Enable_bound_preserving_limiter>false</Enable_bound_preserving_limiter>
  <Reinitialization_cadence_steps>1</Reinitialization_cadence_steps>
  <Volume_correction_cadence_steps>1</Volume_correction_cadence_steps>
  <Enable_reinitialization>false</Enable_reinitialization>
  <Enable_volume_correction>false</Enable_volume_correction>
  <Output type=\"Spatial\">
    <Level_set>true</Level_set>
    <Generated_interface>true</Generated_interface>
    <Surface_position>true</Surface_position>
  </Output>
  <Output type=\"Volume_integral\"><Volume>true</Volume></Output>
  <LS type=\"Direct\">
    <Linear_algebra type=\"eigen\"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
</Add_equation>
<Add_equation type=\"fluid\">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>8</Max_iterations>
  <Tolerance>1.0e-6</Tolerance>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>
  <Density>1.0</Density>
  <Force_x>0.0</Force_x><Force_y>0.0</Force_y><Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>false</Hydrostatic_pressure_initialization>
  <Viscosity model=\"Constant\"><Value>0.1</Value></Viscosity>
  <Output type=\"Spatial\">
    <Velocity>true</Velocity><Pressure>true</Pressure><Divergence>true</Divergence>
  </Output>
  <Output type=\"Volume_integral\"><Volume>true</Volume></Output>
  <LS type=\"Direct\">
    <Linear_algebra type=\"eigen\"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>1</Max_iterations>
    <Krylov_space_dimension>1</Krylov_space_dimension>
    <Tolerance>1.0e-8</Tolerance>
    <Absolute_tolerance>1.0e-10</Absolute_tolerance>
  </LS>
{chr(10).join(wall_blocks)}
  <Add_BC name=\"free_surface\">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>static_capillary_surface_3d</Generated_interface_domain_id>
    <Generated_interface_geometry>LinearCorner</Generated_interface_geometry>
    <Level_set_isovalue>0.0</Level_set_isovalue>
    <Active_domain>LevelSetNegative</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>{surface_tension:.16g}</Surface_tension>
    <Surface_tension_form>SurfaceStress</Surface_tension_form>
    <Enable_velocity_extension>false</Enable_velocity_extension>
    <Enable_cut_cell_stabilization>true</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>true</Use_cut_metadata_scale>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>{contact_block}
  </Add_BC>
</Add_equation>
</svMultiPhysicsFile>
"""
    (case_dir / "solver.xml").write_text(solver, encoding="utf-8")


def _nearest_liquid_node(grid: pv.UnstructuredGrid,
                         target: np.ndarray) -> int:
    phi = np.asarray(grid.point_data["phi"], dtype=float)
    liquid = np.flatnonzero(phi < 0.0)
    if liquid.size == 0:
        raise ValueError("spatial capillary mesh has no liquid vertex")
    local = int(np.argmin(np.linalg.norm(
        np.asarray(grid.points)[liquid] - target, axis=1)))
    return int(liquid[local])


def _write_case_metadata(case_dir: Path,
                         benchmark: dict[str, Any],
                         gauge_node: int,
                         pressure_jump: float) -> None:
    benchmark["pressure_gauge"] = {
        "node_id": gauge_node,
        "expected_initial_hydrostatic_pressure": pressure_jump,
        "constraint_applied": False,
        "role": (
            "read-only interior pressure probe; free-surface traction "
            "anchors pressure"),
    }
    (case_dir / "pressure_gauge.csv").write_text(
        f"node_id,pressure\n{gauge_node},{pressure_jump:.16g}\n",
        encoding="utf-8",
    )
    (case_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_sphere_case(case_dir: Path,
                      steps: int,
                      resolution: int,
                      radius: float,
                      surface_tension: float,
                      time_step_size: float,
                      level_set_positive_scale: float = 1.0) -> None:
    """Write a centered closed-sphere equilibrium case."""
    if (not math.isfinite(level_set_positive_scale) or
            level_set_positive_scale <= 0.0):
        raise ValueError("level-set positive scale must be positive and finite")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("sphere radius must be positive and finite")
    if not math.isfinite(surface_tension) or surface_tension <= 0.0:
        raise ValueError("surface tension must be positive and finite")
    case_dir.mkdir(parents=True, exist_ok=True)
    center = np.asarray([0.5, 0.5, 0.5], dtype=float)
    pressure_jump = 2.0 * surface_tension / radius
    grid = write_tetrahedral_box(
        case_dir,
        resolution,
        resolution,
        resolution,
        lambda points: level_set_positive_scale * (
            np.linalg.norm(points - center, axis=1) - radius),
        pressure_jump,
    )
    _write_solver_xml(
        case_dir, steps, time_step_size, surface_tension, contact=None)
    gauge_node = _nearest_liquid_node(grid, center)
    _write_case_metadata(case_dir, {
        "benchmark": "synthetic zero-gravity closed spherical equilibrium",
        "representation": "unfitted_level_set",
        "spatial_dimension": 3,
        "capillary_geometry": "sphere_3d",
        "capillary_radius": radius,
        "capillary_curvature_factor": 2.0,
        "sphere_center": center.tolist(),
        "initial_active_pressure": pressure_jump,
        "density": 1.0,
        "viscosity": 0.1,
        "surface_tension": surface_tension,
        "level_set_positive_scale": level_set_positive_scale,
        "expected_liquid_volume": 4.0 * math.pi * radius ** 3 / 3.0,
        "expected_liquid_gas_area": 4.0 * math.pi * radius ** 2,
        "mesh_resolution": {
            "nx": resolution,
            "ny": resolution,
            "nz": resolution,
            "h": 1.0 / resolution,
        },
        "initial_pressure_extension": (
            "constant two-gamma-over-radius on background support; inactive "
            "pressure coefficients are constrained by active support"),
    }, gauge_node, pressure_jump)


def write_sessile_sphere_case(
        case_dir: Path,
        steps: int,
        resolution: int,
        contact_angle_degrees: float,
        radius: float,
        surface_tension: float,
        time_step_size: float,
        wall_face: str,
        level_set_positive_scale: float = 1.0,
) -> None:
    """Write a prescribed-angle spherical cap on a unit-cube wall."""
    if (not math.isfinite(level_set_positive_scale) or
            level_set_positive_scale <= 0.0):
        raise ValueError("level-set positive scale must be positive and finite")
    if not math.isfinite(surface_tension) or surface_tension <= 0.0:
        raise ValueError("surface tension must be positive and finite")
    frame = axis_wall_frame(wall_face)
    geometry = spherical_cap_geometry(contact_angle_degrees, radius)
    axis = int(frame["wall_axis"])
    coordinate = float(frame["wall_coordinate"])
    normal = np.asarray(frame["wall_normal"], dtype=float)
    inward = -normal
    center = np.full(3, 0.5, dtype=float)
    center[axis] = (
        coordinate + geometry["center_inward_distance"] * inward[axis])
    pressure_jump = 2.0 * surface_tension / radius
    case_dir.mkdir(parents=True, exist_ok=True)
    grid = write_tetrahedral_box(
        case_dir,
        resolution,
        resolution,
        resolution,
        lambda points: level_set_positive_scale * (
            np.linalg.norm(points - center, axis=1) - radius),
        pressure_jump,
    )
    contact = {
        "wall": wall_face,
        "wall_axis": axis,
        "wall_coordinate": coordinate,
        "wall_normal": list(frame["wall_normal"]),
        "wall_tangent_axes": list(frame["wall_tangent_axes"]),
        "equilibrium_contact_angle_degrees": contact_angle_degrees,
    }
    _write_solver_xml(
        case_dir, steps, time_step_size, surface_tension, contact=contact)
    apex = np.asarray(center, dtype=float)
    apex += radius * inward
    gauge_target = np.asarray(center, dtype=float)
    gauge_target[axis] = 0.5 * (coordinate + apex[axis])
    gauge_node = _nearest_liquid_node(grid, gauge_target)
    _write_case_metadata(case_dir, {
        "benchmark": "synthetic prescribed-angle sessile spherical equilibrium",
        "representation": "unfitted_level_set",
        "spatial_dimension": 3,
        "capillary_geometry": "sessile_spherical_cap_3d",
        "capillary_radius": radius,
        "capillary_curvature_factor": 2.0,
        "initial_active_pressure": pressure_jump,
        "density": 1.0,
        "viscosity": 0.1,
        "surface_tension": surface_tension,
        "level_set_positive_scale": level_set_positive_scale,
        "mesh_resolution": {
            "nx": resolution,
            "ny": resolution,
            "nz": resolution,
            "h": 1.0 / resolution,
        },
        "sessile_contact": {
            **contact,
            "active_domain": "LevelSetNegative",
            "circle_center": center.tolist(),
            "circle_radius": radius,
            "level_set_positive_scale": level_set_positive_scale,
            "expected_initial_liquid_volume": geometry["volume"],
            "expected_initial_base_radius": geometry["base_radius"],
            "expected_initial_apex_height": geometry["height"],
            "expected_initial_liquid_gas_area": geometry["liquid_gas_area"],
            "expected_initial_wetted_wall_area": geometry["wetted_wall_area"],
            "expected_initial_contact_line_measure": (
                geometry["contact_line_measure"]),
            "initial_contact_angle_degrees": contact_angle_degrees,
            "dynamic": False,
            "contact_line_model": "PrescribedContactAngle",
            "level_set_geometry_owner": "accepted_state_wall_aware_repair",
            "momentum_owner": "young_wall_energy",
        },
        "initial_pressure_extension": (
            "constant two-gamma-over-radius on background support; inactive "
            "pressure coefficients are constrained by active support"),
    }, gauge_node, pressure_jump)


def _clean_zero_surface(dataset: pv.DataSet) -> pv.PolyData:
    if "phi" not in dataset.point_data:
        raise ValueError("saved spatial state is missing phi")
    surface = dataset.contour(isosurfaces=[0.0], scalars="phi").triangulate()
    if surface.n_cells == 0 or surface.n_points < 4:
        raise ValueError("saved spatial state has no usable zero isosurface")
    scale = max(1.0, float(np.max(np.ptp(
        np.asarray(surface.points, dtype=float), axis=0))))
    return surface.clean(
        point_merging=True,
        tolerance=128.0 * np.finfo(float).eps * scale,
        absolute=True,
    )


def _fit_sphere(points: np.ndarray) -> dict[str, Any]:
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("sphere fit requires at least four spatial points")
    system = np.column_stack((2.0 * points, np.ones(points.shape[0])))
    right = np.sum(points * points, axis=1)
    solution, _residuals, rank, singular_values = np.linalg.lstsq(
        system, right, rcond=None)
    if rank < 4 or singular_values.size < 4:
        raise ValueError("sphere fit is rank deficient")
    center = solution[:3]
    radius_squared = float(solution[3] + np.dot(center, center))
    if radius_squared <= 0.0 or not math.isfinite(radius_squared):
        raise ValueError("sphere fit produced an invalid radius")
    radius = math.sqrt(radius_squared)
    radial_error = np.linalg.norm(points - center, axis=1) - radius
    return {
        "center": center.tolist(),
        "radius": radius,
        "rmse": float(math.sqrt(np.mean(radial_error * radial_error))),
        "max_absolute_error": float(np.max(np.abs(radial_error))),
    }


def _surface_area(surface: pv.PolyData) -> float:
    sized = surface.compute_cell_sizes(area=True)
    values = np.asarray(sized.cell_data["Area"], dtype=float)
    area = float(np.sum(values))
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("zero isosurface has invalid area")
    return area


def _liquid_volume(dataset: pv.DataSet) -> float:
    liquid = dataset.clip_scalar(scalars="phi", value=0.0, invert=True)
    if liquid.n_cells == 0:
        raise ValueError("saved spatial state has no clipped liquid")
    sized = liquid.compute_cell_sizes(volume=True)
    volume = float(np.sum(np.asarray(sized.cell_data["Volume"], dtype=float)))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("saved spatial liquid volume is invalid")
    return volume


def _pressure_and_speed(dataset: pv.DataSet,
                        h: float) -> dict[str, Any]:
    phi = np.asarray(dataset.point_data["phi"], dtype=float).reshape(-1)
    result: dict[str, Any] = {}
    band = 0.5 * h
    liquid = phi < -band
    gas = phi > band
    if "Pressure" in dataset.point_data and np.any(liquid) and np.any(gas):
        pressure = np.asarray(
            dataset.point_data["Pressure"], dtype=float).reshape(-1)
        liquid_pressure = float(np.median(pressure[liquid]))
        gas_pressure = float(np.median(pressure[gas]))
        result.update({
            "liquid_pressure_median": liquid_pressure,
            "gas_pressure_median": gas_pressure,
            "pressure_jump": liquid_pressure - gas_pressure,
        })
    if "Velocity" in dataset.point_data and np.any(phi <= 0.0):
        velocity = np.asarray(dataset.point_data["Velocity"], dtype=float)
        if velocity.ndim == 1:
            velocity = velocity.reshape((-1, 1))
        speed = np.linalg.norm(velocity, axis=1)
        result["max_liquid_speed"] = float(np.max(speed[phi <= 0.0]))
        result["mean_liquid_speed"] = float(np.mean(speed[phi <= 0.0]))
    return result


def _contact_metrics(surface: pv.PolyData,
                     fit_center: np.ndarray,
                     contact: dict[str, Any],
                     h: float) -> dict[str, Any]:
    frame = axis_wall_frame(str(contact["wall"]))
    axis = int(frame["wall_axis"])
    coordinate = float(frame["wall_coordinate"])
    normal = np.asarray(frame["wall_normal"], dtype=float)
    inward = -normal
    tangent_axes = tuple(int(value) for value in frame["wall_tangent_axes"])
    points = np.asarray(surface.points, dtype=float)
    tolerance = max(1.0e-12, 1.0e-8 * h)
    on_wall = np.abs(points[:, axis] - coordinate) <= tolerance

    edges = surface.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    edge_points = np.asarray(edges.points, dtype=float)
    lines = np.asarray(edges.lines, dtype=np.int64).reshape(-1)
    contact_lengths: list[float] = []
    contact_points: list[np.ndarray] = []
    cursor = 0
    while cursor < lines.size:
        count = int(lines[cursor])
        ids = lines[cursor + 1:cursor + 1 + count]
        cursor += count + 1
        if count != 2:
            continue
        coordinates = edge_points[ids]
        if np.all(np.abs(coordinates[:, axis] - coordinate) <= tolerance):
            contact_lengths.append(float(np.linalg.norm(
                coordinates[1] - coordinates[0])))
            contact_points.extend(coordinates)
    if not contact_lengths or not contact_points:
        raise ValueError("zero isosurface has no wall contact line")
    unique_contact = np.unique(
        np.round(np.asarray(contact_points) / tolerance).astype(np.int64),
        axis=0,
    ).astype(float) * tolerance
    tangent = unique_contact[:, tangent_axes]
    tangent_center = np.mean(tangent, axis=0)
    tangent_radii = np.linalg.norm(tangent - tangent_center, axis=1)

    weighted_cosine = 0.0
    total_weight = 0.0
    for cell_id in range(surface.n_cells):
        point_ids = np.asarray(
            surface.get_cell(cell_id).point_ids, dtype=np.int64)
        if point_ids.size != 3 or np.count_nonzero(on_wall[point_ids]) < 2:
            continue
        triangle = points[point_ids]
        candidate = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(candidate))
        if norm <= 0.0:
            continue
        candidate /= norm
        centroid = np.mean(triangle, axis=0)
        if float(np.dot(candidate, centroid - fit_center)) < 0.0:
            candidate *= -1.0
        wall_ids = point_ids[on_wall[point_ids]]
        edge_length = max(
            float(np.linalg.norm(points[right] - points[left]))
            for index, left in enumerate(wall_ids)
            for right in wall_ids[index + 1:]
        )
        weighted_cosine += edge_length * float(np.dot(candidate, normal))
        total_weight += edge_length
    if total_weight <= 0.0:
        raise ValueError("contact facets do not provide a generated angle")
    cosine = max(-1.0, min(1.0, weighted_cosine / total_weight))
    angle = math.degrees(math.acos(-cosine))
    inward_distance = (points - coordinate * np.eye(3)[axis]) @ inward
    return {
        "base_radius": float(np.mean(tangent_radii)),
        "base_radius_spread": float(np.ptp(tangent_radii)),
        "apex_height": float(np.max(inward_distance)),
        "contact_line_measure": math.fsum(contact_lengths),
        "operator_dynamic_angle_degrees_mean": angle,
        "operator_dynamic_cos_mean": cosine,
        "operator_contact_geometry_available": True,
        "operator_contact_geometry_source": (
            "LinearCorner_generated_triangle_normal_at_phi_zero_wall_edges"),
    }


def spatial_capillary_state_metrics(dataset: pv.DataSet,
                                    benchmark: dict[str, Any]) -> dict[str, Any]:
    """Measure a saved closed sphere or sessile spherical-cap state."""
    try:
        surface = _clean_zero_surface(dataset)
        points = np.asarray(surface.points, dtype=float)
        fit = _fit_sphere(points)
        h = float(benchmark["mesh_resolution"]["h"])
        state: dict[str, Any] = {
            "available": True,
            "fitted_sphere_center": fit["center"],
            "fitted_sphere_radius": fit["radius"],
            "fitted_sphere_rmse": fit["rmse"],
            "fitted_sphere_max_absolute_error": fit["max_absolute_error"],
            "liquid_gas_area": _surface_area(surface),
            "liquid_volume": _liquid_volume(dataset),
        }
        state.update(_pressure_and_speed(dataset, h))
        contact = benchmark.get("sessile_contact")
        if isinstance(contact, dict):
            state.update(_contact_metrics(
                surface,
                np.asarray(fit["center"], dtype=float),
                contact,
                h,
            ))
        return state
    except (KeyError, TypeError, ValueError) as exc:
        return {"available": False, "error": str(exc)}
