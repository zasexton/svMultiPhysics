#!/usr/bin/env python3
"""Generate literature benchmark meshes for open-vessel free-surface cases."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pyvista as pv


ROOT = Path(__file__).resolve().parent
WATER_DENSITY = 998.2
WATER_VISCOSITY = 1.003e-3
GRAVITY = 9.81
SURFACE_TENSION = 0.0728
UNFITTED_SURFACE_TENSION = 0.0
LINEAR_SOLVER_TOLERANCE = "1.0e-6"
LINEAR_SOLVER_ABSOLUTE_TOLERANCE = "1.0e-10"
FLUID_NONLINEAR_TOLERANCE = "1.0e-6"
LEVEL_SET_NONLINEAR_TOLERANCE = "1.0e-6"
TEST05_BLOCKSCHUR_LINEAR_TOLERANCE = "1.0e-4"
TEST05_BLOCKSCHUR_LINEAR_ABSOLUTE_TOLERANCE = "1.0e-4"
TEST05_BLOCKSCHUR_FLUID_NONLINEAR_TOLERANCE = "2.0e-2"
TEST05_BLOCKSCHUR_LEVEL_SET_NONLINEAR_TOLERANCE = "1.0e-4"
TEST05_BLOCKSCHUR_FLUID_MAX_ITERATIONS = "12"
TEST05_BLOCKSCHUR_LINEAR_MAX_ITERATIONS = "100"
TEST05_BLOCKSCHUR_KRYLOV_SPACE_DIMENSION = "80"
TEST05_BLOCKSCHUR_INNER_MAX_ITERATIONS = "150"
TEST05_BLOCKSCHUR_MIN_OUTER_ITERATIONS = "1"
TEST05_ADAPTIVE_TIME_LOOP_MIN_DT = "1.5625e-5"
TEST05_ADAPTIVE_TIME_LOOP_MAX_DT = "5.0e-4"
TEST05_ADAPTIVE_TIME_LOOP_MAX_RETRIES = "8"
TEST05_ADAPTIVE_TIME_LOOP_DECREASE_FACTOR = "0.5"
TEST05_ADAPTIVE_TIME_LOOP_INCREASE_FACTOR = "1.5"
TEST05_ADAPTIVE_TIME_LOOP_TARGET_NEWTON_ITERATIONS = "6"
TEST05_ADAPTIVE_TIME_LOOP_MAX_STEPS_MULTIPLIER = "64"
TEST05_PREVIOUS_INVALID_D18_GAUGE = {
    "node_id": 279,
    "initial_phi": -0.001806,
    "full_volume_hydrostatic_pressure": 17.6869,
    "hydrostatic_error_range": [-17.6869, 0.0],
}
TEST05_REFERENCE_PROFILE_TIMES = {
    18: [0.156, 0.219, 0.281, 0.343, 0.406, 0.468, 0.531],
    38: [0.156, 0.219, 0.281, 0.343, 0.406, 0.468, 0.531, 0.593],
}

warnings.filterwarnings("ignore", message="Meshio doesn't know keyword.*")


@dataclass(frozen=True)
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
            0.5 * (self.zmin + self.zmax),
        )

    def contains(self, points: np.ndarray, pad: float = 0.0) -> np.ndarray:
        return (
            (points[:, 0] >= self.xmin - pad)
            & (points[:, 0] <= self.xmax + pad)
            & (points[:, 1] >= self.ymin - pad)
            & (points[:, 1] <= self.ymax + pad)
            & (points[:, 2] >= self.zmin - pad)
            & (points[:, 2] <= self.zmax + pad)
        )


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    predicate: Callable[[np.ndarray, np.ndarray], bool]


def box_surface(box: Box, *, include_bottom: bool = True) -> pv.PolyData:
    x0, x1, y0, y1, z0, z1 = box.bounds
    points = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )

    triangles: list[tuple[int, int, int]] = [
        (3, 2, 6),
        (3, 6, 7),
        (0, 3, 7),
        (0, 7, 4),
        (1, 5, 6),
        (1, 6, 2),
        (0, 1, 2),
        (0, 2, 3),
        (4, 7, 6),
        (4, 6, 5),
    ]
    if include_bottom:
        triangles.extend([(0, 4, 5), (0, 5, 1)])

    faces = np.array([[3, *tri] for tri in triangles], dtype=np.int64).ravel()
    return pv.PolyData(points, faces).triangulate().clean(tolerance=0.0)


def merge_surfaces(surfaces: Iterable[pv.PolyData]) -> pv.PolyData:
    surfaces = list(surfaces)
    merged = surfaces[0].copy()
    for surface in surfaces[1:]:
        merged = merged.merge(surface, merge_points=False)
    return merged.triangulate().clean(tolerance=0.0)


def tetra_grid(points: np.ndarray, elements: np.ndarray) -> pv.UnstructuredGrid:
    cells = np.hstack(
        [
            np.full((elements.shape[0], 1), 4, dtype=np.int64),
            elements.astype(np.int64),
        ]
    ).ravel()
    cell_types = np.full(elements.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points)


def orient_tets_positive(points: np.ndarray, elements: np.ndarray) -> np.ndarray:
    oriented = elements.copy()
    a = points[oriented[:, 0]]
    b = points[oriented[:, 1]]
    c = points[oriented[:, 2]]
    d = points[oriented[:, 3]]
    volumes = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    negative = volumes < 0.0
    oriented[negative, 2], oriented[negative, 3] = (
        oriented[negative, 3].copy(),
        oriented[negative, 2].copy(),
    )
    return oriented


def segmented_coordinates(values: Iterable[float], max_spacing: float) -> np.ndarray:
    anchors = sorted({float(value) for value in values})
    coords: list[float] = []
    for start, stop in zip(anchors[:-1], anchors[1:]):
        count = max(1, int(np.ceil((stop - start) / max_spacing)))
        segment = np.linspace(start, stop, count + 1)
        if coords:
            segment = segment[1:]
        coords.extend(float(value) for value in segment)
    return np.array(coords, dtype=float)


def centered_cut_segmented_coordinates(
    start: float,
    stop: float,
    cut_values: Iterable[float],
    *,
    max_spacing: float,
) -> np.ndarray:
    anchors = [float(start), float(stop)]
    half_width = 0.5 * max_spacing
    for cut in cut_values:
        cut = float(cut)
        lower = max(float(start), cut - half_width)
        upper = min(float(stop), cut + half_width)
        if lower < cut < upper:
            anchors.extend([lower, upper])
    return segmented_coordinates(anchors, max_spacing)


def structured_tet_grid(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    *,
    mirror_z_midplane: bool = False,
) -> pv.UnstructuredGrid:
    nx = len(x_coords)
    ny = len(y_coords)
    nz = len(z_coords)
    points = np.array(
        [[x, y, z] for z in z_coords for y in y_coords for x in x_coords],
        dtype=float,
    )

    def node(i: int, j: int, k: int) -> int:
        return k * ny * nx + j * nx + i

    elements: list[tuple[int, int, int, int]] = []
    z_mid = 0.5 * (float(z_coords[0]) + float(z_coords[-1]))
    for k in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n000 = node(i, j, k)
                n100 = node(i + 1, j, k)
                n010 = node(i, j + 1, k)
                n110 = node(i + 1, j + 1, k)
                n001 = node(i, j, k + 1)
                n101 = node(i + 1, j, k + 1)
                n011 = node(i, j + 1, k + 1)
                n111 = node(i + 1, j + 1, k + 1)
                if mirror_z_midplane and 0.5 * (z_coords[k] + z_coords[k + 1]) > z_mid:
                    elements.extend(
                        [
                            (n001, n000, n010, n110),
                            (n001, n010, n011, n110),
                            (n001, n011, n111, n110),
                            (n001, n111, n101, n110),
                            (n001, n101, n100, n110),
                            (n001, n100, n000, n110),
                        ]
                    )
                else:
                    elements.extend(
                        [
                            (n000, n001, n011, n111),
                            (n000, n011, n010, n111),
                            (n000, n010, n110, n111),
                            (n000, n110, n100, n111),
                            (n000, n100, n101, n111),
                            (n000, n101, n001, n111),
                        ]
                    )

    tets = orient_tets_positive(points, np.array(elements, dtype=np.int64))
    return tetra_grid(points, tets)


def grid_tets(grid: pv.UnstructuredGrid) -> np.ndarray:
    cells = grid.cells.reshape((-1, 5))
    return cells[:, 1:].astype(np.int64)


def remove_cells_inside(grid: pv.UnstructuredGrid, obstacles: list[Box]) -> pv.UnstructuredGrid:
    if not obstacles:
        return grid

    centers = grid.cell_centers().points
    keep = np.ones(grid.n_cells, dtype=bool)
    for obstacle in obstacles:
        keep &= ~obstacle.contains(centers, pad=1.0e-12)
    return grid.extract_cells(keep).clean(tolerance=1.0e-12)


def build_tetgen_grid(
    domain: Box,
    *,
    max_volume: float,
    obstacles: list[Box] | None = None,
) -> pv.UnstructuredGrid:
    import tetgen

    obstacles = obstacles or []
    surfaces = [box_surface(domain)]
    surfaces.extend(box_surface(obstacle, include_bottom=False) for obstacle in obstacles)

    try:
        surface = merge_surfaces(surfaces)
        tet = tetgen.TetGen(surface)
        result = tet.tetrahedralize(
            plc=True,
            quality=True,
            minratio=1.4,
            mindihedral=8.0,
            maxvolume=max_volume,
            steinerleft=200000,
            quiet=True,
        )
    except RuntimeError:
        if not obstacles:
            raise
        tet = tetgen.TetGen(box_surface(domain))
        result = tet.tetrahedralize(
            plc=True,
            quality=True,
            minratio=1.4,
            mindihedral=8.0,
            maxvolume=max_volume,
            steinerleft=200000,
            quiet=True,
        )
    points, elements = result[0], result[1]
    return remove_cells_inside(tetra_grid(points, elements), obstacles)


def run_mmg(grid: pv.UnstructuredGrid, *, hmin: float, hmax: float, hausd: float) -> pv.UnstructuredGrid:
    import meshio
    from svv.utils.remeshing import mmg

    with tempfile.TemporaryDirectory(prefix="svmp-free-surface-mmg-") as temp_dir:
        temp = Path(temp_dir)
        in_path = temp / "input.mesh"
        out_path = temp / "output.mesh"
        meshio.write(
            in_path,
            meshio.Mesh(points=grid.points, cells=[("tetra", grid_tets(grid))]),
            file_format="medit",
        )
        mmg.run_mmg(
            "mmg3d",
            [
                "-in",
                str(in_path),
                "-out",
                str(out_path),
                "-hmin",
                f"{hmin:.8g}",
                "-hmax",
                f"{hmax:.8g}",
                "-hausd",
                f"{hausd:.8g}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            remeshed = meshio.read(out_path)

    tets = None
    for block in remeshed.cells:
        if block.type == "tetra":
            tets = np.asarray(block.data, dtype=np.int64)
            break
    if tets is None:
        raise RuntimeError("MMG output did not contain tetrahedral cells")
    return tetra_grid(np.asarray(remeshed.points, dtype=float), tets)


def clear_arrays(grid: pv.UnstructuredGrid) -> None:
    for name in list(grid.point_data.keys()):
        del grid.point_data[name]
    for name in list(grid.cell_data.keys()):
        del grid.cell_data[name]


def add_solution_arrays(
    grid: pv.UnstructuredGrid,
    *,
    phi: Callable[[np.ndarray], np.ndarray] | None = None,
    pressure: Callable[[np.ndarray], np.ndarray] | None = None,
    fitted: bool = False,
) -> None:
    clear_arrays(grid)
    npoints = grid.n_points
    grid.point_data["GlobalNodeID"] = np.arange(npoints, dtype=np.int32)
    if phi is not None:
        grid.point_data["phi"] = phi(grid.points).astype(float)
    grid.point_data["Velocity"] = np.zeros((npoints, 3), dtype=float)
    grid.point_data["Pressure"] = (
        pressure(grid.points).astype(float) if pressure is not None else np.zeros(npoints, dtype=float)
    )
    if fitted:
        grid.point_data["mesh_displacement"] = np.zeros((npoints, 3), dtype=float)
        grid.point_data["mesh_velocity"] = np.zeros((npoints, 3), dtype=float)
    grid.cell_data["GlobalElementID"] = np.arange(grid.n_cells, dtype=np.int32)


def signed_distance_to_box(points: np.ndarray, box: Box) -> np.ndarray:
    center = np.array(box.center)
    half = np.array(
        [
            0.5 * (box.xmax - box.xmin),
            0.5 * (box.ymax - box.ymin),
            0.5 * (box.zmax - box.zmin),
        ]
    )
    q = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum.reduce(q, axis=1), 0.0)
    return outside + inside


def boundary_faces(tets: np.ndarray) -> list[tuple[int, int, int]]:
    counts: dict[tuple[int, int, int], list[int | tuple[int, int, int]]] = {}
    for tet in tets:
        faces = (
            (tet[0], tet[2], tet[1]),
            (tet[0], tet[1], tet[3]),
            (tet[1], tet[2], tet[3]),
            (tet[2], tet[0], tet[3]),
        )
        for face in faces:
            key = tuple(sorted(int(i) for i in face))
            if key not in counts:
                counts[key] = [0, tuple(int(i) for i in face)]
            counts[key][0] += 1
    return [value[1] for value in counts.values() if value[0] == 1]  # type: ignore[misc]


def polydata_from_faces(
    points: np.ndarray,
    faces: list[tuple[int, int, int]],
) -> pv.PolyData:
    used = sorted({node for face in faces for node in face})
    local = {node: i for i, node in enumerate(used)}
    vtk_faces = np.array(
        [[3, local[a], local[b], local[c]] for a, b, c in faces],
        dtype=np.int64,
    ).ravel()
    poly = pv.PolyData(points[np.array(used, dtype=np.int64)], vtk_faces)
    poly.point_data["GlobalNodeID"] = np.array(used, dtype=np.int32)
    poly.cell_data["GlobalElementID"] = np.arange(len(faces), dtype=np.int32)
    return poly


def write_surfaces(
    grid: pv.UnstructuredGrid,
    specs: list[SurfaceSpec],
    surface_dir: Path,
) -> None:
    surface_dir.mkdir(parents=True, exist_ok=True)
    points = grid.points
    faces = boundary_faces(grid_tets(grid))

    for spec in specs:
        selected = [
            face
            for face in faces
            if spec.predicate(points[np.array(face, dtype=np.int64)], points[np.array(face, dtype=np.int64)].mean(axis=0))
        ]
        if not selected:
            raise RuntimeError(f"surface {spec.name!r} did not select any boundary faces")
        polydata_from_faces(points, selected).save(surface_dir / f"{spec.name}.vtp", binary=False)


def plane_predicate(axis: int, value: float, tol: float) -> Callable[[np.ndarray, np.ndarray], bool]:
    def predicate(face_points: np.ndarray, center: np.ndarray) -> bool:
        del face_points
        return abs(center[axis] - value) <= tol

    return predicate


def obstacle_predicate(obstacle: Box, tol: float) -> Callable[[np.ndarray, np.ndarray], bool]:
    def predicate(face_points: np.ndarray, center: np.ndarray) -> bool:
        del face_points
        in_span = (
            obstacle.xmin - tol <= center[0] <= obstacle.xmax + tol
            and obstacle.ymin - tol <= center[1] <= obstacle.ymax + tol
            and obstacle.zmin - tol <= center[2] <= obstacle.zmax + tol
        )
        on_vertical = (
            abs(center[0] - obstacle.xmin) <= tol
            or abs(center[0] - obstacle.xmax) <= tol
            or abs(center[2] - obstacle.zmin) <= tol
            or abs(center[2] - obstacle.zmax) <= tol
        ) and center[1] >= obstacle.ymin + tol
        on_top = abs(center[1] - obstacle.ymax) <= tol
        return in_span and (on_vertical or on_top)

    return predicate


def hydrostatic_pressure_at(point: np.ndarray, fill_height: float) -> float:
    return WATER_DENSITY * GRAVITY * (fill_height - float(point[1]))


def dam_break_wet_bed_phi(
    points: np.ndarray,
    *,
    gate_x: float,
    dam_height: float,
    wet_depth: float,
) -> np.ndarray:
    bed_phi = points[:, 1] - wet_depth
    column_phi = np.maximum(points[:, 0] - gate_x, points[:, 1] - dam_height)
    return np.minimum(bed_phi, column_phi)


def dam_break_wet_bed_pressure(
    points: np.ndarray,
    *,
    gate_x: float,
    dam_height: float,
    wet_depth: float,
) -> np.ndarray:
    surface_height = np.where(points[:, 0] <= gate_x, dam_height, wet_depth)
    return WATER_DENSITY * GRAVITY * np.maximum(surface_height - points[:, 1], 0.0)


def test05_vertical_coordinates(
    domain: Box,
    wet_depth: float,
    dam_height: float,
) -> np.ndarray:
    fine_spacing = 0.006
    coarse_spacing = 0.010
    fine_top = min(domain.ymax, wet_depth + 0.5 * fine_spacing)
    anchors = [domain.ymin, domain.ymax]
    for cut, spacing in ((wet_depth, fine_spacing), (dam_height, coarse_spacing)):
        half_width = 0.5 * spacing
        lower = max(domain.ymin, cut - half_width)
        upper = min(domain.ymax, cut + half_width)
        if lower < cut < upper:
            anchors.extend([lower, upper])
    anchors.append(fine_top)
    anchors = sorted({float(value) for value in anchors})

    coords: list[float] = []
    for start, stop in zip(anchors[:-1], anchors[1:]):
        max_spacing = fine_spacing if stop <= fine_top else coarse_spacing
        count = max(1, int(np.ceil((stop - start) / max_spacing)))
        segment = np.linspace(start, stop, count + 1)
        if coords:
            segment = segment[1:]
        coords.extend(float(value) for value in segment)
    cut_values = (wet_depth, dam_height)
    coords = [
        value
        for value in coords
        if value in (domain.ymin, domain.ymax)
        or not any(abs(value - cut) <= 1.0e-12 for cut in cut_values)
    ]
    return np.array(coords, dtype=float)


def write_pressure_gauge(
    case_dir: Path,
    grid: pv.UnstructuredGrid,
    point: tuple[float, float, float],
    pressure: float | Callable[[np.ndarray], float] = 0.0,
) -> dict:
    distances = np.linalg.norm(grid.points - np.array(point), axis=1)
    node_id = int(np.argmin(distances))
    selected_point = grid.points[node_id]
    pressure_value = float(pressure(selected_point) if callable(pressure) else pressure)
    with (case_dir / "pressure_gauge.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["node_id", "pressure"])
        writer.writerow([node_id, f"{pressure_value:.12g}"])

    metadata = {
        "node_id": node_id,
        "coordinates": [float(value) for value in selected_point],
        "expected_initial_hydrostatic_pressure": pressure_value,
    }
    if "phi" in grid.point_data:
        metadata["initial_phi"] = float(grid.point_data["phi"][node_id])
    return metadata


def write_metadata(case_dir: Path, metadata: dict) -> None:
    with (case_dir / "benchmark.json").open("w") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


def hydrostatic_pressure_at_point(point: np.ndarray, fill_height: float) -> float:
    return WATER_DENSITY * GRAVITY * max(0.0, fill_height - float(point[1]))


def write_solver_xml(
    case_dir: Path,
    *,
    mesh_path: str,
    faces: list[str],
    fitted: bool,
    fill_height: float,
    time_step: float,
    time_steps: int,
    include_top_wall_bc: bool = False,
    include_obstacle_bc: bool = False,
    active_domain: str | None = None,
    enable_cut_cell_stabilization: bool = True,
    use_cut_metadata_scale: bool = False,
    use_blockschur_solver: bool = False,
    use_direct_fluid_solver: bool = False,
    include_pressure_constraints: bool = True,
    hydrostatic_pressure_field_name: str | None = None,
    enable_level_set_maintenance: bool = True,
    wet_extension_advection_velocity_method: str | None = None,
    enable_adaptive_time_loop: bool = False,
    adaptive_time_loop_min_dt: str = TEST05_ADAPTIVE_TIME_LOOP_MIN_DT,
    adaptive_time_loop_max_dt: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_DT,
    adaptive_time_loop_max_retries: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_RETRIES,
    adaptive_time_loop_decrease_factor: str = TEST05_ADAPTIVE_TIME_LOOP_DECREASE_FACTOR,
    adaptive_time_loop_increase_factor: str = TEST05_ADAPTIVE_TIME_LOOP_INCREASE_FACTOR,
    adaptive_time_loop_target_newton_iterations: str = TEST05_ADAPTIVE_TIME_LOOP_TARGET_NEWTON_ITERATIONS,
    adaptive_time_loop_max_steps_multiplier: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_STEPS_MULTIPLIER,
) -> None:
    face_blocks = "\n".join(
        f"""  <Add_face name="{name}">
    <Face_file_path>{mesh_path.rsplit('/', 1)[0]}/mesh-surfaces/{name}.vtp</Face_file_path>
  </Add_face>"""
        for name in faces
    )
    wall_bcs = [
        name
        for name in faces
        if name.startswith("wall_") and (include_top_wall_bc or name != "wall_top")
    ]
    if include_obstacle_bc:
        wall_bcs.append("obstacle")
    wall_bc_blocks = "\n".join(
        f"""  <Add_BC name="{name}">
    <Type>Dir</Type>
    <Value>0.0</Value>
  </Add_BC>"""
        for name in wall_bcs
    )
    active_domain_block = ""
    if active_domain is not None:
        active_domain_block = f"""
    <Active_domain>{active_domain}</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>
    <Enable_velocity_extension>true</Enable_velocity_extension>
    <Velocity_extension_diffusivity>1.0</Velocity_extension_diffusivity>"""
    cut_metadata_scale_text = "true" if use_cut_metadata_scale else "false"
    cut_cell_stabilization_text = "true" if enable_cut_cell_stabilization else "false"
    use_relaxed_unfitted_controls = use_blockschur_solver or use_direct_fluid_solver
    unfitted_fluid_solver_type = (
        "Direct" if use_direct_fluid_solver else ("NS" if use_blockschur_solver else "GMRES")
    )
    unfitted_fluid_linear_algebra_block = (
        """    <Linear_algebra type="eigen">
      <Preconditioner>none</Preconditioner>
    </Linear_algebra>"""
        if use_direct_fluid_solver
        else """    <Linear_algebra type="fsils">
      <Preconditioner>fsils</Preconditioner>
    </Linear_algebra>"""
    )
    unfitted_level_set_nonlinear_tolerance = (
        TEST05_BLOCKSCHUR_LEVEL_SET_NONLINEAR_TOLERANCE
        if use_relaxed_unfitted_controls
        else LEVEL_SET_NONLINEAR_TOLERANCE
    )
    unfitted_fluid_nonlinear_tolerance = (
        TEST05_BLOCKSCHUR_FLUID_NONLINEAR_TOLERANCE
        if use_relaxed_unfitted_controls
        else FLUID_NONLINEAR_TOLERANCE
    )
    unfitted_fluid_max_iterations = (
        TEST05_BLOCKSCHUR_FLUID_MAX_ITERATIONS if use_relaxed_unfitted_controls else "8"
    )
    unfitted_linear_max_iterations = (
        TEST05_BLOCKSCHUR_LINEAR_MAX_ITERATIONS if use_relaxed_unfitted_controls else "100"
    )
    unfitted_krylov_space_dimension = (
        TEST05_BLOCKSCHUR_KRYLOV_SPACE_DIMENSION if use_relaxed_unfitted_controls else "80"
    )
    unfitted_linear_tolerance = (
        TEST05_BLOCKSCHUR_LINEAR_TOLERANCE
        if use_relaxed_unfitted_controls
        else LINEAR_SOLVER_TOLERANCE
    )
    unfitted_linear_absolute_tolerance = (
        TEST05_BLOCKSCHUR_LINEAR_ABSOLUTE_TOLERANCE
        if use_relaxed_unfitted_controls
        else LINEAR_SOLVER_ABSOLUTE_TOLERANCE
    )
    unfitted_blockschur_controls = ""
    if use_blockschur_solver:
        unfitted_blockschur_controls = f"""
    <NS_GM_max_iterations>{TEST05_BLOCKSCHUR_INNER_MAX_ITERATIONS}</NS_GM_max_iterations>
    <NS_GM_tolerance>{unfitted_linear_tolerance}</NS_GM_tolerance>
    <NS_CG_max_iterations>{TEST05_BLOCKSCHUR_INNER_MAX_ITERATIONS}</NS_CG_max_iterations>
    <NS_CG_tolerance>{unfitted_linear_tolerance}</NS_CG_tolerance>
    <NS_min_outer_iterations>{TEST05_BLOCKSCHUR_MIN_OUTER_ITERATIONS}</NS_min_outer_iterations>
    <NS_Schur_preconditioner>blockdiag-l</NS_Schur_preconditioner>
    <NS_Momentum_approximation>ilu-k</NS_Momentum_approximation>
    <NS_Use_coupled_outer_FGMRES>true</NS_Use_coupled_outer_FGMRES>"""
    hydrostatic_pressure_field_block = (
        f"\n  <Hydrostatic_pressure_field_name>{hydrostatic_pressure_field_name}</Hydrostatic_pressure_field_name>"
        if hydrostatic_pressure_field_name
        else ""
    )
    pressure_constraint_block = (
        """
  <Node_pressure_constraints>
    <Id_type>Global_vertex_gid</Id_type>
    <Values_file_path>pressure_gauge.csv</Values_file_path>
  </Node_pressure_constraints>"""
        if include_pressure_constraints
        else ""
    )
    jit_module_options = "jit=true; jit_specialization=true"
    enable_level_set_maintenance_text = (
        "true" if enable_level_set_maintenance else "false"
    )
    if wet_extension_advection_velocity_method is None:
        level_set_velocity_block = """  <Velocity_source>coupled_field</Velocity_source>
  <Velocity_field_name>Velocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>"""
    else:
        level_set_velocity_block = f"""  <Velocity_source>prescribed_data</Velocity_source>
  <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Use_wet_extension_advection_velocity>true</Use_wet_extension_advection_velocity>
  <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  <Wet_extension_advection_velocity_method>{wet_extension_advection_velocity_method}</Wet_extension_advection_velocity_method>"""

    if fitted:
        equations = f"""
<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>{unfitted_fluid_max_iterations}</Max_iterations>
  <Tolerance>{unfitted_fluid_nonlinear_tolerance}</Tolerance>
  <Module_options>{jit_module_options}</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>

  <Enable_ALE>true</Enable_ALE>
  <Mesh_velocity_source>coupled_displacement</Mesh_velocity_source>
  <Mesh_velocity_field>mesh_velocity</Mesh_velocity_field>
  <Mesh_displacement_field>mesh_displacement</Mesh_displacement_field>
  <Auto_register_mesh_displacement_field>true</Auto_register_mesh_displacement_field>
  <Moving_mesh_tangent_path>SymbolicRequired</Moving_mesh_tangent_path>

  <Density>{WATER_DENSITY}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>-9.81</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 {fill_height:.6g} 0.0</Hydrostatic_pressure_reference_point>{hydrostatic_pressure_field_block}{pressure_constraint_block}
  <Viscosity model="Constant">
    <Value>{WATER_VISCOSITY}</Value>
  </Viscosity>

  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
    <Mesh_displacement>true</Mesh_displacement>
    <Mesh_velocity>true</Mesh_velocity>
    <Surface_position>true</Surface_position>
  </Output>

  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>

  <LS type="GMRES">
    <Linear_algebra type="eigen"><Preconditioner>none</Preconditioner></Linear_algebra>
    <Max_iterations>100</Max_iterations>
    <Krylov_space_dimension>80</Krylov_space_dimension>
    <Tolerance>{LINEAR_SOLVER_TOLERANCE}</Tolerance>
    <Absolute_tolerance>{LINEAR_SOLVER_ABSOLUTE_TOLERANCE}</Absolute_tolerance>
  </LS>

{wall_bc_blocks}

  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>FittedALE</Implementation>
    <External_pressure>0.0</External_pressure>
    <Surface_tension>{SURFACE_TENSION}</Surface_tension>
    <Use_current_geometry_curvature>true</Use_current_geometry_curvature>
    <Kinematic_enforcement>Nitsche</Kinematic_enforcement>
    <Normal_kinematic_policy>MatchFluidNormalVelocity</Normal_kinematic_policy>
    <Tangential_mesh_policy>SmoothingOnly</Tangential_mesh_policy>
    <Kinematic_nitsche_gamma>20.0</Kinematic_nitsche_gamma>
  </Add_BC>
</Add_equation>

<Add_equation type="mesh_motion">
  <Coupled>true</Coupled>
  <Module_options>{jit_module_options}</Module_options>
  <Model>Harmonic</Model>
  <Field_name>mesh_displacement</Field_name>
  <Operator_tag>equations</Operator_tag>
  <Kappa>1.0</Kappa>
  <Moving_mesh_tangent_path>SymbolicRequired</Moving_mesh_tangent_path>

{wall_bc_blocks}
</Add_equation>"""
    else:
        equations = f"""
<Add_equation type="level_set">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>4</Max_iterations>
  <Tolerance>{unfitted_level_set_nonlinear_tolerance}</Tolerance>
  <Module_options>{jit_module_options}</Module_options>

  <Level_set_field_name>phi</Level_set_field_name>
  <Operator_tag>equations</Operator_tag>
  <Level_set_source>prescribed_data</Level_set_source>
{level_set_velocity_block}
  <Enable_SUPG>true</Enable_SUPG>
  <SUPG_tau_scale>0.5</SUPG_tau_scale>
  <Enable_reinitialization>{enable_level_set_maintenance_text}</Enable_reinitialization>
  <Reinitialization_method>projection</Reinitialization_method>
  <Reinitialization_cadence_steps>5</Reinitialization_cadence_steps>
  <Reinitialization_max_iterations>4</Reinitialization_max_iterations>
  <Enable_volume_correction>{enable_level_set_maintenance_text}</Enable_volume_correction>
  <Volume_correction_use_initial_volume>true</Volume_correction_use_initial_volume>
  <Volume_correction_cadence_steps>5</Volume_correction_cadence_steps>
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

  <LS type="GMRES">
    <Linear_algebra type="fsils">
      <Preconditioner>fsils</Preconditioner>
    </Linear_algebra>
    <Max_iterations>50</Max_iterations>
    <Krylov_space_dimension>50</Krylov_space_dimension>
    <Tolerance>{unfitted_linear_tolerance}</Tolerance>
    <Absolute_tolerance>{LINEAR_SOLVER_ABSOLUTE_TOLERANCE}</Absolute_tolerance>
  </LS>
</Add_equation>

<Add_equation type="fluid">
  <Coupled>true</Coupled>
  <Min_iterations>1</Min_iterations>
  <Max_iterations>{unfitted_fluid_max_iterations}</Max_iterations>
  <Tolerance>{unfitted_fluid_nonlinear_tolerance}</Tolerance>
  <Module_options>{jit_module_options}</Module_options>
  <Backflow_stabilization_coefficient>0.0</Backflow_stabilization_coefficient>

  <Density>{WATER_DENSITY}</Density>
  <Force_x>0.0</Force_x>
  <Force_y>-9.81</Force_y>
  <Force_z>0.0</Force_z>
  <Hydrostatic_pressure_initialization>true</Hydrostatic_pressure_initialization>
  <Hydrostatic_pressure_reference>0.0</Hydrostatic_pressure_reference>
  <Hydrostatic_pressure_reference_point>0.0 {fill_height:.6g} 0.0</Hydrostatic_pressure_reference_point>{hydrostatic_pressure_field_block}{pressure_constraint_block}
  <Viscosity model="Constant">
    <Value>{WATER_VISCOSITY}</Value>
  </Viscosity>

  <Output type="Spatial">
    <Velocity>true</Velocity>
    <Pressure>true</Pressure>
    <Divergence>true</Divergence>
  </Output>

  <Output type="Volume_integral">
    <Volume>true</Volume>
  </Output>

  <LS type="{unfitted_fluid_solver_type}">
{unfitted_fluid_linear_algebra_block}
    <Max_iterations>{unfitted_linear_max_iterations}</Max_iterations>
    <Krylov_space_dimension>{unfitted_krylov_space_dimension}</Krylov_space_dimension>
    <Tolerance>{unfitted_linear_tolerance}</Tolerance>
    <Absolute_tolerance>{unfitted_linear_absolute_tolerance}</Absolute_tolerance>{unfitted_blockschur_controls}
  </LS>

{wall_bc_blocks}

  <Add_BC name="free_surface">
    <Type>Free_surface</Type>
    <Implementation>UnfittedLevelSet</Implementation>
    <Level_set_field_name>phi</Level_set_field_name>
    <Generated_interface_domain_id>open_vessel_surface</Generated_interface_domain_id>
    <Level_set_isovalue>0.0</Level_set_isovalue>{active_domain_block}
    <External_pressure>0.0</External_pressure>
    <Surface_tension>{UNFITTED_SURFACE_TENSION}</Surface_tension>
    <Enable_cut_cell_stabilization>{cut_cell_stabilization_text}</Enable_cut_cell_stabilization>
    <Use_cut_metadata_scale>{cut_metadata_scale_text}</Use_cut_metadata_scale>
    <Cut_cell_velocity_gradient_penalty>1.0</Cut_cell_velocity_gradient_penalty>
    <Cut_cell_pressure_gradient_penalty>1.0</Cut_cell_pressure_gradient_penalty>
  </Add_BC>
</Add_equation>"""

    adaptive_time_loop_block = ""
    if enable_adaptive_time_loop:
        adaptive_time_loop_block = f"""
  <Enable_adaptive_time_loop>true</Enable_adaptive_time_loop>
  <Adaptive_time_loop_min_dt>{adaptive_time_loop_min_dt}</Adaptive_time_loop_min_dt>
  <Adaptive_time_loop_max_dt>{adaptive_time_loop_max_dt}</Adaptive_time_loop_max_dt>
  <Adaptive_time_loop_max_retries>{adaptive_time_loop_max_retries}</Adaptive_time_loop_max_retries>
  <Adaptive_time_loop_decrease_factor>{adaptive_time_loop_decrease_factor}</Adaptive_time_loop_decrease_factor>
  <Adaptive_time_loop_increase_factor>{adaptive_time_loop_increase_factor}</Adaptive_time_loop_increase_factor>
  <Adaptive_time_loop_target_newton_iterations>{adaptive_time_loop_target_newton_iterations}</Adaptive_time_loop_target_newton_iterations>
  <Adaptive_time_loop_max_steps_multiplier>{adaptive_time_loop_max_steps_multiplier}</Adaptive_time_loop_max_steps_multiplier>"""

    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>3</Number_of_spatial_dimensions>
  <Number_of_time_steps>{time_steps}</Number_of_time_steps>
  <Time_step_size>{time_step:.8g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>{adaptive_time_loop_block}

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
  <Mesh_file_path>{mesh_path}</Mesh_file_path>

{face_blocks}
</Add_mesh>
{equations}

</svMultiPhysicsFile>
"""
    (case_dir / "solver.xml").write_text(xml)


def prepare_case_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def surface_specs_for_box(box: Box, tol: float, *, fitted: bool) -> list[SurfaceSpec]:
    specs = [
        SurfaceSpec("wall_left", plane_predicate(0, box.xmin, tol)),
        SurfaceSpec("wall_right", plane_predicate(0, box.xmax, tol)),
        SurfaceSpec("wall_bottom", plane_predicate(1, box.ymin, tol)),
        SurfaceSpec("wall_front", plane_predicate(2, box.zmin, tol)),
        SurfaceSpec("wall_back", plane_predicate(2, box.zmax, tol)),
    ]
    if fitted:
        specs.append(SurfaceSpec("free_surface", plane_predicate(1, box.ymax, tol)))
    else:
        specs.append(SurfaceSpec("wall_top", plane_predicate(1, box.ymax, tol)))
    return specs


def write_case(
    *,
    case_dir: Path,
    mesh_subdir: str,
    domain: Box,
    phi: Callable[[np.ndarray], np.ndarray] | None,
    pressure: Callable[[np.ndarray], np.ndarray] | None = None,
    grid_factory: Callable[[], pv.UnstructuredGrid] | None = None,
    fill_height: float,
    gauge_point: tuple[float, float, float],
    metadata: dict,
    h: float,
    fitted: bool,
    time_step: float,
    time_steps: int,
    obstacles: list[Box] | None = None,
    include_top_wall_bc: bool = False,
    active_domain: str | None = None,
    enable_cut_cell_stabilization: bool = True,
    use_cut_metadata_scale: bool = False,
    use_blockschur_solver: bool = False,
    include_pressure_constraints: bool = True,
    hydrostatic_pressure_field_name: str | None = None,
    enable_level_set_maintenance: bool = True,
    gauge_pressure: float | Callable[[np.ndarray], float] = 0.0,
    record_gauge_metadata: bool = False,
    pressure_gauge_verification: Callable[[dict], dict] | None = None,
) -> None:
    obstacles = obstacles or []
    prepare_case_dir(case_dir)
    mesh_dir = case_dir / "mesh" / mesh_subdir
    surface_dir = mesh_dir / "mesh-surfaces"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if grid_factory is not None:
        if obstacles:
            raise ValueError("grid_factory cases do not support obstacles")
        grid = grid_factory()
    else:
        max_volume = h**3 / 6.0
        grid = build_tetgen_grid(domain, max_volume=max_volume, obstacles=obstacles)
        grid = run_mmg(grid, hmin=0.35 * h, hmax=1.25 * h, hausd=0.15 * h)
        grid = remove_cells_inside(grid, obstacles)
    add_solution_arrays(grid, phi=phi, pressure=pressure, fitted=fitted)

    mesh_path = mesh_dir / "mesh-complete.mesh.vtu"
    grid.save(mesh_path, binary=False)

    tol = max(0.35 * h, 1.0e-8)
    specs = surface_specs_for_box(domain, tol, fitted=fitted)
    if obstacles:
        specs.append(SurfaceSpec("obstacle", obstacle_predicate(obstacles[0], tol)))
    write_surfaces(grid, specs, surface_dir)

    gauge_metadata = write_pressure_gauge(case_dir, grid, gauge_point, gauge_pressure)
    if record_gauge_metadata:
        metadata = dict(metadata)
        metadata["pressure_gauge"] = gauge_metadata
        if pressure_gauge_verification is not None:
            metadata["pressure_gauge_verification"] = pressure_gauge_verification(gauge_metadata)
    write_metadata(case_dir, metadata)
    write_solver_xml(
        case_dir,
        mesh_path=f"mesh/{mesh_subdir}/mesh-complete.mesh.vtu",
        faces=[spec.name for spec in specs],
        fitted=fitted,
        fill_height=fill_height,
        time_step=time_step,
        time_steps=time_steps,
        include_top_wall_bc=include_top_wall_bc,
        include_obstacle_bc=bool(obstacles),
        active_domain=active_domain,
        enable_cut_cell_stabilization=enable_cut_cell_stabilization,
        use_cut_metadata_scale=use_cut_metadata_scale,
        use_blockschur_solver=use_blockschur_solver,
        include_pressure_constraints=include_pressure_constraints,
        hydrostatic_pressure_field_name=hydrostatic_pressure_field_name,
        enable_level_set_maintenance=enable_level_set_maintenance,
    )

    print(
        f"{case_dir.relative_to(ROOT)}: {grid.n_points} points, {grid.n_cells} tetrahedra"
    )


def sloshing_metadata(name: str, fitted: bool, fill_height: float) -> dict:
    return {
        "benchmark": name,
        "representation": "fitted_ale" if fitted else "unfitted_level_set",
        "source_urls": ["https://www.spheric-sph.org/tests/test-10"],
        "mesh_tools": ["PyVista", "TetGen", "MMG"],
        "dimensions_m": {
            "tank_length": 0.900,
            "tank_breadth_1x": 0.062,
            "tank_height": 0.508,
            "initial_fill_height": fill_height,
        },
        "notes": [
            "The mesh encodes the 1x rectangular tank and still-water fill level.",
            "The published roll-angle history and pressure records are external benchmark data.",
        ],
    }


def sloshing_pressure_gauge_verification(gauge_metadata: dict) -> dict:
    current_pressure = float(gauge_metadata["expected_initial_hydrostatic_pressure"])
    return {
        "current_prescribed_pressure_matches_initial_hydrostatic": True,
        "initial_pressure_error_after_constraint": 0.0,
        "previous_surface_target_pressure": 0.0,
        "previous_surface_target_pressure_error": -current_pressure,
    }


def generate_spheric_test10() -> None:
    tank = Box(0.0, 0.900, 0.0, 0.508, 0.0, 0.062)
    lateral_fill = 0.093
    water = Box(tank.xmin, tank.xmax, tank.ymin, lateral_fill, tank.zmin, tank.zmax)

    write_case(
        case_dir=ROOT / "fitted_ale" / "spheric_test10_lateral_water_1x",
        mesh_subdir="water",
        domain=water,
        phi=None,
        fill_height=lateral_fill,
        gauge_point=(0.45, lateral_fill, 0.031),
        gauge_pressure=lambda point: hydrostatic_pressure_at_point(point, lateral_fill),
        record_gauge_metadata=True,
        pressure_gauge_verification=sloshing_pressure_gauge_verification,
        metadata=sloshing_metadata("SPHERIC Test 10 lateral water 1x", True, lateral_fill),
        h=0.045,
        fitted=True,
        time_step=0.001,
        time_steps=40,
    )

    write_case(
        case_dir=ROOT / "unfitted_level_set" / "spheric_test10_lateral_water_1x",
        mesh_subdir="background",
        domain=tank,
        phi=lambda points: points[:, 1] - lateral_fill,
        fill_height=lateral_fill,
        gauge_point=(0.45, lateral_fill, 0.031),
        gauge_pressure=lambda point: hydrostatic_pressure_at_point(point, lateral_fill),
        record_gauge_metadata=True,
        pressure_gauge_verification=sloshing_pressure_gauge_verification,
        metadata=sloshing_metadata("SPHERIC Test 10 lateral water 1x", False, lateral_fill),
        h=0.055,
        fitted=False,
        time_step=0.001,
        time_steps=40,
        include_top_wall_bc=True,
    )


def test05_pressure_gauge_verification(gauge_metadata: dict) -> dict:
    current_pressure = float(gauge_metadata["expected_initial_hydrostatic_pressure"])
    previous_pressure = TEST05_PREVIOUS_INVALID_D18_GAUGE["full_volume_hydrostatic_pressure"]
    previous_range = TEST05_PREVIOUS_INVALID_D18_GAUGE["hydrostatic_error_range"]
    return {
        "current_prescribed_pressure_matches_initial_hydrostatic": True,
        "initial_pressure_error_after_constraint": 0.0,
        "previous_invalid_d18_node_id": TEST05_PREVIOUS_INVALID_D18_GAUGE["node_id"],
        "previous_invalid_d18_initial_phi": TEST05_PREVIOUS_INVALID_D18_GAUGE["initial_phi"],
        "previous_invalid_d18_full_volume_hydrostatic_pressure": previous_pressure,
        "previous_invalid_d18_hydrostatic_error_range": previous_range,
        "current_pressure_matches_previous_invalid_offset": False,
        "current_pressure_matches_previous_invalid_error_range": (
            previous_range[0] <= current_pressure <= previous_range[1]
        ),
        "current_pressure_minus_previous_invalid_pressure": current_pressure - previous_pressure,
    }


def test05_reference_profiles(wet_depth_mm: int) -> list[dict[str, object]]:
    return [
        {
            "time_s": time_s,
            "path": (
                "tests/cases/fluid/open_vessel_free_surface/reference_profiles/"
                f"spheric_test05_wet_bed/d{wet_depth_mm}_{index}.dat"
            ),
        }
        for index, time_s in enumerate(TEST05_REFERENCE_PROFILE_TIMES[wet_depth_mm], start=1)
    ]


def generate_spheric_test05() -> None:
    domain = Box(0.0, 1.20, 0.0, 0.18, 0.0, 0.03)
    gate_x = 0.38
    dam_height = 0.15
    source_urls = ["https://www.spheric-sph.org/tests/test-05"]

    for wet_depth_mm, wet_depth in ((18, 0.018), (38, 0.038)):
        x_coords = centered_cut_segmented_coordinates(
            domain.xmin,
            domain.xmax,
            (gate_x,),
            max_spacing=0.0225,
        )
        y_coords = test05_vertical_coordinates(domain, wet_depth, dam_height)
        z_coords = segmented_coordinates(
            (domain.zmin, 0.5 * (domain.zmin + domain.zmax), domain.zmax),
            max_spacing=0.015,
        )

        def grid_factory(
            x_coords: np.ndarray = x_coords,
            y_coords: np.ndarray = y_coords,
            z_coords: np.ndarray = z_coords,
        ) -> pv.UnstructuredGrid:
            return structured_tet_grid(
                x_coords,
                y_coords,
                z_coords,
                mirror_z_midplane=True,
            )

        def phi(points: np.ndarray, wet_depth: float = wet_depth) -> np.ndarray:
            return dam_break_wet_bed_phi(
                points,
                gate_x=gate_x,
                dam_height=dam_height,
                wet_depth=wet_depth,
            )

        def pressure(points: np.ndarray, wet_depth: float = wet_depth) -> np.ndarray:
            return dam_break_wet_bed_pressure(
                points,
                gate_x=gate_x,
                dam_height=dam_height,
                wet_depth=wet_depth,
            )

        write_case(
            case_dir=ROOT / "unfitted_level_set" / f"spheric_test05_wet_bed_d{wet_depth_mm}",
            mesh_subdir="background",
            domain=domain,
            phi=phi,
            pressure=pressure,
            grid_factory=grid_factory,
            fill_height=dam_height,
            gauge_point=(0.10, 0.075, 0.015),
            gauge_pressure=lambda point, wet_depth=wet_depth: float(
                pressure(point.reshape(1, 3))[0]
            ),
            record_gauge_metadata=True,
            pressure_gauge_verification=test05_pressure_gauge_verification,
            metadata={
                "benchmark": f"SPHERIC Test 05 wet-bed dam break d={wet_depth_mm} mm",
                "representation": "unfitted_level_set",
                "source_urls": source_urls,
                "reference_profiles": test05_reference_profiles(wet_depth_mm),
                "mesh_tools": ["PyVista", "mirrored structured thin-extrusion tetrahedral grid"],
                "dimensions_m": {
                    "initial_column_height": dam_height,
                    "wet_bed_depth": wet_depth,
                    "profile_window_x_min": 0.38,
                    "profile_window_x_max": 1.04,
                    "computational_length": domain.xmax,
                    "computational_height": domain.ymax,
                    "extrusion_breadth": domain.zmax,
                },
                "notes": [
                    "The three-dimensional mesh is a thin extrusion of the published two-dimensional setup.",
                    "The tetrahedral split is mirrored about the extrusion mid-plane to suppress transverse bias.",
                    "The initial level-set field is the union of the wet bed and the retained water column.",
                    "Negative level-set values denote the water side for active-domain assembly.",
                ],
            },
            h=0.006,
            fitted=False,
            time_step=0.0005,
            time_steps=312,
            include_top_wall_bc=True,
            active_domain="LevelSetNegative",
            enable_cut_cell_stabilization=True,
            use_cut_metadata_scale=False,
            use_direct_fluid_solver=True,
            include_pressure_constraints=True,
            hydrostatic_pressure_field_name="Pressure",
            enable_level_set_maintenance=False,
            wet_extension_advection_velocity_method="nearest_interface_point",
            enable_adaptive_time_loop=True,
        )


def generate_spheric_test02() -> None:
    tank = Box(0.0, 3.22, 0.0, 1.00, 0.0, 1.00)
    water_column = Box(3.22 - 0.58, 3.22, 0.0, 0.55, 0.0, 1.00)
    obstacle = Box(3.22 - 2.40 - 0.20, 3.22 - 2.40 + 0.20, 0.0, 0.16, 0.42, 0.58)

    write_case(
        case_dir=ROOT / "unfitted_level_set" / "spheric_test02_dambreak_obstacle",
        mesh_subdir="background",
        domain=tank,
        phi=lambda points: signed_distance_to_box(points, water_column),
        fill_height=0.55,
        gauge_point=(3.22 - 0.58, 0.55, 0.50),
        metadata={
            "benchmark": "SPHERIC Test 02 three-dimensional dam break with obstacle",
            "representation": "unfitted_level_set",
            "source_urls": [
                "https://www.spheric-sph.org/tests/test-02",
                "https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.00346/full",
            ],
            "mesh_tools": ["PyVista", "TetGen", "MMG"],
            "dimensions_m": {
                "tank_length": 3.22,
                "tank_width": 1.00,
                "tank_height": 1.00,
                "initial_column_length": 0.58,
                "initial_column_x_min": 3.22 - 0.58,
                "initial_column_x_max": 3.22,
                "initial_column_height": 0.55,
                "obstacle_length": 0.40,
                "obstacle_width": 0.16,
                "obstacle_height": 0.16,
                "obstacle_center_x": 3.22 - 2.40,
                "obstacle_center_z": 0.50,
            },
            "notes": [
                "The obstacle is represented as an internal no-slip boundary in the background mesh.",
                "The initial level-set field is negative inside the retained water column.",
            ],
        },
        h=0.16,
        fitted=False,
        time_step=0.001,
        time_steps=120,
        obstacles=[obstacle],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["all", "spheric_test10", "spheric_test05", "spheric_test02"],
        default="all",
    )
    args = parser.parse_args()

    if args.case in {"all", "spheric_test10"}:
        generate_spheric_test10()
    if args.case in {"all", "spheric_test05"}:
        generate_spheric_test05()
    if args.case in {"all", "spheric_test02"}:
        generate_spheric_test02()


if __name__ == "__main__":
    main()
