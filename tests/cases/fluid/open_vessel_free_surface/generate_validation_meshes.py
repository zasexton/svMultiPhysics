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
# Triggered interface-preserving redistancing (Enable_interface_preserving_
# redistancing, decision D39): after the kinematic reconciliation, once the
# interface has moved one cell, the level set is replaced by the distance to its
# own zero contour and the interface is restored; the trigger of the sessile and
# capillary-rise protocols, with no parameter of its own.  With the
# reconciliation (D14) and the sign-definite patch bounds (D21) it is the
# accepted-step maintenance of the M1-M4 protocols (D35 item 1; D39 item 4: on in
# the generators).  "--interface-preserving-redistancing off" writes the decks of
# b94af982 byte for byte.
INTERFACE_PRESERVING_REDISTANCING = True
# Wall model of the SPHERIC validation decks (decision D58): free-slip on every
# wall, the Test 10 lid and the Test 02 obstacle included.  Only the normal
# velocity is fixed: a zero Dirichlet value whose Effective_direction selects the
# component along the face normal, so the obstacle is written as one face file per
# face plane.  No-slip pins the contact lines of the sharp level set at the walls.
# "--wall-condition no-slip" writes the no-slip decks of 8f145c4c byte for byte.
SPHERIC_FREE_SLIP_WALLS = True
# Ghost layers of the 3D SPHERIC decks.  Distributed small-cut aggregation needs
# every rank that condenses a slave DOF to hold the masters of its root cell: the
# halo must reach 1 + S + 8 face layers, with S the face-layer depth of a vertex
# star and 8 the root-path guard.  S measured on the meshes (S8, 2026-10-26): 6 on
# plain Kuhn tetrahedra (15 layers), 7 on the Test 02 mesh (16), 9 on mirrored Kuhn
# meshes (18).  18 covers them all and matches the solver's derived 3D default.
# The solver's earlier 3D value (12) covered roots up to 5 steps; Test 02 at
# 4 ranks stopped at 0.358 s with roots 7-8 steps away whose masters the
# condensing rank did not hold (job 324).  While the halo covers the stencil,
# ghost layers move no value (12, 15 and 18 layers are bitwise equal at 4 ranks);
# 0 writes no key, the solver's default, as the decks of 9802903f.
SPHERIC_3D_GHOST_LAYERS = 18
# The coordinate axis of each face file's normal (free-slip component masks).
FACE_NORMAL_AXIS = {
    "wall_left": 0, "wall_right": 0,
    "wall_bottom": 1, "wall_top": 1,
    "wall_front": 2, "wall_back": 2,
    "obstacle_xmin": 0, "obstacle_xmax": 0,
    "obstacle_top": 1,
    "obstacle_zmin": 2, "obstacle_zmax": 2,
}


def free_slip_effective_directions(face_names: Iterable[str], spatial_dimension: int) -> dict[str, str]:
    """Effective_direction of a free-slip wall: the component along the face normal."""
    directions = {}
    for name in face_names:
        if name not in FACE_NORMAL_AXIS:
            raise ValueError(f"face {name!r} has no known normal axis for a free-slip condition")
        axis = FACE_NORMAL_AXIS[name]
        if axis >= spatial_dimension:
            raise ValueError(f"face {name!r} has a normal along axis {axis} in {spatial_dimension}D")
        directions[name] = " ".join("1" if d == axis else "0" for d in range(spatial_dimension))
    return directions
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
TEST02_TANK_LENGTH = 3.22
TEST02_TANK_WIDTH = 1.00
TEST02_TANK_HEIGHT = 1.00
TEST02_INITIAL_COLUMN_LENGTH = 1.228
TEST02_INITIAL_COLUMN_HEIGHT = 0.55
TEST02_GATE_X = TEST02_TANK_LENGTH - TEST02_INITIAL_COLUMN_LENGTH
TEST02_OBSTACLE_X_MIN = 0.6635
TEST02_OBSTACLE_LENGTH = 0.161
TEST02_OBSTACLE_WIDTH = 0.403
TEST02_OBSTACLE_HEIGHT = 0.161
TEST02_OBSTACLE_X_MAX = TEST02_OBSTACLE_X_MIN + TEST02_OBSTACLE_LENGTH
TEST02_OBSTACLE_Z_MIN = 0.5 * (TEST02_TANK_WIDTH - TEST02_OBSTACLE_WIDTH)
TEST02_OBSTACLE_Z_MAX = 0.5 * (TEST02_TANK_WIDTH + TEST02_OBSTACLE_WIDTH)
TEST02_HEIGHT_PROBES = {
    "H1": 0.496,
    "H2": 0.992,
    "H3": 1.488,
    "H4": 2.632,
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


def aligned_and_centered_coordinates(
    start: float,
    stop: float,
    *,
    aligned_values: Iterable[float] = (),
    centered_cut_values: Iterable[float] = (),
    max_spacing: float,
    centering_gap: float | None = None,
) -> np.ndarray:
    anchors = [float(start), float(stop), *(float(value) for value in aligned_values)]
    half_width = 0.5 * (max_spacing if centering_gap is None else centering_gap)
    for cut in centered_cut_values:
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


def structured_triangle_grid(x_coords: np.ndarray, y_coords: np.ndarray) -> pv.UnstructuredGrid:
    """A 2D structured grid of the plane z = 0 split into right triangles,
    the diagonal alternating with the (i + j) parity of the cell (as the 2D
    free-surface benchmarks do), every triangle counterclockwise."""
    nx = len(x_coords)
    ny = len(y_coords)
    points = np.array([[x, y, 0.0] for y in y_coords for x in x_coords], dtype=float)

    def node(i: int, j: int) -> int:
        return j * nx + i

    elements: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            if (i + j) % 2 == 0:
                elements.extend([(a, b, c), (a, c, d)])
            else:
                elements.extend([(a, b, d), (b, c, d)])
    triangles = np.array(elements, dtype=np.int64)
    cells = np.hstack([np.full((triangles.shape[0], 1), 3, dtype=np.int64), triangles]).ravel()
    cell_types = np.full(triangles.shape[0], int(pv.CellType.TRIANGLE), dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, points)


def grid_tets(grid: pv.UnstructuredGrid) -> np.ndarray:
    cells = grid.cells.reshape((-1, 5))
    return cells[:, 1:].astype(np.int64)


def grid_spatial_dimension(grid: pv.UnstructuredGrid) -> int:
    """3 for a tetrahedral grid, 2 for a triangle grid of the plane z = 0."""
    cell_types = set(int(value) for value in np.unique(grid.celltypes))
    if cell_types == {int(pv.CellType.TETRA)}:
        return 3
    if cell_types == {int(pv.CellType.TRIANGLE)}:
        return 2
    raise ValueError(f"unsupported cell types {sorted(cell_types)}: tetrahedra or triangles only")


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


def boundary_edges(triangles: np.ndarray) -> list[tuple[tuple[int, int], int]]:
    """The boundary edges of a triangle grid, each with its parent triangle."""
    counts: dict[tuple[int, int], list] = {}
    for parent, tri in enumerate(triangles):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(int(a), int(b)), max(int(a), int(b)))
            if key not in counts:
                counts[key] = [0, (int(a), int(b)), parent]
            counts[key][0] += 1
    return [(value[1], value[2]) for value in counts.values() if value[0] == 1]


def polydata_from_edges(
    points: np.ndarray,
    edges: list[tuple[tuple[int, int], int]],
) -> pv.PolyData:
    """Line face file of a 2D grid: GlobalNodeID per point and the parent
    triangle as GlobalElementID per line (as the 2D benchmark face files)."""
    used = sorted({node for edge, _ in edges for node in edge})
    local = {node: i for i, node in enumerate(used)}
    lines = np.array([[2, local[a], local[b]] for (a, b), _ in edges], dtype=np.int64).ravel()
    poly = pv.PolyData(points[np.array(used, dtype=np.int64)], lines=lines)
    poly.point_data["GlobalNodeID"] = np.array(used, dtype=np.int32)
    poly.cell_data["GlobalElementID"] = np.array([parent for _, parent in edges], dtype=np.int32)
    return poly


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
    *,
    allow_overlapping_face_files: bool = False,
    require_complete_coverage: bool = False,
) -> None:
    surface_dir.mkdir(parents=True, exist_ok=True)
    points = grid.points
    dimension = grid_spatial_dimension(grid)
    if dimension == 3:
        facets = [(face, None) for face in boundary_faces(grid_tets(grid))]
    else:
        facets = boundary_edges(grid.cells.reshape((-1, 4))[:, 1:].astype(np.int64))

    selections: list[tuple[SurfaceSpec, list]] = []
    owners: dict[tuple[int, ...], list[str]] = {}
    for spec in specs:
        selected = [
            (facet, parent)
            for facet, parent in facets
            if spec.predicate(points[np.array(facet, dtype=np.int64)], points[np.array(facet, dtype=np.int64)].mean(axis=0))
        ]
        if not selected:
            raise RuntimeError(f"surface {spec.name!r} did not select any boundary faces")
        for facet, _ in selected:
            owners.setdefault(tuple(sorted(facet)), []).append(spec.name)
        selections.append((spec, selected))
    if require_complete_coverage:
        # Every boundary face belongs to a face file: an unlabeled face would
        # get no boundary condition.
        missing = [facet for facet, _ in facets if tuple(sorted(facet)) not in owners]
        if missing:
            raise RuntimeError(
                f"{surface_dir}: {len(missing)} boundary faces are in no face file "
                f"(first: GlobalNodeID {sorted(missing[0])})"
            )
    # The centroid predicates below use a tolerance of 0.35 h, so faces of one
    # plane near an edge are also selected for the adjacent plane.  The solver
    # keeps one boundary label per face (the last face file listed), so such
    # faces would lose the conditions of the other files; the solver rejects
    # overlapping face files unless the deck opts out (decision D35).  Fail
    # closed here as well; --allow-overlapping-face-files reproduces the
    # legacy decks, which declare the solver opt-out.  The fitted MeshNitsche
    # decks use generate_spheric_test10_fitted_decks.py, which writes disjoint
    # face sets.
    message = face_file_overlap_message(surface_dir, owners)
    if message:
        if not allow_overlapping_face_files:
            raise RuntimeError(
                message
                + " A boundary face carries one label in the solver (the last listed face file),"
                " so the boundary conditions of the other files would miss these faces."
                " Make the face selections disjoint, or pass --allow-overlapping-face-files"
                " to reproduce the legacy decks (they declare"
                " <Allow_overlapping_face_files>true</Allow_overlapping_face_files>)."
            )
        warnings.warn(message + " (legacy decks: allowed by --allow-overlapping-face-files)")
    for spec, selected in selections:
        if dimension == 3:
            poly = polydata_from_faces(points, [facet for facet, _ in selected])
        else:
            poly = polydata_from_edges(points, selected)
        poly.save(surface_dir / f"{spec.name}.vtp", binary=False)


def face_file_overlap_message(surface_dir: Path, owners: dict[tuple[int, ...], list[str]]) -> str:
    """Describe the boundary faces listed in more than one face file ("" if none)."""
    shared: dict[tuple[str, ...], list[tuple[int, ...]]] = {}
    for face, names in owners.items():
        if len(set(names)) > 1:
            shared.setdefault(tuple(dict.fromkeys(names)), []).append(face)
    if not shared:
        return ""
    groups = "; ".join(
        f"{len(group_faces)} boundary faces are in "
        + ", ".join(f"{name}.vtp" for name in names)
        + f" (first: GlobalNodeID {list(min(group_faces))})"
        for names, group_faces in sorted(shared.items())
    )
    return f"{surface_dir}: face files overlap: {groups}."


def plane_predicate(axis: int, value: float, tol: float) -> Callable[[np.ndarray, np.ndarray], bool]:
    def predicate(face_points: np.ndarray, center: np.ndarray) -> bool:
        del face_points
        return abs(center[axis] - value) <= tol

    return predicate


def plane_face_predicate(axis: int, value: float, tol: float) -> Callable[[np.ndarray, np.ndarray], bool]:
    """A boundary face (or 2D edge) on the plane: every vertex on it.  On a
    structured box grid each boundary face lies in exactly one plane, so the
    selections are disjoint; a face at an edge of the box has vertices on two
    planes but not all of them on both."""

    def predicate(face_points: np.ndarray, center: np.ndarray) -> bool:
        del center
        return bool(np.all(np.abs(face_points[:, axis] - value) <= tol))

    return predicate


def obstacle_face_predicate(obstacle: Box, tol: float) -> Callable[[np.ndarray, np.ndarray], bool]:
    """A face of the obstacle: every vertex within its bounds and on one of
    its vertical planes or on its top (the obstacle stands on the floor)."""

    def predicate(face_points: np.ndarray, center: np.ndarray) -> bool:
        del center
        lower = np.array([obstacle.xmin, obstacle.ymin, obstacle.zmin]) - tol
        upper = np.array([obstacle.xmax, obstacle.ymax, obstacle.zmax]) + tol
        if not np.all((face_points >= lower) & (face_points <= upper)):
            return False
        planes = (
            (0, obstacle.xmin), (0, obstacle.xmax),
            (2, obstacle.zmin), (2, obstacle.zmax),
            (1, obstacle.ymax),
        )
        return any(bool(np.all(np.abs(face_points[:, axis] - value) <= tol)) for axis, value in planes)

    return predicate


def obstacle_plane_specs(obstacle: Box, tol: float) -> list["SurfaceSpec"]:
    """One face file per face plane of the obstacle (it stands on the floor, so it
    has no bottom face): every face vertex within the obstacle bounds and on the
    plane.  A free-slip condition fixes one velocity component per file (D58)."""
    lower = np.array([obstacle.xmin, obstacle.ymin, obstacle.zmin]) - tol
    upper = np.array([obstacle.xmax, obstacle.ymax, obstacle.zmax]) + tol

    def on_plane(axis: int, value: float) -> Callable[[np.ndarray, np.ndarray], bool]:
        def predicate(face_points: np.ndarray, center: np.ndarray) -> bool:
            del center
            return bool(np.all((face_points >= lower) & (face_points <= upper))
                        and np.all(np.abs(face_points[:, axis] - value) <= tol))
        return predicate

    return [
        SurfaceSpec("obstacle_xmin", on_plane(0, obstacle.xmin)),
        SurfaceSpec("obstacle_xmax", on_plane(0, obstacle.xmax)),
        SurfaceSpec("obstacle_zmin", on_plane(2, obstacle.zmin)),
        SurfaceSpec("obstacle_zmax", on_plane(2, obstacle.zmax)),
        SurfaceSpec("obstacle_top", on_plane(1, obstacle.ymax)),
    ]


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


def dam_break_dry_bed_column_phi(
    points: np.ndarray,
    *,
    gate_x: float,
    dam_height: float,
) -> np.ndarray:
    return np.maximum(gate_x - points[:, 0], points[:, 1] - dam_height)


def dam_break_dry_bed_column_pressure(
    points: np.ndarray,
    *,
    gate_x: float,
    dam_height: float,
) -> np.ndarray:
    """Signed hydrostatic continuation for the retained liquid column.

    The pressure unknown exists only on the retained CutVolume support, but
    dry vertices of cut cells still provide P1 interpolation coefficients.
    Continuing the liquid pressure through those vertices makes the trace at
    the horizontal free surface exactly zero and preserves the liquid-side
    hydrostatic trace on the newly exposed vertical dam face.  Clipping the
    dry coefficients to zero would move both traces inside the cut cell.
    """
    del gate_x  # The gate selects phi; the liquid-column pressure is vertical.
    return WATER_DENSITY * GRAVITY * (dam_height - points[:, 1])


def dam_break_wet_bed_pressure(
    points: np.ndarray,
    *,
    gate_x: float,
    dam_height: float,
    wet_depth: float,
) -> np.ndarray:
    surface_height = np.where(points[:, 0] <= gate_x, dam_height, wet_depth)
    # Retain a signed continuation on the dry vertices of cut cells so P1
    # interpolation reaches zero at each local horizontal free surface.
    return WATER_DENSITY * GRAVITY * (surface_height - points[:, 1])


def test05_vertical_coordinates(
    domain: Box,
    wet_depth: float,
    dam_height: float,
    *,
    fine_spacing: float = 0.006,
    coarse_spacing: float = 0.010,
) -> np.ndarray:
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
    spatial_dimension: int = 3,
    level_set_maintenance: bool = False,
    interface_preserving_redistancing: bool = False,
    mesh_path: str,
    faces: list[str],
    fitted: bool,
    fill_height: float,
    time_step: float,
    time_steps: int,
    ghost_layers: int = 0,
    include_top_wall_bc: bool = False,
    include_obstacle_bc: bool = False,
    level_set_outflow_faces: list[str] | None = None,
    wall_effective_directions: dict[str, str] | None = None,
    active_domain: str | None = None,
    enable_cut_cell_stabilization: bool = True,
    use_cut_metadata_scale: bool = False,
    use_blockschur_solver: bool = False,
    use_direct_fluid_solver: bool = False,
    include_pressure_constraints: bool = True,
    hydrostatic_pressure_field_name: str | None = None,
    momentum_source_file_path: str | None = None,
    rotating_frame_angular_velocity_file_path: str | None = None,
    advection_velocity_extension_method: str | None = None,
    enable_adaptive_time_loop: bool = False,
    adaptive_time_loop_min_dt: str = TEST05_ADAPTIVE_TIME_LOOP_MIN_DT,
    adaptive_time_loop_max_dt: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_DT,
    adaptive_time_loop_max_retries: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_RETRIES,
    adaptive_time_loop_decrease_factor: str = TEST05_ADAPTIVE_TIME_LOOP_DECREASE_FACTOR,
    adaptive_time_loop_increase_factor: str = TEST05_ADAPTIVE_TIME_LOOP_INCREASE_FACTOR,
    adaptive_time_loop_target_newton_iterations: str = TEST05_ADAPTIVE_TIME_LOOP_TARGET_NEWTON_ITERATIONS,
    adaptive_time_loop_max_steps_multiplier: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_STEPS_MULTIPLIER,
    newton_line_search_fail_on_no_reduction: bool = False,
    newton_line_search_max_iterations: str = "10",
    allow_overlapping_face_files: bool = False,
) -> None:
    if ghost_layers < 0:
        raise ValueError("ghost_layers must be nonnegative")
    face_overlap_opt_out = (
        "\n  <Allow_overlapping_face_files>true</Allow_overlapping_face_files>"
        if allow_overlapping_face_files
        else ""
    )
    # Without an explicit count the solver derives the ghost depth.
    ghost_layers_block = f"\n  <Ghost_layers>{ghost_layers}</Ghost_layers>" if ghost_layers > 0 else ""
    level_set_outflow_faces = level_set_outflow_faces or []
    unknown_level_set_outflows = set(level_set_outflow_faces).difference(faces)
    if unknown_level_set_outflows:
        raise ValueError(
            "level_set_outflow_faces must name declared mesh faces: "
            + ", ".join(sorted(unknown_level_set_outflows))
        )
    wall_effective_directions = wall_effective_directions or {}
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
        # One obstacle face file, or one per face plane (free-slip, D58).
        wall_bcs.extend(name for name in faces if name == "obstacle" or name.startswith("obstacle_"))

    def wall_bc_block(name: str) -> str:
        direction = wall_effective_directions.get(name)
        direction_block = (
            f"\n    <Effective_direction>{direction}</Effective_direction>"
            if direction is not None
            else ""
        )
        return f"""  <Add_BC name="{name}">
    <Type>Dir</Type>
    <Value>0.0</Value>{direction_block}
  </Add_BC>"""

    wall_bc_blocks = "\n".join(
        wall_bc_block(name)
        for name in wall_bcs
    )
    level_set_outflow_blocks = "".join(
        f"""
  <Add_BC name="{name}">
    <Type>LevelSetOutflow</Type>
  </Add_BC>"""
        for name in level_set_outflow_faces
    )
    if spatial_dimension not in (2, 3):
        raise ValueError("spatial_dimension must be 2 or 3")
    active_domain_block = ""
    if active_domain is not None:
        # The Navier-Stokes-side velocity extension of the free-surface BC
        # is legacy and fails closed; the level set carries the PDE
        # extension (D55).
        active_domain_block = f"""
    <Active_domain>{active_domain}</Active_domain>
    <Active_domain_method>CutVolume</Active_domain_method>"""
    # The current accepted-step maintenance of the M1-M4 protocols: kinematic
    # reconciliation (D14), sign-definite patch bounds (D21) and, unless switched
    # off, triggered interface-preserving redistancing (D39).
    if interface_preserving_redistancing and not level_set_maintenance:
        raise ValueError("interface-preserving redistancing is part of the level-set maintenance")
    redistancing_line = ("""
  <Enable_interface_preserving_redistancing>true</Enable_interface_preserving_redistancing>"""
                         if interface_preserving_redistancing else "")
    level_set_maintenance_block = (
        f"""
  <Enable_kinematic_reconciliation>true</Enable_kinematic_reconciliation>{redistancing_line}
  <Enable_sign_definite_patch_bounds>true</Enable_sign_definite_patch_bounds>"""
        if level_set_maintenance
        else ""
    )
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
    if advection_velocity_extension_method is None:
        level_set_velocity_block = """  <Velocity_source>coupled_field</Velocity_source>
  <Velocity_field_name>Velocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>"""
    else:
        level_set_velocity_block = f"""  <Velocity_source>prescribed_data</Velocity_source>
  <Velocity_field_name>LevelSetAdvectionVelocity</Velocity_field_name>
  <Auto_register_velocity_field>true</Auto_register_velocity_field>
  <Source_velocity_field_name>Velocity</Source_velocity_field_name>
  <Advection_velocity_extension_method>{advection_velocity_extension_method}</Advection_velocity_extension_method>
  <Advection_velocity_extension_coupling>monolithic</Advection_velocity_extension_coupling>"""
    momentum_source_block = (
        "\n  <Momentum_source_temporal_and_spatial_values_file_path>"
        f"{momentum_source_file_path}"
        "</Momentum_source_temporal_and_spatial_values_file_path>"
        if momentum_source_file_path is not None
        else ""
    )
    rotating_frame_angular_velocity_block = (
        "\n  <Rotating_frame_angular_velocity_temporal_values_file_path>"
        f"{rotating_frame_angular_velocity_file_path}"
        "</Rotating_frame_angular_velocity_temporal_values_file_path>"
        if rotating_frame_angular_velocity_file_path is not None
        else ""
    )

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
  <Force_z>0.0</Force_z>{momentum_source_block}{rotating_frame_angular_velocity_block}
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
  <SUPG_transient_scale>2.0</SUPG_transient_scale>{level_set_maintenance_block}

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
  </LS>{level_set_outflow_blocks}
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
  <Force_z>0.0</Force_z>{momentum_source_block}{rotating_frame_angular_velocity_block}
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
    <Small_cut_aggregation>true</Small_cut_aggregation>
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
    newton_line_search_block = ""
    if newton_line_search_fail_on_no_reduction:
        newton_line_search_block = f"""
  <Newton_line_search_fail_on_no_reduction>true</Newton_line_search_fail_on_no_reduction>
  <Newton_line_search_max_iterations>{newton_line_search_max_iterations}</Newton_line_search_max_iterations>"""

    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Use_new_OOP_solver>true</Use_new_OOP_solver>
  <Continue_previous_simulation>false</Continue_previous_simulation>
  <Number_of_spatial_dimensions>{spatial_dimension}</Number_of_spatial_dimensions>
  <Number_of_time_steps>{time_steps}</Number_of_time_steps>
  <Time_step_size>{time_step:.8g}</Time_step_size>
  <Spectral_radius_of_infinite_time_step>0.50</Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop>STOP_SIM</Searched_file_name_to_trigger_stop>{adaptive_time_loop_block}{newton_line_search_block}

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
  <Mesh_file_path>{mesh_path}</Mesh_file_path>{ghost_layers_block}{face_overlap_opt_out}

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


def surface_specs_for_box(
    box: Box,
    tol: float,
    *,
    fitted: bool,
    exact_planes: bool = False,
    spatial_dimension: int = 3,
) -> list[SurfaceSpec]:
    """The face files of a box: centroid within tol of a plane (legacy: the
    files overlap at the edges), or with exact_planes every face vertex
    within tol of the plane (disjoint).  A 2D rectangle has no front/back."""
    plane = plane_face_predicate if exact_planes else plane_predicate
    specs = [
        SurfaceSpec("wall_left", plane(0, box.xmin, tol)),
        SurfaceSpec("wall_right", plane(0, box.xmax, tol)),
        SurfaceSpec("wall_bottom", plane(1, box.ymin, tol)),
    ]
    if spatial_dimension == 3:
        specs += [
            SurfaceSpec("wall_front", plane(2, box.zmin, tol)),
            SurfaceSpec("wall_back", plane(2, box.zmax, tol)),
        ]
    if fitted:
        specs.append(SurfaceSpec("free_surface", plane(1, box.ymax, tol)))
    else:
        specs.append(SurfaceSpec("wall_top", plane(1, box.ymax, tol)))
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
    ghost_layers: int = 0,
    obstacles: list[Box] | None = None,
    include_top_wall_bc: bool = False,
    level_set_outflow_faces: list[str] | None = None,
    wall_effective_directions: dict[str, str] | None = None,
    active_domain: str | None = None,
    enable_cut_cell_stabilization: bool = True,
    use_cut_metadata_scale: bool = False,
    use_blockschur_solver: bool = False,
    use_direct_fluid_solver: bool = False,
    include_pressure_constraints: bool = True,
    hydrostatic_pressure_field_name: str | None = None,
    momentum_source_file_path: str | None = None,
    rotating_frame_angular_velocity_file_path: str | None = None,
    advection_velocity_extension_method: str | None = None,
    enable_adaptive_time_loop: bool = False,
    adaptive_time_loop_min_dt: str = TEST05_ADAPTIVE_TIME_LOOP_MIN_DT,
    adaptive_time_loop_max_dt: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_DT,
    adaptive_time_loop_max_retries: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_RETRIES,
    adaptive_time_loop_decrease_factor: str = TEST05_ADAPTIVE_TIME_LOOP_DECREASE_FACTOR,
    adaptive_time_loop_increase_factor: str = TEST05_ADAPTIVE_TIME_LOOP_INCREASE_FACTOR,
    adaptive_time_loop_target_newton_iterations: str = TEST05_ADAPTIVE_TIME_LOOP_TARGET_NEWTON_ITERATIONS,
    adaptive_time_loop_max_steps_multiplier: str = TEST05_ADAPTIVE_TIME_LOOP_MAX_STEPS_MULTIPLIER,
    newton_line_search_fail_on_no_reduction: bool = False,
    newton_line_search_max_iterations: str = "10",
    gauge_pressure: float | Callable[[np.ndarray], float] = 0.0,
    record_gauge_metadata: bool = False,
    pressure_gauge_verification: Callable[[dict], dict] | None = None,
    allow_overlapping_face_files: bool = False,
    level_set_maintenance: bool = False,
    interface_preserving_redistancing: bool = False,
    free_slip_walls: bool = False,
    readme: str | None = None,
) -> None:
    obstacles = obstacles or []
    if free_slip_walls and (wall_effective_directions or allow_overlapping_face_files):
        raise ValueError(
            "free-slip walls (decision D58) derive the Effective_direction of every disjoint wall face "
            "file; the legacy overlapping face files need --wall-condition no-slip")
    if grid_factory is not None:
        grid = grid_factory()
        grid = remove_cells_inside(grid, obstacles)
    else:
        max_volume = h**3 / 6.0
        grid = build_tetgen_grid(domain, max_volume=max_volume, obstacles=obstacles)
        grid = run_mmg(grid, hmin=0.35 * h, hmax=1.25 * h, hausd=0.15 * h)
        grid = remove_cells_inside(grid, obstacles)
    add_solution_arrays(grid, phi=phi, pressure=pressure, fitted=fitted)

    prepare_case_dir(case_dir)
    mesh_dir = case_dir / "mesh" / mesh_subdir
    surface_dir = mesh_dir / "mesh-surfaces"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = mesh_dir / "mesh-complete.mesh.vtu"
    grid.save(mesh_path, binary=False)

    spatial_dimension = grid_spatial_dimension(grid)
    if allow_overlapping_face_files:
        # Legacy decks: centroid within 0.35 h of a plane, so the files
        # overlap at the box edges (D35).
        tol = max(0.35 * h, 1.0e-8)
        specs = surface_specs_for_box(domain, tol, fitted=fitted)
        if obstacles:
            specs.append(SurfaceSpec("obstacle", obstacle_predicate(obstacles[0], tol)))
    else:
        # Every face vertex on the plane: one file per boundary face.
        extent = max(domain.xmax - domain.xmin, domain.ymax - domain.ymin, domain.zmax - domain.zmin)
        tol = 1.0e-9 * extent
        specs = surface_specs_for_box(
            domain, tol, fitted=fitted, exact_planes=True, spatial_dimension=spatial_dimension
        )
        if obstacles and free_slip_walls:
            specs += obstacle_plane_specs(obstacles[0], tol)
        elif obstacles:
            specs.append(SurfaceSpec("obstacle", obstacle_face_predicate(obstacles[0], tol)))
    if free_slip_walls:
        # Free-slip (D58): every wall face file fixes the velocity component
        # along its normal only.
        wall_effective_directions = free_slip_effective_directions(
            [spec.name for spec in specs if spec.name != "free_surface"], spatial_dimension)
    # Face files this generator writes but this deck no longer has (the single
    # obstacle file of a no-slip deck, or the per-plane files of a free-slip one).
    current = {spec.name for spec in specs}
    if surface_dir.is_dir():
        for stale in surface_dir.glob("*.vtp"):
            if stale.stem not in current and (stale.stem.startswith(("wall_", "obstacle"))
                                              or stale.stem == "free_surface"):
                stale.unlink()
    write_surfaces(
        grid,
        specs,
        surface_dir,
        allow_overlapping_face_files=allow_overlapping_face_files,
        require_complete_coverage=not allow_overlapping_face_files,
    )

    gauge_metadata = write_pressure_gauge(case_dir, grid, gauge_point, gauge_pressure)
    if record_gauge_metadata:
        metadata = dict(metadata)
        metadata["pressure_gauge"] = gauge_metadata
        if pressure_gauge_verification is not None:
            metadata["pressure_gauge_verification"] = pressure_gauge_verification(gauge_metadata)
    write_metadata(case_dir, metadata)
    if readme is not None:
        (case_dir / "README.md").write_text(readme)
    write_solver_xml(
        case_dir,
        spatial_dimension=spatial_dimension,
        level_set_maintenance=level_set_maintenance,
        interface_preserving_redistancing=interface_preserving_redistancing,
        mesh_path=f"mesh/{mesh_subdir}/mesh-complete.mesh.vtu",
        faces=[spec.name for spec in specs],
        fitted=fitted,
        fill_height=fill_height,
        time_step=time_step,
        time_steps=time_steps,
        ghost_layers=ghost_layers,
        include_top_wall_bc=include_top_wall_bc,
        level_set_outflow_faces=level_set_outflow_faces,
        wall_effective_directions=wall_effective_directions,
        include_obstacle_bc=bool(obstacles),
        active_domain=active_domain,
        enable_cut_cell_stabilization=enable_cut_cell_stabilization,
        use_cut_metadata_scale=use_cut_metadata_scale,
        use_blockschur_solver=use_blockschur_solver,
        use_direct_fluid_solver=use_direct_fluid_solver,
        include_pressure_constraints=include_pressure_constraints,
        hydrostatic_pressure_field_name=hydrostatic_pressure_field_name,
        momentum_source_file_path=momentum_source_file_path,
        rotating_frame_angular_velocity_file_path=rotating_frame_angular_velocity_file_path,
        advection_velocity_extension_method=advection_velocity_extension_method,
        enable_adaptive_time_loop=enable_adaptive_time_loop,
        adaptive_time_loop_min_dt=adaptive_time_loop_min_dt,
        adaptive_time_loop_max_dt=adaptive_time_loop_max_dt,
        adaptive_time_loop_max_retries=adaptive_time_loop_max_retries,
        adaptive_time_loop_decrease_factor=adaptive_time_loop_decrease_factor,
        adaptive_time_loop_increase_factor=adaptive_time_loop_increase_factor,
        adaptive_time_loop_target_newton_iterations=adaptive_time_loop_target_newton_iterations,
        adaptive_time_loop_max_steps_multiplier=adaptive_time_loop_max_steps_multiplier,
        newton_line_search_fail_on_no_reduction=newton_line_search_fail_on_no_reduction,
        newton_line_search_max_iterations=newton_line_search_max_iterations,
        allow_overlapping_face_files=allow_overlapping_face_files,
    )

    try:
        display_path = case_dir.relative_to(ROOT)
    except ValueError:
        display_path = case_dir
    cell_name = "tetrahedra" if spatial_dimension == 3 else "triangles"
    print(f"{display_path}: {grid.n_points} points, {grid.n_cells} {cell_name}")


def sloshing_metadata(
    name: str,
    fitted: bool,
    fill_height: float,
    *,
    mesh_tools: list[str] | None = None,
) -> dict:
    # SPHERIC Test10 Figure 1 gives Sensor 1 at the low-fill lateral-impact
    # wall location: left wall, 93 mm above the bottom, centered through the
    # uploaded 1x breadth.
    sensor1 = {
        "name": "Sensor1",
        "case": "lateral_water_1x",
        "coordinates": [0.0, fill_height, 0.031],
        "source": "SPHERIC Test10 Figure 1",
        "role": "literature_pressure_history_sensor",
    }
    return {
        "benchmark": name,
        "representation": "fitted_ale" if fitted else "unfitted_level_set",
        "source_urls": ["https://www.spheric-sph.org/tests/test-10"],
        "mesh_tools": mesh_tools or ["PyVista", "TetGen", "MMG"],
        "dimensions_m": {
            "tank_length": 0.900,
            "tank_breadth_1x": 0.062,
            "tank_height": 0.508,
            "initial_fill_height": fill_height,
        },
        "notes": [
            "The mesh encodes the 1x rectangular tank and still-water fill level.",
            "The published roll-angle history and pressure records are external benchmark data.",
            (
                "The pressure_gauge entry is an interior pressure anchor; pressure history comparisons use pressure_sensor."
                if fitted
                else "The pressure_gauge entry is an initial-condition diagnostic sample, not an algebraic pressure constraint; pressure history comparisons use pressure_sensor."
            ),
        ],
        "rotation_axis": {
            "point": [0.45, 0.0, 0.031],
            "direction": [0.0, 0.0, 1.0],
            "source": "SPHERIC Test10 Figure 1: rotation axis at the center of the bottom line",
        },
        "pressure_sensor": sensor1,
    }


def sloshing_pressure_gauge_verification(gauge_metadata: dict) -> dict:
    current_pressure = float(gauge_metadata["expected_initial_hydrostatic_pressure"])
    return {
        "current_prescribed_pressure_matches_initial_hydrostatic": True,
        "initial_pressure_error_after_constraint": 0.0,
        "previous_surface_target_pressure": 0.0,
        "previous_surface_target_pressure_error": -current_pressure,
    }


def field_initialized_pressure_diagnostic(gauge_metadata: dict) -> dict:
    """Metadata for a sampled initial pressure that is not an algebraic pin."""
    return {
        "pressure_constraint_enabled": False,
        "pressure_initialization_source": "mesh_vertex_field",
        "diagnostic_point_initial_pressure_matches_mesh_field": True,
        "diagnostic_point_initial_pressure_error": 0.0,
        "diagnostic_point_pressure": float(
            gauge_metadata["expected_initial_hydrostatic_pressure"]
        ),
        "diagnostic_role": "initial_condition_sample_not_pressure_constraint",
    }


def write_legacy_fitted_spheric_test10(
    water: Box,
    lateral_fill: float,
    x_coords: np.ndarray,
    fitted_y_coords: np.ndarray,
    z_coords: np.ndarray,
    structured_max_spacing: float,
) -> None:
    """The legacy fitted_ale Test 10 deck (overlapping face files, D35)."""

    def fitted_grid_factory() -> pv.UnstructuredGrid:
        return structured_tet_grid(
            x_coords,
            fitted_y_coords,
            z_coords,
            mirror_z_midplane=True,
        )

    write_case(
        allow_overlapping_face_files=True,
        case_dir=ROOT / "fitted_ale" / "spheric_test10_lateral_water_1x",
        mesh_subdir="water",
        domain=water,
        phi=None,
        fill_height=lateral_fill,
        grid_factory=fitted_grid_factory,
        gauge_point=(0.45, 0.5 * lateral_fill, 0.031),
        gauge_pressure=lambda point: hydrostatic_pressure_at_point(point, lateral_fill),
        record_gauge_metadata=True,
        pressure_gauge_verification=sloshing_pressure_gauge_verification,
        metadata=sloshing_metadata(
            "SPHERIC Test 10 lateral water 1x",
            True,
            lateral_fill,
            mesh_tools=["PyVista", "structured_tet_grid"],
        ),
        h=structured_max_spacing,
        fitted=True,
        time_step=0.001,
        time_steps=40,
    )


TEST10_CRITERIA = {
    "status": "proposal (D1): the working criteria of the SPHERIC campaign until the user sets tolerances",
    "completion_time_s": 1.5,
    "completion_justification": (
        "past the June stop at 0.89-0.90 s (line-search trial cut fractions near 1e-8) and the "
        "first Sensor1 impact peak at 0.953 s"
    ),
    "sensor1_first_peak_time_s": 0.953,
    "sensor1_first_peak_time_tolerance_s": 0.05,
    "sensor1_first_peak_time_justification": (
        "about 5% of the time to the first impact; the fitted 2D MeshNitsche deck peaks 0.028 s "
        "early (0.925 s), so the unfitted deck is held to the same order"
    ),
    "sensor1_first_peak_mbar": 3.86,
    "sensor1_first_peak_relative_tolerance": 0.3,
    "sensor1_first_peak_justification": (
        "the fitted 2D deck is 12% low (3.38 mbar); 30% leaves room for this coarse 3D mesh "
        "(h = 45 mm) and still rejects the June accepted pressures of about 2-3 times the reference"
    ),
    "reported": "Sensor1 RMS difference over 0-1.4 s; the 7.3 s peak (tracker M6); volume drift",
}


def level_set_protocol_note(redistancing: bool) -> str:
    """The level-set sentence of the SPHERIC decks' benchmark.json notes."""
    maintenance = ("kinematic reconciliation (D14), sign-definite patch bounds (D21) and triggered "
                   "interface-preserving redistancing (D39)" if redistancing else
                   "kinematic reconciliation (D14) and sign-definite patch bounds (D21)")
    return f"Level set: harmonic PDE velocity extension, monolithic (D55), with {maintenance}."


def generate_spheric_test10(
    *,
    allow_overlapping_face_files: bool = False,
    include_legacy_fitted: bool = False,
    redistancing: bool = INTERFACE_PRESERVING_REDISTANCING,
    free_slip_walls: bool = SPHERIC_FREE_SLIP_WALLS,
    ghost_layers: int = SPHERIC_3D_GHOST_LAYERS,
) -> None:
    """The unfitted Test 10 deck; with include_legacy_fitted also the legacy
    fitted_ale deck, which keeps its overlapping face files (the fitted
    MeshNitsche decks of generate_spheric_test10_fitted_decks.py replace it)."""
    tank = Box(0.0, 0.900, 0.0, 0.508, 0.0, 0.062)
    lateral_fill = 0.093
    water = Box(tank.xmin, tank.xmax, tank.ymin, lateral_fill, tank.zmin, tank.zmax)
    structured_max_spacing = 0.045
    interface_centering_gap = 0.04
    x_coords = segmented_coordinates(
        (tank.xmin, 0.5 * (tank.xmin + tank.xmax), tank.xmax),
        max_spacing=structured_max_spacing,
    )
    z_coords = segmented_coordinates(
        (tank.zmin, 0.5 * (tank.zmin + tank.zmax), tank.zmax),
        max_spacing=0.031,
    )
    fitted_y_coords = segmented_coordinates(
        (water.ymin, 0.5 * lateral_fill, water.ymax),
        max_spacing=structured_max_spacing,
    )
    unfitted_y_coords = aligned_and_centered_coordinates(
        tank.ymin,
        tank.ymax,
        centered_cut_values=(lateral_fill,),
        max_spacing=structured_max_spacing,
        centering_gap=interface_centering_gap,
    )

    def unfitted_grid_factory() -> pv.UnstructuredGrid:
        return structured_tet_grid(
            x_coords,
            unfitted_y_coords,
            z_coords,
            mirror_z_midplane=True,
        )

    if include_legacy_fitted:
        if not allow_overlapping_face_files:
            raise ValueError(
                "the legacy fitted Test 10 deck keeps its overlapping face files: "
                "pass --allow-overlapping-face-files with --legacy-fitted-test10"
            )
        write_legacy_fitted_spheric_test10(
            water, lateral_fill, x_coords, fitted_y_coords, z_coords, structured_max_spacing
        )
    unfitted_metadata = sloshing_metadata(
        "SPHERIC Test 10 lateral water 1x",
        False,
        lateral_fill,
        mesh_tools=["PyVista", "structured_tet_grid"],
    )
    unfitted_metadata["notes"] = unfitted_metadata["notes"] + [
        "The face files are disjoint (every face vertex on the wall).  The June deck's face files "
        "overlapped at the tank edges between no-slip walls, so the relabeling changes no condition.",
        level_set_protocol_note(redistancing),
        "The roll forcing (body force and rotating-frame angular velocity tables under bc/) is derived "
        "from the published motion record by generate_spheric_test10_roll_body_force.py; it is not "
        "committed.",
    ] + ([
        "Walls: free-slip on all six walls, the lid included; only the normal velocity is fixed "
        "(decision D58).  With no-slip walls the end-wall contact lines stayed pinned at the fill "
        "height for the whole 7.5 s run (2026-10-26), with Sensor 1 on the pinned line.",
    ] if free_slip_walls else [])
    unfitted_metadata["qualification_criteria"] = TEST10_CRITERIA

    write_case(
        allow_overlapping_face_files=allow_overlapping_face_files,
        case_dir=ROOT / "unfitted_level_set" / "spheric_test10_lateral_water_1x",
        mesh_subdir="background",
        domain=tank,
        phi=lambda points: points[:, 1] - lateral_fill,
        # Retained cut-cell pressure DOFs include the first dry vertex layer.
        # Use the signed hydrostatic continuation there so the continuous P1
        # trace interpolates p=0 at the actual phi=0 surface. Far-dry values
        # are ignored by active-support initialization.
        pressure=lambda points: WATER_DENSITY
        * GRAVITY
        * (lateral_fill - points[:, 1]),
        fill_height=lateral_fill,
        grid_factory=unfitted_grid_factory,
        gauge_point=(0.45, 0.5 * lateral_fill, 0.031),
        gauge_pressure=lambda point: hydrostatic_pressure_at_point(point, lateral_fill),
        record_gauge_metadata=True,
        pressure_gauge_verification=field_initialized_pressure_diagnostic,
        metadata=unfitted_metadata,
        h=structured_max_spacing,
        fitted=False,
        time_step=0.001,
        time_steps=40,
        level_set_maintenance=True,
        interface_preserving_redistancing=redistancing,
        free_slip_walls=free_slip_walls,
        ghost_layers=ghost_layers,
        include_top_wall_bc=True,
        active_domain="LevelSetNegative",
        use_direct_fluid_solver=True,
        include_pressure_constraints=False,
        hydrostatic_pressure_field_name="Pressure",
        advection_velocity_extension_method="pde_harmonic",
        momentum_source_file_path="bc/test10_lateral_water_1x_roll_body_force.dat",
        rotating_frame_angular_velocity_file_path="bc/test10_lateral_water_1x_roll_angular_velocity.dat",
    )


def test05_pressure_gauge_verification(gauge_metadata: dict) -> dict:
    current_pressure = float(gauge_metadata["expected_initial_hydrostatic_pressure"])
    previous_pressure = TEST05_PREVIOUS_INVALID_D18_GAUGE["full_volume_hydrostatic_pressure"]
    previous_range = TEST05_PREVIOUS_INVALID_D18_GAUGE["hydrostatic_error_range"]
    return {
        "pressure_constraint_enabled": False,
        "pressure_initialization_source": "mesh_vertex_field",
        "diagnostic_point_initial_pressure_matches_mesh_field": True,
        "diagnostic_point_initial_pressure_error": 0.0,
        "previous_invalid_d18_node_id": TEST05_PREVIOUS_INVALID_D18_GAUGE["node_id"],
        "previous_invalid_d18_initial_phi": TEST05_PREVIOUS_INVALID_D18_GAUGE["initial_phi"],
        "previous_invalid_d18_full_volume_hydrostatic_pressure": previous_pressure,
        "previous_invalid_d18_hydrostatic_error_range": previous_range,
        "diagnostic_pressure_matches_previous_invalid_offset": False,
        "diagnostic_pressure_matches_previous_invalid_error_range": (
            previous_range[0] <= current_pressure <= previous_range[1]
        ),
        "diagnostic_pressure_minus_previous_invalid_pressure": current_pressure - previous_pressure,
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


TEST05_MESH_SPACING = 0.006
TEST05_TIME_STEPS = 600  # 600 x 5e-4 s = 0.3 s (tracker M1); runs to the last profile override it
TEST05_FALSE_WETTING_HEIGHT = 0.16
# The June three-dimensional decks (a thin extrusion, replaced by D51 item 2).
TEST05_JUNE_EXTRUSION = {
    "extrusion_breadth_m": 0.03,
    "cells_across": 2,
    "tetrahedra": {18: 13608, 38: 14256},
    "spacing_m": {"x": 0.022, "y": [0.005, 0.010], "z": 0.015},
    "stop": "step 193 (t = 0.0965 s) on the small-cut aggregation guard (root path 9 > 8)",
    # The overlapping face files shared 112 faces: 108 front/back faces next to
    # the bottom (labeled front/back, their own wall) and 4 left/right faces
    # next to the bottom (labeled bottom, no-slip as their own wall).
    "overlapping_faces": 112,
    "faces_with_a_different_condition": 0,
}


def test05_readme(wet_depth_mm: int, n_points: int, n_triangles: int, nx: int, ny: int,
                  redistancing: bool = INTERFACE_PRESERVING_REDISTANCING,
                  free_slip_walls: bool = SPHERIC_FREE_SLIP_WALLS) -> str:
    walls = ("""Walls left, right and bottom are free-slip (only the normal velocity is fixed, decision
  D58); the top is open (natural traction, level-set outflow).  Zero surface tension: a
  transport and impermeability benchmark, not a wetting one.""" if free_slip_walls else
             """Walls left, right and bottom are no-slip; the top is open (natural traction, level-set
  outflow).  Zero surface tension: a transport and impermeability benchmark, not a
  wetting one.""")
    maintenance = ("""kinematic
  reconciliation (D14), sign-definite patch bounds (D21) and triggered interface-preserving
  redistancing (D39), as the M1-M4 protocols.""" if redistancing else """kinematic
  reconciliation (D14) and sign-definite patch bounds (D21), as the M1-M4 protocols.""")
    return f"""# SPHERIC Test 05 wet-bed dam break, d = {wet_depth_mm} mm (unfitted level set, 2D)

Written by `../../generate_validation_meshes.py --case spheric_test05`.  Status: regenerated
2026-10-26 for the SPHERIC campaign (tracker M1, M6); the profiles of this deck have not
been compared with experiment yet.

## Setup

- The published two-dimensional flume section: 1.2 m long, 0.18 m high; a 0.15 m water
  column behind the gate at x = 0.38 m and a wet bed of {wet_depth_mm} mm downstream.
- The gate is removed instantaneously at t = 0 (the experiment lifts it at 1.5 m/s).
- {walls}
- Level set: harmonic PDE velocity extension with monolithic coupling (D55); {maintenance}
- Mesh: {nx} x {ny} cells of near-isotropic size h = {TEST05_MESH_SPACING * 1000:.0f} mm (the
  June deck's finest spacing), each split into two Triangle3 with the diagonal alternating
  with the cell parity; {n_points} points, {n_triangles} triangles.  The gate, the wet-bed
  height and the column height lie midway between mesh lines.  The four face files are
  disjoint (every face vertex on the wall).

## The June deck it replaces (decision D51 item 2)

The June deck was a three-dimensional thin extrusion of this section: 30 mm across with
2 cells (h_z = 15 mm), h_x = 22 mm, h_y = 5-10 mm, {TEST05_JUNE_EXTRUSION['tetrahedra'][wet_depth_mm]}
tetrahedra, with free-slip front and back planes.  On the current protocol it stopped at
step 193 (t = 0.0965 s) on the small-cut aggregation guard: at the surge front the liquid
formed wedges on the two free-slip planes, and with only 2 cells across no full cell lay
within 8 face steps of a cut cell.  Its face files also overlapped at the tank edges (the
0.35 h centroid selection, kept by the `Allow_overlapping_face_files` opt-out of D35).
A face keeps the label of the last face file listed.  Of the 112 shared faces, 108 were
front/back faces next to the bottom, which kept their own free-slip label, and 4 were
left/right faces next to the bottom, which took the bottom label, no-slip like their own
wall; checked face by face on the committed files, no face had a different condition
(the tracker's M5 item expected bottom faces with the free-slip label; there were none).
The 2D section has no spanwise direction and its face files are disjoint.

## Criteria (tracker M1)

- Free-surface profile RMSE against the experimental profile at most 0.02 m at 0.156,
  0.219 and 0.281 s (`compare_test05_profiles.py`).  June, on the extruded deck with
  cadence-1 volume correction: 0.0184 m (d18) and 0.0158 m (d38) at 0.156 s.
- No false wetting: no vertex with phi < 0 above y = {TEST05_FALSE_WETTING_HEIGHT} m (June d18 wetted the
  lid corners after 0.22 s).
- The later profiles (to {TEST05_REFERENCE_PROFILE_TIMES[wet_depth_mm][-1]} s, tracker M6) are reported.
"""


def generate_spheric_test05(*, allow_overlapping_face_files: bool = False,
                            redistancing: bool = INTERFACE_PRESERVING_REDISTANCING,
                            free_slip_walls: bool = SPHERIC_FREE_SLIP_WALLS) -> None:
    if allow_overlapping_face_files:
        raise ValueError(
            "the SPHERIC 05 decks are two-dimensional sections with disjoint face files "
            "(decision D51 item 2); the legacy extruded decks are not regenerated"
        )
    domain = Box(0.0, 1.20, 0.0, 0.18, 0.0, 0.0)
    gate_x = 0.38
    dam_height = 0.15
    source_urls = ["https://www.spheric-sph.org/tests/test-05"]

    for wet_depth_mm, wet_depth in ((18, 0.018), (38, 0.038)):
        x_coords = centered_cut_segmented_coordinates(
            domain.xmin,
            domain.xmax,
            (gate_x,),
            max_spacing=TEST05_MESH_SPACING,
        )
        y_coords = test05_vertical_coordinates(
            domain,
            wet_depth,
            dam_height,
            fine_spacing=TEST05_MESH_SPACING,
            coarse_spacing=TEST05_MESH_SPACING,
        )

        def grid_factory(
            x_coords: np.ndarray = x_coords,
            y_coords: np.ndarray = y_coords,
        ) -> pv.UnstructuredGrid:
            return structured_triangle_grid(x_coords, y_coords)

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

        nx = len(x_coords) - 1
        ny = len(y_coords) - 1
        write_case(
            allow_overlapping_face_files=False,
            case_dir=ROOT / "unfitted_level_set" / f"spheric_test05_wet_bed_d{wet_depth_mm}",
            mesh_subdir="background",
            domain=domain,
            phi=phi,
            pressure=pressure,
            grid_factory=grid_factory,
            fill_height=dam_height,
            gauge_point=(0.10, 0.075, 0.0),
            gauge_pressure=lambda point, wet_depth=wet_depth: float(
                pressure(point.reshape(1, 3))[0]
            ),
            record_gauge_metadata=True,
            pressure_gauge_verification=test05_pressure_gauge_verification,
            metadata={
                "benchmark": f"SPHERIC Test 05 wet-bed dam break d={wet_depth_mm} mm",
                "representation": "unfitted_level_set",
                "spatial_dimension": 2,
                "source_urls": source_urls,
                "reference_profiles": test05_reference_profiles(wet_depth_mm),
                "mesh_tools": ["PyVista", "structured triangle grid, diagonal alternating with the cell parity"],
                "mesh": {
                    "cells_x": nx,
                    "cells_y": ny,
                    "triangles": 2 * nx * ny,
                    "spacing_m": TEST05_MESH_SPACING,
                },
                "dimensions_m": {
                    "initial_column_height": dam_height,
                    "wet_bed_depth": wet_depth,
                    "gate_x": gate_x,
                    "profile_window_x_min": 0.38,
                    "profile_window_x_max": 1.04,
                    "computational_length": domain.xmax,
                    "computational_height": domain.ymax,
                },
                "qualification_criteria": {
                    "source": "tracker M1 (SPHERIC 05 D18/D38)",
                    "profile_rmse_max_m": 0.02,
                    "profile_rmse_times_s": [0.156, 0.219, 0.281],
                    "profile_rmse_justification": (
                        "no regression from the June RMSE of about 0.02 m "
                        "(0.0184 m d18, 0.0158 m d38 at 0.156 s)"
                    ),
                    "false_wetting": (
                        f"no vertex with phi < 0 above y = {TEST05_FALSE_WETTING_HEIGHT} m"
                    ),
                    "reported_profile_times_s": TEST05_REFERENCE_PROFILE_TIMES[wet_depth_mm],
                },
                "replaces": {
                    "decision": "D51 item 2",
                    "june_three_dimensional_extrusion": {
                        "extrusion_breadth_m": TEST05_JUNE_EXTRUSION["extrusion_breadth_m"],
                        "cells_across": TEST05_JUNE_EXTRUSION["cells_across"],
                        "tetrahedra": TEST05_JUNE_EXTRUSION["tetrahedra"][wet_depth_mm],
                        "spacing_m": TEST05_JUNE_EXTRUSION["spacing_m"],
                        "stop_on_current_protocol": TEST05_JUNE_EXTRUSION["stop"],
                        "overlapping_faces": TEST05_JUNE_EXTRUSION["overlapping_faces"],
                        "faces_with_a_different_condition": TEST05_JUNE_EXTRUSION[
                            "faces_with_a_different_condition"
                        ],
                    },
                },
                "notes": [
                    "The published two-dimensional setup, meshed as a two-dimensional Triangle3 section (decision D51 item 2).",
                    "The face files are disjoint: a boundary edge belongs to the wall all of its vertices lie on.",
                    ("The left, right and bottom walls are free-slip: only the normal velocity is fixed (decision D58)" if free_slip_walls else "The left, right and bottom walls are no-slip") + "; zero surface tension makes this a dam-break transport and impermeability benchmark, not a moving-contact-line wetting validation.",
                    "The wall_top face is an open natural-traction boundary with a level-set outflow condition and no velocity Dirichlet condition, matching the open top of the published flume.",
                    "The deck approximates the gate as removed instantaneously; the published Test 05 specification raises it at 1.5 m/s.",
                    "The initial level-set field is the union of the wet bed and the retained water column; negative values denote the water side.",
                    "Pressure uses the local column-height hydrostatic law continued with sign onto dry cut-support vertices; those coefficients are interpolation data, not gas pressure.",
                ],
            },
            h=TEST05_MESH_SPACING,
            fitted=False,
            time_step=0.0005,
            time_steps=TEST05_TIME_STEPS,
            include_top_wall_bc=False,
            level_set_outflow_faces=["wall_top"],
            active_domain="LevelSetNegative",
            enable_cut_cell_stabilization=True,
            use_cut_metadata_scale=False,
            use_direct_fluid_solver=True,
            include_pressure_constraints=False,
            hydrostatic_pressure_field_name="Pressure",
            advection_velocity_extension_method="pde_harmonic",
            enable_adaptive_time_loop=True,
            newton_line_search_fail_on_no_reduction=True,
            newton_line_search_max_iterations="6",
            level_set_maintenance=True,
            interface_preserving_redistancing=redistancing,
            free_slip_walls=free_slip_walls,
            readme=test05_readme(
                wet_depth_mm,
                (nx + 1) * (ny + 1),
                2 * nx * ny,
                nx,
                ny,
                redistancing,
                free_slip_walls,
            ),
        )


TEST02_COLUMN_HYDROSTATIC_PA = WATER_DENSITY * GRAVITY * TEST02_INITIAL_COLUMN_HEIGHT
TEST02_CRITERIA = {
    "status": "proposal (D1): the working criteria of the SPHERIC campaign until the user sets tolerances",
    "completion_time_s": 1.0,
    "completion_justification": (
        "the front passes the obstacle and reaches the far wall well before 1 s; the June runs "
        "showed their pressure spikes in this window"
    ),
    "maximum_sensor_pressure_pa": 50000.0,
    "maximum_sensor_pressure_justification": (
        "about 4.5 times the largest measured obstacle pressure (P1, 11.2 kPa at 0.415 s) and "
        "9 times the column's hydrostatic pressure rho g H0 = 5.4 kPa; the June spikes reached "
        "MPa, so 50 kPa separates a spike from a physical impact pressure"
    ),
    "minimum_front_speed_ratio": 0.8,
    "front_speed_ratio_definition": (
        "measured over the 1.0 m from the gate to the H2 probe (x = 0.992 m): the experimental "
        "arrival time (H2 above 1 cm at 0.358 s) divided by the simulated one"
    ),
    "minimum_front_speed_ratio_justification": (
        "the June front ran at 0.43 of the reference speed; 0.8 tolerates the coarse mesh "
        "(h = 0.08-0.10 m) while rejecting a front held back by the cut-cell treatment"
    ),
    "reported": "H2 and H4 heights and P1, P3, P5, P7 pressures against experiment (RMS); volume drift",
}


def generate_spheric_test02(*, allow_overlapping_face_files: bool = False,
                            redistancing: bool = INTERFACE_PRESERVING_REDISTANCING,
                            free_slip_walls: bool = SPHERIC_FREE_SLIP_WALLS,
                            ghost_layers: int = SPHERIC_3D_GHOST_LAYERS) -> None:
    tank = Box(0.0, TEST02_TANK_LENGTH, 0.0, TEST02_TANK_HEIGHT, 0.0, TEST02_TANK_WIDTH)
    obstacle = Box(
        TEST02_OBSTACLE_X_MIN,
        TEST02_OBSTACLE_X_MAX,
        0.0,
        TEST02_OBSTACLE_HEIGHT,
        TEST02_OBSTACLE_Z_MIN,
        TEST02_OBSTACLE_Z_MAX,
    )
    structured_max_spacing = 0.10
    interface_centering_gap = 0.08
    x_coords = aligned_and_centered_coordinates(
        tank.xmin,
        tank.xmax,
        aligned_values=(TEST02_OBSTACLE_X_MIN, TEST02_OBSTACLE_X_MAX, *TEST02_HEIGHT_PROBES.values()),
        centered_cut_values=(TEST02_GATE_X,),
        max_spacing=structured_max_spacing,
        centering_gap=interface_centering_gap,
    )
    y_coords = aligned_and_centered_coordinates(
        tank.ymin,
        tank.ymax,
        aligned_values=(TEST02_OBSTACLE_HEIGHT,),
        centered_cut_values=(TEST02_INITIAL_COLUMN_HEIGHT,),
        max_spacing=structured_max_spacing,
        centering_gap=interface_centering_gap,
    )
    z_coords = segmented_coordinates(
        (tank.zmin, TEST02_OBSTACLE_Z_MIN, 0.5, TEST02_OBSTACLE_Z_MAX, tank.zmax),
        max_spacing=structured_max_spacing,
    )

    def grid_factory() -> pv.UnstructuredGrid:
        return structured_tet_grid(x_coords, y_coords, z_coords, mirror_z_midplane=True)

    write_case(
        allow_overlapping_face_files=allow_overlapping_face_files,
        case_dir=ROOT / "unfitted_level_set" / "spheric_test02_dambreak_obstacle",
        mesh_subdir="background",
        domain=tank,
        phi=lambda points: dam_break_dry_bed_column_phi(
            points, gate_x=TEST02_GATE_X, dam_height=TEST02_INITIAL_COLUMN_HEIGHT
        ),
        pressure=lambda points: dam_break_dry_bed_column_pressure(
            points,
            gate_x=TEST02_GATE_X,
            dam_height=TEST02_INITIAL_COLUMN_HEIGHT,
        ),
        fill_height=TEST02_INITIAL_COLUMN_HEIGHT,
        gauge_point=(2.632, 0.10, 0.50),
        gauge_pressure=lambda point: hydrostatic_pressure_at_point(
            point, TEST02_INITIAL_COLUMN_HEIGHT
        ),
        record_gauge_metadata=True,
        pressure_gauge_verification=field_initialized_pressure_diagnostic,
        metadata={
            "benchmark": "SPHERIC Test 02 three-dimensional dam break with obstacle",
            "representation": "unfitted_level_set",
            "source_urls": [
                "https://www.spheric-sph.org/tests/test-02",
                "https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.00346/full",
            ],
            "mesh_tools": ["PyVista", "structured_tet_grid"],
            "dimensions_m": {
                "tank_length": TEST02_TANK_LENGTH,
                "tank_width": TEST02_TANK_WIDTH,
                "tank_height": TEST02_TANK_HEIGHT,
                "initial_column_length": TEST02_INITIAL_COLUMN_LENGTH,
                "initial_column_x_min": TEST02_GATE_X,
                "initial_column_x_max": TEST02_TANK_LENGTH,
                "initial_column_height": TEST02_INITIAL_COLUMN_HEIGHT,
                "obstacle_flow_direction_length": TEST02_OBSTACLE_LENGTH,
                "obstacle_lateral_width": TEST02_OBSTACLE_WIDTH,
                "obstacle_height": TEST02_OBSTACLE_HEIGHT,
                "obstacle_x_min": TEST02_OBSTACLE_X_MIN,
                "obstacle_x_max": TEST02_OBSTACLE_X_MAX,
                "obstacle_center_x": 0.5
                * (TEST02_OBSTACLE_X_MIN + TEST02_OBSTACLE_X_MAX),
                "obstacle_z_min": TEST02_OBSTACLE_Z_MIN,
                "obstacle_z_max": TEST02_OBSTACLE_Z_MAX,
                "obstacle_center_z": 0.50,
                "height_probe_x_positions": TEST02_HEIGHT_PROBES,
                "structured_max_spacing": structured_max_spacing,
                "interface_centering_gap": interface_centering_gap,
                "previous_incorrect_initial_column_length": 0.58,
                "previous_incorrect_obstacle_flow_direction_length": 0.40,
                "previous_incorrect_obstacle_lateral_width": 0.16,
            },
            "notes": [
                "The geometry follows the official SPHERIC Test02/Kleefsman setup: water initially occupies x >= 1.992 m and height <= 0.55 m.",
                "The structured mesh aligns tank and obstacle geometry while centering the top and gate free-surface cuts between mesh planes.",
                ("The obstacle is an internal free-slip boundary of the background mesh, written as one face file per face plane (obstacle_xmin, obstacle_xmax, obstacle_zmin, obstacle_zmax, obstacle_top) so that each fixes the velocity component along its normal (decision D58)." if free_slip_walls else "The obstacle is represented as an internal no-slip boundary in the background mesh."),
                "The initial level-set field is negative inside the retained water column and remains negative on contacted tank walls.",
                "The retained-column hydrostatic pressure is continued through dry cut-support vertices so the top trace is zero and the newly exposed vertical dam face retains its liquid-side initial trace.",
                "The zero level set is limited to the top and gate-facing air-water interfaces; a closed-box signed distance incorrectly creates free-surface cuts on wet walls.",
                "Earlier repository metadata used a 0.58 m retained column and swapped obstacle footprint; those settings are not the SPHERIC Test02 benchmark geometry.",
                level_set_protocol_note(redistancing),
                "The face files are disjoint (every face vertex on the wall or obstacle).  The June deck's files overlapped at the tank edges, and a face keeps the label of the last file listed: 94 faces of the top row of the side walls (left 12, right 12, front 35, back 35) carried the open wall_top label (no velocity condition) instead of no-slip; the other relabeled faces (between walls, and 16 obstacle faces next to the floor labeled bottom) kept a no-slip condition.",
            ] + ([
                "Walls: free-slip on the five tank walls and the obstacle; only the normal velocity is fixed (decision D58); the top is open.  With no-slip walls the floor contact line stayed at the gate and the liquid ran out as a one-cell tongue above a dry floor row (2026-10-26).",
            ] if free_slip_walls else []),
            "qualification_criteria": TEST02_CRITERIA,
        },
        grid_factory=grid_factory,
        h=structured_max_spacing,
        fitted=False,
        time_step=0.001,
        time_steps=120,
        obstacles=[obstacle],
        active_domain="LevelSetNegative",
        enable_cut_cell_stabilization=True,
        use_cut_metadata_scale=False,
        use_direct_fluid_solver=True,
        include_pressure_constraints=False,
        hydrostatic_pressure_field_name="Pressure",
        advection_velocity_extension_method="pde_harmonic",
        enable_adaptive_time_loop=True,
        adaptive_time_loop_min_dt="1.0e-7",
        adaptive_time_loop_max_dt="1.0e-3",
        adaptive_time_loop_max_retries="10",
        adaptive_time_loop_target_newton_iterations="6",
        adaptive_time_loop_max_steps_multiplier="256",
        newton_line_search_fail_on_no_reduction=True,
        newton_line_search_max_iterations="6",
        level_set_maintenance=True,
        interface_preserving_redistancing=redistancing,
        free_slip_walls=free_slip_walls,
        ghost_layers=ghost_layers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["all", "spheric_test10", "spheric_test05", "spheric_test02"],
        default="all",
    )
    parser.add_argument(
        "--allow-overlapping-face-files",
        action="store_true",
        help=(
            "legacy: write face files that share boundary faces (the centroid selection with a "
            "0.35 h tolerance does so at the tank edges) and declare "
            "<Allow_overlapping_face_files>true</Allow_overlapping_face_files> in the decks, as the "
            "committed decks do; without it overlapping face files are an error (decision D35)"
        ),
    )
    parser.add_argument(
        "--legacy-fitted-test10",
        action="store_true",
        help=(
            "also write the legacy fitted_ale Test 10 deck (with --allow-overlapping-face-files); "
            "the fitted MeshNitsche decks of generate_spheric_test10_fitted_decks.py replace it"
        ),
    )
    parser.add_argument(
        "--interface-preserving-redistancing",
        choices=["on", "off"],
        default="on" if INTERFACE_PRESERVING_REDISTANCING else "off",
        help=(
            "triggered interface-preserving redistancing in the unfitted SPHERIC decks (decision "
            "D39, the protocol value); off writes the decks of b94af982 byte for byte"
        ),
    )
    parser.add_argument(
        "--wall-condition",
        choices=["free-slip", "no-slip"],
        default="free-slip" if SPHERIC_FREE_SLIP_WALLS else "no-slip",
        help=(
            "velocity condition on the walls of the unfitted SPHERIC decks: free-slip (decision D58, "
            "only the normal velocity fixed, the Test 02 obstacle written per face plane) or no-slip, "
            "which writes the decks of 8f145c4c byte for byte"
        ),
    )
    parser.add_argument(
        "--ghost-layers-3d",
        type=int,
        default=SPHERIC_3D_GHOST_LAYERS,
        help=(
            "<Ghost_layers> of the 3D SPHERIC decks (Test 02 and 10): 1 + S + 8 layers cover the "
            "small-cut aggregation stencil at the root-path guard 8, S the vertex-star depth (7 on "
            "the Test 02 mesh, 9 on mirrored Kuhn meshes); 18 covers both; 0 writes no key (the "
            "solver's default), as the decks of 9802903f"
        ),
    )
    args = parser.parse_args()
    legacy = {"allow_overlapping_face_files": args.allow_overlapping_face_files}
    protocol = {"redistancing": args.interface_preserving_redistancing == "on",
                "free_slip_walls": args.wall_condition == "free-slip"}

    if args.case in {"all", "spheric_test10"}:
        generate_spheric_test10(**legacy, include_legacy_fitted=args.legacy_fitted_test10, **protocol,
                                ghost_layers=args.ghost_layers_3d)
    if args.case in {"all", "spheric_test05"}:
        generate_spheric_test05(**legacy, **protocol)
    if args.case in {"all", "spheric_test02"}:
        generate_spheric_test02(**legacy, **protocol, ghost_layers=args.ghost_layers_3d)


if __name__ == "__main__":
    main()
