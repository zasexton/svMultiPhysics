#!/usr/bin/env python3
"""Generate a long rectangular Hex8 beam mesh for Ustruct tests.

TetGen only creates tetrahedral volume meshes, so this script uses PyVista for
the requested Hex8 volume mesh and TetGen as a closed-surface PLC validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv

try:
    import tetgen
except ModuleNotFoundError:
    tetgen = None


THIS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BeamSpec:
    length: float = 10.0
    width: float = 1.0
    height: float = 1.0
    nx: int = 20
    ny: int = 2
    nz: int = 2


SURFACE_IDS = {
    "left": 1,
    "right": 2,
    "front": 3,
    "back": 4,
    "bottom": 5,
    "top": 6,
}


def node_id(spec: BeamSpec, i: int, j: int, k: int) -> int:
    return (i * (spec.ny + 1) + j) * (spec.nz + 1) + k


def build_points(spec: BeamSpec) -> np.ndarray:
    points = []
    for i in range(spec.nx + 1):
        x = spec.length * float(i) / float(spec.nx)
        for j in range(spec.ny + 1):
            y = spec.width * float(j) / float(spec.ny)
            for k in range(spec.nz + 1):
                z = spec.height * float(k) / float(spec.nz)
                points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


def build_hex_connectivity(spec: BeamSpec) -> np.ndarray:
    hexes = []
    for i in range(spec.nx):
        for j in range(spec.ny):
            for k in range(spec.nz):
                hexes.append(
                    (
                        node_id(spec, i, j, k),
                        node_id(spec, i + 1, j, k),
                        node_id(spec, i + 1, j + 1, k),
                        node_id(spec, i, j + 1, k),
                        node_id(spec, i, j, k + 1),
                        node_id(spec, i + 1, j, k + 1),
                        node_id(spec, i + 1, j + 1, k + 1),
                        node_id(spec, i, j + 1, k + 1),
                    )
                )
    return np.asarray(hexes, dtype=np.int64)


def build_surface_quads(spec: BeamSpec) -> dict[str, np.ndarray]:
    surfaces: dict[str, list[tuple[int, int, int, int]]] = {
        "left": [],
        "right": [],
        "front": [],
        "back": [],
        "bottom": [],
        "top": [],
    }

    for j in range(spec.ny):
        for k in range(spec.nz):
            surfaces["left"].append(
                (
                    node_id(spec, 0, j, k),
                    node_id(spec, 0, j, k + 1),
                    node_id(spec, 0, j + 1, k + 1),
                    node_id(spec, 0, j + 1, k),
                )
            )
            surfaces["right"].append(
                (
                    node_id(spec, spec.nx, j, k),
                    node_id(spec, spec.nx, j + 1, k),
                    node_id(spec, spec.nx, j + 1, k + 1),
                    node_id(spec, spec.nx, j, k + 1),
                )
            )

    for i in range(spec.nx):
        for k in range(spec.nz):
            surfaces["front"].append(
                (
                    node_id(spec, i, 0, k),
                    node_id(spec, i + 1, 0, k),
                    node_id(spec, i + 1, 0, k + 1),
                    node_id(spec, i, 0, k + 1),
                )
            )
            surfaces["back"].append(
                (
                    node_id(spec, i, spec.ny, k),
                    node_id(spec, i, spec.ny, k + 1),
                    node_id(spec, i + 1, spec.ny, k + 1),
                    node_id(spec, i + 1, spec.ny, k),
                )
            )

    for i in range(spec.nx):
        for j in range(spec.ny):
            surfaces["bottom"].append(
                (
                    node_id(spec, i, j, 0),
                    node_id(spec, i, j + 1, 0),
                    node_id(spec, i + 1, j + 1, 0),
                    node_id(spec, i + 1, j, 0),
                )
            )
            surfaces["top"].append(
                (
                    node_id(spec, i, j, spec.nz),
                    node_id(spec, i + 1, j, spec.nz),
                    node_id(spec, i + 1, j + 1, spec.nz),
                    node_id(spec, i, j + 1, spec.nz),
                )
            )

    return {
        name: np.asarray(quads, dtype=np.int64)
        for name, quads in surfaces.items()
    }


def make_hex_grid(points: np.ndarray, hexes: np.ndarray) -> pv.UnstructuredGrid:
    cells = np.column_stack(
        [np.full(hexes.shape[0], 8, dtype=np.int64), hexes]
    )
    cell_types = np.full(hexes.shape[0], pv.CellType.HEXAHEDRON, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    grid.point_data["GlobalNodeID"] = np.arange(1, grid.n_points + 1, dtype=np.int32)
    grid.cell_data["ModelRegionID"] = np.ones(grid.n_cells, dtype=np.int32)
    grid.cell_data["GlobalElementID"] = np.arange(1, grid.n_cells + 1, dtype=np.int32)
    return grid


def polydata_from_global_quads(points: np.ndarray,
                               quads: np.ndarray,
                               model_face_id: int) -> pv.PolyData:
    used_ids = np.asarray(sorted({int(index) for quad in quads for index in quad}),
                          dtype=np.int64)
    local_id = {int(global_id): i for i, global_id in enumerate(used_ids)}
    local_quads = np.asarray(
        [[local_id[int(index)] for index in quad] for quad in quads],
        dtype=np.int64,
    )
    cells = np.column_stack(
        [np.full(local_quads.shape[0], 4, dtype=np.int64), local_quads]
    )
    surface = pv.PolyData(points[used_ids], cells.ravel())
    surface.point_data["GlobalNodeID"] = used_ids.astype(np.int32) + 1
    surface.cell_data["ModelFaceID"] = np.full(surface.n_cells, model_face_id, dtype=np.int32)
    surface.cell_data["GlobalElementID"] = np.arange(1, surface.n_cells + 1, dtype=np.int32)
    return surface


def exterior_surface(points: np.ndarray,
                     surfaces: dict[str, np.ndarray]) -> pv.PolyData:
    quads = np.concatenate([surfaces[name] for name in SURFACE_IDS], axis=0)
    surface = polydata_from_global_quads(points, quads, model_face_id=0)
    surface.cell_data["ModelFaceID"] = np.concatenate(
        [
            np.full(surfaces[name].shape[0], SURFACE_IDS[name], dtype=np.int32)
            for name in SURFACE_IDS
        ],
        axis=0,
    )
    return surface


def triangulate_quads(quads: Iterable[Iterable[int]]) -> np.ndarray:
    triangles = []
    for quad in quads:
        a, b, c, d = [int(index) for index in quad]
        triangles.append((a, b, c))
        triangles.append((a, c, d))
    return np.asarray(triangles, dtype=np.int64)


def tetgen_closed_surface_check(points: np.ndarray,
                                surfaces: dict[str, np.ndarray]) -> dict[str, int | str]:
    if tetgen is None:
        raise RuntimeError(
            "tetgen is required for the default closed-surface validation. "
            "Install tetgen or pass --skip-tetgen-check."
        )

    all_quads = np.concatenate([surfaces[name] for name in SURFACE_IDS], axis=0)
    all_triangles = triangulate_quads(all_quads)
    used_ids = np.asarray(
        sorted({int(index) for triangle in all_triangles for index in triangle}),
        dtype=np.int64,
    )
    local_id = {int(global_id): i for i, global_id in enumerate(used_ids)}
    local_triangles = np.asarray(
        [[local_id[int(index)] for index in triangle] for triangle in all_triangles],
        dtype=np.int64,
    )
    boundary_points = points[used_ids]
    boundary_surface = pv.PolyData(
        boundary_points,
        np.column_stack(
            [
                np.full(local_triangles.shape[0], 3, dtype=np.int64),
                local_triangles,
            ]
        ).ravel(),
    )
    if boundary_surface.n_open_edges != 0:
        raise RuntimeError(
            f"beam exterior is not closed; open edges={boundary_surface.n_open_edges}"
        )

    generator = tetgen.TetGen(boundary_points, local_triangles)
    result = generator.tetrahedralize(
        plc=True,
        quality=True,
        nobisect=True,
        quiet=True,
    )
    tet_points = result[0]
    tets = result[1]
    return {
        "status": "passed",
        "boundary_points": int(boundary_points.shape[0]),
        "boundary_triangles": int(local_triangles.shape[0]),
        "tetgen_points": int(tet_points.shape[0]),
        "tetgen_tetrahedra": int(tets.shape[0]),
    }


def write_mesh(output_dir: Path,
               spec: BeamSpec,
               skip_tetgen_check: bool) -> dict[str, object]:
    mesh_dir = output_dir / "mesh"
    surfaces_dir = mesh_dir / "mesh-surfaces"
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    points = build_points(spec)
    hexes = build_hex_connectivity(spec)
    surfaces = build_surface_quads(spec)
    grid = make_hex_grid(points, hexes)

    if grid.n_cells != spec.nx * spec.ny * spec.nz:
        raise RuntimeError("unexpected Hex8 cell count")
    if not np.all(grid.celltypes == pv.CellType.HEXAHEDRON):
        raise RuntimeError("generated volume mesh contains non-Hex8 cells")

    grid.save(mesh_dir / "mesh-complete.mesh.vtu", binary=False)
    exterior_surface(points, surfaces).save(mesh_dir / "mesh-complete.exterior.vtp", binary=False)
    for name, quads in surfaces.items():
        polydata_from_global_quads(points, quads, SURFACE_IDS[name]).save(
            surfaces_dir / f"{name}.vtp",
            binary=False,
        )

    tetgen_check: dict[str, int | str]
    if skip_tetgen_check:
        tetgen_check = {"status": "skipped"}
    else:
        tetgen_check = tetgen_closed_surface_check(points, surfaces)

    manifest: dict[str, object] = {
        "description": "Long rectangular Ustruct beam test mesh with Hex8 elements.",
        "mesh_file": "mesh/mesh-complete.mesh.vtu",
        "exterior_file": "mesh/mesh-complete.exterior.vtp",
        "surface_files": {
            name: f"mesh/mesh-surfaces/{name}.vtp"
            for name in SURFACE_IDS
        },
        "surface_ids": SURFACE_IDS,
        "spec": asdict(spec),
        "points": int(grid.n_points),
        "hex8_elements": int(grid.n_cells),
        "bounds": [float(value) for value in grid.bounds],
        "tetgen_closed_surface_check": tetgen_check,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=THIS_DIR)
    parser.add_argument("--length", type=float, default=BeamSpec.length)
    parser.add_argument("--width", type=float, default=BeamSpec.width)
    parser.add_argument("--height", type=float, default=BeamSpec.height)
    parser.add_argument("--nx", type=int, default=BeamSpec.nx)
    parser.add_argument("--ny", type=int, default=BeamSpec.ny)
    parser.add_argument("--nz", type=int, default=BeamSpec.nz)
    parser.add_argument(
        "--skip-tetgen-check",
        action="store_true",
        help="Write the Hex8 mesh without TetGen closed-surface validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = BeamSpec(
        length=args.length,
        width=args.width,
        height=args.height,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
    )
    manifest = write_mesh(args.output_dir, spec, args.skip_tetgen_check)
    print(
        "Wrote {hex8_elements} Hex8 cells and {points} points to {mesh_file}".format(
            **manifest
        )
    )


if __name__ == "__main__":
    main()
