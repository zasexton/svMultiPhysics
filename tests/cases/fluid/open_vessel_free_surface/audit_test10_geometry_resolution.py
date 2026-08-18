#!/usr/bin/env python3
"""Audit SPHERIC Test10 unfitted mesh, sensor, and level-set cut quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = (
    SCRIPT_DIR
    / "unfitted_level_set"
    / "spheric_test10_lateral_water_1x"
)


def tetra_volume(points: np.ndarray) -> float:
    return abs(float(np.dot(points[1] - points[0], np.cross(points[2] - points[0], points[3] - points[0])))) / 6.0


def interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def clipped_negative_volume(points: np.ndarray, values: np.ndarray, tolerance: float) -> float:
    measure = tetra_volume(points)
    negative = values <= tolerance
    negative_count = int(np.count_nonzero(negative))
    if negative_count == 0:
        return 0.0
    if negative_count == 4:
        return measure

    inside = [index for index, is_inside in enumerate(negative) if is_inside]
    outside = [index for index, is_inside in enumerate(negative) if not is_inside]

    def interface_point(i: int, j: int) -> np.ndarray:
        vi = float(values[i])
        vj = float(values[j])
        if abs(vi) <= tolerance:
            return points[i]
        if abs(vj) <= tolerance:
            return points[j]
        denom = vi - vj
        t = vi / denom if denom != 0.0 else 0.0
        return interpolate(points[i], points[j], min(max(t, 0.0), 1.0))

    if negative_count == 1:
        a = inside[0]
        p0 = interface_point(a, outside[0])
        p1 = interface_point(a, outside[1])
        p2 = interface_point(a, outside[2])
        return min(max(tetra_volume(np.array([points[a], p0, p1, p2])), 0.0), measure)
    if negative_count == 3:
        p = outside[0]
        q0 = interface_point(inside[0], p)
        q1 = interface_point(inside[1], p)
        q2 = interface_point(inside[2], p)
        dry = tetra_volume(np.array([points[p], q0, q1, q2]))
        return min(max(measure - dry, 0.0), measure)

    a, b = inside
    c, d = outside
    p_ac = interface_point(a, c)
    p_ad = interface_point(a, d)
    p_bc = interface_point(b, c)
    p_bd = interface_point(b, d)
    volume = (
        tetra_volume(np.array([points[a], points[b], p_bd, p_bc]))
        + tetra_volume(np.array([points[a], p_ac, p_bc, p_bd]))
        + tetra_volume(np.array([points[a], p_ac, p_bd, p_ad]))
    )
    return min(max(volume, 0.0), measure)


def tetra_connectivity(grid: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(grid.cells, dtype=np.int64).reshape((-1, 5))
    if np.any(cells[:, 0] != 4):
        raise RuntimeError("Test10 audit expects tetrahedral cells only")
    return cells[:, 1:]


def volume_metrics(grid: pv.UnstructuredGrid, phi: np.ndarray, expected_volume: float, tolerance: float) -> dict[str, Any]:
    tets = tetra_connectivity(grid)
    points = np.asarray(grid.points, dtype=float)
    active_volume = 0.0
    total_volume = 0.0
    fractions: list[float] = []
    for tet in tets:
        cell_points = points[tet]
        measure = tetra_volume(cell_points)
        active = clipped_negative_volume(cell_points, phi[tet], tolerance)
        fraction = active / measure if measure > 0.0 else math.nan
        active_volume += active
        total_volume += measure
        fractions.append(fraction)

    fraction_array = np.asarray(fractions, dtype=float)
    cut = (fraction_array > 0.0) & (fraction_array < 1.0)
    cut_fractions = fraction_array[cut]
    return {
        "active_volume_m3": active_volume,
        "total_mesh_volume_m3": total_volume,
        "expected_initial_volume_m3": expected_volume,
        "active_volume_relative_error": (active_volume - expected_volume) / expected_volume,
        "full_cell_count": int(np.count_nonzero(fraction_array >= 1.0)),
        "dry_cell_count": int(np.count_nonzero(fraction_array <= 0.0)),
        "cut_cell_count": int(np.count_nonzero(cut)),
        "active_min_volume_fraction": float(np.min(cut_fractions)) if cut_fractions.size else None,
        "active_max_volume_fraction": float(np.max(cut_fractions)) if cut_fractions.size else None,
        "cut_fraction_below_1e_2": int(np.count_nonzero(cut_fractions < 1.0e-2)),
        "cut_fraction_below_1e_4": int(np.count_nonzero(cut_fractions < 1.0e-4)),
        "cut_fraction_below_1e_6": int(np.count_nonzero(cut_fractions < 1.0e-6)),
    }


def point_sample_status(grid: pv.UnstructuredGrid, point: np.ndarray) -> dict[str, Any]:
    points = np.asarray(grid.points, dtype=float)
    distances = np.linalg.norm(points - point.reshape(1, 3), axis=1)
    nearest_index = int(np.argmin(distances))
    sample = pv.PolyData(point.reshape(1, 3)).sample(grid, tolerance=1.0e-8)
    valid = bool(
        "vtkValidPointMask" in sample.point_data
        and int(np.asarray(sample.point_data["vtkValidPointMask"]).reshape(-1)[0]) == 1
    )
    return {
        "target_point": point.tolist(),
        "interpolated_point_sample_valid": valid,
        "containing_cell": int(grid.find_containing_cell(point)),
        "nearest_node_index": nearest_index,
        "nearest_node_point": points[nearest_index].tolist(),
        "nearest_node_distance_m": float(distances[nearest_index]),
    }


def audit(case_dir: Path, tolerance: float) -> dict[str, Any]:
    benchmark = json.loads((case_dir / "benchmark.json").read_text(encoding="utf-8"))
    dims = benchmark["dimensions_m"]
    fill_height = float(dims["initial_fill_height"])
    expected_volume = (
        float(dims["tank_length"])
        * float(dims["tank_breadth_1x"])
        * fill_height
    )
    mesh_path = case_dir / "mesh" / "background" / "mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    checked_phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    plane_phi = points[:, 1] - fill_height
    delta = checked_phi - plane_phi
    metrics = volume_metrics(grid, plane_phi, expected_volume, tolerance)

    pressure_sensor = benchmark.get("pressure_sensor", {})
    pressure_anchor = benchmark.get("pressure_gauge", {})
    sensor_point = np.asarray(pressure_sensor.get("coordinates", [math.nan, math.nan, math.nan]), dtype=float)
    anchor_point = np.asarray(pressure_anchor.get("coordinates", [math.nan, math.nan, math.nan]), dtype=float)

    min_fraction = metrics["active_min_volume_fraction"]
    if min_fraction is not None and min_fraction < 1.0e-4:
        mesh_finding = (
            "The current Test10 unfitted mesh has tiny free-surface cut fractions; "
            "this is a mesh/topology accuracy blocker, not literature-validation evidence."
        )
    else:
        mesh_finding = (
            "The current Test10 unfitted mesh does not show tiny initial cut fractions "
            "at the configured fill plane."
        )

    return {
        "case_dir": str(case_dir),
        "mesh_path": str(mesh_path),
        "mesh_points": int(grid.n_points),
        "mesh_cells": int(grid.n_cells),
        "checked_phi_matches_fill_plane": bool(np.max(np.abs(delta)) <= 1.0e-12),
        "checked_phi_max_abs_delta_from_fill_plane": float(np.max(np.abs(delta))),
        "fill_height_m": fill_height,
        "expected_initial_volume_m3": expected_volume,
        "zero_phi_node_count": int(np.count_nonzero(np.abs(plane_phi) <= tolerance)),
        "volume_metrics": metrics,
        "pressure_sensor": pressure_sensor,
        "pressure_sensor_sample": point_sample_status(grid, sensor_point),
        "pressure_anchor": pressure_anchor,
        "pressure_anchor_sample": point_sample_status(grid, anchor_point),
        "mesh_resolution_finding": mesh_finding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--tolerance", type=float, default=1.0e-10)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = audit(args.case_dir, args.tolerance)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
