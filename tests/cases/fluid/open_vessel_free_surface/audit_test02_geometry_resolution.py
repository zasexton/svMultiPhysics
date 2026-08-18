#!/usr/bin/env python3
"""Audit SPHERIC Test02 level-set geometry and current mesh resolution."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_DIR = (
    SCRIPT_DIR
    / "unfitted_level_set"
    / "spheric_test02_dambreak_obstacle"
)
TEST02_TANK_LENGTH = 3.22
TEST02_TANK_WIDTH = 1.00
TEST02_INITIAL_COLUMN_LENGTH = 1.228
TEST02_INITIAL_COLUMN_HEIGHT = 0.55
TEST02_GATE_X = TEST02_TANK_LENGTH - TEST02_INITIAL_COLUMN_LENGTH
EXPECTED_INITIAL_VOLUME_M3 = (
    TEST02_INITIAL_COLUMN_LENGTH * TEST02_INITIAL_COLUMN_HEIGHT * TEST02_TANK_WIDTH
)


@dataclass(frozen=True)
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.xmin + self.xmax),
                0.5 * (self.ymin + self.ymax),
                0.5 * (self.zmin + self.zmax),
            ],
            dtype=float,
        )

    @property
    def half_width(self) -> np.ndarray:
        return np.array(
            [
                0.5 * (self.xmax - self.xmin),
                0.5 * (self.ymax - self.ymin),
                0.5 * (self.zmax - self.zmin),
            ],
            dtype=float,
        )


def signed_distance_to_box(points: np.ndarray, box: Box) -> np.ndarray:
    q = np.abs(points - box.center.reshape(1, 3)) - box.half_width.reshape(1, 3)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum.reduce(q, axis=1), 0.0)
    return outside + inside


def corrected_test02_phi(points: np.ndarray, gate_x: float, height: float) -> np.ndarray:
    return np.maximum(gate_x - points[:, 0], points[:, 1] - height)


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
        raise RuntimeError("Test02 audit expects tetrahedral cells only")
    return cells[:, 1:]


def cell_edge_lengths(points: np.ndarray) -> list[float]:
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    return [float(np.linalg.norm(points[i] - points[j])) for i, j in edges]


def volume_metrics(grid: pv.UnstructuredGrid, phi: np.ndarray, tolerance: float) -> dict[str, Any]:
    tets = tetra_connectivity(grid)
    points = np.asarray(grid.points, dtype=float)
    active_volume = 0.0
    total_volume = 0.0
    fractions: list[float] = []
    edge_lengths: list[float] = []
    for tet in tets:
        cell_points = points[tet]
        measure = tetra_volume(cell_points)
        active = clipped_negative_volume(cell_points, phi[tet], tolerance)
        fraction = active / measure if measure > 0.0 else math.nan
        active_volume += active
        total_volume += measure
        fractions.append(fraction)
        if 0.0 < fraction < 1.0:
            edge_lengths.extend(cell_edge_lengths(cell_points))

    fraction_array = np.asarray(fractions, dtype=float)
    cut = (fraction_array > 0.0) & (fraction_array < 1.0)
    cut_fractions = fraction_array[cut]
    return {
        "active_volume_m3": active_volume,
        "total_mesh_volume_m3": total_volume,
        "expected_initial_volume_m3": EXPECTED_INITIAL_VOLUME_M3,
        "active_volume_relative_error": (
            active_volume - EXPECTED_INITIAL_VOLUME_M3
        ) / EXPECTED_INITIAL_VOLUME_M3,
        "full_cell_count": int(np.count_nonzero(fraction_array >= 1.0)),
        "dry_cell_count": int(np.count_nonzero(fraction_array <= 0.0)),
        "cut_cell_count": int(np.count_nonzero(cut)),
        "active_min_volume_fraction": float(np.min(cut_fractions)) if cut_fractions.size else None,
        "active_max_volume_fraction": float(np.max(cut_fractions)) if cut_fractions.size else None,
        "cut_fraction_below_1e_2": int(np.count_nonzero(cut_fractions < 1.0e-2)),
        "cut_fraction_below_1e_4": int(np.count_nonzero(cut_fractions < 1.0e-4)),
        "cut_fraction_below_1e_6": int(np.count_nonzero(cut_fractions < 1.0e-6)),
        "cut_edge_length_min_m": float(min(edge_lengths)) if edge_lengths else None,
        "cut_edge_length_max_m": float(max(edge_lengths)) if edge_lengths else None,
        "cut_edge_length_mean_m": float(sum(edge_lengths) / len(edge_lengths)) if edge_lengths else None,
    }


def wall_contact_metrics(
    points: np.ndarray,
    phi: np.ndarray,
    *,
    gate_x: float,
    water_height: float,
    tolerance: float,
) -> dict[str, Any]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    contact = {
        "bottom": np.isclose(y, 0.0, atol=tolerance) & (x >= gate_x - tolerance),
        "right": np.isclose(x, 3.22, atol=tolerance) & (y <= water_height + tolerance),
        "front": np.isclose(z, 0.0, atol=tolerance) & (x >= gate_x - tolerance) & (y <= water_height + tolerance),
        "back": np.isclose(z, 1.0, atol=tolerance) & (x >= gate_x - tolerance) & (y <= water_height + tolerance),
    }
    result: dict[str, Any] = {}
    for name, mask in contact.items():
        values = phi[mask]
        result[name] = {
            "node_count": int(values.size),
            "zero_phi_nodes": int(np.count_nonzero(np.abs(values) <= tolerance)),
            "negative_phi_nodes": int(np.count_nonzero(values < -tolerance)),
            "positive_phi_nodes": int(np.count_nonzero(values > tolerance)),
        }
    return result


def audit(case_dir: Path, tolerance: float) -> dict[str, Any]:
    mesh_path = case_dir / "mesh/background/mesh-complete.mesh.vtu"
    grid = pv.read(mesh_path)
    points = np.asarray(grid.points, dtype=float)
    checked_phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    gate_x = TEST02_GATE_X
    water_height = TEST02_INITIAL_COLUMN_HEIGHT
    closed_box_phi = signed_distance_to_box(
        points,
        Box(gate_x, TEST02_TANK_LENGTH, 0.0, water_height, 0.0, TEST02_TANK_WIDTH),
    )
    plane_phi = corrected_test02_phi(points, gate_x, water_height)
    checked_delta = checked_phi - plane_phi
    checked_matches_plane = bool(np.max(np.abs(checked_delta)) <= 1.0e-12)
    plane_metrics = volume_metrics(grid, plane_phi, tolerance)
    closed_box_metrics = volume_metrics(grid, closed_box_phi, tolerance)
    min_fraction = plane_metrics["active_min_volume_fraction"]
    if (
        min_fraction is not None
        and min_fraction >= 1.0e-2
        and plane_metrics["cut_fraction_below_1e_2"] == 0
    ):
        mesh_finding = (
            "This mesh avoids the tiny generated cut fractions seen in the "
            "checked coarse fixture and is suitable as a topology diagnostic, "
            "but it is still not a SPHERIC Test02 validation on its own."
        )
    else:
        mesh_finding = (
            "This mesh still has small cut fractions and is not mesh-independent "
            "evidence for Test02."
        )
    return {
        "case_dir": str(case_dir),
        "mesh_path": str(mesh_path),
        "mesh_points": int(grid.n_points),
        "mesh_cells": int(grid.n_cells),
        "gate_x_m": gate_x,
        "water_height_m": water_height,
        "checked_phi_matches_plane_formula": checked_matches_plane,
        "checked_phi_max_abs_delta_from_plane": float(np.max(np.abs(checked_delta))),
        "plane_phi": {
            "volume_metrics": plane_metrics,
            "wall_contact_metrics": wall_contact_metrics(
                points,
                plane_phi,
                gate_x=gate_x,
                water_height=water_height,
                tolerance=tolerance,
            ),
            "zero_phi_node_count": int(np.count_nonzero(np.abs(plane_phi) <= tolerance)),
        },
        "closed_box_signed_distance_phi": {
            "volume_metrics": closed_box_metrics,
            "wall_contact_metrics": wall_contact_metrics(
                points,
                closed_box_phi,
                gate_x=gate_x,
                water_height=water_height,
                tolerance=tolerance,
            ),
            "zero_phi_node_count": int(np.count_nonzero(np.abs(closed_box_phi) <= tolerance)),
        },
        "finding": (
            "The checked plane level-set removes wall-contact zero cuts from the "
            f"closed-box signed-distance field. {mesh_finding}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--tolerance", type=float, default=1.0e-12)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = audit(args.case_dir, args.tolerance)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
