#!/usr/bin/env python3
"""Verify the small flat hydrostatic unfitted free-surface case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent
ZERO_TOL = 1.0e-12


@dataclass(frozen=True)
class Crossing:
    point: np.ndarray
    edge: tuple[int, int]
    t: float


def latest_result(case_dir: Path) -> Path:
    def step(path: Path) -> int:
        match = re.search(r"_(\d+)\.p?vtu$", path.name)
        return int(match.group(1)) if match else -1

    candidates = sorted([*case_dir.glob("result_*.vtu"), *case_dir.glob("result_*.pvtu")], key=step)
    if not candidates:
        raise FileNotFoundError(f"no result_*.vtu or result_*.pvtu files found in {case_dir}")
    return candidates[-1]


def resolve_result_path(result: Path | None, case_dir: Path) -> Path:
    if result is None:
        return latest_result(case_dir)
    if result.exists() or result.is_absolute():
        return result
    candidate = case_dir / result
    return candidate if candidate.exists() else result


def load_expected(case_dir: Path) -> dict:
    with (case_dir / "expected_results.json").open() as stream:
        return json.load(stream)


def quad_cells(grid: pv.UnstructuredGrid) -> np.ndarray:
    cells = grid.cells.reshape((-1, 5))
    if not np.all(cells[:, 0] == 4):
        raise RuntimeError("expected a pure quadrilateral VTK mesh")
    return cells[:, 1:].astype(np.int64)


def unique_edges(cells: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[0])):
            key = tuple(sorted((int(a), int(b))))
            if key in seen:
                continue
            seen.add(key)
            out.append((int(a), int(b)))
    return out


def edge_crossings(points: np.ndarray, phi: np.ndarray, cells: np.ndarray) -> list[Crossing]:
    crossings: list[Crossing] = []
    for a, b in unique_edges(cells):
        pa = float(phi[a])
        pb = float(phi[b])
        if abs(pa) <= ZERO_TOL:
            crossings.append(Crossing(points[a, :2], (a, b), 0.0))
        elif abs(pb) <= ZERO_TOL:
            crossings.append(Crossing(points[b, :2], (a, b), 1.0))
        elif pa * pb < 0.0:
            t = pa / (pa - pb)
            crossings.append(Crossing((1.0 - t) * points[a, :2] + t * points[b, :2], (a, b), t))
    if not crossings:
        raise RuntimeError("could not find phi=0 crossings")
    return crossings


def polygon_area(poly: list[np.ndarray]) -> float:
    if len(poly) < 3:
        return 0.0
    pts = np.asarray(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: list[np.ndarray]) -> np.ndarray:
    if len(poly) < 3:
        return np.zeros(2)
    pts = np.asarray(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = float(np.sum(cross))
    if abs(area2) <= 1.0e-14:
        return np.zeros(2)
    return np.array([
        float(np.sum((x + np.roll(x, -1)) * cross) / (3.0 * area2)),
        float(np.sum((y + np.roll(y, -1)) * cross) / (3.0 * area2)),
    ])


def clip_negative(vertices: list[np.ndarray], values: list[float]) -> list[np.ndarray]:
    clipped: list[np.ndarray] = []
    for i, current in enumerate(vertices):
        previous = vertices[i - 1]
        cv = values[i]
        pv = values[i - 1]
        current_inside = cv <= 0.0
        previous_inside = pv <= 0.0
        if current_inside != previous_inside:
            t = pv / (pv - cv)
            clipped.append((1.0 - t) * previous + t * current)
        if current_inside:
            clipped.append(current)
    return clipped


def clipped_area_centroid(points: np.ndarray, phi: np.ndarray, cells: np.ndarray) -> tuple[float, np.ndarray]:
    total = 0.0
    moment = np.zeros(2)
    for cell in cells:
        vertices = [points[index, :2] for index in cell]
        values = [float(phi[index]) for index in cell]
        clipped = clip_negative(vertices, values)
        area = polygon_area(clipped)
        total += area
        moment += area * polygon_centroid(clipped)
    if total <= 0.0:
        return 0.0, np.zeros(2)
    return total, moment / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path)
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    args = parser.parse_args()

    expected = load_expected(args.case_dir)
    result_path = resolve_result_path(args.result, args.case_dir)
    grid = pv.read(result_path)
    points = np.asarray(grid.points, dtype=float)
    cells = quad_cells(grid)
    phi = np.asarray(grid.point_data["phi"], dtype=float)
    velocity = np.asarray(grid.point_data["Velocity"], dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)

    solution = expected["analytic_solution"]
    fluid = expected["fluid"]
    tolerances = expected["verification"]["suggested_tolerances"]
    height = float(solution["free_surface_height"])
    area_expected = float(solution["area"])
    centroid_expected = np.asarray(solution["centroid"][:2], dtype=float)

    crossings = edge_crossings(points, phi, cells)
    crossing_points = np.asarray([crossing.point for crossing in crossings], dtype=float)
    area, centroid = clipped_area_centroid(points, phi, cells)

    wet = phi <= -float(expected["verification"].get("wet_pressure_margin", 0.0))
    finite_wet = wet & np.isfinite(pressure) & np.all(np.isfinite(velocity[:, :2]), axis=1)
    if not np.any(finite_wet):
        raise RuntimeError("no finite wet vertices available for verification")

    exact_pressure = float(fluid["density"]) * float(fluid["gravity"]) * (height - points[:, 1])
    pressure_error = pressure[finite_wet] - exact_pressure[finite_wet]
    wet_speed = np.linalg.norm(velocity[finite_wet, :2], axis=1)
    metrics = {
        "result": str(result_path),
        "interface_crossing_count": int(len(crossings)),
        "interface_height_mean": float(np.mean(crossing_points[:, 1])),
        "interface_height_max_abs_error": float(np.max(np.abs(crossing_points[:, 1] - height))),
        "area": float(area),
        "area_error": float(area - area_expected),
        "area_relative_error": float(abs(area - area_expected) / area_expected),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_error": float(np.linalg.norm(centroid - centroid_expected)),
        "wet_vertex_count": int(np.count_nonzero(wet)),
        "pressure_nonfinite_count": int(np.count_nonzero(~np.isfinite(pressure))),
        "velocity_nonfinite_count": int(np.count_nonzero(~np.isfinite(velocity))),
        "pressure_wet_rms_error": float(np.sqrt(np.mean(pressure_error * pressure_error))),
        "pressure_wet_max_abs_error": float(np.max(np.abs(pressure_error))),
        "velocity_wet_max": float(np.max(wet_speed)),
    }
    checks = {
        "interface_height_abs": metrics["interface_height_max_abs_error"] <= tolerances["interface_height_abs"],
        "area_abs": abs(metrics["area_error"]) <= tolerances["area_abs"],
        "centroid_abs": metrics["centroid_error"] <= tolerances["centroid_abs"],
        "pressure_wet_rms_abs": metrics["pressure_wet_rms_error"] <= tolerances["pressure_wet_rms_abs"],
        "pressure_wet_max_abs": metrics["pressure_wet_max_abs_error"] <= tolerances["pressure_wet_max_abs"],
        "velocity_wet_max": metrics["velocity_wet_max"] <= tolerances["velocity_wet_max"],
    }
    metrics["failed_checks"] = [name for name, ok in checks.items() if not ok]
    metrics["passed"] = not metrics["failed_checks"]
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
