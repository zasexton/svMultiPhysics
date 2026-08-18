#!/usr/bin/env python3
"""Check a tilted-square run against the analytic wet-domain equilibrium."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent
ZERO_TOL = 1.0e-12
DEDUP_TOL = 1.0e-10


@dataclass(frozen=True)
class InterfaceCrossing:
    point: np.ndarray
    edge: tuple[int, int]
    t: float


@dataclass(frozen=True)
class InterfaceSegment:
    point_a: np.ndarray
    point_b: np.ndarray
    pressure_a: float
    pressure_b: float
    length: float


def latest_result(case_dir: Path) -> Path:
    def step(path: Path) -> int:
        match = re.search(r"_(\d+)\.p?vtu$", path.name)
        return int(match.group(1)) if match else -1

    candidates = sorted(
        [*case_dir.glob("result_*.vtu"), *case_dir.glob("result_*.pvtu")],
        key=step,
    )
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
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[0])):
            edge = tuple(sorted((int(a), int(b))))
            if edge in seen:
                continue
            seen.add(edge)
            edges.append((int(a), int(b)))
    return edges


def edge_zero_crossings(
    points: np.ndarray,
    phi: np.ndarray,
    cells: np.ndarray,
    *,
    zero_tol: float = ZERO_TOL,
    dedup_tol: float = DEDUP_TOL,
) -> list[InterfaceCrossing]:
    crossings: list[InterfaceCrossing] = []
    point_keys: set[tuple[int, int]] = set()

    def add_crossing(a: int, b: int, t: float) -> None:
        point = (1.0 - t) * points[a, :2] + t * points[b, :2]
        key = tuple(np.round(point / dedup_tol).astype(np.int64))
        if key in point_keys:
            return
        point_keys.add(key)
        crossings.append(InterfaceCrossing(point=point, edge=(a, b), t=float(t)))

    for a, b in unique_edges(cells):
        pa = float(phi[a])
        pb = float(phi[b])
        a_zero = abs(pa) <= zero_tol
        b_zero = abs(pb) <= zero_tol
        if a_zero and b_zero:
            add_crossing(a, b, 0.0)
            add_crossing(a, b, 1.0)
        elif a_zero:
            add_crossing(a, b, 0.0)
        elif b_zero:
            add_crossing(a, b, 1.0)
        elif pa * pb < 0.0:
            add_crossing(a, b, pa / (pa - pb))

    if not crossings:
        raise RuntimeError("could not find a phi=0 interface in the result")
    return crossings


def crossing_points(crossings: list[InterfaceCrossing]) -> np.ndarray:
    return np.array([crossing.point for crossing in crossings], dtype=float)


def interpolate_crossing_values(values: np.ndarray, crossings: list[InterfaceCrossing]) -> np.ndarray:
    interpolated = []
    for crossing in crossings:
        a, b = crossing.edge
        interpolated.append((1.0 - crossing.t) * values[a] + crossing.t * values[b])
    return np.asarray(interpolated, dtype=float)


def domain_bounds(points: np.ndarray) -> tuple[float, float, float, float]:
    return (
        float(np.min(points[:, 0])),
        float(np.max(points[:, 0])),
        float(np.min(points[:, 1])),
        float(np.max(points[:, 1])),
    )


def boundary_tolerance(points: np.ndarray) -> float:
    x_min, x_max, y_min, y_max = domain_bounds(points)
    extent = max(x_max - x_min, y_max - y_min, 1.0)
    return 1.0e-10 * extent


def minimum_mesh_spacing(points: np.ndarray) -> float:
    x_min, x_max, y_min, y_max = domain_bounds(points)
    extent = max(x_max - x_min, y_max - y_min, 1.0)
    spacing_tol = 1.0e-12 * extent
    candidates: list[float] = []
    for axis in (0, 1):
        coordinates = np.unique(np.round(points[:, axis].astype(float), decimals=14))
        differences = np.diff(np.sort(coordinates))
        candidates.extend(float(diff) for diff in differences if diff > spacing_tol)
    if candidates:
        return min(candidates)
    return extent


def interface_pressure_near_boundary_width(points: np.ndarray, expected: dict) -> float:
    verification = expected.get("verification", {})
    configured_width = verification.get("interface_pressure_near_boundary_width")
    if configured_width is not None:
        return max(0.0, float(configured_width))
    width_in_cells = float(verification.get("interface_pressure_near_boundary_cells", 1.0))
    return max(0.0, width_in_cells * minimum_mesh_spacing(points))


def distance_to_domain_boundary(
    point: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> float:
    x_min, x_max, y_min, y_max = bounds
    return min(
        abs(float(point[0]) - x_min),
        abs(float(point[0]) - x_max),
        abs(float(point[1]) - y_min),
        abs(float(point[1]) - y_max),
    )


def point_on_domain_boundary(
    point: np.ndarray,
    bounds: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return (
        abs(float(point[0]) - x_min) <= tolerance
        or abs(float(point[0]) - x_max) <= tolerance
        or abs(float(point[1]) - y_min) <= tolerance
        or abs(float(point[1]) - y_max) <= tolerance
    )


def cell_interface_segments(
    points: np.ndarray,
    phi: np.ndarray,
    pressure: np.ndarray,
    cells: np.ndarray,
    *,
    zero_tol: float = ZERO_TOL,
    dedup_tol: float = DEDUP_TOL,
) -> list[InterfaceSegment]:
    segments: list[InterfaceSegment] = []

    for cell in cells:
        crossings: list[tuple[np.ndarray, float]] = []
        crossing_keys: set[tuple[int, int]] = set()

        def add_crossing(a: int, b: int, t: float) -> None:
            point = (1.0 - t) * points[a, :2] + t * points[b, :2]
            key = tuple(np.round(point / dedup_tol).astype(np.int64))
            if key in crossing_keys:
                return
            crossing_keys.add(key)
            pressure_value = (1.0 - t) * pressure[a] + t * pressure[b]
            crossings.append((point, float(pressure_value)))

        for a, b in ((cell[0], cell[1]), (cell[1], cell[2]), (cell[2], cell[3]), (cell[3], cell[0])):
            pa = float(phi[a])
            pb = float(phi[b])
            a_zero = abs(pa) <= zero_tol
            b_zero = abs(pb) <= zero_tol
            if a_zero and b_zero:
                add_crossing(int(a), int(b), 0.0)
                add_crossing(int(a), int(b), 1.0)
            elif a_zero:
                add_crossing(int(a), int(b), 0.0)
            elif b_zero:
                add_crossing(int(a), int(b), 1.0)
            elif pa * pb < 0.0:
                add_crossing(int(a), int(b), pa / (pa - pb))

        for left, right in zip(crossings[::2], crossings[1::2]):
            point_a, pressure_a = left
            point_b, pressure_b = right
            length = float(np.linalg.norm(point_b - point_a))
            if length <= zero_tol:
                continue
            segments.append(
                InterfaceSegment(
                    point_a=point_a,
                    point_b=point_b,
                    pressure_a=pressure_a,
                    pressure_b=pressure_b,
                    length=length,
                )
            )

    return segments


def edge_to_cells(cells: np.ndarray) -> dict[tuple[int, int], list[int]]:
    adjacent: dict[tuple[int, int], list[int]] = {}
    for cell_index, cell in enumerate(cells):
        for a, b in (
            (cell[0], cell[1]),
            (cell[1], cell[2]),
            (cell[2], cell[3]),
            (cell[3], cell[0]),
        ):
            edge = tuple(sorted((int(a), int(b))))
            adjacent.setdefault(edge, []).append(cell_index)
    return adjacent


def bilinear_value(values: np.ndarray, xi: float, eta: float) -> float:
    weights = np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            xi * (1.0 - eta),
            xi * eta,
            (1.0 - xi) * eta,
        ],
        dtype=float,
    )
    return float(np.dot(weights, values))


def bilinear_gradient(values: np.ndarray, dx: float, dy: float, xi: float, eta: float) -> np.ndarray:
    dxi = np.array([-(1.0 - eta), 1.0 - eta, eta, -eta], dtype=float)
    deta = np.array([-(1.0 - xi), -xi, xi, 1.0 - xi], dtype=float)
    return np.array(
        [
            float(np.dot(dxi, values) / dx),
            float(np.dot(deta, values) / dy),
        ],
        dtype=float,
    )


def axis_aligned_quad_local_coordinates(
    cell_points: np.ndarray,
    point: np.ndarray,
) -> tuple[float, float, float, float] | None:
    x_min = float(np.min(cell_points[:, 0]))
    x_max = float(np.max(cell_points[:, 0]))
    y_min = float(np.min(cell_points[:, 1]))
    y_max = float(np.max(cell_points[:, 1]))
    dx = x_max - x_min
    dy = y_max - y_min
    if dx <= ZERO_TOL or dy <= ZERO_TOL:
        return None

    expected_order = np.array(
        [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
        ],
        dtype=float,
    )
    if np.max(np.abs(cell_points[:, :2] - expected_order)) > 1.0e-9:
        return None

    xi = float((point[0] - x_min) / dx)
    eta = float((point[1] - y_min) / dy)
    if xi < -1.0e-9 or xi > 1.0 + 1.0e-9 or eta < -1.0e-9 or eta > 1.0 + 1.0e-9:
        return None
    return min(1.0, max(0.0, xi)), min(1.0, max(0.0, eta)), dx, dy


def fit_interface_line(crossings: list[InterfaceCrossing] | np.ndarray) -> tuple[float, float]:
    points = crossing_points(crossings) if isinstance(crossings, list) else crossings
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def line_residual_metrics(points: np.ndarray, slope: float, intercept: float) -> tuple[float, float]:
    residual = points[:, 1] - (slope * points[:, 0] + intercept)
    return float(np.sqrt(np.mean(residual * residual))), float(np.max(np.abs(residual)))


def final_equilibrium_required(expected: dict) -> bool:
    verification = expected.get("verification", {})
    if "final_equilibrium_required" in verification:
        return bool(verification["final_equilibrium_required"])
    return expected.get("initial_condition", {}).get("mode") == "equilibrium"


def verification_flag(expected: dict, name: str, default: bool) -> bool:
    return bool(expected.get("verification", {}).get(name, default))


def interface_pressure_check_scope(expected: dict) -> str:
    scope = str(expected.get("verification", {}).get("interface_pressure_check_scope", "all"))
    if scope not in {"all", "interior", "core"}:
        raise RuntimeError(f"unsupported interface_pressure_check_scope: {scope}")
    return scope


def slope_progress_fraction(slope: float, expected: dict) -> float:
    initial_line = expected.get("initial_condition", {}).get("free_surface_line", {})
    initial_slope = float(initial_line.get("slope", 0.0))
    target_slope = float(expected["analytic_equilibrium"]["free_surface_line"]["slope"])
    denominator = target_slope - initial_slope
    if abs(denominator) <= ZERO_TOL:
        return 1.0 if abs(slope - target_slope) <= ZERO_TOL else 0.0
    return float((slope - initial_slope) / denominator)


def polygon_area(poly: list[np.ndarray]) -> float:
    if len(poly) < 3:
        return 0.0
    pts = np.array(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(poly: list[np.ndarray]) -> np.ndarray:
    if len(poly) < 3:
        return np.zeros(2)
    pts = np.array(poly)
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = float(np.sum(cross))
    if abs(area2) < 1.0e-14:
        return np.zeros(2)
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (3.0 * area2))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (3.0 * area2))
    return np.array([cx, cy])


def clip_polygon_negative(vertices: list[np.ndarray], values: list[float]) -> list[np.ndarray]:
    clipped: list[np.ndarray] = []
    for i, current in enumerate(vertices):
        previous = vertices[i - 1]
        current_value = values[i]
        previous_value = values[i - 1]
        current_inside = current_value <= 0.0
        previous_inside = previous_value <= 0.0
        if current_inside != previous_inside:
            t = previous_value / (previous_value - current_value)
            clipped.append((1.0 - t) * previous + t * current)
        if current_inside:
            clipped.append(current)
    return clipped


def clipped_area_and_centroid(
    points: np.ndarray,
    phi: np.ndarray,
    cells: np.ndarray,
) -> tuple[float, np.ndarray]:
    total_area = 0.0
    first_moment = np.zeros(2)
    for cell in cells:
        vertices = [points[index, :2] for index in cell]
        values = [float(phi[index]) for index in cell]
        clipped = clip_polygon_negative(vertices, values)
        area = polygon_area(clipped)
        centroid = polygon_centroid(clipped)
        total_area += area
        first_moment += area * centroid
    if total_area == 0.0:
        return 0.0, np.zeros(2)
    return total_area, first_moment / total_area


def expected_pressure(points: np.ndarray, expected: dict) -> np.ndarray:
    fluid = expected["fluid"]
    equilibrium = expected["analytic_equilibrium"]
    body_force = np.array(fluid["body_force"], dtype=float)
    reference = np.array(equilibrium["reference_point"], dtype=float)
    return fluid["density"] * np.einsum("ij,j->i", points - reference, body_force)


def wet_pressure_mask(phi: np.ndarray, margin: float) -> np.ndarray:
    wet = phi < -margin
    if not np.any(wet):
        raise RuntimeError("no wet points selected for pressure verification")
    return wet


def pressure_error(
    points: np.ndarray,
    phi: np.ndarray,
    pressure: np.ndarray,
    expected: dict,
    margin: float,
) -> tuple[float, float, int]:
    wet = wet_pressure_mask(phi, margin)
    exact = expected_pressure(points, expected)
    error = pressure[wet] - exact[wet]
    rms = float(np.sqrt(np.mean(error * error)))
    scale = max(1.0, float(np.sqrt(np.mean(exact[wet] * exact[wet]))))
    return rms, rms / scale, int(np.count_nonzero(wet))


def pressure_gradient_metrics(
    points: np.ndarray,
    phi: np.ndarray,
    pressure: np.ndarray,
    expected: dict,
    margin: float,
) -> dict[str, float]:
    wet = wet_pressure_mask(phi, margin)
    design = np.column_stack([points[wet, 0], points[wet, 1], np.ones(np.count_nonzero(wet))])
    a, b, c = np.linalg.lstsq(design, pressure[wet], rcond=None)[0]
    expected_gradient = expected["fluid"]["density"] * np.array(expected["fluid"]["body_force"][:2], dtype=float)
    gradient = np.array([a, b], dtype=float)
    gradient_error = gradient - expected_gradient
    residual = pressure[wet] - design @ np.array([a, b, c])
    return {
        "pressure_gradient_x": float(a),
        "pressure_gradient_y": float(b),
        "pressure_gradient_x_error": float(gradient_error[0]),
        "pressure_gradient_y_error": float(gradient_error[1]),
        "pressure_gradient_relative_error": float(
            np.linalg.norm(gradient_error) / max(1.0, np.linalg.norm(expected_gradient))
        ),
        "pressure_fit_intercept": float(c),
        "pressure_fit_rms_residual": float(np.sqrt(np.mean(residual * residual))),
    }


def point_pressure_subset_metrics(
    prefix: str,
    values: np.ndarray,
    sample_points: np.ndarray,
) -> dict[str, float]:
    finite = np.isfinite(values)
    metrics: dict[str, float] = {
        f"{prefix}_sample_count": int(values.size),
        f"{prefix}_finite_count": int(np.count_nonzero(finite)),
    }
    if not np.any(finite):
        metrics.update(
            {
                f"{prefix}_rms": math.nan,
                f"{prefix}_max_abs": math.nan,
                f"{prefix}_max_abs_x": math.nan,
                f"{prefix}_max_abs_y": math.nan,
                f"{prefix}_max_abs_value": math.nan,
                f"{prefix}_mean": math.nan,
                f"{prefix}_rms_after_mean_removal": math.nan,
                f"{prefix}_max_abs_after_mean_removal": math.nan,
            }
        )
        return metrics

    finite_values = values[finite]
    finite_points = sample_points[finite]
    value_mean = float(np.mean(finite_values))
    centered_values = finite_values - value_mean
    max_abs_index = int(np.argmax(np.abs(finite_values)))
    max_abs_point = finite_points[max_abs_index]
    metrics.update(
        {
            f"{prefix}_rms": float(np.sqrt(np.mean(finite_values * finite_values))),
            f"{prefix}_max_abs": float(np.max(np.abs(finite_values))),
            f"{prefix}_max_abs_x": float(max_abs_point[0]),
            f"{prefix}_max_abs_y": float(max_abs_point[1]),
            f"{prefix}_max_abs_value": float(finite_values[max_abs_index]),
            f"{prefix}_mean": value_mean,
            f"{prefix}_rms_after_mean_removal": float(
                np.sqrt(np.mean(centered_values * centered_values))
            ),
            f"{prefix}_max_abs_after_mean_removal": float(
                np.max(np.abs(centered_values))
            ),
        }
    )
    return metrics


def interface_pressure_metrics(
    points: np.ndarray,
    phi: np.ndarray,
    pressure: np.ndarray,
    crossings: list[InterfaceCrossing],
    cells: np.ndarray,
    expected: dict,
) -> dict[str, float]:
    external_pressure = float(expected["analytic_equilibrium"].get("external_pressure", 0.0))
    interface_pressure = interpolate_crossing_values(pressure, crossings) - external_pressure
    finite = np.isfinite(interface_pressure)
    nonfinite_endpoint_count = 0
    for crossing in crossings:
        a, b = crossing.edge
        if not (np.isfinite(pressure[a]) and np.isfinite(pressure[b])):
            nonfinite_endpoint_count += 1
    metrics = {
        "interface_pressure_interpolation_count": int(interface_pressure.size),
        "interface_pressure_nonfinite_endpoint_count": int(nonfinite_endpoint_count),
        **length_weighted_interface_pressure_metrics(points, phi, pressure, cells, expected),
    }
    if not np.any(finite):
        metrics.update(
            {
                "interface_pressure_finite_count": 0,
                "interface_pressure_rms": math.nan,
                "interface_pressure_max_abs": math.nan,
                "interface_pressure_max_abs_x": math.nan,
                "interface_pressure_max_abs_y": math.nan,
                "interface_pressure_max_abs_value": math.nan,
                "interface_pressure_mean": math.nan,
                "interface_pressure_min": math.nan,
                "interface_pressure_max": math.nan,
                "interface_pressure_rms_after_mean_removal": math.nan,
                "interface_pressure_max_abs_after_mean_removal": math.nan,
            }
        )
        return metrics
    finite_pressure = interface_pressure[finite]
    finite_points = crossing_points(crossings)[finite]
    pressure_mean = float(np.mean(finite_pressure))
    centered_pressure = finite_pressure - pressure_mean
    max_abs_index = int(np.argmax(np.abs(finite_pressure)))
    max_abs_point = finite_points[max_abs_index]
    metrics.update(
        {
            "interface_pressure_finite_count": int(np.count_nonzero(finite)),
            "interface_pressure_rms": float(np.sqrt(np.mean(finite_pressure * finite_pressure))),
            "interface_pressure_max_abs": float(np.max(np.abs(finite_pressure))),
            "interface_pressure_max_abs_x": float(max_abs_point[0]),
            "interface_pressure_max_abs_y": float(max_abs_point[1]),
            "interface_pressure_max_abs_value": float(finite_pressure[max_abs_index]),
            "interface_pressure_mean": pressure_mean,
            "interface_pressure_min": float(np.min(finite_pressure)),
            "interface_pressure_max": float(np.max(finite_pressure)),
            "interface_pressure_rms_after_mean_removal": float(
                np.sqrt(np.mean(centered_pressure * centered_pressure))
            ),
            "interface_pressure_max_abs_after_mean_removal": float(
                np.max(np.abs(centered_pressure))
            ),
        }
    )
    bounds = domain_bounds(points)
    tolerance = boundary_tolerance(points)
    near_boundary_width = interface_pressure_near_boundary_width(points, expected)
    boundary = np.array(
        [
            point_on_domain_boundary(point, bounds, tolerance)
            for point in finite_points
        ],
        dtype=bool,
    )
    boundary_distances = np.asarray(
        [distance_to_domain_boundary(point, bounds) for point in finite_points],
        dtype=float,
    )
    near_boundary = (~boundary) & (boundary_distances <= near_boundary_width + tolerance)
    core = (~boundary) & (~near_boundary)
    metrics["interface_pressure_near_boundary_width"] = near_boundary_width
    metrics.update(
        point_pressure_subset_metrics(
            "interface_pressure_boundary",
            finite_pressure[boundary],
            finite_points[boundary],
        )
    )
    metrics.update(
        point_pressure_subset_metrics(
            "interface_pressure_interior",
            finite_pressure[~boundary],
            finite_points[~boundary],
        )
    )
    metrics.update(
        point_pressure_subset_metrics(
            "interface_pressure_near_boundary",
            finite_pressure[near_boundary],
            finite_points[near_boundary],
        )
    )
    metrics.update(
        point_pressure_subset_metrics(
            "interface_pressure_core",
            finite_pressure[core],
            finite_points[core],
        )
    )
    return metrics


def length_weighted_interface_pressure_metrics(
    points: np.ndarray,
    phi: np.ndarray,
    pressure: np.ndarray,
    cells: np.ndarray,
    expected: dict,
) -> dict[str, float]:
    external_pressure = float(expected["analytic_equilibrium"].get("external_pressure", 0.0))
    segments = cell_interface_segments(points, phi, pressure, cells)
    finite_segments = [
        segment
        for segment in segments
        if np.isfinite(segment.pressure_a) and np.isfinite(segment.pressure_b)
    ]
    bounds = domain_bounds(points)
    tolerance = boundary_tolerance(points)
    near_boundary_width = interface_pressure_near_boundary_width(points, expected)
    boundary_segments = [
        segment
        for segment in finite_segments
        if point_on_domain_boundary(segment.point_a, bounds, tolerance)
        or point_on_domain_boundary(segment.point_b, bounds, tolerance)
    ]
    near_boundary_segments = [
        segment
        for segment in finite_segments
        if not (
            point_on_domain_boundary(segment.point_a, bounds, tolerance)
            or point_on_domain_boundary(segment.point_b, bounds, tolerance)
        )
        and min(
            distance_to_domain_boundary(segment.point_a, bounds),
            distance_to_domain_boundary(segment.point_b, bounds),
        )
        <= near_boundary_width + tolerance
    ]
    interior_segments = [
        segment
        for segment in finite_segments
        if not (
            point_on_domain_boundary(segment.point_a, bounds, tolerance)
            or point_on_domain_boundary(segment.point_b, bounds, tolerance)
        )
    ]
    core_segments = [
        segment
        for segment in finite_segments
        if not (
            point_on_domain_boundary(segment.point_a, bounds, tolerance)
            or point_on_domain_boundary(segment.point_b, bounds, tolerance)
        )
        and min(
            distance_to_domain_boundary(segment.point_a, bounds),
            distance_to_domain_boundary(segment.point_b, bounds),
        )
        > near_boundary_width + tolerance
    ]
    if not finite_segments:
        return {
            "interface_pressure_near_boundary_width": near_boundary_width,
            "interface_pressure_segment_count": len(segments),
            "interface_pressure_finite_segment_count": 0,
            "interface_pressure_total_segment_length": math.nan,
            "interface_pressure_length_weighted_rms": math.nan,
            "interface_pressure_length_weighted_mean": math.nan,
            "interface_pressure_length_weighted_rms_after_mean_removal": math.nan,
            "interface_pressure_length_weighted_max_abs": math.nan,
            "interface_pressure_length_weighted_max_abs_x": math.nan,
            "interface_pressure_length_weighted_max_abs_y": math.nan,
            "interface_pressure_length_weighted_max_abs_value": math.nan,
            "interface_pressure_length_weighted_max_abs_after_mean_removal": math.nan,
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_boundary",
                boundary_segments,
                external_pressure,
            ),
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_interior",
                interior_segments,
                external_pressure,
            ),
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_near_boundary",
                near_boundary_segments,
                external_pressure,
            ),
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_core",
                core_segments,
                external_pressure,
            ),
        }

    total_length = sum(segment.length for segment in finite_segments)
    if total_length <= ZERO_TOL:
        return {
            "interface_pressure_near_boundary_width": near_boundary_width,
            "interface_pressure_segment_count": len(segments),
            "interface_pressure_finite_segment_count": len(finite_segments),
            "interface_pressure_total_segment_length": total_length,
            "interface_pressure_length_weighted_rms": math.nan,
            "interface_pressure_length_weighted_mean": math.nan,
            "interface_pressure_length_weighted_rms_after_mean_removal": math.nan,
            "interface_pressure_length_weighted_max_abs": math.nan,
            "interface_pressure_length_weighted_max_abs_x": math.nan,
            "interface_pressure_length_weighted_max_abs_y": math.nan,
            "interface_pressure_length_weighted_max_abs_value": math.nan,
            "interface_pressure_length_weighted_max_abs_after_mean_removal": math.nan,
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_boundary",
                boundary_segments,
                external_pressure,
            ),
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_interior",
                interior_segments,
                external_pressure,
            ),
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_near_boundary",
                near_boundary_segments,
                external_pressure,
            ),
            **weighted_segment_pressure_subset_metrics(
                "interface_pressure_core",
                core_segments,
                external_pressure,
            ),
        }

    integral = 0.0
    squared_integral = 0.0
    endpoint_values: list[float] = []
    endpoint_points: list[np.ndarray] = []
    for segment in finite_segments:
        pa = segment.pressure_a - external_pressure
        pb = segment.pressure_b - external_pressure
        endpoint_values.extend([pa, pb])
        endpoint_points.extend([segment.point_a, segment.point_b])
        integral += segment.length * 0.5 * (pa + pb)
        squared_integral += segment.length * (pa * pa + pa * pb + pb * pb) / 3.0

    mean = integral / total_length
    centered_squared_integral = max(
        0.0,
        squared_integral - 2.0 * mean * integral + mean * mean * total_length,
    )
    endpoints = np.asarray(endpoint_values, dtype=float)
    endpoint_points_array = np.asarray(endpoint_points, dtype=float)
    centered_endpoints = endpoints - mean
    max_abs_index = int(np.argmax(np.abs(endpoints)))
    max_abs_point = endpoint_points_array[max_abs_index]
    return {
        "interface_pressure_near_boundary_width": near_boundary_width,
        "interface_pressure_segment_count": len(segments),
        "interface_pressure_finite_segment_count": len(finite_segments),
        "interface_pressure_total_segment_length": total_length,
        "interface_pressure_length_weighted_rms": float(math.sqrt(squared_integral / total_length)),
        "interface_pressure_length_weighted_mean": float(mean),
        "interface_pressure_length_weighted_rms_after_mean_removal": float(
            math.sqrt(centered_squared_integral / total_length)
        ),
        "interface_pressure_length_weighted_max_abs": float(np.max(np.abs(endpoints))),
        "interface_pressure_length_weighted_max_abs_x": float(max_abs_point[0]),
        "interface_pressure_length_weighted_max_abs_y": float(max_abs_point[1]),
        "interface_pressure_length_weighted_max_abs_value": float(endpoints[max_abs_index]),
        "interface_pressure_length_weighted_max_abs_after_mean_removal": float(
            np.max(np.abs(centered_endpoints))
        ),
        **weighted_segment_pressure_subset_metrics(
            "interface_pressure_boundary",
            boundary_segments,
            external_pressure,
        ),
        **weighted_segment_pressure_subset_metrics(
            "interface_pressure_interior",
            interior_segments,
            external_pressure,
        ),
        **weighted_segment_pressure_subset_metrics(
            "interface_pressure_near_boundary",
            near_boundary_segments,
            external_pressure,
        ),
        **weighted_segment_pressure_subset_metrics(
            "interface_pressure_core",
            core_segments,
            external_pressure,
        ),
    }


def weighted_segment_pressure_subset_metrics(
    prefix: str,
    segments: list[InterfaceSegment],
    external_pressure: float,
) -> dict[str, float]:
    total_length = sum(segment.length for segment in segments)
    if not segments or total_length <= ZERO_TOL:
        return {
            f"{prefix}_segment_count": len(segments),
            f"{prefix}_total_segment_length": total_length,
            f"{prefix}_length_weighted_rms": math.nan,
            f"{prefix}_length_weighted_mean": math.nan,
            f"{prefix}_length_weighted_rms_after_mean_removal": math.nan,
            f"{prefix}_length_weighted_max_abs": math.nan,
            f"{prefix}_length_weighted_max_abs_x": math.nan,
            f"{prefix}_length_weighted_max_abs_y": math.nan,
            f"{prefix}_length_weighted_max_abs_value": math.nan,
            f"{prefix}_length_weighted_max_abs_after_mean_removal": math.nan,
        }

    integral = 0.0
    squared_integral = 0.0
    endpoint_values: list[float] = []
    endpoint_points: list[np.ndarray] = []
    for segment in segments:
        pa = segment.pressure_a - external_pressure
        pb = segment.pressure_b - external_pressure
        endpoint_values.extend([pa, pb])
        endpoint_points.extend([segment.point_a, segment.point_b])
        integral += segment.length * 0.5 * (pa + pb)
        squared_integral += segment.length * (pa * pa + pa * pb + pb * pb) / 3.0

    mean = integral / total_length
    centered_squared_integral = max(
        0.0,
        squared_integral - 2.0 * mean * integral + mean * mean * total_length,
    )
    endpoints = np.asarray(endpoint_values, dtype=float)
    endpoint_points_array = np.asarray(endpoint_points, dtype=float)
    centered_endpoints = endpoints - mean
    max_abs_index = int(np.argmax(np.abs(endpoints)))
    max_abs_point = endpoint_points_array[max_abs_index]
    return {
        f"{prefix}_segment_count": len(segments),
        f"{prefix}_total_segment_length": total_length,
        f"{prefix}_length_weighted_rms": float(math.sqrt(squared_integral / total_length)),
        f"{prefix}_length_weighted_mean": float(mean),
        f"{prefix}_length_weighted_rms_after_mean_removal": float(
            math.sqrt(centered_squared_integral / total_length)
        ),
        f"{prefix}_length_weighted_max_abs": float(np.max(np.abs(endpoints))),
        f"{prefix}_length_weighted_max_abs_x": float(max_abs_point[0]),
        f"{prefix}_length_weighted_max_abs_y": float(max_abs_point[1]),
        f"{prefix}_length_weighted_max_abs_value": float(endpoints[max_abs_index]),
        f"{prefix}_length_weighted_max_abs_after_mean_removal": float(
            np.max(np.abs(centered_endpoints))
        ),
    }


def interface_stress_metrics(
    points: np.ndarray,
    cells: np.ndarray,
    phi: np.ndarray,
    pressure: np.ndarray,
    velocity: np.ndarray,
    crossings: list[InterfaceCrossing],
    expected: dict,
) -> dict[str, float]:
    edge_cells = edge_to_cells(cells)
    fluid = expected["fluid"]
    mu = float(fluid.get("dynamic_viscosity", fluid.get("viscosity", 0.0)))
    external_pressure = float(expected["analytic_equilibrium"].get("external_pressure", 0.0))
    normal_residuals: list[float] = []
    normal_viscous: list[float] = []
    tangential_tractions: list[float] = []

    for crossing in crossings:
        edge = tuple(sorted(crossing.edge))
        crossing_normal_residuals: list[float] = []
        crossing_normal_viscous: list[float] = []
        crossing_tangential: list[float] = []
        for cell_index in edge_cells.get(edge, []):
            cell = cells[cell_index]
            local = axis_aligned_quad_local_coordinates(points[cell, :2], crossing.point)
            if local is None:
                continue
            xi, eta, dx, dy = local
            grad_phi = bilinear_gradient(phi[cell], dx, dy, xi, eta)
            normal_norm = float(np.linalg.norm(grad_phi))
            if normal_norm <= ZERO_TOL:
                continue
            normal = grad_phi / normal_norm
            tangent = np.array([-normal[1], normal[0]], dtype=float)
            grad_u = np.vstack(
                [
                    bilinear_gradient(velocity[cell, component], dx, dy, xi, eta)
                    for component in range(2)
                ]
            )
            strain = 0.5 * (grad_u + grad_u.T)
            pressure_trace = bilinear_value(pressure[cell], xi, eta)
            viscous_normal = float(2.0 * mu * normal @ strain @ normal)
            normal_residual = float(-pressure_trace + viscous_normal + external_pressure)
            tangential = float(2.0 * mu * tangent @ strain @ normal)
            crossing_normal_residuals.append(normal_residual)
            crossing_normal_viscous.append(viscous_normal)
            crossing_tangential.append(tangential)
        if crossing_normal_residuals:
            normal_residuals.append(float(np.mean(crossing_normal_residuals)))
            normal_viscous.append(float(np.mean(crossing_normal_viscous)))
            tangential_tractions.append(float(np.mean(crossing_tangential)))

    if not normal_residuals:
        return {
            "interface_stress_sample_count": 0,
            "interface_normal_traction_residual_rms": math.nan,
            "interface_normal_traction_residual_max_abs": math.nan,
            "interface_viscous_normal_stress_rms": math.nan,
            "interface_viscous_normal_stress_max_abs": math.nan,
            "interface_tangential_traction_rms": math.nan,
            "interface_tangential_traction_max_abs": math.nan,
        }

    normal_array = np.asarray(normal_residuals, dtype=float)
    viscous_array = np.asarray(normal_viscous, dtype=float)
    tangential_array = np.asarray(tangential_tractions, dtype=float)
    return {
        "interface_stress_sample_count": int(normal_array.size),
        "interface_normal_traction_residual_rms": float(
            np.sqrt(np.mean(normal_array * normal_array))
        ),
        "interface_normal_traction_residual_max_abs": float(np.max(np.abs(normal_array))),
        "interface_viscous_normal_stress_rms": float(
            np.sqrt(np.mean(viscous_array * viscous_array))
        ),
        "interface_viscous_normal_stress_max_abs": float(np.max(np.abs(viscous_array))),
        "interface_tangential_traction_rms": float(
            np.sqrt(np.mean(tangential_array * tangential_array))
        ),
        "interface_tangential_traction_max_abs": float(np.max(np.abs(tangential_array))),
    }


def probe_pressure_metrics(grid: pv.UnstructuredGrid, expected: dict) -> dict[str, float]:
    probes = expected["verification"].get("probe_points", [])
    if not probes:
        return {}

    probe_points = np.array([probe["coordinates"] for probe in probes], dtype=float)
    sampled = pv.PolyData(probe_points).sample(grid, tolerance=1.0e-9)
    valid = np.asarray(sampled.point_data.get("vtkValidPointMask", np.ones(len(probes))), dtype=bool)
    pressure = np.asarray(sampled.point_data["Pressure"], dtype=float)

    metrics: dict[str, float] = {}
    for index, probe in enumerate(probes):
        name = str(probe["name"])
        if not valid[index]:
            metrics[f"probe_{name}_valid"] = 0.0
            metrics[f"probe_{name}_pressure"] = math.nan
            metrics[f"probe_{name}_pressure_error"] = math.nan
            metrics[f"probe_{name}_pressure_relative_error"] = math.nan
            continue
        expected_pressure_value = float(probe["expected_final_pressure"])
        error = float(pressure[index] - expected_pressure_value)
        metrics[f"probe_{name}_valid"] = 1.0
        metrics[f"probe_{name}_pressure"] = float(pressure[index])
        metrics[f"probe_{name}_pressure_error"] = error
        metrics[f"probe_{name}_pressure_relative_error"] = abs(error) / max(1.0, abs(expected_pressure_value))
    return metrics


def text_float(parent: ET.Element, tag: str) -> float:
    element = parent.find(tag)
    if element is None or element.text is None:
        raise RuntimeError(f"solver.xml is missing <{tag}>")
    return float(element.text.strip())


def text_int(parent: ET.Element, tag: str) -> int:
    element = parent.find(tag)
    if element is None or element.text is None:
        raise RuntimeError(f"solver.xml is missing <{tag}>")
    return int(element.text.strip())


def text_bool(parent: ET.Element, tag: str) -> bool:
    element = parent.find(tag)
    if element is None or element.text is None:
        raise RuntimeError(f"solver.xml is missing <{tag}>")
    return element.text.strip().lower() in {"true", "1", "yes"}


def solver_consistency(case_dir: Path, expected: dict) -> tuple[dict[str, float | int | bool], list[tuple[str, bool]]]:
    root = ET.parse(case_dir / "solver.xml").getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise RuntimeError("solver.xml is missing <GeneralSimulationParameters>")
    fluid = next((eq for eq in root.findall("Add_equation") if eq.attrib.get("type") == "fluid"), None)
    if fluid is None:
        raise RuntimeError("solver.xml is missing <Add_equation type=\"fluid\">")
    viscosity = fluid.find("Viscosity")
    if viscosity is None:
        raise RuntimeError("solver.xml is missing fluid <Viscosity>")

    body_force = expected["fluid"]["body_force"]
    metrics: dict[str, float | int | bool] = {
        "solver_time_step": text_float(general, "Time_step_size"),
        "solver_time_steps": text_int(general, "Number_of_time_steps"),
        "solver_save_increment": text_int(general, "Increment_in_saving_VTK_files"),
        "solver_combine_time_series": text_bool(general, "Combine_time_series"),
        "solver_density": text_float(fluid, "Density"),
        "solver_viscosity": text_float(viscosity, "Value"),
        "solver_force_x": text_float(fluid, "Force_x"),
        "solver_force_y": text_float(fluid, "Force_y"),
        "solver_force_z": text_float(fluid, "Force_z"),
    }
    run = expected["run"]
    checks = [
        ("solver_time_step_consistency", abs(metrics["solver_time_step"] - run["time_step"]) <= 1.0e-12),
        ("solver_time_steps_consistency", metrics["solver_time_steps"] == run["time_steps"]),
        ("solver_save_every_step", metrics["solver_save_increment"] == 1),
        ("solver_combine_time_series", bool(metrics["solver_combine_time_series"])),
        ("solver_density_consistency", abs(metrics["solver_density"] - expected["fluid"]["density"]) <= 1.0e-10),
        (
            "solver_viscosity_consistency",
            abs(metrics["solver_viscosity"] - expected["fluid"]["dynamic_viscosity"]) <= 1.0e-12,
        ),
        ("solver_force_x_consistency", abs(metrics["solver_force_x"] - body_force[0]) <= 1.0e-8),
        ("solver_force_y_consistency", abs(metrics["solver_force_y"] - body_force[1]) <= 1.0e-8),
        ("solver_force_z_consistency", abs(metrics["solver_force_z"] - body_force[2]) <= 1.0e-8),
    ]
    return metrics, checks


def analytic_phi(points: np.ndarray, expected: dict) -> np.ndarray:
    line = expected["analytic_equilibrium"]["free_surface_line"]
    return points[:, 1] - (line["intercept"] + line["slope"] * points[:, 0])


def analytic_self_test(case_dir: Path, expected: dict) -> dict[str, float]:
    grid = pv.read(case_dir / "mesh/background/mesh-complete.mesh.vtu")
    cells = quad_cells(grid)
    phi = analytic_phi(grid.points, expected)
    area, centroid = clipped_area_and_centroid(grid.points, phi, cells)
    centroid_expected = expected["analytic_equilibrium"]["centroid"]
    centroid_target = np.array([centroid_expected["x"], centroid_expected["y"]], dtype=float)
    return {
        "analytic_self_test_area": area,
        "analytic_self_test_area_error": abs(area - expected["analytic_equilibrium"]["area"]),
        "analytic_self_test_centroid_x": float(centroid[0]),
        "analytic_self_test_centroid_y": float(centroid[1]),
        "analytic_self_test_centroid_error": float(np.linalg.norm(centroid - centroid_target)),
    }


def add_probe_checks(metrics: dict, tolerances: dict) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = []
    probe_abs = tolerances["probe_pressure_abs"]
    probe_rel = tolerances["probe_pressure_relative"]
    for key, value in metrics.items():
        if not key.endswith("_pressure_error") or not key.startswith("probe_"):
            continue
        prefix = key.removesuffix("_pressure_error")
        rel_key = f"{prefix}_pressure_relative_error"
        valid_key = f"{prefix}_valid"
        ok = (
            metrics.get(valid_key, 0.0) == 1.0
            and (abs(value) <= probe_abs or metrics[rel_key] <= probe_rel)
        )
        checks.append((f"{prefix}_pressure", bool(ok)))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, help="result_*.vtu or result_*.pvtu file to check")
    parser.add_argument("--case-dir", type=Path, default=CASE_DIR)
    parser.add_argument("--self-test", action="store_true", help="check analytic phi_eq area and centroid on the mesh")
    args = parser.parse_args()

    expected = load_expected(args.case_dir)
    tolerances = expected["verification"]["suggested_tolerances"]

    if args.self_test:
        metrics = analytic_self_test(args.case_dir, expected)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        failed = [
            name
            for name, ok in [
                ("analytic_self_test_area", metrics["analytic_self_test_area_error"] <= 1.0e-12),
                ("analytic_self_test_centroid", metrics["analytic_self_test_centroid_error"] <= 1.0e-12),
            ]
            if not ok
        ]
        if failed:
            raise SystemExit(f"failed checks: {', '.join(failed)}")
        return

    result_path = resolve_result_path(args.result, args.case_dir)
    grid = pv.read(result_path)

    required_arrays = {"phi", "Velocity", "Pressure"}
    missing = required_arrays - set(grid.point_data.keys())
    if missing:
        raise RuntimeError(f"{result_path} is missing point arrays: {sorted(missing)}")

    points = grid.points
    phi = np.asarray(grid.point_data["phi"], dtype=float)
    velocity = np.asarray(grid.point_data["Velocity"], dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float)
    cells = quad_cells(grid)
    wet_margin = float(expected["verification"].get("wet_pressure_margin", 5.0e-2))

    crossings = edge_zero_crossings(points, phi, cells)
    points_on_interface = crossing_points(crossings)
    slope, intercept = fit_interface_line(crossings)
    line_rms, line_max = line_residual_metrics(points_on_interface, slope, intercept)
    expected_line = expected["analytic_equilibrium"]["free_surface_line"]
    expected_line_rms, expected_line_max = line_residual_metrics(
        points_on_interface,
        expected_line["slope"],
        expected_line["intercept"],
    )
    slope_error = abs(slope - expected_line["slope"])
    require_final_equilibrium = final_equilibrium_required(expected)
    require_slope_progress = verification_flag(
        expected,
        "interface_slope_progress_required",
        not require_final_equilibrium,
    )
    require_intercept = verification_flag(expected, "interface_intercept_required", True)
    require_pressure_gradient = verification_flag(expected, "pressure_gradient_required", True)
    pressure_scope = interface_pressure_check_scope(expected)
    require_boundary_pressure_guard = verification_flag(
        expected,
        "interface_pressure_boundary_guard_required",
        False,
    )
    require_near_boundary_pressure_guard = verification_flag(
        expected,
        "interface_pressure_near_boundary_guard_required",
        False,
    )
    slope_progress = slope_progress_fraction(slope, expected)
    intercept_error = abs(intercept - expected_line["intercept"])

    area, centroid = clipped_area_and_centroid(points, phi, cells)
    centroid_expected = expected["analytic_equilibrium"]["centroid"]
    centroid_error = float(
        np.linalg.norm(
            centroid
            - np.array([centroid_expected["x"], centroid_expected["y"]], dtype=float)
        )
    )
    area_error = abs(area - expected["analytic_equilibrium"]["area"])

    wet = phi <= 0.0
    velocity_norm = np.linalg.norm(velocity[:, :2], axis=1)
    velocity_max = float(np.max(velocity_norm[wet])) if np.any(wet) else math.nan
    active_fluid = np.asarray(grid.point_data.get("ActiveFluid", np.full_like(phi, np.nan)), dtype=float)
    active_output_present = "ActiveFluid" in grid.point_data
    pressure_rms, pressure_relative, wet_pressure_points = pressure_error(
        points,
        phi,
        pressure,
        expected,
        wet_margin,
    )

    metrics: dict[str, float | int | str | bool] = {
        "result": str(result_path),
        "interface_crossing_count": len(crossings),
        "interface_slope": slope,
        "interface_slope_error": slope_error,
        "interface_slope_progress_fraction": slope_progress,
        "interface_slope_progress_required": require_slope_progress,
        "final_equilibrium_required": require_final_equilibrium,
        "verification_profile": str(expected.get("verification", {}).get("profile", "legacy")),
        "interface_pressure_check_scope": pressure_scope,
        "interface_pressure_boundary_guard_required": require_boundary_pressure_guard,
        "interface_pressure_near_boundary_guard_required": require_near_boundary_pressure_guard,
        "interface_intercept": intercept,
        "interface_intercept_error": intercept_error,
        "interface_intercept_required": require_intercept,
        "interface_line_rms_residual": line_rms,
        "interface_line_max_abs_residual": line_max,
        "interface_expected_line_rms_residual": expected_line_rms,
        "interface_expected_line_max_abs_residual": expected_line_max,
        "area": area,
        "area_error": area_error,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_error": centroid_error,
        "velocity_max": velocity_max,
        "velocity_max_all_points": float(np.nanmax(velocity_norm)),
        "active_fluid_output_present": bool(active_output_present),
        "active_fluid_dry_vertex_count": int(np.count_nonzero(active_fluid <= 0.0)) if active_output_present else 0,
        "active_fluid_wet_vertex_count": int(np.count_nonzero(active_fluid > 0.0)) if active_output_present else 0,
        "pressure_nonfinite_count": int(np.count_nonzero(~np.isfinite(pressure))),
        "pressure_wet_nonfinite_count": int(np.count_nonzero(~np.isfinite(pressure[wet]))),
        "velocity_nonfinite_count": int(np.count_nonzero(~np.isfinite(velocity))),
        "velocity_wet_nonfinite_count": int(
            np.count_nonzero(np.any(~np.isfinite(velocity[wet]), axis=1))
        ),
        "pressure_wet_point_count": wet_pressure_points,
        "pressure_rms_error": pressure_rms,
        "pressure_rms_relative_error": pressure_relative,
        "pressure_gradient_required": require_pressure_gradient,
    }
    metrics.update(pressure_gradient_metrics(points, phi, pressure, expected, wet_margin))
    metrics.update(interface_pressure_metrics(points, phi, pressure, crossings, cells, expected))
    metrics.update(interface_stress_metrics(points, cells, phi, pressure, velocity, crossings, expected))
    metrics.update(probe_pressure_metrics(grid, expected))
    solver_metrics, solver_checks = solver_consistency(args.case_dir, expected)
    metrics.update(solver_metrics)

    pressure_metric_contract = {
        "all": {
            "rms": "interface_pressure_rms",
            "max": "interface_pressure_max_abs",
            "sample_count": "interface_pressure_interpolation_count",
            "rms_check": "interface_pressure_rms_abs",
            "max_check": "interface_pressure_max_abs",
        },
        "interior": {
            "rms": "interface_pressure_interior_rms",
            "max": "interface_pressure_interior_max_abs",
            "sample_count": "interface_pressure_interior_sample_count",
            "rms_check": "interface_pressure_interior_rms_abs",
            "max_check": "interface_pressure_interior_max_abs",
        },
        "core": {
            "rms": "interface_pressure_core_rms",
            "max": "interface_pressure_core_max_abs",
            "sample_count": "interface_pressure_core_sample_count",
            "rms_check": "interface_pressure_core_rms_abs",
            "max_check": "interface_pressure_core_max_abs",
        },
    }[pressure_scope]
    pressure_rms_metric = pressure_metric_contract["rms"]
    pressure_max_metric = pressure_metric_contract["max"]
    pressure_sample_count_metric = pressure_metric_contract["sample_count"]
    pressure_rms_check = pressure_metric_contract["rms_check"]
    pressure_max_check = pressure_metric_contract["max_check"]

    checks = [
        *solver_checks,
        (
            "interface_slope_abs",
            (not require_final_equilibrium) or slope_error <= tolerances["interface_slope_abs"],
        ),
        (
            "interface_slope_progress_min",
            (not require_slope_progress)
            or require_final_equilibrium
            or slope_progress >= tolerances.get("interface_slope_progress_min", 0.5),
        ),
        ("interface_intercept_abs", (not require_intercept) or intercept_error <= tolerances["interface_intercept_abs"]),
        (
            "interface_line_rms_residual_abs",
            line_rms <= tolerances["interface_line_rms_residual_abs"],
        ),
        (
            "interface_line_max_abs_residual_abs",
            line_max <= tolerances["interface_line_max_abs_residual_abs"],
        ),
        ("area_abs", area_error <= tolerances["area_abs"]),
        ("centroid_abs", centroid_error <= tolerances["centroid_abs"]),
        ("velocity_max", velocity_max <= tolerances["velocity_max"]),
        ("active_fluid_output_present", bool(metrics["active_fluid_output_present"])),
        ("pressure_wet_finite", metrics["pressure_wet_nonfinite_count"] == 0),
        ("velocity_wet_finite", metrics["velocity_wet_nonfinite_count"] == 0),
        (
            "interface_pressure_all_samples_finite",
            metrics["interface_pressure_finite_count"] == metrics["interface_pressure_interpolation_count"],
        ),
        (
            "interface_pressure_all_endpoints_finite",
            metrics["interface_pressure_nonfinite_endpoint_count"] == 0,
        ),
        (
            f"{pressure_scope}_interface_pressure_samples_present",
            metrics[pressure_sample_count_metric] > 0,
        ),
        (
            pressure_rms_check,
            math.isfinite(metrics[pressure_rms_metric])
            and metrics[pressure_rms_metric] <= tolerances["interface_pressure_rms_abs"],
        ),
        (
            pressure_max_check,
            math.isfinite(metrics[pressure_max_metric])
            and metrics[pressure_max_metric] <= tolerances["interface_pressure_max_abs"],
        ),
        (
            "interface_pressure_boundary_max_abs_guard",
            (not require_boundary_pressure_guard)
            or (
                metrics["interface_pressure_boundary_sample_count"] > 0
                and math.isfinite(metrics["interface_pressure_boundary_max_abs"])
                and metrics["interface_pressure_boundary_max_abs"]
                <= tolerances["interface_pressure_boundary_max_abs_guard"]
            ),
        ),
        (
            "interface_pressure_near_boundary_max_abs_guard",
            (not require_near_boundary_pressure_guard)
            or (
                metrics["interface_pressure_near_boundary_sample_count"] > 0
                and math.isfinite(metrics["interface_pressure_near_boundary_max_abs"])
                and metrics["interface_pressure_near_boundary_max_abs"]
                <= tolerances["interface_pressure_near_boundary_max_abs_guard"]
            ),
        ),
        ("pressure_rms_relative", pressure_relative <= tolerances["pressure_rms_relative"]),
        (
            "pressure_gradient_x_abs",
            (not require_pressure_gradient)
            or abs(metrics["pressure_gradient_x_error"]) <= tolerances["pressure_gradient_abs"],
        ),
        (
            "pressure_gradient_y_abs",
            (not require_pressure_gradient)
            or abs(metrics["pressure_gradient_y_error"]) <= tolerances["pressure_gradient_abs"],
        ),
        (
            "pressure_gradient_relative",
            (not require_pressure_gradient)
            or metrics["pressure_gradient_relative_error"] <= tolerances["pressure_gradient_relative"],
        ),
        *add_probe_checks(metrics, tolerances),
    ]
    failed = [name for name, ok in checks if not ok]

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
