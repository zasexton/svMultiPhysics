#!/usr/bin/env python3
"""Verify the 2D traveling-interface MMS free-surface case."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


CASE_DIR = Path(__file__).resolve().parent
ZERO_TOL = 1.0e-12
DEDUP_TOL = 1.0e-10
OMEGA_TOL = 1.0e-14


@dataclass(frozen=True)
class InterfaceCrossing:
    point: np.ndarray
    edge: tuple[int, ...]
    t: float


def latest_result(case_dir: Path) -> Path:
    def step(path: Path) -> int:
        match = re.search(r"_(\d+)\.p?vtu$", path.name)
        return int(match.group(1)) if match else -1

    candidates = sorted([*case_dir.glob("result_*.vtu"), *case_dir.glob("result_*.pvtu")], key=step)
    if not candidates:
        raise FileNotFoundError(f"no result_*.vtu or result_*.pvtu files found in {case_dir}")
    return candidates[-1]


def resolve_result_path(result: Path | None) -> Path:
    if result is None:
        return latest_result(CASE_DIR)
    if result.exists() or result.is_absolute():
        return result
    candidate = CASE_DIR / result
    return candidate if candidate.exists() else result


def load_expected() -> dict:
    with (CASE_DIR / "expected_results.json").open() as stream:
        return json.load(stream)


def infer_time(path: Path, expected: dict, explicit_time: float | None) -> float:
    if explicit_time is not None:
        return explicit_time
    if path.name == "mesh-complete.mesh.vtu":
        return 0.0
    match = re.search(r"_(\d+)\.p?vtu$", path.name)
    if match:
        return int(match.group(1)) * float(expected["run"]["time_step"])
    return float(expected["run"]["final_time"])


def quad_cells(grid: pv.UnstructuredGrid) -> list[np.ndarray]:
    cells: list[np.ndarray] = []
    offset = 0
    raw = grid.cells
    while offset < raw.size:
        node_count = int(raw[offset])
        conn = raw[offset + 1: offset + 1 + node_count].astype(np.int64)
        if node_count not in (4, 8, 9):
            raise RuntimeError("expected a pure quadrilateral VTK mesh")
        cells.append(conn)
        offset += node_count + 1
    return cells


def cell_corners(cell: np.ndarray) -> np.ndarray:
    return cell[:4]


def cell_edges(cell: np.ndarray) -> list[tuple[int, ...]]:
    if cell.size >= 8:
        return [
            (int(cell[0]), int(cell[4]), int(cell[1])),
            (int(cell[1]), int(cell[5]), int(cell[2])),
            (int(cell[2]), int(cell[6]), int(cell[3])),
            (int(cell[3]), int(cell[7]), int(cell[0])),
        ]
    return [
        (int(cell[0]), int(cell[1])),
        (int(cell[1]), int(cell[2])),
        (int(cell[2]), int(cell[3])),
        (int(cell[3]), int(cell[0])),
    ]


def unique_edges(cells: list[np.ndarray]) -> list[tuple[int, ...]]:
    edges: list[tuple[int, ...]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        for edge_nodes in cell_edges(cell):
            key = tuple(sorted((edge_nodes[0], edge_nodes[-1])))
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge_nodes)
    return edges


def quadratic_root_parameters(f0: float, fm: float, f1: float) -> list[float]:
    if max(abs(f0), abs(fm), abs(f1)) <= ZERO_TOL:
        return [0.0, 1.0]
    a = 0.5 * (f0 + f1) - fm
    b = 0.5 * (f1 - f0)
    c = fm
    if abs(a) <= ZERO_TOL:
        if abs(b) <= ZERO_TOL:
            return []
        s = -c / b
        return [0.5 * (s + 1.0)] if -1.0 <= s <= 1.0 else []
    disc = b * b - 4.0 * a * c
    if disc < -ZERO_TOL:
        return []
    disc = max(0.0, disc)
    roots = []
    for s in ((-b - math.sqrt(disc)) / (2.0 * a), (-b + math.sqrt(disc)) / (2.0 * a)):
        if -1.0 - ZERO_TOL <= s <= 1.0 + ZERO_TOL:
            roots.append(min(1.0, max(0.0, 0.5 * (s + 1.0))))
    return roots


def edge_point(points: np.ndarray, edge_nodes: tuple[int, ...], t: float) -> np.ndarray:
    if len(edge_nodes) == 2:
        a, b = edge_nodes
        return (1.0 - t) * points[a, :2] + t * points[b, :2]
    a, m, b = edge_nodes
    s = 2.0 * t - 1.0
    return (
        0.5 * s * (s - 1.0) * points[a, :2] +
        (1.0 - s * s) * points[m, :2] +
        0.5 * s * (s + 1.0) * points[b, :2]
    )


def edge_value(values: np.ndarray, edge_nodes: tuple[int, ...], t: float) -> np.ndarray:
    if len(edge_nodes) == 2:
        a, b = edge_nodes
        return (1.0 - t) * values[a] + t * values[b]
    a, m, b = edge_nodes
    s = 2.0 * t - 1.0
    return (
        0.5 * s * (s - 1.0) * values[a] +
        (1.0 - s * s) * values[m] +
        0.5 * s * (s + 1.0) * values[b]
    )


def edge_root_parameters(phi: np.ndarray, edge_nodes: tuple[int, ...]) -> list[float]:
    if len(edge_nodes) == 3:
        f0 = float(phi[edge_nodes[0]])
        fm = float(phi[edge_nodes[1]])
        f1 = float(phi[edge_nodes[2]])
        return quadratic_root_parameters(f0, fm, f1)

    a, b = edge_nodes
    pa = float(phi[a])
    pb = float(phi[b])
    a_zero = abs(pa) <= ZERO_TOL
    b_zero = abs(pb) <= ZERO_TOL
    if a_zero and b_zero:
        return [0.0, 1.0]
    if a_zero:
        return [0.0]
    if b_zero:
        return [1.0]
    if pa * pb < 0.0:
        return [pa / (pa - pb)]
    return []


def edge_zero_crossings(points: np.ndarray, phi: np.ndarray, cells: list[np.ndarray]) -> list[InterfaceCrossing]:
    crossings: list[InterfaceCrossing] = []
    keys: set[tuple[int, int]] = set()

    def add(edge_nodes: tuple[int, ...], t: float) -> None:
        p = edge_point(points, edge_nodes, t)
        key = tuple(np.round(p / DEDUP_TOL).astype(np.int64))
        if key in keys:
            return
        keys.add(key)
        crossings.append(InterfaceCrossing(point=p, edge=edge_nodes, t=float(t)))

    for edge_nodes in unique_edges(cells):
        for t in edge_root_parameters(phi, edge_nodes):
            add(edge_nodes, t)
    if not crossings:
        raise RuntimeError("no phi=0 crossings found")
    return crossings


def crossing_points(crossings: list[InterfaceCrossing]) -> np.ndarray:
    return np.array([c.point for c in crossings], dtype=float)


def interpolate_crossing_values(values: np.ndarray, crossings: list[InterfaceCrossing]) -> np.ndarray:
    out = []
    for crossing in crossings:
        out.append(edge_value(values, crossing.edge, crossing.t))
    return np.asarray(out, dtype=float)


def nonfinite_crossing_endpoint_count(values: np.ndarray, crossings: list[InterfaceCrossing]) -> int:
    count = 0
    for crossing in crossings:
        if not all(np.all(np.isfinite(values[node])) for node in crossing.edge):
            count += 1
    return count


def fit_mode(points: np.ndarray, k: float) -> tuple[float, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack([np.ones_like(x), np.cos(k * x), np.sin(k * x)])
    c0, c_cos, c_sin = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(c0), float(c_cos), float(c_sin)


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


def clipped_area_and_centroid(points: np.ndarray, phi: np.ndarray, cells: list[np.ndarray]) -> tuple[float, np.ndarray]:
    total_area = 0.0
    first_moment = np.zeros(2)
    for cell in cells:
        corners = cell_corners(cell)
        vertices = [points[index, :2] for index in corners]
        values = [float(phi[index]) for index in corners]
        clipped = clip_polygon_negative(vertices, values)
        area = polygon_area(clipped)
        centroid = polygon_centroid(clipped)
        total_area += area
        first_moment += area * centroid
    if total_area == 0.0:
        return 0.0, np.zeros(2)
    return total_area, first_moment / total_area


def sol_params(expected: dict) -> tuple[dict, dict]:
    return expected["analytic_solution"], expected["fluid"]


def constant_translation_motion(expected: dict) -> bool:
    sol, _ = sol_params(expected)
    return (
        sol.get("motion_model") == "constant_translation"
        or abs(float(sol["Omega"])) <= OMEGA_TOL
    )


def shift(t: float, expected: dict) -> float:
    sol, _ = sol_params(expected)
    if constant_translation_motion(expected):
        return sol["U0"] * t
    return (sol["U0"] / sol["Omega"]) * math.sin(sol["Omega"] * t)


def uniform_velocity(t: float, expected: dict) -> float:
    sol, _ = sol_params(expected)
    if constant_translation_motion(expected):
        return sol["U0"]
    return sol["U0"] * math.cos(sol["Omega"] * t)


def uniform_acceleration(t: float, expected: dict) -> float:
    sol, _ = sol_params(expected)
    if constant_translation_motion(expected):
        return 0.0
    return -sol["U0"] * sol["Omega"] * math.sin(sol["Omega"] * t)


def exact_height(x: np.ndarray, t: float, expected: dict) -> np.ndarray:
    sol, _ = sol_params(expected)
    return sol["H0"] + sol["amplitude"] * np.cos(sol["k"] * (x - shift(t, expected)))


def exact_height_x(x: np.ndarray, t: float, expected: dict) -> np.ndarray:
    sol, _ = sol_params(expected)
    return -sol["amplitude"] * sol["k"] * np.sin(sol["k"] * (x - shift(t, expected)))


def exact_phi(points: np.ndarray, t: float, expected: dict) -> np.ndarray:
    return points[:, 1] - exact_height(points[:, 0], t, expected)


def exact_velocity(points: np.ndarray, t: float, expected: dict) -> np.ndarray:
    out = np.zeros((points.shape[0], 3), dtype=float)
    out[:, 0] = uniform_velocity(t, expected)
    return out


def exact_pressure(points: np.ndarray, t: float, expected: dict) -> np.ndarray:
    _, fluid = sol_params(expected)
    return fluid["density"] * fluid["gravity"] * (exact_height(points[:, 0], t, expected) - points[:, 1])


def source_x(points: np.ndarray, t: float, expected: dict) -> np.ndarray:
    _, fluid = sol_params(expected)
    return uniform_acceleration(t, expected) + fluid["gravity"] * exact_height_x(points[:, 0], t, expected)


def wrap_error(delta: float, period: float) -> float:
    return (delta + 0.5 * period) % period - 0.5 * period


def interface_phase_defined(sol: dict[str, float], *, floor: float = 1.0e-12) -> bool:
    return abs(sol["amplitude"]) > floor


def rms(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def quad_shape_functions(node_count: int, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if node_count == 4:
        n = 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ],
            dtype=float,
        )
        dxi = 0.25 * np.array(
            [
                -(1.0 - eta),
                1.0 - eta,
                1.0 + eta,
                -(1.0 + eta),
            ],
            dtype=float,
        )
        deta = 0.25 * np.array(
            [
                -(1.0 - xi),
                -(1.0 + xi),
                1.0 + xi,
                1.0 - xi,
            ],
            dtype=float,
        )
        return n, dxi, deta

    if node_count == 8:
        n = np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta) * (-xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 - eta) * (xi - eta - 1.0),
                0.25 * (1.0 + xi) * (1.0 + eta) * (xi + eta - 1.0),
                0.25 * (1.0 - xi) * (1.0 + eta) * (-xi + eta - 1.0),
                0.5 * (1.0 - xi * xi) * (1.0 - eta),
                0.5 * (1.0 + xi) * (1.0 - eta * eta),
                0.5 * (1.0 - xi * xi) * (1.0 + eta),
                0.5 * (1.0 - xi) * (1.0 - eta * eta),
            ],
            dtype=float,
        )
        dxi = np.array(
            [
                0.25 * (1.0 - eta) * (2.0 * xi + eta),
                0.25 * (1.0 - eta) * (2.0 * xi - eta),
                0.25 * (1.0 + eta) * (2.0 * xi + eta),
                0.25 * (1.0 + eta) * (2.0 * xi - eta),
                -xi * (1.0 - eta),
                0.5 * (1.0 - eta * eta),
                -xi * (1.0 + eta),
                -0.5 * (1.0 - eta * eta),
            ],
            dtype=float,
        )
        deta = np.array(
            [
                0.25 * (1.0 - xi) * (xi + 2.0 * eta),
                0.25 * (1.0 + xi) * (-xi + 2.0 * eta),
                0.25 * (1.0 + xi) * (xi + 2.0 * eta),
                0.25 * (1.0 - xi) * (-xi + 2.0 * eta),
                -0.5 * (1.0 - xi * xi),
                -(1.0 + xi) * eta,
                0.5 * (1.0 - xi * xi),
                -(1.0 - xi) * eta,
            ],
            dtype=float,
        )
        return n, dxi, deta

    if node_count == 9:
        lx = np.array(
            [
                0.5 * xi * (xi - 1.0),
                1.0 - xi * xi,
                0.5 * xi * (xi + 1.0),
            ],
            dtype=float,
        )
        ly = np.array(
            [
                0.5 * eta * (eta - 1.0),
                1.0 - eta * eta,
                0.5 * eta * (eta + 1.0),
            ],
            dtype=float,
        )
        dlx = np.array([xi - 0.5, -2.0 * xi, xi + 0.5], dtype=float)
        dly = np.array([eta - 0.5, -2.0 * eta, eta + 0.5], dtype=float)
        order = (
            (0, 0),
            (2, 0),
            (2, 2),
            (0, 2),
            (1, 0),
            (2, 1),
            (1, 2),
            (0, 1),
            (1, 1),
        )
        n = np.array([lx[i] * ly[j] for i, j in order], dtype=float)
        dxi = np.array([dlx[i] * ly[j] for i, j in order], dtype=float)
        deta = np.array([lx[i] * dly[j] for i, j in order], dtype=float)
        return n, dxi, deta

    raise RuntimeError(f"unsupported quadrilateral cell with {node_count} nodes")


def residual_audits(points: np.ndarray, t: float, expected: dict) -> dict[str, float]:
    _, fluid = sol_params(expected)
    hx = exact_height_x(points[:, 0], t, expected)
    sx = source_x(points, t, expected)
    rx = uniform_acceleration(t, expected) + fluid["gravity"] * hx - sx
    ry = -fluid["gravity"] - fluid["body_force"][1]

    sol, _ = sol_params(expected)
    u = uniform_velocity(t, expected)
    phi_t = -sol["amplitude"] * sol["k"] * u * np.sin(sol["k"] * (points[:, 0] - shift(t, expected)))
    phi_x = sol["amplitude"] * sol["k"] * np.sin(sol["k"] * (points[:, 0] - shift(t, expected)))
    rphi = phi_t + u * phi_x
    return {
        "manufactured_residual_x_max": float(np.max(np.abs(rx))),
        "manufactured_residual_y_max": float(abs(ry)),
        "level_set_residual_max": float(np.max(np.abs(rphi))),
    }


def field_norm_metrics(
    prefix: str,
    mask: np.ndarray,
    *,
    velocity: np.ndarray,
    pressure: np.ndarray,
    velocity_exact_values: np.ndarray,
    pressure_exact_values: np.ndarray,
    expected_velocity_x: float,
    velocity_scale_floor: float,
    pressure_scale_floor: float,
) -> dict[str, object]:
    if not np.any(mask):
        raise RuntimeError(f"no finite nodes available for {prefix or 'field'} verification")

    vel_error = velocity[mask, :2] - velocity_exact_values[mask, :2]
    vel_error_norm = np.linalg.norm(vel_error, axis=1)
    vel_exact_norm = np.linalg.norm(velocity_exact_values[mask, :2], axis=1)
    velocity_scale = max(rms(vel_exact_norm), velocity_scale_floor, 1.0e-14)

    p_error = pressure[mask] - pressure_exact_values[mask]
    pressure_offset = float(np.mean(p_error))
    p_error_shifted = p_error - pressure_offset
    pressure_scale = max(rms(pressure_exact_values[mask]), pressure_scale_floor, 1.0)

    stem = f"{prefix}_" if prefix else ""
    return {
        f"{stem}wet_node_count": int(np.count_nonzero(mask)),
        f"{stem}velocity_l2_error": rms(vel_error_norm),
        f"{stem}velocity_rms_error": rms(vel_error_norm),
        f"{stem}velocity_relative_l2_error": rms(vel_error_norm) / velocity_scale,
        f"{stem}velocity_max_abs_error": float(np.max(np.abs(vel_error))),
        f"{stem}velocity_mean_x": float(np.mean(velocity[mask, 0])),
        f"{stem}velocity_mean_y": float(np.mean(velocity[mask, 1])),
        f"{stem}velocity_mean_x_error": float(np.mean(velocity[mask, 0]) - expected_velocity_x),
        f"{stem}velocity_mean_y_error": float(np.mean(velocity[mask, 1])),
        f"{stem}pressure_rms_error": rms(p_error),
        f"{stem}pressure_relative_rms_error": rms(p_error) / pressure_scale,
        f"{stem}pressure_max_abs_error": float(np.max(np.abs(p_error))),
        f"{stem}pressure_rms_error_after_constant_offset_removal": rms(p_error_shifted),
        f"{stem}pressure_relative_rms_error_after_constant_offset_removal": (
            rms(p_error_shifted) / pressure_scale
        ),
        f"{stem}pressure_constant_offset": pressure_offset,
    }


def weighted_rms(error_sq: float, measure: float) -> float:
    if measure <= 0.0:
        return float("nan")
    return math.sqrt(max(0.0, error_sq / measure))


def finite_scale(*values: float) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return max(finite_values) if finite_values else float("nan")


def scalar_l2_metrics(
    prefix: str,
    values: np.ndarray,
    exact_values: np.ndarray,
    mask: np.ndarray,
    *,
    scale_floor: float,
) -> dict[str, object]:
    stem = f"{prefix}_" if prefix else ""
    finite_mask = mask & np.isfinite(values) & np.isfinite(exact_values)
    if not np.any(finite_mask):
        return {
            f"{stem}node_count": 0,
            f"{stem}l2_error": math.nan,
            f"{stem}rms_error": math.nan,
            f"{stem}relative_l2_error": math.nan,
            f"{stem}max_abs_error": math.nan,
        }
    error = values[finite_mask] - exact_values[finite_mask]
    exact = exact_values[finite_mask]
    exact_scale = max(rms(exact), scale_floor, 1.0e-14)
    error_rms = rms(error)
    return {
        f"{stem}node_count": int(np.count_nonzero(finite_mask)),
        f"{stem}l2_error": error_rms,
        f"{stem}rms_error": error_rms,
        f"{stem}relative_l2_error": error_rms / exact_scale,
        f"{stem}max_abs_error": float(np.max(np.abs(error))),
    }


def quadrature_field_norm_metrics(
    prefix: str,
    stats: dict[str, float],
    *,
    velocity_scale_floor: float,
    pressure_scale_floor: float,
) -> dict[str, object]:
    stem = f"{prefix}_" if prefix else ""
    total_area = stats["area"]
    finite_area = stats["finite_area"]
    finite_fraction = finite_area / total_area if total_area > 0.0 else float("nan")
    velocity_l2 = weighted_rms(stats["velocity_error_sq"], finite_area)
    velocity_exact_rms = weighted_rms(stats["velocity_exact_sq"], finite_area)
    velocity_scale = finite_scale(velocity_exact_rms, velocity_scale_floor, 1.0e-14)
    pressure_rms = weighted_rms(stats["pressure_error_sq"], finite_area)
    pressure_exact_rms = weighted_rms(stats["pressure_exact_sq"], finite_area)
    pressure_scale = finite_scale(pressure_exact_rms, pressure_scale_floor, 1.0)
    pressure_offset = (
        stats["pressure_error_sum"] / finite_area if finite_area > 0.0 else float("nan")
    )
    pressure_shifted_sq = (
        max(0.0, stats["pressure_error_sq"] - pressure_offset * pressure_offset * finite_area)
        if finite_area > 0.0
        else float("nan")
    )
    pressure_shifted_rms = weighted_rms(pressure_shifted_sq, finite_area)

    return {
        f"{stem}quadrature_field_area": total_area,
        f"{stem}quadrature_field_finite_area": finite_area,
        f"{stem}quadrature_field_finite_area_fraction": finite_fraction,
        f"{stem}quadrature_field_sample_count": int(stats["sample_count"]),
        f"{stem}quadrature_field_finite_sample_count": int(stats["finite_sample_count"]),
        f"{stem}quadrature_velocity_l2_error": velocity_l2,
        f"{stem}quadrature_velocity_relative_l2_error": velocity_l2 / velocity_scale,
        f"{stem}quadrature_pressure_rms_error": pressure_rms,
        f"{stem}quadrature_pressure_relative_rms_error": pressure_rms / pressure_scale,
        f"{stem}quadrature_pressure_rms_error_after_constant_offset_removal": pressure_shifted_rms,
        f"{stem}quadrature_pressure_relative_rms_error_after_constant_offset_removal": (
            pressure_shifted_rms / pressure_scale
        ),
        f"{stem}quadrature_pressure_constant_offset": pressure_offset,
    }


def solve_weighted_mode(stats: dict[str, object]) -> np.ndarray | None:
    finite_count = int(stats["finite_sample_count"])
    if finite_count < 3:
        return None
    matrix = np.asarray(stats["normal_matrix"], dtype=float)
    rhs = np.asarray(stats["rhs"], dtype=float)
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
        return coeffs


def implied_interface_mode_metrics(
    prefix: str,
    stats: dict[str, object],
    *,
    expected: dict,
    time: float,
) -> dict[str, object]:
    sol, _ = sol_params(expected)
    stem = f"{prefix}_" if prefix else ""
    area = float(stats["area"])
    finite_area = float(stats["finite_area"])
    coeffs = solve_weighted_mode(stats)
    height_l2 = weighted_rms(float(stats["height_error_sq"]), finite_area)
    expected_shift = shift(time, expected)
    expected_cos = sol["amplitude"] * math.cos(sol["k"] * expected_shift)
    expected_sin = sol["amplitude"] * math.sin(sol["k"] * expected_shift)

    out: dict[str, object] = {
        f"{stem}quadrature_implied_interface_area": area,
        f"{stem}quadrature_implied_interface_finite_area": finite_area,
        f"{stem}quadrature_implied_interface_sample_count": int(stats["sample_count"]),
        f"{stem}quadrature_implied_interface_finite_sample_count": int(stats["finite_sample_count"]),
        f"{stem}quadrature_implied_interface_height_l2_error": height_l2,
    }
    if coeffs is None:
        out.update(
            {
                f"{stem}quadrature_implied_interface_mean_error": math.nan,
                f"{stem}quadrature_implied_interface_amplitude_error": math.nan,
                f"{stem}quadrature_implied_interface_shift_error": math.nan,
            }
        )
        return out

    c0, c_cos, c_sin = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    amplitude = math.hypot(c_cos, c_sin)
    if interface_phase_defined(sol):
        measured_phase = math.atan2(c_sin, c_cos)
        measured_shift = measured_phase / sol["k"]
        shift_error = wrap_error(measured_shift - expected_shift, sol["L"])
    else:
        measured_shift = 0.0
        shift_error = 0.0
    out.update(
        {
            f"{stem}quadrature_implied_interface_mean": c0,
            f"{stem}quadrature_implied_interface_cos_coeff": c_cos,
            f"{stem}quadrature_implied_interface_sin_coeff": c_sin,
            f"{stem}quadrature_implied_interface_amplitude": amplitude,
            f"{stem}quadrature_implied_interface_shift_measured": measured_shift,
            f"{stem}quadrature_implied_interface_mean_error": c0 - sol["H0"],
            f"{stem}quadrature_implied_interface_cos_error": c_cos - expected_cos,
            f"{stem}quadrature_implied_interface_sin_error": c_sin - expected_sin,
            f"{stem}quadrature_implied_interface_amplitude_error": amplitude - sol["amplitude"],
            f"{stem}quadrature_implied_interface_shift_error": shift_error,
        }
    )
    return out


def gradient_stats_template() -> dict[str, float]:
    return {
        "area": 0.0,
        "finite_area": 0.0,
        "grad_error_sq": 0.0,
        "grad_exact_sq": 0.0,
        "grad_x_error_sq": 0.0,
        "grad_x_exact_sq": 0.0,
        "grad_y_error_sq": 0.0,
        "grad_y_exact_sq": 0.0,
        "spatial_residual_error_sq": 0.0,
        "spatial_residual_exact_sq": 0.0,
        "sample_count": 0.0,
        "finite_sample_count": 0.0,
    }


def accumulate_gradient_stats(
    stats: dict[str, float],
    *,
    weight: float,
    grad_q: np.ndarray,
    grad_exact_q: np.ndarray,
    spatial_residual_q: float,
    spatial_residual_exact_q: float,
) -> None:
    stats["area"] += weight
    stats["sample_count"] += 1.0
    if (
        not np.all(np.isfinite(grad_q))
        or not np.all(np.isfinite(grad_exact_q))
        or not math.isfinite(spatial_residual_q)
        or not math.isfinite(spatial_residual_exact_q)
    ):
        return
    grad_error = grad_q - grad_exact_q
    spatial_error = spatial_residual_q - spatial_residual_exact_q
    stats["finite_area"] += weight
    stats["finite_sample_count"] += 1.0
    stats["grad_error_sq"] += weight * float(np.dot(grad_error, grad_error))
    stats["grad_exact_sq"] += weight * float(np.dot(grad_exact_q, grad_exact_q))
    stats["grad_x_error_sq"] += weight * float(grad_error[0] * grad_error[0])
    stats["grad_x_exact_sq"] += weight * float(grad_exact_q[0] * grad_exact_q[0])
    stats["grad_y_error_sq"] += weight * float(grad_error[1] * grad_error[1])
    stats["grad_y_exact_sq"] += weight * float(grad_exact_q[1] * grad_exact_q[1])
    stats["spatial_residual_error_sq"] += weight * spatial_error * spatial_error
    stats["spatial_residual_exact_sq"] += (
        weight * spatial_residual_exact_q * spatial_residual_exact_q
    )


def gradient_norm_metrics(prefix: str, stats: dict[str, float]) -> dict[str, object]:
    stem = f"{prefix}_" if prefix else ""
    grad_l2 = weighted_rms(stats["grad_error_sq"], stats["finite_area"])
    grad_exact = weighted_rms(stats["grad_exact_sq"], stats["finite_area"])
    grad_x_l2 = weighted_rms(stats["grad_x_error_sq"], stats["finite_area"])
    grad_x_exact = weighted_rms(stats["grad_x_exact_sq"], stats["finite_area"])
    grad_y_l2 = weighted_rms(stats["grad_y_error_sq"], stats["finite_area"])
    grad_y_exact = weighted_rms(stats["grad_y_exact_sq"], stats["finite_area"])
    spatial_l2 = weighted_rms(stats["spatial_residual_error_sq"], stats["finite_area"])
    spatial_exact = weighted_rms(stats["spatial_residual_exact_sq"], stats["finite_area"])
    return {
        f"{stem}quadrature_phi_grad_area": stats["area"],
        f"{stem}quadrature_phi_grad_finite_area": stats["finite_area"],
        f"{stem}quadrature_phi_grad_sample_count": int(stats["sample_count"]),
        f"{stem}quadrature_phi_grad_finite_sample_count": int(stats["finite_sample_count"]),
        f"{stem}quadrature_phi_grad_l2_error": grad_l2,
        f"{stem}quadrature_phi_grad_relative_l2_error": grad_l2 / finite_scale(grad_exact, 1.0e-14),
        f"{stem}quadrature_phi_grad_x_l2_error": grad_x_l2,
        f"{stem}quadrature_phi_grad_x_relative_l2_error": (
            grad_x_l2 / finite_scale(grad_x_exact, 1.0e-14)
        ),
        f"{stem}quadrature_phi_grad_y_l2_error": grad_y_l2,
        f"{stem}quadrature_phi_grad_y_relative_l2_error": (
            grad_y_l2 / finite_scale(grad_y_exact, 1.0e-14)
        ),
        f"{stem}quadrature_level_set_spatial_residual_l2_error": spatial_l2,
        f"{stem}quadrature_level_set_spatial_residual_relative_l2_error": (
            spatial_l2 / finite_scale(spatial_exact, 1.0e-14)
        ),
    }


def quadrature_norm_metrics(
    cells: list[np.ndarray],
    points: np.ndarray,
    phi: np.ndarray,
    velocity: np.ndarray | None,
    pressure: np.ndarray | None,
    time: float,
    expected: dict,
    *,
    bulk_clearance: float,
    side_wall_clearance: float,
) -> dict[str, object]:
    sol, fluid = sol_params(expected)
    order = int(expected.get("verification", {}).get("field_quadrature_order", 4))
    order = max(1, order)
    qpts, qwts = np.polynomial.legendre.leggauss(order)

    h_mesh = float(expected["mesh"]["element_size"])
    velocity_scale_floor = abs(sol["U0"])
    pressure_scale_floor = fluid["density"] * fluid["gravity"] * sol["H0"]
    coords = points[:, :2]
    x_min = float(np.min(coords[:, 0]))
    x_max = float(np.max(coords[:, 0]))

    phi_area = 0.0
    phi_error_sq = 0.0
    phi_exact_sq = 0.0
    phi_sample_count = 0
    phi_finite_sample_count = 0
    interior_phi_area = 0.0
    interior_phi_error_sq = 0.0
    interior_phi_exact_sq = 0.0
    interior_phi_sample_count = 0
    interior_phi_finite_sample_count = 0
    field_stats = {
        "area": 0.0,
        "finite_area": 0.0,
        "velocity_error_sq": 0.0,
        "velocity_exact_sq": 0.0,
        "pressure_error_sq": 0.0,
        "pressure_exact_sq": 0.0,
        "pressure_error_sum": 0.0,
        "sample_count": 0.0,
        "finite_sample_count": 0.0,
    }
    bulk_stats = dict(field_stats)
    implied_mode_stats = {
        "area": 0.0,
        "finite_area": 0.0,
        "height_error_sq": 0.0,
        "sample_count": 0.0,
        "finite_sample_count": 0.0,
        "normal_matrix": np.zeros((3, 3), dtype=float),
        "rhs": np.zeros(3, dtype=float),
    }
    interior_implied_mode_stats = {
        "area": 0.0,
        "finite_area": 0.0,
        "height_error_sq": 0.0,
        "sample_count": 0.0,
        "finite_sample_count": 0.0,
        "normal_matrix": np.zeros((3, 3), dtype=float),
        "rhs": np.zeros(3, dtype=float),
    }
    gradient_stats = gradient_stats_template()
    interior_gradient_stats = gradient_stats_template()

    include_field_metrics = velocity is not None and pressure is not None
    expected_velocity_x = uniform_velocity(time, expected)
    for cell in cells:
        cell_points = coords[cell, :]
        cell_phi = phi[cell]
        cell_velocity = velocity[cell, :2] if include_field_metrics else None
        cell_pressure = pressure[cell] if include_field_metrics else None
        for xi, wx in zip(qpts, qwts):
            for eta, wy in zip(qpts, qwts):
                shape, shape_xi, shape_eta = quad_shape_functions(cell.size, float(xi), float(eta))
                x_q = shape @ cell_points
                dx_dxi = shape_xi @ cell_points
                dx_deta = shape_eta @ cell_points
                det_j = float(dx_dxi[0] * dx_deta[1] - dx_dxi[1] * dx_deta[0])
                jac = abs(det_j)
                weight = float(wx * wy * jac)
                if weight <= 0.0 or not math.isfinite(weight):
                    continue

                point_q = np.array([[x_q[0], x_q[1], 0.0]], dtype=float)
                phi_exact_q = float(exact_phi(point_q, time, expected)[0])
                height_exact_q = float(exact_height(np.array([x_q[0]]), time, expected)[0])
                height_x_exact_q = float(exact_height_x(np.array([x_q[0]]), time, expected)[0])
                phi_q = float(shape @ cell_phi)
                grad_q = np.array([math.nan, math.nan], dtype=float)
                if math.isfinite(det_j) and abs(det_j) > ZERO_TOL:
                    dphi_ref = np.array(
                        [
                            float(shape_xi @ cell_phi),
                            float(shape_eta @ cell_phi),
                        ],
                        dtype=float,
                    )
                    jacobian = np.array(
                        [
                            [dx_dxi[0], dx_deta[0]],
                            [dx_dxi[1], dx_deta[1]],
                        ],
                        dtype=float,
                    )
                    try:
                        grad_q = np.linalg.solve(jacobian.T, dphi_ref)
                    except np.linalg.LinAlgError:
                        grad_q = np.array([math.nan, math.nan], dtype=float)
                grad_exact_q = np.array([-height_x_exact_q, 1.0], dtype=float)
                spatial_residual_q = expected_velocity_x * float(grad_q[0])
                spatial_residual_exact_q = expected_velocity_x * float(grad_exact_q[0])
                accumulate_gradient_stats(
                    gradient_stats,
                    weight=weight,
                    grad_q=grad_q,
                    grad_exact_q=grad_exact_q,
                    spatial_residual_q=spatial_residual_q,
                    spatial_residual_exact_q=spatial_residual_exact_q,
                )
                phi_area += weight
                phi_sample_count += 1
                implied_mode_stats["area"] = float(implied_mode_stats["area"]) + weight
                implied_mode_stats["sample_count"] = float(implied_mode_stats["sample_count"]) + 1.0
                if math.isfinite(phi_q):
                    phi_error_sq += weight * (phi_q - phi_exact_q) ** 2
                    phi_exact_sq += weight * phi_exact_q * phi_exact_q
                    phi_finite_sample_count += 1
                    height_q = x_q[1] - phi_q
                    mode = np.array([1.0, math.cos(sol["k"] * x_q[0]), math.sin(sol["k"] * x_q[0])])
                    implied_mode_stats["finite_area"] = float(implied_mode_stats["finite_area"]) + weight
                    implied_mode_stats["finite_sample_count"] = (
                        float(implied_mode_stats["finite_sample_count"]) + 1.0
                    )
                    implied_mode_stats["height_error_sq"] = (
                        float(implied_mode_stats["height_error_sq"])
                        + weight * (height_q - height_exact_q) ** 2
                    )
                    implied_mode_stats["normal_matrix"] = (
                        np.asarray(implied_mode_stats["normal_matrix"]) + weight * np.outer(mode, mode)
                    )
                    implied_mode_stats["rhs"] = np.asarray(implied_mode_stats["rhs"]) + weight * mode * height_q
                if (
                    x_q[0] > x_min + side_wall_clearance
                    and x_q[0] < x_max - side_wall_clearance
                ):
                    accumulate_gradient_stats(
                        interior_gradient_stats,
                        weight=weight,
                        grad_q=grad_q,
                        grad_exact_q=grad_exact_q,
                        spatial_residual_q=spatial_residual_q,
                        spatial_residual_exact_q=spatial_residual_exact_q,
                    )
                    interior_phi_area += weight
                    interior_phi_sample_count += 1
                    interior_implied_mode_stats["area"] = (
                        float(interior_implied_mode_stats["area"]) + weight
                    )
                    interior_implied_mode_stats["sample_count"] = (
                        float(interior_implied_mode_stats["sample_count"]) + 1.0
                    )
                    if math.isfinite(phi_q):
                        interior_phi_error_sq += weight * (phi_q - phi_exact_q) ** 2
                        interior_phi_exact_sq += weight * phi_exact_q * phi_exact_q
                        interior_phi_finite_sample_count += 1
                        interior_implied_mode_stats["finite_area"] = (
                            float(interior_implied_mode_stats["finite_area"]) + weight
                        )
                        interior_implied_mode_stats["finite_sample_count"] = (
                            float(interior_implied_mode_stats["finite_sample_count"]) + 1.0
                        )
                        interior_implied_mode_stats["height_error_sq"] = (
                            float(interior_implied_mode_stats["height_error_sq"])
                            + weight * (height_q - height_exact_q) ** 2
                        )
                        interior_implied_mode_stats["normal_matrix"] = (
                            np.asarray(interior_implied_mode_stats["normal_matrix"])
                            + weight * np.outer(mode, mode)
                        )
                        interior_implied_mode_stats["rhs"] = (
                            np.asarray(interior_implied_mode_stats["rhs"]) + weight * mode * height_q
                        )

                if not include_field_metrics:
                    continue

                active_stats = []
                if phi_exact_q < -2.0 * h_mesh:
                    active_stats.append(field_stats)
                if phi_exact_q < -bulk_clearance:
                    active_stats.append(bulk_stats)
                if not active_stats:
                    continue

                velocity_q = shape @ cell_velocity
                pressure_q = float(shape @ cell_pressure)
                pressure_exact_q = float(exact_pressure(point_q, time, expected)[0])
                velocity_error_sq = float(
                    (velocity_q[0] - expected_velocity_x) ** 2 + velocity_q[1] ** 2
                )
                velocity_exact_sq = expected_velocity_x * expected_velocity_x
                pressure_error = pressure_q - pressure_exact_q
                finite_fields = (
                    np.all(np.isfinite(velocity_q))
                    and math.isfinite(pressure_q)
                    and math.isfinite(pressure_error)
                )
                for stats in active_stats:
                    stats["area"] += weight
                    stats["sample_count"] += 1.0
                    if not finite_fields:
                        continue
                    stats["finite_area"] += weight
                    stats["velocity_error_sq"] += weight * velocity_error_sq
                    stats["velocity_exact_sq"] += weight * velocity_exact_sq
                    stats["pressure_error_sq"] += weight * pressure_error * pressure_error
                    stats["pressure_exact_sq"] += weight * pressure_exact_q * pressure_exact_q
                    stats["pressure_error_sum"] += weight * pressure_error
                    stats["finite_sample_count"] += 1.0

    phi_l2 = weighted_rms(phi_error_sq, phi_area)
    phi_exact_rms = weighted_rms(phi_exact_sq, phi_area)
    phi_scale = finite_scale(phi_exact_rms, sol["H0"], 1.0e-14)
    interior_phi_l2 = weighted_rms(interior_phi_error_sq, interior_phi_area)
    interior_phi_exact_rms = weighted_rms(interior_phi_exact_sq, interior_phi_area)
    interior_phi_scale = finite_scale(interior_phi_exact_rms, sol["H0"], 1.0e-14)
    out: dict[str, object] = {
        "field_quadrature_order": order,
        "quadrature_domain_area": phi_area,
        "quadrature_phi_sample_count": phi_sample_count,
        "quadrature_phi_finite_sample_count": phi_finite_sample_count,
        "quadrature_phi_l2_error": phi_l2,
        "quadrature_phi_relative_l2_error": phi_l2 / phi_scale,
        "quadrature_interior_phi_area": interior_phi_area,
        "quadrature_interior_phi_sample_count": interior_phi_sample_count,
        "quadrature_interior_phi_finite_sample_count": interior_phi_finite_sample_count,
        "quadrature_interior_phi_l2_error": interior_phi_l2,
        "quadrature_interior_phi_relative_l2_error": interior_phi_l2 / interior_phi_scale,
    }
    out.update(
        implied_interface_mode_metrics(
            "",
            implied_mode_stats,
            expected=expected,
            time=time,
        )
    )
    out.update(
        implied_interface_mode_metrics(
            "interior",
            interior_implied_mode_stats,
            expected=expected,
            time=time,
        )
    )
    out.update(gradient_norm_metrics("", gradient_stats))
    out.update(gradient_norm_metrics("interior", interior_gradient_stats))
    if include_field_metrics:
        out.update(
            quadrature_field_norm_metrics(
                "",
                field_stats,
                velocity_scale_floor=velocity_scale_floor,
                pressure_scale_floor=pressure_scale_floor,
            )
        )
        out.update(
            quadrature_field_norm_metrics(
                "bulk",
                bulk_stats,
                velocity_scale_floor=velocity_scale_floor,
                pressure_scale_floor=pressure_scale_floor,
            )
        )
    return out


def compute_metrics(result_path: Path, time: float, expected: dict) -> dict[str, object]:
    grid = pv.read(result_path)
    cells = quad_cells(grid)
    points = np.asarray(grid.points, dtype=float)
    phi = np.asarray(grid.point_data["phi"], dtype=float)
    checked_fields = set(expected.get("verification", {}).get("checked_fields", ["phi", "Velocity", "Pressure"]))
    verify_velocity_pressure = "Velocity" in checked_fields or "Pressure" in checked_fields
    velocity: np.ndarray | None = None
    pressure: np.ndarray | None = None
    if verify_velocity_pressure:
        if "Velocity" not in grid.point_data or "Pressure" not in grid.point_data:
            missing = [
                name
                for name in ("Velocity", "Pressure")
                if name not in grid.point_data
            ]
            raise RuntimeError(f"missing checked field(s): {', '.join(missing)}")
        velocity = np.asarray(grid.point_data["Velocity"], dtype=float)
        pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)

    sol, fluid = sol_params(expected)
    tolerances = expected["verification"]["suggested_tolerances"]
    phi_ex = exact_phi(points, time, expected)
    phi_error = phi - phi_ex

    crossings = edge_zero_crossings(points, phi, cells)
    xpoints = crossing_points(crossings)
    c0, c_cos, c_sin = fit_mode(xpoints, sol["k"])
    amplitude = math.hypot(c_cos, c_sin)
    expected_shift = shift(time, expected)
    if interface_phase_defined(sol):
        measured_phase = math.atan2(c_sin, c_cos)
        measured_shift = measured_phase / sol["k"]
        shift_error = wrap_error(measured_shift - expected_shift, sol["L"])
    else:
        measured_shift = 0.0
        shift_error = 0.0
    expected_cos = sol["amplitude"] * math.cos(sol["k"] * expected_shift)
    expected_sin = sol["amplitude"] * math.sin(sol["k"] * expected_shift)
    exact_h_cross = exact_height(xpoints[:, 0], time, expected)
    height_error = xpoints[:, 1] - exact_h_cross

    area, centroid = clipped_area_and_centroid(points, phi, cells)
    area_expected = float(sol["expected_area"])
    centroid_y_expected = float(sol["expected_y_centroid"])

    h_mesh = float(expected["mesh"]["element_size"])
    bulk_clearance = float(
        expected.get("verification", {}).get("bulk_field_clearance", 0.05 * sol["H0"])
    )
    bulk_clearance = max(bulk_clearance, 1.0e-12)
    side_wall_clearance = float(
        expected.get("verification", {}).get("side_wall_clearance", 0.05 * sol["L"])
    )
    side_wall_clearance = max(side_wall_clearance, 0.0)
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    interior_phi_mask = (
        (points[:, 0] > x_min + side_wall_clearance)
        & (points[:, 0] < x_max - side_wall_clearance)
    )
    field_metrics: dict[str, object] = {}
    bulk_field_metrics: dict[str, object] = {}
    bulk_fallback_to_legacy = False
    if verify_velocity_pressure:
        assert velocity is not None
        assert pressure is not None
        finite_velocity = np.all(np.isfinite(velocity[:, :2]), axis=1)
        finite_pressure = np.isfinite(pressure)
        wet = (phi_ex < -2.0 * h_mesh) & finite_velocity & finite_pressure
        if not np.any(wet):
            wet = (phi <= -h_mesh) & finite_velocity & finite_pressure
        if not np.any(wet):
            raise RuntimeError("no wet finite nodes available for field verification")

        vel_exact = exact_velocity(points, time, expected)
        p_exact = exact_pressure(points, time, expected)
        expected_velocity_x = uniform_velocity(time, expected)
        pressure_scale_floor = fluid["density"] * fluid["gravity"] * sol["H0"]
        field_metrics = field_norm_metrics(
            "",
            wet,
            velocity=velocity,
            pressure=pressure,
            velocity_exact_values=vel_exact,
            pressure_exact_values=p_exact,
            expected_velocity_x=expected_velocity_x,
            velocity_scale_floor=abs(sol["U0"]),
            pressure_scale_floor=pressure_scale_floor,
        )

        bulk_wet = (phi_ex < -bulk_clearance) & finite_velocity & finite_pressure
        if not np.any(bulk_wet):
            bulk_wet = wet
            bulk_fallback_to_legacy = True
        bulk_field_metrics = field_norm_metrics(
            "bulk",
            bulk_wet,
            velocity=velocity,
            pressure=pressure,
            velocity_exact_values=vel_exact,
            pressure_exact_values=p_exact,
            expected_velocity_x=expected_velocity_x,
            velocity_scale_floor=abs(sol["U0"]),
            pressure_scale_floor=pressure_scale_floor,
        )

    quadrature_metrics = quadrature_norm_metrics(
        cells,
        points,
        phi,
        velocity,
        pressure,
        time,
        expected,
        bulk_clearance=bulk_clearance,
        side_wall_clearance=side_wall_clearance,
    )

    p_interface_finite = np.asarray([], dtype=float)
    p_interface_nonfinite_endpoint_count = 0
    interface_pressure_abs_required = False
    interface_pressure_rms = math.nan
    interface_pressure_max_abs = math.nan
    p_interface = np.asarray([], dtype=float)
    if verify_velocity_pressure:
        assert pressure is not None
        p_interface = interpolate_crossing_values(pressure, crossings)
        p_interface_finite = p_interface[np.isfinite(p_interface)]
        p_interface_nonfinite_endpoint_count = nonfinite_crossing_endpoint_count(pressure, crossings)
        interface_pressure_abs_required = bool(
            expected.get("verification", {}).get("interface_pressure_abs_required", True)
        )
        interface_pressure_rms = rms(p_interface_finite) if p_interface_finite.size else math.nan
        interface_pressure_max_abs = (
            float(np.max(np.abs(p_interface_finite))) if p_interface_finite.size else math.nan
        )

    audits = residual_audits(points, time, expected)
    metrics: dict[str, object] = {
        "result_path": str(result_path),
        "time": time,
        "solver_xml_is_full_exact_mms": bool(expected["solver_feature_status"]["solver_xml_is_full_exact_mms"]),
        "solver_feature_blocker": expected["solver_feature_status"]["blocker"],
        "checked_fields": sorted(checked_fields),
        "phi_l2_error": rms(phi_error),
        "phi_rms_error": rms(phi_error),
        "phi_max_abs_error": float(np.max(np.abs(phi_error))),
        "interface_crossing_count": len(crossings),
        "interface_mean": c0,
        "interface_cos_coeff": c_cos,
        "interface_sin_coeff": c_sin,
        "interface_amplitude": amplitude,
        "interface_shift_measured": measured_shift,
        "interface_shift_expected": expected_shift,
        "interface_mean_error": c0 - sol["H0"],
        "interface_cos_error": c_cos - expected_cos,
        "interface_sin_error": c_sin - expected_sin,
        "interface_amplitude_error": amplitude - sol["amplitude"],
        "interface_shift_error": shift_error,
        "interface_l2_height_error": rms(height_error),
        "interface_max_height_error": float(np.max(np.abs(height_error))),
        "area": area,
        "area_error": area - area_expected,
        "area_relative_error": abs(area - area_expected) / area_expected,
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_y_error": float(centroid[1] - centroid_y_expected),
        "bulk_field_clearance": bulk_clearance,
        "bulk_wet_fallback_to_legacy_mask": bulk_fallback_to_legacy,
        "side_wall_clearance": side_wall_clearance,
    }
    metrics.update(
        scalar_l2_metrics(
            "interior_phi",
            phi,
            phi_ex,
            interior_phi_mask,
            scale_floor=sol["H0"],
        )
    )
    if verify_velocity_pressure:
        metrics.update(
            {
                "interface_pressure_interpolation_count": int(p_interface.size),
                "interface_pressure_finite_count": int(p_interface_finite.size),
                "interface_pressure_nonfinite_endpoint_count": int(p_interface_nonfinite_endpoint_count),
                "interface_pressure_abs_required": interface_pressure_abs_required,
                "interface_pressure_rms": interface_pressure_rms,
                "interface_pressure_max_abs": interface_pressure_max_abs,
            }
        )
    metrics.update(field_metrics)
    metrics.update(bulk_field_metrics)
    metrics.update(quadrature_metrics)
    metrics.update(audits)

    checks = {
        "phi_l2_abs": metrics["phi_l2_error"] <= tolerances["phi_l2_abs"],
        "phi_max_abs": metrics["phi_max_abs_error"] <= tolerances["phi_max_abs"],
        "interface_mean_abs": abs(metrics["interface_mean_error"]) <= tolerances["interface_mean_abs"],
        "interface_amplitude_relative": abs(metrics["interface_amplitude_error"]) / max(sol["amplitude"], 1.0e-14)
        <= tolerances["interface_amplitude_relative"],
        "interface_shift_abs": abs(metrics["interface_shift_error"]) <= tolerances["interface_shift_abs"],
        "interface_l2_height_abs": metrics["interface_l2_height_error"] <= tolerances["interface_l2_height_abs"],
        "interface_max_height_abs": metrics["interface_max_height_error"] <= tolerances["interface_max_height_abs"],
        "area_relative": metrics["area_relative_error"] <= tolerances["area_relative"],
        "centroid_y_abs": abs(metrics["centroid_y_error"]) <= tolerances["centroid_y_abs"],
        "manufactured_residual_x_abs": metrics["manufactured_residual_x_max"] <= tolerances["manufactured_residual_abs"],
        "manufactured_residual_y_abs": metrics["manufactured_residual_y_max"] <= tolerances["manufactured_residual_abs"],
        "level_set_residual_abs": metrics["level_set_residual_max"] <= tolerances["level_set_residual_abs"],
    }
    if verify_velocity_pressure:
        checks.update(
            {
                "velocity_relative_l2": metrics["velocity_relative_l2_error"] <= tolerances["velocity_relative_l2"],
                "velocity_mean_abs": max(abs(metrics["velocity_mean_x_error"]), abs(metrics["velocity_mean_y_error"]))
                <= tolerances["velocity_mean_abs"],
                "pressure_relative_rms": metrics["pressure_relative_rms_error"] <= tolerances["pressure_relative_rms"],
                "pressure_rms_after_offset_relative": metrics["pressure_relative_rms_error_after_constant_offset_removal"]
                <= tolerances["pressure_rms_after_offset_relative"],
                "interface_pressure_all_samples_finite": metrics["interface_pressure_finite_count"]
                == metrics["interface_pressure_interpolation_count"],
                "interface_pressure_all_endpoints_finite": metrics["interface_pressure_nonfinite_endpoint_count"] == 0,
                "interface_pressure_abs": (
                    not interface_pressure_abs_required
                    or (
                        math.isfinite(metrics["interface_pressure_max_abs"])
                        and metrics["interface_pressure_max_abs"] <= tolerances["interface_pressure_abs"]
                    )
                ),
            }
        )
    failed = [name for name, ok in checks.items() if not ok]
    metrics["failed_checks"] = failed
    metrics["passed"] = not failed
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", type=Path, help="VTU/PVTU result file. Defaults to latest result_*.vtu.")
    parser.add_argument("--time", type=float, default=None, help="Physical time for the result. Inferred from filename if omitted.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = load_expected()
    result_path = resolve_result_path(args.result)
    time = infer_time(result_path, expected, args.time)
    metrics = compute_metrics(result_path, time, expected)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
