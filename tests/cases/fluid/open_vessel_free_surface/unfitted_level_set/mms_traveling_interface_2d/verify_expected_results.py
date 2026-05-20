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


def shift(t: float, expected: dict) -> float:
    sol, _ = sol_params(expected)
    return (sol["U0"] / sol["Omega"]) * math.sin(sol["Omega"] * t)


def uniform_velocity(t: float, expected: dict) -> float:
    sol, _ = sol_params(expected)
    return sol["U0"] * math.cos(sol["Omega"] * t)


def uniform_acceleration(t: float, expected: dict) -> float:
    sol, _ = sol_params(expected)
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


def rms(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


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


def compute_metrics(result_path: Path, time: float, expected: dict) -> dict[str, object]:
    grid = pv.read(result_path)
    cells = quad_cells(grid)
    points = np.asarray(grid.points, dtype=float)
    phi = np.asarray(grid.point_data["phi"], dtype=float)
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
    measured_phase = math.atan2(c_sin, c_cos)
    measured_shift = measured_phase / sol["k"]
    expected_shift = shift(time, expected)
    shift_error = wrap_error(measured_shift - expected_shift, sol["L"])
    expected_cos = sol["amplitude"] * math.cos(sol["k"] * expected_shift)
    expected_sin = sol["amplitude"] * math.sin(sol["k"] * expected_shift)
    exact_h_cross = exact_height(xpoints[:, 0], time, expected)
    height_error = xpoints[:, 1] - exact_h_cross

    area, centroid = clipped_area_and_centroid(points, phi, cells)
    area_expected = float(sol["expected_area"])
    centroid_y_expected = float(sol["expected_y_centroid"])

    h_mesh = float(expected["mesh"]["element_size"])
    finite_velocity = np.all(np.isfinite(velocity[:, :2]), axis=1)
    finite_pressure = np.isfinite(pressure)
    wet = (phi_ex < -2.0 * h_mesh) & finite_velocity & finite_pressure
    if not np.any(wet):
        wet = (phi <= -h_mesh) & finite_velocity & finite_pressure
    if not np.any(wet):
        raise RuntimeError("no wet finite nodes available for field verification")

    vel_exact = exact_velocity(points, time, expected)
    vel_error = velocity[wet, :2] - vel_exact[wet, :2]
    vel_error_norm = np.linalg.norm(vel_error, axis=1)
    vel_exact_norm = np.linalg.norm(vel_exact[wet, :2], axis=1)
    velocity_scale = max(rms(vel_exact_norm), abs(sol["U0"]), 1.0e-14)

    p_exact = exact_pressure(points, time, expected)
    p_error = pressure[wet] - p_exact[wet]
    pressure_offset = float(np.mean(p_error))
    p_error_shifted = p_error - pressure_offset
    pressure_scale = max(rms(p_exact[wet]), fluid["density"] * fluid["gravity"] * sol["H0"], 1.0)

    p_interface = interpolate_crossing_values(pressure, crossings)
    p_interface_finite = p_interface[np.isfinite(p_interface)]
    p_interface_nonfinite_endpoint_count = nonfinite_crossing_endpoint_count(pressure, crossings)
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
        "wet_node_count": int(np.count_nonzero(wet)),
        "velocity_l2_error": rms(vel_error_norm),
        "velocity_rms_error": rms(vel_error_norm),
        "velocity_relative_l2_error": rms(vel_error_norm) / velocity_scale,
        "velocity_max_abs_error": float(np.max(np.abs(vel_error))),
        "velocity_mean_x": float(np.mean(velocity[wet, 0])),
        "velocity_mean_y": float(np.mean(velocity[wet, 1])),
        "velocity_mean_x_error": float(np.mean(velocity[wet, 0]) - uniform_velocity(time, expected)),
        "velocity_mean_y_error": float(np.mean(velocity[wet, 1])),
        "pressure_rms_error": rms(p_error),
        "pressure_relative_rms_error": rms(p_error) / pressure_scale,
        "pressure_max_abs_error": float(np.max(np.abs(p_error))),
        "pressure_rms_error_after_constant_offset_removal": rms(p_error_shifted),
        "pressure_relative_rms_error_after_constant_offset_removal": rms(p_error_shifted) / pressure_scale,
        "pressure_constant_offset": pressure_offset,
        "interface_pressure_interpolation_count": int(p_interface.size),
        "interface_pressure_finite_count": int(p_interface_finite.size),
        "interface_pressure_nonfinite_endpoint_count": int(p_interface_nonfinite_endpoint_count),
        "interface_pressure_rms": interface_pressure_rms,
        "interface_pressure_max_abs": interface_pressure_max_abs,
    }
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
        "velocity_relative_l2": metrics["velocity_relative_l2_error"] <= tolerances["velocity_relative_l2"],
        "velocity_mean_abs": max(abs(metrics["velocity_mean_x_error"]), abs(metrics["velocity_mean_y_error"]))
        <= tolerances["velocity_mean_abs"],
        "pressure_relative_rms": metrics["pressure_relative_rms_error"] <= tolerances["pressure_relative_rms"],
        "pressure_rms_after_offset_relative": metrics["pressure_relative_rms_error_after_constant_offset_removal"]
        <= tolerances["pressure_rms_after_offset_relative"],
        "interface_pressure_all_samples_finite": metrics["interface_pressure_finite_count"]
        == metrics["interface_pressure_interpolation_count"],
        "interface_pressure_all_endpoints_finite": metrics["interface_pressure_nonfinite_endpoint_count"] == 0,
        "interface_pressure_abs": math.isfinite(metrics["interface_pressure_max_abs"])
        and metrics["interface_pressure_max_abs"] <= tolerances["interface_pressure_abs"],
        "manufactured_residual_x_abs": metrics["manufactured_residual_x_max"] <= tolerances["manufactured_residual_abs"],
        "manufactured_residual_y_abs": metrics["manufactured_residual_y_max"] <= tolerances["manufactured_residual_abs"],
        "level_set_residual_abs": metrics["level_set_residual_max"] <= tolerances["level_set_residual_abs"],
    }
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
