#!/usr/bin/env python3
"""Audit SPHERIC Test02 obstacle-face water coverage and pressure stack."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

import verify_spheric_test02_histories as verifier


OBSTACLE_FRONT_X_M = 0.8245
OBSTACLE_CENTER_Z_M = 0.5
OBSTACLE_TOP_Y_M = 0.161
FACE_SENSOR_HEIGHTS_M = {
    "P1": 0.021,
    "P2": 0.061,
    "P3": 0.101,
    "P4": 0.141,
}
TOP_SENSOR_X_M = {
    "P5": 0.8035,
    "P7": 0.7235,
}
PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")


def sample_scalar_at_point(
    grid: pv.DataSet,
    point: tuple[float, float, float],
    scalar_name: str,
) -> dict[str, Any]:
    sample = pv.PolyData(np.asarray(point, dtype=float).reshape(1, 3)).sample(
        grid,
        tolerance=1.0e-9,
    )
    valid = bool(
        "vtkValidPointMask" in sample.point_data
        and int(np.asarray(sample.point_data["vtkValidPointMask"]).reshape(-1)[0]) == 1
    )
    value = None
    if valid and scalar_name in sample.point_data:
        value = float(np.asarray(sample.point_data[scalar_name]).reshape(-1)[0])
    return {
        "value": value,
        "sample_valid": valid,
        "containing_cell": int(grid.find_containing_cell(np.asarray(point, dtype=float))),
    }


def vertical_height_from_phi(
    grid: pv.DataSet,
    *,
    x: float,
    z: float,
    y_min: float = 0.0,
    y_max: float = 0.8,
    sample_count: int = 401,
) -> dict[str, Any]:
    y_values = np.linspace(y_min, y_max, sample_count)
    points = np.column_stack(
        [
            np.full_like(y_values, x),
            y_values,
            np.full_like(y_values, z),
        ]
    )
    sample = pv.PolyData(points).sample(grid, tolerance=1.0e-9)
    valid_mask = (
        np.asarray(sample.point_data.get("vtkValidPointMask", np.zeros(sample_count)))
        .reshape(-1)
        .astype(bool)
    )
    if "phi" not in sample.point_data or not np.any(valid_mask):
        return {
            "height_m": None,
            "status": "no_valid_phi_samples",
            "valid_sample_count": int(np.count_nonzero(valid_mask)),
        }

    valid_y = y_values[valid_mask]
    phi = np.asarray(sample.point_data["phi"], dtype=float).reshape(-1)[valid_mask]
    if np.all(phi > 0.0):
        return {
            "height_m": 0.0,
            "status": "dry_column",
            "valid_sample_count": int(valid_y.size),
        }
    if np.all(phi <= 0.0):
        return {
            "height_m": float(np.max(valid_y)),
            "status": "wet_to_sample_top",
            "valid_sample_count": int(valid_y.size),
        }

    crossings: list[float] = []
    for index in range(len(valid_y) - 1):
        y0 = float(valid_y[index])
        y1 = float(valid_y[index + 1])
        p0 = float(phi[index])
        p1 = float(phi[index + 1])
        if p0 == 0.0:
            crossings.append(y0)
        if p0 * p1 < 0.0:
            crossings.append(y0 - p0 * (y1 - y0) / (p1 - p0))
    if phi[-1] == 0.0:
        crossings.append(float(valid_y[-1]))
    if not crossings:
        return {
            "height_m": None,
            "status": "no_crossing",
            "valid_sample_count": int(valid_y.size),
        }
    return {
        "height_m": float(max(crossings)),
        "status": "crossing",
        "valid_sample_count": int(valid_y.size),
    }


def first_time_at_or_above(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:
    finite = np.isfinite(values)
    indices = np.flatnonzero(finite & (values >= threshold))
    if not indices.size:
        return None
    return float(times[int(indices[0])])


def peak_value(times: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"time_s": None, "value": None}
    finite_times = times[finite]
    finite_values = values[finite]
    index = int(np.argmax(finite_values))
    return {
        "time_s": float(finite_times[index]),
        "value": float(finite_values[index]),
    }


def late_window_trend(
    times: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float | None = None,
    window_s: float = 0.05,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"available": False, "reason": "no finite samples"}
    finite_times = times[finite]
    finite_values = values[finite]
    final_time = float(finite_times[-1])
    window = finite_times >= final_time - window_s - 1.0e-12
    if np.count_nonzero(window) < 2:
        return {"available": False, "reason": "fewer than two late-window samples"}

    trend_times = finite_times[window]
    trend_values = finite_values[window]
    slope, intercept = np.polyfit(trend_times, trend_values, 1)
    report: dict[str, Any] = {
        "available": True,
        "window_s": window_s,
        "sample_count": int(trend_times.size),
        "time_start_s": float(trend_times[0]),
        "time_end_s": final_time,
        "value_start": float(trend_values[0]),
        "value_end": float(trend_values[-1]),
        "slope_per_s": float(slope),
    }
    if threshold is not None:
        extrapolated_time: float | None = None
        if float(trend_values[-1]) >= threshold:
            extrapolated_time = final_time
        elif slope > 0.0:
            candidate = float((threshold - intercept) / slope)
            if candidate >= final_time:
                extrapolated_time = candidate
        report.update(
            {
                "threshold": threshold,
                "extrapolated_threshold_time_s": extrapolated_time,
                "extrapolated_seconds_after_final": (
                    extrapolated_time - final_time if extrapolated_time is not None else None
                ),
            }
        )
    return report


def pressure_history_metrics(
    times: np.ndarray,
    values: np.ndarray,
    reference: dict[str, np.ndarray],
    trace: str,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"available": False}
    finite_times = times[finite]
    finite_values = values[finite]
    reference_values = np.interp(finite_times, reference["Time"], reference[trace])
    errors = finite_values - reference_values
    peak = peak_value(finite_times, finite_values)
    return {
        "available": True,
        "first_ge_100Pa_s": first_time_at_or_above(finite_times, finite_values, 100.0),
        "simulated_peak_over_sample_window_pa": peak["value"],
        "simulated_peak_time_s": peak["time_s"],
        "reference_peak_over_sample_window_pa": float(np.max(reference_values)),
        "rmse_pa": float(math.sqrt(float(np.mean(errors * errors)))),
        "final_value_pa": float(finite_values[-1]),
        "final_reference_pa": float(reference_values[-1]),
    }


def reference_pressure_stack(reference: dict[str, np.ndarray], sample_end_s: float) -> dict[str, Any]:
    ref_times = reference["Time"]
    in_window = ref_times <= sample_end_s + 1.0e-12
    result: dict[str, Any] = {}
    for trace in PRIMARY_PRESSURE_TRACES:
        values = reference[trace][in_window]
        times = ref_times[in_window]
        peak = peak_value(times, values)
        result[trace] = {
            "first_ge_100Pa_s": first_time_at_or_above(ref_times, reference[trace], 100.0),
            "peak_over_sample_window_pa": peak["value"],
            "peak_time_over_sample_window_s": peak["time_s"],
        }
    p1_peak = result["P1"]["peak_over_sample_window_pa"]
    for trace in ("P3", "P5", "P7"):
        peak = result[trace]["peak_over_sample_window_pa"]
        result[trace]["peak_over_P1_peak_ratio"] = (
            float(peak / p1_peak) if p1_peak not in (None, 0.0) else None
        )
    return result


def result_times(case_dir: Path, prefix: str, solver_log: Path | None) -> tuple[dict[str, float], str]:
    pvd_times = verifier.result_times_from_pvd(case_dir, prefix)
    if pvd_times:
        return pvd_times, f"{prefix}.pvd"
    log_times = verifier.result_times_from_solver_log(solver_log, prefix)
    if log_times:
        return log_times, str(solver_log)
    return {}, "result_step_times_time_step_size"


def audit_case(
    case_dir: Path,
    reference_csv: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
) -> dict[str, Any]:
    setup = verifier.parse_solver_xml(case_dir / "solver.xml")
    reference = verifier.load_reference_csv(reference_csv)
    results = verifier.output_results(case_dir, result_prefix)
    time_by_result, time_source = result_times(case_dir, result_prefix, solver_log)
    sensors = verifier.pressure_sensors()
    column_offsets = (0.0, 0.005, 0.01, 0.02, 0.05)
    column_names = [f"x_plus_{offset:g}m" for offset in column_offsets]
    times: list[float] = []
    heights: dict[str, list[float]] = {name: [] for name in column_names}
    pressure: dict[str, list[float]] = {name: [] for name in PRIMARY_PRESSURE_TRACES}
    phi_at_sensor: dict[str, list[float]] = {name: [] for name in PRIMARY_PRESSURE_TRACES}

    for result in results:
        grid = pv.read(result)
        step = verifier.result_step(result, result_prefix)
        times.append(time_by_result.get(result.name, step * setup["time_step_size_s"]))
        for name, offset in zip(column_names, column_offsets):
            height = vertical_height_from_phi(
                grid,
                x=OBSTACLE_FRONT_X_M + offset,
                z=OBSTACLE_CENTER_Z_M,
            )["height_m"]
            heights[name].append(float(height) if height is not None else float("nan"))
        for trace in PRIMARY_PRESSURE_TRACES:
            point = sensors[trace].point
            pressure_sample = verifier.pressure_at_point(grid, point)
            pressure[trace].append(float(pressure_sample["pressure_pa"]))
            phi_sample = sample_scalar_at_point(grid, point, "phi")
            phi = phi_sample["value"]
            phi_at_sensor[trace].append(float(phi) if phi is not None else float("nan"))

    time_array = np.asarray(times, dtype=float)
    column_reports: dict[str, Any] = {}
    for name in column_names:
        values = np.asarray(heights[name], dtype=float)
        report = {
            "x_m": OBSTACLE_FRONT_X_M + column_offsets[column_names.index(name)],
            "peak_height": peak_value(time_array, values),
            "final_height_m": float(values[-1]) if np.isfinite(values[-1]) else None,
            "late_window_height_trend_to_P3": late_window_trend(
                time_array,
                values,
                threshold=FACE_SENSOR_HEIGHTS_M["P3"],
            ),
            "late_window_height_trend_to_obstacle_top": late_window_trend(
                time_array,
                values,
                threshold=OBSTACLE_TOP_Y_M,
            ),
            "first_height_ge_P1_y_s": first_time_at_or_above(
                time_array,
                values,
                FACE_SENSOR_HEIGHTS_M["P1"],
            ),
            "first_height_ge_P3_y_s": first_time_at_or_above(
                time_array,
                values,
                FACE_SENSOR_HEIGHTS_M["P3"],
            ),
            "first_height_ge_obstacle_top_s": first_time_at_or_above(
                time_array,
                values,
                OBSTACLE_TOP_Y_M,
            ),
        }
        column_reports[name] = report

    pressure_reports: dict[str, Any] = {}
    for trace in PRIMARY_PRESSURE_TRACES:
        values = np.asarray(pressure[trace], dtype=float)
        phi_values = np.asarray(phi_at_sensor[trace], dtype=float)
        pressure_reports[trace] = {
            **pressure_history_metrics(time_array, values, reference, trace),
            "late_window_pressure_trend": late_window_trend(time_array, values),
            "late_window_phi_trend_to_wet": late_window_trend(
                time_array,
                -phi_values,
                threshold=0.0,
            ),
            "first_phi_nonpositive_s": first_time_at_or_above(
                time_array,
                -phi_values,
                0.0,
            ),
            "final_phi": float(phi_values[-1]) if np.isfinite(phi_values[-1]) else None,
        }

    p1_peak = pressure_reports["P1"].get("simulated_peak_over_sample_window_pa")
    for trace in ("P3", "P5", "P7"):
        peak = pressure_reports[trace].get("simulated_peak_over_sample_window_pa")
        pressure_reports[trace]["simulated_peak_over_P1_peak_ratio"] = (
            float(peak / p1_peak) if p1_peak not in (None, 0.0) and peak is not None else None
        )

    reference_stack = reference_pressure_stack(reference, float(time_array[-1]))
    face_column = column_reports["x_plus_0.005m"]
    face_p3_trend = face_column["late_window_height_trend_to_P3"]
    face_top_trend = face_column["late_window_height_trend_to_obstacle_top"]
    p3_ratio = pressure_reports["P3"]["simulated_peak_over_P1_peak_ratio"]
    ref_p3_ratio = reference_stack["P3"]["peak_over_P1_peak_ratio"]
    face_final_height = face_column["final_height_m"]
    if face_final_height is None:
        face_p3_status = "not evaluated against P3 height"
    elif face_final_height >= FACE_SENSOR_HEIGHTS_M["P3"]:
        face_p3_status = "at or above P3 height by the saved cutoff"
    else:
        face_p3_status = "still below P3 height at the saved cutoff"
    finding = (
        "Obstacle-face coverage and pressure-stack diagnostics show whether the "
        "P3 pressure error is a wetting/coverage issue or a pressure-distribution "
        "issue. "
        f"At x={face_column['x_m']} m the interpolated face-adjacent water column is "
        f"{face_p3_status}, with final height {face_final_height} m. "
        "The late-window trend extrapolates "
        "P3-height wetting to "
        f"{face_p3_trend['extrapolated_threshold_time_s']} s and obstacle-top "
        f"run-up to {face_top_trend['extrapolated_threshold_time_s']} s. "
        f"The saved-window simulated P3/P1 peak ratio is {p3_ratio}, versus "
        f"reference-window ratio {ref_p3_ratio}."
    )

    return {
        "case_dir": str(case_dir),
        "reference_csv": str(reference_csv),
        "result_count": len(results),
        "sampled_time_start_s": float(time_array[0]) if len(time_array) else None,
        "sampled_time_end_s": float(time_array[-1]) if len(time_array) else None,
        "time_source": time_source,
        "sampled_histories": {
            "time_s": [float(value) for value in time_array],
            "obstacle_face_height_m": {
                name: [float(value) if math.isfinite(float(value)) else None for value in values]
                for name, values in heights.items()
            },
            "pressure_pa": {
                name: [float(value) if math.isfinite(float(value)) else None for value in values]
                for name, values in pressure.items()
            },
            "phi_at_pressure_sensor": {
                name: [float(value) if math.isfinite(float(value)) else None for value in values]
                for name, values in phi_at_sensor.items()
            },
        },
        "obstacle_face_columns": column_reports,
        "pressure_stack": pressure_reports,
        "reference_pressure_stack_over_sample_window": reference_stack,
        "finding": finding,
        "status": "diagnostic_obstacle_face_dynamics_not_validation_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = audit_case(
        args.case_dir,
        args.reference_csv,
        result_prefix=args.result_prefix,
        solver_log=args.solver_log,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
