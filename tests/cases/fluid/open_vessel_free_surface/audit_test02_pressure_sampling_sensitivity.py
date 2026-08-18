#!/usr/bin/env python3
"""Audit SPHERIC Test02 pressure-sensor sampling sensitivity.

This is a diagnostic companion to verify_spheric_test02_histories.py. It keeps
the official sensor coordinates as the primary comparison, then samples nearby
fluid-side offsets and local node neighborhoods to check whether pressure
history errors are caused by exact boundary sampling choices.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

import verify_spheric_test02_histories as verifier


PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")
PRESSURE_THRESHOLD_PA = verifier.PRESSURE_RESPONSE_THRESHOLD_PA


def point_variants(name: str, point: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
    x, y, z = point
    variants = {"target": point}
    if name in {"P1", "P2", "P3", "P4"}:
        for offset in (0.005, 0.01, 0.02, 0.05):
            variants[f"fluid_side_x_plus_{offset:g}m"] = (x + offset, y, z)
    else:
        for offset in (0.005, 0.01, 0.02, 0.05):
            variants[f"fluid_side_y_plus_{offset:g}m"] = (x, y + offset, z)
    return variants


def sample_point_pressure(grid: pv.DataSet, point: tuple[float, float, float]) -> dict[str, Any]:
    sample = verifier.pressure_at_point(grid, point)
    return {
        "pressure_pa": sample["pressure_pa"],
        "sample_valid": bool(sample.get("sample_valid", False)),
        "containing_cell": int(sample.get("containing_cell", -1)),
        "nearest_node": sample.get("nearest_node", sample),
    }


def local_node_pressure(
    grid: pv.DataSet,
    sensor_name: str,
    point: tuple[float, float, float],
    *,
    radius: float,
) -> dict[str, Any]:
    points = np.asarray(grid.points, dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    phi = (
        np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
        if "phi" in grid.point_data
        else np.full(len(points), -1.0)
    )
    x, y, z = point
    if sensor_name in {"P1", "P2", "P3", "P4"}:
        mask = (
            (points[:, 0] >= x - 1.0e-12)
            & (points[:, 0] <= x + radius + 1.0e-12)
            & (np.abs(points[:, 1] - y) <= radius + 1.0e-12)
            & (np.abs(points[:, 2] - z) <= radius + 1.0e-12)
        )
    else:
        mask = (
            (points[:, 1] >= y - 1.0e-12)
            & (points[:, 1] <= y + radius + 1.0e-12)
            & (np.abs(points[:, 0] - x) <= radius + 1.0e-12)
            & (np.abs(points[:, 2] - z) <= radius + 1.0e-12)
        )
    wet_mask = mask & (phi <= 0.0)
    selected = np.flatnonzero(wet_mask if np.any(wet_mask) else mask)
    if selected.size == 0:
        return {
            "pressure_pa": None,
            "selected_count": 0,
            "used_wet_filter": bool(np.any(wet_mask)),
            "radius_m": radius,
        }
    local_pressure = pressure[selected]
    local_index = int(selected[int(np.argmax(local_pressure))])
    return {
        "pressure_pa": float(pressure[local_index]),
        "selected_count": int(selected.size),
        "used_wet_filter": bool(np.any(wet_mask)),
        "radius_m": radius,
        "selected_index": local_index,
        "selected_point": points[local_index].tolist(),
        "sensor_distance_m": float(np.linalg.norm(points[local_index] - np.asarray(point))),
    }


def first_time_at_or_above(times: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    finite = np.isfinite(values)
    indices = np.flatnonzero(finite & (values >= threshold))
    if not indices.size:
        return None
    return float(times[int(indices[0])])


def history_metrics(
    times: np.ndarray,
    values: np.ndarray,
    reference: dict[str, np.ndarray],
    trace: str,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"available": False, "reason": "no finite samples"}
    sample_times = times[finite]
    sample_values = values[finite]
    reference_values = np.interp(sample_times, reference["Time"], reference[trace])
    errors = sample_values - reference_values
    peak_index = int(np.argmax(sample_values))
    return {
        "available": True,
        "sample_count": int(sample_values.size),
        "first_ge_100Pa_s": first_time_at_or_above(sample_times, sample_values, PRESSURE_THRESHOLD_PA),
        "simulated_peak_over_sample_window_pa": float(sample_values[peak_index]),
        "simulated_peak_time_s": float(sample_times[peak_index]),
        "reference_peak_over_sample_window_pa": float(np.max(reference_values)),
        "rmse_pa": float(math.sqrt(float(np.mean(errors * errors)))),
        "final_value_pa": float(sample_values[-1]),
        "final_reference_pa": float(reference_values[-1]),
    }


def reference_event(reference: dict[str, np.ndarray], trace: str) -> dict[str, Any]:
    times = reference["Time"]
    values = reference[trace]
    peak_index = int(np.argmax(values))
    return {
        "first_ge_100Pa_s": first_time_at_or_above(times, values, PRESSURE_THRESHOLD_PA),
        "peak_time_s": float(times[peak_index]),
        "peak_value_pa": float(values[peak_index]),
    }


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
    pressure_traces: tuple[str, ...],
) -> dict[str, Any]:
    setup = verifier.parse_solver_xml(case_dir / "solver.xml")
    reference = verifier.load_reference_csv(reference_csv)
    results = verifier.output_results(case_dir, result_prefix)
    time_by_result, time_source = result_times(case_dir, result_prefix, solver_log)
    sensors = verifier.pressure_sensors()
    times: list[float] = []
    series: dict[str, dict[str, list[float]]] = {}
    sample_metadata: dict[str, dict[str, Any]] = {}
    variants_by_sensor = {
        name: point_variants(name, sensors[name].point) for name in pressure_traces
    }

    for result in results:
        grid = pv.read(result)
        step = verifier.result_step(result, result_prefix)
        times.append(time_by_result.get(result.name, step * setup["time_step_size_s"]))
        for name in pressure_traces:
            series.setdefault(name, {})
            sample_metadata.setdefault(name, {})
            for variant_name, point in variants_by_sensor[name].items():
                method_name = f"point:{variant_name}"
                sample = sample_point_pressure(grid, point)
                series[name].setdefault(method_name, []).append(float(sample["pressure_pa"]))
                sample_metadata[name].setdefault(method_name, sample)
                nearest = sample["nearest_node"]
                nearest_name = f"nearest_node_for_{variant_name}"
                series[name].setdefault(nearest_name, []).append(float(nearest["pressure_pa"]))
                sample_metadata[name].setdefault(nearest_name, nearest)
            for radius in (0.05, 0.1):
                method_name = f"local_fluid_side_node_max_r{radius:g}m"
                sample = local_node_pressure(grid, name, sensors[name].point, radius=radius)
                value = sample["pressure_pa"]
                series[name].setdefault(method_name, []).append(
                    float(value) if value is not None else float("nan")
                )
                sample_metadata[name].setdefault(method_name, sample)

    time_array = np.asarray(times, dtype=float)
    sensor_reports: dict[str, Any] = {}
    for name in pressure_traces:
        methods: dict[str, Any] = {}
        best_method = None
        best_peak = -math.inf
        for method_name, values in series.get(name, {}).items():
            metrics = history_metrics(
                time_array,
                np.asarray(values, dtype=float),
                reference,
                name,
            )
            metrics["representative_sample_metadata"] = sample_metadata[name][method_name]
            methods[method_name] = metrics
            if metrics.get("available") and metrics["simulated_peak_over_sample_window_pa"] > best_peak:
                best_peak = float(metrics["simulated_peak_over_sample_window_pa"])
                best_method = method_name
        target_peak = methods.get("point:target", {}).get("simulated_peak_over_sample_window_pa")
        best_peak_value = methods.get(best_method, {}).get("simulated_peak_over_sample_window_pa") if best_method else None
        sensor_reports[name] = {
            "reference_event": reference_event(reference, name),
            "official_target_point_m": list(sensors[name].point),
            "methods": methods,
            "best_peak_method": best_method,
            "best_peak_over_target_peak_ratio": (
                float(best_peak_value / target_peak)
                if target_peak not in (None, 0.0) and best_peak_value is not None
                else None
            ),
        }

    findings: list[str] = []
    for name, report in sensor_reports.items():
        target = report["methods"].get("point:target", {})
        best = report["methods"].get(report["best_peak_method"], {})
        ref_window_peak = target.get("reference_peak_over_sample_window_pa")
        if not target.get("available") or not best.get("available") or not ref_window_peak:
            continue
        target_fraction = target["simulated_peak_over_sample_window_pa"] / ref_window_peak
        best_fraction = best["simulated_peak_over_sample_window_pa"] / ref_window_peak
        if best_fraction >= 0.5 and target_fraction < 0.25:
            findings.append(
                f"{name}: nearby sampling recovers a substantial pressure peak "
                f"({best_fraction:.3g} of reference-window peak versus "
                f"{target_fraction:.3g} at the official point), so sampling is material."
            )
        else:
            findings.append(
                f"{name}: strongest nearby sample reaches {best_fraction:.3g} of "
                f"the reference-window peak; exact-point sampling alone does not explain the error."
            )

    return {
        "case_dir": str(case_dir),
        "reference_csv": str(reference_csv),
        "result_count": len(results),
        "sampled_time_start_s": float(time_array[0]) if len(time_array) else None,
        "sampled_time_end_s": float(time_array[-1]) if len(time_array) else None,
        "time_source": time_source,
        "pressure_traces": list(pressure_traces),
        "sensors": sensor_reports,
        "finding": " ".join(findings),
        "status": "diagnostic_pressure_sampling_sensitivity_not_validation_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--pressure-traces", default=",".join(PRIMARY_PRESSURE_TRACES))
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    pressure_traces = tuple(
        name.strip() for name in args.pressure_traces.split(",") if name.strip()
    )
    report = audit_case(
        args.case_dir,
        args.reference_csv,
        result_prefix=args.result_prefix,
        solver_log=args.solver_log,
        pressure_traces=pressure_traces,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
