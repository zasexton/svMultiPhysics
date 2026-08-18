#!/usr/bin/env python3
"""Sample SPHERIC Test02 obstacle-face pressure profiles at selected times."""

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
OBSTACLE_TOP_Y_M = 0.161
OBSTACLE_CENTER_Z_M = 0.5
OBSTACLE_LENGTH_M = 0.161
RHO_WATER_KG_PER_M3 = 1000.0
GRAVITY_M_PER_S2 = 9.81
PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")
FACE_PRESSURE_TRACES = ("P1", "P2", "P3", "P4")
TOP_PRESSURE_TRACES = ("P5", "P6", "P7", "P8")
ALL_PRESSURE_TRACES = FACE_PRESSURE_TRACES + TOP_PRESSURE_TRACES
DEFAULT_TARGET_TIMES_S = (
    0.43725,
    0.453,
    0.507,
    0.5191875,
    0.525,
    0.54,
)


def result_times(
    case_dir: Path,
    prefix: str,
    solver_log: Path | None,
) -> tuple[dict[str, float], str]:
    pvd_times = verifier.result_times_from_pvd(case_dir, prefix)
    if pvd_times:
        return pvd_times, f"{prefix}.pvd"
    log_times = verifier.result_times_from_solver_log(solver_log, prefix)
    if log_times:
        return log_times, str(solver_log)
    return {}, "result_step_times_time_step_size"


def result_time(
    result_path: Path,
    *,
    prefix: str,
    dt: float,
    time_by_result: dict[str, float],
) -> float:
    step = verifier.result_step(result_path, prefix)
    return float(time_by_result.get(result_path.name, step * dt))


def select_results_by_time(
    results: list[Path],
    times: np.ndarray,
    targets: tuple[float, ...],
) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    used: set[int] = set()
    for target in targets:
        index = int(np.argmin(np.abs(times - target)))
        if index in used:
            continue
        used.add(index)
        selections.append(
            {
                "result": results[index],
                "time_s": float(times[index]),
                "target_time_s": float(target),
                "target_time_error_s": float(times[index] - target),
            }
        )
    return selections


def sample_polyline(
    grid: pv.DataSet,
    points: np.ndarray,
) -> dict[str, np.ndarray]:
    sample = pv.PolyData(points).sample(grid, tolerance=1.0e-9)
    valid = (
        np.asarray(sample.point_data.get("vtkValidPointMask", np.zeros(points.shape[0])))
        .reshape(-1)
        .astype(bool)
    )
    arrays: dict[str, np.ndarray] = {"valid": valid}
    for name in ("Pressure", "phi", "ActiveFluid", "Velocity"):
        if name in sample.point_data:
            arrays[name] = np.asarray(sample.point_data[name], dtype=float)
    return arrays


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def zero_crossing_height(y_values: np.ndarray, phi: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    valid_y = y_values[valid]
    valid_phi = phi[valid]
    if not valid_y.size:
        return {"height_m": None, "status": "no_valid_samples"}
    if np.all(valid_phi > 0.0):
        return {"height_m": 0.0, "status": "dry_line"}
    if np.all(valid_phi <= 0.0):
        return {"height_m": float(np.max(valid_y)), "status": "wet_to_sample_top"}

    crossings: list[float] = []
    for index in range(len(valid_y) - 1):
        y0 = float(valid_y[index])
        y1 = float(valid_y[index + 1])
        p0 = float(valid_phi[index])
        p1 = float(valid_phi[index + 1])
        if p0 == 0.0:
            crossings.append(y0)
        if p0 * p1 < 0.0:
            crossings.append(y0 - p0 * (y1 - y0) / (p1 - p0))
    if valid_phi[-1] == 0.0:
        crossings.append(float(valid_y[-1]))
    if not crossings:
        return {"height_m": None, "status": "no_crossing"}
    return {"height_m": float(max(crossings)), "status": "crossing"}


def sample_point_arrays(
    grid: pv.DataSet,
    point: tuple[float, float, float],
) -> dict[str, Any]:
    target = np.asarray(point, dtype=float).reshape(1, 3)
    sample = pv.PolyData(target).sample(grid, tolerance=1.0e-9)
    valid = bool(
        "vtkValidPointMask" in sample.point_data
        and int(np.asarray(sample.point_data["vtkValidPointMask"]).reshape(-1)[0]) == 1
    )
    result: dict[str, Any] = {
        "point_m": [float(value) for value in target.reshape(3).tolist()],
        "sample_valid": valid,
        "containing_cell": int(grid.find_containing_cell(target.reshape(3))),
    }
    for name in ("Pressure", "phi", "ActiveFluid"):
        if valid and name in sample.point_data:
            result[name] = float(np.asarray(sample.point_data[name]).reshape(-1)[0])
    if valid and "Velocity" in sample.point_data:
        velocity = np.asarray(sample.point_data["Velocity"], dtype=float).reshape(-1)
        result["Velocity"] = [float(value) for value in velocity.tolist()]
        result["speed_m_per_s"] = float(np.linalg.norm(velocity))
    return result


def pressure_stats(
    coordinate: np.ndarray,
    pressure: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    finite = valid & np.isfinite(pressure)
    if not np.any(finite):
        return {"available": False, "reason": "no finite valid pressure samples"}
    selected_pressure = pressure[finite]
    selected_coordinate = coordinate[finite]
    max_index = int(np.argmax(selected_pressure))
    min_index = int(np.argmin(selected_pressure))
    return {
        "available": True,
        "valid_sample_count": int(np.count_nonzero(finite)),
        "pressure_max_pa": float(selected_pressure[max_index]),
        "pressure_max_coordinate_m": float(selected_coordinate[max_index]),
        "pressure_min_pa": float(selected_pressure[min_index]),
        "pressure_min_coordinate_m": float(selected_coordinate[min_index]),
        "pressure_mean_pa": float(np.mean(selected_pressure)),
    }


def vertex_pressure_stats(pressure: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = pressure[mask & np.isfinite(pressure)]
    if selected.size == 0:
        return {"available": False, "count": 0}
    return {
        "available": True,
        "count": int(selected.size),
        "pressure_min_pa": float(np.min(selected)),
        "pressure_max_pa": float(np.max(selected)),
        "pressure_mean_pa": float(np.mean(selected)),
    }


def hydrostatic_pressure(height_m: float | None, y_m: float) -> float | None:
    if height_m is None or height_m <= y_m:
        return 0.0
    return float(RHO_WATER_KG_PER_M3 * GRAVITY_M_PER_S2 * (height_m - y_m))


def vertical_profile(
    grid: pv.DataSet,
    *,
    x_m: float,
    y_values: np.ndarray,
) -> dict[str, Any]:
    points = np.column_stack(
        [
            np.full_like(y_values, x_m),
            y_values,
            np.full_like(y_values, OBSTACLE_CENTER_Z_M),
        ]
    )
    arrays = sample_polyline(grid, points)
    valid = arrays["valid"]
    pressure = arrays.get("Pressure", np.full_like(y_values, np.nan, dtype=float)).reshape(-1)
    phi = arrays.get("phi", np.full_like(y_values, np.nan, dtype=float)).reshape(-1)
    height = zero_crossing_height(y_values, phi, valid)
    sensors = verifier.pressure_sensors()
    sensor_samples: dict[str, Any] = {}
    for trace in FACE_PRESSURE_TRACES:
        sensor_y = sensors[trace].point[1]
        sample = sample_point_arrays(grid, (x_m, sensor_y, OBSTACLE_CENTER_Z_M))
        pressure_pa = sample.get("Pressure")
        hydrostatic_pa = hydrostatic_pressure(height["height_m"], sensor_y)
        sensor_samples[trace] = {
            **sample,
            "hydrostatic_from_local_height_pa": hydrostatic_pa,
            "pressure_minus_hydrostatic_pa": (
                float(pressure_pa - hydrostatic_pa)
                if pressure_pa is not None and hydrostatic_pa is not None
                else None
            ),
        }
    p1 = sensor_samples["P1"].get("Pressure")
    p3 = sensor_samples["P3"].get("Pressure")
    return {
        "x_m": float(x_m),
        "height_from_phi": height,
        "pressure_stats": pressure_stats(y_values, pressure, valid),
        "sensor_samples": sensor_samples,
        "P3_over_P1_pressure_ratio": (
            float(p3 / p1) if p1 not in (None, 0.0) and p3 is not None else None
        ),
        "sampled_y_m": [float(value) for value in y_values.tolist()],
        "pressure_pa": [finite_or_none(value) for value in pressure.tolist()],
        "phi": [finite_or_none(value) for value in phi.tolist()],
        "valid": [bool(value) for value in valid.tolist()],
    }


def sensor_support_cell(
    grid: pv.DataSet,
    point: tuple[float, float, float],
    *,
    reference_pressure_pa: float | None,
) -> dict[str, Any]:
    target = np.asarray(point, dtype=float)
    cell_id = int(grid.find_containing_cell(target))
    report: dict[str, Any] = {
        "point_m": [float(value) for value in target.tolist()],
        "containing_cell": cell_id,
    }
    if cell_id < 0:
        report["available"] = False
        report["reason"] = "sensor point is outside the result grid"
        return report

    cell = grid.get_cell(cell_id)
    point_ids = np.asarray(cell.point_ids, dtype=int)
    points = np.asarray(grid.points, dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    phi = (
        np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
        if "phi" in grid.point_data
        else np.full(grid.n_points, np.nan)
    )
    active = (
        np.asarray(grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
        if "ActiveFluid" in grid.point_data
        else np.full(grid.n_points, np.nan)
    )
    vertex_pressure = pressure[point_ids]
    vertex_phi = phi[point_ids]
    vertex_active = active[point_ids]
    wet_mask = vertex_phi <= 0.0
    active_mask = vertex_active > 0.5
    active_wet_mask = active_mask & wet_mask
    all_mask = np.ones(point_ids.shape[0], dtype=bool)

    vertices = []
    for local_index, point_id in enumerate(point_ids):
        vertices.append(
            {
                "local_index": int(local_index),
                "point_id": int(point_id),
                "point_m": [float(value) for value in points[point_id].tolist()],
                "Pressure": float(pressure[point_id]),
                "phi": finite_or_none(phi[point_id]),
                "ActiveFluid": finite_or_none(active[point_id]),
                "wet_phi_le_0": bool(wet_mask[local_index]),
                "active_fluid_gt_0p5": bool(active_mask[local_index]),
            }
        )

    cell_wet_fraction = None
    if "WetVolumeFraction" in grid.cell_data:
        cell_wet_fraction = float(np.asarray(grid.cell_data["WetVolumeFraction"]).reshape(-1)[cell_id])

    stats = {
        "all_vertices": vertex_pressure_stats(vertex_pressure, all_mask),
        "active_vertices": vertex_pressure_stats(vertex_pressure, active_mask),
        "wet_vertices": vertex_pressure_stats(vertex_pressure, wet_mask),
        "active_wet_vertices": vertex_pressure_stats(vertex_pressure, active_wet_mask),
        "dry_or_inactive_vertices": vertex_pressure_stats(
            vertex_pressure,
            ~(active_wet_mask),
        ),
    }
    active_stats = stats["active_wet_vertices"]
    if not active_stats.get("available"):
        active_stats = stats["active_vertices"]
    if reference_pressure_pa not in (None, 0.0) and active_stats.get("available"):
        active_stats["pressure_mean_over_reference"] = float(
            active_stats["pressure_mean_pa"] / reference_pressure_pa
        )
        active_stats["pressure_max_over_reference"] = float(
            active_stats["pressure_max_pa"] / reference_pressure_pa
        )

    return {
        **report,
        "available": True,
        "cell_type": int(cell.type),
        "point_ids": [int(value) for value in point_ids.tolist()],
        "cell_wet_volume_fraction": cell_wet_fraction,
        "vertex_count": int(point_ids.size),
        "active_vertex_count": int(np.count_nonzero(active_mask)),
        "wet_vertex_count": int(np.count_nonzero(wet_mask)),
        "active_wet_vertex_count": int(np.count_nonzero(active_wet_mask)),
        "reference_pressure_pa": reference_pressure_pa,
        "pressure_stats": stats,
        "vertices": vertices,
    }


def top_profile(
    grid: pv.DataSet,
    *,
    x_values: np.ndarray,
) -> dict[str, Any]:
    points = np.column_stack(
        [
            x_values,
            np.full_like(x_values, OBSTACLE_TOP_Y_M),
            np.full_like(x_values, OBSTACLE_CENTER_Z_M),
        ]
    )
    arrays = sample_polyline(grid, points)
    valid = arrays["valid"]
    pressure = arrays.get("Pressure", np.full_like(x_values, np.nan, dtype=float)).reshape(-1)
    phi = arrays.get("phi", np.full_like(x_values, np.nan, dtype=float)).reshape(-1)
    sensors = verifier.pressure_sensors()
    sensor_samples: dict[str, Any] = {}
    for trace in TOP_PRESSURE_TRACES:
        sensor_samples[trace] = sample_point_arrays(grid, sensors[trace].point)
    p5 = sensor_samples["P5"].get("Pressure")
    p7 = sensor_samples["P7"].get("Pressure")
    return {
        "y_m": OBSTACLE_TOP_Y_M,
        "x_min_m": float(np.min(x_values)),
        "x_max_m": float(np.max(x_values)),
        "pressure_stats": pressure_stats(x_values, pressure, valid),
        "sensor_samples": sensor_samples,
        "P7_over_P5_pressure_ratio": (
            float(p7 / p5) if p5 not in (None, 0.0) and p7 is not None else None
        ),
        "sampled_x_m": [float(value) for value in x_values.tolist()],
        "pressure_pa": [finite_or_none(value) for value in pressure.tolist()],
        "phi": [finite_or_none(value) for value in phi.tolist()],
        "valid": [bool(value) for value in valid.tolist()],
    }


def reference_at_time(reference: dict[str, np.ndarray], time_s: float) -> dict[str, Any]:
    values = {
        trace: float(np.interp(time_s, reference["Time"], reference[trace]))
        for trace in ALL_PRESSURE_TRACES
        if trace in reference
    }
    p1 = values.get("P1")
    p5 = values.get("P5")
    values["P3_over_P1_pressure_ratio"] = (
        float(values["P3"] / p1) if p1 not in (None, 0.0) and "P3" in values else None
    )
    values["P7_over_P5_pressure_ratio"] = (
        float(values["P7"] / p5) if p5 not in (None, 0.0) and "P7" in values else None
    )
    return values


def sample_event(
    grid: pv.DataSet,
    *,
    time_s: float,
    target_time_s: float,
    target_time_error_s: float,
    result: Path,
    reference: dict[str, np.ndarray],
    vertical_x_offsets_m: tuple[float, ...],
    y_sample_count: int,
    x_sample_count: int,
) -> dict[str, Any]:
    y_values = np.linspace(0.0, OBSTACLE_TOP_Y_M, y_sample_count)
    top_x = np.linspace(
        OBSTACLE_FRONT_X_M - OBSTACLE_LENGTH_M,
        OBSTACLE_FRONT_X_M,
        x_sample_count,
    )
    vertical = {
        f"x_plus_{offset:g}m": vertical_profile(
            grid,
            x_m=OBSTACLE_FRONT_X_M + offset,
            y_values=y_values,
        )
        for offset in vertical_x_offsets_m
    }
    reference_pressure = reference_at_time(reference, time_s)
    official_sensor_pressure = {
        trace: sample_point_arrays(grid, verifier.pressure_sensors()[trace].point)
        for trace in PRIMARY_PRESSURE_TRACES
    }
    for trace, sample in official_sensor_pressure.items():
        pressure_pa = sample.get("Pressure")
        reference_pa = reference_pressure[trace]
        sample["reference_pressure_pa"] = reference_pa
        sample["pressure_error_pa"] = (
            float(pressure_pa - reference_pa) if pressure_pa is not None else None
        )
    sensor_support_cells = {
        trace: sensor_support_cell(
            grid,
            verifier.pressure_sensors()[trace].point,
            reference_pressure_pa=reference_pressure.get(trace),
        )
        for trace in ALL_PRESSURE_TRACES
    }
    p1 = official_sensor_pressure["P1"].get("Pressure")
    p3 = official_sensor_pressure["P3"].get("Pressure")
    p5 = official_sensor_pressure["P5"].get("Pressure")
    p7 = official_sensor_pressure["P7"].get("Pressure")
    return {
        "result": str(result),
        "time_s": float(time_s),
        "target_time_s": float(target_time_s),
        "target_time_error_s": float(target_time_error_s),
        "reference_pressure_pa": reference_pressure,
        "official_sensor_pressure": official_sensor_pressure,
        "sensor_support_cells": sensor_support_cells,
        "official_P3_over_P1_pressure_ratio": (
            float(p3 / p1) if p1 not in (None, 0.0) and p3 is not None else None
        ),
        "official_P7_over_P5_pressure_ratio": (
            float(p7 / p5) if p5 not in (None, 0.0) and p7 is not None else None
        ),
        "vertical_profiles": vertical,
        "top_profile": top_profile(grid, x_values=top_x),
    }


def event_with_max_sensor_pressure(
    events: list[dict[str, Any]],
    trace: str,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_value = -math.inf
    for event in events:
        value = event["official_sensor_pressure"][trace].get("Pressure")
        if value is not None and value > best_value:
            best_value = float(value)
            best = event
    return best


def audit_case(
    case_dir: Path,
    reference_csv: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
    target_times_s: tuple[float, ...],
    vertical_x_offsets_m: tuple[float, ...],
    y_sample_count: int,
    x_sample_count: int,
) -> dict[str, Any]:
    setup = verifier.parse_solver_xml(case_dir / "solver.xml")
    reference = verifier.load_reference_csv(reference_csv)
    results = verifier.output_results(case_dir, result_prefix)
    if not results:
        raise RuntimeError(f"no result files found in {case_dir}")

    time_by_result, time_source = result_times(case_dir, result_prefix, solver_log)
    times = np.asarray(
        [
            result_time(
                result,
                prefix=result_prefix,
                dt=setup["time_step_size_s"],
                time_by_result=time_by_result,
            )
            for result in results
        ],
        dtype=float,
    )
    selections = select_results_by_time(results, times, target_times_s)
    events = [
        sample_event(
            pv.read(selection["result"]),
            time_s=selection["time_s"],
            target_time_s=selection["target_time_s"],
            target_time_error_s=selection["target_time_error_s"],
            result=selection["result"],
            reference=reference,
            vertical_x_offsets_m=vertical_x_offsets_m,
            y_sample_count=y_sample_count,
            x_sample_count=x_sample_count,
        )
        for selection in selections
    ]

    p1_event = event_with_max_sensor_pressure(events, "P1")
    p3_event = event_with_max_sensor_pressure(events, "P3")
    final_event = events[-1]
    finding_parts = [
        "Official Test02 pressure-sensor layout matches the benchmark PDF: "
        "front-face P1-P4 start 0.021 m above the obstacle bottom with 0.04 m spacing, "
        "and top-face P5-P8 start 0.021 m behind the front edge with 0.04 m spacing."
    ]
    if p1_event is not None:
        finding_parts.append(
            "At the sampled P1-peak event "
            f"t={p1_event['time_s']} s, simulated P3/P1 is "
            f"{p1_event['official_P3_over_P1_pressure_ratio']} while the time-matched "
            f"reference ratio is {p1_event['reference_pressure_pa']['P3_over_P1_pressure_ratio']}."
        )
    if p3_event is not None:
        finding_parts.append(
            "At the sampled P3-peak event "
            f"t={p3_event['time_s']} s, simulated P3/P1 is "
            f"{p3_event['official_P3_over_P1_pressure_ratio']} while the time-matched "
            f"reference ratio is {p3_event['reference_pressure_pa']['P3_over_P1_pressure_ratio']}."
        )
    finding_parts.append(
        "At the final sampled event "
        f"t={final_event['time_s']} s, the face-line local height at x=front is "
        f"{final_event['vertical_profiles']['x_plus_0m']['height_from_phi']['height_m']} m "
        "and the official P3/P1 ratio is "
        f"{final_event['official_P3_over_P1_pressure_ratio']} versus time-matched reference "
        f"{final_event['reference_pressure_pa']['P3_over_P1_pressure_ratio']}."
    )
    final_p3_support = final_event["sensor_support_cells"]["P3"]
    final_p3_active_stats = final_p3_support["pressure_stats"]["active_wet_vertices"]
    if not final_p3_active_stats.get("available"):
        final_p3_active_stats = final_p3_support["pressure_stats"]["active_vertices"]
    final_p3_reference = final_event["reference_pressure_pa"].get("P3")
    final_p3_active_ratio = (
        float(final_p3_active_stats["pressure_mean_pa"] / final_p3_reference)
        if final_p3_active_stats.get("available") and final_p3_reference not in (None, 0.0)
        else None
    )
    finding_parts.append(
        "The final P3 containing cell has "
        f"{final_p3_support['active_wet_vertex_count']} active-wet vertices; their mean "
        f"pressure/reference ratio is {final_p3_active_ratio}. This distinguishes the "
        "remaining pressure-stack error from a pure inactive-vertex interpolation artifact."
    )

    return {
        "case_dir": str(case_dir),
        "reference_csv": str(reference_csv),
        "result_count": len(results),
        "sampled_time_start_s": float(times[0]),
        "sampled_time_end_s": float(times[-1]),
        "time_source": time_source,
        "target_times_s": [float(value) for value in target_times_s],
        "vertical_x_offsets_m": [float(value) for value in vertical_x_offsets_m],
        "y_sample_count": int(y_sample_count),
        "x_sample_count": int(x_sample_count),
        "events": events,
        "summary": {
            "sampled_P1_peak_event_time_s": (
                p1_event["time_s"] if p1_event is not None else None
            ),
            "sampled_P3_peak_event_time_s": (
                p3_event["time_s"] if p3_event is not None else None
            ),
            "sampled_P1_peak_P3_over_P1_ratio": (
                p1_event["official_P3_over_P1_pressure_ratio"]
                if p1_event is not None
                else None
            ),
            "sampled_P3_peak_P3_over_P1_ratio": (
                p3_event["official_P3_over_P1_pressure_ratio"]
                if p3_event is not None
                else None
            ),
            "final_sample_time_s": final_event["time_s"],
            "final_sample_P3_over_P1_ratio": final_event[
                "official_P3_over_P1_pressure_ratio"
            ],
            "final_sample_reference_P3_over_P1_ratio": final_event[
                "reference_pressure_pa"
            ]["P3_over_P1_pressure_ratio"],
            "final_face_height_at_front_m": final_event["vertical_profiles"][
                "x_plus_0m"
            ]["height_from_phi"]["height_m"],
            "final_P3_active_wet_vertex_pressure_mean_over_reference": final_p3_active_ratio,
            "final_P3_active_wet_vertex_count": final_p3_support["active_wet_vertex_count"],
        },
        "finding": " ".join(finding_parts),
        "status": "diagnostic_obstacle_pressure_profile_not_validation_gate",
    }


def parse_float_tuple(values: list[float] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if not values:
        return default
    return tuple(float(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=verifier.DEFAULT_REFERENCE_CSV,
    )
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--target-time", type=float, action="append", dest="target_times")
    parser.add_argument(
        "--vertical-x-offset",
        type=float,
        action="append",
        dest="vertical_x_offsets",
    )
    parser.add_argument("--y-sample-count", type=int, default=81)
    parser.add_argument("--x-sample-count", type=int, default=81)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    report = audit_case(
        args.case_dir,
        args.reference_csv,
        result_prefix=args.result_prefix,
        solver_log=args.solver_log,
        target_times_s=parse_float_tuple(args.target_times, DEFAULT_TARGET_TIMES_S),
        vertical_x_offsets_m=parse_float_tuple(args.vertical_x_offsets, (0.0, 0.005, 0.02)),
        y_sample_count=args.y_sample_count,
        x_sample_count=args.x_sample_count,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
