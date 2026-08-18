#!/usr/bin/env python3
"""Compare SPHERIC Test02 result histories with official H/P traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import xml.etree.ElementTree as ET
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
DEFAULT_REFERENCE_CSV = (
    SCRIPT_DIR.parents[3]
    / "Documentation"
    / "qualification_logs"
    / "open_vessel_free_surface_remaining_20260526"
    / "test02_reference_histories_20260602.csv"
)
HEIGHT_PROBES = {
    "H1": 0.496,
    "H2": 0.992,
    "H3": 1.488,
    "H4": 2.632,
}
PRIMARY_HEIGHT_TRACES = ("H4", "H2")
PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")
HEIGHT_RESPONSE_THRESHOLD_M = 0.005
PRESSURE_RESPONSE_THRESHOLD_PA = 100.0


@dataclass(frozen=True)
class PressureSensor:
    name: str
    point: tuple[float, float, float]


def pressure_sensors() -> dict[str, PressureSensor]:
    obstacle_x_front = 0.8245
    obstacle_top = 0.161
    center_z = 0.5
    face_heights = {
        "P1": 0.021,
        "P2": 0.061,
        "P3": 0.101,
        "P4": 0.141,
    }
    top_offsets = {
        "P5": 0.021,
        "P6": 0.061,
        "P7": 0.101,
        "P8": 0.141,
    }
    sensors = {
        name: PressureSensor(name, (obstacle_x_front, height, center_z))
        for name, height in face_heights.items()
    }
    sensors.update(
        {
            name: PressureSensor(
                name,
                (obstacle_x_front - offset, obstacle_top, center_z),
            )
            for name, offset in top_offsets.items()
        }
    )
    return sensors


def child_text(parent: ET.Element, path: str) -> str | None:
    child = parent.find(path)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_solver_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    general = root.find("GeneralSimulationParameters")
    if general is None:
        raise RuntimeError(f"solver.xml is missing GeneralSimulationParameters: {path}")
    return {
        "number_of_time_steps": int(child_text(general, "Number_of_time_steps") or 0),
        "time_step_size_s": float(child_text(general, "Time_step_size") or 0.0),
    }


def output_results(case_dir: Path, prefix: str) -> list[Path]:
    def step(path: Path) -> int:
        match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(
        [*case_dir.glob(f"{prefix}_*.vtu"), *case_dir.glob(f"{prefix}_*.pvtu")],
        key=step,
    )


def result_step(path: Path, prefix: str) -> int:
    match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", path.name)
    if not match:
        raise RuntimeError(f"result name does not match prefix {prefix!r}: {path}")
    return int(match.group(1))


def result_times_from_pvd(case_dir: Path, prefix: str) -> dict[str, float]:
    pvd_path = case_dir / f"{prefix}.pvd"
    if not pvd_path.exists():
        return {}

    root = ET.parse(pvd_path).getroot()
    times: dict[str, float] = {}
    for dataset in root.findall(".//DataSet"):
        file_name = dataset.attrib.get("file")
        time_text = dataset.attrib.get("timestep")
        if not file_name or time_text is None:
            continue
        times[Path(file_name).name] = float(time_text)
    return times


def result_times_from_solver_log(path: Path | None, prefix: str) -> dict[str, float]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    times: dict[str, float] = {}
    for match in re.finditer(
        r"step_accepted step=(\d+) time=([0-9.eE+-]+)",
        text,
    ):
        step = int(match.group(1))
        times[f"{prefix}_{step:03d}.vtu"] = float(match.group(2))
        times[f"{prefix}_{step:03d}.pvtu"] = float(match.group(2))
    return times


def load_reference_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"reference CSV has no rows: {path}")
    names = rows[0].keys()
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in names
    }


def nearest_probe_column(
    points: np.ndarray,
    *,
    x: float,
    z: float,
    tolerance: float,
) -> np.ndarray:
    distance = np.hypot(points[:, 0] - x, points[:, 2] - z)
    selected = np.flatnonzero(distance <= tolerance)
    if selected.size:
        return selected
    nearest = float(np.min(distance))
    return np.flatnonzero(distance <= nearest + 1.0e-12)


def height_from_phi_column(
    points: np.ndarray,
    phi: np.ndarray,
    *,
    x: float,
    z: float = 0.5,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    column = nearest_probe_column(points, x=x, z=z, tolerance=tolerance)
    column_points = points[column]
    column_phi = phi[column]
    order = np.argsort(column_points[:, 1])
    y_values = column_points[order, 1]
    phi_values = column_phi[order]
    unique_y: list[float] = []
    unique_phi: list[float] = []
    for y, value in zip(y_values, phi_values):
        if unique_y and abs(float(y) - unique_y[-1]) <= 1.0e-12:
            unique_phi[-1] = min(unique_phi[-1], float(value))
        else:
            unique_y.append(float(y))
            unique_phi.append(float(value))

    y_array = np.asarray(unique_y, dtype=float)
    phi_array = np.asarray(unique_phi, dtype=float)
    if np.all(phi_array > 0.0):
        return {
            "height_m": 0.0,
            "status": "dry_column",
            "sample_count": int(column.size),
            "nearest_xz_distance_m": float(
                np.min(np.hypot(points[column, 0] - x, points[column, 2] - z))
            ),
        }
    if np.all(phi_array <= 0.0):
        return {
            "height_m": float(np.max(y_array)),
            "status": "wet_to_top",
            "sample_count": int(column.size),
            "nearest_xz_distance_m": float(
                np.min(np.hypot(points[column, 0] - x, points[column, 2] - z))
            ),
        }

    crossings: list[float] = []
    for left in range(len(y_array) - 1):
        y0 = float(y_array[left])
        y1 = float(y_array[left + 1])
        p0 = float(phi_array[left])
        p1 = float(phi_array[left + 1])
        if p0 == 0.0:
            crossings.append(y0)
        if p0 * p1 < 0.0:
            crossings.append(y0 - p0 * (y1 - y0) / (p1 - p0))
    if phi_array[-1] == 0.0:
        crossings.append(float(y_array[-1]))
    if not crossings:
        return {
            "height_m": None,
            "status": "no_crossing",
            "sample_count": int(column.size),
            "nearest_xz_distance_m": float(
                np.min(np.hypot(points[column, 0] - x, points[column, 2] - z))
            ),
        }
    return {
        "height_m": float(max(crossings)),
        "status": "crossing",
        "sample_count": int(column.size),
        "nearest_xz_distance_m": float(
            np.min(np.hypot(points[column, 0] - x, points[column, 2] - z))
        ),
    }


def pressure_at_point(grid: pv.DataSet, point: tuple[float, float, float]) -> dict[str, Any]:
    if "Pressure" not in grid.point_data:
        raise RuntimeError("result file does not contain Pressure point data")
    points = np.asarray(grid.points, dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    target = np.asarray(point, dtype=float)
    index = int(np.argmin(np.linalg.norm(points - target.reshape(1, 3), axis=1)))
    nearest = {
        "pressure_pa": float(pressure[index]),
        "selected_index": index,
        "selected_point": points[index].tolist(),
        "sensor_distance_m": float(np.linalg.norm(points[index] - target)),
    }
    sample = pv.PolyData(target.reshape(1, 3)).sample(grid, tolerance=1.0e-9)
    valid = bool(
        "vtkValidPointMask" in sample.point_data
        and int(np.asarray(sample.point_data["vtkValidPointMask"]).reshape(-1)[0]) == 1
    )
    if valid and "Pressure" in sample.point_data:
        return {
            "pressure_pa": float(np.asarray(sample.point_data["Pressure"]).reshape(-1)[0]),
            "selection": "interpolated_point_sample",
            "target_point": target.tolist(),
            "sample_valid": True,
            "containing_cell": int(grid.find_containing_cell(target)),
            "nearest_node": nearest,
        }
    return {
        **nearest,
        "selection": "nearest_node_fallback",
        "target_point": target.tolist(),
        "sample_valid": valid,
        "containing_cell": int(grid.find_containing_cell(target)),
    }


def sample_result(
    result_path: Path,
    *,
    prefix: str,
    dt: float,
    result_times: dict[str, float],
    height_traces: tuple[str, ...],
    pressure_traces: tuple[str, ...],
) -> dict[str, Any]:
    grid = pv.read(result_path)
    if "phi" not in grid.point_data:
        raise RuntimeError(f"result file does not contain phi point data: {result_path}")
    points = np.asarray(grid.points, dtype=float)
    phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    step = result_step(result_path, prefix)
    sensors = pressure_sensors()
    time_s = result_times.get(result_path.name, step * dt)
    return {
        "result": str(result_path),
        "step": step,
        "time_s": time_s,
        "height": {
            name: height_from_phi_column(points, phi, x=HEIGHT_PROBES[name])
            for name in height_traces
        },
        "pressure": {
            name: pressure_at_point(grid, sensors[name].point)
            for name in pressure_traces
        },
    }


def compare_samples(
    samples: list[dict[str, Any]],
    reference: dict[str, np.ndarray],
    traces: tuple[str, ...],
    *,
    kind: str,
) -> dict[str, Any]:
    if not samples:
        return {"available": False, "reason": "no samples"}
    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    ref_times = reference["Time"]
    result: dict[str, Any] = {"available": True, "trace_count": len(traces)}
    for trace in traces:
        if trace not in reference:
            result[trace] = {"available": False, "reason": "trace missing from reference"}
            continue
        if kind == "height":
            values = np.asarray(
                [sample["height"][trace]["height_m"] for sample in samples],
                dtype=float,
            )
        else:
            values = np.asarray(
                [sample["pressure"][trace]["pressure_pa"] for sample in samples],
                dtype=float,
            )
        reference_values = np.interp(times, ref_times, reference[trace])
        errors = values - reference_values
        result[trace] = {
            "sample_count": int(len(samples)),
            "rmse": float(math.sqrt(float(np.mean(errors * errors)))),
            "max_abs_error": float(np.max(np.abs(errors))),
            "final_value": float(values[-1]),
            "final_reference": float(reference_values[-1]),
            "final_error": float(errors[-1]),
            "reference_peak_over_sample_window": float(np.max(reference_values)),
            "simulated_peak_over_sample_window": float(np.max(values)),
        }
    return result


def reference_summary(reference: dict[str, np.ndarray]) -> dict[str, Any]:
    times = reference["Time"]
    return {
        "rows": int(len(times)),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "dt_median_s": float(np.median(np.diff(times))),
        "height_traces": [name for name in reference if name.startswith("H")],
        "pressure_traces": [name for name in reference if name.startswith("P")],
    }


def first_time_at_or_above(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    if not indices.size:
        return None
    return float(times[int(indices[0])])


def trace_event_summary(
    reference: dict[str, np.ndarray],
    *,
    height_traces: tuple[str, ...],
    pressure_traces: tuple[str, ...],
) -> dict[str, Any]:
    times = reference["Time"]
    traces: dict[str, Any] = {}
    required_peak_times: list[float] = []
    required_response_times: list[float] = []

    for name in height_traces:
        if name not in reference:
            continue
        values = reference[name]
        peak_index = int(np.argmax(values))
        peak_time = float(times[peak_index])
        response_time = first_time_at_or_above(
            times, values, HEIGHT_RESPONSE_THRESHOLD_M
        )
        traces[name] = {
            "kind": "height",
            "response_threshold_m": HEIGHT_RESPONSE_THRESHOLD_M,
            "first_response_time_s": response_time,
            "peak_time_s": peak_time,
            "peak_value_m": float(values[peak_index]),
        }
        required_peak_times.append(peak_time)
        if response_time is not None:
            required_response_times.append(response_time)

    for name in pressure_traces:
        if name not in reference:
            continue
        values = reference[name]
        peak_index = int(np.argmax(values))
        peak_time = float(times[peak_index])
        response_time = first_time_at_or_above(
            times, values, PRESSURE_RESPONSE_THRESHOLD_PA
        )
        traces[name] = {
            "kind": "pressure",
            "response_threshold_pa": PRESSURE_RESPONSE_THRESHOLD_PA,
            "first_response_time_s": response_time,
            "peak_time_s": peak_time,
            "peak_value_pa": float(values[peak_index]),
        }
        required_peak_times.append(peak_time)
        if response_time is not None:
            required_response_times.append(response_time)

    return {
        "traces": traces,
        "max_required_first_response_time_s": (
            max(required_response_times) if required_response_times else None
        ),
        "max_required_peak_time_s": max(required_peak_times) if required_peak_times else None,
    }


def validation_window(
    samples: list[dict[str, Any]],
    reference: dict[str, np.ndarray],
    *,
    height_traces: tuple[str, ...],
    pressure_traces: tuple[str, ...],
    minimum_reference_coverage: float,
) -> dict[str, Any]:
    ref_times = reference["Time"]
    events = trace_event_summary(
        reference,
        height_traces=height_traces,
        pressure_traces=pressure_traces,
    )
    if not samples:
        return {
            "sampled": False,
            "minimum_reference_coverage": minimum_reference_coverage,
            "reference_time_start_s": float(ref_times[0]),
            "reference_time_end_s": float(ref_times[-1]),
            "reference_duration_s": float(ref_times[-1] - ref_times[0]),
            "reference_events": events,
            "reference_coverage_fraction": 0.0,
        }

    start = float(samples[0]["time_s"])
    end = float(samples[-1]["time_s"])
    reference_duration = float(ref_times[-1] - ref_times[0])
    return {
        "sampled": True,
        "sample_time_start_s": start,
        "sample_time_end_s": end,
        "sample_duration_s": max(0.0, end - start),
        "minimum_reference_coverage": minimum_reference_coverage,
        "reference_time_start_s": float(ref_times[0]),
        "reference_time_end_s": float(ref_times[-1]),
        "reference_duration_s": reference_duration,
        "reference_events": events,
        "reference_coverage_fraction": (
            max(0.0, end - float(ref_times[0])) / reference_duration
            if reference_duration > 0.0
            else 0.0
        ),
    }


def validation_blocking_reasons(window: dict[str, Any]) -> list[str]:
    if not window.get("sampled", False):
        return ["no result_*.vtu or result_*.pvtu files are available to sample"]

    reasons: list[str] = []
    sample_end = float(window["sample_time_end_s"])
    coverage = float(window["reference_coverage_fraction"])
    events = window["reference_events"]
    response_end = events.get("max_required_first_response_time_s")
    peak_end = events.get("max_required_peak_time_s")
    if response_end is not None and sample_end < float(response_end):
        reasons.append(
            "solver history ends before all requested primary traces reach their "
            f"first response thresholds ({sample_end:g}s < {float(response_end):g}s)"
        )
    if peak_end is not None and sample_end < float(peak_end):
        reasons.append(
            "solver history ends before the latest requested primary-trace peak "
            f"({sample_end:g}s < {float(peak_end):g}s)"
        )
    if coverage < float(window["minimum_reference_coverage"]):
        reasons.append(
            "solver history covers only "
            f"{coverage:.3f} of the supplied reference horizon"
        )
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--height-traces", default=",".join(PRIMARY_HEIGHT_TRACES))
    parser.add_argument("--pressure-traces", default=",".join(PRIMARY_PRESSURE_TRACES))
    parser.add_argument(
        "--minimum-reference-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum fraction of the supplied reference time horizon required "
            "before reporting ready_for_history_gate."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--solver-log",
        type=Path,
        help=(
            "Optional solver stdout log used to recover adaptive step_accepted "
            "times when result.pvd is unavailable."
        ),
    )
    args = parser.parse_args()

    setup = parse_solver_xml(args.case_dir / "solver.xml")
    reference = load_reference_csv(args.reference_csv)
    height_traces = tuple(name.strip() for name in args.height_traces.split(",") if name.strip())
    pressure_traces = tuple(
        name.strip() for name in args.pressure_traces.split(",") if name.strip()
    )
    results = output_results(args.case_dir, args.result_prefix)
    result_times = result_times_from_pvd(args.case_dir, args.result_prefix)
    time_source = f"{args.result_prefix}.pvd" if result_times else None
    log_times = result_times_from_solver_log(args.solver_log, args.result_prefix)
    if not result_times and log_times:
        result_times = log_times
        time_source = str(args.solver_log)
    samples = [
        sample_result(
            result,
            prefix=args.result_prefix,
            dt=setup["time_step_size_s"],
            result_times=result_times,
            height_traces=height_traces,
            pressure_traces=pressure_traces,
        )
        for result in results
    ]
    comparison = {
        "height": compare_samples(samples, reference, height_traces, kind="height"),
        "pressure": compare_samples(samples, reference, pressure_traces, kind="pressure"),
    }
    window = validation_window(
        samples,
        reference,
        height_traces=height_traces,
        pressure_traces=pressure_traces,
        minimum_reference_coverage=args.minimum_reference_coverage,
    )
    blocking_reasons = validation_blocking_reasons(window)
    validation_ready = bool(samples) and not blocking_reasons
    report = {
        "case_dir": str(args.case_dir),
        "reference_csv": str(args.reference_csv),
        "setup": setup,
        "time_source": time_source or "result_step_times_time_step_size",
        "reference": reference_summary(reference),
        "result_count": len(results),
        "sampled_time_start_s": samples[0]["time_s"] if samples else None,
        "sampled_time_end_s": samples[-1]["time_s"] if samples else None,
        "height_probe_x_positions_m": {name: HEIGHT_PROBES[name] for name in height_traces},
        "pressure_sensor_coordinates_m": {
            name: list(pressure_sensors()[name].point) for name in pressure_traces
        },
        "pressure_sampling": {
            "method": "interpolated_point_sample_with_nearest_node_metadata",
            "all_requested_samples_valid": bool(samples)
            and all(
                sample["pressure"][name].get("sample_valid", False)
                for sample in samples
                for name in pressure_traces
            ),
        },
        "samples": samples,
        "comparison": comparison,
        "validation_window": window,
        "validation_ready": validation_ready,
        "validation_status": (
            "ready_for_history_gate" if validation_ready else "not_validation_ready"
        ),
        "blocking_reasons": blocking_reasons,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
