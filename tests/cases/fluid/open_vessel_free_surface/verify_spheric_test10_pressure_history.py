#!/usr/bin/env python3
"""Verify SPHERIC Test10 pressure history against pressure/motion records."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[3]
DEFAULT_CASE_DIR = (
    SCRIPT_DIR
    / "unfitted_level_set"
    / "spheric_test10_lateral_water_1x"
)
DEFAULT_REFERENCE_MEMBER = "SPHERIC_TestCase10/data_files/lateral_water_1x.txt"
GLOBAL_ID_FIELDS = (
    "GlobalVertexID",
    "GlobalNodeID",
    "GlobalPointID",
    "Global_vertex_gid",
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fetch_spheric_test10_reference import (  # noqa: E402
    TEST10_ZIP_URL,
    extract_entry,
    parse_central_directory,
    request_headers,
)


@dataclass(frozen=True)
class ReferenceSeries:
    time_s: np.ndarray
    pressure_pa: np.ndarray
    roll_position_deg: np.ndarray
    roll_velocity_deg_per_s: np.ndarray
    roll_acceleration_deg_per_s2: np.ndarray


def child_text(parent: ET.Element, path: str) -> str | None:
    child = parent.find(path)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def parse_solver_xml(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    general = root.find("GeneralSimulationParameters")
    fluid = root.find("./Add_equation[@type='fluid']")
    mesh_motion = root.find("./Add_equation[@type='mesh_motion']")
    if general is None or fluid is None:
        raise RuntimeError(f"solver.xml is missing required sections: {path}")

    pressure_constraint = fluid.find("Node_pressure_constraints")
    constraint_path = None
    constraint_id_type = None
    if pressure_constraint is not None:
        constraint_path = child_text(pressure_constraint, "Values_file_path")
        constraint_id_type = child_text(pressure_constraint, "Id_type")
    source_path = None
    for tag in (
        "Momentum_source_temporal_and_spatial_values_file_path",
        "MomentumSourceTemporalAndSpatialValuesFilePath",
        "Body_force_temporal_and_spatial_values_file_path",
        "BodyForceTemporalAndSpatialValuesFilePath",
    ):
        source_path = child_text(fluid, tag)
        if source_path:
            break

    text = path.read_text(encoding="utf-8")
    lower_text = text.lower()
    roll_keywords = (
        "roll",
        "rotation",
        "angular",
        "position_smooth",
        "lateral_water_1x",
        "time_function",
        "temporal",
        "prescribed_motion",
        "momentum_source_temporal_and_spatial_values_file_path",
        "body_force_temporal_and_spatial_values_file_path",
    )
    zero_dirichlet_bcs = [
        bc.get("name")
        for bc in fluid.findall("Add_BC")
        if child_text(bc, "Type") == "Dir" and (child_text(bc, "Value") or "") == "0.0"
    ]

    return {
        "number_of_time_steps": int(child_text(general, "Number_of_time_steps") or 0),
        "time_step_size_s": float(child_text(general, "Time_step_size") or 0.0),
        "pressure_constraint": {
            "configured": pressure_constraint is not None,
            "id_type": constraint_id_type,
            "values_file_path": constraint_path,
        },
        "momentum_source_temporal_and_spatial_values_file_path": source_path,
        "mesh_motion_equation_present": mesh_motion is not None,
        "roll_forcing_detected": any(keyword in lower_text for keyword in roll_keywords),
        "zero_dirichlet_fluid_boundaries": zero_dirichlet_bcs,
    }


def load_benchmark(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_constraint_nodes(case_dir: Path, relative_path: str | None) -> set[int]:
    if relative_path is None:
        return set()
    path = case_dir / relative_path
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["node_id"]) for row in csv.DictReader(stream)}


def fetch_reference_member(member: str) -> bytes:
    headers = request_headers(TEST10_ZIP_URL)
    archive_size = int(headers["content-length"])
    entries = parse_central_directory(TEST10_ZIP_URL, archive_size)
    by_name = {entry.name: entry for entry in entries}
    if member not in by_name:
        raise RuntimeError(f"reference member not found in Test10 archive: {member}")
    return extract_entry(TEST10_ZIP_URL, by_name[member])


def load_reference_series(path: Path | None, *, fetch: bool, member: str) -> ReferenceSeries | None:
    if path is None and not fetch:
        return None
    if path is not None:
        data = path.read_bytes()
    else:
        data = fetch_reference_member(member)
    text = data.decode("latin1")
    rows = csv.DictReader(text.splitlines(), delimiter="\t")
    time_s: list[float] = []
    pressure_pa: list[float] = []
    roll_position: list[float] = []
    roll_velocity: list[float] = []
    roll_acceleration: list[float] = []
    for row in rows:
        time_s.append(float(row["Time[s]"]))
        pressure_pa.append(100.0 * float(row["Pressure[mbar]"]))
        roll_position.append(float(row["Position_smooth_splines [deg]"]))
        roll_velocity.append(float(row["Velocity[deg\\s]"]))
        roll_acceleration.append(float(row["Aceleration[deg\\s2]"]))
    if not time_s:
        raise RuntimeError("SPHERIC Test10 reference table has no rows")
    return ReferenceSeries(
        time_s=np.asarray(time_s, dtype=float),
        pressure_pa=np.asarray(pressure_pa, dtype=float),
        roll_position_deg=np.asarray(roll_position, dtype=float),
        roll_velocity_deg_per_s=np.asarray(roll_velocity, dtype=float),
        roll_acceleration_deg_per_s2=np.asarray(roll_acceleration, dtype=float),
    )


def output_results(case_dir: Path, prefix: str) -> list[Path]:
    def step(path: Path) -> int:
        match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(
        [*case_dir.glob(f"{prefix}_*.vtu"), *case_dir.glob(f"{prefix}_*.pvtu")],
        key=step,
    )


def pvd_times(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    root = ET.parse(path).getroot()
    out: dict[str, float] = {}
    for data_set in root.findall(".//DataSet"):
        file_name = data_set.get("file")
        timestep = data_set.get("timestep")
        if file_name is not None and timestep is not None:
            out[Path(file_name).name] = float(timestep)
    return out


def global_ids(grid: pv.DataSet) -> np.ndarray | None:
    for name in GLOBAL_ID_FIELDS:
        if name in grid.point_data:
            return np.asarray(grid.point_data[name], dtype=int).reshape(-1)
    return None


def pressure_at_sensor(
    result_path: Path,
    *,
    sensor_node_id: int | None,
    sensor_coordinates: np.ndarray | None,
) -> dict[str, Any]:
    grid = pv.read(result_path)
    if "Pressure" not in grid.point_data:
        raise RuntimeError(f"result file does not contain point-data Pressure: {result_path}")
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    if sensor_coordinates is None:
        raise RuntimeError("sensor coordinates are required for pressure-history sampling")
    target = np.asarray(sensor_coordinates, dtype=float)
    points = np.asarray(grid.points, dtype=float)
    distances = np.linalg.norm(points - target.reshape(1, 3), axis=1)
    nearest_index = int(np.argmin(distances))
    nearest = {
        "pressure_pa": float(pressure[nearest_index]),
        "selected_index": nearest_index,
        "selected_point": points[nearest_index].tolist(),
        "sensor_distance_m": float(distances[nearest_index]),
    }

    sample = pv.PolyData(target.reshape(1, 3)).sample(grid, tolerance=1.0e-8)
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

    ids = global_ids(grid)
    selected_index = None
    selection = "global_node_id_fallback"
    if sensor_node_id is not None and ids is not None:
        matches = np.flatnonzero(ids == int(sensor_node_id))
        if matches.size:
            selected_index = int(matches[0])
            return {
                "pressure_pa": float(pressure[selected_index]),
                "selection": selection,
                "target_point": target.tolist(),
                "sample_valid": valid,
                "containing_cell": int(grid.find_containing_cell(target)),
                "nearest_node": {
                    "pressure_pa": float(pressure[selected_index]),
                    "selected_index": selected_index,
                    "selected_point": points[selected_index].tolist(),
                    "sensor_distance_m": float(np.linalg.norm(points[selected_index] - target)),
                },
            }
    if selected_index is None:
        selected_index = nearest_index
    return {
        "pressure_pa": float(pressure[selected_index]),
        "selected_index": selected_index,
        "selection": "nearest_node_fallback",
        "target_point": target.tolist(),
        "sample_valid": valid,
        "containing_cell": int(grid.find_containing_cell(target)),
        "selected_point": points[selected_index].tolist(),
        "sensor_distance_m": float(np.linalg.norm(points[selected_index] - target)),
        "nearest_node": nearest,
    }


def sample_pressure_history(
    case_dir: Path,
    *,
    prefix: str,
    dt: float,
    sensor_node_id: int | None,
    sensor_coordinates: np.ndarray | None,
) -> dict[str, Any]:
    results = output_results(case_dir, prefix)
    pvd = pvd_times(case_dir / f"{prefix}.pvd")
    samples: list[dict[str, Any]] = []
    for result in results:
        match = re.match(rf"{re.escape(prefix)}_(\d+)\.p?vtu$", result.name)
        step = int(match.group(1)) if match else len(samples)
        time_s = pvd.get(result.name, step * dt)
        sample = pressure_at_sensor(
            result,
            sensor_node_id=sensor_node_id,
            sensor_coordinates=sensor_coordinates,
        )
        sample.update({"result": str(result), "step": step, "time_s": time_s})
        samples.append(sample)
    return {
        "found": bool(results),
        "result_count": len(results),
        "samples": samples,
    }


def compare_pressure_history(samples: list[dict[str, Any]], reference: ReferenceSeries) -> dict[str, Any]:
    if not samples:
        return {"available": False, "reason": "no solver pressure samples"}
    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    pressure = np.asarray([sample["pressure_pa"] for sample in samples], dtype=float)
    reference_pressure = np.interp(times, reference.time_s, reference.pressure_pa)
    errors = pressure - reference_pressure
    reference_scale = max(float(np.max(np.abs(reference_pressure))), 1.0)
    peak_index = int(np.argmax(pressure))
    reference_peak_index = int(np.argmax(reference_pressure))
    return {
        "available": True,
        "sample_count": int(len(samples)),
        "rmse_pa": float(math.sqrt(float(np.mean(errors * errors)))),
        "rmse_relative_to_reference_peak": float(
            math.sqrt(float(np.mean(errors * errors))) / reference_scale
        ),
        "max_abs_error_pa": float(np.max(np.abs(errors))),
        "simulated_peak_pressure_pa": float(pressure[peak_index]),
        "simulated_peak_time_s": float(times[peak_index]),
        "reference_peak_pressure_pa": float(reference_pressure[reference_peak_index]),
        "reference_peak_time_s": float(times[reference_peak_index]),
        "peak_pressure_error_pa": float(
            pressure[peak_index] - reference_pressure[reference_peak_index]
        ),
        "peak_time_error_s": float(times[peak_index] - times[reference_peak_index]),
    }


def validation_window(
    samples: list[dict[str, Any]],
    reference: ReferenceSeries | None,
    *,
    minimum_reference_coverage: float,
) -> tuple[dict[str, Any], list[str]]:
    if not samples:
        return {"sampled": False}, []

    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    sampled_start = float(np.min(times))
    sampled_end = float(np.max(times))
    sampled_duration = sampled_end - sampled_start
    window: dict[str, Any] = {
        "sampled": True,
        "sample_time_start_s": sampled_start,
        "sample_time_end_s": sampled_end,
        "sample_duration_s": sampled_duration,
        "minimum_reference_coverage": float(minimum_reference_coverage),
    }
    blockers: list[str] = []

    if reference is None:
        return window, blockers

    reference_start = float(reference.time_s[0])
    reference_end = float(reference.time_s[-1])
    reference_duration = reference_end - reference_start
    reference_peak_index = int(np.argmax(reference.pressure_pa))
    reference_peak_time = float(reference.time_s[reference_peak_index])
    coverage = (
        (sampled_end - reference_start) / reference_duration
        if reference_duration > 0.0
        else 0.0
    )
    window.update(
        {
            "reference_time_start_s": reference_start,
            "reference_time_end_s": reference_end,
            "reference_duration_s": reference_duration,
            "reference_peak_pressure_time_s": reference_peak_time,
            "reference_coverage_fraction": float(coverage),
        }
    )
    if sampled_end < reference_peak_time:
        blockers.append(
            "solver pressure history ends before the reference pressure peak time "
            f"({sampled_end:.6g}s < {reference_peak_time:.6g}s)"
        )
    if coverage < minimum_reference_coverage:
        blockers.append(
            "solver pressure history covers only "
            f"{coverage:.3f} of the supplied reference horizon"
        )
    return window, blockers


def reference_summary(reference: ReferenceSeries | None) -> dict[str, Any]:
    if reference is None:
        return {"available": False}
    dt = np.diff(reference.time_s)
    return {
        "available": True,
        "rows": int(reference.time_s.size),
        "time_start_s": float(reference.time_s[0]),
        "time_end_s": float(reference.time_s[-1]),
        "dt_median_s": float(np.median(dt)) if dt.size else None,
        "pressure_min_pa": float(np.min(reference.pressure_pa)),
        "pressure_max_pa": float(np.max(reference.pressure_pa)),
        "roll_position_min_deg": float(np.min(reference.roll_position_deg)),
        "roll_position_max_deg": float(np.max(reference.roll_position_deg)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--fetch-reference", action="store_true")
    parser.add_argument("--reference-member", default=DEFAULT_REFERENCE_MEMBER)
    parser.add_argument("--prefix", default="result")
    parser.add_argument(
        "--minimum-reference-coverage",
        type=float,
        default=0.95,
        help=(
            "Minimum fraction of the supplied reference time horizon required before "
            "marking the run ready for a pressure-history gate."
        ),
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    case_dir = args.case_dir
    benchmark = load_benchmark(case_dir / "benchmark.json")
    solver = parse_solver_xml(case_dir / "solver.xml")
    pressure_gauge = benchmark.get("pressure_gauge", {})
    pressure_sensor = benchmark.get("pressure_sensor", {})
    sensor_node_id = pressure_sensor.get("node_id")
    sensor_coordinates = pressure_sensor.get("coordinates")
    sensor_source = "pressure_sensor"
    if sensor_coordinates is None:
        sensor_node_id = pressure_gauge.get("node_id")
        sensor_coordinates = pressure_gauge.get("coordinates")
        sensor_source = "pressure_gauge_fallback"
    sensor_coordinates_array = (
        np.asarray(sensor_coordinates, dtype=float) if sensor_coordinates is not None else None
    )
    constraint_nodes = load_constraint_nodes(
        case_dir,
        solver["pressure_constraint"]["values_file_path"],
    )
    sensor_is_constrained = (
        sensor_node_id is not None and int(sensor_node_id) in constraint_nodes
    )
    pressure_anchor_node_id = pressure_gauge.get("node_id")
    pressure_anchor_is_constrained = (
        pressure_anchor_node_id is not None and int(pressure_anchor_node_id) in constraint_nodes
    )

    reference = load_reference_series(
        args.reference_file,
        fetch=args.fetch_reference,
        member=args.reference_member,
    )
    history = sample_pressure_history(
        case_dir,
        prefix=args.prefix,
        dt=solver["time_step_size_s"],
        sensor_node_id=int(sensor_node_id) if sensor_node_id is not None else None,
        sensor_coordinates=sensor_coordinates_array,
    )
    comparison = compare_pressure_history(history["samples"], reference) if reference else {
        "available": False,
        "reason": "no full reference table supplied",
    }
    window, window_blocking_reasons = validation_window(
        history["samples"],
        reference,
        minimum_reference_coverage=args.minimum_reference_coverage,
    )

    blocking_reasons: list[str] = []
    if not solver["roll_forcing_detected"]:
        blocking_reasons.append("published roll-angle forcing is not configured in solver.xml")
    if sensor_is_constrained:
        blocking_reasons.append(
            "pressure sensor node is configured as a pressure constraint, not an output-only sensor"
        )
    if not history["found"]:
        blocking_reasons.append("no result_*.vtu or result_*.pvtu files are available to sample")
    if reference is None:
        blocking_reasons.append("full SPHERIC Test10 reference pressure table was not supplied")
    blocking_reasons.extend(window_blocking_reasons)

    validation_ready = not blocking_reasons
    report = {
        "case_dir": str(case_dir),
        "reference_member": args.reference_member,
        "setup": {
            "sensor_node_id": sensor_node_id,
            "sensor_coordinates": sensor_coordinates,
            "sensor_source": sensor_source,
            "pressure_sensor": pressure_sensor,
            "sensor_is_pressure_constraint": sensor_is_constrained,
            "pressure_anchor_node_id": pressure_anchor_node_id,
            "pressure_anchor_coordinates": pressure_gauge.get("coordinates"),
            "pressure_anchor_is_pressure_constraint": pressure_anchor_is_constrained,
            "pressure_constraint": solver["pressure_constraint"],
            "momentum_source_temporal_and_spatial_values_file_path": solver[
                "momentum_source_temporal_and_spatial_values_file_path"
            ],
            "mesh_motion_equation_present": solver["mesh_motion_equation_present"],
            "roll_forcing_detected": solver["roll_forcing_detected"],
            "zero_dirichlet_fluid_boundaries": solver["zero_dirichlet_fluid_boundaries"],
        },
        "reference": reference_summary(reference),
        "pressure_history": {
            "found": history["found"],
            "result_count": history["result_count"],
            "sample_count": len(history["samples"]),
            "samples": history["samples"],
        },
        "comparison": comparison,
        "validation_window": window,
        "validation_ready": validation_ready,
        "validation_status": (
            "ready_for_pressure_history_gate" if validation_ready else "not_validation_ready"
        ),
        "blocking_reasons": blocking_reasons,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
