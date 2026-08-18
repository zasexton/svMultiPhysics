#!/usr/bin/env python3
"""Localize SPHERIC Test02 pressure extrema in level-set result files."""

from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

import verify_spheric_test02_histories as verifier


PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")
TINY_WET_FRACTION = 1.0e-4


def result_times(case_dir: Path, prefix: str, solver_log: Path | None) -> tuple[dict[str, float], str]:
    pvd_times = verifier.result_times_from_pvd(case_dir, prefix)
    if pvd_times:
        return pvd_times, f"{prefix}.pvd"
    log_times = verifier.result_times_from_solver_log(solver_log, prefix)
    if log_times:
        return log_times, str(solver_log)
    return {}, "result_step_times_time_step_size"


def incident_cell_ids(grid: pv.DataSet, point_index: int) -> list[int]:
    cells = np.asarray(grid.cells, dtype=int)
    incident: list[int] = []
    offset = 0
    cell_id = 0
    while offset < cells.size:
        node_count = int(cells[offset])
        point_ids = cells[offset + 1 : offset + 1 + node_count]
        if np.any(point_ids == point_index):
            incident.append(cell_id)
        offset += node_count + 1
        cell_id += 1
    return incident


def incident_cell_summary(grid: pv.DataSet, point_index: int) -> dict[str, Any]:
    cell_ids = incident_cell_ids(grid, point_index)
    if not cell_ids:
        return {"incident_cell_count": 0}

    report: dict[str, Any] = {"incident_cell_count": len(cell_ids)}
    for name in ("WetVolumeFraction", "WetVolumeMeasure", "RegionID", "GlobalCellID"):
        if name not in grid.cell_data:
            continue
        values = np.asarray(grid.cell_data[name]).reshape(-1)[cell_ids]
        if name in {"RegionID", "GlobalCellID"}:
            report[name] = [int(value) for value in values.tolist()]
        else:
            report[f"{name}_min"] = float(np.min(values))
            report[f"{name}_max"] = float(np.max(values))
            report[f"{name}_mean"] = float(np.mean(values))
    return report


def node_report(
    grid: pv.DataSet,
    *,
    point_index: int,
    time_s: float,
    step: int,
    result_name: str,
) -> dict[str, Any]:
    points = np.asarray(grid.points, dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    active = (
        np.asarray(grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
        if "ActiveFluid" in grid.point_data
        else np.full_like(phi, np.nan)
    )
    velocity = (
        np.asarray(grid.point_data["Velocity"], dtype=float)
        if "Velocity" in grid.point_data
        else np.full((len(points), 3), np.nan)
    )
    point = points[point_index]
    return {
        "result": result_name,
        "step": step,
        "time_s": time_s,
        "point_index": int(point_index),
        "point_m": [float(value) for value in point.tolist()],
        "pressure_pa": float(pressure[point_index]),
        "phi": float(phi[point_index]),
        "active_fluid": float(active[point_index]),
        "velocity_m_per_s": [float(value) for value in velocity[point_index].tolist()],
        "speed_m_per_s": float(np.linalg.norm(velocity[point_index])),
        "incident_cells": incident_cell_summary(grid, point_index),
    }


def extrema_report(
    grid: pv.DataSet,
    *,
    mask: np.ndarray,
    time_s: float,
    step: int,
    result_name: str,
) -> dict[str, Any] | None:
    if not np.any(mask):
        return None
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    selected = np.flatnonzero(mask)
    point_index = int(selected[int(np.argmax(pressure[selected]))])
    return node_report(
        grid,
        point_index=point_index,
        time_s=time_s,
        step=step,
        result_name=result_name,
    )


def local_node_max(
    grid: pv.DataSet,
    point: tuple[float, float, float],
    *,
    radius: float,
    wet_only: bool,
) -> dict[str, Any] | None:
    points = np.asarray(grid.points, dtype=float)
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    distances = np.linalg.norm(points - np.asarray(point, dtype=float), axis=1)
    mask = distances <= radius + 1.0e-12
    if wet_only:
        mask = mask & (phi <= 0.0)
    if not np.any(mask):
        return None
    selected = np.flatnonzero(mask)
    point_index = int(selected[int(np.argmax(pressure[selected]))])
    return {
        "point_index": point_index,
        "pressure_pa": float(pressure[point_index]),
        "point_m": [float(value) for value in points[point_index].tolist()],
        "sensor_distance_m": float(distances[point_index]),
        "phi": float(phi[point_index]),
        "incident_cells": incident_cell_summary(grid, point_index),
    }


def reference_pressure_peaks(reference_csv: Path | None, final_time_s: float) -> dict[str, Any]:
    if reference_csv is None:
        return {}
    reference = verifier.load_reference_csv(reference_csv)
    times = reference["Time"]
    in_window = times <= final_time_s + 1.0e-12
    peaks: dict[str, Any] = {}
    for trace in PRIMARY_PRESSURE_TRACES:
        values = reference[trace][in_window]
        trace_times = times[in_window]
        if values.size == 0:
            continue
        index = int(np.argmax(values))
        peaks[trace] = {
            "peak_over_sample_window_pa": float(values[index]),
            "peak_time_over_sample_window_s": float(trace_times[index]),
        }
    return peaks


def audit_case(
    case_dir: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
    reference_csv: Path | None,
    pressure_traces: tuple[str, ...],
    local_radius_m: float,
) -> dict[str, Any]:
    setup = verifier.parse_solver_xml(case_dir / "solver.xml")
    results = verifier.output_results(case_dir, result_prefix)
    time_by_result, time_source = result_times(case_dir, result_prefix, solver_log)
    sensors = verifier.pressure_sensors()

    samples: list[dict[str, Any]] = []
    global_max_wet: dict[str, Any] | None = None
    global_max_active: dict[str, Any] | None = None
    global_max_all: dict[str, Any] | None = None
    tiny_cut_wet_spike_count = 0

    sensor_peaks: dict[str, dict[str, Any]] = {
        name: {
            "target_peak_pa": -math.inf,
            "target_peak_event": None,
            "nearest_node_peak_pa": -math.inf,
            "nearest_node_peak_event": None,
            "local_wet_node_peak_pa": -math.inf,
            "local_wet_node_peak_event": None,
            "local_any_node_peak_pa": -math.inf,
            "local_any_node_peak_event": None,
        }
        for name in pressure_traces
    }

    for result in results:
        grid = pv.read(result)
        step = verifier.result_step(result, result_prefix)
        time_s = time_by_result.get(result.name, step * setup["time_step_size_s"])
        pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
        phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
        active = (
            np.asarray(grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
            if "ActiveFluid" in grid.point_data
            else np.zeros_like(phi)
        )

        all_event = extrema_report(
            grid,
            mask=np.ones_like(phi, dtype=bool),
            time_s=time_s,
            step=step,
            result_name=result.name,
        )
        wet_event = extrema_report(
            grid,
            mask=phi <= 0.0,
            time_s=time_s,
            step=step,
            result_name=result.name,
        )
        active_event = extrema_report(
            grid,
            mask=active > 0.5,
            time_s=time_s,
            step=step,
            result_name=result.name,
        )

        for name, candidate in (
            ("global_max_all", all_event),
            ("global_max_wet", wet_event),
            ("global_max_active", active_event),
        ):
            if candidate is None:
                continue
            current = locals()[name]
            if current is None or candidate["pressure_pa"] > current["pressure_pa"]:
                if name == "global_max_all":
                    global_max_all = candidate
                elif name == "global_max_wet":
                    global_max_wet = candidate
                else:
                    global_max_active = candidate

        wet_fraction = (
            wet_event.get("incident_cells", {}).get("WetVolumeFraction_max")
            if wet_event is not None
            else None
        )
        if wet_event is not None and wet_fraction is not None and wet_fraction < TINY_WET_FRACTION:
            tiny_cut_wet_spike_count += 1

        sample: dict[str, Any] = {
            "result": result.name,
            "step": step,
            "time_s": float(time_s),
            "max_pressure_all_pa": all_event["pressure_pa"] if all_event else None,
            "max_pressure_wet_pa": wet_event["pressure_pa"] if wet_event else None,
            "max_pressure_active_pa": active_event["pressure_pa"] if active_event else None,
            "max_wet_point_m": wet_event["point_m"] if wet_event else None,
            "max_wet_phi": wet_event["phi"] if wet_event else None,
            "max_wet_incident_wet_volume_fraction_max": wet_fraction,
        }

        for trace in pressure_traces:
            sensor = sensors[trace]
            target = verifier.pressure_at_point(grid, sensor.point)
            target_pressure = float(target["pressure_pa"])
            nearest = target.get("nearest_node", {})
            local_wet = local_node_max(grid, sensor.point, radius=local_radius_m, wet_only=True)
            local_any = local_node_max(grid, sensor.point, radius=local_radius_m, wet_only=False)
            sample[f"{trace}_target_pressure_pa"] = target_pressure
            for key, event in (
                ("target", {"pressure_pa": target_pressure, "sample": target}),
                ("nearest_node", nearest),
                ("local_wet_node", local_wet),
                ("local_any_node", local_any),
            ):
                if event is None:
                    continue
                pressure_value = float(event["pressure_pa"])
                peak_key = f"{key}_peak_pa"
                event_key = f"{key}_peak_event"
                if pressure_value > sensor_peaks[trace][peak_key]:
                    sensor_peaks[trace][peak_key] = pressure_value
                    sensor_peaks[trace][event_key] = {
                        "result": result.name,
                        "step": step,
                        "time_s": float(time_s),
                        **event,
                    }
        samples.append(sample)

    final_time = float(samples[-1]["time_s"]) if samples else 0.0
    reference_peaks = reference_pressure_peaks(reference_csv, final_time)
    p1_reference_peak = reference_peaks.get("P1", {}).get("peak_over_sample_window_pa")

    for report in sensor_peaks.values():
        for key in list(report):
            if key.endswith("_peak_pa") and report[key] == -math.inf:
                report[key] = None
        if p1_reference_peak:
            for key in (
                "target_peak_pa",
                "nearest_node_peak_pa",
                "local_wet_node_peak_pa",
                "local_any_node_peak_pa",
            ):
                value = report.get(key)
                report[f"{key}_over_reference_P1_peak"] = (
                    float(value / p1_reference_peak) if value is not None else None
                )

    finding = "Pressure-extrema localization did not find result files."
    if global_max_wet is not None:
        incident = global_max_wet.get("incident_cells", {})
        wet_fraction_max = incident.get("WetVolumeFraction_max")
        if wet_fraction_max is not None and wet_fraction_max < TINY_WET_FRACTION:
            finding = (
                "The largest active/wet pressure extrema localize to barely wet cut cells, "
                f"with max wet pressure {global_max_wet['pressure_pa']} Pa at "
                f"t={global_max_wet['time_s']} s and incident WetVolumeFraction max "
                f"{wet_fraction_max}. This identifies a tiny-cut pressure spike separate "
                "from official pressure-sensor interpolation."
            )
        else:
            finding = (
                "The largest active/wet pressure extrema do not localize to the configured "
                "tiny-cut threshold; inspect the per-sample maxima before changing the "
                "pressure stabilization path."
            )

    return {
        "case_dir": str(case_dir),
        "result_count": len(results),
        "sampled_time_start_s": samples[0]["time_s"] if samples else None,
        "sampled_time_end_s": samples[-1]["time_s"] if samples else None,
        "time_source": time_source,
        "pressure_traces": list(pressure_traces),
        "local_radius_m": local_radius_m,
        "tiny_wet_fraction_threshold": TINY_WET_FRACTION,
        "reference_pressure_peaks_over_sample_window": reference_peaks,
        "global_max_pressure_all": global_max_all,
        "global_max_pressure_wet": global_max_wet,
        "global_max_pressure_active": global_max_active,
        "tiny_cut_wet_spike_count": tiny_cut_wet_spike_count,
        "sensor_peaks": sensor_peaks,
        "samples": samples,
        "finding": finding,
        "status": "diagnostic_pressure_extrema_not_validation_gate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--pressure-traces", default=",".join(PRIMARY_PRESSURE_TRACES))
    parser.add_argument("--local-radius-m", type=float, default=0.05)
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    pressure_traces = tuple(
        name.strip() for name in args.pressure_traces.split(",") if name.strip()
    )
    report = audit_case(
        args.case_dir,
        result_prefix=args.result_prefix,
        solver_log=args.solver_log,
        reference_csv=args.reference_csv,
        pressure_traces=pressure_traces,
        local_radius_m=args.local_radius_m,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
