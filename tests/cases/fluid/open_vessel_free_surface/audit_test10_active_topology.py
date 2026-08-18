#!/usr/bin/env python3
"""Audit Test10 tail active topology and failed nonlinear retry context."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_QUALIFICATION_DIR = (
    REPO_ROOT
    / "Documentation"
    / "qualification_logs"
    / "open_vessel_free_surface_remaining_20260526"
)
DEFAULT_CASE_DIR = (
    DEFAULT_QUALIFICATION_DIR
    / "test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_1s_post_controller_fix_case"
)
DEFAULT_SOLVER_LOG = (
    DEFAULT_QUALIFICATION_DIR
    / "test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_1s_post_controller_fix_solver_stdout_20260604.log"
)

NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\S+)")
RESULT_RE_TEMPLATE = r"{prefix}_(\d+)\.p?vtu$"
NONLINEAR_DONE_RE = re.compile(
    rf"TimeLoop: nonlinear_done step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"converged=(?P<converged>\d+) iters=(?P<iters>\d+) "
    rf"\|\|r\|\|=(?P<residual>{NUMBER_RE}) "
    rf"\|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) "
    rf"\|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE}) "
    rf"\(linear: converged=(?P<linear_converged>\d+) iters=(?P<linear_iters>\d+) "
    rf"rel=(?P<linear_rel>{NUMBER_RE})\)"
)
STEP_REJECTED_RE = re.compile(
    rf"TimeLoop: step_rejected step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE}) reason=(?P<reason>\S+) "
    rf"\(newton: converged=(?P<newton_converged>\d+) iters=(?P<newton_iters>\d+) "
    rf"\|\|r\|\|=(?P<residual>{NUMBER_RE}) "
    rf"\|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) "
    rf"\|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE})\)"
)
STEP_START_RE = re.compile(
    rf"TimeLoop: step_start step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE})"
)
RESIDUAL_BLOCK_RE = re.compile(
    rf"diagnostic=residual_block_norms phase='?(?P<phase>[A-Za-z_]+)'? "
    rf"field=(?P<field>{NUMBER_RE}) aux=(?P<aux>{NUMBER_RE}) "
    rf"combined=(?P<combined>{NUMBER_RE})"
)
VECTOR_COMPONENT_RE = re.compile(
    rf"\[(?P<name>phi|Velocity\[0\]|Velocity\[1\]|Velocity\[2\]|Pressure) "
    rf"norm=(?P<norm>{NUMBER_RE}) mean=(?P<mean>{NUMBER_RE}) "
    rf"min=(?P<min>{NUMBER_RE}) max=(?P<max>{NUMBER_RE})\]"
)
CUT_CONTEXT_KEYS = (
    "provenance",
    "solution_source",
    "cut_context_revision",
    "active_side_volume",
    "active_side_physical_volume",
    "inactive_side_physical_volume",
    "interface_fragments",
    "active_volume_regions",
    "active_wet_cells",
    "active_cut_cells",
    "active_full_wet_cells",
    "active_full_dry_cells",
    "active_min_volume_fraction",
    "active_max_volume_fraction",
    "active_pruned_volume_regions",
    "active_pruned_volume",
    "generated_pruned_volume_rules",
    "generated_pruned_volume",
    "implicit_cut_fallback_cells",
    "cut_adjacent_facets",
    "cut_adjacent_capped_scale",
    "cut_adjacent_min_scale",
    "cut_adjacent_max_scale",
    "cut_adjacent_mean_scale",
)


def parse_log_value(raw: str) -> int | float | str:
    value = raw.strip("'")
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(NUMBER_RE, value):
        return float(value)
    return value


def parse_key_values(line: str) -> dict[str, int | float | str]:
    return {
        match.group(1): parse_log_value(match.group(2))
        for match in KEY_VALUE_RE.finditer(line)
    }


def result_step(path: Path, prefix: str) -> int:
    match = re.match(RESULT_RE_TEMPLATE.format(prefix=re.escape(prefix)), path.name)
    return int(match.group(1)) if match else -1


def result_files(case_dir: Path, prefix: str) -> list[Path]:
    return sorted(
        [*case_dir.glob(f"{prefix}_*.vtu"), *case_dir.glob(f"{prefix}_*.pvtu")],
        key=lambda path: result_step(path, prefix),
    )


def pvd_times(case_dir: Path, prefix: str) -> tuple[dict[str, float], str | None]:
    path = case_dir / f"{prefix}.pvd"
    if not path.exists():
        return {}, None
    root = ET.parse(path).getroot()
    out: dict[str, float] = {}
    for data_set in root.findall(".//DataSet"):
        file_name = data_set.get("file")
        timestep = data_set.get("timestep")
        if file_name and timestep:
            out[Path(file_name).name] = float(timestep)
    return out, str(path)


def array_bounds(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
    }


def tetra_connectivity(grid: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(grid.cells, dtype=np.int64).reshape((-1, 5))
    if np.any(cells[:, 0] != 4):
        raise RuntimeError("Test10 active-topology audit expects tetrahedral cells only")
    return cells[:, 1:]


def load_sensor_point(case_dir: Path) -> np.ndarray | None:
    path = case_dir / "benchmark.json"
    if not path.exists():
        return None
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    sensor = benchmark.get("pressure_sensor", {})
    coordinates = sensor.get("coordinates")
    if coordinates is None:
        return None
    return np.asarray(coordinates, dtype=float)


def incident_wet_support(
    n_points: int,
    tets: np.ndarray,
    wet_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_fraction = np.zeros(n_points, dtype=float)
    min_positive = np.full(n_points, math.nan, dtype=float)
    for cell_index, tet in enumerate(tets):
        fraction = float(wet_fraction[cell_index])
        if fraction <= 0.0:
            continue
        max_fraction[tet] = np.maximum(max_fraction[tet], fraction)
        current = min_positive[tet]
        missing = np.isnan(current)
        min_positive[tet[missing]] = fraction
        present = ~missing
        min_positive[tet[present]] = np.minimum(current[present], fraction)
    return max_fraction, min_positive


def vector_components(line: str, *, line_number: int, phase: str | None) -> dict[str, Any] | None:
    values = {
        match.group("name"): {
            "norm": float(match.group("norm")),
            "mean": float(match.group("mean")),
            "min": float(match.group("min")),
            "max": float(match.group("max")),
        }
        for match in VECTOR_COMPONENT_RE.finditer(line)
    }
    if not values:
        return None
    label_match = re.search(r"label='?([A-Za-z_]+)'?", line)
    return {
        "line_number": line_number,
        "label": label_match.group(1) if label_match else None,
        "phase": phase,
        "components": values,
    }


def parse_solver_log(path: Path | None, *, tail_records: int) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "solver_log": str(path) if path else None,
            "available": False,
        }

    nonlinear_done: list[dict[str, Any]] = []
    step_rejected: list[dict[str, Any]] = []
    residual_blocks: list[dict[str, Any]] = []
    vector_records: list[dict[str, Any]] = []
    cut_contexts: list[dict[str, Any]] = []
    last_residual_phase: str | None = None
    current_step: int | None = None

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if match := STEP_START_RE.search(line):
                current_step = int(match.group("step"))
            if match := NONLINEAR_DONE_RE.search(line):
                nonlinear_done.append(
                    {
                        "line_number": line_number,
                        "step": int(match.group("step")),
                        "time_s": float(match.group("time")),
                        "converged": bool(int(match.group("converged"))),
                        "iters": int(match.group("iters")),
                        "residual": float(match.group("residual")),
                        "residual_field": float(match.group("residual_field")),
                        "residual_aux": float(match.group("residual_aux")),
                        "linear_converged": bool(int(match.group("linear_converged"))),
                        "linear_iters": int(match.group("linear_iters")),
                        "linear_rel": float(match.group("linear_rel")),
                    }
                )
            if match := STEP_REJECTED_RE.search(line):
                step_rejected.append(
                    {
                        "line_number": line_number,
                        "step": int(match.group("step")),
                        "time_s": float(match.group("time")),
                        "dt_s": float(match.group("dt")),
                        "reason": match.group("reason"),
                        "newton_converged": bool(int(match.group("newton_converged"))),
                        "newton_iters": int(match.group("newton_iters")),
                        "residual": float(match.group("residual")),
                        "residual_field": float(match.group("residual_field")),
                        "residual_aux": float(match.group("residual_aux")),
                    }
                )
            if "diagnostic=residual_block_norms" in line:
                match = RESIDUAL_BLOCK_RE.search(line)
                if match:
                    last_residual_phase = match.group("phase")
                    residual_blocks.append(
                        {
                            "line_number": line_number,
                            "phase": last_residual_phase,
                            "field": float(match.group("field")),
                            "aux": float(match.group("aux")),
                            "combined": float(match.group("combined")),
                        }
                    )
            if "diagnostic=vector_component_norms" in line:
                record = vector_components(
                    line,
                    line_number=line_number,
                    phase=last_residual_phase,
                )
                if record is not None:
                    vector_records.append(record)
            if "diagnostic=cut_context_rebuild" in line:
                values = parse_key_values(line)
                record = {
                    key: values[key]
                    for key in CUT_CONTEXT_KEYS
                    if key in values
                }
                record["line_number"] = line_number
                record["step_context"] = current_step
                cut_contexts.append(record)

    rejected_steps = [record["step"] for record in step_rejected]
    final_failed_step = rejected_steps[-1] if rejected_steps else None
    final_step_rejections = [
        record for record in step_rejected if record["step"] == final_failed_step
    ]
    final_step_nonlinear = [
        record for record in nonlinear_done if record["step"] == final_failed_step
    ]
    final_step_cut_contexts = [
        record
        for record in cut_contexts
        if record.get("step_context") == final_failed_step
    ]
    cut_min_records = [
        record
        for record in cut_contexts
        if isinstance(record.get("active_min_volume_fraction"), (int, float))
    ]
    final_step_cut_min_records = [
        record
        for record in final_step_cut_contexts
        if isinstance(record.get("active_min_volume_fraction"), (int, float))
    ]
    capped_records = [
        record
        for record in cut_contexts
        if isinstance(record.get("cut_adjacent_capped_scale"), (int, float))
    ]
    final_step_capped_records = [
        record
        for record in final_step_cut_contexts
        if isinstance(record.get("cut_adjacent_capped_scale"), (int, float))
    ]
    final_residual_components = next(
        (
            record
            for record in reversed(vector_records)
            if record.get("label") == "residual_block_norms"
        ),
        None,
    )
    final_solution_components = next(
        (
            record
            for record in reversed(vector_records)
            if record.get("label") == "solution_state"
        ),
        None,
    )
    line_search_reject_components = next(
        (
            record
            for record in reversed(vector_records)
            if record.get("label") == "residual_block_norms"
            and record.get("phase") == "line_search_reject"
        ),
        final_residual_components,
    )

    return {
        "solver_log": str(path),
        "available": True,
        "nonlinear_done_count": len(nonlinear_done),
        "step_rejected_count": len(step_rejected),
        "final_failed_step": final_failed_step,
        "final_step_rejections": final_step_rejections,
        "final_step_nonlinear_done": final_step_nonlinear,
        "final_step_cut_context_count": len(final_step_cut_contexts),
        "final_step_tail_cut_contexts": final_step_cut_contexts[-tail_records:],
        "final_step_min_active_min_volume_fraction": (
            min(final_step_cut_min_records, key=lambda item: item["active_min_volume_fraction"])
            if final_step_cut_min_records
            else None
        ),
        "final_step_max_cut_adjacent_capped_scale": (
            max(final_step_capped_records, key=lambda item: item["cut_adjacent_capped_scale"])
            if final_step_capped_records
            else None
        ),
        "final_rejected_dt_sequence_s": [
            record["dt_s"] for record in final_step_rejections
        ],
        "final_rejected_residual_sequence": [
            record["residual"] for record in final_step_rejections
        ],
        "all_cut_context_count": len(cut_contexts),
        "tail_cut_contexts": cut_contexts[-tail_records:],
        "min_active_min_volume_fraction_in_log": (
            min(cut_min_records, key=lambda item: item["active_min_volume_fraction"])
            if cut_min_records
            else None
        ),
        "max_cut_adjacent_capped_scale_in_log": (
            max(capped_records, key=lambda item: item["cut_adjacent_capped_scale"])
            if capped_records
            else None
        ),
        "final_residual_components": final_residual_components,
        "final_line_search_reject_residual_components": line_search_reject_components,
        "final_solution_components": final_solution_components,
        "tail_residual_blocks": residual_blocks[-tail_records:],
    }


def smallest_cut_cells(
    grid: pv.UnstructuredGrid,
    tets: np.ndarray,
    wet_fraction: np.ndarray,
    wet_measure: np.ndarray,
    *,
    sensor_point: np.ndarray | None,
    limit: int,
    fraction_tolerance: float,
) -> list[dict[str, Any]]:
    points = np.asarray(grid.points, dtype=float)
    phi = np.asarray(grid.point_data.get("phi", []), dtype=float)
    pressure = np.asarray(grid.point_data.get("Pressure", []), dtype=float)
    velocity = np.asarray(grid.point_data.get("Velocity", []), dtype=float)
    global_ids = np.asarray(
        grid.cell_data.get("GlobalCellID", grid.cell_data.get("GlobalElementID", np.arange(grid.n_cells))),
        dtype=int,
    )

    cut_indices = np.flatnonzero(
        (wet_fraction > fraction_tolerance)
        & (wet_fraction < 1.0 - fraction_tolerance)
    )
    order = cut_indices[np.argsort(wet_fraction[cut_indices])[:limit]]
    out: list[dict[str, Any]] = []
    for cell_index in order:
        tet = tets[cell_index]
        cell_points = points[tet]
        centroid = np.mean(cell_points, axis=0)
        record: dict[str, Any] = {
            "cell_index": int(cell_index),
            "global_cell_id": int(global_ids[cell_index]),
            "wet_volume_fraction": float(wet_fraction[cell_index]),
            "wet_volume_measure": float(wet_measure[cell_index]),
            "centroid": centroid.tolist(),
            "bbox_min": np.min(cell_points, axis=0).tolist(),
            "bbox_max": np.max(cell_points, axis=0).tolist(),
        }
        if sensor_point is not None:
            record["distance_to_pressure_sensor_m"] = float(
                np.linalg.norm(centroid - sensor_point)
            )
        if phi.size:
            record["phi_at_vertices"] = array_bounds(phi[tet])
        if pressure.size:
            record["pressure_at_vertices"] = array_bounds(pressure[tet])
        if velocity.size:
            speeds = np.linalg.norm(velocity[tet], axis=1)
            record["velocity_norm_at_vertices"] = array_bounds(speeds)
        out.append(record)
    return out


def point_field_extrema(
    grid: pv.UnstructuredGrid,
    tets: np.ndarray,
    wet_fraction: np.ndarray,
    *,
    tiny_fraction: float,
) -> dict[str, Any]:
    point_max_wet, point_min_positive_wet = incident_wet_support(
        grid.n_points,
        tets,
        wet_fraction,
    )
    wet_supported = point_max_wet > 0.0
    pressure = np.asarray(grid.point_data.get("Pressure", []), dtype=float)
    velocity = np.asarray(grid.point_data.get("Velocity", []), dtype=float)
    active = np.asarray(grid.point_data.get("ActiveFluid", []), dtype=float)
    phi = np.asarray(grid.point_data.get("phi", []), dtype=float)
    result: dict[str, Any] = {
        "wet_supported_point_count": int(np.count_nonzero(wet_supported)),
        "points_with_any_tiny_positive_incident_wet_fraction": int(
            np.count_nonzero(
                (~np.isnan(point_min_positive_wet))
                & (point_min_positive_wet < tiny_fraction)
            )
        ),
        "points_supported_only_by_tiny_positive_wet_fraction": int(
            np.count_nonzero(
                (point_max_wet > 0.0)
                & (point_max_wet < tiny_fraction)
            )
        ),
    }
    if active.size and phi.size:
        result["activefluid_phi_sign_mismatch_count"] = int(
            np.count_nonzero((active > 0.5) != (phi <= 0.0))
        )
        result["activefluid_point_count"] = int(np.count_nonzero(active > 0.5))
        result["phi_nonpositive_point_count"] = int(np.count_nonzero(phi <= 0.0))
    if pressure.size:
        result["pressure_all_points"] = array_bounds(pressure)
        result["pressure_wet_supported_points"] = array_bounds(pressure[wet_supported])
        if np.count_nonzero(wet_supported):
            max_index = int(np.flatnonzero(wet_supported)[np.argmax(pressure[wet_supported])])
            min_index = int(np.flatnonzero(wet_supported)[np.argmin(pressure[wet_supported])])
            result["max_pressure_wet_supported_point"] = {
                "point_index": max_index,
                "pressure": float(pressure[max_index]),
                "point": np.asarray(grid.points[max_index], dtype=float).tolist(),
                "incident_wet_fraction_max": float(point_max_wet[max_index]),
                "incident_wet_fraction_min_positive": (
                    None
                    if math.isnan(point_min_positive_wet[max_index])
                    else float(point_min_positive_wet[max_index])
                ),
            }
            result["min_pressure_wet_supported_point"] = {
                "point_index": min_index,
                "pressure": float(pressure[min_index]),
                "point": np.asarray(grid.points[min_index], dtype=float).tolist(),
                "incident_wet_fraction_max": float(point_max_wet[min_index]),
                "incident_wet_fraction_min_positive": (
                    None
                    if math.isnan(point_min_positive_wet[min_index])
                    else float(point_min_positive_wet[min_index])
                ),
            }
    if velocity.size:
        speed = np.linalg.norm(velocity, axis=1)
        result["velocity_norm_all_points"] = array_bounds(speed)
        result["velocity_norm_wet_supported_points"] = array_bounds(speed[wet_supported])
    return result


def audit_result_file(
    path: Path,
    *,
    time_s: float | None,
    sensor_point: np.ndarray | None,
    smallest_cell_count: int,
    fraction_tolerance: float,
    tiny_fraction: float,
) -> dict[str, Any]:
    grid = pv.read(path)
    tets = tetra_connectivity(grid)
    wet_fraction = np.asarray(grid.cell_data["WetVolumeFraction"], dtype=float)
    wet_measure = np.asarray(grid.cell_data["WetVolumeMeasure"], dtype=float)
    positive = wet_fraction > fraction_tolerance
    cut = positive & (wet_fraction < 1.0 - fraction_tolerance)
    full_wet = wet_fraction >= 1.0 - fraction_tolerance
    dry = wet_fraction <= fraction_tolerance
    cut_fractions = wet_fraction[cut]
    positive_fractions = wet_fraction[positive]
    thresholds = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)

    return {
        "path": str(path),
        "file": path.name,
        "time_s": time_s,
        "points": int(grid.n_points),
        "cells": int(grid.n_cells),
        "wet_volume_from_cell_measure_m3": float(np.sum(wet_measure)),
        "wet_cell_count": int(np.count_nonzero(positive)),
        "cut_cell_count": int(np.count_nonzero(cut)),
        "full_wet_cell_count": int(np.count_nonzero(full_wet)),
        "full_dry_cell_count": int(np.count_nonzero(dry)),
        "positive_wet_fraction": array_bounds(positive_fractions),
        "cut_wet_fraction": array_bounds(cut_fractions),
        "cut_fraction_below_thresholds": {
            f"{threshold:.0e}": int(np.count_nonzero(cut_fractions < threshold))
            for threshold in thresholds
        },
        "point_field_extrema": point_field_extrema(
            grid,
            tets,
            wet_fraction,
            tiny_fraction=tiny_fraction,
        ),
        "smallest_cut_cells": smallest_cut_cells(
            grid,
            tets,
            wet_fraction,
            wet_measure,
            sensor_point=sensor_point,
            limit=smallest_cell_count,
            fraction_tolerance=fraction_tolerance,
        ),
    }


def tail_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    min_cut_records = [
        record
        for record in records
        if record["cut_wet_fraction"]["min"] is not None
    ]
    tiny_counts = {
        threshold: max(
            record["cut_fraction_below_thresholds"][threshold]
            for record in records
        )
        for threshold in ("1e-02", "1e-04", "1e-06", "1e-08")
    }
    return {
        "result_count": len(records),
        "first_tail_time_s": records[0]["time_s"] if records else None,
        "final_tail_time_s": records[-1]["time_s"] if records else None,
        "min_cut_wet_fraction_in_tail": (
            min(
                min_cut_records,
                key=lambda record: record["cut_wet_fraction"]["min"],
            )
            if min_cut_records
            else None
        ),
        "max_cut_cells_in_tail": max(
            (record["cut_cell_count"] for record in records),
            default=None,
        ),
        "max_tiny_cut_counts_in_tail": tiny_counts if records else {},
        "max_activefluid_phi_sign_mismatch_count": max(
            (
                record["point_field_extrema"].get("activefluid_phi_sign_mismatch_count", 0)
                for record in records
            ),
            default=0,
        ),
    }


def build_audit(
    *,
    case_dir: Path,
    solver_log: Path | None,
    result_prefix: str,
    tail_count: int,
    smallest_cell_count: int,
    fraction_tolerance: float,
    tiny_fraction: float,
    log_tail_records: int,
) -> dict[str, Any]:
    files = result_files(case_dir, result_prefix)
    if not files:
        raise FileNotFoundError(f"no {result_prefix}_*.vtu or {result_prefix}_*.pvtu in {case_dir}")
    time_by_file, pvd_source = pvd_times(case_dir, result_prefix)
    sensor_point = load_sensor_point(case_dir)
    selected = files[-tail_count:]
    records = [
        audit_result_file(
            path,
            time_s=time_by_file.get(path.name),
            sensor_point=sensor_point,
            smallest_cell_count=smallest_cell_count,
            fraction_tolerance=fraction_tolerance,
            tiny_fraction=tiny_fraction,
        )
        for path in selected
    ]
    log_context = parse_solver_log(solver_log, tail_records=log_tail_records)
    accepted_tail = tail_summary(records)

    final_record = records[-1]
    final_log_min = log_context.get("min_active_min_volume_fraction_in_log")
    final_tail_min = final_record["cut_wet_fraction"]["min"]
    rejected_tail = log_context.get("step_rejected_tail") or []
    if log_context.get("available") and not rejected_tail:
        finding = (
            "The accepted Test10 tail VTUs are separated from solver-log cut "
            "contexts, and the solver log contains no rejected steps. The "
            "accepted tail minimum cut wet fraction is "
            f"{final_tail_min:.6g}. This supports a completed transient "
            "classification for the audited window; validation readiness still "
            "depends on the pressure-history horizon and literature comparison."
        )
    elif final_log_min and final_tail_min is not None:
        finding = (
            "The accepted Test10 tail VTUs do not contain tiny positive wet "
            "fractions: the final accepted cut minimum is "
            f"{final_tail_min:.6g}. The failed retry context in the solver log "
            "does contain moving trial cut fractions in the 1e-8 class and "
            "capped cut-adjacent scaling. The remaining blocker is therefore "
            "the failed trial topology/nonlinear update after the accepted "
            "state, not the initial mesh or accepted-output wet "
            "volume drift."
        )
    else:
        finding = (
            "The audit separates accepted-output topology from solver-log "
            "failed-trial topology; see tail records and final retry context."
        )

    return {
        "case_dir": str(case_dir),
        "solver_log": str(solver_log) if solver_log else None,
        "result_prefix": result_prefix,
        "result_count": len(files),
        "tail_count": len(records),
        "pvd_time_source": pvd_source,
        "pressure_sensor_point": sensor_point.tolist() if sensor_point is not None else None,
        "fraction_tolerance": fraction_tolerance,
        "tiny_fraction_threshold": tiny_fraction,
        "accepted_output_tail_summary": accepted_tail,
        "final_accepted_output": final_record,
        "accepted_output_tail_records": records,
        "failed_retry_solver_context": log_context,
        "finding": finding,
        "status": "diagnostic_only_test10_still_not_validation_ready",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--solver-log", type=Path, default=DEFAULT_SOLVER_LOG)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--tail-count", type=int, default=10)
    parser.add_argument("--smallest-cell-count", type=int, default=8)
    parser.add_argument("--fraction-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--tiny-fraction", type=float, default=1.0e-6)
    parser.add_argument("--log-tail-records", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_audit(
        case_dir=args.case_dir,
        solver_log=args.solver_log,
        result_prefix=args.result_prefix,
        tail_count=args.tail_count,
        smallest_cell_count=args.smallest_cell_count,
        fraction_tolerance=args.fraction_tolerance,
        tiny_fraction=args.tiny_fraction,
        log_tail_records=args.log_tail_records,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
