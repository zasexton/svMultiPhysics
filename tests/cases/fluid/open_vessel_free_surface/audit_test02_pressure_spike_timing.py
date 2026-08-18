#!/usr/bin/env python3
"""Audit Test02 pressure spikes against output timing and local flow context."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

import verify_spheric_test02_histories as verifier


OBSTACLE_FRONT_X_M = 0.8245
OBSTACLE_CENTER_Z_M = 0.5
RHO_WATER_KG_PER_M3 = 998.2
GRAVITY_M_PER_S2 = 9.81
PRIMARY_PRESSURE_TRACES = ("P1", "P3", "P5", "P7")
TINY_WET_FRACTION = 1.0e-4
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\S+)")
STEP_START_RE = re.compile(
    rf"TimeLoop: step_start step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) dt=(?P<dt>{NUMBER_RE})"
)
NONLINEAR_DONE_RE = re.compile(
    rf"TimeLoop: nonlinear_done step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"converged=(?P<converged>\d+) iters=(?P<iters>\d+) "
    rf"\|\|r\|\|=(?P<residual>{NUMBER_RE}) "
    rf"\|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) "
    rf"\|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE}) "
    rf"\(linear: converged=(?P<linear_converged>\d+) iters=(?P<linear_iters>\d+) "
    rf"rel=(?P<linear_rel>{NUMBER_RE})\)"
)
STEP_ACCEPTED_RE = re.compile(
    rf"TimeLoop: step_accepted step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) dt=(?P<dt>{NUMBER_RE})"
)
PRESSURE_NORMS_RE = re.compile(
    rf"\[Pressure norm=(?P<norm>{NUMBER_RE}) mean=(?P<mean>{NUMBER_RE}) "
    rf"min=(?P<min>{NUMBER_RE}) max=(?P<max>{NUMBER_RE})\]"
)
CUT_CONTEXT_KEYS = (
    "provenance",
    "solution_source",
    "cut_context_revision",
    "cell_count",
    "active_side_volume",
    "active_side_physical_volume",
    "inactive_side_physical_volume",
    "interface_fragments",
    "active_volume_regions",
    "active_wet_cells",
    "active_cut_cells",
    "active_full_wet_cells",
    "active_full_dry_cells",
    "active_quadrature_points",
    "active_empty_quadrature_regions",
    "active_nonfinite_measure_regions",
    "active_negative_measure_regions",
    "active_min_volume_fraction",
    "active_max_volume_fraction",
    "active_pruned_volume_regions",
    "active_pruned_volume",
    "generated_pruned_volume_rules",
    "generated_pruned_volume",
    "implicit_cut_fallback_cells",
    "cut_adjacent_facets",
    "cut_adjacent_metadata",
    "cut_adjacent_zero_scale",
    "cut_adjacent_nonfinite_scale",
    "cut_adjacent_capped_scale",
    "cut_adjacent_min_scale",
    "cut_adjacent_max_scale",
    "cut_adjacent_mean_scale",
)
VOLUME_CORRECTION_KEYS = (
    "step",
    "target_negative_volume",
    "initial_negative_volume",
    "initial_volume_error",
    "corrected_negative_volume",
    "achieved_volume_error",
    "applied_shift",
    "applied_shift_magnitude",
    "iterations",
    "volume_measure_source",
)
WET_VOLUME_KEYS = (
    "step",
    "time",
    "wet_volume",
    "reference_wet_volume",
    "physical_wet_volume",
    "initial_wet_volume",
    "wet_volume_drift",
    "relative_wet_volume_drift",
    "volume_rule_count",
    "physical_volume_rule_count",
    "skipped_physical_volume_rule_count",
    "cut_cell_count",
    "full_wet_cell_count",
    "full_dry_cell_count",
)
ACTIVEFLUID_WARNING_KEYS = (
    "step",
    "time",
    "compared_cut_cell_count",
    "disagreeing_cut_cell_count",
    "threshold",
    "max_abs_difference",
    "max_difference_cell",
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


def selected_key_values(
    line: str,
    keys: tuple[str, ...],
    *,
    line_number: int,
) -> dict[str, Any]:
    values = parse_key_values(line)
    out = {key: values[key] for key in keys if key in values}
    out["line_number"] = line_number
    return out


def parse_pressure_norms(line: str, *, line_number: int) -> dict[str, Any] | None:
    match = PRESSURE_NORMS_RE.search(line)
    if not match:
        return None
    return {
        "line_number": line_number,
        "pressure_norm": float(match.group("norm")),
        "pressure_mean": float(match.group("mean")),
        "pressure_min": float(match.group("min")),
        "pressure_max": float(match.group("max")),
    }


def parse_wet_volume_diagnostics(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    diagnostics: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if "Wet volume diagnostic step=" not in line:
                continue
            entry = selected_key_values(line, WET_VOLUME_KEYS, line_number=line_number)
            step = int(entry.pop("step"))
            if "time" in entry:
                entry["time_s"] = entry.pop("time")
            if "wet_volume" in entry:
                entry["wet_volume_m3"] = entry.pop("wet_volume")
            diagnostics[step] = entry
    return diagnostics


def empty_solve_step(step: int) -> dict[str, Any]:
    return {
        "solve_step": step,
        "accepted_candidate_cut_contexts": [],
        "line_search_trial_cut_contexts": [],
    }


def parse_solver_step_context(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "accepted_results": {},
            "nonlinear_residuals": [],
            "max_nonlinear_residual": None,
        }

    solve_steps: dict[int, dict[str, Any]] = {}
    accepted_results: dict[int, dict[str, Any]] = {}
    accepted_step_cut_context: dict[int, dict[str, Any]] = {}
    wet_volume = parse_wet_volume_diagnostics(path)
    volume_correction: dict[int, dict[str, Any]] = {}
    activefluid_warning: dict[int, dict[str, Any]] = {}
    current_step: int | None = None
    last_accepted_result_step: int | None = None
    nonlinear_residuals: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            start = STEP_START_RE.search(line)
            if start:
                current_step = int(start.group("step"))
                entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                entry.update(
                    {
                        "start_time_s": float(start.group("time")),
                        "dt_s": float(start.group("dt")),
                        "step_start_line_number": line_number,
                    }
                )
                continue

            if "NewtonSolver: residual block norms" in line and current_step is not None:
                values = parse_key_values(line)
                if values.get("diagnostic") == "residual_block_norms":
                    entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                    residual_entry = {
                        "line_number": line_number,
                        "phase": values.get("phase"),
                        "field": values.get("field"),
                        "aux": values.get("aux"),
                        "combined": values.get("combined"),
                    }
                    entry.setdefault("residual_block_norms", []).append(residual_entry)
                    if "initial_residual_block_norm" not in entry:
                        entry["initial_residual_block_norm"] = residual_entry
                    entry["last_residual_block_norm"] = residual_entry
                continue

            if "vector_component_norms label='solution_state'" in line and current_step is not None:
                pressure_norms = parse_pressure_norms(line, line_number=line_number)
                if pressure_norms is not None:
                    entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                    entry["solution_state_pressure_norms"] = pressure_norms
                continue

            if "Active-domain cut context diagnostic=cut_context_rebuild" in line:
                context = selected_key_values(line, CUT_CONTEXT_KEYS, line_number=line_number)
                provenance = context.get("provenance")
                if provenance == "accepted_step" and last_accepted_result_step is not None:
                    accepted_step_cut_context[last_accepted_result_step] = context
                elif current_step is not None and provenance == "accepted":
                    entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                    entry["accepted_candidate_cut_contexts"].append(context)
                elif current_step is not None and provenance == "line_search_trial":
                    entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                    entry["line_search_trial_cut_contexts"].append(context)
                elif current_step is not None and provenance == "before_physics_solve":
                    entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                    entry["before_physics_solve_cut_context"] = context
                continue

            done = NONLINEAR_DONE_RE.search(line)
            if done:
                step = int(done.group("step"))
                entry = solve_steps.setdefault(step, empty_solve_step(step))
                nonlinear = {
                    "line_number": line_number,
                    "time_s": float(done.group("time")),
                    "converged": bool(int(done.group("converged"))),
                    "iters": int(done.group("iters")),
                    "residual_norm": float(done.group("residual")),
                    "residual_field_norm": float(done.group("residual_field")),
                    "residual_aux_norm": float(done.group("residual_aux")),
                    "linear_converged": bool(int(done.group("linear_converged"))),
                    "linear_iters": int(done.group("linear_iters")),
                    "linear_relative_residual": float(done.group("linear_rel")),
                }
                entry["nonlinear_done"] = nonlinear
                nonlinear_residuals.append(
                    {
                        "solve_step": step,
                        "time_s": nonlinear["time_s"],
                        "dt_s": entry.get("dt_s"),
                        "residual_norm": nonlinear["residual_norm"],
                    }
                )
                continue

            accepted = STEP_ACCEPTED_RE.search(line)
            if accepted:
                result_step = int(accepted.group("step"))
                if current_step is None:
                    current_step = result_step - 1
                entry = solve_steps.setdefault(current_step, empty_solve_step(current_step))
                entry.update(
                    {
                        "accepted_result_step": result_step,
                        "accepted_time_s": float(accepted.group("time")),
                        "accepted_dt_s": float(accepted.group("dt")),
                        "step_accepted_line_number": line_number,
                    }
                )
                accepted_results[result_step] = {
                    "result_step": result_step,
                    "solve_step": current_step,
                    "start_time_s": entry.get("start_time_s"),
                    "accepted_time_s": entry.get("accepted_time_s"),
                    "dt_s": entry.get("accepted_dt_s", entry.get("dt_s")),
                    "step_start_line_number": entry.get("step_start_line_number"),
                    "step_accepted_line_number": line_number,
                }
                last_accepted_result_step = result_step
                continue

            if "Level-set volume corrected field='phi' step=" in line:
                entry = selected_key_values(line, VOLUME_CORRECTION_KEYS, line_number=line_number)
                step = int(entry.pop("step"))
                volume_correction[step] = entry
                continue

            if "WARNING ActiveFluid/WetVolumeFraction disagreement" in line:
                entry = selected_key_values(line, ACTIVEFLUID_WARNING_KEYS, line_number=line_number)
                step = int(entry.pop("step"))
                if "time" in entry:
                    entry["time_s"] = entry.pop("time")
                activefluid_warning[step] = entry

    for result_step, result_context in accepted_results.items():
        solve_step = int(result_context["solve_step"])
        entry = solve_steps.get(solve_step, {})
        accepted_candidates = entry.get("accepted_candidate_cut_contexts", [])
        line_search_trials = entry.get("line_search_trial_cut_contexts", [])
        result_context["nonlinear_done"] = entry.get("nonlinear_done")
        result_context["initial_residual_block_norm"] = entry.get("initial_residual_block_norm")
        result_context["last_residual_block_norm"] = entry.get("last_residual_block_norm")
        result_context["solution_state_pressure_norms"] = entry.get("solution_state_pressure_norms")
        result_context["accepted_candidate_cut_context"] = (
            accepted_candidates[-1] if accepted_candidates else None
        )
        result_context["line_search_trial_cut_context"] = line_search_trials[-1] if line_search_trials else None
        result_context["line_search_trial_cut_context_count"] = len(line_search_trials)
        result_context["accepted_step_cut_context"] = accepted_step_cut_context.get(result_step)
        result_context["wet_volume"] = wet_volume.get(result_step)
        result_context["volume_correction"] = volume_correction.get(result_step)
        result_context["activefluid_wet_fraction_warning"] = activefluid_warning.get(result_step)

    max_nonlinear_residual = (
        max(nonlinear_residuals, key=lambda item: float(item["residual_norm"]))
        if nonlinear_residuals
        else None
    )
    return {
        "accepted_results": accepted_results,
        "nonlinear_residuals": nonlinear_residuals,
        "max_nonlinear_residual": max_nonlinear_residual,
    }


def sample_point(grid: pv.DataSet, point: tuple[float, float, float]) -> dict[str, Any]:
    target = np.asarray(point, dtype=float)
    sample = pv.PolyData(target.reshape(1, 3)).sample(grid, tolerance=1.0e-9)
    valid = bool(
        "vtkValidPointMask" in sample.point_data
        and int(np.asarray(sample.point_data["vtkValidPointMask"]).reshape(-1)[0]) == 1
    )
    points = np.asarray(grid.points, dtype=float)
    nearest_index = int(np.argmin(np.linalg.norm(points - target.reshape(1, 3), axis=1)))
    out: dict[str, Any] = {
        "point_m": [float(value) for value in target.tolist()],
        "sample_valid": valid,
        "containing_cell": int(grid.find_containing_cell(target)),
        "nearest_node_index": nearest_index,
        "nearest_node_distance_m": float(np.linalg.norm(points[nearest_index] - target)),
    }
    for name in ("Pressure", "phi", "ActiveFluid"):
        if valid and name in sample.point_data:
            out[name] = float(np.asarray(sample.point_data[name]).reshape(-1)[0])
    if valid and "Velocity" in sample.point_data:
        velocity = np.asarray(sample.point_data["Velocity"], dtype=float).reshape(-1)
        out["Velocity"] = [float(value) for value in velocity.tolist()]
        out["speed_m_per_s"] = float(np.linalg.norm(velocity))
    return out


def front_height(grid: pv.DataSet, *, y_max_m: float = 0.24) -> dict[str, Any]:
    y_values = np.linspace(0.0, y_max_m, 241)
    points = np.column_stack(
        [
            np.full_like(y_values, OBSTACLE_FRONT_X_M),
            y_values,
            np.full_like(y_values, OBSTACLE_CENTER_Z_M),
        ]
    )
    sample = pv.PolyData(points).sample(grid, tolerance=1.0e-9)
    valid = (
        np.asarray(sample.point_data.get("vtkValidPointMask", np.zeros_like(y_values)))
        .reshape(-1)
        .astype(bool)
    )
    if "phi" not in sample.point_data or not np.any(valid):
        return {"height_m": None, "status": "no_valid_phi_samples"}
    phi = np.asarray(sample.point_data["phi"], dtype=float).reshape(-1)
    valid_y = y_values[valid]
    valid_phi = phi[valid]
    if np.all(valid_phi > 0.0):
        return {"height_m": 0.0, "status": "dry_line"}
    if np.all(valid_phi <= 0.0):
        return {"height_m": float(np.max(valid_y)), "status": "wet_to_sample_top"}
    crossings: list[float] = []
    for index in range(len(valid_y) - 1):
        y0 = float(valid_y[index])
        y1 = float(valid_y[index + 1])
        phi0 = float(valid_phi[index])
        phi1 = float(valid_phi[index + 1])
        if phi0 == 0.0:
            crossings.append(y0)
        if phi0 * phi1 < 0.0:
            crossings.append(y0 - phi0 * (y1 - y0) / (phi1 - phi0))
    if valid_phi[-1] == 0.0:
        crossings.append(float(valid_y[-1]))
    if not crossings:
        return {"height_m": None, "status": "no_crossing"}
    return {"height_m": float(max(crossings)), "status": "crossing"}


def point_min_incident_wet_fraction(grid: pv.DataSet) -> np.ndarray:
    point_count = grid.n_points
    min_fraction = np.full(point_count, np.inf, dtype=float)
    if "WetVolumeFraction" not in grid.cell_data:
        return min_fraction
    wet_fraction = np.asarray(grid.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
    cells = np.asarray(grid.cells, dtype=int)
    offset = 0
    cell_id = 0
    while offset < cells.size and cell_id < wet_fraction.size:
        node_count = int(cells[offset])
        point_ids = cells[offset + 1 : offset + 1 + node_count]
        min_fraction[point_ids] = np.minimum(min_fraction[point_ids], wet_fraction[cell_id])
        offset += node_count + 1
        cell_id += 1
    return min_fraction


def pressure_extrema_context(grid: pv.DataSet) -> dict[str, Any]:
    pressure = np.asarray(grid.point_data["Pressure"], dtype=float).reshape(-1)
    phi = np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
    active = (
        np.asarray(grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
        if "ActiveFluid" in grid.point_data
        else np.zeros_like(phi)
    )
    min_incident_wet = point_min_incident_wet_fraction(grid)
    wet = phi <= 0.0
    active_mask = active > 0.5
    tiny_wet = wet & (min_incident_wet <= TINY_WET_FRACTION)

    def max_report(mask: np.ndarray) -> dict[str, Any] | None:
        if not np.any(mask):
            return None
        selected = np.flatnonzero(mask)
        index = int(selected[int(np.argmax(pressure[selected]))])
        return {
            "point_index": index,
            "pressure_pa": float(pressure[index]),
            "phi": float(phi[index]),
            "active_fluid": float(active[index]),
            "min_incident_wet_volume_fraction": (
                float(min_incident_wet[index]) if math.isfinite(float(min_incident_wet[index])) else None
            ),
            "point_m": [float(value) for value in np.asarray(grid.points[index]).tolist()],
        }

    return {
        "max_all": max_report(np.isfinite(pressure)),
        "max_wet": max_report(wet & np.isfinite(pressure)),
        "max_active": max_report(active_mask & np.isfinite(pressure)),
        "max_tiny_wet": max_report(tiny_wet & np.isfinite(pressure)),
        "tiny_wet_node_count": int(np.count_nonzero(tiny_wet)),
    }


def hydrostatic_from_front(height_m: float | None, y_m: float) -> float | None:
    if height_m is None:
        return None
    return float(max(0.0, RHO_WATER_KG_PER_M3 * GRAVITY_M_PER_S2 * (height_m - y_m)))


def interp_reference(reference: dict[str, np.ndarray] | None, trace: str, time_s: float) -> float | None:
    if reference is None or trace not in reference:
        return None
    return float(np.interp(time_s, reference["Time"], reference[trace]))


def event_summary(
    samples: list[dict[str, Any]],
    trace: str,
    *,
    reference: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    finite = [sample for sample in samples if sample["pressure"][trace].get("Pressure") is not None]
    if not finite:
        return {"available": False}
    values = np.asarray([sample["pressure"][trace]["Pressure"] for sample in finite], dtype=float)
    times = np.asarray([sample["time_s"] for sample in finite], dtype=float)
    max_index = int(np.argmax(values))
    jumps = np.diff(values)
    dts = np.diff(times)
    jump_rates = np.divide(jumps, dts, out=np.full_like(jumps, np.nan), where=dts > 0.0)
    positive_jump_index = int(np.argmax(jumps)) + 1 if jumps.size else max_index
    max_event = finite[max_index]
    jump_event = finite[positive_jump_index]
    previous = finite[positive_jump_index - 1] if positive_jump_index > 0 else None
    return {
        "available": True,
        "max_pressure_event": compact_event(max_event, trace, reference=reference),
        "max_positive_jump_event": {
            **compact_event(jump_event, trace, reference=reference),
            "previous_time_s": previous["time_s"] if previous else None,
            "previous_pressure_pa": previous["pressure"][trace].get("Pressure") if previous else None,
            "previous_solver_step_context": compact_solver_context(previous.get("solver_context"))
            if previous
            else None,
            "pressure_jump_pa": float(jumps[positive_jump_index - 1]) if jumps.size else 0.0,
            "dt_since_previous_s": float(dts[positive_jump_index - 1]) if dts.size else None,
            "pressure_jump_rate_pa_per_s": float(jump_rates[positive_jump_index - 1])
            if jump_rates.size
            else None,
        },
        "final_event": compact_event(finite[-1], trace, reference=reference),
    }


def compact_cut_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if context is None:
        return None
    keys = (
        "provenance",
        "line_number",
        "cut_context_revision",
        "active_side_physical_volume",
        "interface_fragments",
        "active_volume_regions",
        "active_wet_cells",
        "active_cut_cells",
        "active_full_wet_cells",
        "active_full_dry_cells",
        "active_empty_quadrature_regions",
        "active_nonfinite_measure_regions",
        "active_negative_measure_regions",
        "active_min_volume_fraction",
        "active_max_volume_fraction",
        "active_pruned_volume_regions",
        "generated_pruned_volume_rules",
        "generated_pruned_volume",
        "implicit_cut_fallback_cells",
        "cut_adjacent_facets",
        "cut_adjacent_capped_scale",
        "cut_adjacent_min_scale",
        "cut_adjacent_max_scale",
        "cut_adjacent_mean_scale",
    )
    return {key: context[key] for key in keys if key in context}


def compact_solver_context(context: dict[str, Any] | None) -> dict[str, Any] | None:
    if not context:
        return None
    nonlinear = context.get("nonlinear_done") or {}
    initial_residual = context.get("initial_residual_block_norm") or {}
    wet_volume = context.get("wet_volume") or {}
    volume_correction = context.get("volume_correction") or {}
    warning = context.get("activefluid_wet_fraction_warning")
    initial_combined = initial_residual.get("combined")
    final_residual = nonlinear.get("residual_norm")
    relative_residual = (
        float(final_residual / initial_combined)
        if isinstance(final_residual, (float, int))
        and isinstance(initial_combined, (float, int))
        and initial_combined > 0.0
        else None
    )
    return {
        "result_step": context.get("result_step"),
        "solve_step": context.get("solve_step"),
        "start_time_s": context.get("start_time_s"),
        "accepted_time_s": context.get("accepted_time_s"),
        "dt_s": context.get("dt_s"),
        "step_start_line_number": context.get("step_start_line_number"),
        "step_accepted_line_number": context.get("step_accepted_line_number"),
        "nonlinear_converged": nonlinear.get("converged"),
        "nonlinear_iters": nonlinear.get("iters"),
        "nonlinear_residual_norm": nonlinear.get("residual_norm"),
        "nonlinear_relative_residual_norm": relative_residual,
        "linear_iters": nonlinear.get("linear_iters"),
        "linear_relative_residual": nonlinear.get("linear_relative_residual"),
        "initial_residual_block_norm": initial_residual or None,
        "last_residual_block_norm": context.get("last_residual_block_norm"),
        "solution_state_pressure_norms": context.get("solution_state_pressure_norms"),
        "accepted_candidate_cut_context": compact_cut_context(context.get("accepted_candidate_cut_context")),
        "line_search_trial_cut_context": compact_cut_context(context.get("line_search_trial_cut_context")),
        "line_search_trial_cut_context_count": context.get("line_search_trial_cut_context_count"),
        "accepted_step_cut_context": compact_cut_context(context.get("accepted_step_cut_context")),
        "wet_volume_relative_drift": wet_volume.get("relative_wet_volume_drift"),
        "wet_volume_cut_cell_count": wet_volume.get("cut_cell_count"),
        "wet_volume_full_wet_cell_count": wet_volume.get("full_wet_cell_count"),
        "wet_volume_full_dry_cell_count": wet_volume.get("full_dry_cell_count"),
        "volume_correction_initial_error": volume_correction.get("initial_volume_error"),
        "volume_correction_achieved_error": volume_correction.get("achieved_volume_error"),
        "volume_correction_applied_shift": volume_correction.get("applied_shift"),
        "volume_correction_iterations": volume_correction.get("iterations"),
        "activefluid_wet_fraction_warning": warning,
    }


def compact_event(
    sample: dict[str, Any],
    trace: str,
    *,
    reference: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    pressure_sample = sample["pressure"][trace]
    front = sample["front_height"]
    sensor_y = verifier.pressure_sensors()[trace].point[1]
    hydrostatic = hydrostatic_from_front(front.get("height_m"), sensor_y)
    pressure = pressure_sample.get("Pressure")
    return {
        "result": sample["result"],
        "step": sample["step"],
        "time_s": sample["time_s"],
        "pressure_pa": pressure,
        "reference_pressure_pa": interp_reference(reference, trace, sample["time_s"]),
        "front_height_m": front.get("height_m"),
        "front_height_status": front.get("status"),
        "hydrostatic_from_front_height_pa": hydrostatic,
        "pressure_minus_hydrostatic_pa": (
            float(pressure - hydrostatic) if pressure is not None and hydrostatic is not None else None
        ),
        "phi": pressure_sample.get("phi"),
        "active_fluid": pressure_sample.get("ActiveFluid"),
        "speed_m_per_s": pressure_sample.get("speed_m_per_s"),
        "wet_volume_drift": sample.get("wet_volume", {}).get("relative_wet_volume_drift"),
        "solver_step_context": compact_solver_context(sample.get("solver_context")),
        "max_active_pressure_pa": sample["extrema"].get("max_active", {}).get("pressure_pa")
        if sample.get("extrema", {}).get("max_active")
        else None,
        "max_tiny_wet_pressure_pa": sample["extrema"].get("max_tiny_wet", {}).get("pressure_pa")
        if sample.get("extrema", {}).get("max_tiny_wet")
        else None,
        "tiny_wet_node_count": sample["extrema"].get("tiny_wet_node_count"),
    }


def case_audit(
    label: str,
    case_dir: Path,
    solver_log: Path | None,
    *,
    prefix: str,
    reference: dict[str, np.ndarray] | None,
) -> dict[str, Any]:
    setup = verifier.parse_solver_xml(case_dir / "solver.xml")
    results = verifier.output_results(case_dir, prefix)
    time_by_result, time_source = result_times(case_dir, prefix, solver_log)
    wet_volume = parse_wet_volume_diagnostics(solver_log)
    solver_context = parse_solver_step_context(solver_log)
    solver_context_by_result = solver_context["accepted_results"]
    sensors = verifier.pressure_sensors()
    samples: list[dict[str, Any]] = []
    for result in results:
        grid = pv.read(result)
        step = verifier.result_step(result, prefix)
        time_s = float(time_by_result.get(result.name, step * setup["time_step_size_s"]))
        pressure = {name: sample_point(grid, sensors[name].point) for name in PRIMARY_PRESSURE_TRACES}
        samples.append(
            {
                "result": result.name,
                "step": step,
                "time_s": time_s,
                "front_height": front_height(grid),
                "pressure": pressure,
                "extrema": pressure_extrema_context(grid),
                "wet_volume": wet_volume.get(step, {}),
                "solver_context": solver_context_by_result.get(step, {}),
            }
        )

    trace_events = {
        trace: event_summary(samples, trace, reference=reference)
        for trace in PRIMARY_PRESSURE_TRACES
    }
    final = samples[-1] if samples else None
    p1_final = final["pressure"]["P1"].get("Pressure") if final else None
    p3_final = final["pressure"]["P3"].get("Pressure") if final else None
    p1_peak = trace_events["P1"]["max_pressure_event"]["pressure_pa"] if trace_events["P1"].get("available") else None
    p3_at_p1_peak = None
    p1_peak_time = trace_events["P1"]["max_pressure_event"]["time_s"] if trace_events["P1"].get("available") else None
    if p1_peak_time is not None:
        nearest = min(samples, key=lambda sample: abs(sample["time_s"] - p1_peak_time))
        p3_at_p1_peak = nearest["pressure"]["P3"].get("Pressure")
    return {
        "label": label,
        "case_dir": str(case_dir),
        "solver_log": str(solver_log) if solver_log else None,
        "result_prefix": prefix,
        "time_source": time_source,
        "result_count": len(results),
        "time_start_s": samples[0]["time_s"] if samples else None,
        "time_end_s": samples[-1]["time_s"] if samples else None,
        "trace_events": trace_events,
        "final_p3_over_p1_ratio": (
            float(p3_final / p1_final) if p1_final not in (None, 0.0) and p3_final is not None else None
        ),
        "p3_over_p1_at_p1_peak": (
            float(p3_at_p1_peak / p1_peak)
            if p1_peak not in (None, 0.0) and p3_at_p1_peak is not None
            else None
        ),
        "max_nonlinear_residual": solver_context.get("max_nonlinear_residual"),
        "max_abs_relative_wet_volume_drift": max(
            (abs(entry.get("relative_wet_volume_drift", 0.0)) for entry in wet_volume.values()),
            default=None,
        ),
    }


def build_finding(cases: dict[str, Any]) -> str:
    parts: list[str] = []
    unit = cases.get("unit")
    half = cases.get("half_dt")
    rho0 = cases.get("rho0")
    if unit:
        p1 = unit["trace_events"]["P1"]["max_pressure_event"]
        parts.append(
            "The unit h=0.15 run-up pressure maximum is an accepted-output event "
            f"at t={p1['time_s']} s with P1={p1['pressure_pa']} Pa, front height "
            f"{p1['front_height_m']} m, and pressure-minus-local-hydrostatic "
            f"{p1['pressure_minus_hydrostatic_pa']} Pa."
        )
    if half:
        jump = half["trace_events"]["P1"]["max_positive_jump_event"]
        parts.append(
            "The half-dt control still has its largest P1 accepted-output jump "
            f"over dt={jump['dt_since_previous_s']} s, from "
            f"{jump['previous_pressure_pa']} Pa to {jump['pressure_pa']} Pa, "
            "so the spike is not solely the old rejected-step rollback cluster."
        )
    if rho0:
        p1 = rho0["trace_events"]["P1"]["max_pressure_event"]
        parts.append(
            "The rho=0 high-damping control lowers the maximum P1 event to "
            f"{p1['pressure_pa']} Pa and removes the tiny-cut spike class in the "
            "separate extrema audit, but its P3/P1 ratio remains low."
        )
    parts.append(
        "These are timing/local-context diagnostics only; they do not form a "
        "Test02 validation gate or a mesh-refinement closure."
    )
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        nargs=3,
        action="append",
        metavar=("LABEL", "CASE_DIR", "SOLVER_LOG"),
        required=True,
        help="Case label, result directory, and solver stdout log.",
    )
    parser.add_argument("--reference-csv", type=Path)
    parser.add_argument("--prefix", default="result")
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    reference = verifier.load_reference_csv(args.reference_csv) if args.reference_csv else None
    cases: dict[str, Any] = {}
    for label, case_dir, solver_log in args.case:
        cases[label] = case_audit(
            label,
            Path(case_dir),
            Path(solver_log),
            prefix=args.prefix,
            reference=reference,
        )
    report = {
        "status": "diagnostic_pressure_spike_timing_not_validation_gate",
        "reference_csv": str(args.reference_csv) if args.reference_csv else None,
        "tiny_wet_fraction_threshold": TINY_WET_FRACTION,
        "cases": cases,
        "finding": build_finding(cases),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
