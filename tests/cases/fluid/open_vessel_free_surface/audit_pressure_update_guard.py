#!/usr/bin/env python3
"""Audit accepted pressure updates on active/wet support in saved VTU windows."""

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


NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RESULT_RE_TEMPLATE = r"{prefix}_(\d+)\.p?vtu$"
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\S+)")
STEP_START_RE = re.compile(
    rf"TimeLoop: step_start step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE})"
)
STEP_ACCEPTED_RE = re.compile(
    rf"TimeLoop: step_accepted step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE})"
)
STEP_REJECTED_RE = re.compile(
    rf"TimeLoop: step_rejected step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE}) reason=(?P<reason>\S+) "
    rf"\(newton: converged=(?P<converged>[01]) iters=(?P<iters>\d+) "
    rf"\|\|r\|\|=(?P<residual>{NUMBER_RE}) "
    rf"\|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) "
    rf"\|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE})\)"
)
NONLINEAR_DONE_RE = re.compile(
    rf"TimeLoop: nonlinear_done step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"converged=(?P<converged>[01]) iters=(?P<iters>\d+) "
    rf"\|\|r\|\|=(?P<residual>{NUMBER_RE}) "
    rf"\|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) "
    rf"\|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE}) "
    rf"\(linear: converged=(?P<linear_converged>[01]) "
    rf"iters=(?P<linear_iters>\d+) rel=(?P<linear_rel>{NUMBER_RE})\)"
)
CUT_CONTEXT_KEYS = (
    "provenance",
    "solution_source",
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
    "cut_adjacent_facets",
    "cut_adjacent_capped_scale",
    "cut_adjacent_min_scale",
    "cut_adjacent_max_scale",
    "cut_adjacent_mean_scale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare consecutive result VTUs and report pressure increments on "
            "active/wet/cut-supported nodes. This is a diagnostic guard for "
            "accepted Test02/Test10 pressure jumps, not a validation gate."
        )
    )
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument(
        "--previous-result",
        type=Path,
        help=(
            "Direct-pair mode: previous VTU/PVTU result to compare. Requires "
            "--current-result and bypasses case-directory result discovery."
        ),
    )
    parser.add_argument(
        "--current-result",
        type=Path,
        help=(
            "Direct-pair mode: current VTU/PVTU result to compare. Requires "
            "--previous-result and bypasses case-directory result discovery."
        ),
    )
    parser.add_argument("--previous-time", type=float)
    parser.add_argument("--current-time", type=float)
    parser.add_argument("--result-prefix", default="result")
    parser.add_argument("--solver-log", type=Path)
    parser.add_argument("--start-step", type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--top-events", type=int, default=12)
    parser.add_argument("--active-fluid-threshold", type=float, default=0.5)
    parser.add_argument("--tiny-wet-fraction", type=float, default=1.0e-4)
    parser.add_argument("--full-wet-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--absolute-threshold-pa", type=float, default=1.0e3)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--fail-on-trigger",
        action="store_true",
        help="Exit nonzero when any active/wet update exceeds the threshold.",
    )
    return parser.parse_args()


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


def parse_solver_log(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}

    accepted: dict[int, dict[str, Any]] = {}
    rejected_by_attempt_step: dict[int, list[dict[str, Any]]] = {}
    current_start: dict[str, Any] | None = None
    last_nonlinear: dict[int, dict[str, Any]] = {}
    cut_contexts: dict[int, list[dict[str, Any]]] = {}
    pressure_update_guards: dict[int, dict[str, Any]] = {}

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            start = STEP_START_RE.search(line)
            if start:
                current_start = {
                    "line_number": line_number,
                    "attempt_step": int(start.group("step")),
                    "attempt_time_s": float(start.group("time")),
                    "dt_s": float(start.group("dt")),
                }
                continue

            nonlinear = NONLINEAR_DONE_RE.search(line)
            if nonlinear:
                step = int(nonlinear.group("step"))
                last_nonlinear[step] = {
                    "line_number": line_number,
                    "converged": bool(int(nonlinear.group("converged"))),
                    "iters": int(nonlinear.group("iters")),
                    "residual": float(nonlinear.group("residual")),
                    "residual_field": float(nonlinear.group("residual_field")),
                    "residual_aux": float(nonlinear.group("residual_aux")),
                    "linear_converged": bool(int(nonlinear.group("linear_converged"))),
                    "linear_iters": int(nonlinear.group("linear_iters")),
                    "linear_rel": float(nonlinear.group("linear_rel")),
                }
                continue

            rejected = STEP_REJECTED_RE.search(line)
            if rejected:
                step = int(rejected.group("step"))
                rejected_by_attempt_step.setdefault(step, []).append(
                    {
                        "line_number": line_number,
                        "time_s": float(rejected.group("time")),
                        "dt_s": float(rejected.group("dt")),
                        "reason": rejected.group("reason"),
                        "newton_converged": bool(int(rejected.group("converged"))),
                        "newton_iters": int(rejected.group("iters")),
                        "residual": float(rejected.group("residual")),
                        "residual_field": float(rejected.group("residual_field")),
                        "residual_aux": float(rejected.group("residual_aux")),
                    }
                )
                continue

            if "diagnostic=cut_context_rebuild" in line and current_start is not None:
                values = parse_key_values(line)
                record = {
                    "line_number": line_number,
                    **{key: values[key] for key in CUT_CONTEXT_KEYS if key in values},
                }
                cut_contexts.setdefault(current_start["attempt_step"], []).append(record)
                continue

            if "diagnostic=accepted_pressure_update_guard" in line:
                values = parse_key_values(line)
                step = values.get("step")
                if isinstance(step, int):
                    record = {
                        "line_number": line_number,
                        **values,
                    }
                    pressure_update_guards[step] = record
                    if step in accepted:
                        accepted[step]["pressure_update_guard"] = record
                continue

            step_accepted = STEP_ACCEPTED_RE.search(line)
            if step_accepted:
                to_step = int(step_accepted.group("step"))
                attempt_step = (
                    int(current_start["attempt_step"])
                    if current_start is not None
                    else to_step - 1
                )
                accepted[to_step] = {
                    "line_number": line_number,
                    "to_step": to_step,
                    "to_time_s": float(step_accepted.group("time")),
                    "dt_s": float(step_accepted.group("dt")),
                    "attempt_step": attempt_step,
                    "step_start": current_start,
                    "nonlinear": last_nonlinear.get(attempt_step),
                    "preceding_rejected_attempts": rejected_by_attempt_step.get(
                        attempt_step, []
                    ),
                    "cut_context_rebuilds": cut_contexts.get(attempt_step, []),
                    "pressure_update_guard": pressure_update_guards.get(to_step),
                }
                continue

    return accepted


def cell_point_ids(grid: pv.DataSet) -> list[np.ndarray]:
    cells = np.asarray(grid.cells, dtype=np.int64)
    out: list[np.ndarray] = []
    offset = 0
    while offset < cells.size:
        node_count = int(cells[offset])
        out.append(cells[offset + 1 : offset + 1 + node_count])
        offset += node_count + 1
    return out


def point_wet_support(grid: pv.DataSet) -> dict[str, np.ndarray]:
    n_points = int(grid.n_points)
    max_fraction = np.full(n_points, math.nan, dtype=float)
    min_positive = np.full(n_points, math.nan, dtype=float)
    incident_count = np.zeros(n_points, dtype=np.int64)
    positive_count = np.zeros(n_points, dtype=np.int64)

    if "WetVolumeFraction" not in grid.cell_data:
        return {
            "incident_wet_fraction_max": max_fraction,
            "incident_wet_fraction_min_positive": min_positive,
            "incident_cell_count": incident_count,
            "positive_wet_incident_cell_count": positive_count,
        }

    wet_fraction = np.asarray(grid.cell_data["WetVolumeFraction"], dtype=float).reshape(-1)
    for cell_index, points in enumerate(cell_point_ids(grid)):
        if cell_index >= wet_fraction.size:
            break
        fraction = float(wet_fraction[cell_index])
        for point in points:
            point_index = int(point)
            incident_count[point_index] += 1
            if not math.isfinite(fraction) or fraction <= 0.0:
                continue
            positive_count[point_index] += 1
            current_max = max_fraction[point_index]
            max_fraction[point_index] = (
                fraction
                if math.isnan(current_max)
                else max(current_max, fraction)
            )
            current_min = min_positive[point_index]
            min_positive[point_index] = (
                fraction
                if math.isnan(current_min)
                else min(current_min, fraction)
            )

    return {
        "incident_wet_fraction_max": max_fraction,
        "incident_wet_fraction_min_positive": min_positive,
        "incident_cell_count": incident_count,
        "positive_wet_incident_cell_count": positive_count,
    }


def support_class(
    *,
    phi: float | None,
    active_fluid: float | None,
    incident_wet_fraction_max: float | None,
    incident_wet_fraction_min_positive: float | None,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> str:
    active_by_field = active_fluid is not None and active_fluid > active_threshold
    active_by_phi = phi is not None and phi <= 0.0
    has_wet_fraction = (
        incident_wet_fraction_max is not None
        and math.isfinite(incident_wet_fraction_max)
        and incident_wet_fraction_max > 0.0
    )
    if has_wet_fraction:
        if incident_wet_fraction_max <= tiny_wet_fraction:
            return "tiny_cut_supported"
        if (
            incident_wet_fraction_min_positive is not None
            and math.isfinite(incident_wet_fraction_min_positive)
            and incident_wet_fraction_min_positive >= 1.0 - full_wet_tolerance
        ):
            return "full_wet_supported"
        return "cut_supported"
    if active_by_field or active_by_phi:
        return "active_without_wet_fraction_data"
    return "dry_or_inactive"


def finite_float(value: float | np.floating[Any]) -> float | None:
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def event_report(
    *,
    grid: pv.DataSet,
    point_index: int,
    delta: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    previous_step: int,
    current_step: int,
    previous_time_s: float | None,
    current_time_s: float | None,
    support: dict[str, np.ndarray],
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, Any]:
    points = np.asarray(grid.points, dtype=float)
    phi = (
        np.asarray(grid.point_data["phi"], dtype=float).reshape(-1)
        if "phi" in grid.point_data
        else np.full(grid.n_points, math.nan, dtype=float)
    )
    active = (
        np.asarray(grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
        if "ActiveFluid" in grid.point_data
        else np.full(grid.n_points, math.nan, dtype=float)
    )
    velocity = (
        np.asarray(grid.point_data["Velocity"], dtype=float)
        if "Velocity" in grid.point_data
        else None
    )
    max_wet = support["incident_wet_fraction_max"][point_index]
    min_positive = support["incident_wet_fraction_min_positive"][point_index]
    phi_value = finite_float(phi[point_index])
    active_value = finite_float(active[point_index])
    max_wet_value = finite_float(max_wet)
    min_positive_value = finite_float(min_positive)
    report = {
        "from_step": previous_step,
        "to_step": current_step,
        "from_time_s": previous_time_s,
        "to_time_s": current_time_s,
        "point_index": int(point_index),
        "point_m": [float(value) for value in points[point_index].tolist()],
        "pressure_delta_pa": float(delta[point_index]),
        "abs_pressure_delta_pa": float(abs(delta[point_index])),
        "from_pressure_pa": float(previous_pressure[point_index]),
        "to_pressure_pa": float(current_pressure[point_index]),
        "phi": phi_value,
        "active_fluid": active_value,
        "support_class": support_class(
            phi=phi_value,
            active_fluid=active_value,
            incident_wet_fraction_max=max_wet_value,
            incident_wet_fraction_min_positive=min_positive_value,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        ),
        "incident_cell_count": int(support["incident_cell_count"][point_index]),
        "positive_wet_incident_cell_count": int(
            support["positive_wet_incident_cell_count"][point_index]
        ),
        "incident_wet_fraction_max": max_wet_value,
        "incident_wet_fraction_min_positive": min_positive_value,
    }
    if velocity is not None:
        report["velocity_m_per_s"] = [
            float(value) for value in velocity[point_index].tolist()
        ]
        report["speed_m_per_s"] = float(np.linalg.norm(velocity[point_index]))
    return report


def max_event_for_mask(
    *,
    mask: np.ndarray,
    grid: pv.DataSet,
    delta: np.ndarray,
    previous_pressure: np.ndarray,
    current_pressure: np.ndarray,
    previous_step: int,
    current_step: int,
    previous_time_s: float | None,
    current_time_s: float | None,
    support: dict[str, np.ndarray],
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
) -> dict[str, Any] | None:
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    point_index = int(indices[int(np.argmax(np.abs(delta[indices])))])
    return event_report(
        grid=grid,
        point_index=point_index,
        delta=delta,
        previous_pressure=previous_pressure,
        current_pressure=current_pressure,
        previous_step=previous_step,
        current_step=current_step,
        previous_time_s=previous_time_s,
        current_time_s=current_time_s,
        support=support,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
    )


def pressure_delta_statistics(mask: np.ndarray, delta: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "count": 0,
            "mean_delta_pa": None,
            "median_delta_pa": None,
            "min_delta_pa": None,
            "max_delta_pa": None,
            "rms_delta_pa": None,
            "max_abs_delta_pa": None,
            "max_abs_after_mean_removal_pa": None,
            "max_abs_after_median_removal_pa": None,
            "median_removed_to_raw_max_ratio": None,
        }

    values = np.asarray(delta[mask], dtype=float)
    mean = float(np.mean(values))
    median = float(np.median(values))
    max_abs = float(np.max(np.abs(values)))
    max_abs_after_mean = float(np.max(np.abs(values - mean)))
    max_abs_after_median = float(np.max(np.abs(values - median)))
    return {
        "count": int(values.size),
        "mean_delta_pa": mean,
        "median_delta_pa": median,
        "min_delta_pa": float(np.min(values)),
        "max_delta_pa": float(np.max(values)),
        "rms_delta_pa": float(np.sqrt(np.mean(values * values))),
        "max_abs_delta_pa": max_abs,
        "max_abs_after_mean_removal_pa": max_abs_after_mean,
        "max_abs_after_median_removal_pa": max_abs_after_median,
        "median_removed_to_raw_max_ratio": (
            max_abs_after_median / max_abs if max_abs > 0.0 else None
        ),
    }


def transition_report(
    previous_result: Path,
    current_result: Path,
    *,
    result_prefix: str,
    time_by_result: dict[str, float],
    log_context_by_step: dict[int, dict[str, Any]],
    top_events: int,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    previous_time_override: float | None = None,
    current_time_override: float | None = None,
) -> dict[str, Any]:
    previous_grid = pv.read(previous_result)
    current_grid = pv.read(current_result)
    if previous_grid.n_points != current_grid.n_points:
        raise RuntimeError(
            f"Cannot compare {previous_result} and {current_result}: "
            "point counts differ"
        )
    if "Pressure" not in previous_grid.point_data or "Pressure" not in current_grid.point_data:
        raise RuntimeError("Both result files must contain point-data Pressure")

    previous_step = result_step(previous_result, result_prefix)
    current_step = result_step(current_result, result_prefix)
    previous_time_s = (
        previous_time_override
        if previous_time_override is not None
        else time_by_result.get(previous_result.name)
    )
    current_time_s = (
        current_time_override
        if current_time_override is not None
        else time_by_result.get(current_result.name)
    )
    previous_pressure = np.asarray(
        previous_grid.point_data["Pressure"], dtype=float
    ).reshape(-1)
    current_pressure = np.asarray(
        current_grid.point_data["Pressure"], dtype=float
    ).reshape(-1)
    delta = current_pressure - previous_pressure

    phi = (
        np.asarray(current_grid.point_data["phi"], dtype=float).reshape(-1)
        if "phi" in current_grid.point_data
        else np.full(current_grid.n_points, math.nan, dtype=float)
    )
    active = (
        np.asarray(current_grid.point_data["ActiveFluid"], dtype=float).reshape(-1)
        if "ActiveFluid" in current_grid.point_data
        else np.full(current_grid.n_points, math.nan, dtype=float)
    )
    support = point_wet_support(current_grid)
    max_wet = support["incident_wet_fraction_max"]
    min_positive = support["incident_wet_fraction_min_positive"]

    active_or_phi_wet = (active > active_threshold) | (phi <= 0.0)
    wet_supported = np.isfinite(max_wet) & (max_wet > 0.0)
    active_wet_supported = active_or_phi_wet | wet_supported
    full_wet_supported = (
        wet_supported
        & np.isfinite(min_positive)
        & (min_positive >= 1.0 - full_wet_tolerance)
    )
    cut_supported = wet_supported & ~full_wet_supported
    tiny_cut_supported = wet_supported & (max_wet <= tiny_wet_fraction)

    category_masks = {
        "all_points": np.ones(current_grid.n_points, dtype=bool),
        "active_or_wet_supported": active_wet_supported,
        "full_wet_supported": full_wet_supported,
        "cut_supported": cut_supported,
        "tiny_cut_supported": tiny_cut_supported,
    }
    max_by_category = {
        name: max_event_for_mask(
            mask=mask,
            grid=current_grid,
            delta=delta,
            previous_pressure=previous_pressure,
            current_pressure=current_pressure,
            previous_step=previous_step,
            current_step=current_step,
            previous_time_s=previous_time_s,
            current_time_s=current_time_s,
            support=support,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        )
        for name, mask in category_masks.items()
    }
    delta_statistics_by_category = {
        name: pressure_delta_statistics(mask, delta)
        for name, mask in category_masks.items()
    }

    top_indices = np.argsort(-np.abs(delta))[:top_events]
    top_pressure_updates = [
        event_report(
            grid=current_grid,
            point_index=int(point_index),
            delta=delta,
            previous_pressure=previous_pressure,
            current_pressure=current_pressure,
            previous_step=previous_step,
            current_step=current_step,
            previous_time_s=previous_time_s,
            current_time_s=current_time_s,
            support=support,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        )
        for point_index in top_indices
    ]

    return {
        "from_result": previous_result.name,
        "to_result": current_result.name,
        "from_result_path": str(previous_result),
        "to_result_path": str(current_result),
        "from_step": previous_step,
        "to_step": current_step,
        "from_time_s": previous_time_s,
        "to_time_s": current_time_s,
        "point_count": int(current_grid.n_points),
        "support_counts": {
            "active_or_phi_wet_points": int(np.count_nonzero(active_or_phi_wet)),
            "wet_supported_points": int(np.count_nonzero(wet_supported)),
            "active_or_wet_supported_points": int(
                np.count_nonzero(active_wet_supported)
            ),
            "full_wet_supported_points": int(np.count_nonzero(full_wet_supported)),
            "cut_supported_points": int(np.count_nonzero(cut_supported)),
            "tiny_cut_supported_points": int(np.count_nonzero(tiny_cut_supported)),
        },
        "max_by_category": max_by_category,
        "delta_statistics_by_category": delta_statistics_by_category,
        "top_pressure_updates": top_pressure_updates,
        "solver_log_context": log_context_by_step.get(current_step),
    }


def update_worst(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return (
        candidate
        if candidate["abs_pressure_delta_pa"] > current["abs_pressure_delta_pa"]
        else current
    )


def summarize_transitions(
    *,
    transitions: list[dict[str, Any]],
    case_dir: Path | None,
    result_prefix: str,
    solver_log: Path | None,
    time_source: str,
    start_step: int | None,
    end_step: int | None,
    absolute_threshold_pa: float,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    worst_by_category: dict[str, dict[str, Any] | None] = {
        "all_points": None,
        "active_or_wet_supported": None,
        "full_wet_supported": None,
        "cut_supported": None,
        "tiny_cut_supported": None,
    }
    triggered_transitions: list[dict[str, Any]] = []

    for transition in transitions:
        for category, event in transition["max_by_category"].items():
            worst_by_category[category] = update_worst(worst_by_category[category], event)
        active_event = transition["max_by_category"]["active_or_wet_supported"]
        if (
            active_event is not None
            and active_event["abs_pressure_delta_pa"] > absolute_threshold_pa
        ):
            triggered_transitions.append(
                {
                    "from_step": transition["from_step"],
                    "to_step": transition["to_step"],
                    "from_time_s": transition["from_time_s"],
                    "to_time_s": transition["to_time_s"],
                    "abs_pressure_delta_pa": active_event["abs_pressure_delta_pa"],
                    "support_class": active_event["support_class"],
                    "point_index": active_event["point_index"],
                    "point_m": active_event["point_m"],
                }
            )

    worst_active = worst_by_category["active_or_wet_supported"]
    finding = "No active/wet pressure update exceeded the configured threshold."
    status = "diagnostic_pressure_update_guard_no_threshold_trigger"
    if triggered_transitions:
        status = "diagnostic_pressure_update_guard_triggered"
        finding = (
            f"{len(triggered_transitions)} transition(s) exceeded "
            f"{absolute_threshold_pa:g} Pa on active/wet support."
        )
        if worst_active is not None:
            finding += (
                f" Worst active/wet update was "
                f"{worst_active['abs_pressure_delta_pa']:.6g} Pa "
                f"from step {worst_active['from_step']} to "
                f"{worst_active['to_step']} on {worst_active['support_class']}."
            )

    report = {
        "case_dir": str(case_dir) if case_dir else None,
        "result_prefix": result_prefix,
        "solver_log": str(solver_log) if solver_log else None,
        "time_source": time_source,
        "start_step": start_step,
        "end_step": end_step,
        "transition_count": len(transitions),
        "absolute_threshold_pa": absolute_threshold_pa,
        "active_fluid_threshold": active_threshold,
        "tiny_wet_fraction": tiny_wet_fraction,
        "full_wet_tolerance": full_wet_tolerance,
        "status": status,
        "finding": finding,
        "triggered_transition_count": len(triggered_transitions),
        "triggered_transitions": triggered_transitions,
        "worst_by_category": worst_by_category,
        "transitions": transitions,
    }
    if extra_fields:
        report.update(extra_fields)
    return report


def audit_case(
    case_dir: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
    start_step: int | None,
    end_step: int | None,
    top_events: int,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    absolute_threshold_pa: float,
) -> dict[str, Any]:
    files = result_files(case_dir, result_prefix)
    if len(files) < 2:
        raise RuntimeError(f"Need at least two result files in {case_dir}")
    time_by_result, time_source = pvd_times(case_dir, result_prefix)
    log_context_by_step = parse_solver_log(solver_log)

    transitions: list[dict[str, Any]] = []

    for previous_result, current_result in zip(files, files[1:]):
        current_step = result_step(current_result, result_prefix)
        if start_step is not None and current_step < start_step:
            continue
        if end_step is not None and current_step > end_step:
            continue
        transition = transition_report(
            previous_result,
            current_result,
            result_prefix=result_prefix,
            time_by_result=time_by_result,
            log_context_by_step=log_context_by_step,
            top_events=top_events,
            active_threshold=active_threshold,
            tiny_wet_fraction=tiny_wet_fraction,
            full_wet_tolerance=full_wet_tolerance,
        )
        transitions.append(transition)

    return summarize_transitions(
        transitions=transitions,
        case_dir=case_dir,
        result_prefix=result_prefix,
        solver_log=solver_log,
        time_source=time_source or "result_step_only",
        start_step=start_step,
        end_step=end_step,
        absolute_threshold_pa=absolute_threshold_pa,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
        extra_fields={"comparison_mode": "case_window"},
    )


def audit_direct_pair(
    previous_result: Path,
    current_result: Path,
    *,
    result_prefix: str,
    solver_log: Path | None,
    previous_time: float | None,
    current_time: float | None,
    top_events: int,
    active_threshold: float,
    tiny_wet_fraction: float,
    full_wet_tolerance: float,
    absolute_threshold_pa: float,
) -> dict[str, Any]:
    if not previous_result.exists():
        raise FileNotFoundError(previous_result)
    if not current_result.exists():
        raise FileNotFoundError(current_result)

    log_context_by_step = parse_solver_log(solver_log)
    transition = transition_report(
        previous_result,
        current_result,
        result_prefix=result_prefix,
        time_by_result={},
        log_context_by_step=log_context_by_step,
        top_events=top_events,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
        previous_time_override=previous_time,
        current_time_override=current_time,
    )
    return summarize_transitions(
        transitions=[transition],
        case_dir=None,
        result_prefix=result_prefix,
        solver_log=solver_log,
        time_source="direct_pair_arguments"
        if previous_time is not None or current_time is not None
        else "result_step_only",
        start_step=None,
        end_step=None,
        absolute_threshold_pa=absolute_threshold_pa,
        active_threshold=active_threshold,
        tiny_wet_fraction=tiny_wet_fraction,
        full_wet_tolerance=full_wet_tolerance,
        extra_fields={
            "comparison_mode": "direct_pair",
            "previous_result": str(previous_result),
            "current_result": str(current_result),
            "previous_time_s": previous_time,
            "current_time_s": current_time,
        },
    )


def main() -> int:
    args = parse_args()
    direct_pair_requested = (
        args.previous_result is not None or args.current_result is not None
    )
    if direct_pair_requested:
        if args.previous_result is None or args.current_result is None:
            raise SystemExit(
                "--previous-result and --current-result must be provided together"
            )
        report = audit_direct_pair(
            args.previous_result,
            args.current_result,
            result_prefix=args.result_prefix,
            solver_log=args.solver_log,
            previous_time=args.previous_time,
            current_time=args.current_time,
            top_events=args.top_events,
            active_threshold=args.active_fluid_threshold,
            tiny_wet_fraction=args.tiny_wet_fraction,
            full_wet_tolerance=args.full_wet_tolerance,
            absolute_threshold_pa=args.absolute_threshold_pa,
        )
    else:
        if args.case_dir is None:
            raise SystemExit(
                "--case-dir is required unless --previous-result/--current-result "
                "direct-pair mode is used"
            )
        report = audit_case(
            args.case_dir,
            result_prefix=args.result_prefix,
            solver_log=args.solver_log,
            start_step=args.start_step,
            end_step=args.end_step,
            top_events=args.top_events,
            active_threshold=args.active_fluid_threshold,
            tiny_wet_fraction=args.tiny_wet_fraction,
            full_wet_tolerance=args.full_wet_tolerance,
            absolute_threshold_pa=args.absolute_threshold_pa,
        )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.fail_on_trigger and report["triggered_transition_count"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
