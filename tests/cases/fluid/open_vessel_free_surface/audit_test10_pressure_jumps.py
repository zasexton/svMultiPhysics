#!/usr/bin/env python3
"""Audit accepted Test10 pressure-history jumps against the SPHERIC record."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TIME_TOLERANCE = 1.0e-10
RESIDUAL_ABS_TOLERANCE = 2.0e-2
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
STEP_START_RE = re.compile(
    rf"TimeLoop: step_start step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) dt=(?P<dt>{NUMBER_RE})"
)
STEP_ACCEPTED_RE = re.compile(
    rf"TimeLoop: step_accepted step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) dt=(?P<dt>{NUMBER_RE})"
)
STEP_REJECTED_RE = re.compile(
    rf"TimeLoop: step_rejected step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) dt=(?P<dt>{NUMBER_RE}) "
    rf"reason=(?P<reason>\S+) \(newton: converged=(?P<converged>[01]) iters=(?P<iters>\d+) "
    rf"\|\|r\|\|=(?P<residual>{NUMBER_RE}) \|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) "
    rf"\|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE})\)"
)
NONLINEAR_DONE_RE = re.compile(
    rf"TimeLoop: nonlinear_done step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"converged=(?P<converged>[01]) iters=(?P<iters>\d+) \|\|r\|\|=(?P<residual>{NUMBER_RE}) "
    rf"\|\|r_field\|\|=(?P<residual_field>{NUMBER_RE}) \|\|r_aux\|\|=(?P<residual_aux>{NUMBER_RE}) "
    rf"\(linear: converged=(?P<linear_converged>[01]) iters=(?P<linear_iters>\d+) "
    rf"rel=(?P<linear_relative_residual>{NUMBER_RE})\)"
)
RESIDUAL_BLOCK_RE = re.compile(
    rf"NewtonSolver: residual block norms diagnostic=residual_block_norms "
    rf"phase='(?P<phase>[^']+)' field=(?P<field>{NUMBER_RE}) "
    rf"aux=(?P<aux>{NUMBER_RE}) combined=(?P<combined>{NUMBER_RE})"
)
VECTOR_COMPONENT_RE = re.compile(
    r"NewtonSolver: vector component norms diagnostic=vector_component_norms "
    r"label='(?P<label>[^']+)' (?P<components>.*)"
)
COMPONENT_RE = re.compile(
    rf"\[(?P<name>.+?) norm=(?P<norm>{NUMBER_RE}) mean=(?P<mean>{NUMBER_RE}) "
    rf"min=(?P<min>{NUMBER_RE}) max=(?P<max>{NUMBER_RE})\]"
)
CUT_CONTEXT_INT_KEYS = (
    "active_wet_cells",
    "active_cut_cells",
    "active_full_wet_cells",
    "active_full_dry_cells",
    "active_quadrature_points",
    "active_volume_rule_count",
    "active_pruned_volume_regions",
    "generated_pruned_volume_rules",
    "cut_adjacent_capped_scale",
)
CUT_CONTEXT_FLOAT_KEYS = (
    "active_min_volume_fraction",
    "active_max_volume_fraction",
    "active_side_physical_volume",
    "active_side_raw_volume",
    "active_pruned_volume",
    "generated_pruned_volume",
    "cut_adjacent_min_scale",
    "cut_adjacent_max_scale",
    "cut_adjacent_mean_scale",
)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_spheric_test10_pressure_history import (  # noqa: E402
    DEFAULT_REFERENCE_MEMBER,
    load_reference_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect accepted Sensor1 pressure jumps from a Test10 pressure "
            "comparison JSON and compare them with the official SPHERIC record."
        )
    )
    parser.add_argument(
        "--pressure-comparison",
        required=True,
        type=Path,
        help="JSON produced by verify_spheric_test10_pressure_history.py.",
    )
    parser.add_argument(
        "--active-topology-audit",
        type=Path,
        help="Optional JSON produced by audit_test10_active_topology.py.",
    )
    parser.add_argument(
        "--solver-log",
        type=Path,
        help="Optional solver stdout log containing TimeLoop diagnostics.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        help="Optional local SPHERIC Test10 lateral_water_1x reference table.",
    )
    parser.add_argument(
        "--fetch-reference",
        action="store_true",
        help="Fetch the official SPHERIC Test10 reference table if no local file is provided.",
    )
    parser.add_argument(
        "--reference-member",
        default=DEFAULT_REFERENCE_MEMBER,
        help="Archive member to fetch when --fetch-reference is used.",
    )
    parser.add_argument(
        "--top-jumps",
        type=int,
        default=8,
        help="Number of largest accepted jumps to include.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional output path for the audit JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close_time(left: float, right: float) -> bool:
    return abs(left - right) <= TIME_TOLERANCE * max(1.0, abs(left), abs(right))


def to_float(match: re.Match[str], key: str) -> float:
    return float(match.group(key))


def to_int(match: re.Match[str], key: str) -> int:
    return int(match.group(key))


def token_text(line: str, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}=(?P<value>\S+)", line)
    return None if match is None else match.group("value")


def token_float(line: str, key: str) -> float | None:
    value = token_text(line, key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def token_int(line: str, key: str) -> int | None:
    value = token_text(line, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_cut_context(line_no: int, line: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "line_no": line_no,
        "provenance": token_text(line, "provenance"),
    }
    for key in CUT_CONTEXT_INT_KEYS:
        value = token_int(line, key)
        if value is not None:
            record[key] = value
    for key in CUT_CONTEXT_FLOAT_KEYS:
        value = token_float(line, key)
        if value is not None:
            record[key] = value
    return record


def parse_vector_components(text: str) -> dict[str, dict[str, float]]:
    components: dict[str, dict[str, float]] = {}
    for match in COMPONENT_RE.finditer(text):
        components[match.group("name")] = {
            "norm": to_float(match, "norm"),
            "mean": to_float(match, "mean"),
            "min": to_float(match, "min"),
            "max": to_float(match, "max"),
        }
    return components


def residual_block_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "line_no": record["line_no"],
        "phase": record["phase"],
        "field_norm": record["field_norm"],
        "aux_norm": record["aux_norm"],
        "combined_norm": record["combined_norm"],
    }


def solution_snapshot_summary(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    out: dict[str, Any] = {
        "line_no": snapshot["line_no"],
        "phase": snapshot.get("phase"),
        "components": snapshot["components"],
    }
    residual_block = residual_block_summary(snapshot.get("residual_block"))
    if residual_block is not None:
        out["residual_block_norm"] = residual_block
    return out


def first_record_with_phase(records: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("phase") == phase), None)


def last_record_with_phase(records: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    return next((record for record in reversed(records) if record.get("phase") == phase), None)


def solution_component_change(
    initial: dict[str, Any] | None,
    accepted: dict[str, Any] | None,
    component: str,
) -> dict[str, float] | None:
    if initial is None or accepted is None:
        return None
    initial_component = initial.get("components", {}).get(component)
    accepted_component = accepted.get("components", {}).get(component)
    if initial_component is None or accepted_component is None:
        return None
    return {
        "initial_norm": initial_component["norm"],
        "accepted_norm": accepted_component["norm"],
        "norm_delta": accepted_component["norm"] - initial_component["norm"],
        "initial_mean": initial_component["mean"],
        "accepted_mean": accepted_component["mean"],
        "mean_delta": accepted_component["mean"] - initial_component["mean"],
        "initial_min": initial_component["min"],
        "accepted_min": accepted_component["min"],
        "min_delta": accepted_component["min"] - initial_component["min"],
        "initial_max": initial_component["max"],
        "accepted_max": accepted_component["max"],
        "max_delta": accepted_component["max"] - initial_component["max"],
    }


def active_min_is_clean(cut_context: dict[str, Any] | None) -> bool | None:
    if cut_context is None or "active_min_volume_fraction" not in cut_context:
        return None
    return bool(cut_context["active_min_volume_fraction"] >= 1.0e-2)


def attempt_context_summary(attempt: dict[str, Any] | None) -> dict[str, Any] | None:
    if attempt is None:
        return None

    residual_blocks = attempt.get("residual_block_norms", [])
    solution_snapshots = attempt.get("solution_state_snapshots", [])
    initial_residual = first_record_with_phase(residual_blocks, "jacobian_and_residual")
    accepted_residual = last_record_with_phase(residual_blocks, "line_search")
    if accepted_residual is None and residual_blocks:
        accepted_residual = residual_blocks[-1]
    initial_solution = first_record_with_phase(solution_snapshots, "jacobian_and_residual")
    accepted_solution = last_record_with_phase(solution_snapshots, "line_search")
    if accepted_solution is None and solution_snapshots:
        accepted_solution = solution_snapshots[-1]

    before_context = attempt.get("before_physics_cut_context")
    trial_contexts = attempt.get("line_search_trial_cut_contexts", [])
    accepted_trial_context = trial_contexts[-1] if trial_contexts else None
    return {
        "step_start": {
            "line_no": attempt["line_no"],
            "step": attempt["step"],
            "time_s": attempt["time_s"],
            "dt_s": attempt["dt_s"],
        },
        "before_physics_cut_context": before_context,
        "before_physics_active_min_volume_fraction_ge_1e_minus_2": active_min_is_clean(before_context),
        "accepted_trial_cut_context": accepted_trial_context,
        "accepted_trial_active_min_volume_fraction_ge_1e_minus_2": active_min_is_clean(accepted_trial_context),
        "initial_residual_block_norm": residual_block_summary(initial_residual),
        "accepted_trial_residual_block_norm": residual_block_summary(accepted_residual),
        "initial_solution_state": solution_snapshot_summary(initial_solution),
        "accepted_trial_solution_state": solution_snapshot_summary(accepted_solution),
        "solution_component_changes": {
            component: change
            for component in ("phi", "Velocity[0]", "Velocity[1]", "Velocity[2]", "Pressure")
            if (change := solution_component_change(initial_solution, accepted_solution, component)) is not None
        },
    }


def nonlinear_summary(record: dict[str, Any]) -> dict[str, Any]:
    out = {
        "line_no": record["line_no"],
        "step": record["step"],
        "time_s": record["time_s"],
        "converged": record["converged"],
        "iters": record["iters"],
        "residual_norm": record["residual_norm"],
        "residual_field_norm": record["residual_field_norm"],
        "residual_aux_norm": record["residual_aux_norm"],
    }
    if "dt_s" in record:
        out["dt_s"] = record["dt_s"]
    if "linear_converged" in record:
        out.update(
            {
                "linear_converged": record["linear_converged"],
                "linear_iters": record["linear_iters"],
                "linear_relative_residual": record["linear_relative_residual"],
            }
        )
    return out


def rejected_summary(record: dict[str, Any]) -> dict[str, Any]:
    out = {
        "line_no": record["line_no"],
        "step": record["step"],
        "time_s": record["time_s"],
        "dt_s": record["dt_s"],
        "reason": record["reason"],
        "converged": record["converged"],
        "iters": record["iters"],
        "residual_norm": record["residual_norm"],
        "residual_field_norm": record["residual_field_norm"],
        "residual_aux_norm": record["residual_aux_norm"],
    }
    if record.get("nonlinear") is not None:
        out["nonlinear"] = nonlinear_summary(record["nonlinear"])
    return out


def accepted_summary(record: dict[str, Any]) -> dict[str, Any]:
    out = {
        "line_no": record["line_no"],
        "step": record["step"],
        "time_s": record["time_s"],
        "dt_s": record["dt_s"],
    }
    if record.get("nonlinear") is not None:
        out["nonlinear"] = nonlinear_summary(record["nonlinear"])
    return out


def find_last_matching_nonlinear(
    nonlinear_records: list[dict[str, Any]],
    step: int,
    time_s: float,
    dt_s: float,
    line_no: int,
) -> dict[str, Any] | None:
    for record in reversed(nonlinear_records):
        if record["line_no"] > line_no:
            continue
        if record["step"] != step:
            continue
        if not close_time(record["time_s"], time_s):
            continue
        if "dt_s" in record and not close_time(record["dt_s"], dt_s):
            continue
        return record
    return None


def parse_solver_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False}

    step_starts: list[dict[str, Any]] = []
    nonlinear_records: list[dict[str, Any]] = []
    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    current_attempt: dict[str, Any] | None = None
    last_residual_block: dict[str, Any] | None = None

    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if match := STEP_START_RE.search(line):
            current_attempt = {
                "line_no": line_no,
                "step": to_int(match, "step"),
                "time_s": to_float(match, "time"),
                "dt_s": to_float(match, "dt"),
                "before_physics_cut_context": None,
                "line_search_trial_cut_contexts": [],
                "residual_block_norms": [],
                "solution_state_snapshots": [],
            }
            last_residual_block = None
            step_starts.append(current_attempt)
            continue

        if "diagnostic=cut_context_rebuild" in line and current_attempt is not None:
            cut_context = parse_cut_context(line_no, line)
            provenance = cut_context.get("provenance")
            if provenance == "before_physics_solve":
                current_attempt["before_physics_cut_context"] = cut_context
            elif provenance == "line_search_trial":
                current_attempt["line_search_trial_cut_contexts"].append(cut_context)
            continue

        if match := RESIDUAL_BLOCK_RE.search(line):
            record = {
                "line_no": line_no,
                "phase": match.group("phase"),
                "field_norm": to_float(match, "field"),
                "aux_norm": to_float(match, "aux"),
                "combined_norm": to_float(match, "combined"),
            }
            last_residual_block = record
            if current_attempt is not None:
                current_attempt["residual_block_norms"].append(record)
            continue

        if match := VECTOR_COMPONENT_RE.search(line):
            if match.group("label") == "solution_state" and current_attempt is not None:
                current_attempt["solution_state_snapshots"].append(
                    {
                        "line_no": line_no,
                        "phase": None if last_residual_block is None else last_residual_block.get("phase"),
                        "residual_block": last_residual_block,
                        "components": parse_vector_components(match.group("components")),
                    }
                )
            continue

        if match := NONLINEAR_DONE_RE.search(line):
            record: dict[str, Any] = {
                "line_no": line_no,
                "step": to_int(match, "step"),
                "time_s": to_float(match, "time"),
                "converged": bool(to_int(match, "converged")),
                "iters": to_int(match, "iters"),
                "residual_norm": to_float(match, "residual"),
                "residual_field_norm": to_float(match, "residual_field"),
                "residual_aux_norm": to_float(match, "residual_aux"),
                "linear_converged": bool(to_int(match, "linear_converged")),
                "linear_iters": to_int(match, "linear_iters"),
                "linear_relative_residual": to_float(match, "linear_relative_residual"),
            }
            if (
                current_attempt is not None
                and current_attempt["step"] == record["step"]
                and close_time(current_attempt["time_s"], record["time_s"])
            ):
                record["dt_s"] = current_attempt["dt_s"]
                record["step_start_line_no"] = current_attempt["line_no"]
                record["attempt_context"] = current_attempt
            nonlinear_records.append(record)
            continue

        if match := STEP_REJECTED_RE.search(line):
            record = {
                "line_no": line_no,
                "step": to_int(match, "step"),
                "time_s": to_float(match, "time"),
                "dt_s": to_float(match, "dt"),
                "reason": match.group("reason"),
                "converged": bool(to_int(match, "converged")),
                "iters": to_int(match, "iters"),
                "residual_norm": to_float(match, "residual"),
                "residual_field_norm": to_float(match, "residual_field"),
                "residual_aux_norm": to_float(match, "residual_aux"),
            }
            record["nonlinear"] = find_last_matching_nonlinear(
                nonlinear_records,
                record["step"],
                record["time_s"],
                record["dt_s"],
                line_no,
            )
            rejected_records.append(record)
            continue

        if match := STEP_ACCEPTED_RE.search(line):
            step = to_int(match, "step")
            time_s = to_float(match, "time")
            dt_s = to_float(match, "dt")
            start_time_s = time_s - dt_s
            nonlinear = find_last_matching_nonlinear(
                nonlinear_records,
                step - 1,
                start_time_s,
                dt_s,
                line_no,
            )
            record = {
                "line_no": line_no,
                "step": step,
                "time_s": time_s,
                "dt_s": dt_s,
                "nonlinear": nonlinear,
            }
            accepted_records.append(record)

    return {
        "available": True,
        "artifact": path.name,
        "step_start_count": len(step_starts),
        "nonlinear_done_count": len(nonlinear_records),
        "step_accepted_count": len(accepted_records),
        "step_rejected_count": len(rejected_records),
        "accepted_records": accepted_records,
        "rejected_records": rejected_records,
    }


def attach_solver_context_to_jump(jump: dict[str, Any], solver_log: dict[str, Any]) -> None:
    if not solver_log.get("available"):
        return

    accepted_records = solver_log.get("accepted_records", [])
    accepted = next(
        (
            record
            for record in accepted_records
            if record["step"] == jump["to_step"]
            and close_time(record["time_s"], jump["to_time_s"])
            and close_time(record["dt_s"], jump["dt_s"])
        ),
        None,
    )
    if accepted is None:
        return

    previous_accepted_line = max(
        (record["line_no"] for record in accepted_records if record["line_no"] < accepted["line_no"]),
        default=0,
    )
    next_accepted_line = min(
        (record["line_no"] for record in accepted_records if record["line_no"] > accepted["line_no"]),
        default=sys.maxsize,
    )
    rejected_records = solver_log.get("rejected_records", [])
    preceding_rejections = [
        record
        for record in rejected_records
        if previous_accepted_line < record["line_no"] < accepted["line_no"]
        and close_time(record["time_s"], jump["from_time_s"])
    ]
    following_rejections = [
        record
        for record in rejected_records
        if accepted["line_no"] < record["line_no"] < next_accepted_line
        and close_time(record["time_s"], jump["to_time_s"])
    ]

    accepted_nonlinear = accepted.get("nonlinear")
    residual_norm = None if accepted_nonlinear is None else accepted_nonlinear.get("residual_norm")
    accepted_attempt_context = (
        None if accepted_nonlinear is None else attempt_context_summary(accepted_nonlinear.get("attempt_context"))
    )
    jump["solver_log_context"] = {
        "available": True,
        "artifact": solver_log.get("artifact"),
        "accepted_event": accepted_summary(accepted),
        "accepted_nonlinear_abs_residual_below_0p02": (
            None if residual_norm is None else bool(residual_norm < RESIDUAL_ABS_TOLERANCE)
        ),
        "accepted_attempt_context": accepted_attempt_context,
        "preceding_rejected_attempt_count": len(preceding_rejections),
        "preceding_rejected_attempts": [rejected_summary(record) for record in preceding_rejections],
        "following_rejected_attempt_count": len(following_rejections),
        "following_rejected_attempts": [rejected_summary(record) for record in following_rejections],
    }


def add_solver_context(jumps: list[dict[str, Any]], top_jumps: list[dict[str, Any]], solver_log: dict[str, Any]) -> None:
    records: list[dict[str, Any]] = []
    if top_jumps:
        records.extend(top_jumps)
    if jumps:
        records.append(jumps[-1])

    seen: set[int] = set()
    for record in records:
        record_id = id(record)
        if record_id in seen:
            continue
        seen.add(record_id)
        attach_solver_context_to_jump(record, solver_log)


def solver_log_summary(solver_log: dict[str, Any]) -> dict[str, Any]:
    if not solver_log.get("available"):
        return {"available": False}
    return {
        "available": True,
        "artifact": solver_log.get("artifact"),
        "step_start_count": solver_log.get("step_start_count"),
        "nonlinear_done_count": solver_log.get("nonlinear_done_count"),
        "step_accepted_count": solver_log.get("step_accepted_count"),
        "step_rejected_count": solver_log.get("step_rejected_count"),
        "accepted_abs_residual_threshold": RESIDUAL_ABS_TOLERANCE,
    }


def pressure_samples(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    samples = comparison.get("pressure_history", {}).get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise RuntimeError("pressure comparison JSON must contain at least two pressure_history.samples")
    out = []
    for sample in samples:
        if "time_s" not in sample or "pressure_pa" not in sample:
            raise RuntimeError("pressure history sample is missing time_s or pressure_pa")
        out.append(
            {
                "step": int(sample.get("step", len(out) + 1)),
                "time_s": float(sample["time_s"]),
                "pressure_pa": float(sample["pressure_pa"]),
                "selection": sample.get("selection"),
                "sample_valid": sample.get("sample_valid"),
                "result": sample.get("result"),
            }
        )
    return sorted(out, key=lambda item: (item["time_s"], item["step"]))


def reference_at_times(args: argparse.Namespace, times: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    if args.reference_file is None and not args.fetch_reference:
        return None, {"available": False, "reason": "no reference file or fetch requested"}

    reference = load_reference_series(
        args.reference_file,
        fetch=args.fetch_reference,
        member=args.reference_member,
    )
    if reference is None:
        return None, {"available": False, "reason": "reference loader returned no data"}

    pressure = np.interp(times, reference.time_s, reference.pressure_pa)
    return pressure, {
        "available": True,
        "member": args.reference_member,
        "time_start_s": float(reference.time_s[0]),
        "time_end_s": float(reference.time_s[-1]),
        "pressure_min_pa": float(np.min(reference.pressure_pa)),
        "pressure_max_pa": float(np.max(reference.pressure_pa)),
    }


def topology_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"available": False}
    audit = load_json(path)
    summary = audit.get("accepted_output_tail_summary", {})
    final = audit.get("final_accepted_output", {})
    return {
        "available": True,
        "artifact": path.name,
        "tail_max_tiny_cut_counts": summary.get("max_tiny_cut_counts_in_tail"),
        "tail_min_cut_wet_fraction": (
            summary.get("min_cut_wet_fraction_in_tail", {})
            .get("cut_wet_fraction", {})
            .get("min")
        ),
        "final_accepted_time_s": final.get("time_s"),
        "final_accepted_pressure_all_points": final.get("point_field_extrema", {}).get("pressure_all_points"),
        "final_activefluid_phi_sign_mismatch_count": final.get("point_field_extrema", {}).get(
            "activefluid_phi_sign_mismatch_count"
        ),
        "final_points_with_any_tiny_positive_incident_wet_fraction": final.get("point_field_extrema", {}).get(
            "points_with_any_tiny_positive_incident_wet_fraction"
        ),
    }


def jump_records(
    samples: list[dict[str, Any]],
    reference_pressure: np.ndarray | None,
    *,
    top_jumps: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jumps: list[dict[str, Any]] = []
    for idx in range(1, len(samples)):
        prev = samples[idx - 1]
        curr = samples[idx]
        dt = curr["time_s"] - prev["time_s"]
        if dt <= 0.0:
            continue
        delta = curr["pressure_pa"] - prev["pressure_pa"]
        record: dict[str, Any] = {
            "from_step": prev["step"],
            "to_step": curr["step"],
            "from_time_s": prev["time_s"],
            "to_time_s": curr["time_s"],
            "dt_s": dt,
            "from_pressure_pa": prev["pressure_pa"],
            "to_pressure_pa": curr["pressure_pa"],
            "delta_pressure_pa": delta,
            "abs_delta_pressure_pa": abs(delta),
            "pressure_rate_pa_per_s": delta / dt,
            "abs_pressure_rate_pa_per_s": abs(delta / dt),
        }
        if reference_pressure is not None:
            ref_delta = float(reference_pressure[idx] - reference_pressure[idx - 1])
            record.update(
                {
                    "reference_from_pressure_pa": float(reference_pressure[idx - 1]),
                    "reference_to_pressure_pa": float(reference_pressure[idx]),
                    "reference_delta_pressure_pa": ref_delta,
                    "reference_pressure_rate_pa_per_s": ref_delta / dt,
                    "to_error_pa": curr["pressure_pa"] - float(reference_pressure[idx]),
                }
            )
        jumps.append(record)
    return jumps, sorted(jumps, key=lambda item: item["abs_delta_pressure_pa"], reverse=True)[:top_jumps]


def first_sampled_peak_exceedance(
    samples: list[dict[str, Any]],
    reference_pressure: np.ndarray | None,
    sampled_reference_peak: float | None,
) -> dict[str, Any] | None:
    if sampled_reference_peak is None:
        return None
    for idx, sample in enumerate(samples):
        if sample["pressure_pa"] > sampled_reference_peak:
            return {
                "step": sample["step"],
                "time_s": sample["time_s"],
                "pressure_pa": sample["pressure_pa"],
                "reference_pressure_pa": None if reference_pressure is None else float(reference_pressure[idx]),
                "sampled_reference_peak_pa": sampled_reference_peak,
            }
    return None


def first_abs_error_exceedance(
    samples: list[dict[str, Any]],
    reference_pressure: np.ndarray | None,
    threshold_pa: float,
) -> dict[str, Any] | None:
    if reference_pressure is None:
        return None
    for idx, sample in enumerate(samples):
        error = sample["pressure_pa"] - float(reference_pressure[idx])
        if abs(error) > threshold_pa:
            return {
                "step": sample["step"],
                "time_s": sample["time_s"],
                "pressure_pa": sample["pressure_pa"],
                "reference_pressure_pa": float(reference_pressure[idx]),
                "error_pa": error,
                "abs_error_pa": abs(error),
                "threshold_pa": threshold_pa,
            }
    return None


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    comparison_path = args.pressure_comparison
    comparison = load_json(comparison_path)
    samples = pressure_samples(comparison)
    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    pressures = np.asarray([sample["pressure_pa"] for sample in samples], dtype=float)

    reference_pressure, reference_summary = reference_at_times(args, times)
    jumps, top_jumps = jump_records(samples, reference_pressure, top_jumps=args.top_jumps)
    solver_log = parse_solver_log(args.solver_log)
    add_solver_context(jumps, top_jumps, solver_log)

    comparison_metrics = comparison.get("comparison", {})
    sampled_reference_peak = comparison_metrics.get("reference_peak_pressure_pa")
    if sampled_reference_peak is not None:
        sampled_reference_peak = float(sampled_reference_peak)

    final_reference = None if reference_pressure is None else float(reference_pressure[-1])
    final_error = None if final_reference is None else float(pressures[-1] - final_reference)
    max_jump = top_jumps[0] if top_jumps else None
    final_jump = jumps[-1] if jumps else None

    endpoint_pressure_to_peak = None
    if sampled_reference_peak not in (None, 0.0):
        endpoint_pressure_to_peak = float(pressures[-1] / sampled_reference_peak)

    finding_parts = [
        f"The accepted Sensor1 history reaches {pressures[-1]:.6g} Pa at {times[-1]:.16g} s",
    ]
    if final_reference is not None:
        finding_parts.append(f"while the interpolated SPHERIC reference there is {final_reference:.6g} Pa")
    if final_jump is not None:
        finding_parts.append(
            f"and the final accepted jump is {final_jump['delta_pressure_pa']:.6g} Pa over "
            f"{final_jump['dt_s']:.6g} s"
        )
    if max_jump is not None and max_jump is not final_jump:
        finding_parts.append(
            f"the largest accepted jump is {max_jump['delta_pressure_pa']:.6g} Pa from "
            f"step {max_jump['from_step']} to {max_jump['to_step']}"
        )
    if final_jump is not None:
        final_solver_context = final_jump.get("solver_log_context", {})
        final_accepted_nonlinear = final_solver_context.get("accepted_event", {}).get("nonlinear")
        if final_accepted_nonlinear is not None:
            finding_parts.append(
                f"that final jump was accepted after {final_accepted_nonlinear['iters']} Newton iteration(s) "
                f"with ||r||={final_accepted_nonlinear['residual_norm']:.6g}"
            )
    finding = (
        "; ".join(finding_parts)
        + ". This is accepted-output pressure-path evidence, separate from the later failed retry."
    )

    return {
        "pressure_comparison_artifact": comparison_path.name,
        "sample_count": len(samples),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "pressure_start_pa": float(pressures[0]),
        "pressure_end_pa": float(pressures[-1]),
        "reference": reference_summary,
        "comparison_metrics": {
            "reference_coverage_fraction": comparison.get("validation_window", {}).get("reference_coverage_fraction"),
            "sampled_reference_peak_pressure_pa": sampled_reference_peak,
            "sampled_reference_peak_time_s": comparison_metrics.get("reference_peak_time_s"),
            "simulated_peak_pressure_pa": comparison_metrics.get("simulated_peak_pressure_pa"),
            "simulated_peak_time_s": comparison_metrics.get("simulated_peak_time_s"),
            "rmse_pa": comparison_metrics.get("rmse_pa"),
            "validation_status": comparison.get("validation_status"),
        },
        "final_sample": {
            **samples[-1],
            "interpolated_reference_pressure_pa": final_reference,
            "error_pa": final_error,
            "pressure_to_sampled_reference_peak_ratio": endpoint_pressure_to_peak,
        },
        "first_sampled_reference_peak_exceedance": first_sampled_peak_exceedance(
            samples,
            reference_pressure,
            sampled_reference_peak,
        ),
        "first_abs_error_gt_100pa": first_abs_error_exceedance(samples, reference_pressure, 100.0),
        "max_abs_accepted_jump": max_jump,
        "final_accepted_jump": final_jump,
        "top_accepted_jumps": top_jumps,
        "active_topology_context": topology_summary(args.active_topology_audit),
        "solver_log_context": solver_log_summary(solver_log),
        "finding": finding,
        "status": "diagnostic_only_test10_still_not_validation_ready",
    }


def main() -> int:
    args = parse_args()
    audit = build_audit(args)
    text = json.dumps(audit, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
