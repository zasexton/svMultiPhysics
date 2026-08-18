#!/usr/bin/env python3
"""Audit cut-context lifecycle around an accepted pressure update."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=('[^']*'|\S+)")
STEP_START_RE = re.compile(
    rf"TimeLoop: step_start step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE})"
)
STEP_ACCEPTED_RE = re.compile(
    rf"TimeLoop: step_accepted step=(?P<step>\d+) time=(?P<time>{NUMBER_RE}) "
    rf"dt=(?P<dt>{NUMBER_RE})"
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

FLOAT_CONTEXT_FIELDS = (
    "active_side_volume",
    "active_side_physical_volume",
    "active_side_raw_volume",
    "active_pruned_volume",
    "generated_pruned_volume",
    "active_min_volume_fraction",
    "active_max_volume_fraction",
    "cut_adjacent_min_scale",
    "cut_adjacent_max_scale",
    "cut_adjacent_mean_scale",
)
COUNT_CONTEXT_FIELDS = (
    "active_side_physical_rule_count",
    "active_side_available_cut_volume_rule_count",
    "active_side_retained_cut_volume_rule_count",
    "active_volume_regions",
    "active_volume_rule_count",
    "active_cut_cells",
    "active_full_wet_cells",
    "active_full_dry_cells",
    "active_quadrature_points",
    "active_pruned_volume_regions",
    "generated_pruned_volume_rules",
    "cut_adjacent_facets",
    "cut_adjacent_capped_scale",
    "domain_interface_quadrature_point_count",
    "domain_volume_quadrature_point_count",
)
IDENTITY_CONTEXT_FIELDS = (
    "cut_context_revision",
    "cut_context_topology_key",
    "source_value_revision",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize cut-context rebuilds and accepted pressure-update guard "
            "events around a short Test02/Test10 replay."
        )
    )
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--pressure-update-audit", type=Path)
    parser.add_argument("--case-label", default="case")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--pressure-match-abs-tol-pa",
        type=float,
        default=1.0e-6,
        help="Tolerance for comparing runtime guard pressure update to offline VTU audit.",
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


def parse_solver_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    current_start: dict[str, Any] | None = None
    starts: list[dict[str, Any]] = []
    accepted_steps: list[dict[str, Any]] = []
    nonlinear_done: list[dict[str, Any]] = []
    cut_context_rebuilds: list[dict[str, Any]] = []
    cut_context_skips: list[dict[str, Any]] = []
    pressure_update_guards: list[dict[str, Any]] = []
    field_residual_diagnostics: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            step_start = STEP_START_RE.search(line)
            if step_start:
                current_start = {
                    "line_number": line_number,
                    "attempt_step": int(step_start.group("step")),
                    "attempt_time_s": float(step_start.group("time")),
                    "dt_s": float(step_start.group("dt")),
                }
                starts.append(current_start)
                continue

            nonlinear = NONLINEAR_DONE_RE.search(line)
            if nonlinear:
                nonlinear_done.append(
                    {
                        "line_number": line_number,
                        "attempt_step": int(nonlinear.group("step")),
                        "time_s": float(nonlinear.group("time")),
                        "converged": bool(int(nonlinear.group("converged"))),
                        "iters": int(nonlinear.group("iters")),
                        "residual": float(nonlinear.group("residual")),
                        "residual_field": float(nonlinear.group("residual_field")),
                        "residual_aux": float(nonlinear.group("residual_aux")),
                        "linear_converged": bool(
                            int(nonlinear.group("linear_converged"))
                        ),
                        "linear_iters": int(nonlinear.group("linear_iters")),
                        "linear_rel": float(nonlinear.group("linear_rel")),
                    }
                )
                continue

            if "diagnostic=cut_context_rebuild" in line:
                values = parse_key_values(line)
                cut_context_rebuilds.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_start["attempt_step"]
                            if current_start is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

            if "diagnostic=cut_context_refresh_skip" in line:
                values = parse_key_values(line)
                cut_context_skips.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_start["attempt_step"]
                            if current_start is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

            if "diagnostic=accepted_pressure_update_guard" in line:
                values = parse_key_values(line)
                pressure_update_guards.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_start["attempt_step"]
                            if current_start is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

            if "diagnostic=newton_field_residual" in line:
                values = parse_key_values(line)
                field_residual_diagnostics.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_start["attempt_step"]
                            if current_start is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

            step_accepted = STEP_ACCEPTED_RE.search(line)
            if step_accepted:
                accepted_steps.append(
                    {
                        "line_number": line_number,
                        "to_step": int(step_accepted.group("step")),
                        "to_time_s": float(step_accepted.group("time")),
                        "dt_s": float(step_accepted.group("dt")),
                        "attempt_step": (
                            current_start["attempt_step"]
                            if current_start is not None
                            else None
                        ),
                    }
                )
                continue

    return {
        "step_starts": starts,
        "accepted_steps": accepted_steps,
        "nonlinear_done": nonlinear_done,
        "cut_context_rebuilds": cut_context_rebuilds,
        "cut_context_skips": cut_context_skips,
        "pressure_update_guards": pressure_update_guards,
        "field_residual_diagnostics": field_residual_diagnostics,
    }


def compact_context(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    keep = (
        "line_number",
        "attempt_step",
        "provenance",
        "solution_source",
        "cut_context_revision",
        "cut_context_topology_key",
        "source_value_revision",
        *FLOAT_CONTEXT_FIELDS,
        *COUNT_CONTEXT_FIELDS,
    )
    return {key: record[key] for key in keep if key in record}


def compact_guard(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    keep = (
        "line_number",
        "attempt_step",
        "step",
        "time",
        "dt",
        "global_abs_pressure_delta_pa",
        "local_abs_pressure_delta_pa",
        "local_pressure_delta_pa",
        "local_from_pressure_pa",
        "local_to_pressure_pa",
        "local_worst_vertex",
        "local_worst_dof",
        "support_class",
        "incident_wet_fraction_max",
        "incident_wet_fraction_min_positive",
        "threshold_pa",
        "triggered",
    )
    return {key: record[key] for key in keep if key in record}


def compact_field_residual(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    keep = (
        "line_number",
        "attempt_step",
        "rank",
        "iteration",
        "phase",
        "sync_point",
        "field",
        "field_offset",
        "field_dofs",
        "solve_time",
        "dt",
        "owned_field_dofs",
        "norm",
        "mean",
        "min",
        "max",
        "global_max_abs",
        "local_worst_dof",
        "local_worst_value",
        "local_max_abs",
    )
    return {key: record[key] for key in keep if key in record}


def last_before(
    records: list[dict[str, Any]],
    *,
    line_number: int,
    provenance: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record["line_number"] < line_number
        and (provenance is None or record.get("provenance") == provenance)
    ]
    return candidates[-1] if candidates else None


def last_field_residual_before(
    records: list[dict[str, Any]],
    *,
    line_number: int,
    sync_point: str | None = None,
    phase: str | None = None,
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record["line_number"] < line_number
        and (sync_point is None or record.get("sync_point") == sync_point)
        and (phase is None or record.get("phase") == phase)
    ]
    return candidates[-1] if candidates else None


def first_after(
    records: list[dict[str, Any]],
    *,
    line_number: int,
    provenance: str | None = None,
) -> dict[str, Any] | None:
    for record in records:
        if record["line_number"] > line_number and (
            provenance is None or record.get("provenance") == provenance
        ):
            return record
    return None


def relative_delta(old: float, new: float) -> float | None:
    scale = max(abs(old), abs(new), 1.0e-300)
    return abs(new - old) / scale if math.isfinite(scale) and scale > 0.0 else None


def compare_contexts(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None

    float_deltas: dict[str, dict[str, float | None]] = {}
    for field in FLOAT_CONTEXT_FIELDS:
        if field not in before or field not in after:
            continue
        old = float(before[field])
        new = float(after[field])
        float_deltas[field] = {
            "before": old,
            "after": new,
            "abs_delta": new - old,
            "relative_abs_delta": relative_delta(old, new),
        }

    count_deltas: dict[str, dict[str, int]] = {}
    for field in COUNT_CONTEXT_FIELDS:
        if field not in before or field not in after:
            continue
        old = int(before[field])
        new = int(after[field])
        count_deltas[field] = {
            "before": old,
            "after": new,
            "delta": new - old,
        }

    identity_changes = {
        field: {
            "before": before.get(field),
            "after": after.get(field),
            "changed": before.get(field) != after.get(field),
        }
        for field in IDENTITY_CONTEXT_FIELDS
        if field in before or field in after
    }

    max_relative_float_delta = None
    for delta in float_deltas.values():
        rel = delta["relative_abs_delta"]
        if rel is None:
            continue
        max_relative_float_delta = (
            rel
            if max_relative_float_delta is None
            else max(max_relative_float_delta, rel)
        )

    changed_counts = {
        field: delta
        for field, delta in count_deltas.items()
        if delta["delta"] != 0
    }
    changed_identities = {
        field: change
        for field, change in identity_changes.items()
        if change["changed"]
    }

    return {
        "from_line": before["line_number"],
        "to_line": after["line_number"],
        "from_provenance": before.get("provenance"),
        "to_provenance": after.get("provenance"),
        "float_deltas": float_deltas,
        "count_deltas": count_deltas,
        "identity_changes": identity_changes,
        "changed_counts": changed_counts,
        "changed_identities": changed_identities,
        "max_relative_float_delta": max_relative_float_delta,
    }


def offline_pressure_update(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    transitions = data.get("transitions") or []
    if not transitions:
        return None
    event = transitions[0].get("max_by_category", {}).get("active_or_wet_supported")
    if event is None:
        return None
    stats = transitions[0].get("delta_statistics_by_category", {}).get(
        "active_or_wet_supported"
    )
    return {
        "path": str(path),
        "from_step": event.get("from_step"),
        "to_step": event.get("to_step"),
        "abs_pressure_delta_pa": event.get("abs_pressure_delta_pa"),
        "pressure_delta_pa": event.get("pressure_delta_pa"),
        "point_index": event.get("point_index"),
        "point_m": event.get("point_m"),
        "support_class": event.get("support_class"),
        "incident_wet_fraction_max": event.get("incident_wet_fraction_max"),
        "incident_wet_fraction_min_positive": event.get(
            "incident_wet_fraction_min_positive"
        ),
        "median_removed_active_or_wet_max_pa": (
            stats.get("max_abs_after_median_removal_pa")
            if isinstance(stats, dict)
            else None
        ),
    }


def pressure_match_report(
    guard: dict[str, Any] | None,
    offline: dict[str, Any] | None,
    *,
    abs_tol_pa: float,
) -> dict[str, Any] | None:
    if guard is None or offline is None:
        return None
    runtime = guard.get("global_abs_pressure_delta_pa")
    offline_value = offline.get("abs_pressure_delta_pa")
    if not isinstance(runtime, (int, float)) or not isinstance(
        offline_value, (int, float)
    ):
        return None
    abs_diff = abs(float(runtime) - float(offline_value))
    return {
        "runtime_global_abs_pressure_delta_pa": float(runtime),
        "offline_active_or_wet_abs_pressure_delta_pa": float(offline_value),
        "abs_difference_pa": abs_diff,
        "abs_tolerance_pa": abs_tol_pa,
        "matches": abs_diff <= abs_tol_pa,
    }


def update_residual_ratio(
    guard: dict[str, Any] | None,
    residual: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if guard is None or residual is None:
        return None
    update = guard.get("global_abs_pressure_delta_pa")
    residual_norm = residual.get("norm")
    residual_max = residual.get("global_max_abs")
    if not isinstance(update, (int, float)):
        return None
    out: dict[str, Any] = {
        "global_abs_pressure_delta_pa": float(update),
    }
    if isinstance(residual_norm, (int, float)):
        out["field_residual_norm"] = float(residual_norm)
        out["update_to_field_residual_norm_ratio"] = (
            float(update) / float(residual_norm)
            if float(residual_norm) > 0.0
            else None
        )
    if isinstance(residual_max, (int, float)):
        out["field_residual_global_max_abs"] = float(residual_max)
        out["update_to_field_residual_max_abs_ratio"] = (
            float(update) / float(residual_max)
            if float(residual_max) > 0.0
            else None
        )
    return out


def update_nonlinear_residual_ratio(
    guard: dict[str, Any] | None,
    nonlinear: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if guard is None or nonlinear is None:
        return None
    update = guard.get("global_abs_pressure_delta_pa")
    residual = nonlinear.get("residual")
    residual_field = nonlinear.get("residual_field")
    if not isinstance(update, (int, float)):
        return None
    out: dict[str, Any] = {
        "global_abs_pressure_delta_pa": float(update),
        "nonlinear_converged": nonlinear.get("converged"),
        "nonlinear_iterations": nonlinear.get("iters"),
        "linear_converged": nonlinear.get("linear_converged"),
        "linear_iterations": nonlinear.get("linear_iters"),
        "linear_relative_residual": nonlinear.get("linear_rel"),
    }
    if isinstance(residual, (int, float)):
        out["nonlinear_residual_norm"] = float(residual)
        out["update_to_nonlinear_residual_norm_ratio"] = (
            float(update) / float(residual) if float(residual) > 0.0 else None
        )
    if isinstance(residual_field, (int, float)):
        out["nonlinear_field_residual_norm"] = float(residual_field)
        out["update_to_nonlinear_field_residual_norm_ratio"] = (
            float(update) / float(residual_field)
            if float(residual_field) > 0.0
            else None
        )
    return out


def lifecycle_report(
    parsed: dict[str, list[dict[str, Any]]],
    *,
    case_label: str,
    offline: dict[str, Any] | None,
    pressure_match_abs_tol_pa: float,
) -> dict[str, Any]:
    guards = parsed["pressure_update_guards"]
    accepted_steps = parsed["accepted_steps"]
    contexts = parsed["cut_context_rebuilds"]
    field_residuals = parsed["field_residual_diagnostics"]
    guard = guards[-1] if guards else None
    accepted = accepted_steps[-1] if accepted_steps else None
    reference_line = (
        guard["line_number"]
        if guard is not None
        else accepted["line_number"]
        if accepted is not None
        else math.inf
    )

    initial_context = last_before(
        contexts, line_number=reference_line, provenance="initial"
    )
    line_search_context = last_before(
        contexts, line_number=reference_line, provenance="line_search_trial"
    )
    solve_context = line_search_context or initial_context
    accepted_step_context = (
        first_after(
            contexts,
            line_number=(guard or accepted or {"line_number": 0})["line_number"],
            provenance="accepted_step",
        )
        if guard is not None or accepted is not None
        else None
    )

    guard_before_accepted_step_refresh = (
        guard is not None
        and accepted_step_context is not None
        and guard["line_number"] < accepted_step_context["line_number"]
    )
    pressure_match = pressure_match_report(
        compact_guard(guard), offline, abs_tol_pa=pressure_match_abs_tol_pa
    )
    latest_nonlinear_done = (
        parsed["nonlinear_done"][-1] if parsed["nonlinear_done"] else None
    )
    initial_field_residual = last_field_residual_before(
        field_residuals,
        line_number=reference_line,
        sync_point="jacobian_and_residual",
    )
    solve_field_residual = last_field_residual_before(
        field_residuals,
        line_number=reference_line,
        sync_point="line_search_trial",
    )
    if solve_field_residual is None:
        solve_field_residual = last_field_residual_before(
            field_residuals, line_number=reference_line
        )
    compact_solve_field_residual = compact_field_residual(solve_field_residual)
    maintenance_comparison = compare_contexts(solve_context, accepted_step_context)
    initial_to_solve_comparison = compare_contexts(initial_context, solve_context)
    post_acceptance_refresh_immediate_driver_ruled_out = (
        guard_before_accepted_step_refresh
        and guard is not None
        and guard.get("triggered") == 1
    )

    finding = (
        "Accepted pressure update guard did not appear in the log; lifecycle "
        "ordering cannot classify the pressure jump."
    )
    status = "diagnostic_cut_context_pressure_transition_incomplete"
    if guard is not None:
        status = "diagnostic_cut_context_pressure_transition_guard_found"
        finding = (
            "The accepted pressure update guard fired before the accepted-step "
            "maintenance cut-context rebuild."
            if guard_before_accepted_step_refresh
            else "The accepted pressure update guard was found, but no later "
            "accepted-step maintenance rebuild was found."
        )
        if post_acceptance_refresh_immediate_driver_ruled_out:
            finding += (
                " Post-acceptance maintenance refresh is therefore ruled out "
                "as the immediate source of the accepted pressure increment; "
                "the remaining target is the solve-time active-volume pressure "
                "path on the line-search cut context."
            )

    return {
        "case_label": case_label,
        "status": status,
        "finding": finding,
        "counts": {
            "step_starts": len(parsed["step_starts"]),
            "accepted_steps": len(parsed["accepted_steps"]),
            "nonlinear_done": len(parsed["nonlinear_done"]),
            "cut_context_rebuilds": len(parsed["cut_context_rebuilds"]),
            "cut_context_refresh_skips": len(parsed["cut_context_skips"]),
            "pressure_update_guards": len(parsed["pressure_update_guards"]),
            "field_residual_diagnostics": len(parsed["field_residual_diagnostics"]),
        },
        "accepted_step": accepted,
        "nonlinear_done": latest_nonlinear_done,
        "pressure_update_guard": compact_guard(guard),
        "initial_field_residual": compact_field_residual(initial_field_residual),
        "solve_field_residual": compact_solve_field_residual,
        "pressure_update_to_solve_field_residual": update_residual_ratio(
            compact_guard(guard), compact_solve_field_residual
        ),
        "pressure_update_to_nonlinear_residual": update_nonlinear_residual_ratio(
            compact_guard(guard), latest_nonlinear_done
        ),
        "offline_pressure_update": offline,
        "runtime_offline_pressure_match": pressure_match,
        "initial_context": compact_context(initial_context),
        "solve_context": compact_context(solve_context),
        "accepted_step_maintenance_context": compact_context(accepted_step_context),
        "initial_to_solve_context_delta": initial_to_solve_comparison,
        "solve_to_accepted_step_maintenance_context_delta": maintenance_comparison,
        "guard_before_accepted_step_refresh": guard_before_accepted_step_refresh,
        "post_acceptance_refresh_immediate_driver_ruled_out": (
            post_acceptance_refresh_immediate_driver_ruled_out
        ),
        "cut_context_refresh_skips": parsed["cut_context_skips"],
    }


def audit_transition(
    *,
    solver_log: Path,
    case_label: str,
    pressure_update_audit: Path | None,
    pressure_match_abs_tol_pa: float,
) -> dict[str, Any]:
    parsed = parse_solver_log(solver_log)
    offline = offline_pressure_update(pressure_update_audit)
    return lifecycle_report(
        parsed,
        case_label=case_label,
        offline=offline,
        pressure_match_abs_tol_pa=pressure_match_abs_tol_pa,
    )


def main() -> int:
    args = parse_args()
    report = audit_transition(
        solver_log=args.solver_log,
        case_label=args.case_label,
        pressure_update_audit=args.pressure_update_audit,
        pressure_match_abs_tol_pa=args.pressure_match_abs_tol_pa,
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
