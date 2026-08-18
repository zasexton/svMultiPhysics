#!/usr/bin/env python3
"""Audit cut-adjacent pressure support before accepted pressure-update guards."""

from __future__ import annotations

import argparse
from collections import Counter
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

CUT_CONTEXT_KEYS = (
    "provenance",
    "solution_source",
    "cut_context_revision",
    "cut_context_topology_key",
    "source_value_revision",
    "active_side_retained_cut_volume_rule_count",
    "active_side_available_cut_volume_rule_count",
    "active_side_physical_rule_count",
    "active_volume_regions",
    "active_volume_rule_count",
    "active_wet_cells",
    "active_cut_cells",
    "active_full_wet_cells",
    "active_full_dry_cells",
    "active_pruned_volume_regions",
    "active_pruned_volume",
    "generated_pruned_volume_rules",
    "generated_pruned_volume",
    "active_min_volume_fraction",
    "active_max_volume_fraction",
    "cut_adjacent_facets",
    "cut_adjacent_capped_scale",
    "cut_adjacent_min_scale",
    "cut_adjacent_max_scale",
    "cut_adjacent_mean_scale",
)
PRESSURE_CONSTRAINT_KEYS = (
    "field",
    "support_mode",
    "interface_marker",
    "total_vertices",
    "active_sign_vertices",
    "active_support_cells",
    "active_support_cells_from_volume_support",
    "active_support_cells_from_cut_adjacent_facets",
    "active_support_vertices",
    "active_support_dofs",
    "active_sign_vertices_without_support",
    "inactive_sign_vertices_with_support",
    "inactive_vertices",
    "inactive_dofs",
    "constrained_owned_dofs",
)
CONSTRAINT_REFRESH_KEYS = (
    "provenance",
    "solution_source",
    "synchronized_level_set_fields",
    "support_source",
    "constraints",
)
GUARD_KEYS = (
    "phase",
    "step",
    "time",
    "dt",
    "field",
    "retained_active_volume_rules",
    "active_supported_vertices",
    "compared_vertex_pressure_dofs",
    "local_worst_vertex",
    "local_worst_dof",
    "local_abs_pressure_delta_pa",
    "global_abs_pressure_delta_pa",
    "local_pressure_delta_pa",
    "support_class",
    "incident_wet_fraction_max",
    "incident_wet_fraction_min_positive",
    "threshold_pa",
    "triggered",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join accepted pressure-update guards to the nearest pre-guard "
            "cut-context rebuild and active pressure support diagnostics."
        )
    )
    parser.add_argument(
        "--case-log",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Case label and solver log path. May be supplied more than once.",
    )
    parser.add_argument("--json-output", type=Path)
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


def _summary(record: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    out = {"line_number": record.get("line_number")}
    if record.get("attempt_step") is not None:
        out["attempt_step"] = record.get("attempt_step")
    for key in keys:
        if key in record:
            out[key] = record[key]
    return out


def _number(record: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(record, dict):
        return default
    value = record.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _support_mode(record: dict[str, Any] | None) -> str:
    if not isinstance(record, dict):
        return ""
    mode = record.get("support_mode", "")
    return mode if isinstance(mode, str) else ""


def _nearest_before(
    records: list[dict[str, Any]], guard: dict[str, Any]
) -> dict[str, Any] | None:
    guard_line = guard["line_number"]
    attempt_step = guard.get("attempt_step")
    matches = [
        record
        for record in records
        if record["line_number"] < guard_line
        and (attempt_step is None or record.get("attempt_step") == attempt_step)
    ]
    if not matches:
        matches = [
            record for record in records if record["line_number"] < guard_line
        ]
    return matches[-1] if matches else None


def _first_after(
    records: list[dict[str, Any]], guard: dict[str, Any]
) -> dict[str, Any] | None:
    guard_line = guard["line_number"]
    attempt_step = guard.get("attempt_step")
    matches = [
        record
        for record in records
        if record["line_number"] > guard_line
        and (attempt_step is None or record.get("attempt_step") == attempt_step)
    ]
    if not matches:
        matches = [
            record for record in records if record["line_number"] > guard_line
        ]
    return matches[0] if matches else None


def parse_case_log(path: Path) -> dict[str, list[dict[str, Any]]]:
    current_attempt: dict[str, Any] | None = None
    cut_contexts: list[dict[str, Any]] = []
    pressure_constraints: list[dict[str, Any]] = []
    constraint_refreshes: list[dict[str, Any]] = []
    pressure_update_guards: list[dict[str, Any]] = []

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            start = STEP_START_RE.search(line)
            if start:
                current_attempt = {
                    "attempt_step": int(start.group("step")),
                    "attempt_time_s": float(start.group("time")),
                    "dt_s": float(start.group("dt")),
                    "line_number": line_number,
                }
                continue

            if "diagnostic=cut_context_rebuild" in line:
                values = parse_key_values(line)
                cut_contexts.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_attempt.get("attempt_step")
                            if current_attempt is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

            if (
                "diagnostic=level_set_active_side_vertex_constraint" in line
                and "field='Pressure'" in line
            ):
                values = parse_key_values(line)
                pressure_constraints.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_attempt.get("attempt_step")
                            if current_attempt is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

            if "diagnostic=active_pressure_constraint_refresh" in line:
                values = parse_key_values(line)
                constraint_refreshes.append(
                    {
                        "line_number": line_number,
                        "attempt_step": (
                            current_attempt.get("attempt_step")
                            if current_attempt is not None
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
                            current_attempt.get("attempt_step")
                            if current_attempt is not None
                            else None
                        ),
                        **values,
                    }
                )
                continue

    return {
        "cut_contexts": cut_contexts,
        "pressure_constraints": pressure_constraints,
        "constraint_refreshes": constraint_refreshes,
        "pressure_update_guards": pressure_update_guards,
    }


def classify_guard(
    guard: dict[str, Any],
    *,
    pre_cut_context: dict[str, Any] | None,
    pre_pressure_constraint: dict[str, Any] | None,
    pre_constraint_refresh: dict[str, Any] | None,
    post_cut_context: dict[str, Any] | None,
    post_pressure_constraint: dict[str, Any] | None,
) -> dict[str, Any]:
    mode = _support_mode(pre_pressure_constraint)
    volume_support_cells = _number(
        pre_pressure_constraint, "active_support_cells_from_volume_support"
    )
    cut_adjacent_support_cells = _number(
        pre_pressure_constraint, "active_support_cells_from_cut_adjacent_facets"
    )
    retained_rules = _number(
        pre_cut_context, "active_side_retained_cut_volume_rule_count"
    )
    generated_pruned_rules = _number(pre_cut_context, "generated_pruned_volume_rules")
    generated_pruned_volume = _number(pre_cut_context, "generated_pruned_volume")
    active_pruned_regions = _number(pre_cut_context, "active_pruned_volume_regions")
    active_pruned_volume = _number(pre_cut_context, "active_pruned_volume")

    skipped_no_retained = "cut_adjacent_facets_skipped_no_retained_volume" in mode
    trace_only_support = (
        cut_adjacent_support_cells > 0.0 and volume_support_cells == 0.0
    ) or (
        "cut_adjacent_facets" in mode
        and "retained_cut_volume" not in mode
        and not skipped_no_retained
    )
    retained_volume_support = (
        retained_rules > 0.0
        and volume_support_cells > 0.0
        and "retained_cut_volume" in mode
    )
    pruned_volume_present = (
        generated_pruned_rules > 0.0
        or generated_pruned_volume > 0.0
        or active_pruned_regions > 0.0
        or active_pruned_volume > 0.0
    )

    incident_max = guard.get("incident_wet_fraction_max")
    incident_min = guard.get("incident_wet_fraction_min_positive")
    full_wet = (
        guard.get("support_class") == "full_wet_supported"
        and isinstance(incident_max, (int, float))
        and isinstance(incident_min, (int, float))
        and math.isclose(float(incident_max), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(incident_min), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    )

    if trace_only_support:
        finding = "trace_only_cut_adjacent_support_present_before_guard"
    elif pruned_volume_present and retained_volume_support:
        finding = (
            "pruned_generated_volume_present_but_retained_volume_support_active_"
            "before_guard"
        )
    elif pruned_volume_present:
        finding = "pruned_generated_volume_present_without_trace_only_support_before_guard"
    elif retained_volume_support:
        finding = (
            "retained_volume_support_without_trace_only_or_pruned_generated_"
            "volume_before_guard"
        )
    elif skipped_no_retained:
        finding = "cut_adjacent_support_skipped_without_retained_volume_before_guard"
    else:
        finding = "support_path_unclassified_before_guard"

    return {
        "guard": _summary(guard, GUARD_KEYS),
        "pre_guard_cut_context": _summary(pre_cut_context, CUT_CONTEXT_KEYS),
        "pre_guard_pressure_constraint": _summary(
            pre_pressure_constraint, PRESSURE_CONSTRAINT_KEYS
        ),
        "pre_guard_pressure_constraint_refresh": _summary(
            pre_constraint_refresh, CONSTRAINT_REFRESH_KEYS
        ),
        "post_guard_cut_context": _summary(post_cut_context, CUT_CONTEXT_KEYS),
        "post_guard_pressure_constraint": _summary(
            post_pressure_constraint, PRESSURE_CONSTRAINT_KEYS
        ),
        "pre_guard_retained_volume_support_present": retained_volume_support,
        "pre_guard_cut_adjacent_only_support_present": trace_only_support,
        "pre_guard_cut_adjacent_skipped_no_retained_volume": skipped_no_retained,
        "pre_guard_pruned_generated_volume_present": pruned_volume_present,
        "worst_update_full_wet": full_wet,
        "finding": finding,
    }


def audit_case_log(label: str, path: Path) -> dict[str, Any]:
    parsed = parse_case_log(path)
    guards = parsed["pressure_update_guards"]
    guard_windows = []
    for guard in guards:
        guard_windows.append(
            classify_guard(
                guard,
                pre_cut_context=_nearest_before(parsed["cut_contexts"], guard),
                pre_pressure_constraint=_nearest_before(
                    parsed["pressure_constraints"], guard
                ),
                pre_constraint_refresh=_nearest_before(
                    parsed["constraint_refreshes"], guard
                ),
                post_cut_context=_first_after(parsed["cut_contexts"], guard),
                post_pressure_constraint=_first_after(
                    parsed["pressure_constraints"], guard
                ),
            )
        )

    finding_counts = Counter(window["finding"] for window in guard_windows)
    trace_only = any(
        window["pre_guard_cut_adjacent_only_support_present"]
        for window in guard_windows
    )
    pruned_volume = any(
        window["pre_guard_pruned_generated_volume_present"]
        for window in guard_windows
    )
    retained_support = any(
        window["pre_guard_retained_volume_support_present"]
        for window in guard_windows
    )

    if not guard_windows:
        finding = "no_accepted_pressure_update_guard_found"
    elif trace_only:
        finding = "trace_only_cut_adjacent_support_not_ruled_out"
    elif pruned_volume:
        finding = (
            "trace_only_support_ruled_out_recent_pruned_volume_not_direct_"
            "trace_only_driver"
        )
    else:
        finding = "trace_only_and_recent_pruned_support_absent_before_guards"

    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "guard_count": len(guard_windows),
        "finding": finding,
        "finding_counts": dict(sorted(finding_counts.items())),
        "trace_only_cut_adjacent_support_present_before_any_guard": trace_only,
        "pruned_generated_volume_present_before_any_guard": pruned_volume,
        "retained_volume_support_present_before_any_guard": retained_support,
        "full_wet_guard_count": sum(
            1 for window in guard_windows if window["worst_update_full_wet"]
        ),
        "guard_windows": guard_windows,
    }


def parse_case_log_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"--case-log must be LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    if not label:
        raise ValueError(f"--case-log label is empty in {raw!r}")
    return label, Path(path)


def build_report(case_logs: list[tuple[str, Path]]) -> dict[str, Any]:
    cases = [audit_case_log(label, path) for label, path in case_logs]
    trace_only_cases = [
        case["label"]
        for case in cases
        if case["trace_only_cut_adjacent_support_present_before_any_guard"]
    ]
    pruned_cases = [
        case["label"]
        for case in cases
        if case["pruned_generated_volume_present_before_any_guard"]
    ]
    retained_cases = [
        case["label"]
        for case in cases
        if case["retained_volume_support_present_before_any_guard"]
    ]
    total_guards = sum(case["guard_count"] for case in cases)
    finding_counts = Counter(case["finding"] for case in cases)

    if trace_only_cases:
        finding = "trace_only_cut_adjacent_support_not_ruled_out"
    elif pruned_cases:
        finding = (
            "trace_only_support_ruled_out_recent_pruned_volume_not_direct_"
            "trace_only_driver"
        )
    elif total_guards > 0:
        finding = "trace_only_and_recent_pruned_support_absent_before_guards"
    else:
        finding = "no_accepted_pressure_update_guard_found"

    return {
        "scope": (
            "Accepted pressure-update guard windows joined to active cut-volume "
            "pressure support diagnostics."
        ),
        "finding": finding,
        "case_count": len(cases),
        "guard_count": total_guards,
        "case_finding_counts": dict(sorted(finding_counts.items())),
        "trace_only_cut_adjacent_support_cases": trace_only_cases,
        "pruned_generated_volume_cases": pruned_cases,
        "retained_volume_support_cases": retained_cases,
        "full_wet_guard_count": sum(case["full_wet_guard_count"] for case in cases),
        "trace_only_cut_adjacent_support_ruled_out_before_guards": (
            not trace_only_cases and total_guards > 0
        ),
        "pruned_generated_volume_present_before_some_guard": bool(pruned_cases),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    case_logs = [parse_case_log_arg(raw) for raw in args.case_log]
    report = build_report(case_logs)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
