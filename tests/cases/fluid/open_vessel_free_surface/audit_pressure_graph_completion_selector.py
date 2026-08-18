#!/usr/bin/env python3
"""Audit whether graph-completion selectors cover shifted pressure-update rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read audit_pressure_matrix_support_samples JSON and classify whether "
            "the latest worst pressure-update row is inside the weak-row selector "
            "used by active pressure graph-completion diagnostics."
        )
    )
    parser.add_argument(
        "--support-json",
        action="append",
        default=[],
        help="Support audit JSON as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else default
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return default
        return out if math.isfinite(out) else default
    return default


def int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def value_dict(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    values = record.get("values")
    return values if isinstance(values, dict) else record


def parse_int_sample(value: Any) -> list[int]:
    if not isinstance(value, str) or not value:
        return []
    out: list[int] = []
    for item in value.split("|"):
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def field_sum(row: dict[str, Any], field: str) -> float | None:
    for key in (
        "row_unconstrained_field_abs_sum_by_field",
        "row_field_abs_sum_by_field",
    ):
        values = row.get(key)
        if isinstance(values, dict):
            parsed = numeric(values.get(field))
            if parsed is not None:
                return parsed
    sample = row.get("matrix_sample")
    if isinstance(sample, dict):
        values = sample.get("row_field_abs_sums")
        if isinstance(values, str):
            for item in values.split("|"):
                name, separator, raw = item.partition(":")
                if separator and name == field:
                    return numeric(raw)
    return None


def sampled_row_global_dof(row: dict[str, Any]) -> int | None:
    sample = row.get("matrix_sample")
    if isinstance(sample, dict):
        dof = int_or_none(sample.get("dof"))
        if dof is not None:
            return dof
    return int_or_none(row.get("global_dof"))


def find_sampled_pressure_row(
    report: dict[str, Any],
    global_dof: int | None,
) -> dict[str, Any] | None:
    if global_dof is None:
        return None
    for row in report.get("sampled_pressure_rows", []):
        if isinstance(row, dict) and sampled_row_global_dof(row) == global_dof:
            return row
    return None


def graph_thresholds(report: dict[str, Any]) -> dict[str, float]:
    graph_values = value_dict(report.get("latest_pressure_graph_completion"))
    support_rank_values = value_dict(report.get("latest_support_rank_diagnostic"))
    tolerance = numeric(graph_values.get("tolerance"))
    if tolerance is None:
        tolerance = numeric(support_rank_values.get("tolerance"), 1.0e-14)
    coupling = numeric(graph_values.get("coupling_threshold"))
    if coupling is None:
        coupling = tolerance
    self_threshold = numeric(graph_values.get("self_threshold"), -1.0)
    return {
        "tolerance": tolerance if tolerance is not None else 1.0e-14,
        "coupling_threshold": coupling if coupling is not None else 1.0e-14,
        "self_threshold": self_threshold if self_threshold is not None else -1.0,
    }


def classify_selector_membership(
    *,
    row_velocity_abs_sum: float | None,
    row_pressure_abs_sum: float | None,
    tolerance: float,
    coupling_threshold: float,
    self_threshold: float,
) -> dict[str, Any]:
    zero_coupling = (
        row_velocity_abs_sum is not None and row_velocity_abs_sum <= tolerance
    )
    weak_coupling = (
        row_velocity_abs_sum is not None
        and coupling_threshold >= 0.0
        and row_velocity_abs_sum <= coupling_threshold
    )
    zero_self = row_pressure_abs_sum is not None and row_pressure_abs_sum <= tolerance
    weak_self = (
        row_pressure_abs_sum is not None
        and self_threshold >= 0.0
        and row_pressure_abs_sum <= self_threshold
    )
    selector_eligible = weak_coupling or weak_self
    if row_velocity_abs_sum is None or row_pressure_abs_sum is None:
        reason = "missing_row_support"
    elif selector_eligible:
        reason = "inside_selector_rule"
    elif not weak_coupling and not weak_self:
        reason = "outside_selector_rule_strong_coupling_and_self"
    elif not weak_coupling:
        reason = "outside_selector_rule_strong_coupling"
    else:
        reason = "outside_selector_rule_strong_self"
    return {
        "zero_coupling": zero_coupling,
        "weak_coupling": weak_coupling,
        "zero_self": zero_self,
        "weak_self": weak_self,
        "selector_eligible": selector_eligible,
        "selector_reason": reason,
    }


def threshold_factor_of_current(
    required_threshold: float | None,
    current_threshold: float,
) -> float | None:
    if required_threshold is None or current_threshold <= 0.0:
        return None
    return required_threshold / current_threshold


def selector_threshold_requirements(
    *,
    row_velocity_abs_sum: float | None,
    row_pressure_abs_sum: float | None,
    coupling_threshold: float,
    self_threshold: float,
) -> dict[str, Any]:
    coupling_factor = threshold_factor_of_current(
        row_velocity_abs_sum,
        coupling_threshold,
    )
    self_factor = threshold_factor_of_current(row_pressure_abs_sum, self_threshold)
    options = []
    if row_velocity_abs_sum is not None and coupling_factor is not None:
        options.append(
            {
                "selector": "coupling_threshold",
                "threshold_needed": row_velocity_abs_sum,
                "current_threshold": coupling_threshold,
                "factor_of_current": coupling_factor,
            }
        )
    if row_pressure_abs_sum is not None and self_factor is not None:
        options.append(
            {
                "selector": "self_threshold",
                "threshold_needed": row_pressure_abs_sum,
                "current_threshold": self_threshold,
                "factor_of_current": self_factor,
            }
        )
    least_expansion = (
        min(options, key=lambda option: option["factor_of_current"])
        if options
        else None
    )
    return {
        "selector_thresholds_needed_to_include": {
            "coupling_threshold": row_velocity_abs_sum,
            "self_threshold": row_pressure_abs_sum,
        },
        "selector_threshold_factors_of_current": {
            "coupling_threshold": coupling_factor,
            "self_threshold": self_factor,
        },
        "least_selector_threshold_expansion_to_include": least_expansion,
    }


def audit_selector_coverage(
    label: str,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    graph_values = value_dict(report.get("latest_pressure_graph_completion"))
    update_summary = report.get("pressure_update_support_summary", {})
    if not isinstance(update_summary, dict):
        update_summary = {}
    max_global_dof = int_or_none(update_summary.get("max_update_global_dof"))
    sampled_row = find_sampled_pressure_row(report, max_global_dof)
    thresholds = graph_thresholds(report)
    row_velocity = field_sum(sampled_row, "Velocity") if sampled_row else None
    row_pressure = field_sum(sampled_row, "Pressure") if sampled_row else None
    membership = classify_selector_membership(
        row_velocity_abs_sum=row_velocity,
        row_pressure_abs_sum=row_pressure,
        tolerance=thresholds["tolerance"],
        coupling_threshold=thresholds["coupling_threshold"],
        self_threshold=thresholds["self_threshold"],
    )
    requirements = selector_threshold_requirements(
        row_velocity_abs_sum=row_velocity,
        row_pressure_abs_sum=row_pressure,
        coupling_threshold=thresholds["coupling_threshold"],
        self_threshold=thresholds["self_threshold"],
    )
    candidate_sample = parse_int_sample(graph_values.get("candidate_global_dofs"))
    candidate_sample_contains = (
        max_global_dof in set(candidate_sample) if max_global_dof is not None else False
    )
    if sampled_row is None:
        finding = "max_update_row_not_sampled"
    elif membership["selector_eligible"]:
        finding = "max_update_row_inside_selector_rule"
    else:
        finding = "max_update_row_outside_selector_rule"
    return {
        "label": label,
        "path": str(path),
        "finding": finding,
        "graph_mode": graph_values.get("mode"),
        "requested_mode": graph_values.get("requested_mode"),
        "candidate_row_count": graph_values.get("candidate_row_count"),
        "edge_count": graph_values.get("edge_count"),
        "candidate_log_sample_count": len(candidate_sample),
        "candidate_log_sample_contains_max_update_row": candidate_sample_contains,
        "max_update_global_dof": max_global_dof,
        "max_update_local_dof": update_summary.get("max_update_local_dof"),
        "max_abs_update_pa": update_summary.get("max_abs_update"),
        "sampled_max_update_row": sampled_row is not None,
        "row_velocity_abs_sum": row_velocity,
        "row_pressure_abs_sum": row_pressure,
        "thresholds": thresholds,
        **membership,
        **requirements,
    }


def maximum_numeric(values: list[float | None]) -> float | None:
    numeric_values = [value for value in values if value is not None]
    return max(numeric_values) if numeric_values else None


def casewise_least_selector_threshold_floor(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    floor: dict[str, float | None] = {
        "coupling_threshold": None,
        "self_threshold": None,
    }
    for case in cases:
        option = case.get("least_selector_threshold_expansion_to_include")
        if not isinstance(option, dict):
            continue
        selector = option.get("selector")
        threshold = numeric(option.get("threshold_needed"))
        if selector not in floor or threshold is None:
            continue
        current = floor[selector]
        floor[selector] = threshold if current is None else max(current, threshold)
    return {
        "case_count": len(cases),
        **floor,
    }


def summarize_selector_coverage(
    reports: list[tuple[str, Path, dict[str, Any]]],
) -> dict[str, Any]:
    cases = [
        audit_selector_coverage(label, path, report)
        for label, path, report in reports
    ]
    finding_counts: dict[str, int] = {}
    for case in cases:
        finding = str(case["finding"])
        finding_counts[finding] = finding_counts.get(finding, 0) + 1
    if finding_counts.get("max_update_row_outside_selector_rule", 0) > 0:
        finding = "shifted_pressure_update_rows_escape_weak_row_selector"
    elif finding_counts.get("max_update_row_inside_selector_rule", 0) == len(cases):
        finding = "max_update_rows_remain_inside_weak_row_selector"
    else:
        finding = "selector_coverage_incomplete"
    sampled_outside_cases = [
        case
        for case in cases
        if case["finding"] == "max_update_row_outside_selector_rule"
        and case["sampled_max_update_row"]
    ]
    return {
        "finding": finding,
        "case_count": len(cases),
        "finding_counts": finding_counts,
        "sampled_outside_selector_threshold_floor_if_single_selector_widened": {
            "case_count": len(sampled_outside_cases),
            "coupling_threshold": maximum_numeric(
                [case["row_velocity_abs_sum"] for case in sampled_outside_cases]
            ),
            "self_threshold": maximum_numeric(
                [case["row_pressure_abs_sum"] for case in sampled_outside_cases]
            ),
        },
        "sampled_outside_selector_threshold_floor_if_casewise_least_widened": (
            casewise_least_selector_threshold_floor(sampled_outside_cases)
        ),
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    reports = [
        (label, path, load_json(path))
        for label, path in (
            parse_labeled_path(value) for value in args.support_json
        )
    ]
    summary = summarize_selector_coverage(reports)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
