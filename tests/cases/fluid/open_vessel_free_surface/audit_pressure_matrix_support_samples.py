#!/usr/bin/env python3
"""Summarize sampled pressure matrix-support diagnostics from solver logs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_pressure_update_guard import parse_key_values  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize newton_matrix_support_sample diagnostics."
    )
    parser.add_argument("--solver-log", type=Path, required=True)
    parser.add_argument("--field-name", default="Pressure")
    parser.add_argument("--coupling-field-name", default="Velocity")
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_matrix_sample_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=newton_matrix_support_sample" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "newton_matrix_support_sample":
        return None
    field = values.get("field")
    if field is not None and field != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_operator_matrix_support_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=pressure_row_operator_matrix_support" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "pressure_row_operator_matrix_support":
        return None
    field = values.get("field")
    if field is not None and field != field_name:
        return None
    pressure_field = values.get("pressure_field")
    if pressure_field is not None and pressure_field != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_operator_matrix_summary_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=pressure_row_operator_matrix_summary" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "pressure_row_operator_matrix_summary":
        return None
    pressure_field = values.get("pressure_field")
    if pressure_field is not None and pressure_field != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_constraint_sample_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=level_set_active_side_vertex_constraint_sample" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "level_set_active_side_vertex_constraint_sample":
        return None
    if values.get("field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_accepted_pressure_update_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=accepted_pressure_update_guard" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "accepted_pressure_update_guard":
        return None
    if values.get("field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_support_rank_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=active_pressure_support_rank" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "active_pressure_support_rank":
        return None
    if values.get("pressure_field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_support_rank_clamp_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=active_pressure_support_rank_clamp" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "active_pressure_support_rank_clamp":
        return None
    if values.get("pressure_field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_pressure_update_support_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=active_pressure_update_support" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "active_pressure_update_support":
        return None
    if values.get("pressure_field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def parse_pressure_graph_completion_line(
    line: str,
    *,
    line_number: int,
    field_name: str,
) -> dict[str, Any] | None:
    if "diagnostic=active_pressure_graph_completion" not in line:
        return None
    values = parse_key_values(line)
    if values.get("diagnostic") != "active_pressure_graph_completion":
        return None
    if values.get("pressure_field") != field_name:
        return None
    return {"line_number": line_number, "values": values}


def read_solver_log(
    solver_log: Path,
    *,
    field_name: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    matrix_samples: list[dict[str, Any]] = []
    operator_matrix_support_samples: list[dict[str, Any]] = []
    operator_matrix_summaries: list[dict[str, Any]] = []
    constraint_samples: list[dict[str, Any]] = []
    accepted_updates: list[dict[str, Any]] = []
    support_rank_diagnostics: list[dict[str, Any]] = []
    support_rank_clamps: list[dict[str, Any]] = []
    pressure_graph_completions: list[dict[str, Any]] = []
    pressure_update_support_diagnostics: list[dict[str, Any]] = []
    with solver_log.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            matrix_sample = parse_matrix_sample_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if matrix_sample is not None:
                matrix_samples.append(matrix_sample)
            operator_matrix_support = parse_operator_matrix_support_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if operator_matrix_support is not None:
                operator_matrix_support_samples.append(operator_matrix_support)
            operator_matrix_summary = parse_operator_matrix_summary_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if operator_matrix_summary is not None:
                operator_matrix_summaries.append(operator_matrix_summary)
            constraint_sample = parse_constraint_sample_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if constraint_sample is not None:
                constraint_samples.append(constraint_sample)
            accepted_update = parse_accepted_pressure_update_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if accepted_update is not None:
                accepted_updates.append(accepted_update)
            support_rank = parse_support_rank_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if support_rank is not None:
                support_rank_diagnostics.append(support_rank)
            support_rank_clamp = parse_support_rank_clamp_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if support_rank_clamp is not None:
                support_rank_clamps.append(support_rank_clamp)
            graph_completion = parse_pressure_graph_completion_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if graph_completion is not None:
                pressure_graph_completions.append(graph_completion)
            pressure_update_support = parse_pressure_update_support_line(
                line,
                line_number=line_number,
                field_name=field_name,
            )
            if pressure_update_support is not None:
                pressure_update_support_diagnostics.append(pressure_update_support)
    return (
        matrix_samples,
        operator_matrix_support_samples,
        operator_matrix_summaries,
        constraint_samples,
        accepted_updates,
        support_rank_diagnostics,
        support_rank_clamps,
        pressure_graph_completions,
        pressure_update_support_diagnostics,
    )


def local_dof_for_matrix_sample(values: dict[str, Any]) -> int | None:
    local_dof = values.get("field_local_dof")
    return local_dof if isinstance(local_dof, int) else None


def local_dof_for_operator_matrix_support(values: dict[str, Any]) -> int | None:
    local_dof = values.get("field_local_dof")
    if isinstance(local_dof, int):
        return local_dof
    local_dof = values.get("pressure_local_dof")
    return local_dof if isinstance(local_dof, int) else None


def latest_constraint_samples_before(
    constraint_samples: list[dict[str, Any]],
    *,
    line_number: int,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for sample in constraint_samples:
        if sample["line_number"] >= line_number:
            continue
        local_dof = sample["values"].get("local_dof")
        if not isinstance(local_dof, int):
            continue
        prior = selected.get(local_dof)
        if prior is None or sample["line_number"] > prior["line_number"]:
            selected[local_dof] = sample
    return selected


def is_abs_leq(values: dict[str, Any], key: str, tolerance: float) -> bool:
    value = values.get(key)
    return isinstance(value, (int, float)) and abs(float(value)) <= tolerance


def parse_field_abs_sums(value: Any) -> dict[str, float]:
    if not isinstance(value, str):
        return {}
    result: dict[str, float] = {}
    for item in value.split("|"):
        if ":" not in item:
            continue
        name, number = item.split(":", 1)
        if not name:
            continue
        try:
            result[name] = float(number)
        except ValueError:
            continue
    return result


def field_abs_leq(
    row: dict[str, Any],
    key: str,
    field_name: str,
    tolerance: float,
) -> bool:
    field_sums = row.get(key)
    if not isinstance(field_sums, dict):
        return False
    value = field_sums.get(field_name)
    return isinstance(value, (int, float)) and abs(float(value)) <= tolerance


def field_abs_value(
    row: dict[str, Any],
    key: str,
    field_name: str,
) -> float | None:
    field_sums = row.get(key)
    if not isinstance(field_sums, dict):
        return None
    value = field_sums.get(field_name)
    return float(value) if isinstance(value, (int, float)) else None


def numeric_value(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def truthy_int(value: Any) -> bool:
    return value == 1 or value is True


def positive_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0


def is_identity_pressure_constraint_row(row: dict[str, Any]) -> bool:
    values = row.get("matrix_sample")
    if not isinstance(values, dict):
        return False

    def near(key: str, target: float, tolerance: float = 1.0e-12) -> bool:
        value = values.get(key)
        return isinstance(value, (int, float)) and abs(float(value) - target) <= tolerance

    return (
        near("row_abs_sum", 1.0)
        and near("col_abs_sum", 1.0)
        and near("diag", 1.0)
        and near("row_constrained_abs_sum", 1.0)
        and near("col_constrained_abs_sum", 1.0)
        and near("row_unconstrained_abs_sum", 0.0)
        and near("col_unconstrained_abs_sum", 0.0)
    )


def sorted_local_dof_string(rows: set[int]) -> str:
    if not rows:
        return "none"
    return "|".join(str(row) for row in sorted(rows))


def summarize_operator_matrix_support_by_op(
    operator_support_rows: list[dict[str, Any]],
    *,
    zero_tolerance: float,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in operator_support_rows:
        op = row.get("op")
        local_pressure_row = row.get("local_pressure_row")
        values = row.get("operator_matrix_support", {})
        if not isinstance(op, str) or not isinstance(local_pressure_row, int):
            continue
        if not isinstance(values, dict):
            values = {}
        summary = summaries.setdefault(
            op,
            {
                "sampled_row_count": 0,
                "nonzero_row_count": 0,
                "nonzero_self_row_count": 0,
                "nonzero_coupling_row_count": 0,
                "nonzero_diag_count": 0,
                "_sampled_local_pressure_rows": set(),
                "_nonzero_row_local_pressure_rows": set(),
                "_nonzero_self_local_pressure_rows": set(),
                "_nonzero_coupling_local_pressure_rows": set(),
                "_positive_self_values": [],
                "_positive_coupling_values": [],
                "_positive_diag_values": [],
            },
        )
        summary["sampled_row_count"] += 1
        summary["_sampled_local_pressure_rows"].add(local_pressure_row)

        row_abs_sum = numeric_value(values.get("row_abs_sum"))
        row_self_abs_sum = numeric_value(values.get("row_self_abs_sum"))
        row_coupling_abs_sum = numeric_value(values.get("row_coupling_abs_sum"))
        diag = numeric_value(values.get("diag"))

        if row_abs_sum is not None and abs(row_abs_sum) > zero_tolerance:
            summary["nonzero_row_count"] += 1
            summary["_nonzero_row_local_pressure_rows"].add(local_pressure_row)
        if row_self_abs_sum is not None and abs(row_self_abs_sum) > zero_tolerance:
            summary["nonzero_self_row_count"] += 1
            summary["_nonzero_self_local_pressure_rows"].add(local_pressure_row)
            summary["_positive_self_values"].append(abs(row_self_abs_sum))
        if (
            row_coupling_abs_sum is not None
            and abs(row_coupling_abs_sum) > zero_tolerance
        ):
            summary["nonzero_coupling_row_count"] += 1
            summary["_nonzero_coupling_local_pressure_rows"].add(local_pressure_row)
            summary["_positive_coupling_values"].append(abs(row_coupling_abs_sum))
        if diag is not None and abs(diag) > zero_tolerance:
            summary["nonzero_diag_count"] += 1
            summary["_positive_diag_values"].append(abs(diag))

    out: dict[str, dict[str, Any]] = {}
    for op, summary in sorted(summaries.items()):
        positive_self_values = summary.pop("_positive_self_values")
        positive_coupling_values = summary.pop("_positive_coupling_values")
        positive_diag_values = summary.pop("_positive_diag_values")
        sampled_rows = summary.pop("_sampled_local_pressure_rows")
        nonzero_rows = summary.pop("_nonzero_row_local_pressure_rows")
        nonzero_self_rows = summary.pop("_nonzero_self_local_pressure_rows")
        nonzero_coupling_rows = summary.pop(
            "_nonzero_coupling_local_pressure_rows"
        )
        summary["zero_row_count"] = (
            summary["sampled_row_count"] - summary["nonzero_row_count"]
        )
        summary["zero_self_row_count"] = (
            summary["sampled_row_count"] - summary["nonzero_self_row_count"]
        )
        summary["zero_coupling_row_count"] = (
            summary["sampled_row_count"] - summary["nonzero_coupling_row_count"]
        )
        summary["sampled_local_pressure_rows"] = sorted_local_dof_string(
            sampled_rows
        )
        summary["nonzero_row_local_pressure_rows"] = sorted_local_dof_string(
            nonzero_rows
        )
        summary["nonzero_self_local_pressure_rows"] = sorted_local_dof_string(
            nonzero_self_rows
        )
        summary["nonzero_coupling_local_pressure_rows"] = (
            sorted_local_dof_string(nonzero_coupling_rows)
        )
        summary["min_positive_self_row_abs_sum"] = (
            min(positive_self_values) if positive_self_values else None
        )
        summary["max_self_row_abs_sum"] = (
            max(positive_self_values) if positive_self_values else None
        )
        summary["min_positive_coupling_row_abs_sum"] = (
            min(positive_coupling_values)
            if positive_coupling_values
            else None
        )
        summary["max_coupling_row_abs_sum"] = (
            max(positive_coupling_values)
            if positive_coupling_values
            else None
        )
        summary["min_positive_diag"] = (
            min(positive_diag_values) if positive_diag_values else None
        )
        summary["max_diag"] = (
            max(positive_diag_values) if positive_diag_values else None
        )
        out[op] = summary
    return out


def support_provenance_summary(
    rows: list[dict[str, Any]],
    *,
    field_name: str,
    coupling_field_name: str,
    zero_tolerance: float,
    weak_coupling_threshold: float | None = None,
    weak_self_threshold: float | None = None,
) -> dict[str, Any]:
    weak_coupling_threshold = (
        zero_tolerance if weak_coupling_threshold is None else weak_coupling_threshold
    )
    weak_self_threshold = (
        zero_tolerance if weak_self_threshold is None else weak_self_threshold
    )

    counts: Counter[str] = Counter()
    max_row_coupling_by_class: dict[str, float] = {}
    max_row_self_by_class: dict[str, float] = {}

    def bump(label: str) -> None:
        counts[label] += 1

    def update_max(target: dict[str, float], label: str, value: float | None) -> None:
        if value is None:
            return
        target[label] = max(target.get(label, 0.0), abs(value))

    for row in rows:
        constraint = row.get("constraint_sample")
        if not isinstance(constraint, dict):
            bump("missing_constraint_sample")
            continue

        row_coupling = field_abs_value(
            row,
            "row_field_abs_sum_by_field",
            coupling_field_name,
        )
        row_self = field_abs_value(
            row,
            "row_field_abs_sum_by_field",
            field_name,
        )
        zero_coupling = row_coupling is not None and abs(row_coupling) <= zero_tolerance
        weak_coupling = (
            row_coupling is not None
            and abs(row_coupling) > zero_tolerance
            and abs(row_coupling) <= weak_coupling_threshold
        )
        zero_self = row_self is not None and abs(row_self) <= zero_tolerance
        weak_self = (
            row_self is not None
            and abs(row_self) > zero_tolerance
            and abs(row_self) <= weak_self_threshold
        )

        active_support = truthy_int(constraint.get("active_dof_support"))
        inactive_constraint = truthy_int(constraint.get("inactive_constraint"))
        retained = positive_numeric(constraint.get("retained_rule_count"))
        active_sign = truthy_int(constraint.get("vertex_active_sign"))
        inactive_sign_retained = active_support and retained and not active_sign
        identity_pressure_row = is_identity_pressure_constraint_row(row)

        if active_support:
            bump("active_support")
        if inactive_constraint:
            bump("inactive_constraint")
        if retained:
            bump("retained_rule_support")
        if active_sign:
            bump("vertex_active_sign")
        if inactive_sign_retained:
            bump("inactive_sign_retained_support")
        if identity_pressure_row:
            bump("identity_pressure_row")
            continue

        if zero_coupling:
            bump("zero_coupling_row")
        if weak_coupling:
            bump("weak_coupling_row")
        if zero_self:
            bump("zero_self_row")
        if weak_self:
            bump("weak_self_row")

        classes = []
        if active_support:
            classes.append("active_support")
        if inactive_constraint:
            classes.append("inactive_constraint")
        if retained:
            classes.append("retained")
        if active_sign:
            classes.append("active_sign")
        if inactive_sign_retained:
            classes.append("inactive_sign_retained")
        if zero_coupling:
            classes.append("zero_coupling")
        elif weak_coupling:
            classes.append("weak_coupling")
        if zero_self:
            classes.append("zero_self")
        elif weak_self:
            classes.append("weak_self")

        for label in classes:
            update_max(max_row_coupling_by_class, label, row_coupling)
            update_max(max_row_self_by_class, label, row_self)

        if active_support and retained and zero_coupling:
            bump("retained_zero_coupling_row")
        if active_support and retained and weak_coupling:
            bump("retained_weak_coupling_row")
        if active_support and retained and zero_self:
            bump("retained_zero_self_row")
        if active_support and retained and weak_self:
            bump("retained_weak_self_row")
        if inactive_sign_retained and zero_coupling:
            bump("inactive_sign_retained_zero_coupling_row")
        if inactive_sign_retained and weak_coupling:
            bump("inactive_sign_retained_weak_coupling_row")
        if inactive_sign_retained and zero_self:
            bump("inactive_sign_retained_zero_self_row")
        if inactive_sign_retained and weak_self:
            bump("inactive_sign_retained_weak_self_row")

    return {
        "sampled_row_count": len(rows),
        "zero_tolerance": zero_tolerance,
        "weak_coupling_threshold": weak_coupling_threshold,
        "weak_self_threshold": weak_self_threshold,
        "counts": dict(sorted(counts.items())),
        "max_row_coupling_abs_sum_by_class": dict(
            sorted(max_row_coupling_by_class.items())
        ),
        "max_row_self_abs_sum_by_class": dict(sorted(max_row_self_by_class.items())),
    }


def parse_top_update_details(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or value == "none":
        return []
    rows: list[dict[str, Any]] = []
    for item in value.split("|"):
        parts = item.split(":")
        if len(parts) < 2:
            continue
        try:
            row: dict[str, Any] = {
                "local_pressure_row": int(parts[0]),
                "global_dof": int(parts[1]),
            }
        except ValueError:
            continue
        for part in parts[2:]:
            if "=" not in part:
                continue
            key, raw = part.split("=", 1)
            try:
                row[key] = float(raw)
            except ValueError:
                row[key] = raw
        rows.append(row)
    return rows


def parse_dof_sample_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, str) or value in {"", "none"}:
        return []
    out: list[int] = []
    for item in value.split("|"):
        if item == "...":
            continue
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def pressure_update_support_summary(
    diagnostics: list[dict[str, Any]],
    *,
    zero_tolerance: float,
) -> dict[str, Any] | None:
    if not diagnostics:
        return None
    latest = diagnostics[-1].get("values")
    if not isinstance(latest, dict):
        return None
    details = parse_top_update_details(latest.get("top_update_details"))
    max_local = latest.get("max_update_local_dof")
    max_detail = next(
        (
            row
            for row in details
            if isinstance(max_local, int)
            and row.get("local_pressure_row") == max_local
        ),
        details[0] if details else None,
    )

    def detail_float(key: str) -> float | None:
        if not isinstance(max_detail, dict):
            return None
        return numeric_value(max_detail.get(key))

    def latest_float(key: str) -> float | None:
        return numeric_value(latest.get(key))

    update = detail_float("update")
    diag = detail_float("diag")
    row_self_action = latest_float("max_update_row_self_action")
    row_coupling_action = latest_float("max_update_row_coupling_action")
    row_self_constant_action = latest_float("max_update_row_self_constant_action")
    row_self_nonconstant_action = latest_float(
        "max_update_row_self_nonconstant_action"
    )
    diag_action_abs = (
        abs(update * diag)
        if update is not None and diag is not None
        else None
    )
    self_to_coupling_action_ratio = None
    if (
        row_self_action is not None
        and row_coupling_action is not None
        and abs(row_coupling_action) > zero_tolerance
    ):
        self_to_coupling_action_ratio = (
            abs(row_self_action) / abs(row_coupling_action)
        )
    constant_action_fraction = None
    nonconstant_action_fraction = None
    if row_self_action is not None and abs(row_self_action) > zero_tolerance:
        if row_self_constant_action is not None:
            constant_action_fraction = (
                abs(row_self_constant_action) / abs(row_self_action)
            )
        if row_self_nonconstant_action is not None:
            nonconstant_action_fraction = (
                abs(row_self_nonconstant_action) / abs(row_self_action)
            )

    return {
        "diagnostic_count": len(diagnostics),
        "line_number": diagnostics[-1].get("line_number"),
        "phase": latest.get("phase"),
        "parsed_top_update_count": len(details),
        "top_update_details": details,
        "max_update_local_dof": latest.get("max_update_local_dof"),
        "max_update_global_dof": latest.get("max_update_global_dof"),
        "max_abs_update": latest.get("max_abs_update"),
        "max_update_rhs": latest.get("max_update_rhs"),
        "max_update_row_action": latest.get("max_update_row_action"),
        "max_update_row_linear_residual": latest.get(
            "max_update_row_linear_residual"
        ),
        "max_update_detail": max_detail,
        "max_update_diag_action_abs": diag_action_abs,
        "max_update_self_to_coupling_action_ratio": (
            float(self_to_coupling_action_ratio)
            if self_to_coupling_action_ratio is not None
            else None
        ),
        "max_update_constant_self_action_fraction": (
            float(constant_action_fraction)
            if constant_action_fraction is not None
            else None
        ),
        "max_update_nonconstant_self_action_fraction": (
            float(nonconstant_action_fraction)
            if nonconstant_action_fraction is not None
            else None
        ),
        "same_sign_pressure_action_top_edge_count": latest.get(
            "same_sign_pressure_action_top_edge_count"
        ),
        "same_sign_pressure_action_component_count": latest.get(
            "same_sign_pressure_action_component_count"
        ),
        "same_sign_pressure_action_largest_component_size": latest.get(
            "same_sign_pressure_action_largest_component_size"
        ),
        "same_sign_pressure_action_covered_top_update_count": latest.get(
            "same_sign_pressure_action_covered_top_update_count"
        ),
        "same_sign_pressure_action_isolated_top_update_count": latest.get(
            "same_sign_pressure_action_isolated_top_update_count"
        ),
        "same_sign_pressure_action_largest_component_has_max_update": latest.get(
            "same_sign_pressure_action_largest_component_has_max_update"
        ),
        "same_sign_pressure_action_covered_global_dofs": parse_dof_sample_list(
            latest.get("same_sign_pressure_action_covered_global_dofs")
        ),
        "same_sign_pressure_action_isolated_global_dofs": parse_dof_sample_list(
            latest.get("same_sign_pressure_action_isolated_global_dofs")
        ),
        "same_sign_pressure_action_largest_component_global_dofs": (
            parse_dof_sample_list(
                latest.get("same_sign_pressure_action_largest_component_global_dofs")
            )
        ),
    }


def summarize_pressure_matrix_support(
    *,
    solver_log: Path,
    field_name: str = "Pressure",
    coupling_field_name: str = "Velocity",
    zero_tolerance: float = 1.0e-14,
) -> dict[str, Any]:
    (
        matrix_samples,
        operator_matrix_support_samples,
        operator_matrix_summaries,
        constraint_samples,
        accepted_updates,
        support_rank_diagnostics,
        support_rank_clamps,
        pressure_graph_completions,
        pressure_update_support_diagnostics,
    ) = read_solver_log(
        solver_log,
        field_name=field_name,
    )
    latest_by_local_dof: dict[int, dict[str, Any]] = {}
    for sample in matrix_samples:
        local_dof = local_dof_for_matrix_sample(sample["values"])
        if local_dof is None:
            continue
        prior = latest_by_local_dof.get(local_dof)
        if prior is None or sample["line_number"] > prior["line_number"]:
            latest_by_local_dof[local_dof] = sample

    latest_operator_support_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for sample in operator_matrix_support_samples:
        local_dof = local_dof_for_operator_matrix_support(sample["values"])
        op = sample["values"].get("op")
        if local_dof is None or not isinstance(op, str):
            continue
        key = (op, local_dof)
        prior = latest_operator_support_by_key.get(key)
        if prior is None or sample["line_number"] > prior["line_number"]:
            latest_operator_support_by_key[key] = sample

    operator_support_rows: list[dict[str, Any]] = []
    for op, local_dof in sorted(latest_operator_support_by_key):
        sample = latest_operator_support_by_key[(op, local_dof)]
        operator_support_rows.append(
            {
                "op": op,
                "local_pressure_row": local_dof,
                "line_number": sample["line_number"],
                "operator_matrix_support": sample["values"],
            }
        )

    latest_operator_summary_by_op: dict[str, dict[str, Any]] = {}
    for summary in operator_matrix_summaries:
        values = summary.get("values")
        if not isinstance(values, dict):
            continue
        op = values.get("op")
        if not isinstance(op, str):
            continue
        prior = latest_operator_summary_by_op.get(op)
        if prior is None or summary["line_number"] > prior["line_number"]:
            latest_operator_summary_by_op[op] = summary

    rows: list[dict[str, Any]] = []
    for local_dof in sorted(latest_by_local_dof):
        sample = latest_by_local_dof[local_dof]
        constraint_by_dof = latest_constraint_samples_before(
            constraint_samples,
            line_number=sample["line_number"],
        )
        constraint_sample = constraint_by_dof.get(local_dof)
        record: dict[str, Any] = {
            "local_pressure_row": local_dof,
            "matrix_sample_line_number": sample["line_number"],
            "matrix_sample": sample["values"],
            "constraint_sample_available": constraint_sample is not None,
        }
        row_field_sums = parse_field_abs_sums(
            sample["values"].get("row_field_abs_sums")
        )
        col_field_sums = parse_field_abs_sums(
            sample["values"].get("col_field_abs_sums")
        )
        row_constrained_field_sums = parse_field_abs_sums(
            sample["values"].get("row_constrained_field_abs_sums")
        )
        row_unconstrained_field_sums = parse_field_abs_sums(
            sample["values"].get("row_unconstrained_field_abs_sums")
        )
        col_constrained_field_sums = parse_field_abs_sums(
            sample["values"].get("col_constrained_field_abs_sums")
        )
        col_unconstrained_field_sums = parse_field_abs_sums(
            sample["values"].get("col_unconstrained_field_abs_sums")
        )
        if row_field_sums:
            record["row_field_abs_sum_by_field"] = row_field_sums
        if col_field_sums:
            record["col_field_abs_sum_by_field"] = col_field_sums
        if row_constrained_field_sums:
            record["row_constrained_field_abs_sum_by_field"] = (
                row_constrained_field_sums
            )
        if row_unconstrained_field_sums:
            record["row_unconstrained_field_abs_sum_by_field"] = (
                row_unconstrained_field_sums
            )
        if col_constrained_field_sums:
            record["col_constrained_field_abs_sum_by_field"] = (
                col_constrained_field_sums
            )
        if col_unconstrained_field_sums:
            record["col_unconstrained_field_abs_sum_by_field"] = (
                col_unconstrained_field_sums
            )
        if constraint_sample is not None:
            record["constraint_sample_line_number"] = constraint_sample["line_number"]
            record["constraint_sample"] = constraint_sample["values"]
        rows.append(record)

    status_counts = Counter(
        str(row["matrix_sample"].get("status"))
        for row in rows
    )
    operator_status_counts = Counter(
        str(row["operator_matrix_support"].get("status"))
        for row in operator_support_rows
    )
    zero_row_count = sum(
        1
        for row in rows
        if is_abs_leq(row["matrix_sample"], "row_abs_sum", zero_tolerance)
    )
    zero_col_count = sum(
        1
        for row in rows
        if is_abs_leq(row["matrix_sample"], "col_abs_sum", zero_tolerance)
    )
    zero_diag_count = sum(
        1
        for row in rows
        if is_abs_leq(row["matrix_sample"], "diag", zero_tolerance)
    )
    active_support_count = sum(
        1
        for row in rows
        if row.get("constraint_sample", {}).get("active_dof_support") == 1
    )
    inactive_constraint_count = sum(
        1
        for row in rows
        if row.get("constraint_sample", {}).get("inactive_constraint") == 1
    )
    row_field_block_sample_count = sum(
        1
        for row in rows
        if isinstance(row.get("row_field_abs_sum_by_field"), dict)
    )
    col_field_block_sample_count = sum(
        1
        for row in rows
        if isinstance(row.get("col_field_abs_sum_by_field"), dict)
    )
    zero_coupling_row_block_count = sum(
        1
        for row in rows
        if field_abs_leq(
            row,
            "row_field_abs_sum_by_field",
            coupling_field_name,
            zero_tolerance,
        )
    )
    zero_coupling_col_block_count = sum(
        1
        for row in rows
        if field_abs_leq(
            row,
            "col_field_abs_sum_by_field",
            coupling_field_name,
            zero_tolerance,
        )
    )
    zero_self_row_block_count = sum(
        1
        for row in rows
        if field_abs_leq(
            row,
            "row_field_abs_sum_by_field",
            field_name,
            zero_tolerance,
        )
    )
    zero_self_col_block_count = sum(
        1
        for row in rows
        if field_abs_leq(
            row,
            "col_field_abs_sum_by_field",
            field_name,
            zero_tolerance,
        )
    )
    operator_nonzero_self_row_ops = sorted(
        {
            str(row["op"])
            for row in operator_support_rows
            if not is_abs_leq(
                row["operator_matrix_support"],
                "row_self_abs_sum",
                zero_tolerance,
            )
        }
    )
    operator_nonzero_coupling_row_ops = sorted(
        {
            str(row["op"])
            for row in operator_support_rows
            if not is_abs_leq(
                row["operator_matrix_support"],
                "row_coupling_abs_sum",
                zero_tolerance,
            )
        }
    )

    return {
        "solver_log": str(solver_log),
        "field_name": field_name,
        "coupling_field_name": coupling_field_name,
        "zero_tolerance": zero_tolerance,
        "matrix_sample_count": len(matrix_samples),
        "operator_matrix_support_sample_count": len(
            operator_matrix_support_samples
        ),
        "operator_matrix_summary_count": len(operator_matrix_summaries),
        "latest_operator_matrix_support_sample_count": len(
            operator_support_rows
        ),
        "latest_operator_matrix_summary_count": len(
            latest_operator_summary_by_op
        ),
        "latest_matrix_sampled_row_count": len(rows),
        "constraint_sample_count": len(constraint_samples),
        "accepted_pressure_update_count": len(accepted_updates),
        "support_rank_diagnostic_count": len(support_rank_diagnostics),
        "support_rank_clamp_count": len(support_rank_clamps),
        "pressure_graph_completion_count": len(pressure_graph_completions),
        "pressure_update_support_diagnostic_count": len(
            pressure_update_support_diagnostics
        ),
        "pressure_graph_completions": pressure_graph_completions,
        "pressure_update_support_diagnostics": pressure_update_support_diagnostics,
        "latest_accepted_pressure_update": (
            accepted_updates[-1] if accepted_updates else None
        ),
        "latest_support_rank_diagnostic": (
            support_rank_diagnostics[-1] if support_rank_diagnostics else None
        ),
        "latest_support_rank_clamp": (
            support_rank_clamps[-1] if support_rank_clamps else None
        ),
        "latest_pressure_graph_completion": (
            pressure_graph_completions[-1] if pressure_graph_completions else None
        ),
        "latest_pressure_update_support_diagnostic": (
            pressure_update_support_diagnostics[-1]
            if pressure_update_support_diagnostics
            else None
        ),
        "pressure_update_support_summary": pressure_update_support_summary(
            pressure_update_support_diagnostics,
            zero_tolerance=zero_tolerance,
        ),
        "matrix_sample_status_counts": dict(sorted(status_counts.items())),
        "operator_matrix_support_status_counts": dict(
            sorted(operator_status_counts.items())
        ),
        "operator_matrix_support_ops": sorted(
            {
                str(row["op"])
                for row in operator_support_rows
            }
        ),
        "operator_matrix_support_nonzero_self_row_ops": (
            operator_nonzero_self_row_ops
        ),
        "operator_matrix_support_nonzero_coupling_row_ops": (
            operator_nonzero_coupling_row_ops
        ),
        "operator_matrix_support_by_op": summarize_operator_matrix_support_by_op(
            operator_support_rows,
            zero_tolerance=zero_tolerance,
        ),
        "operator_matrix_summary_by_op": {
            op: summary["values"]
            for op, summary in sorted(latest_operator_summary_by_op.items())
        },
        "matrix_sampled_zero_row_count": int(zero_row_count),
        "matrix_sampled_zero_col_count": int(zero_col_count),
        "matrix_sampled_zero_diag_count": int(zero_diag_count),
        "matrix_sampled_nonzero_row_count": len(rows) - int(zero_row_count),
        "matrix_sampled_nonzero_col_count": len(rows) - int(zero_col_count),
        "matrix_sampled_nonzero_diag_count": len(rows) - int(zero_diag_count),
        "matrix_sampled_active_support_count": int(active_support_count),
        "matrix_sampled_inactive_constraint_count": int(inactive_constraint_count),
        "matrix_sampled_row_field_block_sample_count": int(
            row_field_block_sample_count
        ),
        "matrix_sampled_col_field_block_sample_count": int(
            col_field_block_sample_count
        ),
        "matrix_sampled_zero_coupling_row_block_count": int(
            zero_coupling_row_block_count
        ),
        "matrix_sampled_zero_coupling_col_block_count": int(
            zero_coupling_col_block_count
        ),
        "matrix_sampled_nonzero_coupling_row_block_count": (
            row_field_block_sample_count - int(zero_coupling_row_block_count)
        ),
        "matrix_sampled_nonzero_coupling_col_block_count": (
            col_field_block_sample_count - int(zero_coupling_col_block_count)
        ),
        "matrix_sampled_zero_self_row_block_count": int(
            zero_self_row_block_count
        ),
        "matrix_sampled_zero_self_col_block_count": int(
            zero_self_col_block_count
        ),
        "matrix_sampled_nonzero_self_row_block_count": (
            row_field_block_sample_count - int(zero_self_row_block_count)
        ),
        "matrix_sampled_nonzero_self_col_block_count": (
            col_field_block_sample_count - int(zero_self_col_block_count)
        ),
        "constraint_support_provenance_summary": support_provenance_summary(
            rows,
            field_name=field_name,
            coupling_field_name=coupling_field_name,
            zero_tolerance=zero_tolerance,
            weak_coupling_threshold=1.0e-3,
            weak_self_threshold=1.0e-7,
        ),
        "sampled_pressure_rows": rows,
        "pressure_row_operator_matrix_support_samples": operator_support_rows,
    }


def main() -> int:
    args = parse_args()
    report = summarize_pressure_matrix_support(
        solver_log=args.solver_log,
        field_name=args.field_name,
        coupling_field_name=args.coupling_field_name,
        zero_tolerance=args.zero_tolerance,
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
