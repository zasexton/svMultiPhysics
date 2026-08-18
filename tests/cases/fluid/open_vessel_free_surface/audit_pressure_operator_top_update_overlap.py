#!/usr/bin/env python3
"""Join top pressure-update rows with aggregate operator-support row samples."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


OP_GALERKIN = "equations_diagnostic_ns_galerkin_continuity"
OP_NONPRESSURE = "equations_diagnostic_ns_vms_pspg_nonpressure"
OP_DIRECT_PGRAD = "equations_diagnostic_ns_vms_pspg_pressure_gradient"
OP_WALL_NORMAL_PGRAD = "equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient"
OP_WALL_TANGENTIAL_PGRAD = (
    "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient"
)
OP_GHOST = "equations_diagnostic_ns_pressure_ghost_penalty"

DEFAULT_OPERATORS = (
    OP_GALERKIN,
    OP_NONPRESSURE,
    OP_DIRECT_PGRAD,
)

EXACT_PHYSICAL_OPERATORS = (
    OP_GALERKIN,
    OP_NONPRESSURE,
    OP_DIRECT_PGRAD,
    OP_WALL_NORMAL_PGRAD,
    OP_WALL_TANGENTIAL_PGRAD,
    OP_GHOST,
)

AGGREGATE_SAMPLE_NAMES = (
    "zero_coupling",
    "zero_row",
    "weakest_coupling",
    "weakest_self",
)

AGGREGATE_SAMPLE_FLAGS = tuple(f"{name}_sample" for name in AGGREGATE_SAMPLE_NAMES)

DIRECT_PSPG_PATH_CLASSES = {
    "direct_pspg_weak_self_with_wall_support",
    "direct_pspg_weak_self_no_wall_support",
    "direct_pspg_positive_self",
}

GHOST_PENALTY_PATH_CLASSES = {
    "ghost_penalty_positive_self",
    "ghost_penalty_weak_self",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read audit_pressure_matrix_support_samples JSON files and classify "
            "whether logged top pressure-update rows overlap operator-summary "
            "zero/weak sampled row lists."
        )
    )
    parser.add_argument(
        "--support-json",
        action="append",
        default=[],
        help="Support audit JSON as LABEL=PATH. May be repeated.",
    )
    parser.add_argument(
        "--operator",
        action="append",
        default=[],
        help=(
            "Operator name to include. Defaults to Galerkin continuity, "
            "VMS/PSPG nonpressure, and direct VMS/PSPG pressure-gradient."
        ),
    )
    parser.add_argument("--top-events", type=int, default=12)
    parser.add_argument("--zero-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--weak-velocity-row-sum", type=float, default=1.0e-3)
    parser.add_argument("--weak-pressure-row-sum", type=float, default=1.0e-7)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}")
    return label, Path(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_sample(value: Any) -> list[int]:
    if not isinstance(value, str) or not value or value == "none":
        return []
    out: list[int] = []
    for item in value.split("|"):
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        try:
            out = float(value)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def support_class(
    value: Any,
    *,
    zero_tolerance: float,
    weak_threshold: float,
) -> str:
    number = numeric(value)
    if number is None:
        return "missing"
    magnitude = abs(number)
    if magnitude <= zero_tolerance:
        return "zero"
    if magnitude <= weak_threshold:
        return "weak"
    return "positive"


def classify_row_support(
    row: dict[str, Any],
    *,
    zero_tolerance: float,
    weak_velocity_row_sum: float,
    weak_pressure_row_sum: float,
) -> str:
    coupling = numeric(row.get("row_coupling"))
    pressure = numeric(row.get("row_self"))
    if coupling is None:
        coupling_class = "missing_coupling"
    elif abs(coupling) <= zero_tolerance:
        coupling_class = "zero_coupling"
    elif abs(coupling) <= weak_velocity_row_sum:
        coupling_class = "weak_coupling"
    else:
        coupling_class = "positive_coupling"

    if pressure is None:
        pressure_class = "missing_self"
    elif abs(pressure) <= zero_tolerance:
        pressure_class = "zero_self"
    elif abs(pressure) <= weak_pressure_row_sum:
        pressure_class = "weak_self"
    else:
        pressure_class = "positive_self"

    return f"{coupling_class}:{pressure_class}"


def top_update_rows(
    report: dict[str, Any],
    *,
    top_events: int,
    zero_tolerance: float,
    weak_velocity_row_sum: float,
    weak_pressure_row_sum: float,
) -> list[dict[str, Any]]:
    summary = report.get("pressure_update_support_summary")
    if not isinstance(summary, dict):
        return []
    rows = summary.get("top_update_details")
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows[:top_events], start=1):
        if not isinstance(row, dict):
            continue
        global_dof = row.get("global_dof")
        if not isinstance(global_dof, int):
            continue
        out.append(
            {
                "rank": rank,
                "global_dof": global_dof,
                "local_pressure_row": row.get("local_pressure_row"),
                "abs_update": numeric(row.get("abs_update")),
                "update": numeric(row.get("update")),
                "row_coupling": numeric(row.get("row_coupling")),
                "row_self": numeric(row.get("row_self")),
                "row_support_class": classify_row_support(
                    row,
                    zero_tolerance=zero_tolerance,
                    weak_velocity_row_sum=weak_velocity_row_sum,
                    weak_pressure_row_sum=weak_pressure_row_sum,
                ),
            }
        )
    return out


def operator_sample_sets(summary: dict[str, Any]) -> dict[str, set[int]]:
    return {
        "zero_coupling": set(parse_int_sample(summary.get("zero_coupling_row_global_dofs"))),
        "zero_row": set(parse_int_sample(summary.get("zero_row_global_dofs"))),
        "weakest_coupling": set(
            parse_int_sample(summary.get("weakest_coupling_row_global_dofs"))
        ),
        "weakest_self": set(parse_int_sample(summary.get("weakest_self_row_global_dofs"))),
    }


def value_dict(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    values = record.get("values")
    return values if isinstance(values, dict) else record


def support_rank_zero_coupling_dofs(report: dict[str, Any]) -> list[int]:
    values = value_dict(report.get("latest_support_rank_diagnostic"))
    return parse_int_sample(values.get("zero_coupling_row_global_dofs"))


def no_galerkin_support_summary(
    *,
    support_rank_zero_dofs: list[int],
    gal_zero_dofs: list[int],
    nonpressure_zero_dofs: list[int],
    gal_zero_top_hits: list[int],
    top_update_count: int,
) -> dict[str, Any]:
    support_rank_set = set(support_rank_zero_dofs)
    gal_set = set(gal_zero_dofs)
    nonpressure_set = set(nonpressure_zero_dofs)
    gal_equals_support_rank = gal_set == support_rank_set
    nonpressure_equals_support_rank = nonpressure_set == support_rank_set
    gal_equals_nonpressure = gal_set == nonpressure_set

    if not gal_set and not nonpressure_set:
        finding = "no_galerkin_zero_coupling_absent"
    elif (
        gal_equals_support_rank
        and nonpressure_equals_support_rank
        and len(gal_zero_top_hits) == 0
    ):
        finding = "no_galerkin_support_rank_equivalent_no_top_overlap"
    elif (
        gal_equals_support_rank
        and nonpressure_equals_support_rank
        and len(gal_zero_top_hits) == top_update_count
        and top_update_count > 0
    ):
        finding = "no_galerkin_support_rank_equivalent_all_top_updates"
    elif gal_equals_support_rank and nonpressure_equals_support_rank:
        finding = "no_galerkin_support_rank_equivalent_partial_top_overlap"
    elif gal_equals_nonpressure:
        finding = "no_galerkin_nonpressure_equivalent_support_rank_differs"
    else:
        finding = "no_galerkin_nonpressure_support_rank_selectors_differ"

    return {
        "no_galerkin_support_finding": finding,
        "no_galerkin_zero_coupling_global_dofs": gal_zero_dofs,
        "no_nonpressure_zero_coupling_global_dofs": nonpressure_zero_dofs,
        "support_rank_zero_coupling_global_dofs": support_rank_zero_dofs,
        "no_galerkin_equals_support_rank_zero_coupling": gal_equals_support_rank,
        "no_nonpressure_equals_support_rank_zero_coupling": (
            nonpressure_equals_support_rank
        ),
        "no_galerkin_equals_no_nonpressure_zero_coupling": gal_equals_nonpressure,
        "no_galerkin_top_update_overlap_count": len(gal_zero_top_hits),
        "no_galerkin_top_update_overlap_global_dofs": gal_zero_top_hits,
        "support_rank_minus_no_galerkin_global_dofs": sorted(
            support_rank_set - gal_set
        ),
        "no_galerkin_minus_support_rank_global_dofs": sorted(
            gal_set - support_rank_set
        ),
    }


def exact_operator_support_by_dof(
    report: dict[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    by_dof: dict[int, dict[str, dict[str, Any]]] = {}
    best_line: dict[tuple[int, str], int] = {}
    for row in report.get("pressure_row_operator_matrix_support_samples", []):
        if not isinstance(row, dict):
            continue
        op = row.get("op")
        support = row.get("operator_matrix_support")
        if not isinstance(op, str) or not isinstance(support, dict):
            continue
        dof = int_or_none(support.get("dof"))
        if dof is None:
            continue
        line = int_or_none(row.get("line_number")) or -1
        key = (dof, op)
        if key in best_line and line < best_line[key]:
            continue
        best_line[key] = line
        by_dof.setdefault(dof, {})[op] = support
    return by_dof


def exact_operator_support_summary(
    support: dict[str, Any] | None,
    *,
    zero_tolerance: float,
    weak_velocity_row_sum: float,
    weak_pressure_row_sum: float,
) -> dict[str, Any]:
    if not isinstance(support, dict):
        return {
            "status": "missing",
            "row_abs_sum": None,
            "row_coupling_abs_sum": None,
            "row_self_abs_sum": None,
            "diag": None,
            "row_abs_class": "missing",
            "row_coupling_class": "missing",
            "row_self_class": "missing",
        }
    row_abs = numeric(support.get("row_abs_sum"))
    return {
        "status": support.get("status", "ok"),
        "row_abs_sum": row_abs,
        "row_coupling_abs_sum": numeric(support.get("row_coupling_abs_sum")),
        "row_self_abs_sum": numeric(support.get("row_self_abs_sum")),
        "diag": numeric(support.get("diag")),
        "row_abs_class": support_class(
            row_abs,
            zero_tolerance=zero_tolerance,
            weak_threshold=weak_pressure_row_sum,
        ),
        "row_coupling_class": support_class(
            support.get("row_coupling_abs_sum"),
            zero_tolerance=zero_tolerance,
            weak_threshold=weak_velocity_row_sum,
        ),
        "row_self_class": support_class(
            support.get("row_self_abs_sum"),
            zero_tolerance=zero_tolerance,
            weak_threshold=weak_pressure_row_sum,
        ),
    }


def exact_physical_path_class(operator_support: dict[str, dict[str, Any]]) -> str:
    ghost = operator_support.get(OP_GHOST, {})
    direct = operator_support.get(OP_DIRECT_PGRAD, {})
    wall_normal = operator_support.get(OP_WALL_NORMAL_PGRAD, {})
    wall_tangential = operator_support.get(OP_WALL_TANGENTIAL_PGRAD, {})
    if ghost.get("row_self_class") == "positive":
        return "ghost_penalty_positive_self"
    if ghost.get("row_self_class") == "weak":
        return "ghost_penalty_weak_self"
    wall_classes = {
        wall_normal.get("row_self_class"),
        wall_tangential.get("row_self_class"),
    }
    if direct.get("row_self_class") in {"zero", "weak"} and (
        wall_classes & {"weak", "positive"}
    ):
        return "direct_pspg_weak_self_with_wall_support"
    if direct.get("row_self_class") in {"zero", "weak"}:
        return "direct_pspg_weak_self_no_wall_support"
    if direct.get("row_self_class") == "positive":
        return "direct_pspg_positive_self"
    return "operator_support_incomplete"


def operator_case_overlap(
    *,
    op: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sets = operator_sample_sets(summary)
    row_dofs = [int(row["global_dof"]) for row in rows]
    overlap: dict[str, Any] = {
        "op": op,
        "status": summary.get("status"),
        "unconstrained_pressure_rows": summary.get("unconstrained_pressure_rows"),
        "zero_coupling_row_block_count": summary.get(
            "zero_coupling_row_block_count"
        ),
        "weak_coupling_row_block_count": summary.get(
            "weak_coupling_row_block_count"
        ),
        "pressure_only_row_block_count": summary.get("pressure_only_row_block_count"),
        "weak_self_row_block_count": summary.get("weak_self_row_block_count"),
        "positive_coupling_row_block_count": summary.get(
            "positive_coupling_row_block_count"
        ),
        "positive_self_row_block_count": summary.get("positive_self_row_block_count"),
        "sample_limit": summary.get("sample_limit"),
    }
    for name, dofs in sets.items():
        hits = [dof for dof in row_dofs if dof in dofs]
        overlap[f"top_update_{name}_sample_hit_count"] = len(hits)
        overlap[f"top_update_{name}_sample_hit_global_dofs"] = hits
        overlap[f"{name}_sample_global_dofs"] = sorted(dofs)
    return overlap


def row_operator_memberships(
    row: dict[str, Any],
    operator_overlaps: dict[str, dict[str, Any]],
) -> dict[str, dict[str, bool]]:
    dof = int(row["global_dof"])
    memberships: dict[str, dict[str, bool]] = {}
    for op, overlap in operator_overlaps.items():
        memberships[op] = {
            "zero_coupling_sample": dof
            in set(overlap.get("zero_coupling_sample_global_dofs", [])),
            "zero_row_sample": dof
            in set(overlap.get("zero_row_sample_global_dofs", [])),
            "weakest_coupling_sample": dof
            in set(overlap.get("weakest_coupling_sample_global_dofs", [])),
            "weakest_self_sample": dof
            in set(overlap.get("weakest_self_sample_global_dofs", [])),
        }
    return memberships


def row_has_aggregate_sample(row: dict[str, Any], *, op: str | None = None) -> bool:
    memberships = row.get("operator_sample_membership")
    if not isinstance(memberships, dict):
        return False
    if op is not None:
        op_membership = memberships.get(op)
        if not isinstance(op_membership, dict):
            return False
        return any(bool(op_membership.get(name)) for name in AGGREGATE_SAMPLE_FLAGS)
    for op_membership in memberships.values():
        if not isinstance(op_membership, dict):
            continue
        if any(bool(op_membership.get(name)) for name in AGGREGATE_SAMPLE_FLAGS):
            return True
    return False


def ordered_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def exact_physical_path_dofs(rows: list[dict[str, Any]], classes: set[str]) -> list[int]:
    return ordered_unique(
        [
            int(row["global_dof"])
            for row in rows
            if row.get("exact_physical_path_class") in classes
        ]
    )


def exact_to_aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_sampled_rows = [
        row for row in rows if bool(row.get("exact_operator_sampled"))
    ]
    direct_rows = [
        row
        for row in rows
        if row.get("exact_physical_path_class") in DIRECT_PSPG_PATH_CLASSES
    ]
    ghost_rows = [
        row
        for row in rows
        if row.get("exact_physical_path_class") in GHOST_PENALTY_PATH_CLASSES
    ]
    direct_hits = [
        int(row["global_dof"])
        for row in direct_rows
        if row_has_aggregate_sample(row, op=OP_DIRECT_PGRAD)
    ]
    direct_misses = [
        int(row["global_dof"])
        for row in direct_rows
        if not row_has_aggregate_sample(row, op=OP_DIRECT_PGRAD)
    ]
    any_sample_misses = [
        int(row["global_dof"])
        for row in direct_rows
        if not row_has_aggregate_sample(row)
    ]
    if not exact_sampled_rows:
        finding = "exact_toprow_operator_samples_missing"
    elif direct_rows and direct_misses:
        finding = "exact_direct_pspg_rows_undercovered_by_aggregate_samples"
    elif direct_rows:
        finding = "exact_direct_pspg_rows_covered_by_aggregate_direct_samples"
    elif ghost_rows:
        finding = "exact_top_rows_are_ghost_penalty_path"
    else:
        finding = "exact_toprow_physical_path_incomplete"
    return {
        "finding": finding,
        "exact_operator_sampled_top_row_count": len(exact_sampled_rows),
        "exact_direct_pspg_top_update_count": len(direct_rows),
        "exact_direct_pspg_top_update_global_dofs": [
            int(row["global_dof"]) for row in direct_rows
        ],
        "exact_ghost_penalty_top_update_count": len(ghost_rows),
        "exact_ghost_penalty_top_update_global_dofs": [
            int(row["global_dof"]) for row in ghost_rows
        ],
        "exact_direct_pspg_rows_with_direct_pgrad_aggregate_sample_count": len(
            direct_hits
        ),
        "exact_direct_pspg_rows_with_direct_pgrad_aggregate_sample_global_dofs": (
            direct_hits
        ),
        "exact_direct_pspg_rows_missing_direct_pgrad_aggregate_sample_count": len(
            direct_misses
        ),
        "exact_direct_pspg_rows_missing_direct_pgrad_aggregate_sample_global_dofs": (
            direct_misses
        ),
        "exact_direct_pspg_rows_missing_any_aggregate_sample_count": len(
            any_sample_misses
        ),
        "exact_direct_pspg_rows_missing_any_aggregate_sample_global_dofs": (
            any_sample_misses
        ),
    }


def audit_case(
    label: str,
    path: Path,
    report: dict[str, Any],
    *,
    operators: tuple[str, ...],
    top_events: int,
    zero_tolerance: float,
    weak_velocity_row_sum: float,
    weak_pressure_row_sum: float,
) -> dict[str, Any]:
    rows = top_update_rows(
        report,
        top_events=top_events,
        zero_tolerance=zero_tolerance,
        weak_velocity_row_sum=weak_velocity_row_sum,
        weak_pressure_row_sum=weak_pressure_row_sum,
    )
    summaries = report.get("operator_matrix_summary_by_op")
    if not isinstance(summaries, dict):
        summaries = {}
    exact_support_by_dof = exact_operator_support_by_dof(report)

    overlaps: dict[str, dict[str, Any]] = {}
    for op in operators:
        summary = summaries.get(op)
        if isinstance(summary, dict):
            overlaps[op] = operator_case_overlap(op=op, summary=summary, rows=rows)
        else:
            overlaps[op] = {"op": op, "status": "missing_operator_summary"}

    for row in rows:
        row["operator_sample_membership"] = row_operator_memberships(row, overlaps)
        row["direct_pgrad_aggregate_any_sample"] = row_has_aggregate_sample(
            row, op=OP_DIRECT_PGRAD
        )
        row["any_operator_aggregate_sample"] = row_has_aggregate_sample(row)
        dof = int(row["global_dof"])
        exact_operator_support = {
            op: exact_operator_support_summary(
                exact_support_by_dof.get(dof, {}).get(op),
                zero_tolerance=zero_tolerance,
                weak_velocity_row_sum=weak_velocity_row_sum,
                weak_pressure_row_sum=weak_pressure_row_sum,
            )
            for op in EXACT_PHYSICAL_OPERATORS
        }
        row["exact_operator_sampled"] = dof in exact_support_by_dof
        row["exact_operator_support"] = exact_operator_support
        row["exact_physical_path_class"] = exact_physical_path_class(
            exact_operator_support
        )

    row_class_counts = Counter(row["row_support_class"] for row in rows)
    gal_zero_hits = overlaps.get(OP_GALERKIN, {}).get(
        "top_update_zero_coupling_sample_hit_count", 0
    )
    gal_zero_hit_dofs = overlaps.get(OP_GALERKIN, {}).get(
        "top_update_zero_coupling_sample_hit_global_dofs", []
    )
    if gal_zero_hits == 0:
        finding = "no_top_update_overlap_no_galerkin_zero_coupling_sample"
    elif gal_zero_hits == len(rows):
        finding = "all_top_updates_overlap_no_galerkin_zero_coupling_sample"
    else:
        finding = "partial_top_update_overlap_no_galerkin_zero_coupling_sample"

    exact_summary = exact_to_aggregate_summary(rows)
    exact_physical_counts = Counter(
        str(row.get("exact_physical_path_class")) for row in rows
    )
    aggregate_direct_hits = [
        int(row["global_dof"])
        for row in rows
        if row_has_aggregate_sample(row, op=OP_DIRECT_PGRAD)
    ]
    aggregate_any_hits = [
        int(row["global_dof"]) for row in rows if row_has_aggregate_sample(row)
    ]
    no_galerkin_summary = no_galerkin_support_summary(
        support_rank_zero_dofs=support_rank_zero_coupling_dofs(report),
        gal_zero_dofs=sorted(
            overlaps.get(OP_GALERKIN, {}).get(
                "zero_coupling_sample_global_dofs", []
            )
        ),
        nonpressure_zero_dofs=sorted(
            overlaps.get(OP_NONPRESSURE, {}).get(
                "zero_coupling_sample_global_dofs", []
            )
        ),
        gal_zero_top_hits=[
            int(dof)
            for dof in gal_zero_hit_dofs
            if isinstance(dof, int)
        ],
        top_update_count=len(rows),
    )

    return {
        "label": label,
        "support_json": str(path),
        "finding": finding,
        "limitation": (
            "Operator row overlap uses bounded row samples emitted by the "
            "operator-summary diagnostic; non-overlap with a sampled list is "
            "not proof that a row is absent from the full operator class."
        ),
        "top_update_count": len(rows),
        "row_support_class_counts": dict(sorted(row_class_counts.items())),
        "exact_to_aggregate_sample_finding": exact_summary["finding"],
        "exact_operator_sampled_top_row_count": exact_summary[
            "exact_operator_sampled_top_row_count"
        ],
        "exact_physical_path_class_counts": dict(
            sorted(exact_physical_counts.items())
        ),
        "aggregate_direct_pgrad_any_sample_hit_count": len(aggregate_direct_hits),
        "aggregate_direct_pgrad_any_sample_hit_global_dofs": aggregate_direct_hits,
        "aggregate_any_operator_sample_hit_count": len(aggregate_any_hits),
        "aggregate_any_operator_sample_hit_global_dofs": aggregate_any_hits,
        **no_galerkin_summary,
        **{
            key: value
            for key, value in exact_summary.items()
            if key != "finding"
            and key != "exact_operator_sampled_top_row_count"
        },
        "operator_overlaps": overlaps,
        "top_update_rows": rows,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    finding_counts = Counter(str(case.get("finding")) for case in cases)
    no_galerkin_support_finding_counts = Counter(
        str(case.get("no_galerkin_support_finding")) for case in cases
    )
    partial = sum(
        1
        for case in cases
        if case.get("finding")
        == "partial_top_update_overlap_no_galerkin_zero_coupling_sample"
    )
    none = sum(
        1
        for case in cases
        if case.get("finding")
        == "no_top_update_overlap_no_galerkin_zero_coupling_sample"
    )
    all_hits = sum(
        1
        for case in cases
        if case.get("finding")
        == "all_top_updates_overlap_no_galerkin_zero_coupling_sample"
    )
    if partial and none:
        finding = "mixed_no_galerkin_overlap_partial_for_some_cases_absent_for_others"
    elif partial:
        finding = "no_galerkin_overlap_partial"
    elif all_hits and not none:
        finding = "no_galerkin_overlap_all_sampled_top_rows"
    else:
        finding = "no_galerkin_overlap_absent"
    exact_finding_counts = Counter(
        str(case.get("exact_to_aggregate_sample_finding")) for case in cases
    )
    exact_undercovers = sum(
        1
        for case in cases
        if case.get("exact_to_aggregate_sample_finding")
        == "exact_direct_pspg_rows_undercovered_by_aggregate_samples"
    )
    if exact_undercovers:
        exact_finding = "exact_direct_pspg_top_rows_undercovered_by_aggregate_samples"
    elif any(
        case.get("exact_to_aggregate_sample_finding")
        == "exact_toprow_operator_samples_missing"
        for case in cases
    ):
        exact_finding = "exact_toprow_operator_samples_missing"
    else:
        exact_finding = "exact_toprow_operator_samples_cover_tested_paths"
    if any(
        case.get("no_galerkin_support_finding")
        == "no_galerkin_support_rank_equivalent_partial_top_overlap"
        for case in cases
    ):
        no_galerkin_support_finding = (
            "no_galerkin_support_rank_equivalent_but_partial_top_overlap"
        )
    elif any(
        case.get("no_galerkin_support_finding")
        == "no_galerkin_support_rank_equivalent_all_top_updates"
        for case in cases
    ):
        no_galerkin_support_finding = (
            "no_galerkin_support_rank_equivalent_all_sampled_top_rows"
        )
    elif all(
        case.get("no_galerkin_support_finding")
        == "no_galerkin_zero_coupling_absent"
        for case in cases
    ):
        no_galerkin_support_finding = "no_galerkin_zero_coupling_absent"
    elif all(
        str(case.get("no_galerkin_support_finding")).startswith(
            "no_galerkin_support_rank_equivalent"
        )
        or case.get("no_galerkin_support_finding")
        == "no_galerkin_zero_coupling_absent"
        for case in cases
    ):
        no_galerkin_support_finding = (
            "no_galerkin_support_rank_equivalent_or_absent"
        )
    else:
        no_galerkin_support_finding = (
            "no_galerkin_support_rank_selector_differs_in_some_cases"
        )
    return {
        "finding": finding,
        "exact_to_aggregate_sample_finding": exact_finding,
        "no_galerkin_support_finding": no_galerkin_support_finding,
        "case_count": len(cases),
        "finding_counts": dict(sorted(finding_counts.items())),
        "no_galerkin_support_finding_counts": dict(
            sorted(no_galerkin_support_finding_counts.items())
        ),
        "exact_to_aggregate_sample_finding_counts": dict(
            sorted(exact_finding_counts.items())
        ),
        "cases": cases,
    }


def build_report(
    labeled_reports: list[tuple[str, Path, dict[str, Any]]],
    *,
    operators: tuple[str, ...] = DEFAULT_OPERATORS,
    top_events: int = 12,
    zero_tolerance: float = 1.0e-14,
    weak_velocity_row_sum: float = 1.0e-3,
    weak_pressure_row_sum: float = 1.0e-7,
) -> dict[str, Any]:
    cases = [
        audit_case(
            label,
            path,
            report,
            operators=operators,
            top_events=top_events,
            zero_tolerance=zero_tolerance,
            weak_velocity_row_sum=weak_velocity_row_sum,
            weak_pressure_row_sum=weak_pressure_row_sum,
        )
        for label, path, report in labeled_reports
    ]
    return summarize_cases(cases)


def main() -> int:
    args = parse_args()
    operators = tuple(args.operator) if args.operator else DEFAULT_OPERATORS
    labeled_reports = [
        (label, path, load_json(path))
        for label, path in (parse_labeled_path(value) for value in args.support_json)
    ]
    report = build_report(
        labeled_reports,
        operators=operators,
        top_events=args.top_events,
        zero_tolerance=args.zero_tolerance,
        weak_velocity_row_sum=args.weak_velocity_row_sum,
        weak_pressure_row_sum=args.weak_pressure_row_sum,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
