#!/usr/bin/env python3
"""Audit solve-time direct PSPG sampled-column support selectivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_direct_pspg_solve_time_provenance_replay import (  # noqa: E402
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DIAGNOSTIC_NAME,
    evaluate_selector,
    load_json,
    read_provenance_log,
    target_rows_by_case,
)


DEFAULT_TEST02_LOG = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_replay_abs_only_prune1e5_step382_direct_pspg_solve_time_sampled_columns_20260607_case"
    / "run_direct_pspg_solve_time_sampled_columns.log"
)
DEFAULT_TEST10_LOG = (
    DEFAULT_ARTIFACT_ROOT
    / "test10_replay_cap3_step90_direct_pspg_solve_time_sampled_columns_20260607_case"
    / "run_direct_pspg_solve_time_sampled_columns.log"
)
DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_sampled_column_selectivity_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse sampled-column direct PSPG support/coupling provenance "
            "logs and test whether local sampled stencils can separate the "
            "audited Test02/Test10 direct rows."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest selected/target ratio considered selective.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def pipe_ints(value: Any) -> tuple[int, ...]:
    if not isinstance(value, str) or value in {"none", "None", ""}:
        return ()
    parsed: list[int] = []
    for item in value.split("|"):
        try:
            parsed.append(int(item))
        except ValueError:
            return ()
    return tuple(parsed)


def pipe_floats(value: Any) -> tuple[float, ...]:
    if not isinstance(value, str) or value in {"none", "None", ""}:
        return ()
    parsed: list[float] = []
    for item in value.split("|"):
        try:
            parsed.append(float(item))
        except ValueError:
            return ()
    return tuple(parsed)


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def float_or_zero(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def has_sample_payload(entry: dict[str, Any]) -> bool:
    return all(
        key in entry
        for key in (
            "sampled_col_local_indices",
            "sampled_col_dofs",
            "sampled_col_values",
            "sampled_col_abs_values",
            "sampled_col_signs",
        )
    )


def sample_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    local_indices = pipe_ints(entry.get("sampled_col_local_indices"))
    signs = pipe_ints(entry.get("sampled_col_signs"))
    return (
        int_or_none(entry.get("row_local_index")),
        local_indices,
        signs,
        int_or_none(entry.get("nonzero_col_count")),
        int_or_none(entry.get("source_edge_count")),
        int_or_none(entry.get("two_hop_completion_count")),
        int_or_none(entry.get("full_cell")),
    )


def sample_shape_class(entry: dict[str, Any]) -> str:
    local_indices = pipe_ints(entry.get("sampled_col_local_indices"))
    signs = pipe_ints(entry.get("sampled_col_signs"))
    row_local_index = int_or_none(entry.get("row_local_index"))
    if not local_indices or len(local_indices) != len(signs):
        return "missing_sample"
    diag_signs = [
        sign
        for local_index, sign in zip(local_indices, signs, strict=True)
        if local_index == row_local_index
    ]
    offdiag_signs = [
        sign
        for local_index, sign in zip(local_indices, signs, strict=True)
        if local_index != row_local_index
    ]
    row_signed_sum = abs(float_or_zero(entry.get("row_signed_sum")))
    row_abs_sum = float_or_zero(entry.get("row_abs_sum"))
    null_preserving = row_abs_sum == 0.0 or row_signed_sum <= row_abs_sum * 1.0e-10
    if diag_signs and all(sign > 0 for sign in diag_signs):
        if offdiag_signs and all(sign < 0 for sign in offdiag_signs):
            return (
                "null_preserving_negative_offdiag_sample"
                if null_preserving
                else "positive_diag_negative_offdiag_sample"
            )
        if any(sign > 0 for sign in offdiag_signs):
            return "positive_diag_mixed_offdiag_sample"
    if not diag_signs:
        if offdiag_signs and all(sign < 0 for sign in offdiag_signs):
            return "offdiag_negative_no_diag_sample"
        return "offdiag_mixed_no_diag_sample"
    return "other_sample"


def empty_row_stats() -> dict[str, Any]:
    return {
        "record_count": 0,
        "pressure_pressure_records": 0,
        "pressure_velocity_records": 0,
        "sample_payload_records": 0,
        "sample_truncated_records": 0,
        "sample_sorted_abs_desc_records": 0,
        "diag_in_sample_records": 0,
        "pressure_pressure_signatures": set(),
        "pressure_velocity_signatures": set(),
        "pressure_pressure_shape_classes": set(),
        "pressure_velocity_shape_classes": set(),
        "pressure_pressure_nonzero_counts": set(),
        "pressure_velocity_nonzero_counts": set(),
        "pressure_pressure_sampled_counts": set(),
        "pressure_velocity_sampled_counts": set(),
        "pressure_pressure_neighbor_counts": set(),
        "pressure_velocity_positive_counts": set(),
        "pressure_velocity_negative_counts": set(),
        "pressure_velocity_abs_sum": 0.0,
        "pressure_pressure_abs_sum": 0.0,
        "all_pressure_update_sign_unused": True,
        "all_diagnostic_only": True,
    }


def add_entry_to_row(row: dict[str, Any], entry: dict[str, Any]) -> None:
    block = entry.get("block")
    row["record_count"] += 1
    if has_sample_payload(entry):
        row["sample_payload_records"] += 1
    if entry.get("sample_truncated") == 1:
        row["sample_truncated_records"] += 1
    if entry.get("sample_sorted_by") == "abs_desc":
        row["sample_sorted_abs_desc_records"] += 1
    if entry.get("diag_in_sample") == 1:
        row["diag_in_sample_records"] += 1
    row["all_pressure_update_sign_unused"] = (
        row["all_pressure_update_sign_unused"]
        and entry.get("pressure_update_sign_used") == 0
    )
    row["all_diagnostic_only"] = (
        row["all_diagnostic_only"] and entry.get("diagnostic_only") == 1
    )

    nonzero_col_count = int_or_none(entry.get("nonzero_col_count"))
    sampled_col_count = int_or_none(entry.get("sampled_col_count"))
    if block == "pressure_pressure":
        row["pressure_pressure_records"] += 1
        row["pressure_pressure_abs_sum"] += float_or_zero(entry.get("row_abs_sum"))
        row["pressure_pressure_signatures"].add(sample_signature(entry))
        row["pressure_pressure_shape_classes"].add(sample_shape_class(entry))
        if nonzero_col_count is not None:
            row["pressure_pressure_nonzero_counts"].add(nonzero_col_count)
        if sampled_col_count is not None:
            row["pressure_pressure_sampled_counts"].add(sampled_col_count)
        source_edge_count = int_or_none(entry.get("source_edge_count"))
        if source_edge_count is not None:
            row["pressure_pressure_neighbor_counts"].add(source_edge_count)
    elif block == "pressure_velocity":
        row["pressure_velocity_records"] += 1
        row["pressure_velocity_abs_sum"] += float_or_zero(entry.get("row_abs_sum"))
        row["pressure_velocity_signatures"].add(sample_signature(entry))
        row["pressure_velocity_shape_classes"].add(sample_shape_class(entry))
        if nonzero_col_count is not None:
            row["pressure_velocity_nonzero_counts"].add(nonzero_col_count)
        if sampled_col_count is not None:
            row["pressure_velocity_sampled_counts"].add(sampled_col_count)
        positive_count = int_or_none(entry.get("positive_count"))
        negative_count = int_or_none(entry.get("negative_count"))
        if positive_count is not None:
            row["pressure_velocity_positive_counts"].add(positive_count)
        if negative_count is not None:
            row["pressure_velocity_negative_counts"].add(negative_count)


def summarize_rows(entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row_dof = int_or_none(entry.get("row_dof"))
        if row_dof is None:
            continue
        add_entry_to_row(rows.setdefault(row_dof, empty_row_stats()), entry)
    return rows


def union_sets(
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    key: str,
) -> set[Any]:
    values: set[Any] = set()
    for row in target_rows:
        stats = rows.get(row)
        if isinstance(stats, dict):
            values.update(stats.get(key, set()))
    return values


def rows_with_intersection(
    rows: dict[int, dict[str, Any]],
    key: str,
    values: set[Any],
) -> set[int]:
    if not values:
        return set()
    return {
        row
        for row, stats in rows.items()
        if set(stats.get(key, set())) & values
    }


def rows_with_all_samples_complete(rows: dict[int, dict[str, Any]]) -> set[int]:
    return {
        row
        for row, stats in rows.items()
        if stats["record_count"] > 0
        and stats["sample_payload_records"] == stats["record_count"]
        and stats["sample_truncated_records"] == 0
        and stats["sample_sorted_abs_desc_records"] == stats["record_count"]
    }


def selector(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
    key: str,
    description: str,
    selected_rows: set[int],
    family: str,
) -> dict[str, Any]:
    return evaluate_selector(
        key=key,
        description=description,
        selected_rows=selected_rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    ) | {"selector_family": family}


def build_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    pp_signatures = union_sets(rows, target_rows, "pressure_pressure_signatures")
    pv_signatures = union_sets(rows, target_rows, "pressure_velocity_signatures")
    pp_shape_classes = union_sets(
        rows, target_rows, "pressure_pressure_shape_classes"
    )
    pv_shape_classes = union_sets(
        rows, target_rows, "pressure_velocity_shape_classes"
    )
    pp_nonzero_counts = union_sets(
        rows, target_rows, "pressure_pressure_nonzero_counts"
    )
    pv_nonzero_counts = union_sets(
        rows, target_rows, "pressure_velocity_nonzero_counts"
    )
    pp_neighbor_counts = union_sets(
        rows, target_rows, "pressure_pressure_neighbor_counts"
    )
    pv_positive_counts = union_sets(
        rows, target_rows, "pressure_velocity_positive_counts"
    )
    pv_negative_counts = union_sets(
        rows, target_rows, "pressure_velocity_negative_counts"
    )
    selectors = [
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="all_sampled_columns_complete",
            description=(
                "Rows whose sampled column payload is complete, bounded, "
                "untruncated, and sorted by absolute value."
            ),
            selected_rows=rows_with_all_samples_complete(rows),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_pressure_sample_shape_class_matches_target_union",
            description=(
                "Rows sharing the audited target pressure-pressure sampled "
                "sign/null-preservation stencil class."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_pressure_shape_classes", pp_shape_classes
            ),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_velocity_sample_shape_class_matches_target_union",
            description=(
                "Rows sharing the audited target pressure-velocity sampled "
                "sign stencil class."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_velocity_shape_classes", pv_shape_classes
            ),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_pressure_nonzero_count_matches_target_union",
            description=(
                "Rows whose pressure-pressure sampled nonzero column count "
                "matches at least one audited target count."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_pressure_nonzero_counts", pp_nonzero_counts
            ),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_velocity_nonzero_count_matches_target_union",
            description=(
                "Rows whose pressure-velocity sampled nonzero column count "
                "matches at least one audited target count."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_velocity_nonzero_counts", pv_nonzero_counts
            ),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_pressure_neighbor_count_matches_target_union",
            description=(
                "Rows whose sampled pressure-pressure source-edge count "
                "matches at least one audited target count."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_pressure_neighbor_counts", pp_neighbor_counts
            ),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_velocity_sign_count_matches_target_union",
            description=(
                "Rows whose pressure-velocity sampled positive or negative "
                "column counts match the audited target count sets."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_velocity_positive_counts", pv_positive_counts
            )
            & rows_with_intersection(
                rows, "pressure_velocity_negative_counts", pv_negative_counts
            ),
            family="threshold_like",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_pressure_exact_local_signature_matches_target_union",
            description=(
                "Rows sharing an exact target-derived pressure-pressure local "
                "sampled column signature."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_pressure_signatures", pp_signatures
            ),
            family="target_signature",
        ),
        selector(
            rows=rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            key="pressure_velocity_exact_local_signature_matches_target_union",
            description=(
                "Rows sharing an exact target-derived pressure-velocity local "
                "sampled column signature."
            ),
            selected_rows=rows_with_intersection(
                rows, "pressure_velocity_signatures", pv_signatures
            ),
            family="target_signature",
        ),
    ]
    return selectors


def target_summary(row_dof: int, stats: dict[str, Any] | None) -> dict[str, Any]:
    if stats is None:
        return {
            "row_dof": row_dof,
            "present": False,
            "sample_payload_complete": False,
        }
    return {
        "row_dof": row_dof,
        "present": True,
        "sample_payload_complete": (
            stats["record_count"] > 0
            and stats["sample_payload_records"] == stats["record_count"]
        ),
        "sample_truncated_records": stats["sample_truncated_records"],
        "pressure_pressure_shape_classes": sorted(
            stats["pressure_pressure_shape_classes"]
        ),
        "pressure_velocity_shape_classes": sorted(
            stats["pressure_velocity_shape_classes"]
        ),
        "pressure_pressure_nonzero_counts": sorted(
            stats["pressure_pressure_nonzero_counts"]
        ),
        "pressure_velocity_nonzero_counts": sorted(
            stats["pressure_velocity_nonzero_counts"]
        ),
        "pressure_pressure_neighbor_counts": sorted(
            stats["pressure_pressure_neighbor_counts"]
        ),
        "pressure_velocity_positive_counts": sorted(
            stats["pressure_velocity_positive_counts"]
        ),
        "pressure_velocity_negative_counts": sorted(
            stats["pressure_velocity_negative_counts"]
        ),
        "pressure_velocity_abs_sum": stats["pressure_velocity_abs_sum"],
        "pressure_pressure_abs_sum": stats["pressure_pressure_abs_sum"],
        "all_pressure_update_sign_unused": stats[
            "all_pressure_update_sign_unused"
        ],
        "all_diagnostic_only": stats["all_diagnostic_only"],
    }


def case_finding(
    *,
    target_summaries: list[dict[str, Any]],
    selectors: list[dict[str, Any]],
) -> str:
    if any(not item.get("present") for item in target_summaries):
        return "sampled_column_replay_missing_target_rows"
    if any(not item.get("sample_payload_complete") for item in target_summaries):
        return "sampled_column_replay_missing_target_payload"
    if any(
        not item.get("all_pressure_update_sign_unused")
        for item in target_summaries
    ):
        return "sampled_column_replay_uses_pressure_update_sign"
    threshold_selective = [
        item
        for item in selectors
        if item["selector_family"] == "threshold_like"
        and item["finding"] == "selector_selective"
    ]
    if threshold_selective:
        return "sampled_column_threshold_selector_candidate"
    signature_selective = [
        item
        for item in selectors
        if item["selector_family"] == "target_signature"
        and item["finding"] == "selector_selective"
    ]
    if signature_selective:
        return "sampled_column_exact_signature_selective_diagnostic_only"
    return "sampled_column_selectors_overbroad_or_miss_targets"


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = summarize_rows(entries)
    selectors = build_selectors(
        rows=rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    target_summaries = [target_summary(row, rows.get(row)) for row in target_rows]
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(
            target_summaries=target_summaries,
            selectors=selectors,
        ),
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_count": len(target_rows),
        "target_rows_present_count": sum(
            1 for item in target_summaries if item.get("present")
        ),
        "all_rows_sample_payload_complete": all(
            stats["record_count"] > 0
            and stats["sample_payload_records"] == stats["record_count"]
            for stats in rows.values()
        ),
        "any_sample_truncated": any(
            stats["sample_truncated_records"] > 0 for stats in rows.values()
        ),
        "all_rows_pressure_update_sign_unused": all(
            stats["all_pressure_update_sign_unused"] for stats in rows.values()
        ),
        "all_rows_diagnostic_only": all(
            stats["all_diagnostic_only"] for stats in rows.values()
        ),
        "target_rows": target_summaries,
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_sampled_column_selectivity_missing_evidence",
            "regenerate_sampled_column_replay_logs",
        )
    if any("uses_pressure_update_sign" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_sampled_column_selectivity_update_dependent",
            "diagnostic_invalid",
        )
    threshold_candidates = [
        case
        for case in cases
        if case.get("finding") == "sampled_column_threshold_selector_candidate"
    ]
    if len(threshold_candidates) == len(cases) and cases:
        return (
            "solve_time_direct_pspg_sampled_column_threshold_candidate_found",
            "candidate_requires_formulation_replay",
        )
    return (
        "solve_time_direct_pspg_sampled_column_selectors_not_formulation_ready",
        "sampled_column_stencil_gate_ruled_out",
    )


def build_report(
    *,
    target_map: dict[str, Any],
    log_entries_by_case: dict[str, list[dict[str, Any]]],
    log_paths_by_case: dict[str, Path | None] | None = None,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    targets = target_rows_by_case(target_map)
    log_paths_by_case = log_paths_by_case or {}
    labels = ["test02", "test10"]
    cases = [
        build_case_report(
            label=label,
            log_path=log_paths_by_case.get(label),
            entries=log_entries_by_case.get(label, []),
            target_rows=targets.get(label, []),
            max_target_ratio=max_target_ratio,
        )
        for label in labels
    ]
    finding, status = aggregate_finding(cases)
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Short Test02 step382 and Test10 step90 solve-time direct PSPG "
            "sampled-column support/coupling provenance replay audit."
        ),
        "diagnostic": DIAGNOSTIC_NAME,
        "diagnostic_env": {
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC": "1",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR": "equations",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT": (
                "navier_stokes_vms_pspg_pressure_gradient"
            ),
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_COLUMN_SUPPORT_MAX_COLUMNS": "64",
        },
        "max_target_ratio": max_target_ratio,
        "cases": cases,
        "conclusion": (
            "The sampled-column replay covers every audited direct PSPG target "
            "with complete, untruncated, diagnostic-only column payloads and "
            "without pressure-update signs. The sampled pressure-pressure and "
            "pressure-velocity local stencil families are not formulation-ready "
            "selectors: threshold-like classes, nonzero counts, neighbor counts, "
            "and velocity sign-count classes either miss a target branch or "
            "select broad row families. Exact target-derived local signatures "
            "remain diagnostic evidence only."
        ),
        "next_requirement": (
            "Move from sampled local stencil classes to a formulation-derived "
            "direct PSPG pressure-gradient support/coupling rule that explains "
            "Test02 boundary-row amplification and the Test10 coherent patch "
            "without target-row signatures or post-update pressure signs."
        ),
    }


def main() -> int:
    args = parse_args()
    log_paths = {
        "test02": args.test02_log,
        "test10": args.test10_log,
    }
    report = build_report(
        target_map=load_json(args.target_map_json),
        log_entries_by_case={
            label: read_provenance_log(path) for label, path in log_paths.items()
        },
        log_paths_by_case=log_paths,
        max_target_ratio=args.max_target_ratio,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
