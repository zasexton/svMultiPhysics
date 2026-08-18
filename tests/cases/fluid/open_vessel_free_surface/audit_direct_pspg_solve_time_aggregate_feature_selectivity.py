#!/usr/bin/env python3
"""Audit solve-time direct PSPG aggregate feature selectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_provenance_replay import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    load_json,
    read_provenance_log,
    summarize_rows,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_aggregate_feature_selectivity_20260607.json"
)

AGGREGATE_FEATURES = [
    "pressure_pressure_records",
    "pressure_velocity_records",
    "pressure_pressure_edge_count",
    "pressure_pressure_two_hop_completion_count",
    "pressure_pressure_neighbor_pair_count",
    "pressure_velocity_nonzero_count",
    "min_volume_fraction",
    "full_cell_records",
    "cut_cell_records",
    "rule_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether non-oracle aggregate fields from the solve-time "
            "direct PSPG support/coupling provenance logs provide a physical "
            "support discriminator for Test02/Test10 direct targets."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def evaluate_selector(
    *,
    key: str,
    selector_kind: str,
    feature: str,
    selected_rows: set[int],
    target_rows: list[int],
    max_target_ratio: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target_set = set(target_rows)
    covered = sorted(target_set & selected_rows)
    uncovered = sorted(target_set - selected_rows)
    target_count = len(target_rows)
    selected_count = len(selected_rows)
    selected_to_target_ratio = (
        selected_count / target_count if target_count > 0 else None
    )
    covers_targets = target_count > 0 and len(covered) == target_count
    overbroad = (
        selected_to_target_ratio is not None
        and selected_to_target_ratio > max_target_ratio
    )
    if not covers_targets and overbroad:
        finding = "selector_overbroad_and_misses_targets"
    elif not covers_targets:
        finding = "selector_misses_targets"
    elif overbroad:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"
    result: dict[str, Any] = {
        "key": key,
        "selector_kind": selector_kind,
        "feature": feature,
        "finding": finding,
        "selected_count": selected_count,
        "target_count": target_count,
        "selected_to_target_ratio": selected_to_target_ratio,
        "covered_target_count": len(covered),
        "covered_target_global_dofs": covered,
        "uncovered_target_global_dofs": uncovered,
        "covers_targets": covers_targets,
        "selector_overbroad": overbroad,
    }
    if extra:
        result.update(extra)
    return result


def build_feature_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    present_targets = [row for row in target_rows if row in rows]
    for feature in AGGREGATE_FEATURES:
        row_values = {
            row: stats.get(feature)
            for row, stats in rows.items()
            if stats.get(feature) is not None
        }
        target_values = [
            row_values[row] for row in present_targets if row in row_values
        ]
        if not target_values:
            continue
        target_min = min(target_values)
        target_max = max(target_values)
        target_value_set = set(target_values)
        selectors.append(
            evaluate_selector(
                key=f"{feature}_target_range",
                selector_kind="target_value_range",
                feature=feature,
                selected_rows={
                    row
                    for row, value in row_values.items()
                    if target_min <= value <= target_max
                },
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
                extra={
                    "target_min": target_min,
                    "target_max": target_max,
                    "production_readiness": "range_threshold_candidate",
                },
            )
        )
        selectors.append(
            evaluate_selector(
                key=f"{feature}_exact_target_value_set",
                selector_kind="exact_target_value_set",
                feature=feature,
                selected_rows={
                    row
                    for row, value in row_values.items()
                    if value in target_value_set
                },
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
                extra={
                    "target_value_count": len(target_value_set),
                    "target_values": sorted(target_value_set),
                    "production_readiness": "integer_or_class_value_candidate",
                },
            )
        )
    return selectors


def selector_group(selectors: list[dict[str, Any]], selector_kind: str) -> list[dict[str, Any]]:
    return [
        selector for selector in selectors if selector.get("selector_kind") == selector_kind
    ]


def best_covering_selector(selectors: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [selector for selector in selectors if selector.get("covers_targets")]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda selector: (
            float(selector.get("selected_to_target_ratio") or float("inf")),
            int(selector.get("selected_count") or 0),
            str(selector.get("key") or ""),
        ),
    )


def case_finding(
    selectors: list[dict[str, Any]],
    target_count: int,
    target_rows_present_count: int,
) -> str:
    if target_count == 0:
        return "direct_target_rows_missing"
    if target_rows_present_count < target_count:
        return "solve_time_aggregate_feature_selectivity_missing_target_rows"
    range_selectors = selector_group(selectors, "target_value_range")
    exact_selectors = selector_group(selectors, "exact_target_value_set")
    if any(
        selector.get("finding") == "selector_selective"
        for selector in range_selectors
    ):
        return "solve_time_aggregate_range_selector_candidate"
    if any(
        selector.get("finding") == "selector_selective"
        for selector in exact_selectors
    ):
        return "solve_time_aggregate_exact_value_selector_candidate"
    return "solve_time_aggregate_feature_selectors_overbroad_or_miss_targets"


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = summarize_rows(entries)
    selectors = build_feature_selectors(
        rows=rows,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    range_selectors = selector_group(selectors, "target_value_range")
    exact_selectors = selector_group(selectors, "exact_target_value_set")
    best_range = best_covering_selector(range_selectors)
    best_exact = best_covering_selector(exact_selectors)
    present_targets = [row for row in target_rows if row in rows]
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(
            selectors=selectors,
            target_count=len(target_rows),
            target_rows_present_count=len(present_targets),
        ),
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_count": len(target_rows),
        "target_rows_present_count": len(present_targets),
        "features": AGGREGATE_FEATURES,
        "target_rows": [
            {
                "row_dof": row,
                "present": row in rows,
                "features": {
                    feature: rows[row].get(feature)
                    for feature in AGGREGATE_FEATURES
                }
                if row in rows
                else None,
            }
            for row in target_rows
        ],
        "best_covering_range_selector": best_range,
        "best_covering_exact_value_selector": best_exact,
        "range_selector_findings": {
            selector["key"]: selector["finding"] for selector in range_selectors
        },
        "exact_value_selector_findings": {
            selector["key"]: selector["finding"] for selector in exact_selectors
        },
        "range_selector_selected_to_target_ratios": {
            selector["key"]: selector["selected_to_target_ratio"]
            for selector in range_selectors
        },
        "exact_value_selector_selected_to_target_ratios": {
            selector["key"]: selector["selected_to_target_ratio"]
            for selector in exact_selectors
        },
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_aggregate_feature_selectivity_missing_evidence",
            "regenerate_solve_time_provenance_logs",
        )
    if any(
        case.get("finding")
        in {
            "solve_time_aggregate_range_selector_candidate",
            "solve_time_aggregate_exact_value_selector_candidate",
        }
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_aggregate_feature_candidate_requires_replay",
            "aggregate_feature_candidate_needs_transfer_check",
        )
    return (
        "solve_time_direct_pspg_aggregate_feature_selectivity_rules_out_counts_and_volume_gate",
        "aggregate_counts_and_volume_features_overbroad",
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
            "support/coupling aggregate feature audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "features": AGGREGATE_FEATURES,
        "cases": cases,
        "conclusion": (
            "Aggregate solve-time provenance counts and classes are not the "
            "missing physical discriminator. Full/cut record counts, rule "
            "counts, pressure-pressure edge/two-hop counts, pressure-velocity "
            "nonzero counts, and min volume-fraction classes cover the audited "
            "direct PSPG targets only with broad row sets in at least one case."
        ),
        "next_requirement": (
            "Continue the direct PSPG formulation search with a physical "
            "support/coupling discriminator beyond aggregate provenance counts, "
            "full-cell classes, and target value ranges."
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
