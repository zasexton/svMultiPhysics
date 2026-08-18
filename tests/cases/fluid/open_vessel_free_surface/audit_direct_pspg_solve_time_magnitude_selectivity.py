#!/usr/bin/env python3
"""Audit solve-time direct PSPG support/coupling magnitude selectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_support_coupling_signature import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    read_provenance_log,
    same_parent_nonzero_pv_cells,
    summarize_rows,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_magnitude_selectivity_20260607.json"
)

MAGNITUDE_FEATURES = [
    "pressure_pressure_abs_sum",
    "pressure_velocity_abs_sum",
    "pressure_velocity_to_pressure_pressure_abs_ratio",
    "pressure_pressure_abs_sum_per_parent_cell",
    "pressure_velocity_abs_sum_per_parent_cell",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether solve-time direct PSPG pressure-pressure and "
            "pressure-velocity magnitude features provide a credible "
            "support/coupling discriminator for Test02/Test10 targets."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0.0 else 0.0


def row_magnitude_features(stats: dict[str, Any]) -> dict[str, float]:
    pp_abs = float(stats.get("pressure_pressure_abs_sum") or 0.0)
    pv_abs = float(stats.get("pressure_velocity_abs_sum") or 0.0)
    pp_parent_count = len(stats.get("pressure_pressure_parent_cells") or ())
    pv_parent_count = len(stats.get("pressure_velocity_parent_cells") or ())
    return {
        "pressure_pressure_abs_sum": pp_abs,
        "pressure_velocity_abs_sum": pv_abs,
        "pressure_velocity_to_pressure_pressure_abs_ratio": safe_ratio(
            pv_abs, pp_abs
        ),
        "pressure_pressure_abs_sum_per_parent_cell": safe_ratio(
            pp_abs, float(pp_parent_count)
        ),
        "pressure_velocity_abs_sum_per_parent_cell": safe_ratio(
            pv_abs, float(pv_parent_count)
        ),
        "pressure_pressure_parent_cell_count": float(pp_parent_count),
        "pressure_velocity_parent_cell_count": float(pv_parent_count),
        "same_parent_nonzero_pressure_velocity_parent_cell_count": float(
            len(same_parent_nonzero_pv_cells(stats))
        ),
    }


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
    selected_count = len(selected_rows)
    target_count = len(target_rows)
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
    result = {
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
    row_features: dict[int, dict[str, float]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    present_targets = [row for row in target_rows if row in row_features]
    for feature in MAGNITUDE_FEATURES:
        target_values = [row_features[row][feature] for row in present_targets]
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
                    for row, features in row_features.items()
                    if target_min <= features[feature] <= target_max
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
                    for row, features in row_features.items()
                    if features[feature] in target_value_set
                },
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
                extra={
                    "target_value_count": len(target_value_set),
                    "production_readiness": (
                        "diagnostic_oracle_not_formulation_ready"
                    ),
                },
            )
        )
    return selectors


def case_finding(selectors: list[dict[str, Any]]) -> str:
    range_selectors = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "target_value_range"
    ]
    exact_selectors = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "exact_target_value_set"
    ]
    if any(selector.get("finding") == "selector_selective" for selector in range_selectors):
        return "solve_time_magnitude_range_selector_candidate"
    if any(selector.get("finding") == "selector_selective" for selector in exact_selectors):
        return "exact_magnitude_value_oracles_only_range_selectors_broad"
    return "solve_time_magnitude_selectors_overbroad_or_miss_targets"


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = summarize_rows(entries)
    row_features = {
        row: row_magnitude_features(stats) for row, stats in rows.items()
    }
    selectors = build_feature_selectors(
        row_features=row_features,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    range_selectors = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "target_value_range"
    ]
    exact_selectors = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "exact_target_value_set"
    ]
    present_targets = [row for row in target_rows if row in row_features]
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(selectors),
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_count": len(target_rows),
        "target_rows_present_count": len(present_targets),
        "target_rows": [
            {
                "row_dof": row,
                "present": row in row_features,
                "features": row_features.get(row),
            }
            for row in target_rows
        ],
        "range_selector_findings": {
            selector["key"]: selector["finding"] for selector in range_selectors
        },
        "range_selector_selected_to_target_ratios": {
            selector["key"]: selector["selected_to_target_ratio"]
            for selector in range_selectors
        },
        "exact_value_oracle_selector_keys": [
            selector["key"]
            for selector in exact_selectors
            if selector.get("finding") == "selector_selective"
        ],
        "exact_value_oracle_selected_to_target_ratios": {
            selector["key"]: selector["selected_to_target_ratio"]
            for selector in exact_selectors
            if selector.get("finding") == "selector_selective"
        },
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any(
        case.get("finding") == "solve_time_magnitude_range_selector_candidate"
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_support_coupling_magnitude_candidate_found",
            "range_magnitude_candidate_needs_targeted_replay",
        )
    if any(
        case.get("finding")
        == "exact_magnitude_value_oracles_only_range_selectors_broad"
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_support_coupling_magnitude_selectors_not_formulation_ready",
            "range_thresholds_overbroad_exact_value_oracles_only",
        )
    return (
        "solve_time_direct_pspg_support_coupling_magnitude_selectors_ruled_out",
        "magnitude_selectors_overbroad_or_miss_targets",
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
    cases = [
        build_case_report(
            label=label,
            log_path=log_paths_by_case.get(label),
            entries=log_entries_by_case.get(label, []),
            target_rows=targets.get(label, []),
            max_target_ratio=max_target_ratio,
        )
        for label in ("test02", "test10")
    ]
    finding, status = aggregate_finding(cases)
    return {
        "finding": finding,
        "status": status,
        "scope": (
            "Short Test02 step382 and Test10 step90 solve-time direct PSPG "
            "support/coupling magnitude selectivity audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "features": MAGNITUDE_FEATURES,
        "cases": cases,
        "conclusion": (
            "Solve-time pressure-pressure and pressure-velocity magnitude "
            "features do not provide a credible formulation-side gate. Exact "
            "floating target-value sets can identify some saved-state targets, "
            "but those are diagnostic oracles tied to the replay state. The "
            "corresponding target-value ranges, which are the threshold-like "
            "selectors that could plausibly become a rule, are broad or miss "
            "branch-specific targets."
        ),
        "next_requirement": (
            "Do not promote exact local-matrix magnitude equality as a "
            "support/coupling rule. Continue with topology/physics provenance "
            "or a targeted Test10 aggregated-signature replay."
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
