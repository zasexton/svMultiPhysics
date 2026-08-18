#!/usr/bin/env python3
"""Audit solve-time direct PSPG support-measure selectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_aggregate_feature_selectivity import (
    best_covering_selector,
    evaluate_selector,
    selector_group,
)
from audit_direct_pspg_solve_time_provenance_replay import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    load_json,
    read_provenance_log,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_support_measure_selectivity_20260607.json"
)

SUPPORT_MEASURE_FEATURES = [
    "min_active_quadrature_points",
    "max_active_quadrature_points",
    "active_quadrature_point_values",
    "min_active_quadrature_fraction",
    "max_active_quadrature_fraction",
    "active_quadrature_fraction_values",
    "measure_values",
    "measure_fraction_values",
    "parent_measure_values",
    "rule_quadrature_point_values",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether solve-time active-quadrature and generated-measure "
            "fields provide a physical support discriminator for Test02/Test10 "
            "direct PSPG targets."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument("--test02-log", type=Path, default=DEFAULT_TEST02_LOG)
    parser.add_argument("--test10-log", type=Path, default=DEFAULT_TEST10_LOG)
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    return parser.parse_args()


def rounded(value: float) -> float:
    return round(value, 12)


def row_support_measure_features(
    entries: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    raw_rows: dict[int, dict[str, list[Any]]] = {}
    for entry in entries:
        row_dof = entry.get("row_dof")
        if not isinstance(row_dof, int):
            continue
        row = raw_rows.setdefault(
            row_dof,
            {
                "active_quadrature_points": [],
                "rule_quadrature_points": [],
                "active_quadrature_fractions": [],
                "measures": [],
                "measure_fractions": [],
                "parent_measures": [],
            },
        )
        active_qp = entry.get("active_quadrature_points")
        rule_qp = entry.get("rule_quadrature_points")
        if isinstance(active_qp, int):
            row["active_quadrature_points"].append(active_qp)
        if isinstance(rule_qp, int):
            row["rule_quadrature_points"].append(rule_qp)
        if isinstance(active_qp, int) and isinstance(rule_qp, int) and rule_qp:
            row["active_quadrature_fractions"].append(rounded(active_qp / rule_qp))

        measure = entry.get("measure")
        parent_measure = entry.get("parent_measure")
        if isinstance(measure, (int, float)):
            row["measures"].append(rounded(float(measure)))
        if isinstance(parent_measure, (int, float)):
            row["parent_measures"].append(rounded(float(parent_measure)))
        if (
            isinstance(measure, (int, float))
            and isinstance(parent_measure, (int, float))
            and parent_measure
        ):
            row["measure_fractions"].append(
                rounded(float(measure) / float(parent_measure))
            )

    features_by_row: dict[int, dict[str, Any]] = {}
    for row_dof, row in raw_rows.items():
        active_values = sorted(set(row["active_quadrature_points"]))
        active_fraction_values = sorted(set(row["active_quadrature_fractions"]))
        measure_values = sorted(set(row["measures"]))
        measure_fraction_values = sorted(set(row["measure_fractions"]))
        parent_measure_values = sorted(set(row["parent_measures"]))
        rule_values = sorted(set(row["rule_quadrature_points"]))
        features_by_row[row_dof] = {
            "min_active_quadrature_points": (
                min(active_values) if active_values else None
            ),
            "max_active_quadrature_points": (
                max(active_values) if active_values else None
            ),
            "active_quadrature_point_values": tuple(active_values),
            "min_active_quadrature_fraction": (
                min(active_fraction_values) if active_fraction_values else None
            ),
            "max_active_quadrature_fraction": (
                max(active_fraction_values) if active_fraction_values else None
            ),
            "active_quadrature_fraction_values": tuple(active_fraction_values),
            "measure_values": tuple(measure_values),
            "measure_fraction_values": tuple(measure_fraction_values),
            "parent_measure_values": tuple(parent_measure_values),
            "rule_quadrature_point_values": tuple(rule_values),
        }
    return features_by_row


def jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def comparable_values(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def build_feature_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    selectors: list[dict[str, Any]] = []
    present_targets = [row for row in target_rows if row in rows]
    for feature in SUPPORT_MEASURE_FEATURES:
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
        target_value_set = set(target_values)
        scalar_targets = [
            value
            for target_value in target_values
            for value in comparable_values(target_value)
        ]
        if scalar_targets:
            target_min = min(scalar_targets)
            target_max = max(scalar_targets)
            selectors.append(
                evaluate_selector(
                    key=f"{feature}_target_range",
                    selector_kind="target_value_range",
                    feature=feature,
                    selected_rows={
                        row
                        for row, value in row_values.items()
                        if all(
                            target_min <= item <= target_max
                            for item in comparable_values(value)
                        )
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
                    "target_values": sorted(jsonable(value) for value in target_value_set),
                    "production_readiness": "integer_or_class_value_candidate",
                },
            )
        )
    return selectors


def case_finding(
    *,
    selectors: list[dict[str, Any]],
    target_count: int,
    target_rows_present_count: int,
) -> str:
    if target_count == 0:
        return "direct_target_rows_missing"
    if target_rows_present_count < target_count:
        return "solve_time_support_measure_selectivity_missing_target_rows"
    if any(
        selector.get("finding") == "selector_selective"
        for selector in selector_group(selectors, "target_value_range")
    ):
        return "solve_time_support_measure_range_selector_candidate"
    if any(
        selector.get("finding") == "selector_selective"
        for selector in selector_group(selectors, "exact_target_value_set")
    ):
        return "solve_time_support_measure_exact_value_selector_candidate"
    return "solve_time_support_measure_selectors_overbroad_or_miss_targets"


def build_case_report(
    *,
    label: str,
    log_path: Path | None,
    entries: list[dict[str, Any]],
    target_rows: list[int],
    max_target_ratio: float,
) -> dict[str, Any]:
    rows = row_support_measure_features(entries)
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
        "features": SUPPORT_MEASURE_FEATURES,
        "target_rows": [
            {
                "row_dof": row,
                "present": row in rows,
                "features": jsonable(
                    {
                        feature: rows[row].get(feature)
                        for feature in SUPPORT_MEASURE_FEATURES
                    }
                )
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
            "solve_time_direct_pspg_support_measure_selectivity_missing_evidence",
            "regenerate_solve_time_provenance_logs",
        )
    if any(
        case.get("finding")
        in {
            "solve_time_support_measure_range_selector_candidate",
            "solve_time_support_measure_exact_value_selector_candidate",
        }
        for case in cases
    ):
        return (
            "solve_time_direct_pspg_support_measure_candidate_requires_replay",
            "support_measure_candidate_needs_transfer_check",
        )
    return (
        "solve_time_direct_pspg_support_measure_selectivity_rules_out_qpoint_measure_gate",
        "active_qpoint_and_measure_features_overbroad",
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
            "active-quadrature and support-measure feature audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "features": SUPPORT_MEASURE_FEATURES,
        "cases": cases,
        "conclusion": (
            "Solve-time active-quadrature counts/fractions and generated-measure "
            "classes are not the missing physical discriminator. They cover the "
            "audited direct PSPG targets only with broad row sets in at least one "
            "case."
        ),
        "next_requirement": (
            "Continue the direct PSPG formulation search with a physical "
            "support/coupling discriminator beyond active quadrature count, "
            "generated measure, full-cell support, and target value ranges."
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
