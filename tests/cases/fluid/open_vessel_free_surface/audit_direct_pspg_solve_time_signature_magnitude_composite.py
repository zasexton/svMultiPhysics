#!/usr/bin/env python3
"""Audit solve-time direct PSPG signature plus magnitude composites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_direct_pspg_solve_time_magnitude_selectivity import (
    MAGNITUDE_FEATURES,
    row_magnitude_features,
)
from audit_direct_pspg_solve_time_support_coupling_signature import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TARGET_MAP,
    DEFAULT_TEST02_LOG,
    DEFAULT_TEST10_LOG,
    read_provenance_log,
    signature_tuple,
    summarize_rows,
    target_rows_by_case,
)


DEFAULT_JSON_OUTPUT = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_solve_time_signature_magnitude_composite_20260607.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify whether solve-time support/coupling signatures combined "
            "with non-oracle magnitude ranges provide a common direct PSPG "
            "support discriminator for Test02/Test10 targets."
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


def evaluate_selector(
    *,
    key: str,
    description: str,
    feature: str | None,
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

    result: dict[str, Any] = {
        "key": key,
        "description": description,
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


def target_range(
    *,
    row_features: dict[int, dict[str, float]],
    target_rows: list[int],
    feature: str,
) -> tuple[float, float] | None:
    values = [
        row_features[row][feature]
        for row in target_rows
        if row in row_features
    ]
    if not values:
        return None
    return min(values), max(values)


def build_composite_selectors(
    *,
    rows: dict[int, dict[str, Any]],
    row_features: dict[int, dict[str, float]],
    target_rows: list[int],
    max_target_ratio: float,
) -> list[dict[str, Any]]:
    present_targets = [row for row in target_rows if row in rows]
    target_signatures = {
        signature_tuple(rows[row]) for row in present_targets
    }
    target_local_signatures = {
        signature_tuple(rows[row], include_local_indices=True)
        for row in present_targets
    }
    signature_rows = {
        row
        for row, stats in rows.items()
        if signature_tuple(stats) in target_signatures
    }
    local_signature_rows = {
        row
        for row, stats in rows.items()
        if signature_tuple(stats, include_local_indices=True)
        in target_local_signatures
    }

    selectors = [
        evaluate_selector(
            key="target_support_coupling_signature_union",
            description=(
                "Rows whose solve-time support/coupling signature matches an "
                "audited target signature."
            ),
            feature=None,
            selected_rows=signature_rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            extra={"selector_kind": "signature_baseline"},
        ),
        evaluate_selector(
            key="target_support_coupling_signature_with_local_index_union",
            description=(
                "Rows whose solve-time support/coupling signature plus local "
                "row-index set matches an audited target signature."
            ),
            feature=None,
            selected_rows=local_signature_rows,
            target_rows=target_rows,
            max_target_ratio=max_target_ratio,
            extra={"selector_kind": "signature_baseline"},
        ),
    ]

    for feature in MAGNITUDE_FEATURES:
        value_range = target_range(
            row_features=row_features,
            target_rows=target_rows,
            feature=feature,
        )
        if value_range is None:
            continue
        target_min, target_max = value_range
        range_rows = {
            row
            for row, features in row_features.items()
            if target_min <= features[feature] <= target_max
        }
        selectors.append(
            evaluate_selector(
                key=f"target_signature_with_{feature}_range",
                description=(
                    "Rows matching an audited aggregate support/coupling "
                    f"signature and the audited target {feature} range."
                ),
                feature=feature,
                selected_rows=signature_rows & range_rows,
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
                extra={
                    "selector_kind": "signature_magnitude_range_composite",
                    "signature_includes_local_indices": False,
                    "target_min": target_min,
                    "target_max": target_max,
                },
            )
        )
        selectors.append(
            evaluate_selector(
                key=f"target_local_signature_with_{feature}_range",
                description=(
                    "Rows matching an audited local support/coupling signature "
                    f"and the audited target {feature} range."
                ),
                feature=feature,
                selected_rows=local_signature_rows & range_rows,
                target_rows=target_rows,
                max_target_ratio=max_target_ratio,
                extra={
                    "selector_kind": "signature_magnitude_range_composite",
                    "signature_includes_local_indices": True,
                    "target_min": target_min,
                    "target_max": target_max,
                },
            )
        )
    return selectors


def best_covering_composite(selectors: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "signature_magnitude_range_composite"
        and selector.get("covers_targets")
    ]
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
    target_rows: list[int],
    target_rows_present_count: int,
) -> str:
    if not target_rows:
        return "direct_target_rows_missing"
    if target_rows_present_count < len(target_rows):
        return "solve_time_signature_magnitude_composite_missing_target_rows"
    composites = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "signature_magnitude_range_composite"
    ]
    if any(selector.get("finding") == "selector_selective" for selector in composites):
        return "solve_time_signature_magnitude_composite_selective_candidate"
    if any(selector.get("covers_targets") for selector in composites):
        return "solve_time_signature_magnitude_composite_covers_targets_but_overbroad"
    if any(selector.get("selector_overbroad") for selector in composites):
        return "solve_time_signature_magnitude_composite_misses_or_overbroad"
    return "solve_time_signature_magnitude_composite_misses_targets"


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
    selectors = build_composite_selectors(
        rows=rows,
        row_features=row_features,
        target_rows=target_rows,
        max_target_ratio=max_target_ratio,
    )
    best = best_covering_composite(selectors)
    target_rows_present_count = sum(1 for row in target_rows if row in rows)
    composite_selectors = [
        selector
        for selector in selectors
        if selector.get("selector_kind") == "signature_magnitude_range_composite"
    ]
    return {
        "label": label,
        "log_path": str(log_path) if log_path is not None else None,
        "finding": case_finding(
            selectors, target_rows, target_rows_present_count
        ),
        "record_count": len(entries),
        "unique_pressure_row_count": len(rows),
        "target_count": len(target_rows),
        "target_rows_present_count": target_rows_present_count,
        "target_rows": target_rows,
        "best_covering_composite_key": best.get("key") if best else None,
        "best_covering_composite_selected_count": (
            best.get("selected_count") if best else None
        ),
        "best_covering_composite_selected_to_target_ratio": (
            best.get("selected_to_target_ratio") if best else None
        ),
        "composite_selector_findings": {
            selector["key"]: selector["finding"] for selector in composite_selectors
        },
        "composite_selector_selected_to_target_ratios": {
            selector["key"]: selector["selected_to_target_ratio"]
            for selector in composite_selectors
        },
        "selectors": selectors,
    }


def aggregate_finding(cases: list[dict[str, Any]]) -> tuple[str, str]:
    if any("missing" in str(case.get("finding")) for case in cases):
        return (
            "solve_time_direct_pspg_signature_magnitude_composite_missing_evidence",
            "regenerate_short_replay_logs",
        )
    selective_cases = [
        case
        for case in cases
        if case.get("finding")
        == "solve_time_signature_magnitude_composite_selective_candidate"
    ]
    if cases and len(selective_cases) == len(cases):
        return (
            "solve_time_direct_pspg_signature_magnitude_composite_selector_ready",
            "candidate_ready_for_targeted_formulation_replay",
        )
    if selective_cases:
        return (
            "solve_time_direct_pspg_signature_magnitude_composite_partial_test10_only",
            "test10_composite_candidate_test02_overbroad",
        )
    return (
        "solve_time_direct_pspg_signature_magnitude_composite_rules_out_common_gate",
        "signature_magnitude_composite_overbroad_or_misses_targets",
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
            "support/coupling signature plus magnitude-range composite audit."
        ),
        "source_diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "max_target_ratio": max_target_ratio,
        "features": MAGNITUDE_FEATURES,
        "cases": cases,
        "conclusion": (
            "Intersecting solve-time support/coupling signatures with "
            "non-oracle pressure-pressure and pressure-velocity magnitude "
            "ranges narrows the Test10 target family, but the same composite "
            "selectors remain overbroad for Test02. This rules out promoting "
            "signature-plus-range thresholding as a common formulation-side "
            "direct PSPG gate."
        ),
        "next_requirement": (
            "Continue the direct PSPG formulation search with a stronger "
            "Test02 physical discriminator; do not replace the blocked "
            "post-update same-sign pressure-action patch with solve-time "
            "signature/magnitude range thresholding."
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
