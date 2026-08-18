#!/usr/bin/env python3
"""Audit direct PSPG null/balance selectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_GLOBAL_EMISSION = (
    DEFAULT_ARTIFACT_ROOT
    / "test02_test10_direct_pspg_global_candidate_emission_20260606.json"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)


SELECTOR_DEFINITIONS = [
    {
        "key": "high_direct_self_row_sum_leak",
        "description": (
            "Direct PSPG pressure-gradient rows whose pressure-block row sum is "
            "large relative to absolute row support."
        ),
        "count_key": "high_direct_self_row_sum_leak_candidate_count",
        "covered_key": (
            "high_direct_self_row_sum_leak_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "high_direct_self_row_sum_leak_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "null_preserving_direct_self",
        "description": (
            "Direct PSPG pressure-gradient rows whose pressure-block row sum is "
            "near zero relative to absolute row support."
        ),
        "count_key": "null_preserving_direct_self_candidate_count",
        "covered_key": (
            "null_preserving_direct_self_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "null_preserving_direct_self_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "diag_dominant_direct_self",
        "description": (
            "Direct PSPG pressure-gradient rows dominated by their diagonal "
            "pressure self entry."
        ),
        "count_key": "diag_dominant_direct_self_candidate_count",
        "covered_key": "diag_dominant_direct_self_covered_direct_target_global_dofs",
        "uncovered_key": (
            "diag_dominant_direct_self_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "balanced_diag_direct_self",
        "description": (
            "Direct PSPG pressure-gradient rows whose diagonal share is near "
            "half of absolute pressure self support."
        ),
        "count_key": "balanced_diag_direct_self_candidate_count",
        "covered_key": "balanced_diag_direct_self_covered_direct_target_global_dofs",
        "uncovered_key": (
            "balanced_diag_direct_self_uncovered_direct_target_global_dofs"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct PSPG row-sum and diagonal-balance selectors "
            "against audited Test02/Test10 target rows."
        )
    )
    parser.add_argument(
        "--global-emission-json",
        type=Path,
        default=DEFAULT_GLOBAL_EMISSION,
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--max-target-ratio",
        type=float,
        default=5.0,
        help="Largest candidate/target ratio still considered selective.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ratio(numerator: int | None, denominator: int) -> float | None:
    if numerator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def numeric_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def target_counts(target_map: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            counts[label] = len(as_list(case.get("direct_pspg_target_global_dofs")))
    return counts


def case_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(report.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def evaluate_selector_case(
    *,
    selector: dict[str, str],
    label: str,
    emission_case: dict[str, Any],
    direct_target_count: int,
    max_target_ratio: float,
) -> dict[str, Any]:
    selected_count = numeric_count(emission_case.get(selector["count_key"]))
    covered = as_list(emission_case.get(selector["covered_key"]))
    uncovered = as_list(emission_case.get(selector["uncovered_key"]))
    selected_to_target_ratio = ratio(selected_count, direct_target_count)
    evidence_missing = selected_count is None
    covers_targets = (
        not evidence_missing
        and direct_target_count > 0
        and len(covered) == direct_target_count
        and not uncovered
    )
    overbroad = (
        selected_to_target_ratio is not None
        and selected_to_target_ratio > max_target_ratio
    )
    if evidence_missing:
        finding = "selector_evidence_missing"
    elif not covers_targets and overbroad:
        finding = "selector_overbroad_and_misses_targets"
    elif not covers_targets:
        finding = "selector_misses_targets"
    elif overbroad:
        finding = "selector_overbroad"
    else:
        finding = "selector_selective"
    return {
        "label": label,
        "finding": finding,
        "direct_target_count": direct_target_count,
        "selected_count": selected_count,
        "selected_to_target_ratio": selected_to_target_ratio,
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "covers_targets": covers_targets,
        "selector_overbroad": overbroad,
    }


def aggregate_selector_finding(cases: list[dict[str, Any]]) -> str:
    if any(case["finding"] == "selector_evidence_missing" for case in cases):
        return "selector_evidence_missing"
    if cases and all(case["finding"] == "selector_selective" for case in cases):
        return "selector_selective"
    if any("misses_targets" in str(case["finding"]) for case in cases):
        if any("overbroad" in str(case["finding"]) for case in cases):
            return "selector_overbroad_or_miss_targets"
        return "selector_misses_targets"
    if any(case["finding"] == "selector_overbroad" for case in cases):
        return "selector_overbroad"
    return "selector_inconclusive"


def evaluate_selector(
    *,
    selector: dict[str, str],
    emission_cases: dict[str, dict[str, Any]],
    counts: dict[str, int],
    max_target_ratio: float,
) -> dict[str, Any]:
    cases = [
        evaluate_selector_case(
            selector=selector,
            label=label,
            emission_case=emission_cases.get(label, {}),
            direct_target_count=count,
            max_target_ratio=max_target_ratio,
        )
        for label, count in counts.items()
    ]
    return {
        "key": selector["key"],
        "description": selector["description"],
        "finding": aggregate_selector_finding(cases),
        "count_key": selector["count_key"],
        "cases": cases,
    }


def null_balance_summary(emission_cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "direct_self_row_sum_leak_threshold",
        "direct_self_null_preserving_threshold",
        "direct_self_diag_dominant_threshold",
        "direct_self_balanced_diag_low_threshold",
        "direct_self_balanced_diag_high_threshold",
        "max_direct_self_row_sum_leak_ratio",
        "min_direct_self_diag_abs_ratio",
        "max_direct_self_diag_abs_ratio",
    ]
    return {
        label: {key: case.get(key) for key in keys}
        for label, case in emission_cases.items()
    }


def build_report(
    *,
    global_emission: dict[str, Any],
    target_map: dict[str, Any],
    global_emission_path: Path | None = None,
    target_map_path: Path | None = None,
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    counts = target_counts(target_map)
    emission_cases = case_map(global_emission)
    selectors = [
        evaluate_selector(
            selector=selector,
            emission_cases=emission_cases,
            counts=counts,
            max_target_ratio=max_target_ratio,
        )
        for selector in SELECTOR_DEFINITIONS
    ]
    selective = [
        selector for selector in selectors if selector["finding"] == "selector_selective"
    ]
    missing = [
        selector
        for selector in selectors
        if selector["finding"] == "selector_evidence_missing"
    ]
    misses = [
        selector
        for selector in selectors
        if "miss" in str(selector["finding"])
    ]
    overbroad = [
        selector
        for selector in selectors
        if "overbroad" in str(selector["finding"])
    ]

    if selective:
        finding = "direct_pspg_null_balance_selector_selective"
        next_requirement = (
            "Prototype the selective null/balance pressure-gradient support "
            "gate and run short Test02/Test10 replay windows."
        )
    elif missing:
        finding = "direct_pspg_null_balance_selector_evidence_missing"
        next_requirement = (
            "Regenerate direct PSPG global candidate emission with null/balance "
            "fields enabled."
        )
    elif overbroad or misses:
        finding = "direct_pspg_null_balance_selectors_overbroad_or_miss_targets"
        next_requirement = (
            "Do not promote direct PSPG row-sum or diagonal-balance topology "
            "alone. Add richer formulation-side pressure-gradient support/"
            "coupling provenance before replay."
        )
    else:
        finding = "direct_pspg_null_balance_selectivity_inconclusive"
        next_requirement = (
            "Regenerate null/balance evidence before choosing a formulation "
            "replay."
        )

    return {
        "scope": (
            "Selectivity audit for direct PSPG pressure-gradient row-sum and "
            "diagonal-balance selectors."
        ),
        "global_emission_path": (
            str(global_emission_path) if global_emission_path else None
        ),
        "target_map_path": str(target_map_path) if target_map_path else None,
        "max_target_ratio": max_target_ratio,
        "finding": finding,
        "selector_count": len(selectors),
        "selective_selector_keys": [selector["key"] for selector in selective],
        "overbroad_selector_keys": [selector["key"] for selector in overbroad],
        "miss_selector_keys": [selector["key"] for selector in misses],
        "missing_selector_keys": [selector["key"] for selector in missing],
        "null_balance_by_case": null_balance_summary(emission_cases),
        "selectors": selectors,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        global_emission=load_json(args.global_emission_json),
        target_map=load_json(args.target_map_json),
        global_emission_path=args.global_emission_json,
        target_map_path=args.target_map_json,
        max_target_ratio=args.max_target_ratio,
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
