#!/usr/bin/env python3
"""Audit direct PSPG coupled-patch graph topology selectors."""

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
        "key": "pressure_action_zero_two_hop",
        "description": (
            "Direct PSPG rows with no missing two-hop pressure-action completion "
            "neighbors through the current pressure-action graph."
        ),
        "count_key": "pressure_action_zero_two_hop_candidate_count",
        "covered_key": (
            "pressure_action_zero_two_hop_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_zero_two_hop_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_low_two_hop",
        "description": (
            "Direct PSPG rows with bounded missing two-hop pressure-action "
            "completion neighbors."
        ),
        "count_key": "pressure_action_low_two_hop_candidate_count",
        "covered_key": (
            "pressure_action_low_two_hop_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_low_two_hop_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_high_two_hop",
        "description": (
            "Direct PSPG rows whose two-hop completion count is high relative "
            "to the maximum direct PSPG pressure-action graph count."
        ),
        "count_key": "pressure_action_high_two_hop_candidate_count",
        "covered_key": (
            "pressure_action_high_two_hop_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_high_two_hop_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_zero_clustering",
        "description": (
            "Direct PSPG rows with degree at least two whose pressure-action "
            "neighbors have no pressure-action edges between them."
        ),
        "count_key": "pressure_action_zero_clustering_candidate_count",
        "covered_key": (
            "pressure_action_zero_clustering_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_zero_clustering_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_low_clustering",
        "description": (
            "Direct PSPG rows whose pressure-action neighbor graph has low "
            "local clustering."
        ),
        "count_key": "pressure_action_low_clustering_candidate_count",
        "covered_key": (
            "pressure_action_low_clustering_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_low_clustering_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_high_clustering",
        "description": (
            "Direct PSPG rows whose pressure-action neighbor graph has high "
            "local clustering."
        ),
        "count_key": "pressure_action_high_clustering_candidate_count",
        "covered_key": (
            "pressure_action_high_clustering_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_high_clustering_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_articulation",
        "description": (
            "Direct PSPG rows that are articulation points in the pressure-action "
            "graph."
        ),
        "count_key": "pressure_action_articulation_candidate_count",
        "covered_key": (
            "pressure_action_articulation_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_articulation_uncovered_direct_target_global_dofs"
        ),
    },
    {
        "key": "pressure_action_bridge_endpoint",
        "description": (
            "Direct PSPG rows that are endpoints of pressure-action graph "
            "bridges."
        ),
        "count_key": "pressure_action_bridge_endpoint_candidate_count",
        "covered_key": (
            "pressure_action_bridge_endpoint_covered_direct_target_global_dofs"
        ),
        "uncovered_key": (
            "pressure_action_bridge_endpoint_uncovered_direct_target_global_dofs"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matrix-only direct PSPG pressure-action graph motif "
            "selectors against audited Test02/Test10 target rows."
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


def graph_topology_summary(
    emission_cases: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    keys = [
        "pressure_action_low_two_hop_threshold",
        "pressure_action_high_two_hop_ratio_threshold",
        "matrix_pressure_action_max_two_hop_completion_count",
        "pressure_action_low_clustering_threshold",
        "pressure_action_high_clustering_threshold",
        "pressure_action_clustering_eligible_row_count",
        "matrix_pressure_action_min_clustering_ratio",
        "matrix_pressure_action_max_clustering_ratio",
        "matrix_pressure_action_component_count",
        "matrix_pressure_action_largest_component_size",
        "matrix_pressure_action_covered_count",
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
        finding = "direct_pspg_coupled_patch_graph_selector_selective"
        next_requirement = (
            "Prototype the selective matrix-only coupled-patch graph gate and "
            "run short Test02/Test10 replay windows."
        )
    elif missing:
        finding = "direct_pspg_coupled_patch_graph_selector_evidence_missing"
        next_requirement = (
            "Regenerate direct PSPG global candidate emission with coupled-patch "
            "graph motif fields enabled."
        )
    elif overbroad or misses:
        finding = (
            "direct_pspg_coupled_patch_graph_selectors_overbroad_or_miss_targets"
        )
        next_requirement = (
            "Do not promote matrix-only direct PSPG pressure-action graph "
            "motifs as the formulation gate. Add element/cut-volume pressure-"
            "gradient support provenance before replay."
        )
    else:
        finding = "direct_pspg_coupled_patch_graph_selectivity_inconclusive"
        next_requirement = (
            "Regenerate coupled-patch graph motif evidence before choosing a "
            "formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for matrix-only coupled-patch topology in the "
            "direct PSPG pressure-action graph."
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
        "graph_topology_by_case": graph_topology_summary(emission_cases),
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
