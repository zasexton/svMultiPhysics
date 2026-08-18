#!/usr/bin/env python3
"""Audit direct PSPG cut-volume parent-cell graph topology selectivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable


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
DEFAULT_OPERATOR = "equations_diagnostic_ns_vms_pspg_pressure_gradient"


def _load_local_matrix_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_local_matrix_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_local_matrix_selectivity",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


LM = _load_local_matrix_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a row-parent-cell graph from direct PSPG generated "
            "cut-volume local matrix provenance and compare graph-topology "
            "selectors with audited Test02/Test10 target rows."
        )
    )
    parser.add_argument("--global-emission-json", type=Path, default=DEFAULT_GLOBAL_EMISSION)
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--log",
        action="append",
        type=str,
        default=[],
        help="Case-labelled log path as label=/path/to/run.log.",
    )
    parser.add_argument("--candidate-key", default="preferred_candidate_global_dofs")
    parser.add_argument("--operator", default=DEFAULT_OPERATOR)
    parser.add_argument("--test-field", default="pressure")
    parser.add_argument("--trial-field", default="pressure")
    parser.add_argument("--max-target-ratio", type=float, default=5.0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def candidate_parent_graph_profiles(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
) -> dict[int, dict[str, Any]]:
    rows = [row for row in candidate_rows if row in profiles]
    row_set = set(rows)
    parent_to_rows: dict[int, set[int]] = {}
    for row in rows:
        for parent in profiles[row].get("parent_cells", []):
            if isinstance(parent, int):
                parent_to_rows.setdefault(parent, set()).add(row)

    neighbors: dict[int, dict[int, int]] = {row: {} for row in rows}
    for parent_rows in parent_to_rows.values():
        parent_row_list = sorted(row for row in parent_rows if row in row_set)
        for i, lhs in enumerate(parent_row_list):
            for rhs in parent_row_list[i + 1 :]:
                neighbors[lhs][rhs] = neighbors[lhs].get(rhs, 0) + 1
                neighbors[rhs][lhs] = neighbors[rhs].get(lhs, 0) + 1

    graph_profiles: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_neighbors = set(neighbors[row])
        degree = len(row_neighbors)
        weighted_degree = sum(neighbors[row].values())
        max_edge_weight = max(neighbors[row].values()) if row_neighbors else 0
        if degree >= 2:
            possible_edges = degree * (degree - 1) // 2
            actual_edges = 0
            neighbor_list = sorted(row_neighbors)
            for i, lhs in enumerate(neighbor_list):
                lhs_neighbors = neighbors[lhs]
                for rhs in neighbor_list[i + 1 :]:
                    if rhs in lhs_neighbors:
                        actual_edges += 1
            clustering = actual_edges / possible_edges
        else:
            clustering = 0.0

        two_hop = set()
        for neighbor in row_neighbors:
            two_hop.update(neighbors[neighbor])
        two_hop.discard(row)
        two_hop.difference_update(row_neighbors)

        graph_profiles[row] = {
            "global_dof": row,
            "parent_cell_count": profiles[row].get("parent_cell_count", 0),
            "cut_volume_support_class": profiles[row].get(
                "cut_volume_support_class",
                "unknown",
            ),
            "row_parent_graph_degree": degree,
            "row_parent_graph_weighted_degree": weighted_degree,
            "row_parent_graph_max_edge_weight": max_edge_weight,
            "row_parent_graph_clustering": clustering,
            "row_parent_graph_two_hop_count": len(two_hop),
            "row_parent_graph_neighbor_sample": sorted(row_neighbors)[:24],
            "total_row_abs_sum": safe_float(profiles[row].get("total_row_abs_sum")),
            "max_rule_row_abs_fraction": safe_float(
                profiles[row].get("max_rule_row_abs_fraction")
            ),
        }
    return graph_profiles


def case_thresholds(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
) -> dict[str, float | None]:
    candidate_profiles = [profiles[row] for row in candidate_rows if row in profiles]

    def values(key: str) -> list[float]:
        return [
            float(profile[key])
            for profile in candidate_profiles
            if isinstance(profile.get(key), (int, float))
        ]

    degree = values("row_parent_graph_degree")
    weighted_degree = values("row_parent_graph_weighted_degree")
    clustering = values("row_parent_graph_clustering")
    two_hop = values("row_parent_graph_two_hop_count")
    return {
        "degree_p25": LM.percentile(degree, 0.25),
        "degree_p75": LM.percentile(degree, 0.75),
        "weighted_degree_p25": LM.percentile(weighted_degree, 0.25),
        "weighted_degree_p75": LM.percentile(weighted_degree, 0.75),
        "clustering_p25": LM.percentile(clustering, 0.25),
        "clustering_p75": LM.percentile(clustering, 0.75),
        "two_hop_p25": LM.percentile(two_hop, 0.25),
        "two_hop_p75": LM.percentile(two_hop, 0.75),
    }


def threshold_le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def threshold_ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def selector_definitions(thresholds: dict[str, float | None]) -> list[dict[str, Any]]:
    return [
        {
            "key": "parent_graph_profiled_candidate",
            "description": "Preferred candidates in the row-parent-cell support graph.",
            "threshold_key": None,
            "predicate": lambda profile: profile["row_parent_graph_degree"] >= 0,
        },
        {
            "key": "parent_graph_low_degree_p25",
            "description": "Rows in the bottom quartile of shared-parent graph degree.",
            "threshold_key": "degree_p25",
            "predicate": lambda profile: threshold_le(
                profile, "row_parent_graph_degree", thresholds["degree_p25"]
            ),
        },
        {
            "key": "parent_graph_high_degree_p75",
            "description": "Rows in the top quartile of shared-parent graph degree.",
            "threshold_key": "degree_p75",
            "predicate": lambda profile: threshold_ge(
                profile, "row_parent_graph_degree", thresholds["degree_p75"]
            ),
        },
        {
            "key": "parent_graph_low_weighted_degree_p25",
            "description": "Rows in the bottom quartile of parent-overlap weighted degree.",
            "threshold_key": "weighted_degree_p25",
            "predicate": lambda profile: threshold_le(
                profile,
                "row_parent_graph_weighted_degree",
                thresholds["weighted_degree_p25"],
            ),
        },
        {
            "key": "parent_graph_high_weighted_degree_p75",
            "description": "Rows in the top quartile of parent-overlap weighted degree.",
            "threshold_key": "weighted_degree_p75",
            "predicate": lambda profile: threshold_ge(
                profile,
                "row_parent_graph_weighted_degree",
                thresholds["weighted_degree_p75"],
            ),
        },
        {
            "key": "parent_graph_low_two_hop_p25",
            "description": "Rows in the bottom quartile of two-hop row-parent graph reach.",
            "threshold_key": "two_hop_p25",
            "predicate": lambda profile: threshold_le(
                profile, "row_parent_graph_two_hop_count", thresholds["two_hop_p25"]
            ),
        },
        {
            "key": "parent_graph_high_two_hop_p75",
            "description": "Rows in the top quartile of two-hop row-parent graph reach.",
            "threshold_key": "two_hop_p75",
            "predicate": lambda profile: threshold_ge(
                profile, "row_parent_graph_two_hop_count", thresholds["two_hop_p75"]
            ),
        },
        {
            "key": "parent_graph_low_clustering_p25",
            "description": "Rows in the bottom quartile of row-parent graph clustering.",
            "threshold_key": "clustering_p25",
            "predicate": lambda profile: threshold_le(
                profile, "row_parent_graph_clustering", thresholds["clustering_p25"]
            ),
        },
        {
            "key": "parent_graph_high_clustering_p75",
            "description": "Rows in the top quartile of row-parent graph clustering.",
            "threshold_key": "clustering_p75",
            "predicate": lambda profile: threshold_ge(
                profile, "row_parent_graph_clustering", thresholds["clustering_p75"]
            ),
        },
        {
            "key": "parent_graph_degree_tail",
            "description": "Rows in either low-degree or high-degree tail.",
            "threshold_key": "degree_p25|degree_p75",
            "predicate": lambda profile: (
                threshold_le(profile, "row_parent_graph_degree", thresholds["degree_p25"])
                or threshold_ge(profile, "row_parent_graph_degree", thresholds["degree_p75"])
            ),
        },
        {
            "key": "parent_graph_low_degree_high_clustering",
            "description": "Low-degree rows with high local row-parent graph clustering.",
            "threshold_key": "degree_p25|clustering_p75",
            "predicate": lambda profile: (
                threshold_le(profile, "row_parent_graph_degree", thresholds["degree_p25"])
                and threshold_ge(
                    profile,
                    "row_parent_graph_clustering",
                    thresholds["clustering_p75"],
                )
            ),
        },
        {
            "key": "parent_graph_high_degree_low_clustering",
            "description": "High-degree rows with low local row-parent graph clustering.",
            "threshold_key": "degree_p75|clustering_p25",
            "predicate": lambda profile: (
                threshold_ge(profile, "row_parent_graph_degree", thresholds["degree_p75"])
                and threshold_le(
                    profile,
                    "row_parent_graph_clustering",
                    thresholds["clustering_p25"],
                )
            ),
        },
    ]


def evaluate_selector_case(
    *,
    label: str,
    selector: dict[str, Any],
    candidate_rows: list[int],
    target_rows: list[int],
    profiles: dict[int, dict[str, Any]],
    thresholds: dict[str, float | None],
    max_target_ratio: float,
) -> dict[str, Any]:
    predicate: Callable[[dict[str, Any]], bool] = selector["predicate"]
    selected = [
        row for row in candidate_rows if row in profiles and predicate(profiles[row])
    ]
    selected_set = set(selected)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    threshold_key = selector.get("threshold_key")
    threshold_value: Any = None
    if isinstance(threshold_key, str):
        if "|" in threshold_key:
            threshold_value = {key: thresholds.get(key) for key in threshold_key.split("|")}
        else:
            threshold_value = thresholds.get(threshold_key)
    return {
        "label": label,
        "key": selector["key"],
        "description": selector["description"],
        "threshold_key": threshold_key,
        "threshold_value": threshold_value,
        "finding": LM.selector_finding(
            selected_count=len(selected),
            covered=covered,
            uncovered=uncovered,
            direct_target_count=len(target_rows),
            max_target_ratio=max_target_ratio,
        ),
        "direct_target_count": len(target_rows),
        "selected_count": len(selected),
        "selected_to_target_ratio": LM.ratio(len(selected), len(target_rows)),
        "covered_direct_target_count": len(covered),
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "selected_global_dofs": selected,
    }


def profile_summary(
    *,
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(target_set.intersection(profiles)),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "target_profiles": {
            str(row): profiles[row] for row in target_rows if row in profiles
        },
    }


def build_report(
    *,
    global_emission: dict[str, Any],
    target_map: dict[str, Any],
    global_emission_path: Path | None = None,
    target_map_path: Path | None = None,
    explicit_logs: list[str] | None = None,
    candidate_key: str = "preferred_candidate_global_dofs",
    operator: str = DEFAULT_OPERATOR,
    test_field: str = "pressure",
    trial_field: str = "pressure",
    max_target_ratio: float = 5.0,
) -> dict[str, Any]:
    emission_cases = LM.case_map(global_emission)
    target_cases = LM.target_case_map(target_map)
    log_paths = LM.default_log_paths(emission_cases, explicit_logs or [])
    cases: dict[str, dict[str, Any]] = {}
    selector_defs_by_case: dict[str, list[dict[str, Any]]] = {}

    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = LM.latest_local_matrix_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        base_profiles = LM.row_profiles_from_entries(entries)
        profiles = candidate_parent_graph_profiles(base_profiles, candidate_rows)
        thresholds = case_thresholds(profiles, candidate_rows)
        selector_defs = selector_definitions(thresholds)
        selector_defs_by_case[label] = selector_defs
        selector_cases = [
            evaluate_selector_case(
                label=label,
                selector=selector,
                candidate_rows=candidate_rows,
                target_rows=target_rows,
                profiles=profiles,
                thresholds=thresholds,
                max_target_ratio=max_target_ratio,
            )
            for selector in selector_defs
        ]
        cases[label] = {
            "label": label,
            "candidate_key": candidate_key,
            "candidate_count": len(candidate_rows),
            "direct_target_count": len(target_rows),
            "log_evidence": evidence,
            "thresholds": thresholds,
            "profile_summary": profile_summary(
                profiles=profiles,
                candidate_rows=candidate_rows,
                target_rows=target_rows,
            ),
            "selectors": selector_cases,
        }

    selectors = []
    first_label = next(iter(target_cases), None)
    selector_count = len(selector_defs_by_case[first_label]) if first_label else 0
    for selector_index in range(selector_count):
        case_results = [cases[label]["selectors"][selector_index] for label in target_cases]
        selector_template = selector_defs_by_case[first_label][selector_index]
        selectors.append(
            {
                "key": selector_template["key"],
                "description": selector_template["description"],
                "finding": LM.aggregate_selector_finding(case_results),
                "cases": case_results,
            }
        )

    selective = [
        selector for selector in selectors if selector["finding"] == "selector_selective"
    ]
    overbroad = [
        selector for selector in selectors if "overbroad" in str(selector["finding"])
    ]
    misses = [selector for selector in selectors if "miss" in str(selector["finding"])]
    missing_cases = [
        label
        for label, case in cases.items()
        if case["log_evidence"].get("status") != "ok"
    ]

    if missing_cases:
        finding = "direct_pspg_cut_volume_parent_graph_evidence_missing"
        next_requirement = (
            "Regenerate cut-volume local matrix provenance before evaluating "
            "parent-support graph topology."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_parent_graph_selector_selective"
        next_requirement = (
            "Prototype the selective row-parent graph topology gate and run "
            "the same short Test02/Test10 replay windows."
        )
    elif overbroad or misses:
        finding = (
            "direct_pspg_cut_volume_parent_graph_selectors_overbroad_or_miss_targets"
        )
        next_requirement = (
            "Do not promote row-parent-cell graph degree, clustering, or two-hop "
            "reach alone; the remaining discriminator must combine spatial graph "
            "context with stronger formulation physics."
        )
    else:
        finding = "direct_pspg_cut_volume_parent_graph_selectivity_inconclusive"
        next_requirement = "Regenerate graph evidence before selecting a formulation replay."

    return {
        "scope": (
            "Selectivity audit for row-parent-cell graph topology in direct PSPG "
            "generated cut-volume pressure-pressure local matrix provenance."
        ),
        "global_emission_path": str(global_emission_path) if global_emission_path else None,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "candidate_key": candidate_key,
        "operator": operator,
        "test_field": test_field,
        "trial_field": trial_field,
        "max_target_ratio": max_target_ratio,
        "finding": finding,
        "missing_case_labels": missing_cases,
        "selective_selector_keys": [selector["key"] for selector in selective],
        "overbroad_selector_keys": [selector["key"] for selector in overbroad],
        "miss_selector_keys": [selector["key"] for selector in misses],
        "cases": list(cases.values()),
        "selectors": selectors,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    report = build_report(
        global_emission=LM.load_json(args.global_emission_json),
        target_map=LM.load_json(args.target_map_json),
        global_emission_path=args.global_emission_json,
        target_map_path=args.target_map_json,
        explicit_logs=args.log,
        candidate_key=args.candidate_key,
        operator=args.operator,
        test_field=args.test_field,
        trial_field=args.trial_field,
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
