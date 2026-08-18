#!/usr/bin/env python3
"""Audit signed direct PSPG cut-volume column-support selectivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
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
DEFAULT_OPERATOR = "equations_diagnostic_ns_vms_pspg_pressure_gradient"


def _load_column_support_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_column_support_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


CS = _load_column_support_module()
LM = CS.LM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build signed sampled-column support graph and edge-magnitude "
            "profiles from the direct PSPG cut-volume pressure-gradient "
            "diagnostic, then compare fixed selectors against audited "
            "Test02/Test10 target rows."
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


def edge_abs_by_row(entries: list[dict[str, Any]]) -> dict[int, dict[int, float]]:
    edges: dict[int, dict[int, float]] = {}
    for entry in entries:
        row = entry.get("row_dof")
        if not isinstance(row, int):
            continue
        col_dofs = CS.as_int_list(entry.get("sampled_col_dofs"))
        col_values = CS.as_float_list(entry.get("sampled_col_values"))
        row_edges = edges.setdefault(row, {})
        for col_dof, value in zip(col_dofs, col_values):
            if col_dof == row or value == 0.0:
                continue
            row_edges[col_dof] = row_edges.get(col_dof, 0.0) + abs(value)
    return edges


def connected_component_sizes(
    adjacency: dict[int, set[int]],
    rows: set[int],
) -> dict[int, int]:
    seen: set[int] = set()
    sizes: dict[int, int] = {}
    for row in sorted(rows):
        if row in seen:
            continue
        stack = [row]
        component: list[int] = []
        seen.add(row)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor in rows and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        for member in component:
            sizes[member] = len(component)
    return sizes


def graph_clustering(
    row: int,
    adjacency: dict[int, set[int]],
) -> float:
    neighbors = sorted(adjacency.get(row, set()))
    possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
    if possible_edges == 0:
        return 0.0
    actual_edges = 0
    for index, lhs in enumerate(neighbors):
        for rhs in neighbors[index + 1 :]:
            if rhs in adjacency.get(lhs, set()) or lhs in adjacency.get(rhs, set()):
                actual_edges += 1
    return actual_edges / possible_edges


def two_hop_count(
    row: int,
    adjacency: dict[int, set[int]],
) -> int:
    neighbors = set(adjacency.get(row, set()))
    two_hop: set[int] = set()
    for neighbor in neighbors:
        two_hop.update(adjacency.get(neighbor, set()))
    two_hop.discard(row)
    two_hop.difference_update(neighbors)
    return len(two_hop)


def signed_column_graph_profiles(
    *,
    entries: list[dict[str, Any]],
    candidate_rows: list[int],
) -> dict[int, dict[str, Any]]:
    base_profiles = CS.row_profiles_from_column_entries(entries)
    edges = edge_abs_by_row(entries)
    candidate_set = set(candidate_rows)
    profiled_candidates = [row for row in candidate_rows if row in base_profiles]

    directed_candidate_adjacency: dict[int, set[int]] = {
        row: {
            col
            for col in edges.get(row, {})
            if col in candidate_set and col != row
        }
        for row in profiled_candidates
    }
    undirected_candidate_adjacency: dict[int, set[int]] = {
        row: set() for row in profiled_candidates
    }
    for row, neighbors in directed_candidate_adjacency.items():
        for neighbor in neighbors:
            if neighbor not in candidate_set:
                continue
            undirected_candidate_adjacency.setdefault(row, set()).add(neighbor)
            undirected_candidate_adjacency.setdefault(neighbor, set()).add(row)

    component_sizes = connected_component_sizes(
        undirected_candidate_adjacency,
        set(profiled_candidates),
    )

    profiles: dict[int, dict[str, Any]] = {}
    for row in profiled_candidates:
        base = base_profiles[row]
        row_edges = edges.get(row, {})
        edge_values = [value for value in row_edges.values() if value > 0.0]
        edge_abs_sum = sum(edge_values)
        max_edge_abs = max(edge_values) if edge_values else 0.0
        min_edge_abs = min(edge_values) if edge_values else 0.0
        candidate_neighbors = directed_candidate_adjacency.get(row, set())
        offcandidate_neighbors = set(row_edges).difference(candidate_set)
        reciprocal_edges = sum(
            1
            for neighbor in candidate_neighbors
            if row in directed_candidate_adjacency.get(neighbor, set())
        )

        profiles[row] = {
            "global_dof": row,
            "column_support_class": base.get("column_support_class", "unknown"),
            "rule_count": base.get("rule_count", 0),
            "parent_cell_count": base.get("parent_cell_count", 0),
            "sampled_col_count": base.get("sampled_col_count", 0),
            "sampled_offdiag_col_count": base.get("sampled_offdiag_col_count", 0),
            "positive_sampled_offdiag_col_count": base.get(
                "positive_sampled_offdiag_col_count",
                0,
            ),
            "negative_sampled_offdiag_col_count": base.get(
                "negative_sampled_offdiag_col_count",
                0,
            ),
            "pressure_row_abs_sum": safe_float(base.get("pressure_row_abs_sum")),
            "diag_abs": safe_float(base.get("diag_abs")),
            "sampled_offdiag_abs_sum": safe_float(
                base.get("sampled_offdiag_abs_sum")
            ),
            "pressure_row_signed_sum_ratio": safe_float(
                base.get("pressure_row_signed_sum_ratio")
            ),
            "sampled_offdiag_signed_balance_ratio": safe_float(
                base.get("sampled_offdiag_signed_balance_ratio")
            ),
            "candidate_negative_offdiag_col_count": len(candidate_neighbors),
            "offcandidate_negative_offdiag_col_count": len(offcandidate_neighbors),
            "reciprocal_candidate_negative_edge_count": reciprocal_edges,
            "nonreciprocal_candidate_negative_edge_count": (
                len(candidate_neighbors) - reciprocal_edges
            ),
            "column_graph_component_size": component_sizes.get(row, 0),
            "column_graph_two_hop_count": two_hop_count(
                row,
                directed_candidate_adjacency,
            ),
            "column_graph_clustering": graph_clustering(
                row,
                undirected_candidate_adjacency,
            ),
            "edge_abs_concentration": (
                max_edge_abs / edge_abs_sum if edge_abs_sum > 0.0 else 0.0
            ),
            "edge_min_to_max_abs_ratio": (
                min_edge_abs / max_edge_abs if max_edge_abs > 0.0 else 0.0
            ),
            "mean_edge_abs": (
                edge_abs_sum / len(edge_values) if edge_values else 0.0
            ),
            "diag_to_offdiag_abs_ratio": (
                safe_float(base.get("diag_abs")) / edge_abs_sum
                if edge_abs_sum > 0.0
                else 0.0
            ),
            "candidate_negative_offdiag_col_dofs": sorted(candidate_neighbors)[:24],
            "offcandidate_negative_offdiag_col_dofs": sorted(
                offcandidate_neighbors
            )[:24],
        }
    return profiles


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

    return {
        "candidate_degree_p25": LM.percentile(
            values("candidate_negative_offdiag_col_count"),
            0.25,
        ),
        "candidate_degree_p75": LM.percentile(
            values("candidate_negative_offdiag_col_count"),
            0.75,
        ),
        "two_hop_p25": LM.percentile(values("column_graph_two_hop_count"), 0.25),
        "two_hop_p75": LM.percentile(values("column_graph_two_hop_count"), 0.75),
        "component_size_p25": LM.percentile(values("column_graph_component_size"), 0.25),
        "component_size_p75": LM.percentile(values("column_graph_component_size"), 0.75),
        "edge_concentration_p25": LM.percentile(values("edge_abs_concentration"), 0.25),
        "edge_concentration_p75": LM.percentile(values("edge_abs_concentration"), 0.75),
        "edge_min_to_max_ratio_p25": LM.percentile(
            values("edge_min_to_max_abs_ratio"),
            0.25,
        ),
        "edge_min_to_max_ratio_p75": LM.percentile(
            values("edge_min_to_max_abs_ratio"),
            0.75,
        ),
        "mean_edge_abs_p25": LM.percentile(values("mean_edge_abs"), 0.25),
        "mean_edge_abs_p75": LM.percentile(values("mean_edge_abs"), 0.75),
    }


def le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def selector_definitions(
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "column_profiled_candidate",
            "description": "Preferred candidates with signed sampled column-support profiles.",
            "threshold_key": None,
            "predicate": lambda profile: True,
        },
        {
            "key": "column_null_preserving_negative_offdiag_class",
            "description": "Rows classified as null-preserving negative-offdiagonal sampled stencils.",
            "threshold_key": "fixed:null_preserving_negative_offdiag_stencil",
            "predicate": lambda profile: (
                profile["column_support_class"]
                == "null_preserving_negative_offdiag_stencil"
            ),
        },
        {
            "key": "column_candidate_neighbor_closed",
            "description": "Rows whose sampled negative offdiagonal columns are all preferred candidates.",
            "threshold_key": "fixed:0_offcandidate_neighbors",
            "predicate": lambda profile: (
                profile["offcandidate_negative_offdiag_col_count"] == 0
                and profile["candidate_negative_offdiag_col_count"] > 0
            ),
        },
        {
            "key": "column_all_candidate_edges_reciprocal",
            "description": "Rows whose sampled negative candidate edges are all reciprocated.",
            "threshold_key": "fixed:reciprocal_all_candidate_edges",
            "predicate": lambda profile: (
                profile["candidate_negative_offdiag_col_count"]
                == profile["reciprocal_candidate_negative_edge_count"]
                and profile["candidate_negative_offdiag_col_count"] > 0
            ),
        },
        {
            "key": "column_low_candidate_degree_p25",
            "description": "Rows in the bottom quartile of sampled candidate-neighbor count.",
            "threshold_key": "candidate_degree_p25",
            "predicate": lambda profile: le(
                profile,
                "candidate_negative_offdiag_col_count",
                thresholds["candidate_degree_p25"],
            ),
        },
        {
            "key": "column_high_candidate_degree_p75",
            "description": "Rows in the top quartile of sampled candidate-neighbor count.",
            "threshold_key": "candidate_degree_p75",
            "predicate": lambda profile: ge(
                profile,
                "candidate_negative_offdiag_col_count",
                thresholds["candidate_degree_p75"],
            ),
        },
        {
            "key": "column_candidate_degree_tail",
            "description": "Rows in either low or high sampled candidate-neighbor count tail.",
            "threshold_key": "candidate_degree_p25|candidate_degree_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "candidate_negative_offdiag_col_count",
                    thresholds["candidate_degree_p25"],
                )
                or ge(
                    profile,
                    "candidate_negative_offdiag_col_count",
                    thresholds["candidate_degree_p75"],
                )
            ),
        },
        {
            "key": "column_single_connected_component",
            "description": "Rows in the largest sampled candidate column-support component.",
            "threshold_key": "component_size_p75",
            "predicate": lambda profile: ge(
                profile,
                "column_graph_component_size",
                thresholds["component_size_p75"],
            ),
        },
        {
            "key": "column_low_two_hop_p25",
            "description": "Rows in the bottom quartile of sampled candidate two-hop reach.",
            "threshold_key": "two_hop_p25",
            "predicate": lambda profile: le(
                profile,
                "column_graph_two_hop_count",
                thresholds["two_hop_p25"],
            ),
        },
        {
            "key": "column_high_two_hop_p75",
            "description": "Rows in the top quartile of sampled candidate two-hop reach.",
            "threshold_key": "two_hop_p75",
            "predicate": lambda profile: ge(
                profile,
                "column_graph_two_hop_count",
                thresholds["two_hop_p75"],
            ),
        },
        {
            "key": "column_two_hop_tail",
            "description": "Rows in either low or high sampled candidate two-hop reach tail.",
            "threshold_key": "two_hop_p25|two_hop_p75",
            "predicate": lambda profile: (
                le(profile, "column_graph_two_hop_count", thresholds["two_hop_p25"])
                or ge(
                    profile,
                    "column_graph_two_hop_count",
                    thresholds["two_hop_p75"],
                )
            ),
        },
        {
            "key": "column_low_edge_concentration_p25",
            "description": "Rows in the bottom quartile of strongest sampled-edge concentration.",
            "threshold_key": "edge_concentration_p25",
            "predicate": lambda profile: le(
                profile,
                "edge_abs_concentration",
                thresholds["edge_concentration_p25"],
            ),
        },
        {
            "key": "column_high_edge_concentration_p75",
            "description": "Rows in the top quartile of strongest sampled-edge concentration.",
            "threshold_key": "edge_concentration_p75",
            "predicate": lambda profile: ge(
                profile,
                "edge_abs_concentration",
                thresholds["edge_concentration_p75"],
            ),
        },
        {
            "key": "column_edge_concentration_tail",
            "description": "Rows in either low or high strongest sampled-edge concentration tail.",
            "threshold_key": "edge_concentration_p25|edge_concentration_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "edge_abs_concentration",
                    thresholds["edge_concentration_p25"],
                )
                or ge(
                    profile,
                    "edge_abs_concentration",
                    thresholds["edge_concentration_p75"],
                )
            ),
        },
        {
            "key": "column_high_edge_uniformity_p75",
            "description": "Rows in the top quartile of weakest-to-strongest sampled-edge ratio.",
            "threshold_key": "edge_min_to_max_ratio_p75",
            "predicate": lambda profile: ge(
                profile,
                "edge_min_to_max_abs_ratio",
                thresholds["edge_min_to_max_ratio_p75"],
            ),
        },
        {
            "key": "column_low_edge_uniformity_p25",
            "description": "Rows in the bottom quartile of weakest-to-strongest sampled-edge ratio.",
            "threshold_key": "edge_min_to_max_ratio_p25",
            "predicate": lambda profile: le(
                profile,
                "edge_min_to_max_abs_ratio",
                thresholds["edge_min_to_max_ratio_p25"],
            ),
        },
        {
            "key": "column_mean_edge_abs_tail",
            "description": "Rows in either low or high mean sampled-edge magnitude tail.",
            "threshold_key": "mean_edge_abs_p25|mean_edge_abs_p75",
            "predicate": lambda profile: (
                le(profile, "mean_edge_abs", thresholds["mean_edge_abs_p25"])
                or ge(profile, "mean_edge_abs", thresholds["mean_edge_abs_p75"])
            ),
        },
        {
            "key": "column_low_degree_or_high_mean_edge",
            "description": "Low sampled degree rows plus high mean sampled-edge rows.",
            "threshold_key": "candidate_degree_p25|mean_edge_abs_p75",
            "predicate": lambda profile: (
                le(
                    profile,
                    "candidate_negative_offdiag_col_count",
                    thresholds["candidate_degree_p25"],
                )
                or ge(profile, "mean_edge_abs", thresholds["mean_edge_abs_p75"])
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
    selected = [
        row
        for row in candidate_rows
        if row in profiles and selector["predicate"](profiles[row])
    ]
    selected_set = set(selected)
    covered = [row for row in target_rows if row in selected_set]
    uncovered = [row for row in target_rows if row not in selected_set]
    threshold_key = selector.get("threshold_key")
    threshold_value: Any = None
    if isinstance(threshold_key, str):
        if "|" in threshold_key:
            threshold_value = {
                key: thresholds.get(key)
                for key in threshold_key.split("|")
                if not key.startswith("fixed:")
            }
        elif threshold_key.startswith("fixed:"):
            threshold_value = threshold_key.removeprefix("fixed:")
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
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    return {
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len([row for row in target_rows if row in profiles]),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "target_profiles": {
            str(row): profiles[row] for row in target_rows if row in profiles
        },
    }


def default_column_log_paths(
    emission_cases: dict[str, dict[str, Any]],
    explicit_logs: list[str],
) -> dict[str, Path]:
    paths = LM.default_log_paths(emission_cases, [])
    for label, path in list(paths.items()):
        sibling = path.with_name("run_direct_pspg_cut_volume_column_support.log")
        if sibling.exists():
            paths[label] = sibling
    for value in explicit_logs:
        label, path = LM.parse_log_arg(value)
        paths[label] = path
    return paths


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
    log_paths = default_column_log_paths(emission_cases, explicit_logs or [])

    cases: list[dict[str, Any]] = []
    selector_cases: dict[str, list[dict[str, Any]]] = {}
    missing_cases: list[str] = []

    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = CS.latest_column_support_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        if evidence.get("status") != "ok":
            missing_cases.append(label)
        profiles = signed_column_graph_profiles(
            entries=entries,
            candidate_rows=candidate_rows,
        )
        thresholds = case_thresholds(profiles, candidate_rows)
        selectors = selector_definitions(thresholds)
        for selector in selectors:
            selector_cases.setdefault(selector["key"], []).append(
                evaluate_selector_case(
                    label=label,
                    selector=selector,
                    candidate_rows=candidate_rows,
                    target_rows=target_rows,
                    profiles=profiles,
                    thresholds=thresholds,
                    max_target_ratio=max_target_ratio,
                )
            )

        cases.append(
            {
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
            }
        )

    selectors = [
        {
            "key": key,
            "finding": LM.aggregate_selector_finding(case_results),
            "cases": case_results,
        }
        for key, case_results in selector_cases.items()
    ]
    selective = [
        selector["key"]
        for selector in selectors
        if selector["finding"] == "selector_selective"
    ]
    overbroad = [
        selector["key"]
        for selector in selectors
        if selector["finding"] == "selector_overbroad"
    ]
    miss = [
        selector["key"]
        for selector in selectors
        if "miss" in selector["finding"]
    ]

    if missing_cases:
        finding = "direct_pspg_cut_volume_column_support_selectivity_evidence_missing"
        next_requirement = (
            "Rerun the short Test02/Test10 column-support windows or pass "
            "explicit --log paths with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_COLUMN_SUPPORT_DIAGNOSTIC=1."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_column_support_selector_identified"
        next_requirement = (
            "Translate the selective signed column-support selector into a "
            "bounded formulation-side replay probe."
        )
    else:
        finding = "direct_pspg_cut_volume_column_support_selectors_overbroad_or_miss_targets"
        next_requirement = (
            "Move beyond coarse signed column topology and sampled edge "
            "magnitude tails toward element-local pressure-gradient geometry "
            "or a formulation-derived support/coupling rule."
        )

    return {
        "scope": (
            "Selectivity audit for signed sampled column-support topology and "
            "edge magnitudes from direct PSPG cut-volume pressure-gradient rows."
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
        "selective_selector_keys": selective,
        "overbroad_selector_keys": overbroad,
        "miss_selector_keys": miss,
        "cases": cases,
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
