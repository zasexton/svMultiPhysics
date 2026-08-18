#!/usr/bin/env python3
"""Audit edge-level direct PSPG gradient-column graph selectivity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
NONZERO_TOLERANCE = 1.0e-12


def _load_gradient_balance_module():
    script = Path(__file__).with_name(
        "audit_direct_pspg_cut_volume_gradient_balance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(script.stem, script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


GB = _load_gradient_balance_module()
LM = GB.LM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build edge-level sampled pressure-gradient Gram/cosine support "
            "graphs from direct PSPG cut-volume rows and compare fixed "
            "selectors against audited Test02/Test10 target rows."
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


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_float(value: Any, default: float = 0.0) -> float:
    result = finite_float(value)
    return result if result is not None else default


def finite_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    parsed: list[float] = []
    for value in values:
        result = finite_float(value)
        if result is not None:
            parsed.append(result)
    return parsed


def as_int_list(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int)]


def sign(value: float) -> int:
    if value > NONZERO_TOLERANCE:
        return 1
    if value < -NONZERO_TOLERANCE:
        return -1
    return 0


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


def graph_clustering(row: int, adjacency: dict[int, set[int]]) -> float:
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


def two_hop_count(row: int, adjacency: dict[int, set[int]]) -> int:
    neighbors = set(adjacency.get(row, set()))
    two_hop: set[int] = set()
    for neighbor in neighbors:
        two_hop.update(adjacency.get(neighbor, set()))
    two_hop.discard(row)
    two_hop.difference_update(neighbors)
    return len(two_hop)


def gradient_column_graph_profiles(
    *,
    entries: list[dict[str, Any]],
    candidate_rows: list[int],
) -> dict[int, dict[str, Any]]:
    candidate_set = set(candidate_rows)
    raw_profiles: dict[int, dict[str, Any]] = {}
    directed_candidate_adjacency: dict[int, set[int]] = {}

    for entry in entries:
        row = entry.get("row_dof")
        if not isinstance(row, int) or row not in candidate_set:
            continue
        col_dofs = as_int_list(entry.get("sampled_col_dofs"))
        values = finite_values(entry.get("sampled_col_values"))
        grams = finite_values(entry.get("sampled_col_gradient_gram_values"))
        cosines = finite_values(entry.get("sampled_col_gradient_cosines"))
        profile = raw_profiles.setdefault(
            row,
            {
                "global_dof": row,
                "rule_count": 0,
                "sampled_offdiag_edge_count": 0,
                "candidate_edge_sample_count": 0,
                "candidate_neighbors": set(),
                "negative_gradient_neighbors": set(),
                "positive_gradient_neighbors": set(),
                "reciprocal_candidate_neighbor_count": 0,
                "nonreciprocal_candidate_neighbor_count": 0,
                "matrix_gradient_sign_mismatch_count": 0,
                "negative_gradient_edge_count": 0,
                "positive_gradient_edge_count": 0,
                "zero_gradient_edge_count": 0,
                "candidate_gradient_edge_count": 0,
                "candidate_negative_gradient_edge_count": 0,
                "candidate_positive_gradient_edge_count": 0,
                "total_gradient_gram_abs_sum": 0.0,
                "candidate_gradient_gram_abs_sum": 0.0,
                "max_abs_gradient_gram": 0.0,
                "matrix_edge_abs_sum": 0.0,
                "candidate_matrix_edge_abs_sum": 0.0,
                "abs_gradient_cosine_sum": 0.0,
                "max_abs_gradient_cosine": 0.0,
                "sample_truncated_rule_count": 0,
            },
        )
        profile["rule_count"] += 1
        if entry.get("sample_truncated") == 1:
            profile["sample_truncated_rule_count"] += 1

        for col_dof, value, gram, cosine in zip(
            col_dofs,
            values,
            grams,
            cosines,
            strict=False,
        ):
            if col_dof == row:
                continue
            value_sign = sign(value)
            gram_sign = sign(gram)
            abs_gram = abs(gram)
            abs_value = abs(value)
            abs_cosine = abs(cosine)
            profile["sampled_offdiag_edge_count"] += 1
            profile["matrix_edge_abs_sum"] += abs_value
            profile["total_gradient_gram_abs_sum"] += abs_gram
            profile["max_abs_gradient_gram"] = max(
                profile["max_abs_gradient_gram"],
                abs_gram,
            )
            profile["abs_gradient_cosine_sum"] += abs_cosine
            profile["max_abs_gradient_cosine"] = max(
                profile["max_abs_gradient_cosine"],
                abs_cosine,
            )
            if value_sign and gram_sign and value_sign != gram_sign:
                profile["matrix_gradient_sign_mismatch_count"] += 1
            if gram_sign < 0:
                profile["negative_gradient_edge_count"] += 1
            elif gram_sign > 0:
                profile["positive_gradient_edge_count"] += 1
            else:
                profile["zero_gradient_edge_count"] += 1

            if col_dof not in candidate_set or value_sign == 0:
                continue
            profile["candidate_edge_sample_count"] += 1
            profile["candidate_neighbors"].add(col_dof)
            directed_candidate_adjacency.setdefault(row, set()).add(col_dof)
            profile["candidate_matrix_edge_abs_sum"] += abs_value
            profile["candidate_gradient_gram_abs_sum"] += abs_gram
            profile["candidate_gradient_edge_count"] += 1
            if gram_sign < 0:
                profile["candidate_negative_gradient_edge_count"] += 1
                profile["negative_gradient_neighbors"].add(col_dof)
            elif gram_sign > 0:
                profile["candidate_positive_gradient_edge_count"] += 1
                profile["positive_gradient_neighbors"].add(col_dof)

    profiled_rows = set(raw_profiles)
    undirected_adjacency: dict[int, set[int]] = {row: set() for row in profiled_rows}
    for row, neighbors in directed_candidate_adjacency.items():
        for neighbor in neighbors:
            if neighbor not in profiled_rows:
                continue
            undirected_adjacency.setdefault(row, set()).add(neighbor)
            undirected_adjacency.setdefault(neighbor, set()).add(row)
    component_sizes = connected_component_sizes(undirected_adjacency, profiled_rows)

    profiles: dict[int, dict[str, Any]] = {}
    profiled_candidate_count = len([row for row in candidate_rows if row in raw_profiles])
    for row, profile in raw_profiles.items():
        candidate_neighbors = sorted(profile.pop("candidate_neighbors"))
        negative_neighbors = sorted(profile.pop("negative_gradient_neighbors"))
        positive_neighbors = sorted(profile.pop("positive_gradient_neighbors"))
        reciprocal = sum(
            1
            for neighbor in candidate_neighbors
            if row in directed_candidate_adjacency.get(neighbor, set())
        )
        profile["candidate_neighbors"] = candidate_neighbors[:32]
        profile["negative_gradient_neighbors"] = negative_neighbors[:32]
        profile["positive_gradient_neighbors"] = positive_neighbors[:32]
        profile["candidate_neighbor_count"] = len(candidate_neighbors)
        profile["reciprocal_candidate_neighbor_count"] = reciprocal
        profile["nonreciprocal_candidate_neighbor_count"] = (
            len(candidate_neighbors) - reciprocal
        )
        profile["candidate_neighbor_fraction"] = (
            profile["candidate_edge_sample_count"]
            / profile["sampled_offdiag_edge_count"]
            if profile["sampled_offdiag_edge_count"]
            else 0.0
        )
        profile["reciprocal_candidate_neighbor_fraction"] = (
            reciprocal / len(candidate_neighbors) if candidate_neighbors else 0.0
        )
        profile["negative_gradient_edge_fraction"] = (
            profile["negative_gradient_edge_count"]
            / profile["sampled_offdiag_edge_count"]
            if profile["sampled_offdiag_edge_count"]
            else 0.0
        )
        profile["candidate_negative_gradient_edge_fraction"] = (
            profile["candidate_negative_gradient_edge_count"]
            / profile["candidate_gradient_edge_count"]
            if profile["candidate_gradient_edge_count"]
            else 0.0
        )
        profile["candidate_gradient_gram_abs_fraction"] = (
            profile["candidate_gradient_gram_abs_sum"]
            / profile["total_gradient_gram_abs_sum"]
            if profile["total_gradient_gram_abs_sum"]
            else 0.0
        )
        profile["gradient_gram_abs_concentration"] = (
            profile["max_abs_gradient_gram"]
            / profile["total_gradient_gram_abs_sum"]
            if profile["total_gradient_gram_abs_sum"]
            else 0.0
        )
        profile["matrix_gradient_sign_mismatch_fraction"] = (
            profile["matrix_gradient_sign_mismatch_count"]
            / profile["sampled_offdiag_edge_count"]
            if profile["sampled_offdiag_edge_count"]
            else 0.0
        )
        profile["mean_abs_gradient_cosine"] = (
            profile["abs_gradient_cosine_sum"]
            / profile["sampled_offdiag_edge_count"]
            if profile["sampled_offdiag_edge_count"]
            else 0.0
        )
        profile["candidate_component_size"] = component_sizes.get(row, 0)
        profile["candidate_component_fraction"] = (
            profile["candidate_component_size"] / profiled_candidate_count
            if profiled_candidate_count
            else 0.0
        )
        profile["candidate_graph_two_hop_count"] = two_hop_count(
            row,
            directed_candidate_adjacency,
        )
        profile["candidate_graph_clustering"] = graph_clustering(
            row,
            undirected_adjacency,
        )
        if (
            profile["negative_gradient_edge_fraction"] == 1.0
            and profile["reciprocal_candidate_neighbor_fraction"] == 1.0
            and profile["candidate_neighbor_count"] > 0
        ):
            edge_class = "reciprocal_all_negative_gradient_stencil"
        elif profile["candidate_neighbor_count"] > 0:
            edge_class = "candidate_gradient_stencil"
        else:
            edge_class = "missing_candidate_gradient_stencil"
        profile["gradient_column_edge_class"] = edge_class
        profiles[row] = profile
    return profiles


def metric_values(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in candidate_rows:
        profile = profiles.get(row)
        if not profile:
            continue
        value = profile.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def case_thresholds(
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
) -> dict[str, float | None]:
    keys = [
        "candidate_edge_sample_count",
        "candidate_neighbor_count",
        "candidate_neighbor_fraction",
        "candidate_graph_two_hop_count",
        "candidate_graph_clustering",
        "candidate_component_fraction",
        "candidate_gradient_gram_abs_fraction",
        "gradient_gram_abs_concentration",
        "mean_abs_gradient_cosine",
        "max_abs_gradient_cosine",
        "matrix_gradient_sign_mismatch_fraction",
    ]
    thresholds: dict[str, float | None] = {}
    for key in keys:
        values = metric_values(profiles, candidate_rows, key)
        thresholds[f"{key}_p10"] = LM.percentile(values, 0.10)
        thresholds[f"{key}_p25"] = LM.percentile(values, 0.25)
        thresholds[f"{key}_p75"] = LM.percentile(values, 0.75)
        thresholds[f"{key}_p90"] = LM.percentile(values, 0.90)
    return thresholds


def le(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) <= threshold


def ge(profile: dict[str, Any], key: str, threshold: float | None) -> bool:
    return threshold is not None and safe_float(profile.get(key)) >= threshold


def tail_selector(
    profile: dict[str, Any],
    key: str,
    thresholds: dict[str, float | None],
) -> bool:
    return le(profile, key, thresholds[f"{key}_p25"]) or ge(
        profile,
        key,
        thresholds[f"{key}_p75"],
    )


def selector_definitions(
    thresholds: dict[str, float | None],
) -> list[dict[str, Any]]:
    return [
        {
            "key": "gradient_column_graph_profiled_candidate",
            "description": "Preferred candidates with sampled gradient-column graph profiles.",
            "threshold_key": None,
            "predicate": lambda profile: profile["sampled_offdiag_edge_count"] > 0,
        },
        {
            "key": "gradient_column_graph_reciprocal_negative_stencil",
            "description": "Rows with reciprocal candidate edges and all sampled gradient Gram edges negative.",
            "threshold_key": "fixed:reciprocal_all_negative_gradient_stencil",
            "predicate": lambda profile: (
                profile["gradient_column_edge_class"]
                == "reciprocal_all_negative_gradient_stencil"
            ),
        },
        {
            "key": "gradient_column_graph_single_component",
            "description": "Rows in one sampled candidate gradient-column graph component.",
            "threshold_key": "fixed:candidate_component_fraction_eq_1",
            "predicate": lambda profile: profile["candidate_component_fraction"] == 1.0,
        },
        {
            "key": "gradient_column_graph_edge_count_tail",
            "description": "Rows in either low or high candidate gradient-edge count tail.",
            "threshold_key": (
                "candidate_edge_sample_count_p25|candidate_edge_sample_count_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_edge_sample_count",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_neighbor_fraction_tail",
            "description": "Rows in either low or high candidate-edge fraction tail.",
            "threshold_key": (
                "candidate_neighbor_fraction_p25|candidate_neighbor_fraction_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_neighbor_fraction",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_neighbor_count_tail",
            "description": "Rows in either low or high unique candidate-neighbor count tail.",
            "threshold_key": "candidate_neighbor_count_p25|candidate_neighbor_count_p75",
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_neighbor_count",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_two_hop_tail",
            "description": "Rows in either low or high sampled gradient-column two-hop tail.",
            "threshold_key": (
                "candidate_graph_two_hop_count_p25|"
                "candidate_graph_two_hop_count_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_graph_two_hop_count",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_clustering_tail",
            "description": "Rows in either low or high sampled gradient-column clustering tail.",
            "threshold_key": (
                "candidate_graph_clustering_p25|candidate_graph_clustering_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_graph_clustering",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_component_fraction_tail",
            "description": "Rows in either low or high sampled component-fraction tail.",
            "threshold_key": (
                "candidate_component_fraction_p25|candidate_component_fraction_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_component_fraction",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_candidate_gram_fraction_tail",
            "description": "Rows in either low or high candidate-gradient-Gram fraction tail.",
            "threshold_key": (
                "candidate_gradient_gram_abs_fraction_p25|"
                "candidate_gradient_gram_abs_fraction_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "candidate_gradient_gram_abs_fraction",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_gram_concentration_tail",
            "description": "Rows in either low or high edge-level gradient-Gram concentration tail.",
            "threshold_key": (
                "gradient_gram_abs_concentration_p25|"
                "gradient_gram_abs_concentration_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "gradient_gram_abs_concentration",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_abs_cosine_tail",
            "description": "Rows in either low or high sampled gradient-cosine tail.",
            "threshold_key": (
                "mean_abs_gradient_cosine_p25|mean_abs_gradient_cosine_p75"
            ),
            "predicate": lambda profile: tail_selector(
                profile,
                "mean_abs_gradient_cosine",
                thresholds,
            ),
        },
        {
            "key": "gradient_column_graph_sign_mismatch",
            "description": "Rows where sampled matrix-edge signs disagree with gradient Gram signs.",
            "threshold_key": "fixed:matrix_gradient_sign_mismatch_fraction_gt_0",
            "predicate": lambda profile: (
                profile["matrix_gradient_sign_mismatch_fraction"] > 0.0
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
            threshold_value = {
                key: thresholds.get(key) for key in threshold_key.split("|")
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


def count_by_key(profiles: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for profile in profiles:
        value = str(profile.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def profile_summary(
    *,
    profiles: dict[int, dict[str, Any]],
    candidate_rows: list[int],
    target_rows: list[int],
) -> dict[str, Any]:
    candidate_set = set(candidate_rows)
    target_set = set(target_rows)
    profiled_candidates = [row for row in candidate_rows if row in profiles]
    target_profiles = {str(row): profiles[row] for row in target_rows if row in profiles}
    profiled_target_rows = target_set.intersection(profiles)
    return {
        "profiled_row_count": len(profiles),
        "profiled_candidate_count": len(profiled_candidates),
        "profiled_target_count": len(profiled_target_rows),
        "unprofiled_candidate_count": len(candidate_set.difference(profiles)),
        "unprofiled_target_global_dofs": [
            row for row in target_rows if row not in profiles
        ],
        "candidate_edge_class_counts": count_by_key(
            [profiles[row] for row in profiled_candidates],
            "gradient_column_edge_class",
        ),
        "target_edge_class_counts": count_by_key(
            [profiles[row] for row in target_rows if row in profiles],
            "gradient_column_edge_class",
        ),
        "target_profiles": target_profiles,
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
    log_paths = GB.default_gradient_log_paths(emission_cases, explicit_logs or [])

    cases: dict[str, dict[str, Any]] = {}
    selector_defs_by_case: dict[str, list[dict[str, Any]]] = {}
    for label, target_rows in target_cases.items():
        emission_case = emission_cases.get(label, {})
        candidate_rows = LM.int_list(emission_case.get(candidate_key))
        log_path = log_paths.get(label, Path(""))
        entries, evidence = GB.latest_gradient_balance_batch(
            log_path,
            operator=operator,
            test_field=test_field,
            trial_field=trial_field,
        )
        profiles = gradient_column_graph_profiles(
            entries=entries,
            candidate_rows=candidate_rows,
        )
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
    selector_count = (
        len(selector_defs_by_case[first_label]) if first_label is not None else 0
    )
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
        label for label, case in cases.items() if case["log_evidence"].get("status") != "ok"
    ]

    if missing_cases:
        finding = "direct_pspg_cut_volume_gradient_column_graph_evidence_missing"
        next_requirement = (
            "Regenerate Test02/Test10 short replay logs with "
            "SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_GRADIENT_BALANCE_DIAGNOSTIC=1 and "
            f"SVMP_FE_CUT_VOLUME_LOCAL_MATRIX_PROVENANCE_OPERATOR={operator}."
        )
    elif selective:
        finding = "direct_pspg_cut_volume_gradient_column_graph_selector_selective"
        next_requirement = (
            "Prototype the selective sampled pressure-gradient edge topology gate "
            "inside the direct PSPG formulation and run the short Test02/Test10 windows."
        )
    elif overbroad or misses:
        finding = (
            "direct_pspg_cut_volume_gradient_column_graph_selectors_"
            "overbroad_or_miss_targets"
        )
        next_requirement = (
            "Do not promote sampled pressure-gradient edge topology, reciprocity, "
            "component, Gram-fraction, or cosine tails directly; the remaining "
            "rule must be stronger than sampled edge-level graph thresholding."
        )
    else:
        finding = "direct_pspg_cut_volume_gradient_column_graph_selectivity_inconclusive"
        next_requirement = (
            "Regenerate edge-level gradient-balance logs before selecting a "
            "formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for edge-level sampled pressure-gradient Gram/"
            "cosine column graph topology in direct PSPG cut-volume rows."
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
