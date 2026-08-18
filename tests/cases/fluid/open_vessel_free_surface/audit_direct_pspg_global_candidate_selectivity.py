#!/usr/bin/env python3
"""Audit selectivity of globally emitted direct PSPG formulation candidates."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare globally emitted direct PSPG formulation candidates with "
            "audited target rows and classify whether the raw selector is "
            "narrow enough to promote."
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


def as_count(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def target_counts(target_map: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if not isinstance(label, str):
            continue
        counts[label] = len(as_list(case.get("direct_pspg_target_global_dofs")))
    return counts


def ratio_exceeds(value: float | None, limit: float) -> bool:
    return value is not None and value > limit


def gate_finding(
    *,
    has_evidence: bool,
    covered_target_count: int,
    target_count: int,
    candidate_ratio: float | None,
    max_target_ratio: float,
    prefix: str,
) -> str:
    if not has_evidence:
        return f"{prefix}_missing"
    if covered_target_count != target_count:
        return f"{prefix}_misses_targets"
    if ratio_exceeds(candidate_ratio, max_target_ratio):
        return f"{prefix}_overbroad"
    return f"{prefix}_selective"


def aggregate_gate_finding(case_findings: dict[str, Any], *, prefix: str) -> str:
    findings = list(case_findings.values())
    if any(finding == f"{prefix}_missing" for finding in findings):
        return f"{prefix}_missing"
    if any(finding == f"{prefix}_misses_targets" for finding in findings):
        return f"{prefix}_misses_targets"
    if any(finding == f"{prefix}_overbroad" for finding in findings):
        return f"{prefix}_overbroad"
    if findings and all(finding == f"{prefix}_selective" for finding in findings):
        return f"{prefix}_selective"
    return f"{prefix}_inconclusive"


def evaluate_case(
    *,
    case: dict[str, Any],
    target_count: int,
    max_target_ratio: float,
) -> dict[str, Any]:
    preferred = as_count(case.get("preferred_candidate_count"))
    sparse = as_count(case.get("sparse_direct_self_candidate_count"))
    low_ratio = as_count(case.get("low_direct_self_ratio_candidate_count"))
    moderate_ratio = as_count(
        case.get("moderate_direct_self_ratio_candidate_count")
    )
    sparse_or_moderate_ratio = as_count(
        case.get("sparse_or_moderate_direct_self_ratio_candidate_count")
    )
    sparse_seeded_radius1 = as_count(
        case.get("sparse_seeded_pressure_action_radius1_candidate_count")
    )
    sparse_seeded_radius2 = as_count(
        case.get("sparse_seeded_pressure_action_radius2_candidate_count")
    )
    graph_local_low = as_count(
        case.get("graph_local_low_direct_self_ratio_candidate_count")
    )
    graph_local_moderate = as_count(
        case.get("graph_local_moderate_direct_self_ratio_candidate_count")
    )
    pressure_action_moderate_degree = as_count(
        case.get("pressure_action_moderate_degree_candidate_count")
    )
    pressure_action_moderate_sum = as_count(
        case.get("pressure_action_moderate_sum_ratio_candidate_count")
    )
    pressure_action_self_dominant = as_count(
        case.get("pressure_action_self_dominant_candidate_count")
    )
    matrix_action = as_count(case.get("matrix_pressure_action_covered_count"))
    matrix_isolated = as_count(case.get("matrix_pressure_action_isolated_count"))
    direct_positive = as_count(case.get("direct_self_positive_row_count"))
    covered_targets = len(as_list(case.get("covered_direct_target_global_dofs")))
    sparse_seeded_component = as_count(
        case.get("sparse_seeded_matrix_pressure_action_component_dof_count")
    )
    sparse_seeded_component_covered_targets = len(
        as_list(
            case.get(
                "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs"
            )
        )
    )
    low_ratio_covered_targets = len(
        as_list(case.get("low_direct_self_ratio_covered_direct_target_global_dofs"))
    )
    moderate_ratio_covered_targets = len(
        as_list(
            case.get("moderate_direct_self_ratio_covered_direct_target_global_dofs")
        )
    )
    sparse_or_moderate_ratio_covered_targets = len(
        as_list(
            case.get(
                "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs"
            )
        )
    )
    sparse_seeded_radius1_covered_targets = len(
        as_list(
            case.get(
                "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs"
            )
        )
    )
    sparse_seeded_radius2_covered_targets = len(
        as_list(
            case.get(
                "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs"
            )
        )
    )
    graph_local_low_covered_targets = len(
        as_list(
            case.get(
                "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs"
            )
        )
    )
    graph_local_moderate_covered_targets = len(
        as_list(
            case.get(
                "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs"
            )
        )
    )
    pressure_action_moderate_degree_covered_targets = len(
        as_list(
            case.get(
                "pressure_action_moderate_degree_covered_direct_target_global_dofs"
            )
        )
    )
    pressure_action_moderate_sum_covered_targets = len(
        as_list(
            case.get(
                "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs"
            )
        )
    )
    pressure_action_self_dominant_covered_targets = len(
        as_list(
            case.get(
                "pressure_action_self_dominant_covered_direct_target_global_dofs"
            )
        )
    )
    has_sparse_seeded_component_evidence = (
        "sparse_seeded_matrix_pressure_action_component_dof_count" in case
    )
    has_direct_self_ratio_evidence = (
        "sparse_or_moderate_direct_self_ratio_candidate_count" in case
    )
    has_sparse_seeded_radius_evidence = (
        "sparse_seeded_pressure_action_radius1_candidate_count" in case
    )
    has_graph_local_evidence = (
        "graph_local_moderate_direct_self_ratio_candidate_count" in case
    )
    has_pressure_action_topology_evidence = (
        "pressure_action_moderate_degree_candidate_count" in case
    )

    preferred_ratio = ratio(preferred, target_count)
    sparse_ratio = ratio(sparse, target_count)
    low_direct_self_ratio = ratio(low_ratio, target_count)
    moderate_direct_self_ratio = ratio(moderate_ratio, target_count)
    sparse_or_moderate_direct_self_ratio = ratio(
        sparse_or_moderate_ratio, target_count
    )
    sparse_seeded_radius1_ratio = ratio(sparse_seeded_radius1, target_count)
    sparse_seeded_radius2_ratio = ratio(sparse_seeded_radius2, target_count)
    graph_local_low_direct_self_ratio = ratio(graph_local_low, target_count)
    graph_local_moderate_direct_self_ratio = ratio(
        graph_local_moderate, target_count
    )
    pressure_action_moderate_degree_ratio = ratio(
        pressure_action_moderate_degree, target_count
    )
    pressure_action_moderate_sum_ratio = ratio(
        pressure_action_moderate_sum, target_count
    )
    pressure_action_self_dominant_ratio = ratio(
        pressure_action_self_dominant, target_count
    )
    matrix_ratio = ratio(matrix_action, target_count)
    sparse_seeded_component_ratio = ratio(sparse_seeded_component, target_count)
    matrix_covers_all_direct = (
        direct_positive > 0
        and matrix_action == direct_positive
        and matrix_isolated == 0
    )
    raw_preferred_overbroad = ratio_exceeds(preferred_ratio, max_target_ratio)
    raw_sparse_overbroad = ratio_exceeds(sparse_ratio, max_target_ratio)
    raw_matrix_overbroad = (
        ratio_exceeds(matrix_ratio, max_target_ratio)
        or matrix_covers_all_direct
    )
    sparse_seeded_component_overbroad = (
        has_sparse_seeded_component_evidence
        and ratio_exceeds(sparse_seeded_component_ratio, max_target_ratio)
    )
    sparse_seeded_component_covers_targets = (
        has_sparse_seeded_component_evidence
        and sparse_seeded_component_covered_targets == target_count
    )
    low_direct_self_ratio_covers_targets = (
        has_direct_self_ratio_evidence
        and low_ratio_covered_targets == target_count
    )
    moderate_direct_self_ratio_covers_targets = (
        has_direct_self_ratio_evidence
        and moderate_ratio_covered_targets == target_count
    )
    sparse_or_moderate_direct_self_ratio_covers_targets = (
        has_direct_self_ratio_evidence
        and sparse_or_moderate_ratio_covered_targets == target_count
    )
    sparse_or_moderate_direct_self_ratio_overbroad = (
        has_direct_self_ratio_evidence
        and ratio_exceeds(sparse_or_moderate_direct_self_ratio, max_target_ratio)
    )
    graph_local_low_direct_self_ratio_covers_targets = (
        has_graph_local_evidence
        and graph_local_low_covered_targets == target_count
    )
    graph_local_moderate_direct_self_ratio_covers_targets = (
        has_graph_local_evidence
        and graph_local_moderate_covered_targets == target_count
    )
    graph_local_moderate_direct_self_ratio_overbroad = (
        has_graph_local_evidence
        and ratio_exceeds(graph_local_moderate_direct_self_ratio, max_target_ratio)
    )

    if not has_direct_self_ratio_evidence:
        direct_self_support_ratio_gate_finding = "direct_self_support_ratio_gate_missing"
    elif not sparse_or_moderate_direct_self_ratio_covers_targets:
        direct_self_support_ratio_gate_finding = (
            "sparse_or_moderate_direct_self_ratio_gate_misses_targets"
        )
    elif sparse_or_moderate_direct_self_ratio_overbroad:
        direct_self_support_ratio_gate_finding = (
            "sparse_or_moderate_direct_self_ratio_gate_overbroad"
        )
    else:
        direct_self_support_ratio_gate_finding = (
            "sparse_or_moderate_direct_self_ratio_gate_selective"
        )

    if not has_graph_local_evidence:
        graph_local_support_ratio_gate_finding = (
            "graph_local_support_ratio_gate_missing"
        )
    elif not graph_local_moderate_direct_self_ratio_covers_targets:
        graph_local_support_ratio_gate_finding = (
            "graph_local_moderate_direct_self_ratio_gate_misses_targets"
        )
    elif graph_local_moderate_direct_self_ratio_overbroad:
        graph_local_support_ratio_gate_finding = (
            "graph_local_moderate_direct_self_ratio_gate_overbroad"
        )
    else:
        graph_local_support_ratio_gate_finding = (
            "graph_local_moderate_direct_self_ratio_gate_selective"
        )
    pressure_action_moderate_degree_gate_finding = gate_finding(
        has_evidence=has_pressure_action_topology_evidence,
        covered_target_count=pressure_action_moderate_degree_covered_targets,
        target_count=target_count,
        candidate_ratio=pressure_action_moderate_degree_ratio,
        max_target_ratio=max_target_ratio,
        prefix="pressure_action_moderate_degree_gate",
    )
    pressure_action_moderate_sum_ratio_gate_finding = gate_finding(
        has_evidence=has_pressure_action_topology_evidence,
        covered_target_count=pressure_action_moderate_sum_covered_targets,
        target_count=target_count,
        candidate_ratio=pressure_action_moderate_sum_ratio,
        max_target_ratio=max_target_ratio,
        prefix="pressure_action_moderate_sum_ratio_gate",
    )
    pressure_action_self_dominant_gate_finding = gate_finding(
        has_evidence=has_pressure_action_topology_evidence,
        covered_target_count=pressure_action_self_dominant_covered_targets,
        target_count=target_count,
        candidate_ratio=pressure_action_self_dominant_ratio,
        max_target_ratio=max_target_ratio,
        prefix="pressure_action_self_dominant_gate",
    )
    sparse_seeded_radius1_gate_finding = gate_finding(
        has_evidence=has_sparse_seeded_radius_evidence,
        covered_target_count=sparse_seeded_radius1_covered_targets,
        target_count=target_count,
        candidate_ratio=sparse_seeded_radius1_ratio,
        max_target_ratio=max_target_ratio,
        prefix="sparse_seeded_pressure_action_radius1_gate",
    )
    sparse_seeded_radius2_gate_finding = gate_finding(
        has_evidence=has_sparse_seeded_radius_evidence,
        covered_target_count=sparse_seeded_radius2_covered_targets,
        target_count=target_count,
        candidate_ratio=sparse_seeded_radius2_ratio,
        max_target_ratio=max_target_ratio,
        prefix="sparse_seeded_pressure_action_radius2_gate",
    )

    if case.get("finding") != "candidate_emitted_covers_audited_targets":
        finding = "candidate_emission_not_coverage_complete"
    elif (
        raw_preferred_overbroad
        or raw_sparse_overbroad
        or raw_matrix_overbroad
        or sparse_seeded_component_overbroad
    ):
        finding = "raw_global_candidate_selector_overbroad"
    else:
        finding = "raw_global_candidate_selector_selective"

    return {
        "label": case.get("label"),
        "finding": finding,
        "direct_target_count": target_count,
        "covered_direct_target_count": covered_targets,
        "preferred_candidate_count": preferred,
        "sparse_direct_self_candidate_count": sparse,
        "low_direct_self_ratio_candidate_count": low_ratio,
        "moderate_direct_self_ratio_candidate_count": moderate_ratio,
        "sparse_or_moderate_direct_self_ratio_candidate_count": (
            sparse_or_moderate_ratio
        ),
        "sparse_seeded_pressure_action_radius1_candidate_count": (
            sparse_seeded_radius1
        ),
        "sparse_seeded_pressure_action_radius2_candidate_count": (
            sparse_seeded_radius2
        ),
        "graph_local_low_direct_self_ratio_candidate_count": graph_local_low,
        "graph_local_moderate_direct_self_ratio_candidate_count": (
            graph_local_moderate
        ),
        "pressure_action_moderate_degree_candidate_count": (
            pressure_action_moderate_degree
        ),
        "pressure_action_moderate_sum_ratio_candidate_count": (
            pressure_action_moderate_sum
        ),
        "pressure_action_self_dominant_candidate_count": (
            pressure_action_self_dominant
        ),
        "matrix_pressure_action_covered_count": matrix_action,
        "matrix_pressure_action_isolated_count": matrix_isolated,
        "direct_self_positive_row_count": direct_positive,
        "sparse_seeded_matrix_pressure_action_component_dof_count": (
            sparse_seeded_component
        ),
        "sparse_seeded_matrix_pressure_action_component_covered_direct_target_count": (
            sparse_seeded_component_covered_targets
        ),
        "low_direct_self_ratio_covered_direct_target_count": (
            low_ratio_covered_targets
        ),
        "moderate_direct_self_ratio_covered_direct_target_count": (
            moderate_ratio_covered_targets
        ),
        "sparse_or_moderate_direct_self_ratio_covered_direct_target_count": (
            sparse_or_moderate_ratio_covered_targets
        ),
        "sparse_seeded_pressure_action_radius1_covered_direct_target_count": (
            sparse_seeded_radius1_covered_targets
        ),
        "sparse_seeded_pressure_action_radius2_covered_direct_target_count": (
            sparse_seeded_radius2_covered_targets
        ),
        "graph_local_low_direct_self_ratio_covered_direct_target_count": (
            graph_local_low_covered_targets
        ),
        "graph_local_moderate_direct_self_ratio_covered_direct_target_count": (
            graph_local_moderate_covered_targets
        ),
        "pressure_action_moderate_degree_covered_direct_target_count": (
            pressure_action_moderate_degree_covered_targets
        ),
        "pressure_action_moderate_sum_ratio_covered_direct_target_count": (
            pressure_action_moderate_sum_covered_targets
        ),
        "pressure_action_self_dominant_covered_direct_target_count": (
            pressure_action_self_dominant_covered_targets
        ),
        "preferred_to_target_ratio": preferred_ratio,
        "sparse_direct_self_to_target_ratio": sparse_ratio,
        "low_direct_self_ratio_to_target_ratio": low_direct_self_ratio,
        "moderate_direct_self_ratio_to_target_ratio": moderate_direct_self_ratio,
        "sparse_or_moderate_direct_self_ratio_to_target_ratio": (
            sparse_or_moderate_direct_self_ratio
        ),
        "sparse_seeded_pressure_action_radius1_to_target_ratio": (
            sparse_seeded_radius1_ratio
        ),
        "sparse_seeded_pressure_action_radius2_to_target_ratio": (
            sparse_seeded_radius2_ratio
        ),
        "graph_local_low_direct_self_ratio_to_target_ratio": (
            graph_local_low_direct_self_ratio
        ),
        "graph_local_moderate_direct_self_ratio_to_target_ratio": (
            graph_local_moderate_direct_self_ratio
        ),
        "pressure_action_moderate_degree_to_target_ratio": (
            pressure_action_moderate_degree_ratio
        ),
        "pressure_action_moderate_sum_ratio_to_target_ratio": (
            pressure_action_moderate_sum_ratio
        ),
        "pressure_action_self_dominant_to_target_ratio": (
            pressure_action_self_dominant_ratio
        ),
        "matrix_pressure_action_to_target_ratio": matrix_ratio,
        "sparse_seeded_matrix_pressure_action_component_to_target_ratio": (
            sparse_seeded_component_ratio
        ),
        "matrix_pressure_action_covers_all_direct_rows": matrix_covers_all_direct,
        "sparse_seeded_matrix_pressure_action_component_covers_targets": (
            sparse_seeded_component_covers_targets
        ),
        "low_direct_self_ratio_covers_targets": (
            low_direct_self_ratio_covers_targets
        ),
        "moderate_direct_self_ratio_covers_targets": (
            moderate_direct_self_ratio_covers_targets
        ),
        "sparse_or_moderate_direct_self_ratio_covers_targets": (
            sparse_or_moderate_direct_self_ratio_covers_targets
        ),
        "sparse_seeded_pressure_action_radius1_covers_targets": (
            sparse_seeded_radius1_covered_targets == target_count
            if has_sparse_seeded_radius_evidence
            else False
        ),
        "sparse_seeded_pressure_action_radius2_covers_targets": (
            sparse_seeded_radius2_covered_targets == target_count
            if has_sparse_seeded_radius_evidence
            else False
        ),
        "graph_local_low_direct_self_ratio_covers_targets": (
            graph_local_low_direct_self_ratio_covers_targets
        ),
        "graph_local_moderate_direct_self_ratio_covers_targets": (
            graph_local_moderate_direct_self_ratio_covers_targets
        ),
        "pressure_action_moderate_degree_covers_targets": (
            pressure_action_moderate_degree_covered_targets == target_count
            if has_pressure_action_topology_evidence
            else False
        ),
        "pressure_action_moderate_sum_ratio_covers_targets": (
            pressure_action_moderate_sum_covered_targets == target_count
            if has_pressure_action_topology_evidence
            else False
        ),
        "pressure_action_self_dominant_covers_targets": (
            pressure_action_self_dominant_covered_targets == target_count
            if has_pressure_action_topology_evidence
            else False
        ),
        "direct_self_support_ratio_gate_finding": (
            direct_self_support_ratio_gate_finding
        ),
        "graph_local_support_ratio_gate_finding": (
            graph_local_support_ratio_gate_finding
        ),
        "pressure_action_moderate_degree_gate_finding": (
            pressure_action_moderate_degree_gate_finding
        ),
        "pressure_action_moderate_sum_ratio_gate_finding": (
            pressure_action_moderate_sum_ratio_gate_finding
        ),
        "pressure_action_self_dominant_gate_finding": (
            pressure_action_self_dominant_gate_finding
        ),
        "sparse_seeded_pressure_action_radius1_gate_finding": (
            sparse_seeded_radius1_gate_finding
        ),
        "sparse_seeded_pressure_action_radius2_gate_finding": (
            sparse_seeded_radius2_gate_finding
        ),
        "raw_preferred_selector_overbroad": raw_preferred_overbroad,
        "raw_sparse_direct_self_selector_overbroad": raw_sparse_overbroad,
        "raw_matrix_pressure_action_selector_overbroad": raw_matrix_overbroad,
        "sparse_or_moderate_direct_self_ratio_selector_overbroad": (
            sparse_or_moderate_direct_self_ratio_overbroad
        ),
        "sparse_seeded_pressure_action_radius1_selector_overbroad": (
            ratio_exceeds(sparse_seeded_radius1_ratio, max_target_ratio)
        ),
        "sparse_seeded_pressure_action_radius2_selector_overbroad": (
            ratio_exceeds(sparse_seeded_radius2_ratio, max_target_ratio)
        ),
        "graph_local_moderate_direct_self_ratio_selector_overbroad": (
            graph_local_moderate_direct_self_ratio_overbroad
        ),
        "pressure_action_moderate_degree_selector_overbroad": ratio_exceeds(
            pressure_action_moderate_degree_ratio, max_target_ratio
        ),
        "pressure_action_moderate_sum_ratio_selector_overbroad": ratio_exceeds(
            pressure_action_moderate_sum_ratio, max_target_ratio
        ),
        "pressure_action_self_dominant_selector_overbroad": ratio_exceeds(
            pressure_action_self_dominant_ratio, max_target_ratio
        ),
        "sparse_seeded_matrix_pressure_action_component_selector_overbroad": (
            sparse_seeded_component_overbroad
        ),
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
    emission_cases = [
        case for case in as_list(global_emission.get("cases"))
        if isinstance(case, dict)
    ]
    cases = [
        evaluate_case(
            case=case,
            target_count=counts.get(str(case.get("label")), 0),
            max_target_ratio=max_target_ratio,
        )
        for case in emission_cases
    ]

    coverage_ready = (
        global_emission.get("finding") == "candidate_emission_covers_audited_targets"
        and bool(cases)
    )
    overbroad_cases = [
        case for case in cases
        if case["finding"] == "raw_global_candidate_selector_overbroad"
    ]
    selective_cases = [
        case for case in cases
        if case["finding"] == "raw_global_candidate_selector_selective"
    ]
    support_ratio_case_findings = {
        str(case.get("label")): case.get("direct_self_support_ratio_gate_finding")
        for case in cases
    }
    if any(
        finding == "direct_self_support_ratio_gate_missing"
        for finding in support_ratio_case_findings.values()
    ):
        direct_self_support_ratio_gate_finding = (
            "direct_self_support_ratio_gate_missing"
        )
    elif any(
        finding == "sparse_or_moderate_direct_self_ratio_gate_misses_targets"
        for finding in support_ratio_case_findings.values()
    ):
        direct_self_support_ratio_gate_finding = (
            "direct_self_support_ratio_gate_misses_targets"
        )
    elif any(
        finding == "sparse_or_moderate_direct_self_ratio_gate_overbroad"
        for finding in support_ratio_case_findings.values()
    ):
        direct_self_support_ratio_gate_finding = (
            "direct_self_support_ratio_gate_overbroad"
        )
    elif cases and all(
        finding == "sparse_or_moderate_direct_self_ratio_gate_selective"
        for finding in support_ratio_case_findings.values()
    ):
        direct_self_support_ratio_gate_finding = (
            "direct_self_support_ratio_gate_selective"
        )
    else:
        direct_self_support_ratio_gate_finding = (
            "direct_self_support_ratio_gate_inconclusive"
        )
    graph_local_case_findings = {
        str(case.get("label")): case.get("graph_local_support_ratio_gate_finding")
        for case in cases
    }
    if any(
        finding == "graph_local_support_ratio_gate_missing"
        for finding in graph_local_case_findings.values()
    ):
        graph_local_support_ratio_gate_finding = (
            "graph_local_support_ratio_gate_missing"
        )
    elif any(
        finding == "graph_local_moderate_direct_self_ratio_gate_misses_targets"
        for finding in graph_local_case_findings.values()
    ):
        graph_local_support_ratio_gate_finding = (
            "graph_local_support_ratio_gate_misses_targets"
        )
    elif any(
        finding == "graph_local_moderate_direct_self_ratio_gate_overbroad"
        for finding in graph_local_case_findings.values()
    ):
        graph_local_support_ratio_gate_finding = (
            "graph_local_support_ratio_gate_overbroad"
        )
    elif cases and all(
        finding == "graph_local_moderate_direct_self_ratio_gate_selective"
        for finding in graph_local_case_findings.values()
    ):
        graph_local_support_ratio_gate_finding = (
            "graph_local_support_ratio_gate_selective"
        )
    else:
        graph_local_support_ratio_gate_finding = (
            "graph_local_support_ratio_gate_inconclusive"
        )
    pressure_action_moderate_degree_case_findings = {
        str(case.get("label")): case.get(
            "pressure_action_moderate_degree_gate_finding"
        )
        for case in cases
    }
    pressure_action_moderate_sum_ratio_case_findings = {
        str(case.get("label")): case.get(
            "pressure_action_moderate_sum_ratio_gate_finding"
        )
        for case in cases
    }
    pressure_action_self_dominant_case_findings = {
        str(case.get("label")): case.get(
            "pressure_action_self_dominant_gate_finding"
        )
        for case in cases
    }
    sparse_seeded_pressure_action_radius1_case_findings = {
        str(case.get("label")): case.get(
            "sparse_seeded_pressure_action_radius1_gate_finding"
        )
        for case in cases
    }
    sparse_seeded_pressure_action_radius2_case_findings = {
        str(case.get("label")): case.get(
            "sparse_seeded_pressure_action_radius2_gate_finding"
        )
        for case in cases
    }
    pressure_action_moderate_degree_gate_finding = aggregate_gate_finding(
        pressure_action_moderate_degree_case_findings,
        prefix="pressure_action_moderate_degree_gate",
    )
    pressure_action_moderate_sum_ratio_gate_finding = aggregate_gate_finding(
        pressure_action_moderate_sum_ratio_case_findings,
        prefix="pressure_action_moderate_sum_ratio_gate",
    )
    pressure_action_self_dominant_gate_finding = aggregate_gate_finding(
        pressure_action_self_dominant_case_findings,
        prefix="pressure_action_self_dominant_gate",
    )
    sparse_seeded_pressure_action_radius1_gate_finding = aggregate_gate_finding(
        sparse_seeded_pressure_action_radius1_case_findings,
        prefix="sparse_seeded_pressure_action_radius1_gate",
    )
    sparse_seeded_pressure_action_radius2_gate_finding = aggregate_gate_finding(
        sparse_seeded_pressure_action_radius2_case_findings,
        prefix="sparse_seeded_pressure_action_radius2_gate",
    )

    if not coverage_ready:
        finding = "global_candidate_emission_not_ready_for_selectivity"
        next_requirement = (
            "Prove global candidate emission covers the audited direct PSPG "
            "targets before using selectivity evidence."
        )
    elif overbroad_cases:
        finding = (
            "global_candidate_selector_overbroad_matrix_proxy_not_formulation_ready"
        )
        next_requirement = (
            "Do not promote raw global emitted candidates. Add a formulation-side "
            "physical provenance gate, such as active PSPG pressure-gradient "
            "support topology, boundary-support deficiency, or coupled patch "
            "structure, then replay Test02/Test10."
        )
    elif len(selective_cases) == len(cases):
        finding = "global_candidate_selector_selective_for_formulation_replay"
        next_requirement = (
            "Prototype the selective formulation-side candidate and replay the "
            "short Test02/Test10 windows."
        )
    else:
        finding = "global_candidate_selector_selectivity_inconclusive"
        next_requirement = (
            "Regenerate the global candidate emission artifact with full counts "
            "and target coverage before choosing a formulation replay."
        )

    return {
        "scope": (
            "Selectivity audit for globally emitted direct PSPG formulation "
            "candidate sets."
        ),
        "global_emission_path": (
            str(global_emission_path) if global_emission_path is not None else None
        ),
        "target_map_path": str(target_map_path) if target_map_path is not None else None,
        "max_target_ratio": max_target_ratio,
        "finding": finding,
        "direct_self_support_ratio_gate_finding": (
            direct_self_support_ratio_gate_finding
        ),
        "direct_self_support_ratio_case_findings": support_ratio_case_findings,
        "graph_local_support_ratio_gate_finding": (
            graph_local_support_ratio_gate_finding
        ),
        "graph_local_support_ratio_case_findings": graph_local_case_findings,
        "pressure_action_moderate_degree_gate_finding": (
            pressure_action_moderate_degree_gate_finding
        ),
        "pressure_action_moderate_degree_case_findings": (
            pressure_action_moderate_degree_case_findings
        ),
        "pressure_action_moderate_sum_ratio_gate_finding": (
            pressure_action_moderate_sum_ratio_gate_finding
        ),
        "pressure_action_moderate_sum_ratio_case_findings": (
            pressure_action_moderate_sum_ratio_case_findings
        ),
        "pressure_action_self_dominant_gate_finding": (
            pressure_action_self_dominant_gate_finding
        ),
        "pressure_action_self_dominant_case_findings": (
            pressure_action_self_dominant_case_findings
        ),
        "sparse_seeded_pressure_action_radius1_gate_finding": (
            sparse_seeded_pressure_action_radius1_gate_finding
        ),
        "sparse_seeded_pressure_action_radius1_case_findings": (
            sparse_seeded_pressure_action_radius1_case_findings
        ),
        "sparse_seeded_pressure_action_radius2_gate_finding": (
            sparse_seeded_pressure_action_radius2_gate_finding
        ),
        "sparse_seeded_pressure_action_radius2_case_findings": (
            sparse_seeded_pressure_action_radius2_case_findings
        ),
        "case_count": len(cases),
        "cases": cases,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    global_emission = load_json(args.global_emission_json)
    target_map = load_json(args.target_map_json)
    report = build_report(
        global_emission=global_emission,
        target_map=target_map,
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
