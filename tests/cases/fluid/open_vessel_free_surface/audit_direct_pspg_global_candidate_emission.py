#!/usr/bin/env python3
"""Audit direct PSPG formulation candidate emission from replay logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_TARGET_MAP = (
    DEFAULT_ARTIFACT_ROOT / "test02_test10_direct_pspg_formulation_target_20260606.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse direct_pspg_formulation_candidate diagnostics from short "
            "Test02/Test10 replay logs and compare emitted candidates with the "
            "audited direct PSPG target rows."
        )
    )
    parser.add_argument("--target-map-json", type=Path, default=DEFAULT_TARGET_MAP)
    parser.add_argument(
        "--log",
        action="append",
        type=str,
        default=[],
        help="Case-labelled log path as label=/path/to/run.log.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_scalar(value: str) -> Any:
    if value in {"none", ""}:
        return [] if value == "none" else value
    if "|" in value:
        return parse_dof_list(value)
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_dof_list(value: str) -> list[int | str]:
    if value in {"", "none"}:
        return []
    parsed: list[int | str] = []
    for token in value.split("|"):
        if token == "...":
            parsed.append(token)
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            parsed.append(token)
    return parsed


def parse_key_values(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for token in shlex.split(line):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        result[key] = parse_scalar(value)
    return result


def candidate_entries(log_path: Path) -> list[dict[str, Any]]:
    entries = []
    if not log_path.exists():
        return entries
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "diagnostic=direct_pspg_formulation_candidate" not in line:
            continue
        entries.append(parse_key_values(line))
    return entries


def target_case_map(target_map: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for case in as_list(target_map.get("cases")):
        if not isinstance(case, dict):
            continue
        label = case.get("label")
        if isinstance(label, str):
            cases[label] = case
    return cases


def parse_log_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, path = value.split("=", 1)
    return label, Path(path)


def latest_ok_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok_entries = [entry for entry in entries if entry.get("status") == "ok"]
    if not ok_entries:
        return None
    return ok_entries[-1]


def int_list_without_truncation(value: Any) -> tuple[list[int], bool]:
    if isinstance(value, int):
        return [value], False
    values = as_list(value)
    truncated = "..." in values
    ints = [item for item in values if isinstance(item, int)]
    return ints, truncated


def evaluate_case(
    *,
    label: str,
    log_path: Path,
    target_case: dict[str, Any],
) -> dict[str, Any]:
    entries = candidate_entries(log_path)
    latest = latest_ok_entry(entries)
    target_rows = as_list(target_case.get("direct_pspg_target_global_dofs"))
    if latest is None:
        return {
            "label": label,
            "path": str(log_path),
            "exists": log_path.exists(),
            "entry_count": len(entries),
            "finding": "direct_pspg_candidate_diagnostic_missing",
            "direct_target_count": len(target_rows),
            "covered_direct_target_global_dofs": [],
            "uncovered_direct_target_global_dofs": target_rows,
            "candidate_list_truncated": False,
        }

    candidates, truncated = int_list_without_truncation(
        latest.get("preferred_candidate_global_dofs")
    )
    high_row_sum_leak_candidates, high_row_sum_leak_truncated = (
        int_list_without_truncation(
            latest.get("high_direct_self_row_sum_leak_global_dofs")
        )
    )
    null_preserving_candidates, null_preserving_truncated = (
        int_list_without_truncation(
            latest.get("null_preserving_direct_self_global_dofs")
        )
    )
    diag_dominant_candidates, diag_dominant_truncated = (
        int_list_without_truncation(
            latest.get("diag_dominant_direct_self_global_dofs")
        )
    )
    balanced_diag_candidates, balanced_diag_truncated = (
        int_list_without_truncation(
            latest.get("balanced_diag_direct_self_global_dofs")
        )
    )
    sparse_candidates, sparse_truncated = int_list_without_truncation(
        latest.get("sparse_direct_self_global_dofs")
    )
    constrained_pressure_neighbor_candidates, constrained_pressure_neighbor_truncated = (
        int_list_without_truncation(
            latest.get("constrained_pressure_neighbor_global_dofs")
        )
    )
    (
        high_constrained_pressure_neighbor_ratio_candidates,
        high_constrained_pressure_neighbor_ratio_truncated,
    ) = int_list_without_truncation(
        latest.get("high_constrained_pressure_neighbor_ratio_global_dofs")
    )
    sparse_unconstrained_direct_self_candidates, sparse_unconstrained_truncated = (
        int_list_without_truncation(
            latest.get("sparse_unconstrained_direct_self_global_dofs")
        )
    )
    (
        constrained_or_sparse_unconstrained_candidates,
        constrained_or_sparse_unconstrained_truncated,
    ) = int_list_without_truncation(
        latest.get("constrained_or_sparse_unconstrained_direct_self_global_dofs")
    )
    matrix_action_candidates, matrix_action_truncated = int_list_without_truncation(
        latest.get("matrix_pressure_action_covered_global_dofs")
    )
    pressure_action_low_degree_candidates, pressure_action_low_degree_truncated = (
        int_list_without_truncation(
            latest.get("pressure_action_low_degree_global_dofs")
        )
    )
    (
        pressure_action_moderate_degree_candidates,
        pressure_action_moderate_degree_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_moderate_degree_global_dofs")
    )
    pressure_action_low_sum_candidates, pressure_action_low_sum_truncated = (
        int_list_without_truncation(
            latest.get("pressure_action_low_sum_ratio_global_dofs")
        )
    )
    (
        pressure_action_moderate_sum_candidates,
        pressure_action_moderate_sum_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_moderate_sum_ratio_global_dofs")
    )
    (
        pressure_action_self_dominant_candidates,
        pressure_action_self_dominant_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_self_dominant_global_dofs")
    )
    (
        pressure_action_zero_two_hop_candidates,
        pressure_action_zero_two_hop_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_zero_two_hop_global_dofs")
    )
    (
        pressure_action_low_two_hop_candidates,
        pressure_action_low_two_hop_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_low_two_hop_global_dofs")
    )
    (
        pressure_action_high_two_hop_candidates,
        pressure_action_high_two_hop_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_high_two_hop_global_dofs")
    )
    (
        pressure_action_zero_clustering_candidates,
        pressure_action_zero_clustering_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_zero_clustering_global_dofs")
    )
    (
        pressure_action_low_clustering_candidates,
        pressure_action_low_clustering_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_low_clustering_global_dofs")
    )
    (
        pressure_action_high_clustering_candidates,
        pressure_action_high_clustering_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_high_clustering_global_dofs")
    )
    (
        pressure_action_articulation_candidates,
        pressure_action_articulation_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_articulation_global_dofs")
    )
    (
        pressure_action_bridge_endpoint_candidates,
        pressure_action_bridge_endpoint_truncated,
    ) = int_list_without_truncation(
        latest.get("pressure_action_bridge_endpoint_global_dofs")
    )
    sparse_seeded_components, sparse_seeded_components_truncated = (
        int_list_without_truncation(
            latest.get("sparse_seeded_matrix_pressure_action_component_global_dofs")
        )
    )
    residual_sign_action_candidates, residual_sign_action_truncated = (
        int_list_without_truncation(
            latest.get("residual_sign_pressure_action_covered_global_dofs")
        )
    )
    (
        sparse_seeded_residual_sign_components,
        sparse_seeded_residual_sign_components_truncated,
    ) = int_list_without_truncation(
        latest.get("sparse_seeded_residual_sign_pressure_action_component_global_dofs")
    )
    (
        sparse_or_residual_sign_candidates,
        sparse_or_residual_sign_truncated,
    ) = int_list_without_truncation(
        latest.get("sparse_direct_self_or_residual_sign_pressure_action_global_dofs")
    )
    low_ratio_candidates, low_ratio_truncated = int_list_without_truncation(
        latest.get("low_direct_self_ratio_global_dofs")
    )
    moderate_ratio_candidates, moderate_ratio_truncated = int_list_without_truncation(
        latest.get("moderate_direct_self_ratio_global_dofs")
    )
    sparse_or_moderate_candidates, sparse_or_moderate_truncated = (
        int_list_without_truncation(
            latest.get("sparse_or_moderate_direct_self_ratio_global_dofs")
        )
    )
    sparse_seeded_radius1_candidates, sparse_seeded_radius1_truncated = (
        int_list_without_truncation(
            latest.get("sparse_seeded_pressure_action_radius1_global_dofs")
        )
    )
    sparse_seeded_radius2_candidates, sparse_seeded_radius2_truncated = (
        int_list_without_truncation(
            latest.get("sparse_seeded_pressure_action_radius2_global_dofs")
        )
    )
    graph_local_low_candidates, graph_local_low_truncated = (
        int_list_without_truncation(
            latest.get("graph_local_low_direct_self_ratio_global_dofs")
        )
    )
    graph_local_moderate_candidates, graph_local_moderate_truncated = (
        int_list_without_truncation(
            latest.get("graph_local_moderate_direct_self_ratio_global_dofs")
        )
    )
    candidate_set = set(candidates)
    high_row_sum_leak_set = set(high_row_sum_leak_candidates)
    null_preserving_set = set(null_preserving_candidates)
    diag_dominant_set = set(diag_dominant_candidates)
    balanced_diag_set = set(balanced_diag_candidates)
    sparse_seeded_component_set = set(sparse_seeded_components)
    constrained_pressure_neighbor_set = set(constrained_pressure_neighbor_candidates)
    high_constrained_pressure_neighbor_ratio_set = set(
        high_constrained_pressure_neighbor_ratio_candidates
    )
    sparse_unconstrained_direct_self_set = set(
        sparse_unconstrained_direct_self_candidates
    )
    constrained_or_sparse_unconstrained_set = set(
        constrained_or_sparse_unconstrained_candidates
    )
    low_ratio_candidate_set = set(low_ratio_candidates)
    moderate_ratio_candidate_set = set(moderate_ratio_candidates)
    sparse_or_moderate_candidate_set = set(sparse_or_moderate_candidates)
    sparse_seeded_radius1_candidate_set = set(sparse_seeded_radius1_candidates)
    sparse_seeded_radius2_candidate_set = set(sparse_seeded_radius2_candidates)
    graph_local_low_candidate_set = set(graph_local_low_candidates)
    graph_local_moderate_candidate_set = set(graph_local_moderate_candidates)
    pressure_action_low_degree_set = set(pressure_action_low_degree_candidates)
    pressure_action_moderate_degree_set = set(
        pressure_action_moderate_degree_candidates
    )
    pressure_action_low_sum_set = set(pressure_action_low_sum_candidates)
    pressure_action_moderate_sum_set = set(pressure_action_moderate_sum_candidates)
    pressure_action_self_dominant_set = set(
        pressure_action_self_dominant_candidates
    )
    pressure_action_zero_two_hop_set = set(
        pressure_action_zero_two_hop_candidates
    )
    pressure_action_low_two_hop_set = set(pressure_action_low_two_hop_candidates)
    pressure_action_high_two_hop_set = set(
        pressure_action_high_two_hop_candidates
    )
    pressure_action_zero_clustering_set = set(
        pressure_action_zero_clustering_candidates
    )
    pressure_action_low_clustering_set = set(
        pressure_action_low_clustering_candidates
    )
    pressure_action_high_clustering_set = set(
        pressure_action_high_clustering_candidates
    )
    pressure_action_articulation_set = set(pressure_action_articulation_candidates)
    pressure_action_bridge_endpoint_set = set(
        pressure_action_bridge_endpoint_candidates
    )
    residual_sign_action_set = set(residual_sign_action_candidates)
    sparse_seeded_residual_sign_component_set = set(
        sparse_seeded_residual_sign_components
    )
    sparse_or_residual_sign_set = set(sparse_or_residual_sign_candidates)
    covered = [row for row in target_rows if row in candidate_set]
    uncovered = [row for row in target_rows if row not in candidate_set]
    high_row_sum_leak_covered = [
        row for row in target_rows if row in high_row_sum_leak_set
    ]
    high_row_sum_leak_uncovered = [
        row for row in target_rows if row not in high_row_sum_leak_set
    ]
    null_preserving_covered = [
        row for row in target_rows if row in null_preserving_set
    ]
    null_preserving_uncovered = [
        row for row in target_rows if row not in null_preserving_set
    ]
    diag_dominant_covered = [
        row for row in target_rows if row in diag_dominant_set
    ]
    diag_dominant_uncovered = [
        row for row in target_rows if row not in diag_dominant_set
    ]
    balanced_diag_covered = [
        row for row in target_rows if row in balanced_diag_set
    ]
    balanced_diag_uncovered = [
        row for row in target_rows if row not in balanced_diag_set
    ]
    sparse_seeded_component_covered = [
        row for row in target_rows if row in sparse_seeded_component_set
    ]
    sparse_seeded_component_uncovered = [
        row for row in target_rows if row not in sparse_seeded_component_set
    ]
    residual_sign_action_covered = [
        row for row in target_rows if row in residual_sign_action_set
    ]
    residual_sign_action_uncovered = [
        row for row in target_rows if row not in residual_sign_action_set
    ]
    sparse_seeded_residual_sign_component_covered = [
        row for row in target_rows if row in sparse_seeded_residual_sign_component_set
    ]
    sparse_seeded_residual_sign_component_uncovered = [
        row
        for row in target_rows
        if row not in sparse_seeded_residual_sign_component_set
    ]
    sparse_or_residual_sign_covered = [
        row for row in target_rows if row in sparse_or_residual_sign_set
    ]
    sparse_or_residual_sign_uncovered = [
        row for row in target_rows if row not in sparse_or_residual_sign_set
    ]
    constrained_pressure_neighbor_covered = [
        row for row in target_rows if row in constrained_pressure_neighbor_set
    ]
    constrained_pressure_neighbor_uncovered = [
        row for row in target_rows if row not in constrained_pressure_neighbor_set
    ]
    high_constrained_pressure_neighbor_ratio_covered = [
        row
        for row in target_rows
        if row in high_constrained_pressure_neighbor_ratio_set
    ]
    high_constrained_pressure_neighbor_ratio_uncovered = [
        row
        for row in target_rows
        if row not in high_constrained_pressure_neighbor_ratio_set
    ]
    sparse_unconstrained_direct_self_covered = [
        row for row in target_rows if row in sparse_unconstrained_direct_self_set
    ]
    sparse_unconstrained_direct_self_uncovered = [
        row for row in target_rows if row not in sparse_unconstrained_direct_self_set
    ]
    constrained_or_sparse_unconstrained_covered = [
        row for row in target_rows if row in constrained_or_sparse_unconstrained_set
    ]
    constrained_or_sparse_unconstrained_uncovered = [
        row
        for row in target_rows
        if row not in constrained_or_sparse_unconstrained_set
    ]
    low_ratio_covered = [
        row for row in target_rows if row in low_ratio_candidate_set
    ]
    low_ratio_uncovered = [
        row for row in target_rows if row not in low_ratio_candidate_set
    ]
    moderate_ratio_covered = [
        row for row in target_rows if row in moderate_ratio_candidate_set
    ]
    moderate_ratio_uncovered = [
        row for row in target_rows if row not in moderate_ratio_candidate_set
    ]
    sparse_or_moderate_covered = [
        row for row in target_rows if row in sparse_or_moderate_candidate_set
    ]
    sparse_or_moderate_uncovered = [
        row for row in target_rows if row not in sparse_or_moderate_candidate_set
    ]
    sparse_seeded_radius1_covered = [
        row for row in target_rows if row in sparse_seeded_radius1_candidate_set
    ]
    sparse_seeded_radius1_uncovered = [
        row for row in target_rows if row not in sparse_seeded_radius1_candidate_set
    ]
    sparse_seeded_radius2_covered = [
        row for row in target_rows if row in sparse_seeded_radius2_candidate_set
    ]
    sparse_seeded_radius2_uncovered = [
        row for row in target_rows if row not in sparse_seeded_radius2_candidate_set
    ]
    graph_local_low_covered = [
        row for row in target_rows if row in graph_local_low_candidate_set
    ]
    graph_local_low_uncovered = [
        row for row in target_rows if row not in graph_local_low_candidate_set
    ]
    graph_local_moderate_covered = [
        row for row in target_rows if row in graph_local_moderate_candidate_set
    ]
    graph_local_moderate_uncovered = [
        row for row in target_rows if row not in graph_local_moderate_candidate_set
    ]
    pressure_action_low_degree_covered = [
        row for row in target_rows if row in pressure_action_low_degree_set
    ]
    pressure_action_low_degree_uncovered = [
        row for row in target_rows if row not in pressure_action_low_degree_set
    ]
    pressure_action_moderate_degree_covered = [
        row for row in target_rows if row in pressure_action_moderate_degree_set
    ]
    pressure_action_moderate_degree_uncovered = [
        row for row in target_rows if row not in pressure_action_moderate_degree_set
    ]
    pressure_action_low_sum_covered = [
        row for row in target_rows if row in pressure_action_low_sum_set
    ]
    pressure_action_low_sum_uncovered = [
        row for row in target_rows if row not in pressure_action_low_sum_set
    ]
    pressure_action_moderate_sum_covered = [
        row for row in target_rows if row in pressure_action_moderate_sum_set
    ]
    pressure_action_moderate_sum_uncovered = [
        row for row in target_rows if row not in pressure_action_moderate_sum_set
    ]
    pressure_action_self_dominant_covered = [
        row for row in target_rows if row in pressure_action_self_dominant_set
    ]
    pressure_action_self_dominant_uncovered = [
        row for row in target_rows if row not in pressure_action_self_dominant_set
    ]
    pressure_action_zero_two_hop_covered = [
        row for row in target_rows if row in pressure_action_zero_two_hop_set
    ]
    pressure_action_zero_two_hop_uncovered = [
        row for row in target_rows if row not in pressure_action_zero_two_hop_set
    ]
    pressure_action_low_two_hop_covered = [
        row for row in target_rows if row in pressure_action_low_two_hop_set
    ]
    pressure_action_low_two_hop_uncovered = [
        row for row in target_rows if row not in pressure_action_low_two_hop_set
    ]
    pressure_action_high_two_hop_covered = [
        row for row in target_rows if row in pressure_action_high_two_hop_set
    ]
    pressure_action_high_two_hop_uncovered = [
        row for row in target_rows if row not in pressure_action_high_two_hop_set
    ]
    pressure_action_zero_clustering_covered = [
        row for row in target_rows if row in pressure_action_zero_clustering_set
    ]
    pressure_action_zero_clustering_uncovered = [
        row for row in target_rows if row not in pressure_action_zero_clustering_set
    ]
    pressure_action_low_clustering_covered = [
        row for row in target_rows if row in pressure_action_low_clustering_set
    ]
    pressure_action_low_clustering_uncovered = [
        row for row in target_rows if row not in pressure_action_low_clustering_set
    ]
    pressure_action_high_clustering_covered = [
        row for row in target_rows if row in pressure_action_high_clustering_set
    ]
    pressure_action_high_clustering_uncovered = [
        row for row in target_rows if row not in pressure_action_high_clustering_set
    ]
    pressure_action_articulation_covered = [
        row for row in target_rows if row in pressure_action_articulation_set
    ]
    pressure_action_articulation_uncovered = [
        row for row in target_rows if row not in pressure_action_articulation_set
    ]
    pressure_action_bridge_endpoint_covered = [
        row for row in target_rows if row in pressure_action_bridge_endpoint_set
    ]
    pressure_action_bridge_endpoint_uncovered = [
        row for row in target_rows if row not in pressure_action_bridge_endpoint_set
    ]
    if truncated and uncovered:
        finding = "candidate_emitted_but_coverage_sample_limited"
    elif uncovered:
        finding = "candidate_emitted_but_misses_targets"
    elif truncated:
        finding = "candidate_emitted_coverage_seen_but_list_truncated"
    else:
        finding = "candidate_emitted_covers_audited_targets"

    return {
        "label": label,
        "path": str(log_path),
        "exists": log_path.exists(),
        "entry_count": len(entries),
        "finding": finding,
        "selector": latest.get("selector"),
        "op": latest.get("op"),
        "phase": latest.get("phase"),
        "pressure_offset": latest.get("pressure_offset"),
        "pressure_dofs": latest.get("pressure_dofs"),
        "direct_target_count": len(target_rows),
        "direct_self_positive_row_count": latest.get(
            "direct_self_positive_row_count"
        ),
        "direct_self_row_sum_leak_threshold": latest.get(
            "direct_self_row_sum_leak_threshold"
        ),
        "direct_self_null_preserving_threshold": latest.get(
            "direct_self_null_preserving_threshold"
        ),
        "direct_self_diag_dominant_threshold": latest.get(
            "direct_self_diag_dominant_threshold"
        ),
        "direct_self_balanced_diag_low_threshold": latest.get(
            "direct_self_balanced_diag_low_threshold"
        ),
        "direct_self_balanced_diag_high_threshold": latest.get(
            "direct_self_balanced_diag_high_threshold"
        ),
        "max_direct_self_row_sum_leak_ratio": latest.get(
            "max_direct_self_row_sum_leak_ratio"
        ),
        "min_direct_self_diag_abs_ratio": latest.get(
            "min_direct_self_diag_abs_ratio"
        ),
        "max_direct_self_diag_abs_ratio": latest.get(
            "max_direct_self_diag_abs_ratio"
        ),
        "high_direct_self_row_sum_leak_candidate_count": latest.get(
            "high_direct_self_row_sum_leak_candidate_count"
        ),
        "high_direct_self_row_sum_leak_global_dofs": (
            high_row_sum_leak_candidates
        ),
        "high_direct_self_row_sum_leak_covered_direct_target_global_dofs": (
            high_row_sum_leak_covered
        ),
        "high_direct_self_row_sum_leak_uncovered_direct_target_global_dofs": (
            high_row_sum_leak_uncovered
        ),
        "high_direct_self_row_sum_leak_list_truncated": (
            high_row_sum_leak_truncated
        ),
        "null_preserving_direct_self_candidate_count": latest.get(
            "null_preserving_direct_self_candidate_count"
        ),
        "null_preserving_direct_self_global_dofs": null_preserving_candidates,
        "null_preserving_direct_self_covered_direct_target_global_dofs": (
            null_preserving_covered
        ),
        "null_preserving_direct_self_uncovered_direct_target_global_dofs": (
            null_preserving_uncovered
        ),
        "null_preserving_direct_self_list_truncated": (
            null_preserving_truncated
        ),
        "diag_dominant_direct_self_candidate_count": latest.get(
            "diag_dominant_direct_self_candidate_count"
        ),
        "diag_dominant_direct_self_global_dofs": diag_dominant_candidates,
        "diag_dominant_direct_self_covered_direct_target_global_dofs": (
            diag_dominant_covered
        ),
        "diag_dominant_direct_self_uncovered_direct_target_global_dofs": (
            diag_dominant_uncovered
        ),
        "diag_dominant_direct_self_list_truncated": diag_dominant_truncated,
        "balanced_diag_direct_self_candidate_count": latest.get(
            "balanced_diag_direct_self_candidate_count"
        ),
        "balanced_diag_direct_self_global_dofs": balanced_diag_candidates,
        "balanced_diag_direct_self_covered_direct_target_global_dofs": (
            balanced_diag_covered
        ),
        "balanced_diag_direct_self_uncovered_direct_target_global_dofs": (
            balanced_diag_uncovered
        ),
        "balanced_diag_direct_self_list_truncated": balanced_diag_truncated,
        "sparse_direct_self_candidate_count": latest.get(
            "sparse_direct_self_candidate_count"
        ),
        "sparse_direct_self_global_dofs": sparse_candidates,
        "sparse_direct_self_list_truncated": sparse_truncated,
        "max_unconstrained_direct_self_numeric_entries": latest.get(
            "max_unconstrained_direct_self_numeric_entries"
        ),
        "constrained_pressure_neighbor_candidate_count": latest.get(
            "constrained_pressure_neighbor_candidate_count"
        ),
        "constrained_pressure_neighbor_global_dofs": (
            constrained_pressure_neighbor_candidates
        ),
        "constrained_pressure_neighbor_covered_direct_target_global_dofs": (
            constrained_pressure_neighbor_covered
        ),
        "constrained_pressure_neighbor_uncovered_direct_target_global_dofs": (
            constrained_pressure_neighbor_uncovered
        ),
        "constrained_pressure_neighbor_list_truncated": (
            constrained_pressure_neighbor_truncated
        ),
        "constrained_pressure_neighbor_ratio_threshold": latest.get(
            "constrained_pressure_neighbor_ratio_threshold"
        ),
        "high_constrained_pressure_neighbor_ratio_candidate_count": latest.get(
            "high_constrained_pressure_neighbor_ratio_candidate_count"
        ),
        "high_constrained_pressure_neighbor_ratio_global_dofs": (
            high_constrained_pressure_neighbor_ratio_candidates
        ),
        "high_constrained_pressure_neighbor_ratio_covered_direct_target_global_dofs": (
            high_constrained_pressure_neighbor_ratio_covered
        ),
        "high_constrained_pressure_neighbor_ratio_uncovered_direct_target_global_dofs": (
            high_constrained_pressure_neighbor_ratio_uncovered
        ),
        "high_constrained_pressure_neighbor_ratio_list_truncated": (
            high_constrained_pressure_neighbor_ratio_truncated
        ),
        "sparse_unconstrained_direct_self_candidate_count": latest.get(
            "sparse_unconstrained_direct_self_candidate_count"
        ),
        "sparse_unconstrained_direct_self_global_dofs": (
            sparse_unconstrained_direct_self_candidates
        ),
        "sparse_unconstrained_direct_self_covered_direct_target_global_dofs": (
            sparse_unconstrained_direct_self_covered
        ),
        "sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs": (
            sparse_unconstrained_direct_self_uncovered
        ),
        "sparse_unconstrained_direct_self_list_truncated": (
            sparse_unconstrained_truncated
        ),
        "constrained_or_sparse_unconstrained_direct_self_candidate_count": (
            latest.get(
                "constrained_or_sparse_unconstrained_direct_self_candidate_count"
            )
        ),
        "constrained_or_sparse_unconstrained_direct_self_global_dofs": (
            constrained_or_sparse_unconstrained_candidates
        ),
        "constrained_or_sparse_unconstrained_direct_self_covered_direct_target_global_dofs": (
            constrained_or_sparse_unconstrained_covered
        ),
        "constrained_or_sparse_unconstrained_direct_self_uncovered_direct_target_global_dofs": (
            constrained_or_sparse_unconstrained_uncovered
        ),
        "constrained_or_sparse_unconstrained_direct_self_list_truncated": (
            constrained_or_sparse_unconstrained_truncated
        ),
        "direct_self_low_ratio_threshold": latest.get(
            "direct_self_low_ratio_threshold"
        ),
        "direct_self_moderate_ratio_threshold": latest.get(
            "direct_self_moderate_ratio_threshold"
        ),
        "low_direct_self_ratio_candidate_count": latest.get(
            "low_direct_self_ratio_candidate_count"
        ),
        "low_direct_self_ratio_global_dofs": low_ratio_candidates,
        "low_direct_self_ratio_covered_direct_target_global_dofs": (
            low_ratio_covered
        ),
        "low_direct_self_ratio_uncovered_direct_target_global_dofs": (
            low_ratio_uncovered
        ),
        "low_direct_self_ratio_list_truncated": low_ratio_truncated,
        "moderate_direct_self_ratio_candidate_count": latest.get(
            "moderate_direct_self_ratio_candidate_count"
        ),
        "moderate_direct_self_ratio_global_dofs": moderate_ratio_candidates,
        "moderate_direct_self_ratio_covered_direct_target_global_dofs": (
            moderate_ratio_covered
        ),
        "moderate_direct_self_ratio_uncovered_direct_target_global_dofs": (
            moderate_ratio_uncovered
        ),
        "moderate_direct_self_ratio_list_truncated": moderate_ratio_truncated,
        "sparse_or_moderate_direct_self_ratio_candidate_count": latest.get(
            "sparse_or_moderate_direct_self_ratio_candidate_count"
        ),
        "sparse_or_moderate_direct_self_ratio_global_dofs": (
            sparse_or_moderate_candidates
        ),
        "sparse_or_moderate_direct_self_ratio_covered_direct_target_global_dofs": (
            sparse_or_moderate_covered
        ),
        "sparse_or_moderate_direct_self_ratio_uncovered_direct_target_global_dofs": (
            sparse_or_moderate_uncovered
        ),
        "sparse_or_moderate_direct_self_ratio_list_truncated": (
            sparse_or_moderate_truncated
        ),
        "sparse_seeded_pressure_action_radius1_candidate_count": latest.get(
            "sparse_seeded_pressure_action_radius1_candidate_count"
        ),
        "sparse_seeded_pressure_action_radius1_global_dofs": (
            sparse_seeded_radius1_candidates
        ),
        "sparse_seeded_pressure_action_radius1_covered_direct_target_global_dofs": (
            sparse_seeded_radius1_covered
        ),
        "sparse_seeded_pressure_action_radius1_uncovered_direct_target_global_dofs": (
            sparse_seeded_radius1_uncovered
        ),
        "sparse_seeded_pressure_action_radius1_list_truncated": (
            sparse_seeded_radius1_truncated
        ),
        "sparse_seeded_pressure_action_radius2_candidate_count": latest.get(
            "sparse_seeded_pressure_action_radius2_candidate_count"
        ),
        "sparse_seeded_pressure_action_radius2_global_dofs": (
            sparse_seeded_radius2_candidates
        ),
        "sparse_seeded_pressure_action_radius2_covered_direct_target_global_dofs": (
            sparse_seeded_radius2_covered
        ),
        "sparse_seeded_pressure_action_radius2_uncovered_direct_target_global_dofs": (
            sparse_seeded_radius2_uncovered
        ),
        "sparse_seeded_pressure_action_radius2_list_truncated": (
            sparse_seeded_radius2_truncated
        ),
        "graph_local_direct_self_low_ratio_threshold": latest.get(
            "graph_local_direct_self_low_ratio_threshold"
        ),
        "graph_local_direct_self_moderate_ratio_threshold": latest.get(
            "graph_local_direct_self_moderate_ratio_threshold"
        ),
        "graph_local_neighbor_positive_row_count": latest.get(
            "graph_local_neighbor_positive_row_count"
        ),
        "graph_local_low_direct_self_ratio_candidate_count": latest.get(
            "graph_local_low_direct_self_ratio_candidate_count"
        ),
        "graph_local_low_direct_self_ratio_global_dofs": (
            graph_local_low_candidates
        ),
        "graph_local_low_direct_self_ratio_covered_direct_target_global_dofs": (
            graph_local_low_covered
        ),
        "graph_local_low_direct_self_ratio_uncovered_direct_target_global_dofs": (
            graph_local_low_uncovered
        ),
        "graph_local_low_direct_self_ratio_list_truncated": (
            graph_local_low_truncated
        ),
        "graph_local_moderate_direct_self_ratio_candidate_count": latest.get(
            "graph_local_moderate_direct_self_ratio_candidate_count"
        ),
        "graph_local_moderate_direct_self_ratio_global_dofs": (
            graph_local_moderate_candidates
        ),
        "graph_local_moderate_direct_self_ratio_covered_direct_target_global_dofs": (
            graph_local_moderate_covered
        ),
        "graph_local_moderate_direct_self_ratio_uncovered_direct_target_global_dofs": (
            graph_local_moderate_uncovered
        ),
        "graph_local_moderate_direct_self_ratio_list_truncated": (
            graph_local_moderate_truncated
        ),
        "matrix_pressure_action_component_count": latest.get(
            "matrix_pressure_action_component_count"
        ),
        "matrix_pressure_action_largest_component_size": latest.get(
            "matrix_pressure_action_largest_component_size"
        ),
        "matrix_pressure_action_covered_count": latest.get(
            "matrix_pressure_action_covered_count"
        ),
        "matrix_pressure_action_covered_global_dofs": matrix_action_candidates,
        "matrix_pressure_action_covered_list_truncated": matrix_action_truncated,
        "matrix_pressure_action_max_degree": latest.get(
            "matrix_pressure_action_max_degree"
        ),
        "matrix_pressure_action_max_abs_sum": latest.get(
            "matrix_pressure_action_max_abs_sum"
        ),
        "pressure_action_low_degree_threshold": latest.get(
            "pressure_action_low_degree_threshold"
        ),
        "pressure_action_moderate_degree_threshold": latest.get(
            "pressure_action_moderate_degree_threshold"
        ),
        "pressure_action_low_degree_candidate_count": latest.get(
            "pressure_action_low_degree_candidate_count"
        ),
        "pressure_action_low_degree_global_dofs": (
            pressure_action_low_degree_candidates
        ),
        "pressure_action_low_degree_covered_direct_target_global_dofs": (
            pressure_action_low_degree_covered
        ),
        "pressure_action_low_degree_uncovered_direct_target_global_dofs": (
            pressure_action_low_degree_uncovered
        ),
        "pressure_action_low_degree_list_truncated": (
            pressure_action_low_degree_truncated
        ),
        "pressure_action_moderate_degree_candidate_count": latest.get(
            "pressure_action_moderate_degree_candidate_count"
        ),
        "pressure_action_moderate_degree_global_dofs": (
            pressure_action_moderate_degree_candidates
        ),
        "pressure_action_moderate_degree_covered_direct_target_global_dofs": (
            pressure_action_moderate_degree_covered
        ),
        "pressure_action_moderate_degree_uncovered_direct_target_global_dofs": (
            pressure_action_moderate_degree_uncovered
        ),
        "pressure_action_moderate_degree_list_truncated": (
            pressure_action_moderate_degree_truncated
        ),
        "pressure_action_low_sum_ratio_threshold": latest.get(
            "pressure_action_low_sum_ratio_threshold"
        ),
        "pressure_action_moderate_sum_ratio_threshold": latest.get(
            "pressure_action_moderate_sum_ratio_threshold"
        ),
        "pressure_action_low_sum_ratio_candidate_count": latest.get(
            "pressure_action_low_sum_ratio_candidate_count"
        ),
        "pressure_action_low_sum_ratio_global_dofs": (
            pressure_action_low_sum_candidates
        ),
        "pressure_action_low_sum_ratio_covered_direct_target_global_dofs": (
            pressure_action_low_sum_covered
        ),
        "pressure_action_low_sum_ratio_uncovered_direct_target_global_dofs": (
            pressure_action_low_sum_uncovered
        ),
        "pressure_action_low_sum_ratio_list_truncated": (
            pressure_action_low_sum_truncated
        ),
        "pressure_action_moderate_sum_ratio_candidate_count": latest.get(
            "pressure_action_moderate_sum_ratio_candidate_count"
        ),
        "pressure_action_moderate_sum_ratio_global_dofs": (
            pressure_action_moderate_sum_candidates
        ),
        "pressure_action_moderate_sum_ratio_covered_direct_target_global_dofs": (
            pressure_action_moderate_sum_covered
        ),
        "pressure_action_moderate_sum_ratio_uncovered_direct_target_global_dofs": (
            pressure_action_moderate_sum_uncovered
        ),
        "pressure_action_moderate_sum_ratio_list_truncated": (
            pressure_action_moderate_sum_truncated
        ),
        "pressure_action_self_dominant_threshold": latest.get(
            "pressure_action_self_dominant_threshold"
        ),
        "pressure_action_self_dominant_candidate_count": latest.get(
            "pressure_action_self_dominant_candidate_count"
        ),
        "pressure_action_self_dominant_global_dofs": (
            pressure_action_self_dominant_candidates
        ),
        "pressure_action_self_dominant_covered_direct_target_global_dofs": (
            pressure_action_self_dominant_covered
        ),
        "pressure_action_self_dominant_uncovered_direct_target_global_dofs": (
            pressure_action_self_dominant_uncovered
        ),
        "pressure_action_self_dominant_list_truncated": (
            pressure_action_self_dominant_truncated
        ),
        "pressure_action_low_two_hop_threshold": latest.get(
            "pressure_action_low_two_hop_threshold"
        ),
        "pressure_action_high_two_hop_ratio_threshold": latest.get(
            "pressure_action_high_two_hop_ratio_threshold"
        ),
        "matrix_pressure_action_max_two_hop_completion_count": latest.get(
            "matrix_pressure_action_max_two_hop_completion_count"
        ),
        "pressure_action_zero_two_hop_candidate_count": latest.get(
            "pressure_action_zero_two_hop_candidate_count"
        ),
        "pressure_action_zero_two_hop_global_dofs": (
            pressure_action_zero_two_hop_candidates
        ),
        "pressure_action_zero_two_hop_covered_direct_target_global_dofs": (
            pressure_action_zero_two_hop_covered
        ),
        "pressure_action_zero_two_hop_uncovered_direct_target_global_dofs": (
            pressure_action_zero_two_hop_uncovered
        ),
        "pressure_action_zero_two_hop_list_truncated": (
            pressure_action_zero_two_hop_truncated
        ),
        "pressure_action_low_two_hop_candidate_count": latest.get(
            "pressure_action_low_two_hop_candidate_count"
        ),
        "pressure_action_low_two_hop_global_dofs": (
            pressure_action_low_two_hop_candidates
        ),
        "pressure_action_low_two_hop_covered_direct_target_global_dofs": (
            pressure_action_low_two_hop_covered
        ),
        "pressure_action_low_two_hop_uncovered_direct_target_global_dofs": (
            pressure_action_low_two_hop_uncovered
        ),
        "pressure_action_low_two_hop_list_truncated": (
            pressure_action_low_two_hop_truncated
        ),
        "pressure_action_high_two_hop_candidate_count": latest.get(
            "pressure_action_high_two_hop_candidate_count"
        ),
        "pressure_action_high_two_hop_global_dofs": (
            pressure_action_high_two_hop_candidates
        ),
        "pressure_action_high_two_hop_covered_direct_target_global_dofs": (
            pressure_action_high_two_hop_covered
        ),
        "pressure_action_high_two_hop_uncovered_direct_target_global_dofs": (
            pressure_action_high_two_hop_uncovered
        ),
        "pressure_action_high_two_hop_list_truncated": (
            pressure_action_high_two_hop_truncated
        ),
        "pressure_action_low_clustering_threshold": latest.get(
            "pressure_action_low_clustering_threshold"
        ),
        "pressure_action_high_clustering_threshold": latest.get(
            "pressure_action_high_clustering_threshold"
        ),
        "pressure_action_clustering_eligible_row_count": latest.get(
            "pressure_action_clustering_eligible_row_count"
        ),
        "matrix_pressure_action_min_clustering_ratio": latest.get(
            "matrix_pressure_action_min_clustering_ratio"
        ),
        "matrix_pressure_action_max_clustering_ratio": latest.get(
            "matrix_pressure_action_max_clustering_ratio"
        ),
        "pressure_action_zero_clustering_candidate_count": latest.get(
            "pressure_action_zero_clustering_candidate_count"
        ),
        "pressure_action_zero_clustering_global_dofs": (
            pressure_action_zero_clustering_candidates
        ),
        "pressure_action_zero_clustering_covered_direct_target_global_dofs": (
            pressure_action_zero_clustering_covered
        ),
        "pressure_action_zero_clustering_uncovered_direct_target_global_dofs": (
            pressure_action_zero_clustering_uncovered
        ),
        "pressure_action_zero_clustering_list_truncated": (
            pressure_action_zero_clustering_truncated
        ),
        "pressure_action_low_clustering_candidate_count": latest.get(
            "pressure_action_low_clustering_candidate_count"
        ),
        "pressure_action_low_clustering_global_dofs": (
            pressure_action_low_clustering_candidates
        ),
        "pressure_action_low_clustering_covered_direct_target_global_dofs": (
            pressure_action_low_clustering_covered
        ),
        "pressure_action_low_clustering_uncovered_direct_target_global_dofs": (
            pressure_action_low_clustering_uncovered
        ),
        "pressure_action_low_clustering_list_truncated": (
            pressure_action_low_clustering_truncated
        ),
        "pressure_action_high_clustering_candidate_count": latest.get(
            "pressure_action_high_clustering_candidate_count"
        ),
        "pressure_action_high_clustering_global_dofs": (
            pressure_action_high_clustering_candidates
        ),
        "pressure_action_high_clustering_covered_direct_target_global_dofs": (
            pressure_action_high_clustering_covered
        ),
        "pressure_action_high_clustering_uncovered_direct_target_global_dofs": (
            pressure_action_high_clustering_uncovered
        ),
        "pressure_action_high_clustering_list_truncated": (
            pressure_action_high_clustering_truncated
        ),
        "pressure_action_articulation_candidate_count": latest.get(
            "pressure_action_articulation_candidate_count"
        ),
        "pressure_action_articulation_global_dofs": (
            pressure_action_articulation_candidates
        ),
        "pressure_action_articulation_covered_direct_target_global_dofs": (
            pressure_action_articulation_covered
        ),
        "pressure_action_articulation_uncovered_direct_target_global_dofs": (
            pressure_action_articulation_uncovered
        ),
        "pressure_action_articulation_list_truncated": (
            pressure_action_articulation_truncated
        ),
        "pressure_action_bridge_endpoint_candidate_count": latest.get(
            "pressure_action_bridge_endpoint_candidate_count"
        ),
        "pressure_action_bridge_endpoint_global_dofs": (
            pressure_action_bridge_endpoint_candidates
        ),
        "pressure_action_bridge_endpoint_covered_direct_target_global_dofs": (
            pressure_action_bridge_endpoint_covered
        ),
        "pressure_action_bridge_endpoint_uncovered_direct_target_global_dofs": (
            pressure_action_bridge_endpoint_uncovered
        ),
        "pressure_action_bridge_endpoint_list_truncated": (
            pressure_action_bridge_endpoint_truncated
        ),
        "matrix_pressure_action_isolated_count": latest.get(
            "matrix_pressure_action_isolated_count"
        ),
        "sparse_seeded_matrix_pressure_action_component_count": latest.get(
            "sparse_seeded_matrix_pressure_action_component_count"
        ),
        "sparse_seeded_matrix_pressure_action_component_dof_count": latest.get(
            "sparse_seeded_matrix_pressure_action_component_dof_count"
        ),
        "sparse_seeded_matrix_pressure_action_component_covered_direct_target_global_dofs": (
            sparse_seeded_component_covered
        ),
        "sparse_seeded_matrix_pressure_action_component_uncovered_direct_target_global_dofs": (
            sparse_seeded_component_uncovered
        ),
        "sparse_seeded_matrix_pressure_action_component_list_truncated": (
            sparse_seeded_components_truncated
        ),
        "residual_sign_threshold": latest.get("residual_sign_threshold"),
        "residual_nonzero_direct_row_count": latest.get(
            "residual_nonzero_direct_row_count"
        ),
        "residual_positive_direct_row_count": latest.get(
            "residual_positive_direct_row_count"
        ),
        "residual_negative_direct_row_count": latest.get(
            "residual_negative_direct_row_count"
        ),
        "residual_zero_direct_row_count": latest.get(
            "residual_zero_direct_row_count"
        ),
        "residual_nonfinite_direct_row_count": latest.get(
            "residual_nonfinite_direct_row_count"
        ),
        "min_positive_residual_abs": latest.get("min_positive_residual_abs"),
        "max_residual_abs": latest.get("max_residual_abs"),
        "residual_sign_pressure_action_edge_count": latest.get(
            "residual_sign_pressure_action_edge_count"
        ),
        "residual_opposite_sign_pressure_action_edge_count": latest.get(
            "residual_opposite_sign_pressure_action_edge_count"
        ),
        "residual_zero_or_missing_sign_pressure_action_edge_count": latest.get(
            "residual_zero_or_missing_sign_pressure_action_edge_count"
        ),
        "residual_sign_pressure_action_component_count": latest.get(
            "residual_sign_pressure_action_component_count"
        ),
        "residual_sign_pressure_action_largest_component_size": latest.get(
            "residual_sign_pressure_action_largest_component_size"
        ),
        "residual_sign_pressure_action_covered_count": latest.get(
            "residual_sign_pressure_action_covered_count"
        ),
        "residual_sign_pressure_action_covered_global_dofs": (
            residual_sign_action_candidates
        ),
        "residual_sign_pressure_action_covered_direct_target_global_dofs": (
            residual_sign_action_covered
        ),
        "residual_sign_pressure_action_uncovered_direct_target_global_dofs": (
            residual_sign_action_uncovered
        ),
        "residual_sign_pressure_action_covered_list_truncated": (
            residual_sign_action_truncated
        ),
        "sparse_seeded_residual_sign_pressure_action_component_count": latest.get(
            "sparse_seeded_residual_sign_pressure_action_component_count"
        ),
        "sparse_seeded_residual_sign_pressure_action_component_dof_count": latest.get(
            "sparse_seeded_residual_sign_pressure_action_component_dof_count"
        ),
        "sparse_seeded_residual_sign_pressure_action_component_covered_direct_target_global_dofs": (
            sparse_seeded_residual_sign_component_covered
        ),
        "sparse_seeded_residual_sign_pressure_action_component_uncovered_direct_target_global_dofs": (
            sparse_seeded_residual_sign_component_uncovered
        ),
        "sparse_seeded_residual_sign_pressure_action_component_list_truncated": (
            sparse_seeded_residual_sign_components_truncated
        ),
        "sparse_direct_self_or_residual_sign_pressure_action_candidate_count": (
            latest.get(
                "sparse_direct_self_or_residual_sign_pressure_action_candidate_count"
            )
        ),
        "sparse_direct_self_or_residual_sign_pressure_action_global_dofs": (
            sparse_or_residual_sign_candidates
        ),
        "sparse_direct_self_or_residual_sign_pressure_action_covered_direct_target_global_dofs": (
            sparse_or_residual_sign_covered
        ),
        "sparse_direct_self_or_residual_sign_pressure_action_uncovered_direct_target_global_dofs": (
            sparse_or_residual_sign_uncovered
        ),
        "sparse_direct_self_or_residual_sign_pressure_action_list_truncated": (
            sparse_or_residual_sign_truncated
        ),
        "preferred_candidate_count": latest.get("preferred_candidate_count"),
        "preferred_candidate_global_dofs": candidates,
        "covered_direct_target_global_dofs": covered,
        "uncovered_direct_target_global_dofs": uncovered,
        "candidate_list_truncated": truncated,
        "artifact_limitation": latest.get("artifact_limitation"),
    }


def build_report(
    *,
    target_map: dict[str, Any],
    logs: list[tuple[str, Path]],
    target_map_path: Path | None = None,
) -> dict[str, Any]:
    targets = target_case_map(target_map)
    cases = [
        evaluate_case(
            label=label,
            log_path=path,
            target_case=targets.get(label, {}),
        )
        for label, path in logs
    ]
    missing_cases = [
        label
        for label in targets
        if label not in {case["label"] for case in cases}
    ]
    complete_cases = [
        case
        for case in cases
        if case["finding"] == "candidate_emitted_covers_audited_targets"
    ]
    sample_limited_cases = [
        case for case in cases if "sample_limited" in case["finding"]
    ]
    miss_cases = [
        case
        for case in cases
        if case["finding"] == "candidate_emitted_but_misses_targets"
    ]

    if missing_cases:
        finding = "candidate_emission_logs_missing_cases"
    elif len(complete_cases) == len(cases) and cases:
        finding = "candidate_emission_covers_audited_targets"
    elif sample_limited_cases:
        finding = "candidate_emission_present_but_sample_limited"
    elif miss_cases:
        finding = "candidate_emission_misses_audited_targets"
    else:
        finding = "candidate_emission_not_observed"

    if finding == "candidate_emission_covers_audited_targets":
        next_requirement = (
            "Convert the globally emitted sparse-direct-self plus matrix "
            "pressure-action candidate into formulation-side PSPG "
            "support/coupling and replay Test02/Test10 without broad "
            "post-assembly graph mutation."
        )
    elif finding == "candidate_emission_present_but_sample_limited":
        next_requirement = (
            "Rerun the diagnostic with a wider candidate sample or targeted "
            "target-row reporting so coverage can be proved or refuted."
        )
    elif finding == "candidate_emission_misses_audited_targets":
        next_requirement = (
            "Revise the global candidate selector because the emitted "
            "candidate set misses audited direct PSPG target rows."
        )
    else:
        next_requirement = (
            "Run the diagnostic on short Test02/Test10 windows and use the "
            "emitted candidate breadth/coverage to decide whether a targeted "
            "formulation replay is justified."
        )

    return {
        "scope": (
            "Global pre-update direct PSPG formulation-candidate diagnostic "
            "emission parsed from short replay logs."
        ),
        "target_map_path": str(target_map_path) if target_map_path else None,
        "finding": finding,
        "missing_case_labels": missing_cases,
        "case_count": len(cases),
        "cases": cases,
        "next_requirement": next_requirement,
    }


def main() -> int:
    args = parse_args()
    target_map = load_json(args.target_map_json)
    logs = [parse_log_arg(value) for value in args.log]
    report = build_report(
        target_map=target_map,
        logs=logs,
        target_map_path=args.target_map_json,
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
