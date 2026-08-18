#!/usr/bin/env python3
"""Summarize Test02/Test10 root-cause hypothesis status from current artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_ROOT = Path(
    "Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526"
)
DEFAULT_ROOT_REPORT = Path(
    "Documentation/open_vessel_free_surface_test02_test10_root_cause_report_20260605.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact machine-readable status matrix for the Test02/Test10 "
            "open-vessel root-cause hypotheses from the current replay artifacts."
        )
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--root-report", type=Path, default=DEFAULT_ROOT_REPORT)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_json(root: Path, name: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    path = root / name
    evidence = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return None, evidence
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        evidence["json_error"] = str(exc)
        return None, evidence
    return data, evidence


def values(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    nested = record.get("values")
    return nested if isinstance(nested, dict) else record


def case_by_label(report: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    for case in report.get("cases", []):
        if (
            isinstance(case, dict)
            and (case.get("label") == label or case.get("name") == label)
        ):
            return case
    return {}


def pressure_update_case_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    worst_by_category = report.get("worst_by_category")
    if not isinstance(worst_by_category, dict):
        worst_by_category = {}
    active_wet = worst_by_category.get("active_or_wet_supported")
    if not isinstance(active_wet, dict):
        active_wet = {}
    return {
        "status": report.get("status"),
        "finding": report.get("finding"),
        "absolute_threshold_pa": report.get("absolute_threshold_pa"),
        "triggered_transition_count": report.get("triggered_transition_count"),
        "worst_active_or_wet_update_pa": active_wet.get("abs_pressure_delta_pa"),
        "worst_active_or_wet_point_index": active_wet.get("point_index"),
        "worst_active_or_wet_support_class": active_wet.get("support_class"),
        "worst_active_or_wet_active_fluid": active_wet.get("active_fluid"),
        "worst_active_or_wet_fraction_min_positive": active_wet.get(
            "incident_wet_fraction_min_positive"
        ),
    }


def graph_completion_selector_coverage_summary(
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    cases = [case for case in report.get("cases", []) if isinstance(case, dict)]
    return {
        "finding": report.get("finding"),
        "case_count": report.get("case_count"),
        "finding_counts": report.get("finding_counts"),
        "case_findings": {
            case.get("label"): case.get("finding") for case in cases
        },
        "max_update_global_dofs": {
            case.get("label"): case.get("max_update_global_dof") for case in cases
        },
        "max_abs_updates_pa": {
            case.get("label"): case.get("max_abs_update_pa") for case in cases
        },
        "selector_reasons": {
            case.get("label"): case.get("selector_reason") for case in cases
        },
        "least_selector_thresholds_to_include": {
            case.get("label"): case.get("least_selector_threshold_expansion_to_include")
            for case in cases
        },
        "casewise_least_widened_threshold_floor": report.get(
            "sampled_outside_selector_threshold_floor_if_casewise_least_widened"
        ),
        "single_selector_widened_threshold_floor": report.get(
            "sampled_outside_selector_threshold_floor_if_single_selector_widened"
        ),
    }


def explicit_balance_selector_replay_summary(
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {
            "finding": None,
            "status": None,
            "ruleout_flags": None,
            "boundary_provenance": None,
            "ruled_out_by_variant": None,
            "case_findings_by_variant": None,
            "accepted_pressure_updates_by_variant": None,
            "balance_candidate_counts_by_variant": None,
            "next_requirement": None,
        }
    variants = [
        variant for variant in report.get("variants", []) if isinstance(variant, dict)
    ]
    return {
        "finding": report.get("finding"),
        "status": report.get("status"),
        "ruleout_flags": report.get("ruleout_flags"),
        "boundary_provenance": report.get("boundary_provenance"),
        "ruled_out_by_variant": report.get("ruled_out_by_variant"),
        "case_findings_by_variant": {
            variant.get("key"): {
                case.get("label"): case.get("finding")
                for case in variant.get("cases", [])
                if isinstance(case, dict)
            }
            for variant in variants
        },
        "accepted_pressure_updates_by_variant": {
            variant.get("key"): {
                case.get("label"): case.get("accepted_pressure_update_pa")
                for case in variant.get("cases", [])
                if isinstance(case, dict)
            }
            for variant in variants
        },
        "balance_candidate_counts_by_variant": {
            variant.get("key"): {
                case.get("label"): case.get("balance_candidate_row_count")
                for case in variant.get("cases", [])
                if isinstance(case, dict)
            }
            for variant in variants
        },
        "next_requirement": report.get("next_requirement"),
    }


def status_item(
    *,
    key: str,
    question: str,
    status: str,
    conclusion: str,
    evidence: list[dict[str, Any]],
    remaining_risk: str = "",
) -> dict[str, Any]:
    missing = [
        item["path"]
        for item in evidence
        if isinstance(item, dict) and not item.get("exists", True)
    ]
    return {
        "key": key,
        "question": question,
        "status": status,
        "conclusion": conclusion,
        "evidence": evidence,
        "missing_evidence": missing,
        "remaining_risk": remaining_risk,
    }


def build_status_report(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    root_report: Path = DEFAULT_ROOT_REPORT,
) -> dict[str, Any]:
    report_text = root_report.read_text(encoding="utf-8") if root_report.exists() else ""
    root_evidence = {
        "path": str(root_report),
        "exists": root_report.exists(),
    }

    linear_patch, linear_patch_evidence = optional_json(
        artifact_root,
        "linear_pressure_cut_volume_patch_audit_20260605.json",
    )
    top_provenance, top_provenance_evidence = optional_json(
        artifact_root,
        "test02_test10_pressure_operator_toprow_provenance_20260606.json",
    )
    direct_pspg_target, direct_pspg_target_evidence = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_formulation_target_20260606.json",
    )
    (
        graph_completion_candidate_readiness,
        graph_completion_candidate_readiness_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_graph_completion_candidate_readiness_20260606.json",
    )
    (
        graph_completion_replay_family,
        graph_completion_replay_family_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_graph_completion_replay_family_20260607.json",
    )
    (
        graph_completion_stability_tradeoff,
        graph_completion_stability_tradeoff_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_graph_completion_stability_tradeoff_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_active_support_completion_replays,
        direct_pspg_active_support_completion_replays_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_active_support_completion_replays_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_explicit_balance_selector_replays,
        direct_pspg_explicit_balance_selector_replays_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_explicit_balance_selector_replays_"
            "20260607.json"
        ),
    )
    (
        formulation_side_candidate_predicates,
        formulation_side_candidate_predicates_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_formulation_side_candidate_predicates_20260606.json",
    )
    direct_pspg_global_candidate_emission, direct_pspg_global_candidate_emission_evidence = (
        optional_json(
            artifact_root,
            "test02_test10_direct_pspg_global_candidate_emission_20260606.json",
        )
    )
    (
        direct_pspg_global_candidate_selectivity,
        direct_pspg_global_candidate_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_global_candidate_selectivity_20260607.json",
    )
    (
        direct_pspg_boundary_provenance_selectivity,
        direct_pspg_boundary_provenance_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_boundary_provenance_selectivity_20260607.json",
    )
    (
        direct_pspg_named_face_provenance_selectivity,
        direct_pspg_named_face_provenance_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_named_face_provenance_selectivity_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_cut_state_provenance_selectivity,
        direct_pspg_cut_state_provenance_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_state_provenance_selectivity_20260607.json",
    )
    (
        direct_pspg_same_sign_dependency_readiness,
        direct_pspg_same_sign_dependency_readiness_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_same_sign_dependency_readiness_20260607.json",
    )
    (
        direct_pspg_coupled_patch_dependency_barrier,
        direct_pspg_coupled_patch_dependency_barrier_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_coupled_patch_dependency_barrier_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_provenance_support,
        direct_pspg_solve_time_provenance_support_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_provenance_support_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_provenance_replay,
        direct_pspg_solve_time_provenance_replay_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_provenance_replay_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_aggregate_feature_selectivity,
        direct_pspg_solve_time_aggregate_feature_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_aggregate_feature_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_support_measure_selectivity,
        direct_pspg_solve_time_support_measure_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_support_measure_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_parent_rule_component_selectivity,
        direct_pspg_solve_time_parent_rule_component_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_parent_rule_component_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_sampled_column_selectivity,
        direct_pspg_solve_time_sampled_column_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_sampled_column_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_same_rule_cross_block_signature,
        direct_pspg_solve_time_same_rule_cross_block_signature_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_same_rule_cross_block_"
            "signature_20260607.json"
        ),
    )
    (
        direct_pspg_same_rule_cross_block_row_filter_replays,
        direct_pspg_same_rule_cross_block_row_filter_replays_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_row_filter_"
            "replays_20260607.json"
        ),
    )
    (
        direct_pspg_same_rule_cross_block_parent_cell_scope,
        direct_pspg_same_rule_cross_block_parent_cell_scope_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_"
            "scope_20260607.json"
        ),
    )
    (
        direct_pspg_same_rule_cross_block_parent_cell_replays,
        direct_pspg_same_rule_cross_block_parent_cell_replays_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_"
            "replays_20260607.json"
        ),
    )
    (
        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_"
            "parent_cell_scope_20260607.json"
        ),
    )
    (
        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_"
            "parent_cell_replays_20260607.json"
        ),
    )
    (
        direct_pspg_same_rule_cross_block_broad_union_branch_shift,
        direct_pspg_same_rule_cross_block_broad_union_branch_shift_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_union_"
            "branch_shift_20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_support_coupling_signature,
        direct_pspg_solve_time_support_coupling_signature_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_support_coupling_signature_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_magnitude_selectivity,
        direct_pspg_solve_time_magnitude_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_magnitude_selectivity_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_solve_time_signature_magnitude_composite,
        direct_pspg_solve_time_signature_magnitude_composite_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_solve_time_signature_magnitude_"
            "composite_20260607.json"
        ),
    )
    (
        direct_pspg_test10_signature_replay_readiness,
        direct_pspg_test10_signature_replay_readiness_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_test10_signature_replay_readiness_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_test10_signature_row_filter_replay,
        direct_pspg_test10_signature_row_filter_replay_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    )
    (
        direct_pspg_test10_signature_row_filter_replays,
        direct_pspg_test10_signature_row_filter_replays_evidence,
    ) = optional_json(
        artifact_root,
        "test10_direct_pspg_signature_row_filter_replays_20260607.json",
    )
    (
        direct_pspg_ghost_branch_signature_interaction,
        direct_pspg_ghost_branch_signature_interaction_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_ghost_branch_signature_interaction_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_active_pressure_support_selectivity,
        direct_pspg_active_pressure_support_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_active_pressure_support_selectivity_20260607.json",
    )
    (
        direct_pspg_residual_sign_selectivity,
        direct_pspg_residual_sign_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_residual_sign_selectivity_20260607.json",
    )
    (
        direct_pspg_null_balance_selectivity,
        direct_pspg_null_balance_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_null_balance_selectivity_20260607.json",
    )
    (
        direct_pspg_coupled_patch_graph_selectivity,
        direct_pspg_coupled_patch_graph_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_coupled_patch_graph_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_row_provenance_selectivity,
        direct_pspg_cut_volume_row_provenance_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_row_provenance_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_local_matrix_selectivity,
        direct_pspg_cut_volume_local_matrix_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_local_matrix_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_local_coupling_selectivity,
        direct_pspg_cut_volume_local_coupling_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_local_coupling_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_parent_graph_selectivity,
        direct_pspg_cut_volume_parent_graph_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_parent_graph_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_composite_selectivity,
        direct_pspg_cut_volume_composite_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_composite_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_column_support_readiness,
        direct_pspg_cut_volume_column_support_readiness_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_column_support_readiness_20260607.json",
    )
    (
        direct_pspg_cut_volume_column_support_selectivity,
        direct_pspg_cut_volume_column_support_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_column_support_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_column_geometry_selectivity,
        direct_pspg_cut_volume_column_geometry_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_column_geometry_selectivity_20260607.json",
    )
    (
        direct_pspg_cut_volume_quadrature_geometry_selectivity,
        direct_pspg_cut_volume_quadrature_geometry_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_cut_volume_quadrature_geometry_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_cut_volume_gradient_balance_selectivity,
        direct_pspg_cut_volume_gradient_balance_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_cut_volume_gradient_balance_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_cut_volume_gradient_column_graph_selectivity,
        direct_pspg_cut_volume_gradient_column_graph_selectivity_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_cut_volume_gradient_column_graph_"
            "selectivity_20260607.json"
        ),
    )
    (
        direct_pspg_cut_volume_local_schur_completion,
        direct_pspg_cut_volume_local_schur_completion_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_cut_volume_local_schur_"
            "completion_20260607.json"
        ),
    )
    (
        direct_pspg_cut_volume_local_edge_balance,
        direct_pspg_cut_volume_local_edge_balance_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_cut_volume_local_edge_balance_20260607.json",
    )
    (
        graph_completion_selector_coverage,
        graph_completion_selector_coverage_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_graph_completion_selector_coverage_20260606.json",
    )
    (
        direct_pspg_formulation_vocabulary_support,
        direct_pspg_formulation_vocabulary_support_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_formulation_vocabulary_support_20260607.json",
    )
    direct_pspg_assembly_api_support, direct_pspg_assembly_api_support_evidence = (
        optional_json(
            artifact_root,
            "test02_test10_direct_pspg_assembly_api_support_20260607.json",
        )
    )
    (
        direct_pspg_topology_policy_replay_pair,
        direct_pspg_topology_policy_replay_pair_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_topology_policy_replay_pair_20260607.json",
    )
    (
        direct_pspg_topology_policy_mode_replays,
        direct_pspg_topology_policy_mode_replays_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_topology_policy_mode_replays_20260607.json",
    )
    (
        direct_pspg_topology_policy_application_effect,
        direct_pspg_topology_policy_application_effect_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_topology_policy_application_effect_"
            "20260607.json"
        ),
    )
    (
        direct_pspg_topology_policy_scope_scale,
        direct_pspg_topology_policy_scope_scale_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_topology_policy_scope_scale_20260607.json",
    )
    (
        direct_pspg_topology_policy_parent_scope,
        direct_pspg_topology_policy_parent_scope_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_topology_policy_parent_scope_20260607.json",
    )
    (
        direct_pspg_topology_policy_parent_subset_readiness,
        direct_pspg_topology_policy_parent_subset_readiness_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_test10_direct_pspg_topology_policy_parent_subset_replay_"
            "readiness_20260607.json"
        ),
    )
    (
        direct_pspg_topology_policy_parent_subset_replay,
        direct_pspg_topology_policy_parent_subset_replay_evidence,
    ) = optional_json(
        artifact_root,
        "test10_direct_pspg_topology_policy_parent_subset_replay_20260607.json",
    )
    (
        active_pressure_support_cutoff_relevance,
        active_pressure_support_cutoff_relevance_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_active_pressure_support_cutoff_relevance_20260607.json",
    )
    test02_shape_tangent, test02_shape_tangent_evidence = optional_json(
        artifact_root,
        (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "shape_tangent_pressure_update_audit_20260606.json"
        ),
    )
    test02_cut_volume_scale_cap16, test02_cut_volume_scale_cap16_evidence = (
        optional_json(
            artifact_root,
            (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                "cut_volume_scale_cap16_pressure_update_audit_20260606.json"
            ),
        )
    )
    test10_cut_volume_scale_cap16, test10_cut_volume_scale_cap16_evidence = (
        optional_json(
            artifact_root,
            (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                "cut_volume_scale_cap16_pressure_update_audit_20260606.json"
            ),
        )
    )
    (
        test02_free_surface_tangential_pressure_update,
        test02_free_surface_tangential_pressure_update_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "free_surface_tangential_scale1_pressure_update_audit_20260606.json"
        ),
    )
    (
        test10_free_surface_tangential_pressure_update,
        test10_free_surface_tangential_pressure_update_evidence,
    ) = optional_json(
        artifact_root,
        (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_"
            "free_surface_tangential_scale1_pressure_update_audit_20260606.json"
        ),
    )
    cut_adjacent_support_window, cut_adjacent_support_window_evidence = optional_json(
        artifact_root,
        "test02_test10_cut_adjacent_support_pressure_window_20260606.json",
    )
    (
        pressure_stabilization_driver_windows,
        pressure_stabilization_driver_windows_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_pressure_stabilization_driver_windows_20260607.json",
    )
    top_overlap, top_overlap_evidence = optional_json(
        artifact_root,
        "test02_test10_pressure_operator_top_update_overlap_20260606.json",
    )
    (
        no_galerkin_gate_relevance,
        no_galerkin_gate_relevance_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_direct_pspg_no_galerkin_gate_relevance_20260607.json",
    )
    boundary_provenance, boundary_provenance_evidence = optional_json(
        artifact_root,
        (
            "test02_test10_graph_completion_shared_row_schur_low_degree_"
            "edge_balance_deg3_boundary_provenance_20260606.json"
        ),
    )
    coupling_outcome, coupling_outcome_evidence = optional_json(
        artifact_root,
        "test02_test10_graph_completion_shared_row_schur_coupling_edge_balance_20260606_outcome.json",
    )
    low_degree_outcome, low_degree_outcome_evidence = optional_json(
        artifact_root,
        (
            "test02_test10_graph_completion_shared_row_schur_low_degree_"
            "edge_balance_deg3_20260606_outcome.json"
        ),
    )
    support_gap_patch_outcome, support_gap_patch_outcome_evidence = optional_json(
        artifact_root,
        "test02_test10_graph_completion_support_gap_patch_20260606_outcome.json",
    )
    (
        support_gap_patch_schur_only_outcome,
        support_gap_patch_schur_only_outcome_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_graph_completion_support_gap_patch_schur_only_20260606_outcome.json",
    )
    (
        support_gap_local_patch_schur_only_outcome,
        support_gap_local_patch_schur_only_outcome_evidence,
    ) = optional_json(
        artifact_root,
        "test02_test10_graph_completion_support_gap_local_patch_schur_only_depth1_20260606_outcome.json",
    )
    test02_components, test02_components_evidence = optional_json(
        artifact_root,
        "test02_update_support_components_20260606.json",
    )
    test10_components, test10_components_evidence = optional_json(
        artifact_root,
        "test10_update_support_components_20260606.json",
    )
    support_rank_guard, support_rank_guard_evidence = optional_json(
        artifact_root,
        "test10_replay_cap3_step90_support_rank_guard_audit_20260605.json",
    )
    constraint_coverage, constraint_coverage_evidence = optional_json(
        artifact_root,
        "test10_replay_cap3_step90_vms_disabled_pressure_constraint_coverage_audit_20260605.json",
    )
    free_surface_reference, free_surface_reference_evidence = optional_json(
        artifact_root,
        "test10_replay_cap3_step90_pressure_reference_probe_penalty1em6_support_audit_20260605.json",
    )
    free_surface_tangential, free_surface_tangential_evidence = optional_json(
        artifact_root,
        (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_"
            "free_surface_tangential_scale1_support_audit_20260606.json"
        ),
    )
    shape_tangent, shape_tangent_evidence = optional_json(
        artifact_root,
        "test10_replay_cap3_step90_pspg_wall_full_gradient_shape_tangent_pressure_update_audit_20260606.json",
    )
    full_cell, full_cell_evidence = optional_json(
        artifact_root,
        "test10_replay_cap3_step90_pspg_full_cell_support_wall_full_gradient_pressure_update_audit_20260606.json",
    )
    pressure_update_rejection_replay, pressure_update_rejection_replay_evidence = (
        optional_json(
            artifact_root,
            "test02_test10_pressure_update_rejection_replay_20260607.json",
        )
    )
    pressure_update_residual_context, pressure_update_residual_context_evidence = (
        optional_json(
            artifact_root,
            "test02_test10_pressure_update_residual_context_20260607.json",
        )
    )

    top_finding = top_provenance.get("finding") if isinstance(top_provenance, dict) else None
    direct_pspg_target_finding = (
        direct_pspg_target.get("finding")
        if isinstance(direct_pspg_target, dict)
        else None
    )
    graph_completion_candidate_readiness_finding = (
        graph_completion_candidate_readiness.get("finding")
        if isinstance(graph_completion_candidate_readiness, dict)
        else None
    )
    graph_completion_replay_family_finding = (
        graph_completion_replay_family.get("finding")
        if isinstance(graph_completion_replay_family, dict)
        else None
    )
    graph_completion_stability_tradeoff_finding = (
        graph_completion_stability_tradeoff.get("finding")
        if isinstance(graph_completion_stability_tradeoff, dict)
        else None
    )
    direct_pspg_active_support_completion_replays_finding = (
        direct_pspg_active_support_completion_replays.get("finding")
        if isinstance(direct_pspg_active_support_completion_replays, dict)
        else None
    )
    direct_pspg_explicit_balance_selector_replays_finding = (
        direct_pspg_explicit_balance_selector_replays.get("finding")
        if isinstance(direct_pspg_explicit_balance_selector_replays, dict)
        else None
    )
    formulation_side_candidate_predicates_finding = (
        formulation_side_candidate_predicates.get("finding")
        if isinstance(formulation_side_candidate_predicates, dict)
        else None
    )
    direct_pspg_global_candidate_emission_finding = (
        direct_pspg_global_candidate_emission.get("finding")
        if isinstance(direct_pspg_global_candidate_emission, dict)
        else None
    )
    direct_pspg_global_candidate_selectivity_finding = (
        direct_pspg_global_candidate_selectivity.get("finding")
        if isinstance(direct_pspg_global_candidate_selectivity, dict)
        else None
    )
    direct_pspg_boundary_provenance_selectivity_finding = (
        direct_pspg_boundary_provenance_selectivity.get("finding")
        if isinstance(direct_pspg_boundary_provenance_selectivity, dict)
        else None
    )
    direct_pspg_named_face_provenance_selectivity_finding = (
        direct_pspg_named_face_provenance_selectivity.get("finding")
        if isinstance(direct_pspg_named_face_provenance_selectivity, dict)
        else None
    )
    direct_pspg_cut_state_provenance_selectivity_finding = (
        direct_pspg_cut_state_provenance_selectivity.get("finding")
        if isinstance(direct_pspg_cut_state_provenance_selectivity, dict)
        else None
    )
    direct_pspg_same_sign_dependency_readiness_finding = (
        direct_pspg_same_sign_dependency_readiness.get("finding")
        if isinstance(direct_pspg_same_sign_dependency_readiness, dict)
        else None
    )
    direct_pspg_coupled_patch_dependency_barrier_finding = (
        direct_pspg_coupled_patch_dependency_barrier.get("finding")
        if isinstance(direct_pspg_coupled_patch_dependency_barrier, dict)
        else None
    )
    direct_pspg_solve_time_provenance_support_finding = (
        direct_pspg_solve_time_provenance_support.get("finding")
        if isinstance(direct_pspg_solve_time_provenance_support, dict)
        else None
    )
    direct_pspg_solve_time_provenance_replay_finding = (
        direct_pspg_solve_time_provenance_replay.get("finding")
        if isinstance(direct_pspg_solve_time_provenance_replay, dict)
        else None
    )
    direct_pspg_solve_time_aggregate_feature_selectivity_finding = (
        direct_pspg_solve_time_aggregate_feature_selectivity.get("finding")
        if isinstance(direct_pspg_solve_time_aggregate_feature_selectivity, dict)
        else None
    )
    direct_pspg_solve_time_support_measure_selectivity_finding = (
        direct_pspg_solve_time_support_measure_selectivity.get("finding")
        if isinstance(direct_pspg_solve_time_support_measure_selectivity, dict)
        else None
    )
    direct_pspg_solve_time_parent_rule_component_selectivity_finding = (
        direct_pspg_solve_time_parent_rule_component_selectivity.get("finding")
        if isinstance(
            direct_pspg_solve_time_parent_rule_component_selectivity, dict
        )
        else None
    )
    direct_pspg_solve_time_sampled_column_selectivity_finding = (
        direct_pspg_solve_time_sampled_column_selectivity.get("finding")
        if isinstance(direct_pspg_solve_time_sampled_column_selectivity, dict)
        else None
    )
    direct_pspg_solve_time_same_rule_cross_block_signature_finding = (
        direct_pspg_solve_time_same_rule_cross_block_signature.get("finding")
        if isinstance(direct_pspg_solve_time_same_rule_cross_block_signature, dict)
        else None
    )
    direct_pspg_same_rule_cross_block_row_filter_replays_finding = (
        direct_pspg_same_rule_cross_block_row_filter_replays.get("finding")
        if isinstance(direct_pspg_same_rule_cross_block_row_filter_replays, dict)
        else None
    )
    direct_pspg_same_rule_cross_block_parent_cell_scope_finding = (
        direct_pspg_same_rule_cross_block_parent_cell_scope.get("finding")
        if isinstance(direct_pspg_same_rule_cross_block_parent_cell_scope, dict)
        else None
    )
    direct_pspg_same_rule_cross_block_parent_cell_replays_finding = (
        direct_pspg_same_rule_cross_block_parent_cell_replays.get("finding")
        if isinstance(direct_pspg_same_rule_cross_block_parent_cell_replays, dict)
        else None
    )
    direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope_finding = (
        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
            "finding"
        )
        if isinstance(
            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
            dict,
        )
        else None
    )
    direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays_finding = (
        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
            "finding"
        )
        if isinstance(
            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
            dict,
        )
        else None
    )
    direct_pspg_same_rule_cross_block_broad_union_branch_shift_finding = (
        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get("finding")
        if isinstance(
            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
            dict,
        )
        else None
    )
    direct_pspg_solve_time_support_coupling_signature_finding = (
        direct_pspg_solve_time_support_coupling_signature.get("finding")
        if isinstance(direct_pspg_solve_time_support_coupling_signature, dict)
        else None
    )
    direct_pspg_solve_time_magnitude_selectivity_finding = (
        direct_pspg_solve_time_magnitude_selectivity.get("finding")
        if isinstance(direct_pspg_solve_time_magnitude_selectivity, dict)
        else None
    )
    direct_pspg_solve_time_signature_magnitude_composite_finding = (
        direct_pspg_solve_time_signature_magnitude_composite.get("finding")
        if isinstance(
            direct_pspg_solve_time_signature_magnitude_composite, dict
        )
        else None
    )
    direct_pspg_test10_signature_replay_readiness_finding = (
        direct_pspg_test10_signature_replay_readiness.get("finding")
        if isinstance(direct_pspg_test10_signature_replay_readiness, dict)
        else None
    )
    direct_pspg_test10_signature_row_filter_replay_summary = (
        pressure_update_case_summary(
            direct_pspg_test10_signature_row_filter_replay
        )
    )
    direct_pspg_test10_signature_row_filter_replays_finding = (
        direct_pspg_test10_signature_row_filter_replays.get("finding")
        if isinstance(direct_pspg_test10_signature_row_filter_replays, dict)
        else None
    )
    direct_pspg_ghost_branch_signature_interaction_finding = (
        direct_pspg_ghost_branch_signature_interaction.get("finding")
        if isinstance(direct_pspg_ghost_branch_signature_interaction, dict)
        else None
    )
    direct_pspg_active_pressure_support_selectivity_finding = (
        direct_pspg_active_pressure_support_selectivity.get("finding")
        if isinstance(direct_pspg_active_pressure_support_selectivity, dict)
        else None
    )
    direct_pspg_residual_sign_selectivity_finding = (
        direct_pspg_residual_sign_selectivity.get("finding")
        if isinstance(direct_pspg_residual_sign_selectivity, dict)
        else None
    )
    direct_pspg_null_balance_selectivity_finding = (
        direct_pspg_null_balance_selectivity.get("finding")
        if isinstance(direct_pspg_null_balance_selectivity, dict)
        else None
    )
    direct_pspg_coupled_patch_graph_selectivity_finding = (
        direct_pspg_coupled_patch_graph_selectivity.get("finding")
        if isinstance(direct_pspg_coupled_patch_graph_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_row_provenance_selectivity_finding = (
        direct_pspg_cut_volume_row_provenance_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_row_provenance_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_local_matrix_selectivity_finding = (
        direct_pspg_cut_volume_local_matrix_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_local_matrix_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_local_coupling_selectivity_finding = (
        direct_pspg_cut_volume_local_coupling_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_local_coupling_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_parent_graph_selectivity_finding = (
        direct_pspg_cut_volume_parent_graph_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_parent_graph_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_composite_selectivity_finding = (
        direct_pspg_cut_volume_composite_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_composite_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_column_support_readiness_finding = (
        direct_pspg_cut_volume_column_support_readiness.get("finding")
        if isinstance(direct_pspg_cut_volume_column_support_readiness, dict)
        else None
    )
    direct_pspg_cut_volume_column_support_selectivity_finding = (
        direct_pspg_cut_volume_column_support_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_column_support_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_column_geometry_selectivity_finding = (
        direct_pspg_cut_volume_column_geometry_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_column_geometry_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_quadrature_geometry_selectivity_finding = (
        direct_pspg_cut_volume_quadrature_geometry_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_quadrature_geometry_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_gradient_balance_selectivity_finding = (
        direct_pspg_cut_volume_gradient_balance_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_gradient_balance_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_gradient_column_graph_selectivity_finding = (
        direct_pspg_cut_volume_gradient_column_graph_selectivity.get("finding")
        if isinstance(direct_pspg_cut_volume_gradient_column_graph_selectivity, dict)
        else None
    )
    direct_pspg_cut_volume_local_schur_completion_finding = (
        direct_pspg_cut_volume_local_schur_completion.get("finding")
        if isinstance(direct_pspg_cut_volume_local_schur_completion, dict)
        else None
    )
    direct_pspg_cut_volume_local_edge_balance_finding = (
        direct_pspg_cut_volume_local_edge_balance.get("finding")
        if isinstance(direct_pspg_cut_volume_local_edge_balance, dict)
        else None
    )
    direct_pspg_formulation_vocabulary_support_finding = (
        direct_pspg_formulation_vocabulary_support.get("finding")
        if isinstance(direct_pspg_formulation_vocabulary_support, dict)
        else None
    )
    direct_pspg_assembly_api_support_finding = (
        direct_pspg_assembly_api_support.get("finding")
        if isinstance(direct_pspg_assembly_api_support, dict)
        else None
    )
    direct_pspg_topology_policy_replay_pair_finding = (
        direct_pspg_topology_policy_replay_pair.get("finding")
        if isinstance(direct_pspg_topology_policy_replay_pair, dict)
        else None
    )
    direct_pspg_topology_policy_mode_replays_finding = (
        direct_pspg_topology_policy_mode_replays.get("finding")
        if isinstance(direct_pspg_topology_policy_mode_replays, dict)
        else None
    )
    direct_pspg_topology_policy_application_effect_finding = (
        direct_pspg_topology_policy_application_effect.get("finding")
        if isinstance(direct_pspg_topology_policy_application_effect, dict)
        else None
    )
    direct_pspg_topology_policy_scope_scale_finding = (
        direct_pspg_topology_policy_scope_scale.get("finding")
        if isinstance(direct_pspg_topology_policy_scope_scale, dict)
        else None
    )
    direct_pspg_topology_policy_parent_scope_finding = (
        direct_pspg_topology_policy_parent_scope.get("finding")
        if isinstance(direct_pspg_topology_policy_parent_scope, dict)
        else None
    )
    direct_pspg_topology_policy_parent_subset_readiness_finding = (
        direct_pspg_topology_policy_parent_subset_readiness.get("finding")
        if isinstance(direct_pspg_topology_policy_parent_subset_readiness, dict)
        else None
    )
    direct_pspg_topology_policy_parent_subset_replay_finding = (
        direct_pspg_topology_policy_parent_subset_replay.get("finding")
        if isinstance(direct_pspg_topology_policy_parent_subset_replay, dict)
        else None
    )
    active_pressure_support_cutoff_relevance_finding = (
        active_pressure_support_cutoff_relevance.get("finding")
        if isinstance(active_pressure_support_cutoff_relevance, dict)
        else None
    )
    cut_adjacent_support_window_finding = (
        cut_adjacent_support_window.get("finding")
        if isinstance(cut_adjacent_support_window, dict)
        else None
    )
    pressure_stabilization_driver_windows_finding = (
        pressure_stabilization_driver_windows.get("finding")
        if isinstance(pressure_stabilization_driver_windows, dict)
        else None
    )
    pressure_update_rejection_replay_finding = (
        pressure_update_rejection_replay.get("finding")
        if isinstance(pressure_update_rejection_replay, dict)
        else None
    )
    pressure_update_residual_context_finding = (
        pressure_update_residual_context.get("finding")
        if isinstance(pressure_update_residual_context, dict)
        else None
    )
    no_galerkin_gate_relevance_finding = (
        no_galerkin_gate_relevance.get("finding")
        if isinstance(no_galerkin_gate_relevance, dict)
        else None
    )
    test02_top = case_by_label(top_provenance, "test02")
    test10_top = case_by_label(top_provenance, "test10")
    test02_component_values = values(
        test02_components.get("latest_pressure_update_support_diagnostic")
        if isinstance(test02_components, dict)
        else None
    )
    test10_component_values = values(
        test10_components.get("latest_pressure_update_support_diagnostic")
        if isinstance(test10_components, dict)
        else None
    )
    support_rank_values = values(
        support_rank_guard.get("latest_support_rank_diagnostic")
        if isinstance(support_rank_guard, dict)
        else None
    )
    retained_patch = case_by_label(linear_patch, "retained_cut_volume_support")
    full_volume_patch = case_by_label(linear_patch, "full_volume_one_cell_boundary_topology")
    retained_direct_gap_patch = (
        retained_patch.get("pspg_hydrostatic_balance", {}).get(
            "direct_support_gap_or_same_sign_patch_completion"
        )
        if isinstance(retained_patch, dict)
        else None
    )
    full_volume_direct_gap_patch = (
        full_volume_patch.get("pspg_hydrostatic_balance", {}).get(
            "direct_support_gap_or_same_sign_patch_completion"
        )
        if isinstance(full_volume_patch, dict)
        else None
    )

    hypotheses = [
        status_item(
            key="direct_pspg_pressure_gradient_support_topology",
            question=(
                "Does the active cut-volume direct PSPG pressure-gradient path "
                "create nonphysical pressure updates on the moving interface?"
            ),
            status="supported_unresolved_primary_target",
            conclusion=(
                "Exact top-row provenance classifies Test10 as direct PSPG with "
                "zero ghost-penalty support and Test02 as mixed direct PSPG plus "
                "ghost-positive branch rows. The solver-side component refresh "
                "shows Test10 has no isolated top updates, while Test02's primary "
                "row remains isolated in the bounded top list. The formulation "
                "target map reduces this to isolated Test02-style direct rows "
                "plus a coherent Test10-style direct PSPG patch. The closest "
                "pre-linear graph-completion candidates clear Test10 only with "
                "overbroad pressure patches that destabilize Test02; the "
                "narrowest audited formulation-side candidate is sparse "
                "direct-self entries plus same-sign pressure-action patch "
                "coverage. The global pre-update emission diagnostic now covers "
                "all audited direct targets, but its raw emitted selector is "
                "overbroad. The remaining gap is a formulation-side physical "
                "provenance gate rather than target discovery. Literal mesh "
                "boundary/incident support, named wall/obstacle face "
                "membership, and simple source cut-state provenance are not "
                "selective gates, and the same-sign patch "
                "predicate is blocked as a production rule because the exact "
                "evidence uses post-update pressure signs while tested pre-update "
                "proxies miss targets or overselect. Direct constrained-pressure "
                "neighbor exposure is also not the missing active-support gate: "
                "it selects no audited targets in either case, while sparse "
                "unconstrained pressure-neighbor topology remains the same "
                "overbroad/incomplete sparse-self proxy. Pre-update operator "
                "residual signs do not replace the post-update same-sign oracle "
                "either; the residual-sign pressure-action selectors miss "
                "targets or remain too broad. Direct pressure-gradient row-sum "
                "and diagonal-balance selectors also fail as standalone "
                "physical gates. Matrix-only coupled-patch graph motifs "
                "(two-hop completion, local clustering, articulation rows, and "
                "bridge endpoints) now miss audited targets or cover them only "
                "with overbroad direct PSPG graph subsets. Exact assembly-time "
                "cut-volume row provenance also fails as a simple gate: partial "
                "or low-fraction generated cut-volume support misses every "
                "audited target, while full-cell-only generated-volume support "
                "covers all targets only with broad Test02/Test10 candidate sets. "
                "Assembly-time local cut-volume matrix row-action strength and "
                "single-rule concentration fail too: low/high total action, "
                "high concentration, full-cell-dominant action, low diagonal "
                "fraction, and low support-count selectors miss targets or "
                "remain overbroad. The same is true for local cross-field "
                "pressure/velocity coupling magnitude and ratio: Test02 targets "
                "are high-ratio outliers while Test10 targets mix zero-coupled "
                "and mid-ratio rows, so single scalar coupling gates miss one "
                "branch or stay broad. Row-parent-cell support graph degree, "
                "weighted degree, clustering, and two-hop reach likewise fail "
                "alone: the isolated and coherent branches occupy opposite graph "
                "tails, and tail unions are broad. Bounded composites of those "
                "parent-graph tails with local pressure row action and "
                "pressure/velocity coupling ratio do not close the gap either: "
                "the complete selectors remain broad, while the narrower "
                "ratio/action composites miss the Test10 mid-ratio branch or "
                "the Test02 isolated row. The signed column-support replay now "
                "covers all audited targets, but its coarse support class is "
                "nonselective: every profiled Test02/Test10 candidate and "
                "target row is a null-preserving negative-offdiagonal stencil. "
                "The signed column-support selectivity audit rules out the "
                "next topology/magnitude families too: candidate-neighbor "
                "closure, reciprocal negative edges, the single connected "
                "sampled-column component, degree/two-hop tails, and sampled "
                "edge-concentration or mean-edge-magnitude tails are broad or "
                "miss branch-specific targets. Sampled reference-node edge "
                "geometry also fails as the missing discriminator: complete "
                "finite geometry and diagonal-edge selectors are broad, while "
                "edge-length, axis/diagonal fraction, row-origin, and finite-"
                "edge-count tails miss branch-specific targets. Cut-volume "
                "quadrature geometry closes that follow-up as another simple "
                "gate: all targets have full-cell q-point geometry, while "
                "full-cell, weight, radius, span, parent-cell, and row-to-"
                "centroid selectors are broad or miss branches. Physical "
                "shape-gradient moments and gradient Gram-stencil balance now "
                "fail as the next standalone discriminator too: all audited "
                "targets are full-cell-only gradient-support rows, but that "
                "class is broad, and resultant-gradient, energy, Gram diagonal, "
                "row-sum leakage, matrix-to-Gram scale, sign-mismatch, "
                "negative-Gram, and cosine selectors either miss branches or "
                "remain overbroad. Edge-level sampled pressure-gradient column "
                "graph topology fails as the follow-up too: all targets share "
                "reciprocal all-negative gradient stencils, but that class "
                "still selects broad Test02/Test10 candidate sets, while "
                "component, edge-count, two-hop, Gram-fraction, concentration, "
                "cosine, and sign-mismatch selectors miss targets or remain "
                "overbroad. An element-local Schur support-completion "
                "diagnostic at the cut-volume local-matrix point also covers "
                "every audited target only by touching every preferred direct "
                "PSPG candidate in both cases, so local Schur topology alone "
                "is not a selective formulation rule. Element-local existing-"
                "edge balance is also overbroad. Residual-level shape tangents, "
                "direct PSPG cut-volume scale capping, and free-surface "
                "tangential pressure-gradient probes still exceed the Test02/"
                "Test10 pressure-update guards, while graph-completion selector "
                "coverage shows the shifted maxima escape the weak-row selector "
                "unless its thresholds are widened by case-specific amounts. "
                "The form-vocabulary audit confirms the current public Forms "
                "DSL exposes scalar cut-volume metadata and measures, but not "
                "the active pressure graph, local Schur, edge-balance, or "
                "pressure-action topology handles needed by the remaining "
                "evidence. The assembly API audit now closes the generic "
                "cut-volume source-provenance gap: source-component tags can "
                "flow through FormInstallOptions, the operator registry, "
                "FESystem planning, assembly diagnostic context, and logs. The "
                "production Navier-Stokes direct PSPG pressure-gradient subterm "
                "is now split/tagged under `equations` while preserving its "
                "velocity tangent dependency, and a disabled-by-default local "
                "topology policy can now mutate that tagged pressure-pressure "
                "local matrix before global insertion while preserving the "
                "constant-pressure null. Source-tagged composite cut-volume "
                "groups now also retain diagnostic context by grouping on "
                "source_component_tag. The first API-backed "
                "local_schur_edge_balance replay pair exercises that hook in "
                "the fused production path (3352 Test02 and 720 Test10 policy "
                "applications), but still triggers both accepted pressure-update "
                "guards: Test02 reaches 176844.214 Pa on a tiny-cut-supported "
                "point, and Test10 reaches 522.417 Pa on a full-wet point. This "
                "rules out broad local Schur plus existing-edge balance as a "
                "complete production fix, even though the Test10 same-case "
                "comparison improves from 622.609 Pa without the hook. "
                "The follow-up local-mode replay matrix then separates the "
                "two stages and rules out each as a standalone fix too: "
                "local_schur_completion leaves Test02 at 176849.840 Pa and "
                "Test10 at 590.729 Pa, while local_edge_balance leaves Test02 "
                "at 176848.029 Pa and Test10 at 530.319 Pa. The edge-balance "
                "stage is the Test10-improving part, but it remains far above "
                "the 100 Pa guard; Test02 is essentially unchanged on the same "
                "tiny-cut-supported branch across all local modes. "
                "The active pressure-support cutoff relevance audit now checks "
                "the constraint implementation directly: retained generated-"
                "volume support activation is unconditional, retained volume "
                "fractions are diagnostic-only, and a tiny retained active "
                "fraction can keep pressure DOFs unconstrained. That identifies "
                "a real tiny-cut branch hazard but rules out a retained-"
                "fraction cutoff alone as the complete fix, because adaptive "
                "Test02 rejection shifts to a full-wet row and Test10 remains "
                "full-wet-supported in the latest evidence. "
                "The next discriminator must be a formulation-derived support/"
                "coupling rule beyond sampled column graph, reference-edge, "
                "quadrature geometry, shape-gradient Gram-balance, edge-level "
                "gradient-column graph features, pure element-local Schur "
                "completion, pure existing-edge balance, or another scalar "
                "multiplier in the current form DSL. The coupled-patch "
                "dependency barrier now joins the remaining subsignals and "
                "classifies the current family as requiring new solve-time "
                "direct PSPG pressure-gradient support/coupling provenance "
                "that does not use pressure-update signs. That diagnostic hook "
                "is now source-ready and replayed in the tagged production "
                "direct PSPG assembly path. The short Test02/Test10 provenance "
                "replays cover every audited direct PSPG target row without "
                "using pressure-update signs, but they rule out a simple scalar "
                "pressure-velocity to pressure-pressure coupling gate: the low "
                "Test02 threshold that covers all targets selects 338 rows, the "
                "high Test02 threshold isolates only row 10676, and Test10 "
                "splits between zero-coupling targets and nonzero-coupled "
                "boundary rows. Aggregate solve-time counts and classes are "
                "also not selective: PP/PV record counts, PP edge/two-hop "
                "counts, PV nonzero counts, full/cut record classes, min "
                "volume fraction, and rule counts cover the targets only with "
                "broad row sets. Active-quadrature counts/fractions and "
                "generated-measure classes also stay broad, so the support-"
                "measure path is not the missing gate either. Any rule derived "
                "from this evidence must therefore be a richer topology/coupling "
                "formulation rule, narrower than the tested broad local topology "
                "policy and not a replay of post-update same-sign evidence. The sampled-column "
                "replay then covers every audited target with complete, "
                "untruncated payloads but rules out simple sampled local-stencil "
                "classes as the gate: pressure-pressure shape class, "
                "pressure-velocity sign class, sampled nonzero counts, neighbor "
                "counts, and exact local target signatures all remain broad. "
                "The follow-up "
                "same-parent PP/PV support-coupling signature audit finds a "
                "selective Test10 candidate family (48 rows for 12 targets), "
                "but the matching Test02 target-signature selector is still "
                "broad (276 rows for 7 targets). This rules out a common "
                "parent-cell support/coupling signature gate as the complete "
                "fix. The solve-time signature plus non-oracle magnitude-range "
                "composite audit narrows Test10 further (best 22 rows for 12 "
                "targets) but Test02 remains overbroad (best 53 rows for 7 "
                "targets), so thresholded signature/magnitude composites are "
                "also ruled out as a common formulation gate. The "
                "Test10-specific aggregated-signature row list is now "
                "replayable through the solve-time FE row filter, but the "
                "targeted 48-row replay family still triggers the 100 Pa Test10 "
                "guard in all local modes: Schur-only reaches 619.617 Pa, "
                "edge-balance-only reaches 607.517 Pa, and the combined mode "
                "reaches 604.713 Pa on the same full-wet point. That rules out "
                "exact row-list local topology replay as sufficient and leaves "
                "a stronger Test02 physical discriminator plus a formulation-"
                "side support/coupling rule beyond row-list mutation as the "
                "next step."
            ),
            evidence=[
                top_provenance_evidence | {"finding": top_finding},
                direct_pspg_target_evidence
                | {
                    "finding": direct_pspg_target_finding,
                    "class_counts": (
                        direct_pspg_target.get("formulation_target_class_counts")
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                    "direct_target_case_count": (
                        direct_pspg_target.get("direct_target_case_count")
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                    "ghost_branch_case_count": (
                        direct_pspg_target.get("ghost_branch_case_count")
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                    "recommended_next_predicate": (
                        direct_pspg_target.get("recommended_next_predicate")
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                    "predicate_derivation_readiness": (
                        direct_pspg_target.get("predicate_derivation_readiness")
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                    "complete_diagnostic_candidate_keys": (
                        direct_pspg_target.get("complete_diagnostic_candidate_keys")
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                    "complete_formulation_ready_candidate_keys": (
                        direct_pspg_target.get(
                            "complete_formulation_ready_candidate_keys"
                        )
                        if isinstance(direct_pspg_target, dict)
                        else None
                    ),
                },
                test02_components_evidence
                | {
                    "max_update_global_dof": test02_component_values.get(
                        "max_update_global_dof"
                    ),
                    "isolated_top_update_count": test02_component_values.get(
                        "same_sign_pressure_action_isolated_top_update_count"
                    ),
                },
                test10_components_evidence
                | {
                    "max_update_global_dof": test10_component_values.get(
                        "max_update_global_dof"
                    ),
                    "isolated_top_update_count": test10_component_values.get(
                        "same_sign_pressure_action_isolated_top_update_count"
                    ),
                },
                linear_patch_evidence
                | {
                    "retained_direct_support_gap_patch_ratio": (
                        retained_direct_gap_patch.get(
                            "balanced_max_to_strongest_support_target_response_ratio"
                        )
                        if isinstance(retained_direct_gap_patch, dict)
                        else None
                    ),
                    "full_volume_direct_support_gap_patch_ratio": (
                        full_volume_direct_gap_patch.get(
                            "balanced_max_to_strongest_support_target_response_ratio"
                        )
                        if isinstance(full_volume_direct_gap_patch, dict)
                        else None
                    ),
                    "retained_preserves_hydrostatic_balance": (
                        retained_direct_gap_patch.get(
                            "preserves_hydrostatic_balance"
                        )
                        if isinstance(retained_direct_gap_patch, dict)
                        else None
                    ),
                    "retained_preserves_constant_pressure_null": (
                        retained_direct_gap_patch.get(
                            "preserves_constant_pressure_null"
                        )
                        if isinstance(retained_direct_gap_patch, dict)
                        else None
                    ),
                },
                cut_adjacent_support_window_evidence
                | {
                    "finding": cut_adjacent_support_window_finding,
                    "trace_only_cut_adjacent_support_ruled_out_before_guards": (
                        cut_adjacent_support_window.get(
                            "trace_only_cut_adjacent_support_ruled_out_before_guards"
                        )
                        if isinstance(cut_adjacent_support_window, dict)
                        else None
                    ),
                    "pruned_generated_volume_present_before_some_guard": (
                        cut_adjacent_support_window.get(
                            "pruned_generated_volume_present_before_some_guard"
                        )
                        if isinstance(cut_adjacent_support_window, dict)
                        else None
                    ),
                    "trace_only_cut_adjacent_support_cases": (
                        cut_adjacent_support_window.get(
                            "trace_only_cut_adjacent_support_cases"
                        )
                        if isinstance(cut_adjacent_support_window, dict)
                        else None
                    ),
                    "pruned_generated_volume_cases": (
                        cut_adjacent_support_window.get(
                            "pruned_generated_volume_cases"
                        )
                        if isinstance(cut_adjacent_support_window, dict)
                        else None
                    ),
                    "retained_volume_support_cases": (
                        cut_adjacent_support_window.get(
                            "retained_volume_support_cases"
                        )
                        if isinstance(cut_adjacent_support_window, dict)
                        else None
                    ),
                },
                graph_completion_candidate_readiness_evidence
                | {
                    "finding": graph_completion_candidate_readiness_finding,
                    "overbroad_modes": (
                        graph_completion_candidate_readiness.get("overbroad_modes")
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                    "test02_unstable_modes": (
                        graph_completion_candidate_readiness.get(
                            "test02_unstable_modes"
                        )
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                    "test10_guard_clear_modes": (
                        graph_completion_candidate_readiness.get(
                            "test10_guard_clear_modes"
                        )
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                    "direct_target_counts": (
                        graph_completion_candidate_readiness.get(
                            "direct_target_counts"
                        )
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                },
                formulation_side_candidate_predicates_evidence
                | {
                    "finding": formulation_side_candidate_predicates_finding,
                    "preferred_next_candidate": (
                        formulation_side_candidate_predicates.get(
                            "preferred_next_candidate"
                        )
                        if isinstance(formulation_side_candidate_predicates, dict)
                        else None
                    ),
                    "exact_audited_candidate_keys": (
                        formulation_side_candidate_predicates.get(
                            "exact_audited_candidate_keys"
                        )
                        if isinstance(formulation_side_candidate_predicates, dict)
                        else None
                    ),
                    "partial_candidate_keys": (
                        formulation_side_candidate_predicates.get(
                            "partial_candidate_keys"
                        )
                        if isinstance(formulation_side_candidate_predicates, dict)
                        else None
                    ),
                    "direct_target_counts": (
                        formulation_side_candidate_predicates.get(
                            "direct_target_counts"
                        )
                        if isinstance(formulation_side_candidate_predicates, dict)
                        else None
                    ),
                    "current_artifact_limitation": (
                        formulation_side_candidate_predicates.get(
                            "current_artifact_limitation"
                        )
                        if isinstance(formulation_side_candidate_predicates, dict)
                        else None
                    ),
                },
                direct_pspg_global_candidate_emission_evidence
                | {
                    "finding": direct_pspg_global_candidate_emission_finding,
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_global_candidate_emission.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_emission, dict)
                        else None
                    ),
                    "preferred_candidate_counts": (
                        {
                            case.get("label"): case.get("preferred_candidate_count")
                            for case in direct_pspg_global_candidate_emission.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_emission, dict)
                        else None
                    ),
                    "covered_direct_target_counts": (
                        {
                            case.get("label"): len(
                                case.get("covered_direct_target_global_dofs", [])
                            )
                            for case in direct_pspg_global_candidate_emission.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_emission, dict)
                        else None
                    ),
                    "candidate_list_truncated": (
                        {
                            case.get("label"): case.get("candidate_list_truncated")
                            for case in direct_pspg_global_candidate_emission.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_emission, dict)
                        else None
                    ),
                    "missing_case_labels": (
                        direct_pspg_global_candidate_emission.get(
                            "missing_case_labels"
                        )
                        if isinstance(direct_pspg_global_candidate_emission, dict)
                        else None
                    ),
                },
                direct_pspg_global_candidate_selectivity_evidence
                | {
                    "finding": direct_pspg_global_candidate_selectivity_finding,
                    "preferred_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "preferred_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_direct_self_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "sparse_direct_self_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "direct_self_support_ratio_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "direct_self_support_ratio_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "direct_self_support_ratio_case_findings": (
                        direct_pspg_global_candidate_selectivity.get(
                            "direct_self_support_ratio_case_findings"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_or_moderate_direct_self_ratio_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "sparse_or_moderate_direct_self_ratio_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_or_moderate_direct_self_ratio_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "sparse_or_moderate_direct_self_ratio_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_or_moderate_direct_self_ratio_selector_overbroad": (
                        {
                            case.get("label"): case.get(
                                "sparse_or_moderate_direct_self_ratio_selector_overbroad"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "graph_local_support_ratio_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "graph_local_support_ratio_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "graph_local_support_ratio_case_findings": (
                        direct_pspg_global_candidate_selectivity.get(
                            "graph_local_support_ratio_case_findings"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "graph_local_moderate_direct_self_ratio_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "graph_local_moderate_direct_self_ratio_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "graph_local_moderate_direct_self_ratio_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "graph_local_moderate_direct_self_ratio_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "graph_local_moderate_direct_self_ratio_selector_overbroad": (
                        {
                            case.get("label"): case.get(
                                "graph_local_moderate_direct_self_ratio_selector_overbroad"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius1_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "sparse_seeded_pressure_action_radius1_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius2_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "sparse_seeded_pressure_action_radius2_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius1_case_findings": (
                        direct_pspg_global_candidate_selectivity.get(
                            "sparse_seeded_pressure_action_radius1_case_findings"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius2_case_findings": (
                        direct_pspg_global_candidate_selectivity.get(
                            "sparse_seeded_pressure_action_radius2_case_findings"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius1_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_pressure_action_radius1_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius2_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_pressure_action_radius2_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius1_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_pressure_action_radius1_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_pressure_action_radius2_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_pressure_action_radius2_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_moderate_degree_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "pressure_action_moderate_degree_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_moderate_sum_ratio_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "pressure_action_moderate_sum_ratio_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_self_dominant_gate_finding": (
                        direct_pspg_global_candidate_selectivity.get(
                            "pressure_action_self_dominant_gate_finding"
                        )
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_moderate_degree_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "pressure_action_moderate_degree_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_moderate_degree_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "pressure_action_moderate_degree_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_moderate_sum_ratio_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "pressure_action_moderate_sum_ratio_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_moderate_sum_ratio_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "pressure_action_moderate_sum_ratio_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_self_dominant_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "pressure_action_self_dominant_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "pressure_action_self_dominant_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "pressure_action_self_dominant_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "matrix_pressure_action_covers_all_direct_rows": (
                        {
                            case.get("label"): case.get(
                                "matrix_pressure_action_covers_all_direct_rows"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_matrix_pressure_action_component_counts": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_matrix_pressure_action_component_dof_count"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_matrix_pressure_action_component_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_matrix_pressure_action_component_to_target_ratio"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_matrix_pressure_action_component_covers_targets": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_matrix_pressure_action_component_covers_targets"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "sparse_seeded_matrix_pressure_action_component_selector_overbroad": (
                        {
                            case.get("label"): case.get(
                                "sparse_seeded_matrix_pressure_action_component_selector_overbroad"
                            )
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_global_candidate_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_global_candidate_selectivity, dict)
                        else None
                    ),
                },
                direct_pspg_boundary_provenance_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_boundary_provenance_selectivity_finding
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_boundary_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_boundary_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_boundary_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_boundary_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_boundary_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_boundary_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "profile_status": (
                        {
                            label: evidence.get("profile_status")
                            for label, evidence in direct_pspg_boundary_provenance_selectivity.get(
                                "profile_evidence", {}
                            ).items()
                            if isinstance(evidence, dict)
                        }
                        if isinstance(
                            direct_pspg_boundary_provenance_selectivity, dict
                        )
                        else None
                    ),
                },
                direct_pspg_cut_state_provenance_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_state_provenance_selectivity_finding
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_state_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_state_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_state_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_state_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_state_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_state_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "target_wet_support_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "target_wet_support_class_counts"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_state_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_state_provenance_selectivity, dict
                        )
                        else None
                    ),
                    "profile_status": (
                        {
                            label: evidence.get("profile_status")
                            for label, evidence in direct_pspg_cut_state_provenance_selectivity.get(
                                "profile_evidence", {}
                            ).items()
                            if isinstance(evidence, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_state_provenance_selectivity, dict
                        )
                        else None
                    ),
                },
                direct_pspg_same_sign_dependency_readiness_evidence
                | {
                    "finding": (
                        direct_pspg_same_sign_dependency_readiness_finding
                    ),
                    "preferred_candidate_depends_on_pressure_update": (
                        direct_pspg_same_sign_dependency_readiness.get(
                            "dependency_summary", {}
                        ).get("preferred_candidate_depends_on_pressure_update")
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "all_exact_candidates_depend_on_pressure_update": (
                        direct_pspg_same_sign_dependency_readiness.get(
                            "dependency_summary", {}
                        ).get("all_exact_candidates_depend_on_pressure_update")
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "complete_non_update_dependent_candidate_keys": (
                        direct_pspg_same_sign_dependency_readiness.get(
                            "dependency_summary", {}
                        ).get("complete_non_update_dependent_candidate_keys")
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "all_preupdate_proxy_gates_failed": (
                        direct_pspg_same_sign_dependency_readiness.get(
                            "preupdate_proxy_summary", {}
                        ).get("all_preupdate_proxy_gates_failed")
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "failed_preupdate_proxy_gate_keys": (
                        direct_pspg_same_sign_dependency_readiness.get(
                            "preupdate_proxy_summary", {}
                        ).get("failed_gate_keys")
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "cross_policy_patch_finding": (
                        direct_pspg_same_sign_dependency_readiness.get(
                            "cross_policy_patch_summary", {}
                        ).get("finding")
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "cross_policy_patch_case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_same_sign_dependency_readiness.get(
                                "cross_policy_patch_summary", {}
                            ).get("cases", [])
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                    "cross_policy_patch_dofs": (
                        {
                            case.get("label"): case.get(
                                "pressure_disabled_direct_patch_global_dofs"
                            )
                            for case in direct_pspg_same_sign_dependency_readiness.get(
                                "cross_policy_patch_summary", {}
                            ).get("cases", [])
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_sign_dependency_readiness, dict
                        )
                        else None
                    ),
                },
                direct_pspg_active_pressure_support_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_active_pressure_support_selectivity_finding
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_active_pressure_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_active_pressure_support_selectivity, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_active_pressure_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_active_pressure_support_selectivity, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_active_pressure_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_active_pressure_support_selectivity, dict
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_active_pressure_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_active_pressure_support_selectivity, dict
                        )
                        else None
                    ),
                },
                direct_pspg_residual_sign_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_residual_sign_selectivity_finding
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_residual_sign_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_residual_sign_selectivity, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_residual_sign_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_residual_sign_selectivity, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_residual_sign_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_residual_sign_selectivity, dict
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_residual_sign_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_residual_sign_selectivity, dict
                        )
                        else None
                    ),
                    "residual_signal_by_case": (
                        direct_pspg_residual_sign_selectivity.get(
                            "residual_signal_by_case"
                        )
                        if isinstance(
                            direct_pspg_residual_sign_selectivity, dict
                        )
                        else None
                    ),
                },
                direct_pspg_null_balance_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_null_balance_selectivity_finding
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_null_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_null_balance_selectivity, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_null_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_null_balance_selectivity, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_null_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_null_balance_selectivity, dict
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_null_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_null_balance_selectivity, dict
                        )
                        else None
                    ),
                    "null_balance_by_case": (
                        direct_pspg_null_balance_selectivity.get(
                            "null_balance_by_case"
                        )
                        if isinstance(
                            direct_pspg_null_balance_selectivity, dict
                        )
                        else None
                    ),
                },
                direct_pspg_coupled_patch_graph_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_coupled_patch_graph_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_coupled_patch_graph_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_coupled_patch_graph_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_coupled_patch_graph_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_coupled_patch_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_coupled_patch_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_coupled_patch_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                    "graph_topology_by_case": (
                        direct_pspg_coupled_patch_graph_selectivity.get(
                            "graph_topology_by_case"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_graph_selectivity, dict
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_row_provenance_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_row_provenance_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_row_provenance_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_row_provenance_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_row_provenance_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "candidate_support_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("candidate_support_class_counts")
                            for case in direct_pspg_cut_volume_row_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_row_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_row_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_row_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_row_provenance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_row_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_local_matrix_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_local_matrix_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_local_matrix_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_local_matrix_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_local_matrix_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_local_matrix_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_local_matrix_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_local_matrix_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_matrix_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_matrix_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_matrix_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_local_coupling_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_local_coupling_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_local_coupling_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_local_coupling_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_local_coupling_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_local_coupling_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_local_coupling_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_local_coupling_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_coupling_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_coupling_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_coupling_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_parent_graph_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_parent_graph_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_parent_graph_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_parent_graph_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_parent_graph_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_parent_graph_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_parent_graph_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_parent_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_parent_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_parent_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_parent_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_composite_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_composite_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_composite_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_composite_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_composite_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_composite_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_composite_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_composite_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_composite_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_composite_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_composite_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_composite_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_column_support_readiness_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_column_support_readiness_finding
                    ),
                    "missing_case_labels": (
                        direct_pspg_cut_volume_column_support_readiness.get(
                            "missing_case_labels"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "case_log_status": (
                        {
                            case.get("label"): case.get(
                                "log_evidence", {}
                            ).get("status")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "latest_batch_entry_counts": (
                        {
                            case.get("label"): case.get(
                                "log_evidence", {}
                            ).get("latest_batch_entry_count")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "profiled_candidate_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("profiled_candidate_count")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "profiled_target_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("profiled_target_count")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "unprofiled_target_global_dofs_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("unprofiled_target_global_dofs")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "candidate_column_support_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("candidate_column_support_class_counts")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "target_column_support_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_column_support_class_counts")
                            for case in direct_pspg_cut_volume_column_support_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_column_support_readiness.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_support_readiness,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_column_support_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_column_support_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_column_support_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_column_support_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_column_support_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_column_support_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_column_support_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_column_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_column_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_column_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_column_support_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_column_support_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_support_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_column_geometry_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_column_geometry_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_column_geometry_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_column_geometry_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_column_geometry_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "geometry_field_entry_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "log_evidence", {}
                            ).get("geometry_field_entry_count")
                            for case in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "candidate_reference_geometry_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("candidate_reference_geometry_class_counts")
                            for case in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_reference_geometry_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_reference_geometry_class_counts")
                            for case in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_column_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_column_geometry_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_column_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_quadrature_geometry_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_quadrature_geometry_selectivity_finding
                    ),
                    "selective_selector_keys": (
                        direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                            "selective_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "overbroad_selector_keys": (
                        direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                            "overbroad_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "miss_selector_keys": (
                        direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                            "miss_selector_keys"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "cut_qpoint_field_entry_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "log_evidence", {}
                            ).get("cut_qpoint_field_entry_count")
                            for case in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "thresholds_by_case": (
                        {
                            case.get("label"): case.get("thresholds")
                            for case in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "candidate_cut_qpoint_geometry_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("candidate_cut_qpoint_geometry_class_counts")
                            for case in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_cut_qpoint_geometry_class_counts_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_cut_qpoint_geometry_class_counts")
                            for case in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_profiles_by_case": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_profiles")
                            for case in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_quadrature_geometry_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_quadrature_geometry_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_gradient_balance_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_gradient_balance_selectivity_finding
                    ),
                    "case_log_status": (
                        {
                            case.get("label"): case.get("log_evidence", {}).get(
                                "status"
                            )
                            for case in direct_pspg_cut_volume_gradient_balance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "profiled_target_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("profiled_target_count")
                            for case in direct_pspg_cut_volume_gradient_balance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_gradient_support_class_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_gradient_support_class_counts")
                            for case in direct_pspg_cut_volume_gradient_balance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_gradient_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_gradient_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_gradient_balance_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_gradient_balance_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_gradient_balance_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_gradient_column_graph_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_gradient_column_graph_selectivity_finding
                    ),
                    "case_log_status": (
                        {
                            case.get("label"): case.get("log_evidence", {}).get(
                                "status"
                            )
                            for case in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "profiled_target_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("profiled_target_count")
                            for case in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_edge_class_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("target_edge_class_counts")
                            for case in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "candidate_edge_class_counts": (
                        {
                            case.get("label"): case.get(
                                "profile_summary", {}
                            ).get("candidate_edge_class_counts")
                            for case in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_gradient_column_graph_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_gradient_column_graph_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                graph_completion_replay_family_evidence
                | {
                    "finding": graph_completion_replay_family_finding,
                    "variant_findings": (
                        graph_completion_replay_family.get("variant_findings")
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "test10_guard_clear_variants": (
                        graph_completion_replay_family.get(
                            "test10_guard_clear_variants"
                        )
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "test02_unstable_variants": (
                        graph_completion_replay_family.get(
                            "test02_unstable_variants"
                        )
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "test10_still_trigger_variants": (
                        graph_completion_replay_family.get(
                            "test10_still_trigger_variants"
                        )
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "candidate_counts_by_variant": (
                        {
                            variant.get("key"): {
                                case.get("label"): case.get("candidate_row_count")
                                for case in variant.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for variant in graph_completion_replay_family.get(
                                "variants", []
                            )
                            if isinstance(variant, dict)
                        }
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "accepted_pressure_updates_by_variant": (
                        {
                            variant.get("key"): {
                                case.get("label"): case.get(
                                    "accepted_pressure_update_pa"
                                )
                                for case in variant.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for variant in graph_completion_replay_family.get(
                                "variants", []
                            )
                            if isinstance(variant, dict)
                        }
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "case_findings_by_variant": (
                        {
                            variant.get("key"): {
                                case.get("label"): case.get("finding")
                                for case in variant.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for variant in graph_completion_replay_family.get(
                                "variants", []
                            )
                            if isinstance(variant, dict)
                        }
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "next_requirement": (
                        graph_completion_replay_family.get("next_requirement")
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                },
                direct_pspg_cut_volume_local_schur_completion_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_local_schur_completion_finding
                    ),
                    "aggregate_selector_finding": (
                        direct_pspg_cut_volume_local_schur_completion.get(
                            "aggregate_selector_finding"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                    "case_log_status": (
                        {
                            case.get("label"): case.get("log_evidence", {}).get(
                                "status"
                            )
                            for case in direct_pspg_cut_volume_local_schur_completion.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                    "selected_counts_by_case": (
                        {
                            case.get("label"): case.get("selector", {}).get(
                                "selected_count"
                            )
                            for case in direct_pspg_cut_volume_local_schur_completion.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_case": (
                        {
                            case.get("label"): case.get("selector", {}).get(
                                "covered_direct_target_count"
                            )
                            for case in direct_pspg_cut_volume_local_schur_completion.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_case": (
                        {
                            case.get("label"): case.get("selector", {}).get(
                                "selected_to_target_ratio"
                            )
                            for case in direct_pspg_cut_volume_local_schur_completion.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                    "summary_metrics_by_case": (
                        {
                            case.get("label"): case.get("summary_metrics")
                            for case in direct_pspg_cut_volume_local_schur_completion.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_local_schur_completion.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_schur_completion, dict
                        )
                        else None
                    ),
                },
                direct_pspg_cut_volume_local_edge_balance_evidence
                | {
                    "finding": (
                        direct_pspg_cut_volume_local_edge_balance_finding
                    ),
                    "aggregate_selector_findings": (
                        direct_pspg_cut_volume_local_edge_balance.get(
                            "aggregate_selector_findings"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "selector_findings": (
                        {
                            selector.get("key"): selector.get("finding")
                            for selector in direct_pspg_cut_volume_local_edge_balance.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "case_log_status": (
                        {
                            case.get("label"): case.get("log_evidence", {}).get(
                                "status"
                            )
                            for case in direct_pspg_cut_volume_local_edge_balance.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "selected_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get("selected_count")
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_edge_balance.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "covered_target_counts_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "covered_direct_target_count"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_edge_balance.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "selected_to_target_ratios_by_selector": (
                        {
                            selector.get("key"): {
                                case.get("label"): case.get(
                                    "selected_to_target_ratio"
                                )
                                for case in selector.get("cases", [])
                                if isinstance(case, dict)
                            }
                            for selector in direct_pspg_cut_volume_local_edge_balance.get(
                                "selectors", []
                            )
                            if isinstance(selector, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "summary_metrics_by_case": (
                        {
                            case.get("label"): case.get("summary_metrics")
                            for case in direct_pspg_cut_volume_local_edge_balance.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_cut_volume_local_edge_balance.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_cut_volume_local_edge_balance, dict
                        )
                        else None
                    ),
                },
                test02_shape_tangent_evidence
                | {
                    "case": "test02",
                    "control_variant": "residual_shape_tangent",
                    **pressure_update_case_summary(test02_shape_tangent),
                },
                shape_tangent_evidence
                | {
                    "case": "test10",
                    "control_variant": "residual_shape_tangent",
                    **pressure_update_case_summary(shape_tangent),
                },
                test02_cut_volume_scale_cap16_evidence
                | {
                    "case": "test02",
                    "control_variant": "direct_pspg_cut_volume_scale_cap16",
                    **pressure_update_case_summary(test02_cut_volume_scale_cap16),
                },
                test10_cut_volume_scale_cap16_evidence
                | {
                    "case": "test10",
                    "control_variant": "direct_pspg_cut_volume_scale_cap16",
                    **pressure_update_case_summary(test10_cut_volume_scale_cap16),
                },
                test02_free_surface_tangential_pressure_update_evidence
                | {
                    "case": "test02",
                    "control_variant": (
                        "free_surface_tangential_pressure_gradient"
                    ),
                    **pressure_update_case_summary(
                        test02_free_surface_tangential_pressure_update
                    ),
                },
                test10_free_surface_tangential_pressure_update_evidence
                | {
                    "case": "test10",
                    "control_variant": (
                        "free_surface_tangential_pressure_gradient"
                    ),
                    **pressure_update_case_summary(
                        test10_free_surface_tangential_pressure_update
                    ),
                },
                graph_completion_selector_coverage_evidence
                | graph_completion_selector_coverage_summary(
                    graph_completion_selector_coverage
                ),
                direct_pspg_formulation_vocabulary_support_evidence
                | {
                    "finding": (
                        direct_pspg_formulation_vocabulary_support_finding
                    ),
                    "status": (
                        direct_pspg_formulation_vocabulary_support.get("status")
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "direct_pspg_expression_summary": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "direct_pspg_expression_summary"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "public_cut_cell_helpers": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "public_cut_cell_helpers"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "public_measures": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "public_measures"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "required_topology_handles_missing": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "required_topology_handles_missing"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "missing_required_topology_handle_count": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "missing_required_topology_handle_count"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "required_topology_handle_count": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "required_topology_handle_count"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_formulation_vocabulary_support.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_formulation_vocabulary_support, dict
                        )
                        else None
                    ),
                },
                direct_pspg_assembly_api_support_evidence
                | {
                    "finding": direct_pspg_assembly_api_support_finding,
                    "status": (
                        direct_pspg_assembly_api_support.get("status")
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "assembly_diagnostic_context_fields": (
                        direct_pspg_assembly_api_support.get(
                            "assembly_diagnostic_context_fields"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "planned_cut_volume_term_fields": (
                        direct_pspg_assembly_api_support.get(
                            "planned_cut_volume_term_fields"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "assembly_api_features": (
                        direct_pspg_assembly_api_support.get(
                            "assembly_api_features"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "required_api_handles_missing": (
                        direct_pspg_assembly_api_support.get(
                            "required_api_handles_missing"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "missing_required_api_handle_count": (
                        direct_pspg_assembly_api_support.get(
                            "missing_required_api_handle_count"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "required_api_handle_count": (
                        direct_pspg_assembly_api_support.get(
                            "required_api_handle_count"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_assembly_api_support.get(
                            "next_requirement"
                        )
                        if isinstance(direct_pspg_assembly_api_support, dict)
                        else None
                    ),
                },
                direct_pspg_topology_policy_replay_pair_evidence
                | {
                    "finding": direct_pspg_topology_policy_replay_pair_finding,
                    "status": (
                        direct_pspg_topology_policy_replay_pair.get("status")
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                    "policy": (
                        direct_pspg_topology_policy_replay_pair.get("policy")
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                    "policy_hook_exercised": (
                        direct_pspg_topology_policy_replay_pair.get(
                            "policy_hook_exercised"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                    "policy_log_counts": (
                        direct_pspg_topology_policy_replay_pair.get(
                            "policy_log_counts"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                    "pressure_update_guard_cleared": (
                        direct_pspg_topology_policy_replay_pair.get(
                            "pressure_update_guard_cleared"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                    "case_summaries": (
                        [
                            {
                                "label": case.get("label"),
                                "guard_status": case.get("guard_status"),
                                "policy_log_count": case.get("policy_log_count"),
                                "worst_active_or_wet_update_pa": case.get(
                                    "worst_active_or_wet_update_pa"
                                ),
                                "worst_active_or_wet_support_class": case.get(
                                    "worst_active_or_wet_support_class"
                                ),
                                "absolute_threshold_pa": case.get(
                                    "absolute_threshold_pa"
                                ),
                            }
                            for case in direct_pspg_topology_policy_replay_pair.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        ]
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_replay_pair.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_replay_pair, dict
                        )
                        else None
                    ),
                },
                direct_pspg_topology_policy_mode_replays_evidence
                | {
                    "finding": direct_pspg_topology_policy_mode_replays_finding,
                    "status": (
                        direct_pspg_topology_policy_mode_replays.get("status")
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "policies_tested": (
                        direct_pspg_topology_policy_mode_replays.get(
                            "policies_tested"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "policy_hook_exercised": (
                        direct_pspg_topology_policy_mode_replays.get(
                            "policy_hook_exercised"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "policy_log_counts": (
                        direct_pspg_topology_policy_mode_replays.get(
                            "policy_log_counts"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "pressure_update_guard_cleared": (
                        direct_pspg_topology_policy_mode_replays.get(
                            "pressure_update_guard_cleared"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "case_policy_results": (
                        [
                            {
                                "case": result.get("case"),
                                "policy": result.get("policy"),
                                "guard_status": result.get("guard_status"),
                                "policy_log_count": result.get(
                                    "policy_log_count"
                                ),
                                "worst_active_or_wet_update_pa": result.get(
                                    "worst_active_or_wet_update_pa"
                                ),
                                "worst_active_or_wet_support_class": result.get(
                                    "worst_active_or_wet_support_class"
                                ),
                                "absolute_threshold_pa": result.get(
                                    "absolute_threshold_pa"
                                ),
                            }
                            for result in direct_pspg_topology_policy_mode_replays.get(
                                "case_policy_results", []
                            )
                            if isinstance(result, dict)
                        ]
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "mode_interpretation": (
                        direct_pspg_topology_policy_mode_replays.get(
                            "mode_interpretation"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_mode_replays.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_mode_replays, dict
                        )
                        else None
                    ),
                },
                direct_pspg_topology_policy_application_effect_evidence
                | {
                    "finding": (
                        direct_pspg_topology_policy_application_effect_finding
                    ),
                    "status": (
                        direct_pspg_topology_policy_application_effect.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_application_effect,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_topology_policy_application_effect.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_application_effect,
                            dict,
                        )
                        else None
                    ),
                    "all_test10_signature_replays_mutate_selected_records": (
                        direct_pspg_topology_policy_application_effect.get(
                            "all_test10_signature_replays_mutate_selected_records"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_application_effect,
                            dict,
                        )
                        else None
                    ),
                    "best_updates": (
                        {
                            "test02_broad": {
                                "policy": direct_pspg_topology_policy_application_effect.get(
                                    "best_test02_broad_policy"
                                ),
                                "update_pa": direct_pspg_topology_policy_application_effect.get(
                                    "best_test02_broad_update_pa"
                                ),
                            },
                            "test10_broad": {
                                "policy": direct_pspg_topology_policy_application_effect.get(
                                    "best_test10_broad_policy"
                                ),
                                "update_pa": direct_pspg_topology_policy_application_effect.get(
                                    "best_test10_broad_update_pa"
                                ),
                            },
                            "test10_signature": {
                                "policy": direct_pspg_topology_policy_application_effect.get(
                                    "best_test10_signature_policy"
                                ),
                                "update_pa": direct_pspg_topology_policy_application_effect.get(
                                    "best_test10_signature_update_pa"
                                ),
                            },
                        }
                        if isinstance(
                            direct_pspg_topology_policy_application_effect,
                            dict,
                        )
                        else None
                    ),
                    "test10_broad_vs_signature_row_filter": (
                        direct_pspg_topology_policy_application_effect.get(
                            "test10_broad_vs_signature_row_filter"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_application_effect,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_application_effect.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_application_effect,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_topology_policy_scope_scale_evidence
                | {
                    "finding": direct_pspg_topology_policy_scope_scale_finding,
                    "status": (
                        direct_pspg_topology_policy_scope_scale.get("status")
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                    "same_case_no_policy_test10_update_pa": (
                        direct_pspg_topology_policy_scope_scale.get(
                            "same_case_no_policy_test10_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_topology_policy_scope_scale.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                    "signature_rows_worse_than_broad_for_all_test10_modes": (
                        direct_pspg_topology_policy_scope_scale.get(
                            "signature_rows_worse_than_broad_for_all_test10_modes"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                    "test10_broad_vs_signature_row_filter": (
                        direct_pspg_topology_policy_scope_scale.get(
                            "test10_broad_vs_signature_row_filter"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                    "test02_broad_policy_scope": (
                        direct_pspg_topology_policy_scope_scale.get(
                            "test02_broad_policy_scope"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_scope_scale.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_scope_scale, dict
                        )
                        else None
                    ),
                },
                direct_pspg_topology_policy_parent_scope_evidence
                | {
                    "finding": direct_pspg_topology_policy_parent_scope_finding,
                    "status": (
                        direct_pspg_topology_policy_parent_scope.get("status")
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "same_case_no_policy_test10_update_pa": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "same_case_no_policy_test10_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "all_test10_signature_parent_rule_sets_are_strict_broad_subsets": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "all_test10_signature_parent_rule_sets_are_strict_broad_subsets"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "all_test10_broad_only_rule_weight_share_above_half": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "all_test10_broad_only_rule_weight_share_above_half"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "signature_rows_worse_than_broad_for_all_test10_modes": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "signature_rows_worse_than_broad_for_all_test10_modes"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "test10_parent_rule_scope": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "test10_parent_rule_scope"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "test02_broad_parent_rule_scope": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "test02_broad_parent_rule_scope"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_parent_scope.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_scope, dict
                        )
                        else None
                    ),
                },
                direct_pspg_topology_policy_parent_subset_readiness_evidence
                | {
                    "finding": (
                        direct_pspg_topology_policy_parent_subset_readiness_finding
                    ),
                    "status": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                    "source_hook": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "source_hook"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                    "same_signature_parent_set_all_policies": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "same_signature_parent_set_all_policies"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                    "signature_parent_cell_count": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "signature_parent_cell_count"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                    "signature_parent_cell_ranges": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "signature_parent_cell_ranges"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                    "parent_scope": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "parent_scope"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_parent_subset_readiness.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_readiness,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_topology_policy_parent_subset_replay_evidence
                | {
                    "finding": (
                        direct_pspg_topology_policy_parent_subset_replay_finding
                    ),
                    "status": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "signature_parent_filter_full_local_confirmed": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "signature_parent_filter_full_local_confirmed"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "signature_parent_filter_update_pa": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "signature_parent_filter_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "broad_policy_update_pa": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "broad_policy_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "signature_row_filter_update_pa": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "signature_row_filter_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "same_case_no_policy_update_pa": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "same_case_no_policy_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "parent_minus_broad_update_pa": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "parent_minus_broad_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "parent_minus_signature_row_update_pa": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "parent_minus_signature_row_update_pa"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "pressure_update_guard_cleared": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "pressure_update_guard_cleared"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "replays": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "replays"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_topology_policy_parent_subset_replay.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_topology_policy_parent_subset_replay,
                            dict,
                        )
                        else None
                    ),
                },
                active_pressure_support_cutoff_relevance_evidence
                | {
                    "finding": active_pressure_support_cutoff_relevance_finding,
                    "status": (
                        active_pressure_support_cutoff_relevance.get("status")
                        if isinstance(
                            active_pressure_support_cutoff_relevance, dict
                        )
                        else None
                    ),
                    "constraint_source": (
                        active_pressure_support_cutoff_relevance.get(
                            "constraint_source"
                        )
                        if isinstance(
                            active_pressure_support_cutoff_relevance, dict
                        )
                        else None
                    ),
                    "classification": (
                        active_pressure_support_cutoff_relevance.get(
                            "classification"
                        )
                        if isinstance(
                            active_pressure_support_cutoff_relevance, dict
                        )
                        else None
                    ),
                    "topology_policy_replay_summary": (
                        active_pressure_support_cutoff_relevance.get(
                            "topology_policy_replay_summary"
                        )
                        if isinstance(
                            active_pressure_support_cutoff_relevance, dict
                        )
                        else None
                    ),
                    "pressure_update_rejection_summary": (
                        active_pressure_support_cutoff_relevance.get(
                            "pressure_update_rejection_summary"
                        )
                        if isinstance(
                            active_pressure_support_cutoff_relevance, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        active_pressure_support_cutoff_relevance.get(
                            "next_requirement"
                        )
                        if isinstance(
                            active_pressure_support_cutoff_relevance, dict
                        )
                        else None
                    ),
                },
                direct_pspg_coupled_patch_dependency_barrier_evidence
                | {
                    "finding": (
                        direct_pspg_coupled_patch_dependency_barrier_finding
                    ),
                    "status": (
                        direct_pspg_coupled_patch_dependency_barrier.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_dependency_barrier, dict
                        )
                        else None
                    ),
                    "blocker_summary": (
                        direct_pspg_coupled_patch_dependency_barrier.get(
                            "blocker_summary"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_dependency_barrier, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_coupled_patch_dependency_barrier.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_coupled_patch_dependency_barrier, dict
                        )
                        else None
                    ),
                },
                direct_pspg_solve_time_provenance_support_evidence
                | {
                    "finding": direct_pspg_solve_time_provenance_support_finding,
                    "status": (
                        direct_pspg_solve_time_provenance_support.get("status")
                        if isinstance(
                            direct_pspg_solve_time_provenance_support, dict
                        )
                        else None
                    ),
                    "features": (
                        direct_pspg_solve_time_provenance_support.get(
                            "features"
                        )
                        if isinstance(
                            direct_pspg_solve_time_provenance_support, dict
                        )
                        else None
                    ),
                    "diagnostic_env": (
                        direct_pspg_solve_time_provenance_support.get(
                            "diagnostic_env"
                        )
                        if isinstance(
                            direct_pspg_solve_time_provenance_support, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_provenance_support.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_provenance_support, dict
                        )
                        else None
                    ),
                },
                direct_pspg_solve_time_provenance_replay_evidence
                | {
                    "finding": direct_pspg_solve_time_provenance_replay_finding,
                    "status": (
                        direct_pspg_solve_time_provenance_replay.get("status")
                        if isinstance(
                            direct_pspg_solve_time_provenance_replay, dict
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_provenance_replay.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_solve_time_provenance_replay, dict)
                        else None
                    ),
                    "record_counts": (
                        {
                            case.get("label"): case.get("record_count")
                            for case in direct_pspg_solve_time_provenance_replay.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_solve_time_provenance_replay, dict)
                        else None
                    ),
                    "target_rows_present_counts": (
                        {
                            case.get("label"): case.get(
                                "target_rows_present_count"
                            )
                            for case in direct_pspg_solve_time_provenance_replay.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_solve_time_provenance_replay, dict)
                        else None
                    ),
                    "max_target_ratio_rows": (
                        {
                            case.get("label"): case.get("max_target_ratio_rows")
                            for case in direct_pspg_solve_time_provenance_replay.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_solve_time_provenance_replay, dict)
                        else None
                    ),
                    "zero_pressure_velocity_target_global_dofs": (
                        {
                            case.get("label"): case.get(
                                "zero_pressure_velocity_target_global_dofs"
                            )
                            for case in direct_pspg_solve_time_provenance_replay.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(direct_pspg_solve_time_provenance_replay, dict)
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_provenance_replay.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_provenance_replay, dict
                        )
                        else None
                    ),
                },
                direct_pspg_solve_time_sampled_column_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_sampled_column_selectivity_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_sampled_column_selectivity.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_sampled_column_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_sampled_column_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_sampled_column_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "record_counts": (
                        {
                            case.get("label"): case.get("record_count")
                            for case in direct_pspg_solve_time_sampled_column_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_sampled_column_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_rows_present_counts": (
                        {
                            case.get("label"): case.get(
                                "target_rows_present_count"
                            )
                            for case in direct_pspg_solve_time_sampled_column_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_sampled_column_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "any_sample_truncated": (
                        {
                            case.get("label"): case.get("any_sample_truncated")
                            for case in direct_pspg_solve_time_sampled_column_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_sampled_column_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_sampled_column_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_sampled_column_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_solve_time_support_coupling_signature_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_support_coupling_signature_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_support_coupling_signature.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_support_coupling_signature,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_support_coupling_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_coupling_signature,
                            dict,
                        )
                        else None
                    ),
                    "target_support_class_counts": (
                        {
                            case.get("label"): case.get(
                                "target_same_parent_pressure_velocity_support_class_counts"
                            )
                            for case in direct_pspg_solve_time_support_coupling_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_coupling_signature,
                            dict,
                        )
                        else None
                    ),
                    "exact_local_signature_selected_counts": (
                        {
                            case.get("label"): case.get(
                                "exact_local_signature_selected_count"
                            )
                            for case in direct_pspg_solve_time_support_coupling_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_coupling_signature,
                            dict,
                        )
                        else None
                    ),
                    "exact_local_signature_selected_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "exact_local_signature_selected_to_target_ratio"
                            )
                            for case in direct_pspg_solve_time_support_coupling_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_coupling_signature,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_support_coupling_signature.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_support_coupling_signature,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_solve_time_magnitude_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_magnitude_selectivity_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_magnitude_selectivity.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_magnitude_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_magnitude_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_magnitude_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "range_selector_selected_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "range_selector_selected_to_target_ratios"
                            )
                            for case in direct_pspg_solve_time_magnitude_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_magnitude_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "exact_value_oracle_selector_keys": (
                        {
                            case.get("label"): case.get(
                                "exact_value_oracle_selector_keys"
                            )
                            for case in direct_pspg_solve_time_magnitude_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_magnitude_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_magnitude_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_magnitude_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_solve_time_signature_magnitude_composite_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_signature_magnitude_composite_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_signature_magnitude_composite.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_signature_magnitude_composite,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_signature_magnitude_composite.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_signature_magnitude_composite,
                            dict,
                        )
                        else None
                    ),
                    "best_covering_composite_selected_counts": (
                        {
                            case.get("label"): case.get(
                                "best_covering_composite_selected_count"
                            )
                            for case in direct_pspg_solve_time_signature_magnitude_composite.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_signature_magnitude_composite,
                            dict,
                        )
                        else None
                    ),
                    "best_covering_composite_selected_to_target_ratios": (
                        {
                            case.get("label"): case.get(
                                "best_covering_composite_selected_to_target_ratio"
                            )
                            for case in direct_pspg_solve_time_signature_magnitude_composite.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_signature_magnitude_composite,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_signature_magnitude_composite.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_signature_magnitude_composite,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_test10_signature_replay_readiness_evidence
                | {
                    "finding": (
                        direct_pspg_test10_signature_replay_readiness_finding
                    ),
                    "status": (
                        direct_pspg_test10_signature_replay_readiness.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "case_selector_findings": (
                        {
                            case.get("label"): values(
                                case.get("exact_local_signature_selector")
                            ).get("finding")
                            for case in direct_pspg_test10_signature_replay_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "case_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("exact_local_signature_selector")
                            ).get("selected_count")
                            for case in direct_pspg_test10_signature_replay_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "case_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("exact_local_signature_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_test10_signature_replay_readiness.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "test10_signature_candidate_global_dofs": (
                        case_by_label(
                            direct_pspg_test10_signature_replay_readiness,
                            "test10",
                        ).get("signature_candidate_global_dofs")
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "fe_topology_signature_or_row_selector_present": (
                        values(
                            direct_pspg_test10_signature_replay_readiness.get(
                                "hook_summary"
                            )
                        ).get("fe_topology_signature_or_row_selector_present")
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "post_assembly_explicit_row_path_present": (
                        values(
                            direct_pspg_test10_signature_replay_readiness.get(
                                "hook_summary"
                            )
                        ).get("post_assembly_explicit_row_path_present")
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_test10_signature_replay_readiness.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_replay_readiness,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_test10_signature_row_filter_replay_evidence
                | direct_pspg_test10_signature_row_filter_replay_summary
                | {
                    "policy": "local_schur_edge_balance",
                    "row_filter_global_dof_count": 48,
                    "row_filter_source": (
                        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS"
                    ),
                    "conclusion": (
                        "Targeted Test10 signature-row replay did not clear "
                        "the 100 Pa pressure-update guard."
                    ),
                },
                direct_pspg_test10_signature_row_filter_replays_evidence
                | {
                    "finding": direct_pspg_test10_signature_row_filter_replays_finding,
                    "status": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "policies_tested": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "policies_tested"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "row_filter_global_dof_counts": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "row_filter_global_dof_counts"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "best_policy_by_worst_update": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "best_policy_by_worst_update"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "best_worst_active_or_wet_update_pa": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "best_worst_active_or_wet_update_pa"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "policy_worst_active_or_wet_updates_pa": (
                        {
                            item.get("policy"): item.get(
                                "worst_active_or_wet_update_pa"
                            )
                            for item in direct_pspg_test10_signature_row_filter_replays.get(
                                "replays", []
                            )
                            if isinstance(item, dict)
                        }
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "policy_row_filter_log_counts": (
                        {
                            item.get("policy"): values(
                                item.get("topology_log")
                            ).get("row_filter_log_count")
                            for item in direct_pspg_test10_signature_row_filter_replays.get(
                                "replays", []
                            )
                            if isinstance(item, dict)
                        }
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_test10_signature_row_filter_replays.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_test10_signature_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_ghost_branch_signature_interaction_evidence
                | {
                    "finding": (
                        direct_pspg_ghost_branch_signature_interaction_finding
                    ),
                    "status": (
                        direct_pspg_ghost_branch_signature_interaction.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_ghost_branch_signature_interaction.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "row_10676_baseline_update_pa": (
                        case_by_label(
                            direct_pspg_ghost_branch_signature_interaction,
                            "test02",
                        ).get("row_10676_baseline_update_pa")
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "row_10676_pressure_disabled_update_pa": (
                        case_by_label(
                            direct_pspg_ghost_branch_signature_interaction,
                            "test02",
                        ).get("row_10676_pressure_disabled_update_pa")
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "signature_ratios": (
                        {
                            case.get("label"): values(
                                case.get("signature")
                            ).get(
                                "exact_local_signature_selected_to_target_ratio"
                            )
                            for case in direct_pspg_ghost_branch_signature_interaction.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "pressure_disabled_still_triggers": (
                        {
                            case.get("label"): values(
                                case.get("branch_policy")
                            ).get("pressure_disabled_still_triggers")
                            for case in direct_pspg_ghost_branch_signature_interaction.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_ghost_branch_signature_interaction.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_named_face_provenance_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_named_face_provenance_selectivity_finding
                    ),
                    "status": (
                        direct_pspg_named_face_provenance_selectivity.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_named_face_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_named_faces_by_case": (
                        {
                            case.get("label"): case.get("target_named_faces")
                            for case in direct_pspg_named_face_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_face_classes_by_case": (
                        {
                            case.get("label"): case.get("target_face_classes")
                            for case in direct_pspg_named_face_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "selected_counts_by_case_selector": (
                        {
                            case.get("label"): {
                                selector.get("key"): selector.get(
                                    "selected_count"
                                )
                                for selector in case.get("selectors", [])
                                if isinstance(selector, dict)
                            }
                            for case in direct_pspg_named_face_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "covered_target_counts_by_case_selector": (
                        {
                            case.get("label"): {
                                selector.get("key"): selector.get(
                                    "covered_target_count"
                                )
                                for selector in case.get("selectors", [])
                                if isinstance(selector, dict)
                            }
                            for case in direct_pspg_named_face_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "profile_status": (
                        {
                            case.get("label"): case.get(
                                "profile_evidence", {}
                            ).get("profile_status")
                            for case in direct_pspg_named_face_provenance_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_named_face_provenance_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_named_face_provenance_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
                graph_completion_stability_tradeoff_evidence
                | {
                    "finding": graph_completion_stability_tradeoff_finding,
                    "status": (
                        graph_completion_stability_tradeoff.get("status")
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "tradeoff_flags": (
                        graph_completion_stability_tradeoff.get(
                            "tradeoff_flags"
                        )
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "least_selector_tradeoff": (
                        graph_completion_stability_tradeoff.get(
                            "least_selector_tradeoff"
                        )
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "localized_balance_variant_findings": (
                        {
                            variant.get("key"): {
                                "test02_nonlinear_failed": variant.get(
                                    "test02_nonlinear_failed"
                                ),
                                "test10_guard_triggered": variant.get(
                                    "test10_guard_triggered"
                                ),
                                "test10_update_pa": variant.get(
                                    "test10", {}
                                ).get("accepted_pressure_update_pa"),
                            }
                            for variant in graph_completion_stability_tradeoff.get(
                                "localized_balance_variants", []
                            )
                            if isinstance(variant, dict)
                        }
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        graph_completion_stability_tradeoff.get(
                            "next_requirement"
                        )
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                },
                direct_pspg_explicit_balance_selector_replays_evidence
                | explicit_balance_selector_replay_summary(
                    direct_pspg_explicit_balance_selector_replays
                ),
            ],
            remaining_risk=(
                "Still needs a formulation-side topology/coupling rule; current "
                "post-assembly and pre-linear graph-completion mutations are "
                "diagnostic-only or too broad. The audited target predicate now "
                "has global pre-update emission coverage, but the raw global "
                "candidate selector is too broad to promote without an active "
                "PSPG pressure-gradient physical provenance gate. Simple cut-state "
                "provenance, literal mesh boundary support, named wall/obstacle "
                "face membership, and the post-update same-sign pressure-action "
                "oracle are ruled out as that gate. "
                "Constrained pressure-neighbor exposure is absent on the audited "
                "targets, and sparse unconstrained pressure-neighbor topology is "
                "overbroad or incomplete. Residual-sign pressure-action topology "
                "is also insufficient as a pre-update same-sign substitute. "
                "Direct pressure-gradient row-sum leakage and diagonal-balance "
                "selectors do not supply the missing gate by themselves. "
                "Matrix-only coupled-patch graph motifs are now ruled out too. "
                "Exact cut-volume row provenance shows every audited target is "
                "assembled through full-cell equivalent generated-volume "
                "support, but that class is too broad to promote and partial/"
                "low-fraction cut-volume selectors miss the targets. Local "
                "cut-volume matrix row-action strength and single-rule "
                "concentration likewise miss targets or stay overbroad. Local "
                "pressure/velocity coupling magnitude and velocity-to-pressure "
                "row-action ratio also fail as standalone gates. Row-parent-cell "
                "graph degree, clustering, and two-hop reach fail by themselves "
                "as well, and bounded graph/action/coupling composites either "
                "miss branch-specific targets or stay broad. The signed "
                "column-support replay is now ready and rules out support-sign "
                "class by itself: all profiled candidates and targets share the "
                "null-preserving negative-offdiagonal class. Signed sampled-"
                "column graph topology and edge-magnitude tails are also "
                "insufficient: candidate-neighbor closure, reciprocal negative "
                "edges, single-component support, degree/two-hop tails, and "
                "edge-concentration or mean-edge-magnitude tails all remain "
                "broad or miss targets. Sampled reference-node edge geometry "
                "does not isolate the targets either: complete/diagonal-edge "
                "selectors are broad, and edge-length, axis/diagonal fraction, "
                "row-origin, and finite-edge-count tails miss branch-specific "
                "targets. The next unresolved evidence gap is therefore a "
                "stronger formulation-side pressure-gradient support "
                "discriminator beyond raw cut/full volume provenance, local "
                "row-action metrics, per-row cross-field coupling ratios, "
                "parent-support graph metrics, thresholded composites, coarse "
                "signed-stencil class counts, sampled signed-column graph "
                "features, sampled reference-node edge geometry, or sampled "
                "cut-volume q-point geometry, or physical shape-gradient Gram "
                "balance, or sampled edge-level gradient-column graph "
                "topology. The new element-local Schur completion probe is "
                "constant-null preserving and diagnostic-only, but it touches "
                "all preferred direct PSPG candidates in both cases, so local "
                "Schur topology alone is now ruled out as the selective rule. "
                "The element-local existing-edge balance probe is also "
                "constant-null preserving and diagnostic-only, but its candidate "
                "and touched-row selectors cover all targets only by selecting "
                "nearly every preferred direct PSPG candidate. The paired replay "
                "controls now also rule out residual shape tangents, a direct "
                "PSPG cut-volume scale cap, and free-surface tangential pressure "
                "gradient terms as complete fixes; graph-completion selector "
                "coverage supplies exact threshold floors for the shifted rows "
                "but also shows the current weak-row selector is incomplete. "
                "The current public Forms vocabulary cannot express the "
                "remaining support topology directly, so a real formulation "
                "fix now needs an FE Forms or assembly API extension rather "
                "than another scalar form multiplier. The assembly API also "
                "has a disabled-by-default solve-affecting direct PSPG "
                "subterm/topology hook. The first fused-path replay pair "
                "exercised that hook, but broad local_schur_edge_balance still "
                "triggered both Test02/Test10 guards. The separated "
                "local_schur_completion and local_edge_balance replay modes "
                "also trigger both guards, so the remaining step is a "
                "formulation-side physical support/coupling rule or a "
                "Test02-focused guard/rejection mechanism rather than promoting "
                "this local graph mutation family. The replay-family stability "
                "tradeoff now also rules out threshold-selected post-assembly "
                "Schur fill or existing-edge balance: broad topology clears "
                "Test10 only with Test02 nonlinear failure, while localized "
                "balance gates still leave Test10 above guard and fail Test02. "
                "Explicit direct-row lists, shifted-row lists, operator top-row "
                "lists, cross-policy patch seeds, and one/two-ring pressure "
                "neighborhood balance selectors are ruled out as stable "
                "formulation selectors too. "
                "The retained-volume-fraction "
                "support cutoff branch is now classified as diagnostic-only: it "
                "may explain the latest tiny-cut Test02 rows, but it cannot "
                "address the full-wet Test02/Test10 branches by itself. The "
                "coupled-patch dependency barrier now makes the remaining "
                "blocker explicit: the next diagnostic must be solve-time "
                "pressure-gradient support/coupling provenance rather than "
                "post-update same-sign or pre-update proxy promotion. The "
                "source-level provenance hook is replayed now: scalar PP/PV "
                "coupling, aggregate count/volume fields, active-qpoint measure "
                "fields, and raw connected parent/rule co-support components "
                "are ruled out, while same-parent support/coupling signatures "
                "are only Test10-selective and Test02 remains overbroad. "
                "The same-rule cross-block sampled PP/PV signature plus "
                "magnitude range audit now exports a smaller Test02/Test10 "
                "candidate row family. The row-filter and derived parent-cell "
                "replays improve both short-window baselines but still trigger "
                "both guards, so this family is useful evidence for broader "
                "local support/coupling and not a sufficient fix. The "
                "broad-minus-same-rule complement replay also triggers both "
                "guards and is worse than the same-rule parent replay, while "
                "broad policy is better than both isolated parts; the helpful "
                "effect is therefore a broad union/synergy effect, not a "
                "sufficient subset rule. "
                "Solve-time magnitude features are not the missing "
                "gate either: range/threshold-like PP and PV magnitude selectors "
                "are overbroad, and only exact floating-value target oracles are "
                "selective. Signature-plus-magnitude range composites narrow "
                "Test10 but remain overbroad for Test02, so they are not the "
                "common support/coupling discriminator either. The selective "
                "Test10 exact-local signature row set is now exported and "
                "replayed through the solve-time FE topology row filter; all "
                "three targeted local modes still trigger the Test10 guard, so "
                "exact signature-row local topology mutation is ruled out as a "
                "sufficient fix. The parent/rule-scope audit also shows that "
                "the exact signature rows are only a strict subset of broad "
                "Test10 support and that broad-only rule keys carry most broad "
                "topology weight, so the helpful effect needs connected "
                "co-support but broad mutation is still insufficient. The "
                "parent-cell filter hook made the exact full-local test "
                "available on the 264 signature parent cells, without global "
                "row-filter attenuation. That replay now "
                "also triggers the Test10 guard, so exact parent-subset "
                "full-local mutation is ruled out as a sufficient fix. "
                "Pressure ghost-branch membership cannot supply the "
                "missing Test02 discriminator either: pressure-disabled "
                "provenance removes ghost rows while row 10676 persists and "
                "worsens, and Test10 has no ghost rows in the compared top-row "
                "branches. A common fix still needs a new physical support/"
                "coupling discriminator that handles both Test02's boundary-row "
                "amplification and Test10's coherent full-wet boundary mode "
                "before any formulation rule can be promoted."
            ),
        ),
        status_item(
            key="direct_pspg_solve_time_aggregate_features",
            question=(
                "Do solve-time direct PSPG aggregate support/coupling counts "
                "supply the missing physical discriminator?"
            ),
            status="aggregate_counts_and_volume_features_ruled_out",
            conclusion=(
                "Solve-time aggregate support/coupling counts, full/cut record "
                "classes, min volume fraction, PP edge/two-hop counts, PV "
                "nonzero counts, and rule counts cover audited direct targets "
                "only with broad row sets; they are not a common formulation gate."
            ),
            evidence=[
                direct_pspg_solve_time_aggregate_feature_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_aggregate_feature_selectivity_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_aggregate_feature_selectivity.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "features": (
                        direct_pspg_solve_time_aggregate_feature_selectivity.get(
                            "features"
                        )
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_exact_selector_keys": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_exact_value_selector")
                            ).get("key")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_range_selector_keys": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_range_selector")
                            ).get("key")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_exact_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_exact_value_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_range_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_range_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_exact_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_exact_value_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_range_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_range_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_aggregate_feature_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_aggregate_feature_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_aggregate_feature_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "This rules out the audited aggregate counts and classes only; "
                "a richer physical support/coupling topology rule is still the "
                "primary unresolved target."
            ),
        ),
        status_item(
            key="direct_pspg_solve_time_support_measure_features",
            question=(
                "Do solve-time active-quadrature and generated-measure fields "
                "supply the missing physical support discriminator?"
            ),
            status="active_qpoint_and_measure_features_ruled_out",
            conclusion=(
                "Solve-time active quadrature counts/fractions, generated "
                "measure classes, measure fractions, parent measures, and rule "
                "quadrature counts cover audited direct targets only with broad "
                "row sets; they are not a common formulation gate."
            ),
            evidence=[
                direct_pspg_solve_time_support_measure_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_support_measure_selectivity_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_support_measure_selectivity.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "features": (
                        direct_pspg_solve_time_support_measure_selectivity.get(
                            "features"
                        )
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_exact_selector_keys": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_exact_value_selector")
                            ).get("key")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_range_selector_keys": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_range_selector")
                            ).get("key")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_exact_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_exact_value_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_range_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_range_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_exact_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_exact_value_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_range_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_range_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_support_measure_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_support_measure_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_support_measure_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "This rules out the audited solve-time support-measure fields "
                "only; the unresolved path still needs richer physical direct "
                "PSPG support/coupling topology."
            ),
        ),
        status_item(
            key="direct_pspg_solve_time_parent_rule_components",
            question=(
                "Does raw connected co-support over solve-time parent cells and "
                "rule indices supply the missing direct PSPG topology gate?"
            ),
            status="parent_rule_component_closure_ruled_out",
            conclusion=(
                "Solve-time parent-cell/rule-index co-support components "
                "collapse the audited direct PSPG rows into broad connected row "
                "sets; raw connected support-patch closure is not a common "
                "formulation gate."
            ),
            evidence=[
                direct_pspg_solve_time_parent_rule_component_selectivity_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_parent_rule_component_selectivity_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_parent_rule_component_selectivity.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "graph_modes": (
                        direct_pspg_solve_time_parent_rule_component_selectivity.get(
                            "graph_modes"
                        )
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_parent_rule_component_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "component_counts": (
                        {
                            case.get("label"): case.get("component_counts")
                            for case in direct_pspg_solve_time_parent_rule_component_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "target_component_sizes": (
                        {
                            case.get("label"): case.get("target_component_sizes")
                            for case in direct_pspg_solve_time_parent_rule_component_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_component_selector_keys": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_component_selector")
                            ).get("key")
                            for case in direct_pspg_solve_time_parent_rule_component_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_component_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_component_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_parent_rule_component_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "best_component_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_component_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_parent_rule_component_selectivity.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_parent_rule_component_selectivity.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_parent_rule_component_selectivity,
                            dict,
                        )
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "This rules out raw connected parent/rule support closure only; "
                "the unresolved path still needs a physical support/coupling "
                "rule beyond connected-component membership."
            ),
        ),
        status_item(
            key="direct_pspg_solve_time_same_rule_cross_block_signature",
            question=(
                "Does same-rule sampled PP/PV local support topology identify a "
                "smaller direct PSPG row family for targeted replay?"
            ),
            status="same_rule_cross_block_broad_union_consistent_replayed_insufficient",
            conclusion=(
                "Same-rule pressure-pressure/pressure-velocity local signature "
                "pairs plus non-update pressure-velocity magnitude ranges cover "
                "all audited Test02/Test10 direct PSPG targets with small row "
                "sets. The targeted local_schur_edge_balance row-filter replay "
                "and the derived parent-cell replay both improve the no-policy "
                "baseline in both windows, but both still trigger pressure-update "
                "guards. The broad-minus-same-rule parent-cell complement also "
                "triggers both guards and is worse than the same-rule parent-cell "
                "replay, while broad policy is better than both isolated parts. "
                "The transition-consistent broad-union point audit shows broad "
                "policy only slightly improves the Test02 full-wet reference "
                "branch relative to the same-rule parent-cell replay and still "
                "leaves both Test02 and Test10 full-wet guard triggers. Neither "
                "exact row lists, parent-scoped full-local mutation, the broad-"
                "only complement, nor the broad union alone are sufficient "
                "formulation fixes."
            ),
            evidence=[
                direct_pspg_solve_time_same_rule_cross_block_signature_evidence
                | {
                    "finding": (
                        direct_pspg_solve_time_same_rule_cross_block_signature_finding
                    ),
                    "status": (
                        direct_pspg_solve_time_same_rule_cross_block_signature.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "features": (
                        direct_pspg_solve_time_same_rule_cross_block_signature.get(
                            "features"
                        )
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "shape_pair_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("shape_pair_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "base_signature_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("base_same_rule_signature_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "base_signature_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("base_same_rule_signature_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "best_composite_selector_keys": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_composite_selector")
                            ).get("key")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "best_composite_features": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_composite_selector")
                            ).get("feature")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "best_composite_selected_counts": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_composite_selector")
                            ).get("selected_count")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "best_composite_selected_to_target_ratios": (
                        {
                            case.get("label"): values(
                                case.get("best_covering_composite_selector")
                            ).get("selected_to_target_ratio")
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "best_composite_selected_global_dofs": (
                        {
                            case.get("label"): case.get(
                                "best_covering_composite_selected_global_dofs"
                            )
                            for case in direct_pspg_solve_time_same_rule_cross_block_signature.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_solve_time_same_rule_cross_block_signature.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_solve_time_same_rule_cross_block_signature,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_same_rule_cross_block_row_filter_replays_evidence
                | {
                    "finding": (
                        direct_pspg_same_rule_cross_block_row_filter_replays_finding
                    ),
                    "status": (
                        direct_pspg_same_rule_cross_block_row_filter_replays.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "row_filters_match_candidate_counts": (
                        direct_pspg_same_rule_cross_block_row_filter_replays.get(
                            "row_filters_match_candidate_counts"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_improve_no_policy_baseline": (
                        direct_pspg_same_rule_cross_block_row_filter_replays.get(
                            "all_replays_improve_no_policy_baseline"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_same_rule_cross_block_row_filter_replays.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "triggered_cases": (
                        direct_pspg_same_rule_cross_block_row_filter_replays.get(
                            "triggered_cases"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "candidate_row_counts": (
                        {
                            case.get("label"): case.get(
                                "expected_candidate_row_count"
                            )
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "worst_active_or_wet_update_pa": (
                        {
                            case.get("label"): values(
                                case.get("pressure_update")
                            ).get("worst_active_or_wet_update_pa")
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "worst_active_or_wet_support_class": (
                        {
                            case.get("label"): values(
                                case.get("pressure_update")
                            ).get("worst_active_or_wet_support_class")
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "improvement_vs_baseline_pa": (
                        {
                            case.get("label"): case.get(
                                "improvement_vs_baseline_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "replay_to_baseline_update_ratio": (
                        {
                            case.get("label"): case.get(
                                "replay_to_baseline_update_ratio"
                            )
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "replay_to_broad_policy_update_ratio": (
                        {
                            case.get("label"): case.get(
                                "replay_to_broad_policy_update_ratio"
                            )
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "matrix_mutated_counts": (
                        {
                            case.get("label"): values(
                                case.get("topology_log")
                            ).get("matrix_mutated_count")
                            for case in direct_pspg_same_rule_cross_block_row_filter_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_same_rule_cross_block_row_filter_replays.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_row_filter_replays,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_same_rule_cross_block_parent_cell_scope_evidence
                | {
                    "finding": (
                        direct_pspg_same_rule_cross_block_parent_cell_scope_finding
                    ),
                    "status": (
                        direct_pspg_same_rule_cross_block_parent_cell_scope.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "all_cases_ready_for_parent_cell_replay": (
                        direct_pspg_same_rule_cross_block_parent_cell_scope.get(
                            "all_cases_ready_for_parent_cell_replay"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "parent_cell_counts": (
                        {
                            case.get("label"): case.get("parent_cell_count")
                            for case in direct_pspg_same_rule_cross_block_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "parent_expanded_row_counts": (
                        {
                            case.get("label"): case.get(
                                "parent_expanded_row_count"
                            )
                            for case in direct_pspg_same_rule_cross_block_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "parent_expanded_to_candidate_ratios": (
                        {
                            case.get("label"): case.get(
                                "parent_expanded_to_candidate_ratio"
                            )
                            for case in direct_pspg_same_rule_cross_block_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_same_rule_cross_block_parent_cell_scope.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_same_rule_cross_block_parent_cell_replays_evidence
                | {
                    "finding": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays_finding
                    ),
                    "status": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "parent_filters_match_scope_counts": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "parent_filters_match_scope_counts"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "row_filters_disabled": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "row_filters_disabled"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_improve_no_policy_baseline": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "all_replays_improve_no_policy_baseline"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_improve_row_filter_replay": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "all_replays_improve_row_filter_replay"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "triggered_cases": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "triggered_cases"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "parent_cell_counts": (
                        {
                            case.get("label"): values(
                                case.get("parent_scope")
                            ).get("expected_parent_cell_count")
                            for case in direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "worst_active_or_wet_update_pa": (
                        {
                            case.get("label"): values(
                                case.get("pressure_update")
                            ).get("worst_active_or_wet_update_pa")
                            for case in direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "improvement_vs_baseline_pa": (
                        {
                            case.get("label"): case.get(
                                "improvement_vs_baseline_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "improvement_vs_row_filter_pa": (
                        {
                            case.get("label"): case.get(
                                "improvement_vs_row_filter_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "replay_to_broad_policy_update_ratio": (
                        {
                            case.get("label"): case.get(
                                "replay_to_broad_policy_update_ratio"
                            )
                            for case in direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "matrix_mutated_counts": (
                        {
                            case.get("label"): values(
                                case.get("topology_log")
                            ).get("matrix_mutated_count")
                            for case in direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_same_rule_cross_block_parent_cell_replays.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope_evidence
                | {
                    "finding": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope_finding
                    ),
                    "status": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "all_cases_ready_for_broad_minus_parent_cell_replay": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                            "all_cases_ready_for_broad_minus_parent_cell_replay"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "broad_parent_cell_counts": (
                        {
                            case.get("label"): case.get("broad_parent_cell_count")
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "same_rule_parent_cell_counts": (
                        {
                            case.get("label"): case.get(
                                "same_rule_parent_cell_count"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "broad_only_parent_cell_counts": (
                        {
                            case.get("label"): case.get(
                                "broad_only_parent_cell_count"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "broad_only_to_broad_parent_ratios": (
                        {
                            case.get("label"): case.get(
                                "broad_only_to_broad_parent_ratio"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_scope,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays_evidence
                | {
                    "finding": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays_finding
                    ),
                    "status": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "parent_filters_match_scope_counts": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "parent_filters_match_scope_counts"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "row_filters_disabled": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "row_filters_disabled"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "all_replays_trigger_guard": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "all_replays_trigger_guard"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "broad_policy_better_than_isolated_parts": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "broad_policy_better_than_isolated_parts"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "complement_worse_than_same_rule_parent_cell": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "complement_worse_than_same_rule_parent_cell"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "triggered_cases": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "triggered_cases"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "broad_only_parent_cell_counts": (
                        {
                            case.get("label"): values(
                                case.get("broad_minus_scope")
                            ).get("expected_parent_cell_count")
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "worst_active_or_wet_update_pa": (
                        {
                            case.get("label"): values(
                                case.get("pressure_update")
                            ).get("worst_active_or_wet_update_pa")
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "improvement_vs_baseline_pa": (
                        {
                            case.get("label"): case.get(
                                "improvement_vs_baseline_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "improvement_vs_same_rule_parent_cell_pa": (
                        {
                            case.get("label"): case.get(
                                "improvement_vs_same_rule_parent_cell_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "replay_to_broad_policy_update_ratio": (
                        {
                            case.get("label"): case.get(
                                "replay_to_broad_policy_update_ratio"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "matrix_mutated_counts": (
                        {
                            case.get("label"): values(
                                case.get("topology_log")
                            ).get("matrix_mutated_count")
                            for case in direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_minus_parent_cell_replays,
                            dict,
                        )
                        else None
                    ),
                },
                direct_pspg_same_rule_cross_block_broad_union_branch_shift_evidence
                | {
                    "finding": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift_finding
                    ),
                    "status": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "case_findings"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "all_variants_guard_triggered": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "all_variants_guard_triggered"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "test02_branch_shift_supported": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "test02_branch_shift_supported"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "test02_consistent_full_wet_residual_supported": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "test02_consistent_full_wet_residual_supported"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "test10_broad_union_residual_guard_supported": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "test10_broad_union_residual_guard_supported"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "reference_points": (
                        {
                            case.get("label"): case.get("reference_point")
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "broad_reference_abs_pressure_delta_pa": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "broad_reference_abs_pressure_delta_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "isolated_reference_abs_pressure_delta_pa": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "isolated_reference_abs_pressure_delta_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "broad_reference_improvement_vs_isolated_pa": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "broad_reference_improvement_vs_isolated_pa"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "broad_policy_worst_points": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "broad_policy_worst_point"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "broad_policy_worst_support_classes": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "broad_policy_worst_support_class"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "broad_policy_clears_reference_point_guard": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "broad_policy_clears_reference_point_guard"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "broad_policy_guard_triggered": (
                        {
                            case.get("label"): values(case.get("flags")).get(
                                "broad_policy_guard_triggered"
                            )
                            for case in direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_same_rule_cross_block_broad_union_branch_shift.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_same_rule_cross_block_broad_union_branch_shift,
                            dict,
                        )
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "The candidate still depends on audited target magnitude ranges "
                "and its targeted row-list, parent-cell, and broad-minus "
                "complement topology replays do not clear Test02/Test10; the "
                "transition-consistent broad union helps the full-wet branch "
                "only modestly and still fails both full-wet guards. "
                "The unresolved path needs a broader physics-derived support/"
                "coupling predicate rather than exact subsets, broad-only "
                "complements, or the broad policy by itself."
            ),
        ),
        status_item(
            key="pressure_ghost_penalty_direct_driver",
            question=(
                "Does pressure ghost penalty directly drive the sampled Test02/Test10 "
                "accepted pressure jumps?"
            ),
            status="ruled_out_as_direct_test10_or_sampled_max_row_driver",
            conclusion=(
                "Pressure ghost penalty is absent from Test10 exact top rows and "
                "is not the sampled max-row self-stencil. It remains a Test02 "
                "branch shaper because Test02's exact top rows are mixed."
            ),
            evidence=[
                top_provenance_evidence
                | {
                    "test02_finding": test02_top.get("finding"),
                    "test10_finding": test10_top.get("finding"),
                    "finding_counts": (
                        top_provenance.get("finding_counts")
                        if isinstance(top_provenance, dict)
                        else None
                    ),
                },
                root_evidence
                | {
                    "mentions_pressure_disabled_control": (
                        "pressure-disabled" in report_text
                        and "ghost" in report_text
                    )
                },
                direct_pspg_ghost_branch_signature_interaction_evidence
                | {
                    "finding": (
                        direct_pspg_ghost_branch_signature_interaction_finding
                    ),
                    "status": (
                        direct_pspg_ghost_branch_signature_interaction.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in direct_pspg_ghost_branch_signature_interaction.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "row_10676_baseline_update_pa": (
                        case_by_label(
                            direct_pspg_ghost_branch_signature_interaction,
                            "test02",
                        ).get("row_10676_baseline_update_pa")
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                    "row_10676_pressure_disabled_update_pa": (
                        case_by_label(
                            direct_pspg_ghost_branch_signature_interaction,
                            "test02",
                        ).get("row_10676_pressure_disabled_update_pa")
                        if isinstance(
                            direct_pspg_ghost_branch_signature_interaction,
                            dict,
                        )
                        else None
                    ),
                },
                pressure_stabilization_driver_windows_evidence
                | {
                    "finding": pressure_stabilization_driver_windows_finding,
                    "status": (
                        pressure_stabilization_driver_windows.get("status")
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                    "all_saved_window_worst_updates_nonincident": (
                        pressure_stabilization_driver_windows.get(
                            "all_saved_window_worst_updates_nonincident"
                        )
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                    "any_saved_window_worst_update_incident": (
                        pressure_stabilization_driver_windows.get(
                            "any_saved_window_worst_update_incident"
                        )
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                    "case_findings": (
                        {
                            case.get("label"): case.get("finding")
                            for case in pressure_stabilization_driver_windows.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                    "case_incident_cut_adjacent_face_counts": (
                        {
                            case.get("label"): case.get(
                                "incident_cut_adjacent_face_count"
                            )
                            for case in pressure_stabilization_driver_windows.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                    "case_worst_updates_pa": (
                        {
                            case.get("label"): case.get(
                                "worst_update_abs_pressure_delta_pa"
                            )
                            for case in pressure_stabilization_driver_windows.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                    "next_requirement": (
                        pressure_stabilization_driver_windows.get(
                            "next_requirement"
                        )
                        if isinstance(pressure_stabilization_driver_windows, dict)
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "Ghost-penalty branch interaction is still in scope for Test02, "
                "but the joined branch/signature evidence rules out using "
                "ghost-positive membership as the missing common support/"
                "coupling discriminator. The saved-window cut-adjacent "
                "stabilization proxy also rules out direct ghost-penalty "
                "incidence at the worst active/wet Test02/Test10 updates, but "
                "it is not an exact assembled residual dump and does not exclude "
                "branch shaping."
            ),
        ),
        status_item(
            key="hydrostatic_linear_cut_volume_patch_consistency",
            question=(
                "Does a hydrostatic or linear-pressure cut-volume patch expose a "
                "residual/update inconsistency?"
            ),
            status="ruled_out_narrow_inconsistency_supported_amplification_proxy",
            conclusion=(
                "The retained patch preserves linear pressure, matched hydrostatic "
                "cancellation, and the constant-pressure null, but still exposes "
                "boundary-row solve amplification and trace-only support hazards."
            ),
            evidence=[
                linear_patch_evidence
                | {
                    "passed": (
                        linear_patch.get("passed")
                        if isinstance(linear_patch, dict)
                        else None
                    ),
                    "hazard_detected": (
                        linear_patch.get("hazard_detected")
                        if isinstance(linear_patch, dict)
                        else None
                    ),
                    "pspg_hydrostatic_hazard_detected": (
                        linear_patch.get("pspg_hydrostatic_hazard_detected")
                        if isinstance(linear_patch, dict)
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "Patch evidence is a proxy; the production rule must transfer to "
                "moving 3D replay rows without breaking Test02."
            ),
        ),
        status_item(
            key="inactive_pressure_constraint_omission",
            question=(
                "Are accepted pressure jumps caused by missing inactive pressure "
                "support constraints?"
            ),
            status="ruled_out_as_constraint_coverage_omission",
            conclusion=(
                "The VMS-disabled Test10 zero rows map to active supported pressure "
                "DOFs outside inactive constraint runs, so the issue is active "
                "pressure support/rank rather than an inactive identity omission."
            ),
            evidence=[
                constraint_coverage_evidence
                | {
                    "constraint_vertex_dof_mapping_status": (
                        constraint_coverage.get("constraint_vertex_dof_mapping_status")
                        if isinstance(constraint_coverage, dict)
                        else None
                    ),
                    "reported_zero_row_count": (
                        constraint_coverage.get("reported_zero_row_count")
                        if isinstance(constraint_coverage, dict)
                        else None
                    ),
                },
                support_rank_guard_evidence
                | {
                    "zero_coupling_row_block_count": support_rank_values.get(
                        "zero_coupling_row_block_count"
                    ),
                    "pressure_only_row_block_count": support_rank_values.get(
                        "pressure_only_row_block_count"
                    ),
                },
            ],
            remaining_risk=(
                "A production active-support criterion is still needed for zero "
                "and weak-but-nonzero pressure rows."
            ),
        ),
        status_item(
            key="natural_free_surface_pressure_anchor",
            question=(
                "Are the jumps caused by missing natural free-surface pressure "
                "anchor/support interaction?"
            ),
            status="ruled_out_for_sampled_max_rows",
            conclusion=(
                "The direct free-surface pressure-reference and generated-interface "
                "tangential pressure-gradient probes do not support the sampled "
                "bad rows or worsen/leave the guarded branches."
            ),
            evidence=[
                free_surface_reference_evidence,
                free_surface_tangential_evidence,
                test02_free_surface_tangential_pressure_update_evidence
                | {
                    "case": "test02",
                    **pressure_update_case_summary(
                        test02_free_surface_tangential_pressure_update
                    ),
                },
                test10_free_surface_tangential_pressure_update_evidence
                | {
                    "case": "test10",
                    **pressure_update_case_summary(
                        test10_free_surface_tangential_pressure_update
                    ),
                },
                root_evidence
                | {
                    "mentions_free_surface_reference_exclusion": (
                        "missing direct generated-interface pressure trace reference"
                        in report_text
                    ),
                    "mentions_free_surface_tangential_exclusion": (
                        "direct generated-interface tangential pressure-gradient support"
                        in report_text
                    ),
                },
            ],
            remaining_risk=(
                "This excludes the sampled replay rows, not every possible "
                "free-surface pressure treatment issue."
            ),
        ),
        status_item(
            key="geometry_refresh_frozen_quadrature_or_shape_tangent",
            question=(
                "Are accepted jumps caused by geometry refresh, frozen quadrature, "
                "or missing residual-level shape tangents?"
            ),
            status="ruled_out_as_immediate_source_or_complete_fix",
            conclusion=(
                "Cut-context transition and shape-tangent controls do not move the "
                "known pressure-update branches enough to explain or fix them."
            ),
            evidence=[
                test02_shape_tangent_evidence
                | {
                    "case": "test02",
                    **pressure_update_case_summary(test02_shape_tangent),
                },
                shape_tangent_evidence
                | {
                    "case": "test10",
                    "status": (
                        shape_tangent.get("status")
                        if isinstance(shape_tangent, dict)
                        else None
                    ),
                    **pressure_update_case_summary(shape_tangent),
                },
                root_evidence
                | {
                    "mentions_cut_context_transition": (
                        "cut-context" in report_text
                        or "post-acceptance maintenance refresh" in report_text
                    ),
                },
            ],
            remaining_risk=(
                "The remaining geometry question is formulation consistency on "
                "the solve-time active-volume pressure operator."
            ),
        ),
        status_item(
            key="timestep_acceptance_logic",
            question=(
                "Are the pressure jumps caused by a timestep acceptance gap or too-large dt?"
            ),
            status="guard_supported_dt_reduction_ruled_out_as_fix",
            conclusion=(
                "The pre-commit guard prevents silent acceptance, but adaptive "
                "retry evidence shows smaller dt can increase the same Test10 "
                "row and can make Test02 shift from tiny-cut-supported rows "
                "back to the full-wet branch while growing by orders of "
                "magnitude."
            ),
            evidence=[
                pressure_update_rejection_replay_evidence
                | {
                    "finding": pressure_update_rejection_replay_finding,
                    "status": (
                        pressure_update_rejection_replay.get("status")
                        if isinstance(pressure_update_rejection_replay, dict)
                        else None
                    ),
                    "guard": (
                        pressure_update_rejection_replay.get("guard")
                        if isinstance(pressure_update_rejection_replay, dict)
                        else None
                    ),
                    "fixed_step_replays": (
                        [
                            {
                                "case": replay.get("case"),
                                "threshold_pa": replay.get("threshold_pa"),
                                "step_accepted": replay.get("step_accepted"),
                                "step_rejected_count": replay.get(
                                    "step_rejected_count"
                                ),
                                "worst_pre_commit_update_pa": replay.get(
                                    "worst_pre_commit_update_pa"
                                ),
                                "worst_pre_commit_dof": replay.get(
                                    "worst_pre_commit_dof"
                                ),
                                "worst_pre_commit_support_class": replay.get(
                                    "worst_pre_commit_support_class"
                                ),
                            }
                            for replay in pressure_update_rejection_replay.get(
                                "fixed_step_replays", []
                            )
                            if isinstance(replay, dict)
                        ]
                        if isinstance(pressure_update_rejection_replay, dict)
                        else None
                    ),
                    "adaptive_replays": (
                        [
                            {
                                "case": replay.get("case"),
                                "threshold_pa": replay.get("threshold_pa"),
                                "step_accepted": replay.get("step_accepted"),
                                "step_rejected_count": replay.get(
                                    "step_rejected_count"
                                ),
                                "first_update_pa": replay.get("first_update_pa"),
                                "last_update_pa": replay.get("last_update_pa"),
                                "update_growth_factor": replay.get(
                                    "update_growth_factor"
                                ),
                                "support_branch_shift": replay.get(
                                    "support_branch_shift"
                                ),
                            }
                            for replay in pressure_update_rejection_replay.get(
                                "adaptive_replays", []
                            )
                            if isinstance(replay, dict)
                        ]
                        if isinstance(pressure_update_rejection_replay, dict)
                        else None
                    ),
                    "next_requirement": (
                        pressure_update_rejection_replay.get("next_requirement")
                        if isinstance(pressure_update_rejection_replay, dict)
                        else None
                    ),
                },
                pressure_update_residual_context_evidence
                | {
                    "finding": pressure_update_residual_context_finding,
                    "status": (
                        pressure_update_residual_context.get("status")
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "large_ratio_threshold": (
                        pressure_update_residual_context.get(
                            "large_ratio_threshold"
                        )
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "all_cases_accepted_converged_large_update_residual_gap": (
                        pressure_update_residual_context.get(
                            "all_cases_accepted_converged_large_update_residual_gap"
                        )
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "all_cases_post_acceptance_refresh_ruled_out": (
                        pressure_update_residual_context.get(
                            "all_cases_post_acceptance_refresh_ruled_out"
                        )
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "case_update_to_nonlinear_field_residual_ratios": (
                        {
                            case.get("label"): case.get(
                                "update_to_nonlinear_field_residual_norm_ratio"
                            )
                            for case in pressure_update_residual_context.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "case_pressure_updates_pa": (
                        {
                            case.get("label"): case.get(
                                "global_abs_pressure_delta_pa"
                            )
                            for case in pressure_update_residual_context.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "case_nonlinear_field_residual_norms": (
                        {
                            case.get("label"): case.get(
                                "nonlinear_field_residual_norm"
                            )
                            for case in pressure_update_residual_context.get(
                                "cases", []
                            )
                            if isinstance(case, dict)
                        }
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                    "next_requirement": (
                        pressure_update_residual_context.get("next_requirement")
                        if isinstance(pressure_update_residual_context, dict)
                        else None
                    ),
                },
                root_evidence
                | {
                    "mentions_pre_commit_rejection": (
                        "pre-commit pressure-update rejection" in report_text
                    ),
                    "mentions_dt_halving_worsens": (
                        "as `dt` halves" in report_text
                        or "as dt halves" in report_text
                        or "dt is halved" in report_text
                    ),
                },
            ],
            remaining_risk=(
                "Keep as a diagnostic safety gate, not a physics or formulation "
                "fix; the pressure-path support inconsistency remains. The "
                "residual-context audit shows the accepted updates can be many "
                "orders larger than the converged nonlinear residual, so "
                "residual convergence alone is not a safe acceptance criterion "
                "for these pressure modes."
            ),
        ),
        status_item(
            key="post_assembly_graph_completion_family",
            question=(
                "Is the smallest credible fix a thresholded post-assembly graph "
                "completion or edge-balance mutation?"
            ),
            status="ruled_out_as_production_fix_supported_as_diagnostic",
            conclusion=(
                "Graph-completion variants prove topology and edge balance are "
                "causal, but selector, support-gap patch, row-list, degree, "
                "coupling, neighborhood, and all-row variants either leave a "
                "guard-triggering branch or destabilize Test02."
            ),
            evidence=[
                coupling_outcome_evidence
                | {
                    "test10_outcome": (
                        coupling_outcome.get("test10_step90", {}).get("outcome")
                        if isinstance(coupling_outcome, dict)
                        else None
                    ),
                    "test02_outcome": (
                        coupling_outcome.get("test02_step382", {}).get("outcome")
                        if isinstance(coupling_outcome, dict)
                        else None
                    ),
                },
                low_degree_outcome_evidence
                | {
                    "test10_outcome": (
                        low_degree_outcome.get("test10_step90", {}).get("outcome")
                        if isinstance(low_degree_outcome, dict)
                        else None
                    ),
                    "test02_outcome": (
                        low_degree_outcome.get("test02_step382", {}).get("outcome")
                        if isinstance(low_degree_outcome, dict)
                        else None
                    ),
                },
                support_gap_patch_outcome_evidence
                | {
                    "finding": (
                        support_gap_patch_outcome.get("finding")
                        if isinstance(support_gap_patch_outcome, dict)
                        else None
                    ),
                    "test10_outcome": (
                        support_gap_patch_outcome.get("test10_step90", {}).get(
                            "outcome"
                        )
                        if isinstance(support_gap_patch_outcome, dict)
                        else None
                    ),
                    "test10_accepted_pressure_update_pa": (
                        support_gap_patch_outcome.get("test10_step90", {}).get(
                            "accepted_pressure_update_pa"
                        )
                        if isinstance(support_gap_patch_outcome, dict)
                        else None
                    ),
                    "test02_outcome": (
                        support_gap_patch_outcome.get("test02_step382", {}).get(
                            "outcome"
                        )
                        if isinstance(support_gap_patch_outcome, dict)
                        else None
                    ),
                    "test02_final_residual_norm": (
                        support_gap_patch_outcome.get("test02_step382", {}).get(
                            "final_residual_norm"
                        )
                        if isinstance(support_gap_patch_outcome, dict)
                        else None
                    ),
                },
                support_gap_patch_schur_only_outcome_evidence
                | {
                    "finding": (
                        support_gap_patch_schur_only_outcome.get("finding")
                        if isinstance(support_gap_patch_schur_only_outcome, dict)
                        else None
                    ),
                    "test10_outcome": (
                        support_gap_patch_schur_only_outcome.get(
                            "test10_step90", {}
                        ).get("outcome")
                        if isinstance(support_gap_patch_schur_only_outcome, dict)
                        else None
                    ),
                    "test10_accepted_pressure_update_pa": (
                        support_gap_patch_schur_only_outcome.get(
                            "test10_step90", {}
                        ).get("accepted_pressure_update_pa")
                        if isinstance(support_gap_patch_schur_only_outcome, dict)
                        else None
                    ),
                    "test02_outcome": (
                        support_gap_patch_schur_only_outcome.get(
                            "test02_step382", {}
                        ).get("outcome")
                        if isinstance(support_gap_patch_schur_only_outcome, dict)
                        else None
                    ),
                    "test02_final_residual_norm": (
                        support_gap_patch_schur_only_outcome.get(
                            "test02_step382", {}
                        ).get("final_residual_norm")
                        if isinstance(support_gap_patch_schur_only_outcome, dict)
                        else None
                    ),
                },
                support_gap_local_patch_schur_only_outcome_evidence
                | {
                    "finding": (
                        support_gap_local_patch_schur_only_outcome.get(
                            "finding"
                        )
                        if isinstance(
                            support_gap_local_patch_schur_only_outcome, dict
                        )
                        else None
                    ),
                    "pressure_neighbor_depth": (
                        support_gap_local_patch_schur_only_outcome.get(
                            "pressure_neighbor_depth"
                        )
                        if isinstance(
                            support_gap_local_patch_schur_only_outcome, dict
                        )
                        else None
                    ),
                    "test10_outcome": (
                        support_gap_local_patch_schur_only_outcome.get(
                            "test10_step90", {}
                        ).get("outcome")
                        if isinstance(
                            support_gap_local_patch_schur_only_outcome, dict
                        )
                        else None
                    ),
                    "test10_accepted_pressure_update_pa": (
                        support_gap_local_patch_schur_only_outcome.get(
                            "test10_step90", {}
                        ).get("accepted_pressure_update_pa")
                        if isinstance(
                            support_gap_local_patch_schur_only_outcome, dict
                        )
                        else None
                    ),
                    "test02_outcome": (
                        support_gap_local_patch_schur_only_outcome.get(
                            "test02_step382", {}
                        ).get("outcome")
                        if isinstance(
                            support_gap_local_patch_schur_only_outcome, dict
                        )
                        else None
                    ),
                    "test02_final_residual_norm": (
                        support_gap_local_patch_schur_only_outcome.get(
                            "test02_step382", {}
                        ).get("final_residual_norm")
                        if isinstance(
                            support_gap_local_patch_schur_only_outcome, dict
                        )
                        else None
                    ),
                },
                boundary_provenance_evidence
                | {
                    "finding": (
                        boundary_provenance.get("finding")
                        if isinstance(boundary_provenance, dict)
                        else None
                    ),
                    "boundary_topology_finding": (
                        boundary_provenance.get("boundary_topology_finding")
                        if isinstance(boundary_provenance, dict)
                        else None
                    ),
                },
                graph_completion_candidate_readiness_evidence
                | {
                    "finding": graph_completion_candidate_readiness_finding,
                    "overbroad_modes": (
                        graph_completion_candidate_readiness.get("overbroad_modes")
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                    "test02_unstable_modes": (
                        graph_completion_candidate_readiness.get(
                            "test02_unstable_modes"
                        )
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                    "test10_guard_clear_modes": (
                        graph_completion_candidate_readiness.get(
                            "test10_guard_clear_modes"
                        )
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                    "direct_target_counts": (
                        graph_completion_candidate_readiness.get(
                            "direct_target_counts"
                        )
                        if isinstance(graph_completion_candidate_readiness, dict)
                        else None
                    ),
                },
                graph_completion_replay_family_evidence
                | {
                    "finding": graph_completion_replay_family_finding,
                    "variant_findings": (
                        graph_completion_replay_family.get("variant_findings")
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "test10_guard_clear_variants": (
                        graph_completion_replay_family.get(
                            "test10_guard_clear_variants"
                        )
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "test02_unstable_variants": (
                        graph_completion_replay_family.get(
                            "test02_unstable_variants"
                        )
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "test10_still_trigger_variants": (
                        graph_completion_replay_family.get(
                            "test10_still_trigger_variants"
                        )
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "case_findings_by_variant": (
                        {
                            variant.get("key"): {
                                case.get("label"): case.get("finding")
                                for case in variant.get("cases", [])
                            }
                            for variant in graph_completion_replay_family.get(
                                "variants", []
                            )
                        }
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                    "next_requirement": (
                        graph_completion_replay_family.get("next_requirement")
                        if isinstance(graph_completion_replay_family, dict)
                        else None
                    ),
                },
                graph_completion_stability_tradeoff_evidence
                | {
                    "finding": graph_completion_stability_tradeoff_finding,
                    "status": (
                        graph_completion_stability_tradeoff.get("status")
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "tradeoff_flags": (
                        graph_completion_stability_tradeoff.get(
                            "tradeoff_flags"
                        )
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "least_selector_tradeoff": (
                        graph_completion_stability_tradeoff.get(
                            "least_selector_tradeoff"
                        )
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        graph_completion_stability_tradeoff.get(
                            "next_requirement"
                        )
                        if isinstance(
                            graph_completion_stability_tradeoff, dict
                        )
                        else None
                    ),
                },
                direct_pspg_explicit_balance_selector_replays_evidence
                | explicit_balance_selector_replay_summary(
                    direct_pspg_explicit_balance_selector_replays
                ),
                direct_pspg_active_support_completion_replays_evidence
                | {
                    "finding": (
                        direct_pspg_active_support_completion_replays_finding
                    ),
                    "status": (
                        direct_pspg_active_support_completion_replays.get(
                            "status"
                        )
                        if isinstance(
                            direct_pspg_active_support_completion_replays, dict
                        )
                        else None
                    ),
                    "all_replays_guard_triggered": (
                        direct_pspg_active_support_completion_replays.get(
                            "all_replays_guard_triggered"
                        )
                        if isinstance(
                            direct_pspg_active_support_completion_replays, dict
                        )
                        else None
                    ),
                    "all_replays_accepted_one_step": (
                        direct_pspg_active_support_completion_replays.get(
                            "all_replays_accepted_one_step"
                        )
                        if isinstance(
                            direct_pspg_active_support_completion_replays, dict
                        )
                        else None
                    ),
                    "case_updates_pa": (
                        direct_pspg_active_support_completion_replays.get(
                            "case_updates_pa"
                        )
                        if isinstance(
                            direct_pspg_active_support_completion_replays, dict
                        )
                        else None
                    ),
                    "cap_removal": (
                        direct_pspg_active_support_completion_replays.get(
                            "cap_removal"
                        )
                        if isinstance(
                            direct_pspg_active_support_completion_replays, dict
                        )
                        else None
                    ),
                    "next_requirement": (
                        direct_pspg_active_support_completion_replays.get(
                            "next_requirement"
                        )
                        if isinstance(
                            direct_pspg_active_support_completion_replays, dict
                        )
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "Use this family as evidence for a formulation-derived PSPG support "
                "rule, not as the final implementation."
            ),
        ),
        status_item(
            key="aggregate_no_galerkin_support_selector",
            question=(
                "Can the no-Galerkin/nonpressure zero-coupling selector explain "
                "the moving top rows?"
            ),
            status="supported_partial_for_test10_ruled_out_as_complete_selector",
            conclusion=(
                "The selector reaches part of Test10 and none of Test02, exact "
                "direct PSPG top rows are undercovered by bounded aggregate "
                "samples, and the combined no-Galerkin plus same-sign predicate "
                "still misses the isolated Test02 direct PSPG row."
            ),
            evidence=[
                top_overlap_evidence
                | {
                    "finding": (
                        top_overlap.get("finding")
                        if isinstance(top_overlap, dict)
                        else None
                    ),
                    "no_galerkin_support_finding": (
                        top_overlap.get("no_galerkin_support_finding")
                        if isinstance(top_overlap, dict)
                        else None
                    ),
                    "exact_to_aggregate_sample_finding": (
                        top_overlap.get("exact_to_aggregate_sample_finding")
                        if isinstance(top_overlap, dict)
                        else None
                    ),
                },
                no_galerkin_gate_relevance_evidence
                | {
                    "finding": no_galerkin_gate_relevance_finding,
                    "status": (
                        no_galerkin_gate_relevance.get("status")
                        if isinstance(no_galerkin_gate_relevance, dict)
                        else None
                    ),
                    "classification": (
                        no_galerkin_gate_relevance.get("classification")
                        if isinstance(no_galerkin_gate_relevance, dict)
                        else None
                    ),
                    "top_overlap": (
                        no_galerkin_gate_relevance.get("top_overlap")
                        if isinstance(no_galerkin_gate_relevance, dict)
                        else None
                    ),
                    "formulation_candidate": (
                        no_galerkin_gate_relevance.get(
                            "formulation_candidate"
                        )
                        if isinstance(no_galerkin_gate_relevance, dict)
                        else None
                    ),
                    "next_requirement": (
                        no_galerkin_gate_relevance.get("next_requirement")
                        if isinstance(no_galerkin_gate_relevance, dict)
                        else None
                    ),
                },
            ],
            remaining_risk=(
                "Likely necessary for Test10 rank robustness, but insufficient "
                "without the direct PSPG topology/coupling fix; keep it as a "
                "Test10 sub-signal, not the production formulation gate."
            ),
        ),
        status_item(
            key="broad_full_cell_or_scalar_support_extension",
            question=(
                "Can broad full-cell PSPG support or scalar support scaling fix the issue?"
            ),
            status="ruled_out_as_complete_fix",
            conclusion=(
                "Broad full-cell support and scalar controls leave guarded branches "
                "or worsen one case; they are not credible production fixes."
            ),
            evidence=[
                full_cell_evidence,
                root_evidence
                | {
                    "mentions_full_cell_ruled_out": (
                        "full-cell VMS/PSPG continuity support" in report_text
                    ),
                    "mentions_uniform_multiplier_ruled_out": (
                        "uniform global multiplier" in report_text
                        or "one scalar tuning knob" in report_text
                    ),
                },
            ],
            remaining_risk=(
                "The eventual formulation may still include local support changes, "
                "but not a broad scalar or full-cell sweep."
            ),
        ),
    ]

    status_counts = Counter(item["status"] for item in hypotheses)
    missing_evidence = sorted(
        {
            path
            for item in hypotheses
            for path in item["missing_evidence"]
        }
    )
    unresolved = [
        item["key"]
        for item in hypotheses
        if "unresolved" in item["status"]
    ]

    return {
        "scope": "Test02/Test10 open-vessel free-surface root-cause status",
        "root_report": root_evidence,
        "artifact_root": str(artifact_root),
        "hypothesis_count": len(hypotheses),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence": missing_evidence,
        "unresolved_hypotheses": unresolved,
        "overall_status": (
            "primary_formulation_target_unresolved"
            if unresolved
            else "all_tracked_hypotheses_resolved_or_ruled_out"
        ),
        "recommended_next_target": (
            "Formulation-side direct PSPG pressure-gradient support topology/"
            "coupling rule on solve-time active cut volumes, with the pressure "
            "update guard retained as a safety diagnostic."
        ),
        "hypotheses": hypotheses,
    }


def main() -> None:
    args = parse_args()
    report = build_status_report(
        artifact_root=args.artifact_root,
        root_report=args.root_report,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
