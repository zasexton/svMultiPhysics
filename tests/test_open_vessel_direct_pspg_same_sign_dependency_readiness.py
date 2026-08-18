import importlib.util
from pathlib import Path
import sys


def _load_audit_module():
    repo = Path(__file__).resolve().parents[1]
    script = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
        / "audit_direct_pspg_same_sign_dependency_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_same_sign_dependency_readiness",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_map():
    return {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [100, 101]},
            {
                "label": "test10",
                "direct_pspg_target_global_dofs": [200, 201, 202],
            },
        ]
    }


def _predicate_report(*, include_ready_candidate=False):
    candidates = [
        {
            "key": "same_sign_pressure_action_patch",
            "row_sources": ["same_sign_pressure_action"],
            "production_readiness": "diagnostic_only_partial_expected",
            "derivation_status": "needs_preupdate_direct_pspg_pressure_action_graph",
            "depends_on_pressure_update_values_in_current_artifact": True,
            "finding": "partial_audited_coverage",
            "covers_all_audited_targets": False,
            "exact_audited_coverage": False,
        },
        {
            "key": "sparse_direct_self_or_same_sign_pressure_action_patch",
            "row_sources": ["sparse_direct_self_entry", "same_sign_pressure_action"],
            "production_readiness": (
                "formulation_candidate_pending_global_solve_time_emission"
            ),
            "derivation_status": (
                "derive_from_direct_pspg_pressure_gradient_self_topology_and_action_graph"
            ),
            "depends_on_pressure_update_values_in_current_artifact": True,
            "finding": "exact_audited_coverage",
            "covers_all_audited_targets": True,
            "exact_audited_coverage": True,
        },
    ]
    exact_keys = ["sparse_direct_self_or_same_sign_pressure_action_patch"]
    complete_keys = ["sparse_direct_self_or_same_sign_pressure_action_patch"]
    if include_ready_candidate:
        candidates.append(
            {
                "key": "formulation_side_physical_patch",
                "row_sources": ["active_pspg_pressure_gradient_support"],
                "production_readiness": "formulation_ready",
                "derivation_status": "available_before_pressure_update",
                "depends_on_pressure_update_values_in_current_artifact": False,
                "finding": "exact_audited_coverage",
                "covers_all_audited_targets": True,
                "exact_audited_coverage": True,
            }
        )
        exact_keys.append("formulation_side_physical_patch")
        complete_keys.append("formulation_side_physical_patch")

    return {
        "finding": "narrow_formulation_side_candidate_identified_needs_global_emission",
        "preferred_next_candidate": {
            "key": "sparse_direct_self_or_same_sign_pressure_action_patch"
        },
        "exact_audited_candidate_keys": exact_keys,
        "complete_audited_candidate_keys": complete_keys,
        "candidates": candidates,
    }


def _global_selectivity():
    return {
        "finding": "global_candidate_selector_overbroad_matrix_proxy_not_formulation_ready",
        "direct_self_support_ratio_gate_finding": (
            "direct_self_support_ratio_gate_misses_targets"
        ),
        "graph_local_support_ratio_gate_finding": (
            "graph_local_support_ratio_gate_overbroad"
        ),
        "pressure_action_moderate_degree_gate_finding": (
            "pressure_action_moderate_degree_gate_misses_targets"
        ),
        "pressure_action_moderate_sum_ratio_gate_finding": (
            "pressure_action_moderate_sum_ratio_gate_misses_targets"
        ),
        "pressure_action_self_dominant_gate_finding": (
            "pressure_action_self_dominant_gate_misses_targets"
        ),
        "sparse_seeded_pressure_action_radius1_gate_finding": (
            "sparse_seeded_pressure_action_radius1_gate_overbroad"
        ),
        "sparse_seeded_pressure_action_radius2_gate_finding": (
            "sparse_seeded_pressure_action_radius2_gate_overbroad"
        ),
        "cases": [
            {
                "label": "test02",
                "finding": "raw_global_candidate_selector_overbroad",
                "direct_target_count": 2,
                "preferred_to_target_ratio": 40.0,
                "sparse_seeded_pressure_action_radius1_to_target_ratio": 15.0,
                "sparse_seeded_pressure_action_radius1_covers_targets": True,
                "pressure_action_moderate_sum_ratio_covers_targets": False,
            },
            {
                "label": "test10",
                "finding": "raw_global_candidate_selector_overbroad",
                "direct_target_count": 3,
                "preferred_to_target_ratio": 12.0,
                "sparse_seeded_pressure_action_radius1_to_target_ratio": 12.0,
                "sparse_seeded_pressure_action_radius1_covers_targets": True,
                "pressure_action_moderate_sum_ratio_covers_targets": True,
            },
        ],
    }


def _toprow_provenance():
    return {
        "cross_policy_neighbor_comparisons": [],
        "cases": [
            {
                "label": "test02",
                "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs": [
                    100
                ],
                "top_update_rows": [
                    {
                        "global_dof": 100,
                        "direct_pspg_patch_neighbor_profile": {
                            "pressure_action_neighbor_dofs": [100, 101],
                            "direct_pgrad_row_neighbor_dofs": [100, 101, 102],
                        },
                    }
                ],
            },
            {
                "label": "test10",
                "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs": [],
                "top_update_rows": [],
            },
        ],
    }


def _pressure_disabled_toprow_provenance():
    return {
        "cross_policy_neighbor_comparisons": [],
        "cases": [
            {
                "label": "test02_pressure_disabled",
                "direct_pspg_same_sign_pressure_action_component_count": 1,
                "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                    100,
                    101,
                    102,
                ],
                "top_update_rows": [
                    {
                        "global_dof": 100,
                        "direct_pspg_patch_neighbor_profile": {
                            "same_sign_pressure_action_top_update_neighbor_dofs": [
                                101
                            ],
                            "direct_pgrad_direct_pspg_top_neighbor_dofs": [
                                101,
                                102,
                            ],
                        },
                    }
                ],
            },
            {
                "label": "test10_pressure_disabled",
                "direct_pspg_same_sign_pressure_action_component_count": 1,
                "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                    200,
                    201,
                    202,
                ],
                "top_update_rows": [],
            },
        ],
    }


def test_same_sign_dependency_audit_blocks_update_dependent_exact_patch():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        predicate_report=_predicate_report(),
        global_selectivity=_global_selectivity(),
        full_toprow_provenance=_toprow_provenance(),
        pressure_disabled_toprow_provenance=_pressure_disabled_toprow_provenance(),
    )

    assert report["finding"] == (
        "same_sign_patch_blocked_by_pressure_update_dependency_and_"
        "preupdate_proxies"
    )
    dependency = report["dependency_summary"]
    assert dependency["preferred_candidate_depends_on_pressure_update"]
    assert dependency["all_exact_candidates_depend_on_pressure_update"]
    assert dependency["complete_non_update_dependent_candidate_keys"] == []
    assert dependency["same_sign_update_dependent_candidate_keys"] == [
        "same_sign_pressure_action_patch",
        "sparse_direct_self_or_same_sign_pressure_action_patch",
    ]

    proxy = report["preupdate_proxy_summary"]
    assert proxy["all_preupdate_proxy_gates_failed"]
    assert "sparse_seeded_pressure_action_radius1_gate_finding" in proxy[
        "failed_gate_keys"
    ]
    cases = {case["label"]: case for case in proxy["cases"]}
    assert cases["test02"]["candidate_to_target_ratios"][
        "preferred_to_target_ratio"
    ] == 40.0
    assert cases["test10"]["covers_targets"][
        "pressure_action_moderate_sum_ratio"
    ]

    cross_policy = report["cross_policy_patch_summary"]
    assert cross_policy["finding"] == (
        "cross_policy_patch_evidence_is_post_update_diagnostic_only"
    )
    test02 = {case["label"]: case for case in cross_policy["cases"]}["test02"]
    assert test02["finding"] == (
        "cross_policy_patch_visible_only_after_pressure_disabled_update"
    )
    assert test02["pressure_disabled_direct_patch_global_dofs"] == [100, 101]
    assert "pressure-update signs" in report["conclusion"]


def test_same_sign_dependency_audit_allows_complete_non_update_candidate():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        predicate_report=_predicate_report(include_ready_candidate=True),
        global_selectivity=_global_selectivity(),
        full_toprow_provenance=_toprow_provenance(),
        pressure_disabled_toprow_provenance=_pressure_disabled_toprow_provenance(),
    )

    assert report["finding"] == "formulation_ready_candidate_available"
    assert report["dependency_summary"][
        "complete_non_update_dependent_candidate_keys"
    ] == ["formulation_side_physical_patch"]
