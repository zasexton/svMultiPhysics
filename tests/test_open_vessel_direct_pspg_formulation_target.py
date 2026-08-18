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
        / "audit_direct_pspg_formulation_target.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_formulation_target",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_pspg_formulation_target_classifies_isolated_and_coherent_cases():
    audit = _load_audit_module()
    report = audit.build_formulation_target_report(
        {
            "finding": "top_rows_split_between_direct_pspg_and_ghost_penalty_paths",
            "cross_policy_neighbor_comparisons": [
                {
                    "base_label": "test02",
                    "current_top_isolated_cross_policy_patch_global_dofs": [
                        100,
                        103,
                    ],
                }
            ],
            "cases": [
                {
                    "label": "test02",
                    "finding": "mixed_direct_pspg_and_ghost_penalty_top_rows",
                    "direct_pspg_balance_global_dofs": [100, 101],
                    "ghost_penalty_balance_global_dofs": [90],
                    "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs": [
                        100
                    ],
                    "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                        101
                    ],
                    "direct_pspg_same_sign_pressure_action_component_count": 1,
                    "direct_pspg_same_sign_pressure_action_components": [
                        {
                            "component_index": 1,
                            "global_dofs": [101],
                            "direct_pspg_global_dofs": [101],
                            "ghost_penalty_global_dofs": [],
                            "size": 1,
                            "same_sign_pressure_action_edge_count": 0,
                            "contains_rank1": False,
                            "boundary_class_counts": {"boundary_face": 1},
                            "incident_support_class_counts": {
                                "shared_boundary_support": 1
                            },
                        }
                    ],
                    "direct_pspg_sparse_direct_self_entry_global_dofs": [100],
                    "direct_pspg_low_direct_self_ratio_global_dofs": [100],
                    "direct_pspg_missing_wall_normal_self_global_dofs": [100],
                    "direct_pspg_missing_wall_tangential_self_global_dofs": [101],
                    "boundary_class_counts": {"boundary_face": 2},
                    "incident_support_class_counts": {"shared_boundary_support": 2},
                    "physical_path_class_counts": {
                        "direct_pspg_weak_self_with_wall_support": 2,
                        "ghost_penalty_positive_self": 1,
                    },
                },
                {
                    "label": "test10",
                    "finding": "direct_pspg_top_rows_without_ghost_penalty",
                    "direct_pspg_balance_global_dofs": [200, 201, 202],
                    "ghost_penalty_balance_global_dofs": [],
                    "direct_pspg_same_sign_pressure_action_isolated_direct_global_dofs": [],
                    "direct_pspg_same_sign_pressure_action_direct_coverage_global_dofs": [
                        200,
                        201,
                        202,
                    ],
                    "direct_pspg_same_sign_pressure_action_component_count": 1,
                    "direct_pspg_same_sign_pressure_action_components": [
                        {
                            "component_index": 1,
                            "global_dofs": [200, 201, 202],
                            "direct_pspg_global_dofs": [200, 201, 202],
                            "ghost_penalty_global_dofs": [],
                            "size": 3,
                            "same_sign_pressure_action_edge_count": 3,
                            "contains_rank1": True,
                        }
                    ],
                    "direct_pspg_sparse_direct_self_entry_global_dofs": [201],
                    "direct_pspg_low_direct_self_ratio_global_dofs": [],
                    "direct_pspg_missing_wall_normal_self_global_dofs": [202],
                    "direct_pspg_missing_wall_tangential_self_global_dofs": [],
                },
            ],
        }
    )

    assert report["finding"] == (
        "mixed_isolated_and_coherent_direct_pspg_formulation_targets"
    )
    assert report["remaining_hypothesis"] == (
        "direct_pspg_pressure_gradient_support_topology"
    )
    assert report["direct_target_case_count"] == 2
    assert report["ghost_branch_case_count"] == 1
    assert report["formulation_target_class_counts"] == {
        "coherent_direct_pspg_pressure_action_patch": 1,
        "isolated_direct_pspg_row_with_ghost_penalty_branch": 1,
    }
    coverage = {item["key"]: item for item in report["candidate_coverage"]}
    assert coverage["same_sign_pressure_action_patch_only"][
        "covers_all_cases"
    ] is False
    assert coverage["same_sign_pressure_action_patch_only"]["cases"][0][
        "uncovered_direct_target_global_dofs"
    ] == [100]
    assert coverage["direct_support_gap_rows_only"]["covers_all_cases"] is False
    assert coverage["direct_support_gap_rows_only"]["cases"][1][
        "uncovered_direct_target_global_dofs"
    ] == [200]
    assert coverage["direct_support_gap_or_same_sign_pressure_action_patch"][
        "covers_all_cases"
    ]
    assert report["recommended_next_predicate"]["key"] == (
        "direct_support_gap_or_same_sign_pressure_action_patch"
    )
    assert report["recommended_next_predicate"]["production_readiness"] == (
        "coverage_complete_but_diagnostic_only"
    )
    assert report["predicate_derivation_readiness"] == (
        "coverage_complete_but_no_formulation_side_derivation"
    )
    assert report["complete_diagnostic_candidate_keys"] == [
        "isolated_or_same_sign_direct_targets",
        "direct_support_gap_or_same_sign_pressure_action_patch",
    ]
    assert report["complete_formulation_ready_candidate_keys"] == []
    assert (
        "solve-time active cut-volume direct PSPG pressure-gradient support"
        in report["next_derivation_requirement"]
    )
    assert coverage["direct_support_gap_or_same_sign_pressure_action_patch"][
        "derivation"
    ]["depends_on_top_update_rows"]
    assert coverage["direct_support_gap_or_same_sign_pressure_action_patch"][
        "derivation"
    ]["depends_on_pressure_update_values"]

    cases = {case["label"]: case for case in report["cases"]}
    test02 = cases["test02"]
    assert test02["formulation_target_class"] == (
        "isolated_direct_pspg_row_with_ghost_penalty_branch"
    )
    assert test02["direct_pspg_target_global_dofs"] == [100, 101]
    assert test02["ghost_penalty_branch_global_dofs"] == [90]
    assert test02["direct_pspg_isolated_global_dofs"] == [100]
    assert test02["cross_policy_isolated_patch_global_dofs"] == [100, 103]
    assert test02["direct_pspg_support_gap_global_dofs"] == [100, 101]
    assert any(
        "isolated direct rows" in requirement
        for requirement in test02["formulation_requirements"]
    )
    assert any(
        "ghost-penalty branch" in requirement
        for requirement in test02["formulation_requirements"]
    )

    test10 = cases["test10"]
    assert test10["formulation_target_class"] == (
        "coherent_direct_pspg_pressure_action_patch"
    )
    assert test10["direct_pspg_target_global_dofs"] == [200, 201, 202]
    assert test10["ghost_penalty_branch_global_dofs"] == []
    assert test10["direct_pspg_same_sign_covered_global_dofs"] == [
        200,
        201,
        202,
    ]
    assert any(
        "coherent direct-PSPG patch" in requirement
        for requirement in test10["formulation_requirements"]
    )
