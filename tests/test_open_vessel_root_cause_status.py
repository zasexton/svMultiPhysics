import importlib.util
import json
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
        / "audit_open_vessel_root_cause_status.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_open_vessel_root_cause_status",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(root, name, data):
    path = root / name
    path.write_text(json.dumps(data), encoding="utf-8")


def _evidence_by_suffix(item, suffix):
    return next(
        evidence
        for evidence in item["evidence"]
        if evidence.get("path", "").endswith(suffix)
    )


def _pressure_update_fixture(threshold, update_pa, point_index, support_class):
    return {
        "absolute_threshold_pa": threshold,
        "status": "diagnostic_pressure_update_guard_triggered",
        "finding": (
            f"1 transition(s) exceeded {threshold:g} Pa on active/wet support. "
            f"Worst active/wet update was {update_pa:.6g} Pa."
        ),
        "triggered_transition_count": 1,
        "worst_by_category": {
            "active_or_wet_supported": {
                "abs_pressure_delta_pa": update_pa,
                "point_index": point_index,
                "support_class": support_class,
                "active_fluid": 1.0,
                "incident_wet_fraction_min_positive": 1.0,
            },
        },
    }


def test_root_cause_status_matrix_classifies_current_hypotheses(tmp_path):
    audit = _load_audit_module()
    root_report = tmp_path / "root.md"
    root_report.write_text(
        "\n".join(
            [
                "pressure-disabled controls are not \"turn off ghost penalty\".",
                "The missing direct generated-interface pressure trace reference is ruled out.",
                "The direct generated-interface tangential pressure-gradient support is ruled out.",
                "The cut-context transition is not the source.",
                "The pre-commit pressure-update rejection catches the row.",
                "The row grows as dt halves.",
                "full-cell VMS/PSPG continuity support is not enough.",
                "This is not one scalar tuning knob.",
            ]
        ),
        encoding="utf-8",
    )

    _write_json(
        tmp_path,
        "linear_pressure_cut_volume_patch_audit_20260605.json",
        {
            "passed": True,
            "hazard_detected": True,
            "pspg_hydrostatic_hazard_detected": True,
            "cases": [
                {
                    "name": "retained_cut_volume_support",
                    "pspg_hydrostatic_balance": {
                        "direct_support_gap_or_same_sign_patch_completion": {
                            "balanced_max_to_strongest_support_target_response_ratio": 2.16,
                            "preserves_hydrostatic_balance": True,
                            "preserves_constant_pressure_null": True,
                        },
                    },
                },
                {
                    "name": "full_volume_one_cell_boundary_topology",
                    "pspg_hydrostatic_balance": {
                        "direct_support_gap_or_same_sign_patch_completion": {
                            "balanced_max_to_strongest_support_target_response_ratio": 2.17,
                            "preserves_hydrostatic_balance": True,
                            "preserves_constant_pressure_null": True,
                        },
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_pressure_operator_toprow_provenance_20260606.json",
        {
            "finding": "top_rows_split_between_direct_pspg_and_ghost_penalty_paths",
            "finding_counts": {
                "mixed_direct_pspg_and_ghost_penalty_top_rows": 1,
                "direct_pspg_top_rows_without_ghost_penalty": 1,
            },
            "cases": [
                {
                    "label": "test02",
                    "finding": "mixed_direct_pspg_and_ghost_penalty_top_rows",
                },
                {
                    "label": "test10",
                    "finding": "direct_pspg_top_rows_without_ghost_penalty",
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_formulation_target_20260606.json",
        {
            "finding": "mixed_isolated_and_coherent_direct_pspg_formulation_targets",
            "direct_target_case_count": 2,
            "ghost_branch_case_count": 1,
            "formulation_target_class_counts": {
                "coherent_direct_pspg_pressure_action_patch": 1,
                "isolated_direct_pspg_row_with_ghost_penalty_branch": 1,
            },
            "recommended_next_predicate": {
                "key": "direct_support_gap_or_same_sign_pressure_action_patch",
                "covers_all_cases": True,
                "production_readiness": "coverage_complete_but_diagnostic_only",
                "solve_time_derivation_status": (
                    "requires_formulation_side_topology_replacement"
                ),
            },
            "predicate_derivation_readiness": (
                "coverage_complete_but_no_formulation_side_derivation"
            ),
            "complete_diagnostic_candidate_keys": [
                "isolated_or_same_sign_direct_targets",
                "direct_support_gap_or_same_sign_pressure_action_patch",
            ],
            "complete_formulation_ready_candidate_keys": [],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_cut_adjacent_support_pressure_window_20260606.json",
        {
            "finding": (
                "trace_only_support_ruled_out_recent_pruned_volume_not_direct_"
                "trace_only_driver"
            ),
            "trace_only_cut_adjacent_support_ruled_out_before_guards": True,
            "pruned_generated_volume_present_before_some_guard": True,
            "trace_only_cut_adjacent_support_cases": [],
            "pruned_generated_volume_cases": ["test10"],
            "retained_volume_support_cases": ["test02", "test10"],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_pressure_stabilization_driver_windows_20260607.json",
        {
            "finding": (
                "cut_adjacent_pressure_stabilization_not_direct_worst_update_driver"
            ),
            "status": (
                "ghost_penalty_direct_worst_update_path_ruled_out_for_saved_windows"
            ),
            "all_saved_window_worst_updates_nonincident": True,
            "any_saved_window_worst_update_incident": False,
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "worst_update_not_incident_to_cut_adjacent_stabilization"
                    ),
                    "incident_cut_adjacent_face_count": 0,
                    "worst_update_abs_pressure_delta_pa": 2112204.128955333,
                },
                {
                    "label": "test10",
                    "finding": (
                        "worst_update_not_incident_to_cut_adjacent_stabilization"
                    ),
                    "incident_cut_adjacent_face_count": 0,
                    "worst_update_abs_pressure_delta_pa": 1075.2113565356985,
                },
            ],
            "next_requirement": (
                "Treat cut-adjacent pressure ghost penalty as branch-shaping "
                "evidence only."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_graph_completion_candidate_readiness_20260606.json",
        {
            "finding": (
                "support_gap_graph_completion_selectors_overbroad_and_"
                "test02_unstable"
            ),
            "overbroad_modes": [
                "shared_row_schur_support_gap_patch_completion",
            ],
            "test02_unstable_modes": [
                "shared_row_schur_support_gap_patch_completion",
            ],
            "test10_guard_clear_modes": [
                "shared_row_schur_support_gap_patch_completion",
            ],
            "direct_target_counts": {"test02": 7, "test10": 12},
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_graph_completion_replay_family_20260607.json",
        {
            "finding": (
                "direct_pspg_graph_completion_replay_family_rules_out_"
                "post_assembly_selector_variants"
            ),
            "variant_findings": {
                "least_selector_schur_only": "both_guards_still_trigger",
                "least_selector_schur_edge_balance": (
                    "test10_clears_but_test02_unstable"
                ),
                "support_rank_neighborhood_depth1": "both_guards_still_trigger",
            },
            "test10_guard_clear_variants": [
                "least_selector_schur_edge_balance",
            ],
            "test02_unstable_variants": [
                "least_selector_schur_edge_balance",
            ],
            "test10_still_trigger_variants": [
                "least_selector_schur_only",
                "support_rank_neighborhood_depth1",
            ],
            "next_requirement": (
                "Move the Schur/topology and edge-balance evidence into a "
                "formulation-side direct PSPG pressure-gradient support/coupling "
                "rule."
            ),
            "variants": [
                {
                    "key": "least_selector_schur_only",
                    "cases": [
                        {
                            "label": "test02",
                            "finding": "guard_still_triggered",
                            "candidate_row_count": 304,
                            "accepted_pressure_update_pa": 319684.78410933603,
                        },
                        {
                            "label": "test10",
                            "finding": "guard_still_triggered",
                            "candidate_row_count": 68,
                            "accepted_pressure_update_pa": 122.46838944778688,
                        },
                    ],
                },
                {
                    "key": "least_selector_schur_edge_balance",
                    "cases": [
                        {
                            "label": "test02",
                            "finding": "nonlinear_failed_with_overbroad_patch",
                            "candidate_row_count": 304,
                            "accepted_pressure_update_pa": None,
                        },
                        {
                            "label": "test10",
                            "finding": "guard_cleared",
                            "candidate_row_count": 68,
                            "accepted_pressure_update_pa": 15.254558181653124,
                        },
                    ],
                },
                {
                    "key": "support_rank_neighborhood_depth1",
                    "cases": [
                        {
                            "label": "test02",
                            "finding": "guard_still_triggered",
                            "candidate_row_count": 0,
                            "accepted_pressure_update_pa": 366719.9658064514,
                        },
                        {
                            "label": "test10",
                            "finding": "guard_still_triggered",
                            "candidate_row_count": 21,
                            "accepted_pressure_update_pa": 319.85947884033067,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_graph_completion_stability_tradeoff_"
            "20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_graph_completion_stability_tradeoff_rules_out_"
                "post_assembly_fix"
            ),
            "status": "post_assembly_schur_balance_tradeoff_ruled_out",
            "tradeoff_flags": {
                (
                    "broad_topology_clears_test10_but_destabilizes_test02"
                ): True,
                (
                    "least_selector_schur_stable_but_insufficient_balance_"
                    "clears_test10_but_destabilizes_test02"
                ): True,
                (
                    "localized_balance_gates_fail_test10_and_destabilize_"
                    "test02"
                ): True,
                "support_rank_neighborhood_expansion_too_local": True,
            },
            "least_selector_tradeoff": {
                "schur_only": {
                    "test02": {
                        "finding": "guard_still_triggered",
                        "accepted_pressure_update_pa": 319684.78410933603,
                    },
                    "test02_guard_triggered": True,
                    "test02_nonlinear_failed": False,
                    "test10": {
                        "finding": "guard_still_triggered",
                        "accepted_pressure_update_pa": 122.46838944778688,
                    },
                    "test10_guard_triggered": True,
                },
                "schur_edge_balance": {
                    "test02": {
                        "finding": "nonlinear_failed_with_overbroad_patch",
                        "accepted_pressure_update_pa": None,
                        "final_residual_norm": 27095.229891115217,
                    },
                    "test02_nonlinear_failed": True,
                    "test10": {
                        "finding": "guard_cleared",
                        "accepted_pressure_update_pa": 15.254558181653124,
                    },
                    "test10_guard_cleared": True,
                },
            },
            "localized_balance_variants": [
                {
                    "key": "coupling_deficient_balance",
                    "test02_nonlinear_failed": True,
                    "test10_guard_triggered": True,
                    "test10": {
                        "accepted_pressure_update_pa": 120.94274868680577
                    },
                },
                {
                    "key": "low_pressure_degree_balance",
                    "test02_nonlinear_failed": True,
                    "test10_guard_triggered": True,
                    "test10": {
                        "accepted_pressure_update_pa": 120.9238982982647
                    },
                },
            ],
            "next_requirement": (
                "Do not promote threshold-selected post-assembly Schur fill or "
                "existing-edge balance as the formulation fix."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_active_support_completion_replays_"
            "20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_active_support_completion_replays_rule_out_raw_"
                "active_support_completion"
            ),
            "status": (
                "raw_active_support_completion_directional_but_insufficient"
            ),
            "all_replays_guard_triggered": True,
            "all_replays_accepted_one_step": True,
            "case_updates_pa": {
                "active_support_neigh64": {
                    "test02": 186507.92759082434,
                    "test10": 201.1556177587019,
                },
                "active_support_all": {
                    "test02": 155956.10179486268,
                    "test10": 203.0459932023828,
                },
            },
            "cap_removal": {
                "cap64_neighbor_cap_limited_all_cases": True,
                "uncapped_still_triggers_all_cases": True,
                "by_case": {
                    "test02": {
                        "cap64_neighbor_row_count": 368,
                        "uncapped_neighbor_row_count": 879,
                        "cap64_edge_count": 19456,
                        "uncapped_edge_count": 220856,
                        "uncapped_minus_cap64_update_pa": (
                            -30551.82579596166
                        ),
                    },
                    "test10": {
                        "cap64_neighbor_row_count": 132,
                        "uncapped_neighbor_row_count": 251,
                        "cap64_edge_count": 4352,
                        "uncapped_edge_count": 14722,
                        "uncapped_minus_cap64_update_pa": (
                            1.8903754436808825
                        ),
                    },
                },
            },
            "next_requirement": (
                "Move to a formulation-derived physical support/coupling rule."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_explicit_balance_selector_replays_"
            "20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_explicit_balance_selectors_rule_out_row_lists_and_"
                "pressure_neighborhoods"
            ),
            "status": "explicit_balance_selectors_ruled_out",
            "boundary_provenance": {
                "finding": (
                    "latest_bad_rows_can_be_candidates_without_balance_coverage"
                ),
                "boundary_topology_finding": (
                    "boundary_top_update_candidates_missing_balance"
                ),
            },
            "ruleout_flags": {
                "boundary_balance_predicate_misses_latest_bad_rows": True,
                "explicit_row_lists_ruled_out": True,
                "current_pressure_neighborhoods_ruled_out": True,
            },
            "ruled_out_by_variant": {
                "explicit_direct_rows": True,
                "explicit_shifted_rows": True,
                "explicit_cross_policy_patch": True,
                "explicit_operator_top_rows": True,
                "explicit_neighborhood_depth1": True,
                "explicit_neighborhood_depth2": True,
            },
            "variants": [
                {
                    "key": "explicit_direct_rows",
                    "cases": [
                        {
                            "label": "test02",
                            "finding": "guard_still_triggered",
                            "accepted_pressure_update_pa": 102071.75239899695,
                            "balance_candidate_row_count": 7,
                        },
                        {
                            "label": "test10",
                            "finding": "guard_still_triggered",
                            "accepted_pressure_update_pa": 120.642165923368,
                            "balance_candidate_row_count": 12,
                        },
                    ],
                },
                {
                    "key": "explicit_shifted_rows",
                    "cases": [
                        {
                            "label": "test02",
                            "finding": "nonlinear_failed",
                            "accepted_pressure_update_pa": None,
                            "balance_candidate_row_count": 8,
                        },
                        {
                            "label": "test10",
                            "finding": "guard_still_triggered",
                            "accepted_pressure_update_pa": 106.81906280077567,
                            "balance_candidate_row_count": 13,
                        },
                    ],
                },
                {
                    "key": "explicit_neighborhood_depth2",
                    "cases": [
                        {
                            "label": "test02",
                            "finding": "guard_still_triggered",
                            "accepted_pressure_update_pa": 103141.83046458055,
                            "balance_candidate_row_count": 113,
                        },
                        {
                            "label": "test10",
                            "finding": "guard_still_triggered",
                            "accepted_pressure_update_pa": 118.70937283831643,
                            "balance_candidate_row_count": 50,
                        },
                    ],
                },
            ],
            "next_requirement": (
                "Do not promote explicit direct-row lists, shifted-row lists, "
                "exact operator top-row lists, cross-policy patch seeds, or "
                "one/two-ring current-pressure-neighborhood balance selectors."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_formulation_side_candidate_predicates_20260606.json",
        {
            "finding": (
                "narrow_formulation_side_candidate_identified_needs_global_"
                "emission"
            ),
            "preferred_next_candidate": {
                "key": (
                    "sparse_direct_self_or_same_sign_pressure_action_patch"
                ),
                "production_readiness": (
                    "formulation_candidate_pending_global_solve_time_emission"
                ),
                "derivation_status": (
                    "derive_from_direct_pspg_pressure_gradient_self_topology_"
                    "and_action_graph"
                ),
            },
            "exact_audited_candidate_keys": [
                "sparse_direct_self_or_same_sign_pressure_action_patch"
            ],
            "partial_candidate_keys": [
                "same_sign_pressure_action_patch",
                "zero_galerkin_nonpressure_or_same_sign_pressure_action_patch",
            ],
            "direct_target_counts": {"test02": 7, "test10": 12},
            "current_artifact_limitation": (
                "The preferred candidate is still proven only on exact sampled "
                "top rows."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_global_candidate_emission_20260606.json",
        {
            "finding": "candidate_emission_covers_audited_targets",
            "missing_case_labels": [],
            "cases": [
                {
                    "label": "test02",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "preferred_candidate_count": 866,
                    "covered_direct_target_global_dofs": [
                        10676,
                        10952,
                        12211,
                        10954,
                        12213,
                        10953,
                        12212,
                    ],
                    "candidate_list_truncated": False,
                },
                {
                    "label": "test10",
                    "finding": "candidate_emitted_covers_audited_targets",
                    "preferred_candidate_count": 251,
                    "covered_direct_target_global_dofs": [
                        3526,
                        3456,
                        3925,
                        3455,
                        3924,
                        3454,
                        3923,
                        3451,
                        3920,
                        3525,
                        3919,
                        3450,
                    ],
                    "candidate_list_truncated": False,
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_global_candidate_selectivity_20260607.json",
        {
            "finding": (
                "global_candidate_selector_overbroad_matrix_proxy_not_"
                "formulation_ready"
            ),
            "direct_self_support_ratio_gate_finding": (
                "direct_self_support_ratio_gate_misses_targets"
            ),
            "direct_self_support_ratio_case_findings": {
                "test02": "sparse_or_moderate_direct_self_ratio_gate_misses_targets",
                "test10": "sparse_or_moderate_direct_self_ratio_gate_overbroad",
            },
            "graph_local_support_ratio_gate_finding": (
                "graph_local_support_ratio_gate_misses_targets"
            ),
            "graph_local_support_ratio_case_findings": {
                "test02": "graph_local_moderate_direct_self_ratio_gate_misses_targets",
                "test10": "graph_local_moderate_direct_self_ratio_gate_overbroad",
            },
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
            "sparse_seeded_pressure_action_radius1_case_findings": {
                "test02": "sparse_seeded_pressure_action_radius1_gate_overbroad",
                "test10": "sparse_seeded_pressure_action_radius1_gate_overbroad",
            },
            "sparse_seeded_pressure_action_radius2_case_findings": {
                "test02": "sparse_seeded_pressure_action_radius2_gate_overbroad",
                "test10": "sparse_seeded_pressure_action_radius2_gate_overbroad",
            },
            "cases": [
                {
                    "label": "test02",
                    "finding": "raw_global_candidate_selector_overbroad",
                    "preferred_to_target_ratio": 866 / 7,
                    "sparse_direct_self_to_target_ratio": 545 / 7,
                    "sparse_or_moderate_direct_self_ratio_to_target_ratio": 572 / 7,
                    "sparse_or_moderate_direct_self_ratio_covers_targets": False,
                    (
                        "sparse_or_moderate_direct_self_ratio_"
                        "selector_overbroad"
                    ): True,
                    "graph_local_moderate_direct_self_ratio_to_target_ratio": (
                        584 / 7
                    ),
                    "graph_local_moderate_direct_self_ratio_covers_targets": False,
                    (
                        "graph_local_moderate_direct_self_ratio_"
                        "selector_overbroad"
                    ): True,
                    "sparse_seeded_pressure_action_radius1_to_target_ratio": (
                        818 / 7
                    ),
                    "sparse_seeded_pressure_action_radius2_to_target_ratio": (
                        866 / 7
                    ),
                    "sparse_seeded_pressure_action_radius1_covers_targets": True,
                    "sparse_seeded_pressure_action_radius2_covers_targets": True,
                    "pressure_action_moderate_degree_to_target_ratio": 167 / 7,
                    "pressure_action_moderate_degree_covers_targets": False,
                    "pressure_action_moderate_sum_ratio_to_target_ratio": 561 / 7,
                    "pressure_action_moderate_sum_ratio_covers_targets": False,
                    "pressure_action_self_dominant_to_target_ratio": 1 / 7,
                    "pressure_action_self_dominant_covers_targets": False,
                    "matrix_pressure_action_covers_all_direct_rows": True,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 866,
                    (
                        "sparse_seeded_matrix_pressure_action_component_"
                        "to_target_ratio"
                    ): 866 / 7,
                    (
                        "sparse_seeded_matrix_pressure_action_component_"
                        "covers_targets"
                    ): True,
                    (
                        "sparse_seeded_matrix_pressure_action_component_"
                        "selector_overbroad"
                    ): True,
                },
                {
                    "label": "test10",
                    "finding": "raw_global_candidate_selector_overbroad",
                    "preferred_to_target_ratio": 251 / 12,
                    "sparse_direct_self_to_target_ratio": 217 / 12,
                    "sparse_or_moderate_direct_self_ratio_to_target_ratio": 217 / 12,
                    "sparse_or_moderate_direct_self_ratio_covers_targets": True,
                    (
                        "sparse_or_moderate_direct_self_ratio_"
                        "selector_overbroad"
                    ): True,
                    "graph_local_moderate_direct_self_ratio_to_target_ratio": (
                        211 / 12
                    ),
                    "graph_local_moderate_direct_self_ratio_covers_targets": True,
                    (
                        "graph_local_moderate_direct_self_ratio_"
                        "selector_overbroad"
                    ): True,
                    "sparse_seeded_pressure_action_radius1_to_target_ratio": (
                        251 / 12
                    ),
                    "sparse_seeded_pressure_action_radius2_to_target_ratio": (
                        251 / 12
                    ),
                    "sparse_seeded_pressure_action_radius1_covers_targets": True,
                    "sparse_seeded_pressure_action_radius2_covers_targets": True,
                    "pressure_action_moderate_degree_to_target_ratio": 99 / 12,
                    "pressure_action_moderate_degree_covers_targets": False,
                    "pressure_action_moderate_sum_ratio_to_target_ratio": 212 / 12,
                    "pressure_action_moderate_sum_ratio_covers_targets": True,
                    "pressure_action_self_dominant_to_target_ratio": 0.0,
                    "pressure_action_self_dominant_covers_targets": False,
                    "matrix_pressure_action_covers_all_direct_rows": True,
                    "sparse_seeded_matrix_pressure_action_component_dof_count": 251,
                    (
                        "sparse_seeded_matrix_pressure_action_component_"
                        "to_target_ratio"
                    ): 251 / 12,
                    (
                        "sparse_seeded_matrix_pressure_action_component_"
                        "covers_targets"
                    ): True,
                    (
                        "sparse_seeded_matrix_pressure_action_component_"
                        "selector_overbroad"
                    ): True,
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_boundary_provenance_selectivity_20260607.json",
        {
            "finding": (
                "mesh_boundary_incident_support_selectors_miss_audited_targets"
            ),
            "profile_evidence": {
                "test02": {"profile_status": "ok"},
                "test10": {"profile_status": "ok"},
            },
            "selectors": [
                {
                    "key": "preferred_boundary_only",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 358,
                            "covered_direct_target_count": 3,
                        },
                        {
                            "label": "test10",
                            "selected_count": 188,
                            "covered_direct_target_count": 9,
                        },
                    ],
                },
                {
                    "key": "preferred_one_cell_boundary",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
                {
                    "key": "sparse_or_moderate_direct_self_boundary",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 237,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 168,
                            "covered_direct_target_count": 9,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_named_face_provenance_selectivity_"
            "20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_named_face_provenance_selectors_not_formulation_ready"
            ),
            "status": "named_face_boundary_gate_ruled_out",
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "named_face_provenance_selectors_overbroad_or_miss_targets"
                    ),
                    "profile_evidence": {"profile_status": "ok"},
                    "target_rows_present_count": 7,
                    "target_named_faces": [
                        "obstacle",
                        "wall_front",
                        "wall_top",
                    ],
                    "target_face_classes": [
                        "named_face_intersection",
                        "no_named_face",
                        "single_named_face",
                    ],
                    "selectors": [
                        {
                            "key": "preferred_target_named_face_union",
                            "selected_count": 264,
                            "covered_target_count": 5,
                        },
                        {
                            "key": "preferred_target_named_face_signature",
                            "selected_count": 576,
                            "covered_target_count": 7,
                        },
                    ],
                },
                {
                    "label": "test10",
                    "finding": (
                        "named_face_provenance_selectors_overbroad_or_miss_targets"
                    ),
                    "profile_evidence": {"profile_status": "ok"},
                    "target_rows_present_count": 12,
                    "target_named_faces": [
                        "wall_back",
                        "wall_front",
                        "wall_right",
                    ],
                    "target_face_classes": [
                        "multi_face_intersection",
                        "named_face_intersection",
                        "no_named_face",
                        "single_named_face",
                    ],
                    "selectors": [
                        {
                            "key": "preferred_target_named_face_union",
                            "selected_count": 187,
                            "covered_target_count": 9,
                        },
                        {
                            "key": "preferred_target_named_face_signature",
                            "selected_count": 153,
                            "covered_target_count": 12,
                        },
                    ],
                },
            ],
            "next_requirement": (
                "Continue the direct PSPG support/coupling search with a "
                "physical topology discriminator beyond named wall/obstacle "
                "face membership."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_cut_state_provenance_selectivity_20260607.json",
        {
            "finding": (
                "cut_state_provenance_selectors_overbroad_or_miss_targets"
            ),
            "profile_evidence": {
                "test02": {"profile_status": "ok"},
                "test10": {"profile_status": "ok"},
            },
            "selectors": [
                {
                    "key": "preferred_inactive_point",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 634,
                            "covered_direct_target_count": 7,
                            "target_wet_support_class_counts": {
                                "dry_only_incident_support": 6,
                                "mixed_cut_dry_incident_support": 1,
                            },
                        },
                        {
                            "label": "test10",
                            "selected_count": 120,
                            "covered_direct_target_count": 12,
                            "target_wet_support_class_counts": {
                                "dry_only_incident_support": 12,
                            },
                        },
                    ],
                },
                {
                    "key": "preferred_dry_only_incident_support",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 566,
                            "covered_direct_target_count": 6,
                            "target_wet_support_class_counts": {
                                "dry_only_incident_support": 6,
                                "mixed_cut_dry_incident_support": 1,
                            },
                        },
                        {
                            "label": "test10",
                            "selected_count": 99,
                            "covered_direct_target_count": 12,
                            "target_wet_support_class_counts": {
                                "dry_only_incident_support": 12,
                            },
                        },
                    ],
                },
                {
                    "key": "preferred_cut_incident_support",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 131,
                            "covered_direct_target_count": 1,
                            "target_wet_support_class_counts": {
                                "dry_only_incident_support": 6,
                                "mixed_cut_dry_incident_support": 1,
                            },
                        },
                        {
                            "label": "test10",
                            "selected_count": 59,
                            "covered_direct_target_count": 0,
                            "target_wet_support_class_counts": {
                                "dry_only_incident_support": 12,
                            },
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_same_sign_dependency_readiness_20260607.json",
        {
            "finding": (
                "same_sign_patch_blocked_by_pressure_update_dependency_and_"
                "preupdate_proxies"
            ),
            "dependency_summary": {
                "preferred_candidate_depends_on_pressure_update": True,
                "all_exact_candidates_depend_on_pressure_update": True,
                "complete_non_update_dependent_candidate_keys": [],
            },
            "preupdate_proxy_summary": {
                "all_preupdate_proxy_gates_failed": True,
                "failed_gate_keys": [
                    "direct_self_support_ratio_gate_finding",
                    "graph_local_support_ratio_gate_finding",
                    "pressure_action_moderate_degree_gate_finding",
                    "pressure_action_moderate_sum_ratio_gate_finding",
                    "pressure_action_self_dominant_gate_finding",
                    "sparse_seeded_pressure_action_radius1_gate_finding",
                    "sparse_seeded_pressure_action_radius2_gate_finding",
                ],
            },
            "cross_policy_patch_summary": {
                "finding": (
                    "cross_policy_patch_evidence_is_post_update_diagnostic_only"
                ),
                "cases": [
                    {
                        "label": "test02",
                        "finding": (
                            "cross_policy_patch_visible_only_after_pressure_disabled_update"
                        ),
                        "pressure_disabled_direct_patch_global_dofs": [
                            10676,
                            10668,
                            10677,
                            10680,
                        ],
                    },
                    {
                        "label": "test10",
                        "finding": "no_full_gradient_isolated_direct_rows",
                        "pressure_disabled_direct_patch_global_dofs": [],
                    },
                ],
            },
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_coupled_patch_dependency_barrier_20260607.json",
        {
            "finding": (
                "coupled_patch_dependency_barrier_requires_solve_time_provenance"
            ),
            "status": (
                "remaining_gate_requires_new_assembly_provenance_diagnostic"
            ),
            "blocker_summary": {
                "same_sign_exact_candidates_update_dependent": True,
                "same_sign_complete_candidates_update_dependent": True,
                "same_sign_has_non_update_dependent_complete_candidate": False,
                "preupdate_proxy_gates_all_failed": True,
                "cross_policy_patch_is_post_update_diagnostic_only": True,
                "no_galerkin_complete_gate_ruled_out": True,
                "retained_fraction_cutoff_not_complete_fix": True,
                "requires_new_solve_time_provenance": True,
            },
            "next_requirement": (
                "Add solve-time direct PSPG pressure-gradient support/coupling "
                "provenance that does not use pressure-update signs."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_solve_time_provenance_support_20260607.json",
        {
            "finding": (
                "solve_time_direct_pspg_support_coupling_provenance_ready"
            ),
            "status": "diagnostic_ready_replay_pending",
            "features": {
                "env_flag_present": True,
                "operator_filter_env_present": True,
                "source_component_filter_env_present": True,
                "emits_pressure_pressure_block": True,
                "emits_pressure_velocity_block": True,
                "emits_sampled_column_payload": True,
                "uses_bounded_column_sample": True,
                "records_sample_order_and_diag_membership": True,
                "records_pressure_update_sign_not_used": True,
                "called_before_topology_policy": True,
                "does_not_mutate_matrix": True,
            },
            "diagnostic_env": {
                "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_PROVENANCE_DIAGNOSTIC": "1",
                "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR": (
                    "equations"
                ),
                "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_SOURCE_COMPONENT": (
                    "navier_stokes_vms_pspg_pressure_gradient"
                ),
            },
            "next_requirement": (
                "Run short Test02/Test10 replay windows with the direct PSPG "
                "support/coupling provenance diagnostic."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_solve_time_provenance_replay_20260607.json",
        {
            "finding": (
                "solve_time_direct_pspg_support_coupling_replay_rules_out_simple_pp_pv_gate"
            ),
            "status": "replay_evidence_supports_coupling_split_no_selector",
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "solve_time_provenance_covers_targets_but_simple_pp_pv_selectors_fail"
                    ),
                    "record_count": 26864,
                    "target_rows_present_count": 7,
                    "max_target_ratio_rows": [10676],
                    "zero_pressure_velocity_target_global_dofs": [],
                },
                {
                    "label": "test10",
                    "finding": (
                        "solve_time_provenance_target_family_splits_zero_and_nonzero_coupling"
                    ),
                    "record_count": 5760,
                    "target_rows_present_count": 12,
                    "max_target_ratio_rows": [3450, 3919],
                    "zero_pressure_velocity_target_global_dofs": [
                        3526,
                        3456,
                        3925,
                    ],
                },
            ],
            "next_requirement": (
                "Derive a formulation-side topology/coupling rule from this "
                "solve-time provenance."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_aggregate_feature_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_aggregate_feature_selectivity_"
                "rules_out_counts_and_volume_gate"
            ),
            "status": "aggregate_counts_and_volume_features_overbroad",
            "features": [
                "pressure_pressure_records",
                "pressure_velocity_records",
                "pressure_pressure_edge_count",
                "pressure_pressure_two_hop_completion_count",
                "pressure_pressure_neighbor_pair_count",
                "pressure_velocity_nonzero_count",
                "min_volume_fraction",
                "full_cell_records",
                "cut_cell_records",
                "rule_count",
            ],
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "solve_time_aggregate_feature_selectors_overbroad_or_"
                        "miss_targets"
                    ),
                    "best_covering_exact_value_selector": {
                        "key": "full_cell_records_exact_target_value_set",
                        "selected_count": 214,
                        "selected_to_target_ratio": 214 / 7,
                    },
                    "best_covering_range_selector": {
                        "key": "cut_cell_records_target_range",
                        "selected_count": 427,
                        "selected_to_target_ratio": 61.0,
                    },
                },
                {
                    "label": "test10",
                    "finding": (
                        "solve_time_aggregate_feature_selectors_overbroad_or_"
                        "miss_targets"
                    ),
                    "best_covering_exact_value_selector": {
                        "key": "full_cell_records_exact_target_value_set",
                        "selected_count": 109,
                        "selected_to_target_ratio": 109 / 12,
                    },
                    "best_covering_range_selector": {
                        "key": "full_cell_records_target_range",
                        "selected_count": 111,
                        "selected_to_target_ratio": 111 / 12,
                    },
                },
            ],
            "next_requirement": (
                "Continue the direct PSPG formulation search with a physical "
                "support/coupling discriminator beyond aggregate provenance "
                "counts, full-cell classes, and target value ranges."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_support_measure_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_support_measure_selectivity_rules_out_"
                "qpoint_measure_gate"
            ),
            "status": "active_qpoint_and_measure_features_overbroad",
            "features": [
                "min_active_quadrature_points",
                "max_active_quadrature_points",
                "active_quadrature_point_values",
                "min_active_quadrature_fraction",
                "max_active_quadrature_fraction",
                "active_quadrature_fraction_values",
                "measure_values",
                "measure_fraction_values",
                "parent_measure_values",
                "rule_quadrature_point_values",
            ],
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "solve_time_support_measure_selectors_overbroad_or_"
                        "miss_targets"
                    ),
                    "best_covering_exact_value_selector": {
                        "key": (
                            "active_quadrature_fraction_values_exact_target_"
                            "value_set"
                        ),
                        "selected_count": 427,
                        "selected_to_target_ratio": 61.0,
                    },
                    "best_covering_range_selector": {
                        "key": "active_quadrature_fraction_values_target_range",
                        "selected_count": 427,
                        "selected_to_target_ratio": 61.0,
                    },
                },
                {
                    "label": "test10",
                    "finding": (
                        "solve_time_support_measure_selectors_overbroad_or_"
                        "miss_targets"
                    ),
                    "best_covering_exact_value_selector": {
                        "key": (
                            "active_quadrature_fraction_values_exact_target_"
                            "value_set"
                        ),
                        "selected_count": 126,
                        "selected_to_target_ratio": 10.5,
                    },
                    "best_covering_range_selector": {
                        "key": "active_quadrature_fraction_values_target_range",
                        "selected_count": 126,
                        "selected_to_target_ratio": 10.5,
                    },
                },
            ],
            "next_requirement": (
                "Continue the direct PSPG formulation search with a physical "
                "support/coupling discriminator beyond active quadrature count, "
                "generated measure, full-cell support, and target value ranges."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_parent_rule_component_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_parent_rule_components_rule_out_"
                "connected_cosupport_closure"
            ),
            "status": "parent_rule_component_closure_overbroad",
            "graph_modes": [
                "parent_cell",
                "rule_index",
                "parent_or_rule",
                "parent_rule_local_index",
            ],
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "solve_time_parent_rule_components_overbroad_or_"
                        "miss_targets"
                    ),
                    "component_counts": {
                        "parent_cell": 1,
                        "rule_index": 1,
                        "parent_or_rule": 1,
                        "parent_rule_local_index": 1,
                    },
                    "target_component_sizes": {
                        "parent_cell": [880],
                        "rule_index": [880],
                        "parent_or_rule": [880],
                        "parent_rule_local_index": [880],
                    },
                    "best_covering_component_selector": {
                        "key": "parent_cell_target_component_union",
                        "selected_count": 880,
                        "selected_to_target_ratio": 880 / 7,
                    },
                },
                {
                    "label": "test10",
                    "finding": (
                        "solve_time_parent_rule_components_overbroad_or_"
                        "miss_targets"
                    ),
                    "component_counts": {
                        "parent_cell": 1,
                        "rule_index": 1,
                        "parent_or_rule": 1,
                        "parent_rule_local_index": 1,
                    },
                    "target_component_sizes": {
                        "parent_cell": [252],
                        "rule_index": [252],
                        "parent_or_rule": [252],
                        "parent_rule_local_index": [252],
                    },
                    "best_covering_component_selector": {
                        "key": "parent_cell_target_component_union",
                        "selected_count": 252,
                        "selected_to_target_ratio": 21.0,
                    },
                },
            ],
            "next_requirement": (
                "Continue the direct PSPG formulation search with a physical "
                "support/coupling rule beyond raw connected parent/rule "
                "co-support closure or exact row/parent replay of current "
                "local matrix deltas."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_sampled_column_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_sampled_column_selectors_not_formulation_ready"
            ),
            "status": "sampled_column_stencil_gate_ruled_out",
            "cases": [
                {
                    "label": "test02",
                    "finding": "sampled_column_selectors_overbroad_or_miss_targets",
                    "record_count": 26864,
                    "target_rows_present_count": 7,
                    "any_sample_truncated": False,
                },
                {
                    "label": "test10",
                    "finding": "sampled_column_selectors_overbroad_or_miss_targets",
                    "record_count": 5760,
                    "target_rows_present_count": 12,
                    "any_sample_truncated": False,
                },
            ],
            "next_requirement": (
                "Move from sampled local stencil classes to a formulation-derived "
                "direct PSPG pressure-gradient support/coupling rule."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_same_rule_cross_block_"
            "signature_20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_same_rule_cross_block_signature_"
                "magnitude_candidate_found"
            ),
            "status": "same_rule_cross_block_candidate_requires_replay",
            "features": [
                "pressure_pressure_abs_sum",
                "pressure_velocity_abs_sum",
                "pressure_velocity_to_pressure_pressure_abs_ratio",
                "pressure_pressure_abs_sum_per_parent_cell",
                "pressure_velocity_abs_sum_per_parent_cell",
            ],
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "same_rule_cross_block_signature_magnitude_candidate"
                    ),
                    "shape_pair_selector": {
                        "selected_count": 879,
                        "selected_to_target_ratio": 879 / 7,
                    },
                    "base_same_rule_signature_selector": {
                        "selected_count": 56,
                        "selected_to_target_ratio": 8.0,
                    },
                    "best_covering_composite_selector": {
                        "key": (
                            "same_rule_cross_block_signature_with_"
                            "pressure_velocity_abs_sum_range"
                        ),
                        "feature": "pressure_velocity_abs_sum",
                        "selected_count": 20,
                        "selected_to_target_ratio": 20 / 7,
                    },
                    "best_covering_composite_selected_global_dofs": [
                        10658,
                        10659,
                        10676,
                        10945,
                        10946,
                        10947,
                        10952,
                        10953,
                        10954,
                        10959,
                        10972,
                        11946,
                        11947,
                        12204,
                        12205,
                        12206,
                        12211,
                        12212,
                        12213,
                        12218,
                    ],
                },
                {
                    "label": "test10",
                    "finding": (
                        "same_rule_cross_block_signature_magnitude_candidate"
                    ),
                    "shape_pair_selector": {
                        "selected_count": 252,
                        "selected_to_target_ratio": 21.0,
                    },
                    "base_same_rule_signature_selector": {
                        "selected_count": 60,
                        "selected_to_target_ratio": 5.0,
                    },
                    "best_covering_composite_selector": {
                        "key": (
                            "same_rule_cross_block_signature_with_"
                            "pressure_velocity_abs_sum_range"
                        ),
                        "feature": "pressure_velocity_abs_sum",
                        "selected_count": 21,
                        "selected_to_target_ratio": 1.75,
                    },
                    "best_covering_composite_selected_global_dofs": [
                        3450,
                        3451,
                        3454,
                        3455,
                        3456,
                        3458,
                        3459,
                        3466,
                        3525,
                        3526,
                        3529,
                        3822,
                        3824,
                        3919,
                        3920,
                        3923,
                        3924,
                        3925,
                        3927,
                        3928,
                        3935,
                    ],
                },
            ],
            "next_requirement": (
                "Run a targeted Test02/Test10 row-filter replay for the "
                "exported same-rule cross-block candidate rows, or derive a "
                "formulation-side support/coupling rule that reproduces the "
                "same row family without target-fitted magnitude ranges."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_row_filter_"
            "replays_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_row_filter_replays_do_not_"
                "clear_guards"
            ),
            "status": "same_rule_cross_block_replay_insufficient",
            "row_filters_match_candidate_counts": True,
            "all_replays_improve_no_policy_baseline": True,
            "all_replays_trigger_guard": True,
            "triggered_cases": ["test02", "test10"],
            "cases": [
                {
                    "label": "test02",
                    "expected_candidate_row_count": 20,
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 357449.7849043233,
                        "worst_active_or_wet_support_class": (
                            "full_wet_supported"
                        ),
                    },
                    "improvement_vs_baseline_pa": 9270.180902128108,
                    "replay_to_baseline_update_ratio": 0.9747213629840358,
                    "replay_to_broad_policy_update_ratio": 2.021269323569583,
                    "topology_log": {
                        "matrix_mutated_count": 300,
                    },
                },
                {
                    "label": "test10",
                    "expected_candidate_row_count": 21,
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 582.6183066757754,
                        "worst_active_or_wet_support_class": (
                            "full_wet_supported"
                        ),
                    },
                    "improvement_vs_baseline_pa": 39.99110318614055,
                    "replay_to_baseline_update_ratio": 0.9357685532009388,
                    "replay_to_broad_policy_update_ratio": 1.1152355333088089,
                    "topology_log": {
                        "matrix_mutated_count": 86,
                    },
                },
            ],
            "next_requirement": (
                "Do not promote the same-rule row list as a fix. Use the replay "
                "result to derive a formulation-side rule that keeps the helpful "
                "same-rule PP/PV coupling signal while adding the missing broader "
                "support/coupling mechanism, or test the next physical candidate "
                "against the same short-window guards."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_"
            "scope_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_parent_cell_scope_ready_"
                "for_replay"
            ),
            "status": "run_same_rule_cross_block_parent_cell_replay",
            "all_cases_ready_for_parent_cell_replay": True,
            "cases": [
                {
                    "label": "test02",
                    "candidate_row_count": 20,
                    "parent_cell_count": 360,
                    "parent_expanded_row_count": 157,
                    "parent_expanded_to_candidate_ratio": 7.85,
                    "ready_for_parent_cell_replay": True,
                },
                {
                    "label": "test10",
                    "candidate_row_count": 21,
                    "parent_cell_count": 86,
                    "parent_expanded_row_count": 57,
                    "parent_expanded_to_candidate_ratio": 57 / 21,
                    "ready_for_parent_cell_replay": True,
                },
            ],
            "next_requirement": (
                "Run local_schur_edge_balance with the derived parent-cell "
                "filters and no global row filter; compare against the "
                "row-list replay and same-case no-policy baseline."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_"
            "replays_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_parent_cell_replays_do_"
                "not_clear_guards"
            ),
            "status": "same_rule_cross_block_parent_cell_replay_insufficient",
            "parent_filters_match_scope_counts": True,
            "row_filters_disabled": True,
            "all_replays_improve_no_policy_baseline": True,
            "all_replays_improve_row_filter_replay": True,
            "all_replays_trigger_guard": True,
            "triggered_cases": ["test02", "test10"],
            "cases": [
                {
                    "label": "test02",
                    "parent_scope": {
                        "expected_parent_cell_count": 360,
                    },
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 321290.80382374703,
                    },
                    "improvement_vs_baseline_pa": 45429.16198270436,
                    "improvement_vs_row_filter_pa": 36158.98108057625,
                    "replay_to_broad_policy_update_ratio": 1.8168013330537556,
                    "topology_log": {
                        "matrix_mutated_count": 360,
                    },
                },
                {
                    "label": "test10",
                    "parent_scope": {
                        "expected_parent_cell_count": 86,
                    },
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 570.6844972203451,
                    },
                    "improvement_vs_baseline_pa": 51.92491264157093,
                    "improvement_vs_row_filter_pa": 11.933809455430378,
                    "replay_to_broad_policy_update_ratio": 1.0923920898400148,
                    "topology_log": {
                        "matrix_mutated_count": 86,
                    },
                },
            ],
            "next_requirement": (
                "Do not promote parent-cell replay as a fix. Use the "
                "improvement over the row-list replay as evidence that some "
                "broader local support coupling is relevant."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_"
            "parent_cell_scope_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_"
                "ready_for_replay"
            ),
            "status": "run_broad_minus_same_rule_parent_cell_replay",
            "all_cases_ready_for_broad_minus_parent_cell_replay": True,
            "cases": [
                {
                    "label": "test02",
                    "broad_parent_cell_count": 3352,
                    "same_rule_parent_cell_count": 360,
                    "broad_only_parent_cell_count": 2992,
                    "broad_only_to_broad_parent_ratio": 2992 / 3352,
                },
                {
                    "label": "test10",
                    "broad_parent_cell_count": 720,
                    "same_rule_parent_cell_count": 86,
                    "broad_only_parent_cell_count": 634,
                    "broad_only_to_broad_parent_ratio": 634 / 720,
                },
            ],
            "next_requirement": (
                "Run local_schur_edge_balance with the broad-minus parent-cell "
                "filters and no global row filter."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_"
            "parent_cell_replays_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_"
                "do_not_clear_guards"
            ),
            "status": "broad_minus_parent_replay_insufficient",
            "parent_filters_match_scope_counts": True,
            "row_filters_disabled": True,
            "all_replays_trigger_guard": True,
            "broad_policy_better_than_isolated_parts": True,
            "complement_worse_than_same_rule_parent_cell": True,
            "triggered_cases": ["test02", "test10"],
            "cases": [
                {
                    "label": "test02",
                    "broad_minus_scope": {
                        "expected_parent_cell_count": 2992,
                    },
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 366324.79523179174,
                    },
                    "improvement_vs_baseline_pa": 395.17057465965627,
                    "improvement_vs_same_rule_parent_cell_pa": (
                        -45033.991408044705
                    ),
                    "replay_to_broad_policy_update_ratio": 2.0714547954284535,
                    "topology_log": {
                        "matrix_mutated_count": 2992,
                    },
                },
                {
                    "label": "test10",
                    "broad_minus_scope": {
                        "expected_parent_cell_count": 634,
                    },
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 575.8357642247117,
                    },
                    "improvement_vs_baseline_pa": 46.773645637204254,
                    "improvement_vs_same_rule_parent_cell_pa": (
                        -5.151267004366673
                    ),
                    "replay_to_broad_policy_update_ratio": 1.1022525352448447,
                    "topology_log": {
                        "matrix_mutated_count": 634,
                    },
                },
            ],
            "next_requirement": (
                "Do not promote same-rule parent cells or the broad-only "
                "complement as a fix."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_union_"
            "branch_shift_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_same_rule_cross_block_broad_union_consistent_"
                "replays_do_not_clear_guards"
            ),
            "status": "broad_union_consistent_replay_insufficient",
            "case_findings": {
                "test02": (
                    "broad_union_reduces_full_wet_reference_but_guard_remains"
                ),
                "test10": (
                    "broad_union_reduces_shared_full_wet_reference_but_guard_"
                    "remains"
                ),
            },
            "all_variants_guard_triggered": True,
            "test02_branch_shift_supported": False,
            "test02_consistent_full_wet_residual_supported": True,
            "test10_broad_union_residual_guard_supported": True,
            "cases": [
                {
                    "label": "test02",
                    "reference_point": 1172,
                    "flags": {
                        "broad_reference_abs_pressure_delta_pa": (
                            321110.9963650234
                        ),
                        "isolated_reference_abs_pressure_delta_pa": {
                            "no_policy": 366719.9658064514,
                            "same_rule_parent": 321290.80382374703,
                            "broad_minus_parent": 366324.79523179174,
                        },
                        "broad_reference_improvement_vs_isolated_pa": {
                            "no_policy": 45608.969441427966,
                            "same_rule_parent": 179.8074587236042,
                            "broad_minus_parent": 45213.79886676831,
                        },
                        "broad_policy_worst_point": 1172,
                        "broad_policy_worst_support_class": "full_wet_supported",
                        "broad_policy_clears_reference_point_guard": False,
                        "broad_policy_guard_triggered": True,
                    },
                },
                {
                    "label": "test10",
                    "reference_point": 83,
                    "flags": {
                        "broad_reference_abs_pressure_delta_pa": (
                            522.4172735486616
                        ),
                        "isolated_reference_abs_pressure_delta_pa": {
                            "no_policy": 622.609409861916,
                            "same_rule_parent": 570.6844972203451,
                            "broad_minus_parent": 575.8357642247117,
                        },
                        "broad_reference_improvement_vs_isolated_pa": {
                            "no_policy": 100.19213631325442,
                            "same_rule_parent": 48.267223671683496,
                            "broad_minus_parent": 53.41849067605017,
                        },
                        "broad_policy_worst_point": 83,
                        "broad_policy_worst_support_class": "full_wet_supported",
                        "broad_policy_clears_reference_point_guard": False,
                        "broad_policy_guard_triggered": True,
                    },
                },
            ],
            "next_requirement": (
                "The next formulation candidate should preserve broad-union "
                "support/coupling improvement at the full-wet branch, but must "
                "reduce the residual full-wet Test02/Test10 updates below the "
                "guards."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_support_coupling_signature_"
            "20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_support_coupling_signature_partial_"
                "test10_only"
            ),
            "status": "test10_signature_candidate_test02_overbroad",
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "solve_time_support_coupling_signature_covers_targets_"
                        "but_overbroad"
                    ),
                    "target_same_parent_pressure_velocity_support_class_counts": {
                        "none": 0,
                        "partial": 0,
                        "full": 7,
                    },
                    "exact_local_signature_selected_count": 276,
                    "exact_local_signature_selected_to_target_ratio": (
                        276 / 7
                    ),
                },
                {
                    "label": "test10",
                    "finding": (
                        "solve_time_support_coupling_signature_selective_"
                        "candidate"
                    ),
                    "target_same_parent_pressure_velocity_support_class_counts": {
                        "none": 3,
                        "partial": 3,
                        "full": 6,
                    },
                    "exact_local_signature_selected_count": 48,
                    "exact_local_signature_selected_to_target_ratio": 4.0,
                },
            ],
            "next_requirement": (
                "Either add a solve-time aggregation API for a targeted Test10 "
                "support/coupling-signature replay, or find an additional "
                "Test02 physical discriminator."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_magnitude_selectivity_"
            "20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_support_coupling_magnitude_"
                "selectors_not_formulation_ready"
            ),
            "status": "range_thresholds_overbroad_exact_value_oracles_only",
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "exact_magnitude_value_oracles_only_range_"
                        "selectors_broad"
                    ),
                    "range_selector_selected_to_target_ratios": {
                        "pressure_pressure_abs_sum_target_range": 84.0,
                        "pressure_velocity_abs_sum_target_range": (
                            107 / 7
                        ),
                        (
                            "pressure_velocity_to_pressure_pressure_abs_ratio_"
                            "target_range"
                        ): 338 / 7,
                    },
                    "exact_value_oracle_selector_keys": [
                        "pressure_velocity_abs_sum_exact_target_value_set",
                        (
                            "pressure_velocity_to_pressure_pressure_abs_ratio_"
                            "exact_target_value_set"
                        ),
                    ],
                },
                {
                    "label": "test10",
                    "finding": (
                        "exact_magnitude_value_oracles_only_range_"
                        "selectors_broad"
                    ),
                    "range_selector_selected_to_target_ratios": {
                        "pressure_pressure_abs_sum_target_range": 134 / 12,
                        "pressure_velocity_abs_sum_target_range": 89 / 12,
                        (
                            "pressure_velocity_to_pressure_pressure_abs_ratio_"
                            "target_range"
                        ): 9.0,
                    },
                    "exact_value_oracle_selector_keys": [
                        "pressure_velocity_abs_sum_exact_target_value_set",
                    ],
                },
            ],
            "next_requirement": (
                "Do not promote exact local-matrix magnitude equality as a "
                "support/coupling rule."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_solve_time_signature_magnitude_"
            "composite_20260607.json"
        ),
        {
            "finding": (
                "solve_time_direct_pspg_signature_magnitude_composite_partial_"
                "test10_only"
            ),
            "status": "test10_composite_candidate_test02_overbroad",
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "solve_time_signature_magnitude_composite_covers_"
                        "targets_but_overbroad"
                    ),
                    "best_covering_composite_selected_count": 53,
                    "best_covering_composite_selected_to_target_ratio": 53 / 7,
                },
                {
                    "label": "test10",
                    "finding": (
                        "solve_time_signature_magnitude_composite_selective_"
                        "candidate"
                    ),
                    "best_covering_composite_selected_count": 22,
                    "best_covering_composite_selected_to_target_ratio": 22 / 12,
                },
            ],
            "next_requirement": (
                "Continue the direct PSPG formulation search with a stronger "
                "Test02 physical discriminator."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_test10_signature_replay_readiness_"
            "20260607.json"
        ),
        {
            "finding": (
                "test10_signature_replay_candidate_ready_for_solve_time_"
                "replay"
            ),
            "status": "run_targeted_test10_signature_replay",
            "cases": [
                {
                    "label": "test02",
                    "exact_local_signature_selector": {
                        "finding": "selector_overbroad",
                        "selected_count": 276,
                        "selected_to_target_ratio": 276 / 7,
                    },
                    "signature_candidate_global_dofs": [10676, 10952],
                },
                {
                    "label": "test10",
                    "exact_local_signature_selector": {
                        "finding": "selector_selective",
                        "selected_count": 48,
                        "selected_to_target_ratio": 4.0,
                    },
                    "signature_candidate_global_dofs": [
                        3277,
                        3278,
                        3450,
                        3451,
                        3525,
                        3526,
                        3919,
                        3920,
                        3925,
                    ],
                },
            ],
            "hook_summary": {
                "fe_topology_signature_or_row_selector_present": True,
                "post_assembly_explicit_row_path_present": True,
            },
            "next_requirement": (
                "Run a targeted Test10 replay with the exported signature "
                "candidate rows through the solve-time direct PSPG topology "
                "row filter."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
        {
            "status": "diagnostic_pressure_update_guard_triggered",
            "finding": (
                "1 transition(s) exceeded 100 Pa on active/wet support. "
                "Worst active/wet update was 604.713 Pa from step 90 to 1 "
                "on full_wet_supported."
            ),
            "absolute_threshold_pa": 100.0,
            "triggered_transition_count": 1,
            "worst_by_category": {
                "active_or_wet_supported": {
                    "abs_pressure_delta_pa": 604.7126561932914,
                    "point_index": 83,
                    "support_class": "full_wet_supported",
                    "active_fluid": 1.0,
                    "incident_wet_fraction_min_positive": 1.0,
                },
            },
        },
    )
    _write_json(
        tmp_path,
        "test10_direct_pspg_signature_row_filter_replays_20260607.json",
        {
            "finding": "test10_signature_row_filter_local_modes_do_not_clear_guard",
            "status": "signature_row_filter_local_modes_ruled_out_as_sufficient_fix",
            "policies_tested": [
                "local_schur_completion",
                "local_edge_balance",
                "local_schur_edge_balance",
            ],
            "row_filter_global_dof_counts": [48],
            "all_replays_trigger_guard": True,
            "best_policy_by_worst_update": "local_schur_edge_balance",
            "best_worst_active_or_wet_update_pa": 604.7126561932914,
            "replays": [
                {
                    "policy": "local_schur_completion",
                    "worst_active_or_wet_update_pa": 619.6167550623924,
                    "topology_log": {"row_filter_log_count": 264},
                },
                {
                    "policy": "local_edge_balance",
                    "worst_active_or_wet_update_pa": 607.5173052131886,
                    "topology_log": {"row_filter_log_count": 264},
                },
                {
                    "policy": "local_schur_edge_balance",
                    "worst_active_or_wet_update_pa": 604.7126561932914,
                    "topology_log": {"row_filter_log_count": 264},
                },
            ],
            "next_requirement": (
                "Do not promote exact signature-row local topology replay as "
                "the formulation fix."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_ghost_branch_signature_interaction_"
            "20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_ghost_branch_signature_interaction_rules_out_"
                "common_gate"
            ),
            "status": (
                "ghost_branch_is_branch_shaper_not_support_coupling_"
                "signature_fix"
            ),
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "ghost_branch_shapes_test02_but_cannot_narrow_"
                        "signature"
                    ),
                    "baseline_ghost_penalty_global_dofs": [10624],
                    "pressure_disabled_ghost_penalty_global_dofs": [],
                    "row_10676_baseline_update_pa": 366719.9658064514,
                    "row_10676_pressure_disabled_update_pa": (
                        1298098.542745239
                    ),
                    "signature": {
                        "exact_local_signature_selected_to_target_ratio": (
                            276 / 7
                        )
                    },
                    "branch_policy": {
                        "pressure_disabled_still_triggers": True,
                    },
                },
                {
                    "label": "test10",
                    "finding": (
                        "ghost_absent_test10_signature_candidate_remains_"
                        "partial_fix"
                    ),
                    "baseline_ghost_penalty_global_dofs": [],
                    "pressure_disabled_ghost_penalty_global_dofs": [],
                    "signature": {
                        "exact_local_signature_selected_to_target_ratio": 4.0,
                    },
                    "branch_policy": {
                        "pressure_disabled_still_triggers": True,
                    },
                },
            ],
            "next_requirement": (
                "Do not use ghost-positive branch membership to narrow the "
                "Test02 support/coupling signature."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_active_pressure_support_selectivity_20260607.json",
        {
            "finding": (
                "active_pressure_support_topology_selectors_overbroad_or_miss_targets"
            ),
            "selectors": [
                {
                    "key": "constrained_pressure_neighbor",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                    ],
                },
                {
                    "key": "high_constrained_pressure_neighbor_ratio",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                    ],
                },
                {
                    "key": "sparse_unconstrained_direct_self",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 545,
                            "covered_direct_target_count": 1,
                            "selected_to_target_ratio": 545 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 217,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 217 / 12,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_residual_sign_selectivity_20260607.json",
        {
            "finding": (
                "residual_sign_pressure_action_selectors_overbroad_or_miss_targets"
            ),
            "residual_signal_by_case": {
                "test02": {
                    "residual_nonzero_direct_row_count": 866,
                    "residual_sign_pressure_action_edge_count": 1218,
                },
                "test10": {
                    "residual_nonzero_direct_row_count": 251,
                    "residual_sign_pressure_action_edge_count": 354,
                },
            },
            "selectors": [
                {
                    "key": "residual_sign_pressure_action",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 500,
                            "covered_direct_target_count": 6,
                            "selected_to_target_ratio": 500 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
                {
                    "key": (
                        "sparse_seeded_residual_sign_pressure_action_component"
                    ),
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 500,
                            "covered_direct_target_count": 6,
                            "selected_to_target_ratio": 500 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
                {
                    "key": (
                        "sparse_direct_self_or_residual_sign_pressure_action"
                    ),
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 545,
                            "covered_direct_target_count": 1,
                            "selected_to_target_ratio": 545 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_null_balance_selectivity_20260607.json",
        {
            "finding": (
                "direct_pspg_null_balance_selectors_overbroad_or_miss_targets"
            ),
            "null_balance_by_case": {
                "test02": {
                    "max_direct_self_row_sum_leak_ratio": 0.03,
                    "min_direct_self_diag_abs_ratio": 0.5,
                    "max_direct_self_diag_abs_ratio": 0.5,
                },
                "test10": {
                    "max_direct_self_row_sum_leak_ratio": 0.12,
                    "min_direct_self_diag_abs_ratio": 0.5,
                    "max_direct_self_diag_abs_ratio": 0.56,
                },
            },
            "selectors": [
                {
                    "key": "high_direct_self_row_sum_leak",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                    ],
                },
                {
                    "key": "null_preserving_direct_self",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 866,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 866 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
                {
                    "key": "balanced_diag_direct_self",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 866,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 866 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_coupled_patch_graph_selectivity_20260607.json",
        {
            "finding": (
                "direct_pspg_coupled_patch_graph_selectors_overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "pressure_action_high_two_hop",
                "pressure_action_zero_clustering",
                "pressure_action_low_clustering",
            ],
            "miss_selector_keys": [
                "pressure_action_zero_two_hop",
                "pressure_action_low_two_hop",
                "pressure_action_high_two_hop",
                "pressure_action_high_clustering",
                "pressure_action_articulation",
                "pressure_action_bridge_endpoint",
            ],
            "graph_topology_by_case": {
                "test02": {
                    "matrix_pressure_action_max_two_hop_completion_count": 18,
                    "matrix_pressure_action_max_clustering_ratio": 0.0,
                },
                "test10": {
                    "matrix_pressure_action_max_two_hop_completion_count": 15,
                    "matrix_pressure_action_max_clustering_ratio": 0.0,
                },
            },
            "selectors": [
                {
                    "key": "pressure_action_low_two_hop",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 4,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
                {
                    "key": "pressure_action_high_two_hop",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 796,
                            "covered_direct_target_count": 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 239,
                            "covered_direct_target_count": 9,
                        },
                    ],
                },
                {
                    "key": "pressure_action_zero_clustering",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 864,
                            "covered_direct_target_count": 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                        },
                    ],
                },
                {
                    "key": "pressure_action_articulation",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 2,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
                {
                    "key": "pressure_action_bridge_endpoint",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 4,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_pressure_operator_top_update_overlap_20260606.json",
        {
            "finding": "mixed_no_galerkin_overlap_partial_for_some_cases_absent_for_others",
            "no_galerkin_support_finding": (
                "no_galerkin_support_rank_selector_differs_in_some_cases"
            ),
            "exact_to_aggregate_sample_finding": (
                "exact_direct_pspg_top_rows_undercovered_by_aggregate_samples"
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_no_galerkin_gate_relevance_20260607.json",
        {
            "finding": (
                "no_galerkin_nonpressure_gate_ruled_out_as_complete_"
                "formulation_gate"
            ),
            "status": "partial_test10_signal_ruled_out_as_complete_gate",
            "classification": {
                "overlap_missing_cases": ["test02"],
                "overlap_partial_cases": ["test10"],
                "candidate_uncovered_cases": ["test02"],
                "support_rank_mismatch_cases": ["test10"],
                "complete_gate_candidate": False,
            },
            "top_overlap": {
                "finding": (
                    "mixed_no_galerkin_overlap_partial_for_some_cases_absent_"
                    "for_others"
                ),
                "no_galerkin_support_finding": (
                    "no_galerkin_support_rank_selector_differs_in_some_cases"
                ),
                "cases": [
                    {
                        "label": "test02",
                        "direct_target_count": 7,
                        "no_galerkin_top_update_overlap_count": 0,
                        "no_galerkin_top_update_overlap_ratio": 0.0,
                    },
                    {
                        "label": "test10",
                        "direct_target_count": 12,
                        "no_galerkin_top_update_overlap_count": 3,
                        "no_galerkin_top_update_overlap_ratio": 0.25,
                    },
                ],
            },
            "formulation_candidate": {
                "key": (
                    "zero_galerkin_nonpressure_or_same_sign_pressure_action_"
                    "patch"
                ),
                "finding": "partial_audited_coverage",
                "covers_all_audited_targets": False,
                "cases": [
                    {
                        "label": "test02",
                        "direct_target_count": 7,
                        "selected_count": 6,
                        "uncovered_direct_target_global_dofs": [10676],
                    },
                    {
                        "label": "test10",
                        "direct_target_count": 12,
                        "selected_count": 12,
                        "uncovered_direct_target_global_dofs": [],
                    },
                ],
            },
            "next_requirement": (
                "Keep no-Galerkin/nonpressure zero coupling as a Test10 "
                "sub-signal, but derive the remaining gate from direct PSPG "
                "pressure-gradient support/coupling topology."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_graph_completion_shared_row_schur_low_degree_"
            "edge_balance_deg3_boundary_provenance_20260606.json"
        ),
        {
            "finding": "latest_bad_rows_can_be_candidates_without_balance_coverage",
            "boundary_topology_finding": (
                "boundary_top_update_candidates_missing_balance"
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_graph_completion_shared_row_schur_coupling_edge_balance_20260606_outcome.json",
        {
            "test10_step90": {"outcome": "accepted_guard_triggered"},
            "test02_step382": {"outcome": "nonlinear_failed"},
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_graph_completion_shared_row_schur_low_degree_"
            "edge_balance_deg3_20260606_outcome.json"
        ),
        {
            "test10_step90": {"outcome": "accepted_guard_triggered"},
            "test02_step382": {"outcome": "nonlinear_failed"},
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_graph_completion_support_gap_patch_20260606_outcome.json",
        {
            "finding": (
                "support_gap_patch_selector_reproduces_test10_all_row_clearance_"
                "but_expands_to_all_pressure_rows_and_fails_test02_nonlinear_"
                "convergence"
            ),
            "test10_step90": {
                "outcome": "accepted_guard_not_triggered",
                "accepted_pressure_update_pa": 6.7753020523015266,
            },
            "test02_step382": {
                "outcome": "nonlinear_failed",
                "final_residual_norm": 29088.54648995749,
            },
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_row_provenance_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_row_provenance_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "cut_volume_profiled_candidate",
                "cut_volume_full_cell_only_support",
            ],
            "miss_selector_keys": [
                "cut_volume_partial_rule_support",
                "cut_volume_low_min_fraction",
            ],
            "cases": [
                {
                    "label": "test02",
                    "profile_summary": {
                        "candidate_support_class_counts": {
                            "full_cell_only_support": 426,
                            "mixed_partial_and_full_cell_support": 210,
                            "partial_cut_only_support": 230,
                        },
                        "target_profiles": {
                            "10676": {
                                "cut_volume_support_class": (
                                    "full_cell_only_support"
                                ),
                                "min_volume_fraction": 1.0,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "profile_summary": {
                        "candidate_support_class_counts": {
                            "full_cell_only_support": 125,
                            "mixed_partial_and_full_cell_support": 64,
                            "partial_cut_only_support": 62,
                        },
                        "target_profiles": {
                            "3526": {
                                "cut_volume_support_class": (
                                    "full_cell_only_support"
                                ),
                                "min_volume_fraction": 1.0,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "cut_volume_full_cell_only_support",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 426,
                            "covered_direct_target_count": 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 125,
                            "covered_direct_target_count": 12,
                        },
                    ],
                },
                {
                    "key": "cut_volume_partial_rule_support",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 440,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 126,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_local_matrix_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_local_matrix_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "local_matrix_profiled_candidate",
                "local_matrix_low_total_abs_sum_p25",
                "local_matrix_full_cell_dominant_abs_fraction",
            ],
            "miss_selector_keys": [
                "local_matrix_low_total_abs_sum_p10",
                "local_matrix_low_parent_cell_support",
            ],
            "cases": [
                {
                    "label": "test02",
                    "thresholds": {
                        "total_row_abs_sum_p25": 7.885742e-10,
                        "max_rule_row_abs_fraction_p75": 0.21238702821778546,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "10676": {
                                "total_row_abs_sum": 4.40018e-10,
                                "max_rule_row_abs_fraction": 0.255944075015113,
                                "diag_abs_fraction": 0.5000009090537206,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "thresholds": {
                        "total_row_abs_sum_p25": 1.72991e-08,
                        "max_rule_row_abs_fraction_p75": 0.2087948233093213,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "3526": {
                                "total_row_abs_sum": 1.720504e-08,
                                "max_rule_row_abs_fraction": 0.2950280847937581,
                                "diag_abs_fraction": 0.5,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "local_matrix_low_total_abs_sum_p25",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 217,
                            "covered_direct_target_count": 1,
                        },
                        {
                            "label": "test10",
                            "selected_count": 63,
                            "covered_direct_target_count": 3,
                        },
                    ],
                },
                {
                    "key": "local_matrix_full_cell_dominant_abs_fraction",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 426,
                            "covered_direct_target_count": 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 125,
                            "covered_direct_target_count": 12,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_local_coupling_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_local_coupling_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "cross_field_profiled_candidate",
                "cross_field_high_velocity_pressure_ratio_p90",
            ],
            "miss_selector_keys": [
                "cross_field_zero_velocity_action",
                "cross_field_low_velocity_pressure_ratio_p25",
            ],
            "cases": [
                {
                    "label": "test02",
                    "thresholds": {
                        "velocity_to_pressure_abs_ratio_p90": 1.0562530747823181e-05,
                        "velocity_total_row_abs_sum_p25": 1.1068422900499998e-15,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "10676": {
                                "pressure_total_row_abs_sum": 4.40018e-10,
                                "velocity_total_row_abs_sum": 2.332753e-13,
                                "velocity_to_pressure_abs_ratio": 0.0005301494484316551,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "thresholds": {
                        "velocity_to_pressure_abs_ratio_p90": 0.00016555671658536833,
                        "velocity_total_row_abs_sum_p25": 3.6093891910400004e-13,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "3526": {
                                "pressure_total_row_abs_sum": 1.720504e-08,
                                "velocity_total_row_abs_sum": 0.0,
                                "velocity_to_pressure_abs_ratio": 0.0,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "cross_field_zero_velocity_action",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 9,
                            "covered_direct_target_count": 3,
                        },
                    ],
                },
                {
                    "key": "cross_field_high_velocity_pressure_ratio_p90",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 88,
                            "covered_direct_target_count": 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 26,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_parent_graph_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_parent_graph_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "parent_graph_profiled_candidate",
                "parent_graph_degree_tail",
            ],
            "miss_selector_keys": [
                "parent_graph_low_degree_high_clustering",
                "parent_graph_high_degree_low_clustering",
            ],
            "cases": [
                {
                    "label": "test02",
                    "thresholds": {
                        "degree_p25": 10.0,
                        "degree_p75": 14.0,
                        "clustering_p25": 0.3956043956043956,
                        "clustering_p75": 0.4666666666666667,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "10676": {
                                "row_parent_graph_degree": 5,
                                "row_parent_graph_clustering": 0.8,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "thresholds": {
                        "degree_p25": 8.0,
                        "degree_p75": 10.0,
                        "clustering_p25": 0.42857142857142855,
                        "clustering_p75": 0.5357142857142857,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "3526": {
                                "row_parent_graph_degree": 6,
                                "row_parent_graph_clustering": 0.6,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "parent_graph_degree_tail",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 809,
                            "covered_direct_target_count": 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 247,
                            "covered_direct_target_count": 12,
                        },
                    ],
                },
                {
                    "key": "parent_graph_high_degree_low_clustering",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 309,
                            "covered_direct_target_count": 6,
                        },
                        {
                            "label": "test10",
                            "selected_count": 59,
                            "covered_direct_target_count": 0,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_composite_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_composite_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "composite_graph_bimodal_tail",
                "composite_isolated_or_high_ratio_coherent",
            ],
            "miss_selector_keys": [
                "composite_graph_tail_and_ratio_tail",
                "composite_twohop_graph_ratio_tail",
            ],
            "cases": [
                {
                    "label": "test02",
                    "thresholds": {
                        "degree_p25": 10.0,
                        "degree_p75": 14.0,
                        "velocity_ratio_p90": 1.0562530747823181e-05,
                        "pressure_abs_p75": 2.4224653e-09,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "10676": {
                                "row_parent_graph_degree": 5,
                                "row_parent_graph_clustering": 0.8,
                                "velocity_to_pressure_abs_ratio": 0.0005301494484316551,
                                "pressure_total_row_abs_sum": 4.40018e-10,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "thresholds": {
                        "degree_p25": 8.0,
                        "degree_p75": 10.0,
                        "velocity_ratio_p90": 0.00016555671658536833,
                        "pressure_abs_p75": 3.8852440000000004e-08,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "3526": {
                                "row_parent_graph_degree": 6,
                                "row_parent_graph_clustering": 0.6,
                                "velocity_to_pressure_abs_ratio": 0.0,
                                "pressure_total_row_abs_sum": 1.720504e-08,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "composite_graph_bimodal_tail",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 793,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 113.28571428571429,
                        },
                        {
                            "label": "test10",
                            "selected_count": 169,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 14.083333333333334,
                        },
                    ],
                },
                {
                    "key": "composite_graph_tail_and_ratio_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 172,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 24.571428571428573,
                        },
                        {
                            "label": "test10",
                            "selected_count": 51,
                            "covered_direct_target_count": 5,
                            "selected_to_target_ratio": 4.25,
                        },
                    ],
                },
                {
                    "key": "composite_twohop_graph_ratio_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 97,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 13.857142857142858,
                        },
                        {
                            "label": "test10",
                            "selected_count": 35,
                            "covered_direct_target_count": 5,
                            "selected_to_target_ratio": 2.9166666666666665,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_column_support_"
            "readiness_20260607.json"
        ),
        {
            "finding": "direct_pspg_cut_volume_column_support_evidence_ready",
            "missing_case_labels": [],
            "next_requirement": (
                "Use signed sampled column neighborhoods to derive a "
                "formulation-side direct PSPG pressure-gradient support rule "
                "instead of scalar row thresholds."
            ),
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {
                        "status": "ok",
                        "entry_count": 26864,
                        "batch_count": 2,
                        "latest_batch_entry_count": 13432,
                    },
                    "profile_summary": {
                        "profiled_candidate_count": 866,
                        "profiled_target_count": 7,
                        "unprofiled_target_global_dofs": [],
                        "candidate_column_support_class_counts": {
                            "null_preserving_negative_offdiag_stencil": 866,
                        },
                        "target_column_support_class_counts": {
                            "null_preserving_negative_offdiag_stencil": 7,
                        },
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {
                        "status": "ok",
                        "entry_count": 5760,
                        "batch_count": 2,
                        "latest_batch_entry_count": 2880,
                    },
                    "profile_summary": {
                        "profiled_candidate_count": 251,
                        "profiled_target_count": 12,
                        "unprofiled_target_global_dofs": [],
                        "candidate_column_support_class_counts": {
                            "null_preserving_negative_offdiag_stencil": 251,
                        },
                        "target_column_support_class_counts": {
                            "null_preserving_negative_offdiag_stencil": 12,
                        },
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_column_support_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_column_support_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "column_profiled_candidate",
                "column_null_preserving_negative_offdiag_class",
                "column_candidate_neighbor_closed",
                "column_all_candidate_edges_reciprocal",
                "column_candidate_degree_tail",
                "column_single_connected_component",
            ],
            "miss_selector_keys": [
                "column_low_candidate_degree_p25",
                "column_high_candidate_degree_p75",
                "column_two_hop_tail",
                "column_edge_concentration_tail",
                "column_mean_edge_abs_tail",
            ],
            "next_requirement": (
                "Move beyond coarse signed column topology and sampled edge "
                "magnitude tails toward element-local pressure-gradient geometry."
            ),
            "cases": [
                {
                    "label": "test02",
                    "thresholds": {
                        "candidate_degree_p25": 5.0,
                        "candidate_degree_p75": 6.0,
                        "component_size_p75": 866.0,
                        "two_hop_p25": 12.0,
                        "two_hop_p75": 16.0,
                        "edge_concentration_p25": 0.2658199127765325,
                        "edge_concentration_p75": 0.5399942640068196,
                        "mean_edge_abs_p25": 8.78618662e-11,
                        "mean_edge_abs_p75": 2.0187203333333334e-10,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "10676": {
                                "candidate_negative_offdiag_col_count": 4,
                                "offcandidate_negative_offdiag_col_count": 0,
                                "reciprocal_candidate_negative_edge_count": 4,
                                "column_graph_component_size": 866,
                                "column_graph_two_hop_count": 9,
                                "column_graph_clustering": 0.0,
                                "edge_abs_concentration": 0.3124297190397882,
                                "mean_edge_abs": 5.5002449999999997e-11,
                            }
                        }
                    },
                },
                {
                    "label": "test10",
                    "thresholds": {
                        "candidate_degree_p25": 4.0,
                        "candidate_degree_p75": 5.0,
                        "component_size_p75": 251.0,
                        "two_hop_p25": 9.0,
                        "two_hop_p75": 12.0,
                        "edge_concentration_p25": 0.3744658306882016,
                        "edge_concentration_p75": 0.5192773149025679,
                        "mean_edge_abs_p25": 2.150633e-09,
                        "mean_edge_abs_p75": 3.8852472e-09,
                    },
                    "profile_summary": {
                        "target_profiles": {
                            "3526": {
                                "candidate_negative_offdiag_col_count": 4,
                                "offcandidate_negative_offdiag_col_count": 0,
                                "reciprocal_candidate_negative_edge_count": 4,
                                "column_graph_component_size": 251,
                                "column_graph_two_hop_count": 7,
                                "column_graph_clustering": 0.0,
                                "edge_abs_concentration": 0.4945334699132767,
                                "mean_edge_abs": 2.150633e-09,
                            }
                        }
                    },
                },
            ],
            "selectors": [
                {
                    "key": "column_null_preserving_negative_offdiag_class",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 866,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 123.71428571428571,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 20.916666666666668,
                        },
                    ],
                },
                {
                    "key": "column_candidate_neighbor_closed",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 848,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 121.14285714285714,
                        },
                        {
                            "label": "test10",
                            "selected_count": 245,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 20.416666666666668,
                        },
                    ],
                },
                {
                    "key": "column_low_candidate_degree_p25",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 545,
                            "covered_direct_target_count": 1,
                            "selected_to_target_ratio": 77.85714285714286,
                        },
                        {
                            "label": "test10",
                            "selected_count": 99,
                            "covered_direct_target_count": 11,
                            "selected_to_target_ratio": 8.25,
                        },
                    ],
                },
                {
                    "key": "column_two_hop_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 654,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 93.42857142857143,
                        },
                        {
                            "label": "test10",
                            "selected_count": 193,
                            "covered_direct_target_count": 11,
                            "selected_to_target_ratio": 16.083333333333332,
                        },
                    ],
                },
                {
                    "key": "column_mean_edge_abs_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 434,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 62.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 140,
                            "covered_direct_target_count": 3,
                            "selected_to_target_ratio": 11.666666666666666,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_column_geometry_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_column_geometry_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "geometry_profiled_candidate",
                "geometry_complete_reference_edges",
                "geometry_has_diagonal_edges",
                "geometry_high_max_ref_edge_length_p75",
                "geometry_unique_length_count_tail",
            ],
            "miss_selector_keys": [
                "geometry_mixed_axis_diagonal_edges",
                "geometry_mean_ref_edge_length_tail",
                "geometry_axis_fraction_tail",
                "geometry_row_origin_fraction_tail",
            ],
            "next_requirement": (
                "Reference-node edge geometry did not isolate the direct PSPG "
                "target rows; move to quadrature/cut-interface geometry or a "
                "formulation-derived support/coupling balance."
            ),
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {
                        "status": "ok",
                        "geometry_field_entry_count": 13432,
                    },
                    "thresholds": {
                        "axis_aligned_edge_fraction_p25": 0.25,
                        "axis_aligned_edge_fraction_p75": 0.5555555555555556,
                        "mean_ref_edge_length_p25": 1.1840933333333334,
                        "mean_ref_edge_length_p75": 1.3106575000000003,
                    },
                    "profile_summary": {
                        "candidate_reference_geometry_class_counts": {
                            "axis_only_reference_edges": 2,
                            "diagonal_only_reference_edges": 38,
                            "mixed_axis_diagonal_reference_edges": 826,
                        },
                        "target_reference_geometry_class_counts": {
                            "mixed_axis_diagonal_reference_edges": 7,
                        },
                        "target_profiles": {
                            "10676": {
                                "reference_geometry_class": (
                                    "mixed_axis_diagonal_reference_edges"
                                ),
                                "finite_geometry_edge_sample_count": 8,
                                "mean_ref_edge_length": 1.2071049999999999,
                                "weighted_mean_ref_edge_length": 1.2489739293340567,
                                "axis_aligned_edge_fraction": 0.5,
                                "diagonal_edge_fraction": 0.5,
                                "row_origin_fraction": 0.0,
                                "unique_ref_edge_lengths": [1.0, 1.41421],
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {
                        "status": "ok",
                        "geometry_field_entry_count": 2880,
                    },
                    "thresholds": {
                        "axis_aligned_edge_fraction_p25": 0.1111111111111111,
                        "axis_aligned_edge_fraction_p75": 0.5555555555555556,
                        "mean_ref_edge_length_p25": 1.1840933333333334,
                        "mean_ref_edge_length_p75": 1.368186666666666,
                    },
                    "profile_summary": {
                        "candidate_reference_geometry_class_counts": {
                            "axis_only_reference_edges": 2,
                            "diagonal_only_reference_edges": 26,
                            "mixed_axis_diagonal_reference_edges": 223,
                        },
                        "target_reference_geometry_class_counts": {
                            "diagonal_only_reference_edges": 1,
                            "mixed_axis_diagonal_reference_edges": 11,
                        },
                        "target_profiles": {
                            "3526": {
                                "reference_geometry_class": (
                                    "diagonal_only_reference_edges"
                                ),
                                "finite_geometry_edge_sample_count": 8,
                                "mean_ref_edge_length": 1.4142100000000002,
                                "weighted_mean_ref_edge_length": 1.4142099999999997,
                                "axis_aligned_edge_fraction": 0.0,
                                "diagonal_edge_fraction": 1.0,
                                "row_origin_fraction": 0.0,
                                "unique_ref_edge_lengths": [1.41421],
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "geometry_has_diagonal_edges",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 864,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 123.42857142857143,
                        },
                        {
                            "label": "test10",
                            "selected_count": 249,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 20.75,
                        },
                    ],
                },
                {
                    "key": "geometry_mixed_axis_diagonal_edges",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 826,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 118.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 223,
                            "covered_direct_target_count": 11,
                            "selected_to_target_ratio": 18.583333333333332,
                        },
                    ],
                },
                {
                    "key": "geometry_mean_ref_edge_length_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 505,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 72.14285714285714,
                        },
                        {
                            "label": "test10",
                            "selected_count": 183,
                            "covered_direct_target_count": 9,
                            "selected_to_target_ratio": 15.25,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_quadrature_geometry_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_quadrature_geometry_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "overbroad_selector_keys": [
                "qgeom_profiled_candidate",
                "qgeom_complete_fields",
                "qgeom_uniform_full_cell_class",
                "qgeom_parent_cell_count_tail",
                "qgeom_weight_concentration_tail",
                "qgeom_parent_centroid_x_range_tail",
            ],
            "miss_selector_keys": [
                "qgeom_nonzero_level_set_residual",
                "qgeom_nonzero_gradient_norm",
                "qgeom_radius_tail",
                "qgeom_span_x_tail",
                "qgeom_row_to_centroid_distance_tail",
            ],
            "next_requirement": (
                "Cut-volume q-point geometry did not isolate the direct PSPG "
                "target rows; move to formulation-derived pressure-gradient "
                "support/coupling balance or a richer cut-interface proximity "
                "field."
            ),
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {
                        "status": "ok",
                        "cut_qpoint_field_entry_count": 13432,
                    },
                    "thresholds": {
                        "cut_qpoint_max_radius_max_p75": 0.535766,
                        "parent_cell_count_p75": 24.0,
                        "row_to_cut_qpoint_centroid_distance_mean_p75": (
                            0.82915619758885
                        ),
                    },
                    "profile_summary": {
                        "candidate_cut_qpoint_geometry_class_counts": {
                            "mixed_qpoint_geometry": 408,
                            "uniform_full_cell_qpoint_geometry": 458,
                        },
                        "target_cut_qpoint_geometry_class_counts": {
                            "uniform_full_cell_qpoint_geometry": 7,
                        },
                        "target_profiles": {
                            "10676": {
                                "cut_qpoint_geometry_class": (
                                    "uniform_full_cell_qpoint_geometry"
                                ),
                                "cut_qpoint_field_rule_count": 4,
                                "cut_qpoint_counts": [16],
                                "parent_cell_count": 4,
                                "cut_qpoint_weight_sum_total": 0.666668,
                                "cut_qpoint_max_radius_max": 0.435037,
                                "cut_qpoint_level_set_max_abs_max": 0.0,
                                "cut_qpoint_gradient_norm_max": 0.0,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {
                        "status": "ok",
                        "cut_qpoint_field_entry_count": 2880,
                    },
                    "thresholds": {
                        "cut_qpoint_max_radius_max_p75": 0.524314,
                        "parent_cell_count_p75": 12.0,
                        "row_to_cut_qpoint_centroid_distance_mean_p75": (
                            0.8359192559908185
                        ),
                    },
                    "profile_summary": {
                        "candidate_cut_qpoint_geometry_class_counts": {
                            "mixed_qpoint_geometry": 123,
                            "uniform_full_cell_qpoint_geometry": 128,
                        },
                        "target_cut_qpoint_geometry_class_counts": {
                            "uniform_full_cell_qpoint_geometry": 12,
                        },
                        "target_profiles": {
                            "3526": {
                                "cut_qpoint_geometry_class": (
                                    "uniform_full_cell_qpoint_geometry"
                                ),
                                "cut_qpoint_field_rule_count": 4,
                                "cut_qpoint_counts": [16],
                                "parent_cell_count": 4,
                                "cut_qpoint_weight_sum_total": 0.666668,
                                "cut_qpoint_max_radius_max": 0.435037,
                                "cut_qpoint_level_set_max_abs_max": 0.0,
                                "cut_qpoint_gradient_norm_max": 0.0,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "qgeom_uniform_full_cell_class",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 458,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 65.42857142857143,
                        },
                        {
                            "label": "test10",
                            "selected_count": 128,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 10.666666666666666,
                        },
                    ],
                },
                {
                    "key": "qgeom_radius_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 518,
                            "covered_direct_target_count": 1,
                            "selected_to_target_ratio": 74.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 172,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 14.333333333333334,
                        },
                    ],
                },
                {
                    "key": "qgeom_row_to_centroid_distance_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 481,
                            "covered_direct_target_count": 1,
                            "selected_to_target_ratio": 68.71428571428571,
                        },
                        {
                            "label": "test10",
                            "selected_count": 126,
                            "covered_direct_target_count": 8,
                            "selected_to_target_ratio": 10.5,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_gradient_balance_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_gradient_balance_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "next_requirement": (
                "Use the gradient-balance evidence to decide whether physical "
                "shape-gradient support can supply the missing formulation-side "
                "direct PSPG support/coupling gate."
            ),
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {"status": "ok"},
                    "profile_summary": {
                        "profiled_target_count": 7,
                        "target_gradient_support_class_counts": {
                            "full_cell_only_gradient_support": 7,
                        },
                        "target_profiles": {
                            "10676": {
                                "gradient_support_class": (
                                    "full_cell_only_gradient_support"
                                ),
                                "matrix_to_gram_abs_ratio": (
                                    1.967343434930542e-09
                                ),
                                "row_grad_resultant_ratio": 0.5482549566800144,
                                "gram_diag_abs_fraction": 0.5000004471052172,
                                "sampled_sign_mismatch_fraction": 0.0,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {"status": "ok"},
                    "profile_summary": {
                        "profiled_target_count": 12,
                        "target_gradient_support_class_counts": {
                            "full_cell_only_gradient_support": 12,
                        },
                        "target_profiles": {
                            "3526": {
                                "gradient_support_class": (
                                    "full_cell_only_gradient_support"
                                ),
                                "matrix_to_gram_abs_ratio": (
                                    1.669672490693332e-07
                                ),
                                "row_grad_resultant_ratio": 0.7619837673244342,
                                "gram_diag_abs_fraction": 0.5,
                                "sampled_sign_mismatch_fraction": 0.0,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "gradient_balance_full_cell_only",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 426,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 426 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 125,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 125 / 12,
                        },
                    ],
                },
                {
                    "key": "gradient_balance_resultant_ratio_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 434,
                            "covered_direct_target_count": 6,
                            "selected_to_target_ratio": 434 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 148,
                            "covered_direct_target_count": 3,
                            "selected_to_target_ratio": 148 / 12,
                        },
                    ],
                },
                {
                    "key": "gradient_balance_sampled_sign_mismatch",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_gradient_column_graph_"
            "selectivity_20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_cut_volume_gradient_column_graph_selectors_"
                "overbroad_or_miss_targets"
            ),
            "selective_selector_keys": [],
            "next_requirement": (
                "Do not promote sampled pressure-gradient edge topology, "
                "reciprocity, component, Gram-fraction, or cosine tails directly; "
                "the remaining rule must be stronger than sampled edge-level "
                "graph thresholding."
            ),
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {"status": "ok"},
                    "profile_summary": {
                        "profiled_target_count": 7,
                        "candidate_edge_class_counts": {
                            "missing_candidate_gradient_stencil": 16,
                            "reciprocal_all_negative_gradient_stencil": 850,
                        },
                        "target_edge_class_counts": {
                            "reciprocal_all_negative_gradient_stencil": 7,
                        },
                        "target_profiles": {
                            "10676": {
                                "gradient_column_edge_class": (
                                    "reciprocal_all_negative_gradient_stencil"
                                ),
                                "candidate_edge_sample_count": 8,
                                "candidate_neighbor_count": 4,
                                "candidate_component_size": 850,
                                "candidate_component_fraction": (
                                    0.9815242494226328
                                ),
                                "candidate_graph_clustering": 0.0,
                                "matrix_gradient_sign_mismatch_fraction": 0.0,
                            }
                        },
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {"status": "ok"},
                    "profile_summary": {
                        "profiled_target_count": 12,
                        "candidate_edge_class_counts": {
                            "reciprocal_all_negative_gradient_stencil": 251,
                        },
                        "target_edge_class_counts": {
                            "reciprocal_all_negative_gradient_stencil": 12,
                        },
                        "target_profiles": {
                            "3526": {
                                "gradient_column_edge_class": (
                                    "reciprocal_all_negative_gradient_stencil"
                                ),
                                "candidate_edge_sample_count": 8,
                                "candidate_neighbor_count": 4,
                                "candidate_component_size": 251,
                                "candidate_component_fraction": 1.0,
                                "candidate_graph_clustering": 0.0,
                                "matrix_gradient_sign_mismatch_fraction": 0.0,
                            }
                        },
                    },
                },
            ],
            "selectors": [
                {
                    "key": "gradient_column_graph_reciprocal_negative_stencil",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 850,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 850 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
                {
                    "key": "gradient_column_graph_edge_count_tail",
                    "finding": "selector_overbroad_or_miss_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 523,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 523 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 228,
                            "covered_direct_target_count": 11,
                            "selected_to_target_ratio": 228 / 12,
                        },
                    ],
                },
                {
                    "key": "gradient_column_graph_sign_mismatch",
                    "finding": "selector_misses_targets",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                        {
                            "label": "test10",
                            "selected_count": 0,
                            "covered_direct_target_count": 0,
                            "selected_to_target_ratio": 0.0,
                        },
                    ],
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_cut_volume_local_schur_"
            "completion_20260607.json"
        ),
        {
            "finding": "direct_pspg_cut_volume_local_schur_completion_overbroad",
            "aggregate_selector_finding": "selector_overbroad",
            "next_requirement": (
                "Do not promote local Schur completion alone; it touches "
                "audited targets only by selecting a broad direct PSPG "
                "candidate set."
            ),
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {"status": "ok"},
                    "summary_metrics": {
                        "constant_pressure_null_preserving_all": True,
                        "diagnostic_only_all": True,
                        "summary_count": 3358,
                        "local_row_count_sum": 13432,
                        "source_edge_count_sum": 10074,
                        "schur_hub_count_sum": 6716,
                        "schur_contribution_count_sum": 6716,
                        "schur_edge_count_sum": 6716,
                        "touched_row_count_sum": 13432,
                    },
                    "selector": {
                        "selected_count": 866,
                        "covered_direct_target_count": 7,
                        "selected_to_target_ratio": 866 / 7,
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {"status": "ok"},
                    "summary_metrics": {
                        "constant_pressure_null_preserving_all": True,
                        "diagnostic_only_all": True,
                        "summary_count": 720,
                        "local_row_count_sum": 2880,
                        "source_edge_count_sum": 2160,
                        "schur_hub_count_sum": 1440,
                        "schur_contribution_count_sum": 1440,
                        "schur_edge_count_sum": 1440,
                        "touched_row_count_sum": 2880,
                    },
                    "selector": {
                        "selected_count": 251,
                        "covered_direct_target_count": 12,
                        "selected_to_target_ratio": 251 / 12,
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_cut_volume_local_edge_balance_20260607.json",
        {
            "finding": "direct_pspg_cut_volume_local_edge_balance_overbroad",
            "aggregate_selector_findings": [
                "selector_overbroad",
                "selector_overbroad",
            ],
            "next_requirement": (
                "Do not promote local existing-edge balance alone; it covers "
                "audited targets only by selecting a broad direct PSPG "
                "candidate set."
            ),
            "selectors": [
                {
                    "key": "local_edge_balance_candidate_rows",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 865,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 865 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 248,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 248 / 12,
                        },
                    ],
                },
                {
                    "key": "local_edge_balance_touched_rows",
                    "finding": "selector_overbroad",
                    "cases": [
                        {
                            "label": "test02",
                            "selected_count": 866,
                            "covered_direct_target_count": 7,
                            "selected_to_target_ratio": 866 / 7,
                        },
                        {
                            "label": "test10",
                            "selected_count": 251,
                            "covered_direct_target_count": 12,
                            "selected_to_target_ratio": 251 / 12,
                        },
                    ],
                },
            ],
            "cases": [
                {
                    "label": "test02",
                    "log_evidence": {"status": "ok"},
                    "summary_metrics": {
                        "constant_pressure_null_preserving_all": True,
                        "diagnostic_only_all": True,
                        "summary_count": 3358,
                        "local_row_count_sum": 13432,
                        "source_edge_count_sum": 10074,
                        "balance_candidate_row_count_sum": 10074,
                        "balance_edge_count_sum": 10074,
                        "touched_row_count_sum": 13432,
                    },
                },
                {
                    "label": "test10",
                    "log_evidence": {"status": "ok"},
                    "summary_metrics": {
                        "constant_pressure_null_preserving_all": True,
                        "diagnostic_only_all": True,
                        "summary_count": 720,
                        "local_row_count_sum": 2880,
                        "source_edge_count_sum": 2160,
                        "balance_candidate_row_count_sum": 2160,
                        "balance_edge_count_sum": 2160,
                        "touched_row_count_sum": 2880,
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_graph_completion_support_gap_patch_schur_only_20260606_outcome.json",
        {
            "finding": (
                "support_gap_patch_schur_only_clears_test10_guard_without_"
                "edge_balance_but_full_patch_expansion_still_fails_test02_"
                "nonlinear_convergence"
            ),
            "test10_step90": {
                "outcome": "accepted_guard_not_triggered",
                "accepted_pressure_update_pa": 93.45299901163241,
            },
            "test02_step382": {
                "outcome": "nonlinear_failed",
                "final_residual_norm": 2542.9071873675475,
            },
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_graph_completion_support_gap_local_patch_schur_only_depth1_20260606_outcome.json",
        {
            "finding": (
                "support_gap_local_patch_depth1_schur_only_keeps_test10_"
                "below_guard_but_still_fails_test02_nonlinear_convergence"
            ),
            "pressure_neighbor_depth": 1,
            "test10_step90": {
                "outcome": "accepted_guard_not_triggered",
                "accepted_pressure_update_pa": 93.87750523707791,
            },
            "test02_step382": {
                "outcome": "nonlinear_failed",
                "final_residual_norm": 2513.2066824913213,
            },
        },
    )
    for name, max_dof, isolated in (
        ("test02_update_support_components_20260606.json", 10676, 7),
        ("test10_update_support_components_20260606.json", 3356, 0),
    ):
        _write_json(
            tmp_path,
            name,
            {
                "latest_pressure_update_support_diagnostic": {
                    "values": {
                        "max_update_global_dof": max_dof,
                        "same_sign_pressure_action_isolated_top_update_count": (
                            isolated
                        ),
                    },
                },
            },
        )
    _write_json(
        tmp_path,
        "test10_replay_cap3_step90_support_rank_guard_audit_20260605.json",
        {
            "latest_support_rank_diagnostic": {
                "values": {
                    "zero_coupling_row_block_count": 9,
                    "pressure_only_row_block_count": 9,
                },
            },
        },
    )
    _write_json(
        tmp_path,
        "test10_replay_cap3_step90_vms_disabled_pressure_constraint_coverage_audit_20260605.json",
        {
            "constraint_vertex_dof_mapping_status": "ok",
            "reported_zero_row_count": 5,
        },
    )
    for name in (
        "test10_replay_cap3_step90_pressure_reference_probe_penalty1em6_support_audit_20260605.json",
        (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_"
            "free_surface_tangential_scale1_support_audit_20260606.json"
        ),
        (
            "test10_replay_cap3_step90_pspg_full_cell_support_"
            "wall_full_gradient_pressure_update_audit_20260606.json"
        ),
    ):
        _write_json(tmp_path, name, {"status": "diagnostic_pressure_update_guard_triggered"})
    for name, threshold, update_pa, point_index, support_class in (
        (
            (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                "shape_tangent_pressure_update_audit_20260606.json"
            ),
            100000.0,
            366720.75479080016,
            1172,
            "full_wet_supported",
        ),
        (
            (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                "shape_tangent_pressure_update_audit_20260606.json"
            ),
            100.0,
            622.70494029817,
            83,
            "full_wet_supported",
        ),
        (
            (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                "cut_volume_scale_cap16_pressure_update_audit_20260606.json"
            ),
            100000.0,
            366739.55884526786,
            1172,
            "full_wet_supported",
        ),
        (
            (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                "cut_volume_scale_cap16_pressure_update_audit_20260606.json"
            ),
            100.0,
            629.8487957714515,
            83,
            "full_wet_supported",
        ),
        (
            (
                "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
                "free_surface_tangential_scale1_pressure_update_audit_20260606.json"
            ),
            100000.0,
            347889.7856780469,
            1172,
            "full_wet_supported",
        ),
        (
            (
                "test10_replay_cap3_step90_pspg_wall_full_gradient_"
                "free_surface_tangential_scale1_pressure_update_audit_20260606.json"
            ),
            100.0,
            862.2837968627545,
            7,
            "full_wet_supported",
        ),
    ):
        _write_json(
            tmp_path,
            name,
            _pressure_update_fixture(
                threshold,
                update_pa,
                point_index,
                support_class,
            ),
        )
    _write_json(
        tmp_path,
        "test02_test10_graph_completion_selector_coverage_20260606.json",
        {
            "finding": "shifted_pressure_update_rows_escape_weak_row_selector",
            "case_count": 4,
            "finding_counts": {
                "max_update_row_not_sampled": 1,
                "max_update_row_outside_selector_rule": 3,
            },
            "sampled_outside_selector_threshold_floor_if_casewise_least_widened": {
                "case_count": 4,
                "coupling_threshold": 9.687318259562922e-09,
                "self_threshold": 1.759508340111003e-07,
            },
            "sampled_outside_selector_threshold_floor_if_single_selector_widened": {
                "case_count": 4,
                "coupling_threshold": 0.0003266812746844929,
                "self_threshold": 0.8026219518508216,
            },
            "cases": [
                {
                    "label": "test02_existing_edge",
                    "finding": "max_update_row_not_sampled",
                    "max_update_global_dof": 10624,
                    "max_abs_update_pa": 226147.9528838232,
                    "selector_reason": "missing_row_support",
                    "least_selector_threshold_expansion_to_include": None,
                },
                {
                    "label": "test02_existing_edge_cap256",
                    "finding": "max_update_row_outside_selector_rule",
                    "max_update_global_dof": 10624,
                    "max_abs_update_pa": 223708.2137905913,
                    "selector_reason": "row_support_outside_selector",
                    "least_selector_threshold_expansion_to_include": {
                        "selector": "coupling_threshold",
                        "threshold_needed": 9.687318259562922e-09,
                        "factor_of_current": 968731.8259562922,
                    },
                },
                {
                    "label": "test10_existing_edge_cap256",
                    "finding": "max_update_row_outside_selector_rule",
                    "max_update_global_dof": 3354,
                    "max_abs_update_pa": 236.128309,
                    "selector_reason": "row_support_outside_selector",
                    "least_selector_threshold_expansion_to_include": {
                        "selector": "self_threshold",
                        "threshold_needed": 1.759508340111003e-07,
                        "factor_of_current": 17.59508340111003,
                    },
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_formulation_vocabulary_support_20260607.json",
        {
            "finding": (
                "form_vocabulary_lacks_direct_pspg_support_topology_handles"
            ),
            "status": "requires_fe_forms_or_assembly_api_extension",
            "public_cut_cell_helpers": [
                "cutEmbeddedNormal",
                "cutStabilizationScale",
                "cutVolumeFraction",
            ],
            "public_measures": ["dx", "ds", "dS", "dI", "dCutVolume"],
            "direct_pspg_expression_summary": {
                "direct_pressure_gradient_integrand_installed": True,
                "active_volume_measure_used": True,
                "cut_volume_fraction_scale_available": True,
                "free_surface_boundary_terms_separate": True,
            },
            "required_topology_handles_missing": {
                "active_pressure_graph_connectivity": True,
                "element_local_schur_completion": True,
                "existing_pressure_edge_balance": True,
                "direct_pspg_local_matrix_provenance": True,
                "post_update_same_sign_pressure_action": True,
            },
            "missing_required_topology_handle_count": 5,
            "required_topology_handle_count": 5,
            "next_requirement": (
                "Do not search for another scalar form multiplier in the "
                "current DSL; add an FE Forms/assembly API."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_assembly_api_support_20260607.json",
        {
            "finding": (
                "assembly_api_has_direct_pspg_topology_policy_hook_replay_pending"
            ),
            "status": "topology_policy_hook_available_replay_pending",
            "assembly_diagnostic_context_fields": [
                "operator_tag",
                "source_component_tag",
                "test_field_name",
                "trial_field_name",
            ],
            "planned_cut_volume_term_fields": [
                "col_dof_map",
                "col_dof_offset",
                "kernel",
                "marker",
                "matrix_capable",
                "row_dof_map",
                "row_dof_offset",
                "side",
                "source_component_tag",
                "test_field",
                "test_space",
                "trial_field",
                "trial_space",
                "vector_capable",
            ],
            "assembly_api_features": {
                "add_cut_volume_kernel_has_source_component_argument": True,
                "diagnostic_context_is_documented_non_mutating": False,
                "diagnostic_context_has_source_component_tag": True,
                "diagnostic_context_lacks_topology_policy_handle": True,
                "diagnostic_context_only_operator_and_fields": False,
                "cut_volume_context_built_from_request_op_and_fields": True,
                "cut_volume_context_includes_source_component_tag": True,
                "fused_composite_terms_may_drop_per_term_diagnostic_context": False,
                "fused_composite_terms_preserve_source_component_diagnostic_context": True,
                "planned_cut_volume_term_has_source_component_tag": True,
                "planned_cut_volume_term_lacks_source_component_tag": False,
                "add_cut_volume_kernel_lacks_source_component_argument": False,
                "forms_installer_forwards_cut_volume_only_op_fields_kernel": False,
                "forms_installer_forwards_source_component_tag_to_cut_volumes": True,
                "operator_registry_cut_volume_term_has_source_component_tag": True,
                "system_setup_preserves_cut_volume_source_component_tag": True,
                "diagnostic_logs_include_source_component_tag": True,
                "direct_pspg_local_topology_diagnostics_log_before_insert": True,
                "direct_pspg_local_topology_diagnostics_mark_diagnostic_only": True,
                "direct_pspg_topology_policy_api_env_gated": True,
                "direct_pspg_topology_policy_scoped_to_equations_operator": True,
                "direct_pspg_topology_policy_scoped_to_production_source_component": True,
                "direct_pspg_topology_policy_requires_pressure_pressure_block": True,
                "direct_pspg_topology_policy_default_partial_cut_only": True,
                "direct_pspg_topology_policy_constant_null_preserving": True,
                "direct_pspg_topology_policy_log_marks_solve_affecting": True,
                "direct_pspg_topology_policy_mutates_before_global_insert": True,
                "production_direct_pspg_subterm_has_source_component_tag": True,
                "production_direct_pspg_split_preserves_velocity_tangent": True,
                "production_direct_pspg_subterm_lacks_source_component_tag": False,
                "production_equations_installed": True,
                "direct_pspg_diagnostic_operator_installed": True,
                "pressure_row_contribution_diagnostics_env_gated": True,
            },
            "required_api_handles_missing": {
                "production_subterm_provenance_tag": False,
                "solve_affecting_local_matrix_mutation_hook": False,
                "direct_pspg_topology_policy_api": False,
                "planned_cut_volume_source_component_tag": False,
                "add_cut_volume_kernel_source_component_argument": False,
                "forms_installer_source_component_forwarding": False,
                "system_setup_source_component_propagation": False,
                "assembly_diagnostic_source_component_context": False,
                "legacy_planned_cut_volume_source_component_tag_absent": False,
                "composite_term_provenance_for_fused_cut_volume_blocks": False,
            },
            "missing_required_api_handle_count": 0,
            "required_api_handle_count": 10,
            "next_requirement": (
                "Run short Test02/Test10 replay windows with the API-backed "
                "direct PSPG topology policy modes enabled."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_topology_policy_replay_pair_20260607.json",
        {
            "finding": (
                "direct_pspg_topology_policy_schur_edge_balance_"
                "replay_pair_does_not_clear_guards"
            ),
            "status": "policy_hook_exercised_mode_ruled_out_as_complete_fix",
            "policy": "local_schur_edge_balance",
            "policy_hook_exercised": True,
            "policy_log_counts": {
                "test02": 3352,
                "test10": 720,
            },
            "pressure_update_guard_cleared": {
                "test02": False,
                "test10": False,
            },
            "cases": [
                {
                    "label": "test02",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 3352,
                    "absolute_threshold_pa": 100000.0,
                    "worst_active_or_wet_update_pa": 176844.2140471727,
                    "worst_active_or_wet_support_class": "tiny_cut_supported",
                },
                {
                    "label": "test10",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 720,
                    "absolute_threshold_pa": 100.0,
                    "worst_active_or_wet_update_pa": 522.4172735486616,
                    "worst_active_or_wet_support_class": "full_wet_supported",
                },
            ],
            "next_requirement": (
                "Do not promote broad local_schur_edge_balance as the "
                "production fix."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_topology_policy_mode_replays_20260607.json",
        {
            "finding": "direct_pspg_topology_policy_local_modes_do_not_clear_guards",
            "status": "local_topology_policy_family_ruled_out_as_complete_fix",
            "policies_tested": [
                "local_schur_completion",
                "local_edge_balance",
                "local_schur_edge_balance",
            ],
            "policy_hook_exercised": True,
            "policy_log_counts": {
                "local_schur_completion": {
                    "test02": 3352,
                    "test10": 720,
                },
                "local_edge_balance": {
                    "test02": 3352,
                    "test10": 720,
                },
                "local_schur_edge_balance": {
                    "test02": 3352,
                    "test10": 720,
                },
            },
            "pressure_update_guard_cleared": {
                "local_schur_completion": {
                    "test02": False,
                    "test10": False,
                },
                "local_edge_balance": {
                    "test02": False,
                    "test10": False,
                },
                "local_schur_edge_balance": {
                    "test02": False,
                    "test10": False,
                },
            },
            "case_policy_results": [
                {
                    "case": "test02",
                    "policy": "local_schur_completion",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 3352,
                    "absolute_threshold_pa": 100000.0,
                    "worst_active_or_wet_update_pa": 176849.84039557964,
                    "worst_active_or_wet_support_class": "tiny_cut_supported",
                },
                {
                    "case": "test02",
                    "policy": "local_edge_balance",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 3352,
                    "absolute_threshold_pa": 100000.0,
                    "worst_active_or_wet_update_pa": 176848.02921204976,
                    "worst_active_or_wet_support_class": "tiny_cut_supported",
                },
                {
                    "case": "test02",
                    "policy": "local_schur_edge_balance",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 3352,
                    "absolute_threshold_pa": 100000.0,
                    "worst_active_or_wet_update_pa": 176844.2140471727,
                    "worst_active_or_wet_support_class": "tiny_cut_supported",
                },
                {
                    "case": "test10",
                    "policy": "local_schur_completion",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 720,
                    "absolute_threshold_pa": 100.0,
                    "worst_active_or_wet_update_pa": 590.7292901816519,
                    "worst_active_or_wet_support_class": "full_wet_supported",
                },
                {
                    "case": "test10",
                    "policy": "local_edge_balance",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 720,
                    "absolute_threshold_pa": 100.0,
                    "worst_active_or_wet_update_pa": 530.3194043612839,
                    "worst_active_or_wet_support_class": "full_wet_supported",
                },
                {
                    "case": "test10",
                    "policy": "local_schur_edge_balance",
                    "guard_status": "diagnostic_pressure_update_guard_triggered",
                    "policy_log_count": 720,
                    "absolute_threshold_pa": 100.0,
                    "worst_active_or_wet_update_pa": 522.4172735486616,
                    "worst_active_or_wet_support_class": "full_wet_supported",
                },
            ],
            "mode_interpretation": {
                "test02": "All local topology policies leave the accepted maximum on the same tiny-cut-supported point.",
                "test10": "Edge balance improves the update, but remains above the guard.",
            },
            "next_requirement": (
                "Move to a narrower formulation-derived pressure-gradient "
                "support/coupling rule."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_topology_policy_application_effect_"
            "20260607.json"
        ),
        {
            "finding": (
                "direct_pspg_topology_policy_application_effect_rules_out_"
                "underapplication"
            ),
            "status": "local_matrix_policy_applies_but_is_not_sufficient_fix",
            "all_replays_trigger_guard": True,
            "all_test10_signature_replays_mutate_selected_records": True,
            "best_test02_broad_policy": "local_schur_edge_balance",
            "best_test02_broad_update_pa": 176844.2140471727,
            "best_test10_broad_policy": "local_schur_edge_balance",
            "best_test10_broad_update_pa": 522.4172735486616,
            "best_test10_signature_policy": "local_schur_edge_balance",
            "best_test10_signature_update_pa": 604.7126561932914,
            "test10_broad_vs_signature_row_filter": {
                "local_schur_completion": {
                    "broad_policy_log_count": 720,
                    "signature_policy_log_count": 264,
                    "broad_matrix_mutated_count": 720,
                    "signature_matrix_mutated_count": 84,
                    "signature_selected_records_matrix_mutated_count": 84,
                    "broad_update_pa": 590.7292901816519,
                    "signature_row_filter_update_pa": 619.6167550623924,
                    "signature_minus_broad_update_pa": 28.88746488074048,
                },
                "local_edge_balance": {
                    "broad_policy_log_count": 720,
                    "signature_policy_log_count": 264,
                    "broad_matrix_mutated_count": 720,
                    "signature_matrix_mutated_count": 258,
                    "signature_selected_records_matrix_mutated_count": 258,
                    "broad_update_pa": 530.3194043612839,
                    "signature_row_filter_update_pa": 607.5173052131886,
                    "signature_minus_broad_update_pa": 77.1979008519047,
                },
                "local_schur_edge_balance": {
                    "broad_policy_log_count": 720,
                    "signature_policy_log_count": 264,
                    "broad_matrix_mutated_count": 720,
                    "signature_matrix_mutated_count": 258,
                    "signature_selected_records_matrix_mutated_count": 258,
                    "broad_update_pa": 522.4172735486616,
                    "signature_row_filter_update_pa": 604.7126561932914,
                    "signature_minus_broad_update_pa": 82.29538264462985,
                },
            },
            "next_requirement": (
                "Do not treat the current local topology-policy failure as a "
                "hook execution or row-filter coverage issue."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_topology_policy_scope_scale_20260607.json",
        {
            "finding": (
                "direct_pspg_topology_policy_scope_scale_rules_out_exact_row_filter"
            ),
            "status": "broad_cosupport_mutation_helpful_but_insufficient",
            "same_case_no_policy_test10_update_pa": 622.6094100310928,
            "all_replays_trigger_guard": True,
            "signature_rows_worse_than_broad_for_all_test10_modes": True,
            "test10_broad_vs_signature_row_filter": {
                "local_schur_completion": {
                    "broad_update_pa": 590.7292901816519,
                    "signature_row_filter_update_pa": 619.6167550623924,
                    "signature_minus_broad_update_pa": 28.88746488074048,
                    "no_policy_to_broad_improvement_pa": 31.880119849440916,
                    "signature_to_broad_policy_log_fraction": (
                        0.36666666666666664
                    ),
                    "signature_to_broad_touched_row_fraction": 0.0625,
                    "signature_to_broad_topology_edge_weight_fraction": (
                        0.0683742309892418
                    ),
                },
                "local_edge_balance": {
                    "broad_update_pa": 530.3194043612839,
                    "signature_row_filter_update_pa": 607.5173052131886,
                    "signature_minus_broad_update_pa": 77.1979008519047,
                    "no_policy_to_broad_improvement_pa": 92.29000566980892,
                    "signature_to_broad_policy_log_fraction": (
                        0.36666666666666664
                    ),
                    "signature_to_broad_touched_row_fraction": (
                        0.2111111111111111
                    ),
                    "signature_to_broad_topology_edge_weight_fraction": (
                        0.2031175152469853
                    ),
                },
                "local_schur_edge_balance": {
                    "broad_update_pa": 522.4172735486616,
                    "signature_row_filter_update_pa": 604.7126561932914,
                    "signature_minus_broad_update_pa": 82.29538264462985,
                    "no_policy_to_broad_improvement_pa": 100.19213648243124,
                    "signature_to_broad_policy_log_fraction": (
                        0.36666666666666664
                    ),
                    "signature_to_broad_touched_row_fraction": (
                        0.21388888888888888
                    ),
                    "signature_to_broad_topology_edge_weight_fraction": (
                        0.18883839042222791
                    ),
                },
            },
            "test02_broad_policy_scope": {
                "local_schur_edge_balance": {
                    "update_pa": 176844.2140471727,
                    "support_class": "tiny_cut_supported",
                    "policy_log_count": 3352,
                    "matrix_mutated_count": 3352,
                    "touched_row_count_sum": 13408.0,
                    "topology_edge_weight_sum_total": 6.012408200777215e-05,
                    "max_row_abs_delta": 3.56123e-08,
                },
            },
            "next_requirement": (
                "A credible formulation fix must act on the coupled direct PSPG "
                "support patch or physical boundary support rule."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_direct_pspg_topology_policy_parent_scope_20260607.json",
        {
            "finding": (
                "direct_pspg_topology_policy_parent_scope_rules_out_exact_parent_subset"
            ),
            "status": "broad_parent_cosupport_required_but_insufficient",
            "same_case_no_policy_test10_update_pa": 622.6094100310928,
            "all_replays_trigger_guard": True,
            "all_test10_signature_parent_rule_sets_are_strict_broad_subsets": (
                True
            ),
            "all_test10_broad_only_rule_weight_share_above_half": True,
            "signature_rows_worse_than_broad_for_all_test10_modes": True,
            "test10_parent_rule_scope": {
                "local_schur_edge_balance": {
                    "broad_update_pa": 522.4172735486616,
                    "signature_row_filter_update_pa": 604.7126561932914,
                    "signature_minus_broad_update_pa": 82.29538264462985,
                    "no_policy_to_broad_improvement_pa": 100.19213648243124,
                    "rule_scope": {
                        "broad_key_count": 720,
                        "signature_key_count": 264,
                        "overlap_key_count": 264,
                        "broad_only_key_count": 456,
                        "signature_only_key_count": 0,
                        "signature_to_broad_key_fraction": (
                            0.36666666666666664
                        ),
                        "broad_only_topology_edge_weight_sum_fraction": (
                            0.5734631284834748
                        ),
                        "signature_to_broad_overlap_topology_edge_weight_sum_fraction": (
                            0.44272465766165725
                        ),
                        "signature_to_broad_topology_edge_weight_sum_fraction": (
                            0.18883839042222791
                        ),
                    },
                },
            },
            "test02_broad_parent_rule_scope": {
                "local_schur_edge_balance": {
                    "update_pa": 176844.2140471727,
                    "support_class": "tiny_cut_supported",
                    "rule_scope": {
                        "broad_key_count": 3352,
                        "broad_topology_edge_weight_sum": 6.012408200777215e-05,
                        "broad_cut_cell_record_count": 1110.0,
                        "broad_full_cell_record_count": 2242.0,
                    },
                },
            },
            "next_requirement": (
                "A credible direct PSPG formulation fix should express a "
                "connected support-patch or physical boundary-support closure."
            ),
        },
    )
    _write_json(
        tmp_path,
        (
            "test02_test10_direct_pspg_topology_policy_parent_subset_replay_"
            "readiness_20260607.json"
        ),
        {
            "finding": "direct_pspg_signature_parent_subset_replay_ready",
            "status": "run_signature_parent_full_local_replay",
            "source_hook": {
                "parent_cell_filter_api_present": True,
                "row_filter_api_present": True,
                "topology_policy_api_present": True,
            },
            "parent_scope": {
                "strict_parent_rule_subset": True,
                "broad_only_rule_weight_majority": True,
                "combined_rule_scope": {
                    "broad_key_count": 720,
                    "signature_key_count": 264,
                    "broad_only_key_count": 456,
                    "signature_to_broad_overlap_topology_edge_weight_sum_fraction": (
                        0.44272465766165725
                    ),
                },
            },
            "same_signature_parent_set_all_policies": True,
            "signature_parent_cell_count": 264,
            "signature_parent_cell_ranges": (
                "1-4,6-125,240-245,356-357,1441-1444,1446-1565,"
                "1680-1685,1796-1797"
            ),
            "next_requirement": (
                "Run the Test10 step90 topology-policy replay with "
                "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELLS set "
                "and no global row DOF filter."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test10_direct_pspg_topology_policy_parent_subset_replay_20260607.json",
        {
            "finding": (
                "direct_pspg_signature_parent_subset_full_local_replay_"
                "does_not_clear_test10_guard"
            ),
            "status": "exact_parent_subset_ruled_out_as_sufficient_fix",
            "signature_parent_filter_full_local_confirmed": True,
            "signature_parent_filter_update_pa": 578.9424523317655,
            "broad_policy_update_pa": 522.4172735486616,
            "signature_row_filter_update_pa": 604.7126561932914,
            "same_case_no_policy_update_pa": 622.609409861916,
            "parent_minus_broad_update_pa": 56.52517878310391,
            "parent_minus_signature_row_update_pa": -25.770203861525943,
            "pressure_update_guard_cleared": {
                "same_case_no_policy": False,
                "broad_policy": False,
                "signature_row_filter": False,
                "signature_parent_filter": False,
            },
            "replays": {
                "signature_parent_filter": {
                    "pressure_update": {
                        "worst_active_or_wet_update_pa": 578.9424523317655,
                        "worst_active_or_wet_point_index": 83,
                        "worst_active_or_wet_support_class": (
                            "full_wet_supported"
                        ),
                    },
                    "policy_log": {
                        "record_count": 264,
                        "matrix_mutated_count": 264,
                        "row_filter_enabled_values": [0],
                        "parent_filter_enabled_values": [1],
                        "parent_filter_parent_cell_count_values": [264],
                        "parent_filter_selected_count": 264,
                        "selected_local_row_count_sum": 1056.0,
                    },
                },
            },
            "next_requirement": (
                "Move away from exact-row or exact-parent replay of the "
                "current local matrix deltas and test a physical support-patch "
                "closure."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_active_pressure_support_cutoff_relevance_20260607.json",
        {
            "finding": (
                "active_pressure_support_cutoff_not_complete_fix_from_branch_shift"
            ),
            "status": "support_cutoff_diagnostic_only_not_complete_fix",
            "constraint_source": {
                "constraint_source_has_retained_volume_fraction_diagnostic": True,
                "detected_cutoff_env_terms": [],
                "retained_generated_volume_support_activation_is_unconditional": True,
                "retained_generated_volume_support_uses_volume_fraction_cutoff": False,
                "sample_dof_env_supported": True,
            },
            "classification": {
                "tiny_cut_supported_branch_present": True,
                "full_wet_supported_branch_present": True,
                "retained_fraction_cutoff_is_complete_fix_candidate": False,
                "retained_fraction_cutoff_is_diagnostic_only": True,
            },
            "topology_policy_replay_summary": {
                "finding": "direct_pspg_topology_policy_local_modes_do_not_clear_guards",
                "status": "local_topology_policy_family_ruled_out_as_complete_fix",
                "test02_policy_support_class_counts": {
                    "tiny_cut_supported": 3,
                },
                "test10_policy_support_class_counts": {
                    "full_wet_supported": 3,
                },
                "test02_min_tiny_cut_fraction_positive": 5.023213675652029e-7,
            },
            "pressure_update_rejection_summary": {
                "finding": (
                    "pressure_update_rejection_catches_both_cases_dt_reduction_not_fix"
                ),
                "status": "pre_commit_guard_supported_dt_reduction_ruled_out",
                "fixed_step_support_class_counts": {
                    "tiny_cut_supported": 1,
                    "full_wet_supported": 1,
                },
                "test02_adaptive_support_branch_shift": (
                    "tiny_cut_supported_to_full_wet_supported"
                ),
                "test02_adaptive_support_sequence": [
                    "tiny_cut_supported",
                    "full_wet_supported",
                ],
                "test10_adaptive_support_sequence": [
                    "full_wet_supported",
                ],
            },
            "next_requirement": (
                "Use the active pressure-support sample diagnostics to "
                "characterize tiny-cut rows, but pursue a formulation-side "
                "pressure-gradient support/coupling rule that also handles the "
                "full-wet boundary rows."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_pressure_update_rejection_replay_20260607.json",
        {
            "finding": (
                "pressure_update_rejection_catches_both_cases_dt_reduction_not_fix"
            ),
            "status": "pre_commit_guard_supported_dt_reduction_ruled_out",
            "guard": {
                "environment": "SVMP_ACTIVE_PRESSURE_UPDATE_REJECT_ON_TRIGGER=1",
                "diagnostic": "accepted_pressure_update_guard",
                "phase": "pre_commit",
                "reject_reason": "ErrorTooLarge",
                "default_behavior": "disabled unless requested by environment",
            },
            "fixed_step_replays": [
                {
                    "case": "test02",
                    "threshold_pa": 100000.0,
                    "step_accepted": False,
                    "step_rejected_count": 1,
                    "worst_pre_commit_update_pa": 105591.14535324997,
                    "worst_pre_commit_dof": 11875,
                    "worst_pre_commit_support_class": "tiny_cut_supported",
                },
                {
                    "case": "test10",
                    "threshold_pa": 1000.0,
                    "step_accepted": False,
                    "step_rejected_count": 1,
                    "worst_pre_commit_update_pa": 1075.5582119407377,
                    "worst_pre_commit_dof": 3356,
                    "worst_pre_commit_support_class": "full_wet_supported",
                },
            ],
            "adaptive_replays": [
                {
                    "case": "test02",
                    "threshold_pa": 100000.0,
                    "step_accepted": False,
                    "step_rejected_count": 11,
                    "first_update_pa": 105593.66490062946,
                    "last_update_pa": 14137282.618001418,
                    "update_growth_factor": 133.6925183418554,
                    "support_branch_shift": (
                        "tiny_cut_supported_to_full_wet_supported"
                    ),
                },
                {
                    "case": "test10",
                    "threshold_pa": 1070.0,
                    "step_accepted": False,
                    "step_rejected_count": 4,
                    "first_update_pa": 1075.5582119407377,
                    "last_update_pa": 9103.459989150644,
                    "update_growth_factor": 8.463008330177967,
                    "support_branch_shift": "none",
                },
            ],
            "next_requirement": (
                "Retain the rejection hook as a diagnostic safety gate."
            ),
        },
    )
    _write_json(
        tmp_path,
        "test02_test10_pressure_update_residual_context_20260607.json",
        {
            "finding": "accepted_pressure_updates_converged_with_large_residual_gap",
            "status": "residual_convergence_acceptance_gap_supported",
            "large_ratio_threshold": 1000.0,
            "all_cases_accepted_converged_large_update_residual_gap": True,
            "all_cases_post_acceptance_refresh_ruled_out": True,
            "cases": [
                {
                    "label": "test02",
                    "finding": (
                        "accepted_converged_large_pressure_update_residual_gap"
                    ),
                    "global_abs_pressure_delta_pa": 2112209.8407043177,
                    "nonlinear_field_residual_norm": 1.8964751204792418,
                    "update_to_nonlinear_field_residual_norm_ratio": (
                        1113750.000597943
                    ),
                },
                {
                    "label": "test10",
                    "finding": (
                        "accepted_converged_large_pressure_update_residual_gap"
                    ),
                    "global_abs_pressure_delta_pa": 1075.5582134176257,
                    "nonlinear_field_residual_norm": 0.0011455415013957783,
                    "update_to_nonlinear_field_residual_norm_ratio": (
                        938426.6400308398
                    ),
                },
            ],
            "next_requirement": (
                "Keep the pressure-update guard as a diagnostic safety gate."
            ),
        },
    )

    report = audit.build_status_report(
        artifact_root=tmp_path,
        root_report=root_report,
    )

    assert report["overall_status"] == "primary_formulation_target_unresolved"
    assert report["missing_evidence"] == []
    assert report["unresolved_hypotheses"] == [
        "direct_pspg_pressure_gradient_support_topology"
    ]
    statuses = {item["key"]: item["status"] for item in report["hypotheses"]}
    assert statuses["direct_pspg_pressure_gradient_support_topology"] == (
        "supported_unresolved_primary_target"
    )
    assert statuses["direct_pspg_solve_time_aggregate_features"] == (
        "aggregate_counts_and_volume_features_ruled_out"
    )
    assert statuses["direct_pspg_solve_time_support_measure_features"] == (
        "active_qpoint_and_measure_features_ruled_out"
    )
    assert statuses["direct_pspg_solve_time_parent_rule_components"] == (
        "parent_rule_component_closure_ruled_out"
    )
    assert statuses["direct_pspg_solve_time_same_rule_cross_block_signature"] == (
        "same_rule_cross_block_broad_union_consistent_replayed_insufficient"
    )
    direct_pspg = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "direct_pspg_pressure_gradient_support_topology"
    )
    assert direct_pspg["evidence"][1]["finding"] == (
        "mixed_isolated_and_coherent_direct_pspg_formulation_targets"
    )
    assert direct_pspg["evidence"][1]["class_counts"] == {
        "coherent_direct_pspg_pressure_action_patch": 1,
        "isolated_direct_pspg_row_with_ghost_penalty_branch": 1,
    }
    assert direct_pspg["evidence"][1]["recommended_next_predicate"]["key"] == (
        "direct_support_gap_or_same_sign_pressure_action_patch"
    )
    assert direct_pspg["evidence"][1]["predicate_derivation_readiness"] == (
        "coverage_complete_but_no_formulation_side_derivation"
    )
    assert direct_pspg["evidence"][1][
        "complete_formulation_ready_candidate_keys"
    ] == []
    assert direct_pspg["evidence"][4][
        "retained_direct_support_gap_patch_ratio"
    ] == 2.16
    assert direct_pspg["evidence"][4][
        "full_volume_direct_support_gap_patch_ratio"
    ] == 2.17
    assert direct_pspg["evidence"][4]["retained_preserves_hydrostatic_balance"]
    assert direct_pspg["evidence"][4]["retained_preserves_constant_pressure_null"]
    assert direct_pspg["evidence"][5]["finding"] == (
        "trace_only_support_ruled_out_recent_pruned_volume_not_direct_"
        "trace_only_driver"
    )
    assert direct_pspg["evidence"][5][
        "trace_only_cut_adjacent_support_ruled_out_before_guards"
    ]
    assert direct_pspg["evidence"][5]["pruned_generated_volume_cases"] == [
        "test10"
    ]
    assert direct_pspg["evidence"][6]["finding"] == (
        "support_gap_graph_completion_selectors_overbroad_and_test02_unstable"
    )
    assert direct_pspg["evidence"][6]["overbroad_modes"] == [
        "shared_row_schur_support_gap_patch_completion"
    ]
    assert direct_pspg["evidence"][6]["test02_unstable_modes"] == [
        "shared_row_schur_support_gap_patch_completion"
    ]
    assert direct_pspg["evidence"][6]["test10_guard_clear_modes"] == [
        "shared_row_schur_support_gap_patch_completion"
    ]
    assert direct_pspg["evidence"][6]["direct_target_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][7]["finding"] == (
        "narrow_formulation_side_candidate_identified_needs_global_emission"
    )
    assert direct_pspg["evidence"][7]["preferred_next_candidate"]["key"] == (
        "sparse_direct_self_or_same_sign_pressure_action_patch"
    )
    assert direct_pspg["evidence"][7]["preferred_next_candidate"][
        "production_readiness"
    ] == "formulation_candidate_pending_global_solve_time_emission"
    assert direct_pspg["evidence"][7]["exact_audited_candidate_keys"] == [
        "sparse_direct_self_or_same_sign_pressure_action_patch"
    ]
    assert "same_sign_pressure_action_patch" in direct_pspg["evidence"][7][
        "partial_candidate_keys"
    ]
    assert direct_pspg["evidence"][7]["direct_target_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert "sampled top rows" in direct_pspg["evidence"][7][
        "current_artifact_limitation"
    ]
    assert direct_pspg["evidence"][8]["finding"] == (
        "candidate_emission_covers_audited_targets"
    )
    assert direct_pspg["evidence"][8]["case_findings"] == {
        "test02": "candidate_emitted_covers_audited_targets",
        "test10": "candidate_emitted_covers_audited_targets",
    }
    assert direct_pspg["evidence"][8]["preferred_candidate_counts"] == {
        "test02": 866,
        "test10": 251,
    }
    assert direct_pspg["evidence"][8]["covered_direct_target_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][8]["candidate_list_truncated"] == {
        "test02": False,
        "test10": False,
    }
    assert direct_pspg["evidence"][8]["missing_case_labels"] == []
    assert direct_pspg["evidence"][9]["finding"] == (
        "global_candidate_selector_overbroad_matrix_proxy_not_formulation_ready"
    )
    assert direct_pspg["evidence"][9]["preferred_to_target_ratios"] == {
        "test02": 866 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][9]["sparse_direct_self_to_target_ratios"] == {
        "test02": 545 / 7,
        "test10": 217 / 12,
    }
    assert direct_pspg["evidence"][9][
        "direct_self_support_ratio_gate_finding"
    ] == "direct_self_support_ratio_gate_misses_targets"
    assert direct_pspg["evidence"][9][
        "direct_self_support_ratio_case_findings"
    ] == {
        "test02": "sparse_or_moderate_direct_self_ratio_gate_misses_targets",
        "test10": "sparse_or_moderate_direct_self_ratio_gate_overbroad",
    }
    assert direct_pspg["evidence"][9][
        "sparse_or_moderate_direct_self_ratio_to_target_ratios"
    ] == {
        "test02": 572 / 7,
        "test10": 217 / 12,
    }
    assert direct_pspg["evidence"][9][
        "sparse_or_moderate_direct_self_ratio_covers_targets"
    ] == {
        "test02": False,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "sparse_or_moderate_direct_self_ratio_selector_overbroad"
    ] == {
        "test02": True,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "graph_local_support_ratio_gate_finding"
    ] == "graph_local_support_ratio_gate_misses_targets"
    assert direct_pspg["evidence"][9][
        "graph_local_support_ratio_case_findings"
    ] == {
        "test02": "graph_local_moderate_direct_self_ratio_gate_misses_targets",
        "test10": "graph_local_moderate_direct_self_ratio_gate_overbroad",
    }
    assert direct_pspg["evidence"][9][
        "graph_local_moderate_direct_self_ratio_to_target_ratios"
    ] == {
        "test02": 584 / 7,
        "test10": 211 / 12,
    }
    assert direct_pspg["evidence"][9][
        "graph_local_moderate_direct_self_ratio_covers_targets"
    ] == {
        "test02": False,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "graph_local_moderate_direct_self_ratio_selector_overbroad"
    ] == {
        "test02": True,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_pressure_action_radius1_gate_finding"
    ] == "sparse_seeded_pressure_action_radius1_gate_overbroad"
    assert direct_pspg["evidence"][9][
        "sparse_seeded_pressure_action_radius2_gate_finding"
    ] == "sparse_seeded_pressure_action_radius2_gate_overbroad"
    assert direct_pspg["evidence"][9][
        "sparse_seeded_pressure_action_radius1_case_findings"
    ] == {
        "test02": "sparse_seeded_pressure_action_radius1_gate_overbroad",
        "test10": "sparse_seeded_pressure_action_radius1_gate_overbroad",
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_pressure_action_radius1_to_target_ratios"
    ] == {
        "test02": 818 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_pressure_action_radius2_to_target_ratios"
    ] == {
        "test02": 866 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_pressure_action_radius1_covers_targets"
    ] == {
        "test02": True,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "pressure_action_moderate_degree_gate_finding"
    ] == "pressure_action_moderate_degree_gate_misses_targets"
    assert direct_pspg["evidence"][9][
        "pressure_action_moderate_sum_ratio_gate_finding"
    ] == "pressure_action_moderate_sum_ratio_gate_misses_targets"
    assert direct_pspg["evidence"][9][
        "pressure_action_self_dominant_gate_finding"
    ] == "pressure_action_self_dominant_gate_misses_targets"
    assert direct_pspg["evidence"][9][
        "pressure_action_moderate_degree_to_target_ratios"
    ] == {
        "test02": 167 / 7,
        "test10": 99 / 12,
    }
    assert direct_pspg["evidence"][9][
        "pressure_action_moderate_degree_covers_targets"
    ] == {
        "test02": False,
        "test10": False,
    }
    assert direct_pspg["evidence"][9][
        "pressure_action_moderate_sum_ratio_to_target_ratios"
    ] == {
        "test02": 561 / 7,
        "test10": 212 / 12,
    }
    assert direct_pspg["evidence"][9][
        "pressure_action_moderate_sum_ratio_covers_targets"
    ] == {
        "test02": False,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "pressure_action_self_dominant_to_target_ratios"
    ] == {
        "test02": 1 / 7,
        "test10": 0.0,
    }
    assert direct_pspg["evidence"][9][
        "pressure_action_self_dominant_covers_targets"
    ] == {
        "test02": False,
        "test10": False,
    }
    assert direct_pspg["evidence"][9][
        "matrix_pressure_action_covers_all_direct_rows"
    ] == {
        "test02": True,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_matrix_pressure_action_component_counts"
    ] == {
        "test02": 866,
        "test10": 251,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_matrix_pressure_action_component_to_target_ratios"
    ] == {
        "test02": 866 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_matrix_pressure_action_component_covers_targets"
    ] == {
        "test02": True,
        "test10": True,
    }
    assert direct_pspg["evidence"][9][
        "sparse_seeded_matrix_pressure_action_component_selector_overbroad"
    ] == {
        "test02": True,
        "test10": True,
    }
    assert direct_pspg["evidence"][9]["case_findings"] == {
        "test02": "raw_global_candidate_selector_overbroad",
        "test10": "raw_global_candidate_selector_overbroad",
    }
    assert direct_pspg["evidence"][10]["finding"] == (
        "mesh_boundary_incident_support_selectors_miss_audited_targets"
    )
    assert direct_pspg["evidence"][10]["profile_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][10]["selector_findings"][
        "preferred_boundary_only"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][10]["selected_counts_by_selector"][
        "preferred_boundary_only"
    ] == {
        "test02": 358,
        "test10": 188,
    }
    assert direct_pspg["evidence"][10]["covered_target_counts_by_selector"][
        "preferred_boundary_only"
    ] == {
        "test02": 3,
        "test10": 9,
    }
    assert direct_pspg["evidence"][10]["covered_target_counts_by_selector"][
        "preferred_one_cell_boundary"
    ] == {
        "test02": 0,
        "test10": 0,
    }
    named_face_provenance = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_named_face_provenance_selectivity_"
            "20260607.json"
        ),
    )
    assert named_face_provenance["finding"] == (
        "direct_pspg_named_face_provenance_selectors_not_formulation_ready"
    )
    assert named_face_provenance["status"] == (
        "named_face_boundary_gate_ruled_out"
    )
    assert named_face_provenance["case_findings"] == {
        "test02": "named_face_provenance_selectors_overbroad_or_miss_targets",
        "test10": "named_face_provenance_selectors_overbroad_or_miss_targets",
    }
    assert named_face_provenance["target_named_faces_by_case"] == {
        "test02": ["obstacle", "wall_front", "wall_top"],
        "test10": ["wall_back", "wall_front", "wall_right"],
    }
    assert named_face_provenance["target_face_classes_by_case"]["test10"] == [
        "multi_face_intersection",
        "named_face_intersection",
        "no_named_face",
        "single_named_face",
    ]
    assert named_face_provenance["selected_counts_by_case_selector"][
        "test02"
    ]["preferred_target_named_face_union"] == 264
    assert named_face_provenance["covered_target_counts_by_case_selector"][
        "test10"
    ]["preferred_target_named_face_union"] == 9
    assert named_face_provenance["profile_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert "named wall/obstacle face membership" in named_face_provenance[
        "next_requirement"
    ]
    assert direct_pspg["evidence"][11]["finding"] == (
        "cut_state_provenance_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][11]["profile_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][11]["selector_findings"][
        "preferred_inactive_point"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][11]["selected_counts_by_selector"][
        "preferred_inactive_point"
    ] == {
        "test02": 634,
        "test10": 120,
    }
    assert direct_pspg["evidence"][11]["covered_target_counts_by_selector"][
        "preferred_inactive_point"
    ] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][11]["covered_target_counts_by_selector"][
        "preferred_cut_incident_support"
    ] == {
        "test02": 1,
        "test10": 0,
    }
    assert direct_pspg["evidence"][11]["target_wet_support_counts_by_selector"][
        "preferred_inactive_point"
    ] == {
        "test02": {
            "dry_only_incident_support": 6,
            "mixed_cut_dry_incident_support": 1,
        },
        "test10": {"dry_only_incident_support": 12},
    }
    assert direct_pspg["evidence"][12]["finding"] == (
        "same_sign_patch_blocked_by_pressure_update_dependency_and_"
        "preupdate_proxies"
    )
    assert direct_pspg["evidence"][12][
        "preferred_candidate_depends_on_pressure_update"
    ]
    assert direct_pspg["evidence"][12][
        "all_exact_candidates_depend_on_pressure_update"
    ]
    assert direct_pspg["evidence"][12][
        "complete_non_update_dependent_candidate_keys"
    ] == []
    assert direct_pspg["evidence"][12]["all_preupdate_proxy_gates_failed"]
    assert "sparse_seeded_pressure_action_radius1_gate_finding" in direct_pspg[
        "evidence"
    ][12]["failed_preupdate_proxy_gate_keys"]
    assert direct_pspg["evidence"][12]["cross_policy_patch_finding"] == (
        "cross_policy_patch_evidence_is_post_update_diagnostic_only"
    )
    assert direct_pspg["evidence"][12]["cross_policy_patch_case_findings"] == {
        "test02": "cross_policy_patch_visible_only_after_pressure_disabled_update",
        "test10": "no_full_gradient_isolated_direct_rows",
    }
    assert direct_pspg["evidence"][12]["cross_policy_patch_dofs"]["test02"] == [
        10676,
        10668,
        10677,
        10680,
    ]
    assert direct_pspg["evidence"][13]["finding"] == (
        "active_pressure_support_topology_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][13]["selector_findings"][
        "constrained_pressure_neighbor"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][13]["selected_counts_by_selector"][
        "constrained_pressure_neighbor"
    ] == {
        "test02": 0,
        "test10": 0,
    }
    assert direct_pspg["evidence"][13]["covered_target_counts_by_selector"][
        "constrained_pressure_neighbor"
    ] == {
        "test02": 0,
        "test10": 0,
    }
    assert direct_pspg["evidence"][13]["selector_findings"][
        "sparse_unconstrained_direct_self"
    ] == "selector_overbroad_or_miss_targets"
    assert direct_pspg["evidence"][13]["selected_to_target_ratios_by_selector"][
        "sparse_unconstrained_direct_self"
    ] == {
        "test02": 545 / 7,
        "test10": 217 / 12,
    }
    assert direct_pspg["evidence"][14]["finding"] == (
        "residual_sign_pressure_action_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][14]["selector_findings"][
        "residual_sign_pressure_action"
    ] == "selector_overbroad_or_miss_targets"
    assert direct_pspg["evidence"][14]["selected_to_target_ratios_by_selector"][
        "sparse_direct_self_or_residual_sign_pressure_action"
    ] == {
        "test02": 545 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][14]["residual_signal_by_case"]["test10"][
        "residual_sign_pressure_action_edge_count"
    ] == 354
    assert direct_pspg["evidence"][15]["finding"] == (
        "direct_pspg_null_balance_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][15]["selector_findings"][
        "high_direct_self_row_sum_leak"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][15]["selector_findings"][
        "null_preserving_direct_self"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][15]["selected_to_target_ratios_by_selector"][
        "balanced_diag_direct_self"
    ] == {
        "test02": 866 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][15]["null_balance_by_case"]["test02"][
        "max_direct_self_row_sum_leak_ratio"
    ] == 0.03
    assert direct_pspg["evidence"][16]["finding"] == (
        "direct_pspg_coupled_patch_graph_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][16]["selective_selector_keys"] == []
    assert direct_pspg["evidence"][16]["selector_findings"][
        "pressure_action_low_two_hop"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][16]["selector_findings"][
        "pressure_action_zero_clustering"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][16]["selected_counts_by_selector"][
        "pressure_action_high_two_hop"
    ] == {
        "test02": 796,
        "test10": 239,
    }
    assert direct_pspg["evidence"][16]["covered_target_counts_by_selector"][
        "pressure_action_articulation"
    ] == {
        "test02": 0,
        "test10": 0,
    }
    assert direct_pspg["evidence"][16]["graph_topology_by_case"]["test02"][
        "matrix_pressure_action_max_two_hop_completion_count"
    ] == 18
    assert direct_pspg["evidence"][17]["finding"] == (
        "direct_pspg_cut_volume_row_provenance_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][17]["selective_selector_keys"] == []
    assert direct_pspg["evidence"][17]["candidate_support_class_counts_by_case"][
        "test02"
    ] == {
        "full_cell_only_support": 426,
        "mixed_partial_and_full_cell_support": 210,
        "partial_cut_only_support": 230,
    }
    assert direct_pspg["evidence"][17]["selector_findings"][
        "cut_volume_full_cell_only_support"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][17]["selected_counts_by_selector"][
        "cut_volume_full_cell_only_support"
    ] == {
        "test02": 426,
        "test10": 125,
    }
    assert direct_pspg["evidence"][17]["covered_target_counts_by_selector"][
        "cut_volume_partial_rule_support"
    ] == {
        "test02": 0,
        "test10": 0,
    }
    assert direct_pspg["evidence"][18]["finding"] == (
        "direct_pspg_cut_volume_local_matrix_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][18]["selective_selector_keys"] == []
    assert direct_pspg["evidence"][18]["thresholds_by_case"]["test02"][
        "total_row_abs_sum_p25"
    ] == 7.885742e-10
    assert direct_pspg["evidence"][18]["target_profiles_by_case"]["test10"][
        "3526"
    ]["max_rule_row_abs_fraction"] == 0.2950280847937581
    assert direct_pspg["evidence"][18]["selector_findings"][
        "local_matrix_full_cell_dominant_abs_fraction"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][18]["selected_counts_by_selector"][
        "local_matrix_low_total_abs_sum_p25"
    ] == {
        "test02": 217,
        "test10": 63,
    }
    assert direct_pspg["evidence"][18]["covered_target_counts_by_selector"][
        "local_matrix_full_cell_dominant_abs_fraction"
    ] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][19]["finding"] == (
        "direct_pspg_cut_volume_local_coupling_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][19]["selective_selector_keys"] == []
    assert direct_pspg["evidence"][19]["thresholds_by_case"]["test02"][
        "velocity_to_pressure_abs_ratio_p90"
    ] == 1.0562530747823181e-05
    assert direct_pspg["evidence"][19]["target_profiles_by_case"]["test10"][
        "3526"
    ]["velocity_to_pressure_abs_ratio"] == 0.0
    assert direct_pspg["evidence"][19]["selector_findings"][
        "cross_field_zero_velocity_action"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][19]["selected_counts_by_selector"][
        "cross_field_high_velocity_pressure_ratio_p90"
    ] == {
        "test02": 88,
        "test10": 26,
    }
    assert direct_pspg["evidence"][19]["covered_target_counts_by_selector"][
        "cross_field_zero_velocity_action"
    ] == {
        "test02": 0,
        "test10": 3,
    }
    assert direct_pspg["evidence"][20]["finding"] == (
        "direct_pspg_cut_volume_parent_graph_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][20]["selective_selector_keys"] == []
    assert direct_pspg["evidence"][20]["thresholds_by_case"]["test10"][
        "degree_p25"
    ] == 8.0
    assert direct_pspg["evidence"][20]["target_profiles_by_case"]["test02"][
        "10676"
    ]["row_parent_graph_clustering"] == 0.8
    assert direct_pspg["evidence"][20]["selector_findings"][
        "parent_graph_degree_tail"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][20]["selected_counts_by_selector"][
        "parent_graph_degree_tail"
    ] == {
        "test02": 809,
        "test10": 247,
    }
    assert direct_pspg["evidence"][20]["covered_target_counts_by_selector"][
        "parent_graph_high_degree_low_clustering"
    ] == {
        "test02": 6,
        "test10": 0,
    }
    assert direct_pspg["evidence"][21]["finding"] == (
        "direct_pspg_cut_volume_composite_selectors_overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][21]["selective_selector_keys"] == []
    assert direct_pspg["evidence"][21]["thresholds_by_case"]["test02"][
        "velocity_ratio_p90"
    ] == 1.0562530747823181e-05
    assert direct_pspg["evidence"][21]["target_profiles_by_case"]["test10"][
        "3526"
    ]["velocity_to_pressure_abs_ratio"] == 0.0
    assert direct_pspg["evidence"][21]["selector_findings"][
        "composite_graph_bimodal_tail"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][21]["selected_counts_by_selector"][
        "composite_graph_bimodal_tail"
    ] == {
        "test02": 793,
        "test10": 169,
    }
    assert direct_pspg["evidence"][21]["covered_target_counts_by_selector"][
        "composite_graph_tail_and_ratio_tail"
    ] == {
        "test02": 7,
        "test10": 5,
    }
    assert direct_pspg["evidence"][21]["selected_to_target_ratios_by_selector"][
        "composite_twohop_graph_ratio_tail"
    ] == {
        "test02": 13.857142857142858,
        "test10": 2.9166666666666665,
    }
    assert direct_pspg["evidence"][22]["finding"] == (
        "direct_pspg_cut_volume_column_support_evidence_ready"
    )
    assert direct_pspg["evidence"][22]["missing_case_labels"] == []
    assert direct_pspg["evidence"][22]["case_log_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][22]["latest_batch_entry_counts"] == {
        "test02": 13432,
        "test10": 2880,
    }
    assert direct_pspg["evidence"][22]["profiled_candidate_counts"] == {
        "test02": 866,
        "test10": 251,
    }
    assert direct_pspg["evidence"][22]["profiled_target_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][22]["unprofiled_target_global_dofs_by_case"] == {
        "test02": [],
        "test10": [],
    }
    assert direct_pspg["evidence"][22][
        "candidate_column_support_class_counts_by_case"
    ] == {
        "test02": {"null_preserving_negative_offdiag_stencil": 866},
        "test10": {"null_preserving_negative_offdiag_stencil": 251},
    }
    assert direct_pspg["evidence"][22][
        "target_column_support_class_counts_by_case"
    ] == {
        "test02": {"null_preserving_negative_offdiag_stencil": 7},
        "test10": {"null_preserving_negative_offdiag_stencil": 12},
    }
    assert "signed sampled column neighborhoods" in direct_pspg["evidence"][22][
        "next_requirement"
    ]
    assert direct_pspg["evidence"][23]["finding"] == (
        "direct_pspg_cut_volume_column_support_selectors_"
        "overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][23]["selective_selector_keys"] == []
    assert "column_candidate_neighbor_closed" in direct_pspg["evidence"][23][
        "overbroad_selector_keys"
    ]
    assert "column_mean_edge_abs_tail" in direct_pspg["evidence"][23][
        "miss_selector_keys"
    ]
    assert direct_pspg["evidence"][23]["thresholds_by_case"]["test02"][
        "candidate_degree_p25"
    ] == 5.0
    assert direct_pspg["evidence"][23]["target_profiles_by_case"]["test02"][
        "10676"
    ]["column_graph_component_size"] == 866
    assert direct_pspg["evidence"][23]["selector_findings"][
        "column_low_candidate_degree_p25"
    ] == "selector_overbroad_or_miss_targets"
    assert direct_pspg["evidence"][23]["selected_counts_by_selector"][
        "column_candidate_neighbor_closed"
    ] == {
        "test02": 848,
        "test10": 245,
    }
    assert direct_pspg["evidence"][23]["covered_target_counts_by_selector"][
        "column_mean_edge_abs_tail"
    ] == {
        "test02": 7,
        "test10": 3,
    }
    assert direct_pspg["evidence"][23]["selected_to_target_ratios_by_selector"][
        "column_null_preserving_negative_offdiag_class"
    ] == {
        "test02": 123.71428571428571,
        "test10": 20.916666666666668,
    }
    assert "element-local pressure-gradient geometry" in direct_pspg["evidence"][23][
        "next_requirement"
    ]
    assert direct_pspg["evidence"][24]["finding"] == (
        "direct_pspg_cut_volume_column_geometry_selectors_"
        "overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][24]["selective_selector_keys"] == []
    assert "geometry_has_diagonal_edges" in direct_pspg["evidence"][24][
        "overbroad_selector_keys"
    ]
    assert "geometry_mean_ref_edge_length_tail" in direct_pspg["evidence"][24][
        "miss_selector_keys"
    ]
    assert direct_pspg["evidence"][24]["geometry_field_entry_counts_by_case"] == {
        "test02": 13432,
        "test10": 2880,
    }
    assert direct_pspg["evidence"][24][
        "candidate_reference_geometry_class_counts_by_case"
    ]["test02"]["mixed_axis_diagonal_reference_edges"] == 826
    assert direct_pspg["evidence"][24][
        "target_reference_geometry_class_counts_by_case"
    ]["test10"] == {
        "diagonal_only_reference_edges": 1,
        "mixed_axis_diagonal_reference_edges": 11,
    }
    assert direct_pspg["evidence"][24]["thresholds_by_case"]["test10"][
        "axis_aligned_edge_fraction_p25"
    ] == 0.1111111111111111
    assert direct_pspg["evidence"][24]["target_profiles_by_case"]["test10"][
        "3526"
    ]["reference_geometry_class"] == "diagonal_only_reference_edges"
    assert direct_pspg["evidence"][24]["selector_findings"][
        "geometry_mixed_axis_diagonal_edges"
    ] == "selector_overbroad_or_miss_targets"
    assert direct_pspg["evidence"][24]["selected_counts_by_selector"][
        "geometry_has_diagonal_edges"
    ] == {
        "test02": 864,
        "test10": 249,
    }
    assert direct_pspg["evidence"][24]["covered_target_counts_by_selector"][
        "geometry_mean_ref_edge_length_tail"
    ] == {
        "test02": 0,
        "test10": 9,
    }
    assert direct_pspg["evidence"][24]["selected_to_target_ratios_by_selector"][
        "geometry_has_diagonal_edges"
    ] == {
        "test02": 123.42857142857143,
        "test10": 20.75,
    }
    assert "quadrature/cut-interface geometry" in direct_pspg["evidence"][24][
        "next_requirement"
    ]
    assert direct_pspg["evidence"][25]["finding"] == (
        "direct_pspg_cut_volume_quadrature_geometry_selectors_"
        "overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][25]["selective_selector_keys"] == []
    assert "qgeom_uniform_full_cell_class" in direct_pspg["evidence"][25][
        "overbroad_selector_keys"
    ]
    assert "qgeom_radius_tail" in direct_pspg["evidence"][25][
        "miss_selector_keys"
    ]
    assert direct_pspg["evidence"][25][
        "cut_qpoint_field_entry_counts_by_case"
    ] == {
        "test02": 13432,
        "test10": 2880,
    }
    assert direct_pspg["evidence"][25][
        "candidate_cut_qpoint_geometry_class_counts_by_case"
    ]["test02"] == {
        "mixed_qpoint_geometry": 408,
        "uniform_full_cell_qpoint_geometry": 458,
    }
    assert direct_pspg["evidence"][25][
        "target_cut_qpoint_geometry_class_counts_by_case"
    ] == {
        "test02": {"uniform_full_cell_qpoint_geometry": 7},
        "test10": {"uniform_full_cell_qpoint_geometry": 12},
    }
    assert direct_pspg["evidence"][25]["thresholds_by_case"]["test10"][
        "parent_cell_count_p75"
    ] == 12.0
    assert direct_pspg["evidence"][25]["target_profiles_by_case"]["test10"][
        "3526"
    ]["cut_qpoint_geometry_class"] == "uniform_full_cell_qpoint_geometry"
    assert direct_pspg["evidence"][25]["selector_findings"][
        "qgeom_radius_tail"
    ] == "selector_overbroad_or_miss_targets"
    assert direct_pspg["evidence"][25]["selected_counts_by_selector"][
        "qgeom_uniform_full_cell_class"
    ] == {
        "test02": 458,
        "test10": 128,
    }
    assert direct_pspg["evidence"][25]["covered_target_counts_by_selector"][
        "qgeom_row_to_centroid_distance_tail"
    ] == {
        "test02": 1,
        "test10": 8,
    }
    assert direct_pspg["evidence"][25]["selected_to_target_ratios_by_selector"][
        "qgeom_uniform_full_cell_class"
    ] == {
        "test02": 65.42857142857143,
        "test10": 10.666666666666666,
    }
    assert "formulation-derived pressure-gradient support/coupling" in (
        direct_pspg["evidence"][25]["next_requirement"]
    )
    assert direct_pspg["evidence"][26]["finding"] == (
        "direct_pspg_cut_volume_gradient_balance_selectors_"
        "overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][26]["case_log_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][26]["profiled_target_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][26][
        "target_gradient_support_class_counts"
    ] == {
        "test02": {"full_cell_only_gradient_support": 7},
        "test10": {"full_cell_only_gradient_support": 12},
    }
    assert direct_pspg["evidence"][26]["selector_findings"][
        "gradient_balance_full_cell_only"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][26]["selector_findings"][
        "gradient_balance_sampled_sign_mismatch"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][26]["covered_target_counts_by_selector"][
        "gradient_balance_resultant_ratio_tail"
    ] == {
        "test02": 6,
        "test10": 3,
    }
    assert direct_pspg["evidence"][26]["selected_to_target_ratios_by_selector"][
        "gradient_balance_full_cell_only"
    ] == {
        "test02": 60.857142857142854,
        "test10": 10.416666666666666,
    }
    assert "shape-gradient support" in direct_pspg["evidence"][26][
        "next_requirement"
    ]
    assert direct_pspg["evidence"][27]["finding"] == (
        "direct_pspg_cut_volume_gradient_column_graph_selectors_"
        "overbroad_or_miss_targets"
    )
    assert direct_pspg["evidence"][27]["case_log_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][27]["profiled_target_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][27]["target_edge_class_counts"] == {
        "test02": {"reciprocal_all_negative_gradient_stencil": 7},
        "test10": {"reciprocal_all_negative_gradient_stencil": 12},
    }
    assert direct_pspg["evidence"][27]["candidate_edge_class_counts"] == {
        "test02": {
            "missing_candidate_gradient_stencil": 16,
            "reciprocal_all_negative_gradient_stencil": 850,
        },
        "test10": {"reciprocal_all_negative_gradient_stencil": 251},
    }
    assert direct_pspg["evidence"][27]["selector_findings"][
        "gradient_column_graph_reciprocal_negative_stencil"
    ] == "selector_overbroad"
    assert direct_pspg["evidence"][27]["selector_findings"][
        "gradient_column_graph_sign_mismatch"
    ] == "selector_misses_targets"
    assert direct_pspg["evidence"][27]["covered_target_counts_by_selector"][
        "gradient_column_graph_edge_count_tail"
    ] == {
        "test02": 7,
        "test10": 11,
    }
    assert direct_pspg["evidence"][27]["selected_to_target_ratios_by_selector"][
        "gradient_column_graph_reciprocal_negative_stencil"
    ] == {
        "test02": 121.42857142857143,
        "test10": 20.916666666666668,
    }
    assert "edge-level graph thresholding" in direct_pspg["evidence"][27][
        "next_requirement"
    ]
    assert direct_pspg["evidence"][28]["finding"] == (
        "direct_pspg_graph_completion_replay_family_rules_out_"
        "post_assembly_selector_variants"
    )
    assert direct_pspg["evidence"][28]["variant_findings"][
        "least_selector_schur_only"
    ] == "both_guards_still_trigger"
    assert direct_pspg["evidence"][28]["variant_findings"][
        "least_selector_schur_edge_balance"
    ] == "test10_clears_but_test02_unstable"
    assert direct_pspg["evidence"][28]["test10_guard_clear_variants"] == [
        "least_selector_schur_edge_balance"
    ]
    assert direct_pspg["evidence"][28]["test02_unstable_variants"] == [
        "least_selector_schur_edge_balance"
    ]
    assert direct_pspg["evidence"][28]["candidate_counts_by_variant"][
        "least_selector_schur_edge_balance"
    ] == {
        "test02": 304,
        "test10": 68,
    }
    assert direct_pspg["evidence"][28]["accepted_pressure_updates_by_variant"][
        "support_rank_neighborhood_depth1"
    ] == {
        "test02": 366719.9658064514,
        "test10": 319.85947884033067,
    }
    assert direct_pspg["evidence"][28]["case_findings_by_variant"][
        "least_selector_schur_edge_balance"
    ] == {
        "test02": "nonlinear_failed_with_overbroad_patch",
        "test10": "guard_cleared",
    }
    assert "formulation-side direct PSPG pressure-gradient" in (
        direct_pspg["evidence"][28]["next_requirement"]
    )
    graph_tradeoff = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_graph_completion_stability_tradeoff_"
            "20260607.json"
        ),
    )
    assert graph_tradeoff["finding"] == (
        "direct_pspg_graph_completion_stability_tradeoff_rules_out_"
        "post_assembly_fix"
    )
    assert graph_tradeoff["status"] == (
        "post_assembly_schur_balance_tradeoff_ruled_out"
    )
    assert graph_tradeoff["tradeoff_flags"][
        "broad_topology_clears_test10_but_destabilizes_test02"
    ]
    assert graph_tradeoff["tradeoff_flags"][
        (
            "least_selector_schur_stable_but_insufficient_balance_"
            "clears_test10_but_destabilizes_test02"
        )
    ]
    assert graph_tradeoff["least_selector_tradeoff"]["schur_only"][
        "test10"
    ]["accepted_pressure_update_pa"] == 122.46838944778688
    assert graph_tradeoff["least_selector_tradeoff"][
        "schur_edge_balance"
    ]["test10"]["accepted_pressure_update_pa"] == 15.254558181653124
    assert graph_tradeoff["least_selector_tradeoff"][
        "schur_edge_balance"
    ]["test02_nonlinear_failed"]
    assert graph_tradeoff["localized_balance_variant_findings"][
        "low_pressure_degree_balance"
    ]["test10_update_pa"] == 120.9238982982647
    assert "post-assembly Schur fill" in graph_tradeoff["next_requirement"]
    explicit_balance = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_explicit_balance_selector_replays_"
            "20260607.json"
        ),
    )
    assert explicit_balance["finding"] == (
        "direct_pspg_explicit_balance_selectors_rule_out_row_lists_and_"
        "pressure_neighborhoods"
    )
    assert explicit_balance["status"] == "explicit_balance_selectors_ruled_out"
    assert explicit_balance["ruleout_flags"] == {
        "boundary_balance_predicate_misses_latest_bad_rows": True,
        "explicit_row_lists_ruled_out": True,
        "current_pressure_neighborhoods_ruled_out": True,
    }
    assert explicit_balance["boundary_provenance"]["boundary_topology_finding"] == (
        "boundary_top_update_candidates_missing_balance"
    )
    assert explicit_balance["ruled_out_by_variant"]["explicit_shifted_rows"]
    assert explicit_balance["case_findings_by_variant"][
        "explicit_shifted_rows"
    ] == {
        "test02": "nonlinear_failed",
        "test10": "guard_still_triggered",
    }
    assert explicit_balance["accepted_pressure_updates_by_variant"][
        "explicit_direct_rows"
    ] == {
        "test02": 102071.75239899695,
        "test10": 120.642165923368,
    }
    assert explicit_balance["accepted_pressure_updates_by_variant"][
        "explicit_neighborhood_depth2"
    ] == {
        "test02": 103141.83046458055,
        "test10": 118.70937283831643,
    }
    assert explicit_balance["balance_candidate_counts_by_variant"][
        "explicit_neighborhood_depth2"
    ] == {
        "test02": 113,
        "test10": 50,
    }
    assert "current-pressure-neighborhood" in explicit_balance[
        "next_requirement"
    ]
    assert direct_pspg["evidence"][29]["finding"] == (
        "direct_pspg_cut_volume_local_schur_completion_overbroad"
    )
    assert direct_pspg["evidence"][29]["aggregate_selector_finding"] == (
        "selector_overbroad"
    )
    assert direct_pspg["evidence"][29]["case_log_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][29]["selected_counts_by_case"] == {
        "test02": 866,
        "test10": 251,
    }
    assert direct_pspg["evidence"][29]["covered_target_counts_by_case"] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][29]["selected_to_target_ratios_by_case"] == {
        "test02": 866 / 7,
        "test10": 251 / 12,
    }
    assert direct_pspg["evidence"][29]["summary_metrics_by_case"]["test02"][
        "constant_pressure_null_preserving_all"
    ]
    assert direct_pspg["evidence"][29]["summary_metrics_by_case"]["test10"][
        "diagnostic_only_all"
    ]
    assert "local Schur completion alone" in (
        direct_pspg["evidence"][29]["next_requirement"]
    )
    assert direct_pspg["evidence"][30]["finding"] == (
        "direct_pspg_cut_volume_local_edge_balance_overbroad"
    )
    assert direct_pspg["evidence"][30]["aggregate_selector_findings"] == [
        "selector_overbroad",
        "selector_overbroad",
    ]
    assert direct_pspg["evidence"][30]["selector_findings"] == {
        "local_edge_balance_candidate_rows": "selector_overbroad",
        "local_edge_balance_touched_rows": "selector_overbroad",
    }
    assert direct_pspg["evidence"][30]["case_log_status"] == {
        "test02": "ok",
        "test10": "ok",
    }
    assert direct_pspg["evidence"][30]["selected_counts_by_selector"][
        "local_edge_balance_candidate_rows"
    ] == {
        "test02": 865,
        "test10": 248,
    }
    assert direct_pspg["evidence"][30]["selected_counts_by_selector"][
        "local_edge_balance_touched_rows"
    ] == {
        "test02": 866,
        "test10": 251,
    }
    assert direct_pspg["evidence"][30]["covered_target_counts_by_selector"][
        "local_edge_balance_candidate_rows"
    ] == {
        "test02": 7,
        "test10": 12,
    }
    assert direct_pspg["evidence"][30]["selected_to_target_ratios_by_selector"][
        "local_edge_balance_candidate_rows"
    ] == {
        "test02": 865 / 7,
        "test10": 248 / 12,
    }
    assert direct_pspg["evidence"][30]["summary_metrics_by_case"]["test02"][
        "constant_pressure_null_preserving_all"
    ]
    assert direct_pspg["evidence"][30]["summary_metrics_by_case"]["test10"][
        "diagnostic_only_all"
    ]
    assert "local existing-edge balance alone" in (
        direct_pspg["evidence"][30]["next_requirement"]
    )
    shape_tangent_test02 = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "shape_tangent_pressure_update_audit_20260606.json"
        ),
    )
    assert shape_tangent_test02["control_variant"] == "residual_shape_tangent"
    assert shape_tangent_test02["worst_active_or_wet_update_pa"] == (
        366720.75479080016
    )
    assert shape_tangent_test02["worst_active_or_wet_point_index"] == 1172
    shape_tangent_test10 = _evidence_by_suffix(
        direct_pspg,
        (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_"
            "shape_tangent_pressure_update_audit_20260606.json"
        ),
    )
    assert shape_tangent_test10["worst_active_or_wet_update_pa"] == (
        622.70494029817
    )
    cap16_test02 = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "cut_volume_scale_cap16_pressure_update_audit_20260606.json"
        ),
    )
    assert cap16_test02["control_variant"] == (
        "direct_pspg_cut_volume_scale_cap16"
    )
    assert cap16_test02["absolute_threshold_pa"] == 100000.0
    assert cap16_test02["worst_active_or_wet_update_pa"] == 366739.55884526786
    cap16_test10 = _evidence_by_suffix(
        direct_pspg,
        (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_"
            "cut_volume_scale_cap16_pressure_update_audit_20260606.json"
        ),
    )
    assert cap16_test10["worst_active_or_wet_update_pa"] == 629.8487957714515
    tangential_test02 = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_"
            "free_surface_tangential_scale1_pressure_update_audit_20260606.json"
        ),
    )
    assert tangential_test02["control_variant"] == (
        "free_surface_tangential_pressure_gradient"
    )
    assert tangential_test02["worst_active_or_wet_update_pa"] == (
        347889.7856780469
    )
    tangential_test10 = _evidence_by_suffix(
        direct_pspg,
        (
            "test10_replay_cap3_step90_pspg_wall_full_gradient_"
            "free_surface_tangential_scale1_pressure_update_audit_20260606.json"
        ),
    )
    assert tangential_test10["worst_active_or_wet_point_index"] == 7
    assert tangential_test10["worst_active_or_wet_update_pa"] == (
        862.2837968627545
    )
    selector_coverage = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_graph_completion_selector_coverage_20260606.json",
    )
    assert selector_coverage["finding"] == (
        "shifted_pressure_update_rows_escape_weak_row_selector"
    )
    assert selector_coverage["max_update_global_dofs"][
        "test02_existing_edge_cap256"
    ] == 10624
    assert selector_coverage["selector_reasons"][
        "test02_existing_edge"
    ] == "missing_row_support"
    assert selector_coverage["least_selector_thresholds_to_include"][
        "test10_existing_edge_cap256"
    ]["selector"] == "self_threshold"
    assert selector_coverage["casewise_least_widened_threshold_floor"] == {
        "case_count": 4,
        "coupling_threshold": 9.687318259562922e-09,
        "self_threshold": 1.759508340111003e-07,
    }
    assert selector_coverage["single_selector_widened_threshold_floor"] == {
        "case_count": 4,
        "coupling_threshold": 0.0003266812746844929,
        "self_threshold": 0.8026219518508216,
    }
    vocabulary_support = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_formulation_vocabulary_support_20260607.json",
    )
    assert vocabulary_support["finding"] == (
        "form_vocabulary_lacks_direct_pspg_support_topology_handles"
    )
    assert vocabulary_support["status"] == (
        "requires_fe_forms_or_assembly_api_extension"
    )
    assert vocabulary_support["direct_pspg_expression_summary"] == {
        "direct_pressure_gradient_integrand_installed": True,
        "active_volume_measure_used": True,
        "cut_volume_fraction_scale_available": True,
        "free_surface_boundary_terms_separate": True,
    }
    assert vocabulary_support["public_measures"] == [
        "dx",
        "ds",
        "dS",
        "dI",
        "dCutVolume",
    ]
    assert all(vocabulary_support["required_topology_handles_missing"].values())
    assert vocabulary_support["missing_required_topology_handle_count"] == 5
    assert vocabulary_support["required_topology_handle_count"] == 5
    assert "another scalar form multiplier" in (
        vocabulary_support["next_requirement"]
    )
    assembly_api_support = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_assembly_api_support_20260607.json",
    )
    assert assembly_api_support["finding"] == (
        "assembly_api_has_direct_pspg_topology_policy_hook_replay_pending"
    )
    assert assembly_api_support["status"] == "topology_policy_hook_available_replay_pending"
    assert assembly_api_support["assembly_diagnostic_context_fields"] == [
        "operator_tag",
        "source_component_tag",
        "test_field_name",
        "trial_field_name",
    ]
    assert not assembly_api_support["assembly_api_features"][
        "diagnostic_context_is_documented_non_mutating"
    ]
    assert assembly_api_support["assembly_api_features"][
        "direct_pspg_diagnostic_operator_installed"
    ]
    assert assembly_api_support["assembly_api_features"][
        "planned_cut_volume_term_has_source_component_tag"
    ]
    assert assembly_api_support["assembly_api_features"][
        "forms_installer_forwards_source_component_tag_to_cut_volumes"
    ]
    assert assembly_api_support["assembly_api_features"][
        "diagnostic_logs_include_source_component_tag"
    ]
    assert assembly_api_support["assembly_api_features"][
        "production_direct_pspg_subterm_has_source_component_tag"
    ]
    assert assembly_api_support["assembly_api_features"][
        "production_direct_pspg_split_preserves_velocity_tangent"
    ]
    assert assembly_api_support["assembly_api_features"][
        "direct_pspg_topology_policy_log_marks_solve_affecting"
    ]
    assert assembly_api_support["assembly_api_features"][
        "direct_pspg_topology_policy_mutates_before_global_insert"
    ]
    assert "source_component_tag" in (
        assembly_api_support["planned_cut_volume_term_fields"]
    )
    assert not assembly_api_support["required_api_handles_missing"][
        "planned_cut_volume_source_component_tag"
    ]
    assert not assembly_api_support["required_api_handles_missing"][
        "production_subterm_provenance_tag"
    ]
    assert not assembly_api_support["required_api_handles_missing"][
        "direct_pspg_topology_policy_api"
    ]
    assert not assembly_api_support["required_api_handles_missing"][
        "solve_affecting_local_matrix_mutation_hook"
    ]
    assert not assembly_api_support["required_api_handles_missing"][
        "composite_term_provenance_for_fused_cut_volume_blocks"
    ]
    assert assembly_api_support["missing_required_api_handle_count"] == 0
    assert assembly_api_support["required_api_handle_count"] == 10
    assert "replay windows" in (
        assembly_api_support["next_requirement"]
    )
    topology_policy_replay_pair = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_topology_policy_replay_pair_20260607.json",
    )
    assert topology_policy_replay_pair["finding"] == (
        "direct_pspg_topology_policy_schur_edge_balance_"
        "replay_pair_does_not_clear_guards"
    )
    assert topology_policy_replay_pair["status"] == (
        "policy_hook_exercised_mode_ruled_out_as_complete_fix"
    )
    assert topology_policy_replay_pair["policy"] == "local_schur_edge_balance"
    assert topology_policy_replay_pair["policy_hook_exercised"]
    assert topology_policy_replay_pair["policy_log_counts"] == {
        "test02": 3352,
        "test10": 720,
    }
    assert topology_policy_replay_pair["pressure_update_guard_cleared"] == {
        "test02": False,
        "test10": False,
    }
    case_summaries = {
        case["label"]: case
        for case in topology_policy_replay_pair["case_summaries"]
    }
    assert case_summaries["test02"]["worst_active_or_wet_update_pa"] == (
        176844.2140471727
    )
    assert case_summaries["test02"]["worst_active_or_wet_support_class"] == (
        "tiny_cut_supported"
    )
    assert case_summaries["test10"]["worst_active_or_wet_update_pa"] == (
        522.4172735486616
    )
    assert case_summaries["test10"]["worst_active_or_wet_support_class"] == (
        "full_wet_supported"
    )
    topology_policy_mode_replays = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_topology_policy_mode_replays_20260607.json",
    )
    assert topology_policy_mode_replays["finding"] == (
        "direct_pspg_topology_policy_local_modes_do_not_clear_guards"
    )
    assert topology_policy_mode_replays["status"] == (
        "local_topology_policy_family_ruled_out_as_complete_fix"
    )
    assert topology_policy_mode_replays["policies_tested"] == [
        "local_schur_completion",
        "local_edge_balance",
        "local_schur_edge_balance",
    ]
    assert topology_policy_mode_replays["policy_hook_exercised"]
    assert topology_policy_mode_replays["pressure_update_guard_cleared"][
        "local_schur_completion"
    ] == {
        "test02": False,
        "test10": False,
    }
    assert topology_policy_mode_replays["pressure_update_guard_cleared"][
        "local_edge_balance"
    ] == {
        "test02": False,
        "test10": False,
    }
    mode_results = {
        (result["case"], result["policy"]): result
        for result in topology_policy_mode_replays["case_policy_results"]
    }
    assert mode_results[("test02", "local_schur_completion")][
        "worst_active_or_wet_update_pa"
    ] == 176849.84039557964
    assert mode_results[("test02", "local_edge_balance")][
        "worst_active_or_wet_update_pa"
    ] == 176848.02921204976
    assert mode_results[("test10", "local_schur_completion")][
        "worst_active_or_wet_update_pa"
    ] == 590.7292901816519
    assert mode_results[("test10", "local_edge_balance")][
        "worst_active_or_wet_update_pa"
    ] == 530.3194043612839
    topology_policy_application_effect = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_topology_policy_application_effect_"
            "20260607.json"
        ),
    )
    assert topology_policy_application_effect["finding"] == (
        "direct_pspg_topology_policy_application_effect_rules_out_"
        "underapplication"
    )
    assert topology_policy_application_effect["status"] == (
        "local_matrix_policy_applies_but_is_not_sufficient_fix"
    )
    assert topology_policy_application_effect["all_replays_trigger_guard"]
    assert topology_policy_application_effect[
        "all_test10_signature_replays_mutate_selected_records"
    ]
    assert topology_policy_application_effect["best_updates"][
        "test02_broad"
    ] == {
        "policy": "local_schur_edge_balance",
        "update_pa": 176844.2140471727,
    }
    assert topology_policy_application_effect["best_updates"][
        "test10_signature"
    ] == {
        "policy": "local_schur_edge_balance",
        "update_pa": 604.7126561932914,
    }
    test10_policy_effect = topology_policy_application_effect[
        "test10_broad_vs_signature_row_filter"
    ]["local_schur_edge_balance"]
    assert test10_policy_effect["broad_matrix_mutated_count"] == 720
    assert test10_policy_effect["signature_matrix_mutated_count"] == 258
    assert test10_policy_effect[
        "signature_selected_records_matrix_mutated_count"
    ] == 258
    assert test10_policy_effect["signature_minus_broad_update_pa"] == (
        82.29538264462985
    )
    assert "row-filter coverage issue" in topology_policy_application_effect[
        "next_requirement"
    ]
    topology_policy_scope_scale = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_topology_policy_scope_scale_20260607.json",
    )
    assert topology_policy_scope_scale["finding"] == (
        "direct_pspg_topology_policy_scope_scale_rules_out_exact_row_filter"
    )
    assert topology_policy_scope_scale["status"] == (
        "broad_cosupport_mutation_helpful_but_insufficient"
    )
    assert topology_policy_scope_scale[
        "same_case_no_policy_test10_update_pa"
    ] == 622.6094100310928
    assert topology_policy_scope_scale["all_replays_trigger_guard"]
    assert topology_policy_scope_scale[
        "signature_rows_worse_than_broad_for_all_test10_modes"
    ]
    scope_combined = topology_policy_scope_scale[
        "test10_broad_vs_signature_row_filter"
    ]["local_schur_edge_balance"]
    assert scope_combined["no_policy_to_broad_improvement_pa"] == (
        100.19213648243124
    )
    assert scope_combined["signature_minus_broad_update_pa"] == (
        82.29538264462985
    )
    assert scope_combined["signature_to_broad_policy_log_fraction"] == (
        0.36666666666666664
    )
    assert scope_combined["signature_to_broad_topology_edge_weight_fraction"] == (
        0.18883839042222791
    )
    assert topology_policy_scope_scale["test02_broad_policy_scope"][
        "local_schur_edge_balance"
    ]["support_class"] == "tiny_cut_supported"
    assert "physical boundary support rule" in topology_policy_scope_scale[
        "next_requirement"
    ]
    topology_policy_parent_scope = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_topology_policy_parent_scope_20260607.json",
    )
    assert topology_policy_parent_scope["finding"] == (
        "direct_pspg_topology_policy_parent_scope_rules_out_exact_parent_subset"
    )
    assert topology_policy_parent_scope["status"] == (
        "broad_parent_cosupport_required_but_insufficient"
    )
    assert topology_policy_parent_scope[
        "same_case_no_policy_test10_update_pa"
    ] == 622.6094100310928
    assert topology_policy_parent_scope[
        "all_test10_signature_parent_rule_sets_are_strict_broad_subsets"
    ]
    assert topology_policy_parent_scope[
        "all_test10_broad_only_rule_weight_share_above_half"
    ]
    parent_combined = topology_policy_parent_scope[
        "test10_parent_rule_scope"
    ]["local_schur_edge_balance"]
    assert parent_combined["signature_minus_broad_update_pa"] == (
        82.29538264462985
    )
    rule_scope = parent_combined["rule_scope"]
    assert rule_scope["broad_key_count"] == 720
    assert rule_scope["signature_key_count"] == 264
    assert rule_scope["broad_only_key_count"] == 456
    assert rule_scope["signature_to_broad_key_fraction"] == (
        0.36666666666666664
    )
    assert rule_scope["broad_only_topology_edge_weight_sum_fraction"] == (
        0.5734631284834748
    )
    assert rule_scope[
        "signature_to_broad_overlap_topology_edge_weight_sum_fraction"
    ] == 0.44272465766165725
    assert topology_policy_parent_scope["test02_broad_parent_rule_scope"][
        "local_schur_edge_balance"
    ]["rule_scope"]["broad_cut_cell_record_count"] == 1110.0
    assert "connected support-patch" in topology_policy_parent_scope[
        "next_requirement"
    ]
    topology_policy_parent_subset = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_topology_policy_parent_subset_replay_"
            "readiness_20260607.json"
        ),
    )
    assert topology_policy_parent_subset["finding"] == (
        "direct_pspg_signature_parent_subset_replay_ready"
    )
    assert topology_policy_parent_subset["status"] == (
        "run_signature_parent_full_local_replay"
    )
    assert topology_policy_parent_subset["source_hook"][
        "parent_cell_filter_api_present"
    ]
    assert topology_policy_parent_subset[
        "same_signature_parent_set_all_policies"
    ]
    assert topology_policy_parent_subset["signature_parent_cell_count"] == 264
    assert topology_policy_parent_subset["signature_parent_cell_ranges"] == (
        "1-4,6-125,240-245,356-357,1441-1444,1446-1565,"
        "1680-1685,1796-1797"
    )
    assert topology_policy_parent_subset["parent_scope"][
        "strict_parent_rule_subset"
    ]
    assert topology_policy_parent_subset["parent_scope"][
        "combined_rule_scope"
    ]["broad_only_key_count"] == 456
    assert "no global row DOF filter" in topology_policy_parent_subset[
        "next_requirement"
    ]
    topology_policy_parent_subset_replay = _evidence_by_suffix(
        direct_pspg,
        "test10_direct_pspg_topology_policy_parent_subset_replay_20260607.json",
    )
    assert topology_policy_parent_subset_replay["finding"] == (
        "direct_pspg_signature_parent_subset_full_local_replay_"
        "does_not_clear_test10_guard"
    )
    assert topology_policy_parent_subset_replay["status"] == (
        "exact_parent_subset_ruled_out_as_sufficient_fix"
    )
    assert topology_policy_parent_subset_replay[
        "signature_parent_filter_full_local_confirmed"
    ]
    assert topology_policy_parent_subset_replay[
        "signature_parent_filter_update_pa"
    ] == 578.9424523317655
    assert topology_policy_parent_subset_replay["broad_policy_update_pa"] == (
        522.4172735486616
    )
    assert topology_policy_parent_subset_replay[
        "signature_row_filter_update_pa"
    ] == 604.7126561932914
    assert topology_policy_parent_subset_replay[
        "pressure_update_guard_cleared"
    ]["signature_parent_filter"] is False
    parent_filter_replay = topology_policy_parent_subset_replay["replays"][
        "signature_parent_filter"
    ]
    assert parent_filter_replay["policy_log"]["record_count"] == 264
    assert parent_filter_replay["policy_log"]["row_filter_enabled_values"] == [0]
    assert parent_filter_replay["policy_log"]["parent_filter_enabled_values"] == [1]
    assert parent_filter_replay["policy_log"][
        "parent_filter_parent_cell_count_values"
    ] == [264]
    assert "physical support-patch" in topology_policy_parent_subset_replay[
        "next_requirement"
    ]
    support_cutoff = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_active_pressure_support_cutoff_relevance_20260607.json",
    )
    assert support_cutoff["finding"] == (
        "active_pressure_support_cutoff_not_complete_fix_from_branch_shift"
    )
    assert support_cutoff["status"] == (
        "support_cutoff_diagnostic_only_not_complete_fix"
    )
    assert support_cutoff["constraint_source"][
        "retained_generated_volume_support_activation_is_unconditional"
    ]
    assert not support_cutoff["constraint_source"][
        "retained_generated_volume_support_uses_volume_fraction_cutoff"
    ]
    assert support_cutoff["classification"] == {
        "tiny_cut_supported_branch_present": True,
        "full_wet_supported_branch_present": True,
        "retained_fraction_cutoff_is_complete_fix_candidate": False,
        "retained_fraction_cutoff_is_diagnostic_only": True,
    }
    assert support_cutoff["topology_policy_replay_summary"][
        "test02_policy_support_class_counts"
    ] == {
        "tiny_cut_supported": 3,
    }
    assert support_cutoff["pressure_update_rejection_summary"][
        "test02_adaptive_support_branch_shift"
    ] == "tiny_cut_supported_to_full_wet_supported"
    coupled_patch_barrier = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_coupled_patch_dependency_barrier_20260607.json",
    )
    assert coupled_patch_barrier["finding"] == (
        "coupled_patch_dependency_barrier_requires_solve_time_provenance"
    )
    assert coupled_patch_barrier["status"] == (
        "remaining_gate_requires_new_assembly_provenance_diagnostic"
    )
    assert coupled_patch_barrier["blocker_summary"][
        "requires_new_solve_time_provenance"
    ]
    assert (
        "does not use pressure-update signs"
        in coupled_patch_barrier["next_requirement"]
    )
    solve_time_provenance = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_solve_time_provenance_support_20260607.json",
    )
    assert solve_time_provenance["finding"] == (
        "solve_time_direct_pspg_support_coupling_provenance_ready"
    )
    assert solve_time_provenance["status"] == (
        "diagnostic_ready_replay_pending"
    )
    assert solve_time_provenance["features"][
        "records_pressure_update_sign_not_used"
    ]
    assert solve_time_provenance["features"]["emits_sampled_column_payload"]
    assert solve_time_provenance["features"]["uses_bounded_column_sample"]
    assert solve_time_provenance["features"]["does_not_mutate_matrix"]
    assert solve_time_provenance["diagnostic_env"][
        "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_SUPPORT_COUPLING_OPERATOR"
    ] == "equations"
    solve_time_replay = _evidence_by_suffix(
        direct_pspg,
        "test02_test10_direct_pspg_solve_time_provenance_replay_20260607.json",
    )
    assert solve_time_replay["finding"] == (
        "solve_time_direct_pspg_support_coupling_replay_rules_out_simple_pp_pv_gate"
    )
    assert solve_time_replay["status"] == (
        "replay_evidence_supports_coupling_split_no_selector"
    )
    assert solve_time_replay["record_counts"] == {
        "test02": 26864,
        "test10": 5760,
    }
    assert solve_time_replay["target_rows_present_counts"] == {
        "test02": 7,
        "test10": 12,
    }
    assert solve_time_replay["max_target_ratio_rows"]["test02"] == [10676]
    assert solve_time_replay["zero_pressure_velocity_target_global_dofs"][
        "test10"
    ] == [3526, 3456, 3925]
    sampled_column_selectivity = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_solve_time_sampled_column_"
            "selectivity_20260607.json"
        ),
    )
    assert sampled_column_selectivity["finding"] == (
        "solve_time_direct_pspg_sampled_column_selectors_not_formulation_ready"
    )
    assert sampled_column_selectivity["status"] == (
        "sampled_column_stencil_gate_ruled_out"
    )
    assert sampled_column_selectivity["record_counts"] == {
        "test02": 26864,
        "test10": 5760,
    }
    assert sampled_column_selectivity["any_sample_truncated"] == {
        "test02": False,
        "test10": False,
    }
    support_coupling_signature = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_solve_time_support_coupling_signature_"
            "20260607.json"
        ),
    )
    assert support_coupling_signature["finding"] == (
        "solve_time_direct_pspg_support_coupling_signature_partial_test10_only"
    )
    assert support_coupling_signature["status"] == (
        "test10_signature_candidate_test02_overbroad"
    )
    assert support_coupling_signature["case_findings"] == {
        "test02": (
            "solve_time_support_coupling_signature_covers_targets_but_overbroad"
        ),
        "test10": "solve_time_support_coupling_signature_selective_candidate",
    }
    assert support_coupling_signature["target_support_class_counts"] == {
        "test02": {"none": 0, "partial": 0, "full": 7},
        "test10": {"none": 3, "partial": 3, "full": 6},
    }
    assert support_coupling_signature[
        "exact_local_signature_selected_counts"
    ] == {
        "test02": 276,
        "test10": 48,
    }
    assert support_coupling_signature[
        "exact_local_signature_selected_to_target_ratios"
    ] == {
        "test02": 276 / 7,
        "test10": 4.0,
    }
    magnitude_selectivity = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_solve_time_magnitude_selectivity_"
            "20260607.json"
        ),
    )
    assert magnitude_selectivity["finding"] == (
        "solve_time_direct_pspg_support_coupling_magnitude_selectors_not_formulation_ready"
    )
    assert magnitude_selectivity["status"] == (
        "range_thresholds_overbroad_exact_value_oracles_only"
    )
    assert magnitude_selectivity["case_findings"] == {
        "test02": "exact_magnitude_value_oracles_only_range_selectors_broad",
        "test10": "exact_magnitude_value_oracles_only_range_selectors_broad",
    }
    assert magnitude_selectivity["range_selector_selected_to_target_ratios"][
        "test02"
    ]["pressure_pressure_abs_sum_target_range"] == 84.0
    assert magnitude_selectivity["range_selector_selected_to_target_ratios"][
        "test10"
    ]["pressure_velocity_abs_sum_target_range"] == 89 / 12
    assert (
        "pressure_velocity_abs_sum_exact_target_value_set"
        in magnitude_selectivity["exact_value_oracle_selector_keys"]["test02"]
    )
    signature_magnitude = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_solve_time_signature_magnitude_"
            "composite_20260607.json"
        ),
    )
    assert signature_magnitude["finding"] == (
        "solve_time_direct_pspg_signature_magnitude_composite_partial_test10_only"
    )
    assert signature_magnitude["status"] == (
        "test10_composite_candidate_test02_overbroad"
    )
    assert signature_magnitude["case_findings"] == {
        "test02": (
            "solve_time_signature_magnitude_composite_covers_targets_but_overbroad"
        ),
        "test10": (
            "solve_time_signature_magnitude_composite_selective_candidate"
        ),
    }
    assert signature_magnitude["best_covering_composite_selected_counts"] == {
        "test02": 53,
        "test10": 22,
    }
    assert signature_magnitude[
        "best_covering_composite_selected_to_target_ratios"
    ] == {
        "test02": 53 / 7,
        "test10": 22 / 12,
    }
    aggregate_features = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "direct_pspg_solve_time_aggregate_features"
    )
    aggregate_evidence = _evidence_by_suffix(
        aggregate_features,
        (
            "test02_test10_direct_pspg_solve_time_aggregate_feature_"
            "selectivity_20260607.json"
        ),
    )
    assert aggregate_evidence["finding"] == (
        "solve_time_direct_pspg_aggregate_feature_selectivity_"
        "rules_out_counts_and_volume_gate"
    )
    assert aggregate_evidence["status"] == (
        "aggregate_counts_and_volume_features_overbroad"
    )
    assert aggregate_evidence["case_findings"] == {
        "test02": (
            "solve_time_aggregate_feature_selectors_overbroad_or_miss_targets"
        ),
        "test10": (
            "solve_time_aggregate_feature_selectors_overbroad_or_miss_targets"
        ),
    }
    assert aggregate_evidence["best_exact_selector_keys"] == {
        "test02": "full_cell_records_exact_target_value_set",
        "test10": "full_cell_records_exact_target_value_set",
    }
    assert aggregate_evidence["best_range_selector_keys"] == {
        "test02": "cut_cell_records_target_range",
        "test10": "full_cell_records_target_range",
    }
    assert aggregate_evidence["best_exact_selected_counts"] == {
        "test02": 214,
        "test10": 109,
    }
    assert aggregate_evidence["best_range_selected_counts"] == {
        "test02": 427,
        "test10": 111,
    }
    assert aggregate_evidence["best_exact_selected_to_target_ratios"] == {
        "test02": 214 / 7,
        "test10": 109 / 12,
    }
    assert aggregate_evidence["best_range_selected_to_target_ratios"] == {
        "test02": 61.0,
        "test10": 111 / 12,
    }
    assert "physical support/coupling discriminator" in aggregate_evidence[
        "next_requirement"
    ]
    support_measure_features = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "direct_pspg_solve_time_support_measure_features"
    )
    support_measure_evidence = _evidence_by_suffix(
        support_measure_features,
        (
            "test02_test10_direct_pspg_solve_time_support_measure_"
            "selectivity_20260607.json"
        ),
    )
    assert support_measure_evidence["finding"] == (
        "solve_time_direct_pspg_support_measure_selectivity_rules_out_"
        "qpoint_measure_gate"
    )
    assert support_measure_evidence["status"] == (
        "active_qpoint_and_measure_features_overbroad"
    )
    assert support_measure_evidence["case_findings"] == {
        "test02": (
            "solve_time_support_measure_selectors_overbroad_or_miss_targets"
        ),
        "test10": (
            "solve_time_support_measure_selectors_overbroad_or_miss_targets"
        ),
    }
    assert support_measure_evidence["best_exact_selector_keys"] == {
        "test02": "active_quadrature_fraction_values_exact_target_value_set",
        "test10": "active_quadrature_fraction_values_exact_target_value_set",
    }
    assert support_measure_evidence["best_range_selector_keys"] == {
        "test02": "active_quadrature_fraction_values_target_range",
        "test10": "active_quadrature_fraction_values_target_range",
    }
    assert support_measure_evidence["best_exact_selected_counts"] == {
        "test02": 427,
        "test10": 126,
    }
    assert support_measure_evidence["best_range_selected_counts"] == {
        "test02": 427,
        "test10": 126,
    }
    assert support_measure_evidence["best_exact_selected_to_target_ratios"] == {
        "test02": 61.0,
        "test10": 10.5,
    }
    assert support_measure_evidence["best_range_selected_to_target_ratios"] == {
        "test02": 61.0,
        "test10": 10.5,
    }
    assert "active quadrature count" in support_measure_evidence[
        "next_requirement"
    ]
    parent_rule_components = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "direct_pspg_solve_time_parent_rule_components"
    )
    parent_rule_evidence = _evidence_by_suffix(
        parent_rule_components,
        (
            "test02_test10_direct_pspg_solve_time_parent_rule_component_"
            "selectivity_20260607.json"
        ),
    )
    assert parent_rule_evidence["finding"] == (
        "solve_time_direct_pspg_parent_rule_components_rule_out_"
        "connected_cosupport_closure"
    )
    assert parent_rule_evidence["status"] == (
        "parent_rule_component_closure_overbroad"
    )
    assert parent_rule_evidence["case_findings"] == {
        "test02": "solve_time_parent_rule_components_overbroad_or_miss_targets",
        "test10": "solve_time_parent_rule_components_overbroad_or_miss_targets",
    }
    assert parent_rule_evidence["component_counts"] == {
        "test02": {
            "parent_cell": 1,
            "rule_index": 1,
            "parent_or_rule": 1,
            "parent_rule_local_index": 1,
        },
        "test10": {
            "parent_cell": 1,
            "rule_index": 1,
            "parent_or_rule": 1,
            "parent_rule_local_index": 1,
        },
    }
    assert parent_rule_evidence["target_component_sizes"] == {
        "test02": {
            "parent_cell": [880],
            "rule_index": [880],
            "parent_or_rule": [880],
            "parent_rule_local_index": [880],
        },
        "test10": {
            "parent_cell": [252],
            "rule_index": [252],
            "parent_or_rule": [252],
            "parent_rule_local_index": [252],
        },
    }
    assert parent_rule_evidence["best_component_selector_keys"] == {
        "test02": "parent_cell_target_component_union",
        "test10": "parent_cell_target_component_union",
    }
    assert parent_rule_evidence["best_component_selected_counts"] == {
        "test02": 880,
        "test10": 252,
    }
    assert parent_rule_evidence["best_component_selected_to_target_ratios"] == {
        "test02": 880 / 7,
        "test10": 21.0,
    }
    assert "connected parent/rule co-support closure" in parent_rule_evidence[
        "next_requirement"
    ]
    same_rule_cross_block = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "direct_pspg_solve_time_same_rule_cross_block_signature"
    )
    same_rule_evidence = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_solve_time_same_rule_cross_block_"
            "signature_20260607.json"
        ),
    )
    assert same_rule_evidence["finding"] == (
        "solve_time_direct_pspg_same_rule_cross_block_signature_"
        "magnitude_candidate_found"
    )
    assert same_rule_evidence["status"] == (
        "same_rule_cross_block_candidate_requires_replay"
    )
    assert same_rule_evidence["case_findings"] == {
        "test02": "same_rule_cross_block_signature_magnitude_candidate",
        "test10": "same_rule_cross_block_signature_magnitude_candidate",
    }
    assert same_rule_evidence["shape_pair_selected_counts"] == {
        "test02": 879,
        "test10": 252,
    }
    assert same_rule_evidence["base_signature_selected_counts"] == {
        "test02": 56,
        "test10": 60,
    }
    assert same_rule_evidence["base_signature_selected_to_target_ratios"] == {
        "test02": 8.0,
        "test10": 5.0,
    }
    assert same_rule_evidence["best_composite_selector_keys"] == {
        "test02": (
            "same_rule_cross_block_signature_with_pressure_velocity_abs_sum_range"
        ),
        "test10": (
            "same_rule_cross_block_signature_with_pressure_velocity_abs_sum_range"
        ),
    }
    assert same_rule_evidence["best_composite_features"] == {
        "test02": "pressure_velocity_abs_sum",
        "test10": "pressure_velocity_abs_sum",
    }
    assert same_rule_evidence["best_composite_selected_counts"] == {
        "test02": 20,
        "test10": 21,
    }
    assert same_rule_evidence["best_composite_selected_to_target_ratios"] == {
        "test02": 20 / 7,
        "test10": 1.75,
    }
    assert same_rule_evidence["best_composite_selected_global_dofs"][
        "test02"
    ][0] == 10658
    assert same_rule_evidence["best_composite_selected_global_dofs"][
        "test10"
    ][-1] == 3935
    assert "targeted Test02/Test10 row-filter replay" in same_rule_evidence[
        "next_requirement"
    ]
    same_rule_replay_evidence = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_row_filter_"
            "replays_20260607.json"
        ),
    )
    assert same_rule_replay_evidence["finding"] == (
        "direct_pspg_same_rule_cross_block_row_filter_replays_do_not_clear_guards"
    )
    assert same_rule_replay_evidence["status"] == (
        "same_rule_cross_block_replay_insufficient"
    )
    assert same_rule_replay_evidence["row_filters_match_candidate_counts"] is True
    assert (
        same_rule_replay_evidence["all_replays_improve_no_policy_baseline"] is True
    )
    assert same_rule_replay_evidence["all_replays_trigger_guard"] is True
    assert same_rule_replay_evidence["triggered_cases"] == ["test02", "test10"]
    assert same_rule_replay_evidence["candidate_row_counts"] == {
        "test02": 20,
        "test10": 21,
    }
    assert same_rule_replay_evidence["worst_active_or_wet_update_pa"] == {
        "test02": 357449.7849043233,
        "test10": 582.6183066757754,
    }
    assert same_rule_replay_evidence["worst_active_or_wet_support_class"] == {
        "test02": "full_wet_supported",
        "test10": "full_wet_supported",
    }
    assert same_rule_replay_evidence["improvement_vs_baseline_pa"] == {
        "test02": 9270.180902128108,
        "test10": 39.99110318614055,
    }
    assert same_rule_replay_evidence["replay_to_baseline_update_ratio"] == {
        "test02": 0.9747213629840358,
        "test10": 0.9357685532009388,
    }
    assert same_rule_replay_evidence["replay_to_broad_policy_update_ratio"] == {
        "test02": 2.021269323569583,
        "test10": 1.1152355333088089,
    }
    assert same_rule_replay_evidence["matrix_mutated_counts"] == {
        "test02": 300,
        "test10": 86,
    }
    assert "Do not promote the same-rule row list" in same_rule_replay_evidence[
        "next_requirement"
    ]
    same_rule_parent_scope_evidence = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_"
            "scope_20260607.json"
        ),
    )
    assert same_rule_parent_scope_evidence["finding"] == (
        "direct_pspg_same_rule_cross_block_parent_cell_scope_ready_for_replay"
    )
    assert same_rule_parent_scope_evidence["status"] == (
        "run_same_rule_cross_block_parent_cell_replay"
    )
    assert same_rule_parent_scope_evidence[
        "all_cases_ready_for_parent_cell_replay"
    ]
    assert same_rule_parent_scope_evidence["parent_cell_counts"] == {
        "test02": 360,
        "test10": 86,
    }
    assert same_rule_parent_scope_evidence["parent_expanded_row_counts"] == {
        "test02": 157,
        "test10": 57,
    }
    assert same_rule_parent_scope_evidence[
        "parent_expanded_to_candidate_ratios"
    ] == {
        "test02": 7.85,
        "test10": 57 / 21,
    }
    same_rule_parent_replay_evidence = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_parent_cell_"
            "replays_20260607.json"
        ),
    )
    assert same_rule_parent_replay_evidence["finding"] == (
        "direct_pspg_same_rule_cross_block_parent_cell_replays_do_not_clear_guards"
    )
    assert same_rule_parent_replay_evidence["status"] == (
        "same_rule_cross_block_parent_cell_replay_insufficient"
    )
    assert same_rule_parent_replay_evidence["parent_filters_match_scope_counts"]
    assert same_rule_parent_replay_evidence["row_filters_disabled"]
    assert (
        same_rule_parent_replay_evidence[
            "all_replays_improve_no_policy_baseline"
        ]
        is True
    )
    assert same_rule_parent_replay_evidence[
        "all_replays_improve_row_filter_replay"
    ]
    assert same_rule_parent_replay_evidence["all_replays_trigger_guard"]
    assert same_rule_parent_replay_evidence["triggered_cases"] == [
        "test02",
        "test10",
    ]
    assert same_rule_parent_replay_evidence["parent_cell_counts"] == {
        "test02": 360,
        "test10": 86,
    }
    assert same_rule_parent_replay_evidence[
        "worst_active_or_wet_update_pa"
    ] == {
        "test02": 321290.80382374703,
        "test10": 570.6844972203451,
    }
    assert same_rule_parent_replay_evidence["improvement_vs_baseline_pa"] == {
        "test02": 45429.16198270436,
        "test10": 51.92491264157093,
    }
    assert same_rule_parent_replay_evidence["improvement_vs_row_filter_pa"] == {
        "test02": 36158.98108057625,
        "test10": 11.933809455430378,
    }
    assert same_rule_parent_replay_evidence[
        "replay_to_broad_policy_update_ratio"
    ] == {
        "test02": 1.8168013330537556,
        "test10": 1.0923920898400148,
    }
    assert same_rule_parent_replay_evidence["matrix_mutated_counts"] == {
        "test02": 360,
        "test10": 86,
    }
    assert "Do not promote parent-cell replay" in same_rule_parent_replay_evidence[
        "next_requirement"
    ]
    broad_minus_scope_evidence = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_"
            "parent_cell_scope_20260607.json"
        ),
    )
    assert broad_minus_scope_evidence["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_scope_ready_for_replay"
    )
    assert broad_minus_scope_evidence["status"] == (
        "run_broad_minus_same_rule_parent_cell_replay"
    )
    assert broad_minus_scope_evidence[
        "all_cases_ready_for_broad_minus_parent_cell_replay"
    ]
    assert broad_minus_scope_evidence["broad_parent_cell_counts"] == {
        "test02": 3352,
        "test10": 720,
    }
    assert broad_minus_scope_evidence["same_rule_parent_cell_counts"] == {
        "test02": 360,
        "test10": 86,
    }
    assert broad_minus_scope_evidence["broad_only_parent_cell_counts"] == {
        "test02": 2992,
        "test10": 634,
    }
    assert broad_minus_scope_evidence["broad_only_to_broad_parent_ratios"] == {
        "test02": 2992 / 3352,
        "test10": 634 / 720,
    }
    broad_minus_replay_evidence = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_minus_"
            "parent_cell_replays_20260607.json"
        ),
    )
    assert broad_minus_replay_evidence["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_minus_parent_replays_do_not_"
        "clear_guards"
    )
    assert broad_minus_replay_evidence["status"] == (
        "broad_minus_parent_replay_insufficient"
    )
    assert broad_minus_replay_evidence["parent_filters_match_scope_counts"]
    assert broad_minus_replay_evidence["row_filters_disabled"]
    assert broad_minus_replay_evidence["all_replays_trigger_guard"]
    assert broad_minus_replay_evidence["broad_policy_better_than_isolated_parts"]
    assert broad_minus_replay_evidence[
        "complement_worse_than_same_rule_parent_cell"
    ]
    assert broad_minus_replay_evidence["triggered_cases"] == [
        "test02",
        "test10",
    ]
    assert broad_minus_replay_evidence["broad_only_parent_cell_counts"] == {
        "test02": 2992,
        "test10": 634,
    }
    assert broad_minus_replay_evidence["worst_active_or_wet_update_pa"] == {
        "test02": 366324.79523179174,
        "test10": 575.8357642247117,
    }
    assert broad_minus_replay_evidence["improvement_vs_baseline_pa"] == {
        "test02": 395.17057465965627,
        "test10": 46.773645637204254,
    }
    assert broad_minus_replay_evidence[
        "improvement_vs_same_rule_parent_cell_pa"
    ] == {
        "test02": -45033.991408044705,
        "test10": -5.151267004366673,
    }
    assert broad_minus_replay_evidence[
        "replay_to_broad_policy_update_ratio"
    ] == {
        "test02": 2.0714547954284535,
        "test10": 1.1022525352448447,
    }
    assert broad_minus_replay_evidence["matrix_mutated_counts"] == {
        "test02": 2992,
        "test10": 634,
    }
    assert "broad-only complement" in broad_minus_replay_evidence[
        "next_requirement"
    ]
    broad_union_branch_shift = _evidence_by_suffix(
        same_rule_cross_block,
        (
            "test02_test10_direct_pspg_same_rule_cross_block_broad_union_"
            "branch_shift_20260607.json"
        ),
    )
    assert broad_union_branch_shift["finding"] == (
        "direct_pspg_same_rule_cross_block_broad_union_consistent_replays_"
        "do_not_clear_guards"
    )
    assert broad_union_branch_shift["status"] == (
        "broad_union_consistent_replay_insufficient"
    )
    assert broad_union_branch_shift["case_findings"] == {
        "test02": (
            "broad_union_reduces_full_wet_reference_but_guard_remains"
        ),
        "test10": (
            "broad_union_reduces_shared_full_wet_reference_but_guard_remains"
        ),
    }
    assert broad_union_branch_shift["all_variants_guard_triggered"]
    assert not broad_union_branch_shift["test02_branch_shift_supported"]
    assert broad_union_branch_shift[
        "test02_consistent_full_wet_residual_supported"
    ]
    assert broad_union_branch_shift[
        "test10_broad_union_residual_guard_supported"
    ]
    assert broad_union_branch_shift["reference_points"] == {
        "test02": 1172,
        "test10": 83,
    }
    assert broad_union_branch_shift[
        "broad_reference_abs_pressure_delta_pa"
    ] == {
        "test02": 321110.9963650234,
        "test10": 522.4172735486616,
    }
    assert broad_union_branch_shift[
        "isolated_reference_abs_pressure_delta_pa"
    ] == {
        "test02": {
            "no_policy": 366719.9658064514,
            "same_rule_parent": 321290.80382374703,
            "broad_minus_parent": 366324.79523179174,
        },
        "test10": {
            "no_policy": 622.609409861916,
            "same_rule_parent": 570.6844972203451,
            "broad_minus_parent": 575.8357642247117,
        },
    }
    assert broad_union_branch_shift[
        "broad_reference_improvement_vs_isolated_pa"
    ] == {
        "test02": {
            "no_policy": 45608.969441427966,
            "same_rule_parent": 179.8074587236042,
            "broad_minus_parent": 45213.79886676831,
        },
        "test10": {
            "no_policy": 100.19213631325442,
            "same_rule_parent": 48.267223671683496,
            "broad_minus_parent": 53.41849067605017,
        },
    }
    assert broad_union_branch_shift["broad_policy_worst_points"] == {
        "test02": 1172,
        "test10": 83,
    }
    assert broad_union_branch_shift["broad_policy_worst_support_classes"] == {
        "test02": "full_wet_supported",
        "test10": "full_wet_supported",
    }
    assert broad_union_branch_shift[
        "broad_policy_clears_reference_point_guard"
    ] == {
        "test02": False,
        "test10": False,
    }
    assert broad_union_branch_shift["broad_policy_guard_triggered"] == {
        "test02": True,
        "test10": True,
    }
    assert "full-wet Test02/Test10" in broad_union_branch_shift[
        "next_requirement"
    ]
    test10_signature_replay = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_test10_signature_replay_readiness_"
            "20260607.json"
        ),
    )
    assert test10_signature_replay["finding"] == (
        "test10_signature_replay_candidate_ready_for_solve_time_replay"
    )
    assert test10_signature_replay["status"] == "run_targeted_test10_signature_replay"
    assert test10_signature_replay["case_selector_findings"] == {
        "test02": "selector_overbroad",
        "test10": "selector_selective",
    }
    assert test10_signature_replay["case_selected_counts"] == {
        "test02": 276,
        "test10": 48,
    }
    assert test10_signature_replay["case_selected_to_target_ratios"] == {
        "test02": 276 / 7,
        "test10": 4.0,
    }
    assert test10_signature_replay[
        "fe_topology_signature_or_row_selector_present"
    ] is True
    assert test10_signature_replay[
        "post_assembly_explicit_row_path_present"
    ] is True
    assert test10_signature_replay["test10_signature_candidate_global_dofs"][
        :2
    ] == [3277, 3278]
    test10_signature_row_filter_replay = _evidence_by_suffix(
        direct_pspg,
        (
            "test10_replay_cap3_step90_direct_pspg_signature_rows_"
            "schur_edge_balance_pressure_update_audit_20260607.json"
        ),
    )
    assert test10_signature_row_filter_replay["status"] == (
        "diagnostic_pressure_update_guard_triggered"
    )
    assert test10_signature_row_filter_replay["policy"] == (
        "local_schur_edge_balance"
    )
    assert test10_signature_row_filter_replay["row_filter_global_dof_count"] == 48
    assert test10_signature_row_filter_replay[
        "worst_active_or_wet_update_pa"
    ] == 604.7126561932914
    assert test10_signature_row_filter_replay[
        "worst_active_or_wet_support_class"
    ] == "full_wet_supported"
    test10_signature_row_filter_replays = _evidence_by_suffix(
        direct_pspg,
        "test10_direct_pspg_signature_row_filter_replays_20260607.json",
    )
    assert test10_signature_row_filter_replays["finding"] == (
        "test10_signature_row_filter_local_modes_do_not_clear_guard"
    )
    assert test10_signature_row_filter_replays["status"] == (
        "signature_row_filter_local_modes_ruled_out_as_sufficient_fix"
    )
    assert test10_signature_row_filter_replays["policies_tested"] == [
        "local_schur_completion",
        "local_edge_balance",
        "local_schur_edge_balance",
    ]
    assert test10_signature_row_filter_replays[
        "row_filter_global_dof_counts"
    ] == [48]
    assert test10_signature_row_filter_replays["all_replays_trigger_guard"]
    assert test10_signature_row_filter_replays[
        "best_policy_by_worst_update"
    ] == "local_schur_edge_balance"
    assert test10_signature_row_filter_replays[
        "best_worst_active_or_wet_update_pa"
    ] == 604.7126561932914
    assert test10_signature_row_filter_replays[
        "policy_worst_active_or_wet_updates_pa"
    ] == {
        "local_schur_completion": 619.6167550623924,
        "local_edge_balance": 607.5173052131886,
        "local_schur_edge_balance": 604.7126561932914,
    }
    assert test10_signature_row_filter_replays[
        "policy_row_filter_log_counts"
    ] == {
        "local_schur_completion": 264,
        "local_edge_balance": 264,
        "local_schur_edge_balance": 264,
    }
    ghost_signature = _evidence_by_suffix(
        direct_pspg,
        (
            "test02_test10_direct_pspg_ghost_branch_signature_interaction_"
            "20260607.json"
        ),
    )
    assert ghost_signature["finding"] == (
        "direct_pspg_ghost_branch_signature_interaction_rules_out_common_gate"
    )
    assert ghost_signature["status"] == (
        "ghost_branch_is_branch_shaper_not_support_coupling_signature_fix"
    )
    assert ghost_signature["case_findings"] == {
        "test02": "ghost_branch_shapes_test02_but_cannot_narrow_signature",
        "test10": (
            "ghost_absent_test10_signature_candidate_remains_partial_fix"
        ),
    }
    assert ghost_signature["row_10676_baseline_update_pa"] == (
        366719.9658064514
    )
    assert ghost_signature["row_10676_pressure_disabled_update_pa"] == (
        1298098.542745239
    )
    assert ghost_signature["signature_ratios"] == {
        "test02": 276 / 7,
        "test10": 4.0,
    }
    assert ghost_signature["pressure_disabled_still_triggers"] == {
        "test02": True,
        "test10": True,
    }
    assert statuses["timestep_acceptance_logic"] == (
        "guard_supported_dt_reduction_ruled_out_as_fix"
    )
    timestep = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "timestep_acceptance_logic"
    )
    rejection_replay = _evidence_by_suffix(
        timestep,
        "test02_test10_pressure_update_rejection_replay_20260607.json",
    )
    assert rejection_replay["finding"] == (
        "pressure_update_rejection_catches_both_cases_dt_reduction_not_fix"
    )
    assert rejection_replay["status"] == (
        "pre_commit_guard_supported_dt_reduction_ruled_out"
    )
    assert rejection_replay["guard"]["phase"] == "pre_commit"
    fixed_replays = {
        replay["case"]: replay
        for replay in rejection_replay["fixed_step_replays"]
    }
    assert not fixed_replays["test02"]["step_accepted"]
    assert fixed_replays["test02"]["step_rejected_count"] == 1
    assert fixed_replays["test02"]["worst_pre_commit_update_pa"] == (
        105591.14535324997
    )
    assert fixed_replays["test02"]["worst_pre_commit_support_class"] == (
        "tiny_cut_supported"
    )
    assert not fixed_replays["test10"]["step_accepted"]
    assert fixed_replays["test10"]["worst_pre_commit_update_pa"] == (
        1075.5582119407377
    )
    adaptive_replays = {
        replay["case"]: replay
        for replay in rejection_replay["adaptive_replays"]
    }
    assert adaptive_replays["test02"]["step_rejected_count"] == 11
    assert adaptive_replays["test02"]["last_update_pa"] == (
        14137282.618001418
    )
    assert adaptive_replays["test02"]["support_branch_shift"] == (
        "tiny_cut_supported_to_full_wet_supported"
    )
    assert adaptive_replays["test10"]["step_rejected_count"] == 4
    assert adaptive_replays["test10"]["last_update_pa"] == (
        9103.459989150644
    )
    residual_context = _evidence_by_suffix(
        timestep,
        "test02_test10_pressure_update_residual_context_20260607.json",
    )
    assert residual_context["finding"] == (
        "accepted_pressure_updates_converged_with_large_residual_gap"
    )
    assert residual_context["status"] == (
        "residual_convergence_acceptance_gap_supported"
    )
    assert residual_context[
        "all_cases_accepted_converged_large_update_residual_gap"
    ]
    assert residual_context[
        "all_cases_post_acceptance_refresh_ruled_out"
    ]
    assert residual_context[
        "case_update_to_nonlinear_field_residual_ratios"
    ] == {
        "test02": 1113750.000597943,
        "test10": 938426.6400308398,
    }
    assert residual_context["case_pressure_updates_pa"]["test10"] == (
        1075.5582134176257
    )
    assert statuses["pressure_ghost_penalty_direct_driver"] == (
        "ruled_out_as_direct_test10_or_sampled_max_row_driver"
    )
    pressure_ghost = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "pressure_ghost_penalty_direct_driver"
    )
    pressure_ghost_signature = _evidence_by_suffix(
        pressure_ghost,
        (
            "test02_test10_direct_pspg_ghost_branch_signature_interaction_"
            "20260607.json"
        ),
    )
    assert pressure_ghost_signature["status"] == (
        "ghost_branch_is_branch_shaper_not_support_coupling_signature_fix"
    )
    assert pressure_ghost_signature["row_10676_pressure_disabled_update_pa"] == (
        1298098.542745239
    )
    pressure_stabilization_driver = _evidence_by_suffix(
        pressure_ghost,
        "test02_test10_pressure_stabilization_driver_windows_20260607.json",
    )
    assert pressure_stabilization_driver["finding"] == (
        "cut_adjacent_pressure_stabilization_not_direct_worst_update_driver"
    )
    assert pressure_stabilization_driver["status"] == (
        "ghost_penalty_direct_worst_update_path_ruled_out_for_saved_windows"
    )
    assert pressure_stabilization_driver[
        "all_saved_window_worst_updates_nonincident"
    ]
    assert not pressure_stabilization_driver[
        "any_saved_window_worst_update_incident"
    ]
    assert pressure_stabilization_driver[
        "case_incident_cut_adjacent_face_counts"
    ] == {
        "test02": 0,
        "test10": 0,
    }
    assert pressure_stabilization_driver["case_worst_updates_pa"] == {
        "test02": 2112204.128955333,
        "test10": 1075.2113565356985,
    }
    assert statuses["post_assembly_graph_completion_family"] == (
        "ruled_out_as_production_fix_supported_as_diagnostic"
    )
    graph_completion = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "post_assembly_graph_completion_family"
    )
    support_gap_patch = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            "test02_test10_graph_completion_support_gap_patch_20260606_outcome.json"
        )
    )
    assert support_gap_patch["test10_outcome"] == "accepted_guard_not_triggered"
    assert support_gap_patch["test10_accepted_pressure_update_pa"] == (
        6.7753020523015266
    )
    assert support_gap_patch["test02_outcome"] == "nonlinear_failed"
    assert support_gap_patch["test02_final_residual_norm"] == 29088.54648995749
    support_gap_patch_schur_only = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            "test02_test10_graph_completion_support_gap_patch_schur_only_20260606_outcome.json"
        )
    )
    assert support_gap_patch_schur_only["test10_outcome"] == (
        "accepted_guard_not_triggered"
    )
    assert support_gap_patch_schur_only[
        "test10_accepted_pressure_update_pa"
    ] == 93.45299901163241
    assert support_gap_patch_schur_only["test02_outcome"] == "nonlinear_failed"
    assert support_gap_patch_schur_only["test02_final_residual_norm"] == (
        2542.9071873675475
    )
    support_gap_local_patch = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            "test02_test10_graph_completion_support_gap_local_patch_schur_only_depth1_20260606_outcome.json"
        )
    )
    assert support_gap_local_patch["pressure_neighbor_depth"] == 1
    assert support_gap_local_patch["test10_outcome"] == (
        "accepted_guard_not_triggered"
    )
    assert support_gap_local_patch["test10_accepted_pressure_update_pa"] == (
        93.87750523707791
    )
    assert support_gap_local_patch["test02_outcome"] == "nonlinear_failed"
    assert support_gap_local_patch["test02_final_residual_norm"] == (
        2513.2066824913213
    )
    graph_readiness = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            "test02_test10_direct_pspg_graph_completion_candidate_readiness_20260606.json"
        )
    )
    assert graph_readiness["finding"] == (
        "support_gap_graph_completion_selectors_overbroad_and_test02_unstable"
    )
    assert graph_readiness["direct_target_counts"] == {"test02": 7, "test10": 12}
    replay_family = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            "test02_test10_direct_pspg_graph_completion_replay_family_20260607.json"
        )
    )
    assert replay_family["finding"] == (
        "direct_pspg_graph_completion_replay_family_rules_out_"
        "post_assembly_selector_variants"
    )
    assert replay_family["case_findings_by_variant"][
        "support_rank_neighborhood_depth1"
    ] == {
        "test02": "guard_still_triggered",
        "test10": "guard_still_triggered",
    }
    graph_tradeoff = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            (
                "test02_test10_direct_pspg_graph_completion_stability_"
                "tradeoff_20260607.json"
            )
        )
    )
    assert graph_tradeoff["finding"] == (
        "direct_pspg_graph_completion_stability_tradeoff_rules_out_"
        "post_assembly_fix"
    )
    assert graph_tradeoff["status"] == (
        "post_assembly_schur_balance_tradeoff_ruled_out"
    )
    assert graph_tradeoff["tradeoff_flags"][
        "support_rank_neighborhood_expansion_too_local"
    ]
    assert graph_tradeoff["least_selector_tradeoff"][
        "schur_edge_balance"
    ]["test02_nonlinear_failed"]
    active_support_completion = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            (
                "test02_test10_direct_pspg_active_support_completion_"
                "replays_20260607.json"
            )
        )
    )
    assert active_support_completion["finding"] == (
        "direct_pspg_active_support_completion_replays_rule_out_raw_"
        "active_support_completion"
    )
    assert active_support_completion["status"] == (
        "raw_active_support_completion_directional_but_insufficient"
    )
    assert active_support_completion["all_replays_guard_triggered"]
    assert active_support_completion["case_updates_pa"][
        "active_support_all"
    ]["test02"] == 155956.10179486268
    assert active_support_completion["cap_removal"][
        "cap64_neighbor_cap_limited_all_cases"
    ]
    assert active_support_completion["cap_removal"][
        "uncapped_still_triggers_all_cases"
    ]
    explicit_balance = next(
        evidence
        for evidence in graph_completion["evidence"]
        if evidence["path"].endswith(
            (
                "test02_test10_direct_pspg_explicit_balance_selector_replays_"
                "20260607.json"
            )
        )
    )
    assert explicit_balance["finding"] == (
        "direct_pspg_explicit_balance_selectors_rule_out_row_lists_and_"
        "pressure_neighborhoods"
    )
    assert explicit_balance["status"] == "explicit_balance_selectors_ruled_out"
    assert explicit_balance["ruled_out_by_variant"] == {
        "explicit_cross_policy_patch": True,
        "explicit_direct_rows": True,
        "explicit_neighborhood_depth1": True,
        "explicit_neighborhood_depth2": True,
        "explicit_operator_top_rows": True,
        "explicit_shifted_rows": True,
    }
    assert explicit_balance["case_findings_by_variant"][
        "explicit_neighborhood_depth2"
    ] == {
        "test02": "guard_still_triggered",
        "test10": "guard_still_triggered",
    }
    assert statuses["aggregate_no_galerkin_support_selector"] == (
        "supported_partial_for_test10_ruled_out_as_complete_selector"
    )
    no_galerkin = next(
        item
        for item in report["hypotheses"]
        if item["key"] == "aggregate_no_galerkin_support_selector"
    )
    no_galerkin_gate = _evidence_by_suffix(
        no_galerkin,
        "test02_test10_direct_pspg_no_galerkin_gate_relevance_20260607.json",
    )
    assert no_galerkin_gate["finding"] == (
        "no_galerkin_nonpressure_gate_ruled_out_as_complete_formulation_gate"
    )
    assert no_galerkin_gate["status"] == (
        "partial_test10_signal_ruled_out_as_complete_gate"
    )
    assert no_galerkin_gate["classification"] == {
        "overlap_missing_cases": ["test02"],
        "overlap_partial_cases": ["test10"],
        "candidate_uncovered_cases": ["test02"],
        "support_rank_mismatch_cases": ["test10"],
        "complete_gate_candidate": False,
    }
    overlap_cases = {
        case["label"]: case
        for case in no_galerkin_gate["top_overlap"]["cases"]
    }
    assert overlap_cases["test02"]["no_galerkin_top_update_overlap_ratio"] == 0.0
    assert overlap_cases["test10"]["no_galerkin_top_update_overlap_ratio"] == 0.25
    predicate_cases = {
        case["label"]: case
        for case in no_galerkin_gate["formulation_candidate"]["cases"]
    }
    assert predicate_cases["test02"]["uncovered_direct_target_global_dofs"] == [
        10676
    ]
