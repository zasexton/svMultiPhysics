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
        / "audit_direct_pspg_graph_completion_stability_tradeoff.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_graph_completion_stability_tradeoff", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _case(
    label,
    *,
    finding,
    outcome,
    candidate_row_count,
    accepted_pressure_update_pa=None,
    existing_balance_edge_count=0,
):
    return {
        "label": label,
        "finding": finding,
        "outcome": outcome,
        "candidate_row_count": candidate_row_count,
        "candidate_to_direct_target_ratio": (
            candidate_row_count / (7 if label == "test02" else 12)
        ),
        "accepted_pressure_update_pa": accepted_pressure_update_pa,
        "threshold_pa": 100000.0 if label == "test02" else 100.0,
        "newton_iterations": 12 if outcome.startswith("nonlinear") else 1,
        "final_residual_norm": (
            27095.0 if outcome.startswith("nonlinear") else 0.001
        ),
        "shared_row_schur_edge_count": 42,
        "existing_balance_edge_count": existing_balance_edge_count,
    }


def _variant(
    key,
    *,
    test02_finding,
    test02_outcome,
    test10_finding,
    test10_outcome,
    test02_update=None,
    test10_update=None,
    balance_edges=0,
):
    return {
        "key": key,
        "cases": [
            _case(
                "test02",
                finding=test02_finding,
                outcome=test02_outcome,
                candidate_row_count=304,
                accepted_pressure_update_pa=test02_update,
                existing_balance_edge_count=balance_edges,
            ),
            _case(
                "test10",
                finding=test10_finding,
                outcome=test10_outcome,
                candidate_row_count=68,
                accepted_pressure_update_pa=test10_update,
                existing_balance_edge_count=balance_edges,
            ),
        ],
    }


def test_build_report_rules_out_post_assembly_schur_balance_tradeoff():
    audit = _load_audit_module()
    replay_family = {
        "finding": "direct_pspg_graph_completion_replay_family_rules_out_post_assembly_selector_variants",
        "next_requirement": "Move to formulation-side support.",
        "variants": [
            _variant(
                "support_gap_patch_schur_only",
                test02_finding="nonlinear_failed_with_overbroad_patch",
                test02_outcome="nonlinear_failed",
                test10_finding="guard_cleared_with_overbroad_patch",
                test10_outcome="accepted_guard_not_triggered",
                test10_update=93.4,
            ),
            _variant(
                "support_gap_patch_schur_edge_balance",
                test02_finding="nonlinear_failed_with_overbroad_patch",
                test02_outcome="nonlinear_failed",
                test10_finding="guard_cleared_with_overbroad_patch",
                test10_outcome="accepted_guard_not_triggered",
                test10_update=6.8,
                balance_edges=100,
            ),
            _variant(
                "all_unconstrained_schur_edge_balance",
                test02_finding="nonlinear_failed_with_overbroad_patch",
                test02_outcome="nonlinear_failed_before_acceptance",
                test10_finding="guard_cleared_with_overbroad_patch",
                test10_outcome="accepted_guard_not_triggered",
                test10_update=6.8,
                balance_edges=100,
            ),
            _variant(
                "least_selector_schur_only",
                test02_finding="guard_still_triggered",
                test02_outcome="accepted_guard_triggered",
                test10_finding="guard_still_triggered",
                test10_outcome="accepted_guard_triggered",
                test02_update=319000.0,
                test10_update=122.0,
            ),
            _variant(
                "least_selector_schur_edge_balance",
                test02_finding="nonlinear_failed_with_overbroad_patch",
                test02_outcome="nonlinear_failed_before_acceptance",
                test10_finding="guard_cleared",
                test10_outcome="accepted_guard_not_triggered",
                test10_update=15.3,
                balance_edges=100,
            ),
            _variant(
                "support_rank_neighborhood_depth1",
                test02_finding="guard_still_triggered",
                test02_outcome="accepted_guard_triggered",
                test10_finding="guard_still_triggered",
                test10_outcome="accepted_guard_triggered",
                test02_update=366000.0,
                test10_update=319.0,
                balance_edges=50,
            ),
            _variant(
                "support_rank_neighborhood_depth2",
                test02_finding="guard_still_triggered",
                test02_outcome="accepted_guard_triggered",
                test10_finding="guard_still_triggered",
                test10_outcome="accepted_guard_triggered",
                test02_update=366000.0,
                test10_update=320.0,
                balance_edges=50,
            ),
            _variant(
                "coupling_deficient_balance",
                test02_finding="nonlinear_failed_with_overbroad_patch",
                test02_outcome="nonlinear_failed_before_acceptance",
                test10_finding="guard_still_triggered",
                test10_outcome="accepted_guard_triggered",
                test10_update=120.9,
                balance_edges=20,
            ),
            _variant(
                "low_pressure_degree_balance",
                test02_finding="nonlinear_failed_with_overbroad_patch",
                test02_outcome="nonlinear_failed_before_acceptance",
                test10_finding="guard_still_triggered",
                test10_outcome="accepted_guard_triggered",
                test10_update=120.8,
                balance_edges=20,
            ),
        ],
    }

    report = audit.build_report(replay_family)

    assert report["finding"] == (
        "direct_pspg_graph_completion_stability_tradeoff_rules_out_"
        "post_assembly_fix"
    )
    assert report["status"] == "post_assembly_schur_balance_tradeoff_ruled_out"
    assert report["tradeoff_flags"] == {
        "broad_topology_clears_test10_but_destabilizes_test02": True,
        "least_selector_schur_stable_but_insufficient_balance_clears_test10_but_destabilizes_test02": True,
        "localized_balance_gates_fail_test10_and_destabilize_test02": True,
        "support_rank_neighborhood_expansion_too_local": True,
    }
    assert report["least_selector_tradeoff"]["schur_only"][
        "test10_guard_triggered"
    ]
    assert report["least_selector_tradeoff"]["schur_edge_balance"][
        "test10_guard_cleared"
    ]
    assert report["least_selector_tradeoff"]["schur_edge_balance"][
        "test02_nonlinear_failed"
    ]


def test_build_report_requires_replay_family_variants():
    audit = _load_audit_module()
    report = audit.build_report({"variants": []})

    assert report["finding"] == (
        "direct_pspg_graph_completion_stability_tradeoff_missing_evidence"
    )
    assert report["status"] == "missing_replay_family_variants"
    assert "least_selector_schur_only" in report["missing_variants"]
