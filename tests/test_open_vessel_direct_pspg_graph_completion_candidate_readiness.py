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
        / "audit_direct_pspg_graph_completion_candidate_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_graph_completion_candidate_readiness", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_graph_completion_candidate_readiness_flags_overbroad_unstable_selector():
    audit = _load_audit_module()
    target_map = {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [100, 101]},
            {
                "label": "test10",
                "direct_pspg_target_global_dofs": [200, 201, 202],
            },
        ]
    }
    outcome = {
        "mode": "shared_row_schur_support_gap_patch_completion",
        "finding": "synthetic",
        "test10_step90": {
            "outcome": "accepted_guard_not_triggered",
            "accepted": True,
            "converged": True,
            "triggered": False,
            "accepted_pressure_update_pa": 93.0,
            "threshold_pa": 100.0,
            "worst_global_dof": 200,
            "candidate_row_count": 40,
            "support_gap_candidate_count": 20,
            "support_gap_patch_candidate_count": 40,
            "balance_candidate_row_count": 0,
            "edge_count": 100,
            "shared_row_schur_edge_count": 100,
            "existing_balance_edge_count": 0,
        },
        "test02_step382": {
            "outcome": "nonlinear_failed",
            "accepted": False,
            "converged": False,
            "newton_iterations": 12,
            "final_residual_norm": 2500.0,
            "candidate_row_count": 30,
            "support_gap_candidate_count": 15,
            "support_gap_patch_candidate_count": 30,
            "balance_candidate_row_count": 0,
            "edge_count": 80,
            "shared_row_schur_edge_count": 80,
            "existing_balance_edge_count": 0,
        },
    }

    report = audit.build_report(
        target_map=target_map,
        target_map_path=None,
        outcome_paths=[],
    )
    assert report["finding"] == "support_gap_graph_completion_readiness_unclassified"

    outcome_report = audit.outcome_report(
        outcome_path=Path("synthetic.json"),
        outcome=outcome,
        targets_by_label=audit.target_case_map(target_map),
    )
    assert outcome_report["finding"] == "candidate_overbroad_and_test02_unstable"
    cases = {case["label"]: case for case in outcome_report["cases"]}
    assert cases["test10"]["finding"] == "clears_guard_but_candidate_overbroad"
    assert cases["test10"]["candidate_to_direct_target_ratio"] == 40 / 3
    assert cases["test02"]["finding"] == "overbroad_candidate_and_nonlinear_failed"
    assert cases["test02"]["candidate_to_direct_target_ratio"] == 15.0


def test_build_report_summarizes_overbroad_modes(tmp_path):
    audit = _load_audit_module()
    target_map = {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [100]},
            {"label": "test10", "direct_pspg_target_global_dofs": [200]},
        ]
    }
    path = tmp_path / "outcome.json"
    path.write_text(
        """
        {
          "mode": "shared_row_schur_support_gap_patch_edge_balance",
          "test10_step90": {
            "outcome": "accepted_guard_not_triggered",
            "accepted": true,
            "converged": true,
            "triggered": false,
            "candidate_row_count": 12
          },
          "test02_step382": {
            "outcome": "nonlinear_failed",
            "accepted": false,
            "converged": false,
            "candidate_row_count": 12
          }
        }
        """,
        encoding="utf-8",
    )

    report = audit.build_report(
        target_map=target_map,
        target_map_path=Path("target.json"),
        outcome_paths=[path],
    )

    assert report["finding"] == (
        "support_gap_graph_completion_selectors_overbroad_and_test02_unstable"
    )
    assert report["overbroad_modes"] == [
        "shared_row_schur_support_gap_patch_edge_balance"
    ]
    assert report["test02_unstable_modes"] == [
        "shared_row_schur_support_gap_patch_edge_balance"
    ]
    assert report["test10_guard_clear_modes"] == [
        "shared_row_schur_support_gap_patch_edge_balance"
    ]
