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
        / "audit_direct_pspg_topology_policy_scope_scale.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_topology_policy_scope_scale",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _pressure_audit(update: float, *, threshold: float, support: str) -> dict:
    return {
        "status": "diagnostic_pressure_update_guard_triggered",
        "absolute_threshold_pa": threshold,
        "triggered_transition_count": 1,
        "worst_by_category": {
            "active_or_wet_supported": {
                "abs_pressure_delta_pa": update,
                "point_index": 83,
                "support_class": support,
            }
        },
    }


def _write_policy_log(path: Path, spec: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = spec["policy"]
    if spec["variant"] == "signature_row_filter":
        rows = [
            (
                "row_filter_enabled=1 row_filter_global_dof_count=48 "
                "row_filter_selected_local_row_count=1 matrix_mutated=1 "
                "source_edge_weight_sum=1.0 topology_edge_weight_sum=2.0 "
                "touched_row_count=2 schur_contribution_count=1 "
                "balance_candidate_row_count=1 max_delta_weight=0.5 "
                "max_row_abs_delta=1.0 full_cell=1"
            )
        ]
    else:
        rows = [
            (
                "row_filter_enabled=0 row_filter_global_dof_count=0 "
                "row_filter_selected_local_row_count=4 matrix_mutated=1 "
                "source_edge_weight_sum=3.0 topology_edge_weight_sum=4.0 "
                "touched_row_count=4 schur_contribution_count=2 "
                "balance_candidate_row_count=3 max_delta_weight=1.0 "
                "max_row_abs_delta=2.0 full_cell=1"
            ),
            (
                "row_filter_enabled=0 row_filter_global_dof_count=0 "
                "row_filter_selected_local_row_count=4 matrix_mutated=1 "
                "source_edge_weight_sum=5.0 topology_edge_weight_sum=6.0 "
                "touched_row_count=4 schur_contribution_count=2 "
                "balance_candidate_row_count=3 max_delta_weight=1.5 "
                "max_row_abs_delta=3.0 full_cell=1"
            ),
        ]
    path.write_text(
        "\n".join(
            "StandardAssembler: "
            "diagnostic=cut_volume_direct_pspg_topology_policy "
            f"status=applied policy={policy} {row}"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_scope_scale_rules_out_exact_row_filter(tmp_path):
    audit = _load_audit_module()
    updates = {
        ("test02", "broad_policy", "local_schur_completion"): 176849.84039557964,
        ("test02", "broad_policy", "local_edge_balance"): 176848.02921204976,
        ("test02", "broad_policy", "local_schur_edge_balance"): 176844.2140471727,
        ("test10", "broad_policy", "local_schur_completion"): 590.7292901816519,
        ("test10", "broad_policy", "local_edge_balance"): 530.3194043612839,
        ("test10", "broad_policy", "local_schur_edge_balance"): 522.4172735486616,
        ("test10", "signature_row_filter", "local_schur_completion"): (
            619.6167550623924
        ),
        ("test10", "signature_row_filter", "local_edge_balance"): 607.5173052131886,
        ("test10", "signature_row_filter", "local_schur_edge_balance"): (
            604.7126561932914
        ),
    }
    for spec in audit.REPLAYS:
        support = (
            "tiny_cut_supported"
            if spec["label"] == "test02"
            else "full_wet_supported"
        )
        threshold = 100000.0 if spec["label"] == "test02" else 100.0
        _write_json(
            tmp_path / spec["audit_name"],
            _pressure_audit(
                updates[(spec["label"], spec["variant"], spec["policy"])],
                threshold=threshold,
                support=support,
            ),
        )
        _write_policy_log(tmp_path / spec["case_dir"] / spec["log_name"], spec)

    mode_replays = tmp_path / "mode_replays.json"
    _write_json(
        mode_replays,
        {
            "case_policy_results": [
                {
                    "case": "test10",
                    "policy": "local_schur_edge_balance",
                    "same_case_no_policy_worst_active_or_wet_update_pa": (
                        622.6094100310928
                    ),
                }
            ]
        },
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        mode_replays_json=mode_replays,
    )

    assert report["finding"] == (
        "direct_pspg_topology_policy_scope_scale_rules_out_exact_row_filter"
    )
    assert report["status"] == "broad_cosupport_mutation_helpful_but_insufficient"
    assert report["same_case_no_policy_test10_update_pa"] == 622.6094100310928
    assert report["all_replays_trigger_guard"]
    assert report["signature_rows_worse_than_broad_for_all_test10_modes"]
    combined = report["test10_broad_vs_signature_row_filter"][
        "local_schur_edge_balance"
    ]
    assert combined["no_policy_to_broad_improvement_pa"] == (
        622.6094100310928 - 522.4172735486616
    )
    assert combined["signature_minus_broad_update_pa"] == (
        604.7126561932914 - 522.4172735486616
    )
    assert combined["signature_to_broad_policy_log_fraction"] == 0.5
    assert combined["signature_to_broad_topology_edge_weight_fraction"] == 0.2
    assert report["test02_broad_policy_scope"]["local_schur_edge_balance"][
        "support_class"
    ] == "tiny_cut_supported"


def test_scope_scale_reports_missing_evidence(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(
        artifact_root=tmp_path,
        mode_replays_json=tmp_path / "missing_mode.json",
    )

    assert report["finding"] == "direct_pspg_topology_policy_scope_scale_incomplete"
    assert report["status"] == "regenerate_missing_policy_scope_evidence"
    assert len(report["missing_evidence"]) == 18
