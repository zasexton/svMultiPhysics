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
        / "audit_direct_pspg_topology_policy_application_effect.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_topology_policy_application_effect",
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


def _pressure_audit(update: float, threshold: float, support: str) -> dict:
    return {
        "status": "diagnostic_pressure_update_guard_triggered",
        "finding": (
            f"1 transition exceeded {threshold:g} Pa. "
            f"Worst active/wet update was {update:.3f} Pa."
        ),
        "absolute_threshold_pa": threshold,
        "triggered_transition_count": 1,
        "worst_by_category": {
            "active_or_wet_supported": {
                "abs_pressure_delta_pa": update,
                "point_index": 83,
                "support_class": support,
                "active_fluid": 1.0 if support == "full_wet_supported" else 0.0,
                "incident_wet_fraction_min_positive": 1.0,
            },
            "full_wet_supported": {
                "abs_pressure_delta_pa": update,
                "point_index": 83,
                "support_class": support,
                "active_fluid": 1.0 if support == "full_wet_supported" else 0.0,
                "incident_wet_fraction_min_positive": 1.0,
            },
        },
    }


def _write_log(path: Path, spec: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = spec["policy"]
    if spec["variant"] == "signature_row_filter":
        lines = [
            (
                "StandardAssembler: "
                "diagnostic=cut_volume_direct_pspg_topology_policy "
                f"status=applied policy={policy} row_filter_enabled=1 "
                "row_filter_global_dof_count=48 "
                "row_filter_selected_local_row_count=1 matrix_mutated=1 "
                "touched_row_count=2 balance_candidate_row_count=1 "
                "schur_contribution_count=1 max_delta_weight=1.5e-09 "
                "max_row_abs_delta=3.0e-09 full_cell=1"
            ),
            (
                "StandardAssembler: "
                "diagnostic=cut_volume_direct_pspg_topology_policy "
                f"status=applied policy={policy} row_filter_enabled=1 "
                "row_filter_global_dof_count=48 "
                "row_filter_selected_local_row_count=2 matrix_mutated=0 "
                "touched_row_count=0 balance_candidate_row_count=0 "
                "schur_contribution_count=0 max_delta_weight=0 "
                "max_row_abs_delta=0 full_cell=1"
            ),
        ]
    else:
        lines = [
            (
                "StandardAssembler: "
                "diagnostic=cut_volume_direct_pspg_topology_policy "
                f"status=applied policy={policy} matrix_mutated=1 "
                "touched_row_count=4 balance_candidate_row_count=3 "
                "schur_contribution_count=2 max_delta_weight=2.5e-09 "
                "max_row_abs_delta=5.0e-09 full_cell=0"
            )
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_application_effect_rules_out_policy_underapplication(tmp_path):
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
    for spec in audit.BROAD_REPLAYS + audit.SIGNATURE_ROW_REPLAYS:
        key = (spec["label"], spec["variant"], spec["policy"])
        support = (
            "tiny_cut_supported"
            if spec["label"] == "test02"
            else "full_wet_supported"
        )
        threshold = 100000.0 if spec["label"] == "test02" else 100.0
        _write_json(
            tmp_path / spec["audit_name"],
            _pressure_audit(updates[key], threshold, support),
        )
        _write_log(tmp_path / spec["case_dir"] / spec["log_name"], spec)

    report = audit.build_report(artifact_root=tmp_path)

    assert report["finding"] == (
        "direct_pspg_topology_policy_application_effect_rules_out_"
        "underapplication"
    )
    assert report["status"] == "local_matrix_policy_applies_but_is_not_sufficient_fix"
    assert report["all_replays_trigger_guard"]
    assert report["all_test10_signature_replays_mutate_selected_records"]
    assert report["best_test02_broad_policy"] == "local_schur_edge_balance"
    assert report["best_test02_broad_update_pa"] == 176844.2140471727
    assert report["best_test10_broad_policy"] == "local_schur_edge_balance"
    assert report["best_test10_signature_policy"] == "local_schur_edge_balance"
    assert report["test10_broad_vs_signature_row_filter"][
        "local_schur_edge_balance"
    ]["signature_minus_broad_update_pa"] == (
        604.7126561932914 - 522.4172735486616
    )
    assert report["case_policy_matrix"]["test10"]["signature_row_filter"][
        "local_edge_balance"
    ]["matrix_mutated_count"] == 1


def test_application_effect_reports_missing_evidence(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(artifact_root=tmp_path)

    assert report["finding"] == (
        "direct_pspg_topology_policy_application_effect_incomplete"
    )
    assert report["status"] == "regenerate_missing_policy_application_evidence"
    assert len(report["missing_evidence"]) == 18
