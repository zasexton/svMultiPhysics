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
        / "audit_direct_pspg_test10_signature_row_filter_replays.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_test10_signature_row_filter_replays",
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


def _pressure_audit(update: float, *, point: int = 83) -> dict:
    return {
        "status": "diagnostic_pressure_update_guard_triggered",
        "finding": (
            "1 transition(s) exceeded 100 Pa on active/wet support. "
            f"Worst active/wet update was {update:.3f} Pa."
        ),
        "absolute_threshold_pa": 100.0,
        "triggered_transition_count": 1,
        "worst_by_category": {
            "active_or_wet_supported": {
                "abs_pressure_delta_pa": update,
                "point_index": point,
                "support_class": "full_wet_supported",
                "active_fluid": 1.0,
                "incident_wet_fraction_min_positive": 1.0,
            },
            "full_wet_supported": {
                "abs_pressure_delta_pa": update,
                "point_index": point,
                "support_class": "full_wet_supported",
                "active_fluid": 1.0,
                "incident_wet_fraction_min_positive": 1.0,
            },
            "cut_supported": {
                "abs_pressure_delta_pa": 344.0,
                "point_index": 609,
                "support_class": "cut_supported",
                "active_fluid": 0.0,
                "incident_wet_fraction_min_positive": 0.32,
            },
        },
    }


def _write_log(path: Path, policy: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                (
                    "StandardAssembler: "
                    "diagnostic=cut_volume_direct_pspg_topology_policy "
                    f"status=applied policy={policy} "
                    "row_filter_enabled=1 row_filter_global_dof_count=48 "
                    "row_filter_selected_local_row_count=1 "
                    "matrix_mutated=1"
                ),
                (
                    "StandardAssembler: "
                    "diagnostic=cut_volume_direct_pspg_topology_policy "
                    f"status=applied policy={policy} "
                    "row_filter_enabled=1 row_filter_global_dof_count=48 "
                    "row_filter_selected_local_row_count=2 "
                    "matrix_mutated=0"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_signature_row_filter_replay_family_rules_out_triggered_modes(tmp_path):
    audit = _load_audit_module()
    updates = {
        "local_schur_completion": 619.6167550623924,
        "local_edge_balance": 607.5173052131886,
        "local_schur_edge_balance": 604.7126561932914,
    }
    for spec in audit.REPLAYS:
        _write_json(
            tmp_path / spec["audit_name"],
            _pressure_audit(updates[spec["policy"]]),
        )
        _write_log(
            tmp_path / spec["case_dir"] / spec["log_name"],
            spec["policy"],
        )

    report = audit.build_report(artifact_root=tmp_path)

    assert report["finding"] == (
        "test10_signature_row_filter_local_modes_do_not_clear_guard"
    )
    assert report["status"] == (
        "signature_row_filter_local_modes_ruled_out_as_sufficient_fix"
    )
    assert report["row_filter_global_dof_counts"] == [48]
    assert report["all_replays_trigger_guard"]
    assert report["cleared_policies"] == []
    assert report["best_policy_by_worst_update"] == "local_schur_edge_balance"
    assert report["best_worst_active_or_wet_update_pa"] == 604.7126561932914
    assert {item["topology_log"]["row_filter_log_count"] for item in report["replays"]} == {
        2
    }
    assert {item["topology_log"]["matrix_mutated_count"] for item in report["replays"]} == {
        1
    }


def test_signature_row_filter_replay_family_reports_missing_artifacts(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(artifact_root=tmp_path)

    assert report["finding"] == "test10_signature_row_filter_replay_family_incomplete"
    assert report["status"] == "regenerate_missing_replay_artifacts"
    assert len(report["missing_evidence"]) == 6
