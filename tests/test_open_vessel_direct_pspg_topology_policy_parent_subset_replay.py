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
        / "audit_direct_pspg_topology_policy_parent_subset_replay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_topology_policy_parent_subset_replay",
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


def _pressure_audit(update_pa: float, triggered: bool = True) -> dict:
    return {
        "absolute_threshold_pa": 100.0,
        "finding": "triggered" if triggered else "not triggered",
        "status": (
            "diagnostic_pressure_update_guard_triggered"
            if triggered
            else "diagnostic_pressure_update_guard_no_threshold_trigger"
        ),
        "transitions": [
            {
                "delta_statistics_by_category": {
                    "active_or_wet_supported": {"max_abs_delta_pa": update_pa}
                },
                "max_by_category": {
                    "active_or_wet_supported": {
                        "abs_pressure_delta_pa": update_pa,
                        "point_index": 83,
                        "pressure_delta_pa": -update_pa,
                        "support_class": "full_wet_supported",
                    }
                },
            }
        ],
    }


def _policy_line(parent: int) -> str:
    return (
        "StandardAssembler: diagnostic=cut_volume_direct_pspg_topology_policy "
        "status=applied policy=local_schur_edge_balance record=summary "
        "op='equations' "
        "source_component='navier_stokes_vms_pspg_pressure_gradient' "
        f"rule_index={parent} parent_cell={parent} full_cell=1 "
        "volume_fraction=1 measure=0.166667 parent_measure=0.166667 "
        "rule_quadrature_points=16 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=1 "
        "quadrature_policy_key=1 local_row_count=4 "
        "parent_filter_enabled=1 parent_filter_parent_cell_count=264 "
        "parent_filter_selected=1 row_filter_enabled=0 "
        "row_filter_global_dof_count=0 row_filter_selected_local_row_count=4 "
        "source_edge_count=5 source_edge_weight_sum=1e-9 "
        "topology_edge_count=7 topology_edge_weight_sum=2e-9 "
        "schur_hub_count=2 schur_contribution_count=2 "
        "balance_candidate_row_count=3 touched_row_count=4 "
        "max_delta_weight=1e-9 max_row_abs_delta=2e-9 "
        "matrix_mutated=1 solve_affecting=1 "
        "constant_pressure_null_preserving=1 diagnostic_only=0"
    )


def _write_required_inputs(audit, root: Path, parent_triggered: bool = True):
    _write_json(
        root / "readiness.json",
        {
            "finding": "direct_pspg_signature_parent_subset_replay_ready",
            "status": "run_signature_parent_full_local_replay",
            "source_hook": {"parent_cell_filter_api_present": True},
            "same_signature_parent_set_all_policies": True,
            "signature_parent_cell_count": 264,
            "signature_parent_cell_ranges": "1-4",
        },
    )
    _write_json(
        root / "parent_scope.json",
        {
            "finding": (
                "direct_pspg_topology_policy_parent_scope_rules_out_exact_parent_subset"
            ),
            "status": "broad_parent_cosupport_required_but_insufficient",
            "test10_parent_rule_scope": {
                "local_schur_edge_balance": {
                    "rule_scope": {
                        "broad_key_count": 720,
                        "signature_key_count": 264,
                        "broad_only_key_count": 456,
                    }
                }
            },
        },
    )
    updates = {
        "same_case_no_policy": 622.609409861916,
        "broad_policy": 522.4172735486616,
        "signature_row_filter": 604.7126561932914,
        "signature_parent_filter": 578.9424523317655,
    }
    for label, spec in audit.REPLAYS.items():
        _write_json(
            root / spec["audit_name"],
            _pressure_audit(
                updates[label],
                triggered=(label != "signature_parent_filter" or parent_triggered),
            ),
        )
        log_name = spec.get("log_name")
        if log_name:
            path = root / spec["case_dir"] / log_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(_policy_line(parent) for parent in range(1, 265))
                + "\n",
                encoding="utf-8",
            )


def test_parent_subset_replay_rules_out_exact_parent_subset(tmp_path):
    audit = _load_audit_module()
    _write_required_inputs(audit, tmp_path, parent_triggered=True)

    report = audit.build_report(
        artifact_root=tmp_path,
        readiness_json=tmp_path / "readiness.json",
        parent_scope_json=tmp_path / "parent_scope.json",
    )

    assert report["finding"] == (
        "direct_pspg_signature_parent_subset_full_local_replay_"
        "does_not_clear_test10_guard"
    )
    assert report["status"] == "exact_parent_subset_ruled_out_as_sufficient_fix"
    assert report["signature_parent_filter_full_local_confirmed"]
    assert report["signature_parent_filter_update_pa"] == 578.9424523317655
    assert report["broad_policy_update_pa"] == 522.4172735486616
    assert report["signature_row_filter_update_pa"] == 604.7126561932914
    assert report["pressure_update_guard_cleared"]["signature_parent_filter"] is False
    assert "physical support-patch closure" in report["next_requirement"]


def test_parent_subset_replay_cleared_requires_test02_transfer(tmp_path):
    audit = _load_audit_module()
    _write_required_inputs(audit, tmp_path, parent_triggered=False)

    report = audit.build_report(
        artifact_root=tmp_path,
        readiness_json=tmp_path / "readiness.json",
        parent_scope_json=tmp_path / "parent_scope.json",
    )

    assert report["finding"] == (
        "direct_pspg_signature_parent_subset_full_local_replay_clears_test10_guard"
    )
    assert report["status"] == "requires_test02_transfer_check"
    assert report["pressure_update_guard_cleared"]["signature_parent_filter"] is True
    assert "Test02 short replay" in report["next_requirement"]


def test_parent_subset_replay_reports_missing_inputs(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(
        artifact_root=tmp_path,
        readiness_json=tmp_path / "readiness.json",
        parent_scope_json=tmp_path / "parent_scope.json",
    )

    assert report["finding"] == (
        "direct_pspg_signature_parent_subset_full_local_replay_incomplete"
    )
    assert report["status"] == "regenerate_missing_parent_subset_replay_inputs"
    assert len(report["missing_evidence"]) == 9
