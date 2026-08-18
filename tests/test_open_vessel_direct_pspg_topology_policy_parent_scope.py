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
        / "audit_direct_pspg_topology_policy_parent_scope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_topology_policy_parent_scope",
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


def _record(
    *,
    policy: str,
    parent: int,
    topology_weight: float,
    touched_rows: int,
    selected_rows: int,
    matrix_mutated: int,
    full_cell: int,
) -> str:
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_direct_pspg_topology_policy "
        f"status=applied policy={policy} record=summary op='equations' "
        "source_component='navier_stokes_vms_pspg_pressure_gradient' "
        "marker=1601158 side=Negative test='Pressure' trial='Pressure' "
        f"rule_index={parent + 10} parent_cell={parent} "
        f"full_cell={full_cell} volume_fraction=1 measure=0.166667 "
        "parent_measure=0.166667 rule_quadrature_points=16 "
        "active_quadrature_points=4 source_revision=1 "
        f"cut_topology_revision={1000 + parent} "
        "quadrature_policy_key=4160702276957219031 local_row_count=4 "
        f"row_filter_selected_local_row_count={selected_rows} "
        "source_edge_count=5 "
        f"source_edge_weight_sum={topology_weight * 2.0} "
        "topology_edge_count=7 "
        f"topology_edge_weight_sum={topology_weight} "
        "schur_hub_count=2 schur_contribution_count=2 "
        "balance_candidate_row_count=3 "
        f"touched_row_count={touched_rows} max_delta_weight=1 "
        "max_row_abs_delta=2 "
        f"matrix_mutated={matrix_mutated} solve_affecting=1 "
        "constant_pressure_null_preserving=1 diagnostic_only=0"
    )


def _write_policy_log(path: Path, spec: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = spec["policy"]
    if spec["variant"] == "signature_row_filter":
        rows = [
            _record(
                policy=policy,
                parent=1,
                topology_weight=4.0,
                touched_rows=2,
                selected_rows=1,
                matrix_mutated=1,
                full_cell=1,
            ),
            _record(
                policy=policy,
                parent=2,
                topology_weight=6.0,
                touched_rows=2,
                selected_rows=1,
                matrix_mutated=1,
                full_cell=1,
            ),
        ]
    else:
        full_cell = 0 if spec["label"] == "test02" else 1
        rows = [
            _record(
                policy=policy,
                parent=1,
                topology_weight=10.0,
                touched_rows=4,
                selected_rows=4,
                matrix_mutated=1,
                full_cell=full_cell,
            ),
            _record(
                policy=policy,
                parent=2,
                topology_weight=10.0,
                touched_rows=4,
                selected_rows=4,
                matrix_mutated=1,
                full_cell=full_cell,
            ),
            _record(
                policy=policy,
                parent=3,
                topology_weight=40.0,
                touched_rows=4,
                selected_rows=4,
                matrix_mutated=1,
                full_cell=full_cell,
            ),
            _record(
                policy=policy,
                parent=4,
                topology_weight=40.0,
                touched_rows=4,
                selected_rows=4,
                matrix_mutated=1,
                full_cell=full_cell,
            ),
        ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_parent_scope_rules_out_exact_parent_subset(tmp_path):
    audit = _load_audit_module()
    updates = {
        ("test02", "broad_policy", "local_schur_completion"): 176849.0,
        ("test02", "broad_policy", "local_edge_balance"): 176848.0,
        ("test02", "broad_policy", "local_schur_edge_balance"): 176844.0,
        ("test10", "broad_policy", "local_schur_completion"): 800.0,
        ("test10", "broad_policy", "local_edge_balance"): 700.0,
        ("test10", "broad_policy", "local_schur_edge_balance"): 600.0,
        ("test10", "signature_row_filter", "local_schur_completion"): 900.0,
        ("test10", "signature_row_filter", "local_edge_balance"): 850.0,
        ("test10", "signature_row_filter", "local_schur_edge_balance"): 800.0,
    }
    for spec in audit.REPLAYS:
        support = (
            "tiny_cut_supported"
            if spec["label"] == "test02"
            else "full_wet_supported"
        )
        _write_json(
            tmp_path / spec["audit_name"],
            _pressure_audit(
                updates[(spec["label"], spec["variant"], spec["policy"])],
                threshold=100.0,
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
                    "same_case_no_policy_worst_active_or_wet_update_pa": 1000.0,
                }
            ]
        },
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        mode_replays_json=mode_replays,
    )

    assert report["finding"] == (
        "direct_pspg_topology_policy_parent_scope_rules_out_exact_parent_subset"
    )
    assert report["status"] == "broad_parent_cosupport_required_but_insufficient"
    assert report["all_replays_trigger_guard"]
    assert report[
        "all_test10_signature_parent_rule_sets_are_strict_broad_subsets"
    ]
    assert report["all_test10_broad_only_rule_weight_share_above_half"]
    combined = report["test10_parent_rule_scope"]["local_schur_edge_balance"]
    assert combined["signature_minus_broad_update_pa"] == 200.0
    assert combined["no_policy_to_broad_improvement_pa"] == 400.0
    parent_scope = combined["parent_scope"]
    assert parent_scope["broad_key_count"] == 4
    assert parent_scope["signature_key_count"] == 2
    assert parent_scope["broad_only_key_count"] == 2
    assert parent_scope["signature_to_broad_key_fraction"] == 0.5
    assert parent_scope["broad_only_topology_edge_weight_sum_fraction"] == 0.8
    assert parent_scope[
        "signature_to_broad_overlap_topology_edge_weight_sum_fraction"
    ] == 0.5
    assert parent_scope[
        "signature_to_broad_topology_edge_weight_sum_fraction"
    ] == 0.1
    rule_scope = combined["rule_scope"]
    assert rule_scope["broad_only_topology_edge_weight_sum_fraction"] == 0.8
    assert report["test02_broad_parent_rule_scope"]["local_schur_edge_balance"][
        "support_class"
    ] == "tiny_cut_supported"
    assert report["test02_broad_parent_rule_scope"]["local_schur_edge_balance"][
        "parent_scope"
    ]["broad_cut_cell_record_count"] == 4.0
    assert "connected support-patch" in report["next_requirement"]


def test_parent_scope_reports_missing_evidence(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(
        artifact_root=tmp_path,
        mode_replays_json=tmp_path / "missing_mode.json",
    )

    assert report["finding"] == "direct_pspg_topology_policy_parent_scope_incomplete"
    assert report["status"] == "regenerate_missing_parent_scope_evidence"
    assert len(report["missing_evidence"]) == 18
