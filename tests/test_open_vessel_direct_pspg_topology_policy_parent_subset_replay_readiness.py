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
        / "audit_direct_pspg_topology_policy_parent_subset_replay_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_topology_policy_parent_subset_replay_readiness",
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


def _policy_log(policy: str, parent_cells: list[int]) -> str:
    rows = []
    for parent in parent_cells:
        rows.append(
            "StandardAssembler: "
            "diagnostic=cut_volume_direct_pspg_topology_policy "
            f"status=applied policy={policy} record=summary op='equations' "
            "source_component='navier_stokes_vms_pspg_pressure_gradient' "
            "marker=1601158 side=Negative test='Pressure' trial='Pressure' "
            f"rule_index={parent + 10} parent_cell={parent} "
            "full_cell=1 volume_fraction=1 measure=0.166667 "
            "parent_measure=0.166667 rule_quadrature_points=16 "
            "active_quadrature_points=4 source_revision=1 "
            f"cut_topology_revision={1000 + parent} "
            "quadrature_policy_key=4160702276957219031 local_row_count=4 "
            "row_filter_enabled=1 row_filter_global_dof_count=48 "
            "row_filter_selected_local_row_count=1 source_edge_count=2 "
            "source_edge_weight_sum=1 topology_edge_count=2 "
            "topology_edge_weight_sum=1 schur_hub_count=1 "
            "schur_contribution_count=1 balance_candidate_row_count=1 "
            "touched_row_count=3 max_delta_weight=1 max_row_abs_delta=1 "
            "matrix_mutated=1 solve_affecting=1 "
            "constant_pressure_null_preserving=1 diagnostic_only=0"
        )
    return "\n".join(rows) + "\n"


def _write_signature_logs(audit, root: Path, parent_cells: list[int]):
    for spec in audit.SIGNATURE_REPLAYS:
        path = root / spec["case_dir"] / spec["log_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_policy_log(spec["policy"], parent_cells), encoding="utf-8")


def _parent_scope() -> dict:
    return {
        "finding": (
            "direct_pspg_topology_policy_parent_scope_rules_out_exact_parent_subset"
        ),
        "status": "broad_parent_cosupport_required_but_insufficient",
        "all_test10_signature_parent_rule_sets_are_strict_broad_subsets": True,
        "all_test10_broad_only_rule_weight_share_above_half": True,
        "test10_parent_rule_scope": {
            "local_schur_edge_balance": {
                "rule_scope": {
                    "broad_key_count": 6,
                    "signature_key_count": 3,
                    "broad_only_key_count": 3,
                    "broad_only_topology_edge_weight_sum_fraction": 0.75,
                    "signature_to_broad_overlap_topology_edge_weight_sum_fraction": (
                        0.5
                    ),
                    "signature_to_broad_topology_edge_weight_sum_fraction": 0.125,
                },
            },
        },
    }


def _source_with_parent_filter() -> str:
    return "\n".join(
        [
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELLS",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELL_IDS",
            "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_PARENT_CELLS_FILTER",
            "parent_filter_enabled",
            "parent_filter_parent_cell_count",
            "parent_filter_selected=1",
            "cutVolumeDirectPspgTopologyParentCellFilter",
        ]
    )


def test_parent_subset_replay_ready_with_parent_filter(tmp_path):
    audit = _load_audit_module()
    parent_cells = [1, 2, 3]
    _write_signature_logs(audit, tmp_path, parent_cells)
    parent_scope = tmp_path / "parent_scope.json"
    _write_json(parent_scope, _parent_scope())
    source = tmp_path / "StandardAssembler.cpp"
    source.write_text(_source_with_parent_filter(), encoding="utf-8")

    report = audit.build_report(
        artifact_root=tmp_path,
        parent_scope_json=parent_scope,
        standard_assembler=source,
    )

    assert report["finding"] == "direct_pspg_signature_parent_subset_replay_ready"
    assert report["status"] == "run_signature_parent_full_local_replay"
    assert report["source_hook"]["parent_cell_filter_api_present"]
    assert report["same_signature_parent_set_all_policies"]
    assert report["signature_parent_cell_count"] == 3
    assert report["signature_parent_cells"] == parent_cells
    assert report["signature_parent_cells_csv"] == "1,2,3"
    assert report["signature_parent_cell_ranges"] == "1-3"
    assert report["parent_scope"]["strict_parent_rule_subset"]
    assert "no global row DOF filter" in report["next_requirement"]


def test_parent_subset_replay_reports_missing_parent_filter(tmp_path):
    audit = _load_audit_module()
    _write_signature_logs(audit, tmp_path, [1, 2, 3])
    parent_scope = tmp_path / "parent_scope.json"
    _write_json(parent_scope, _parent_scope())
    source = tmp_path / "StandardAssembler.cpp"
    source.write_text(
        "\n".join(
            [
                "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY",
                "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS",
            ]
        ),
        encoding="utf-8",
    )

    report = audit.build_report(
        artifact_root=tmp_path,
        parent_scope_json=parent_scope,
        standard_assembler=source,
    )

    assert report["finding"] == (
        "direct_pspg_signature_parent_subset_replay_blocked_by_filter_api"
    )
    assert report["status"] == "add_parent_cell_filter_to_topology_policy_hook"
    assert not report["source_hook"]["parent_cell_filter_api_present"]


def test_parent_subset_replay_reports_missing_inputs(tmp_path):
    audit = _load_audit_module()

    report = audit.build_report(
        artifact_root=tmp_path,
        parent_scope_json=tmp_path / "missing_parent_scope.json",
        standard_assembler=tmp_path / "missing.cpp",
    )

    assert report["finding"] == (
        "direct_pspg_signature_parent_subset_replay_readiness_incomplete"
    )
    assert report["status"] == "regenerate_missing_parent_subset_inputs"
    assert len(report["missing_evidence"]) == 5
