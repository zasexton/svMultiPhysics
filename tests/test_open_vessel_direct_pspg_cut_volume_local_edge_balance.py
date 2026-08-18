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
        / "audit_direct_pspg_cut_volume_local_edge_balance.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_local_edge_balance",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_map():
    return {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [100]},
        ]
    }


def _global_emission(path, candidates=None):
    return {
        "cases": [
            {
                "label": "test02",
                "path": str(path),
                "preferred_candidate_global_dofs": candidates
                if candidates is not None
                else [100, 101, 102],
            }
        ]
    }


def _summary(op, rule_index, candidate_count, touched_count):
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_direct_pspg_local_edge_balance "
        f"status=ok record=summary op='{op}' marker=7 side=Negative "
        "test='Pressure' trial='Pressure' "
        f"rule_index={rule_index} parent_cell=3 full_cell=1 "
        "volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        "local_row_count=4 source_edge_count=3 source_edge_weight_sum=6 "
        "target_scale=1 max_edge_scale=16 max_local_row_abs_sum=10 "
        "target_self_row_abs_sum=10 "
        f"balance_candidate_row_count={candidate_count} "
        f"touched_row_count={touched_count} balance_edge_count=2 "
        "balance_edge_weight_sum=4 max_row_scale=2 "
        "max_balance_delta_weight=2 max_row_abs_delta=4 "
        "constant_pressure_null_preserving=1 diagnostic_only=1"
    )


def _row(op, rule_index, row_dof, *, candidate=1, delta=4):
    row_scale = 2 if candidate else 1
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_direct_pspg_local_edge_balance "
        f"status=ok record=row op='{op}' marker=7 side=Negative "
        "test='Pressure' trial='Pressure' "
        f"rule_index={rule_index} parent_cell=3 full_cell=1 "
        "volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index=0 row_dof={row_dof} row_abs_sum=5 "
        "row_diag_abs=3 row_offdiag_abs_sum=2 source_edge_count=2 "
        "source_edge_weight_sum=4 target_self_row_abs_sum=10 "
        f"row_scale={row_scale} balance_candidate={candidate} "
        "balance_edge_count=1 balance_edge_weight_sum=2 "
        "balance_diag_delta=2 balance_offdiag_abs_delta=2 "
        f"balance_row_abs_delta={delta} balance_row_abs_ratio=0.8 "
        "constant_pressure_row_sum_delta=0 diagnostic_only=1"
    )


def test_local_edge_balance_audit_uses_latest_rule_index_reset_batch(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run_direct_pspg_cut_volume_local_edge_balance.log"
    log.write_text(
        "\n".join(
            [
                _summary(op, 10, 1, 1),
                _row(op, 10, 999),
                _summary(op, 1, 3, 3),
                _row(op, 1, 100),
                _row(op, 1, 101),
                _row(op, 1, 102),
            ]
        ),
        encoding="utf-8",
    )

    report = audit.build_report(
        global_emission=_global_emission(log),
        target_map=_target_map(),
        explicit_logs=[f"test02={log}"],
        max_target_ratio=2.0,
    )

    case = report["cases"][0]
    assert case["log_evidence"]["batch_count"] == 2
    assert case["summary_metrics"]["balance_candidate_row_count_sum"] == 3
    assert case["profile_summary"]["balance_candidate_target_count"] == 1
    assert case["selectors"][0]["finding"] == "selector_overbroad"
    assert case["selectors"][0]["selected_global_dofs"] == [100, 101, 102]
    assert report["finding"] == "direct_pspg_cut_volume_local_edge_balance_overbroad"


def test_local_edge_balance_audit_flags_missing_logs():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission=_global_emission("missing_row_provenance.log"),
        target_map=_target_map(),
    )

    assert (
        report["finding"]
        == "direct_pspg_cut_volume_local_edge_balance_evidence_missing"
    )
    assert report["missing_case_labels"] == ["test02"]


def test_local_edge_balance_audit_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log = tmp_path / "custom_local_edge_balance.log"
    op = audit.DEFAULT_OPERATOR
    log.write_text(
        "\n".join(
            [
                _summary(op, 1, 1, 1),
                _row(op, 1, 100),
            ]
        ),
        encoding="utf-8",
    )
    global_path.write_text(json.dumps(_global_emission(log, [100])), encoding="utf-8")
    target_path.write_text(json.dumps(_target_map()), encoding="utf-8")

    report = audit.build_report(
        global_emission=json.loads(global_path.read_text(encoding="utf-8")),
        target_map=json.loads(target_path.read_text(encoding="utf-8")),
        global_emission_path=global_path,
        target_map_path=target_path,
        explicit_logs=[f"test02={log}"],
        max_target_ratio=1.0,
    )

    assert report["global_emission_path"] == str(global_path)
    assert report["target_map_path"] == str(target_path)
    assert report["finding"] == "direct_pspg_cut_volume_local_edge_balance_selective"
    assert report["next_requirement"].startswith("Promote the selective local")
