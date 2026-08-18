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
        / "audit_direct_pspg_cut_volume_column_support_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_column_support_readiness",
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


def _entry(op, rule_index, row_dof, col_dofs, values, parent_cell=3):
    row_abs_sum = sum(abs(value) for value in values)
    row_signed_sum = sum(values)
    positive_sum = sum(value for value in values if value > 0.0)
    negative_abs_sum = sum(-value for value in values if value < 0.0)
    diag_value = sum(
        value for col_dof, value in zip(col_dofs, values) if col_dof == row_dof
    )
    diag_abs = sum(
        abs(value) for col_dof, value in zip(col_dofs, values) if col_dof == row_dof
    )
    offdiag_abs = row_abs_sum - diag_abs
    signs = [1 if value > 0.0 else -1 for value in values]
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_local_matrix_column_support "
        f"status=ok op='{op}' marker=7 side=Negative test='Pressure' "
        "trial='Pressure' "
        f"rule_index={rule_index} parent_cell={parent_cell} full_cell=1 "
        "volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index=0 row_dof={row_dof} col_count={len(col_dofs)} "
        f"nonzero_col_count={len(col_dofs)} "
        f"positive_col_count={sum(1 for value in values if value > 0.0)} "
        f"negative_col_count={sum(1 for value in values if value < 0.0)} "
        f"sampled_col_count={len(col_dofs)} sample_truncated=0 "
        "sample_sorted_by=abs_desc "
        f"row_abs_sum={row_abs_sum} row_signed_sum={row_signed_sum} "
        f"positive_sum={positive_sum} negative_abs_sum={negative_abs_sum} "
        f"has_diag={1 if diag_abs else 0} diag_in_sample={1 if diag_abs else 0} "
        f"diag_value={diag_value} diag_abs={diag_abs} "
        f"offdiag_abs_sum={offdiag_abs} "
        "sampled_col_local_indices="
        f"{'|'.join(str(index) for index, _ in enumerate(col_dofs))} "
        f"sampled_col_dofs={'|'.join(str(col_dof) for col_dof in col_dofs)} "
        f"sampled_col_values={'|'.join(str(value) for value in values)} "
        f"sampled_col_abs_values={'|'.join(str(abs(value)) for value in values)} "
        f"sampled_col_signs={'|'.join(str(sign) for sign in signs)}"
    )


def test_column_support_readiness_uses_latest_rule_index_reset_batch(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _entry(op, 10, 100, [100, 101], [20.0, -5.0]),
                _entry(op, 11, 101, [100, 101], [-5.0, 20.0]),
                _entry(op, 1, 100, [100, 101, 102], [2.0, -1.0, -1.0]),
                _entry(op, 2, 101, [100, 101, 102], [1.0, 2.0, -1.0]),
            ]
        ),
        encoding="utf-8",
    )
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "path": str(log),
                "preferred_candidate_global_dofs": [100, 101],
            }
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
    )

    case = report["cases"][0]
    assert report["finding"] == "direct_pspg_cut_volume_column_support_evidence_ready"
    assert case["log_evidence"]["batch_count"] == 2
    target_profile = case["profile_summary"]["target_profiles"]["100"]
    assert target_profile["pressure_row_abs_sum"] == 4.0
    assert target_profile["pressure_row_signed_sum"] == 0.0
    assert target_profile["sampled_col_dofs"] == [100, 101, 102]
    assert target_profile["negative_sampled_offdiag_col_dofs"] == [101, 102]
    assert target_profile["positive_sampled_offdiag_col_count"] == 0
    assert target_profile["sampled_offdiag_signed_balance_ratio"] == 1.0
    assert target_profile["all_rules_diag_in_sample"]


def test_column_support_readiness_flags_missing_logs():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "cases": [
                {
                    "label": "test02",
                    "path": "missing.log",
                    "preferred_candidate_global_dofs": [100, 101],
                }
            ]
        },
        target_map=_target_map(),
    )

    assert report["finding"] == "direct_pspg_cut_volume_column_support_evidence_missing"
    assert report["missing_case_labels"] == ["test02"]


def test_column_support_readiness_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log_path = tmp_path / "run.log"
    op = audit.DEFAULT_OPERATOR
    log_path.write_text(
        _entry(op, 1, 100, [100, 101, 102], [2.0, -1.0, -1.0]),
        encoding="utf-8",
    )
    global_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": "test02",
                        "path": str(log_path),
                        "preferred_candidate_global_dofs": [100],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    target_path.write_text(json.dumps(_target_map()), encoding="utf-8")

    report = audit.build_report(
        global_emission=json.loads(global_path.read_text(encoding="utf-8")),
        target_map=json.loads(target_path.read_text(encoding="utf-8")),
        global_emission_path=global_path,
        target_map_path=target_path,
    )

    assert report["global_emission_path"] == str(global_path)
    assert report["target_map_path"] == str(target_path)
    assert report["cases"][0]["profile_summary"]["profiled_target_count"] == 1
