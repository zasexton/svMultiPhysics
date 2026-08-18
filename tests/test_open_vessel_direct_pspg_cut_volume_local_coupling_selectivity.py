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
        / "audit_direct_pspg_cut_volume_local_coupling_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_local_coupling_selectivity",
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


def _entry(op, trial, rule_index, row_dof, row_abs_sum, parent_cell=3):
    col_count = 4 if trial == "Pressure" else 12
    has_diag = 1 if trial == "Pressure" else 0
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_local_matrix_row_provenance "
        f"status=ok op='{op}' marker=7 side=Negative test='Pressure' "
        f"trial='{trial}' rule_index={rule_index} parent_cell={parent_cell} "
        "full_cell=1 volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index=0 row_dof={row_dof} col_count={col_count} "
        f"row_abs_sum={row_abs_sum} row_abs_fraction=0.5 "
        "row_signed_sum=0 positive_sum=0 negative_abs_sum=0 "
        f"nonzero_count=2 positive_count=1 negative_count=1 has_diag={has_diag} "
        f"diag_value={row_abs_sum / 2.0 if has_diag else 0.0} "
        f"diag_abs={row_abs_sum / 2.0 if has_diag else 0.0} "
        f"offdiag_abs_sum={row_abs_sum / 2.0 if has_diag else row_abs_sum} "
        f"max_abs_entry={row_abs_sum / 2.0} max_abs_col_dof={row_dof} "
        "max_abs_col_local_index=0"
    )


def _case_log(op, rows):
    lines = []
    for rule_index, row, pressure_abs, velocity_abs in rows:
        lines.append(_entry(op, "Pressure", rule_index, row, pressure_abs))
        lines.append(_entry(op, "Velocity", rule_index, row, velocity_abs))
    return "\n".join(lines)


def test_local_coupling_audit_uses_latest_rule_index_reset_batch(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _case_log(op, [(10, 100, 10.0, 5.0), (11, 101, 10.0, 8.0)]),
                _case_log(op, [(1, 100, 10.0, 0.0), (2, 101, 10.0, 5.0)]),
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
        max_target_ratio=1.0,
    )

    case = report["cases"][0]
    assert case["log_evidence"]["pressure"]["batch_count"] == 2
    assert case["log_evidence"]["velocity"]["batch_count"] == 2
    assert case["profile_summary"]["target_profiles"]["100"][
        "velocity_to_pressure_abs_ratio"
    ] == 0.0
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    zero_velocity = selectors["cross_field_zero_velocity_action"]["cases"][0]
    assert zero_velocity["finding"] == "selector_selective"
    assert zero_velocity["covered_direct_target_global_dofs"] == [100]


def test_local_coupling_audit_flags_missing_logs():
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

    assert report["finding"] == "direct_pspg_cut_volume_local_coupling_evidence_missing"
    assert report["missing_case_labels"] == ["test02"]


def test_local_coupling_audit_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log_path = tmp_path / "run.log"
    op = audit.DEFAULT_OPERATOR
    log_path.write_text(
        _case_log(op, [(1, 100, 10.0, 0.0), (2, 101, 10.0, 5.0)]),
        encoding="utf-8",
    )
    global_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": "test02",
                        "path": str(log_path),
                        "preferred_candidate_global_dofs": [100, 101],
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
        max_target_ratio=1.0,
    )

    assert report["global_emission_path"] == str(global_path)
    assert report["target_map_path"] == str(target_path)
    assert report["finding"] == "direct_pspg_cut_volume_local_coupling_selector_selective"
