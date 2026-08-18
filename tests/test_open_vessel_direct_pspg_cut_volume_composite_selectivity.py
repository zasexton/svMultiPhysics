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
        / "audit_direct_pspg_cut_volume_composite_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_composite_selectivity",
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


def _entry(op, trial, rule_index, parent_cell, row_dof, row_abs_sum, row_local_index):
    col_count = 4 if trial == "Pressure" else 12
    has_diag = 1 if trial == "Pressure" else 0
    diag_abs = row_abs_sum / 2.0 if has_diag else 0.0
    offdiag_abs = row_abs_sum - diag_abs
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_local_matrix_row_provenance "
        f"status=ok op='{op}' marker=7 side=Negative test='Pressure' "
        f"trial='{trial}' rule_index={rule_index} parent_cell={parent_cell} "
        "full_cell=1 volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index={row_local_index} row_dof={row_dof} col_count={col_count} "
        f"row_abs_sum={row_abs_sum} row_abs_fraction=0.5 "
        "row_signed_sum=0 positive_sum=0 negative_abs_sum=0 "
        f"nonzero_count=2 positive_count=1 negative_count=1 has_diag={has_diag} "
        f"diag_value={diag_abs} diag_abs={diag_abs} offdiag_abs_sum={offdiag_abs} "
        f"max_abs_entry={row_abs_sum / 2.0} max_abs_col_dof={row_dof} "
        "max_abs_col_local_index=0"
    )


def _cell(op, rule_index, parent_cell, rows, pressure_abs, velocity_abs):
    lines = []
    for index, row in enumerate(rows):
        lines.append(
            _entry(
                op,
                "Pressure",
                rule_index,
                parent_cell,
                row,
                pressure_abs[row],
                index,
            )
        )
        lines.append(
            _entry(
                op,
                "Velocity",
                rule_index,
                parent_cell,
                row,
                velocity_abs[row],
                index,
            )
        )
    return "\n".join(lines)


def _case_log(op, rows, pressure_abs, velocity_abs):
    return "\n".join(
        [
            _cell(op, 1, 10, rows[:3], pressure_abs, velocity_abs),
            _cell(op, 2, 11, rows[2:], pressure_abs, velocity_abs),
        ]
    )


def test_composite_audit_uses_latest_batches_and_builds_features(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _case_log(
                    op,
                    [100, 101, 102, 103, 104],
                    {100: 20.0, 101: 10.0, 102: 10.0, 103: 10.0, 104: 10.0},
                    {100: 5.0, 101: 5.0, 102: 5.0, 103: 5.0, 104: 5.0},
                ),
                _case_log(
                    op,
                    [100, 101, 102, 103, 104],
                    {100: 1.0, 101: 10.0, 102: 10.0, 103: 10.0, 104: 10.0},
                    {100: 100.0, 101: 1.0, 102: 5.0, 103: 5.0, 104: 1.0},
                ),
            ]
        ),
        encoding="utf-8",
    )
    global_emission = {
        "cases": [
            {
                "label": "test02",
                "path": str(log),
                "preferred_candidate_global_dofs": [100, 101, 102, 103, 104],
            }
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
        max_target_ratio=5.0,
    )

    case = report["cases"][0]
    assert case["log_evidence"]["pressure"]["batch_count"] == 2
    assert case["log_evidence"]["velocity"]["batch_count"] == 2
    target_profile = case["profile_summary"]["target_profiles"]["100"]
    assert target_profile["pressure_total_row_abs_sum"] == 1.0
    assert target_profile["velocity_to_pressure_abs_ratio"] == 100.0
    assert target_profile["row_parent_graph_degree"] == 2

    selectors = {selector["key"]: selector for selector in report["selectors"]}
    graph_tail = selectors["composite_graph_bimodal_tail"]["cases"][0]
    assert graph_tail["finding"] == "selector_selective"
    assert graph_tail["covered_direct_target_global_dofs"] == [100]


def test_composite_audit_flags_missing_logs():
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

    assert report["finding"] == "direct_pspg_cut_volume_composite_evidence_missing"
    assert report["missing_case_labels"] == ["test02"]


def test_composite_audit_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log_path = tmp_path / "run.log"
    op = audit.DEFAULT_OPERATOR
    log_path.write_text(
        _case_log(
            op,
            [100, 101, 102, 103, 104],
            {100: 1.0, 101: 10.0, 102: 10.0, 103: 10.0, 104: 10.0},
            {100: 100.0, 101: 1.0, 102: 5.0, 103: 5.0, 104: 1.0},
        ),
        encoding="utf-8",
    )
    global_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "label": "test02",
                        "path": str(log_path),
                        "preferred_candidate_global_dofs": [100, 101, 102, 103, 104],
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
        max_target_ratio=5.0,
    )

    assert report["global_emission_path"] == str(global_path)
    assert report["target_map_path"] == str(target_path)
    assert report["finding"] == "direct_pspg_cut_volume_composite_selector_selective"
