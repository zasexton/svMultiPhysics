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
        / "audit_direct_pspg_cut_volume_parent_graph_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_parent_graph_selectivity",
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


def _entry(op, rule_index, parent_cell, row_dof, row_local_index):
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_local_matrix_row_provenance "
        f"status=ok op='{op}' marker=7 side=Negative test='Pressure' "
        "trial='Pressure' "
        f"rule_index={rule_index} parent_cell={parent_cell} full_cell=1 "
        "volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index={row_local_index} row_dof={row_dof} col_count=4 "
        "row_abs_sum=1 row_abs_fraction=0.25 row_signed_sum=0 "
        "positive_sum=0 negative_abs_sum=0 nonzero_count=2 "
        "positive_count=1 negative_count=1 has_diag=1 diag_value=0.5 "
        "diag_abs=0.5 offdiag_abs_sum=0.5 max_abs_entry=0.5 "
        f"max_abs_col_dof={row_dof} max_abs_col_local_index=0"
    )


def _cell(op, rule_index, parent_cell, rows):
    return "\n".join(
        _entry(op, rule_index, parent_cell, row, i)
        for i, row in enumerate(rows)
    )


def test_parent_graph_audit_builds_shared_parent_degree(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                _cell(op, 1, 10, [100, 101, 102]),
                _cell(op, 2, 11, [101, 102, 103]),
                _cell(op, 3, 12, [102, 103, 104]),
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
        max_target_ratio=1.0,
    )

    case = report["cases"][0]
    target_profile = case["profile_summary"]["target_profiles"]["100"]
    assert target_profile["row_parent_graph_degree"] == 2
    assert target_profile["row_parent_graph_two_hop_count"] == 2
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    low_degree = selectors["parent_graph_low_degree_p25"]["cases"][0]
    assert low_degree["covered_direct_target_global_dofs"] == [100]


def test_parent_graph_audit_flags_missing_logs():
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

    assert report["finding"] == "direct_pspg_cut_volume_parent_graph_evidence_missing"
    assert report["missing_case_labels"] == ["test02"]


def test_parent_graph_audit_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log_path = tmp_path / "run.log"
    op = audit.DEFAULT_OPERATOR
    log_path.write_text(
        "\n".join(
            [
                _cell(op, 1, 10, [100, 101]),
                _cell(op, 2, 11, [101, 102]),
            ]
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
                        "preferred_candidate_global_dofs": [100, 101, 102],
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
    assert report["cases"][0]["profile_summary"]["profiled_target_count"] == 1
