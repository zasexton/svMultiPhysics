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
        / "audit_direct_pspg_cut_volume_gradient_balance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_gradient_balance_selectivity",
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


def _entry(op, rule_index, row_dof, matrix_to_gram, full_cell=1, mismatch=False):
    row_abs = matrix_to_gram * 10.0
    sampled_grams = "-5|-2|-2|-1" if mismatch else "5|-2|-2|-1"
    return (
        "StandardAssembler: "
        "diagnostic=cut_volume_local_matrix_gradient_balance "
        f"status=ok op='{op}' marker=7 side=Negative test='Pressure' "
        f"trial='Pressure' rule_index={rule_index} parent_cell=3 "
        f"full_cell={full_cell} volume_fraction=1 measure=1 parent_measure=1 "
        "rule_quadrature_points=4 active_quadrature_points=4 "
        "source_revision=1 cut_topology_revision=2 quadrature_policy_key=3 "
        f"row_local_index=0 row_dof={row_dof} col_count=4 "
        "nonzero_col_count=4 sampled_col_count=4 sample_truncated=0 "
        f"row_abs_sum={row_abs} row_signed_sum=0 positive_sum=5 "
        "negative_abs_sum=5 diag_abs=5 offdiag_abs_sum=5 "
        "gradient_balance_available=1 gradient_qpoint_count=4 "
        "row_grad_x=1 row_grad_y=0 row_grad_z=0 row_grad_norm=1 "
        "row_grad_abs_integral=1 row_grad_energy=2 row_grad_max_norm=1 "
        "row_grad_directional_ratio=1 row_grad_axis_dominance=1 "
        "row_grad_dominant_axis=0 gram_row_abs_sum=10 "
        "gram_row_abs_fraction=0.25 gram_row_signed_sum=0 "
        "gram_positive_sum=5 gram_negative_abs_sum=5 gram_nonzero_count=4 "
        "gram_positive_count=1 gram_negative_count=3 gram_diag_value=5 "
        "gram_diag_abs=5 gram_diag_abs_fraction=0.5 "
        "gram_offdiag_abs_sum=5 gram_max_abs_entry=5 "
        f"gram_max_abs_col_dof={row_dof} gram_max_abs_col_local_index=0 "
        f"matrix_to_gram_abs_ratio={matrix_to_gram} "
        f"sampled_col_local_indices=0|1|2|3 sampled_col_dofs={row_dof}|101|102|103 "
        "sampled_col_values=5|-2|-2|-1 "
        f"sampled_col_gradient_gram_values={sampled_grams} "
        "sampled_col_gradient_cosines=1|-0.5|-0.5|-0.25"
    )


def test_gradient_balance_audit_uses_latest_rule_index_reset_batch(tmp_path):
    audit = _load_audit_module()
    op = audit.DEFAULT_OPERATOR
    log = tmp_path / "run_direct_pspg_cut_volume_gradient_balance.log"
    log.write_text(
        "\n".join(
            [
                _entry(op, 10, 100, 0.1),
                _entry(op, 11, 101, 0.2),
                _entry(op, 1, 100, 9.0, mismatch=True),
                _entry(op, 2, 101, 0.2),
            ]
        ),
        encoding="utf-8",
    )
    report = audit.build_report(
        global_emission={
            "cases": [
                {
                    "label": "test02",
                    "path": str(log.with_name("run_direct_pspg_cut_volume_row_provenance.log")),
                    "preferred_candidate_global_dofs": [100, 101],
                }
            ]
        },
        target_map=_target_map(),
        max_target_ratio=1.0,
    )

    case = report["cases"][0]
    assert case["log_evidence"]["batch_count"] == 2
    assert case["profile_summary"]["target_profiles"]["100"][
        "matrix_to_gram_abs_ratio"
    ] == 9.0
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    sign_mismatch = selectors["gradient_balance_sampled_sign_mismatch"]["cases"][0]
    assert sign_mismatch["finding"] == "selector_selective"
    assert sign_mismatch["covered_direct_target_global_dofs"] == [100]


def test_gradient_balance_audit_flags_missing_logs():
    audit = _load_audit_module()
    report = audit.build_report(
        global_emission={
            "cases": [
                {
                    "label": "test02",
                    "path": "missing_row_provenance.log",
                    "preferred_candidate_global_dofs": [100, 101],
                }
            ]
        },
        target_map=_target_map(),
    )

    assert report["finding"] == "direct_pspg_cut_volume_gradient_balance_evidence_missing"
    assert report["missing_case_labels"] == ["test02"]


def test_gradient_balance_audit_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log_path = tmp_path / "custom_gradient.log"
    op = audit.DEFAULT_OPERATOR
    log_path.write_text(
        "\n".join(
            [
                _entry(op, 1, 100, 9.0, mismatch=True),
                _entry(op, 2, 101, 0.2),
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
                        "path": "unused.log",
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
        explicit_logs=[f"test02={log_path}"],
        max_target_ratio=1.0,
    )

    assert report["global_emission_path"] == str(global_path)
    assert report["target_map_path"] == str(target_path)
    assert report["selective_selector_keys"]
