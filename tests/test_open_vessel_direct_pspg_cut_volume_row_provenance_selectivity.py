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
        / "audit_direct_pspg_cut_volume_row_provenance_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_cut_volume_row_provenance_selectivity",
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
            {"label": "test02", "direct_pspg_target_global_dofs": [100, 101]},
        ]
    }


def test_cut_volume_row_provenance_uses_latest_batch(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    op = audit.DEFAULT_OPERATOR
    log.write_text(
        "\n".join(
            [
                (
                    "assembleOperator: diagnostic=cut_volume_row_provenance "
                    f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
                    "trial='pressure' want_matrix=1 want_vector=0 rule_index=1 "
                    "parent_cell=3 full_cell=0 volume_fraction=0.02 measure=0.2 "
                    "parent_measure=1 quadrature_points=2 source_revision=11 "
                    "cut_topology_revision=13 quadrature_policy_key=17 "
                    "row_dofs=100|101 col_dofs=100|101"
                ),
                (
                    "assembleOperator: diagnostic=cut_volume_row_provenance_summary "
                    f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
                    "trial='pressure' emitted_rules=1 skipped_rules=0 "
                    "total_rule_slots=1 max_rules=0"
                ),
                (
                    "assembleOperator: diagnostic=cut_volume_row_provenance "
                    f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
                    "trial='pressure' want_matrix=1 want_vector=0 rule_index=2 "
                    "parent_cell=4 full_cell=0 volume_fraction=0.05 measure=0.3 "
                    "parent_measure=1 quadrature_points=3 source_revision=12 "
                    "cut_topology_revision=14 quadrature_policy_key=18 "
                    "row_dofs=100|102 col_dofs=100|102"
                ),
                (
                    "assembleOperator: diagnostic=cut_volume_row_provenance "
                    f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
                    "trial='pressure' want_matrix=1 want_vector=0 rule_index=3 "
                    "parent_cell=5 full_cell=1 volume_fraction=1 measure=1 "
                    "parent_measure=1 quadrature_points=4 source_revision=12 "
                    "cut_topology_revision=15 quadrature_policy_key=18 "
                    "row_dofs=101|102 col_dofs=101|102"
                ),
                (
                    "assembleOperator: diagnostic=cut_volume_row_provenance_summary "
                    f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
                    "trial='pressure' emitted_rules=2 skipped_rules=0 "
                    "total_rule_slots=2 max_rules=0"
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
                "preferred_candidate_global_dofs": [100, 101, 102, 103],
            }
        ]
    }

    report = audit.build_report(
        global_emission=global_emission,
        target_map=_target_map(),
        max_target_ratio=1.0,
    )

    case = report["cases"][0]
    assert case["log_evidence"]["latest_batch_entry_count"] == 2
    target_profiles = case["profile_summary"]["target_profiles"]
    assert target_profiles["100"]["min_volume_fraction"] == 0.05
    assert target_profiles["101"]["cut_volume_support_class"] == (
        "full_cell_only_support"
    )
    selectors = {selector["key"]: selector for selector in report["selectors"]}
    no_full_case = selectors["cut_volume_no_full_cell_support"]["cases"][0]
    assert no_full_case["finding"] == "selector_misses_targets"
    assert no_full_case["covered_direct_target_global_dofs"] == [100]
    low_parent_case = selectors["cut_volume_low_parent_cell_support"]["cases"][0]
    assert low_parent_case["finding"] == "selector_overbroad"


def test_cut_volume_row_provenance_flags_missing_logs():
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

    assert report["finding"] == (
        "direct_pspg_cut_volume_row_provenance_evidence_missing"
    )
    assert report["missing_case_labels"] == ["test02"]


def test_cut_volume_row_provenance_cli_style_report_paths(tmp_path):
    audit = _load_audit_module()
    global_path = tmp_path / "global.json"
    target_path = tmp_path / "target.json"
    log_path = tmp_path / "run.log"
    op = audit.DEFAULT_OPERATOR
    log_path.write_text(
        (
            "assembleOperator: diagnostic=cut_volume_row_provenance "
            f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
            "trial='pressure' want_matrix=1 want_vector=0 rule_index=2 "
            "parent_cell=4 full_cell=0 volume_fraction=0.05 measure=0.3 "
            "parent_measure=1 quadrature_points=3 source_revision=12 "
            "cut_topology_revision=14 quadrature_policy_key=18 "
            "row_dofs=100|101 col_dofs=100|101\n"
            "assembleOperator: diagnostic=cut_volume_row_provenance_summary "
            f"status=ok op='{op}' marker=7 side=Negative test='pressure' "
            "trial='pressure' emitted_rules=1 skipped_rules=0 "
            "total_rule_slots=1 max_rules=0\n"
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
    )

    assert report["global_emission_path"] == str(global_path)
    assert report["target_map_path"] == str(target_path)
    assert report["finding"] == (
        "direct_pspg_cut_volume_row_provenance_selector_selective"
    )
