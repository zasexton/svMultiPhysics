import importlib.util
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
        / "audit_direct_pspg_solve_time_parent_rule_component_selectivity.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_parent_rule_component_selectivity", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _target_map():
    return {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [10, 20]},
            {"label": "test10", "direct_pspg_target_global_dofs": [100, 101]},
        ]
    }


def _entry(row, parent_cell, rule_index, local_index=0):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": "pressure_pressure",
        "row_dof": row,
        "parent_cell": parent_cell,
        "rule_index": rule_index,
        "row_local_index": local_index,
    }


def test_parent_rule_component_selectivity_rules_out_broad_closure():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": [
                _entry(10, 1, 1),
                _entry(20, 1, 1),
                _entry(30, 1, 1),
            ],
            "test10": [
                _entry(100, 2, 2),
                _entry(101, 2, 2),
                _entry(102, 2, 2),
            ],
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_parent_rule_components_rule_out_"
        "connected_cosupport_closure"
    )
    assert report["status"] == "parent_rule_component_closure_overbroad"
    for case in report["cases"]:
        assert case["finding"] == (
            "solve_time_parent_rule_components_overbroad_or_miss_targets"
        )
        assert case["best_covering_component_selector"][
            "selected_to_target_ratio"
        ] > 1.0
    assert "connected parent/rule co-support" in report["next_requirement"]


def test_parent_rule_component_selectivity_reports_candidate_for_replay():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": [
                _entry(10, 1, 1),
                _entry(20, 1, 1),
                _entry(30, 2, 2),
            ],
            "test10": [
                _entry(100, 3, 3),
                _entry(101, 3, 3),
                _entry(102, 4, 4),
            ],
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_parent_rule_component_candidate_requires_replay"
    )
    assert report["status"] == "parent_rule_component_candidate_needs_transfer_check"
    assert any(
        case["finding"] == (
            "solve_time_parent_rule_component_candidate_requires_replay"
        )
        for case in report["cases"]
    )


def test_parent_rule_component_selectivity_reports_missing_targets():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": [_entry(10, 1, 1)], "test10": []},
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_parent_rule_component_selectivity_missing_evidence"
    )
    assert report["status"] == "regenerate_solve_time_provenance_logs"
