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
        / "audit_direct_pspg_solve_time_support_measure_selectivity.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_support_measure_selectivity", script
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


def _entry(
    row,
    *,
    active_quadrature_points=4,
    rule_quadrature_points=16,
    measure=0.25,
    parent_measure=1.0,
):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": "pressure_pressure",
        "row_dof": row,
        "row_abs_sum": 1.0,
        "active_quadrature_points": active_quadrature_points,
        "rule_quadrature_points": rule_quadrature_points,
        "measure": measure,
        "parent_measure": parent_measure,
    }


def test_support_measure_selectivity_rules_out_broad_measure_classes():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": [_entry(10), _entry(20), _entry(30)],
            "test10": [_entry(100), _entry(101), _entry(102)],
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_measure_selectivity_rules_out_"
        "qpoint_measure_gate"
    )
    assert report["status"] == "active_qpoint_and_measure_features_overbroad"
    for case in report["cases"]:
        assert case["finding"] == (
            "solve_time_support_measure_selectors_overbroad_or_miss_targets"
        )
        assert case["best_covering_exact_value_selector"][
            "selected_to_target_ratio"
        ] > 1.0
    assert "active quadrature count" in report["next_requirement"]


def test_support_measure_selectivity_reports_candidate_for_transfer_check():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": [
                _entry(10, active_quadrature_points=4),
                _entry(20, active_quadrature_points=4),
                _entry(30, active_quadrature_points=8),
            ],
            "test10": [
                _entry(100, active_quadrature_points=4),
                _entry(101, active_quadrature_points=4),
                _entry(102, active_quadrature_points=8),
            ],
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_measure_candidate_requires_replay"
    )
    assert report["status"] == "support_measure_candidate_needs_transfer_check"
    assert any(
        case["finding"] == "solve_time_support_measure_range_selector_candidate"
        for case in report["cases"]
    )


def test_support_measure_selectivity_reports_missing_targets():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": [_entry(10)], "test10": []},
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_measure_selectivity_missing_evidence"
    )
    assert report["status"] == "regenerate_solve_time_provenance_logs"
