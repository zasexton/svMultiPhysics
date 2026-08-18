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
        / "audit_direct_pspg_solve_time_magnitude_selectivity.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_magnitude_selectivity",
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
            {"label": "test02", "direct_pspg_target_global_dofs": [10, 20]},
            {"label": "test10", "direct_pspg_target_global_dofs": [100, 101]},
        ]
    }


def _entry(row, block, parent_cell, row_abs_sum):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": block,
        "row_dof": row,
        "parent_cell": parent_cell,
        "row_local_index": 0,
        "row_abs_sum": row_abs_sum,
        "source_edge_count": 1 if block == "pressure_pressure" else 0,
        "neighbor_pair_count": 1 if block == "pressure_pressure" else 0,
        "neighbor_connected_pair_count": 0,
        "two_hop_completion_count": 1 if block == "pressure_pressure" else 0,
        "pressure_update_sign_used": 0,
        "diagnostic_only": 1,
    }


def _row(row, *, pp_abs, pv_abs):
    return [
        _entry(row, "pressure_pressure", row, pp_abs),
        _entry(row, "pressure_velocity", row, pv_abs),
    ]


def test_solve_time_magnitude_selectivity_rejects_exact_value_oracles():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": (
                _row(10, pp_abs=1.0, pv_abs=0.10)
                + _row(20, pp_abs=2.0, pv_abs=0.30)
                + _row(30, pp_abs=1.5, pv_abs=0.20)
            ),
            "test10": (
                _row(100, pp_abs=3.0, pv_abs=0.0)
                + _row(101, pp_abs=4.0, pv_abs=0.4)
                + _row(102, pp_abs=3.5, pv_abs=0.2)
            ),
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_magnitude_selectors_not_formulation_ready"
    )
    assert report["status"] == (
        "range_thresholds_overbroad_exact_value_oracles_only"
    )

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == (
        "exact_magnitude_value_oracles_only_range_selectors_broad"
    )
    assert test02["range_selector_findings"][
        "pressure_velocity_abs_sum_target_range"
    ] == "selector_overbroad"
    assert (
        "pressure_velocity_abs_sum_exact_target_value_set"
        in test02["exact_value_oracle_selector_keys"]
    )
    assert test02["exact_value_oracle_selected_to_target_ratios"][
        "pressure_velocity_abs_sum_exact_target_value_set"
    ] == 1.0


def test_solve_time_magnitude_selectivity_reports_range_candidate():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": _row(10, pp_abs=1.0, pv_abs=0.10)
            + _row(20, pp_abs=2.0, pv_abs=0.20)
            + _row(30, pp_abs=3.0, pv_abs=0.30),
            "test10": _row(100, pp_abs=4.0, pv_abs=0.0)
            + _row(101, pp_abs=5.0, pv_abs=0.5)
            + _row(102, pp_abs=6.0, pv_abs=0.6),
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_magnitude_candidate_found"
    )
    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == "solve_time_magnitude_range_selector_candidate"
    assert test02["range_selector_findings"][
        "pressure_pressure_abs_sum_target_range"
    ] == "selector_selective"
