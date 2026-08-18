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
        / "audit_direct_pspg_solve_time_provenance_replay.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_provenance_replay",
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


def _entry(row, block, row_abs_sum, *, sign_used=0, diagnostic_only=1):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": block,
        "row_dof": row,
        "row_abs_sum": row_abs_sum,
        "source_edge_count": 1 if block == "pressure_pressure" else 0,
        "two_hop_completion_count": 0,
        "neighbor_pair_count": 0,
        "neighbor_connected_pair_count": 0,
        "source_edge_weight_sum": row_abs_sum / 2.0
        if block == "pressure_pressure"
        else 0.0,
        "nonzero_count": 12 if row_abs_sum > 0.0 else 0,
        "volume_fraction": 1.0,
        "full_cell": 1,
        "rule_index": row,
        "pressure_update_sign_used": sign_used,
        "diagnostic_only": diagnostic_only,
    }


def _row_pair(row, pp_abs, pv_abs):
    return [
        _entry(row, "pressure_pressure", pp_abs),
        _entry(row, "pressure_velocity", pv_abs),
    ]


def test_solve_time_provenance_replay_rules_out_simple_pp_pv_gate():
    audit = _load_audit_module()
    test02_entries = []
    test02_entries += _row_pair(10, 10.0, 1.0)
    test02_entries += _row_pair(20, 10.0, 0.1)
    test02_entries += _row_pair(30, 10.0, 0.2)
    test02_entries += _row_pair(31, 10.0, 0.2)
    test02_entries += _row_pair(32, 10.0, 0.2)

    test10_entries = []
    test10_entries += _row_pair(100, 10.0, 0.0)
    test10_entries += _row_pair(101, 10.0, 0.2)
    test10_entries += _row_pair(102, 10.0, 0.3)
    test10_entries += _row_pair(103, 10.0, 0.3)
    test10_entries += _row_pair(104, 10.0, 0.3)

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": test02_entries, "test10": test10_entries},
        max_target_ratio=2.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_replay_rules_out_simple_pp_pv_gate"
    )
    assert report["status"] == "replay_evidence_supports_coupling_split_no_selector"

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["all_rows_pressure_update_sign_unused"]
    assert test02["all_rows_diagnostic_only"]
    assert test02["max_target_ratio_rows"] == [10]
    assert any(
        selector["key"] == "pv_to_pp_ratio_at_or_above_min_target"
        and selector["finding"] == "selector_overbroad"
        for selector in test02["selectors"]
    )
    assert any(
        selector["key"] == "pv_to_pp_ratio_at_or_above_max_target"
        and selector["finding"] == "selector_misses_targets"
        for selector in test02["selectors"]
    )

    test10 = next(case for case in report["cases"] if case["label"] == "test10")
    assert test10["finding"] == (
        "solve_time_provenance_target_family_splits_zero_and_nonzero_coupling"
    )
    assert test10["zero_pressure_velocity_target_global_dofs"] == [100]


def test_solve_time_provenance_replay_rejects_update_sign_dependency():
    audit = _load_audit_module()
    entries = _row_pair(10, 10.0, 1.0)
    entries += _row_pair(20, 10.0, 0.1)
    entries[0]["pressure_update_sign_used"] = 1
    test10_entries = _row_pair(100, 10.0, 0.0) + _row_pair(101, 10.0, 0.2)

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": entries, "test10": test10_entries},
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_replay_invalid_update_dependent"
    )
    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == "solve_time_provenance_uses_pressure_update_sign"
