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
        / "audit_direct_pspg_solve_time_aggregate_feature_selectivity.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_aggregate_feature_selectivity", script
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
    block,
    rule_index,
    *,
    full_cell=1,
    volume_fraction=1.0,
    source_edge_count=1,
    two_hop_completion_count=1,
    nonzero_count=1,
):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": block,
        "row_dof": row,
        "rule_index": rule_index,
        "parent_cell": rule_index,
        "row_local_index": 0,
        "row_abs_sum": 1.0,
        "source_edge_count": source_edge_count if block == "pressure_pressure" else 0,
        "source_edge_weight_sum": 0.5 if block == "pressure_pressure" else 0.0,
        "neighbor_pair_count": two_hop_completion_count
        if block == "pressure_pressure"
        else 0,
        "neighbor_connected_pair_count": 0,
        "two_hop_completion_count": two_hop_completion_count
        if block == "pressure_pressure"
        else 0,
        "nonzero_count": nonzero_count if block == "pressure_velocity" else 2,
        "full_cell": full_cell,
        "volume_fraction": volume_fraction,
        "pressure_update_sign_used": 0,
        "diagnostic_only": 1,
    }


def _row(
    row,
    *,
    rules,
    full_cell=1,
    volume_fraction=1.0,
    source_edge_count=1,
    two_hop_completion_count=1,
    pv_nonzero_count=1,
):
    entries = []
    for rule_index in rules:
        entries.append(
            _entry(
                row,
                "pressure_pressure",
                rule_index,
                full_cell=full_cell,
                volume_fraction=volume_fraction,
                source_edge_count=source_edge_count,
                two_hop_completion_count=two_hop_completion_count,
            )
        )
        entries.append(
            _entry(
                row,
                "pressure_velocity",
                rule_index,
                full_cell=full_cell,
                volume_fraction=volume_fraction,
                nonzero_count=pv_nonzero_count,
            )
        )
    return entries


def test_aggregate_feature_selectivity_rules_out_broad_count_features():
    audit = _load_audit_module()
    test02_rows = (
        _row(10, rules=[1, 2])
        + _row(20, rules=[1, 2])
        + _row(30, rules=[1, 2])
    )
    test10_rows = (
        _row(100, rules=[3])
        + _row(101, rules=[3])
        + _row(102, rules=[3])
    )

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": test02_rows,
            "test10": test10_rows,
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_aggregate_feature_selectivity_rules_out_"
        "counts_and_volume_gate"
    )
    assert report["status"] == "aggregate_counts_and_volume_features_overbroad"
    for case in report["cases"]:
        assert case["finding"] == (
            "solve_time_aggregate_feature_selectors_overbroad_or_miss_targets"
        )
        assert case["best_covering_exact_value_selector"][
            "selected_to_target_ratio"
        ] > 1.0
    assert "aggregate provenance counts" in report["next_requirement"]


def test_aggregate_feature_selectivity_reports_candidate_for_transfer_check():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": (
                _row(10, rules=[1], source_edge_count=2)
                + _row(20, rules=[1], source_edge_count=2)
                + _row(30, rules=[2], source_edge_count=8)
            ),
            "test10": (
                _row(100, rules=[3], source_edge_count=4)
                + _row(101, rules=[3], source_edge_count=4)
                + _row(102, rules=[4], source_edge_count=9)
            ),
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_aggregate_feature_candidate_requires_replay"
    )
    assert report["status"] == "aggregate_feature_candidate_needs_transfer_check"
    assert any(
        case["finding"] == "solve_time_aggregate_range_selector_candidate"
        for case in report["cases"]
    )


def test_aggregate_feature_selectivity_reports_missing_targets():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": _row(10, rules=[1]), "test10": []},
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_aggregate_feature_selectivity_missing_evidence"
    )
    assert report["status"] == "regenerate_solve_time_provenance_logs"
