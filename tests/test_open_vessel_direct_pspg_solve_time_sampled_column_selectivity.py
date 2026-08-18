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
        / "audit_direct_pspg_solve_time_sampled_column_selectivity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_sampled_column_selectivity",
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


def _entry(
    row,
    block,
    *,
    row_local_index=0,
    local_indices="0|1|2",
    signs="1|-1|-1",
    nonzero_count=3,
    source_edge_count=2,
    two_hop_count=1,
    row_abs_sum=10.0,
    row_signed_sum=0.0,
    positive_count=1,
    negative_count=2,
    sign_used=0,
):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": block,
        "row_dof": row,
        "row_local_index": row_local_index,
        "sampled_col_local_indices": local_indices,
        "sampled_col_dofs": "10|11|12",
        "sampled_col_values": "1|-0.5|-0.5",
        "sampled_col_abs_values": "1|0.5|0.5",
        "sampled_col_signs": signs,
        "sampled_col_count": nonzero_count,
        "sample_truncated": 0,
        "sample_sorted_by": "abs_desc",
        "diag_in_sample": 1,
        "nonzero_col_count": nonzero_count,
        "source_edge_count": source_edge_count,
        "two_hop_completion_count": two_hop_count,
        "full_cell": 1,
        "row_abs_sum": row_abs_sum,
        "row_signed_sum": row_signed_sum,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "pressure_update_sign_used": sign_used,
        "diagnostic_only": 1,
    }


def _row_pair(row, *, pp_signature="common", pv_signature="common", sign_used=0):
    if pp_signature == "target_exact":
        pp = _entry(
            row,
            "pressure_pressure",
            row_local_index=3,
            local_indices="3|2",
            signs="1|-1",
            nonzero_count=2,
            source_edge_count=1,
            two_hop_count=0,
            sign_used=sign_used,
        )
    else:
        pp = _entry(row, "pressure_pressure", sign_used=sign_used)
    if pv_signature == "zero":
        pv = _entry(
            row,
            "pressure_velocity",
            local_indices="0|1",
            signs="1|-1",
            nonzero_count=0,
            source_edge_count=0,
            two_hop_count=0,
            row_abs_sum=0.0,
            positive_count=0,
            negative_count=0,
            sign_used=sign_used,
        )
    else:
        pv = _entry(
            row,
            "pressure_velocity",
            local_indices="0|1|2|3",
            signs="1|-1|1|-1",
            nonzero_count=4,
            source_edge_count=0,
            two_hop_count=0,
            positive_count=2,
            negative_count=2,
            sign_used=sign_used,
        )
    return [pp, pv]


def test_sampled_column_selectivity_rules_out_threshold_like_stencil_gate():
    audit = _load_audit_module()
    test02_entries = []
    test02_entries += _row_pair(10, pp_signature="target_exact")
    test02_entries += _row_pair(20)
    test02_entries += _row_pair(30)
    test02_entries += _row_pair(31)
    test02_entries += _row_pair(32)
    test10_entries = []
    test10_entries += _row_pair(100, pv_signature="zero")
    test10_entries += _row_pair(101)
    test10_entries += _row_pair(102)
    test10_entries += _row_pair(103)
    test10_entries += _row_pair(104)

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": test02_entries, "test10": test10_entries},
        max_target_ratio=2.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_sampled_column_selectors_not_formulation_ready"
    )
    assert report["status"] == "sampled_column_stencil_gate_ruled_out"

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["target_rows_present_count"] == 2
    assert test02["all_rows_sample_payload_complete"]
    assert not test02["any_sample_truncated"]
    assert any(
        selector["key"] == "all_sampled_columns_complete"
        and selector["finding"] == "selector_overbroad"
        for selector in test02["selectors"]
    )
    assert any(
        selector["key"]
        == "pressure_pressure_exact_local_signature_matches_target_union"
        and selector["selector_family"] == "target_signature"
        for selector in test02["selectors"]
    )


def test_sampled_column_selectivity_rejects_update_dependent_payload():
    audit = _load_audit_module()
    test02_entries = _row_pair(10, sign_used=1) + _row_pair(20)
    test10_entries = _row_pair(100) + _row_pair(101)

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": test02_entries, "test10": test10_entries},
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_sampled_column_selectivity_update_dependent"
    )
    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == "sampled_column_replay_uses_pressure_update_sign"
