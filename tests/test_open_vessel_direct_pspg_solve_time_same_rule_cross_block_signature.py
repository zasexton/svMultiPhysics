import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tests/cases/fluid/open_vessel_free_surface/"
    "audit_direct_pspg_solve_time_same_rule_cross_block_signature.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_direct_pspg_solve_time_same_rule_cross_block_signature", SCRIPT
)
audit = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit)


def _target_map() -> dict:
    return {
        "cases": [
            {"label": "test02", "direct_pspg_target_global_dofs": [10, 20]},
            {"label": "test10", "direct_pspg_target_global_dofs": [100, 101]},
        ]
    }


def _entry(
    *,
    row: int,
    block: str,
    parent_cell: int,
    rule_index: int,
    row_abs_sum: float,
    local_pattern: str = "target",
) -> dict:
    if local_pattern == "target":
        local_indices = "0|1"
        signs = "1|-1"
        nonzero_count = 2
        source_edge_count = 1
    else:
        local_indices = "0|2|3"
        signs = "1|-1|-1"
        nonzero_count = 3
        source_edge_count = 2
    return {
        "row_dof": row,
        "block": block,
        "parent_cell": parent_cell,
        "rule_index": rule_index,
        "row_local_index": 0,
        "full_cell": 1,
        "sampled_col_local_indices": local_indices,
        "sampled_col_signs": signs,
        "nonzero_col_count": nonzero_count,
        "source_edge_count": source_edge_count,
        "two_hop_completion_count": 0,
        "row_abs_sum": row_abs_sum,
        "row_signed_sum": 0.0,
        "pressure_update_sign_used": 0,
        "diagnostic_only": 1,
    }


def _row_entries(
    row: int,
    *,
    parent_cell: int,
    rule_index: int,
    pp_abs: float,
    pv_abs: float,
    local_pattern: str = "target",
) -> list[dict]:
    return [
        _entry(
            row=row,
            block="pressure_pressure",
            parent_cell=parent_cell,
            rule_index=rule_index,
            row_abs_sum=pp_abs,
            local_pattern=local_pattern,
        ),
        _entry(
            row=row,
            block="pressure_velocity",
            parent_cell=parent_cell,
            rule_index=rule_index,
            row_abs_sum=pv_abs,
            local_pattern=local_pattern,
        ),
    ]


def _candidate_entries(base_row: int) -> list[dict]:
    entries: list[dict] = []
    entries.extend(
        _row_entries(base_row, parent_cell=1, rule_index=1, pp_abs=10.0, pv_abs=1.0)
    )
    entries.extend(
        _row_entries(
            base_row + 10,
            parent_cell=1,
            rule_index=1,
            pp_abs=12.0,
            pv_abs=2.0,
        )
    )
    entries.extend(
        _row_entries(
            base_row + 1,
            parent_cell=1,
            rule_index=1,
            pp_abs=11.0,
            pv_abs=1.5,
        )
    )
    entries.extend(
        _row_entries(
            base_row + 2,
            parent_cell=1,
            rule_index=1,
            pp_abs=50.0,
            pv_abs=9.0,
        )
    )
    entries.extend(
        _row_entries(
            base_row + 3,
            parent_cell=2,
            rule_index=2,
            pp_abs=11.0,
            pv_abs=1.5,
            local_pattern="other",
        )
    )
    return entries


def test_same_rule_cross_block_signature_magnitude_candidate_found() -> None:
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": _candidate_entries(10),
            "test10": _candidate_entries(100),
        },
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_same_rule_cross_block_signature_magnitude_"
        "candidate_found"
    )
    assert report["status"] == "same_rule_cross_block_candidate_requires_replay"
    assert [case["finding"] for case in report["cases"]] == [
        "same_rule_cross_block_signature_magnitude_candidate",
        "same_rule_cross_block_signature_magnitude_candidate",
    ]
    for case in report["cases"]:
        best = case["best_covering_composite_selector"]
        assert best["finding"] == "selector_selective"
        assert best["covered_target_count"] == 2
        assert best["selected_to_target_ratio"] <= report["max_target_ratio"]


def test_same_rule_cross_block_signature_overbroad_without_candidate() -> None:
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": _candidate_entries(10),
            "test10": _candidate_entries(100),
        },
        max_target_ratio=0.5,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_same_rule_cross_block_signature_selectors_not_ready"
    )
    assert report["status"] == (
        "same_rule_cross_block_signature_overbroad_or_misses_targets"
    )
    assert [case["finding"] for case in report["cases"]] == [
        "same_rule_cross_block_signature_magnitude_overbroad_candidate",
        "same_rule_cross_block_signature_magnitude_overbroad_candidate",
    ]


def test_same_rule_cross_block_signature_missing_target_rows() -> None:
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": _candidate_entries(10),
            "test10": [],
        },
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_same_rule_cross_block_signature_missing_evidence"
    )
    assert report["status"] == "regenerate_sampled_column_replay_logs"
