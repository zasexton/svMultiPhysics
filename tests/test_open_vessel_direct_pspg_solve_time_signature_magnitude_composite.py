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
        / "audit_direct_pspg_solve_time_signature_magnitude_composite.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_signature_magnitude_composite",
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


def _entry(row, block, parent_cell, local_index, row_abs_sum):
    return {
        "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
        "block": block,
        "row_dof": row,
        "parent_cell": parent_cell,
        "row_local_index": local_index,
        "row_abs_sum": row_abs_sum,
        "source_edge_count": 1 if block == "pressure_pressure" else 0,
        "neighbor_pair_count": 1 if block == "pressure_pressure" else 0,
        "neighbor_connected_pair_count": 0,
        "two_hop_completion_count": 1 if block == "pressure_pressure" else 0,
        "pressure_update_sign_used": 0,
        "diagnostic_only": 1,
    }


def _row(
    row,
    *,
    pp_parents,
    pv_parents,
    pp_abs,
    pv_abs,
    local_indices=(0, 1),
):
    entries = []
    pp_each = pp_abs / max(len(pp_parents), 1)
    pv_each = pv_abs / max(len(pv_parents), 1)
    for index, parent in zip(local_indices, pp_parents):
        entries.append(_entry(row, "pressure_pressure", parent, index, pp_each))
    for index, parent in zip(local_indices, pv_parents):
        entries.append(_entry(row, "pressure_velocity", parent, index, pv_each))
    return entries


def test_signature_magnitude_composite_keeps_test02_overbroad_test10_selective():
    audit = _load_audit_module()
    shared_signature = {
        "pp_parents": (1, 2),
        "pv_parents": (1, 2),
        "local_indices": (0, 1),
    }
    test02_entries = []
    for row, pv_abs in [
        (10, 0.10),
        (20, 0.20),
        (30, 0.12),
        (31, 0.14),
        (32, 0.16),
    ]:
        test02_entries += _row(
            row,
            **shared_signature,
            pp_abs=2.0,
            pv_abs=pv_abs,
        )

    test10_entries = []
    test10_entries += _row(
        100,
        pp_parents=(10, 11),
        pv_parents=(10, 11),
        local_indices=(0, 1),
        pp_abs=3.0,
        pv_abs=0.30,
    )
    test10_entries += _row(
        101,
        pp_parents=(20, 21),
        pv_parents=(20, 21),
        local_indices=(1, 2),
        pp_abs=4.0,
        pv_abs=0.40,
    )
    test10_entries += _row(
        102,
        pp_parents=(30, 31),
        pv_parents=(30, 31),
        local_indices=(2, 3),
        pp_abs=5.0,
        pv_abs=0.50,
    )

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": test02_entries, "test10": test10_entries},
        max_target_ratio=2.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_signature_magnitude_composite_partial_test10_only"
    )
    assert report["status"] == "test10_composite_candidate_test02_overbroad"

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == (
        "solve_time_signature_magnitude_composite_covers_targets_but_overbroad"
    )
    assert test02["best_covering_composite_selected_count"] == 5
    assert test02["best_covering_composite_selected_to_target_ratio"] == 2.5

    test10 = next(case for case in report["cases"] if case["label"] == "test10")
    assert test10["finding"] == (
        "solve_time_signature_magnitude_composite_selective_candidate"
    )
    assert test10["best_covering_composite_selected_count"] == 2
    assert test10["best_covering_composite_selected_to_target_ratio"] == 1.0


def test_signature_magnitude_composite_can_report_common_candidate():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": _row(
                10,
                pp_parents=(1, 2),
                pv_parents=(1, 2),
                pp_abs=2.0,
                pv_abs=0.10,
            )
            + _row(
                20,
                pp_parents=(3, 4),
                pv_parents=(3, 4),
                local_indices=(2, 3),
                pp_abs=3.0,
                pv_abs=0.20,
            )
            + _row(
                30,
                pp_parents=(5, 6),
                pv_parents=(5, 6),
                local_indices=(4, 5),
                pp_abs=4.0,
                pv_abs=0.30,
            ),
            "test10": _row(
                100,
                pp_parents=(10, 11),
                pv_parents=(10, 11),
                pp_abs=3.0,
                pv_abs=0.30,
            )
            + _row(
                101,
                pp_parents=(20, 21),
                pv_parents=(20, 21),
                local_indices=(2, 3),
                pp_abs=4.0,
                pv_abs=0.40,
            )
            + _row(
                102,
                pp_parents=(30, 31),
                pv_parents=(30, 31),
                local_indices=(4, 5),
                pp_abs=5.0,
                pv_abs=0.50,
            ),
        },
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_signature_magnitude_composite_selector_ready"
    )
    assert report["status"] == "candidate_ready_for_targeted_formulation_replay"


def test_signature_magnitude_composite_reports_missing_targets():
    audit = _load_audit_module()
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": [], "test10": []},
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_signature_magnitude_composite_missing_evidence"
    )
    assert report["status"] == "regenerate_short_replay_logs"
