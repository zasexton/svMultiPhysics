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
        / "audit_direct_pspg_solve_time_support_coupling_signature.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_solve_time_support_coupling_signature",
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
    parent_cell,
    local_index,
    row_abs_sum,
    *,
    sign_used=0,
    diagnostic_only=1,
):
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
        "pressure_update_sign_used": sign_used,
        "diagnostic_only": diagnostic_only,
    }


def _support_row(
    row,
    *,
    pp_parents,
    pv_nonzero_parents=(),
    pv_zero_parents=(),
    local_indices=(0, 1),
    sign_used=0,
):
    entries = []
    for index, parent in zip(local_indices, pp_parents):
        entries.append(
            _entry(
                row,
                "pressure_pressure",
                parent,
                index,
                1.0,
                sign_used=sign_used,
            )
        )
    for index, parent in zip(local_indices, pv_nonzero_parents):
        entries.append(_entry(row, "pressure_velocity", parent, index, 1.0))
    for index, parent in zip(local_indices, pv_zero_parents):
        entries.append(_entry(row, "pressure_velocity", parent, index, 0.0))
    return entries


def test_support_coupling_signature_finds_test10_candidate_but_test02_overbroad():
    audit = _load_audit_module()
    test02_entries = []
    for row in [10, 20, 30, 31, 32]:
        test02_entries += _support_row(
            row,
            pp_parents=(1, 2),
            pv_nonzero_parents=(1, 2),
        )

    test10_entries = []
    test10_entries += _support_row(
        100,
        pp_parents=(10, 11),
        pv_zero_parents=(10, 11),
        local_indices=(0, 1),
    )
    test10_entries += _support_row(
        101,
        pp_parents=(20, 21),
        pv_nonzero_parents=(20,),
        pv_zero_parents=(21,),
        local_indices=(1, 2),
    )
    test10_entries += _support_row(
        102,
        pp_parents=(30, 31),
        pv_nonzero_parents=(30, 31),
        local_indices=(2, 3),
    )

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": test02_entries, "test10": test10_entries},
        max_target_ratio=2.0,
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_signature_partial_test10_only"
    )
    assert report["status"] == "test10_signature_candidate_test02_overbroad"

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == (
        "solve_time_support_coupling_signature_covers_targets_but_overbroad"
    )
    assert test02["exact_local_signature_selected_count"] == 5
    assert test02["target_same_parent_pressure_velocity_support_class_counts"] == {
        "none": 0,
        "partial": 0,
        "full": 2,
    }

    test10 = next(case for case in report["cases"] if case["label"] == "test10")
    assert test10["finding"] == (
        "solve_time_support_coupling_signature_selective_candidate"
    )
    assert test10["exact_local_signature_selected_count"] == 2
    assert test10["target_same_parent_pressure_velocity_support_class_counts"] == {
        "none": 1,
        "partial": 1,
        "full": 0,
    }


def test_support_coupling_signature_rejects_update_sign_dependency():
    audit = _load_audit_module()
    test02_entries = _support_row(
        10,
        pp_parents=(1, 2),
        pv_nonzero_parents=(1, 2),
        sign_used=1,
    ) + _support_row(20, pp_parents=(3, 4), pv_nonzero_parents=(3, 4))
    test10_entries = _support_row(
        100,
        pp_parents=(10, 11),
        pv_zero_parents=(10, 11),
    ) + _support_row(
        101,
        pp_parents=(20, 21),
        pv_nonzero_parents=(20,),
        pv_zero_parents=(21,),
    )

    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={"test02": test02_entries, "test10": test10_entries},
    )

    assert report["finding"] == (
        "solve_time_direct_pspg_support_coupling_signature_invalid_update_dependent"
    )
    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["finding"] == (
        "solve_time_support_coupling_signature_uses_pressure_update_sign"
    )
