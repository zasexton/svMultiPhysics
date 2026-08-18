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
        / "audit_direct_pspg_test10_signature_replay_readiness.py"
    )
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location(
        "audit_direct_pspg_test10_signature_replay_readiness",
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


def _entry(row, *, same_parent_pv_count, local_indices=(0, 1)):
    entries = []
    for local_index, parent in zip(local_indices, range(row, row + len(local_indices))):
        entries.append(
            {
                "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
                "block": "pressure_pressure",
                "row_dof": row,
                "parent_cell": parent,
                "row_local_index": local_index,
                "row_abs_sum": 1.0,
                "source_edge_count": 1,
                "neighbor_pair_count": 1,
                "neighbor_connected_pair_count": 0,
                "two_hop_completion_count": 1,
                "pressure_update_sign_used": 0,
                "diagnostic_only": 1,
            }
        )
        entries.append(
            {
                "diagnostic": "cut_volume_direct_pspg_support_coupling_provenance",
                "block": "pressure_velocity",
                "row_dof": row,
                "parent_cell": parent,
                "row_local_index": local_index,
                "row_abs_sum": 1.0
                if local_index < same_parent_pv_count
                else 0.0,
                "source_edge_count": 0,
                "neighbor_pair_count": 0,
                "neighbor_connected_pair_count": 0,
                "two_hop_completion_count": 0,
                "pressure_update_sign_used": 0,
                "diagnostic_only": 1,
            }
        )
    return entries


def _source(*, has_solve_time_selector=False):
    standard = """
    SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_POLICY
    SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_OPERATOR
    SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_SOURCE_COMPONENT
    SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_APPLY_FULL_CELL
    local_schur_completion
    local_edge_balance
    local_schur_edge_balance
    """
    if has_solve_time_selector:
        standard += "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS"
    newton = """
    SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_BALANCE_GLOBAL_DOFS
    SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_EXPLICIT_BALANCE_GLOBAL_DOFS
    shared_row_schur_explicit_edge_balance
    """
    return standard, newton


def test_test10_signature_replay_readiness_exports_rows_and_blocks_on_api_gap():
    audit = _load_audit_module()
    standard, newton = _source(has_solve_time_selector=False)
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": (
                _entry(10, same_parent_pv_count=2)
                + _entry(20, same_parent_pv_count=2)
                + _entry(30, same_parent_pv_count=2)
            ),
            "test10": (
                _entry(100, same_parent_pv_count=0, local_indices=(1, 2))
                + _entry(101, same_parent_pv_count=1, local_indices=(1, 2))
                + _entry(102, same_parent_pv_count=2, local_indices=(1, 2))
            ),
        },
        standard_assembler_text=standard,
        newton_solver_text=newton,
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "test10_signature_replay_candidate_blocked_by_solve_time_selector_api"
    )
    assert report["status"] == (
        "candidate_rows_exported_replay_requires_signature_selector_api"
    )
    assert not report["hook_summary"][
        "fe_topology_signature_or_row_selector_present"
    ]
    assert report["hook_summary"]["post_assembly_explicit_row_path_present"]

    test10 = next(case for case in report["cases"] if case["label"] == "test10")
    assert test10["exact_local_signature_selector"]["finding"] == (
        "selector_selective"
    )
    assert test10["signature_candidate_global_dofs"] == [100, 101]

    test02 = next(case for case in report["cases"] if case["label"] == "test02")
    assert test02["exact_local_signature_selector"]["finding"] == (
        "selector_overbroad"
    )


def test_test10_signature_replay_readiness_detects_available_solve_time_selector():
    audit = _load_audit_module()
    standard, newton = _source(has_solve_time_selector=True)
    report = audit.build_report(
        target_map=_target_map(),
        log_entries_by_case={
            "test02": (
                _entry(10, same_parent_pv_count=2)
                + _entry(20, same_parent_pv_count=2)
                + _entry(30, same_parent_pv_count=2)
            ),
            "test10": _entry(100, same_parent_pv_count=0, local_indices=(1, 2))
            + _entry(101, same_parent_pv_count=1, local_indices=(1, 2)),
        },
        standard_assembler_text=standard,
        newton_solver_text=newton,
        max_target_ratio=1.0,
    )

    assert report["finding"] == (
        "test10_signature_replay_candidate_ready_for_solve_time_replay"
    )
    assert report["status"] == "run_targeted_test10_signature_replay"


def test_standard_assembler_exposes_solve_time_row_filter_source_api():
    audit = _load_audit_module()
    repo = Path(__file__).resolve().parents[1]
    standard = (
        repo / "Code" / "Source" / "solver" / "FE" / "Assembly"
        / "StandardAssembler.cpp"
    ).read_text(encoding="utf-8", errors="replace")
    summary = audit.source_hook_summary(
        standard_assembler_text=standard,
        newton_solver_text="",
    )

    assert summary["fe_topology_signature_or_row_selector_present"]
    assert "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_GLOBAL_DOFS" in standard
    assert "SVMP_FE_CUT_VOLUME_DIRECT_PSPG_TOPOLOGY_ROW_DOFS" in standard
    assert "row_filter_enabled=" in standard
    assert "row_filter_selected_local_row_count=" in standard
