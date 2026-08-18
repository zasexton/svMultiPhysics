import importlib.util
import sys
from pathlib import Path


def _load_patch_module():
    repo = Path(__file__).resolve().parents[1]
    script_dir = (
        repo
        / "tests"
        / "cases"
        / "fluid"
        / "open_vessel_free_surface"
    )
    script = script_dir / "audit_linear_pressure_cut_volume_patch.py"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "audit_linear_pressure_cut_volume_patch", script
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_linear_pressure_patch_detects_trace_only_support_hazard():
    patch = _load_patch_module()

    report = patch.audit_patch()
    cases = {case["name"]: case for case in report["cases"]}

    assert report["passed"] is True
    assert report["hazard_detected"] is True
    assert report["pspg_hydrostatic_hazard_detected"] is True
    assert cases["retained_cut_volume_support"]["preserves_linear_pressure_state"]
    assert cases["retained_cut_volume_support"]["pspg_hydrostatic_balance"][
        "preserves_hydrostatic_balance"
    ]
    assert cases["retained_cut_volume_support"]["pspg_hydrostatic_balance"][
        "direct_pressure_gradient_has_boundary_action"
    ]
    assert cases["retained_cut_volume_support"]["pspg_hydrostatic_balance"][
        "preserves_constant_pressure_null"
    ]
    retained_amplification = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["boundary_solve_amplification"]
    assert retained_amplification["available"]
    assert retained_amplification["constant_null_preserved_during_solve_proxy"]
    assert retained_amplification["max_response_is_weakest_diag_row"]
    assert (
        retained_amplification[
            "max_to_strongest_support_target_response_ratio"
        ]
        > 5.0
    )
    assert retained_amplification["max_target_response_row"]["target_row"] == 3
    assert (
        retained_amplification["max_target_response_row"]["incident_cell_count"]
        == 1
    )
    assert retained_amplification["uniform_scale_probe"]["scale"] == 10.0
    assert retained_amplification["uniform_scale_probe"][
        "preserves_response_ratio"
    ]
    assert (
        retained_amplification["uniform_scale_probe"][
            "max_to_strongest_support_target_response_ratio"
        ]
        == retained_amplification[
            "max_to_strongest_support_target_response_ratio"
        ]
    )
    retained_completion = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["boundary_pair_completion"]
    retained_shared_completion = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["shared_support_completion"]
    retained_active_completion = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["weak_boundary_active_completion"]
    retained_shared_row_schur_completion = cases[
        "retained_cut_volume_support"
    ]["pspg_hydrostatic_balance"]["shared_row_schur_completion"]
    retained_schur_edge_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["shared_row_schur_existing_edge_balance"]
    retained_weak_boundary_edge_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["weak_boundary_existing_edge_support_balance"]
    retained_schur_shared_support_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["shared_row_schur_shared_support_edge_balance"]
    retained_schur_weak_boundary_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["shared_row_schur_weak_boundary_edge_balance"]
    retained_direct_gap_patch_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["direct_support_gap_or_same_sign_patch_completion"]
    retained_existing_edge_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["existing_edge_support_balance"]
    retained_incident_support_balance = cases["retained_cut_volume_support"][
        "pspg_hydrostatic_balance"
    ]["incident_support_count_balance"]
    assert retained_completion["available"]
    assert retained_completion["kind"] == (
        "diagnostic_one_cell_boundary_pair_completion"
    )
    assert retained_completion["edge_count"] == 1
    assert retained_completion["one_cell_boundary_rows"] == [3, 4]
    assert retained_completion["preserves_hydrostatic_balance"]
    assert retained_completion["preserves_constant_pressure_null"]
    assert retained_completion["reduces_response_ratio"]
    assert retained_completion["reduces_max_target_response"]
    assert (
        retained_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        < retained_amplification[
            "max_to_strongest_support_target_response_ratio"
        ]
    )
    assert (
        retained_completion["completed_boundary_solve_amplification"][
            "max_target_response_row"
        ]["target_row"]
        == 3
    )
    assert retained_shared_completion["available"]
    assert retained_shared_completion["kind"] == (
        "diagnostic_one_cell_to_shared_support_completion"
    )
    assert retained_shared_completion["edge_count"] == 6
    assert retained_shared_completion["one_cell_boundary_rows"] == [3, 4]
    assert retained_shared_completion["shared_support_rows"] == [0, 1, 2]
    assert retained_shared_completion["preserves_hydrostatic_balance"]
    assert retained_shared_completion["preserves_constant_pressure_null"]
    assert retained_shared_completion["reduces_response_ratio"]
    assert retained_shared_completion["reduces_max_target_response"]
    assert (
        retained_shared_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        > retained_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_active_completion["available"]
    assert retained_active_completion["kind"] == (
        "diagnostic_one_cell_to_active_support_completion"
    )
    assert retained_active_completion["edge_count"] == 7
    assert retained_active_completion["contribution_count"] == 8
    assert retained_active_completion["one_cell_boundary_rows"] == [3, 4]
    assert retained_active_completion["active_rows"] == [0, 1, 2, 3, 4]
    assert retained_active_completion["preserves_hydrostatic_balance"]
    assert retained_active_completion["preserves_constant_pressure_null"]
    assert retained_active_completion["reduces_response_ratio"]
    assert retained_active_completion["reduces_max_target_response"]
    assert (
        retained_active_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        < retained_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_shared_row_schur_completion["available"]
    assert retained_shared_row_schur_completion["kind"] == (
        "diagnostic_shared_row_schur_support_completion"
    )
    assert retained_shared_row_schur_completion["edge_count"] == 6
    assert retained_shared_row_schur_completion["contribution_count"] == 6
    assert retained_shared_row_schur_completion["shared_support_rows"] == [
        0,
        1,
        2,
    ]
    assert retained_shared_row_schur_completion["preserves_hydrostatic_balance"]
    assert retained_shared_row_schur_completion[
        "preserves_constant_pressure_null"
    ]
    assert retained_shared_row_schur_completion["reduces_response_ratio"]
    assert retained_shared_row_schur_completion["reduces_max_target_response"]
    assert abs(
        retained_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        - 6.153374233128836
    ) <= 1.0e-12
    assert (
        retained_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        < retained_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert (
        retained_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        > retained_active_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_schur_edge_balance["available"]
    assert retained_schur_edge_balance["kind"] == (
        "diagnostic_shared_row_schur_existing_edge_balance"
    )
    assert retained_schur_edge_balance["edge_count"] == 10
    assert retained_schur_edge_balance["schur_edge_count"] == 6
    assert retained_schur_edge_balance["schur_contribution_count"] == 6
    assert retained_schur_edge_balance["preserves_hydrostatic_balance"]
    assert retained_schur_edge_balance["preserves_constant_pressure_null"]
    assert retained_schur_edge_balance["reduces_response_ratio"]
    assert retained_schur_edge_balance["reduces_max_target_response"]
    assert abs(
        retained_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 2.372933991730586
    ) <= 1.0e-12
    assert (
        retained_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        < retained_active_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_schur_edge_balance["max_edge_scale"] < 8.0
    assert retained_weak_boundary_edge_balance["available"]
    assert retained_weak_boundary_edge_balance["kind"] == (
        "diagnostic_weak_boundary_existing_pressure_edge_support_balance"
    )
    assert retained_weak_boundary_edge_balance["edge_count"] == 4
    assert retained_weak_boundary_edge_balance["eligible_balance_rows"] == [3, 4]
    assert (
        retained_weak_boundary_edge_balance["balance_row_selection"]
        == "one_cell_boundary_rows"
    )
    assert retained_weak_boundary_edge_balance["preserves_hydrostatic_balance"]
    assert retained_weak_boundary_edge_balance[
        "preserves_constant_pressure_null"
    ]
    assert retained_weak_boundary_edge_balance["reduces_response_ratio"]
    assert retained_weak_boundary_edge_balance["reduces_max_target_response"]
    assert abs(
        retained_weak_boundary_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 6.624999999999999
    ) <= 1.0e-12
    assert (
        retained_weak_boundary_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > retained_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_schur_shared_support_balance["available"]
    assert retained_schur_shared_support_balance["kind"] == (
        "diagnostic_shared_row_schur_shared_support_edge_balance"
    )
    assert retained_schur_shared_support_balance["edge_count"] == 10
    assert retained_schur_shared_support_balance["schur_edge_count"] == 6
    assert retained_schur_shared_support_balance["eligible_balance_rows"] == [
        0,
        1,
        2,
    ]
    assert retained_schur_shared_support_balance["preserves_hydrostatic_balance"]
    assert retained_schur_shared_support_balance[
        "preserves_constant_pressure_null"
    ]
    assert retained_schur_shared_support_balance["reduces_response_ratio"]
    assert retained_schur_shared_support_balance["reduces_max_target_response"]
    assert abs(
        retained_schur_shared_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 6.514223194748352
    ) <= 1.0e-12
    assert (
        retained_schur_shared_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > retained_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert (
        retained_schur_shared_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > retained_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_schur_weak_boundary_balance["available"]
    assert retained_schur_weak_boundary_balance["kind"] == (
        "diagnostic_shared_row_schur_weak_boundary_edge_balance"
    )
    assert retained_schur_weak_boundary_balance["edge_count"] == 10
    assert retained_schur_weak_boundary_balance["schur_edge_count"] == 6
    assert retained_schur_weak_boundary_balance[
        "schur_contribution_count"
    ] == 6
    assert retained_schur_weak_boundary_balance["eligible_balance_rows"] == [3, 4]
    assert retained_schur_weak_boundary_balance[
        "preserves_hydrostatic_balance"
    ]
    assert retained_schur_weak_boundary_balance[
        "preserves_constant_pressure_null"
    ]
    assert retained_schur_weak_boundary_balance["reduces_response_ratio"]
    assert retained_schur_weak_boundary_balance["reduces_max_target_response"]
    assert abs(
        retained_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 2.168696625002024
    ) <= 1.0e-12
    assert (
        retained_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        < retained_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert (
        retained_schur_weak_boundary_balance[
            "schur_completed_max_to_strongest_support_target_response_ratio"
        ]
        == retained_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert retained_direct_gap_patch_balance["available"]
    assert retained_direct_gap_patch_balance["kind"] == (
        "diagnostic_direct_support_gap_or_same_sign_patch_completion"
    )
    assert retained_direct_gap_patch_balance["support_gap_rows"] == [3, 4]
    assert retained_direct_gap_patch_balance["same_sign_pressure_patch_rows"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert retained_direct_gap_patch_balance["schur_edge_count"] == 6
    assert retained_direct_gap_patch_balance["schur_contribution_count"] == 6
    assert retained_direct_gap_patch_balance["edge_count"] == 10
    assert retained_direct_gap_patch_balance["eligible_balance_rows"] == [3, 4]
    retained_direct_gap_schur_only = retained_direct_gap_patch_balance[
        "schur_only_completion"
    ]
    assert retained_direct_gap_schur_only["available"]
    assert retained_direct_gap_schur_only["kind"] == (
        "diagnostic_direct_support_gap_or_same_sign_patch_schur_completion"
    )
    assert retained_direct_gap_schur_only["support_gap_rows"] == [3, 4]
    assert retained_direct_gap_schur_only["same_sign_pressure_patch_rows"] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert retained_direct_gap_schur_only["edge_count"] == 6
    assert retained_direct_gap_schur_only["contribution_count"] == 6
    assert retained_direct_gap_schur_only["preserves_hydrostatic_balance"]
    assert retained_direct_gap_schur_only["preserves_constant_pressure_null"]
    assert retained_direct_gap_schur_only["reduces_response_ratio"]
    assert retained_direct_gap_schur_only["reduces_max_target_response"]
    assert abs(
        retained_direct_gap_schur_only[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        - retained_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    ) <= 1.0e-12
    assert retained_direct_gap_patch_balance[
        "balance_stage_further_reduces_schur_response_ratio"
    ]
    assert retained_direct_gap_patch_balance[
        "balance_stage_response_ratio_reduction_factor"
    ] > 2.0
    assert retained_direct_gap_patch_balance["preserves_hydrostatic_balance"]
    assert retained_direct_gap_patch_balance["preserves_constant_pressure_null"]
    assert retained_direct_gap_patch_balance["reduces_response_ratio"]
    assert retained_direct_gap_patch_balance["reduces_max_target_response"]
    assert abs(
        retained_direct_gap_patch_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - retained_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    ) <= 1.0e-12
    assert retained_existing_edge_balance["available"]
    assert retained_existing_edge_balance["kind"] == (
        "diagnostic_existing_pressure_edge_support_balance"
    )
    assert retained_existing_edge_balance["edge_count"] == 4
    assert retained_existing_edge_balance["non_laplacian_offdiag_count"] == 0
    assert retained_existing_edge_balance["preserves_hydrostatic_balance"]
    assert retained_existing_edge_balance["preserves_constant_pressure_null"]
    assert retained_existing_edge_balance["reduces_response_ratio"]
    assert retained_existing_edge_balance["reduces_max_target_response"]
    assert (
        retained_existing_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        < retained_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert abs(
        retained_existing_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 4.75
    ) <= 1.0e-12
    assert all(
        abs(row_abs - retained_existing_edge_balance["target_row_abs_sum"])
        <= 1.0e-12
        for row_abs in retained_existing_edge_balance["balanced_row_abs_sum"][1:]
    )
    assert (
        retained_existing_edge_balance["max_edge_scale"]
        > retained_amplification["uniform_scale_probe"]["scale"]
    )
    assert retained_incident_support_balance["available"]
    assert retained_incident_support_balance["kind"] == (
        "diagnostic_existing_pressure_edge_incident_support_balance"
    )
    assert retained_incident_support_balance["edge_count"] == 4
    assert retained_incident_support_balance["target_incident_cell_count"] == 2
    assert retained_incident_support_balance["max_edge_scale"] == 2.0
    assert retained_incident_support_balance["preserves_hydrostatic_balance"]
    assert retained_incident_support_balance["preserves_constant_pressure_null"]
    assert retained_incident_support_balance["reduces_response_ratio"]
    assert retained_incident_support_balance["reduces_max_target_response"]
    assert abs(
        retained_incident_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 8.317073170731707
    ) <= 1.0e-12
    assert (
        retained_incident_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > retained_existing_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert not cases["pruned_trace_only_cut_adjacent_support"][
        "preserves_linear_pressure_state"
    ]
    assert not cases["pruned_trace_only_cut_adjacent_support"][
        "pspg_hydrostatic_balance"
    ]["preserves_hydrostatic_balance"]
    assert cases["fixed_pruned_cut_adjacent_support_skipped"][
        "preserves_linear_pressure_state"
    ]
    assert cases["fixed_pruned_cut_adjacent_support_skipped"][
        "pspg_hydrostatic_balance"
    ]["preserves_hydrostatic_balance"]
    assert (
        cases["fixed_pruned_cut_adjacent_support_skipped"][
            "pspg_hydrostatic_balance"
        ]["max_abs_pressure_gradient_action"]
        == 0.0
    )
    assert cases["fixed_pruned_cut_adjacent_support_skipped"]["face"] is None
    assert (
        cases["pruned_trace_only_cut_adjacent_support"]["face"][
            "grad_jump_current_norm_pa_per_m"
        ]
        > 0.0
    )
    assert (
        cases["pruned_trace_only_cut_adjacent_support"][
            "pspg_hydrostatic_balance"
        ]["max_abs_total_hydrostatic_action"]
        > 0.0
    )
    full_volume_amplification = cases["full_volume_one_cell_boundary_topology"][
        "pspg_hydrostatic_balance"
    ]["boundary_solve_amplification"]
    assert cases["full_volume_one_cell_boundary_topology"][
        "pspg_hydrostatic_balance"
    ]["preserves_hydrostatic_balance"]
    assert full_volume_amplification["available"]
    assert full_volume_amplification["constant_null_preserved_during_solve_proxy"]
    assert full_volume_amplification["max_response_is_weakest_diag_row"]
    assert abs(
        full_volume_amplification[
            "max_to_strongest_support_target_response_ratio"
        ]
        - 6.0
    ) <= 1.0e-12
    assert full_volume_amplification["max_target_response_row"][
        "incident_cell_count"
    ] == 1
    full_volume_completion = cases["full_volume_one_cell_boundary_topology"][
        "pspg_hydrostatic_balance"
    ]["boundary_pair_completion"]
    full_volume_shared_completion = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["shared_support_completion"]
    full_volume_active_completion = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["weak_boundary_active_completion"]
    full_volume_shared_row_schur_completion = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["shared_row_schur_completion"]
    full_volume_schur_edge_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["shared_row_schur_existing_edge_balance"]
    full_volume_weak_boundary_edge_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["weak_boundary_existing_edge_support_balance"]
    full_volume_schur_shared_support_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["shared_row_schur_shared_support_edge_balance"]
    full_volume_schur_weak_boundary_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["shared_row_schur_weak_boundary_edge_balance"]
    full_volume_direct_gap_patch_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["direct_support_gap_or_same_sign_patch_completion"]
    full_volume_existing_edge_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["existing_edge_support_balance"]
    full_volume_incident_support_balance = cases[
        "full_volume_one_cell_boundary_topology"
    ]["pspg_hydrostatic_balance"]["incident_support_count_balance"]
    assert full_volume_completion["available"]
    assert full_volume_completion["edge_count"] == 1
    assert full_volume_completion["preserves_hydrostatic_balance"]
    assert full_volume_completion["preserves_constant_pressure_null"]
    assert full_volume_completion["reduces_response_ratio"]
    assert full_volume_completion["reduces_max_target_response"]
    assert abs(
        full_volume_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        - 3.5
    ) <= 1.0e-12
    assert (
        full_volume_completion["completed_boundary_solve_amplification"][
            "max_target_response_row"
        ]["incident_cell_count"]
        == 2
    )
    assert full_volume_shared_completion["available"]
    assert full_volume_shared_completion["edge_count"] == 6
    assert full_volume_shared_completion["preserves_hydrostatic_balance"]
    assert full_volume_shared_completion["preserves_constant_pressure_null"]
    assert full_volume_shared_completion["reduces_response_ratio"]
    assert full_volume_shared_completion["reduces_max_target_response"]
    assert (
        full_volume_shared_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_active_completion["available"]
    assert full_volume_active_completion["edge_count"] == 7
    assert full_volume_active_completion["contribution_count"] == 8
    assert full_volume_active_completion["preserves_hydrostatic_balance"]
    assert full_volume_active_completion["preserves_constant_pressure_null"]
    assert full_volume_active_completion["reduces_response_ratio"]
    assert full_volume_active_completion["reduces_max_target_response"]
    assert (
        full_volume_active_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        < full_volume_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_shared_row_schur_completion["available"]
    assert full_volume_shared_row_schur_completion["edge_count"] == 6
    assert full_volume_shared_row_schur_completion["contribution_count"] == 6
    assert full_volume_shared_row_schur_completion["shared_support_rows"] == [
        0,
        1,
        2,
    ]
    assert full_volume_shared_row_schur_completion[
        "preserves_hydrostatic_balance"
    ]
    assert full_volume_shared_row_schur_completion[
        "preserves_constant_pressure_null"
    ]
    assert full_volume_shared_row_schur_completion["reduces_response_ratio"]
    assert full_volume_shared_row_schur_completion["reduces_max_target_response"]
    assert abs(
        full_volume_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        - 3.2058823529411793
    ) <= 1.0e-12
    assert (
        full_volume_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        < full_volume_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert (
        full_volume_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_active_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_schur_edge_balance["available"]
    assert full_volume_schur_edge_balance["edge_count"] == 10
    assert full_volume_schur_edge_balance["schur_edge_count"] == 6
    assert full_volume_schur_edge_balance["schur_contribution_count"] == 6
    assert full_volume_schur_edge_balance["preserves_hydrostatic_balance"]
    assert full_volume_schur_edge_balance["preserves_constant_pressure_null"]
    assert full_volume_schur_edge_balance["reduces_response_ratio"]
    assert full_volume_schur_edge_balance["reduces_max_target_response"]
    assert abs(
        full_volume_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 2.4029720279720284
    ) <= 1.0e-12
    assert (
        full_volume_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        < full_volume_active_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_schur_edge_balance["max_edge_scale"] < 4.0
    assert full_volume_weak_boundary_edge_balance["available"]
    assert full_volume_weak_boundary_edge_balance["edge_count"] == 4
    assert full_volume_weak_boundary_edge_balance["eligible_balance_rows"] == [
        3,
        4,
    ]
    assert full_volume_weak_boundary_edge_balance[
        "preserves_hydrostatic_balance"
    ]
    assert full_volume_weak_boundary_edge_balance[
        "preserves_constant_pressure_null"
    ]
    assert not full_volume_weak_boundary_edge_balance["reduces_response_ratio"]
    assert full_volume_weak_boundary_edge_balance["reduces_max_target_response"]
    assert abs(
        full_volume_weak_boundary_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 6.625000000000003
    ) <= 1.0e-12
    assert (
        full_volume_weak_boundary_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_amplification[
            "max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_schur_shared_support_balance["available"]
    assert full_volume_schur_shared_support_balance["edge_count"] == 10
    assert full_volume_schur_shared_support_balance["schur_edge_count"] == 6
    assert full_volume_schur_shared_support_balance["eligible_balance_rows"] == [
        0,
        1,
        2,
    ]
    assert full_volume_schur_shared_support_balance[
        "preserves_hydrostatic_balance"
    ]
    assert full_volume_schur_shared_support_balance[
        "preserves_constant_pressure_null"
    ]
    assert full_volume_schur_shared_support_balance["reduces_response_ratio"]
    assert full_volume_schur_shared_support_balance["reduces_max_target_response"]
    assert abs(
        full_volume_schur_shared_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 3.3936170212765933
    ) <= 1.0e-12
    assert (
        full_volume_schur_shared_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert (
        full_volume_schur_shared_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_schur_weak_boundary_balance["available"]
    assert full_volume_schur_weak_boundary_balance["edge_count"] == 10
    assert full_volume_schur_weak_boundary_balance["schur_edge_count"] == 6
    assert full_volume_schur_weak_boundary_balance[
        "eligible_balance_rows"
    ] == [3, 4]
    assert full_volume_schur_weak_boundary_balance[
        "preserves_hydrostatic_balance"
    ]
    assert full_volume_schur_weak_boundary_balance[
        "preserves_constant_pressure_null"
    ]
    assert full_volume_schur_weak_boundary_balance["reduces_response_ratio"]
    assert full_volume_schur_weak_boundary_balance["reduces_max_target_response"]
    assert abs(
        full_volume_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 2.1688311688311694
    ) <= 1.0e-12
    assert (
        full_volume_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        < full_volume_schur_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_direct_gap_patch_balance["available"]
    assert full_volume_direct_gap_patch_balance["support_gap_rows"] == [3, 4]
    assert full_volume_direct_gap_patch_balance[
        "same_sign_pressure_patch_rows"
    ] == [0, 1, 2, 3, 4]
    assert full_volume_direct_gap_patch_balance["schur_edge_count"] == 6
    full_volume_direct_gap_schur_only = full_volume_direct_gap_patch_balance[
        "schur_only_completion"
    ]
    assert full_volume_direct_gap_schur_only["available"]
    assert full_volume_direct_gap_schur_only["kind"] == (
        "diagnostic_direct_support_gap_or_same_sign_patch_schur_completion"
    )
    assert full_volume_direct_gap_schur_only["edge_count"] == 6
    assert full_volume_direct_gap_schur_only["contribution_count"] == 6
    assert full_volume_direct_gap_schur_only["preserves_hydrostatic_balance"]
    assert full_volume_direct_gap_schur_only["preserves_constant_pressure_null"]
    assert abs(
        full_volume_direct_gap_schur_only[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
        - full_volume_shared_row_schur_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    ) <= 1.0e-12
    assert full_volume_direct_gap_patch_balance[
        "balance_stage_further_reduces_schur_response_ratio"
    ]
    assert full_volume_direct_gap_patch_balance[
        "balance_stage_response_ratio_reduction_factor"
    ] > 1.0
    assert full_volume_direct_gap_patch_balance["edge_count"] == 10
    assert full_volume_direct_gap_patch_balance["preserves_hydrostatic_balance"]
    assert full_volume_direct_gap_patch_balance["preserves_constant_pressure_null"]
    assert full_volume_direct_gap_patch_balance["reduces_response_ratio"]
    assert full_volume_direct_gap_patch_balance["reduces_max_target_response"]
    assert abs(
        full_volume_direct_gap_patch_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - full_volume_schur_weak_boundary_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
    ) <= 1.0e-12
    assert full_volume_existing_edge_balance["available"]
    assert full_volume_existing_edge_balance["edge_count"] == 4
    assert full_volume_existing_edge_balance["preserves_hydrostatic_balance"]
    assert full_volume_existing_edge_balance["preserves_constant_pressure_null"]
    assert full_volume_existing_edge_balance["reduces_response_ratio"]
    assert full_volume_existing_edge_balance["reduces_max_target_response"]
    assert abs(
        full_volume_existing_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 4.75
    ) <= 1.0e-12
    assert (
        full_volume_existing_edge_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
    assert full_volume_incident_support_balance["available"]
    assert full_volume_incident_support_balance["edge_count"] == 4
    assert full_volume_incident_support_balance["target_incident_cell_count"] == 2
    assert full_volume_incident_support_balance["max_edge_scale"] == 2.0
    assert full_volume_incident_support_balance["preserves_hydrostatic_balance"]
    assert full_volume_incident_support_balance[
        "preserves_constant_pressure_null"
    ]
    assert full_volume_incident_support_balance["reduces_response_ratio"]
    assert full_volume_incident_support_balance["reduces_max_target_response"]
    assert abs(
        full_volume_incident_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        - 4.75
    ) <= 1.0e-12
    assert (
        full_volume_incident_support_balance[
            "balanced_max_to_strongest_support_target_response_ratio"
        ]
        > full_volume_completion[
            "completed_max_to_strongest_support_target_response_ratio"
        ]
    )
