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
        / "audit_pressure_matrix_support_samples.py"
    )
    spec = importlib.util.spec_from_file_location(
        "audit_pressure_matrix_support_samples",
        script,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_matrix_support_sample_summary_merges_constraint_context(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "\n".join(
            [
                "[R0] [INFO] LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint_sample "
                "field='Pressure' local_dof=2 status=ok global_dof=12 "
                "active_dof_support=1 inactive_constraint=0 "
                "retained_rule_count=4 retained_measure=1.5 "
                "retained_min_volume_fraction=0.25 "
                "retained_max_volume_fraction=1 "
                "entity_kind=Vertex entity_id=4 vertex_phi=-0.1 "
                "vertex_active_sign=1",
                "[R0] [INFO] LevelSetActiveSideVertexDirichletConstraint: "
                "diagnostic=level_set_active_side_vertex_constraint_sample "
                "field='Pressure' local_dof=5 status=ok global_dof=15 "
                "active_dof_support=0 inactive_constraint=1 "
                "entity_kind=Vertex entity_id=8 vertex_phi=0.2 "
                "vertex_active_sign=0",
                "[R0] [INFO] NewtonSolver: matrix support diagnostic "
                "diagnostic=newton_matrix_support_sample rank=0 iteration=0 "
                "phase='pre_linear_solve' backend=eigen solve_time=1 dt=0.1 "
                "dof=12 status=ok row_abs_sum=2e-8 row_numeric_entries=2 "
                "row_max_abs=1e-8 col_abs_sum=3e-8 col_numeric_entries=3 "
                "col_max_abs=1.5e-8 row_constrained_abs_sum=4e-9 "
                "row_unconstrained_abs_sum=1.6e-8 "
                "col_constrained_abs_sum=6e-9 "
                "col_unconstrained_abs_sum=2.4e-8 "
                "diag=1e-8 row_first_nonzero=12:1e-8 "
                "col_first_nonzero=12:1e-8 "
                "row_field_abs_sums=phi:0|Velocity:0|Pressure:2e-8 "
                "row_constrained_field_abs_sums=phi:0|Velocity:0|Pressure:4e-9 "
                "row_unconstrained_field_abs_sums=phi:0|Velocity:0|Pressure:1.6e-8 "
                "col_field_abs_sums=phi:0|Velocity:0|Pressure:3e-8 "
                "col_constrained_field_abs_sums=phi:0|Velocity:0|Pressure:6e-9 "
                "col_unconstrained_field_abs_sums=phi:0|Velocity:0|Pressure:2.4e-8 "
                "field='Pressure' field_local_dof=2",
                "[R0] [INFO] NewtonSolver: matrix support diagnostic "
                "diagnostic=newton_matrix_support_sample rank=0 iteration=0 "
                "phase='pre_linear_solve' backend=eigen solve_time=1 dt=0.1 "
                "dof=15 status=ok row_abs_sum=0 row_numeric_entries=0 "
                "row_max_abs=0 col_abs_sum=0 col_numeric_entries=0 "
                "col_max_abs=0 row_constrained_abs_sum=0 "
                "row_unconstrained_abs_sum=0 col_constrained_abs_sum=0 "
                "col_unconstrained_abs_sum=0 diag=0 row_first_nonzero=none "
                "col_first_nonzero=none "
                "row_field_abs_sums=phi:0|Velocity:0|Pressure:0 "
                "row_constrained_field_abs_sums=phi:0|Velocity:0|Pressure:0 "
                "row_unconstrained_field_abs_sums=phi:0|Velocity:0|Pressure:0 "
                "col_field_abs_sums=phi:0|Velocity:0|Pressure:0 "
                "col_constrained_field_abs_sums=phi:0|Velocity:0|Pressure:0 "
                "col_unconstrained_field_abs_sums=phi:0|Velocity:0|Pressure:0 "
                "field='Pressure' field_local_dof=5",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix support "
                "diagnostic=pressure_row_operator_matrix_support rank=0 "
                "iteration=0 phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_vms_pspg' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' pressure_offset=10 pressure_dofs=10 "
                "coupling_offset=20 coupling_dofs=30 dof=12 status=ok "
                "field='Pressure' field_local_dof=2 pressure_local_dof=2 "
                "row_abs_sum=2.3e-8 row_numeric_entries=4 "
                "row_self_abs_sum=1e-8 row_self_numeric_entries=2 "
                "row_self_sum=-1e-9 row_self_offdiag_abs_sum=5e-9 "
                "row_self_signed_abs_ratio=0.1 row_self_diag_abs_ratio=0.5 "
                "row_coupling_abs_sum=3e-4 row_coupling_numeric_entries=2 "
                "col_abs_sum=2e-8 col_numeric_entries=3 "
                "col_self_abs_sum=2e-8 col_self_numeric_entries=2 "
                "col_coupling_abs_sum=4e-4 col_coupling_numeric_entries=1 "
                "diag=5e-9 row_first_nonzero=12:5e-9 "
                "col_first_nonzero=12:5e-9",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix support "
                "diagnostic=pressure_row_operator_matrix_support rank=0 "
                "iteration=0 phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_vms_pspg_pressure_gradient' "
                "backend=eigen solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' pressure_offset=10 pressure_dofs=10 "
                "coupling_offset=20 coupling_dofs=30 dof=12 status=ok "
                "field='Pressure' field_local_dof=2 pressure_local_dof=2 "
                "row_abs_sum=1e-8 row_numeric_entries=2 "
                "row_self_abs_sum=1e-8 row_self_numeric_entries=2 "
                "row_self_sum=-1e-9 row_self_offdiag_abs_sum=5e-9 "
                "row_self_signed_abs_ratio=0.1 row_self_diag_abs_ratio=0.5 "
                "row_coupling_abs_sum=0 row_coupling_numeric_entries=0 "
                "col_abs_sum=2e-8 col_numeric_entries=3 "
                "col_self_abs_sum=2e-8 col_self_numeric_entries=2 "
                "col_coupling_abs_sum=0 col_coupling_numeric_entries=0 "
                "diag=5e-9 row_first_nonzero=12:5e-9 "
                "col_first_nonzero=12:5e-9",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix support "
                "diagnostic=pressure_row_operator_matrix_support rank=0 "
                "iteration=0 phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_vms_pspg_boundary_pressure_flux' "
                "backend=eigen solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' pressure_offset=10 pressure_dofs=10 "
                "coupling_offset=20 coupling_dofs=30 dof=12 status=ok "
                "field='Pressure' field_local_dof=2 pressure_local_dof=2 "
                "row_abs_sum=4e-9 row_numeric_entries=2 "
                "row_self_abs_sum=4e-9 row_self_numeric_entries=2 "
                "row_self_sum=0 row_self_offdiag_abs_sum=2e-9 "
                "row_self_signed_abs_ratio=0 row_self_diag_abs_ratio=0.5 "
                "row_coupling_abs_sum=0 row_coupling_numeric_entries=0 "
                "col_abs_sum=4e-9 col_numeric_entries=2 "
                "col_self_abs_sum=4e-9 col_self_numeric_entries=2 "
                "col_coupling_abs_sum=0 col_coupling_numeric_entries=0 "
                "diag=-2e-9 row_first_nonzero=12:-2e-9 "
                "col_first_nonzero=12:-2e-9",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix support "
                "diagnostic=pressure_row_operator_matrix_support rank=0 "
                "iteration=0 phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient' "
                "backend=eigen solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' pressure_offset=10 pressure_dofs=10 "
                "coupling_offset=20 coupling_dofs=30 dof=12 status=ok "
                "field='Pressure' field_local_dof=2 pressure_local_dof=2 "
                "row_abs_sum=6e-9 row_numeric_entries=2 "
                "row_self_abs_sum=6e-9 row_self_numeric_entries=2 "
                "row_self_sum=0 row_self_offdiag_abs_sum=3e-9 "
                "row_self_signed_abs_ratio=0 row_self_diag_abs_ratio=0.5 "
                "row_coupling_abs_sum=0 row_coupling_numeric_entries=0 "
                "col_abs_sum=6e-9 col_numeric_entries=2 "
                "col_self_abs_sum=6e-9 col_self_numeric_entries=2 "
                "col_coupling_abs_sum=0 col_coupling_numeric_entries=0 "
                "diag=3e-9 row_first_nonzero=12:3e-9 "
                "col_first_nonzero=12:3e-9",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix support "
                "diagnostic=pressure_row_operator_matrix_support rank=0 "
                "iteration=0 phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual' "
                "backend=eigen solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' pressure_offset=10 pressure_dofs=10 "
                "coupling_offset=20 coupling_dofs=30 dof=12 status=ok "
                "field='Pressure' field_local_dof=2 pressure_local_dof=2 "
                "row_abs_sum=2.00006e-4 row_numeric_entries=4 "
                "row_self_abs_sum=6e-9 row_self_numeric_entries=2 "
                "row_self_sum=0 row_self_offdiag_abs_sum=3e-9 "
                "row_self_signed_abs_ratio=0 row_self_diag_abs_ratio=0.5 "
                "row_coupling_abs_sum=2e-4 row_coupling_numeric_entries=2 "
                "col_abs_sum=3.00006e-4 col_numeric_entries=3 "
                "col_self_abs_sum=6e-9 col_self_numeric_entries=2 "
                "col_coupling_abs_sum=3e-4 col_coupling_numeric_entries=1 "
                "diag=3e-9 row_first_nonzero=12:3e-9|22:2e-4 "
                "col_first_nonzero=12:3e-9|23:3e-4",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix support "
                "diagnostic=pressure_row_operator_matrix_support rank=0 "
                "iteration=0 phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_pressure_ghost_penalty' "
                "backend=eigen solve_time=1 dt=0.1 "
                "pressure_field='Pressure' coupling_field='Velocity' "
                "pressure_offset=10 pressure_dofs=10 coupling_offset=20 "
                "coupling_dofs=30 dof=12 status=ok field='Pressure' "
                "field_local_dof=2 pressure_local_dof=2 row_abs_sum=0 "
                "row_numeric_entries=0 row_self_abs_sum=0 "
                "row_self_numeric_entries=0 row_self_sum=0 "
                "row_self_offdiag_abs_sum=0 row_self_signed_abs_ratio=0 "
                "row_self_diag_abs_ratio=0 row_coupling_abs_sum=0 "
                "row_coupling_numeric_entries=0 col_abs_sum=0 "
                "col_numeric_entries=0 col_self_abs_sum=0 "
                "col_self_numeric_entries=0 col_coupling_abs_sum=0 "
                "col_coupling_numeric_entries=0 diag=0 "
                "row_first_nonzero=none col_first_nonzero=none",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix summary "
                "diagnostic=pressure_row_operator_matrix_summary status=ok "
                "rank=0 iteration=0 "
                "phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_galerkin_continuity' "
                "backend=eigen solve_time=1 dt=0.1 "
                "pressure_field='Pressure' coupling_field='Velocity' "
                "pressure_offset=10 pressure_dofs=10 coupling_offset=20 "
                "coupling_dofs=30 constrained_pressure_rows=3 "
                "unconstrained_pressure_rows=7 zero_row_count=0 "
                "zero_col_count=0 zero_diag_count=7 "
                "zero_coupling_row_block_count=0 "
                "zero_coupling_col_block_count=0 "
                "zero_self_row_block_count=7 zero_self_col_block_count=7 "
                "positive_coupling_row_block_count=7 "
                "positive_self_row_block_count=0 "
                "weak_coupling_row_block_count=2 weak_self_row_block_count=0 "
                "weak_coupling_and_self_row_block_count=0 "
                "min_positive_coupling_row_abs_sum=1e-5 "
                "max_coupling_row_abs_sum=2e-3 "
                "min_positive_self_row_abs_sum=0 max_self_row_abs_sum=0 "
                "pressure_only_row_block_count=0 pressure_only_col_block_count=0 "
                "tolerance=1e-14 weak_coupling_threshold=1e-3 "
                "weak_self_threshold=1e-7 sample_limit=16 "
                "zero_coupling_row_local_dofs=none "
                "zero_coupling_row_global_dofs=none "
                "zero_row_local_dofs=none zero_row_global_dofs=none "
                "weakest_coupling_row_local_dofs=2|5 "
                "weakest_coupling_row_global_dofs=12|15 "
                "weakest_self_row_local_dofs=none "
                "weakest_self_row_global_dofs=none",
                "[R0] [INFO] NewtonSolver: pressure row operator matrix summary "
                "diagnostic=pressure_row_operator_matrix_summary status=ok "
                "rank=0 iteration=0 "
                "phase='pressure_row_contribution_matrix:residual' "
                "op='equations_diagnostic_ns_vms_pspg_pressure_gradient' "
                "backend=eigen solve_time=1 dt=0.1 "
                "pressure_field='Pressure' coupling_field='Velocity' "
                "pressure_offset=10 pressure_dofs=10 coupling_offset=20 "
                "coupling_dofs=30 constrained_pressure_rows=3 "
                "unconstrained_pressure_rows=7 zero_row_count=0 "
                "zero_col_count=0 zero_diag_count=0 "
                "zero_coupling_row_block_count=7 "
                "zero_coupling_col_block_count=7 "
                "zero_self_row_block_count=0 zero_self_col_block_count=0 "
                "positive_coupling_row_block_count=0 "
                "positive_self_row_block_count=7 "
                "weak_coupling_row_block_count=0 weak_self_row_block_count=3 "
                "weak_coupling_and_self_row_block_count=0 "
                "min_positive_coupling_row_abs_sum=0 "
                "max_coupling_row_abs_sum=0 "
                "min_positive_self_row_abs_sum=2e-8 "
                "max_self_row_abs_sum=3e-1 "
                "pressure_only_row_block_count=7 pressure_only_col_block_count=7 "
                "tolerance=1e-14 weak_coupling_threshold=1e-3 "
                "weak_self_threshold=1e-7 sample_limit=16 "
                "zero_coupling_row_local_dofs=2|5 "
                "zero_coupling_row_global_dofs=12|15 "
                "zero_row_local_dofs=none zero_row_global_dofs=none "
                "weakest_coupling_row_local_dofs=none "
                "weakest_coupling_row_global_dofs=none "
                "weakest_self_row_local_dofs=2|5 "
                "weakest_self_row_global_dofs=12|15",
                "[svMultiPhysics::Application] Accepted pressure update diagnostic "
                "diagnostic=accepted_pressure_update_guard field='Pressure' "
                "local_worst_dof=12 local_abs_pressure_delta_pa=50 triggered=0",
                "[R0] [INFO] NewtonSolver: active pressure support-rank diagnostic "
                "diagnostic=active_pressure_support_rank rank=0 iteration=0 "
                "phase='pre_linear_solve' backend=eigen solve_time=1 dt=0.1 "
                "pressure_field='Pressure' coupling_field='Velocity' "
                "pressure_offset=10 pressure_dofs=10 coupling_offset=20 "
                "coupling_dofs=30 constrained_pressure_rows=3 "
                "unconstrained_pressure_rows=7 zero_row_count=1 "
                "zero_col_count=1 zero_diag_count=1 "
                "zero_coupling_row_block_count=2 "
                "zero_coupling_col_block_count=2 "
                "zero_self_row_block_count=1 zero_self_col_block_count=1 "
                "positive_coupling_row_block_count=5 "
                "positive_self_row_block_count=6 "
                "min_positive_coupling_row_abs_sum=3e-4 "
                "max_coupling_row_abs_sum=2e-1 "
                "min_positive_self_row_abs_sum=2e-8 "
                "max_self_row_abs_sum=3e-1 "
                "pressure_only_row_block_count=1 "
                "pressure_only_col_block_count=1 tolerance=1e-14 "
                "zero_coupling_row_local_dofs=2|5 "
                "zero_coupling_row_global_dofs=12|15 "
                "zero_row_local_dofs=5 zero_row_global_dofs=15 "
                "weakest_coupling_row_local_dofs=7 "
                "weakest_coupling_row_global_dofs=17 "
                "weakest_self_row_local_dofs=8 "
                "weakest_self_row_global_dofs=18",
                "[R0] [INFO] NewtonSolver: active pressure support-rank clamp "
                "diagnostic=active_pressure_support_rank_clamp rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' clamped_row_count=2 "
                "clamp_coupling_threshold=1e-14 "
                "clamp_self_threshold=1e-8 "
                "constrained_pressure_rows=3 unconstrained_pressure_rows=7 "
                "zero_coupling_row_block_count=2 "
                "positive_coupling_row_block_count=5 "
                "positive_self_row_block_count=6 "
                "min_positive_self_row_abs_sum=2e-8 "
                "max_self_row_abs_sum=3e-1 "
                "pressure_only_row_block_count=1 tolerance=1e-14 "
                "clamped_local_dofs=2|5 clamped_global_dofs=12|15",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' mode='shared_velocity_neighbor' "
                "requested_mode='shared_velocity_neighbor' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 candidate_row_count=2 neighbor_row_count=1 edge_count=1 "
                "edge_weight=5e-9 "
                "edge_weight_rule='min_positive_candidate_diagonal_to_shared_velocity_neighbor' "
                "neighbor_policy='max_shared_velocity_signature_then_row_support' "
                "weight_scale=1 min_positive_candidate_diag_abs=5e-9 "
                "applied=1 candidate_global_dofs=12|15 neighbor_global_dofs=18",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' mode='shared_pressure_neighbor' "
                "requested_mode='shared_pressure_neighbor' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 candidate_row_count=2 neighbor_row_count=2 "
                "edge_count=1 edge_weight=5e-9 "
                "edge_weight_rule='min_positive_candidate_diagonal_to_shared_pressure_neighbor_pair' "
                "neighbor_policy='candidate_pair_with_max_shared_pressure_neighbor_support' "
                "weight_scale=1 max_edge_scale_cap=16 "
                "min_positive_candidate_diag_abs=5e-9 "
                "min_completion_edge_weight=5e-9 "
                "max_completion_edge_weight=5e-9 "
                "min_completion_edge_scale=1 "
                "max_completion_edge_scale=1 "
                "non_laplacian_existing_edge_count=0 "
                "applied=1 candidate_global_dofs=12|15 "
                "neighbor_global_dofs=18|19",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' mode='existing_support_balance' "
                "requested_mode='existing_support_balance' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 candidate_row_count=2 "
                "zero_coupling_candidate_count=1 "
                "weak_coupling_candidate_count=1 "
                "zero_self_candidate_count=0 "
                "weak_self_candidate_count=2 "
                "weak_coupling_and_self_candidate_count=2 "
                "neighbor_row_count=4 edge_count=3 edge_weight=9e-9 "
                "edge_weight_rule='existing_pressure_edges_abs_scaled_to_target_self_row_abs_sum' "
                "neighbor_policy='all_existing_pressure_edges_incident_to_weak_rows' "
                "weight_scale=1 max_edge_scale_cap=16 "
                "min_positive_candidate_diag_abs=5e-9 "
                "target_self_row_abs_sum=3e-1 "
                "min_completion_edge_weight=4e-9 "
                "max_completion_edge_weight=9e-9 "
                "min_completion_edge_scale=3 "
                "max_completion_edge_scale=15 "
                "non_laplacian_existing_edge_count=2 "
                "candidate_with_existing_pressure_edge_count=2 "
                "candidate_with_laplacian_pressure_edge_count=1 "
                "candidate_with_non_laplacian_only_pressure_edge_count=1 "
                "applied=1 candidate_global_dofs=12|15 "
                "neighbor_global_dofs=12|15|18|19",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' mode='active_support_completion' "
                "requested_mode='active_support_completion' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 max_active_neighbors=2 candidate_row_count=2 "
                "zero_coupling_candidate_count=1 "
                "weak_coupling_candidate_count=1 "
                "zero_self_candidate_count=0 "
                "weak_self_candidate_count=2 "
                "weak_coupling_and_self_candidate_count=2 "
                "neighbor_row_count=4 edge_count=4 edge_weight=2.5e-9 "
                "edge_weight_rule='min_positive_candidate_diagonal_distributed_to_active_pressure_support' "
                "neighbor_policy='strongest_unconstrained_pressure_self_rows' "
                "weight_scale=1 max_edge_scale_cap=16 "
                "min_positive_candidate_diag_abs=5e-9 "
                "target_self_row_abs_sum=5e-9 "
                "min_completion_edge_weight=2.5e-9 "
                "max_completion_edge_weight=2.5e-9 "
                "min_completion_edge_scale=1 "
                "max_completion_edge_scale=1 "
                "non_laplacian_existing_edge_count=0 "
                "applied=1 candidate_global_dofs=12|15 "
                "neighbor_global_dofs=12|15|18|19",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' mode='shared_row_schur_completion' "
                "requested_mode='shared_row_schur_completion' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 max_active_neighbors=2 candidate_row_count=2 "
                "zero_coupling_candidate_count=1 "
                "weak_coupling_candidate_count=1 "
                "zero_self_candidate_count=0 "
                "weak_self_candidate_count=2 "
                "weak_coupling_and_self_candidate_count=2 "
                "neighbor_row_count=4 edge_count=5 edge_weight=4e-9 "
                "edge_weight_rule='existing_pressure_laplacian_schur_fill_wi_wj_over_hub_support_sum' "
                "neighbor_policy='weak_candidate_pressure_neighbors_to_shared_row_pressure_neighbors' "
                "weight_scale=1 max_edge_scale_cap=16 "
                "min_positive_candidate_diag_abs=5e-9 "
                "target_self_row_abs_sum=0 "
                "min_completion_edge_weight=1e-9 "
                "max_completion_edge_weight=4e-9 "
                "min_completion_edge_scale=1 "
                "max_completion_edge_scale=1 "
                "non_laplacian_existing_edge_count=0 "
                "shared_row_schur_hub_count=3 "
                "shared_row_schur_candidate_edge_count=4 "
                "shared_row_schur_contribution_count=7 "
                "applied=1 candidate_global_dofs=12|15 "
                "neighbor_global_dofs=12|15|18|19",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' "
                "mode='shared_row_schur_existing_edge_balance' "
                "requested_mode='shared_row_schur_existing_edge_balance' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 max_active_neighbors=2 candidate_row_count=2 "
                "zero_coupling_candidate_count=1 "
                "weak_coupling_candidate_count=1 "
                "zero_self_candidate_count=0 "
                "weak_self_candidate_count=2 "
                "weak_coupling_and_self_candidate_count=2 "
                "neighbor_row_count=4 edge_count=7 edge_weight=9e-9 "
                "edge_weight_rule='shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum' "
                "neighbor_policy='weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_weak_rows' "
                "weight_scale=1 max_edge_scale_cap=16 "
                "min_positive_candidate_diag_abs=5e-9 "
                "target_self_row_abs_sum=3e-1 "
                "min_completion_edge_weight=1e-9 "
                "max_completion_edge_weight=9e-9 "
                "min_completion_edge_scale=1 "
                "max_completion_edge_scale=12 "
                "non_laplacian_existing_edge_count=1 "
                "candidate_with_existing_pressure_edge_count=2 "
                "candidate_with_laplacian_pressure_edge_count=2 "
                "candidate_with_non_laplacian_only_pressure_edge_count=0 "
                "shared_row_schur_hub_count=3 "
                "shared_row_schur_candidate_edge_count=4 "
                "shared_row_schur_contribution_count=7 "
                "shared_row_schur_edge_count=5 "
                "existing_balance_edge_count=2 "
                "applied=1 candidate_global_dofs=12|15 "
                "neighbor_global_dofs=12|15|18|19",
                "[R0] [INFO] NewtonSolver: active pressure graph completion "
                "diagnostic=active_pressure_graph_completion rank=0 "
                "iteration=0 phase='pre_linear_solve' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' mode='existing_edge_balance' "
                "requested_mode='existing_edge_balance' "
                "coupling_threshold=1e-14 self_threshold=1e-8 "
                "max_rows=512 candidate_row_count=2 neighbor_row_count=3 "
                "edge_count=2 edge_weight=8e-9 "
                "edge_weight_rule='existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum' "
                "neighbor_policy='existing_pressure_edges_incident_to_weak_rows' "
                "weight_scale=1 max_edge_scale_cap=16 "
                "min_positive_candidate_diag_abs=5e-9 "
                "target_self_row_abs_sum=3e-1 "
                "min_completion_edge_weight=4e-9 "
                "max_completion_edge_weight=8e-9 "
                "min_completion_edge_scale=3 "
                "max_completion_edge_scale=15 "
                "non_laplacian_existing_edge_count=1 "
                "applied=1 candidate_global_dofs=12|15 "
                "neighbor_global_dofs=12|15|18",
                "[R0] [INFO] NewtonSolver: active pressure update support diagnostic "
                "diagnostic=active_pressure_update_support rank=0 "
                "iteration=0 phase='post_linear_solve_update' backend=eigen "
                "solve_time=1 dt=0.1 pressure_field='Pressure' "
                "coupling_field='Velocity' pressure_offset=10 pressure_dofs=10 "
                "coupling_offset=20 coupling_dofs=30 constrained_pressure_rows=3 "
                "unconstrained_pressure_rows=7 tolerance=1e-14 "
                "weak_coupling_threshold=3.3e-4 weak_self_threshold=1e-7 "
                "action_sample_limit=4 "
                "same_sign_pressure_action_top_edge_count=1 "
                "same_sign_pressure_action_component_count=1 "
                "same_sign_pressure_action_largest_component_size=2 "
                "same_sign_pressure_action_covered_top_update_count=2 "
                "same_sign_pressure_action_isolated_top_update_count=1 "
                "same_sign_pressure_action_largest_component_has_max_update=1 "
                "same_sign_pressure_action_covered_global_dofs=18|19 "
                "same_sign_pressure_action_isolated_global_dofs=21 "
                "same_sign_pressure_action_largest_component_global_dofs=18|19 "
                "zero_coupling_row_block_count=2 "
                "weak_coupling_row_block_count=1 "
                "positive_coupling_row_block_count=4 "
                "zero_self_row_block_count=1 weak_self_row_block_count=2 "
                "positive_self_row_block_count=4 max_abs_update=42 "
                "max_update_local_dof=8 max_update_global_dof=18 "
                "max_update_rhs=1.25 "
                "max_update_row_action=1.250000000001 "
                "max_update_row_coupling_action=0.75 "
                "max_update_row_self_action=0.5 "
                "max_update_row_self_constant_action=0.25 "
                "max_update_row_self_nonconstant_action=0.25 "
                "max_update_row_other_action=0.000000000001 "
                "max_update_row_linear_residual=1e-12 "
                "zero_coupling_max_abs_update=7 "
                "weak_coupling_max_abs_update=42 "
                "positive_coupling_max_abs_update=3 "
                "zero_self_max_abs_update=1 "
                "weak_self_max_abs_update=42 "
                "positive_self_max_abs_update=7 "
                "top_update_details="
                "8:18:update=-42:abs_update=42:rhs=1.25:"
                "row_action=1.250000000001:row_coupling_action=0.75:"
                "row_self_action=0.5:row_self_constant_action=0.25:"
                "row_self_nonconstant_action=0.25:"
                "row_other_action=0.000000000001:"
                "row_linear_residual=1e-12:row=1.1:"
                "row_coupling=3e-4:row_self=1e-8:"
                "row_self_sum=-1e-9:row_self_offdiag=5e-9:"
                "row_self_signed_abs_ratio=0.1:row_self_diag_abs_ratio=0.5:"
                "col=2:col_coupling=6e-4:col_self=2e-8:diag=5e-9:"
                "pressure_action_terms=8/18/m=0.5/u=1/a=0.5:"
                "coupling_action_terms=0/20/m=0.75/u=1/a=0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)

    assert report["matrix_sample_count"] == 2
    assert report["latest_matrix_sampled_row_count"] == 2
    assert report["operator_matrix_support_sample_count"] == 6
    assert report["operator_matrix_summary_count"] == 2
    assert report["latest_operator_matrix_support_sample_count"] == 6
    assert report["latest_operator_matrix_summary_count"] == 2
    assert report["constraint_sample_count"] == 2
    assert report["accepted_pressure_update_count"] == 1
    assert report["support_rank_diagnostic_count"] == 1
    assert report["support_rank_clamp_count"] == 1
    assert report["pressure_graph_completion_count"] == 7
    assert report["pressure_update_support_diagnostic_count"] == 1
    assert report["matrix_sample_status_counts"] == {"ok": 2}
    assert report["operator_matrix_support_status_counts"] == {"ok": 6}
    assert report["operator_matrix_support_ops"] == [
        "equations_diagnostic_ns_pressure_ghost_penalty",
        "equations_diagnostic_ns_vms_pspg",
        "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux",
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual",
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient",
        "equations_diagnostic_ns_vms_pspg_pressure_gradient",
    ]
    assert report["operator_matrix_support_nonzero_self_row_ops"] == [
        "equations_diagnostic_ns_vms_pspg",
        "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux",
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual",
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient",
        "equations_diagnostic_ns_vms_pspg_pressure_gradient",
    ]
    assert report["operator_matrix_support_nonzero_coupling_row_ops"] == [
        "equations_diagnostic_ns_vms_pspg",
        "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual",
    ]
    coverage = report["operator_matrix_support_by_op"]
    assert coverage["equations_diagnostic_ns_vms_pspg"]["sampled_row_count"] == 1
    assert (
        coverage["equations_diagnostic_ns_vms_pspg"][
            "nonzero_self_local_pressure_rows"
        ]
        == "2"
    )
    assert (
        coverage["equations_diagnostic_ns_vms_pspg"][
            "nonzero_coupling_local_pressure_rows"
        ]
        == "2"
    )
    assert (
        coverage["equations_diagnostic_ns_vms_pspg_pressure_gradient"][
            "nonzero_self_row_count"
        ]
        == 1
    )
    assert (
        coverage["equations_diagnostic_ns_vms_pspg_pressure_gradient"][
            "zero_coupling_row_count"
        ]
        == 1
    )
    assert (
        coverage["equations_diagnostic_ns_vms_pspg_pressure_gradient"][
            "min_positive_self_row_abs_sum"
        ]
        == 1.0e-8
    )
    assert (
        coverage[
            "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux"
        ]["nonzero_self_row_count"]
        == 1
    )
    assert (
        coverage[
            "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux"
        ]["zero_coupling_row_count"]
        == 1
    )
    assert (
        coverage[
            "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient"
        ]["nonzero_self_row_count"]
        == 1
    )
    assert (
        coverage[
            "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient"
        ]["zero_coupling_row_count"]
        == 1
    )
    assert (
        coverage[
            "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual"
        ]["nonzero_self_row_count"]
        == 1
    )
    assert (
        coverage[
            "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual"
        ]["nonzero_coupling_row_count"]
        == 1
    )
    assert (
        coverage["equations_diagnostic_ns_pressure_ghost_penalty"][
            "zero_self_row_count"
        ]
        == 1
    )
    summaries = report["operator_matrix_summary_by_op"]
    assert (
        summaries["equations_diagnostic_ns_galerkin_continuity"][
            "positive_coupling_row_block_count"
        ]
        == 7
    )
    assert (
        summaries["equations_diagnostic_ns_galerkin_continuity"][
            "zero_self_row_block_count"
        ]
        == 7
    )
    assert (
        summaries["equations_diagnostic_ns_vms_pspg_pressure_gradient"][
            "zero_coupling_row_block_count"
        ]
        == 7
    )
    assert (
        summaries["equations_diagnostic_ns_vms_pspg_pressure_gradient"][
            "pressure_only_row_block_count"
        ]
        == 7
    )
    assert (
        summaries["equations_diagnostic_ns_vms_pspg_pressure_gradient"][
            "weak_self_row_block_count"
        ]
        == 3
    )
    assert report["matrix_sampled_zero_row_count"] == 1
    assert report["matrix_sampled_zero_col_count"] == 1
    assert report["matrix_sampled_zero_diag_count"] == 1
    assert report["matrix_sampled_nonzero_row_count"] == 1
    assert report["matrix_sampled_active_support_count"] == 1
    assert report["matrix_sampled_inactive_constraint_count"] == 1
    assert report["matrix_sampled_row_field_block_sample_count"] == 2
    assert report["matrix_sampled_col_field_block_sample_count"] == 2
    assert report["matrix_sampled_zero_coupling_row_block_count"] == 2
    assert report["matrix_sampled_zero_coupling_col_block_count"] == 2
    assert report["matrix_sampled_nonzero_self_row_block_count"] == 1
    assert report["matrix_sampled_nonzero_self_col_block_count"] == 1
    provenance = report["constraint_support_provenance_summary"]
    assert provenance["sampled_row_count"] == 2
    assert provenance["weak_coupling_threshold"] == 1.0e-3
    assert provenance["weak_self_threshold"] == 1.0e-7
    assert provenance["counts"] == {
        "active_support": 1,
        "inactive_constraint": 1,
        "retained_rule_support": 1,
        "retained_weak_self_row": 1,
        "retained_zero_coupling_row": 1,
        "vertex_active_sign": 1,
        "weak_self_row": 1,
        "zero_coupling_row": 2,
        "zero_self_row": 1,
    }
    assert provenance["max_row_coupling_abs_sum_by_class"]["retained"] == 0.0
    assert provenance["max_row_self_abs_sum_by_class"]["weak_self"] == 2.0e-8
    first_row = report["sampled_pressure_rows"][0]
    assert first_row["row_field_abs_sum_by_field"]["Pressure"] == 2.0e-8
    assert (
        first_row["row_constrained_field_abs_sum_by_field"]["Pressure"]
        == 4.0e-9
    )
    assert (
        first_row["row_unconstrained_field_abs_sum_by_field"]["Pressure"]
        == 1.6e-8
    )
    assert (
        first_row["col_constrained_field_abs_sum_by_field"]["Pressure"]
        == 6.0e-9
    )
    assert (
        first_row["col_unconstrained_field_abs_sum_by_field"]["Pressure"]
        == 2.4e-8
    )
    assert first_row["constraint_sample"]["entity_id"] == 4
    assert first_row["constraint_sample"]["retained_rule_count"] == 4
    assert first_row["constraint_sample"]["retained_measure"] == 1.5
    assert first_row["constraint_sample"]["retained_min_volume_fraction"] == 0.25
    assert report["sampled_pressure_rows"][1]["matrix_sample"]["row_abs_sum"] == 0
    operator_samples = report["pressure_row_operator_matrix_support_samples"]
    vms_sample = next(
        row
        for row in operator_samples
        if row["op"] == "equations_diagnostic_ns_vms_pspg"
    )
    assert vms_sample["local_pressure_row"] == 2
    assert (
        vms_sample["operator_matrix_support"]["row_self_abs_sum"]
        == 1.0e-8
    )
    assert (
        vms_sample["operator_matrix_support"]["row_self_signed_abs_ratio"]
        == 0.1
    )
    pgrad_sample = next(
        row
        for row in operator_samples
        if row["op"] == "equations_diagnostic_ns_vms_pspg_pressure_gradient"
    )
    assert (
        pgrad_sample["operator_matrix_support"]["row_self_abs_sum"]
        == 1.0e-8
    )
    assert pgrad_sample["operator_matrix_support"]["row_coupling_abs_sum"] == 0
    flux_sample = next(
        row
        for row in operator_samples
        if row["op"]
        == "equations_diagnostic_ns_vms_pspg_boundary_pressure_flux"
    )
    assert (
        flux_sample["operator_matrix_support"]["row_self_abs_sum"]
        == 4.0e-9
    )
    assert flux_sample["operator_matrix_support"]["row_coupling_abs_sum"] == 0
    tangential_sample = next(
        row
        for row in operator_samples
        if row["op"]
        == "equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient"
    )
    assert (
        tangential_sample["operator_matrix_support"]["row_self_abs_sum"]
        == 6.0e-9
    )
    assert (
        tangential_sample["operator_matrix_support"]["row_coupling_abs_sum"]
        == 0
    )
    tangential_momentum_sample = next(
        row
        for row in operator_samples
        if row["op"]
        == "equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual"
    )
    assert (
        tangential_momentum_sample["operator_matrix_support"]["row_self_abs_sum"]
        == 6.0e-9
    )
    assert (
        tangential_momentum_sample["operator_matrix_support"][
            "row_coupling_abs_sum"
        ]
        == 2.0e-4
    )
    ghost_sample = next(
        row
        for row in operator_samples
        if row["op"] == "equations_diagnostic_ns_pressure_ghost_penalty"
    )
    assert ghost_sample["operator_matrix_support"]["row_self_abs_sum"] == 0
    support_rank = report["latest_support_rank_diagnostic"]["values"]
    assert support_rank["unconstrained_pressure_rows"] == 7
    assert support_rank["zero_coupling_row_block_count"] == 2
    assert support_rank["positive_coupling_row_block_count"] == 5
    assert support_rank["positive_self_row_block_count"] == 6
    assert support_rank["min_positive_coupling_row_abs_sum"] == 3.0e-4
    assert support_rank["min_positive_self_row_abs_sum"] == 2.0e-8
    assert support_rank["max_self_row_abs_sum"] == 3.0e-1
    assert support_rank["pressure_only_row_block_count"] == 1
    assert support_rank["zero_coupling_row_local_dofs"] == "2|5"
    assert support_rank["weakest_coupling_row_local_dofs"] == 7
    assert support_rank["weakest_self_row_local_dofs"] == 8
    clamp = report["latest_support_rank_clamp"]["values"]
    assert clamp["clamped_row_count"] == 2
    assert clamp["clamp_coupling_threshold"] == 1.0e-14
    assert clamp["clamp_self_threshold"] == 1.0e-8
    assert clamp["positive_coupling_row_block_count"] == 5
    assert clamp["positive_self_row_block_count"] == 6
    assert clamp["min_positive_self_row_abs_sum"] == 2.0e-8
    assert clamp["clamped_local_dofs"] == "2|5"
    first_graph = report["pressure_graph_completions"][0]["values"]
    assert first_graph["mode"] == "shared_velocity_neighbor"
    assert first_graph["requested_mode"] == "shared_velocity_neighbor"
    assert first_graph["candidate_row_count"] == 2
    assert first_graph["neighbor_row_count"] == 1
    assert first_graph["edge_count"] == 1
    assert first_graph["edge_weight"] == 5.0e-9
    assert (
        first_graph["edge_weight_rule"]
        == "min_positive_candidate_diagonal_to_shared_velocity_neighbor"
    )
    assert (
        first_graph["neighbor_policy"]
        == "max_shared_velocity_signature_then_row_support"
    )
    assert first_graph["applied"] == 1
    assert first_graph["candidate_global_dofs"] == "12|15"
    assert first_graph["neighbor_global_dofs"] == 18
    second_graph = report["pressure_graph_completions"][1]["values"]
    assert second_graph["mode"] == "shared_pressure_neighbor"
    assert second_graph["requested_mode"] == "shared_pressure_neighbor"
    assert second_graph["candidate_row_count"] == 2
    assert second_graph["neighbor_row_count"] == 2
    assert second_graph["edge_count"] == 1
    assert second_graph["edge_weight"] == 5.0e-9
    assert (
        second_graph["edge_weight_rule"]
        == "min_positive_candidate_diagonal_to_shared_pressure_neighbor_pair"
    )
    assert second_graph["neighbor_policy"] == (
        "candidate_pair_with_max_shared_pressure_neighbor_support"
    )
    assert second_graph["min_completion_edge_weight"] == 5.0e-9
    assert second_graph["max_completion_edge_weight"] == 5.0e-9
    assert second_graph["min_completion_edge_scale"] == 1
    assert second_graph["max_completion_edge_scale"] == 1
    assert second_graph["non_laplacian_existing_edge_count"] == 0
    assert second_graph["applied"] == 1
    assert second_graph["candidate_global_dofs"] == "12|15"
    assert second_graph["neighbor_global_dofs"] == "18|19"
    support_graph = report["pressure_graph_completions"][2]["values"]
    assert support_graph["mode"] == "existing_support_balance"
    assert support_graph["requested_mode"] == "existing_support_balance"
    assert support_graph["candidate_row_count"] == 2
    assert support_graph["zero_coupling_candidate_count"] == 1
    assert support_graph["weak_coupling_candidate_count"] == 1
    assert support_graph["zero_self_candidate_count"] == 0
    assert support_graph["weak_self_candidate_count"] == 2
    assert support_graph["weak_coupling_and_self_candidate_count"] == 2
    assert support_graph["neighbor_row_count"] == 4
    assert support_graph["edge_count"] == 3
    assert support_graph["edge_weight"] == 9.0e-9
    assert (
        support_graph["edge_weight_rule"]
        == "existing_pressure_edges_abs_scaled_to_target_self_row_abs_sum"
    )
    assert support_graph["neighbor_policy"] == (
        "all_existing_pressure_edges_incident_to_weak_rows"
    )
    assert support_graph["non_laplacian_existing_edge_count"] == 2
    assert support_graph["candidate_with_existing_pressure_edge_count"] == 2
    assert support_graph["candidate_with_laplacian_pressure_edge_count"] == 1
    assert support_graph["candidate_with_non_laplacian_only_pressure_edge_count"] == 1
    active_graph = report["pressure_graph_completions"][3]["values"]
    assert active_graph["mode"] == "active_support_completion"
    assert active_graph["requested_mode"] == "active_support_completion"
    assert active_graph["candidate_row_count"] == 2
    assert active_graph["max_active_neighbors"] == 2
    assert active_graph["neighbor_row_count"] == 4
    assert active_graph["edge_count"] == 4
    assert active_graph["edge_weight"] == 2.5e-9
    assert (
        active_graph["edge_weight_rule"]
        == "min_positive_candidate_diagonal_distributed_to_active_pressure_support"
    )
    assert active_graph["neighbor_policy"] == (
        "strongest_unconstrained_pressure_self_rows"
    )
    assert active_graph["target_self_row_abs_sum"] == 5.0e-9
    assert active_graph["min_completion_edge_weight"] == 2.5e-9
    assert active_graph["max_completion_edge_weight"] == 2.5e-9
    assert active_graph["min_completion_edge_scale"] == 1
    assert active_graph["max_completion_edge_scale"] == 1
    assert active_graph["applied"] == 1
    assert active_graph["candidate_global_dofs"] == "12|15"
    assert active_graph["neighbor_global_dofs"] == "12|15|18|19"
    schur_graph = report["pressure_graph_completions"][4]["values"]
    assert schur_graph["mode"] == "shared_row_schur_completion"
    assert schur_graph["requested_mode"] == "shared_row_schur_completion"
    assert schur_graph["candidate_row_count"] == 2
    assert schur_graph["max_active_neighbors"] == 2
    assert schur_graph["neighbor_row_count"] == 4
    assert schur_graph["edge_count"] == 5
    assert schur_graph["edge_weight"] == 4.0e-9
    assert schur_graph["edge_weight_rule"] == (
        "existing_pressure_laplacian_schur_fill_wi_wj_over_hub_support_sum"
    )
    assert schur_graph["neighbor_policy"] == (
        "weak_candidate_pressure_neighbors_to_shared_row_pressure_neighbors"
    )
    assert schur_graph["min_completion_edge_weight"] == 1.0e-9
    assert schur_graph["max_completion_edge_weight"] == 4.0e-9
    assert schur_graph["shared_row_schur_hub_count"] == 3
    assert schur_graph["shared_row_schur_candidate_edge_count"] == 4
    assert schur_graph["shared_row_schur_contribution_count"] == 7
    assert schur_graph["applied"] == 1
    assert schur_graph["candidate_global_dofs"] == "12|15"
    assert schur_graph["neighbor_global_dofs"] == "12|15|18|19"
    hybrid_graph = report["pressure_graph_completions"][5]["values"]
    assert hybrid_graph["mode"] == "shared_row_schur_existing_edge_balance"
    assert (
        hybrid_graph["requested_mode"]
        == "shared_row_schur_existing_edge_balance"
    )
    assert hybrid_graph["candidate_row_count"] == 2
    assert hybrid_graph["max_active_neighbors"] == 2
    assert hybrid_graph["neighbor_row_count"] == 4
    assert hybrid_graph["edge_count"] == 7
    assert hybrid_graph["edge_weight"] == 9.0e-9
    assert hybrid_graph["edge_weight_rule"] == (
        "shared_row_schur_completion_then_existing_pressure_laplacian_edges_"
        "scaled_to_target_self_row_abs_sum"
    )
    assert hybrid_graph["neighbor_policy"] == (
        "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_"
        "incident_to_weak_rows"
    )
    assert hybrid_graph["target_self_row_abs_sum"] == 3.0e-1
    assert hybrid_graph["min_completion_edge_weight"] == 1.0e-9
    assert hybrid_graph["max_completion_edge_weight"] == 9.0e-9
    assert hybrid_graph["min_completion_edge_scale"] == 1
    assert hybrid_graph["max_completion_edge_scale"] == 12
    assert hybrid_graph["candidate_with_existing_pressure_edge_count"] == 2
    assert hybrid_graph["candidate_with_laplacian_pressure_edge_count"] == 2
    assert (
        hybrid_graph["candidate_with_non_laplacian_only_pressure_edge_count"]
        == 0
    )
    assert hybrid_graph["shared_row_schur_hub_count"] == 3
    assert hybrid_graph["shared_row_schur_candidate_edge_count"] == 4
    assert hybrid_graph["shared_row_schur_contribution_count"] == 7
    assert hybrid_graph["shared_row_schur_edge_count"] == 5
    assert hybrid_graph["existing_balance_edge_count"] == 2
    assert hybrid_graph["applied"] == 1
    assert hybrid_graph["candidate_global_dofs"] == "12|15"
    assert hybrid_graph["neighbor_global_dofs"] == "12|15|18|19"
    graph = report["latest_pressure_graph_completion"]["values"]
    assert graph["mode"] == "existing_edge_balance"
    assert graph["requested_mode"] == "existing_edge_balance"
    assert graph["candidate_row_count"] == 2
    assert graph["neighbor_row_count"] == 3
    assert graph["edge_count"] == 2
    assert graph["edge_weight"] == 8.0e-9
    assert (
        graph["edge_weight_rule"]
        == "existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum"
    )
    assert graph["neighbor_policy"] == (
        "existing_pressure_edges_incident_to_weak_rows"
    )
    assert graph["max_edge_scale_cap"] == 16
    assert graph["target_self_row_abs_sum"] == 3.0e-1
    assert graph["min_completion_edge_weight"] == 4.0e-9
    assert graph["max_completion_edge_weight"] == 8.0e-9
    assert graph["min_completion_edge_scale"] == 3
    assert graph["max_completion_edge_scale"] == 15
    assert graph["non_laplacian_existing_edge_count"] == 1
    assert graph["applied"] == 1
    assert graph["candidate_global_dofs"] == "12|15"
    assert graph["neighbor_global_dofs"] == "12|15|18"
    update_support = report["latest_pressure_update_support_diagnostic"]["values"]
    assert update_support["weak_coupling_threshold"] == 3.3e-4
    assert update_support["weak_self_threshold"] == 1.0e-7
    assert update_support["same_sign_pressure_action_top_edge_count"] == 1
    assert update_support["same_sign_pressure_action_component_count"] == 1
    assert update_support["same_sign_pressure_action_largest_component_size"] == 2
    assert update_support["same_sign_pressure_action_covered_top_update_count"] == 2
    assert update_support["same_sign_pressure_action_isolated_top_update_count"] == 1
    assert update_support[
        "same_sign_pressure_action_largest_component_has_max_update"
    ] == 1
    assert (
        update_support["same_sign_pressure_action_covered_global_dofs"] == "18|19"
    )
    assert update_support["same_sign_pressure_action_isolated_global_dofs"] == 21
    assert (
        update_support["same_sign_pressure_action_largest_component_global_dofs"]
        == "18|19"
    )
    assert update_support["weak_coupling_row_block_count"] == 1
    assert update_support["weak_self_row_block_count"] == 2
    assert update_support["max_abs_update"] == 42
    assert update_support["max_update_local_dof"] == 8
    assert update_support["max_update_rhs"] == 1.25
    assert update_support["action_sample_limit"] == 4
    assert update_support["max_update_row_coupling_action"] == 0.75
    assert update_support["max_update_row_self_action"] == 0.5
    assert update_support["max_update_row_self_constant_action"] == 0.25
    assert update_support["max_update_row_self_nonconstant_action"] == 0.25
    assert update_support["max_update_row_linear_residual"] == 1.0e-12
    assert update_support["weak_self_max_abs_update"] == 42
    assert (
        update_support["top_update_details"]
        == "8:18:update=-42:abs_update=42:rhs=1.25:"
        "row_action=1.250000000001:row_coupling_action=0.75:"
        "row_self_action=0.5:row_self_constant_action=0.25:"
        "row_self_nonconstant_action=0.25:"
        "row_other_action=0.000000000001:"
        "row_linear_residual=1e-12:row=1.1:"
        "row_coupling=3e-4:row_self=1e-8:"
        "row_self_sum=-1e-9:row_self_offdiag=5e-9:"
        "row_self_signed_abs_ratio=0.1:row_self_diag_abs_ratio=0.5:"
        "col=2:col_coupling=6e-4:col_self=2e-8:diag=5e-9:"
        "pressure_action_terms=8/18/m=0.5/u=1/a=0.5:"
        "coupling_action_terms=0/20/m=0.75/u=1/a=0.75"
    )
    update_summary = report["pressure_update_support_summary"]
    assert update_summary["diagnostic_count"] == 1
    assert update_summary["phase"] == "post_linear_solve_update"
    assert update_summary["parsed_top_update_count"] == 1
    assert update_summary["max_update_local_dof"] == 8
    assert update_summary["max_update_global_dof"] == 18
    assert update_summary["max_update_detail"]["local_pressure_row"] == 8
    assert update_summary["max_update_detail"]["global_dof"] == 18
    assert update_summary["max_update_detail"]["row_self"] == 1.0e-8
    assert update_summary["max_update_detail"]["diag"] == 5.0e-9
    assert update_summary["top_update_details"] == [
        {
            "local_pressure_row": 8,
            "global_dof": 18,
            "update": -42.0,
            "abs_update": 42.0,
            "rhs": 1.25,
            "row_action": 1.250000000001,
            "row_coupling_action": 0.75,
            "row_self_action": 0.5,
            "row_self_constant_action": 0.25,
            "row_self_nonconstant_action": 0.25,
            "row_other_action": 1.0e-12,
            "row_linear_residual": 1.0e-12,
            "row": 1.1,
            "row_coupling": 3.0e-4,
            "row_self": 1.0e-8,
            "row_self_sum": -1.0e-9,
            "row_self_offdiag": 5.0e-9,
            "row_self_signed_abs_ratio": 0.1,
            "row_self_diag_abs_ratio": 0.5,
            "col": 2.0,
            "col_coupling": 6.0e-4,
            "col_self": 2.0e-8,
            "diag": 5.0e-9,
            "pressure_action_terms": "8/18/m=0.5/u=1/a=0.5",
            "coupling_action_terms": "0/20/m=0.75/u=1/a=0.75",
        }
    ]
    assert update_summary["max_update_diag_action_abs"] == 2.1e-7
    assert update_summary["max_update_self_to_coupling_action_ratio"] == (
        0.5 / 0.75
    )
    assert update_summary["max_update_constant_self_action_fraction"] == 0.5
    assert update_summary["max_update_nonconstant_self_action_fraction"] == 0.5
    assert update_summary["same_sign_pressure_action_top_edge_count"] == 1
    assert update_summary["same_sign_pressure_action_component_count"] == 1
    assert update_summary["same_sign_pressure_action_largest_component_size"] == 2
    assert update_summary["same_sign_pressure_action_covered_top_update_count"] == 2
    assert update_summary["same_sign_pressure_action_isolated_top_update_count"] == 1
    assert (
        update_summary["same_sign_pressure_action_largest_component_has_max_update"]
        == 1
    )
    assert update_summary["same_sign_pressure_action_covered_global_dofs"] == [
        18,
        19,
    ]
    assert update_summary["same_sign_pressure_action_isolated_global_dofs"] == [21]
    assert update_summary[
        "same_sign_pressure_action_largest_component_global_dofs"
    ] == [18, 19]


def test_graph_completion_parser_preserves_all_row_selector_fields(tmp_path):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_existing_edge_balance_all' "
        "requested_mode='shared_row_schur_existing_edge_balance_all' "
        "coupling_threshold=1e-14 self_threshold=1e-8 "
        "max_rows=512 max_rows_applied=0 "
        "candidate_selector='all_unconstrained_pressure_rows' "
        "support_rank_candidate_row_count=2 max_active_neighbors=64 "
        "candidate_row_count=7 zero_coupling_candidate_count=1 "
        "weak_coupling_candidate_count=2 zero_self_candidate_count=0 "
        "weak_self_candidate_count=3 weak_coupling_and_self_candidate_count=2 "
        "neighbor_row_count=7 edge_count=23 edge_weight=9e-9 "
        "edge_weight_rule='all_pressure_shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum' "
        "neighbor_policy='all_unconstrained_pressure_rows_schur_fill_then_existing_laplacian_edges_incident_to_all_pressure_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=9e-9 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=12 "
        "non_laplacian_existing_edge_count=1 "
        "candidate_with_existing_pressure_edge_count=7 "
        "candidate_with_laplacian_pressure_edge_count=6 "
        "candidate_with_non_laplacian_only_pressure_edge_count=1 "
        "shared_row_schur_hub_count=7 "
        "shared_row_schur_candidate_edge_count=42 "
        "shared_row_schur_contribution_count=91 "
        "shared_row_schur_edge_count=19 "
        "existing_balance_edge_count=4 "
        "applied=1 candidate_global_dofs=10|11|12|13|14|15|16 "
        "neighbor_global_dofs=10|11|12|13|14|15|16\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_existing_edge_balance_all"
    assert graph["candidate_selector"] == "all_unconstrained_pressure_rows"
    assert graph["max_rows_applied"] == 0
    assert graph["support_rank_candidate_row_count"] == 2
    assert graph["candidate_row_count"] == 7
    assert graph["edge_count"] == 23
    assert graph["shared_row_schur_edge_count"] == 19
    assert graph["existing_balance_edge_count"] == 4


def test_graph_completion_parser_preserves_pressure_neighborhood_selector_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_existing_edge_balance_neighborhood' "
        "requested_mode='schur_existing_edge_balance_neighborhood' "
        "coupling_threshold=1e-14 self_threshold=-1 "
        "max_rows=512 max_rows_applied=1 "
        "candidate_selector='support_rank_rows_plus_pressure_graph_neighbors' "
        "support_rank_candidate_row_count=9 max_active_neighbors=64 "
        "pressure_neighbor_depth=2 "
        "candidate_row_count=31 zero_coupling_candidate_count=9 "
        "weak_coupling_candidate_count=0 zero_self_candidate_count=0 "
        "weak_self_candidate_count=0 weak_coupling_and_self_candidate_count=0 "
        "neighbor_row_count=52 edge_count=144 edge_weight=1e-3 "
        "edge_weight_rule='support_rank_pressure_neighborhood_shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum' "
        "neighbor_policy='support_rank_rows_plus_strongest_pressure_neighbors_schur_fill_then_existing_laplacian_edges_incident_to_expanded_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=16 "
        "non_laplacian_existing_edge_count=2 "
        "candidate_with_existing_pressure_edge_count=31 "
        "candidate_with_laplacian_pressure_edge_count=30 "
        "candidate_with_non_laplacian_only_pressure_edge_count=1 "
        "shared_row_schur_hub_count=45 "
        "shared_row_schur_candidate_edge_count=88 "
        "shared_row_schur_contribution_count=377 "
        "shared_row_schur_edge_count=102 "
        "existing_balance_edge_count=64 "
        "applied=1 candidate_global_dofs=10|11|12 "
        "neighbor_global_dofs=10|11|12|20|21\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == (
        "shared_row_schur_existing_edge_balance_neighborhood"
    )
    assert graph["requested_mode"] == "schur_existing_edge_balance_neighborhood"
    assert graph["candidate_selector"] == (
        "support_rank_rows_plus_pressure_graph_neighbors"
    )
    assert graph["pressure_neighbor_depth"] == 2
    assert graph["support_rank_candidate_row_count"] == 9
    assert graph["candidate_row_count"] == 31
    assert graph["edge_count"] == 144
    assert graph["shared_row_schur_edge_count"] == 102
    assert graph["existing_balance_edge_count"] == 64


def test_graph_completion_parser_preserves_coupling_edge_balance_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_coupling_edge_balance' "
        "requested_mode='schur-coupling-edge-balance' "
        "coupling_threshold=1e-8 self_threshold=1e-7 "
        "max_rows=512 max_rows_applied=1 "
        "candidate_selector='support_rank_zero_or_weak_rows' "
        "support_rank_candidate_row_count=12 max_active_neighbors=64 "
        "pressure_neighbor_depth=0 "
        "candidate_row_count=12 zero_coupling_candidate_count=2 "
        "weak_coupling_candidate_count=4 zero_self_candidate_count=0 "
        "weak_self_candidate_count=8 weak_coupling_and_self_candidate_count=3 "
        "coupling_deficient_balance_candidate_count=6 "
        "balance_candidate_row_count=4 "
        "coupling_deficient_balance_candidate_global_dofs=10|11|12|13|14|15 "
        "low_degree_balance_candidate_global_dofs=none "
        "balance_candidate_global_dofs=10|11|12|13 "
        "neighbor_row_count=18 edge_count=55 edge_weight=1e-3 "
        "edge_weight_rule='shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_coupling_deficient_candidates' "
        "neighbor_policy='weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_coupling_deficient_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=8 "
        "non_laplacian_existing_edge_count=1 "
        "candidate_with_existing_pressure_edge_count=4 "
        "candidate_with_laplacian_pressure_edge_count=4 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=14 "
        "shared_row_schur_candidate_edge_count=22 "
        "shared_row_schur_contribution_count=91 "
        "shared_row_schur_edge_count=47 "
        "existing_balance_edge_count=8 "
        "applied=1 candidate_global_dofs=10|11|12 "
        "neighbor_global_dofs=10|11|12|20|21\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_coupling_edge_balance"
    assert graph["requested_mode"] == "schur-coupling-edge-balance"
    assert graph["candidate_selector"] == "support_rank_zero_or_weak_rows"
    assert graph["candidate_row_count"] == 12
    assert graph["coupling_deficient_balance_candidate_count"] == 6
    assert graph["balance_candidate_row_count"] == 4
    assert (
        graph["coupling_deficient_balance_candidate_global_dofs"]
        == "10|11|12|13|14|15"
    )
    assert graph["low_degree_balance_candidate_global_dofs"] == "none"
    assert graph["balance_candidate_global_dofs"] == "10|11|12|13"
    assert graph["edge_weight_rule"] == (
        "shared_row_schur_completion_then_existing_pressure_laplacian_edges_"
        "scaled_to_target_self_row_abs_sum_for_coupling_deficient_candidates"
    )
    assert graph["neighbor_policy"] == (
        "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_"
        "incident_to_coupling_deficient_rows"
    )
    assert graph["shared_row_schur_edge_count"] == 47
    assert graph["existing_balance_edge_count"] == 8


def test_graph_completion_parser_preserves_low_degree_edge_balance_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_low_degree_edge_balance' "
        "requested_mode='schur-low-degree-edge-balance' "
        "coupling_threshold=1e-8 self_threshold=1e-7 "
        "max_rows=512 max_rows_applied=1 "
        "candidate_selector='support_rank_zero_or_weak_rows' "
        "support_rank_candidate_row_count=12 max_active_neighbors=64 "
        "pressure_neighbor_depth=0 "
        "candidate_row_count=12 zero_coupling_candidate_count=2 "
        "weak_coupling_candidate_count=4 zero_self_candidate_count=0 "
        "weak_self_candidate_count=8 weak_coupling_and_self_candidate_count=3 "
        "coupling_deficient_balance_candidate_count=6 "
        "low_degree_balance_candidate_count=5 "
        "balance_candidate_row_count=4 "
        "coupling_deficient_balance_candidate_global_dofs=10|11|12|13|14|15 "
        "low_degree_balance_candidate_global_dofs=10|11|12|13|14 "
        "balance_candidate_global_dofs=10|11|12|13 "
        "max_balance_pressure_edge_degree=3 "
        "min_candidate_pressure_edge_degree=2 "
        "max_candidate_pressure_edge_degree=9 "
        "neighbor_row_count=18 edge_count=55 edge_weight=1e-3 "
        "edge_weight_rule='shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_low_degree_pressure_candidates' "
        "neighbor_policy='weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_low_degree_pressure_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=8 "
        "non_laplacian_existing_edge_count=1 "
        "candidate_with_existing_pressure_edge_count=4 "
        "candidate_with_laplacian_pressure_edge_count=4 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=14 "
        "shared_row_schur_candidate_edge_count=22 "
        "shared_row_schur_contribution_count=91 "
        "shared_row_schur_edge_count=47 "
        "existing_balance_edge_count=8 "
        "applied=1 candidate_global_dofs=10|11|12 "
        "neighbor_global_dofs=10|11|12|20|21\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_low_degree_edge_balance"
    assert graph["requested_mode"] == "schur-low-degree-edge-balance"
    assert graph["low_degree_balance_candidate_count"] == 5
    assert graph["balance_candidate_row_count"] == 4
    assert (
        graph["coupling_deficient_balance_candidate_global_dofs"]
        == "10|11|12|13|14|15"
    )
    assert graph["low_degree_balance_candidate_global_dofs"] == "10|11|12|13|14"
    assert graph["balance_candidate_global_dofs"] == "10|11|12|13"
    assert graph["max_balance_pressure_edge_degree"] == 3
    assert graph["min_candidate_pressure_edge_degree"] == 2
    assert graph["max_candidate_pressure_edge_degree"] == 9
    assert graph["edge_weight_rule"] == (
        "shared_row_schur_completion_then_existing_pressure_laplacian_edges_"
        "scaled_to_target_self_row_abs_sum_for_low_degree_pressure_candidates"
    )
    assert graph["neighbor_policy"] == (
        "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_"
        "incident_to_low_degree_pressure_rows"
    )
    assert graph["shared_row_schur_edge_count"] == 47
    assert graph["existing_balance_edge_count"] == 8


def test_graph_completion_parser_preserves_explicit_edge_balance_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_explicit_edge_balance' "
        "requested_mode='schur-explicit-edge-balance' "
        "coupling_threshold=1e-8 self_threshold=1e-7 "
        "max_rows=512 max_rows_applied=1 "
        "candidate_selector='support_rank_zero_or_weak_rows_plus_explicit_balance_rows' "
        "support_rank_candidate_row_count=12 max_active_neighbors=64 "
        "pressure_neighbor_depth=0 "
        "candidate_row_count=14 zero_coupling_candidate_count=2 "
        "weak_coupling_candidate_count=4 zero_self_candidate_count=0 "
        "weak_self_candidate_count=8 weak_coupling_and_self_candidate_count=3 "
        "coupling_deficient_balance_candidate_count=6 "
        "low_degree_balance_candidate_count=0 "
        "explicit_balance_candidate_count=2 "
        "balance_candidate_row_count=2 "
        "coupling_deficient_balance_candidate_global_dofs=10|11|12|13|14|15 "
        "low_degree_balance_candidate_global_dofs=none "
        "explicit_balance_requested_global_dofs=18|22|999 "
        "explicit_balance_candidate_global_dofs=18|22 "
        "balance_candidate_global_dofs=18|22 "
        "max_balance_pressure_edge_degree=3 "
        "min_candidate_pressure_edge_degree=0 "
        "max_candidate_pressure_edge_degree=0 "
        "neighbor_row_count=18 edge_count=55 edge_weight=1e-3 "
        "edge_weight_rule='shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_explicit_balance_rows' "
        "neighbor_policy='weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_explicit_balance_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=8 "
        "non_laplacian_existing_edge_count=1 "
        "candidate_with_existing_pressure_edge_count=2 "
        "candidate_with_laplacian_pressure_edge_count=2 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=14 "
        "shared_row_schur_candidate_edge_count=22 "
        "shared_row_schur_contribution_count=91 "
        "shared_row_schur_edge_count=47 "
        "existing_balance_edge_count=8 "
        "applied=1 candidate_global_dofs=10|11|12|18|22 "
        "neighbor_global_dofs=10|11|12|18|20|21|22\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_explicit_edge_balance"
    assert graph["requested_mode"] == "schur-explicit-edge-balance"
    assert graph["candidate_selector"] == (
        "support_rank_zero_or_weak_rows_plus_explicit_balance_rows"
    )
    assert graph["candidate_row_count"] == 14
    assert graph["explicit_balance_candidate_count"] == 2
    assert graph["balance_candidate_row_count"] == 2
    assert graph["explicit_balance_requested_global_dofs"] == "18|22|999"
    assert graph["explicit_balance_candidate_global_dofs"] == "18|22"
    assert graph["balance_candidate_global_dofs"] == "18|22"
    assert graph["edge_weight_rule"] == (
        "shared_row_schur_completion_then_existing_pressure_laplacian_edges_"
        "scaled_to_target_self_row_abs_sum_for_explicit_balance_rows"
    )
    assert graph["neighbor_policy"] == (
        "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_"
        "incident_to_explicit_balance_rows"
    )
    assert graph["shared_row_schur_edge_count"] == 47
    assert graph["existing_balance_edge_count"] == 8


def test_graph_completion_parser_preserves_explicit_neighborhood_balance_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_explicit_neighborhood_edge_balance' "
        "requested_mode='schur-explicit-neighborhood-edge-balance' "
        "coupling_threshold=1e-8 self_threshold=1e-7 "
        "max_rows=512 max_rows_applied=1 "
        "candidate_selector='support_rank_zero_or_weak_rows_plus_explicit_balance_neighborhood_rows' "
        "support_rank_candidate_row_count=12 max_active_neighbors=4 "
        "pressure_neighbor_depth=2 "
        "candidate_row_count=18 zero_coupling_candidate_count=2 "
        "weak_coupling_candidate_count=4 zero_self_candidate_count=0 "
        "weak_self_candidate_count=8 weak_coupling_and_self_candidate_count=3 "
        "coupling_deficient_balance_candidate_count=6 "
        "low_degree_balance_candidate_count=0 "
        "explicit_balance_candidate_count=6 "
        "balance_candidate_row_count=6 "
        "coupling_deficient_balance_candidate_global_dofs=10|11|12|13|14|15 "
        "low_degree_balance_candidate_global_dofs=none "
        "explicit_balance_requested_global_dofs=18|22 "
        "explicit_balance_candidate_global_dofs=16|17|18|20|21|22 "
        "balance_candidate_global_dofs=16|17|18|20|21|22 "
        "max_balance_pressure_edge_degree=3 "
        "min_candidate_pressure_edge_degree=0 "
        "max_candidate_pressure_edge_degree=0 "
        "neighbor_row_count=22 edge_count=75 edge_weight=1e-3 "
        "edge_weight_rule='shared_row_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_explicit_balance_neighborhood_rows' "
        "neighbor_policy='weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_incident_to_explicit_balance_neighborhood_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=8 "
        "non_laplacian_existing_edge_count=1 "
        "candidate_with_existing_pressure_edge_count=6 "
        "candidate_with_laplacian_pressure_edge_count=6 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=14 "
        "shared_row_schur_candidate_edge_count=22 "
        "shared_row_schur_contribution_count=91 "
        "shared_row_schur_edge_count=47 "
        "existing_balance_edge_count=28 "
        "applied=1 candidate_global_dofs=10|11|12|16|17|18|20|21|22 "
        "neighbor_global_dofs=10|11|12|16|17|18|20|21|22\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_explicit_neighborhood_edge_balance"
    assert graph["requested_mode"] == "schur-explicit-neighborhood-edge-balance"
    assert graph["candidate_selector"] == (
        "support_rank_zero_or_weak_rows_plus_explicit_balance_neighborhood_rows"
    )
    assert graph["pressure_neighbor_depth"] == 2
    assert graph["explicit_balance_candidate_count"] == 6
    assert graph["balance_candidate_row_count"] == 6
    assert graph["explicit_balance_requested_global_dofs"] == "18|22"
    assert graph["explicit_balance_candidate_global_dofs"] == (
        "16|17|18|20|21|22"
    )
    assert graph["edge_weight_rule"] == (
        "shared_row_schur_completion_then_existing_pressure_laplacian_edges_"
        "scaled_to_target_self_row_abs_sum_for_explicit_balance_neighborhood_rows"
    )
    assert graph["neighbor_policy"] == (
        "weak_candidate_pressure_schur_fill_then_existing_laplacian_edges_"
        "incident_to_explicit_balance_neighborhood_rows"
    )
    assert graph["existing_balance_edge_count"] == 28


def test_graph_completion_parser_preserves_support_gap_patch_balance_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_support_gap_patch_edge_balance' "
        "requested_mode='schur-support-gap-patch-edge-balance' "
        "coupling_threshold=1e-8 self_threshold=-1 "
        "max_rows=512 max_rows_applied=1 "
        "candidate_selector='pressure_self_support_gap_rows_plus_pressure_graph_patch' "
        "support_rank_candidate_row_count=12 max_active_neighbors=64 "
        "pressure_neighbor_depth=0 "
        "candidate_row_count=9 zero_coupling_candidate_count=1 "
        "weak_coupling_candidate_count=2 zero_self_candidate_count=0 "
        "weak_self_candidate_count=0 weak_coupling_and_self_candidate_count=0 "
        "coupling_deficient_balance_candidate_count=3 "
        "support_gap_candidate_count=2 "
        "support_gap_patch_candidate_count=9 "
        "support_gap_self_threshold=0.125 "
        "support_gap_self_threshold_source='median_positive_pressure_self_row_abs_sum' "
        "support_gap_patch_truncated=0 "
        "low_degree_balance_candidate_count=0 "
        "explicit_balance_candidate_count=0 "
        "balance_candidate_row_count=2 "
        "coupling_deficient_balance_candidate_global_dofs=30|31|32 "
        "support_gap_candidate_global_dofs=40|44 "
        "support_gap_patch_candidate_global_dofs=30|31|32|40|41|42|43|44|45 "
        "low_degree_balance_candidate_global_dofs=none "
        "explicit_balance_requested_global_dofs=none "
        "explicit_balance_candidate_global_dofs=none "
        "balance_candidate_global_dofs=40|44 "
        "max_balance_pressure_edge_degree=3 "
        "min_candidate_pressure_edge_degree=0 "
        "max_candidate_pressure_edge_degree=0 "
        "neighbor_row_count=9 edge_count=31 edge_weight=1e-3 "
        "edge_weight_rule='support_gap_pressure_patch_schur_completion_then_existing_pressure_laplacian_edges_scaled_to_target_self_row_abs_sum_for_support_gap_rows' "
        "neighbor_policy='support_gap_pressure_patch_schur_fill_then_existing_laplacian_edges_incident_to_support_gap_rows' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=3e-1 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=8 "
        "non_laplacian_existing_edge_count=0 "
        "candidate_with_existing_pressure_edge_count=2 "
        "candidate_with_laplacian_pressure_edge_count=2 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=7 "
        "shared_row_schur_candidate_edge_count=18 "
        "shared_row_schur_contribution_count=62 "
        "shared_row_schur_edge_count=29 "
        "existing_balance_edge_count=7 "
        "applied=1 candidate_global_dofs=30|31|32|40|41|42|43|44|45 "
        "neighbor_global_dofs=30|31|32|40|41|42|43|44|45\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_support_gap_patch_edge_balance"
    assert graph["requested_mode"] == "schur-support-gap-patch-edge-balance"
    assert graph["candidate_selector"] == (
        "pressure_self_support_gap_rows_plus_pressure_graph_patch"
    )
    assert graph["candidate_row_count"] == 9
    assert graph["support_gap_candidate_count"] == 2
    assert graph["support_gap_patch_candidate_count"] == 9
    assert graph["support_gap_self_threshold"] == 0.125
    assert (
        graph["support_gap_self_threshold_source"]
        == "median_positive_pressure_self_row_abs_sum"
    )
    assert graph["support_gap_patch_truncated"] == 0
    assert graph["support_gap_candidate_global_dofs"] == "40|44"
    assert graph["support_gap_patch_candidate_global_dofs"] == (
        "30|31|32|40|41|42|43|44|45"
    )
    assert graph["balance_candidate_global_dofs"] == "40|44"
    assert graph["edge_weight_rule"] == (
        "support_gap_pressure_patch_schur_completion_then_existing_pressure_"
        "laplacian_edges_scaled_to_target_self_row_abs_sum_for_support_gap_rows"
    )
    assert graph["neighbor_policy"] == (
        "support_gap_pressure_patch_schur_fill_then_existing_laplacian_edges_"
        "incident_to_support_gap_rows"
    )
    assert graph["shared_row_schur_edge_count"] == 29
    assert graph["existing_balance_edge_count"] == 7


def test_graph_completion_parser_preserves_support_gap_patch_schur_only_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_support_gap_patch_completion' "
        "requested_mode='schur-support-gap-patch' "
        "coupling_threshold=1e-8 self_threshold=-1 "
        "max_rows=-1 max_rows_applied=1 "
        "candidate_selector='pressure_self_support_gap_rows_plus_pressure_graph_patch' "
        "support_rank_candidate_row_count=0 max_active_neighbors=64 "
        "pressure_neighbor_depth=0 "
        "candidate_row_count=9 zero_coupling_candidate_count=0 "
        "weak_coupling_candidate_count=0 zero_self_candidate_count=0 "
        "weak_self_candidate_count=0 weak_coupling_and_self_candidate_count=0 "
        "coupling_deficient_balance_candidate_count=0 "
        "support_gap_candidate_count=2 "
        "support_gap_patch_candidate_count=9 "
        "support_gap_self_threshold=0.125 "
        "support_gap_self_threshold_source='median_positive_pressure_self_row_abs_sum' "
        "support_gap_patch_truncated=0 "
        "low_degree_balance_candidate_count=0 "
        "explicit_balance_candidate_count=0 "
        "balance_candidate_row_count=0 "
        "coupling_deficient_balance_candidate_global_dofs=none "
        "support_gap_candidate_global_dofs=40|44 "
        "support_gap_patch_candidate_global_dofs=30|31|32|40|41|42|43|44|45 "
        "low_degree_balance_candidate_global_dofs=none "
        "explicit_balance_requested_global_dofs=none "
        "explicit_balance_candidate_global_dofs=none "
        "balance_candidate_global_dofs=none "
        "max_balance_pressure_edge_degree=3 "
        "min_candidate_pressure_edge_degree=0 "
        "max_candidate_pressure_edge_degree=0 "
        "neighbor_row_count=9 edge_count=29 edge_weight=1e-3 "
        "edge_weight_rule='support_gap_pressure_patch_schur_completion_wi_wj_over_hub_support_sum' "
        "neighbor_policy='support_gap_pressure_patch_to_shared_row_pressure_neighbors' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=0 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=1 "
        "non_laplacian_existing_edge_count=0 "
        "candidate_with_existing_pressure_edge_count=0 "
        "candidate_with_laplacian_pressure_edge_count=0 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=7 "
        "shared_row_schur_candidate_edge_count=18 "
        "shared_row_schur_contribution_count=62 "
        "shared_row_schur_edge_count=29 "
        "existing_balance_edge_count=0 "
        "applied=1 candidate_global_dofs=30|31|32|40|41|42|43|44|45 "
        "neighbor_global_dofs=30|31|32|40|41|42|43|44|45\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_support_gap_patch_completion"
    assert graph["requested_mode"] == "schur-support-gap-patch"
    assert graph["candidate_selector"] == (
        "pressure_self_support_gap_rows_plus_pressure_graph_patch"
    )
    assert graph["candidate_row_count"] == 9
    assert graph["support_gap_candidate_count"] == 2
    assert graph["support_gap_patch_candidate_count"] == 9
    assert graph["balance_candidate_row_count"] == 0
    assert graph["balance_candidate_global_dofs"] == "none"
    assert graph["edge_weight_rule"] == (
        "support_gap_pressure_patch_schur_completion_wi_wj_over_hub_support_sum"
    )
    assert graph["neighbor_policy"] == (
        "support_gap_pressure_patch_to_shared_row_pressure_neighbors"
    )
    assert graph["shared_row_schur_edge_count"] == 29
    assert graph["existing_balance_edge_count"] == 0


def test_graph_completion_parser_preserves_support_gap_local_patch_fields(
    tmp_path,
):
    audit = _load_audit_module()
    log = tmp_path / "run.log"
    log.write_text(
        "[R0] [INFO] NewtonSolver: active pressure graph completion "
        "diagnostic=active_pressure_graph_completion rank=0 "
        "iteration=0 phase='pre_linear_solve' backend=eigen "
        "solve_time=1 dt=0.1 pressure_field='Pressure' "
        "coupling_field='Velocity' "
        "mode='shared_row_schur_support_gap_local_patch_completion' "
        "requested_mode='schur-support-gap-local-patch' "
        "coupling_threshold=1e-8 self_threshold=-1 "
        "max_rows=-1 max_rows_applied=1 "
        "candidate_selector='pressure_self_support_gap_rows_plus_pressure_graph_local_patch' "
        "support_rank_candidate_row_count=0 max_active_neighbors=64 "
        "pressure_neighbor_depth=1 "
        "candidate_row_count=5 zero_coupling_candidate_count=0 "
        "weak_coupling_candidate_count=0 zero_self_candidate_count=0 "
        "weak_self_candidate_count=0 weak_coupling_and_self_candidate_count=0 "
        "coupling_deficient_balance_candidate_count=0 "
        "support_gap_candidate_count=2 "
        "support_gap_patch_candidate_count=5 "
        "support_gap_self_threshold=0.125 "
        "support_gap_self_threshold_source='median_positive_pressure_self_row_abs_sum' "
        "support_gap_patch_truncated=0 "
        "low_degree_balance_candidate_count=0 "
        "explicit_balance_candidate_count=0 "
        "balance_candidate_row_count=0 "
        "coupling_deficient_balance_candidate_global_dofs=none "
        "support_gap_candidate_global_dofs=40|44 "
        "support_gap_patch_candidate_global_dofs=40|41|43|44|45 "
        "low_degree_balance_candidate_global_dofs=none "
        "explicit_balance_requested_global_dofs=none "
        "explicit_balance_candidate_global_dofs=none "
        "balance_candidate_global_dofs=none "
        "max_balance_pressure_edge_degree=3 "
        "min_candidate_pressure_edge_degree=0 "
        "max_candidate_pressure_edge_degree=0 "
        "neighbor_row_count=5 edge_count=11 edge_weight=1e-3 "
        "edge_weight_rule='support_gap_local_pressure_patch_schur_completion_wi_wj_over_hub_support_sum' "
        "neighbor_policy='support_gap_local_pressure_patch_to_shared_row_pressure_neighbors' "
        "weight_scale=1 max_edge_scale_cap=16 "
        "min_positive_candidate_diag_abs=5e-9 "
        "target_self_row_abs_sum=0 "
        "min_completion_edge_weight=1e-9 "
        "max_completion_edge_weight=1e-3 "
        "min_completion_edge_scale=1 "
        "max_completion_edge_scale=1 "
        "non_laplacian_existing_edge_count=0 "
        "candidate_with_existing_pressure_edge_count=0 "
        "candidate_with_laplacian_pressure_edge_count=0 "
        "candidate_with_non_laplacian_only_pressure_edge_count=0 "
        "shared_row_schur_hub_count=4 "
        "shared_row_schur_candidate_edge_count=9 "
        "shared_row_schur_contribution_count=20 "
        "shared_row_schur_edge_count=11 "
        "existing_balance_edge_count=0 "
        "applied=1 candidate_global_dofs=40|41|43|44|45 "
        "neighbor_global_dofs=40|41|43|44|45\n",
        encoding="utf-8",
    )

    report = audit.summarize_pressure_matrix_support(solver_log=log)
    graph = report["latest_pressure_graph_completion"]["values"]

    assert graph["mode"] == "shared_row_schur_support_gap_local_patch_completion"
    assert graph["requested_mode"] == "schur-support-gap-local-patch"
    assert graph["candidate_selector"] == (
        "pressure_self_support_gap_rows_plus_pressure_graph_local_patch"
    )
    assert graph["pressure_neighbor_depth"] == 1
    assert graph["candidate_row_count"] == 5
    assert graph["support_gap_candidate_count"] == 2
    assert graph["support_gap_patch_candidate_count"] == 5
    assert graph["support_gap_patch_candidate_global_dofs"] == "40|41|43|44|45"
    assert graph["edge_weight_rule"] == (
        "support_gap_local_pressure_patch_schur_completion_wi_wj_over_hub_"
        "support_sum"
    )
    assert graph["neighbor_policy"] == (
        "support_gap_local_pressure_patch_to_shared_row_pressure_neighbors"
    )
    assert graph["shared_row_schur_edge_count"] == 11
    assert graph["existing_balance_edge_count"] == 0
