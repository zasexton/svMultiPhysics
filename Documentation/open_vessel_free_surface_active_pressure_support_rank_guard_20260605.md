# Open-Vessel Active Pressure Support-Rank Guard

Date: 2026-06-05

Scope: targeted follow-up to the Test10 step90 VMS-enabled accepted pressure jump and VMS-disabled singularity. This checks whether active, unconstrained pressure rows have actual Galerkin velocity-block coupling before the linear solve, provides environment-gated guard/clamp probes, records weak positive coupling rows for the accepted pressure-jump state, confirms whether the bad pressure increments are already present in the proposed Newton update before acceptance, records row-action and pressure-subblock null contributors for the accepted maximum, records operator-level pressure matrix-support attribution including a VMS/PSPG pressure-gradient split, records the local pressure-update neighborhood around the accepted maximum, adds a pre-commit pressure-update rejection gate, rules out a direct generated-interface pressure-reference trace contribution for the accepted max row, and adds synthetic patch evidence for a constant-null topology-completing pressure-support direction.

## Code Change

- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  now logs `diagnostic=active_pressure_support_rank` when `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_DIAGNOSTIC=1` or `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_GUARD=1`.
- The diagnostic scans unconstrained rows of the pressure field before `linear.solve()`, after constraints and explicit rank-one matrix updates have been applied. It reports row/column support split into pressure self-block and velocity coupling-block support.
- `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_GUARD=1` turns the diagnostic into a fail-fast guard when unconstrained pressure rows have zero velocity row-block support. The guard is off by default and is not a default formulation change.
- `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP=1` is a diagnostic-only prototype that zeros the Newton update equation and RHS entries for rows with zero velocity row-block support. It is off by default and is not a validated formulation fix.
- `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP_MAX_VELOCITY_ROW_SUM=<value>` extends that clamp to active pressure rows whose velocity row-block support is nonzero but weak. If omitted, the clamp uses the support-rank tolerance and therefore clamps only zero-coupling rows.
- `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP_MAX_PRESSURE_SELF_ROW_SUM=<value>` extends that clamp to active pressure rows whose pressure self-block row support is weak. `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_CLAMP_MAX_SELF_ROW_SUM` is accepted as an alias. If omitted, pressure self-block support does not affect clamp selection.
- The support-rank diagnostic also logs the weakest positive velocity-coupled pressure rows through `weakest_coupling_row_*` fields. This distinguishes structural zero-coupling rows from weak-but-nonzero pressure coupling.
- `SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_DIAGNOSTIC=1` logs `diagnostic=active_pressure_update_support` after the linear solve and Newton increment scaling, before line search/state update. `SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_WEAK_VELOCITY_ROW_SUM`, `SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_WEAK_PRESSURE_SELF_ROW_SUM`, and `SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_SAMPLE_LIMIT` control the support classes and sample count for the proposed pressure increment. `SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_ACTION_SAMPLE_LIMIT` controls bounded row-action contributor samples.
- The same pressure-update/support diagnostic reports pressure-subblock signed row sums, diagonal/off-diagonal ratios, and constant-pressure versus nonconstant-pressure action at sampled rows.
- `SVMP_PRESSURE_ROW_CONTRIBUTION_MATRIX_DIAGNOSTIC=1` extends the pressure-row contribution diagnostic to assemble each diagnostic operator tangent into a scratch matrix and log `diagnostic=pressure_row_operator_matrix_support` at sampled pressure rows. This attributes sampled pressure self/coupling support to Galerkin continuity, active continuity, VMS/PSPG, or pressure ghost penalty without modifying the solve matrix.
- The pressure-row contribution diagnostic now also exposes `equations_diagnostic_ns_vms_pspg_pressure_gradient` and `equations_diagnostic_ns_vms_pspg_nonpressure`, splitting VMS/PSPG into the direct PSPG `grad(p)` residual part and the remaining nonpressure residual part without changing the production residual.
- `SVMP_NS_PSPG_PRESSURE_GRADIENT_SCALE=<nonnegative>` scales the direct PSPG `grad(p)` continuity term for diagnostic controls. `SVMP_PSPG_PRESSURE_GRADIENT_SCALE` is accepted as a fallback. The default scale is `1`, so normal production behavior is unchanged unless the env var is set.
- `SVMP_NS_PSPG_PRESSURE_GRADIENT_FORM=absolute|incremental` switches the direct PSPG pressure-gradient continuity residual between `grad(p)` and `grad(dt_eff * dt(p))` for diagnostic controls. `SVMP_PSPG_PRESSURE_GRADIENT_FORM` is accepted as a fallback. The default form is `absolute`, so normal production behavior is unchanged unless the env var is set.
- `SVMP_NS_PSPG_BOUNDARY_PRESSURE_GRADIENT_SCALE=<positive>` adds a diagnostic-only wall-boundary PSPG pressure-gradient support term on velocity-Dirichlet markers. `SVMP_PSPG_BOUNDARY_PRESSURE_GRADIENT_SCALE` is accepted as a fallback, and the pressure-row contribution diagnostic exposes the term as `equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient`.
- `SVMP_NS_PSPG_BOUNDARY_TANGENTIAL_PRESSURE_GRADIENT_SCALE=<positive>` adds a diagnostic-only wall-tangential PSPG pressure-gradient support term on velocity-Dirichlet markers. `SVMP_PSPG_BOUNDARY_TANGENTIAL_PRESSURE_GRADIENT_SCALE` is accepted as a fallback, and the pressure-row contribution diagnostic exposes the term as `equations_diagnostic_ns_vms_pspg_boundary_tangential_pressure_gradient`.
- `SVMP_NS_PSPG_BOUNDARY_TANGENTIAL_MOMENTUM_RESIDUAL_SCALE=<positive>` adds a diagnostic-only wall-tangential PSPG full momentum-residual support term on velocity-Dirichlet markers. `SVMP_PSPG_BOUNDARY_TANGENTIAL_MOMENTUM_RESIDUAL_SCALE` is accepted as a fallback, and the pressure-row contribution diagnostic exposes the term as `equations_diagnostic_ns_vms_pspg_boundary_tangential_momentum_residual`.
- `SVMP_NS_FREE_SURFACE_PRESSURE_REFERENCE_PROBE_PENALTY=<alpha>` is a disabled-by-default Navier-Stokes probe that adds `alpha * (p - p_ext) * q` on the generated free-surface continuity trace and exposes `equations_diagnostic_ns_free_surface_pressure_reference_probe` in the pressure-row contribution diagnostic.
- `SVMP_ACTIVE_PRESSURE_UPDATE_REJECT_ON_TRIGGER=1` reuses the active/wet pressure-update diagnostic before `TimeLoop` commits a converged candidate. A positive `SVMP_ACTIVE_PRESSURE_UPDATE_THRESHOLD_PA` causes a triggered candidate to be rejected as `ErrorTooLarge`; adaptive runs retry through the configured time-step controller and fixed-step runs fail fast.
- `TimeLoopCallbacks::on_before_step_accept` is the underlying hook. It runs after a converged nonlinear candidate is in `TimeHistory::u()` and before adaptive acceptance, `commitTimeStep()`, and `history.acceptStep()`.
- Optional knobs:
  - `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_ALLOWED_ZERO_VELOCITY_ROWS`
  - `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_TOLERANCE`
  - `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_SAMPLE_LIMIT`
  - `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_PRESSURE_FIELD`
  - `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_COUPLING_FIELD`
  - `SVMP_ACTIVE_PRESSURE_UPDATE_SUPPORT_ACTION_SAMPLE_LIMIT`
- `tests/cases/fluid/open_vessel_free_surface/audit_pressure_matrix_support_samples.py`
  now parses the aggregate support-rank diagnostic, proposed pressure-update/support summaries, and operator matrix-support attribution into JSON. It also summarizes constraint-side support provenance (`active_dof_support`, retained rule counts, active-sign status) against zero/weak velocity coupling and weak pressure self support, excluding unit-diagonal pressure identity rows from support-rank hazard counts.
- `tests/test_open_vessel_pressure_matrix_support_samples.py`
  covers parsing of the new diagnostic line.

## Test10 Replay Evidence

Artifacts:

- VMS-enabled accepted replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_matrix_sample_20260605_case/run_support_rank.log`
- VMS-enabled audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_audit_20260605.json`
- VMS-disabled failed replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_matrix_sample_20260605_case/run_support_rank.log`
- VMS-disabled audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_support_rank_audit_20260605.json`
- VMS-enabled guard replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_matrix_sample_20260605_case/run_support_rank_guard.log`
- VMS-enabled guard audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_guard_audit_20260605.json`
- VMS-enabled all-nine audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_all9_audit_20260605.json`
- VMS-disabled all-nine audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_support_rank_all9_audit_20260605.json`
- VMS-enabled clamp audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_clamp_all9_audit_20260605.json`
- Worst-row plus all-nine audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_worst80_all9_audit_20260605.json`
- Weak-coupling aggregate audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_weak_coupling_audit_20260605.json`
- Weak-coupling clamp audit at `3.1e-4`:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_weak_clamp_3p1e4_audit_20260605.json`
- Weak-coupling clamp audit at `3.3e-4`:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_rank_weak_clamp_3p3e4_audit_20260605.json`
- Weak-front support-measure audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_support_measure_weak_front_audit_20260605.json`
- Weak-front pressure-row contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_weak_front_contribution_audit_20260605.json`
- All-pressure-row support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_all_pressure_support_audit_20260605.json`
- Top pressure-update/support correlation audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_update_support_correlation_20260605.json`
- Pressure self-block clamp audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_self_clamp_1e7_support_rank_audit_20260605.json`
- Pressure self-block clamp pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_self_clamp_1e7_pressure_update_audit_20260605.json`
- Proposed pressure-update/support replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_support_20260605_case/run_update_support.log`
- Proposed pressure-update/support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_support_audit_20260605.json`
- Proposed pressure-update/support pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_support_pressure_update_audit_20260605.json`
- Proposed pressure-update equation audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_equation_audit_20260605.json`
- Proposed pressure-update action-term replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_action_terms_20260605_case/run_update_action_terms.log`
- Proposed pressure-update action-term audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_action_terms_audit_20260605.json`
- Proposed pressure-update action-term pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_action_terms_pressure_update_audit_20260605.json`
- Combined weak-support clamp audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_combined_support_clamp_audit_20260605.json`
- Combined weak-support clamp pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_combined_support_clamp_pressure_update_audit_20260605.json`
- Pressure-update neighborhood audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_neighborhood_20260605.json`
- Free-surface pressure-reference probe replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_reference_probe_penalty1em6_20260605_case/run_pressure_reference_probe_penalty1em6_sampled.log`
- Free-surface pressure-reference probe support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_reference_probe_penalty1em6_sampled_support_audit_20260605.json`
- Free-surface pressure-reference probe contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_reference_probe_penalty1em6_sampled_contribution_audit_20260605.json`
- Free-surface pressure-reference probe pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_reference_probe_penalty1em6_sampled_update_audit_20260605.json`
- Pressure-subblock null replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_subblock_null_20260605_case/run_pressure_subblock_null.log`
- Pressure-subblock null audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_subblock_null_audit_20260605.json`
- Pressure-subblock null pressure-update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_subblock_null_pressure_update_audit_20260605.json`
- Operator matrix-support attribution replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_operator_matrix_support_20260605_case/run_pressure_operator_matrix_support.log`
- Operator matrix-support attribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_operator_matrix_support_audit_20260605.json`
- Operator matrix-support attribution pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pressure_operator_matrix_support_pressure_update_audit_20260605.json`
- VMS/PSPG split operator-support replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_split_operator_support_20260605_case/run_pspg_split_operator_support.log`
- VMS/PSPG split operator-support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_split_operator_support_audit_20260605.json`
- VMS/PSPG split contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_split_operator_contribution_audit_20260605.json`
- VMS/PSPG split pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_split_operator_support_pressure_update_audit_20260605.json`
- PSPG pressure-gradient scale-zero replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale0_20260605_case/run_pspg_pgrad_scale0.log`
- PSPG pressure-gradient scale-zero support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale0_support_audit_20260605.json`
- PSPG pressure-gradient scale-zero contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale0_contribution_audit_20260605.json`
- PSPG pressure-gradient scale-up replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale10_20260605_case/run_pspg_pgrad_scale10.log`
- PSPG pressure-gradient scale-up support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale10_support_audit_20260605.json`
- PSPG pressure-gradient scale-up contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale10_contribution_audit_20260605.json`
- PSPG pressure-gradient scale-up pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_scale10_pressure_update_audit_20260605.json`
- PSPG pressure-gradient incremental-form replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_incremental_20260606_case/run_pspg_pgrad_incremental.log`
- PSPG pressure-gradient incremental-form support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_incremental_support_audit_20260606.json`
- PSPG pressure-gradient incremental-form contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_incremental_contribution_audit_20260606.json`
- PSPG pressure-gradient incremental-form pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_pgrad_incremental_pressure_update_audit_20260606.json`
- PSPG wall-boundary pressure-gradient replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_pgrad_scale1_20260606_case/run_pspg_wall_pgrad_scale1.log`
- PSPG wall-boundary pressure-gradient support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_pgrad_scale1_support_audit_20260606.json`
- PSPG wall-boundary pressure-gradient contribution audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_pgrad_scale1_contribution_audit_20260606.json`
- PSPG wall-boundary pressure-gradient pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_pgrad_scale1_pressure_update_audit_20260606.json`
- PSPG wall-boundary pressure-gradient expanded coverage replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_pgrad_scale1_coverage_20260606_case/run_pspg_wall_pgrad_scale1_coverage.log`
- PSPG wall-boundary pressure-gradient expanded coverage support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_pgrad_scale1_coverage_support_audit_20260606.json`
- Pre-commit pressure-update fixed-step rejection log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_reject_1000pa_20260606_case/run_update_reject_1000pa.log`
- Pre-commit pressure-update adaptive rejection log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_update_reject_adaptive_1070pa_20260606_case/run_update_reject_adaptive_1070pa.log`
- Full-gradient graph-completion Test10 replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_weakself1e8_20260606_case/run_graph_completion_weakself1e8.log`
- Full-gradient graph-completion Test10 support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_weakself1e8_support_audit_20260606.json`
- Full-gradient graph-completion Test10 pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_weakself1e8_pressure_update_audit_20260606.json`
- Full-gradient graph-completion Test02 replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_graph_completion_weakself1e8_20260606_case/run_graph_completion_weakself1e8.log`
- Full-gradient graph-completion Test02 support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_graph_completion_weakself1e8_support_audit_20260606.json`
- Full-gradient graph-completion Test02 pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_graph_completion_weakself1e8_pressure_update_audit_20260606.json`
- Full-gradient pressure-neighbor graph-completion Test10 replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_neighbor_weakself1e8_20260606_case/run_graph_completion_neighbor_weakself1e8.log`
- Full-gradient pressure-neighbor graph-completion Test10 support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_neighbor_weakself1e8_support_audit_20260606.json`
- Full-gradient pressure-neighbor graph-completion Test10 pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_neighbor_weakself1e8_pressure_update_audit_20260606.json`
- Full-gradient pressure-neighbor graph-completion Test02 replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_graph_completion_neighbor_weakself1e8_20260606_case/run_graph_completion_neighbor_weakself1e8.log`
- Full-gradient pressure-neighbor graph-completion Test02 support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_graph_completion_neighbor_weakself1e8_support_audit_20260606.json`
- Full-gradient pressure-neighbor graph-completion Test02 pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_pspg_wall_full_gradient_graph_completion_neighbor_weakself1e8_pressure_update_audit_20260606.json`
- Full-gradient active-support graph-completion Test10 replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_active_support_neigh64_leastselector_20260606_case/run_graph_completion_active_support_neigh64_leastselector.log`
- Full-gradient active-support graph-completion Test10 support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_active_support_neigh64_leastselector_support_audit_20260606.json`
- Full-gradient active-support graph-completion Test10 pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_active_support_neigh64_leastselector_pressure_update_audit_20260606.json`
- Full-gradient uncapped active-support graph-completion Test10 replay log:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_active_support_all_leastselector_20260606_case/run_graph_completion_active_support_all_leastselector.log`
- Full-gradient uncapped active-support graph-completion Test10 support audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_active_support_all_leastselector_support_audit_20260606.json`
- Full-gradient uncapped active-support graph-completion Test10 pressure update audit:
  `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_pspg_wall_full_gradient_graph_completion_active_support_all_leastselector_pressure_update_audit_20260606.json`

| Control | Status | Unconstrained pressure rows | Zero Velocity row-block rows | Pressure-only row-block rows | Zero total pressure rows | Local zero-Velocity rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VMS enabled | accepted | 251 | 9 | 9 | 0 | `180|183|190|193|250|649|652|659|662` |
| VMS disabled | failed before acceptance | 251 | 9 | 4 | 5 | `180|183|190|193|250|649|652|659|662` |
| VMS enabled + guard | failed before acceptance | 251 | 9 | 9 | 0 | `180|183|190|193|250|649|652|659|662` |
| VMS enabled + clamp | accepted | 251 | 9 | 9 | 0 | `180|183|190|193|250|649|652|659|662` |

The VMS-enabled accepted replay still accepts the pressure update diagnostic's worst active pressure increment: `1,075.558213 Pa` at pressure DOF `3356`.

The pre-commit pressure-update rejection probe catches the same row before commit. With `SVMP_ACTIVE_PRESSURE_UPDATE_REJECT_ON_TRIGGER=1` and a 1000 Pa threshold, the fixed-step replay logs `phase='pre_commit'`, `global_abs_pressure_delta_pa=1.0755582119407377e+03`, and `triggered=1`, then rejects the converged candidate as `ErrorTooLarge` with no accepted-step callback. In an adaptive replay with `min_dt=7.8125e-5` and a 1070 Pa threshold, the same row is rejected at every attempted `dt`; the pre-commit maxima are `1,075.558212`, `2,212.155949`, `4,512.193000`, and `9,103.459989 Pa` as `dt` halves. Thus the guard can prevent silent acceptance, but the Test10 step90 state is not fixed by timestep reduction.

The guard replay fails before that accepted update with:

`unconstrained pressure rows with zero Velocity row-block support=9 allowed=0 sample_local_dofs=180|183|190|193|250|649|652|659|662`

The all-nine constraint sample maps every zero-velocity-coupled pressure row to an active-supported unconstrained vertex pressure DOF. Seven are negative-side vertices, and two (`193`, `662`) have positive `phi` but remain active-supported through retained cut-volume support.

The regenerated all-pressure support-provenance summaries sharpen that mapping. Test10 has `252` retained active pressure rows after excluding `568` unit-diagonal pressure identity rows; `9` retained rows have zero `Velocity` row-block support, `223` retained rows are weakly velocity-coupled at the `1e-3` provenance threshold, and `85` retained rows have weak pressure self support at `1e-7`. Only `2/9` retained zero-coupling rows and `59/223` retained weak-coupling rows are inactive-sign retained support, so a constraint rule that targets only positive-side retained vertices would miss most of the Test10 support-rank hazard. The Test02 all-pressure support-provenance summary has `880` retained active pressure rows after excluding `1725` pressure identity rows, with `0` retained zero-coupling rows, `69` retained weak-coupling rows, and `298` retained weak-pressure-self rows. This rules out a zero-coupling retained-row constraint as a complete Test02/Test10 fix, while strengthening the boundary weak-self/weak-coupling formulation target.

The clamp probe is useful as a causality check. It clamps all nine zero-velocity-coupled rows and lowers the accepted worst pressure update from `1,075.558213 Pa` to `711.224854 Pa`, but the worst row remains global DOF `3356`. Therefore the nine zero-coupling rows contribute to rank/conditioning risk, but they are not the complete cause of the accepted pressure jump.

Sampling the accepted worst row shows pressure-local row `80` / global `3356` is active supported and unconstrained with nonzero velocity coupling: `row_field_abs_sums=phi:0|Velocity:0.0003049584994625626|Pressure:3.45981988765098e-08`. The lightweight aggregate replay reports all positive-coupled pressure rows are weakly velocity-coupled in this state: `positive_coupling_row_block_count=242`, `min_positive_coupling_row_abs_sum=1.4006220137698043e-05`, and `max_coupling_row_abs_sum=0.0011717337093909608`.

Targeted weak-coupling clamp controls bracket the accepted update without running a broad sweep:

| Control | Clamp threshold | Clamped rows | Includes original row `80` | Includes shifted row `78` | Accepted worst update | Accepted worst global DOF / point |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Baseline | n/a | 0 | no | no | `1,075.558213 Pa` | 3356 |
| Zero-coupling clamp | `1e-14` | 9 | no | no | `711.224854 Pa` | 3356 |
| Weak clamp | `3.1e-4` | 64 | yes | no | `352.269145 Pa` | 3354 |
| Weak clamp | `3.3e-4` | 85 | yes | yes | `182.117016 Pa` | 3358 |
| Pressure self clamp | `row_self <= 1e-7` plus zero-coupling | 89 | yes | yes | `62.132386 Pa` | point 128 |
| Combined weak-support clamp | `row_self <= 1e-7` or `row_velocity <= 3.3e-4` | 139 | yes | yes | `0.001281 Pa` | point 558 |

The `3.1e-4` threshold was chosen because the baseline worst row has `Velocity` row-block support `3.04958e-4`. The `3.3e-4` threshold was the single follow-up needed to include the shifted worst row `78`, whose support is `3.26681e-4`. After that clamp, the worst row shifts to local row `82`, whose support is `7.26633e-4`.

The support-measure replay samples the weak-coupling front and shows the weak rows are not merely low retained-volume-support rows:

| Local pressure row | Velocity row-block support | Retained measure | Retained rules | Retained volume fraction range |
| ---: | ---: | ---: | ---: | --- |
| 77 | `1.05282e-4` | 0.666667 | 4 | `1..1` |
| 78 | `3.26681e-4` | 1.33333 | 8 | `1..1` |
| 80 | `3.04958e-4` | 1.33333 | 8 | `1..1` |
| 82 | `7.26633e-4` | 4.0 | 24 | `1..1` |

This rules out the narrow explanation that the accepted worst rows are weakly coupled because they have tiny retained active-volume measure. The active support constraint correctly sees full retained-volume support for these rows; the weak pressure/velocity coupling is produced by the assembled pressure equation/coupling scale.

The weak-front pressure-row contribution replay samples the same front rows at the accepted line-search state:

| Local pressure row | Global pressure dof | Galerkin continuity | VMS/PSPG | Pressure ghost penalty | Total Pressure residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 77 | 3353 | `-4.37365e-07` | `+4.37365e-07` | 0 | `-3.57953e-14` |
| 78 | 3354 | `+5.52131e-06` | `-5.52131e-06` | 0 | `-3.41646e-13` |
| 80 | 3356 | `-8.48247e-06` | `+8.48247e-06` | 0 | `-7.44495e-13` |
| 82 | 3358 | `-5.69237e-06` | `+5.69237e-06` | 0 | `-1.19352e-12` |

This rules out a nonzero pressure-row residual source across the weak front. The sampled rows are full retained-support rows whose pressure equations cancel to roundoff, so the remaining mechanism is not local residual forcing; it is pressure/velocity coupling scale and pressure-update/null behavior in the coupled solve.

The pressure self-block clamp gives a stronger Test10 causality probe than the velocity-only clamp: `row_self <= 1e-7` plus the default zero-coupling clamp lowers the accepted worst active/wet update to `62.132386 Pa`, with the full-wet worst update at `62.102670 Pa`. This is not promoted as a default fix because the corresponding Test02 pressure-self probe shifts the accepted maximum to a worse tiny-cut pressure mode, as recorded in `Documentation/open_vessel_free_surface_supportfix_replay_audit_20260605.md`.

The graph-completion prototype tests a less destructive topology mutation than the clamp family. It adds constant-null pressure cycle edges between selected weak-support rows before the linear solve. With full wall-gradient pressure support and `SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_PRESSURE_SELF_ROW_SUM=1e-8`, the Test10 replay selects `9` rows, adds `9` edges at weight `4.41429013362441e-08`, and lowers the accepted update from the full-gradient `622.609410 Pa` control to `474.046852 Pa`, still above the `100 Pa` guard. The Test02 counterpart selects `298` weak-self rows, adds `298` edges at weight `1.266545099351897e-09`, and lowers the full-gradient `366719.965806 Pa` control to `226684.935365 Pa`, but shifts the maximum to a tiny-cut-supported row and still exceeds the `100000 Pa` guard. This confirms pressure graph topology is causal while ruling out the blunt weak-row graph cycle as a default support-rank fix.

The structured `SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MODE=pressure_neighbor` control rules out another tempting support-rank fix. It connects weak candidates to their strongest existing pressure-block neighbor outside the candidate set. Test02 selects the same `298` weak-self rows, adds `141` unique pressure-neighbor edges to `135` neighbors, and remains above guard at `226803.140015 Pa` on tiny-cut-supported row `10624`. Test10 selects `9` candidates, adds `9` edges to `8` neighbors, and worsens to `573.379695 Pa` on full-wet row `3456` with zero velocity coupling. Thus the missing support is not simply an omitted edge to the current pressure graph's strongest neighbor.

The `SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MODE=active_support_completion` least-selector follow-up tests a broader topology-to-active-support direction. In Test10 it selects `68` candidates, finds `132` active-support neighbors, adds `4352` constant-null edges at weight `6.335956786031046e-10`, and lowers the full-gradient branch to `201.155618 Pa` on a cut-supported point, with the full-wet maximum at `145.345067 Pa`. This is directionally better than the simple cycle and pressure-neighbor modes, but it still triggers the `100 Pa` guard, so raw post-assembly active-support completion is diagnostic evidence rather than a production fix.

The uncapped active-support follow-up removes `SVMP_ACTIVE_PRESSURE_GRAPH_COMPLETION_MAX_ACTIVE_NEIGHBORS=64` by setting it to `-1`. In Test10 the same `68` candidates expand from `132` to `251` active-support neighbors and from `4352` to `14722` constant-null edges, with weights `1.6220049372239478e-10..3.2440098744478956e-10`. The direct pressure-update guard still triggers at `203.045993 Pa` on cut-supported point `609`, with the full-wet maximum at `132.079851 Pa`. This rules out the active-neighbor cap as the reason active-support completion stayed above the `100 Pa` Test10 guard.

The proposed Newton update/support replay confirms the `1,075.558213 Pa` Test10 pressure jump is already present in the solved increment before line search or accepted-step maintenance. The largest proposed pressure update is local row `80` / global `3356`, with `row_coupling=3.049584995e-4` and `row_self=3.459819888e-8`. Zero-coupling rows have proposed updates up to `702.668903 Pa`, and rows above the weak-coupling threshold still reach `617.975523 Pa`. The signed row-action fields show the row satisfies `Jdu=rhs` to roundoff: `rhs=-8.1513470004e-6`, `row_action=-8.1513470004e-6`, `row_self_action=-8.0073392197e-6`, `row_coupling_action=-1.4400778067e-7`, and `row_linear_residual=1.04e-18`. The regenerated parser summary records the same max-row diagonal action as `1.8606188485543292e-5` and pressure-self/velocity-action ratio as `55.603517964165334`. This rules out accepted-state output processing, dt scaling, and local linear residual error as the source of the jump and keeps the target on assembled pressure support/rank and pressure-update null behavior.

The action-term replay shows the row equation is pressure-block dominated, not velocity-action dominated. The target diagonal term is `1.729909944e-8 * -1075.558213 = -1.860618849e-5`. Neighboring pressure terms at local rows `3`, `549`, `81`, `77`, and `89` add `+2.64e-6`, `+2.64e-6`, `+2.59e-6`, `+1.58e-6`, and `+1.13e-6`, giving the net pressure self action `-8.007339220e-6`. The net velocity action is only `-1.440077807e-7`. This points the next Test10 fix toward pressure-pressure boundary-stencil/support consistency, with velocity coupling still important for rank but not the dominant max-row action.

The pressure-subblock null replay shows this is not a constant pressure row-sum/gauge leakage mode. At the same row `80` / global `3356`, the signed pressure-subblock row sum is `8.27181e-25`, the signed/absolute pressure row ratio is `2.39e-17`, and the diagonal/absolute pressure row ratio is `0.5`. The constant-pressure action at the row's own update is only `-8.90e-22`, while the nonconstant pressure action is `-8.007339220e-6`, equal to the pressure self action to logged precision. Therefore the pressure block preserves the constant-pressure null row at the sampled maximum; the bad update is a nonconstant boundary pressure mode.

The operator matrix-support attribution replay shows which diagnostic operator owns that pressure self-block. At row `80` / global `3356`, Galerkin continuity has `row_self_abs_sum=0` and pressure ghost penalty has zero row, self-block, coupling-block, and diagonal support. The VMS/PSPG operator has `row_self_abs_sum=3.459819888e-8`, signed pressure row sum `1.65436e-24`, signed/absolute ratio `4.78e-17`, diagonal ratio `0.5`, and velocity coupling support `3.047107657e-4`. The aggregate active-continuity operator contains the same pressure self-block because active continuity is Galerkin plus VMS/PSPG. Therefore the sampled Test10 max-row pressure self-stencil is VMS/PSPG, not pressure ghost penalty.

The VMS/PSPG split attribution replay localizes that further. At the same row, `equations_diagnostic_ns_vms_pspg_pressure_gradient` has `row_self_abs_sum=3.459819888e-8`, `diag=1.729909944e-8`, and only `4.84877e-12` velocity-coupling row support, while `equations_diagnostic_ns_vms_pspg_nonpressure` has `row_self_abs_sum=0`, `diag=0`, and `row_coupling_abs_sum=3.047107668e-4`. The sampled residuals close to the full VMS/PSPG row: `8.67907e-6 + (-1.96594e-7) = 8.48247e-6` to logged precision. Therefore the accepted max-row pressure self-stencil is the direct PSPG pressure-gradient term, while the nonpressure PSPG residual terms provide the velocity-coupling support.

The PSPG pressure-gradient scale-zero replay shows why that term cannot simply be removed. With `SVMP_NS_PSPG_PRESSURE_GRADIENT_SCALE=0`, row `80` / global `3356` has no pressure self support (`row_self_abs_sum=0`, `diag=0`) while retaining velocity coupling (`row_coupling_abs_sum=3.049584997e-4`). The direct solve then fails before acceptance with five pressure zero rows/cols, local rows `180|183|250|649|652`, matching the known VMS-disabled structural-zero subset. The pressure-gradient self-stencil is therefore a defective boundary pressure mode in the accepted baseline, but it is also part of the support/rank path that prevents Test10 singularity.

The PSPG pressure-gradient scale-up replay shows the other side of that constraint. With `SVMP_NS_PSPG_PRESSURE_GRADIENT_SCALE=10`, the same sampled row `80` / global `3356` has `row_self_abs_sum=3.459819891e-7`, `diag=1.729909945e-7`, and nearly unchanged velocity coupling (`row_coupling_abs_sum=3.049626921e-4`). The one-step replay accepts and the original full-wet maximum is reduced, but the worst active/wet update remains `802.897912 Pa` on a cut-supported boundary-edge row. A uniform scale-up is therefore directionally supportive, but it is not a production fix for Test10's boundary pressure mode.

The PSPG pressure-gradient incremental-form replay rules out the narrow old-pressure residual explanation. With `SVMP_NS_PSPG_PRESSURE_GRADIENT_FORM=incremental`, the same row `80` / global `3356` keeps `row_self_abs_sum=3.459819888e-8`, `diag=1.729909944e-8`, and velocity coupling `3.049584997e-4`. The start-state sampled pressure-gradient residual contribution is zero, but the line-search contribution returns as `8.59316e-6`, and the one-step replay accepts a worse `2,078.981232 Pa` full-wet update. A pressure-increment residual form therefore preserves rank but does not fix the nonconstant boundary pressure update mode.

The wall-boundary pressure-gradient support probe adds row self support at the original Test10 max row but does not close the failure. With `SVMP_NS_PSPG_BOUNDARY_PRESSURE_GRADIENT_SCALE=1`, `equations_diagnostic_ns_vms_pspg_boundary_pressure_gradient` contributes `row_self_abs_sum=2.452305634e-8` and `diag=1.226152817e-8` at row `80` / global `3356`; the original sampled row's proposed update drops to `839.670152 Pa`. The accepted maximum shifts instead to pressure-local row `250` / global `3526`, a full-wet zero-velocity-coupled boundary-edge row with `867.810372 Pa`, `row_self=6.316012565e-8`, and `row_coupling=0`. This supports a boundary pressure-support mechanism, but rules out the simple wall-normal term as a complete Test10 fix.

The expanded coverage replay samples the top pressure-update rows and the weakest pressure self-block rows from the existing support correlation. The wall-boundary operator has nonzero pressure self support on `12/19` sampled rows, including the shifted accepted row `250` / global `3526`; the direct PSPG pressure-gradient split has pressure self support on all `19/19`, and pressure ghost penalty has pressure self support on `5/19`. This shows the shifted Test10 maximum is not uncovered by the wall term. The remaining problem is that the row is still zero-velocity-coupled and pressure-block dominated after the added boundary pressure self support.

The combined wall-normal plus wall-tangential pressure-gradient replay adds the full wall-gradient pressure self block and still does not close the support problem. In Test10 it accepts with `622.609410 Pa` on full-wet row `3526`, with zero velocity coupling at the max row. In Test02 it accepts with `366719.965806 Pa` on full-wet row `10676`, with the row still dominated by pressure self action. This rules out the simple full wall-gradient boundary pressure-gradient term as the missing coupling/support mechanism.

The local pressure-edge completion predictor uses the same full-gradient max-row action terms to test whether the synthetic pair-completion idea has real-row leverage. Test02 row `10676` has three same-signed pressure neighbors below the `100000 Pa` guard; an added constant-null edge to those neighbors would need only `1.25e-09` to `1.30e-09`, which is `0.86x` to `0.89x` the row diagonal and `0.43x` to `0.45x` the row pressure self sum. Test10 row `3526` has no logged pressure neighbor below the `100 Pa` guard; the best logged neighbor is still `353.550370 Pa`. This supports pressure-edge topology completion for Test02 but rules out a local edge to the existing sampled neighbors as the complete Test10 mechanism.

The free-surface pressure-reference probe replay with `SVMP_NS_FREE_SURFACE_PRESSURE_REFERENCE_PROBE_PENALTY=1e-6` keeps the accepted max row in the same class. The worst active/wet update remains `1010.774444 Pa` at pressure-local row `80` / global `3356`, with `row_self=3.459819888e-8`, `row_coupling=3.049584995e-4`, and `diag=1.729909944e-8`. The sampled contribution from `equations_diagnostic_ns_free_surface_pressure_reference_probe` at global dof `3356` is exactly `0.0` even though the operator has nonzero norm (`2.93228e-7`). This rules out a missing direct generated-interface pressure-reference trace term as the driver for the Test10 accepted maximum; the bad row is an active boundary pressure row governed by pressure-block/support behavior.

The combined weak-support clamp is the strongest Test10 causality probe so far: it clamps 139 rows and lowers the accepted active/wet pressure update to `0.001281 Pa`. This still should not be promoted as the default fix because the same combined strategy makes the Test02 replay fail nonlinear convergence, as recorded in `Documentation/open_vessel_free_surface_supportfix_replay_audit_20260605.md`.

The pressure-update neighborhood audit shows the original Test10 maximum is not a single isolated point spike. The target point `3` on `x_min` is full-wet supported with `1,075.558213 Pa` update; its 24 nearest neighbors are all same-signed with median `|dp|=502.565019 Pa`, and its incident patch is also fully same-signed with median `|dp|=607.944497 Pa`. The target/patch-median ratio is `1.77`, and the target/largest-patch-neighbor ratio is `1.44`. This supports a coherent weak-support boundary mode with local amplification, consistent with the weak pressure/velocity support-front evidence.

The all-pressure-row support replay samples all `819` pressure rows and all active pressure constraint rows, then joins the top direct-pair pressure updates against matrix support through the sampled vertex mapping. For the top 30 accepted pressure updates:

| Coupling class with `3.3e-4` weak threshold | Top-update rows | Largest absolute pressure update |
| --- | ---: | ---: |
| Weak positive `Velocity` row-block support | 13 | `1,075.558213 Pa` |
| Zero `Velocity` row-block support | 7 | `702.668903 Pa` |
| Positive support above threshold | 10 | `617.975523 Pa` |

All top 30 updates matched sampled matrix rows. The three largest updates are weak positive-coupled rows; rows in the zero-coupling class are also prominent. However, the largest above-threshold positive-coupled update remains about `618 Pa`, so a simple zero/weak threshold clamp is not a complete formulation fix. The correlation supports a pressure/velocity coupling-scale mechanism and also shows the default fix must address coupled pressure-update/null behavior more generally than a single cutoff.

## Interpretation

The previous sampled matrix-support diagnostic showed five active pressure rows with zero velocity-block support. The aggregate scanner shows that the class is larger: nine unconstrained pressure rows have no velocity row-block support in both VMS-enabled and VMS-disabled Test10 step90 replays.

With VMS enabled, all nine are pressure-only rows. With VMS disabled, the same nine rows remain zero-velocity-coupled, but five collapse to fully zero rows and columns. Those five are the direct factorization singularity rows; the other four retain non-VMS pressure-block support.

This makes the next formulation target sharper but also rules out an overly narrow fix. Active pressure support should not leave pressure rows unconstrained solely because their vertices belong to retained active cells if those rows have no Galerkin velocity coupling. However, clamping only the zero-coupling rows does not remove the accepted pressure jump, and targeted weak-coupling clamps move the accepted update along the weakly coupled pressure-row front. The support-measure audit rules out tiny retained support as the driver for the weak front. A default fix needs a principled pressure support/rank and scaling rule for both zero-coupling and weak-coupling active pressure rows; a blanket post-assembly clamp remains a diagnostic probe, not a verified physical pressure formulation.

The top-update correlation, action-term audit, pressure-subblock null audit, operator matrix-support attribution, VMS/PSPG split attribution, scale-zero, scale-up, incremental-form controls, wall-boundary support probe, wall-tangential boundary support probe, full wall-gradient boundary support probe, local pressure-edge predictor, and neighborhood audit reinforce that caution. Zero/weak coupling explains many of the worst pressure updates and the largest one, but not all large updates; the maximum is also part of a broader same-signed boundary patch and its row equation is pressure-block dominated rather than velocity-action dominated. The pressure row-sum at the maximum is effectively zero, and the sampled pressure self-block is carried by the VMS/PSPG pressure-gradient split rather than pressure ghost penalty or the nonpressure PSPG terms. The retained synthetic patch now adds a minimal solve proxy for the same distinction: matched hydrostatic cancellation and constant-null preservation still allow the weakest one-cell boundary row to respond `10.09x` more strongly than the strongest shared row under a zero-mean pressure-row load, and a full-volume topology control still gives a `6.0x` response ratio. A uniform scale-10 patch probe lowers absolute response but preserves the ratio. A constant-null one-cell boundary pair-completion probe changes the pressure support topology and reduces those ratios to `6.1587301587301555` and `3.4999999999999996` while preserving hydrostatic cancellation and the constant-pressure null. The real-row edge predictor supports diagonal-scale local edge completion for Test02, but shows Test10 needs broader topology or coupling because none of its logged pressure neighbors are below guard. Solver-level active-support completion gives that broader topology direction real-row leverage, lowering Test10 to `201.155618 Pa`, but it still triggers the `100 Pa` guard; removing the active-neighbor cap expands the graph to `14722` edges and still leaves `203.045993 Pa`, so the cap is not the missing active-support ingredient. Removing the direct PSPG pressure-gradient split restores a structural-zero pressure-row failure, uniformly strengthening it leaves above-threshold cut-supported updates, switching it to a pressure-increment residual worsens the accepted full-wet update, and adding a simple wall-normal boundary pressure-gradient term only moves the Test10 maximum to another boundary row even though that shifted row has wall-boundary pressure self support. The wall-tangential boundary pressure-gradient term reaches all sampled important rows in Test02 and Test10 and reduces the accepted maxima to `370071.857167 Pa` and `591.865160 Pa`, but both still trigger their guards and the target rows remain pressure-self supported without meaningful velocity coupling. Combining wall-normal and wall-tangential pressure-gradient support leaves Test02 at `366719.965806 Pa` and Test10 at `622.609410 Pa`, so the full wall-gradient pressure self block is not the missing production mechanism. The matched wall-tangential momentum-residual term lowers Test10 further to `472.150770 Pa`, but shifts the maximum to a cut-supported row and adds no velocity coupling on the sampled rows; in Test02 it adds velocity coupling to all `23/23` sampled rows, including `0.009373840196978678` at row `10676`, but worsens the accepted update to `726276.622250 Pa`. Raw wall-marker full-residual coupling is therefore ruled out as the production fix. A production change should be framed as a topology-completing direct PSPG pressure-gradient formulation/support consistency fix for nonconstant boundary pressure modes that preserves pressure-row rank, not as an update clamp calibrated to the observed threshold, a ghost-penalty row fix at the accepted maximum, deletion of the pressure-gradient term, a global pressure-gradient multiplier, a simple pressure-increment residual substitution, a capped or uncapped post-assembly active-support mutation, or a wall-marker boundary term alone.

The pressure-reference probe adds one more exclusion: the accepted max row is not waiting for a direct generated-interface pressure trace contribution. Any production fix should target active cut-volume boundary pressure-row support/null behavior rather than adding a blanket free-surface pressure anchor.

## Verification

- `cmake --build build/svMultiPhysics-build --target svmultiphysics -j 4`
- `python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_pressure_matrix_support_samples.py`
- `python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_linear_pressure_cut_volume_patch.py`
- `pytest -q tests/test_open_vessel_pressure_matrix_support_samples.py`
- `pytest -q tests/test_open_vessel_linear_pressure_cut_volume_patch.py`
- VMS-enabled Test10 step90 support-rank replay: `status=0`
- VMS-disabled Test10 step90 support-rank replay: `status=1` at the known direct factorization failure
- VMS-enabled Test10 step90 guard replay: `status=1` at `active_pressure_support_rank` guard
- VMS-enabled Test10 step90 clamp replay: `status=0`, accepted update reduced to `711.224854 Pa`
- VMS-disabled Test10 step90 clamp replay: `status=1`; the nine rows become identity rows, but the direct solve still does not converge with VMS disabled
- VMS-enabled weak-coupling clamp `3.1e-4`: `status=0`, accepted update reduced to `352.269145 Pa`
- VMS-enabled weak-coupling clamp `3.3e-4`: `status=0`, accepted update reduced to `182.117016 Pa`
- VMS-enabled weak-front support-measure replay: `status=0`; weak rows `77/78/80/82` all have retained volume fraction `1..1`
- VMS-enabled weak-front pressure-row contribution replay: `status=0`; weak rows `77/78/80/82` all have zero pressure ghost-penalty contribution and Galerkin/PSPG cancellation to roundoff
- VMS-enabled all-pressure-row support replay: `status=0`; top 30 accepted pressure updates all match sampled matrix support rows
- VMS-enabled proposed pressure-update/support replay: `status=0`; largest proposed pressure update already matches the accepted full-wet replay jump at row `80` / global `3356`
- VMS-enabled proposed pressure-update equation replay: `status=0`; max row has `Jdu-rhs=1.04e-18`
- VMS-enabled proposed pressure-update action-term replay: `status=0`; max row is pressure-block dominated with small velocity action
- VMS-enabled pressure-subblock null replay: `status=0`; max row has signed pressure row sum `8.27181e-25` and constant-pressure action `-8.90e-22`
- VMS-enabled operator matrix-support attribution replay: `status=0`; max-row pressure self-block is present in VMS/PSPG and zero in pressure ghost penalty
- VMS-enabled VMS/PSPG split attribution replay: `status=0`; max-row pressure self-block is present in the pressure-gradient split and zero in the nonpressure split
- VMS-enabled PSPG pressure-gradient scale-zero replay: `status=1`; pressure-gradient self-block is removed, but the direct solve fails with pressure zero rows `180|183|250|649|652`
- VMS-enabled PSPG pressure-gradient scale-up replay: `status=0`; accepted update reduced from the original maximum but remains `802.897912 Pa` on a cut-supported boundary row
- VMS-enabled PSPG pressure-gradient incremental-form replay: `status=0`; pressure-gradient self-block/rank support is preserved, but accepted update worsens to `2,078.981232 Pa`
- VMS-enabled PSPG wall-boundary pressure-gradient replay: `status=0`; original row `3356` gets additional boundary pressure self support, but the accepted maximum shifts to row `3526` with `867.810372 Pa`
- Wall-tangential PSPG boundary pressure-gradient Test10 replay: `status=0`; accepted update reduced to `591.865160 Pa`, and the diagnostic operator has nonzero pressure self support on all `19/19` sampled important rows
- Wall-tangential PSPG boundary pressure-gradient Test02 replay: `status=0`; accepted update reduced to `370071.857167 Pa`, and the diagnostic operator has nonzero pressure self support on all `23/23` sampled important rows
- Combined wall-normal plus wall-tangential PSPG boundary pressure-gradient Test10 replay: `status=0`; accepted update is `622.609410 Pa` on full-wet row `3526`, with zero velocity coupling at the max row
- Combined wall-normal plus wall-tangential PSPG boundary pressure-gradient Test02 replay: `status=0`; accepted update is `366719.965806 Pa` on full-wet row `10676`, still pressure-self dominated
- Full-gradient graph-completion Test10 replay: `status=0`; candidate graph cycle lowers the accepted update to `474.046852 Pa`, still above guard
- Full-gradient graph-completion Test02 replay: `status=0`; candidate graph cycle lowers the accepted update to `226684.935365 Pa`, still above guard and shifted to tiny-cut-supported row `10624`
- Full-gradient pressure-neighbor graph-completion Test10 replay: `status=0`; strongest-neighbor edges worsen the accepted update to `573.379695 Pa`, still above guard
- Full-gradient pressure-neighbor graph-completion Test02 replay: `status=0`; strongest-neighbor edges leave the accepted update at `226803.140015 Pa`, still above guard and tiny-cut-supported
- Full-gradient active-support graph-completion Test10 replay: `status=0`; active-support edges lower the accepted update to `201.155618 Pa`, still above the `100 Pa` guard
- Full-gradient uncapped active-support graph-completion Test10 replay: `status=0`; removing the active-neighbor cap expands to `14722` edges but leaves the accepted update at `203.045993 Pa`, still above the `100 Pa` guard
- Full-gradient Test02 local pressure-edge predictor: `status=0`; finding is `local_pressure_edge_completion_plausible_for_sampled_max_row`
- Full-gradient Test10 local pressure-edge predictor: `status=0`; finding is `no_logged_pressure_neighbor_below_guard_for_sampled_max_row`
- Wall-tangential PSPG boundary momentum-residual Test10 replay: `status=0`; accepted update reduced to `472.150770 Pa` on cut-supported row `3837`, with diagnostic pressure self support on all `19/19` sampled rows and velocity coupling on `0/19`
- Wall-tangential PSPG boundary momentum-residual Test02 replay: `status=0`; accepted update is `726276.622250 Pa`, with diagnostic pressure self support and velocity coupling on all `23/23` sampled rows, including row `10676` coupling `0.009373840196978678`
- VMS-enabled combined weak-support clamp replay: `status=0`; accepted update reduced to `0.001281 Pa`
- VMS-enabled pressure-update neighborhood audit: `status=0`; max row has same-signed nearest-neighbor and incident-patch support
- VMS-enabled free-surface pressure-reference probe replay: `status=0`; sampled probe contribution at global dof `3356` is `0.0`
- `pytest -q tests/test_open_vessel_pressure_update_support_correlation.py tests/test_open_vessel_pressure_matrix_support_samples.py tests/test_open_vessel_pressure_contribution_samples.py tests/test_open_vessel_pressure_update_neighborhood.py`
