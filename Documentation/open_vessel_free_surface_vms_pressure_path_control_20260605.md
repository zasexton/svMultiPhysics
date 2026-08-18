# Open-Vessel VMS/PSPG Pressure-Path Control

Date: 2026-06-05

Scope: targeted Test02/Test10 one-step replay control after the Newton pressure residual diagnostic. This asks whether the residual-based VMS/PSPG branch is the immediate source of the accepted pressure jumps, or whether removing it exposes a pressure-row support/rank problem.

## Code Change

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
  now supports an environment-only diagnostic override for the residual-based VMS branch.
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  now supports an environment-gated matrix-support sampler for selected global DOFs or pressure-local rows immediately before the linear solve.

Set `SVMP_NS_DISABLE_VMS=1` to force `effective_enable_vms=0`, or set `SVMP_NS_ENABLE_VMS=0/1` to explicitly override the module option. When active, the module logs `diagnostic=navier_stokes_vms_override`. This is a diagnostic control, not a proposed production setting.

The override leaves active cut-volume integration, pressure ghost penalty, inactive pressure constraints, cut-context refresh, and the accepted pressure-update guard unchanged.

## Replay Runs

Control directories:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_vms_disabled_20260605_case/`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_20260605_case/`

Both were run from the same prepared replay inputs as the residual-diagnostic controls, with:

- `SVMP_NS_DISABLE_VMS=1`
- `SVMP_ACTIVE_PRESSURE_UPDATE_DIAGNOSTIC=1`
- `SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC=1`
- `SVMP_ACTIVE_PRESSURE_RESIDUAL_DIAGNOSTIC=1`

## Results

| Case | VMS/PSPG | Outcome | Active/wet pressure update | Support | Solve-time Pressure residual | Notes |
| --- | --- | --- | ---: | --- | ---: | --- |
| Test02 step382 replay | enabled baseline | accepted | 2,112,209.8407 Pa | full wet | 5.20234e-08 | Residual-consistent MPa jump persists in baseline. |
| Test02 step382 replay | disabled | accepted | 4,853,753.8602 Pa | tiny cut | 5.21349e-08 | Full-wet max also rises to 4,740,276.3868 Pa. |
| Test10 step90 replay | enabled baseline | accepted | 1,075.558213 Pa | full wet | 1.65093e-05 | Residual-consistent kPa jump persists in baseline. |
| Test10 step90 replay | disabled | failed before acceptance | n/a | n/a | 2.32425e-05 at first residual assembly | Direct factorization failed with 5 zero pressure rows and 5 zero pressure columns; runtime and matrix sampling map them to active supported vertex pressure DOFs with zero assembled row/column support. |

Test02 with VMS disabled still accepts a pressure state whose `Pressure` residual rows are tiny on the line-search trial. The worst active/wet update changes support class from full-wet to tiny-cut, but the full-wet category also worsens to 4.740 MPa, so removing VMS/PSPG does not remove the pressure excursion.

Test10 with VMS disabled never reaches an accepted-step pressure guard event. The direct solver reports:

- `zero_rows=5`, `zero_cols=5`
- `Pressure{zero_rows=5, zero_cols=5, zero_diag=85, identity_rows=568}`
- `EigenLinearSolver (direct): factorization failed`

## Pressure Row-Support Audit

`tests/cases/fluid/open_vessel_free_surface/audit_pressure_row_support.py` parses the Eigen factorization diagnostic, maps logged scalar `Pressure` block rows back to P1 point indices, and classifies each row using the same support taxonomy as the accepted-step pressure-update audit.

For the Test10 VMS-disabled failure, the diagnostic row list is complete:

| Reported Pressure zero rows | Classified rows | Row/column match | Support classes |
| ---: | ---: | --- | --- |
| 5 | 5 | yes | `dry_or_inactive: 5` |

The five local Pressure rows are `180`, `183`, `250`, `649`, and `652`, matching global rows `3456`, `3459`, `3526`, `3925`, and `3928`. Under a direct local-row-to-point assumption, the corresponding `result_090.vtu` points have `ActiveFluid=0`, positive `phi`, no positive incident `WetVolumeFraction`, and zero saved pressure. The constraint-coverage audit below shows that direct mapping is not verified for this artifact, so this support audit is a clue rather than a proof of dry-row identity.

## Pressure Constraint-Coverage Audit

`Documentation/open_vessel_free_surface_pressure_constraint_coverage_diagnostic_20260605.md` checks the same zero-row set against the solve-time pressure constraint diagnostic immediately before factorization. The selected constraint is line `305` in the VMS-disabled Test10 log:

- `inactive_dof_runs=44-76|121-153|194-223|264-545|590-622|663-818`
- `inactive_dofs=567`
- `active_support_dofs=252`
- `support_mode=retained_cut_volume+cut_adjacent_facets`

None of the zero pressure rows `180`, `183`, `250`, `649`, or `652` fall in those inactive DOF runs. The same constraint reports `inactive_vertex_runs=168-545|630-818`, which differs from the inactive DOF runs, so direct VTU point support labels are marked unverified.

| Audit result | Index space | Count/status |
| --- | --- | ---: |
| Factorization zero pressure rows | pressure DOF space | 5 |
| Zero rows in solve-time inactive pressure DOF runs | pressure DOF space | 0 |
| Zero rows outside solve-time inactive pressure DOF runs | pressure DOF space | 5 |
| Direct local-row to point mapping supported | support label map | false |

The field block has `identity_rows=568`, one more than the parsed `567` inactive pressure DOFs, but that extra identity row does not cover the five factorization zero rows.

## Runtime Constraint Sample

The replay was rerun in:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_constraint_sample_20260605_case/`

with:

- `SVMP_NS_DISABLE_VMS=1`
- `SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS=180|183|250|649|652`

The sampled audit is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_constraint_sample_audit_20260605.json`

| Local pressure row | Global DOF | Entity | Vertex phi | Active DOF support | Inactive constraint |
| ---: | ---: | --- | ---: | ---: | ---: |
| 180 | 3456 | Vertex 82 | -0.0922709 | 1 | 0 |
| 183 | 3459 | Vertex 81 | -0.0558877 | 1 | 0 |
| 250 | 3526 | Vertex 83 | -0.0877238 | 1 | 0 |
| 649 | 3925 | Vertex 587 | -0.0922709 | 1 | 0 |
| 652 | 3928 | Vertex 586 | -0.0558877 | 1 | 0 |

This rules out inactive pressure identity coverage for these rows. They are active supported negative-side vertex pressure DOFs.

## Matrix Support Sample

The replay was rerun in:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_matrix_sample_20260605_case/`

with the same sampled rows plus:

- `SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC=1`

The merged audit is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_matrix_sample_audit_20260605.json`

It reports `matrix_sample_count=5`, `matrix_sampled_zero_rows_zero_row_count=5`, `matrix_sampled_zero_rows_zero_col_count=5`, and `matrix_sampled_zero_rows_zero_diag_count=5`. For local pressure rows `180`, `183`, `250`, `649`, and `652`, the pre-linear-solve matrix samples all have `row_abs_sum=0`, `col_abs_sum=0`, `diag=0`, `row_first_nonzero=none`, and `col_first_nonzero=none`.

This confirms the failure is present in the assembled Jacobian before the direct solver is called.

## VMS-Enabled Counterpart

The accepted baseline was rerun in:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_matrix_sample_20260605_case/`

with the same five sampled rows and `SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC=1`. The summary audit is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_matrix_sample_audit_20260605.json`

All five rows remain active supported and inactive-unconstrained. Unlike the VMS-disabled run, all five have nonzero row, column, and diagonal support immediately before the linear solve:

| Local pressure row | Global DOF | Row abs sum | Column abs sum | Diagonal | Row nonzeros |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 180 | 3456 | 7.87484e-09 | 7.87484e-09 | 3.93742e-09 | 4 |
| 183 | 3459 | 1.72991e-08 | 1.72991e-08 | 8.64955e-09 | 5 |
| 250 | 3526 | 1.72050e-08 | 1.72050e-08 | 8.60252e-09 | 5 |
| 649 | 3925 | 7.87484e-09 | 7.87484e-09 | 3.93742e-09 | 4 |
| 652 | 3928 | 1.72991e-08 | 1.72991e-08 | 8.64955e-09 | 5 |

The row/column field-block sums are `phi:0|Velocity:0|Pressure:<nonzero>` for every sampled row. Since pressure ghost penalty is still enabled in the VMS-disabled replay, the support that appears only in this accepted baseline is residual-based VMS/PSPG pressure-pressure stabilization, not Galerkin velocity-pressure coupling.

Follow-up aggregate support-rank evidence in `Documentation/open_vessel_free_surface_active_pressure_support_rank_guard_20260605.md` shows the sampled rows are part of a larger unsupported class. Both VMS-enabled and VMS-disabled Test10 step90 controls have nine unconstrained pressure rows with zero `Velocity` row-block support: `180|183|190|193|250|649|652|659|662`. With VMS enabled all nine are pressure-only rows; with VMS disabled five become structural zero rows and four retain non-VMS pressure-block support. The env-gated support-rank guard fails the VMS-enabled replay before it accepts the `1,075.558213 Pa` pressure update. A diagnostic clamp of those nine rows reduces the accepted update to `711.224854 Pa` but leaves the worst row at global `3356`; thresholded weak-coupling clamps reduce the same update to `352.269145 Pa` and `182.117016 Pa`. The zero-coupling class is therefore not the complete accepted-spike mechanism, but the weak pressure/velocity coupling scale is now directly implicated.

## Interpretation

The VMS/PSPG branch is not the sole immediate source of the Test02 accepted pressure jump. Disabling it leaves the pressure equations residual-consistent and makes the active/wet pressure excursion larger.

For Test10, VMS/PSPG is carrying or masking essential active pressure-row support/rank in the active-domain system. Removing it exposes active supported vertex pressure rows with zero assembled matrix rows/columns before a nonlinear update is accepted. With VMS enabled, the same singular subset is supported only by a tiny pressure-pressure stabilization stencil. The larger nine-row zero-velocity-coupling class remains present in both controls. The accepted worst row is outside that class but has small nonzero velocity-block support, and targeted clamps show the accepted update follows that weak-coupling scale. Therefore "disable VMS" is ruled out as a fix, and the pressure issue should be treated as active-volume pressure/continuity plus active pressure-row rank and coupling-scale consistency, not a standalone PSPG bug, inactive-constraint omission, or factorization-reporting artifact.

Follow-up: `Documentation/open_vessel_free_surface_pressure_row_contribution_diagnostic_20260605.md` implements the row-level contribution decomposition on the same line-search cut context. It shows the accepted full-wet pressure-jump rows have zero pressure ghost-penalty contribution and are residual-consistent because Galerkin continuity cancels VMS/PSPG to roundoff.

The sharper next target is now coupled pressure-update/null behavior on the same line-search cut context:

- Why the saddle-point/VMS system admits large full-wet pressure increments while satisfying the pressure equations to roundoff.
- Why active supported Test10 vertex pressure rows have zero assembled rows/columns without VMS/PSPG.
- Why weak-but-nonzero pressure/velocity coupling still admits a large accepted update after zero-coupling rows are clamped.
- Whether pressure ghost penalty shapes the bad pressure branch away from the accepted-jump rows.

## Artifacts

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_vms_disabled_20260605_case/run.log`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_vms_disabled_pressure_update_direct_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_vms_disabled_cut_context_pressure_residual_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_20260605_case/run.log`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_cut_context_pressure_residual_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_pressure_row_support_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_pressure_constraint_coverage_audit_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_constraint_sample_20260605_case/run.log`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_constraint_sample_audit_20260605.json`
- `Documentation/open_vessel_free_surface_pressure_constraint_coverage_diagnostic_20260605.md`
- `Documentation/open_vessel_free_surface_pressure_row_contribution_diagnostic_20260605.md`
