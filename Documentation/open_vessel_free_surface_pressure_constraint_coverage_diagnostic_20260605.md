# Open-Vessel Pressure Constraint Coverage Diagnostic

Date: 2026-06-05

Scope: targeted follow-up to the VMS/PSPG disabled Test10 singularity and the pressure-row support audit. This asks whether pressure rows reported by factorization were covered by the solve-time active-side pressure Dirichlet constraint, whether saved VTU point support labels can be trusted for those pressure-row indices, which solve-time entities those pressure DOFs actually belong to, and whether the assembled Jacobian has row/column support for those DOFs immediately before factorization.

## Code Change

- `tests/cases/fluid/open_vessel_free_surface/audit_pressure_constraint_coverage.py`
  parses `LevelSetActiveSideVertexDirichletConstraint`, `newton_matrix_support_sample`, and Eigen factorization block summaries from a solver log. It selects the last matching pressure constraint diagnostic before the selected factorization failure, annotates each logged zero pressure row with solve-time constraint coverage and optional matrix row/column support samples, and marks saved VTU support labels unverified when inactive pressure DOF runs differ from inactive vertex runs.
- `tests/test_open_vessel_pressure_constraint_coverage_audit.py`
  covers run expansion, selection of the last constraint before factorization, merge of optional saved-state row-support JSON, mismatch classification, and matrix-support sample merging.
- `tests/cases/fluid/open_vessel_free_surface/audit_pressure_matrix_support_samples.py`
  summarizes `newton_matrix_support_sample` diagnostics for accepted controls without requiring a factorization failure, including parsed field-block row/column sums, aggregate `active_pressure_support_rank` diagnostics, and zero/nonzero coupling-block counts.
- `tests/test_open_vessel_pressure_matrix_support_samples.py`
  covers standalone matrix-support sample parsing and merge with runtime constraint samples.
- `Code/Source/solver/FE/Constraints/LevelSetActiveSideVertexDirichletConstraint.cpp`
  now has an environment-gated sampled-DOF diagnostic. Set `SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS=<rows>` or `SVMP_LEVEL_SET_ACTIVE_CONSTRAINT_SAMPLE_DOFS=<rows>` to log `diagnostic=level_set_active_side_vertex_constraint_sample` with the local/global DOF, entity kind/id, active support, inactive-constraint state, and vertex phi when the DOF belongs to a vertex.
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  now has an environment-gated matrix-support sampler. Set `SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC=1` with `SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS=<pressure-local-rows>` or `SVMP_NEWTON_MATRIX_SUPPORT_SAMPLE_DOFS=<global-dofs>` to log `diagnostic=newton_matrix_support_sample` with row/column absolute sums, field-block row/column absolute sums, numeric-entry counts, diagonal value, and first nonzero row/column entries immediately before the linear solve.
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  now also has an aggregate active-pressure support-rank diagnostic. Set `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_DIAGNOSTIC=1` to log unconstrained pressure rows with zero velocity-block support; set `SVMP_ACTIVE_PRESSURE_SUPPORT_RANK_GUARD=1` to fail before the linear solve when such rows are present.

These diagnostics do not change solver behavior.

## Result

Artifact:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_pressure_constraint_coverage_audit_20260605.json`

For the VMS-disabled Test10 step90 replay, the selected pressure constraint diagnostic is line `305`, immediately before the line `825` factorization failure. It reports:

- `support_mode=retained_cut_volume+cut_adjacent_facets`
- `total_dofs=819`
- `inactive_dofs=567`
- `inactive_dof_runs=44-76|121-153|194-223|264-545|590-622|663-818`
- `active_support_dofs=252`

The factorization failure reports five local `Pressure` zero rows: `180`, `183`, `250`, `649`, and `652`. None are in the inactive pressure DOF runs above.

The same constraint diagnostic also reports `inactive_vertex_runs=168-545|630-818`, which differs from `inactive_dof_runs`. Therefore a direct `local_pressure_row -> VTU point_index` mapping is not supported for this artifact, even though the pressure field has 819 DOFs and the VTU has 819 points.

| Audit result | Count/status |
| --- | ---: |
| Factorization zero pressure rows | 5 |
| Zero rows in solve-time inactive pressure DOF runs | 0 |
| Zero rows outside solve-time inactive pressure DOF runs | 5 |
| Direct local-row to point mapping supported | false |
| Saved support labels marked unverified | 5 |

The field block also reports `identity_rows=568`, which is one more than the `567` inactive DOFs parsed from the constraint runs. That extra identity row is consistent with a pressure anchor/gauge row, but it does not cover the five factorization zero rows.

## Runtime Sample Result

Control directory:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_constraint_sample_20260605_case/`

The replay was rerun with:

- `SVMP_NS_DISABLE_VMS=1`
- `SVMP_ACTIVE_PRESSURE_CONSTRAINT_SAMPLE_DOFS=180|183|250|649|652`

It failed at the same direct factorization stage. The sampled audit is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_constraint_sample_audit_20260605.json`

| Local pressure row | Global DOF | Entity | Vertex phi | Active sign | Active DOF support | Inactive constraint |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 180 | 3456 | Vertex 82 | -0.0922709 | 1 | 1 | 0 |
| 183 | 3459 | Vertex 81 | -0.0558877 | 1 | 1 | 0 |
| 250 | 3526 | Vertex 83 | -0.0877238 | 1 | 1 | 0 |
| 649 | 3925 | Vertex 587 | -0.0922709 | 1 | 1 | 0 |
| 652 | 3928 | Vertex 586 | -0.0558877 | 1 | 1 | 0 |

All five factorization zero rows are solve-time active supported vertex pressure DOFs. None should be inactive-constrained.

## Matrix Support Sample Result

Control directory:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_matrix_sample_20260605_case/`

The replay was rerun with the same VMS-disabled and constraint-sample settings plus:

- `SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC=1`

It failed at the same direct factorization stage. The merged audit is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_matrix_sample_audit_20260605.json`

The standalone matrix-support summary is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_vms_disabled_matrix_sample_summary_20260605.json`

Summary:

| Audit result | Count/status |
| --- | ---: |
| Factorization zero pressure rows | 5 |
| Runtime sampled zero rows | 5 |
| Runtime sampled rows with active DOF support | 5 |
| Matrix sampled zero rows | 5 |
| Matrix sampled rows with zero row sum | 5 |
| Matrix sampled rows with zero column sum | 5 |
| Matrix sampled rows with zero diagonal | 5 |
| Matrix sampled rows with zero Velocity row-block support | 5 |
| Matrix sampled rows with nonzero Pressure row-block support | 0 |

The matrix sampler reports this immediately before factorization:

| Local pressure row | Global DOF | Row abs sum | Column abs sum | Diagonal | Row nonzeros | Column nonzeros |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 180 | 3456 | 0 | 0 | 0 | 0 | 0 |
| 183 | 3459 | 0 | 0 | 0 | 0 | 0 |
| 250 | 3526 | 0 | 0 | 0 | 0 | 0 |
| 649 | 3925 | 0 | 0 | 0 | 0 | 0 |
| 652 | 3928 | 0 | 0 | 0 | 0 | 0 |

This rules out a factorization-only reporting artifact. The five active supported pressure DOFs have no assembled matrix row, column, or diagonal support before the direct solver is called.

## VMS-Enabled Counterpart

Control directory:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_matrix_sample_20260605_case/`

This reruns the accepted VMS-enabled Test10 step90 replay with the same five sampled pressure-local rows and `SVMP_NEWTON_MATRIX_SUPPORT_DIAGNOSTIC=1`. The summary audit is:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_matrix_sample_audit_20260605.json`

Summary:

| Audit result | Count/status |
| --- | ---: |
| Runtime sampled rows with active DOF support | 5 |
| Runtime sampled rows with inactive constraint | 0 |
| Matrix sampled rows with nonzero row sum | 5 |
| Matrix sampled rows with nonzero column sum | 5 |
| Matrix sampled rows with nonzero diagonal | 5 |
| Matrix sampled rows with zero Velocity row-block support | 5 |
| Matrix sampled rows with nonzero Pressure row-block support | 5 |
| Accepted worst pressure update | 1,075.558213 Pa at dof 3356 |

The pre-linear-solve matrix samples are:

| Local pressure row | Global DOF | Row abs sum | Column abs sum | Diagonal | Row field sums |
| ---: | ---: | ---: | ---: | ---: | --- |
| 180 | 3456 | 7.87484e-09 | 7.87484e-09 | 3.93742e-09 | `phi:0|Velocity:0|Pressure:7.87484e-09` |
| 183 | 3459 | 1.72991e-08 | 1.72991e-08 | 8.64955e-09 | `phi:0|Velocity:0|Pressure:1.72991e-08` |
| 250 | 3526 | 1.72050e-08 | 1.72050e-08 | 8.60252e-09 | `phi:0|Velocity:0|Pressure:1.72050e-08` |
| 649 | 3925 | 7.87484e-09 | 7.87484e-09 | 3.93742e-09 | `phi:0|Velocity:0|Pressure:7.87484e-09` |
| 652 | 3928 | 1.72991e-08 | 1.72991e-08 | 8.64955e-09 | `phi:0|Velocity:0|Pressure:1.72991e-08` |

The row/column field-block sums are zero in `phi` and `Velocity` and nonzero only in `Pressure`. Since pressure ghost penalty remains enabled in both the VMS-enabled and VMS-disabled replays, the support that disappears in the disabled replay is the residual-based VMS/PSPG pressure-pressure stabilization, not Galerkin velocity-pressure coupling and not the pressure ghost penalty.

## Aggregate Support-Rank Guard

The aggregate diagnostic in `Documentation/open_vessel_free_surface_active_pressure_support_rank_guard_20260605.md` scans all unconstrained pressure rows before `linear.solve()`.

| Control | Unconstrained pressure rows | Zero Velocity row-block rows | Pressure-only row-block rows | Zero total pressure rows | Local zero-Velocity rows |
| --- | ---: | ---: | ---: | ---: | --- |
| VMS enabled | 251 | 9 | 9 | 0 | `180|183|190|193|250|649|652|659|662` |
| VMS disabled | 251 | 9 | 4 | 5 | `180|183|190|193|250|649|652|659|662` |
| VMS enabled + guard | 251 | 9 | 9 | 0 | `180|183|190|193|250|649|652|659|662` |

The guard replay fails before accepting the VMS-enabled `1,075.558213 Pa` pressure update. This proves the unsupported pressure-row class is detectable before acceptance. It also shows the five sampled singular rows are a structural-zero subset of a larger nine-row class with no Galerkin velocity coupling.

A diagnostic-only clamp of the nine zero-velocity-coupled rows reduces the accepted Test10 update to `711.224854 Pa`, but does not remove it. The accepted worst row remains global `3356` / pressure-local row `80`, which is active supported and has nonzero but small `Velocity` row-block support. Weak-coupling clamp controls at `3.1e-4` and `3.3e-4` reduce the accepted update further to `352.269145 Pa` and `182.117016 Pa`. Therefore the nine-row class is a real support/rank hazard, but the accepted-jump mechanism also involves weak pressure/velocity coupling scale.

## Interpretation

The earlier pressure-row support audit classified local rows `180`, `183`, `250`, `649`, and `652` as dry/inactive by mapping local pressure rows directly to VTU point indices. This coverage audit shows that mapping is not verified in the solve-time constraint log: inactive vertex runs and inactive pressure DOF runs are different index spaces for this case.

The runtime sample resolves the offline mapping ambiguity. The direct local-row-to-VTU-point support audit was misleading for this artifact because local pressure DOF order differs from VTU point order. The factorization zero rows are not dry/inactive rows; they are active negative-side vertex pressure DOFs with active support.

This rules out inactive pressure identity coverage as the cause of these five zero rows, and the matrix sampler rules out a post-assembly/factorization-only artifact. The VMS-disabled singularity is instead an active pressure-row support/rank failure: rows that the active-support constraint intentionally leaves free have no assembled row or column support when the VMS/PSPG branch is removed. The VMS-enabled counterpart shows the same active pressure rows are supported only by a small pressure-pressure stabilization stencil. The aggregate guard expands the issue to nine unconstrained rows with no velocity-block support. The next target is to replace the active pressure support/rank criterion for zero-coupling rows and address the weak pressure/velocity coupling scale that the thresholded clamp probes now implicate.

## Verification

- `pytest -q tests/test_open_vessel_pressure_constraint_coverage_audit.py tests/test_open_vessel_pressure_row_support_audit.py tests/test_open_vessel_pressure_contribution_samples.py`
- `pytest -q tests/test_open_vessel_pressure_matrix_support_samples.py`
- `python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_pressure_constraint_coverage.py`
- `python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_pressure_matrix_support_samples.py`
- `cmake --build build/svMultiPhysics-build --target svmultiphysics test_fe_constraints -j 4`
- `build/svMultiPhysics-build/bin/test_fe_constraints --gtest_filter='LevelSetActiveSideVertexDirichletConstraint.*'`
