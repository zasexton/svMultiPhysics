# Open-Vessel Pressure-Row Contribution Diagnostic

Date: 2026-06-05

Scope: targeted follow-up to the VMS/PSPG disable and pressure-row support audits. This diagnostic asks which assembled pressure-row contribution is present at the accepted Test02/Test10 pressure-jump rows on the line-search cut context.

## Code Change

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
  now installs diagnostic-only pressure-row operators when `SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC=1`:
  - `equations_diagnostic_ns_galerkin_continuity`
  - `equations_diagnostic_ns_active_continuity`
  - `equations_diagnostic_ns_vms_pspg`
  - `equations_diagnostic_ns_pressure_ghost_penalty`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
  assembles those operators vector-only under the same flag and logs `Pressure` field residual diagnostics before and after constrained-row zeroing. `SVMP_NEWTON_FIELD_RESIDUAL_SAMPLE_DOFS` or `SVMP_PRESSURE_ROW_CONTRIBUTION_SAMPLE_DOFS` adds exact sampled row values.
- `tests/cases/fluid/open_vessel_free_surface/audit_pressure_contribution_samples.py`
  extracts the sampled contribution rows into JSON.

These operators are not production residual changes; they are only installed and assembled for explicit diagnostic runs.

## Replay Runs

Control directories:

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_contribution_diag_20260605_case/`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_contribution_diag_20260605_case/`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_weak_front_contribution_20260605_case/`

Both were run as one-step replays with:

- `SVMP_NS_PRESSURE_ROW_CONTRIBUTION_DIAGNOSTIC=1`
- `SVMP_ACTIVE_PRESSURE_UPDATE_DIAGNOSTIC=1`
- `SVMP_NEWTON_FIELD_RESIDUAL_DIAGNOSTIC=1`
- `SVMP_ACTIVE_PRESSURE_RESIDUAL_DIAGNOSTIC=1`

Sampled rows:

- Test02: accepted-jump dof `10676`, residual-worst dof `11913`, initial/gauge large-row dof `11875`.
- Test10: accepted-jump dof `3356`, residual-worst dof `3369`, gauge dof `3386`.
- Test10 weak front: dofs `3353`, `3354`, `3356`, and `3358`, corresponding to local pressure rows `77`, `78`, `80`, and `82` from the support-rank weak-coupling probes.

## Results

Line-search contribution values at the accepted pressure-jump rows:

| Case | Accepted jump row | Accepted pressure update | Galerkin continuity | VMS/PSPG | Pressure ghost penalty | Total Pressure residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test02 step382 replay | 10676 | 2,112,209.8407 Pa | -3.99815e-04 | +3.99815e-04 | 0 | -7.59551e-12 |
| Test10 step90 replay | 3356 | 1,075.558213 Pa | -8.48247e-06 | +8.48247e-06 | 0 | -7.44495e-13 |

Line-search residual-worst sampled rows:

| Case | Residual-worst sampled row | Galerkin continuity | VMS/PSPG | Pressure ghost penalty | Total Pressure residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test02 | 11913 | -4.84943e-04 | +2.32294e-07 | +4.84693e-04 | -1.78358e-08 |
| Test10 | 3369 | +3.51369e-06 | -1.92990e-07 | -1.15178e-05 | -8.19709e-06 |

The pressure gauge row is removed by constraints in the contribution diagnostic (`sampled_dofs` shows zero after constrained-row zeroing for Test10 dof `3386`). The large pre-constraint `553.267 Pa` row is therefore expected gauge inhomogeneity, not the accepted pressure-jump mechanism.

The weak-front follow-up sampled the rows selected by the support-rank weak-coupling clamp controls. At the accepted line-search state, every sampled row again has zero pressure ghost-penalty contribution and Galerkin continuity cancels VMS/PSPG to near roundoff:

| Test10 global dof | Local pressure row | Galerkin continuity | VMS/PSPG | Pressure ghost penalty | Total Pressure residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3353 | 77 | -4.37365e-07 | +4.37365e-07 | 0 | -3.57953e-14 |
| 3354 | 78 | +5.52131e-06 | -5.52131e-06 | 0 | -3.41646e-13 |
| 3356 | 80 | -8.48247e-06 | +8.48247e-06 | 0 | -7.44495e-13 |
| 3358 | 82 | -5.69237e-06 | +5.69237e-06 | 0 | -1.19352e-12 |

## Interpretation

This rules out the direct explanation that the accepted Test02/Test10 pressure-jump rows are being driven by a nonzero pressure ghost-penalty residual row. At both accepted-jump rows, the pressure ghost-penalty contribution is exactly zero in the sampled line-search diagnostic.

The accepted-jump rows are residual-consistent because Galerkin continuity and VMS/PSPG cancel to roundoff. The pressure ghost penalty is still active elsewhere: in Test02 it cancels most of the active-continuity residual at row `11913`, and in Test10 it dominates the residual-worst sampled row `3369`. That means the ghost penalty remains a branch-shaping mechanism, but not the direct residual source at the accepted full-wet jump rows.

The weak-front contribution replay extends that conclusion from the original Test10 worst row to the rows exposed by the weak-coupling clamp probes. Rows `77`, `78`, `80`, and `82` are not accepting large pressure updates because of a nonzero local pressure residual contribution; the sampled pressure equations are already satisfied to roundoff after the line-search update. Together with the retained-support audit, this rules out both tiny retained support and direct residual forcing for the weak front, leaving pressure/velocity coupling scale and coupled pressure-update/null behavior as the active mechanism.

Follow-up constraint and matrix support coverage in `Documentation/open_vessel_free_surface_pressure_constraint_coverage_diagnostic_20260605.md` also shows that the VMS-disabled Test10 zero rows are active supported negative-side vertex pressure DOFs, not inactive-constraint omissions, and that all five have zero assembled row, column, and diagonal support immediately before factorization. The VMS-enabled counterpart gives the same five rows only a tiny pressure-pressure stencil. The aggregate support-rank guard in `Documentation/open_vessel_free_surface_active_pressure_support_rank_guard_20260605.md` expands this to nine unconstrained Test10 pressure rows with zero `Velocity` row-block support; the five singular rows are the structural-zero subset when VMS/PSPG is disabled. Clamping those nine rows reduces but does not remove the accepted Test10 pressure jump, and the accepted worst row has weak nonzero velocity coupling. Thresholded weak-coupling clamps reduce the same update to `352.269145 Pa` and then `182.117016 Pa`, so the coupling scale is no longer only circumstantial. The results point at the same class of issue from opposite sides: VMS-enabled accepted rows are residual-consistent but physically suspect, while VMS-disabled rows expose active pressure-row support/rank loss in the assembled Jacobian.

The next code target should not be another scalar ghost-penalty sweep. The more credible target is the pressure-update/pressure-null behavior and active pressure-row support of the coupled active-volume solve: why the saddle-point/VMS system admits large pressure increments at full-wet rows while satisfying the pressure equation to roundoff, why active supported pressure rows can be unconstrained despite zero Galerkin velocity coupling, and why weak nonzero velocity coupling remains insufficient for the accepted-jump row. Any formulation fix should preserve hydrostatic/linear patch consistency and be tested against the sampled jump rows above.

## Artifacts

- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_contribution_diag_20260605_case/run.log`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_replay_abs_only_prune1e5_step382_contribution_diag_samples_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_contribution_diag_20260605_case/run.log`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_contribution_diag_samples_20260605.json`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_weak_front_contribution_20260605_case/run_weak_front_contribution.log`
- `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_replay_cap3_step90_weak_front_contribution_audit_20260605.json`
- `Documentation/open_vessel_free_surface_pressure_constraint_coverage_diagnostic_20260605.md`
