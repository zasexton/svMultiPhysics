# JIT Kernel Refactor Qualification — 2026-03-28

This note records the current qualification state for
`Documentation/plan_jit_kernel_optimization_refactor.md`.

## 2026-03-30 Verification Update

- Re-ran `tests/cases/fluid/pipe_simple/solver_perf_oop.xml` on the current tree
  with:
  - default exact production path
  - `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`
  - `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`
    `SVMP_FE_COMPARE_MONOLITHIC_COMPILED=1`
- All three runs now complete cleanly on the current `build-unit` executable.
- The compiled compare hook is still active in `StandardAssembler`, so the
  archived `pipe_simple` mismatch below should now be read as historical
  evidence from 2026-03-28, not as the current tree state.
- Monolithic compiled dispatch remains opt-in, but the remaining blocker is now
  performance qualification rather than a known real-case residual mismatch.
- Coupled helper splitting remains disabled pending its own re-qualification.

## Production Status

- The qualified production mixed-cell path is `MonolithicCellKernel` with shared-geometry exact block execution.
- Monolithic compiled dispatch remains opt-in only:
  `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`.
- Coupled helper splitting inside the compiled monolithic dispatcher is disabled pending re-qualification.
- Current reason: the compiled monolithic path is still materially slower than
  the exact production path on the qualified fluid matrix, and coupled helper
  splitting has not yet been re-qualified.

Archived mismatch artifact:

- `Documentation/qualification_logs/20260328_monolithic_exact_default/pipe_simple_compiled_compare_failure.log`

The default exact path is therefore the qualified production path because it
restores clean nonlinear convergence on the full qualified OOP fluid matrix.

## Archived Artifacts

- Raw qualification logs:
  `Documentation/qualification_logs/20260328_monolithic_exact_default/`
- Phase 0 baseline archive:
  `Documentation/jit_kernel_refactor_phase0_baseline_20260323.md`
- Phase 0 summary report:
  `tests/perf_benchmark_report_20260323.md`

## Unit Re-Verification

The following targeted FE unit coverage was re-run and passed:

- `BackendParity.ResidualPath_JitMonolithicSparsityMatchesFallback`
- `BackendParity.CapabilityOnOff_ResolvedInsertionParity`
- `MixedFormPerformance.InstallFormulation_MonolithicJITParity_VersusPerBlockFallback`
- `MixedFormPerformance.InstallFormulation_MonolithicJITParity_TransientMixedNoVMS`
- `MixedFormPerformance.InstallFormulation_MonolithicJITParity_TransientVMSResidual`
- `MixedFormPerformance.InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientMixedNoVMS`
- `MixedFormPerformance.InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientVMSResidual`
- `MonolithicCoupling.MixedJacobianBlockFDVerification`

Additional focused check:

- With `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1` and
  `SVMP_FE_COMPARE_MONOLITHIC_COMPILED=1`, the unit-scale monolithic parity
  tests still pass.
- On 2026-03-30, the same compare hook was re-run on the real `pipe_simple`
  case and completed cleanly on the current tree, so the archived mismatch no
  longer reproduces.
- The compiled monolithic path is still not production-qualified because it
  remains much slower than the default exact path.

## Qualified Case Matrix

All default qualification runs below were executed with:

- `OMP_NUM_THREADS=1`
- `SVMP_FSILS_GMRES_REORTH=off`
- default exact monolithic production path

All accepted steps in these default runs reported `converged=1`. No qualified
run relied on nonlinear fallback acceptance.

| Case | Warm runtime | Phase 0 warm runtime | Ratio | Newton iters/step | Final residual trace | Qualification result |
|:---|---:|---:|:---|:---|:---|:---|
| `Channel2D` | `45.84 s` | `34.4 s` | `1.33x` slower | `6,4,3,3,3,3,2,2,2,2` | `3.95e-05 -> 1.87e-10` | converges cleanly; runtime target still open |
| `Channel2D_Simple` | `3.31 s` | `1.1 s` | `3.01x` slower | `3,3,3,3,3,3,3,3,3,3` | `4.29e-06 -> 5.42e-09` | converges cleanly; cheap perf-smoke target still open |
| `vortex_shedding` | `25.40 s` | `23.2 s` | `1.10x` slower | `4,5` | `5.73e-08, 9.11e-07` | stable transient convergence; runtime stayed within a small delta |
| `pipe_RCR_3d` | `14.22 s` | `8.2 s` | `1.73x` slower | `4,3` | `5.98e-14, 7.90e-11` | monolithic `RCR` remains excellent; runtime target still open |
| `pipe_RCR_3d_RCRCR` | `17.67 s` | n/a | n/a | `3,2` | `9.58e-14, 2.50e-11` | monolithic `RCRCR` converges excellently |
| `pipe_simple` | `19.32 s` | `10.3 s` | `1.88x` slower | `5,5` | `1.08e-09, 1.96e-08` | stable on the default exact path; fast-path target still open |
| `iliac_artery` | `140.87 s` | `81.6 s` | `1.73x` slower | `5,5` | `4.68e-09, 2.15e-09` | converges cleanly; runtime target still open |

## Comparison Notes

- The final runtime comparison to the archived Phase 0 baseline is now explicit
  for every case that had a Phase 0 warm-run record.
- The final Newton/convergence comparison to the Phase 0 baseline is now
  explicit for all archived baseline cases.
- `pipe_RCR_3d/solver_perf_oop_rcrcr.xml` was not part of the 2026-03-23
  baseline archive, so there is no Phase 0 runtime/Newton reference for that
  case.
- No partitioned `RCRCR` comparison artifact was found in the repo, so the
  partitioned-Newton and partitioned-runtime checklist items remain open.

## Archived Compiled Monolithic Mismatch

The archived 2026-03-28 real-case compare failure was:

- case: `pipe_simple`
- mode:
  `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`
  `SVMP_FE_COMPARE_MONOLITHIC_COMPILED=1`
- first mismatch:
  `cell=0`
  `block=0`
  `test_field=0`
  `trial_field=0`
- diff summary:
  matrix diff `1.11e-16`
  vector diff `3.63e+00`

Interpretation:

- The monolithic compiled Jacobian remains numerically consistent at the first
  compared block entry.
- On 2026-03-28, the compiled monolithic residual diverged from the exact
  fallback on the real fluid formulation.
- On 2026-03-30, that archived compare failure no longer reproduced on the
  current tree.
- The compiled monolithic dispatcher therefore remains behind the explicit
  opt-in env flag for performance reasons and pending helper-splitting
  re-qualification, not because of a currently known `pipe_simple` mismatch.
