# D18 Solver Tolerance Runs - 2026-05-13

## Purpose

This log records the first D18 one-step solver-tolerance attempts after wet-side
cut-volume Navier-Stokes assembly was enabled. The active-domain assembly path
remained active in every run; the open item is the linear and nonlinear solver
floor reached by the current coupled `phi`, `Velocity`, and `Pressure` system.

## Passing Reference

- Run directory: `/tmp/svmp_d18_active_smoke_serial_Hh4ER8`
- Solver XML basis: D18 fixture with one time step and restart output every
  step.
- Nonlinear tolerance: `1.0e-4`
- Linear relative tolerance: `1.0e-4`
- Linear absolute tolerance: `1.0e-4`
- Active-domain log:
  `Active_domain=LevelSetNegative Active_domain_method=CutVolume`
- Wet volume: `1.5869994907267658e-03`
- Cut-cell wet volume: `1.57407e-04`
- Result: converged in 2 nonlinear iterations.
- Final residual norm: `1.2282662820623137e-05`
- Linear summary: converged, 52 iterations, reported relative residual
  `1.1691485632016713e-03`

## Strict Tolerance Attempts

| Run directory | Key settings | Result |
| --- | --- | --- |
| `/tmp/svmp_d18_tight_smoke_serial_Ep2YtO` | `1e-6` nonlinear, `1e-6` linear relative, `1e-10` linear absolute, diagonal FSILS GMRES | Did not converge. True residual check reported `|Ax-b|=3.09354e-05`, relative `5.79653e-03`, target `5.33688e-09`. |
| `/tmp/svmp_d18_ns_smoke_serial_Ao1FKa` | Fluid `LS` changed to `NS` in a temporary file | Did not change the global solver. The coupled solve still used the first equation `LS` block and reported `block_layout=[phi(0:1), Velocity(1:3), Pressure(4:1)]`. |
| `/tmp/svmp_d18_rcs_smoke_serial_NbPZ3c` | Row-column scaling, strict tolerances, 4 Newton iterations | Linear solves reached the `1e-6` relative target, but nonlinear solve stopped at 4 iterations with residual `6.9218570689534820e-01`. |
| `/tmp/svmp_d18_rcs_newton8_smoke_serial_WndXWM` | Row-column scaling, strict tolerances, 8 Newton iterations | Later true residual check reported `|Ax-b|=6.26819e-06`, relative `9.0561e-06`, target `6.92151e-07`. |
| `/tmp/svmp_d18_rcs_budget_smoke_serial_YbphyL` | Row-column scaling, `max_iter=16200`, strict tolerances | Did not converge. True residual check reported `|Ax-b|=1.65728e-05`, relative `4.29542e-05`, target `3.85826e-07`. |
| `/tmp/svmp_d18_absfloor_smoke_serial_4KnEtw` | Strict nonlinear and linear relative tolerance, linear absolute floor `1e-4`, 8 Newton iterations | Linear solves were accepted, but the nonlinear residual stopped at `1.2190292492699640e-05`, above the `1e-6` target. |
| `/tmp/svmp_d18_explicit_rank1_smoke_serial_54exjx` | Strict tolerances, explicit rank-one update switch | Same true-residual result as the strict diagonal run. |
| `/tmp/svmp_d18_pgmres_smoke_serial_o7EblH` | First equation `LS type="PGMRES"` with strict tolerances | Same true-residual result as the strict diagonal GMRES run. |
| `/tmp/svmp_d18_cutscale_smoke_serial_zC0vgf` | `Use_cut_metadata_scale=true` with strict tolerances | Process terminated with a segmentation fault immediately after active-domain cut context setup. |
| `/tmp/svmp_d18_nocutstab_smoke_serial_ACeawl` | Cut-cell penalty terms disabled with strict tolerances | Same true-residual result as the strict diagonal run. |
| `/tmp/svmp_d18_seqlevel_smoke_serial_APR8jr` | Level-set equation marked non-coupled in a temporary file; fluid `LS type="NS"` | Global block layout still included `phi`, `Velocity`, and `Pressure`; same true-residual result as the strict diagonal run. |

## Conclusions

- The D18 active-domain volume diagnostics stayed consistent across the solver
  trials, so these runs did not indicate a wet-volume assembly regression.
- The strict `1e-6` nonlinear and linear relative tolerance path is blocked by
  the current coupled FSILS GMRES solver floor.
- The current D18 fixture should keep the last passing `1e-4` solver tolerances
  until the coupled solver floor is addressed.
- `Use_cut_metadata_scale=true` is not currently a usable stabilization option
  for this active-domain D18 path because it terminates during first-step
  assembly.

## Follow-Up Items

- [ ] Add a robust solver path for the coupled `phi`, `Velocity`, and `Pressure`
      layout, or restructure D18 so the Navier-Stokes solve can use a
      velocity-pressure solver block without the level-set field in the same
      linear system.
- [ ] Fix the active-domain `Use_cut_metadata_scale=true` assembly crash before
      using metadata-scaled cut-cell stabilization for D18.
- [ ] Re-run the `1e-6` nonlinear and linear relative tolerance test after the
      solver path is corrected.
- [ ] Keep recording nonlinear iteration counts, linear iteration counts, and
      residual norms for each D18 and D38 qualification attempt.
