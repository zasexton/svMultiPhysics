# D18 Solver-Control Routing And Strict-Tolerance Attempts - 2026-05-13

## Scope

This note records the D18 one-step solver-control routing check after wet-side
cut-volume Navier-Stokes assembly, hydrostatic initialization, gauge placement,
and MPI active-domain assembly were corrected.

The passing MPI-4 full-run evidence remains:

- `Documentation/qualification_logs/dam_break_d18_full_run_20260513/README.md`
- Commit: `a7020d96d792edb2fb7d99e439da331aa1677fb6`
- D18 settings: active domain `LevelSetNegative`, method `CutVolume`,
  nonlinear tolerance `1e-4`, linear relative tolerance `1e-4`, linear absolute
  tolerance `1e-4`

## Routing Finding

Before this fix, coupled D18 runs with the first equation set to `level_set`
selected the global linear solver backend, solver options, and Newton controls
from that first equation. Changing the fluid equation solver settings in a
temporary XML file therefore did not affect the global coupled solve.

The corrected routing keeps the existing equation order and coupled block
layout, but selects solver controls from the first `fluid` equation when the
first equation is `level_set`.

## Default-Path Verification

The committed D18 fixture was copied to
`/tmp/svmp_d18_route_default_onestep_9FgkJT` and shortened to one time step.
The run completed successfully.

Key log lines:

- Solver controls:
  `method=gmres preconditioner=diagonal rel_tol=1.0e-04 abs_tol=1.0e-04 max_iter=8100`
- Block layout:
  `block_layout=[phi(0:1), Velocity(1:3), Pressure(4:1)]`
- Transient controls:
  `newton(max_it=8, min_it=1, abs_tol=1.0e-04, rel_tol=1.0e-04)`
- One-step result:
  `success=1 steps_taken=1 final_time=5.0e-04`

## Strict-Tolerance Attempts

| Temporary run directory | Key settings | Result |
| --- | --- | --- |
| `/tmp/svmp_d18_strict_outer_onestep_1rTGNv` | BlockSchur layout grouping `phi+Velocity` and `Pressure`, nonlinear `1e-6`, linear relative `1e-6`, linear absolute `1e-10` | Stopped on true residual check: `|Ax-b|=6.17121e-05`, relative `1.00204e-05`, target `6.15864e-06`. |
| `/tmp/svmp_d18_strict_gmres_onestep_SKdq6G` | Monolithic GMRES, diagonal scaling, nonlinear `1e-6`, linear relative `1e-6`, linear absolute `1e-10` | Stopped on true residual check: `|Ax-b|=3.59382e-05`, relative `6.73385e-03`, target `5.33695e-09`. |
| `/tmp/svmp_d18_strict_gmres_fluid_rcs_onestep_JHLR3s` | Monolithic GMRES, row-column scaling, nonlinear `1e-6`, linear relative `1e-6`, linear absolute `1e-10` | Stopped on true residual check: `|Ax-b|=1.65611e-05`, relative `4.29243e-05`, target `3.85821e-07`. |
| Temporary BICGS row-column-scaling runs | BICGS, row-column scaling, strict linear and nonlinear tolerances, larger linear iteration budgets | Linear solves reached strict targets only through pseudo-transient regularization; the nonlinear residual stayed near `9.876e-01` after 8 nonlinear iterations. |
| Temporary nonlinear-only strict run | Existing GMRES `1e-4` linear settings with nonlinear tolerance tightened to `1e-6` | The one-step nonlinear residual reached `1.2515063459340370e-05`, showing the previous linear setting leaves a residual floor above the strict nonlinear target. |

## Conclusion

- Coupled level-set/fluid solver controls now come from the fluid equation for
  D18-style equation ordering.
- The strict nonlinear tolerance, strict linear relative tolerance, and strict
  linear absolute tolerance checklist items remain open.
- The robust solver-path item remains open. The current strict runs show either
  a true-residual floor or pseudo-transient nonlinear stagnation rather than a
  validated strict D18 path.
