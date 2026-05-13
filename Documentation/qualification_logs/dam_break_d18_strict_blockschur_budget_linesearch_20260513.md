# D18 Strict BlockSchur Budget And Line-Search Probe - 2026-05-13

## Scope

This log records the strict one-step D18 probes run after adding the explicit
FSILS BlockSchur layout for the coupled Test05 field order:

- `phi`
- `Velocity`
- `Pressure`

The probes keep the active-domain cut-volume path enabled and use strict solver
settings:

- nonlinear relative tolerance `1.0e-6`
- linear relative tolerance `1.0e-6`
- linear absolute tolerance `1.0e-10`
- fluid BlockSchur outer budget raised to `200`
- nested BlockSchur GM and CG relative tolerances set to `1.0e-6`

## Budget-Only Probe

Temporary run directory:

- `/tmp/svmp_d18_blockschur_strict_outer200_onestep_wXD6EF`

Additional runtime setting:

- `SVMP_FSILS_BLOCKSCHUR_OUTER_CAP=unlimited`

Result:

- The run exited with a nonlinear convergence error.
- BlockSchur outer iteration counts exceeded the previous effective cap:
  `10, 59, 84, 67, 184, 200, 39, 48, 15, 54, 4, 41, 48, 28, 5, 20, 44`.
- The final reported linear solve converged with relative residual
  `7.8136717334026179e-07`.
- The nonlinear solve reached the configured maximum of `8` iterations with
  final residual `1.2696237188898916e+01`.
- Total Newton time was `978.816253 s`, with `918` accumulated linear
  iterations.

Conclusion:

- Raising the BlockSchur outer budget is necessary for some strict linear
  solves, but it is not sufficient for D18 strict qualification.
- With the larger budget, the dominant blocker changes from the immediate
  BlockSchur true-residual stop to a nonconverged nonlinear update history.

## Line-Search Probe

Temporary run directory:

- `/tmp/svmp_d18_strict_linesearch_onestep_35jCCU`

Additional runtime settings:

- `SVMP_FSILS_BLOCKSCHUR_OUTER_CAP=unlimited`
- `SVMP_NEWTON_LINE_SEARCH=1`
- `SVMP_OOP_SOLVER_TRACE=1`
- `SVMP_DEBUG_LINEAR_SOLVE_HISTORY=1`
- `SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS=30`

The run was stopped after the first accepted strict linear solve entered line
search, failed to find a residual-decreasing update, reverted the step, and then
started repeating the same strict residual state.

Key linear history:

- First direct call:
  `converged=0`, `outer=10`, relative residual `2.63233e-02`,
  true-residual norm `1.62233e-01`.
- First PTC retry:
  `converged=0`, `outer=59`, relative residual `7.31123e-03`,
  true-residual norm `4.50596e-02`.
- Second PTC retry:
  `converged=1`, `outer=84`, relative residual `8.00431e-08`,
  true-residual norm `4.93311e-07`.

Line-search trials from the accepted PTC direction:

| Alpha | Residual norm |
| ---: | ---: |
| `1` | `12.5959` |
| `0.5` | `8.67273` |
| `0.25` | `7.12907` |
| `0.125` | `6.55435` |
| `0.0625` | `6.33328` |
| `0.03125` | `6.24153` |
| `0.015625` | `6.2006` |
| `0.0078125` | `6.18141` |
| `0.00390625` | `6.17213` |
| `0.00195312` | `6.16757` |

The starting residual norm was `6.16307`. Since every trial residual was larger
than the starting residual, the solver reverted the update:

- `line search did not reduce residual; reverting to original iterate`

The next direct solve at the same residual state again missed the strict target:

- `outer=72`
- relative residual `6.73886e-03`
- true-residual norm `4.15321e-02`

Conclusion:

- Enabling transient line search does not by itself create a strict D18
  convergence path.
- The strict BlockSchur direction can be non-descent for the active-domain D18
  residual even after a PTC retry produces a strict linear solve.
- The next solver work should target the pressure/constraint block and
  preconditioned direction quality, not only the global outer-iteration cap.

## Next Implementation Direction

The coupled D18 layout still uses a grouped primary block
`LevelSetVelocity(0:4)` and pressure as the constraint block. The strict probes
show large constraint-block update norms and repeated true-residual floors:

- accepted PTC direction `du_norm=9645.88` before the rejected line search
- repeated direct direction `du_norm=47257.3` after reverting

The next implementation step should inspect and improve the pressure constraint
path for the D18 BlockSchur solve. Candidate areas are:

- pressure gauge handling in the BlockSchur pressure block
- pressure nullspace or mean-mode treatment when a gauge constraint is present
- Schur preconditioner consistency for the grouped `phi` plus velocity primary
  block
- a segregated D18 solve path that lets Navier-Stokes use a velocity-pressure
  BlockSchur layout without the level-set field in the same linear system
