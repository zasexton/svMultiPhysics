# D18 MPI Solver Scaling Evidence - 2026-05-13

## Purpose

This note records the MPI-4 D18 solver-scaling checks run after the strict
D18/D38 Test05 inputs were updated to the coupled outer FGMRES BlockSchur path.
The checks distinguish cut-cell stabilization from the remaining MPI solver
floor before changing the profile-comparison workflow.

## Baseline Failure

Checked-in D18 input, one time step, four MPI ranks:

```bash
mpirun -np 4 build/svMultiPhysics-build/bin/svmultiphysics solver.xml
```

Temporary run directory:

- `/tmp/svmp_d18_mpi4_strict_trace_onestep_7gvg0u`

Final error:

- `NewtonSolver: linear solve did not converge`
- `fsils (blockschur: true residual check failed (|Ax-b|=0.0425642, rel=0.00690591, target=6.16345e-06))`

The same strict settings also blocked the 312-step profile run at the first
step:

- `/tmp/svmp_d18_mpi4_profile_b2tEuA`
- Final true-residual check: `|Ax-b|=0.0425642`, relative `0.00690591`,
  target `6.16345e-06`

## Coupled Outer Budget Probe

The strict one-step MPI case was repeated with:

- Fluid `Max_iterations=800`
- Fluid `Krylov_space_dimension=800`
- `SVMP_FSILS_BLOCKSCHUR_OUTER_CAP=800`
- Linear relative tolerance `1.0e-6`
- Linear absolute tolerance `1.0e-10`

Run directory:

- `/tmp/svmp_d18_mpi4_outer800_oJVtnh`

The first coupled outer solve then passed its true-residual check:

- Coupled outer iterations: `70`
- True residual norm: `3.148167e-08`
- True residual target: `1.820469e-07`

The run later reached a small-RHS phase where the absolute floor forced the
linear target to `1.0e-10`. That phase became prohibitively expensive and still
reported true residuals near `1.0e-6`. This identifies the strict
`1.0e-10` absolute linear tolerance as an MPI scaling limit for this active
D18 path.

## Absolute Linear Floor Sweep

All runs below used four MPI ranks, the 800-iteration coupled outer budget, and
the checked-in D18 active-domain assembly.

| Linear absolute tolerance | Newton cap | Result |
| ---: | ---: | --- |
| `1.0e-5` | 8 | Linear solves completed, but nonlinear residual stopped at `9.8933791882910248e-06`. |
| `1.0e-6` | 8 | Nonlinear residual stopped at `6.1434189023711983e-06`. |
| `1.0e-6` | 12 | Nonlinear residual stopped at `6.1434370250073864e-06`. |
| `1.0e-7` | 12 | Nonlinear residual improved to `3.4349784577002280e-06`. |
| `1.0e-8` | 16 | One BlockSchur solve took `62.743431 s` with `2405321` all-reduce calls before the run was stopped. |

The sweep shows a useful accuracy/runtime region around a `1.0e-7` linear
absolute tolerance. Tightening below that enters an expensive small-RHS regime,
while loosening above it increases the nonlinear plateau.

## Stabilization Probe

The cut-adjacent stabilization penalty was raised to `0.1` with:

- Linear absolute tolerance `1.0e-7`
- Fluid `Max_iterations=800`
- Fluid `Krylov_space_dimension=800`
- Newton cap `12`

Run directory:

- `/tmp/svmp_d18_mpi4_penalty01_abs1e7_5JqPbo`

Result:

- Nonlinear residual stopped at `3.4600103213253318e-06`

This is not an improvement over the matching `1.0e-7` floor run without the
penalty change. The current MPI-4 D18 blocker is therefore solver scaling, not
the cut-adjacent stabilization coefficient.

## Conclusion

- Active-domain volume integration and wet-side initialization were already
  verified before these probes.
- The strict MPI-4 D18 failure is controlled by coupled BlockSchur outer budget
  and absolute linear residual floor.
- Stabilization tuning is not supported by the current evidence.
- The next D18 profile attempt should use a documented MPI solver-control
  profile instead of the strict `1.0e-10` absolute linear floor.
