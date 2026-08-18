# BlockSchur Residual and Convergence Gap Findings - 2026-04-21

## Scope

This note records the current findings from the `pipe_RCR_3d` OOP FSILS
BlockSchur parity investigation across serial, MPI-2, and MPI-4.

The goal was to isolate the remaining residual and convergence-count gaps after
fixing the exact reduced Schur preconditioner coarse-matrix identity assembly.
The investigation focused on:

- preconditioned Schur RHS parity
- exact Schur operator application parity
- Schur preconditioner application parity
- constant and near-null pressure mode handling across ranks
- remaining Schur BiCGStab, nested momentum GMRES, and outer BlockSchur stopping
  sensitivity

## Confirmed Fixed Issue

The exact reduced preconditioner coarse matrix previously initialized the
identity on every MPI rank and then summed it with `MPI_Allreduce`. That made
the diagonal effectively:

```text
nRanks + sigma * nS
```

instead of:

```text
1 + sigma * nS
```

After moving the identity addition to after the reduction, the constant/global
pressure mode path became clean to roundoff. The previously large final pressure
solution mismatch collapsed from order `1e1` with `7-11%` relative mismatch to:

```text
MPI-2 vs serial final pressure solution rel diff: 1.13e-5
MPI-4 vs serial final pressure solution rel diff: 4.06e-6
```

The global mean Schur path now compares at roundoff:

```text
operator_global_mean_DGP MPI-2/MPI-4 rel diff:      about 5.3e-16
operator_global_mean_Minv_raw MPI-2/MPI-4 rel diff: about 5.3e-16
```

## Remaining Gap

The remaining production one-step run still shows a small convergence-count gap:

```text
baseline serial: nonlinear=4, final linear iters=3, residual=5.30e-12
baseline MPI-2:  nonlinear=4, final linear iters=2, residual=6.22e-12
baseline MPI-4:  nonlinear=4, final linear iters=3, residual=6.08e-12
```

The remaining absolute nonlinear residuals are close, but MPI-2 crosses the final
outer acceptance threshold one linear iteration earlier than serial and MPI-4.

## Isolation Controls Added

Two env-gated diagnostic controls were added for this investigation. With the
environment variables unset, default behavior is unchanged.

```text
SVMP_FSILS_BLOCKSCHUR_SCHUR_FIXED_ITERS=<N>
```

Forces Schur BiCGStab to continue until at least `N` iterations before honoring
normal convergence. This applies to the scalar legacy Schur path and the newer
multi-constraint Schur BiCGStab path.

```text
SVMP_FSILS_BLOCKSCHUR_MIN_OUTER_ITERS=<N>
```

Requires at least `N` outer BlockSchur iterations before accepting outer
convergence.

## Probe Matrix

All cases used one-step `pipe_RCR_3d/solver_perf_oop.xml` copies with the rebuilt
`svmultiphysics` binary.

Run artifact root:

```text
tests/cases/fluid/_gap_probe_pipe_rcr3d_144701_20260421
```

### Baseline

```text
np1: nonlinear_iters=4 residual=5.301372e-12 final_linear_iters=3 lin_rel=1.340278e-04
     outer_seq=[3,2,3,2,3,2,4,2]
     schur_iters=[78,65,79,64,78,66,72,67]

np2: nonlinear_iters=4 residual=6.216431e-12 final_linear_iters=2 lin_rel=7.729462e-04
     outer_seq=[3,2,3,2,3,2,3,2]
     schur_iters=[76,68,79,67,77,65,73,64]

np4: nonlinear_iters=4 residual=6.077270e-12 final_linear_iters=3 lin_rel=2.226566e-04
     outer_seq=[3,2,3,2,3,2,4,2]
     schur_iters=[77,62,76,67,77,64,71,67]
```

### Fixed Schur BiCGStab Iteration Count

With:

```text
SVMP_FSILS_BLOCKSCHUR_SCHUR_FIXED_ITERS=90
```

the final linear-count gap closes:

```text
np1: nonlinear_iters=4 residual=6.105026e-12 final_linear_iters=3 lin_rel=1.409790e-04
     outer_seq=[3,2,3,2,3,2,4,2]

np2: nonlinear_iters=4 residual=6.549349e-12 final_linear_iters=3 lin_rel=1.995176e-04
     outer_seq=[3,2,3,2,3,2,4,2]

np4: nonlinear_iters=4 residual=6.623545e-12 final_linear_iters=3 lin_rel=2.304343e-04
     outer_seq=[3,2,3,2,3,2,4,2]
```

This is the clearest evidence that the remaining convergence-count gap is
primarily Schur BiCGStab trajectory/stopping sensitivity rather than a remaining
deterministic Schur operator or preconditioner mismatch.

### Tight Nested Momentum GMRES

The `NS_GM_tolerance` was tightened from `1e-3` to `1e-8`, and
`NS_GM_max_iterations` was increased from `10` to `60`.

```text
np1: nonlinear_iters=4 residual=6.256340e-12 final_linear_iters=3 lin_rel=1.520834e-04
     outer_seq=[3,2,3,2,3,2,4,2]

np2: nonlinear_iters=4 residual=6.796400e-12 final_linear_iters=3 lin_rel=3.298242e-04
     outer_seq=[3,2,3,2,3,2,4,2]

np4: nonlinear_iters=4 residual=4.665479e-12 final_linear_iters=4 lin_rel=2.300701e-04
     outer_seq=[3,2,3,2,3,2,5,2]
```

Tightening nested momentum GMRES changes the convergence pattern and closes the
MPI-2 early-exit pattern, but it shifts MPI-4 to one extra final linear
iteration. This means the nested `K^-1` inexactness participates in the
trajectory, but is not a clean standalone root cause.

### Minimum Outer Iteration Gate

With:

```text
SVMP_FSILS_BLOCKSCHUR_MIN_OUTER_ITERS=3
```

the early acceptance is masked, but the result is not a useful fix:

```text
np1: nonlinear_iters=4 residual=2.729011e-12 final_linear_iters=5 lin_rel=1.480441e-04
     outer_seq=[3,3,3,3,4,3,6,3]

np2: nonlinear_iters=4 residual=9.109898e-13 final_linear_iters=6 lin_rel=7.658348e-05
     outer_seq=[3,3,3,3,4,3,7,3]

np4: nonlinear_iters=4 residual=9.095129e-12 final_linear_iters=6 lin_rel=1.090607e-04
     outer_seq=[3,3,3,3,4,3,7,3]
```

This suggests that the outer threshold is where the mismatch becomes visible,
but not where it originates.

## Schur BiCGStab Residual History Evidence

The first Schur solve residual histories remain close early and then diverge
late, consistent with floating-point reduction/order amplification in BiCGStab.

Baseline divergence versus serial:

```text
MPI-2: rel residual-history diff > 1e-6 at iter 28
MPI-2: rel residual-history diff > 1e-3 at iter 36
MPI-2: rel residual-history diff > 1e-1 at iter 38

MPI-4: rel residual-history diff > 1e-6 at iter 29
MPI-4: rel residual-history diff > 1e-3 at iter 34
MPI-4: rel residual-history diff > 1e-1 at iter 38
```

The fixed-iteration Schur run preserves the same late divergence pattern but
prevents different rank counts from exiting the Schur solve at different
effective points in the trajectory.

## Current Conclusion

The remaining residual and convergence-count gap is no longer likely to be in:

- the preconditioned Schur RHS
- the exact Schur operator application
- the exact reduced Schur preconditioner scaling
- the constant/global pressure mode handling

The most likely source is now:

```text
rank-dependent floating-point amplification in Schur BiCGStab,
fed partly by nested momentum GMRES inexactness,
then exposed by the outer BlockSchur convergence threshold.
```

## Follow-Up Options

When resuming this work, the most useful next steps are:

1. Add a deterministic true residual recomputation gate before accepting Schur
   BiCGStab convergence.
2. Evaluate using Schur GMRES/FGMRES for this reduced RCR path instead of
   BiCGStab.
3. Add a rank-robust Schur stopping policy that avoids accepting near the
   threshold when the residual is in the known late-iteration sensitive region.
4. Revisit nested momentum GMRES tolerance policy only after deciding how the
   Schur solve itself should be stabilized.
