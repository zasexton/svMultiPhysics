# D18 Configured-Time No-Output GMRES Probe, 2026-05-15

## Purpose

Verify that the D18 unfitted free-surface case reaches the configured
comparison time without nonlinear stalls after cut-adjacent stabilization scales
were capped for retained active slivers.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 312 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 7200 \
  --qualification-log /tmp/d18_configured_no_output_capped_scale_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --require-process-memory-diagnostics \
  --enable-linear-solve-memory-diagnostics \
  --require-linear-solve-memory-diagnostics \
  --require-basis-cache-diagnostics \
  --max-diagnostic-process-basis-cache-entries 500 \
  --enable-jit-cache-diagnostics \
  --require-jit-cache-diagnostics \
  --max-diagnostic-cut-adjacent-scale 1000 \
  --min-diagnostic-cut-adjacent-capped-scale-count 1 \
  --max-nonlinear-iterations 9 \
  --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 \
  --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 \
  --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 \
  --ns-cg-tolerance 1.0e-4
```

## Parsed Results

- Result: passed.
- Accepted steps: 312, final accepted time 0.156.
- Nonlinear convergence: all 312 records converged.
- Linear convergence: all 312 records converged.
- Nonlinear iterations: maximum 2; distribution 46 records at 1 iteration and
  266 records at 2 iterations.
- Nonlinear residual: maximum 5.907356654952446e-4, mean
  6.928942977290323e-5, minimum 1.675771632572326e-5.
- Linear iterations: maximum 82; total 25385.
- Linear iteration distribution: 304 records at 82 iterations, 3 records at 70,
  and one record each at 25, 41, 48, 56, and 77.
- Linear relative residual: maximum 0.04591015302880207, mean
  0.012883197989063847, minimum 2.127131078546177e-4.
- Minimum retained active volume fraction: 1.493904686224981e-10.
- Generated pruned volume rules: 16.
- Active pruned volume regions: 6.
- Maximum cut-adjacent stabilization scale: 1000.
- Cut-adjacent capped scale count: 68.
- Active pressure support diagnostics: 579 records, maximum active sign
  vertices without support 0.
- Maximum constrained owned pressure DOFs: 2007.
- Active-volume consistency error: 4.988588012793116e-4.
- Active wet-volume change: 2.69399999999996.
- Maximum basis-cache entries: 5.
- Basis-cache entry growth: 4.
- Maximum RSS: 367796 KiB.
- RSS growth: 183304 KiB.
- JIT kernel-cache size: 54 with 0 evictions.
- Solution pressure range: 1409.66346.
- Solution velocity range: 1.0684589599999998.
- VTK outputs: 0.

## Conclusion

The D18 no-output GMRES probe reaches the configured comparison time without
nonlinear or linear solve failure. The tiny retained active sliver path now
reports bounded cut-adjacent scales at the 1000 ceiling, and process memory and
basis-cache entry counts remain stable through the full run.
