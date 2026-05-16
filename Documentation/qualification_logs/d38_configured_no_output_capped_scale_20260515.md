# D38 Configured-Time No-Output GMRES Probe, 2026-05-15

## Purpose

Verify that the D38 unfitted free-surface case reaches the configured
comparison time without nonlinear stalls after cut-adjacent stabilization scales
were capped for retained active slivers.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d38 \
  --steps 312 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 7200 \
  --qualification-log /tmp/d38_configured_no_output_capped_scale_20260515.json \
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
- Nonlinear iterations: maximum 2; distribution 50 records at 1 iteration and
  262 records at 2 iterations.
- Nonlinear residual: maximum 5.898645771659892e-4, mean
  6.866584894545896e-5, minimum 1.4278632642030206e-5.
- Linear iterations: maximum 82; total 25421.
- Linear iteration distribution: 308 records at 82 iterations and one record
  each at 34, 36, 43, and 52.
- Linear relative residual: maximum 0.039850491876843336, mean
  0.013627140271205561, minimum 2.68400647368285e-4.
- Minimum retained active volume fraction: 1.722817776520911e-10.
- Generated pruned volume rules: 12.
- Active pruned volume regions: 6.
- Maximum cut-adjacent stabilization scale: 1000.
- Cut-adjacent capped scale count: 68.
- Active pressure support diagnostics: 575 records, maximum active sign
  vertices without support 0.
- Maximum constrained owned pressure DOFs: 1791.
- Active-volume consistency error: 0.0049776956595906086.
- Active wet-volume change: 3.269999999999982.
- Maximum basis-cache entries: 5.
- Basis-cache entry growth: 4.
- Maximum RSS: 376604 KiB.
- RSS growth: 189876 KiB.
- JIT kernel-cache size: 54 with 0 evictions.
- Solution pressure range: 1406.49049.
- Solution velocity range: 0.838000068.
- VTK outputs: 0.

## Conclusion

The D38 no-output GMRES probe reaches the configured comparison time without
nonlinear or linear solve failure. The retained-sliver path remains bounded by
the cut-adjacent scale ceiling, and process memory and basis-cache entry counts
remain stable through the full run.
