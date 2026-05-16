# D18 Cut-Adjacent Scale Cap Probe, 2026-05-15

## Purpose

Verify that retained active slivers no longer generate unbounded cut-adjacent
stabilization scales in the D18 unfitted free-surface case.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 90 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 2400 \
  --qualification-log /tmp/d18_cut_scale_cap_90step_20260515.json \
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
- Accepted steps: 90, final accepted time 0.045.
- Nonlinear convergence: all records converged.
- Linear convergence: all records converged.
- Nonlinear iterations: maximum 2; distribution 46 records at 1 iteration and
  44 records at 2 iterations.
- Linear iterations: maximum 82; total 7323.
- Minimum retained active volume fraction: 1.295582412072275e-7.
- Maximum cut-adjacent stabilization scale: 1000.
- Cut-adjacent capped scale count: 12.
- Active pressure support diagnostics: 135 records, maximum active sign
  vertices without support 0.
- Maximum constrained owned pressure DOFs: 2007.
- Active-volume consistency error: 4.965552454905264e-4.
- Maximum basis-cache entries: 5.
- Basis-cache entry growth: 4.
- Maximum RSS: 367372 KiB.
- RSS growth: 183300 KiB.
- JIT kernel-cache size: 54 with 0 evictions.
- VTK outputs: 0.

## Conclusion

The D18 no-output GMRES probe now advances past the previous step-74 nonlinear
stall while retaining a cut-volume sliver of order 1e-7. The capped
cut-adjacent scale is parsed by the smoke script, remains at the configured
1000 ceiling, and does not coincide with nonlinear or linear solve failure in
this 90-step run.
