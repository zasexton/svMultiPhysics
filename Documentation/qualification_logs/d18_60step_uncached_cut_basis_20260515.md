# D18 60-Step No-Output GMRES Probe, 2026-05-15

## Purpose

Verify that D18 can advance beyond the short memory regression probe after
transient generated cut-quadrature basis evaluations were removed from the
process-wide basis cache.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 60 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 2400 \
  --qualification-log /tmp/d18_60step_uncached_cut_basis_20260515.json \
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
- Accepted steps: 60, final accepted time 0.03.
- Nonlinear convergence: all records converged.
- Linear convergence: all records converged.
- Nonlinear iterations: maximum 2; distribution 46 records at 1 iteration and
  14 records at 2 iterations.
- Linear iterations: maximum 82; total 4863.
- Maximum basis-cache entries: 5.
- Basis-cache entry growth: 4.
- Maximum RSS: 367520 KiB.
- RSS growth: 183284 KiB.
- JIT kernel-cache size: 54 with 0 evictions.
- Active pressure support diagnostics: 75 records, maximum active sign vertices
  without support 0.
- Cut-context solution sources: missing source count 0.
- Active-volume consistency error: 4.965552454905264e-4.
- Maximum cut-adjacent stabilization scale: 35.69361795636106.

## Conclusion

The long no-output D18 probe confirms that the previous transient basis-cache
growth path is closed: basis-cache entries remained at 5 throughout 60 accepted
steps. Full D18 and D38 configured-time qualification still requires output
space for final VTK diagnostics and comparison metrics.
