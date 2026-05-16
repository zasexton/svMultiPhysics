# D18 Configured-Time No-Output GMRES Attempt, 2026-05-15

## Purpose

Run D18 to the configured comparison interval without VTK output after
transient generated cut-quadrature basis evaluations were removed from the
process-wide basis cache.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 312 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 7200 \
  --qualification-log /tmp/d18_configured_no_output_uncached_basis_20260515.json \
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

- Result: failed with solver return code 1.
- Final reported step: 74 at time 0.037.
- Linear convergence: all linear solves converged.
- Nonlinear convergence: failed at nonlinear iteration limit 9.
- Failing nonlinear residual: 110.72146047872725.
- Failing residual block: velocity dominated; pressure residual norm remained
  near 3.93498e-5.
- Linear iterations: maximum 82; total 6029.
- Maximum basis-cache entries: 5.
- Basis-cache entry growth: 4.
- Maximum RSS: 367920 KiB.
- RSS growth: 183172 KiB.
- JIT kernel-cache size: 54 with 0 evictions.
- Active pressure support diagnostics: 112 records, maximum active sign
  vertices without support 0.
- Cut-context solution sources: missing source count 0.
- Minimum retained active volume fraction: 1.3688149925428837e-7.
- Maximum cut-adjacent stabilization scale: 7305589.180772148.
- Cut-adjacent capped scale count: 0.

## Conclusion

The configured-time no-output D18 attempt no longer shows basis-cache or resident
memory growth. The next blocker is a tiny retained active sliver that generates
an extreme cut-adjacent stabilization scale without being capped or pruned,
followed by velocity-dominated nonlinear residual growth.
