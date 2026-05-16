# D18 Basis Cache Growth Probe

Date: 2026-05-15

## Scope

This step added `basis_cache_entries` to existing compact process-memory
diagnostics on active cut-context rebuilds and linear-solve memory probes. The
smoke script now summarizes maximum basis-cache entries, basis-cache entry
growth, and can require those diagnostics.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 3 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_basis_cache_diagnostics_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --require-process-memory-diagnostics \
  --enable-linear-solve-memory-diagnostics \
  --require-linear-solve-memory-diagnostics \
  --require-basis-cache-diagnostics \
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

## Result

Result: passed.

Key parsed diagnostics:

- Accepted steps: 3, final accepted time `0.0015`.
- Nonlinear records: 3, all converged with one nonlinear iteration each.
- Linear solves: all converged; iteration distribution was 25 for the first
  solve and 82 for the next two solves.
- Basis-cache entries: maximum `9933`, growth `9932`.
- Process memory: maximum RSS `548724` KiB, RSS growth `364164` KiB.
- JIT cache: kernel cache size stayed at 54 with zero evictions.
- Active pressure support diagnostics: 4 records, with 0 active-sign vertices
  lacking wet support.
- Cut-context solution sources: zero missing sources.

The next memory-lifecycle target is avoiding global basis-cache retention for
transient cut-volume quadrature rules whose point coordinates change across
active cut-context rebuilds.
