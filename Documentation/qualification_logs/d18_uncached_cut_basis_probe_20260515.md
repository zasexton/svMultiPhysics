# D18 Transient Cut-Basis Cache Probe, 2026-05-15

## Purpose

Verify that transient generated cut-quadrature basis evaluations do not
accumulate process-wide basis-cache entries during D18 no-output GMRES probes.

## Change Under Test

- Generated cut-volume and generated cut-interface quadrature use uncached
  basis evaluations for geometry, scalar test basis data, and scalar trial basis
  data.
- The assembler invalidates stale geometry basis data when the quadrature rule
  changes.
- The smoke script can require a maximum parsed basis-cache entry count.

## Commands

```bash
cmake --build build/svMultiPhysics-build --target svmultiphysics -j2
```

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 3 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_uncached_cut_basis_probe_20260515.json \
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
- Accepted steps: 3, final accepted time 0.0015.
- Nonlinear iterations: maximum 1.
- Linear iterations: 25, 82, 82.
- Maximum basis-cache entries: 5.
- Basis-cache entry growth: 4.
- Maximum RSS: 367708 KiB.
- RSS growth: 183284 KiB.
- JIT kernel-cache size: 54 with 0 evictions.
- Active pressure support diagnostics: 4 records, maximum active sign vertices
  without support 0.
- Cut-context solution sources: missing source count 0.

## Conclusion

The previous three-step D18 basis-cache probe reached 9933 global basis-cache
entries. The same no-output GMRES probe now stays at 5 entries while preserving
active pressure support diagnostics and cut-context solution-source diagnostics.
Full D18 and D38 configured-time qualification remains blocked by the remaining
long-run resource and output-space constraints.
