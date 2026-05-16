# D18 JIT-Cache Diagnostic Probe, 2026-05-15

## Purpose

This probe verifies the new opt-in JIT-cache diagnostic line and smoke-script
parsing. It was run as a one-step D18 no-output GMRES probe so memory and cache
diagnostics could be checked without waiting for reference-time output.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 1 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --preserve-run-dir \
  --qualification-log /tmp/d18_jit_cache_diagnostics_probe_20260515.json \
  --require-process-memory-diagnostics \
  --enable-jit-cache-diagnostics \
  --require-jit-cache-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --max-nonlinear-iterations 9 \
  --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 \
  --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 \
  --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 \
  --ns-cg-tolerance 1.0e-4
```

Preserved case directory:

```text
/tmp/dam_break_d18_un3hyvf0/spheric_test05_wet_bed_d18
```

## Result

The smoke probe passed all requested parseable checks.

- Accepted steps: 1.
- Final accepted time: 0.0005 s.
- Nonlinear convergence: converged in 1 iteration.
- Linear convergence: converged in 25 iterations.
- JIT-cache diagnostic records parsed: 45.
- Process-memory records parsed: 10.

## Parsed JIT-Cache Diagnostics

- Maximum kernel-cache size: 54.
- Maximum kernel-cache hits: 51.
- Maximum kernel-cache misses: 54.
- Maximum kernel-cache stores: 54.
- Maximum kernel-cache evictions: 0.
- Maximum object-cache entries: 0.
- Maximum object-cache disk hits: 54.
- Maximum object-cache gets: 54.
- Maximum object-cache misses: 0.
- Maximum object-cache bytes read: 1079520.

The cache diagnostics prove the smoke script can now require and summarize the
JIT compiler counters alongside process-memory diagnostics. This run loaded
available object-cache entries from disk and did not show object-cache growth,
so a longer probe can now distinguish object-code accumulation from other
resident-memory sources.

## Other Diagnostics

- Maximum resident set: 408320 KiB.
- Resident-set growth: 223832 KiB.
- Maximum virtual memory: 769268 KiB.
- Pressure range: 1468.85174687.
- Velocity range: 0.06740438.
- Active wet-volume discrepancy between cut context and assembly: 4.807815605545329e-4.
- Cut-adjacent maximum stabilization scale: 34.33082040774677.
- Cut-context solution sources: 2 FE-vector refreshes and 4 ordered
  state-vector refreshes, with no missing source records.
