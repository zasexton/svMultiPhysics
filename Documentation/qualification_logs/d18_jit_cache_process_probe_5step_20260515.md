# D18 Five-Step JIT-Cache And Process-Memory Probe, 2026-05-15

## Purpose

This probe repeats the five-step D18 no-output GMRES memory check with the new
JIT-cache diagnostics enabled. The goal was to determine whether resident-memory
growth follows JIT kernel or object-cache accumulation.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 5 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 1200 \
  --preserve-run-dir \
  --qualification-log /tmp/d18_jit_cache_process_probe_5step_20260515.json \
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
/tmp/dam_break_d18_h78fkmo7/spheric_test05_wet_bed_d18
```

## Result

The smoke probe passed all requested parseable checks.

- Accepted steps: 5.
- Final accepted time: 0.0025 s.
- Nonlinear convergence: all 5 accepted steps converged in 1 iteration.
- Linear convergence: all 5 accepted steps converged.
- Linear iteration distribution: 25 iterations for the first step, 82
  iterations for each later step.
- Total linear iterations: 353.
- JIT-cache diagnostic records parsed: 45.
- Process-memory records parsed: 42.

## JIT-Cache Diagnostics

The cache counters plateaued during the run:

- Maximum kernel-cache size: 54.
- Maximum kernel-cache hits: 51.
- Maximum kernel-cache misses: 54.
- Maximum kernel-cache stores: 54.
- Maximum kernel-cache evictions: 0.
- Maximum object-cache entries: 0.
- Maximum object-cache disk hits: 54.
- Maximum object-cache misses: 0.
- Maximum object-cache notifications compiled: 0.
- Maximum object-cache bytes read: 1079520.
- Maximum object-cache bytes written: 0.

## Process Memory

Resident memory still grew over the same five accepted steps:

- First parsed resident set: 184624 KiB.
- Last parsed resident set: 692196 KiB.
- Resident-set growth: 507572 KiB.
- Maximum parsed virtual memory: 1053428 KiB.

## Interpretation

The five-step memory growth does not follow JIT object-cache accumulation: no
new object-cache entries were retained, the object cache had no misses, and the
kernel-cache size did not grow past 54. The next memory-lifecycle investigation
should focus on generated cut-context data, active-domain assembly traversal
buffers, matrix/vector allocation reuse, or allocator retention during repeated
cut-context rebuilds and assemble-operator calls.
