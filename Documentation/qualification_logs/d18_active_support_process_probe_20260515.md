# D18 Active-Support Process-Memory Probe, 2026-05-15

## Purpose

This probe records the first D18 no-output GMRES run after constraining inactive
pressure vertices from active cell support. The goal was to confirm that the
pressure-support fix removed the direct-factorization pressure-null-row issue
while checking whether the reference-time D18 blocker had moved to resource
growth.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 5 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 1200 \
  --preserve-run-dir \
  --qualification-log /tmp/d18_active_support_process_probe_20260515.json \
  --require-process-memory-diagnostics \
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
/tmp/dam_break_d18_ox1o0jkj/spheric_test05_wet_bed_d18
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
- VTK output: disabled for the probe.

## Parsed Diagnostics

- Pressure range: 1433.750953.
- Velocity range: 0.16232698.
- Active wet-volume discrepancy between cut context and assembly: 4.807815605545329e-4.
- Minimum retained active volume fraction: 2.909908330264437e-2.
- Generated pruned active volume: 0.
- Generated pruned active volume regions: 0.
- Cut-adjacent maximum stabilization scale: 34.3653437326366.
- Cut-adjacent capped scale count: 0.
- Cut-volume exact-order range: 2 to 2.
- Cut-context solution sources: 10 FE-vector refreshes and 20 ordered
  state-vector refreshes, with no missing source records.
- Maximum parsed assembly time: 3.385058 s.
- Maximum parsed cut-volume assembly time: 1.325076 s.
- Maximum parsed interior-face assembly time: 1.914784 s.

## Process Memory

The process-memory diagnostics show continued resident-memory growth during the
short no-output run:

- First parsed resident set: 184364 KiB at cut-context revision 1.
- Last parsed resident set: 691780 KiB at cut-context revision 30.
- Resident-set growth: 507416 KiB.
- Maximum parsed virtual memory: 1053408 KiB.
- Process-memory records parsed: 42.

The last cut-context rebuild still reported the expected active support:

- Active cut cells: 816.
- Active wet cells: 5616.
- Active volume regions: 5616.
- Cut-adjacent maximum scale: 34.3653437326366.

## Interpretation

The pressure-support change moved the short D18 GMRES probe past the invalid
pressure-null-row failure mode: nonlinear and linear solves converged and the
solution fields departed from the old full-domain hydrostatic state. The next
blocker for reference-time D18/D38 qualification is resident-memory growth over
many accepted steps. The next implementation step should inspect memory
lifecycle for generated cut-context data, marked facet traversal metadata, and
JIT specialization resources.
