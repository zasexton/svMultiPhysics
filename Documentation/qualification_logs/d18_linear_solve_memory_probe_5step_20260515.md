# D18 Five-Step Linear-Solve Memory Probe, 2026-05-15

## Purpose

This probe adds before/after linear-solve process-memory diagnostics to the
five-step D18 no-output GMRES run. The goal was to determine whether the
observed resident-memory growth occurs inside the linear solver call or before
that point in cut-context refresh and assembly.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 5 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 1200 \
  --preserve-run-dir \
  --qualification-log /tmp/d18_linear_solve_memory_probe_5step_20260515.json \
  --require-process-memory-diagnostics \
  --enable-linear-solve-memory-diagnostics \
  --require-linear-solve-memory-diagnostics \
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
/tmp/dam_break_d18_z9m9gv5h/spheric_test05_wet_bed_d18
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
- Linear-solve memory records parsed: 10.
- Total process-memory records parsed: 52.

## Linear-Solve Memory

The before/after linear-solve resident set was unchanged for every solve:

| solve | before RSS KiB | after RSS KiB | delta KiB |
| --- | ---: | ---: | ---: |
| 1 | 370360 | 370360 | 0 |
| 2 | 409020 | 409020 | 0 |
| 3 | 481316 | 481316 | 0 |
| 4 | 553340 | 553340 | 0 |
| 5 | 623216 | 623216 | 0 |

## Process-Memory Trend

- First parsed resident set: 184184 KiB.
- Last parsed resident set: 692272 KiB.
- Resident-set growth: 508088 KiB.
- Maximum parsed virtual memory: 1053148 KiB.

The ordered diagnostics show memory growth before the linear solve, mostly
across assemble-operator records and accepted-state cut-context refreshes. The
linear solver call itself did not increase resident memory in this probe.

## Interpretation

The D18 resource blocker is not caused by the GMRES solve call retaining memory
between the start and end of each linear solve. The next implementation step
should target allocation reuse or release in active-domain assembly and
cut-context refresh, including generated cut-volume rule materialization,
generated facet-set traversal data, and sparse matrix assembly views.
