# JIT Kernel Refactor Baseline Archive

This archive captures the pre-refactor performance and convergence baseline that
the refactor is compared against.

## Archived Artifacts

- Summary report: `tests/perf_benchmark_report_20260323.md`
- Raw benchmark output: `tests/perf_benchmark_output.txt`
- Raw result directory: `tests/perf_results_20260323_104947/`

## Coverage Captured On 2026-03-23

The archived benchmark run includes:

- `Channel2D`
- `Channel2D_Simple`
- `vortex_shedding`
- `pipe_RCR_3d`
- `pipe_simple`
- `iliac_artery`

The report contains:

- wall-time comparisons
- Newton iteration summaries
- assembly-time splits
- per-element assembly cost
- cold-vs-warm JIT observations

## Notes For This Refactor

- The archived report predates the `MixedKernelPlan` and `MonolithicCellKernel`
  production path.
- It is the comparison point for the Phase 0 and final qualification checklist
  items in `Documentation/plan_jit_kernel_optimization_refactor.md`.
- `pipe_RCR_3d/solver_perf_oop_rcrcr.xml` is not part of this archived
  2026-03-23 capture and still needs a dedicated qualification rerun before the
  case-matrix items in the refactor plan can be checked off.
