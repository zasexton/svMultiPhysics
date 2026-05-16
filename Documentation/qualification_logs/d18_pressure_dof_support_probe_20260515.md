# D18 Active Pressure DOF Support Probe, 2026-05-15

## Purpose

Verify that inactive active-domain pressure DOFs are constrained from current
active cell support, and that the no-output smoke probe parses DOF-level support
diagnostics for the D18 unfitted free-surface case.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 1 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_pressure_dof_support_probe_20260515.json \
  --require-active-pressure-support-diagnostics \
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
- Accepted steps: 1, final accepted time 0.0005.
- Nonlinear convergence: all records converged.
- Linear convergence: all records converged.
- Active pressure support diagnostics: 2 records.
- Active pressure support cells: 5616.
- Total pressure DOFs: 3630.
- Active pressure DOFs with cell support: 1623.
- Inactive pressure DOFs without active cell support: 2007.
- Constrained owned pressure DOFs: 2007.
- Active sign vertices without active cell support: 0.
- Inactive sign vertices with active cell support: 207.
- Inactive DOF runs were parsed, beginning with
  `173-174|180-195|271-272|278-293|303-304|306-307|313-335|509-510|...`.
- Maximum basis-cache entries: 5.
- Maximum RSS: 367616 KiB.
- Maximum cut-adjacent stabilization scale: 34.33082040774677.

## Conclusion

The D18 no-output probe confirms that active pressure constraints are refreshed
from current active cell support and that every pressure DOF outside that
support is constrained. The parsed DOF-level diagnostics are available without
VTK output, so direct and iterative probes can distinguish inactive pressure
support from later stabilization-scale failures.
