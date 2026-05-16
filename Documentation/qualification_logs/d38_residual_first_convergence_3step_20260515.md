# D38 Residual-First Convergence Probe, 2026-05-15

## Purpose

This is the D38 counterpart to the D18 residual-first convergence probe. It
checks that the no-line-search post-update residual verification path also
avoids throwaway matrix rebuilds for the deeper wet-bed fixture.

## Run

```text
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d38 --steps 3 --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d38_residual_first_convergence_3step_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-newton-assembly-diagnostics \
  --max-diagnostic-newton-matrix-assemblies-per-step 1.1 \
  --require-assembly-topology-consistency \
  --require-assembly-timing-diagnostics \
  --max-diagnostic-cut-context-rebuilds-per-step 4.1 \
  --require-process-memory-diagnostics \
  --enable-linear-solve-memory-diagnostics \
  --require-linear-solve-memory-diagnostics \
  --require-basis-cache-diagnostics \
  --max-diagnostic-process-basis-cache-entries 500 \
  --enable-jit-cache-diagnostics --require-jit-cache-diagnostics \
  --max-diagnostic-cut-adjacent-scale 1000 \
  --max-nonlinear-iterations 9 \
  --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 \
  --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 \
  --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 \
  --ns-cg-tolerance 1.0e-4
```

The probe passed.

## Metrics

- accepted steps: `3`;
- nonlinear iterations: `3`;
- total linear iterations: `207`;
- Newton assembly records: `6`;
- Newton assemblies per accepted step: `2.0`;
- Newton matrix assemblies: `3`;
- Newton matrix assemblies per accepted step: `1.0`;
- Newton vector assemblies: `6`;
- Newton post-first-iteration matrix assemblies: `0`;
- Newton assembly phase counts:
  `jacobian_and_residual=3`, `post_update_convergence_check=3`;
- Newton assembly sync-point counts: `jacobian_and_residual=3`,
  `residual=3`;
- cut-context rebuild provenance:
  `jacobian_and_residual=3`, `residual=3`,
  `before_physics_solve=3`, `accepted_step=3`;
- cut-context rebuilds per accepted step: `4.0`;
- assembly timing records per accepted step: `2.666667`;
- maximum total assembly time in one record: `3.475754 s`;
- latest residual-only convergence-check assembly time: `1.474191 s`;
- latest residual-only convergence-check cut-volume time: `0.481687 s`;
- latest residual-only convergence-check cut-adjacent interior-face time:
  `0.925566 s`;
- maximum cut-adjacent stabilization scale: `34.328125`;
- maximum process basis-cache entries: `5`;
- maximum resident set size: `376732 KiB`.

## Conclusion

D38 shows the same corrected Newton assembly pattern as D18. The residual-first
change removes post-update matrix rebuilds on converged no-line-search steps
without changing the parsed nonlinear or linear convergence behavior in the
three-step probe.
