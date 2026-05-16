# D18 Residual-First Convergence Probe, 2026-05-15

## Purpose

The Newton assembly diagnostics showed that no-line-search D18 steps converged
after one update, but still rebuilt a full matrix in the next iteration to
verify the updated residual. This step changes that post-update convergence
check to assemble the residual first and defer matrix assembly unless the
residual check does not converge.

## Run

```text
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 3 --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_residual_first_convergence_3step_20260515.json \
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
- total linear iterations: `189`;
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
- maximum total assembly time in one record: `3.790601 s`;
- latest residual-only convergence-check assembly time: `1.549369 s`;
- latest residual-only convergence-check cut-volume time: `0.490008 s`;
- latest residual-only convergence-check cut-adjacent interior-face time:
  `0.995005 s`;
- maximum cut-adjacent stabilization scale: `34.342301`;
- maximum process basis-cache entries: `5`;
- maximum resident set size: `368020 KiB`.

## Conclusion

The change removes the avoidable post-update matrix rebuild for converged
no-line-search steps. The remaining D18 cost is still dominated by residual
traversals: each accepted step needs the initial combined assembly and a
post-update residual verification assembly. Further output qualification should
therefore focus on shortening the residual-only cut-volume and cut-adjacent
interior-face paths or proving that the residual check can be safely reused in
specific linearized cases.
