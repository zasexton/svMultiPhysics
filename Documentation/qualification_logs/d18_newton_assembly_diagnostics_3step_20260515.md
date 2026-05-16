# D18 Newton Assembly Diagnostics, 2026-05-15

## Purpose

The stopped D18 final-output run and the no-output assembly-density probes
showed too many expensive assembly traversals per accepted step. This step adds
an opt-in one-line Newton assembly diagnostic and verifies that the smoke script
can parse it.

## Run

```text
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 3 --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_newton_assembly_diag_3step_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-newton-assembly-diagnostics \
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
- assembly timing records: `8`;
- assembly timing records per accepted step: `2.666667`;
- Newton assembly records: `6`;
- Newton assemblies per accepted step: `2.0`;
- Newton matrix assemblies: `6`;
- Newton matrix assemblies per accepted step: `2.0`;
- Newton assembly phase counts: `jacobian_and_residual=6`;
- Newton assembly sync-point counts: `jacobian_and_residual=6`;
- post-first-iteration Newton matrix assemblies: `3`;
- cut-context rebuilds per accepted step: `4.0`;
- maximum total assembly time in one record: `3.643770 s`;
- maximum cut-volume assembly time in one record: `1.419868 s`;
- maximum cut-adjacent interior-face assembly time in one record:
  `2.105423 s`;
- maximum cut-adjacent stabilization scale: `34.342301`;
- maximum process basis-cache entries: `5`;
- maximum resident set size: `367956 KiB`.

## Conclusion

The short D18 run converges each accepted step with one Newton update, but the
no-line-search path enters the next Newton iteration to assemble a combined
Jacobian and residual for convergence verification. Those three post-update
matrix assemblies are unnecessary when the residual check proves convergence
and no second linear solve is needed.

The next implementation step should make the no-line-search post-update
convergence check assemble the residual alone first, then assemble the matrix
only if the residual check does not converge.
