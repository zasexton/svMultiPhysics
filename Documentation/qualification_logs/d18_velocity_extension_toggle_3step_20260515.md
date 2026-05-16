# D18 Velocity-Extension Toggle Probe, 2026-05-15

## Purpose

The previous D18 final-output run was stopped because the case was still in the
solve phase after an impractical wall-clock time. This probe checks whether the
inactive-side free-surface velocity extension is responsible for the repeated
cut-volume and cut-adjacent assembly cost seen in the no-output GMRES probes.

## Run

Temporary smoke-script case copy:

```text
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 3 --linear-solver-type GMRES \
  --disable-vtk-output --disable-velocity-extension \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_disable_velocity_extension_3step_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
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
- maximum linear iterations in one solve: `82`;
- assembly timing records: `8`;
- assembly timing records per accepted step: `2.666667`;
- extra assembly timing records beyond nonlinear iterations per accepted step:
  `1.666667`;
- cut-context rebuilds: `12`;
- cut-context rebuilds per accepted step: `4.0`;
- cut-context rebuild provenance:
  `jacobian_and_residual=6`, `before_physics_solve=3`, `accepted_step=3`;
- maximum total assembly time in one record: `3.563533 s`;
- maximum cut-volume assembly time in one record: `1.325063 s`;
- maximum cut-adjacent interior-face assembly time in one record:
  `2.081367 s`;
- maximum cut-adjacent stabilization scale: `34.342301`;
- maximum process basis-cache entries: `5`;
- maximum resident set size: `368352 KiB`;
- active pressure support diagnostics: `13` records;
- maximum active-sign vertices without pressure support: `0`;
- maximum constrained pressure DOFs: `2007`.

## Conclusion

Disabling free-surface velocity extension in the temporary case copy does not
materially reduce short-run assembly density, cut-volume time, or cut-adjacent
interior-face time relative to the previous no-output GMRES probe. The positive
side cut-volume diagnostics are therefore not sufficient evidence that
inactive-side velocity-extension physics is the throughput blocker.

The remaining runtime issue is the number and cost of repeated full assembly
traversals per accepted step. Further D18/D38 output qualification should wait
until that assembly path is reduced or more precisely attributed.
