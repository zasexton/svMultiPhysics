# D18 Skipped Accepted Cut Refresh Probe

Date: 2026-05-15

## Change

The transient no-line-search Newton callback now skips
`AcceptedNonlinearState` active cut-context refreshes. The callback still
refreshes cut context before nonlinear assembly and at accepted-step output
time, and line-search accepted/restored-state refreshes remain enabled.

## Verification

Command:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 3 --linear-solver-type GMRES --disable-vtk-output \
  --timeout-seconds 600 \
  --qualification-log /tmp/d18_no_line_search_skip_accepted_refresh_3step_20260515.json \
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

Result: passed.

Parsed metrics:

- Accepted steps: 3.
- Nonlinear iterations: 3.
- Linear iterations: 189 total, max 82.
- Assembly timing records: 8.
- Assembly timing records per accepted step: 2.666667.
- Extra assembly timing records per accepted step: 1.666667.
- Cut-context rebuilds: 12.
- Cut-context rebuilds per accepted step: 4.0.
- Cut-context rebuild provenance counts: jacobian_and_residual 6,
  before_physics_solve 3, accepted_step 3.
- Cut-context state refresh count: 6.
- Cut-context vector refresh count: 6.
- Active pressure support diagnostics: 13.
- Maximum active sign vertices without support: 0.
- Maximum constrained owned pressure DOFs: 2007.
- Maximum cut-adjacent scale: 34.342301.
- Maximum basis-cache entries: 5.
- Maximum RSS: 367816 KiB.

The previous 3-step no-line-search probe reported 18 cut-context rebuilds,
with six `accepted` rebuilds. This probe removes those redundant accepted
refreshes and keeps the assembled solution, active pressure support, and
cut-context solution-source checks intact.
