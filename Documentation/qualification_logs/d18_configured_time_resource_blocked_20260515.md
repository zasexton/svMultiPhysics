# D18 Configured-Time Resource-Blocked Probe

Date: 2026-05-15

## Scope

This probe attempted the D18 configured-time GMRES run after dynamic active
pressure support refresh was added. The run used final VTK output only, active
pressure support diagnostics, cut-context solution-source checks, JIT cache
diagnostics, and linear-solve memory diagnostics.

The probe was stopped before final output because `/tmp` is on the root
filesystem and `df -h /tmp` reported only about 237 MiB free. The preserved run
directory was `/tmp/dam_break_d18_rqh5kzcn/spheric_test05_wet_bed_d18`, and the
smoke JSON was `/tmp/d18_configured_time_gmres_20260515.json`.

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 312 \
  --linear-solver-type GMRES \
  --final-output-only \
  --timeout-seconds 7200 \
  --preserve-run-dir \
  --qualification-log /tmp/d18_configured_time_gmres_20260515.json \
  --require-active-pressure-support-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --require-process-memory-diagnostics \
  --enable-linear-solve-memory-diagnostics \
  --require-linear-solve-memory-diagnostics \
  --enable-jit-cache-diagnostics \
  --require-jit-cache-diagnostics \
  --stale-pressure-gauge-tolerance 1.0e-3 \
  --max-wet-fraction-volume-error 1.0e-10 \
  --max-nonlinear-iterations 9 \
  --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 \
  --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 \
  --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 \
  --ns-cg-tolerance 1.0e-4
```

## Parsed Result

The run was terminated with return code `-15` before final VTK output. It did
not satisfy the D18 configured-time acceptance item.

Parsed diagnostics before termination:

- Accepted steps: 47, final accepted time `0.023500000000000017`.
- Nonlinear records: 47, all converged; maximum nonlinear iterations: 2.
- Linear solves: all converged; linear iterations were 25 for the first solve
  and 82 for the remaining 46 solves.
- Active pressure support diagnostics: 50 records.
- Latest active pressure support maxima: 5616 active support cells, 1623 active
  support vertices, 2007 inactive vertices, 2007 constrained owned pressure
  DOFs, 207 inactive-sign vertices with wet support, and 0 active-sign vertices
  without support.
- Cut-context solution sources: zero missing sources, with 95 FE-vector refresh
  records and 194 state-vector refresh records.
- Process memory: maximum RSS `3782892` KiB, RSS growth `3598248` KiB.
- JIT cache: kernel cache size reached 54 with zero evictions.
- Active wet volume changed by about `0.606` over the partial run.

The next configured-time acceptance attempt needs enough free disk for final
VTK output or a no-output acceptance mode that does not require field-based
profile metrics.
