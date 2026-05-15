# D18 GMRES Pressure-Support Timeout, 2026-05-15

Command:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 312 \
  --linear-solver-type GMRES \
  --final-output-only \
  --timeout-seconds 3600 \
  --preserve-run-dir \
  --qualification-log Documentation/qualification_logs/d18_gmres_pressure_support_20260515.json \
  --stale-pressure-gauge-tolerance 100.0 \
  --max-wet-fraction-volume-error 1.0e-8 \
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

Result:

- The run timed out at 3600 s before the configured 312-step comparison time.
- Preserved run directory:
  `/tmp/dam_break_d18_ahvor69w/spheric_test05_wet_bed_d18`.
- The run accepted 12 steps and reached `t=0.006000000000000002`.
- All parsed nonlinear records converged with one nonlinear iteration per
  accepted step.
- All parsed GMRES solves converged; the largest parsed linear iteration count
  was 72 and the total parsed count was 724.
- No direct Eigen factorization diagnostics were emitted on this GMRES path.
- Parsed solution-state ranges at timeout were
  `velocity_range=0.09434347` and `pressure_range=1420.93`.
- After filtering cut-volume diagnostics to the active negative side, the
  parsed active-volume assembly span is `0.13200000000006185` and the
  active-volume consistency error is `0.0004807815605545329`.
- The largest parsed assembly time was `156.8663 s`; the largest parsed
  cut-adjacent interior-face portion was `155.115065 s`, while cut-volume
  assembly was at most `1.599167 s`.

Conclusion:

The inactive pressure support constraint removed the direct-singular-row
blocker seen in direct Eigen probes. The current D18 reference-time blocker is
cut-adjacent interior-face assembly cost, not nonlinear stagnation or linear
solver divergence in the first 12 accepted steps.
