# D18 Active-Side Scale Long-Run Resource Probe, 2026-05-15

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 312 --linear-solver-type GMRES --final-output-only \
  --timeout-seconds 3600 --preserve-run-dir \
  --qualification-log /tmp/d18_gmres_active_side_scale_full_20260515.json \
  --max-wet-fraction-volume-error 1.0e-8 \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --max-nonlinear-iterations 9 --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 --ns-cg-tolerance 1.0e-4
```

## Result

- Run directory: `/tmp/dam_break_d18_xikp_46z/spheric_test05_wet_bed_d18`
- Parsed log: `/tmp/d18_gmres_active_side_scale_full_20260515.json`
- Solver status: stopped with return code `-15` during monitoring
- Accepted steps: `53`
- Final accepted time: `0.02650000000000002`
- Nonlinear iterations: maximum `2`, total `60`
- GMRES iterations: maximum `72`, total `3141`
- Maximum GMRES relative residual: `0.0005633604690844791`
- Maximum assemble-operator time: `3.858421` s
- Maximum interior-face assembly time: `2.219635` s
- Maximum cut-volume assembly time: `1.549614` s
- Active side: `LevelSetNegative`
- Active minimum volume fraction: `0.028069668007427134`
- Maximum cut-adjacent stabilization scale: `35.6256440131534`
- Velocity range: `0.46941931`
- Pressure range: `1416.5`
- VTK outputs: `0`

## Resource Observation

The solver process memory grew monotonically during the run. External monitor
samples showed resident memory near `1.2` GB after about `1:46`, `2.0` GB after
about `3:11`, `3.2` GB after about `5:26`, and `4.65` GB after about `7:44`.
The run was stopped before system memory pressure could disrupt other work.

## Interpretation

The active-side stabilization-scale fix kept D18 scales bounded through the
longer run, and all parsed nonlinear and linear solves converged before the
manual stop. The remaining blocker is retained state or allocation growth during
repeated cut-context rebuilds and assemblies, not a pressure-support singularity
or an unbounded cut-adjacent scale.
