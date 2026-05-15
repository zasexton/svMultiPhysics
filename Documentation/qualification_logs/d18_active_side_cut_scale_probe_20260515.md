# D18 Active-Side Cut Stabilization Probe, 2026-05-15

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 1 --linear-solver-type GMRES --disable-vtk-output \
  --timeout-seconds 900 --preserve-run-dir \
  --qualification-log /tmp/d18_active_side_cut_scale_probe_20260515.json \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --max-nonlinear-iterations 9 --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 --ns-cg-tolerance 1.0e-4
```

## Result

- Run directory: `/tmp/dam_break_d18_luna5um9/spheric_test05_wet_bed_d18`
- Parsed log: `/tmp/d18_active_side_cut_scale_probe_20260515.json`
- Probe status: passed
- Accepted steps: `1`
- Final accepted time: `0.0005`
- Nonlinear iterations: `1`
- GMRES iterations: `22`
- GMRES relative residual: `0.0003924550419360564`
- Active side: `LevelSetNegative`
- Active minimum volume fraction: `0.029094134928455338`
- Maximum cut-adjacent stabilization scale: `34.37118864194021`
- Mean cut-adjacent stabilization scale: `4.736735533484324`
- Cut-adjacent facets: `1764`
- Cut-volume exactness range: order `2` to order `2`
- Active-volume consistency error: `0.0004807815605545329`
- Velocity range: `0.06740539`
- Pressure range: `1468.85`

## Interpretation

The generated cut-adjacent facet set now uses active-side cut cells, and its
scale factors are prebound from the retained active-side cut-volume metadata.
For the D18 one-fluid case, inactive positive-side slivers no longer determine
the wet-support stabilization scale.
