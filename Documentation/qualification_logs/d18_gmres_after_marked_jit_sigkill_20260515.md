# D18 GMRES Post-JIT Stop Evidence, 2026-05-15

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 312 --linear-solver-type GMRES --final-output-only \
  --timeout-seconds 3600 --preserve-run-dir \
  --qualification-log /tmp/d18_gmres_full_after_jit_20260515.json \
  --max-wet-fraction-volume-error 1.0e-8 \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --max-nonlinear-iterations 9 --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 --ns-cg-tolerance 1.0e-4
```

## Result

- Run directory: `/tmp/dam_break_d18_a4lpa3p5/spheric_test05_wet_bed_d18`
- Parsed log: `/tmp/d18_gmres_full_after_jit_20260515.json`
- Solver return code: `-9`
- Accepted steps before stop: `82`
- Final accepted time before stop: `0.04100000000000003`
- Parsed nonlinear records: `82`; all reported converged.
- Parsed linear records: `82`; all reported converged.
- Maximum nonlinear iterations: `4`
- Maximum GMRES iterations: `82`
- Total GMRES iterations: `4821`
- Maximum parsed GMRES relative residual: `0.0032875684747689063`
- Maximum assembly time: `4.119248 s`
- Maximum cut-adjacent interior-face time: `2.218736 s`
- Maximum cut-volume time: `1.75598 s`
- Latest repeated assembly time near the stop: `3.696789 s`
- Active-volume consistency error: `0.0004953372382487942`
- Velocity range before stop: `0.58607414`
- Pressure range before stop: `1548.1`
- Cut-volume exactness range: order `2` to order `2`
- Active minimum volume fraction: `0.027413055469966086`
- Cut-adjacent maximum scale: `2149442.088539577`
- Cut-adjacent capped scale count: `0`
- VTK outputs: `0` because the run used final-only output and stopped before
  step 312.

## Interpretation

The pressure-support singularity and first-use marked interior-face JIT cost no
longer explain the D18 reference-time failure. The run advanced past the prior
12-step timeout point, produced fast repeated assemblies, and had parsed
nonlinear and linear convergence through the final accepted step. The remaining
blocker is a long-run resource stop or the growth of cut-adjacent stabilization
scales to very large values as the interface evolves.
