# D18 Marked Interior-Face JIT Probe, 2026-05-15

Command:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 \
  --steps 1 \
  --linear-solver-type GMRES \
  --disable-vtk-output \
  --timeout-seconds 900 \
  --preserve-run-dir \
  --qualification-log /tmp/d18_marked_interior_jit_probe_20260515.json \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --enable-interior-face-timing \
  --require-interior-face-timing-diagnostics \
  --enable-jit-specialization-trace \
  --require-jit-specialization-trace-diagnostics \
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

- The one-step no-output probe passed.
- Preserved run directory:
  `/tmp/dam_break_d18_v8osteac/spheric_test05_wet_bed_d18`.
- The run accepted step 1 at `t=0.0005`.
- The nonlinear solve converged in 2 iterations.
- The GMRES solve converged with 73 parsed linear iterations and relative
  residual `0.0004712257940750273`.
- Parsed JIT traces included InteriorFace residual and tangent compile/hit
  events, with no marked interior-face fallback runtime-skip records.
- The largest parsed assembly time was `5.539262 s`.
- The largest parsed interior-face assembly time was `5.057048 s`; the latest
  parsed interior-face timing line reported `0.464268 s` total and
  `0.004285 s` kernel time.
- Active negative-side cut-volume assembly stayed consistent with cut-context
  rebuilds: parsed active-volume consistency error was
  `0.0004807815605545329`.

Conclusion:

Marker-specific interior-face JIT dispatch removes the cut-adjacent
stabilization fallback cost that dominated the D18 timeout run. The next
qualification step is to rerun the full D18 GMRES comparison-time probe.
