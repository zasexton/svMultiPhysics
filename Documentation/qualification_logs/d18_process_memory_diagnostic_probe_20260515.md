# D18 Process-Memory Diagnostic Probe, 2026-05-15

## Command

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case d18 --steps 1 --linear-solver-type GMRES --disable-vtk-output \
  --timeout-seconds 900 --preserve-run-dir \
  --qualification-log /tmp/d18_process_memory_probe_20260515.json \
  --require-process-memory-diagnostics \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency \
  --max-nonlinear-iterations 9 --linear-relative-tolerance 6.0e-4 \
  --linear-absolute-tolerance 1.0e-4 --linear-max-iterations 100 \
  --ns-gm-max-iterations 150 --ns-cg-max-iterations 150 \
  --ns-gm-tolerance 1.0e-4 --ns-cg-tolerance 1.0e-4
```

## Result

- Run directory: `/tmp/dam_break_d18_upkm5kax/spheric_test05_wet_bed_d18`
- Parsed log: `/tmp/d18_process_memory_probe_20260515.json`
- Probe status: passed
- Accepted steps: `1`
- Final accepted time: `0.0005`
- Process-memory records parsed: `10`
- Maximum resident memory: `408304` kB
- Resident-memory growth over parsed records: `223880` kB
- Maximum virtual memory: `769496` kB
- Maximum assemble-operator time: `3.320879` s
- Maximum cut-adjacent stabilization scale: `34.37118864194021`

## Interpretation

The smoke script now parses process memory from cut-context rebuild records and
from one compact assemble-operator diagnostic line. The probe verifies that
memory growth can be checked without VTK output, which gives the next D18
resource investigation a direct per-rebuild and per-assembly signal.
