# D18 Full Run Evidence - 2026-05-13

## Scope

This log records the first D18 active-domain MPI run carried to the current
comparison time after the serial, MPI-2, and MPI-4 one-step checks passed.

## Revision And Inputs

- Code revision used for the run: `12b8867765b57fd0fe9d984a0b24deefc9922b08`.
- Solver input:
  `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml`.
- Benchmark metadata:
  `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/benchmark.json`.
- Validation script:
  `tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py`.
- Run directory: `/tmp/svmp_d18_mpi4_full_GexjZk`.
- Solver log saved in this directory as `mpi4_run.log`.
- Field metrics saved in this directory as `final_field_metrics.json`.

The temporary solver XML was identical to the committed D18 solver input.

## Command Lines

```bash
cd /tmp/svmp_d18_mpi4_full_GexjZk
mpiexec -np 4 /home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml > run.log 2>&1
```

```bash
python3 tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py \
  /tmp/svmp_d18_mpi4_full_GexjZk/result_080.pvtu \
  --benchmark-json tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/benchmark.json \
  --initial-wet-volume 0.0015869994907267658 \
  --stale-pressure-gauge-tolerance 1.0 \
  --min-velocity-max 1.0e-6 \
  --output Documentation/qualification_logs/dam_break_d18_full_run_20260513/final_field_metrics.json
```

## Solver Result

- Exit status: `0`.
- Time-loop status: `success=1`.
- Steps taken: `80`.
- Final time: `4.0000000000000001e-02`.
- Nonlinear records: `80`.
- Nonlinear iteration distribution: 4 steps with 1 iteration, 76 steps with
  2 iterations.
- Linear iteration distribution: 1 step with 15 iterations, 2 steps with
  16 iterations, 1 step with 34 iterations, 76 steps with 52 iterations.
- Nonlinear residual range: `1.2301161090270293e-05` to
  `2.1320966676181803e-03`.
- Linear relative residual range: `1.7839884457296328e-05` to
  `1.4192328243099173e-02`.

## Field Diagnostics

- Validation status: passed.
- Wet volume: `0.001574256463463625`.
- Wet volume drift from initial condition: `-1.274302726314092e-05`.
- Pressure range: `-0.4887446012009375` to `1468.8752458421454`.
- Pressure mean: `324.49865173914634`.
- Pressure gauge node: `256`.
- Pressure gauge value: `643.659423052`.
- Gauge hydrostatic-reference error: `0.0`.
- Velocity max: `0.43128009111533566`.
- Velocity mean: `0.025793725723876962`.
- Wet-side velocity mean: `0.09562483807678604`.
- Kinetic energy growth: `0.011892269698111017`.
- Largest velocity location: wet region, point `417`, `phi=-0.014241684460973018`.

The full-run result clears the near-static velocity check and no longer matches
the previous full-volume hydrostatic pressure pattern. The pressure gauge
remains on the wet side and stays at the expected initial hydrostatic reference.

## Remaining Work

- A corrected profile comparison was not run because no digitized SPHERIC
  Test05 D18 profile data file was found in the repository.
- D38 should remain deferred until the D18 profile comparison and strict
  tolerance solver path are completed.
- This full-run evidence used `Use_cut_metadata_scale=false`. A later one-step
  smoke check enabled metadata scaling after per-face constants were
  implemented; a full rerun with that option remains pending for final
  benchmark acceptance.
