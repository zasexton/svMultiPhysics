# D18 Profile Reference Data And Time Alignment - 2026-05-13

## Purpose

This log records the addition of SPHERIC Test05 wet-bed profile reference
tables and the first comparison attempt against the existing corrected D18 full
run output.

## Reference Data

The profile tables were copied from the archive linked on
`https://www.spheric-sph.org/tests/test-05`. The source archive is labeled
`SPHERIC_TestCase5.zip`, and the extracted directory is named
`SPHERIC_TestCase6`.

Added repository files:

- `tests/cases/fluid/open_vessel_free_surface/reference_profiles/spheric_test05_wet_bed/d18_*.dat`
- `tests/cases/fluid/open_vessel_free_surface/reference_profiles/spheric_test05_wet_bed/d38_*.dat`

The source archive stores coordinates in centimeters. The comparison script
converts them to meters.

## Snapshot Times

The archive snapshot images label the D18 profile times as:

| File | Time |
| --- | ---: |
| `d18_1.dat` | `0.156 s` |
| `d18_2.dat` | `0.219 s` |
| `d18_3.dat` | `0.281 s` |
| `d18_4.dat` | `0.343 s` |
| `d18_5.dat` | `0.406 s` |
| `d18_6.dat` | `0.468 s` |
| `d18_7.dat` | `0.531 s` |

The D38 files add `d38_8.dat` at `0.593 s`.

## Existing D18 Full-Run Check

The existing corrected D18 full run ended at `0.040 s`:

- Result: `/tmp/svmp_d18_mpi4_full_GexjZk/result_080.pvtu`
- Recorded in:
  `Documentation/qualification_logs/dam_break_d18_full_run_20260513/README.md`

That time is earlier than the first published D18 profile at `0.156 s`, so the
profile comparison below is a time-alignment diagnostic, not a pass/fail
benchmark comparison.

Command:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py \
  /tmp/svmp_d18_mpi4_full_GexjZk/result_080.pvtu \
  tests/cases/fluid/open_vessel_free_surface/reference_profiles/spheric_test05_wet_bed/d18_1.dat \
  --benchmark-json tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/benchmark.json \
  --front-diagnostic-only \
  --stale-pressure-gauge-tolerance 1.0 \
  --min-velocity-max 0.01 \
  --output Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_1_metrics.json \
  --plot-output Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_1_profile.png
```

Key output:

- Field validation: passed
- Velocity maximum: `0.43128009111533566`
- Largest velocity region: wet
- Profile RMSE against `d18_1.dat`: `0.03276055991180921 m`
- Peak-height error against `d18_1.dat`: `0.01217173138106506 m`
- Front-position role: diagnostic only
- Time-alignment issue: simulation time `0.040 s`, reference time `0.156 s`

## Next Step

Run D18 with the checked-in 312-step input to the first published profile time,
`0.156 s`, and compare `result_312` to `d18_1.dat`.
