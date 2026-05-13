# Dam-Break Validation Script Metrics - 2026-05-13

## Purpose

This log records the field-diagnostic update to
`tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py`.

The script can now run in field-only mode when no reference profile is supplied,
so D18/D38 runs can be checked before a profile-comparison time is reached.

## Added Metrics

- Wet volume and optional wet-volume drift from a supplied initial wet volume.
- Pressure minimum, maximum, and mean.
- Pressure gauge value and gauge error against the benchmark hydrostatic
  reference.
- Velocity maximum, mean, wet-side mean, and the region containing the largest
  velocity.
- Wet-side kinetic energy and optional kinetic-energy growth from a supplied
  initial value.
- Validation-status entries for stale gauge pressure and near-static velocity
  thresholds when those thresholds are supplied.
- A `front_position_role` field that marks the wet-bed front metric as
  `diagnostic_only` for Test05 wet-bed benchmark metadata.
- Optional profile plot output that labels the extracted `phi=0` profile and
  annotates the wet-bed front position as diagnostic-only.

## Smoke Runs

The updated script was run on the existing one-step D18 serial, MPI-2, and MPI-4
outputs with:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py \
  RESULT_FILE \
  --benchmark-json tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/benchmark.json \
  --initial-wet-volume 0.0015869994907267658 \
  --stale-pressure-gauge-tolerance 1.0 \
  --min-velocity-max 1.0e-6 \
  --output field_metrics.json
```

| Run | Validation status | Wet volume | Pressure min | Pressure max | Gauge pressure | Velocity max | Largest-velocity region |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Serial | Passed | `0.001587012742148248` | `-0.004173629881460488` | `1468.8514642144446` | `643.659423052` | `0.04750725550163312` | wet |
| MPI-2 | Passed | `0.0015870057359813985` | `-0.004138792283994896` | `1468.851462771724` | `643.659423052` | `0.047507123642518634` | wet |
| MPI-4 | Passed | `0.0015870075364175366` | `-0.004323321966844228` | `1468.8514938250307` | `643.659423052` | `0.04750712085642338` | wet |

The plot path was smoke-tested with a temporary reference profile and wrote
`/tmp/svmp_d18_profile_plot.png`. The generated JSON reported
`front_position_role=diagnostic_only`.
