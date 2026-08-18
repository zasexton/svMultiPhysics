# Open-Vessel Pressure Update Guard Diagnostic Progress

Date: 2026-06-05

Scope: targeted progress on the Test02/Test10 root-cause goal. This note documents the new accepted-pressure-update diagnostic and the evidence it adds to `Documentation/open_vessel_free_surface_test02_test10_root_cause_report_20260605.md`. It is not a qualification closure.

## What Changed

Added `tests/cases/fluid/open_vessel_free_surface/audit_pressure_update_guard.py`.

The diagnostic compares consecutive saved `result_*.vtu` or `result_*.pvtu` files and reports pressure increments by support class:

- `all_points`
- `active_or_wet_supported`
- `full_wet_supported`
- `cut_supported`
- `tiny_cut_supported`

The support classification uses saved `Pressure`, optional `phi`, optional `ActiveFluid`, optional `Velocity`, and incident positive `WetVolumeFraction` cell data. When a solver log is provided, the report also attaches accepted-step context, nonlinear residual data, rejected attempts, and cut-context rebuild metadata.

Added `tests/test_open_vessel_pressure_update_guard.py` to cover support classification and solver-log parsing.

## Verification

Focused checks passed:

```bash
python -m py_compile tests/cases/fluid/open_vessel_free_surface/audit_pressure_update_guard.py
pytest -q tests/test_open_vessel_pressure_update_guard.py
```

Pytest result: `2 passed in 0.81s`.

## Diagnostic Runs

Test02 terminal window:

```bash
python tests/cases/fluid/open_vessel_free_surface/audit_pressure_update_guard.py \
  --case-dir Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_structured_h0p15_all_solid_normal_only_abs_only_prune1e5_0p54_case \
  --start-step 380 --end-step 383 \
  --absolute-threshold-pa 100000 \
  --json-output Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_abs_only_prune1e5_0p54_pressure_update_guard_20260605.json
```

Test10 step90 window:

```bash
python tests/cases/fluid/open_vessel_free_surface/audit_pressure_update_guard.py \
  --case-dir Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_1s_case \
  --solver-log Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_roll_full_source_dt0p01_tightvol_adaptive_relaxed_ls_max20_metadata_cap3_1s_solver_stdout_20260604.log \
  --start-step 88 --end-step 91 \
  --absolute-threshold-pa 100 \
  --json-output Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_cap3_step90_pressure_update_guard_20260605.json
```

## Evidence

### Test02

Output: `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test02_abs_only_prune1e5_0p54_pressure_update_guard_20260605.json`.

The diagnostic triggered on 2 transitions above the 100000 Pa active/wet threshold. The worst active/wet pressure update is from step 382 to 383:

- pressure increment: 2112204.128955333 Pa
- previous pressure: 15206.866204518465 Pa
- current pressure: 2127410.9951598514 Pa
- support class: `full_wet_supported`
- point index: 1172
- point: `[0.8245, 0.0, 0.5]`
- `phi`: -0.06916536196062417
- `ActiveFluid`: 1.0
- incident `WetVolumeFraction` max: 1.0
- incident positive `WetVolumeFraction` min: 1.0

The worst tiny-cut-supported update in the same transition is also large, but smaller:

- pressure increment: 1125660.528687608 Pa
- support class: `tiny_cut_supported`
- point index: 1171
- point: `[0.6635, 0.0, 0.5]`
- `phi`: 0.6384614226245027
- `ActiveFluid`: 0.0
- incident `WetVolumeFraction` max: 1.648087280335411e-05

Interpretation: the Test02 accepted-output MPa spike is not only an inactive interpolation or tiny-cut artifact. A tiny-cut pressure event is present, but the maximum accepted update is on a fully wet active node.

### Test10

Output: `Documentation/qualification_logs/open_vessel_free_surface_remaining_20260526/test10_cap3_step90_pressure_update_guard_20260605.json`.

The diagnostic triggered on 1 transition above the 100 Pa active/wet threshold. The worst active/wet pressure update is from step 90 to 91:

- pressure increment: 1075.2113565356985 Pa
- previous pressure: 774.7209664196774 Pa
- current pressure: 1849.932322955376 Pa
- support class: `full_wet_supported`
- point index: 3
- point: `[0.0, 0.0365, 0.031]`
- `phi`: -0.05930530155554565
- `ActiveFluid`: 1.0
- incident `WetVolumeFraction` max: 1.0
- incident positive `WetVolumeFraction` min: 1.0

The worst cut-supported update in the same transition is lower:

- pressure increment: 702.472832182176 Pa
- support class: `cut_supported`
- point index: 629
- point: `[0.9, 0.113, 0.062]`
- `phi`: 0.020432590293180418
- `ActiveFluid`: 0.0
- incident `WetVolumeFraction` max: 0.4335481638942627
- incident positive `WetVolumeFraction` min: 0.07500639991719975

There is no `tiny_cut_supported` maximum event in this accepted window.

The attached solver-log context for the step 90 to 91 transition reports an accepted `dt=0.000625`, 1 nonlinear iteration, nonlinear residual `0.0009696672726394497`, and linear relative residual about `1.938e-13`. The accepted attempt follows rejected attempts at `dt=0.01`, `0.005`, `0.0025`, and `0.00125`. The accepted cut context includes `active_min_volume_fraction=0.044697352262951316`, `active_wet_cells=720`, `cut_adjacent_capped_scale=0`, `cut_adjacent_max_scale=22.37`, `generated_pruned_volume_rules=8`, and `generated_pruned_volume=5.67e-10`.

Interpretation: the Test10 accepted pressure jump is a real full-wet active pressure update accepted by the time-step/nonlinear controller, not just sensor interpolation, postprocessing, or a tiny retained cut volume.

## Hypothesis Status

Sampling or postprocessing-only explanation: further ruled out for the accepted pressure-update windows. Both cases show the largest accepted updates on active/full-wet support.

Tiny-cut-only explanation: further ruled out for the largest accepted updates. Test02 still has a secondary tiny-cut pressure update, so tiny cuts remain a stability hazard, but they do not explain the largest accepted full-wet update. Test10 has no tiny-cut-supported event in the audited accepted window.

Timestep acceptance gap: supported as a guard problem. Both cases can accept pressure increments that are large relative to the intended pressure scale while residual checks look acceptable. This motivates an in-solver accepted-step pressure update diagnostic or guard, but the guard should not be treated as the physics fix.

Active cut-volume pressure consistency: supported. The updates occur on active/full-wet pressure support in cases using generated free-surface cut volumes, inactive pressure constraints, and pressure ghost penalty.

Pressure ghost penalty versus active-volume pressure residual versus inactive pressure support constraints: unresolved. The current diagnostic shows where the accepted pressure update appears, but it does not decompose residual contributions.

## Next Narrow Tests

1. Add a cut-adjacent pressure stabilization contribution audit for the saved Test02 terminal state and Test10 step90 state. It should report per-facet pressure ghost-penalty residual contribution, stabilization scale, wet fraction, support mode, and the local pressure increment.

2. Add a hydrostatic or linear-pressure cut-volume patch regression with active `LevelSetNegative` cut volumes, inactive pressure constraints, and pressure ghost penalty enabled. The pass condition should be near-zero pressure/continuity residual and no pressure update for a known consistent state as the interface cuts cells.

3. Prototype an in-solver accepted-step pressure update diagnostic only after the active/full-wet threshold is made defensible. The guard should report support class and residual context, and it should be a safety net rather than a substitute for a pressure-formulation fix.
