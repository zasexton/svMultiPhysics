# Active Pressure Support Constraint Probe, 2026-05-15

## Change

Inactive pressure vertex constraints now use active cell support rather than
the vertex level-set sign alone. A pressure vertex remains unconstrained when
it belongs to a cell with positive active-side volume, even when the vertex
itself is on the dry-sign side of a cut cell.

## Verification

Built the solver target:

```bash
cmake --build build/svMultiPhysics-build --target svmultiphysics -j2
```

Ran the mini direct-Eigen diagnostic probe with failure diagnostics allowed so
the pressure block could be checked separately from known dry velocity rows:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py \
  --case mini2d --steps 1 --disable-vtk-output --timeout-seconds 180 \
  --qualification-log /tmp/mini2d_pressure_support_constraint_allowed_20260515.json \
  --allow-failure-diagnostics --require-eigen-factorization-diagnostics \
  --max-eigen-factorization-pressure-zero-rows 0 \
  --max-eigen-factorization-nonfinite-entries 0 \
  --require-cut-context-solution-source-diagnostics \
  --require-assembly-topology-consistency
```

Result:

- Pressure zero rows: `0`
- Pressure zero columns: `0`
- Nonfinite matrix entries: `0`
- Remaining zero rows are in the velocity block, not the pressure block.

Ran the direct-Eigen `linear_sloshing_2d` case for two steps in a temporary
copy and verified the second output:

```bash
python3 tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/linear_sloshing_2d/verify_expected_results.py \
  /tmp/linear_sloshing_support_KMy12o/case/result_002.vtu
```

Result:

- Verification status: passed
- Pressure relative RMS error: `5.005708253739949e-06`
- Pressure relative RMS error after offset removal: `4.9915607627161905e-06`
- Velocity relative L2 error: `0.013101868384891437`
- Interface pressure RMS: `0.6425151529167369`

The linear-sloshing setup reported `support_mode=cell_patch`,
`active_support_vertices=204`, `inactive_vertices=85`, and
`inactive_sign_vertices_with_support=17`, confirming that dry-sign cut-cell
vertices with wet support were retained.
