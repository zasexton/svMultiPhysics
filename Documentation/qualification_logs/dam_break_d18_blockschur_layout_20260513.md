# D18 BlockSchur Layout Routing - 2026-05-13

## Scope

This log records the D18 solver-layout change that makes the FSILS
Navier-Stokes BlockSchur path usable for the coupled D18 field order.

The coupled D18 unknown order is:

- `phi`
- `Velocity`
- `Pressure`

FSILS BlockSchur requires a primary block plus a constraint block that cover
all per-node unknowns. The D18 layout now groups `phi` and `Velocity` into a
single primary block named `LevelSetVelocity`, with `Pressure` as the
constraint block.

## Implemented Change

- `SimulationBuilder` groups the D18-style FSILS BlockSchur layout from
  `[phi(0:1), Velocity(1:3), Pressure(4:1)]` to
  `[LevelSetVelocity(0:4), Pressure(4:1)]`.
- The D18 and D38 solver XML files use fluid `<LS type="NS">`.
- The D18 and D38 solver XML files set nested BlockSchur GM and CG iteration
  limits and relative tolerances explicitly.
- The validation-mesh generator emits the same solver controls for generated
  Test05 D18/D38 fixtures.
- The open-vessel example test now checks that Test05 fixtures declare the
  BlockSchur solver controls.

## Verification

Commands:

```sh
cmake --build build/svMultiPhysics-build --target svmultiphysics -j2
cmake --build build/svMultiPhysics-build --target test_application -j2
build/svMultiPhysics-build/bin/test_application --gtest_filter='OpenVesselExamples.LiteratureValidationCasesDeclareGeneratedMeshes'
python3 -m py_compile tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py
xmllint --noout \
  tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml \
  tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml
```

Results:

- `svmultiphysics` build passed.
- `test_application` build passed.
- `OpenVesselExamples.LiteratureValidationCasesDeclareGeneratedMeshes` passed.
- `generate_validation_meshes.py` compiled.
- D18 and D38 solver XML files parsed.

## D18 One-Step Runtime Check

Temporary run directory:

- `/tmp/svmp_d18_blockschur_default_onestep_tYMVeg`

Temporary XML changes:

- `Number_of_time_steps=1`
- `Increment_in_saving_restart_files=1`

Key runtime lines:

- Solver route:
  `linear solver method=block-schur preconditioner=diagonal rel_tol=1e-04 abs_tol=1e-04 max_iter=100`
- Block layout:
  `block_layout=[LevelSetVelocity(0:4), Pressure(4:1)] saddle_point=(0,1)`
- Active-domain volume:
  `active_wet_volume=0.001587 cut_cell_active_wet_volume=0.000157407 full_cell_active_wet_volume=0.00142959`
- Result:
  `success=1 steps_taken=1 final_time=5.0e-04`

## Strict Tolerance Recheck

Temporary run directory:

- `/tmp/svmp_d18_blockschur_strict_onestep_ZFXN4h`

Temporary XML changes:

- `Number_of_time_steps=1`
- `Increment_in_saving_restart_files=1`
- nonlinear and linear relative tolerances changed from `1.0e-4` to `1.0e-6`
- linear absolute tolerances changed from `1.0e-4` to `1.0e-10`

The strict run used the intended BlockSchur route and grouped layout, but still
failed the true-residual check:

- `|Ax-b|=0.037536`
- relative residual `0.00609048`
- target `6.16307e-06`

## Current Conclusion

The D18/D38 fixtures now have a concrete FSILS BlockSchur route for the coupled
`phi`, `Velocity`, and `Pressure` layout. The default one-step D18 smoke case
passes through that route.

The strict `1e-6` D18 qualification run remains blocked by a BlockSchur true
residual floor. The next solver step should target the BlockSchur iteration
budget, residual check behavior, or Schur/momentum preconditioner depth for the
active-domain D18 system.
