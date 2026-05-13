# D18 Coupled Outer FGMRES Solver Path - 2026-05-13

## Purpose

This note records the solver-path change used to complete a strict one-step D18
dam-break solve with the grouped `phi`, `Velocity`, and `Pressure` field order.
The prior BlockSchur route used a native fractional-step direction for the
`LevelSetVelocity(0:4), Pressure(4:1)` split. Increasing the legacy BlockSchur
budget alone did not produce an acceptable nonlinear step, and transient line
search rejected the resulting direction.

## Implemented Solver Path

- Added the XML control `NS_Use_coupled_outer_FGMRES`.
- Routed that control through `LinearSolverParameters`, `SimulationBuilder`,
  `SolverOptions`, and `FSILS_lsType`.
- When the scalar-constraint BlockSchur split is active and the control is
  enabled, `ns_solver` uses the existing full-system outer FGMRES solve
  preconditioned by the BlockSchur split.
- Treated the configured coupled-outer route as a larger-budget BlockSchur path,
  so the outer cap is 200 instead of the legacy 50-iteration cap.
- Let the coupled outer FGMRES workspace honor the configured outer iteration
  budget, rather than stopping at the XML Krylov dimension when the outer budget
  is larger.
- Enabled the control for generated and checked-in D18/D38 Test05 BlockSchur
  inputs, with fluid `Max_iterations` set to 200.

## Diagnostic Finding During Implementation

The first strict D18 probe exposed an iteration-accounting bug:

- Coupled outer FGMRES converged internally at 69 outer iterations.
- The backend report still used the legacy 50-iteration BlockSchur sanity cap.
- That cap marked the solve as `breakdown:itr` and zeroed the returned update.

The implemented cap and workspace changes removed that failure mode.

## Verification

Build and parser checks:

- `cmake --build build/svMultiPhysics-build --target svmultiphysics -j2`
- `cmake --build build/svMultiPhysics-build --target test_application -j2`
- `build/svMultiPhysics-build/bin/test_application --gtest_filter='OpenVesselExamples.LiteratureValidationCasesDeclareGeneratedMeshes'`
- `python3 -m py_compile tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py`
- `xmllint --noout tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml`

Strict D18 one-step probe:

- Temporary case: `/tmp/svmp_d18_coupled_strict_onestep_TW3clU`
- Runtime command:
  `SVMP_OOP_SOLVER_TRACE=1 SVMP_FSILS_NS_TRACE=1 build/svMultiPhysics-build/bin/svmultiphysics solver.xml`
- Temporary XML changes:
  `Number_of_time_steps=1`, restart interval `1`, nonlinear tolerances
  `1.0e-6`, linear relative tolerances `1.0e-6`, and absolute linear
  tolerances `1.0e-10`.

Strict D18 result:

- `success=1`
- `steps_taken=1`
- `final_time=5.0e-04`
- Fluid Newton convergence: `iters=3`
- Final nonlinear residual: `4.2207246435489678e-06`
- Final nonlinear relative residual: `6.84841e-07`
- Coupled outer FGMRES BlockSchur iterations: `49`, `69`, `31`
- Final linear solve report: `iters=31`, relative residual
  `5.2072521352929072e-06`
- Active wet volume remained consistent with the D18 initial state:
  `active_wet_volume=0.001587`,
  `cut_cell_active_wet_volume=0.000157407`,
  `full_cell_active_wet_volume=0.00142959`

## Remaining Work

The checked-in D18/D38 fixture tolerances still use the nominal `1.0e-4`
settings. The next remediation step is to tighten the generated qualification
inputs to the strict tolerance profile now that the coupled outer solver path
has a passing D18 one-step probe.
