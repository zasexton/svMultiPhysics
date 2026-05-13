# D18 Strict Tolerance Input Update - 2026-05-13

## Purpose

This note records the update that moves generated and checked-in Test05
unfitted inputs from the temporary strict-tolerance probe settings into the
qualification XML path.

## Input Changes

- Set generated linear solver relative tolerance to `1.0e-6`.
- Set generated linear solver absolute tolerance to `1.0e-10`.
- Set generated fluid nonlinear tolerance to `1.0e-6`.
- Set generated level-set nonlinear tolerance to `1.0e-6`.
- Applied the same tolerance profile to the checked-in D18 and D38 unfitted
  Test05 solver XML files.
- Updated open-vessel generator test expectations for BlockSchur inner GM and
  CG tolerances.

## Verification

Parser and input checks:

- `cmake --build build/svMultiPhysics-build --target test_application -j2`
- `build/svMultiPhysics-build/bin/test_application --gtest_filter='OpenVesselExamples.LiteratureValidationCasesDeclareGeneratedMeshes'`
- `python3 -m py_compile tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py`
- `xmllint --noout tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml`

Checked-in D18 strict one-step probe:

- Temporary case: `/tmp/svmp_d18_checkedin_strict_onestep_ymRr55`
- Only temporary XML edits: `Number_of_time_steps=1` and restart interval `1`.
- Runtime command:
  `SVMP_OOP_SOLVER_TRACE=1 SVMP_FSILS_NS_TRACE=1 build/svMultiPhysics-build/bin/svmultiphysics solver.xml`

Result:

- `success=1`
- `steps_taken=1`
- `final_time=5.0e-04`
- Fluid Newton convergence: `iters=3`
- Final nonlinear residual: `4.2207246435489678e-06`
- Final nonlinear relative residual: `6.84841e-07`
- Coupled outer FGMRES BlockSchur iterations: `49`, `69`, `31`
- Final linear solve report: `iters=31`, relative residual
  `5.2072521352929072e-06`
- Active wet volume remained unchanged:
  `active_wet_volume=0.001587`,
  `cut_cell_active_wet_volume=0.000157407`,
  `full_cell_active_wet_volume=0.00142959`

## Conclusion

The D18 fixture can now carry the strict solver tolerance profile directly.
The D38 fixture has the same tolerance profile, but its runtime validation
remains a separate checklist item.
