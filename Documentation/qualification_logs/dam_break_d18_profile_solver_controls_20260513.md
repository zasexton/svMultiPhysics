# D18 Profile Solver Controls - 2026-05-13

## Purpose

This note records the solver-control decision for running D18 to the first
SPHERIC Test05 profile time after the strict MPI controls proved too expensive
for the multi-step profile run.

## Rejected Profile Paths

Strict coupled outer controls:

- Run directory: `/tmp/svmp_d18_mpi4_profile_controls_nOyY4g`
- Settings: fluid Newton tolerance `5.0e-6`, fluid linear absolute tolerance
  `1.0e-7`, BlockSchur outer budget `800`
- Result: stopped manually during step 4 after a BlockSchur solve took
  `119.502843 s`, made `4490082` all-reduce calls, and hit `1000` Schur
  iterations in each of `60` Schur solves.

Relaxed nested BlockSchur tolerances:

- Run directory: `/tmp/svmp_d18_mpi4_inner1e3_5step_lj5emT`
- Settings: nested `NS_GM_tolerance=1.0e-3`,
  `NS_CG_tolerance=1.0e-3`, `NS_CG_max_iterations=300`
- Result: stopped manually after step 2. Step 2 used `172` outer iterations,
  `172` Schur solves, and `2391297` all-reduce calls.

Plain FSILS GMRES:

- Run directory: `/tmp/svmp_d18_mpi4_gmres_5step_rWpuug`
- Settings: fluid `LS type="GMRES"` with the `1.0e-7` absolute floor
- Result: failed the first nonlinear step after `12` Newton iterations with
  residual `2.8242749841982409e+01`.

Disabled level-set reinitialization and volume correction:

- Run directory: `/tmp/svmp_d18_mpi4_no_reinit_volcorr_6step_oKDYoH`
- Settings: strict coupled outer controls with
  `Enable_reinitialization=false` and `Enable_volume_correction=false`
- Result: stopped manually at the same step-4 BlockSchur stall. The blocker is
  not caused solely by those level-set maintenance operations.

## Accepted Short Probe

Loose profile controls, exploratory branch:

- Run directory: `/tmp/svmp_d18_mpi4_loose_6step_9wj1zy`
- Level-set nonlinear tolerance: `1.0e-4`
- Level-set linear relative tolerance: `1.0e-4`
- Fluid Newton cap: `8`
- Fluid nonlinear tolerance: `1.0e-4`
- Fluid linear relative tolerance: `1.0e-4`
- Fluid linear absolute tolerance: `1.0e-4`
- Coupled outer BlockSchur `Max_iterations`: `100`
- Coupled outer BlockSchur `Krylov_space_dimension`: `80`
- Nested `NS_GM_tolerance`: `1.0e-4`
- Nested `NS_CG_tolerance`: `1.0e-4`

Result:

- `success=1`
- `steps_taken=6`
- `final_time=3.0000000000000001e-03`
- Step 4 crossed the prior strict-control stall point.
- Largest short-probe BlockSchur solve: step 1, `61` outer iterations,
  `12190` all-reduce calls.
- Subsequent accepted steps used one outer iteration and no Schur solves.

## Checked-In Verification

After updating the generator and checked-in D18/D38 XML, the D18 input was run
again with only `Number_of_time_steps` changed to `6` in the temporary copy.

- Run directory: `/tmp/svmp_d18_mpi4_checked_profile_controls_6step_J5YgvT`
- Command: `mpirun -np 4 build/svMultiPhysics-build/bin/svmultiphysics solver.xml`
- Result: `success=1`
- `steps_taken=6`
- `final_time=3.0000000000000001e-03`
- Initial nonlinear residual after step 1:
  `2.1285718483844530e-05`
- Step 1 linear solve: `55` outer iterations, no Schur solves,
  `9297` all-reduce calls
- Steps 3 through 6 each used `4` to `5` outer iterations and no Schur solves.

Static checks:

```bash
python3 -m py_compile tests/cases/fluid/open_vessel_free_surface/generate_validation_meshes.py
xmllint --noout \
  tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml \
  tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml
cmake --build build/svMultiPhysics-build --target test_application -j2
build/svMultiPhysics-build/bin/test_application \
  --gtest_filter='OpenVesselExamples.LiteratureValidationCasesDeclareGeneratedMeshes'
```

## Decision

The checked-in D18/D38 Test05 profile inputs now use the loose profile controls
above. The strict one-step controls remain documented as solver-floor evidence,
but the profile comparison requires a multi-step run that can reach
`result_312` without the step-4 MPI BlockSchur stall.
