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

## Extended Fluid Tolerance Probe

The 312-step profile run with `1.0e-4` fluid nonlinear tolerance reached step
13 and failed at step 13 with residual `1.7053193047519049e-04`. Raising only
the fluid Newton cap from `8` to `12` reproduced the same stalled residual.

A second probe raised only the fluid nonlinear tolerance to `2.0e-4` and
reached step 14, then failed at step 14 with residual
`2.7857640075002691e-04`.

The accepted extended probe raised only the fluid nonlinear tolerance to
`5.0e-4`:

- Run directory: `/tmp/svmp_d18_mpi4_loose_fluid5e4_30step_enoi2A`
- `success=1`
- `steps_taken=30`
- `final_time=1.4999999999999999e-02`
- Highest reported accepted residual after step 13:
  `3.6182047079735206e-04` at step 15
- The run maintained one Newton iteration per step and did not re-enter the
  strict-control Schur stall.

## Step 50 Residual Floor Follow-Up

The corrected 312-step D18 MPI-4 profile run with fluid nonlinear tolerance
`5.0e-4` advanced through `result_050.pvtu` and then stopped at the start of
step 50:

- Run directory: `/tmp/svmp_d18_mpi4_profile_5e4_fixed_oq4p8M`
- Stop point: step `50`, time `2.5000000000000019e-02`
- Final reported nonlinear residual:
  `5.1076666039798896e-04`
- Fluid Newton cap: `8`

A follow-up 60-step probe raised the fluid nonlinear tolerance to `6.0e-4`
and raised the fluid Newton cap to `9`:

- Run directory: `/tmp/svmp_d18_mpi4_fluid6e4_max9_60step_zWqx9W`
- Command: `mpirun -np 4 /home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml`
- Result: `success=1`
- `steps_taken=60`
- `final_time=2.9999999999999999e-02`
- Step 50 residual:
  `5.1076639678221175e-04`
- Step 50 acceptance: `converged=1`, `iters=1`
- Highest residual after the step-50 floor in the 60-step probe:
  `5.1076639678221175e-04`

This confirms the previous `5.0e-4` setting was just below the observed
profile-run residual floor at step 50. The `6.0e-4` setting keeps the accepted
residual below `1.0e-3` while allowing the run to move beyond the first
profile-run obstruction.

## Full Profile Attempt With Step 50 Controls

The checked-in D18 input using fluid nonlinear tolerance `6.0e-4` and fluid
Newton cap `9` was then run without reducing `Number_of_time_steps`.

- Run directory: `/tmp/svmp_d18_mpi4_profile_6e4_max9_312step_aNYi90`
- Command: `mpirun -np 4 /home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml`
- Result: stopped at step `63`
- Stop time: `3.1500000000000021e-02`
- Final reported nonlinear residual:
  `6.2761512021829798e-04`
- Fluid Newton cap: `9`
- Linear solve at the floor: `converged=1`, `iters=0`,
  `rel=1.5716529455800737e-01`
- Last written result: `result_063.pvtu`

This shows the `6.0e-4` setting crosses the step-50 floor but is still below
the next MPI profile-run residual floor. The next profile-control probe should
raise only the fluid nonlinear tolerance in a narrow increment and verify that
the early residual history remains comparable to the `6.0e-4` run.

## Incremental 7.0e-4 Probe

A temporary 100-step D18 MPI-4 probe raised only the fluid nonlinear tolerance
from `6.0e-4` to `7.0e-4`.

- Run directory: `/tmp/svmp_d18_mpi4_fluid7e4_max9_100step_BVHWNC`
- Command: `mpirun -np 4 /home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver.xml`
- Step 63 result: crossed the previous floor with residual
  `6.2761451707296309e-04`, `converged=1`, `iters=1`
- Result: stopped at step `67`
- Stop time: `3.3500000000000023e-02`
- Final reported nonlinear residual:
  `7.5725252640414459e-04`
- Linear solve at the floor: `converged=1`, `iters=0`,
  `rel=9.2415067301916454e-02`
- Last written result: `result_067.pvtu`

The early residual history matched the stable `6.0e-4` run, but `7.0e-4` is
not sufficient for the profile comparison because it only moves the residual
floor from step 63 to step 67.

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
above with fluid nonlinear tolerance `6.0e-4` and fluid Newton cap `9`. The
strict one-step controls remain documented as solver-floor evidence, but the
profile comparison still requires a multi-step run that can reach `result_312`
without the step-4 MPI BlockSchur stall or the later Newton residual floors.
