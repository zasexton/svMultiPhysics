# Free-Surface Transient Benchmark Qualification - 2026-05-13

## Scope

This log records the first benchmark-horizon free-surface transient runs for
the generated open-vessel fixtures after coupled level-set transport was routed
onto the unified transient operator.

Reference benchmark pages:

- SPHERIC Test 05 wet-bed dam-break profile data:
  https://www.spheric-sph.org/tests/test-05
- SPHERIC Test 10 sloshing tank:
  https://www.spheric-sph.org/tests/test-10

## Solver Build

- Executable:
  `build-oop-jit-20260505/svMultiPhysics-build/bin/svmultiphysics`
- Case:
  `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18`
- Temporary run directory:
  `/tmp/svmp_free_surface_d18_operator_fix_mpi4_20260513_011213`

## Test05 D18 Wet-Bed Profile, MPI-4

Run setup:

- MPI ranks: 4
- Time step: `0.0005 s`
- Steps: 312
- Final time: `0.156 s`
- VTK output: final step only
- Reference profile: `d18_1`

Run status:

- Solver completed successfully.
- Log endpoint:
  `TimeLoop: loop.run() returned success=1 steps_taken=312 final_time=1.5600000000000000e-01`.
- FE storage plan included only `operator:equations` for equation assembly,
  confirming that level-set transport was no longer isolated on a separate
  transient operator.
- Final field change by global vertex id:
  - `||phi_final - phi_initial||_inf = 2.004848e-03`
  - `||phi_final - phi_initial||_2 = 1.954269e-04`
  - final `phi` range: `[-1.500448e-02, 1.640048e-01]`

Profile comparison against `d18_1`:

| Metric | Value |
| --- | ---: |
| Coverage fraction | 1.000000 |
| Direct coverage fraction | 0.945455 |
| Simulated front x | 1.037696 m |
| Reference front x | 1.030093 m |
| Front error | 0.007603 m |
| Simulated peak y | 0.024027 m |
| Reference peak y | 0.127070 m |
| Peak y error | -0.103043 m |
| RMSE | 0.039288 m |
| MAE | 0.032967 m |
| Max absolute error | 0.103043 m |

Interpretation:

- The corrected monolithic operator advances the level-set field and produces a
  front location close to the published profile at `0.156 s`.
- The vertical profile remains far below the benchmark profile. The generated
  D18 case is therefore not yet a full quantitative validation case for the
  free-surface shape, even though front propagation is now comparable.
- The next accuracy investigation should focus on vertical free-surface motion:
  pressure/free-surface forcing, vertical velocity magnitude, contact with the
  thin wet bed, and whether the generated coarse mesh can resolve the D18
  run-up profile.

## Fitted ALE Test10 Follow-Up

The fitted ALE sloshing-tank fixture was rerun after adding the normal
kinematic mesh-motion residual. The mesh displacement became nonzero in the
accepted states, confirming that the fitted free-surface mesh is coupled to the
fluid normal velocity. The run still failed during the early transient around
the same physical time, with a large linear residual in the next nonlinear
solve.

Current finding:

- The missing normal kinematic residual was a real startup defect and is fixed.
- Fitted ALE free-surface transients still need a separate robustness
  investigation before the Test10 sloshing case can be used as a full benchmark
  comparison.
- The mesh-velocity VTK output remained zero in the accepted states even while
  mesh displacement changed, so derived mesh-velocity output should be checked
  as part of that follow-up.

## Remaining Qualification Work

- Repeat the D18 benchmark-horizon check in serial and MPI-2.
- Decide whether the D38 wet-bed case should be run after the D18 vertical
  profile gap is understood.
- Extend Test02 only after the generated case reaches the published gauge-data
  time window.
- Use Test10 benchmark comparisons only after the lateral tank-motion forcing
  and fitted ALE robustness gaps are addressed.
