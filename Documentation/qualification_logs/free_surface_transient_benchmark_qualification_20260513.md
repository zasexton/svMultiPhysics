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

Field consistency check at step 312:

- Pressure range: `[-2.937703e+02, 1.468851e+03] Pa`.
- Pressure is nearly identical to the hydrostatic reference
  `rho * 9.81 * (0.15 - y)`, with global error range
  `[-1.76869e+01, 0.0] Pa` and `L2 = 7.894e-01 Pa`.
- Velocity magnitude has maximum `4.034e-02 m/s` and mean
  `1.029e-03 m/s`.
- The level-set-negative side has velocity-magnitude mean
  `6.13e-04 m/s`; the largest speeds occur near the top
  level-set-positive region instead of in the released water column.

Interpretation:

- The corrected monolithic operator advances the level-set field and produces a
  measurable interface displacement at `0.156 s`.
- The velocity and pressure fields show that this is not a valid dam-break
  benchmark solution. The pressure remains almost exactly hydrostatic across
  the full computational volume, and the velocity field is too small and too
  weakly localized to represent the published Test05 transient.
- The profile comparison is therefore only an interface-extraction diagnostic.
  It must not be treated as quantitative free-surface validation until the
  unfitted Navier-Stokes volume terms are restricted or weighted by the
  level-set water region.

Current root-cause finding:

- The unfitted Navier-Stokes residual assembles inertia, convection, viscous,
  pressure, forcing, continuity, and VMS terms on `.dx()` over the full
  computational volume.
- The generated D18 fixture sets external pressure and surface tension to zero
  and leaves kinematic enforcement disabled, so the unfitted free-surface
  boundary branch adds no dynamic or kinematic surface contribution.
- With hydrostatic pressure initialization and a pressure gauge, the current
  system can remain close to a full-volume hydrostatic state while the level
  set moves. A wet-side active-domain integration or material weighting path is
  required before this fixture can be used for benchmark accuracy.

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

- Implement wet-side active-domain integration or equivalent material weighting
  for unfitted level-set free-surface Navier-Stokes forms.
- After that implementation is available, repeat the D18 benchmark-horizon
  check in serial, MPI-2, and MPI-4.
- Decide whether the D38 wet-bed case should be run after the D18 field
  consistency gap is resolved.
- Extend Test02 only after the generated case reaches the published gauge-data
  time window.
- Use Test10 benchmark comparisons only after the lateral tank-motion forcing
  and fitted ALE robustness gaps are addressed.
