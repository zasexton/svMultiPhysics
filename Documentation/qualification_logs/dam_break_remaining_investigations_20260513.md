# Dam-Break Free-Surface Follow-Up - 2026-05-13

## Scope

This note records the issues that remain before the SPHERIC Test05 D18
wet-bed dam-break case can be used as a quantitative free-surface benchmark.
It follows the MPI-4 step-312 review of
`/tmp/svmp_free_surface_d18_operator_fix_mpi4_20260513_011213/result_312.pvtu`.

## Current Result Status

The MPI-4 step-312 result is not a valid dam-break solution yet.

The level-set field advances and the extracted interface moves, but the
velocity and pressure fields do not show the expected collapse and run-up
dynamics. The current front/profile comparison should be treated only as an
interface-extraction diagnostic, not as benchmark validation.

## Observed Field Issues

- Pressure remains almost hydrostatic across the full computational volume.
- Pressure range at step 312 is `[-2.937703e+02, 1.468851e+03] Pa`.
- The hydrostatic reference error is small, with range
  `[-1.76869e+01, 0.0] Pa` and `L2 = 7.894e-01 Pa`.
- Velocity magnitude is too small for the expected transient:
  maximum `4.034e-02 m/s` and mean `1.029e-03 m/s`.
- The level-set-negative side has velocity-magnitude mean `6.13e-04 m/s`.
- The largest speeds occur near the top level-set-positive region instead of
  in the released water column.
- The simulated profile has a front location close to the digitized D18
  profile, but the vertical profile is far too low:
  `peak_y_error = -1.03043e-01 m`.

## Formulation Issues To Resolve

- The unfitted Navier-Stokes residual currently assembles inertia, convection,
  viscous, pressure, forcing, continuity, and VMS terms on `.dx()` over the
  full computational volume.
- The D18 generated fixture sets external pressure and surface tension to zero.
- Kinematic enforcement is disabled for the unfitted free surface in this
  fixture.
- With no active dynamic or kinematic surface contribution, hydrostatic
  initialization and the pressure gauge can keep the solution close to a
  full-volume hydrostatic state.
- The level set can move while the momentum field remains physically
  inconsistent with the dam-break benchmark.

## Investigation Areas

### Wet-Side Active-Domain Integration

Add or enable volume integration over the level-set water region rather than
the full vessel volume. This is the main blocker for the unfitted Test05
benchmark path.

Questions to answer:

- Does FE Forms already expose a cut-volume or level-set-side integration
  measure suitable for Navier-Stokes volume forms?
- If not, where should the active-domain measure be introduced so residuals,
  tangents, VMS terms, and output diagnostics stay consistent?
- How should partially cut cells contribute to mass, momentum, and pressure
  coupling?

### Material Weighting Alternative

If wet-side integration is not immediately available, investigate an equivalent
level-set material-weighting path.

Questions to answer:

- Should density and viscosity be weighted by a smoothed indicator field?
- Should the benchmark be represented as water-only with an inactive exterior
  region, or as a two-material water/gas calculation?
- What stabilization is needed near the interface if a smoothed material
  transition is used?

### Free-Surface Momentum Coupling

Verify that the unfitted free-surface terms drive the momentum solve when the
benchmark physics requires them.

Questions to answer:

- Which dynamic stress terms should be active for the open-vessel dam-break
  benchmark?
- Should unfitted kinematic enforcement remain disabled, or should a weak
  velocity/interface compatibility term be added?
- How should the level-set transport residual and Navier-Stokes residual share
  the same active-domain definition?

### Hydrostatic Initialization And Gauge Placement

Recheck initialization once the active-domain formulation is corrected.

Questions to answer:

- Is the current hydrostatic pressure field valid only in the initial water
  region?
- Should the pressure gauge remain at node `279`, or should the gauge be
  placed in a location that is robust for the active domain?
- Does the initial hydrostatic state mask missing momentum coupling during
  early transient checks?

### Solver Tolerances And Time Integration

The loose absolute residual tolerance may compound the physics gap by accepting
late time steps with little or no linear update.

Questions to answer:

- After the active-domain path is fixed, what nonlinear absolute and relative
  tolerances are required for the D18 time horizon?
- Should benchmark runs reject steps when the velocity or pressure update is
  effectively zero despite interface motion?
- Do serial, MPI-2, and MPI-4 runs produce matching field statistics under the
  tighter criteria?

## Required Validation Before Benchmark Use

Before treating Test05 D18 as a quantitative benchmark, the run should pass
field-level checks in addition to profile extraction:

- Pressure should depart from the initial hydrostatic full-volume state during
  collapse and run-up.
- Velocity magnitude and direction should be largest in physically plausible
  regions of the moving water column.
- Kinetic energy should grow from the released column and then evolve
  consistently across serial, MPI-2, and MPI-4.
- Front position, peak height, RMSE, MAE, and maximum profile error should be
  compared to the digitized SPHERIC D18 profile.
- The same metrics should be rerun for D38 only after the D18 field consistency
  gap is closed.

## Next Implementation Step

The next implementation step is to add wet-side active-domain integration or an
equivalent material-weighting path for unfitted level-set free-surface
Navier-Stokes benchmark cases.
