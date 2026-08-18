# Level-Set Free-Surface Support Status

This note records what the `linear_sloshing_2d` case currently demonstrates,
what has now been accepted for the OOP regression path, and the documented
limits of the unfitted level-set free-surface boundary condition in the new
OOP solver.

## Current Status

The case now exercises the intended single-phase free-surface path:

- incompressible Navier-Stokes integrated only over the active liquid side of
  a generated level-set cut volume
- level-set transport coupled to the fluid `Velocity` field
- weak kinematic coupling on the generated free-surface interface
- natural zero dynamic pressure traction for `External_pressure=0` and
  `Surface_tension=0`
- moving cut-volume active-domain tangents with the level-set field included
  in the monolithic unknown vector
- cut-adjacent ghost stabilization and wet-volume diagnostic output
- output at every time step with the time series collected into `result.pvd`

The latest 100-step diagnostic run showed that the case completes and the
current strict verifier passes. The active fluid classification is consistent
with the computed `phi` sign and with the analytic wet/dry sign at mesh
vertices. The accepted nonlinear residuals are not pressure dominated; by the
last accepted step the residual is primarily in the vertical velocity block,
with pressure several orders smaller.

Representative step-100 metrics from the recent run were:

```text
interface_l2_height_error                         8.87e-5
interface_max_height_error                        2.21e-4
velocity_relative_l2_error                        1.85e-2
pressure_relative_rms_error                       9.48e-5
pressure_relative_rms_error_after_offset_removal  9.23e-5
relative_area_error                               1.29e-5
interface_pressure_max_abs                        9.16
```

The run also reports no interface-analysis errors for the generated marker and
keeps the wet-volume drift below `6.5e-6` over the 100 accepted steps.

## 2026-05-17 Regression Acceptance

The OOP unfitted level-set free-surface regression path is accepted for:

- generated level-set cut-volume assembly for one-phase free-surface flow
- generated free-surface interface contribution lookup by provenance marker
- wet-domain pressure verification with pressure-gauge offset handling
- linear sloshing interface-height, velocity, pressure, and active-area checks
- a compact time-step/mesh refinement table recorded in
  `validation_summary_20260517.json`
- a compact cut-cell stabilization sweep recorded in
  `validation_summary_20260517.json`
- the offset flat open-vessel hydrostatic case, which avoids a mesh-aligned
  degenerate interface and verifies the same generated-interface path on a
  different problem
- serial and `mpi2` execution of the other existing OOP incompressible
  Navier-Stokes smoke fixtures using both `solver_perf_oop.xml` and
  `solver_new.xml`:
  `pipe_simple`, `Channel2D_Simple`, `Channel2D`, `vortex_shedding`,
  `pipe_RCR_3d`, and `iliac_artery`

This is an acceptance of the current OOP support workflow under the documented
linear-reference assumptions below. It is not a claim that the linear sloshing
analytic reference is an exact viscous no-slip Navier-Stokes solution.

## Validation Notes

### Analytic Target Is A Linear Reference

The verifier compares against the small-amplitude inviscid potential-flow
sloshing solution. That is the right regression target for this case, but it
is not the exact solution of viscous nonlinear Navier-Stokes with no-slip wall
physics. The XML currently prescribes exact time/space velocity data on the
left, right, and bottom walls because that is the available OOP fallback for
the analytic slip-wall boundary data.

This case should therefore be interpreted as a linear-regime free-surface
method check, not as a final validation of all physical wall models.

### Interface And Velocity Errors Are Still The Largest Useful Signal

The pressure comparison in the wet interior is already very accurate. The
remaining meaningful discrepancy is the velocity/free-surface motion:

- the free-surface cosine amplitude still lags the analytic value
- the velocity relative L2 error is still around a few percent on the coarse
  default mesh
- the interface error improves when the time step is reduced, which points to
  temporal/interface-advection accuracy rather than a pressure-residual failure

The 2026-05-17 validation study shows the expected useful trends under
time-step and mesh refinement:

```text
variant          iface_l2      iface_max     vel_rel      p_rel_shift   area_rel
baseline_16      1.2849e-4     3.7906e-4     3.230e-2     1.062e-4      1.955e-6
dt_half_16       1.2291e-4     3.6336e-4     2.589e-2     1.540e-4      1.965e-6
mesh_24          9.9878e-5     3.5323e-4     3.482e-2     5.131e-5      5.327e-7
mesh24_dt_half   1.0250e-4     3.6996e-4     2.857e-2     8.033e-5      6.481e-7
```

The smaller time step improves the interface and velocity errors. The finer
mesh improves interface, pressure, and active-area errors. The combined
finer-mesh/smaller-time-step run improves velocity relative to the baseline.
The pressure metric is already near the numerical floor for this coarse
regression and is not expected to be the dominant convergence signal.

### `phi` Should Be Judged By The Interface, Not Whole-Field Signed Distance

The transported `phi` field is currently meaningful primarily through its
zero contour and active-domain sign. A whole-domain comparison against the
analytic signed-distance-like expression can overstate the error because the
method does not currently reinitialize or constrain the full field to remain
an exact signed-distance extension.

The important checks for this case are:

- reconstructed `phi=0` free-surface height
- active wet/dry classification
- active liquid area conservation
- local behavior in a narrow band around the interface

Whole-field `phi` error should not be used as the main pass/fail criterion
unless reinitialization or signed-distance preservation is explicitly part of
the method being tested.

### Interface Pressure Is A Fragile Diagnostic

The pressure residual is no longer the dominant failure mode, and the wet
interior pressure error is small. However, direct pressure interpolation onto
the reconstructed interface remains a fragile diagnostic because interface
edges may involve cut cells and dry-side pressure DOFs that are constrained or
not physically meaningful for the one-phase liquid problem.

For support qualification, pressure should be checked in the active wet
domain, with gauge-offset handling where appropriate. Interface pressure can
remain a diagnostic, but it should not be the only evidence for the dynamic
free-surface condition unless the sampling is restricted to physically active
traces.

### Cut-Cell Stabilization Characterization

Disabling cut-cell stabilization changes the answer and reduces some
interface-pressure diagnostics. Sweeping the nonzero velocity and pressure
gradient penalties over several orders of magnitude showed a weak response in
the current metrics, while enabling versus disabling the stabilization path
has a much larger effect.

Focused unit tests indicate that the generic `cutStabilizationScale()` and
marked cut-adjacent facet JIT/tangent paths are not ignoring the scale. The
case-level behavior is now characterized for this regression: enabling
stabilization regularizes weak cut/dry modes, while the accepted output metrics
remain within the verifier tolerances across the sweep below.

The 2026-05-17 stabilization sweep characterizes the current regression
behavior:

```text
variant                 iface_l2      vel_rel      p_rel_shift   area_rel    interface_pmax
disabled                1.2846e-4     3.103e-2     9.908e-5      1.847e-6    1.60
nearzero                1.2777e-4     3.181e-2     1.037e-4      1.969e-6    3.90
velocity_dominant_eps   1.2777e-4     3.181e-2     1.037e-4      1.969e-6    3.90
pressure_dominant_eps   1.2849e-4     3.230e-2     1.062e-4      1.955e-6    3.84
strong_10               1.2849e-4     3.230e-2     1.062e-4      1.957e-6    3.84
```

All listed variants complete and pass the verifier. The exactly zero pressure
penalty with velocity-only stabilization is singular for the direct-solve
regression because it leaves zero pressure rows; isolated velocity-dominant
checks therefore use a small nonzero pressure penalty.

### Output-Time Semantics For Algebraic Fields

The first-order generalized-alpha loop solves at a stage time and then
extrapolates time-differentiated fields to the end of the step. Algebraic
fields such as pressure are preserved from the stage solve rather than
extrapolated. The verifier therefore treats pressure as an algebraic stage
field and compares it in the active wet domain with gauge-offset handling,
rather than relying on a direct interface pressure trace as the primary
pass/fail signal.

## Goal For Full Support

The long-term goal is for an unfitted level-set free-surface boundary problem
to be a normal supported OOP workflow, not a special-case experiment. At that
point a user should be able to prescribe a one-phase liquid region through a
level-set field and rely on the solver to:

- assemble Navier-Stokes volume terms only over the active fluid domain
- update that active domain consistently as `phi` moves
- include the correct monolithic tangents from the moving active cut volume
- enforce or weakly couple the free-surface kinematic condition
- apply the dynamic free-surface traction condition naturally for zero
  external pressure and surface tension, and explicitly for nonzero external
  pressure or surface tension when those options are requested
- constrain or ignore inactive-region unknowns without contaminating active
  liquid pressure and velocity diagnostics
- stabilize small cut cells without dominating the physical solution
- run with JIT enabled and with fallback paths producing equivalent residuals
  and tangents
- save interpretable `phi`, `ActiveFluid`, `WetVolumeFraction`, `Velocity`,
  and `Pressure` outputs

## Acceptance Criteria Status

The current state against the original acceptance criteria is:

1. Accepted: the linear sloshing case passes
   stricter verification on pressure, velocity, interface height, active
   liquid area, and active fluid classification.
2. Accepted: mesh/time-step refinement trends are recorded in
   `validation_summary_20260517.json` and summarized above.
3. Accepted: pressure checks are wet-domain based,
   with explicit pressure-gauge offset handling and interface pressure kept as
   diagnostic-only.
4. Accepted: the sloshing case exercises generated-interface kinematic
   coupling, and `LevelSetTransport.InterfaceKinematicAddsInterfaceResidual`
   covers the interface kinematic residual independently from the volume
   advection term.
5. Accepted: cut-cell stabilization parameter behavior is documented in
   `validation_summary_20260517.json` and summarized above.
6. Accepted: moving cut-volume tangents and generated interface paths are
   covered by focused FE form/system tests, including fixed-geometry finite
   difference, generated-rule JIT/interpreter parity, and active-domain
   residual sign-change checks.
7. Accepted: the offset flat open-vessel hydrostatic case is a second unfitted
   generated-interface problem and passes its verifier with the current code.
8. Accepted: JIT, interpreter, and marked/generated cut-interface fallback
   paths agree in the focused FE form/system tests listed in
   `validation_summary_20260517.json`.
9. Accepted: diagnostics remain available for active-domain rebuilds,
   interface quadrature, cut-adjacent facet sets, residual block norms, wet
   volume, and output field consistency.

## Non-Blocking Follow-Up

The next useful work is broader validation beyond the current acceptance
criteria: promote a nontrivial second dynamic one-phase level-set free-surface
case, such as square-tank settling or an MMS-style traveling interface, after
it passes fresh current-code runs rather than only saved-output diagnostics.
