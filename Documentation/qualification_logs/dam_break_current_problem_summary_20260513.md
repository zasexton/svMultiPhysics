# Dam-Break Current Problem Summary - 2026-05-13

## Current Status

The D18 wet-bed dam-break case can now run past the earlier MPI-4 solver stop,
but the resulting fields are still not a valid dam-break solution. The solver
marching problem and the physical accuracy problem are separate:

- The previous zero-update residual floor at step `63` was addressed by forcing
  at least one FSILS BlockSchur outer iteration for the coupled D18 layout.
- The full D18 MPI-4 run now reaches `result_312.pvtu` at `0.156 s`.
- The result remains physically wrong: the interface barely evolves, the
  velocity field is sparse and too small, and the pressure field remains close
  to a constrained hydrostatic state.

D18 and D38 qualification should remain blocked.

## Main Failure Signatures

- The `phi=0` profile does not show the expected dam-break collapse and run-up.
  In the full MPI-4 run at `0.156 s`, the extracted D18 profile has RMSE
  `0.03365458346158355 m` against `d18_1.dat`.
- The final D18 interface profile remains too close to the initial retained
  water-column shape. Across sampled outputs from step `50` through step `310`,
  the highest interface point stays near `0.1509 m`.
- The final D18 velocity is far too small for the expected gravity-driven
  collapse:
  - `velocity_max = 0.12408780231518766 m/s`
  - `velocity_mean = 0.0025161748510761743 m/s`
  - `velocity_wet_mean = 0.008466118058423023 m/s`
  - `kinetic_energy = 0.00024711159741521333 J`
- The pressure gauge remains exactly fixed at the initial hydrostatic value:
  `643.659423052 Pa` at node `256` through `result_312`.
- The pressure range remains hydrostatic-like at the final profile time:
  - `pressure_min = -0.00025860192756981036 Pa`
  - `pressure_max = 1469.0132794991368 Pa`
  - wet-side pressure mean near `777.58 Pa`
- The largest velocity is in the wet region, but it is localized and small.
  In `result_310.pvtu`, the largest speed is at GlobalVertexID `452`, with
  speed `0.12408664632689799 m/s`.
- The user-reported node `486` should be rechecked against the exact output
  numbering. In the merged `result_310.pvtu` file, point index `486` is on the
  dry top boundary with speed `5.307178052090245e-04 m/s`, and there is no
  `GlobalNodeID` or `GlobalVertexID` equal to `486` in that merged output.

## Implemented Foundation

The following remediation work is already in place and covered by tests or
logs:

- Active-domain free-surface configuration:
  `Active_domain=LevelSetNegative` and `Active_domain_method=CutVolume`.
- Generated cut-volume rules for wet-side Navier-Stokes volume assembly.
- Navier-Stokes and VMS volume terms routed through the active wet-side
  cut-volume measure.
- Wet-side hydrostatic pressure initialization and dry-side reference pressure
  handling.
- Pressure-gauge metadata validation for active-domain cases.
- D18 gauge changed from the old near-interface node to wet node `256`.
- Profile comparison script now reports field metrics, wet volume, pressure,
  velocity, kinetic energy, and reference profile errors.
- D18 and D38 Test05 generated fixtures now use the grouped BlockSchur layout:
  `LevelSetVelocity(0:4), Pressure(4:1)`.
- FSILS BlockSchur coupled outer FGMRES path exists for the grouped
  `phi`, `Velocity`, and `Pressure` layout.
- FSILS BlockSchur now supports `NS_min_outer_iterations`, routed from XML to
  the coupled outer FGMRES path and legacy loop.

## Solver Probes And Outcomes

### Strict Tolerance Attempts

- Strict D18 one-step settings with nonlinear tolerance `1.0e-6`, linear
  relative tolerance `1.0e-6`, and linear absolute tolerance `1.0e-10` exposed
  a true-residual floor in MPI-4.
- A strict MPI-4 probe failed at the initial solve with true residual
  `|Ax-b|=0.0425642`, relative residual `0.00690591`, and target
  `6.16345e-06`, even after large BlockSchur work.
- Conclusion: strict tolerances cannot yet be restored with the present
  coupled BlockSchur scaling.

### Profile-Run Tolerance Probes

- Fluid nonlinear tolerance `5.0e-4` stopped at step `50` with residual
  `5.1076666039798896e-04`.
- Fluid nonlinear tolerance `6.0e-4` crossed step `50` and completed a
  60-step probe, but the full profile run stopped at step `63` with residual
  `6.2761512021829798e-04`.
- Fluid nonlinear tolerance `7.0e-4` crossed step `63` but stopped at step
  `67` with residual `7.5725252640414459e-04`.
- Fluid nonlinear tolerance `8.0e-4` crossed step `67` but stopped at step
  `78` with residual `8.1397034085718028e-04`.
- Fluid nonlinear tolerance `9.0e-4` reproduced a bad early branch with large
  residuals in steps `1` and `3`.
- Conclusion: raising nonlinear tolerance is not an acceptable primary fix.

### Minimum Outer Iteration Fix

- The step-63 stall showed linear convergence with zero outer iterations and
  no useful Newton update.
- `NS_min_outer_iterations=1` was added and applied to the FSILS BlockSchur
  coupled outer FGMRES path.
- A 70-step D18 MPI-4 probe with the checked-in XML controls reached
  `result_070.pvtu` and returned `success=1`.
- In that probe, step `63` converged with residual
  `2.1432747190868549e-04` and one linear outer iteration.
- The full D18 MPI-4 profile run reached `result_312.pvtu` and returned
  `success=1`.
- Conclusion: the minimum-outer control fixes a solver marching failure, but
  it does not fix the physical solution.

### Full D18 MPI-4 Result At The First Reference Time

Run directory:

```text
/tmp/svmp_d18_mpi4_minouter_xml_312step_lOpa1P
```

Run result:

- Exit status: `0`
- Last output: `result_312.pvtu`
- Final time: `0.156 s`
- `loop.run() returned success=1 steps_taken=312`

Comparison command:

```sh
python3 tests/cases/fluid/open_vessel_free_surface/compare_test05_profiles.py \
  /tmp/svmp_d18_mpi4_minouter_xml_312step_lOpa1P/result_312.pvtu \
  tests/cases/fluid/open_vessel_free_surface/reference_profiles/spheric_test05_wet_bed/d18_1.dat \
  --benchmark-json tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/benchmark.json \
  --front-diagnostic-only \
  --stale-pressure-gauge-tolerance 1.0 \
  --min-velocity-max 0.5 \
  --output Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_metrics.json \
  --plot-output Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_profile.png
```

Comparison result:

- Field validation failed because `velocity_max < 0.5 m/s`.
- Profile RMSE: `0.03365458346158355 m`.
- Peak-height error: `0.02200091339365315 m`.
- Wet volume: `0.0016175924638147652 m^3`.
- Gauge pressure: `643.659423052 Pa`, exactly the prescribed initial value.

## Pressure Constraint Probe

A temporary D18 MPI-4 probe removed the `<Node_pressure_constraints>` block and
ran to six outputs before it was stopped manually:

```text
/tmp/svmp_d18_mpi4_no_pressure_pin_80step_wKoXNc
```

Observed early behavior:

- Step `1`: `velocity_max = 0.049007853337125165 m/s`
- Step `3`: `velocity_max = 0.017214241561358076 m/s`
- Pressure offset became uncontrolled without a point pressure value.

Conclusion: the hard pressure node is suspicious because it is a constraint,
not a passive gauge, but removing it alone does not immediately restore the
expected collapse. A different pressure-nullspace treatment may still be
needed, but the core problem is not solved by simply deleting the pressure node.

## Current Leading Suspects

1. The fluid solve is settling into a near-hydrostatic balance too easily.
   The pressure field remains close to hydrostatic while velocity and kinetic
   energy remain too small.
2. The point pressure value is being used as a hard Dirichlet pressure
   constraint. That keeps the selected wet node fixed at the initial
   hydrostatic pressure throughout the transient.
3. The nonlinear convergence policy accepts large absolute nonlinear residuals
   when the relative criterion is satisfied. For example, early steps can be
   accepted with residuals much larger than the nominal physical tolerance.
4. Strict tolerances currently expose an FSILS true-residual floor rather than
   producing a usable corrected transient.
5. Level-set advection is coupled to the `Velocity` unknown, but the coupled
   velocity field is already too small and sparse. Therefore the interface
   transport equation has little physically meaningful motion to use.
6. Reinitialization and volume correction preserve wet volume, but they may be
   masking poor advection if the velocity field is not producing the expected
   collapse.
7. Active-domain cut-volume diagnostics show nonzero wet volume, but the
   resulting force imbalance may still be wrong if the pressure, gravity, and
   continuity terms are not balanced over the same active support.

## What Has Been Ruled Out

- The old full-volume hydrostatic initialization problem has been addressed at
  setup time: dry-side pressure is initialized to the reference pressure for
  active-domain cases.
- The step-63 MPI-4 profile-run stop was not the fundamental accuracy issue;
  after the minimum-outer fix, the run completes but remains physically wrong.
- Raising the nonlinear tolerance further is not acceptable; it eventually
  produces a bad early branch.
- The profile-front metric alone is not sufficient for validation because the
  wet bed already extends far downstream.
- Removing the point pressure constraint alone is not enough to immediately
  create the expected collapse in the early transient.

## Recommended Next Investigation

1. Add a short diagnostic that computes active-domain integrated force terms
   at the initial state: pressure-gradient contribution, gravity contribution,
   viscous contribution, continuity residual, and VMS contribution.
2. Verify that hydrostatic pressure and gravity do not cancel in the released
   column in a way that prevents horizontal acceleration.
3. Replace the hard point pressure value with a pressure nullspace treatment
   that does not pin a wet pressure node to its initial hydrostatic value
   throughout the transient.
4. Add validation checks that reject a profile run if gauge pressure remains
   exactly equal to its initial hydrostatic value after several steps.
5. Add validation checks that reject a profile run if wet-side kinetic energy
   remains below a documented lower bound at the first D18 reference time.
6. Run a reduced diagnostic case with reinitialization and volume correction
   disabled for a few steps to isolate pure level-set advection from correction
   effects.
7. Inspect cut-volume assembly for the body-force and pressure terms to ensure
   both are integrated over identical wet-side support and have the expected
   signs.
8. Revisit strict solver scaling only after the initial force balance produces
   physically meaningful velocity growth.

## Key Evidence Files

- `Documentation/qualification_logs/dam_break_remaining_investigations_20260513.md`
- `Documentation/qualification_logs/dam_break_accuracy_remediation_plan_20260513.md`
- `Documentation/qualification_logs/dam_break_d18_profile_solver_controls_20260513.md`
- `Documentation/qualification_logs/dam_break_d18_min_outer_iterations_20260513.md`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_metrics.json`
- `Documentation/qualification_logs/dam_break_d18_profile_comparison_20260513/d18_minouter312_profile.png`
