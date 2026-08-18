# Open Vessel Free Surface Remaining Test Case Issues

Date: 2026-05-26

This note tracks the remaining failures from the ordered `open_vessel_free_surface` run sweep and outlines hypotheses, checks, and remediation work needed to get the remaining cases running with one solution output per time step and a PVD time-series file.

Run artifacts:

- Summary: `Documentation/qualification_logs/open_vessel_free_surface_ordered_runs_20260526/RUN_SUMMARY.md`
- Logs and copied run cases: `Documentation/qualification_logs/open_vessel_free_surface_ordered_runs_20260526/`

## Current Baseline

The following cases passed and should remain the regression baseline while fixing the remaining cases:

| Case | Result | Why it matters |
| --- | --- | --- |
| `01_flat_hydrostatic` | 20 VTUs, `result.pvd`, verifier passed | Confirms simple unfitted hydrostatic active-domain setup works. |
| `02_linear_sloshing_2d` | 100 VTUs, `result.pvd`, verifier passed | Confirms smooth coupled level-set/free-surface motion works. |
| `04_square_tank_tilt_settling` | 1000 VTUs, `result.pvd`, verifier exited successfully | Confirms a larger 2D physical unfitted free-surface case can run long-term. |

The remaining failures are not a blanket failure of all unfitted free-surface machinery. They cluster around high-order MMS convergence, SPHERIC-style 3D hydrostatic/pressure setup, and fitted ALE nonlinear stability.

## Remaining Failing Cases

| Case | Failure point | Output status | Failure signature |
| --- | --- | --- | --- |
| `03_mms_record_after_ic_fix_13steps` | Step 0 | No VTUs, no PVD | Nonlinear solve did not converge after 20 Newton iterations. Final residual: `7.9874584309760421e-01`. Linear solve was converged. |
| `05_spheric_test10_lateral_water_1x_unfitted` | Step 0 | No VTUs, no PVD | FSILS true-residual failure, pressure dominated: `|Ax-b|=101.789`, relative residual `1.40968`, target `0.00722071`. Hydrostatic initialization reports gauge mismatch `229.116`. |
| `06_spheric_test10_lateral_water_1x_fitted_ale` | Step 12 | 12 VTUs, manually generated partial `result.pvd` | Nonlinear solve did not converge. Final residual: `8.9043242437600933e-01`. |
| `07_spheric_test05_wet_bed_d18` | Step 0 | No VTUs, no PVD | FSILS true-residual failure, pressure dominated: `|Ax-b|=0.00268439`, relative residual `1.42528e-04`, target `0.00188342`. |
| `08_spheric_test05_wet_bed_d38` | Step 0 | No VTUs, no PVD | FSILS true-residual failure, pressure dominated: `|Ax-b|=0.0021647`, relative residual `1.35551e-04`, target `0.00159696`. |

## Hypotheses

### H1: SPHERIC Test 10 Unfitted Has An Inconsistent Pressure Gauge

Evidence:

- `pressure_gauge.csv` contains `149,0.0`.
- The pressure gauge is wet: signed gap is `-0.0233975`.
- Hydrostatic initialization computes pressure `229.116` at that gauge node, while the gauge constrains it to `0.0`.
- The case then fails immediately with a pressure-dominated FSILS true-residual error.

Potential issue:

The gauge is not located at the hydrostatic reference pressure point/free surface, or the gauge value has not been updated for the current hydrostatic initialization convention. This injects a large pressure correction at step 0.

Checks:

- [ ] Locate node `149` in `05_spheric_test10_lateral_water_1x_unfitted/mesh/background/mesh-complete.mesh.vtu` and confirm coordinates.
- [ ] Compute expected hydrostatic pressure at node `149` from density, gravity, and `<Hydrostatic_pressure_reference_point>0.0 0.093 0.0</Hydrostatic_pressure_reference_point>`.
- [ ] Decide whether to move the gauge to a free-surface/reference-pressure node or set the CSV value to the computed hydrostatic pressure.
- [ ] Rerun a one-step probe after correcting only the gauge.
- [ ] Confirm hydrostatic init reports `gauge_pressure_max_abs_error` near zero.
- [ ] Confirm the first linear solve no longer fails with a large pressure-block true residual.

### H2: SPHERIC Test 05 D18/D38 Have Pressure Gauge Files But No Active Gauge Constraints

Evidence:

- `pressure_gauge.csv` exists:
  - D18: `1765,702.788852769`
  - D38: `1875,697.036707818`
- The solver logs report `gauge_constraints=0` for both D18 and D38 during hydrostatic initialization.
- Both cases fail before output with small but persistent pressure-dominated FSILS true-residual errors.

Potential issue:

The cases may have intended pressure gauges, but the solver XML does not currently apply `<Node_pressure_constraints>`. Without a usable pressure anchor, the pressure block may be near-nullspace dominated or poorly conditioned in the true-residual check.

Checks:

- [ ] Confirm whether D18/D38 intentionally omit `<Node_pressure_constraints>` or whether this is an accidental omission.
- [ ] Validate that node `1765` in D18 and node `1875` in D38 are wet and have active pressure support.
- [ ] Confirm the CSV pressure values match hydrostatic initialization at those nodes.
- [ ] Add `<Node_pressure_constraints>` to copied D18/D38 one-step probes using the existing CSV files.
- [ ] Rerun one-step probes and compare FSILS true-residual diagnostics.
- [ ] If gauges are invalid, choose new wet active-support gauge nodes and regenerate CSV values.

### H3: D18/D38 FSILS True-Residual Check May Be Slightly Stricter Than The Achieved Pressure Solve

Evidence:

- D18 true residual: `0.00268439`, target `0.00188342`.
- D38 true residual: `0.0021647`, target `0.00159696`.
- The misses are about 1.35 to 1.43 times the target, not orders of magnitude.
- The pressure block dominates the true residual.

Potential issue:

The GMRES/preconditioner path is close to the requested tolerance but the unscaled true-residual validation fails after PTC retries. This could be a real pressure conditioning/nullspace problem, a tolerance policy mismatch, or a missing pressure constraint.

Checks:

- [ ] First test H2 with explicit pressure gauges before changing tolerances.
- [ ] Record GMRES iteration counts and pressure block true residual with and without gauge constraints.
- [ ] Run diagnostic one-step probes with tighter linear tolerance and larger Krylov space.
- [ ] Run diagnostic one-step probes with direct linear solve if feasible for the case size.
- [ ] Check whether row-column scaling or pressure scaling changes the true residual but not the physical residual.
- [ ] Add a diagnostic that reports pressure mean/nullspace component before and after the solve.
- [ ] Avoid relaxing true-residual acceptance as a permanent fix until pressure anchoring and scaling are ruled out.

### H4: MMS High-Order Interface Case Is A Nonlinear Consistency Problem, Not A Linear Solver Failure

Evidence:

- `03_mms_record_after_ic_fix_13steps` uses direct solvers.
- Linear solve reports convergence.
- Residual decreases but stalls around `7.99e-01` after 20 Newton iterations.
- Case uses `<Generated_interface_geometry>HighOrderImplicit</Generated_interface_geometry>`.
- Logs show `geometry_tangent_warning=quadrature_sensitivities_omitted`.
- The cut context reports very small volume fractions, around `1e-08`, and cut-adjacent scale capped at `1000`.

Potential issue:

The MMS source/BC/initial condition may be inconsistent with the current OOP high-order implicit geometry path, or the frozen/refreshed quadrature tangent may be insufficient for Newton convergence when sliver cuts are present.

Checks:

- [ ] Verify MMS generated fields, wall velocity files, source terms, and solver time step all correspond to the same manufactured solution.
- [ ] Compare the current case against nearby MMS record directories that previously represented "after constraint", "after spacetime source", and "official after fixes" states.
- [ ] Run a one-step MMS probe with `LinearCorner` generated interface geometry to separate high-order geometry issues from MMS data issues.
- [ ] Run a one-step MMS probe with smaller `Time_step_size` to test transient consistency.
- [ ] Inspect residual component norms over Newton iterations and confirm which velocity component dominates.
- [ ] Check whether sliver cut volumes with volume fractions near `1e-08` correlate with the residual stall.
- [ ] Add or enable diagnostics for manufactured source term evaluation at quadrature points near the interface.
- [ ] Confirm pressure anchoring or pressure mean handling is appropriate for the MMS case.

### H5: Fitted ALE Lateral-Water Failure Suggests A Separate Moving-Mesh/Free-Surface Stability Issue

Evidence:

- The fitted ALE control advances through 12 accepted time steps and writes 12 VTUs.
- It fails at step 12 with nonlinear residual `8.9043242437600933e-01`.
- Final residual block is velocity dominated; pressure residual is small.
- Solution state near failure shows large velocity and pressure magnitudes compared with earlier steps.
- This case does not use unfitted cut-cell integration.

Potential issue:

The fitted ALE free-surface/moving-mesh formulation may become unstable under the current time step, mesh-motion tangent, or free-surface boundary treatment. This is likely separate from the unfitted cut-cell pressure residual failures.

Checks:

- [ ] Inspect `result_011.vtu` and `result_012.vtu` for mesh distortion, velocity spikes, and pressure spikes.
- [ ] Add mesh-quality diagnostics at each accepted ALE step, especially before step 12.
- [ ] Run with a smaller time step and the same final physical time window.
- [ ] Check whether the free-surface boundary pressure and traction are physically reasonable immediately before failure.
- [ ] Confirm `Moving_mesh_tangent_path=SymbolicRequired` is active for both fluid and mesh-motion equations.
- [ ] Compare failure timing with the unfitted lateral-water case after its gauge issue is fixed.
- [ ] If the instability remains at smaller time step, isolate mesh-motion solve from fluid solve with a prescribed/free-surface displacement probe.

### H6: PVD Generation Is Not Robust For Aborted Runs

Evidence:

- Successful cases write `result.pvd` at normal solver completion.
- Aborted cases do not write final PVD.
- The fitted ALE case produced 12 VTUs but required manual PVD generation.

Potential issue:

`Combine_time_series=true` currently depends on normal solver shutdown. For debugging unstable transient cases, partial time series should still be easy to inspect.

Checks:

- [ ] Decide whether the solver should flush/update PVD after each VTK write or only at completion.
- [ ] Add a reusable post-run collator for `result_*.vtu` files.
- [ ] Add this collator to failure triage scripts so partial outputs always get a PVD.
- [ ] Keep solver-written PVD behavior unchanged unless incremental PVD writes are safe for parallel output.

## Prioritized Work Plan

### P0: Pressure Gauge And Pressure Anchor Audit

- [ ] Write a small case-audit script that reports pressure gauge node coordinates, signed level-set value, active support status, and expected hydrostatic pressure.
- [ ] Run the audit on `spheric_test02_dambreak_obstacle`, `spheric_test10_lateral_water_1x_unfitted`, `spheric_test05_wet_bed_d18`, and `spheric_test05_wet_bed_d38`.
- [ ] Fix or regenerate pressure gauges where the gauge is dry, unsupported, or inconsistent with hydrostatic initialization.
- [ ] Add missing `<Node_pressure_constraints>` to D18/D38 if the existing CSV files are intended to be active.
- [ ] Re-run one-step probes before running full transients.

### P1: SPHERIC Unfitted Linear Solver Triage

- [ ] For each SPHERIC unfitted case, capture FSILS true residual by block after pressure-gauge fixes.
- [ ] Test tighter linear tolerances and larger Krylov dimensions in copied run directories.
- [ ] Test direct solve on the smallest failing variant if feasible.
- [ ] Determine whether failure is due to pressure nullspace, pressure scaling, cut-cell pressure stabilization, or a true residual validation threshold.
- [ ] Preserve any successful probe settings in a case-local README or benchmark note.

### P2: MMS High-Order Interface Convergence

- [ ] Re-run the MMS case with high-order interface diagnostics enabled.
- [ ] Compare against `linear_sloshing_2d`, which passes, to isolate MMS-specific source/BC behavior.
- [ ] Compare `HighOrderImplicit` and `LinearCorner` interface geometry on a one-step run.
- [ ] Check source and BC data generation for consistency with `dt=0.02` and the initial time used by the solver.
- [ ] Reduce or regularize extreme sliver cuts only after confirming MMS data consistency.

### P3: Fitted ALE Lateral-Water Stability

- [ ] Review the 12 partial VTUs using the generated `result.pvd`.
- [ ] Add mesh-quality and free-surface traction diagnostics around step 10 to step 12.
- [ ] Run smaller time-step probes to distinguish CFL/time-integration instability from formulation/tangent issues.
- [ ] Compare with the unfitted lateral-water case after its pressure gauge is corrected.

### P4: Output And Qualification Hygiene

- [ ] Keep `Save_results_to_VTK_format=true`, `Increment_in_saving_VTK_files=1`, and `Combine_time_series=true` in all qualification run copies.
- [ ] Add or standardize a partial-PVD collator for failed runs with VTU output.
- [ ] Record every probe in the qualification log directory with command, solver XML diff, VTU count, PVD status, and final failure signature.
- [ ] Promote only passing settings back into the checked-in test cases.

## Acceptance Criteria

A remaining case should be considered fixed only when all applicable checks pass:

- [ ] It completes the requested number of time steps.
- [ ] It writes one VTU per accepted time step.
- [ ] It writes or has a generated `result.pvd` that references every VTU in order.
- [ ] No pressure gauge mismatch is reported during hydrostatic initialization.
- [ ] No FSILS true-residual check fails.
- [ ] No nonlinear solve fails.
- [ ] Existing verifier scripts pass where available.
- [ ] For SPHERIC cases without verifier scripts, benchmark quantities are extracted and compared against the available reference profile or expected qualitative behavior.

## Suggested Next Probe Order

1. `spheric_test10_lateral_water_1x_unfitted`: fix the hydrostatic gauge mismatch first. This is the clearest configuration error.
2. `spheric_test05_wet_bed_d18`: test explicit pressure gauge constraints using the existing CSV.
3. `spheric_test05_wet_bed_d38`: repeat the same pressure-anchor test after D18.
4. `mms_record_after_ic_fix_13steps`: run `LinearCorner` versus `HighOrderImplicit` one-step probes.
5. `spheric_test10_lateral_water_1x_fitted_ale`: inspect partial outputs and test smaller time step.
