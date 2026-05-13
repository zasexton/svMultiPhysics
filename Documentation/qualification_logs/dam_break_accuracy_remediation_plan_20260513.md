# Dam-Break Accuracy Remediation Checklist - 2026-05-13

## Purpose

This checklist turns the dam-break investigation findings into an implementation
and validation plan. The core issue is that the unfitted level-set free-surface
case advances the interface, but the Navier-Stokes volume residual is still
assembled over the full computational tank. D18/D38 qualification should remain
blocked until Navier-Stokes momentum, pressure, continuity, and stabilization are
restricted to the wet active domain or an explicitly accepted equivalent.

## Root-Cause Summary

- [x] Treat the current D18 result as an invalid benchmark solution, not as a
      near miss.
- [x] Record that the `phi=0` contour movement is not enough for validation
      because the initial wet bed can make the front location look plausible.
- [x] Record that pressure currently remains close to full-volume hydrostatic
      initialization.
- [x] Record that velocity remains too small for the expected dam-break
      collapse.
- [x] Record that the missing implementation piece is wet-side volume
      integration for Navier-Stokes and VMS terms.
- [x] Keep the existing investigation note as the evidence log:
      `Documentation/qualification_logs/dam_break_remaining_investigations_20260513.md`.

## Public Configuration Checklist

- [x] Add an explicit active-domain option for unfitted free-surface
      Navier-Stokes.
- [x] Add a code enum equivalent to:
      `FreeSurfaceActiveDomain { None, LevelSetNegative, LevelSetPositive }`.
- [x] Parse the option under the existing free-surface boundary condition block.
- [x] Use this XML spelling for the accepted D18/D38 path:

```xml
<Add_BC name="free_surface">
  <Type>Free_surface</Type>
  <Implementation>UnfittedLevelSet</Implementation>
  <Active_domain>LevelSetNegative</Active_domain>
  <Active_domain_method>CutVolume</Active_domain_method>
</Add_BC>
```

- [x] Keep `Active_domain=None` as the default for backward compatibility.
- [x] Reject `Active_domain` values other than `None`, `LevelSetNegative`, and
      `LevelSetPositive`.
- [x] Reject active-domain configuration on non-`UnfittedLevelSet`
      free-surface boundaries.
- [x] Default `Active_domain_method` to `CutVolume`.
- [x] Allow `Active_domain_method=SmoothedIndicator` only as a diagnostic or
      temporary fallback.
- [x] Document that generated D18/D38 fixtures use negative level-set values as
      the water side.

## Exact Cut-Volume Integration Checklist

- [x] Extend generated level-set interface processing to classify every owned
      cell as full-negative, full-positive, or cut.
- [x] For full-negative and full-positive cells, preserve the selected side as
      normal cell-volume quadrature for the active-domain measure.
- [x] Allow generated level-set domains with full-side volume regions but no
      rank-local interface fragments.
- [x] For cut cells, generate side-specific volume quadrature rules for both
      negative and positive level-set regions.
- [x] Carry parent cell id, interface marker, integration side, and volume
      fraction on each generated volume rule.
- [x] Extend `CutIntegrationContext` so generated volume rules are indexed by
      interface marker and side.
- [x] Keep generated interface surface rules separate from generated volume
      rules.
- [x] Add a Forms integral domain for cut-volume integration, for example
      `IntegralDomain::CutVolume`.
- [x] Add a public Forms measure for the new domain, for example
      `dCutVolume(marker, side)`.
- [x] Update Forms vocabulary/documentation to include the new cut-volume
      measure.
- [x] Route cut-volume kernels through `CutDomainAssembler`.
- [x] Route cut-volume kernels through `ParallelAssembler` for MPI
      active-domain assembly.
- [x] Execute registered cut-volume operator terms from `FESystem::assemble(...)`
      using a registered `CutIntegrationContext`.
- [x] Ensure cut-volume forms do not also assemble duplicate full-cell `.dx()`
      kernels for the same residual terms.
- [x] Ensure JIT-enabled cut-volume Forms kernels use marker/side-aware
      dispatch for cut-volume marker and side context.
- [x] Add interpreter and compiled/JIT support for the new integral domain.
- [x] Add tests proving constant-field cut-volume integrals match expected
      negative and positive volume fractions.

## Navier-Stokes Assembly Checklist

- [x] Factor the existing Navier-Stokes cell integrands before attaching a
      measure.
- [x] When active domain is disabled, preserve the current full-cell `.dx()`
      assembly path.
- [x] When `Active_domain=LevelSetNegative`, assemble the water-side residual
      over `dCutVolume(free_surface_marker, Negative)`.
- [x] Apply the active-domain measure to inertia terms.
- [x] Apply the active-domain measure to convection terms.
- [x] Apply the active-domain measure to viscous terms.
- [x] Apply the active-domain measure to pressure-gradient terms.
- [x] Apply the active-domain measure to body-force terms.
- [x] Apply the active-domain measure to continuity terms.
- [x] Apply the active-domain measure to all VMS residual and stabilization
      terms.
- [x] Keep pressure constraints/gauges compatible with the active wet region.
- [x] Keep zero-surface-tension, zero-external-pressure cases from adding a
      false interface traction.
- [x] Keep D18 kinematic enforcement disabled for the first corrected path.
- [x] Add diagnostics showing which active-domain side and method were used.
- [x] Add diagnostics showing total wet volume and cut-cell volume contribution.

## Material-Weighting Fallback Checklist

- [x] Implement `Active_domain_method=SmoothedIndicator` only after the public
      active-domain option exists.
- [x] Multiply the same Navier-Stokes and VMS volume integrands by a smoothed
      wet indicator when the fallback is selected.
- [x] Use the level-set field and isovalue already configured for the
      free-surface boundary.
- [x] Make the transition width explicit in configuration or derive it from
      local mesh size with a documented default.
- [x] Mark this method as diagnostic, approximate, and not sufficient for final
      D18/D38 qualification.
- [x] Add regression coverage showing the fallback changes the full-tank
      hydrostatic behavior.
- [x] Do not use the fallback as the accepted benchmark path unless the exact
      cut-volume path is explicitly deferred in a later decision.

## Hydrostatic Initialization And Gauge Checklist

- [x] Update hydrostatic pressure initialization for active-domain cases.
- [x] Initialize wet-side pressure consistently with the water column.
- [x] Set dry-side pressure to the reference pressure or otherwise exclude it
      from the active pressure initialization.
- [x] Prevent dry-side full-tank hydrostatic pressure from influencing the
      initial active-domain solve.
- [x] Add a gauge validation check for active-domain cases.
- [x] Reject or warn on pressure gauges placed on the dry side.
- [x] Reject or warn on pressure gauges with `abs(phi)` below a small
      near-interface margin.
- [x] Replace current D18 gauge node `279` with a robust wet-region node in the
      retained water column.
- [x] Record the selected gauge id, coordinates, initial `phi`, and expected
      initial hydrostatic pressure in benchmark metadata.
- [x] Verify the initial pressure-gauge offset no longer matches the previous
      full-volume hydrostatic error range.

## Solver And Time-Integration Checklist

- [x] Keep the current time-step setup unchanged until active-domain assembly is
      verified.
- [ ] After active-domain assembly passes smoke tests, tighten D18 nonlinear
      relative tolerance to `1e-6`.
- [ ] Tighten D18 linear relative tolerance to `1e-6`.
- [ ] Use an absolute linear tolerance no looser than `1e-10` for the
      qualification run, unless solver logs show a documented scaling issue.
- [x] Record the D18 strict-tolerance solver floor in
      `Documentation/qualification_logs/dam_break_d18_solver_tolerance_runs_20260513.md`.
- [ ] Add a robust solver path for the coupled `phi`, `Velocity`, and
      `Pressure` D18 layout, or restructure D18 so the Navier-Stokes solve can
      use a velocity-pressure solver block without the level-set field in the
      same linear system.
- [x] Fix the active-domain `Use_cut_metadata_scale=true` assembly crash before
      using metadata-scaled cut-cell stabilization for D18.
- [ ] Implement per-face metadata-scale constants for cut-adjacent facet
      stabilization before enabling `Use_cut_metadata_scale=true` in D18/D38
      fixtures.
- [ ] Re-run D18 strict tolerance after the solver path is corrected.
- [x] Record nonlinear iteration counts, linear iteration counts, and residual
      norms for each D18/D38 qualification attempt.
- [ ] If active-domain cut cells introduce solver instability, tune
      stabilization only after confirming volume integration and initialization
      are correct.

## Validation Script Checklist

- [x] Update the profile comparison script so wet-bed front position is
      diagnostic-only for D18.
- [x] Report peak height error separately from front-position error.
- [x] Report wet volume at each compared output.
- [x] Report wet-volume drift from the initial condition.
- [x] Report pressure min, max, mean, and hydrostatic-reference error.
- [x] Report velocity max, mean, and wet-side mean.
- [x] Report kinetic energy growth from the initial condition.
- [x] Report whether largest velocities occur in the retained/released water
      region rather than the dry or top exterior region.
- [x] Add a validation failure if pressure remains close to the old full-volume
      hydrostatic state.
- [x] Add a validation failure if velocity remains near-static at the D18
      comparison time.
- [x] Keep the extracted `phi=0` front/profile plot, but label it as an
      interface diagnostic until field checks pass.

## Unit Test Checklist

- [x] Add parser tests for valid active-domain values.
- [x] Add parser tests for invalid active-domain values.
- [x] Add parser tests proving defaults preserve existing cases.
- [x] Add configuration tests rejecting active-domain use on non-unfitted
      free-surface BCs.
- [x] Add cut-volume classification tests for full-negative cells.
- [x] Add cut-volume classification tests for full-positive cells.
- [x] Add cut-volume classification tests for partially cut cells.
- [x] Add constant-integral tests for negative-side volume rules.
- [x] Add constant-integral tests for positive-side volume rules.
- [x] Preserve generated cut-volume regions when building level-set
      interface lifecycle domains.
- [x] Add Forms tests for `IntegralDomain::CutVolume` residual assembly.
- [x] Add Forms tests for cut-volume tangent/Jacobian assembly.
- [x] Add Forms tests covering JIT-enabled cut-volume wrapper fallback.
- [x] Add system-level Forms tests proving cut-volume residual and tangent
      insertion through `FESystem::assemble(...)`.
- [x] Add Forms tests proving full-side cut-volume rules reuse normal
      cell-volume quadrature.
- [x] Add Navier-Stokes installation tests proving active-domain cases install
      cut-volume kernels.
- [x] Add Navier-Stokes installation tests proving active-domain cases do not
      install duplicate full `.dx()` kernels for the same terms.
- [x] Register generated level-set cut-integration context before
      active-domain transient assembly.
- [x] Preserve cut-volume measures when rebuilding symbolic tangent forms.
- [x] Keep mixed-field zero tangent probes shape-compatible for cut-volume
      assembly.
- [x] Add hydrostatic initialization tests for wet-side-only pressure.
- [x] Add gauge validation tests for dry and near-interface nodes.
- [x] Add XML ingestion coverage proving active-domain free-surface boundary
      controls reach the Navier-Stokes translator.
- [x] Add Navier-Stokes tests proving cut-cell stabilization defaults to
      constant scale and rejects explicit metadata scaling on cut-adjacent
      facets.

## Integration And Qualification Checklist

- [x] Run a one-step D18 serial smoke test after active-domain assembly is
      implemented.
- [x] Confirm the log reports `Active_domain=LevelSetNegative` and
      `Active_domain_method=CutVolume`.
- [x] Confirm wet volume is nonzero and physically consistent with the initial
      D18 level set.
- [x] Confirm pressure is no longer initialized as full-tank hydrostatic in the
      dry region.
- [x] Confirm a D18 `Use_cut_metadata_scale=true` smoke case stops with setup
      validation instead of an assembly crash.
- [x] Run the same one-step D18 test with 2 MPI ranks.
- [x] Run the same one-step D18 test with 4 MPI ranks.
- [x] Compare serial, MPI-2, and MPI-4 wet volume, pressure range, and gauge
      pressure.
- [x] Run D18 to the current comparison time after one-step checks pass.
- [x] Confirm velocity and kinetic energy show dam-break collapse dynamics.
- [x] Confirm pressure no longer matches the previous full-volume hydrostatic
      error pattern.
- [ ] Compare corrected D18 profile against digitized SPHERIC Test05 D18 data.
      Not completed in the MPI-4 full-run evidence pass because no digitized
      D18 profile data file was found in the repository.
- [ ] Only after D18 passes, repeat the same workflow for D38.
- [ ] Save solver logs, validation metrics, plots, and command lines in a new
      qualification log directory.

## Acceptance Checklist

- [x] D18 no longer passes or fails based only on the wet-bed front metric.
- [x] D18 active-domain assembly uses wet-side cut-volume rules for all
      Navier-Stokes and VMS volume terms.
- [x] D18 pressure initialization and pressure gauge are both wet-side
      consistent.
- [x] D18 velocity field shows physically meaningful collapse dynamics at the
      comparison time.
- [x] D18 pressure field no longer remains close to full-volume hydrostatic.
- [ ] Serial, MPI-2, and MPI-4 D18 results agree within documented tolerances.
- [ ] D38 is evaluated only after the D18 active-domain path passes.
- [ ] Existing non-active-domain free-surface cases keep their current behavior.
- [ ] The final qualification log names the exact code revision, XML inputs,
      solver settings, output files, and validation script version.

## Assumptions

- [x] Negative level-set values represent water in the generated D18/D38
      fixtures.
- [x] Exact cut-volume integration is the required acceptance path.
- [x] Smoothed material weighting is a fallback diagnostic, not final benchmark
      qualification.
- [x] Open-atmosphere D18 should not add artificial interface traction when
      external pressure and surface tension are zero.
- [x] D18 kinematic enforcement remains off until wet-side Navier-Stokes
      assembly, initialization, and validation have been corrected.
