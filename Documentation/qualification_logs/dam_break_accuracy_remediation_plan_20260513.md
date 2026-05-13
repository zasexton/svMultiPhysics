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
- [ ] Use this XML spelling for the accepted D18/D38 path:

```xml
<Add_BC name="free_surface">
  <Type>UnfittedLevelSet</Type>
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
- [ ] Allow `Active_domain_method=SmoothedIndicator` only as a diagnostic or
      temporary fallback.
- [ ] Document that generated D18/D38 fixtures use negative level-set values as
      the water side.

## Exact Cut-Volume Integration Checklist

- [x] Extend generated level-set interface processing to classify every owned
      cell as full-negative, full-positive, or cut.
- [ ] For full-negative and full-positive cells, preserve the selected side as
      normal cell-volume quadrature for the active-domain measure.
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
- [ ] Route cut-volume kernels through `CutDomainAssembler`.
- [ ] Ensure cut-volume forms do not also assemble duplicate full-cell `.dx()`
      kernels for the same residual terms.
- [ ] Add interpreter and compiled/JIT support for the new integral domain.
- [x] Add tests proving constant-field cut-volume integrals match expected
      negative and positive volume fractions.

## Navier-Stokes Assembly Checklist

- [ ] Factor the existing Navier-Stokes cell integrands before attaching a
      measure.
- [ ] When active domain is disabled, preserve the current full-cell `.dx()`
      assembly path.
- [ ] When `Active_domain=LevelSetNegative`, assemble the water-side residual
      over `dCutVolume(free_surface_marker, Negative)`.
- [ ] Apply the active-domain measure to inertia terms.
- [ ] Apply the active-domain measure to convection terms.
- [ ] Apply the active-domain measure to viscous terms.
- [ ] Apply the active-domain measure to pressure-gradient terms.
- [ ] Apply the active-domain measure to body-force terms.
- [ ] Apply the active-domain measure to continuity terms.
- [ ] Apply the active-domain measure to all VMS residual and stabilization
      terms.
- [ ] Keep pressure constraints/gauges compatible with the active wet region.
- [ ] Keep zero-surface-tension, zero-external-pressure cases from adding a
      false interface traction.
- [ ] Keep D18 kinematic enforcement disabled for the first corrected path.
- [ ] Add diagnostics showing which active-domain side and method were used.
- [ ] Add diagnostics showing total wet volume and cut-cell volume contribution.

## Material-Weighting Fallback Checklist

- [ ] Implement `Active_domain_method=SmoothedIndicator` only after the public
      active-domain option exists.
- [ ] Multiply the same Navier-Stokes and VMS volume integrands by a smoothed
      wet indicator when the fallback is selected.
- [ ] Use the level-set field and isovalue already configured for the
      free-surface boundary.
- [ ] Make the transition width explicit in configuration or derive it from
      local mesh size with a documented default.
- [ ] Mark this method as diagnostic, approximate, and not sufficient for final
      D18/D38 qualification.
- [ ] Add regression coverage showing the fallback changes the full-tank
      hydrostatic behavior.
- [ ] Do not use the fallback as the accepted benchmark path unless the exact
      cut-volume path is explicitly deferred in a later decision.

## Hydrostatic Initialization And Gauge Checklist

- [ ] Update hydrostatic pressure initialization for active-domain cases.
- [ ] Initialize wet-side pressure consistently with the water column.
- [ ] Set dry-side pressure to the reference pressure or otherwise exclude it
      from the active pressure initialization.
- [ ] Prevent dry-side full-tank hydrostatic pressure from influencing the
      initial active-domain solve.
- [ ] Add a gauge validation check for active-domain cases.
- [ ] Reject or warn on pressure gauges placed on the dry side.
- [ ] Reject or warn on pressure gauges with `abs(phi)` below a small
      near-interface margin.
- [ ] Replace current D18 gauge node `279` with a robust wet-region node in the
      retained water column.
- [ ] Record the selected gauge id, coordinates, initial `phi`, and expected
      initial hydrostatic pressure in benchmark metadata.
- [ ] Verify the initial pressure-gauge offset no longer matches the previous
      full-volume hydrostatic error range.

## Solver And Time-Integration Checklist

- [ ] Keep the current time-step setup unchanged until active-domain assembly is
      verified.
- [ ] After active-domain assembly passes smoke tests, tighten D18 nonlinear
      relative tolerance to `1e-6`.
- [ ] Tighten D18 linear relative tolerance to `1e-6`.
- [ ] Use an absolute linear tolerance no looser than `1e-10` for the
      qualification run, unless solver logs show a documented scaling issue.
- [ ] Record nonlinear iteration counts, linear iteration counts, and residual
      norms for each D18/D38 qualification attempt.
- [ ] If active-domain cut cells introduce solver instability, tune
      stabilization only after confirming volume integration and initialization
      are correct.

## Validation Script Checklist

- [ ] Update the profile comparison script so wet-bed front position is
      diagnostic-only for D18.
- [ ] Report peak height error separately from front-position error.
- [ ] Report wet volume at each compared output.
- [ ] Report wet-volume drift from the initial condition.
- [ ] Report pressure min, max, mean, and hydrostatic-reference error.
- [ ] Report velocity max, mean, and wet-side mean.
- [ ] Report kinetic energy growth from the initial condition.
- [ ] Report whether largest velocities occur in the retained/released water
      region rather than the dry or top exterior region.
- [ ] Add a validation failure if pressure remains close to the old full-volume
      hydrostatic state.
- [ ] Add a validation failure if velocity remains near-static at the D18
      comparison time.
- [ ] Keep the extracted `phi=0` front/profile plot, but label it as an
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
- [ ] Add Forms tests for `IntegralDomain::CutVolume` residual assembly.
- [ ] Add Forms tests for cut-volume tangent/Jacobian assembly.
- [ ] Add Navier-Stokes installation tests proving active-domain cases install
      cut-volume kernels.
- [ ] Add Navier-Stokes installation tests proving active-domain cases do not
      install duplicate full `.dx()` kernels for the same terms.
- [ ] Add hydrostatic initialization tests for wet-side-only pressure.
- [ ] Add gauge validation tests for dry and near-interface nodes.

## Integration And Qualification Checklist

- [ ] Run a one-step D18 serial smoke test after active-domain assembly is
      implemented.
- [ ] Confirm the log reports `Active_domain=LevelSetNegative` and
      `Active_domain_method=CutVolume`.
- [ ] Confirm wet volume is nonzero and physically consistent with the initial
      D18 level set.
- [ ] Confirm pressure is no longer initialized as full-tank hydrostatic in the
      dry region.
- [ ] Run the same one-step D18 test with 2 MPI ranks.
- [ ] Run the same one-step D18 test with 4 MPI ranks.
- [ ] Compare serial, MPI-2, and MPI-4 wet volume, pressure range, and gauge
      pressure.
- [ ] Run D18 to the current comparison time after one-step checks pass.
- [ ] Confirm velocity and kinetic energy show dam-break collapse dynamics.
- [ ] Confirm pressure no longer matches the previous full-volume hydrostatic
      error pattern.
- [ ] Compare corrected D18 profile against digitized SPHERIC Test05 D18 data.
- [ ] Only after D18 passes, repeat the same workflow for D38.
- [ ] Save solver logs, validation metrics, plots, and command lines in a new
      qualification log directory.

## Acceptance Checklist

- [ ] D18 no longer passes or fails based only on the wet-bed front metric.
- [ ] D18 active-domain assembly uses wet-side cut-volume rules for all
      Navier-Stokes and VMS volume terms.
- [ ] D18 pressure initialization and pressure gauge are both wet-side
      consistent.
- [ ] D18 velocity field shows physically meaningful collapse dynamics at the
      comparison time.
- [ ] D18 pressure field no longer remains close to full-volume hydrostatic.
- [ ] Serial, MPI-2, and MPI-4 D18 results agree within documented tolerances.
- [ ] D38 is evaluated only after the D18 active-domain path passes.
- [ ] Existing non-active-domain free-surface cases keep their current behavior.
- [ ] The final qualification log names the exact code revision, XML inputs,
      solver settings, output files, and validation script version.

## Assumptions

- [ ] Negative level-set values represent water in the generated D18/D38
      fixtures.
- [ ] Exact cut-volume integration is the required acceptance path.
- [ ] Smoothed material weighting is a fallback diagnostic, not final benchmark
      qualification.
- [ ] Open-atmosphere D18 should not add artificial interface traction when
      external pressure and surface tension are zero.
- [ ] D18 kinematic enforcement remains off until wet-side Navier-Stokes
      assembly, initialization, and validation have been corrected.
