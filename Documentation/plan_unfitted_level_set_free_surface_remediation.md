# Unfitted Level-Set Free-Surface Remediation Checklist

## Purpose

This checklist turns the OOP unfitted level-set free-surface review findings into
concrete implementation and validation work. The immediate failure mode is that
pressure, velocity, and the reported active wet region can be inconsistent with
the moving level set. The remediation goal is to make the level-set interface,
active-domain integration, cut-cell stabilization, and diagnostics match the
current nonlinear state and established unfitted finite-element practice.

## Scope

The checklist covers:

- OOP level-set/free-surface wiring for unfitted meshes.
- Active wet-domain restriction for one-fluid free-surface Navier-Stokes cases.
- Generated level-set interface and cut-volume quadrature.
- Cut-cell/ghost-penalty stabilization locality.
- Level-set maintenance, wet-region diagnostics, and acceptance tests.

The checklist does not replace full two-phase material modeling. It targets the
current one-fluid free-surface formulation with a passive exterior region.

## 1. Make Cut Geometry Consistent With The Current Level Set

Problem: the active cut context is currently rebuilt before the nonlinear solve
from the previous accepted solution, then reused while Newton and line-search
trials update `phi`.

Selected contract: the accepted OOP unfitted free-surface path is
`monolithic-consistent` for generated cut geometry. Residual and Jacobian
assembly synchronize the active cut context from the exact Newton or
line-search trial state being assembled. Rejected line-search trials are
followed by an accepted/restored-state synchronization so trial cut geometry is
not left installed for subsequent assembly or output.

Cache-invalidation contract: generated cut topology, generated volume and
interface quadrature, and generated cut-adjacent facet sets participate in the
cut integration revision snapshot. A topology, quadrature-policy, active-side,
embedded-field value, or conditioning change invalidates cut classification,
cut quadrature, matrix and matrix-free assembly data, preconditioner metadata,
restart metadata, and stabilization hooks. Pure FE layout changes invalidate
matrix, preconditioner, and restart metadata without rebuilding cut
classification or cut quadrature. State-independent full-cell `.dx()` caches
remain reusable when only cut metadata changes.

### Design Checklist

- [x] Decide and document the intended coupling contract:
      `monolithic-consistent`, `outer-fixed-point`, or `explicit-geometry`.
- [x] For the accepted OOP free-surface path, require a geometry update point
      before every residual assembly that can observe a changed level-set field.
- [x] Define whether cut topology changes during Newton invalidate only
      cut-domain kernels or the whole assembler operator cache.
- [x] Define a stable revision key that changes when level-set coefficients,
      generated interface topology, cut-volume quadrature, or cut-adjacent
      facets change.
- [x] Define behavior for failed line-search trials so rejected trial geometry
      is not left installed in `FESystem`.

### Implementation Checklist

- [x] Add a cut-context rebuild callback to the nonlinear assembly path, not only
      the time-loop `on_before_physics_solve` callback.
- [x] Pass the current Newton state or current solution vector into the callback
      before residual assembly.
- [x] Pass the current Newton state or current solution vector into the callback
      before Jacobian assembly.
- [x] Invoke the callback for line-search trial residuals.
- [x] Restore or rebuild the context for the accepted nonlinear state after a
      line-search trial exits.
- [x] Rebuild the active cut context after accepted-step level-set maintenance.
- [x] Rebuild or clear the active cut context before output so diagnostics see
      the maintained accepted `phi`.
- [x] Move `refreshActiveCutIntegrationContext(...)` or its core logic behind a
      reusable service that accepts a state view instead of being tied to
      `ApplicationDriver` time-loop wiring.
- [x] Ensure the rebuild path can gather field coefficients in FE ordering
      without unnecessary full-vector copies on every assembly if the state did
      not change.
- [x] Add logging that reports the cut-context revision, negative volume,
      positive volume, active volume-region count, and interface-fragment count
      at each accepted step.
- [x] Add debug logging for Newton iterations that reports when cut topology
      changes during the nonlinear solve.

### Cache And Invalidation Checklist

Cache dependency inventory: cut-volume rules feed cut-domain quadrature lookup,
cell kernel dispatch, matrix-free data, matrix assembly metadata,
preconditioner metadata, and restart metadata. Interface `dI(...)` rules feed
interface quadrature lookup, interface kernel dispatch, matrix-free data, and
the same matrix/preconditioner/restart metadata. Cut-adjacent facet sets feed
interior-facet traversal, stabilization hooks, matrix-free data, and
preconditioner metadata.

- [x] Identify all assembler caches that depend on cut-volume rules.
- [x] Identify all assembler caches that depend on interface `dI(...)` rules.
- [x] Identify all assembler caches that depend on cut-adjacent facet sets.
- [x] Invalidate cut-dependent kernel dispatch tables when marker/side rules
      change.
- [x] Invalidate cut-dependent sparsity assumptions only if topology changes can
      add or remove test/trial couplings.
- [x] Preserve state-independent full-cell `.dx()` caches when only cut metadata
      changed.
- [x] Add an assertion that `dCutVolume(...)` assembly cannot run with a stale
      cut-context revision for the current nonlinear state.

### Tests

- [x] Add a unit test where a level-set field changes sign in one element during
      Newton and the next residual assembly sees the new active side.
- [x] Add a line-search test where a rejected trial changes the cut topology and
      the accepted state restores the previous topology.
- [x] Add a transient test proving accepted-step reinitialization or volume
      correction is reflected in output-time active volume.
- [x] Add a regression test that fails if `on_before_physics_solve` is the only
      cut-context rebuild point for a coupled level-set/fluid solve.

## 2. Require Or Correctly Default The Active Wet Domain

Problem: some XML fixtures declare `Implementation=UnfittedLevelSet` but omit
`Active_domain`, so Navier-Stokes volume terms assemble over the full background
mesh.

### Configuration Checklist

- [x] Add explicit validation for `Free_surface` plus
      `Implementation=UnfittedLevelSet`.
- [x] Reject missing `Active_domain` unless the input also sets a deliberate
      full-domain opt-in such as `Allow_full_domain_unfitted_free_surface=true`.
- [x] Reject `Active_domain=None` for validation fixtures that claim to be
      one-fluid free-surface unfitted cases.
- [x] Keep `Active_domain_method=CutVolume` as the accepted production path.
- [x] Keep `Active_domain_method=SmoothedIndicator` available only as a
      diagnostic or explicitly approximate fallback.
- [x] Emit a clear diagnostic showing the level-set field, isovalue, active side,
      active-domain method, and generated interface domain id.
- [x] Validate that the configured active side contains at least one wet cell at
      initialization.
- [x] Validate that the configured active side is consistent with benchmark
      documentation, for example negative `phi` is water for D18/D38.

### Fixture Checklist

- [x] Update `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/solver.xml`
      to either set the wet active domain or explicitly mark itself as a
      full-domain diagnostic case.
- [x] Update
      `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test10_lateral_water_1x/solver.xml`
      to set the intended wet active domain.
- [x] Update
      `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test02_dambreak_obstacle/solver.xml`
      to set the intended wet active domain.
- [x] Verify
      `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d18/solver.xml`
      keeps `Active_domain=LevelSetNegative` and
      `Active_domain_method=CutVolume`.
- [x] Verify
      `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/spheric_test05_wet_bed_d38/solver.xml`
      keeps `Active_domain=LevelSetNegative` and
      `Active_domain_method=CutVolume`.
- [x] Update moving-free-surface validation documentation to state that
      unfitted one-fluid cases require an explicit wet active domain.

### Tests

- [x] Add an input-validation test for missing `Active_domain`.
- [x] Add an input-validation test for `Active_domain=None` without an explicit
      full-domain opt-in.
- [x] Add an input-validation test for active-domain options on fitted ALE free
      surfaces.
- [x] Add a smoke test proving a zero-stress unfitted free surface still changes
      Navier-Stokes assembly through `dCutVolume(...)`.
- [x] Add a smoke test proving omitted active-domain configuration does not
      silently produce a full-tank hydrostatic result in validation fixtures.

## 3. Replace Constant Cut-Volume Quadrature

Problem: generated cut-volume rules currently provide one centroid point and
declare exactness only for constants, but the rules are used for full
Navier-Stokes and VMS integrands.

Current production path: generated level-set cut volumes use side-specific
centroid moment fitting for order 0 and order 1 rules. The generic
`makeMomentFittedCutVolumeQuadrature(...)` utility remains available, but the
builder stores the computed side centroid directly so topology metadata and
side fractions stay deterministic.

Degenerate-cut fallback policy: exact no-cut and full-zero cells remain inactive
diagnostics, exact vertex and edge touches are tagged as degenerate, fragments
below the measure tolerance are rejected, and near-tangent cuts with positive
side volume keep active side-volume metadata.

### Quadrature Policy Checklist

- [x] Define a default cut-volume quadrature order as a function of FE order and
      form order.
      The default helper uses the maximum of geometry order, field order, and
      form order as the requested cut-volume quadrature order.
- [x] Define a minimum accepted order for linear velocity-pressure elements.
      Linear geometry, field, and form input requests order 1.
- [x] Define a minimum accepted order for quadratic geometry or quadratic fields.
      Any quadratic geometry, field, or form input requests at least order 2;
      generated rules still report the lower implemented exact order until
      higher-order cut-volume construction is added.
- [x] Decide whether the first production path uses sub-triangulation,
      moment-fitted quadrature, or a hybrid of both.
- [x] Audit the existing `makeMomentFittedCutVolumeQuadrature(...)` utility and
      decide whether it can be promoted to the generated level-set interface
      path.
- [x] Define fallback behavior for degenerate cuts, near-zero volume fractions,
      and interface cuts passing close to vertices.
- [x] Preserve exact total measure for constants even when higher-order
      quadrature is requested.
      Higher requested orders currently reuse the linear moment-fitted volume
      rule and report the implemented exact polynomial order while keeping
      conservative side measures and weights.

### Implementation Checklist

- [x] Replace `constant-level-set-volume` with a policy that can generate more
      than one point per cut side.
      Cut side-volume regions now carry explicit quadrature points, and 2D cut
      side polygons are triangulated so cut quadrilateral side volumes can export
      multiple conservative points.
- [x] Generate negative-side and positive-side quadrature rules with consistent
      parent cell ids, side labels, volume fractions, and provenance.
- [x] For full negative and full positive cells, reuse normal element quadrature
      or mark the generated rule as full-cell-equivalent.
- [x] For cut triangles, generate side-specific sub-triangle or moment-fitted
      rules.
- [x] For cut quads, generate side-specific rules without assuming the cut
      polygon is triangular.
- [x] For cut tetrahedra, generate side-specific sub-tetrahedra or moment-fitted
      rules.
- [x] Decide whether hexahedra, wedges, and pyramids are unsupported with a hard
      diagnostic or supported through the existing extension registry.
      The built-in level-set cutter rejects them with a diagnostic unless an
      extension-registry cutter is explicitly registered for that element type.
- [x] Store the quadrature policy name and exact polynomial order in cut metadata.
- [x] Make `LevelSetGeneratedInterfaceOptions::quadrature_order` meaningful for
      cut-volume rules.
- [x] Ensure `dI(...)` surface rules and `dCutVolume(...)` volume rules can use
      different quadrature orders.
- [x] Ensure MPI-owned and ghost-cell cut-volume metadata remain deterministic.
      Cut-volume rule export is sorted by parent cell, side, marker, and stable
      topology revision, and stable ids do not depend on ownership revision.

### Tests

- [x] Add exact constant integration tests for all supported element types.
- [x] Add exact linear integration tests for cut triangles.
- [x] Add exact linear integration tests for cut quads.
- [x] Add exact linear integration tests for cut tetrahedra.
- [x] Add polynomial convergence tests for quadratic fields where exactness is
      not guaranteed.
      A refinement test integrates a quadratic field over a triangular cut-volume
      region and checks that the generated cut-volume quadrature error decreases.
- [x] Add tests where the interface cuts very close to a vertex.
- [x] Add tests where the interface cuts very close to an edge.
- [x] Add tests for small wet volume fractions.
- [x] Add tests for full-negative and full-positive cells.
- [x] Add tests proving negative and positive measures sum to the parent measure.
- [x] Add a Navier-Stokes assembly test proving a nonconstant pressure or
      velocity field changes when constant-only quadrature is replaced.

## 4. Localize Cut-Cell Stabilization

Problem: the current cut-adjacent stabilization helper lowers to plain `dS()`.
When cut metadata scaling is disabled, this can apply ghost-penalty terms to all
interior facets instead of only the cut neighborhood.

Selected locality rule: generated interfaces use a marker-backed
cut-adjacent interior facet set instead of plain `dS()`. The generated set
contains every interior facet with at least one adjacent cut cell, including
facets shared by two cut cells, and excludes facets whose adjacent cells are
both uncut. Small-fragment conditioning uses the cut cell plus its immediate
cell-neighbor patch as the extension neighborhood; wider patches require an
explicit stabilization policy change.

### Form And Measure Checklist

- [x] Define a real cut-adjacent facet measure or marker-backed facet set for
      generated interfaces.
- [x] Change `cutAdjacentFacetIntegral(...)` so it cannot silently mean all
      interior faces.
- [x] Require an interface marker, generated facet-set marker, or cut-context
      binding when installing cut-cell stabilization.
- [x] Keep plain `dS()` available only for intentionally global DG terms, not
      cut-cell stabilization.
- [x] Decide whether cut-cell stabilization uses only facets with one cut cell or
      a wider extension patch.
- [x] Document the selected extension-patch rule.

### Implementation Checklist

- [x] Generate cut-adjacent interior facet sets whenever the active cut context
      is rebuilt.
- [x] Store the generated facet set in `CutIntegrationContext` with a stable
      marker or handle.
- [x] Route cut-adjacent facet kernels through the assembler using that marker or
      handle.
- [x] Bind cut metadata scale per cut-adjacent facet.
- [x] Ensure non-cut interior faces receive zero cut-cell stabilization
      contribution even when `Use_cut_metadata_scale=false`.
- [x] Make the behavior of `Use_cut_metadata_scale=false` mean unscaled local
      cut-neighborhood stabilization, not global stabilization.
- [x] Include the interface side and active-domain side in stabilization
      diagnostics.
- [x] Fail fast if cut-cell stabilization is enabled but no cut context or
      cut-adjacent facet set exists.

### Tests

- [x] Add a unit test where only one element is cut and only adjacent facets are
      visited by stabilization assembly.
- [x] Add a unit test where a far-field interior face receives zero contribution.
- [x] Add tests with `Use_cut_metadata_scale=true`.
- [x] Add tests with `Use_cut_metadata_scale=false`.
- [x] Add a regression test for D18/D38 stabilization setup with active cut
      metadata.
- [x] Add a solver-level smoke test proving stabilization does not suppress the
      expected dam-break velocity growth through global over-damping.

## 5. Improve Level-Set Maintenance And Wet-Region Diagnostics

Problem: level-set transport is a plain advective equation with optional SUPG,
and output uses vertex sign masking rather than cut-volume wet fractions.

### Maintenance Checklist

- [x] Decide which validation cases require reinitialization and which should run
      without it.
- [x] Decide which validation cases require volume correction and which should
      run without it.
- [x] Document that plain level-set advection is not conservative by itself.
- [x] Add an option for conservative level-set transport or document why it is
      deferred.
- [x] If projection-based reinitialization remains the default, add diagnostics
      reporting interface displacement caused by reinitialization.
- [x] If global-shift volume correction remains the default, add diagnostics
      reporting shift magnitude and achieved volume error.
- [x] Ensure reinitialization and volume correction update the accepted solution,
      previous-state history, and cut context consistently.
- [x] Ensure maintenance does not invalidate the time integrator history in a way
      that changes the next BDF residual unexpectedly.

### Diagnostic Output Checklist

- [x] Add cell-centered wet volume fraction output from generated cut-volume
      metadata.
- [x] Add total wet volume output per accepted step.
- [x] Add wet-volume drift output relative to the initial accepted volume.
- [ ] Add cut-cell count, full-wet cell count, and full-dry cell count output.
- [ ] Keep vertex-sign `ActiveFluid` only as a visualization helper.
- [ ] Rename or document vertex-sign `ActiveFluid` so it is not mistaken for the
      integration active domain.
- [ ] Output the active-domain side, isovalue, and interface marker in run logs.
- [ ] Output a warning when vertex-sign masking and cell wet fraction disagree
      strongly near the interface.

### Tests

- [ ] Add an output test that verifies cell wet fraction is written for cut cells.
- [ ] Add an output test that verifies full-wet and full-dry cells report 1 and 0.
- [ ] Add a maintenance test proving volume correction changes the next cut
      context before output.
- [ ] Add a maintenance test proving reinitialization does not silently move the
      zero contour beyond a configured tolerance.
- [ ] Add a long-run diagnostic test that reports wet-volume drift.

## 6. Integration And Acceptance Tests

These tests should gate the remediation before D18/D38 are treated as qualified
benchmark runs.

### Unit And Component Tests

- [ ] Cut context rebuilds from current Newton state.
- [ ] Cut context rebuilds from line-search trial state.
- [ ] Rejected line-search trial does not leave stale trial cut context installed.
- [ ] Missing active domain is rejected for validation unfitted free-surface
      cases.
- [ ] Cut-volume quadrature integrates constants exactly.
- [ ] Cut-volume quadrature integrates linear fields to the selected exactness
      target.
- [ ] Cut-adjacent stabilization is local to the generated cut-neighborhood.
- [ ] Wet-fraction output matches generated cut-volume metadata.

### Solver Smoke Tests

- [ ] Run a one-step static flat-interface case and confirm zero spurious motion.
- [ ] Run a one-step moving-interface case and confirm the wet volume used by
      Navier-Stokes changes with `phi`.
- [ ] Run a short D18 probe and confirm pressure departs from the invalid
      full-volume hydrostatic state.
- [ ] Run a short D18 probe and confirm velocity grows in the released/retained
      water region.
- [ ] Run a short D18 probe and confirm the active wet region matches cut-volume
      metadata.
- [ ] Run a short D38 probe with the same checks.

### Benchmark Acceptance Tests

- [ ] Complete a D18 run to the configured comparison time without nonlinear
      stalls.
- [ ] Complete a D38 run to the configured comparison time without nonlinear
      stalls.
- [ ] Compare D18 peak height, front position if applicable, pressure range,
      velocity range, and wet-volume drift against acceptance thresholds.
- [ ] Compare D38 peak height, front position if applicable, pressure range,
      velocity range, and wet-volume drift against acceptance thresholds.
- [ ] Record the solver controls, nonlinear iterations, linear iterations,
      wet-volume history, and output diagnostics in a qualification log.
- [ ] Keep D18/D38 qualification blocked until all high-priority checklist items
      above are complete.

## 7. Literature Alignment Checks

Use these as numerical-method guardrails while implementing the fixes.

- [ ] Confirm the chosen geometry update strategy is explicitly monolithic,
      outer-iterated, or explicit, rather than accidentally stale.
- [ ] Confirm cut-cell quadrature is adequate for the finite-element order and
      nonlinear Navier-Stokes/VMS integrands.
- [ ] Confirm ghost penalties are restricted to a cut-neighborhood or documented
      extension patch.
- [ ] Confirm level-set mass loss is either controlled by a conservative method,
      monitored and corrected, or accepted with a documented tolerance.
- [ ] Confirm output diagnostics report the actual integration wet region, not
      only vertex signs.

Relevant methods to compare against include Sussman-Smereka-Osher level-set
transport and reinitialization, Olsson-Kreiss conservative level sets,
CutFEM/XFEM unfitted integration with localized ghost penalties, and
moment-fitted or subcell quadrature for implicitly cut finite elements.

## Recommended Work Order

- [ ] First, implement nonlinear-state-consistent cut-context rebuilding.
- [ ] Second, make missing active-domain configuration a hard validation error
      for unfitted one-fluid free-surface validation cases.
- [ ] Third, localize cut-cell stabilization so solver probes are not affected by
      global artificial damping.
- [ ] Fourth, replace constant-only cut-volume quadrature with higher-order
      cut-cell quadrature.
- [ ] Fifth, add wet-fraction diagnostics and maintenance/cut-context consistency
      checks.
- [ ] Finally, rerun D18/D38 qualification only after the unit and smoke tests
      pass.
