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
centroid moment fitting for order 0 and order 1 rules, and quadratic subcell
rules for cut sides when order 2 or higher is requested. The generic
`makeMomentFittedCutVolumeQuadrature(...)` utility remains available, but the
builder stores the computed side quadrature directly so topology metadata and
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
      Any quadratic geometry, field, or form input requests at least order 2,
      and generated cut-side rules now report quadratic exactness when subcell
      quadrature points are available.
- [x] Decide whether the first production path uses sub-triangulation,
      moment-fitted quadrature, or a hybrid of both.
- [x] Audit the existing `makeMomentFittedCutVolumeQuadrature(...)` utility and
      decide whether it can be promoted to the generated level-set interface
      path.
- [x] Define fallback behavior for degenerate cuts, near-zero volume fractions,
      and interface cuts passing close to vertices.
- [x] Preserve exact total measure for constants even when higher-order
      quadrature is requested.
      Higher requested orders report the implemented exact polynomial order
      while keeping conservative side measures and weights.

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
      Cut triangles, quads, and tetrahedra now use positive-weight quadratic
      subcell rules for cut-side volume integration when requested.
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
- [x] Prune generated active cut-volume slivers below the cut-quadrature
      validity fraction so near-zero retained regions do not enter assembly or
      cut-adjacent stabilization scaling.

### Tests

- [x] Add exact constant integration tests for all supported element types.
- [x] Add exact linear integration tests for cut triangles.
- [x] Add exact linear integration tests for cut quads.
- [x] Add exact linear integration tests for cut tetrahedra.
- [x] Add polynomial tests for quadratic fields.
      Quadratic subcell tests integrate quadratic fields exactly over 2D and 3D
      linear cut-volume regions to match the P1 Navier-Stokes/VMS convective
      integrand order.
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
- [x] Cap generated cut-adjacent stabilization scales for near-zero wet
      fractions and report capped facet counts in cut-context rebuild
      diagnostics.
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
- [x] Add cut-cell count, full-wet cell count, and full-dry cell count output.
- [x] Keep vertex-sign `ActiveFluid` only as a visualization helper.
- [x] Rename or document vertex-sign `ActiveFluid` so it is not mistaken for the
      integration active domain.
- [x] Output the active-domain side, isovalue, and interface marker in run logs.
- [x] Output a warning when vertex-sign masking and cell wet fraction disagree
      strongly near the interface.

### Tests

- [x] Add an output test that verifies cell wet fraction is written for cut cells.
- [x] Add an output test that verifies full-wet and full-dry cells report 1 and 0.
- [x] Add a maintenance test proving volume correction changes the next cut
      context before output.
- [x] Add a maintenance test proving reinitialization does not silently move the
      zero contour beyond a configured tolerance.
- [x] Add a long-run diagnostic test that reports wet-volume drift.

## 6. Integration And Acceptance Tests

These tests should gate the remediation before D18/D38 are treated as qualified
benchmark runs.

### Unit And Component Tests

- [x] Cut context rebuilds from current Newton state.
- [x] Cut context rebuilds from line-search trial state.
- [x] Rejected line-search trial does not leave stale trial cut context installed.
- [x] Missing active domain is rejected for validation unfitted free-surface
      cases.
- [x] Cut-volume quadrature integrates constants exactly.
- [x] Cut-volume quadrature integrates linear fields to the selected exactness
      target.
- [x] Cut-adjacent stabilization is local to the generated cut-neighborhood.
- [x] Wet-fraction output matches generated cut-volume metadata.

### Solver Smoke Tests

- [x] Run a one-step static flat-interface case and confirm zero spurious motion.
- [x] Run a one-step moving-interface case and confirm the wet volume used by
      Navier-Stokes changes with `phi`.
- [x] Add compact parseable diagnostics for D18/D38 failures before further
      solver iteration: cut-context provenance, active wet-region summaries,
      cut-volume quadrature sanity, cut-adjacent stabilization scales,
      residual block norms, and pressure-gauge initialization checks.
- [x] Add a smoke-script switch for D18/D38 solver-control probes that disables
      coupled outer FGMRES in the temporary case copy without editing fixtures.
- [x] Add parseable solution-state component norms and ranges to short D18/D38
      probes so pressure and velocity changes can be checked before VTK output.
- [x] Add no-output D18/D38 smoke checks that evaluate parsed pressure-gauge
      values, solution ranges, and active-volume consistency on timeout.
- [x] Add no-output D18/D38 smoke checks that verify parsed cut-volume
      diagnostics include quadratic exact-order rules after quadratic subcell
      quadrature is enabled.
- [x] Add smoke-script qualification logging that records solver controls,
      nonlinear and linear iteration summaries, wet-volume histories, output
      diagnostics, and parsed timeout diagnostics even when VTK output is
      disabled or unavailable.
- [x] Add smoke-script MPI launcher support so D18/D38 qualification runs can
      use the same multi-rank execution path as the long-run evidence while
      still writing the parsed qualification log.
- [x] Make active cut-context zero-wet-volume validation MPI-global so ranks
      with no local wet cells do not abort when the global active wet volume is
      positive, and report global cut-context volumes and counts in the rebuild
      diagnostic.
- [x] Aggregate rank-local MPI cut-volume assembly diagnostics in the smoke
      script so active-volume, quadrature-order, and wet-volume checks compare
      global assembly summaries against global cut-context rebuild summaries.
- [x] Add parseable FSILS true-residual diagnostics and smoke parsing for
      failed BlockSchur D18/D38 probes so formulation-block residual
      contributions are captured before additional solver changes.
- [x] Add constraint-component mean, RMS, fluctuation, and residual mean
      diagnostics to failed FSILS true-residual records so pressure nullspace
      drift can be separated from pressure-block residual error.
- [x] Add compact FSILS BlockSchur solve-summary diagnostics and smoke parsing
      for nested Schur and momentum iteration limits, tolerances, and internal
      convergence state.
- [x] Add an opt-in bounded BlockSchur true-residual retry for probes where
      native FSILS status reports success but wrapper residual validation
      rejects the solve, and report retry counts in smoke diagnostics.
- [x] Add smoke-script linear and BlockSchur subsolver override switches so
      D18/D38 solver-control probes can be run from temporary case copies
      without editing validation fixtures.
- [x] Add a smoke-script linear solver type override so D18/D38 probes can test
      monolithic and split FSILS paths from temporary case copies without
      editing validation fixtures.
- [x] Add an MMS traveling-interface GMRES diagnostic probe path that can
      require parseable finite-difference Jacobian checks, Newton direction
      checks, linear solve history, and component norms even when the solver
      exits before VTK output.
- [x] Add a component-selectable finite-difference Jacobian diagnostic so MMS
      probes can separate fixed-geometry velocity and pressure assembly checks
      from level-set geometry perturbations.
- [x] Add finite-difference Jacobian component-detail diagnostics that report
      base residual, perturbed residual, FD action, matrix action, mismatch, and
      sign-flip mismatch norms by field component for assembly debugging.
- [x] Add parseable mixed-form block installation diagnostics and top-mismatch
      Jacobian entry diagnostics so MMS probes can confirm the discovered
      residual couplings and the assembled matrix action at the largest
      finite-difference mismatches.
- [x] Route Newton cut-context refreshes through FE-ordered state vectors and
      report the refresh solution source in cut-context rebuild diagnostics so
      MMS probes can separate geometry-state ordering defects from assembly
      defects.
- [x] Add smoke-script validation for cut-context solution-source diagnostics so
      no-output MMS and D18/D38 probes can fail when nonlinear assembly refreshes
      are not fed by FE-ordered state vectors.
- [x] Make cut-context solution-source smoke validation tolerate pre-Newton
      timeouts while still checking every reported nonlinear assembly refresh.
- [x] Parse assemble-operator timing diagnostics in the smoke script so D18/D38
      timeouts can distinguish cut-adjacent interior-face work from cut-volume
      and cell assembly.
- [x] Add smoke-script parsing and controls for interior-face timing diagnostics
      so cut-adjacent stabilization probes can verify faces considered, faces
      assembled, and per-stage interior-face costs.
- [x] Add smoke-script parsing and controls for cut-volume timing diagnostics so
      no-output probes can verify indexed rule traversal, rule counts,
      quadrature-point counts, and per-stage cut-volume costs.
- [x] Make timeout validation accept pre-solution assembly diagnostics when a
      no-output probe times out before nonlinear solution-state records while
      still requiring cut-context, cut-volume, and hydrostatic initialization
      diagnostics.
- [x] Add smoke-script assembly topology consistency checks so no-output probes
      can verify cut-adjacent interior-face counts and cut-volume timing rule
      counts match the generated cut-context diagnostics.
- [x] Add a smoke-script switch for D18/D38 solver-control probes that disables
      cut metadata stabilization scaling in the temporary case copy while
      preserving local cut-neighborhood stabilization.
- [x] Add no-output smoke diagnostics for the cut-adjacent stabilization scale
      cap so D18/D38 probes can verify bounded scale values before VTK output.
- [x] Add no-output smoke parsing for generated cut-volume sliver pruning,
      including retained minimum active volume fraction and pruned region
      counts.
- [x] Run a short D18 probe and confirm pressure departs from the invalid
      full-volume hydrostatic state.
- [x] Run a short D18 probe and confirm velocity grows in the released/retained
      water region.
- [x] Run a short D18 probe and confirm the active wet region matches cut-volume
      metadata.
- [x] Run a short D38 probe with the same checks.

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

- [x] Confirm the chosen geometry update strategy is explicitly monolithic,
      outer-iterated, or explicit, rather than accidentally stale.
- [x] Confirm cut-cell quadrature is adequate for the finite-element order and
      nonlinear Navier-Stokes/VMS integrands.
- [x] Confirm ghost penalties are restricted to a cut-neighborhood or documented
      extension patch.
- [x] Confirm level-set mass loss is either controlled by a conservative method,
      monitored and corrected, or accepted with a documented tolerance.
- [x] Confirm output diagnostics report the actual integration wet region, not
      only vertex signs.

Relevant methods to compare against include Sussman-Smereka-Osher level-set
transport and reinitialization, Olsson-Kreiss conservative level sets,
CutFEM/XFEM unfitted integration with localized ghost penalties, and
moment-fitted or subcell quadrature for implicitly cut finite elements.

## Recommended Work Order

- [x] First, implement nonlinear-state-consistent cut-context rebuilding.
- [x] Second, make missing active-domain configuration a hard validation error
      for unfitted one-fluid free-surface validation cases.
- [x] Third, localize cut-cell stabilization so solver probes are not affected by
      global artificial damping.
- [x] Fourth, replace constant-only cut-volume quadrature with higher-order
      cut-cell quadrature.
- [x] Fifth, add wet-fraction diagnostics and maintenance/cut-context consistency
      checks.
- [ ] Finally, rerun D18/D38 qualification only after the unit and smoke tests
      pass.
