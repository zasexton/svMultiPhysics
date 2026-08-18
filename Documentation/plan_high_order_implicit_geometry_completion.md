# High-Order Implicit Geometry Completion Plan

## Purpose

This plan enumerates the remaining work needed to turn the current OOP
`HighOrderImplicit` level-set geometry path from a milestone implementation into
a fully qualified high-order implicit geometry and quadrature infrastructure for
unfitted meshes.

The current implementation is useful for controlled validation, but it is still
limited by partial reported quadrature order, linearized terminal volume
leaves, incomplete 3D root-polished curved interface reconstruction, no
differentiated quadrature tangent, only experimental mixed-element backend
dispatch, and incomplete surface-output/curvature support.

## Current Evidence

Local source evidence:

- `SayeHyperrectangle` and `HighOrderSubcell` now have experimental
  root-polished 2D interface fragments; `HighOrderSubcell` also has an
  experimental 3D tetrahedral curved-interface path for nondegenerate active
  fragments. Planar 2D terminal volume subcells can emit positive rules up to
  order 5 over linear subcell regions, while curved volume geometry and 3D
  volume rules still use linearized terminal leaves.
- `MomentFit` is exposed in the backend enum path but reports unavailable
  capability until a production implementation exists.
- Backend and domain diagnostics now report root branch counts, root-finder
  iterations, root coordinate tolerance, root iteration caps,
  curved-fragment counts, active and generated domain
  volume/interface/total quadrature point counts, backend elapsed time, curved
  failure counters, fallback status, and requested/possible/achieved/verified
  order metadata.
- Adaptive rectangle, triangle, box, and tetra cuts still call linearized
  terminal leaf cutters at max depth in
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:2608`,
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:2748`,
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:2676`,
  and
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:2817`.
- Requested subdivision depth is capped at 8 in both Saye hyperrectangle and
  high-order subcell paths:
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:1311`
  and
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:1444`.
- `DifferentiatedQuadrature` is rejected during generated-interface validation
  in `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp:411`.
- `RefreshedFrozenQuadrature` explicitly treats quadrature points, weights,
  measures, normals, and topology as fixed during tangent assembly in
  `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp:70`.
- `SayeHyperrectangle` supports only quads/hexes, and `HighOrderSubcell`
  supports only triangles/tets in
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:1489`
  and
  `Code/Source/solver/FE/LevelSet/LevelSetImplicitCutQuadratureBackend.cpp:1516`.
- Generic cut-volume exact order is capped at 2 in
  `Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h:127`; the
  planar 2D subcell-volume helper is capped at order 5 in
  `Code/Source/solver/FE/Interfaces/LevelSetInterfaceDomain.h:132`.
- Current interface fragment quadrature supports stored curved-patch
  quadrature points. Polygon fallback remains order 1. Planar 2D
  linear-subcell volume-region quadrature can report order 5 with positive
  rules, but curved volume-region quadrature and 3D volume quadrature remain
  capped by the linearized subcell path.

Internal planning evidence:

- The existing high-order plan states that the current Saye backend recursively
  subdivides and uses linear leaf cuts until full Saye Gaussian rules and
  root-polishing are implemented:
  `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md:327`.
- The same plan states that the current simplex path uses recursive subdivision
  with linear terminal cut leaves and reports interface order 1 / volume order 2:
  `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md:478`
  and
  `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md:508`.
- The plan documents that `RefreshedFrozenQuadrature` is quasi-Newton because
  the Jacobian treats geometry-dependent quadrature data as fixed:
  `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md:1019`.
- The plan documents that exact geometry sensitivities are required before
  advertising fully consistent `DifferentiatedQuadrature`:
  `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md:1842`.
- The plan also keeps production curvature recovery outside the current
  milestone:
  `Documentation/plan_high_order_curved_implicit_level_set_quadrature.md:1860`.

External method and code references:

- Saye 2015 presents high-order quadrature for implicitly defined curved
  surfaces and volumes in hyperrectangles, using one-dimensional root finding
  and Gaussian quadrature with positive weights:
  https://epubs.siam.org/doi/10.1137/140966290
- Algoim documents two high-order implicit-domain quadrature families, including
  general smooth level-set quadrature and polynomial multi-component quadrature:
  https://algoim.github.io/
- Saye 2022 covers high-order quadrature on multi-component domains defined by
  one or more multivariate polynomials, including junctions and complex geometry:
  https://arxiv.org/abs/2105.08857
- Lederer, Pfeiler, Wintersteiger, and Lehrenfeld 2016 summarize higher-order
  unfitted FEM for Stokes interface problems, including Nitsche interface
  conditions, ghost penalty stabilization, and accurate implicit geometry:
  https://arxiv.org/abs/1605.04085
- NGSolve/ngsxfem documents multiple-level-set unfitted domains, active element
  selection, cut integration symbols, ghost penalties, and Nitsche terms:
  https://ngsolve.org/ngsolve/docs/i-tutorials/unit-8.7-mlset_pde/mlset_pde.html
- GetFEM documents mesh cutting by one or several level sets, adapted
  integration, and discontinuous XFEM fields across level sets:
  https://getfem.org/userdoc/xfem.html

## Target Outcome

A completed implementation should support high-order generated level-set
interfaces on unfitted meshes with:

- requested and achieved quadrature orders above the current interface-order-1
  and volume-order-2 caps;
- curved interface segments/patches rather than linearized terminal leaves;
- root-polished, robust, deterministic interface reconstruction;
- high-order volume and surface quadrature with finite, nonnegative production
  weights;
- exact or explicitly qualified geometry sensitivities for Newton/adjoin
  workflows;
- automatic per-cell backend selection on mixed supported meshes;
- reliable diagnostics, output, curvature recovery hooks, and validation gates.

## Current HighOrderImplicit Support Matrix

This table records the currently advertised contract. "Achieved order" is the
runtime capability cap exposed by
`implicitCutQuadratureBackendCapability(...)`; generated cell results and
domain summaries reduce the achieved order further when the actual generated
rules are lower. All high-order implicit entries currently use
`RefreshedFrozenQuadrature`, so geometry-dependent quadrature data is refreshed
between contexts but frozen in the tangent. `DifferentiatedQuadrature` remains
unavailable until the geometry sensitivities in Phase 7 are implemented and
verified.

| Backend | Element families | Level-set order status | Achieved interface / volume order | Tangent policy | Validation status |
|---|---|---|---|---|---|
| `LinearCorner` | 2D `Triangle3`, `Triangle6`, `Quad4`, `Quad8`, `Quad9`; 3D `Tetra4`, `Tetra10` | Production-qualified linear corner geometry; no high-order geometry | 2D 5 / 5 for generated planar 2D rules; 3D 1 / 2 | Frozen geometry only | Production fallback and linear-equivalence tests; not a production high-order path |
| `SayeHyperrectangle` | 2D `Quad4`, `Quad8`, `Quad9`; 3D `Hex8`, `Hex20`, `Hex27` | Experimental high-order milestone | 2D 5 / 5; 3D 1 / 2 | `RefreshedFrozenQuadrature` | 2D root-polished interface and planar volume-rule tests; 3D still blocked by terminal linearized leaves and missing qualified height-function rules |
| `HighOrderSubcell` | 2D `Triangle3`, `Triangle6`; 3D `Tetra4`, `Tetra10` | Experimental high-order milestone | 2D 5 / 5; 3D 2 / 2 for qualified tetra sphere-cap fixture | `RefreshedFrozenQuadrature` | 2D root-polished interface and planar volume-rule tests; 3D tetra interface fixture is experimental and 3D volume rules remain linearized |
| `Auto` | Dispatches quads/hexes to `SayeHyperrectangle`; triangles/tetrahedra to `HighOrderSubcell` | Experimental dispatch wrapper | Inherits selected backend | `RefreshedFrozenQuadrature` | Mixed quad/triangle and hex/tetra dispatch tests, unsupported wedge/pyramid fail-closed tests, and two-rank mixed quad/triangle MPI rank-determinism coverage |
| `MomentFit` | None enabled | Unavailable | -1 / -1 | None | Fail-closed capability diagnostics only; moment construction and exactness tests remain open |
| Wedges and pyramids | `Wedge6`, `Wedge15`, `Wedge18`, `Pyramid5`, `Pyramid13`, `Pyramid14` | Unsupported | -1 / -1 | None | Configuration must fail closed unless a future backend explicitly qualifies these families |

## Implementation Progress

2026-05-18:

- Default high-order implicit fallback behavior now fails closed when a backend
  downgrades a requested rule; explicit `LinearCorner` fallback is counted and
  reported.
- Generated cut rules now carry requested and achieved order, backend name,
  geometry mode, root residual tolerance, root coordinate tolerance, root
  iteration caps, fallback status, and topology/source revision provenance.
- Interface and volume quadrature points now preserve parent coordinates,
  reference measure factors, level-set residuals, and gradient norms for debug
  and output paths.
- `SayeHyperrectangle` has 2D root-polished curved interface fragments on
  adaptive quadrilateral leaves and can report interface order up to 5 when all
  active fragments use stored curved quadrature.
- `HighOrderSubcell` has 2D root-polished curved interface fragments on
  adaptive triangular leaves and can report interface order up to 5 when all
  active fragments use stored curved quadrature.
- `HighOrderSubcell` has an experimental 3D tetrahedral curved-interface path
  that root-polishes edge intersections and surface quadrature points, reports
  interface order 2 for the qualified P2 sphere-cap fixture, and classifies
  boundary-only curved slivers as zero-measure degenerate fragments instead of
  silently falling back to linearized interface patches.
- Planar 2D cut-volume regions now have requested-order-aware positive
  quadrature up to order 5 over linearized subcell polygons, triangles, and
  rectangles. Backend capability metadata for 2D `SayeHyperrectangle`,
  `HighOrderSubcell`, and `Auto` reports volume order 5, while 3D volume rules
  remain capped at order 2.
- Backend cell results and generated-interface lifecycle results now aggregate
  active volume/interface quadrature point counts and backend elapsed time, and
  backend diagnostic strings include those per-cell runtime counters.
- Cut scalar operator points now preserve generated-rule quadrature policy keys
  for both cut-volume and cut-interface integration, so assembly diagnostics and
  coupling metadata see the same policy identity as rule provenance.
- OOP active-domain cut-context rebuild diagnostics now surface
  backend-internal elapsed time, generated volume/interface quadrature point
  counts, active cut cells, active quadrature points, process memory, and basis
  cache size in the refresh report used by nonlinear/topology-change and
  accepted-step logging.
- Generated interface domain summaries now distinguish legacy interface
  `quadrature_point_count` from `volume_quadrature_point_count` and
  `total_quadrature_point_count`; the OOP active-cut refresh report/logs expose
  those generated-domain interface, volume, and total counters separately from
  backend-generation and active-side volume counters.
- Backend achieved-order probes now require mesh dimension, element type, and
  cut request, and are capability-limited through
  `implicitCutQuadratureBackendCapability(...)` rather than context-free
  milestone caps. `LinearCorner` advertises the 2D generated planar rule cap of
  order 5, keeps the 3D linear caps, and reports unavailable order for
  unsupported element families; generated-interface lifecycle order summaries
  now reduce from per-cell backend result metadata rather than from a
  context-free backend guess.
- Backend cell result achieved/verified orders now reduce from the generated
  active interface and volume quadrature rule provenance, so the cell summary
  cannot advertise an order higher than the rules ultimately handed to
  assembly. Lifecycle tests assert that generated-domain achieved-order
  summaries match the minimum achieved order in the generated rule provenance.
- Generated interface and volume quadrature rule provenance now carries an
  explicit `selected_implicit_quadrature_backend` string. This keeps `Auto`
  mixed-cell dispatch auditable per rule while preserving existing
  `implicit_quadrature_backend` compatibility. Focused lifecycle tests verify
  quad rules record `SayeHyperrectangle` and triangle rules record
  `HighOrderSubcell` after `Auto` dispatch.
- Generated level-set fragments and volume regions now carry cell-level
  fallback status in rule provenance, while request-level fallback status
  remains the compatibility default for manually constructed cuts. VTP cell
  data records requested interface quadrature order, achieved interface
  quadrature order, and fallback-status code (`0` unknown, `1` none, `2` used,
  `3` other) on every emitted fragment and stored quadrature-point cell, so
  visualization output exposes the same per-cell order/fallback diagnostics as
  rule provenance.
- `Auto` implicit-cut backend dispatch is implemented through the backend
  driver and generated-interface lifecycle. Quads/hexes dispatch to
  `SayeHyperrectangle`, triangles/tetrahedra dispatch to `HighOrderSubcell`,
  unsupported wedge/pyramid families fail closed through capability diagnostics,
  and lifecycle tests now cover both mixed Quad4/Triangle3 and mixed
  Hex8/Tetra4 generated-interface domains plus Wedge6/Pyramid5 fail-closed
  generated-interface diagnostics.
- Generated-interface lifecycle builds now validate exported interface and
  volume rule provenance before returning a domain. Active generated rules must
  carry embedded/source id, topology id/revision, parent entity, marker,
  requested/achieved quadrature order, selected backend, geometry mode,
  fallback policy/status, tangent policy, and root-polishing controls. Backend
  full-volume region creation now stamps deterministic topology ids and stable
  ids so fallback/full-cell volume rules satisfy the same contract.
- Backend validation now rejects generated interface/volume quadrature whose
  positive point weights do not sum back to the fragment or region measure,
  rejects volume regions whose measure and volume fraction disagree, and rejects
  active root-polished fragments whose residuals or gradient metadata violate
  the configured root tolerance.
- Root-polishing controls are now explicit request/provenance/configuration
  metadata: `Implicit_cut_root_tolerance` controls signed-level-set residual
  convergence, `Implicit_cut_root_coordinate_tolerance` controls reference-space
  coordinate convergence, and `Implicit_cut_root_max_iterations` bounds each
  branch solve. The same fields participate in generated-interface and
  application active-cut policy keys, restart capture/restore, backend
  diagnostics, and JSON debug output.
- OOP case-matrix qualification now has a reusable runner at
  `tools/run_oop_case_matrix.py`. The runner validates that each case XML uses
  the new OOP solver, executes serial or MPI ranks from isolated case copies,
  captures stdout/stderr logs, parses nonlinear convergence and timing markers
  from both streams, parses active cut-context rebuild diagnostics for
  high-order implicit backend timing, fallback cells, achieved orders, active
  cut cells, active quadrature points, RSS, and basis-cache counters, and emits
  machine-readable `summary.json` plus human-readable `summary.md` reports.
  Optional qualification gates can now fail a run on cut fallback,
  basis-cache ceilings, or RSS ceilings.
- Backend diagnostics now distinguish root-polishing absence from a zero-iteration
  polish by emitting `root_polishing=not_performed` unless root-finder iterations
  were actually recorded. Requested high-order rules downgraded to terminal
  linearized leaves also carry `high_order_downgrade=true` in addition to the
  explicit fallback reason.
- Capability tests now cover the supported and unsupported FE element-family
  variants for `LinearCorner`, `SayeHyperrectangle`, `HighOrderSubcell`,
  `Auto`, and unavailable `MomentFit` in both 2D and 3D.
- 3D tetra diagnostics include curved-fragment failure counters, edge-root
  mismatch counters, and `curved_boundary_degenerate` counts so remaining
  topology and robustness gaps are visible in generated-interface diagnostics.
- 2D root-polished curved fragments now emit deterministic, unique branch and
  topology ids per parent-cell fragment for both `SayeHyperrectangle` and
  `HighOrderSubcell`; lifecycle tests assert uniqueness in active fragments and
  generated interface-rule provenance.
- Two-rank MPI focused tests now include generated-rule provenance IDs in
  deterministic hashes for `SayeHyperrectangle` and assert identical
  `HighOrderSubcell` curved-fragment topology provenance on owned and ghost
  ranks.
- VTP/JSON debug output includes root-polish metadata, root iteration counts,
  residuals, gradient norms, fallback status, root residual tolerance, root
  coordinate tolerance, and root iteration caps.
- Verification added on 2026-05-18:
  `./build-fe-check/test_fe_levelset` passes 101 tests;
  `./build-fe-check/test_fe_geometry` passes 175 tests;
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`
  passes; `cmake --build build-unit/svMultiPhysics-build --target
  svmultiphysics` passes after the order-5 2D volume-rule slice; the rebuilt
  OOP solver completed `Channel2D_Simple` step 0 from a temporary copy with
  residual `2.5060183147769096e-08` and zero cut-volume / interface assembly
  timing before the multi-step smoke was stopped because this build's assembly
  analysis made each step very expensive. Focused lifecycle tests for backend
  runtime counters also pass:
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.LinearBackendDriverReportsSupportAndOrders:LevelSetInterfaceLifecycle.AutoBackendDriverReportsDispatchSupport:LevelSetInterfaceLifecycle.RecursiveBackendReportsStructuredMilestoneCounters:LevelSetInterfaceLifecycle.HighOrderImplicitExplicitFallbackPolicyCountsCell:LevelSetInterfaceLifecycle.MomentFitBackendReportsUnavailableCapability'`;
  the generated-rule policy-key regression passes:
  `./build-fe-check/test_fe_systems --gtest_filter='CutIntegrationInfrastructure.ImportsGeneratedLevelSetInterfaceDomainByMarker:CutIntegrationInfrastructure.FixedGeometryDiagnosticsCountHighOrderRules'`;
  after the runtime-counter and policy-key slice,
  `./build-fe-check/test_fe_levelset` still passes 101 tests and
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`
  still passes; `cmake --build build-unit/svMultiPhysics-build --target
  svmultiphysics` still rebuilds the OOP solver successfully. After the OOP
  active-domain summary slice, `cmake --build
  build-unit/svMultiPhysics-build --target svapplication`,
  `./build-unit/svMultiPhysics-build/bin/test_application --gtest_filter='LevelSetCutConfiguration.*'`,
  and `cmake --build build-unit/svMultiPhysics-build --target
  svmultiphysics` pass. After the generated-domain quadrature summary slice,
  `cmake --build build-fe-check --target test_fe_geometry`,
  `cmake --build build-fe-check --target test_fe_levelset`,
  `cmake --build build-unit/svMultiPhysics-build --target svapplication`,
  `./build-fe-check/test_fe_geometry`,
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.BuildsDomainFromScalarField:LevelSetInterfaceLifecycle.RecursiveBackendReportsStructuredMilestoneCounters:LevelSetInterfaceLifecycle.HighOrderImplicitExplicitFallbackPolicyCountsCell'`,
  `./build-unit/svMultiPhysics-build/bin/test_application --gtest_filter='LevelSetCutConfiguration.*'`,
  and
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`
  pass. After the Phase 1 backend capability/order cleanup, `cmake --build
  build-fe-check --target test_fe_levelset`, `cmake --build
  build-unit/svMultiPhysics-build --target svapplication`,
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.LinearBackendDriverReportsSupportAndOrders:LevelSetInterfaceLifecycle.BackendCapabilityReportsMilestoneContract:LevelSetInterfaceLifecycle.AutoBackendDriverReportsDispatchSupport:LevelSetInterfaceLifecycle.RecursiveBackendReportsStructuredMilestoneCounters:LevelSetInterfaceLifecycle.HighOrderImplicitExplicitFallbackPolicyCountsCell'`,
  `./build-unit/svMultiPhysics-build/bin/test_application --gtest_filter='LevelSetCutConfiguration.*'`,
  `./build-fe-check/test_fe_levelset`, and
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`
  pass. After the root-polishing/downgrade diagnostic and all-family capability
  test slice, `cmake --build build-fe-check --target test_fe_levelset`,
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.BackendCapabilityReportsMilestoneContract:LevelSetInterfaceLifecycle.BackendCapabilityCoversAllElementFamilies:LevelSetInterfaceLifecycle.RecursiveBackendReportsStructuredMilestoneCounters:LevelSetInterfaceLifecycle.HighOrderImplicitFallbackPolicyFailsClosed:LevelSetInterfaceLifecycle.HighOrderImplicitExplicitFallbackPolicyCountsCell:LevelSetInterfaceLifecycle.MomentFitBackendReportsUnavailableCapability'`,
  `./build-fe-check/test_fe_levelset`,
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`,
  `cmake --build build-unit/svMultiPhysics-build --target svmultiphysics`,
  and `./build-unit/svMultiPhysics-build/bin/test_application --gtest_filter='LevelSetCutConfiguration.*'`
  pass. After the rule-derived achieved-order reduction slice, `cmake --build
  build-fe-check --target test_fe_levelset`,
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.GeneratedOrderSummariesMatchRuleProvenance:LevelSetInterfaceLifecycle.RecursiveBackendReportsStructuredMilestoneCounters:LevelSetInterfaceLifecycle.HighOrderImplicitExplicitFallbackPolicyCountsCell:LevelSetInterfaceLifecycle.SayeHyperrectangleP2CircleApproximatesAreaAndLength:LevelSetInterfaceLifecycle.HighOrderSubcellP2CircleSegmentApproximatesAreaAndLength'`,
  `./build-fe-check/test_fe_levelset`,
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`,
  `cmake --build build-unit/svMultiPhysics-build --target svmultiphysics`,
  and `./build-unit/svMultiPhysics-build/bin/test_application --gtest_filter='LevelSetCutConfiguration.*'`
  pass. After the generated-rule validation tightening slice, `cmake --build
  build-fe-check --target test_fe_levelset`,
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.LinearBackendOutputPassesCommonValidation:LevelSetInterfaceLifecycle.InvalidBackendOutputIsRejected:LevelSetInterfaceLifecycle.BackendValidationRejectsInconsistentInterfaceNormals:LevelSetInterfaceLifecycle.BackendValidationRejectsInconsistentQuadratureWeights:LevelSetInterfaceLifecycle.BackendValidationRejectsBadRootPolishMetadata:LevelSetInterfaceLifecycle.BackendIndependentValidationFixturesCheckRuleInvariants:LevelSetInterfaceLifecycle.GeneratedOrderSummariesMatchRuleProvenance:LevelSetInterfaceLifecycle.HighOrderSubcellP2SphereCapApproximatesVolumeAndArea:LevelSetInterfaceLifecycle.SayeHyperrectangleP2CircleApproximatesAreaAndLength'`,
  `./build-fe-check/test_fe_levelset`,
  `ctest --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`,
  `cmake --build build-unit/svMultiPhysics-build --target svmultiphysics`,
  and `./build-unit/svMultiPhysics-build/bin/test_application --gtest_filter='LevelSetCutConfiguration.*'`
  pass. After the OOP case-matrix runner slice,
  `python3 -m py_compile tools/run_oop_case_matrix.py` passes;
  `python3 tools/run_oop_case_matrix.py --dry-run --ranks 1,2` validates and
  enumerates the twelve-entry serial/MPI matrix for `Channel2D`,
  `Channel2D_Simple`, `vortex_shedding`, `iliac_artery`, `pipe_simple`, and
  `pipe_RCR_3d`; `python3 tools/run_oop_case_matrix.py --case
  Channel2D_Simple --ranks 1 --timeout-seconds 120 --output-dir
  Documentation/qualification_logs/oop_case_matrix_channel2d_simple_smoke_pass_20260518`
  passes with 10 accepted steps, final residual
  `2.9693548950937454e-07`, wall time `3.17534 s`, and solver-loop time
  `0.766537 s`; and `python3 tools/run_oop_case_matrix.py --case
  Channel2D_Simple --ranks 2 --timeout-seconds 180 --output-dir
  Documentation/qualification_logs/oop_case_matrix_channel2d_simple_mpi2_smoke_20260518`
  passes with 10 accepted steps, final residual
  `8.68958e-07`, wall time `1.06654 s`, and solver-loop time `0.476671 s`.
  After rebuilding the performance solver with `cmake --build
  build/svMultiPhysics-build --target svmultiphysics --parallel`, the full
  command `python3 tools/run_oop_case_matrix.py --ranks 1,2 --timeout-seconds
  900 --output-dir
  Documentation/qualification_logs/oop_case_matrix_full_rebuilt_20260518`
  passes all twelve serial/MPI2 entries for `Channel2D`, `Channel2D_Simple`,
  `vortex_shedding`, `iliac_artery`, `pipe_simple`, and `pipe_RCR_3d`; the
  generated summary is
  `Documentation/qualification_logs/oop_case_matrix_full_rebuilt_20260518/summary.md`.
  After the cut-context parser slice, `python3 -m py_compile
  tools/run_oop_case_matrix.py` passes; `python3 tools/run_oop_case_matrix.py
  --dry-run --ranks 1,2 --fail-on-cut-fallback --max-basis-cache-entries 100
  --max-rss-kb 100000000` passes; parsing the existing high-order free-surface
  log
  `tests/cases/fluid/open_vessel_free_surface/unfitted_level_set/mms_traveling_interface_2d_record_run_20260518_after_ic_fix_13steps/solver_run_after_ic_fix_13steps.log`
  reports 75 cut-context rebuilds, `0.969233301 s` total implicit backend time,
  max 4 active cut cells, max 4057 active quadrature points, max RSS
  `137232 KB`, max basis-cache entries 3, zero fallback cells, and minimum
  achieved interface/volume orders 1/2; and `python3
  tools/run_oop_case_matrix.py --case Channel2D_Simple --ranks 1
  --timeout-seconds 120 --fail-on-cut-fallback --max-basis-cache-entries 100
  --max-rss-kb 100000000 --output-dir
  Documentation/qualification_logs/oop_case_matrix_channel2d_simple_cutparser_20260518`
  passes. After the support-matrix and differentiated-tangent diagnostic slice,
  `cmake --build build-fe-check --target test_fe_levelset --parallel` passes,
  and `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.RejectsDifferentiatedQuadratureTangentPolicyUntilSensitivitiesExist:LevelSetInterfaceLifecycle.ReportsQuadratureSensitivityCapabilityStub:LevelSetInterfaceLifecycle.BackendCapabilityReportsMilestoneContract:LevelSetInterfaceLifecycle.BackendCapabilityCoversAllElementFamilies:LevelSetInterfaceLifecycle.AutoBackendDriverReportsDispatchSupport:LevelSetInterfaceLifecycle.AutoBackendDispatchesMixedQuadTriangleCells:LevelSetInterfaceLifecycle.BackendDiagnosticStatusNamesAreStable'`
  passes. The rebuilt OOP solver target `cmake --build
  build/svMultiPhysics-build --target svmultiphysics --parallel` passes, and
  `python3 tools/run_oop_case_matrix.py --case Channel2D_Simple --ranks 1,2
  --timeout-seconds 120 --fail-on-cut-fallback --max-basis-cache-entries 100
  --max-rss-kb 100000000 --output-dir
  Documentation/qualification_logs/oop_case_matrix_channel2d_simple_post_tangent_20260518`
  passes.
- After the OOP qualification regression-gate slice, the case-matrix runner
  parses cut-context RSS and basis-cache growth from active-domain rebuild
  diagnostics, can fail runs on `--max-rss-growth-kb` and
  `--max-basis-cache-entry-growth`, and can compare wall time, solver-loop
  time, final nonlinear residual, total Newton iterations, total linear
  iterations, and implicit-cut backend time against a previous `summary.json`
  through explicit ratio gates. Verification: `python3 -m py_compile
  tools/run_oop_case_matrix.py` passes; `python3 tools/run_oop_case_matrix.py
  --dry-run --ranks 1,2 --fail-on-cut-fallback
  --max-basis-cache-entries 100 --max-basis-cache-entry-growth 10
  --max-rss-kb 100000000 --max-rss-growth-kb 1000000
  --baseline-summary
  Documentation/qualification_logs/oop_case_matrix_full_rebuilt_20260518/summary.json
  --max-wall-ratio 10 --max-solver-loop-ratio 10
  --max-final-residual-ratio 100 --max-total-newton-iters-ratio 10
  --max-total-linear-iters-ratio 10` passes; parsing the existing MMS
  high-order free-surface log reports 75 cut-context rebuilds, peak RSS growth
  `23124 KB`, peak basis-cache growth `2`, max RSS `137232 KB`, and max
  basis-cache entries `3`; `python3 tools/run_oop_case_matrix.py --case
  Channel2D_Simple --ranks 1 --timeout-seconds 120 --baseline-summary
  Documentation/qualification_logs/oop_case_matrix_full_rebuilt_20260518/summary.json
  --max-wall-ratio 3 --max-solver-loop-ratio 3 --max-final-residual-ratio 10
  --max-total-newton-iters-ratio 2 --max-total-linear-iters-ratio 2
  --output-dir
  Documentation/qualification_logs/oop_case_matrix_channel2d_simple_baseline_gate_20260518`
  passes with 10 accepted steps, final residual `2.9693548950937454e-07`,
  wall time `3.67914 s`, and solver-loop time `0.745181 s`.
- After the curved-interface VTP output slice, generated level-set interface
  VTP output includes stored interface quadrature points as `Verts` in addition
  to fragment line/polygon topology, with point/cell arrays for quadrature
  flags, weights, reference measure factors, level-set residuals, gradient
  norms, root-polish metadata, curvature estimates, normals, markers, and
  parent cells. A follow-up VTP provenance slice adds point parent coordinates
  and cell arrays for stable fragment ids, fragment kind, curved-patch flags,
  local fragment indices, and conditioning status, so stored curved quadrature
  points and their visualization topology can be traced back to the generated
  patch/component and root-conditioning path.
  Verification: `cmake --build build-fe-check --target
  test_fe_geometry --parallel` passes; `./build-fe-check/test_fe_geometry
  --gtest_filter='LevelSetInterfaceGeometryWriter.*'` passes 6 tests; and
  `ctest --test-dir build-fe-check -R '^FE_Geometry_Tests$'
  --output-on-failure` passes 176 tests. Cross-checks after rebuilding the OOP
  solver target: `cmake --build build/svMultiPhysics-build --target
  svmultiphysics --parallel` passes; `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes; and
  `python3 tools/run_oop_case_matrix.py --case Channel2D_Simple --ranks 1,2
  --timeout-seconds 120 --baseline-summary
  Documentation/qualification_logs/oop_case_matrix_full_rebuilt_20260518/summary.json
  --max-wall-ratio 3 --max-solver-loop-ratio 3 --max-final-residual-ratio 10
  --max-total-newton-iters-ratio 2 --max-total-linear-iters-ratio 2
  --output-dir
  Documentation/qualification_logs/oop_case_matrix_channel2d_simple_vtp_output_20260519`
  passes. The parent-coordinate/fragment-id follow-up slice also passes the
  same `Channel2D_Simple` serial/MPI2 baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_vtp_parent_ids_20260519`,
  with serial wall `3.12851 s`, MPI2 wall `1.06677 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  adding VTP conditioning status, the refreshed baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_vtp_conditioning_20260519`
  passes with serial wall `3.07522 s`, MPI2 wall `1.01673 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  adding explicit selected-backend quadrature provenance, the refreshed
  baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_selected_backend_provenance_20260519`
  passes with serial wall `3.63029 s`, MPI2 wall `1.16871 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  adding VTP requested/achieved order and fallback-status cell data, the
  refreshed baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_vtp_per_cell_status_20260519`
  passes with serial wall `3.22882 s`, MPI2 wall `1.11718 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  adding mixed Hex8/Tetra4 `Auto` dispatch lifecycle coverage,
  `cmake --build build-fe-check --target test_fe_levelset --parallel` passes;
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.AutoBackendDispatchesMixed*'`
  passes the mixed 2D and 3D dispatch tests; `ctest --test-dir build-fe-check
  -R '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes;
  and the refreshed baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_auto_hex_tet_test_20260519`
  passes with serial wall `3.07921 s`, MPI2 wall `1.06676 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  routing Wedge6/Pyramid5 lifecycle failures through explicit `Auto`
  unsupported-cell diagnostics, `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.AutoBackend*'` passes 4 focused
  tests; `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes; and
  the refreshed baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_auto_wedge_pyramid_20260519`
  passes with serial wall `3.07894 s`, MPI2 wall `1.06728 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  enforcing complete generated rule provenance at lifecycle return,
  `cmake --build build-fe-check --target test_fe_levelset --parallel` passes;
  `./build-fe-check/test_fe_levelset --gtest_filter='LevelSetInterfaceLifecycle.*Generated*:LevelSetInterfaceLifecycle.*AutoBackend*:LevelSetInterfaceLifecycle.HighOrderImplicit*'`
  passes 9 focused tests; `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes; and
  the refreshed baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_provenance_enforced_20260519`
  passes with serial wall `3.13003 s`, MPI2 wall `1.06704 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts. After
  adding reverse-cell-iteration mixed-backend ordering coverage,
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.AutoBackend*'` passes 6 focused
  tests; `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes; and
  the refreshed baseline-gated smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_auto_ordering_20260519`
  passes with serial wall `3.18076 s`, MPI2 wall `1.06714 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts.
- Two-rank MPI coverage now verifies `Auto` per-cell backend dispatch on a
  shared mixed Quad4/Triangle3 mesh with rank-dependent ownership and opposite
  cell iteration order. The MPI test asserts that the quad cell selects
  `SayeHyperrectangle`, the triangle cell selects `HighOrderSubcell`, both
  interface and volume rule-signature hashes are identical on the two ranks,
  and active fragment/volume-region counts match across ranks. Verification:
  `cmake --build build-fe-check --target test_fe_levelset_mpi --parallel`
  passes; `mpiexec -n 2 ./build-fe-check/test_fe_levelset_mpi
  --gtest_filter=LevelSetCellEvaluatorMPI.AutoBackendMixedQuadTriangleDispatchIsRankDeterministic`
  passes; `ctest --test-dir build-fe-check -R
  '^test_fe_levelset_mpi_mpi_2$' --output-on-failure` passes; `ctest
  --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes; and
  the refreshed `Channel2D_Simple` serial/MPI2 smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_auto_mpi_dispatch_20260519`
  passes with serial wall `3.13142 s`, MPI2 wall `1.06708 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts.
- Two-rank MPI validation now also compares the shared mixed Quad4/Triangle3
  `Auto` generated-domain measures and deterministic interface/volume rule
  signatures against a rank-local serial reference built from the same cells.
  The MPI result matches serial negative volume, positive volume, interface
  measure, active fragment count, active volume-region count, and rule hashes
  on both ranks, and the MPI measures/rule hashes are rank-deterministic.
  Verification: `cmake --build build-fe-check --target test_fe_levelset_mpi
  --parallel` passes; `mpiexec -n 2 ./build-fe-check/test_fe_levelset_mpi
  --gtest_filter='LevelSetCellEvaluatorMPI.AutoBackendMixedQuadTriangle*'`
  passes; `ctest --test-dir build-fe-check -R
  '^test_fe_levelset_mpi_mpi_2$' --output-on-failure` passes; and `ctest
  --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- Explicit global high-order backends now preflight all generated-interface
  cells before generation. `SayeHyperrectangle`, `HighOrderSubcell`, and
  unavailable `MomentFit` fail closed when a mesh contains an unsupported cell
  family, with diagnostics naming the backend, cell id, element type, fallback
  policy, capability state, and possible orders. If
  `Implicit_cut_fallback_policy=LinearCorner` is explicitly configured and the
  unsupported cell family has a linear-corner cutter and the requested backend
  is implemented, the lifecycle uses counted `LinearCorner` fallback for that
  cell and stamps fallback-status provenance on the generated rules; unavailable
  backends such as `MomentFit` still fail closed. Verification:
  `cmake --build build-fe-check --target
  test_fe_levelset --parallel` passes; `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.GlobalBackend*:LevelSetInterfaceLifecycle.RejectsHighOrderImplicitModeOnUnsupportedTetra:LevelSetInterfaceLifecycle.AutoBackendDispatchesMixedQuadTriangleCells:LevelSetInterfaceLifecycle.AutoBackendMixedQuadTriangleRuleOrderIndependentOfCellIteration:LevelSetInterfaceLifecycle.MomentFitConfigurationFailsClosedWithCapabilityDiagnostic'`
  passes; `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes; `ctest
  --test-dir build-fe-check -R '^test_fe_levelset_mpi_mpi_2$'
  --output-on-failure` passes; `cmake --build build/svMultiPhysics-build
  --target svmultiphysics --parallel` passes; and the refreshed
  `Channel2D_Simple` serial/MPI2 smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_global_backend_validation_strict_20260519`
  passes with serial wall `3.17971 s`, MPI2 wall `1.06789 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts.
- The FE level-set user guide now documents high-order implicit generated
  interface controls in `Code/Source/solver/FE/Docs/LevelSet.md`, including
  backend selection, supported element families, current achieved-order caps,
  default fail-closed fallback behavior, explicit counted `LinearCorner`
  fallback, root-polishing controls, subdivision depth as topology isolation,
  `RefreshedFrozenQuadrature` as the current quasi-Newton tangent policy,
  rejected `DifferentiatedQuadrature`, diagnostics, VTP/JSON output fields, and
  example XML snippets for mixed `Auto` and explicit fallback runs.
- Generated curved interface quadrature now has direct normal-orientation
  coverage against `grad(phi_h)` for both active high-order paths:
  `HighOrderSubcellP2SphereCapApproximatesVolumeAndArea` checks tetra/simplex
  sphere normals and `SayeHyperrectangleP2CircleApproximatesAreaAndLength`
  checks quad/Saye circle normals. Each point must carry a unit normal,
  gradient-norm metadata matching the analytic H1 field gradient, and
  near-parallel positive alignment. Verification: `cmake --build
  build-fe-check --target test_fe_levelset --parallel` passes;
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.HighOrderSubcellP2SphereCapApproximatesVolumeAndArea:LevelSetInterfaceLifecycle.SayeHyperrectangleP2CircleApproximatesAreaAndLength'`
  passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- P1/affine generated-interface validation now covers every currently
  advertised supported high-order cell variant: Saye `Quad4`, `Quad8`,
  `Quad9`, `Hex8`, `Hex20`, and `Hex27`, plus HighOrderSubcell `Triangle3`,
  `Triangle6`, `Tetra4`, and `Tetra10`. The all-cell lifecycle regression uses
  a nondegenerate affine plane, verifies analytic negative/positive/interface
  measures, zero fallback cells, parent-measure partition, finite positive
  generated rules, and selected backend provenance. This also required the FE
  DOF distribution path to honor scalar C0 serendipity layouts for `Quad8` and
  `Hex20` instead of forcing complete Lagrange layouts. Verification:
  `cmake --build build-fe-check --target test_fe_levelset --parallel` passes;
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.AffineCutsMatchLinearMeasuresForSupportedHighOrderCells:LevelSetInterfaceLifecycle.SayeHyperrectangleP1LineMatchesLinearMeasures:LevelSetInterfaceLifecycle.SayeHyperrectangleP1PlaneMatchesHexMeasures:LevelSetInterfaceLifecycle.HighOrderSubcellP1LineMatchesLinearTriangleMeasures:LevelSetInterfaceLifecycle.HighOrderSubcellP1PlaneMatchesLinearTetraMeasures'`
  passes; `ctest --test-dir build-fe-check -R
  '^(FE_Spaces_Tests|FE_Dofs_Tests|FE_Geometry_Tests|FE_LevelSet_Tests|FE_Assembly_Tests|FE_Systems_Tests)$'
  --output-on-failure` passes; `ctest --test-dir build-fe-check -R
  '^test_fe_levelset_mpi_mpi_2$' --output-on-failure` passes; `cmake --build
  build/svMultiPhysics-build --target svmultiphysics --parallel` passes; and
  the baseline-gated `Channel2D_Simple` serial/MPI2 OOP smoke at
  `Documentation/qualification_logs/oop_case_matrix_channel2d_simple_serendipity_p1_equivalence_20260519`
  passes with serial wall `3.17595 s`, MPI2 wall `1.06789 s`, final residuals
  `2.96935e-07` and `8.68958e-07`, and unchanged accepted-step counts.
- P2/P3 level-set DOF influence is now covered from evaluator through
  generated geometry and quadrature. Existing evaluator tests verify P2 edge
  and P3 interior DOFs change `phi_h` evaluations. Existing lifecycle tests
  verify P2 triangle/tetra edge DOFs change generated cut volume, topology
  revision, and restored-state measures. New lifecycle tests verify cubic
  Saye quad and HighOrderSubcell triangle edge DOFs change generated volume
  measures, interface quadrature moments, and generated rule signatures.
  Verification: `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetCellEvaluator.P2RespondsToEdgeDofsAtInteriorNodes:LevelSetCellEvaluator.P3RespondsToInteriorDofsWhereAvailable:LevelSetInterfaceLifecycle.HighOrderSubcellTriangleP2EdgeDofMovesCutVolume:LevelSetInterfaceLifecycle.HighOrderSubcellTetraP2EdgeDofMovesCutVolume:LevelSetInterfaceLifecycle.SayeHyperrectangleQuadP3DofMovesGeometryAndQuadrature:LevelSetInterfaceLifecycle.HighOrderSubcellTriangleP3DofMovesGeometryAndQuadrature'`
  passes, and `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes.
- 2D circle/ellipse convergence is now explicitly covered for the active
  generated-interface paths. Saye hyperrectangle tests cover circle h- and
  p-refinement plus ellipse h-refinement against analytic area and perimeter
  references. HighOrderSubcell triangle tests cover quadratic circle-segment
  h-refinement against analytic area and arc-length references. Verification:
  `cmake --build build-fe-check --target test_fe_levelset --parallel` passes;
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.HighOrderSubcellTriangleCircleErrorsDecreaseUnderHRefinement:LevelSetInterfaceLifecycle.SayeHyperrectangleEllipseErrorsDecreaseUnderHRefinement'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- 3D sphere/ellipsoid convergence is now explicitly covered for the active
  generated-interface paths. Saye hyperrectangle tests cover Hex27 sphere and
  prolate ellipsoid h-refinement against analytic volume and surface-area
  references while preserving the current explicit 3D fallback policy.
  HighOrderSubcell tests cover Tetra10 spherical cap h-refinement against
  analytic volume and surface-area references. Verification: `cmake --build
  build-fe-check --target test_fe_levelset --parallel` passes;
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.HighOrderSubcellTetraSphereCapErrorsDecreaseUnderHRefinement:LevelSetInterfaceLifecycle.SayeHyperrectangleSphereAndEllipsoidErrorsDecreaseUnderHRefinement'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- Affine plane-cut exactness on high-order physical cells is now covered by
  exact measure checks and generated quadrature point checks. The all-cell
  affine regression covers Saye `Quad4/8/9`, `Hex8/20/27`, and
  HighOrderSubcell `Triangle3/6`, `Tetra4/10`. A focused high-order physical
  cell regression verifies `Quad9`, `Hex27`, and `Tetra10` interface
  quadrature points lie on the affine plane, retain zero residuals, and
  preserve exact negative/positive/interface measures. Verification:
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.AffineCutsMatchLinearMeasuresForSupportedHighOrderCells:LevelSetInterfaceLifecycle.AffinePlaneCutsStayPlanarOnHighOrderPhysicalCells'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- Curved physical geometry mapping with a non-affine parent cell is covered by
  the Hex27 midplane mapping regression. The test builds Saye reference-frame
  generated rules, maps them through a non-affine quadratic Hex27 physical
  mapping, verifies mapped negative/positive physical volumes against the
  curved Jacobian, and verifies the mapped interface area against an
  independent quadrature reference. Verification:
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.SayeHyperrectangleReferenceRulesMapToCurvedHexGeometry'`
  passes; this is also included in the passing `FE_LevelSet_Tests` and
  `FE_Geometry_Tests|FE_LevelSet_Tests` runs above.
- Small-volume and near-tangent robustness is now covered for the active
  generated-interface paths. The small-volume regression verifies Saye `Quad9`
  and HighOrderSubcell `Triangle6` cuts return finite positive generated rules,
  preserve parent-measure partitions, and approximate tiny analytic measures.
  Existing near-tangent/degenerate regressions verify Saye circle tangent
  handling returns either finite rules or an explicit tangent/fallback
  diagnostic, and HighOrderSubcell vertex/edge touches return degenerate
  diagnostics without fallback. Verification: `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.SmallVolumeCutsReturnFinitePositiveRules:LevelSetInterfaceLifecycle.SayeHyperrectangleNearTangentCircleReturnsRulesOrDiagnostic:LevelSetInterfaceLifecycle.HighOrderSubcellTriangleVertexAndEdgeTouchesAreDegenerate'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- Multiple-component parent-cell coverage now exercises the active
  generated-interface paths for separated boundary-intersecting components,
  and same-sign-corner interior islands are no longer treated as categorically
  unsupported on the qualified 2D quadrilateral path. A quartic Saye `Quad4`
  level set creates two separated corner circle arcs in one parent, a quartic
  HighOrderSubcell `Triangle3` level set creates two separated vertex circle
  arcs in one parent, and the small-volume Saye `Quad9` regression exercises a
  fully interior circular island with positive corner and edge values and a
  negative interior high-order value. The regression verifies finite rules,
  exact parent-measure partitioning, total measures, and separate left/right
  interface measures. Verification: `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.MultipleComponentsInSingleParentCellProduceSeparatedRules'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
  Remaining component-discovery qualification is still incomplete for broad 3D
  multi-component, branching, high-curvature, and unsupported mixed-cell
  families.
- Polynomial exactness coverage now covers the currently advertised
  generated-rule orders. Saye `Quad9` and HighOrderSubcell `Triangle6` affine
  cuts integrate degree-5 volume and interface moments against analytic
  references at requested/achieved order 5. Saye `Hex27` affine cuts integrate
  advertised 3D volume/interface moments at achieved volume/interface orders
  `2/1`, and HighOrderSubcell `Tetra10` affine cuts integrate advertised
  degree-2 volume/interface moments at achieved order `2/2`. Full 3D box and
  tetra generated-volume subregions now emit degree-2-capable volume rules
  instead of centroid-only rules, so the advertised 3D volume exactness is
  backed by moment checks. Verification: `cmake --build build-fe-check
  --target test_fe_levelset --parallel` passes; `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.Affine2DCutRulesIntegratePolynomialMomentsToAdvertisedOrder:LevelSetInterfaceLifecycle.Affine3DCutRulesIntegratePolynomialMomentsToAdvertisedOrder:LevelSetInterfaceLifecycle.Affine2DCutRulesImprovePolynomialMomentsUnderQRefinement'`
  passes three tests; and `ctest --test-dir build-fe-check -R
  '^FE_LevelSet_Tests$' --output-on-failure` passes.
- h-refinement at fixed one-dimensional quadrature order is covered by the
  generated-interface convergence regressions above. Saye circle, Saye
  ellipse, Saye sphere/ellipsoid, HighOrderSubcell triangle circle, and
  HighOrderSubcell tetra sphere-cap tests keep interface and volume
  quadrature orders fixed while increasing subdivision depth, and verify
  reduced analytic measure errors.
- q-refinement at fixed geometry is covered for 2D generated rules. Saye
  `Quad9` and HighOrderSubcell `Triangle6` affine cuts reuse the same plane
  geometry and compare low requested quadrature order against order 5 for the
  same degree-5 volume and interface moments, verifying the higher-order rules
  reduce the analytic moment errors to exactness tolerance. Verification:
  `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.Affine2DCutRulesImprovePolynomialMomentsUnderQRefinement'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- Positive generated-rule weights and parent-measure conservation are covered
  by a representative invariant sweep across Saye `Quad9`, Saye `Hex27`,
  HighOrderSubcell `Triangle6`, and HighOrderSubcell `Tetra10`. The sweep
  checks every exported generated volume/interface rule has finite positive
  quadrature weights and that each parent-cell volume split sums back to the
  recorded parent measure. Verification: `./build-fe-check/test_fe_levelset
  --gtest_filter='LevelSetInterfaceLifecycle.GeneratedRulesConserveParentMeasureAndUsePositiveWeights'`
  passes; `ctest --test-dir build-fe-check -R '^FE_LevelSet_Tests$'
  --output-on-failure` passes; and `ctest --test-dir build-fe-check -R
  '^(FE_Geometry_Tests|FE_LevelSet_Tests)$' --output-on-failure` passes.
- High-order cut assembly is now covered for the fixed-geometry tangent policy.
  Volume regressions assemble many-point high-order generated rules into mass
  matrices and polynomial-moment matrices, verify nonlinear residual/tangent
  matrices against central finite differences, and retain CutDomainAssembler
  AD/symbolic residual+tangent parity. Interface regressions assemble
  many-point curved-rule surface mass and traction forms, verify nonlinear
  interface tangents against central finite differences, and compare
  many-point high-order volume/interface rules through JIT and interpreter
  paths. Verification: `cmake --build build-fe-check --target test_fe_forms
  --parallel` passes, and `./build-fe-check/test_fe_forms
  --gtest_filter='CutCellForms.HighOrderCutVolumeTangentMatchesFixedGeometryFiniteDifference:CutCellForms.HighOrderCutVolumeManyPointRuleKeepsBasisEvaluation:CutCellForms.HighOrderCutVolumePolynomialMomentsMatchGeneratedRule:CutCellForms.HighOrderCutInterfaceTangentMatchesFixedGeometryFiniteDifference:CutCellForms.HighOrderCutInterfaceManyPointRuleRemapsBasisEvaluation:CutCellForms.HighOrderCutInterfaceSurfaceTractionMatchesGeneratedRule:CutCellForms.HighOrderGeneratedCutRulesJITMatchInterpreter:CutCellForms.CutDomainAssemblerUsesCutRulesForResidualAndTangentKernels'`
  passes eight tests.
- The compact high-order MMS traveling-interface smoke now has current
  solver-level evidence. Using the committed MMS fixture via `--source-ref
  HEAD`, `Generated_interface_geometry=HighOrderImplicit`,
  `Implicit_cut_quadrature_backend=SayeHyperrectangle`, fallback policy `Fail`,
  and a compact `2x2` P2 mesh, the one-step run accepts with zero implicit-cut
  fallback cells, achieved interface/volume orders `2/2`, max RSS `142240 KB`,
  and max basis-cache entries `3`. The run still exposes performance debt under
  `RefreshedFrozenQuadrature`: 28 nonlinear iterations, 108 cut-context rebuilds
  per accepted step, 79 assembly timing records per accepted step, and 51 extra
  assembly timing records per accepted step. Verification:
  `python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py
  --solver build/svMultiPhysics-build/bin/svmultiphysics --source-ref HEAD
  --case mms2d --steps 1 --disable-vtk-output
  --use-high-order-implicit-cuts --max-nonlinear-iterations 60
  --max-diagnostic-assembly-timings-per-step 100
  --max-diagnostic-extra-assembly-timings-per-step 100
  --max-diagnostic-cut-context-rebuilds-per-step 120
  --require-time-loop-convergence --qualification-log
  Documentation/qualification_logs/high_order_implicit_mms2d_smoke_20260519.json
  --timeout-seconds 240` passes.
- The root open-vessel high-order smoke is now wired into the unfitted free
  surface harness as `--case open2d` with an explicit root-case copy whitelist
  (`solver.xml`, `pressure_gauge.csv`, and `mesh`) so temporary runs do not
  capture nested validation cases or old result artifacts. The one-step
  high-order implicit run accepts with zero implicit-cut fallback cells,
  achieved interface/volume orders `2/2`, stable reference wet-volume history
  at `7.2`, 1 nonlinear iteration, 2 linear iterations, pressure range
  `4896.167`, velocity range `8.66842e-19`, max RSS `113584 KB`, and max
  basis-cache entries `2`. Verification:
  `python3 tests/cases/fluid/open_vessel_free_surface/run_test05_velocity_growth_smoke.py
  --solver build/svMultiPhysics-build/bin/svmultiphysics --case open2d
  --steps 1 --disable-vtk-output --use-high-order-implicit-cuts
  --disable-cut-metadata-scale --disable-velocity-extension
  --max-nonlinear-iterations 40
  --max-diagnostic-assembly-timings-per-step 120
  --max-diagnostic-extra-assembly-timings-per-step 80
  --max-diagnostic-cut-context-rebuilds-per-step 120
  --require-time-loop-convergence --qualification-log
  Documentation/qualification_logs/high_order_open_vessel2d_smoke_20260519.json
  --timeout-seconds 240` passes, and the qualification JSON validates with
  `python3 -m json.tool`.

Major remaining gaps include production 3D curved surface rules beyond the
experimental tetra path, high-order curved volume quadrature beyond the current
2D planar linearized-subcell order-5 slice and 3D order-2 cap,
differentiated quadrature sensitivities, 3D generated-rule polynomial moment
exactness,
`MomentFit`, production curvature recovery, broad 3D multi-component and
branch/topology qualification,
higher-rank benchmark coverage, committed high-order benchmark ceilings, and
CI-enforced trend comparison beyond the passing serial/MPI2 OOP case matrix.

## Phase 0: Freeze The Contract And Acceptance Criteria

- [ ] Define the first production-supported order targets for each backend:
      `SayeHyperrectangle` interface order, `SayeHyperrectangle` volume order,
      `HighOrderSubcell` interface order, and `HighOrderSubcell` volume order.
      Partial: the current support matrix above defines the advertised
      milestone capability caps and status for each backend. First
      production-qualified high-order targets remain open because
      `SayeHyperrectangle` and `HighOrderSubcell` are still experimental.
- [ ] Define the minimum order policy as
      `max(parent_geometry_order, level_set_interpolation_order, form_integrand_order)`.
- [ ] Define whether the first production target is exact polynomial order,
      asymptotic h-convergence order, or both.
- [ ] Define the required behavior when a requested order cannot be achieved:
      fail closed by default, or accept only with explicit fallback policy and
      counted diagnostics.
- [ ] Require zero unexpected fallback cells in high-order qualification runs.
- [x] Require all generated rules to carry requested order, achieved order,
      backend name, geometry mode, root tolerance, fallback status, and topology
      revision.
- [ ] Define separate acceptance gates for geometry-only tests, assembly tests,
      nonlinear solver tests, and benchmark qualification.
- [x] Add a `HighOrderImplicit` support matrix table that lists element family,
      level-set order, backend, achieved order, tangent policy, and validation
      status.

## Phase 1: Backend Capability And API Cleanup

- [ ] Replace hard-coded capability caps for `SayeHyperrectangle` and
      `HighOrderSubcell` with backend-specific runtime capability data.
      Partial: public capability metadata is now the source of truth for
      achieved-order probes and backend possible-order metadata, including
      supported and unsupported element-family distinctions. The broader
      production capability matrix and status transitions remain open.
- [ ] Make `achievedInterfaceQuadratureOrder(...)` and
      `achievedVolumeQuadratureOrder(...)` reflect the actual generated rule,
      not the milestone cap.
      Partial: achieved-order probes are now request-, dimension-, and
      element-aware and capability-limited; backend cell results and lifecycle
      summaries are reduced from actual generated rule provenance. A broader
      public API split between capability probes and generated-cell achieved
      order remains open.
- [x] Distinguish requested order, possible order, achieved order, and verified
      order in backend result metadata.
- [x] Add backend result fields for root-branch count, root-finder iteration
      count, curved-fragment count, linearized-leaf count, fallback reason,
      active volume/interface quadrature point counts, and backend elapsed time.
- [x] Change diagnostics so `root_finder_iterations=0` is emitted only by
      backends that genuinely did no root polishing.
- [x] Add explicit diagnostics for "requested high-order rule downgraded to
      linearized leaf".
- [x] Keep `MomentFit` unavailable by default until it has an implementation and
      tests, but change public diagnostics from "not implemented yet" to a
      capability report that distinguishes unavailable, experimental, and
      production-qualified states.
- [x] Add unit tests for capability reporting across all supported and
      unsupported element families.

## Phase 2: SayeHyperrectangle True High-Order Rules

Scope: quadrilateral and hexahedral reference cells.

- [ ] Implement reference-space branch discovery for the zero contour/surface
      of `phi_h - isovalue`.
- [ ] Implement one-dimensional root finding along Saye-style height-function
      branches.
- [ ] Add root polishing with absolute signed-level-set residual tolerance,
      relative coordinate tolerance, and max iterations per branch.
      Partial: residual tolerance, reference-coordinate tolerance, and
      per-branch iteration caps are now explicit configuration/request fields
      and diagnostics; full Saye height-function branch polishing remains open.
- [ ] Detect and diagnose nonfinite evaluations, nonconverged roots,
      branch exits from the parent reference cell, and ambiguous roots.
- [ ] Implement dimension-reduction quadrature for 2D surface curves and 3D
      interface patches.
- [ ] Generate positive-weight Gaussian quadrature points for cut surfaces.
- [ ] Generate positive-weight Gaussian quadrature points for negative and
      positive cut volumes.
      Partial: 2D terminal rectangle and polygon subcell volume regions now use
      positive requested-order-aware rules up to order 5, but this is still
      linearized terminal geometry rather than true Saye volume quadrature.
- [ ] Preserve exact parent measure partition:
      `negative_measure + positive_measure == parent_measure` within tolerance.
- [ ] Store curved fragment data with parent coordinates, physical points,
      normals, measures, branch ids, and root residuals.
- [ ] Support `Quad4`, `Quad8`, `Quad9`, `Hex8`, `Hex20`, and `Hex27` without
      terminal linearized leaves in production mode.
- [ ] Add a diagnostic mode that can still emit subdivision leaves for debugging,
      but mark them as diagnostic visualization rather than production geometry.
- [ ] Add h-refinement and q-refinement convergence tests for lines/circles,
      ellipses, planes, spheres, and ellipsoids.
- [ ] Add tests for near-tangent cuts, cuts close to vertices/edges/faces,
      small retained volume fractions, and high-curvature resolved interfaces.

## Phase 3: HighOrderSubcell Curved Simplex Rules

Scope: triangles and tetrahedra.

- [ ] Decide whether production simplex rules use root-polished curved subcells,
      polynomial implicit quadrature, or a qualified positive-weight subcell
      refinement strategy.
- [ ] Implement root-polished curved leaves for `Triangle6+` and `Tetra10+`
      instead of ending in linear `Triangle3` / `Tetra4` leaf cuts.
      Partial: 2D triangular leaves and experimental 3D tetrahedral interface
      leaves now root-polish edge roots and active surface quadrature points;
      tetra volume regions still come from linearized subcells.
- [ ] Generate high-order interface quadrature for curved simplex cuts.
      Partial: 2D triangles can report interface order up to 5; 3D tets can
      report interface order 2 for the P2 sphere-cap fixture when all
      nondegenerate active fragments are curved and residual-checked.
- [ ] Generate high-order volume quadrature for both active sides of curved
      simplex cuts.
      Partial: 2D linearized simplex subcell volume regions now use positive
      requested-order-aware triangle rules up to order 5; curved volume
      geometry and 3D tetra volume rules remain open.
- [ ] Preserve positive production weights for Navier-Stokes/VMS paths.
      Partial: backend validation now rejects nonpositive interface/volume
      quadrature point weights and rejects stored quadrature whose point weights
      do not sum back to fragment or region measures. Full production
      qualification still depends on replacing terminal linearized leaves.
- [ ] Add deterministic branch ids and topology ids for multiple cut components
      inside one simplex.
      Partial: 2D triangular root-polished fragments now carry unique branch
      and topology ids and lifecycle tests verify generated rule provenance.
      3D tetra curved and boundary-degenerate fragments now get deterministic
      topology ids, but broad multi-component 3D qualification is still open.
- [ ] Handle repeated roots, near-tangent branches, and singular gradients with
      fail-closed diagnostics unless an explicit fallback is selected.
- [ ] Keep the existing recursive linearized-subcell mode as a diagnostic
      fallback path, not the production path for high-order qualification.
- [ ] Add convergence tests for quadratic/cubic level-set fields on triangles
      and tetrahedra.
- [ ] Add mesh-family parity tests comparing equivalent quad/tri and hex/tet
      manufactured geometries.

## Phase 4: MomentFit Backend Decision And Implementation

- [ ] Decide whether `MomentFit` is a production backend, a research backend, or
      removed from public configuration until implemented.
- [ ] If retained, define supported cells, supported level-set orders, and weight
      sign policy.
- [ ] Implement moment system construction for target polynomial moments over
      cut volumes and surfaces.
- [ ] Add conditioning diagnostics for moment matrices.
- [ ] Enforce nonnegative weights for production Navier-Stokes/VMS use, or keep
      signed weights behind an experimental flag.
- [ ] Add exactness tests for polynomial integrands up to the advertised order.
- [ ] Add comparison tests against Saye/Algoim-style rules on hyperrectangles and
      against simplex curved rules on triangles/tetrahedra.
- [ ] Add fallback behavior for singular or ill-conditioned moment systems.
- [ ] Document when moment fitting is preferred over Saye or subcell rules.

## Phase 5: Curved Fragment And Patch Data Model

- [ ] Extend `CutInterfaceFragmentKind::CurvedPatch` from placeholder metadata to
      a production carrier for curved interface fragments.
- [ ] Store parent-parametric quadrature coordinates for every curved surface and
      volume point.
- [ ] Store physical mapped coordinates after parent-geometry mapping.
- [ ] Store normal vectors derived from reference gradients and geometry
      Jacobians, not from linear leaf geometry.
- [ ] Store measure/Jacobian factors separately from quadrature weights so
      diagnostics can distinguish reference and physical measures.
- [ ] Add stable branch/component ids for multiple fragments in one parent cell.
      Partial: 2D root-polished rectangle and triangle fragments now expose
      unique branch/topology ids through generated interface rules.
- [ ] Record root residuals, gradient norms, and local conditioning flags in
      debug metadata.
      Partial: VTP cell data now records root residuals, gradient norms, and a
      numeric conditioning status derived from each fragment's conditioning
      diagnostic. Broader fail-closed conditioning policy and JSON fragment
      conditioning output remain open.
- [ ] Update `toCutQuadratureRule(...)` so curved patches support high-order
      surface quadrature, not order 0.
- [ ] Update VTP/JSON geometry writers to output exact curved-patch quadrature
      points and metadata rather than only leaf-fragment visualization.
      Partial: VTP now writes stored fragment quadrature points as vertex cells
      with quadrature-point flags, weights, reference measure factors,
      residuals, gradient norms, normals, root-polish metadata, markers, and
      parent cells, plus point parent coordinates and fragment stable ids,
      kind ids, curved-patch flags, local fragment indices, and conditioning
      status, requested/achieved interface quadrature orders, and fallback
      status on all emitted VTP cell topology; fallback status is cell-level
      for generated cuts and request-level only for legacy/manual cuts that do
      not stamp fragments. JSON already records
      generated-rule quadrature points and root/provenance metadata. Production
      curved-patch tessellation and curvature recovery output remain open.

## Phase 6: Remove Subdivision-Depth Accuracy Dependence

- [ ] Treat adaptive subdivision as topology discovery and robustness support,
      not the primary accuracy mechanism.
- [ ] Replace terminal calls to `appendLinearizedRectangleCut`,
      `appendLinearizedBoxCut`, `appendLinearizedTriangleCut`, and
      `appendLinearizedTetrahedronCut` in production high-order paths.
- [ ] Keep a bounded subdivision cap for branch isolation, but make requested
      quadrature order achievable without increasing subdivision depth for
      smooth resolved interfaces.
- [ ] Add diagnostics that report whether accuracy came from root-polished
      curved rules or from linearized fallback leaves.
- [ ] Fail high-order qualification if any production cell uses linearized leaves
      unexpectedly.
- [ ] Add regression tests proving that increasing requested quadrature order
      changes achieved order without needing to raise max subdivision depth.

## Phase 7: Geometry Derivatives And DifferentiatedQuadrature

- [ ] Define the differentiated geometry state: level-set DOFs, geometry DOFs,
      isovalue, root coordinates, quadrature weights, normals, measures, active
      side classification, and topology.
- [ ] Derive sensitivities of root coordinates with respect to level-set DOFs
      using the implicit function theorem on each branch.
- [ ] Derive sensitivities of physical point locations with respect to parent
      geometry DOFs.
- [ ] Derive sensitivities of surface and volume quadrature weights.
- [ ] Derive sensitivities of normals and curvature-related normal gradients.
- [ ] Define topology-transition behavior: fail, freeze, one-sided derivative,
      or semismooth event handling.
- [x] Implement `GeometryQuadratureSensitivitySupport` so it reports complete
      support only when point, weight, measure, normal, and topology policies are
      implemented.
- [ ] Enable `geometry_tangent_policy=DifferentiatedQuadrature` only after fixed
      and moving-geometry finite-difference checks pass.
- [ ] Add finite-difference tests perturbing level-set DOFs, geometry DOFs,
      isovalue, and material/interface parameters.
- [ ] Add adjoint/Jacobian verification tests for cut-volume and cut-interface
      forms using differentiated geometry.
- [ ] Keep `RefreshedFrozenQuadrature` available and documented as quasi-Newton.

## Phase 8: Mixed-Element Backend Dispatch

- [x] Add an `Auto` implicit-cut backend selection mode.
- [x] Route quads/hexes to `SayeHyperrectangle`.
- [x] Route triangles/tetrahedra to `HighOrderSubcell` or its production
      successor.
- [x] Define high-order support for wedges and pyramids, or fail closed with
      explicit unsupported-cell diagnostics.
- [ ] Decide whether wedges/pyramids are decomposed into supported subcells or get
      native curved rules.
      Current policy: they are not decomposed in the high-order implicit path;
      `Auto` fails closed with explicit unsupported-cell diagnostics until a
      native or decomposed rule is designed and qualified.
- [x] Preserve deterministic ordering when different backends contribute rules to
      one generated interface domain.
- [x] Include selected backend per cell in `CutQuadratureProvenance`.
- [ ] Add mixed quad/tri, hex/tet, and hex/wedge/pyramid tests.
      Partial: generated-interface lifecycle tests now cover mixed
      Quad4/Triangle3 and mixed Hex8/Tetra4 `Auto` dispatch with selected
      backend and fallback-status provenance checks. Dedicated Wedge6 and
      Pyramid5 generated-interface lifecycle tests verify that unsupported
      mixed-cell families fail closed through `Auto` backend diagnostics.
      A combined hex/wedge/pyramid mesh fixture remains open if native or
      decomposed wedge/pyramid support is later added.
- [x] Add MPI tests proving per-cell backend dispatch is rank deterministic.
      A two-rank shared mixed Quad4/Triangle3 `Auto` dispatch test now varies
      cell ownership and cell iteration order by rank, then compares selected
      backend provenance and deterministic interface/volume rule hashes across
      ranks.
- [x] Add configuration validation that rejects a global backend when the mesh
      contains unsupported high-order cut cells unless a fallback threshold is
      explicitly configured.
      Explicit `SayeHyperrectangle`, `HighOrderSubcell`, and unavailable
      `MomentFit` configurations now preflight cell-family support. Unsupported
      cells fail before generation by default; explicitly configured
      `LinearCorner` fallback is allowed only for implemented backends and cell
      families supported by the linear fallback cutter, and is counted/stamped
      in rule provenance. Unavailable backends still fail closed.

## Phase 9: Surface Output, Curvature, And Interface Fields

- [ ] Replace diagnostic leaf-fragment output with curved interface output that
      includes quadrature points, curved patch ids, normals, and measures.
      Partial: VTP output now includes stored curved-interface quadrature
      points, normals, weights, residuals, gradient norms, root-polish metadata,
      markers, parent cells, parent coordinates, fragment stable ids, fragment
      kind ids, curved-patch flags, local fragment indices, and conditioning
      status, requested/achieved interface quadrature orders, and fallback
      status alongside fragment topology. Generated cuts stamp fallback status
      per cell before fragment/region insertion. Production tessellation and
      curvature recovery output remain open.
- [ ] Add optional tessellated visualization of curved patches with a documented
      visualization resolution independent from production quadrature.
- [ ] Implement FE projection or patch recovery for curvature on generated
      interfaces.
- [ ] Smooth recovered curvature with bounded local operators and deterministic
      support patches.
- [ ] Add curvature diagnostics for circles/spheres and ellipses/ellipsoids.
- [ ] Add capillary-wave or static-drop tests before enabling production
      high-order surface tension.
- [ ] Keep raw pointwise level-set curvature rejected for production surface
      tension unless a separate qualification plan approves it.
- [x] Add interface-field output for normals, curvature, local root residual,
      achieved quadrature order, and fallback status.
      VTP cell data now emits normals, curvature estimates, root residuals,
      requested/achieved interface quadrature orders, and numeric fallback
      status for fragment and stored quadrature-point cells.

## Phase 10: Assembly And Solver Integration

- [ ] Verify cut-volume assembly accepts arbitrary high-order quadrature point
      counts without assuming legacy element or face point counts.
- [ ] Verify cut-interface assembly accepts curved patch rules and remaps normals
      and weights through parent physical geometry.
- [ ] Verify JIT cache keys include marker, side, backend, point count, DOF
      counts, geometry mode, and baked-basis hash.
- [ ] Add JIT/interpreter parity tests for many-point cut-volume and
      cut-interface rules.
- [ ] Add pressure constraint tests for high-order pressure DOFs on dry sides.
- [ ] Add cut-adjacent stabilization tests for higher-order spaces and curved
      interface cuts.
- [ ] Add nonlinear line-search tests where high-order cut topology changes in a
      rejected trial and the accepted context is restored.
- [ ] Add solver tests that compare `RefreshedFrozenQuadrature` quasi-Newton and
      `DifferentiatedQuadrature` exact-tangent behavior once available.

## Phase 11: Performance And Memory Work

- [x] Add per-cell counters for root branches, root iterations, quadrature points,
      curved fragments, fallback leaves, and backend time.
- [ ] Add per-step summaries for cut-context rebuild count, total backend time,
      total quadrature points, active cut cells, memory growth, and basis-cache
      growth.
      Partial: generated-interface lifecycle results now aggregate total backend
      time and total backend volume/interface quadrature points; OOP
      active-domain rebuild/topology/accepted-step diagnostics now include
      backend time, backend quadrature point totals, active cut cells, active
      quadrature points, generated-domain interface/volume/total quadrature
      point totals, process memory, and basis-cache size. The OOP case-matrix
      runner now records peak RSS and basis-cache growth from the first observed
      rebuild and can fail runs on explicit growth thresholds. CI-enforced
      thresholds for the high-order MMS/open-vessel cases remain open.
- [ ] Avoid process-wide caching of transient generated quadrature rules unless
      the key includes all geometry and level-set state revisions.
- [ ] Keep long-lived caches limited to stable FE basis tabulations keyed by
      element family, order, derivative order, and reference point set.
- [ ] Add narrow-band or dirty-cell refresh support so unchanged far-field cells
      do not rebuild high-order cuts every nonlinear state.
- [ ] Add optional rule reuse for identical state revisions.
- [ ] Profile recursive branch discovery versus root-polished quadrature cost.
- [x] Add memory ceilings for high-order MMS and open-vessel smoke tests.
      Covered by the high-order smoke harness defaults requiring process-memory
      and basis-cache diagnostics with ceilings for peak RSS, RSS growth, cache
      entries, and cache-entry growth. The MMS and open-vessel qualification
      logs both passed those gates:
      `Documentation/qualification_logs/high_order_implicit_mms2d_smoke_20260519.json`
      and
      `Documentation/qualification_logs/high_order_open_vessel2d_smoke_20260519.json`.
- [ ] Add performance regression tests that fail on unbounded basis-cache growth.
      Partial: `tools/run_oop_case_matrix.py` can now fail runs with
      `--max-basis-cache-entries`, `--max-basis-cache-entry-growth`,
      `--max-rss-kb`, and `--max-rss-growth-kb` when cut-context diagnostics
      are present. It also supports explicit ratio gates against a prior
      `summary.json` for wall time, solver-loop time, final residual, total
      Newton iterations, total linear iterations, and implicit-cut backend
      time. Committed benchmark baselines and CI-enforced ceilings for the
      high-order MMS/open-vessel cases remain open.
- [ ] Record backend timings in benchmark qualification logs.
      Partial: `tools/run_oop_case_matrix.py` captures full solver stdout/stderr
      and records wall time, completed time-loop duration, accepted steps,
      nonlinear residuals, Newton iterations, and linear iterations in JSON and
      Markdown summaries. It also parses active cut-context rebuild diagnostics
      into dedicated JSON fields for implicit backend time, fallback cells,
      active cut cells, active quadrature points, RSS, and basis-cache counters,
      records RSS/cache growth from the first observed rebuild to the peak
      observed rebuild, adds Markdown cut-context and growth sections when
      those diagnostics are present, and can compare selected metrics against a
      prior qualification summary. CI baselines and trend comparison remain
      open.

## Phase 12: Validation Matrix

Geometry-only tests:

- [x] P1 linear equivalence for all supported cells.
      Covered by lifecycle affine-cut measure checks for Saye Quad4/8/9 and
      Hex8/20/27 and HighOrderSubcell Triangle3/6 and Tetra4/10, including
      serendipity Quad8/Hex20 scalar C0 DOF distribution.
- [x] P2/P3 level-set DOFs affect geometry and quadrature.
      Covered by P2 evaluator and lifecycle edge-DOF motion checks plus P3
      evaluator and generated-rule perturbation checks for Saye and simplex
      paths.
- [x] Circle/ellipse area and perimeter convergence in 2D.
      Covered by Saye circle h-/p-refinement and ellipse h-refinement checks,
      plus HighOrderSubcell triangle circle h-refinement checks, all against
      analytic area and perimeter/arc-length references.
- [x] Sphere/ellipsoid volume and surface-area convergence in 3D.
      Covered by Saye Hex27 sphere and prolate ellipsoid h-refinement checks
      plus a HighOrderSubcell Tetra10 spherical-cap h-refinement check, all
      against analytic volume and surface-area references.
- [x] Plane-cut exactness for affine level sets on high-order physical cells.
      Covered by exact affine-cut measure checks for all supported high-order
      cell variants plus `Quad9`, `Hex27`, and `Tetra10` interface point
      planarity/residual checks.
- [x] Curved physical geometry mapping tests with non-affine parent cells.
      Covered by the Hex27 midplane regression that maps generated reference
      rules through a non-affine quadratic physical mapping and checks physical
      volumes and interface area against independent references.
- [x] Small-volume and near-tangent robustness tests.
      Covered by small-volume Saye `Quad9` and HighOrderSubcell `Triangle6`
      finite-rule checks plus Saye near-tangent and HighOrderSubcell
      vertex/edge-touch diagnostics.
- [x] Multiple-component cell tests where one parent cell contains more than one
      interface component.
      Covered for separated boundary-intersecting components in Saye `Quad4`
      and HighOrderSubcell `Triangle3`. A same-sign-corner interior island is
      covered on the Saye `Quad9` path by
      `SmallVolumeCutsReturnFinitePositiveRules`. Remaining qualification is
      broad 3D multi-component and branching topology coverage.

Quadrature tests:

- [x] Exactness tests for polynomial integrands up to advertised order.
      Covered by 2D Saye `Quad9` and HighOrderSubcell `Triangle6` degree-5
      moment checks, plus 3D Saye `Hex27` and HighOrderSubcell `Tetra10`
      moment checks at their advertised 3D order caps.
- [x] h-refinement tests at fixed one-dimensional quadrature order.
      Covered by the circle/ellipse/sphere/ellipsoid convergence tests that
      hold requested interface and volume quadrature orders fixed while
      increasing subdivision depth.
- [x] q-refinement tests at fixed geometry.
      Covered by 2D Saye `Quad9` and HighOrderSubcell `Triangle6` affine-cut
      moment tests that hold geometry fixed while increasing requested
      interface and volume quadrature order from 1 to 5.
- [x] Positive-weight checks for all production rules.
      Covered by a representative Saye/HighOrderSubcell, 2D/3D generated-rule
      invariant sweep over exported volume and interface rules.
- [x] Parent-measure conservation checks for every volume split.
      Covered by the same representative generated-rule invariant sweep,
      plus the geometry-only regressions that check parent partitioning for
      affine, curved, small-volume, and multi-component cuts.
- [x] Interface normal orientation checks against `grad(phi_h)`.
      Covered for Saye quad/circle and HighOrderSubcell tetra/sphere generated
      curved interface quadrature points with unit-normal, gradient-norm, and
      positive near-parallel alignment checks.

Assembly tests:

- [x] Cut-volume matrix/vector assembly with high-order rules.
      Covered by high-order volume many-point basis/moment assembly checks,
      nonlinear residual+tangent finite-difference checks, and the
      CutDomainAssembler AD/symbolic residual+tangent parity regression.
- [x] Cut-interface surface assembly with curved rules.
      Covered by high-order many-point interface mass and normal-traction
      assembly checks plus nonlinear interface finite-difference tangent
      coverage.
- [x] JIT/interpreter parity with many-point rules.
      Covered by the high-order generated volume/interface many-point parity
      regression with LLVM JIT enabled.
- [x] Fixed-geometry finite-difference tangent checks.
      Covered for high-order cut-volume and cut-interface nonlinear forms under
      the current refreshed/frozen geometry tangent policy.
- [ ] Moving-geometry finite-difference tangent checks once differentiated
      quadrature is implemented.

Solver tests:

- [x] Compact high-order MMS traveling-interface case.
      Covered by the one-step compact MMS high-order implicit smoke at
      `Documentation/qualification_logs/high_order_implicit_mms2d_smoke_20260519.json`.
      The smoke is a correctness/diagnostic pass, not a production performance
      baseline; refreshed/frozen moving-geometry convergence still needs
      optimization and tighter gates.
- [x] Open-vessel high-order smoke case with recorded fallback count,
      wet-volume history, nonlinear iterations, linear iterations, pressure
      range, velocity range, RSS, and basis-cache counters.
      Covered by the one-step root open-vessel high-order implicit smoke at
      `Documentation/qualification_logs/high_order_open_vessel2d_smoke_20260519.json`.
      The log records zero fallback cells, stable wet-volume histories, 1
      nonlinear iteration, 2 linear iterations, pressure and velocity ranges,
      peak RSS, and basis-cache counters.
- [ ] MPI serial-parallel deterministic measure comparison.
      Partial: two-rank focused tests now compare deterministic generated-rule
      provenance for 2D Saye and high-order subcell curved paths. The mixed
      Quad4/Triangle3 `Auto` dispatch path now also compares generated-domain
      measures and deterministic interface/volume rule hashes against a
      rank-local serial reference on both MPI ranks. Current verification after
      the 3D full-region quadrature update: `cmake --build build-fe-check
      --target test_fe_levelset_mpi --parallel` passes, and `ctest --test-dir
      build-fe-check -R '^test_fe_levelset_mpi_mpi_2$' --output-on-failure`
      passes. A dedicated 3D serial-parallel deterministic fixture remains
      open.
- [x] Full OOP fluid case matrix for `Channel2D`, `Channel2D_Simple`,
      `vortex_shedding`, `iliac_artery`, `pipe_simple`, and `pipe_RCR_3d` in
      serial and MPI.
      The ranks 1 and 2 `solver_perf_oop.xml` matrix passes all twelve entries
      with logged residual, iteration, wall-time, and solver-loop summaries in
      `Documentation/qualification_logs/oop_case_matrix_full_rebuilt_20260518/summary.md`.
      Higher-rank and performance-baseline comparisons remain under benchmark
      requalification.
- [ ] Line-search topology restore regression.
- [ ] Benchmark requalification only after all analytic and smoke gates pass.

## Phase 13: Documentation And User-Facing Controls

- [x] Document backend selection: `SayeHyperrectangle`,
      `HighOrderSubcell`, `MomentFit`, and `Auto`.
      Documented in `Code/Source/solver/FE/Docs/LevelSet.md`.
- [x] Document each backend's supported element families and achieved orders.
      Documented in `Code/Source/solver/FE/Docs/LevelSet.md`.
- [x] Document fallback policy and default fail-closed behavior.
      Documented in `Code/Source/solver/FE/Docs/LevelSet.md`.
- [x] Document root tolerance, coordinate tolerance, and root-iteration controls.
- [x] Document subdivision depth as topology isolation, not an accuracy dial.
- [x] Document `RefreshedFrozenQuadrature` versus `DifferentiatedQuadrature`.
- [x] Document what diagnostics mean: root iterations, branch count,
      linearized leaves, fallback cells, achieved order, and curved patch count.
- [ ] Add examples for quad/hex Saye, tri/tet simplex, mixed-element auto mode,
      and curvature-output diagnostics.
      Partial: the FE level-set guide now includes mixed `Auto` and explicit
      `SayeHyperrectangle` fallback snippets. Dedicated simplex and curvature
      output examples remain open.
- [ ] Update validation case docs to state when high-order implicit geometry is
      production-qualified versus diagnostic.

## Definition Of Done

The work is complete when all of the following are true:

- [ ] `HighOrderImplicit` can generate curved interface and volume quadrature
      above interface order 1 and volume order 2 for supported high-order cells.
- [ ] Production high-order runs do not rely on linearized terminal leaves unless
      an explicit fallback policy is configured and counted.
- [ ] Root-polished branch reconstruction is implemented, diagnosed, and tested.
- [ ] `MomentFit` is either implemented and qualified or removed/hidden from
      production configuration.
- [ ] `DifferentiatedQuadrature` is implemented and passes geometry-sensitive
      tangent checks, or the code continues to clearly advertise only
      quasi-Newton support.
- [ ] Mixed supported meshes can be handled by automatic per-cell backend
      dispatch, with deterministic MPI behavior.
- [ ] Curved surface output and curvature recovery are available and validated
      before high-order surface tension is advertised.
- [ ] Geometry-only, quadrature, assembly, nonlinear, MPI, and benchmark smoke
      tests all pass with logged requested/achieved orders, fallback counts,
      root-iteration counts, memory counters, and backend timings.
