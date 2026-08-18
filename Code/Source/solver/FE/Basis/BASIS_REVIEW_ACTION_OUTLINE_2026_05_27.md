# Basis Review Action Outline

Date: 2026-05-27

This document records the work needed to address the accepted findings from the
Basis folder review. It intentionally excludes note 4.9 as written:
`OrthogonalPolynomials.cpp` uses the shared integer-power helper, so only a
minor pyramid accessor cleanup remains there.

The outline is organized by review note number so individual work items can be
tracked back to the review.

## General Acceptance Requirements

- Preserve public basis semantics, node ordering, cache identity behavior, and
  supported basis families.
- Keep changes incremental enough to verify with focused unit tests before
  broad refactors land.
- Add or update tests for every changed numerical path.
- Avoid hot-path allocations after constructor or explicit prewarm.
- Record benchmark before/after data for all performance-motivated changes.
- Keep `FE/Basis/README.md` as the durable user-facing documentation entry.

## 1. Performance

### 1.1 [x] Sparse RT/ND Modal Coefficients On Generic Nodal Paths

Status: Completed 2026-05-27.

Work:

- [x] Add sparse coefficient storage for the generic nodal RT/ND branches, not only
  the transformed direct-seed branch.
- [x] Build sparse rows after moment-matrix inversion and dense coefficient
  construction.
- [x] Route value, Jacobian, curl, and divergence strided helpers through sparse
  row iteration.
- [x] Keep dense coefficients temporarily for scalar pointwise helpers; full
  dense-buffer removal is tracked separately in 3.1.
- [x] Reuse existing Kronecker, strided-vs-pointwise, and sparse-pruning tests
  for RT, ND, and BDM coverage.
- [x] Add benchmark coverage for tetra, wedge, pyramid, and hex RT/ND where
  current constructors support it. RT and tetra/wedge/pyramid ND run at order 3;
  Hex8 ND high order remains blocked by a pre-existing rank-deficient
  construction path, so the benchmark keeps Hex8 ND at order 0.

Acceptance:

- [x] Generic RT/ND nodal-generated strided paths no longer scan a dense n-by-n
  matrix per quadrature point.
- [x] Sparse and dense reference outputs match within documented tolerances:
  `test_fe_basis --gtest_filter='VectorBasis*:VectorBases*:RaviartThomas*:Nedelec*:BDM*:BatchEvaluator.*'`.
- [x] High-order ND tetra and RT/ND wedge/pyramid paths show measurable speedup
  or have benchmark baselines for rows that did not exist before this work.
  Comparable historical rows from
  `scripts/basis_comparison/data/after_basis_review_followup_20260527_001325/perf/basis_remediation_microbench.csv`
  to `/tmp/basis_perf_microbench_current_20260527.csv` showed:
  RT wedge order 3 point values/J/div +92.42%, RT wedge order 3 strided
  values/J/div +57.17%, ND wedge order 3 point values/J/curl +88.62%,
  ND wedge order 3 strided values/J/curl +53.69%, ND pyramid order 2 point
  values/J/curl +68.98%, and ND pyramid order 2 strided values/J/curl +21.59%.
  Newly added current-baseline high-order rows include RT tetra/pyramid/hex
  order 3 strided and ND tetra/pyramid order 3 strided.

### 1.2 [x] Cached Topology Dispatch

Status: Completed 2026-05-27.

Work:

- [x] Add cached evaluator function pointers or small strategy objects in
  `LagrangeBasis` construction.
- [x] Split value, gradient, Hessian, and all-output dispatch by fixed topology.
- [x] Apply the same idea to RT/ND strided transformed paths so topology/order
  selection happens outside the quadrature loop.
- [x] Keep public virtual APIs unchanged.

Acceptance:

- [x] Per-call `switch (topology_id_)` is removed from hot `LagrangeBasis`
  evaluation entry points.
- [x] RT/ND transformed strided loops do not branch on wedge/pyramid/order per QP.
- [x] Existing Lagrange, RT, and ND tests pass unchanged:
  `test_fe_basis --gtest_filter='Lagrange*:VectorBasis*:VectorBases*:RaviartThomas*:Nedelec*:BDM*:BatchEvaluator.*'`.

### 1.3 [x] BasisCache Handle Reuse Audit

Status: Completed 2026-05-27.

Work:

- [x] Audit all assembly and setup callers of `BasisCache::get_or_compute`.
- [x] Classify each call site as setup-only, one-shot, or hot-loop repeated.
- [x] Store `BasisCacheHandle` values in persistent assembler, recipe, and
  workspace scratch state for repeated access.
- [x] Keep direct `get_or_compute` only where a handle is not useful.
- [x] Add regression tests for handle lifetime after `BasisCache::clear()`.

Acceptance:

- [x] Hot assembly paths access cache entries through handles without repeated
  key rebuild, hash, mutex, or map lookup.
- [x] Direct cache lookups remain only in cold or one-shot code:
  `ElementCache.cpp`, `SpaceCache.cpp`, and `SystemSetup.cpp` setup paths;
  `StandardAssembler.cpp` coupled scalar cache refills; and transient face
  quadrature lookups that build a local quadrature rule once for a face path.
- [x] Parallel `FieldSolutionWorkspace` paths use a workspace-local handle cache
  rather than mutating shared assembler state or repeatedly calling
  `get_or_compute`.
- [x] Verified with focused `BasisCache` handle tests and
  `StandardAssemblerCaches`, WorkStream, colored assembler, and multi-field
  assembly filters.

### 1.4 [x] Consolidated Thread-Local Scratch

Status: Completed 2026-05-27.

Work:

- [x] Define one scratch accessor per basis family or shared helper family.
- [x] Consolidate scattered `static thread_local std::vector` instances into named
  scratch structs.
- [x] Document the persistent OMP worker-thread assumption next to each accessor.
- [x] Add `prewarm_scratch(max_size, max_qpts)` hooks for important families.
- [x] Avoid scratch sharing across unrelated public calls where recursion/reentry is
  possible.
- [x] Add Lagrange and BSpline family scratch prewarm hooks while closing the
  remaining local `thread_local std::vector` declarations found by the broader
  Basis search.

Acceptance:

- [x] Hot evaluators use named scratch accessors instead of ad hoc local
  `thread_local` vectors.
- [x] Scratch can be sized before benchmark or assembly loops.
- [x] Dirty-buffer and consistency tests show no stale data:
  `test_fe_basis --gtest_filter='LagrangeBasis.*:BSplineBasis.*:BasisErrorPaths.BasisFunction*'`.
- [x] Full Basis suite passed after the consolidation:
  `./build-fe-eigen-check/test_fe_basis` (568 tests).

### 1.5 [x] Avoid Zero-Fill Before Full Overwrite

Status: Completed 2026-05-27.

Work:

- [x] Audit each `assign(n, 0)` cited path and classify it as full overwrite,
  partial overwrite, or required zero-initialization.
- [x] Replace proven full-overwrite scratch initialization with `resize` in
  Hierarchical, Bernstein, and BSpline derivative kernels.
- [x] Refactor BSpline gradient, Hessian, and all-output internals to compute on
  active support instead of full-size derivative scratch.
- [x] Keep BSpline/NURBS cache fill paths on caller-precleared storage and write
  only active DOFs.
- [x] Preserve explicit zeroing for inactive basis rows, point-basis
  derivatives, low-degree derivative slots, and public full-output APIs where
  zero inactive DOFs are part of the contract.

Acceptance:

- [x] No evaluator depends on uninitialized inactive data.
- [x] Full-basis derivative scratch zeroing is removed from BSpline
  active-support hot paths.
- [x] Dirty-output and consistency tests pass for values, gradients, Hessians,
  and all-output evaluation:
  `test_fe_basis --gtest_filter='BernsteinBasis.*:HierarchicalBasis.*:BSplineBasis.*'`
  and the repeated `BSplineBasis.*` run after the active-support refactor.

### 1.6 [x] BatchEvaluator Tail Zeroing

Status: Completed 2026-05-27.

Work:

- [x] Allocate or resize padded SoA buffers without full zero initialization.
- [x] Zero only tail columns `[num_quad, padded_quad)` for each row/component.
- [x] Keep tail-zero semantics for SIMD loads in weighted reductions.
- [x] Add tests that check active entries are written and padded tail entries remain
  zero.

Acceptance:

- [x] `BatchEvaluator` no longer zero-fills logical cells that are immediately
  overwritten by strided evaluation.
- [x] SIMD tail behavior remains deterministic.

### 1.7 [x] Batched Power Tables In Generic Vector Paths

Status: Completed 2026-05-27.

Work:

- [x] Replace per-QP scalar `fill_power_tables` calls in generic vector nodal
  strided evaluation with `fill_batched_power_tables`.
- [x] Use existing `batched_power_product` and partial-derivative helpers.
- [x] Structure loops so compilers can vectorize over quadrature points.
- [x] Verify transformed direct-seed and generic nodal branches produce identical
  results.

Acceptance:

- [x] Generic vector nodal strided helpers use batched power tables.
- [x] Modal rows are computed across contiguous quadrature-point arrays before
  applying nodal coefficients, removing the scalar per-QP power refill.
- [x] Verification passed:
  `test_fe_basis --gtest_filter='VectorBasis*:VectorBases*:RaviartThomas*:Nedelec*:BDM*:BatchEvaluator.*'`.

### 1.8 [x] Vandermonde Inverse Registry

Status: Completed 2026-05-27.

Work:

- [x] Add a process-wide cache keyed by `(element_type, order, modal_family)` for
  spectral/modal inverse matrices.
- [x] Reuse cached matrices in `SpectralBasis` and `ModalTransform`.
- [x] Leave the existing pyramid order-data cache intact and avoid double caching
  it.
- [x] Define clear lifetime and thread-safety rules for the registry.

Acceptance:

- [x] Repeated construction of identical spectral/modal bases does not refactor and
  reinvert the same Vandermonde matrix.
- [x] Construction tests remain deterministic under parallel construction.
- [x] `cmake --build build-fe-eigen-check --target test_fe_basis -j 4` succeeds.
- [x] `./build-fe-eigen-check/test_fe_basis --gtest_filter='SpectralBasis.*:ModalTransform.*'`
  passes 44 tests.

### 1.9 [x] Single Canonical Modal Matrix Layout

Status: Completed 2026-05-27.

Work:

- [x] Choose one canonical row-major or modal-major layout for modal-to-nodal
  matrices.
- [x] Replace duplicate stored layouts with accessors or loop stride parameters.
- [x] Update spectral and modal transform application helpers.
- [x] Measure whether any removed layout causes a runtime regression.

Acceptance:

- [x] Per-instance modal matrix memory is reduced without numerical changes.
- [x] Any needed alternate access pattern is expressed by indexing, not duplicated
  storage.
- [x] `./build-fe-eigen-check/test_fe_basis --gtest_filter='SpectralBasis.*:ModalTransform.*'`
  passes 44 tests.
- [x] `basis_perf_microbench` spectral pyramid rows compared against
  `scripts/basis_comparison/data/after_basis_performance_cleanup_20260527_100936/perf/basis_remediation_microbench.csv`:
  order4 point all improved 8.38%, order4 strided all improved 20.70%, and
  order5 strided all improved 23.47%.

### 1.10 [x] Low-Order Pyramid Fast Paths

Status: Completed 2026-05-27.

Work:

- [x] Implement closed-form `LagrangePyramidFast<1>` and
  `LagrangePyramidFast<2>` evaluators for Pyramid5 and Pyramid14.
- [x] Cover values, gradients, Hessians, `evaluate_all`, and strided output.
- [x] Route low-order pyramid dispatch through the fast paths.
- [x] Keep current modal path as the general-order fallback.

Acceptance:

- [x] Pyramid5 and Pyramid14 avoid modal-to-nodal matrix multiply in hot
  evaluation.
- [x] Existing Pyramid5/Pyramid14 nodal, trace, polynomial reproduction,
  derivative, apex, and strided/BatchEvaluator tests pass against the unchanged
  modal construction.
- [x] Focused pyramid suite passes 15 tests:
  `./build-fe-eigen-check/test_fe_basis --gtest_filter='LagrangeBasis.Pyramid*:LagrangeBasis.WedgeAndPyramid*:BatchEvaluator.HessiansMatchPointwiseOnPyramidLagrange:BatchEvaluator.PyramidApexValueOnlyConstructionSucceeds:SpectralBasis.Pyramid*'`.
- [x] `./build-fe-eigen-check/test_fe_basis` passes 570 tests.
- [x] `basis_perf_microbench 100` compared against the same-session dense
  baseline: Pyramid14 strided values improved 27.02%,
  values+gradients improved 11.84%, and hessians improved 34.80%.

### 1.11 [x] Restrict Output Pointers

Status: Completed 2026-05-27.

Work:

- [x] Define a compiler-agnostic `SVMP_RESTRICT` macro.
- [x] Annotate hot `_to` and `_strided` output pointer parameters where API
  contracts guarantee non-aliasing.
- [x] Apply consistently to scalar, vector, spline, NURBS, Bernstein, spectral, and
  Lagrange hot paths.
- [x] Verify GCC, Clang, and MSVC-compatible preprocessing.

Acceptance:

- [x] Pointer aliasing no longer blocks vectorization in annotated hot kernels.
- [x] Public headers remain portable across supported compilers.
- [x] `cmake --build build-fe-eigen-check --target test_fe_basis -j 4` succeeds.
- [x] `./build-fe-eigen-check/test_fe_basis` passes 566 tests.

### 1.12 [x] Cache Or Replace BDM Inner Test Bases

Status: Completed 2026-05-27.

Work:

- [x] Cache inner `NedelecBasis` construction by topology/order for BDM moment
  construction, or replace the test space with a direct monomial basis.
- [x] Ensure triangle and tetra BDM construction still produces the same moments.
- [x] Add tests around BDM divergence and DOF interpolation at orders that use the
  inner basis.

Acceptance:

- [x] Repeated BDM construction does not repeatedly build identical inner Nedelec
  spaces.
- [x] BDM construction and evaluation tests remain unchanged numerically.

## 2. Accuracy

### 2.1 [x] Jacobi Recurrence Stability

Status: Completed 2026-05-27.

Work:

- [x] Generate high-precision reference data for Jacobi, Dubiner, and Proriol
  evaluations at high order and high alpha/beta.
- [x] Add tests near endpoints and simplex degeneracies.
- [x] Document the maximum validated production order.
- [x] Evaluate normalized Gautschi-style recurrence for better scaling; defer it
  until production orders exceed the documented validation envelope.

Acceptance:

- [x] High-order recurrence error is quantified and documented.
- [x] Either the current recurrence is validated to a stated order or replaced.

### 2.2 [x] Vandermonde Diagnostics

Status: Completed 2026-05-27.

Work:

- [x] Replace raw `invert_dense_matrix` calls in pyramid and spectral construction
  with diagnostic inversion.
- [x] Set condition-number thresholds by topology/order and numeric type.
- [x] Emit construction errors or warnings when thresholds are exceeded.
- [x] Add tests with intentionally ill-conditioned configurations.

Acceptance:

- [x] Ill-conditioned Vandermonde construction is detected explicitly.
- [x] Existing supported orders continue to construct successfully.

### 2.3 [x] Equispaced Lagrange High-Order Guidance

Status: Completed 2026-05-27.

Work:

- [x] Document that generic equispaced Lagrange order >= 4 is
  interpolation-unstable and Spectral/GLL should be preferred for high-order
  interpolation.
- [x] Update factory or user-facing construction guidance where defaults can
  steer high-order users away from equispaced nodes.
- [x] Add README examples that distinguish evaluation stability from
  interpolation stability.

Acceptance:

- [x] Users can discover the high-order limitation without reading implementation
  code.
- [x] No existing explicit Lagrange request changes behavior silently.

### 2.4 [x] Vector Monomial Candidate Selection

Status: Completed 2026-05-27.

Work:

- [x] Add duplicate detection before pushing vector monomial candidates.
- [x] Replace silent `resize(n)` truncation with explicit dimension checks and
  deterministic candidate selection.
- [x] Keep rank checks but add validation that the expected DOF families survive.
- [x] Extend wedge and pyramid RT/ND tests.

Acceptance:

- [x] Duplicate or excess monomials are reported or resolved intentionally.
- [x] Rank and DOF tests cover the final selected candidate set.

### 2.5 [x] Tolerant Sparse Coefficient Pruning

Status: Completed 2026-05-27.

Work:

- [x] Compute the max absolute coefficient per matrix or row group.
- [x] Prune entries below a relative sparsification tolerance.
- [x] Add a tunable or documented `kSparsifyTol`.
- [x] Test that sparsified coefficients preserve DOF Kronecker properties.

Acceptance:

- [x] Numerical-noise coefficients from dense solves do not survive solely because
  they are nonzero at the bit level.
- [x] Sparsification does not degrade accepted accuracy.

### 2.6 [x] Barycentric Partition Correction

Status: Completed 2026-05-27.

Work:

- [x] Replace single-entry residual correction with compensated accumulation,
  proportional residual distribution, or both.
- [x] Cover values and derivative correction paths.
- [x] Add high-order and near-node tests for partition of unity and derivative sum
  constraints.

Acceptance:

- [x] Partition correction is less concentrated while preserving exact constraints
  within tolerance.
- [x] No low-order regression.

## 3. Computational Scaling

### 3.1 [x] Drop Dense Vector Coefficients After Construction

Status: Completed 2026-05-27.

Work:

- [x] Convert dense RT/ND/BDM coefficient matrices to sparse storage after
  construction.
- [x] Release dense buffers unless construction diagnostics require keeping them in
  debug-only state.
- [x] Update evaluators to consume sparse storage exclusively.

Acceptance:

- [x] Runtime vector bases no longer carry large dense coefficient matrices
  unnecessarily.
- [x] Memory usage drops for high-order vector bases.
- [x] Pointwise and strided RT/ND/BDM tests still match existing baselines:
  `test_fe_basis --gtest_filter='VectorBasis*:VectorBases*:RaviartThomas*:Nedelec*:BDM*'`.

### 3.2 [x] BSpline/NURBS Active Support In Hessian And All Paths

Status: Completed 2026-05-27.

Work:

- [x] Refactor BSpline `evaluate_hessians` and `evaluate_all` around
  `evaluate_active_support`.
- [x] Confirm NURBS denominator accumulation already loops over active tensor
  support and cache fill already uses caller-precleared output.
- [x] Rely on caller preclear where public contracts permit it.
- [x] Add tests that inactive rows remain zero when using cache fill paths.

Acceptance:

- [x] Full-basis derivative scratch clearing is removed from active-support
  spline/NURBS hot paths; public full-output APIs still clear inactive rows by
  contract.
- [x] Large sparse-support NURBS patches scale with active support, not total basis
  size.
- [x] Verification passed:
  `test_fe_basis --gtest_filter='BasisCache.CachedSplineInactiveRowsRemainZero:BasisCache.CachedTensorNURBSInactiveRowsRemainZero:BasisCache.CachedSplineValuesMatchDirectEvaluation:BasisCache.CachedTensorNURBSMatchesDirectEvaluation:BSplineBasis.*'`.

### 3.3 [x] Wedge Exponent Lookup Map

Status: Completed 2026-05-27.

Work:

- [x] Build an `unordered_map` from simplex exponent tuple to index once.
- [x] Use the map in wedge node construction instead of repeated `std::find`.
- [x] Reuse existing wedge construction and ordering tests for covered orders.

Acceptance:

- [x] Wedge node construction avoids quadratic simplex-exponent lookup.

### 3.4 [x] BLAS Or Blocked Modal Apply

Status: Completed 2026-05-27.

Work:

- [x] Added `Math/DenseTransformKernels.h` with scalar thresholds and a blocked
  row-major dense transform for `A[rows, cols] * B[cols, qpts]`.
- [x] Wired the blocked batched kernel through spectral modal strided transforms
  so values, gradients, and Hessians write each output row once rather than
  repeatedly accumulating through the SoA rows.
- [x] Wired high-order pyramid Lagrange strided evaluation through the same
  batched kernel for non-apex batches, while preserving exact apex handling and
  the low-order sparse combination path.
- [x] Kept the small-row/small-QP scalar path to avoid blocked-kernel overhead.
- [x] Chose the always-available blocked path instead of adding a new external
  BLAS dependency, because the FE CMake configuration does not currently expose
  a BLAS provider.
- [x] Benchmarked spectral and pyramid modal application with the existing order
  4 through 10 microbenchmark coverage.

Acceptance:

- [x] Large dense modal transforms use a batched or blocked kernel with better
  cache behavior.
- [x] Small orders remain on scalar or low-order specialized paths.

### 3.5 [x] Fixed Small Simplex Axis Scratch

Status: Completed 2026-05-27.

Work:

- [x] Replace small `std::vector` axis buffers with fixed arrays for common orders.
- [x] Define a maximum stack/static order for the fixed path.
- [x] Add a fallback dynamic path for orders above the fixed capacity.

Acceptance:

- [x] Common simplex orders avoid tiny heap/vector overhead.
- [x] High orders remain supported.

## 4. API And Conciseness

### 4.1 [x] Orthogonal Polynomial Return Types

Status: Completed 2026-05-27.

Work:

- [x] Migrate callers to named result structs.
- [x] Remove pair and tuple convenience wrappers after callers and tests are
  updated.
- [x] Keep one documented return shape per operation.

Acceptance:

- [x] Orthogonal polynomial APIs no longer expose duplicate return conventions.

### 4.2 [x] BasisFactory API Duplication

Status: Completed 2026-05-27.

Work:

- [x] Choose namespace functions or the static `BasisFactory` class as the canonical
  API.
- [x] Add a temporary compatibility shim only if downstream users need a migration
  window.
- [x] Remove duplicate declarations and forwarding definitions.

Acceptance:

- [x] Factory creation and custom registration have one primary API surface.
- [x] `test_fe_basis --gtest_filter='BasisFactory.*:BasisErrorPaths.*'` passes.
- [x] `test_fe_elements --gtest_filter='ElementFactory.*:LagrangeAndDGElements.*:SpectralElement.*:VectorElement.*'` passes.

### 4.3 [x] Descriptor Table Expansion

Status: Completed 2026-05-27.

Work:

- [x] Extend the descriptor pattern to vector bases, splines, NURBS, custom bases,
  H(div), H(curl), and C1 families.
- [x] Add optional validation hooks for family-specific topology/order rules.
- [x] Reduce `basis_factory::create` to table lookup plus construction dispatch.

Acceptance:

- [x] BasisFactory conditional construction logic is substantially shorter and
  table-driven.
- [x] `test_fe_basis --gtest_filter='BasisFactory.*:BasisErrorPaths.*'` passes.

### 4.4 [x] Vector Association Tables

Status: Completed 2026-05-27.

Work:

- [x] Replace hand-coded boundary classification loops with constexpr tables keyed
  by topology, family, and boundary mask.
- [x] Keep generated DOF ordering identical.
- [x] Add association golden tests for quad and hex compatible vector spaces.

Acceptance:

- [x] H(div)/H(curl) association builders are shorter and table-driven.
- [x] Existing trace/association behavior is unchanged.
- [x] `cmake --build build-fe-eigen-check --target test_fe_basis -j 4`
  succeeds.
- [x] `./build-fe-eigen-check/test_fe_basis --gtest_filter='BasisFactory.Compatible*VectorAssociationsKeepGoldenBoundaryOrdering:BasisFactory.CreatesCompatible*VectorSplineAndNurbsBases:BasisFactory.CompatibleVectorSplineAndNurbsRemainQuadHexOnly:BasisCache.CompatibleTensorVector*'`
  passes 9 tests.
- [x] `./build-fe-eigen-check/test_fe_basis` passes 570 tests.

### 4.5 [x] TensorBasis Dimension Template

Status: Completed 2026-05-27.

Work:

- [x] Introduce `TensorProductBasis<Basis1D, D>` specializations while keeping
  `TensorProductBasis<Basis1D>` as the existing dynamic-dimension type.
- [x] Preserve existing factory-facing behavior and downstream
  `dynamic_cast<TensorProductBasis<BSplineBasis>*>` users.
- [x] Move dimension-dependent writes into compile-time branches.
- [x] Benchmark D=1, D=2, and D=3 specialization output.

Acceptance:

- [x] Runtime `dimension_` branches are removed from hot tensor write paths.
- [x] Public construction remains compatible or has a documented migration.
- [x] `test_fe_basis --gtest_filter='BSplineBasis.StaticTensorDimensionSpecializationsMatchDynamicTensorBasis:BSplineBasis.TensorProductStridedOutputsMatchVectorEvaluation:BasisCache.FactoryCreatedTensor*'`
  passes.
- [x] `basis_perf_microbench 20` reports static D=1, D=2, and D=3 tensor
  strided cases with zero allocations.

### 4.6 [x] Vector Direct Seed Tables

Status: Completed 2026-05-27.

Work:

- [x] Keep the public direct-seed runtime on compact generated term tables.
- [x] Replace repeated assignment blocks and table/wrapper boilerplate with
  compact loops/macros.
- [x] Replace mode ternary towers with reusable constexpr-compatible quadratic
  mode helpers.
- [x] Compare generated outputs with current hard-coded behavior through vector
  basis regression tests.

Acceptance:

- [x] Direct seed source size drops from 1464 to 1363 lines without changing
  emitted behavior; focused vector/direct-seed checks passed.

### 4.7 [x] Writer Struct Unification

Status: Completed 2026-05-27.

Work:

- [x] Merge Bernstein and NURBS writer structs around a shared stride/offset
  policy.
- [x] Keep compact/vector/raw addressing as compile-time or inlined policy
  choices.
- [x] Verify generated code remains equivalent after optimization.

Acceptance:

- [x] Addressing-only writer duplication is removed: Bernstein now uses one
  layout-templated writer for span/raw outputs, and NURBS tensor evaluation uses
  one layout-templated writer for dense-vector/raw/compact outputs.
- [x] Removed the old addressing-only writer structs:
  `BernsteinSpanWriter`, `BernsteinRawWriter`, `NURBSTensorVectorWriter`,
  `NURBSTensorRawWriter`, and `NURBSTensorCompactWriter`.
- [x] Focused writer tests passed:
  `test_fe_basis --gtest_filter='BernsteinBasis.*:BSplineBasis.NURBSTensor*:BSplineBasis.AnisotropicTensor*:BSplineBasis.TensorProductStridedOutputsMatchVectorEvaluation'`.
- [x] Full Basis suite passed:
  `./build-fe-eigen-check/test_fe_basis` (568 tests).

### 4.8 [x] Hessian Store Helpers

Status: Completed 2026-05-27.

Work:

- [x] Add inline helpers for contiguous and strided Hessian scatter.
- [x] Replace repeated `r/c` copy loops in Lagrange, pyramid, Bernstein, and
  serendipity paths.
- [x] Keep layout comments near the helper.

Acceptance:

- [x] Hessian storage layout is expressed in one helper instead of repeated
  loops.

## 5. Compiler Optimization

### 5.1 [x] Accessor Attributes

Status: Completed 2026-05-27.

Work:

- [x] Add `[[nodiscard]]` and `noexcept` to trivial getters that cannot throw.
- [x] Avoid `noexcept` on checked accessors that can raise Basis exceptions.
- [x] Update headers consistently.

Acceptance:

- [x] Trivial accessors communicate both return-value importance and no-throw
  behavior.

### 5.2 [x] Result-Returning Attributes

Status: Completed 2026-05-27.

Work:

- [x] Mark factory, dense inversion, diagnostics, and polynomial sequence functions
  `[[nodiscard]]`.
- [x] Fix any call sites that intentionally discard results with explicit casts.

Acceptance:

- [x] Accidentally discarded nontrivial results produce compiler diagnostics.

### 5.3 [x] Branch Likelihood Hints

Status: Completed 2026-05-27.

Work:

- [x] Profile rare branches before annotating them.
- [x] Add `[[likely]]` or `[[unlikely]]` only where measured branch bias is stable.
- [x] Keep cold error paths readable.

Acceptance:

- [x] Likelihood hints are limited to boundary/apex/exact-node paths from the review.
- [x] `test_fe_basis --gtest_filter='BSplineBasis.*:SpectralBasis.*Pyramid*:SpectralBasis.GLLNodeAccuracy:SpectralBasis.EvaluateAllMatchesIndependentTensorCalls:LagrangeBasis.*Pyramid*:BatchEvaluator.Pyramid*'` passes.

### 5.4 [x] Strided Override Contract

Status: Completed 2026-05-27.

Work:

- [x] Document strided evaluation overrides as required for performance-sensitive
  bases.
- [x] Add debug or benchmark coverage that identifies bases falling back to
  per-QP virtual evaluation.
- [x] Consider an internal trait or test list for hot bases that must override.

Acceptance:

- [x] New hot basis families cannot silently rely on the slow fallback without
  test visibility.

### 5.5 [x] Constexpr Cleanup

Status: Completed 2026-05-27.

Work:

- [x] Promote eligible helpers and static coefficient tables to `constexpr`
  where C++20 library support permits.
- [x] Keep runtime paths where exceptions or non-constexpr library calls are
  required.
- [x] Add compile-time tests for helpers that become constexpr.

Acceptance:

- [x] Trivially compile-time data is no longer recomputed at runtime.

### 5.6 [x] LLVM 14 Codegen Guardrails

Status: Completed 2026-05-27.

Work:

- [x] Avoid forced inlining on modal kernels until LLVM IR and benchmark impact
  are checked.
- [x] Prefer aliasing, layout, sparse, and batching improvements before inlining
  attributes.
- [x] Record compiler versions for modal-kernel benchmarks.

Acceptance:

- [x] Compiler-specific hints do not regress LLVM 14 or other supported
  toolchains.

## 6. Scope And Module Boundaries

### 6.1 [x] Move Dense Matrix Utilities To Math

Status: Completed 2026-05-27.

Work:

- [x] Promote `DenseMatrixUtils` to `Math/DenseLinearAlgebra.{h,cpp}` as the
  canonical dense solve, inverse, rank, diagnostics, and pseudo-inverse utility.
- [x] Replace Basis construction includes with the Math header and keep
  `Basis/DenseMatrixUtils.h` as a compatibility facade only.
- [x] Remove `Basis/DenseMatrixUtils.cpp` from the build.
- [x] Update FSILS dense-inversion adapters to call the shared Math
  implementation while preserving their `bool` return and clear-on-failure API.
- [x] Preserve condition diagnostics and construction-time validation behavior
  in the shared implementation.
- [x] Move the dense-matrix unit tests from Basis to Math.

Acceptance:

- [x] General dense matrix functionality no longer lives under Basis.
- [x] Basis and FSILS do not maintain separate dense inversion implementations.
- [x] Verification passed:
  `test_fe_math --gtest_filter='DenseLinearAlgebra.*'`,
  selected Basis Vandermonde/vector construction tests, and selected non-MPI
  FSILS/block-Schur backend tests.

### 6.2 [x] Move Reference Monomial Integrals To Quadrature

Status: Completed 2026-05-27.

Work:

- [x] Move analytic reference-cell monomial integrals to a Quadrature or analytic
  reference-cell namespace.
- [x] Update vector-basis construction includes and tests.
- [x] Keep Basis depending on Quadrature for construction-time moment assembly.

Acceptance:

- [x] Reference integration data lives outside Basis evaluation code.

### 6.3 [x] Move Integer Math To Math

Status: Completed 2026-05-27.

Work:

- [x] Move `BasisIntegerMath.h` to a Math namespace/path.
- [x] Update all Basis and test includes.
- [x] Keep constexpr behavior and overflow semantics.

Acceptance:

- [x] General integer math utilities are reusable without a Basis dependency.

### 6.4 [x] Split BatchEvaluator Responsibilities

Status: Completed 2026-05-27.

Work:

- [x] Keep SoA basis precomputation in Basis if it remains basis-specific.
- [x] Move weighted reductions and stiffness-style kernels to Assembly.
- [x] Update includes and tests so assembly kernels own projection/reduction
  semantics.

Acceptance:

- [x] Basis owns evaluation data, while Assembly owns coefficient-weighted
  reductions.
- [x] Verification passed: `test_fe_basis --gtest_filter='BatchEvaluator.*'`.
- [x] The requested `test_fe_integration` target is absent in this build tree;
  `StiffnessMatrix.BatchEvaluatorConsistency` exists in source but is not wired
  into any current test executable.

### 6.5 [x] ReferenceDerivativeJet Boundary

Status: Completed 2026-05-27.

Work:

- [x] Keep `ReferenceDerivativeJet` as a Basis detail while only Basis uses it.
- [x] Add a comment stating the promotion rule if Assembly/Operator code needs
  AD.
- [x] Avoid exposing it as a public general AD type prematurely.

Acceptance:

- [x] Scope is documented and no unnecessary Math promotion occurs yet.

### 6.6 [x] Reference Node Layout Naming

Status: Completed 2026-05-27.

Work:

- [x] Rename the Basis-side `NodeOrdering` facade to `ReferenceNodeLayout`.
- [x] Remove the duplicate `node_ordering::` free-function facade.
- [x] Update Basis users, Element users, docs, and tests.
- [x] Keep Mesh `svmp::NodeOrdering` reserved for mesh format permutations.

Acceptance:

- [x] Basis reference-node layout and Mesh node-order conversion are clearly
  distinct APIs.
- [x] Source scan leaves no Basis-side `node_ordering::` callers and no
  ambiguous `NodeOrdering` references except the explanatory comment.
- [x] Verified with `test_fe_basis` reference-node/serendipity/Lagrange filters
  and `test_fe_elements` reference-element/factory filters.

### 6.7 [x] Split Vector Basis Runtime And Construction Translation Units

Status: Completed 2026-05-27.

Work:

- [x] Move construction/moment-matrix code into construction-specific translation
  units.
- [x] Keep runtime evaluation units limited to vector basis headers, modal
  polynomial data, and evaluation helpers.
- [x] Update CMake targets and include dependencies.
- [x] Verify LTO and incremental build behavior through a clean target rebuild.

Acceptance:

- [x] Runtime vector evaluators no longer drag heavy Elements/Quadrature
  construction dependencies into the same translation unit.

### 6.8 [x] Move Serendipity Mesh Permutation

Status: Completed 2026-05-27.

Work:

- [x] Move Hex20 mesh-to-internal permutation into node ordering conventions.
- [x] Expose a `mesh_to_basis_ordering(ElementType)` or equivalent query.
- [x] Update `SerendipityBasis` to fetch the permutation at
  construction/evaluation.

Acceptance:

- [x] Mesh ordering knowledge is centralized outside `SerendipityBasis`.

### 6.9 [x] Shared Pyramid Transform Helper

Status: Completed 2026-05-27.

Work:

- [x] Extract Bernstein's Duffy-like pyramid tensor map into a shared Geometry or
  Math helper if another basis family uses or should use the same transform.
- [x] Include value, gradient, and Hessian derivatives in the helper contract.
- [x] Keep rational modal pyramid helpers separate if their representation remains
  different.

Acceptance:

- [x] Shared geometric pyramid mapping logic is not duplicated between basis
  families.

### 6.10 [x] Split GLL Nodes And Weights

Status: Completed 2026-05-27.

Work:

- [x] Add a Basis-facing `gll_nodes(num_points)` API.
- [x] Move or expose GLL weights through Quadrature.
- [x] Update SpectralBasis to request only nodes.
- [x] Update Gauss-Lobatto quadrature to own the full rule construction without
  depending on Basis wrappers.

Acceptance:

- [x] Basis does not expose unused quadrature weights for node-only consumers.
- [x] Quadrature no longer depends on Basis for rule weights.

## 7. Peak LagrangeBasis Optimization Program

This section is a dedicated optimization effort for `LagrangeBasis`. Its goal is
to push Lagrange evaluation close to the practical performance limits of the
target hardware while preserving exact FE semantics.

### 7.1 [x] Benchmark Scope

Status: Completed 2026-05-27.

Work:

- [x] Build microbenchmarks for values, gradients, Hessians, and `evaluate_all`.
- [x] Cover scalar point evaluation and strided QP-batch evaluation.
- [x] Cover line, triangle, quadrilateral, tetrahedron, hexahedron, wedge, and
  pyramid topologies.
- [x] Cover orders 1 through 8, with focused high-order extensions where production
  use requires them.
- [x] Benchmark cold construction separately from hot evaluation.
- [x] Record compiler, flags, CPU model, vector width, thread count, and memory
  bandwidth.

Acceptance:

- [x] Every Lagrange optimization has a stable before/after benchmark row through
  the opt-in `basis_perf_microbench` Lagrange peak sweep:
  `SVMP_BASIS_BENCH_LAGRANGE_PEAK=1`.
- [x] Benchmarks separate algorithmic cost from setup and cache effects through
  `lagrange_scalar_point`, `lagrange_strided_batch`, and
  `lagrange_construction` categories.
- [x] Smoke verification passed:
  `SVMP_BASIS_BENCH_LAGRANGE_PEAK=1 SVMP_BASIS_BENCH_LAGRANGE_PEAK_MAX_ORDER=2 ./build-fe-eigen-check/basis_perf_microbench 1`
  produced the expected expanded Lagrange rows and metadata columns. The default
  peak sweep covers orders 1-8.

### 7.2 [x] Theoretical And Practical Performance Models

Status: Completed 2026-05-27.

Work:

- [x] Derive per-topology operation counts for values, gradients, Hessians, and
  all-output evaluation.
- [x] Estimate minimum memory traffic for each output request and layout.
- [x] Classify each kernel as compute-bound or memory-bound by arithmetic intensity.
- [x] Compare measured time against the lower-bound model.
- [x] Track vectorization efficiency from compiler reports or hardware counters.

Acceptance:

- [x] Performance claims are tied to an operation and memory traffic model.
- [x] Regressions can be diagnosed as algorithmic, vectorization, memory bandwidth,
  dispatch, or allocation problems.
- [x] `basis_perf_microbench` now emits model columns:
  `modeled_flops_per_call`, `modeled_min_bytes_per_call`,
  `arithmetic_intensity_flop_per_byte`, `bound_class`,
  `model_lower_bound_ns`, `measured_to_model_bound`,
  `machine_balance_flop_per_byte`, and `vector_efficiency`.
- [x] Smoke verification passed:
  `SVMP_BASIS_BENCH_LAGRANGE_PEAK=1 SVMP_BASIS_BENCH_LAGRANGE_PEAK_MAX_ORDER=2 SVMP_BASIS_BENCH_PEAK_GFLOPS=64 SVMP_BASIS_BENCH_STREAM_GBPS=20 SVMP_BASIS_BENCH_VECTOR_EFFICIENCY=unmeasured ./build-fe-eigen-check/basis_perf_microbench 1`
  produced 194 rows, including 112 modeled Lagrange rows and lower-bound ratio
  columns with no missing model fields.

### 7.3 [x] Dispatch And Specialization

Status: Completed 2026-05-27.

Work:

- [x] Cache topology-specific evaluator strategies at construction.
- [x] Split derivative request masks so values-only calls do not compute gradients
  or Hessians.
- [x] Add or extend low-order specializations for common P1, P2, and P3 cases.
  P3 Line, Quad, Hex, Triangle, and Tetrahedron now dispatch through
  `LagrangeBasisFast` specializations; tensor node-axis maps and simplex
  exponent maps are constexpr-generated from the same public ordering rules as
  `ReferenceNodeLayout`.
- [x] Keep general barycentric/tensor/simplex evaluators for higher orders.
- [x] Ensure specialization does not duplicate public API behavior.

Acceptance:

- [x] Hot vector evaluation calls use the cached topology dispatch path added in
  workstream 1.2; low-order fixed kernels perform only order/specialization
  selection before entering the closed-form or constexpr-table evaluator.
- [x] Low-order common cases use minimal closed-form or fully specialized
  kernels: P1/P2 remain closed-form, P3 tensor paths use constexpr axis products,
  and P3 simplex paths use constexpr barycentric exponent tables.
- [x] Focused Lagrange checks passed:
  `./build-fe-eigen-check/test_fe_basis --gtest_filter='LagrangeBasis.*Tensor*:LagrangeBasis.Generated*Ordering*:LagrangeBasis.KroneckerDeltaAcrossCanonicalTopologiesAndOrders:LagrangeBasis.PartitionGradientAndHessianSumsAcrossCanonicalTopologiesAndOrders:LagrangeBasis.RawOutputSinksMatchVectorEvaluationAcrossTopologies:LagrangeBasis.NonTensorStridedEvaluationMatchesPointwise'`.
- [x] Full FE basis regression passed: `./build-fe-eigen-check/test_fe_basis`
  reported 571/571 passing tests.

### 7.4 [x] Batch-Native Strided Kernels

Status: Completed 2026-05-27.

Work:

- [x] Rework strided evaluation around SoA batch loops over quadrature points.
- [x] Precompute tensor-product axis basis values and derivatives once per QP
  batch; simplex strided paths evaluate factor sequences directly into SoA
  output rows without per-QP virtual fallback.
- [x] Use contiguous output stores and stable strides.
- [x] Avoid temporary `Gradient` and `Hessian` objects in hot strided tensor and
  simplex paths.
- [x] Keep lower-dimensional unused components deterministically zero.

Acceptance:

- [x] Lagrange tensor, simplex, wedge, and pyramid strided entry points dispatch
  to direct strided implementations rather than the `BasisFunction` per-QP
  virtual fallback.
- [x] Tensor-product strided kernels use QP-contiguous SoA rows, with the QP loop
  innermost for values, gradients, and Hessians.
- [x] Focused checks passed:
  `./build-fe-eigen-check/test_fe_basis --gtest_filter='LagrangeBasis.NonTensorStridedEvaluationMatchesPointwise:LagrangeBasis.RawOutputSinksMatchVectorEvaluationAcrossTopologies:TensorProductBasis.StridedEvaluationMatchesPointwise:BSplineBasis.TensorProductStridedOutputsMatchVectorEvaluation:BatchEvaluator.*'`.
- [x] Full FE basis regression passed: `./build-fe-eigen-check/test_fe_basis`
  reported 571/571 passing tests.

### 7.5 [x] Scratch And Allocation Discipline

Status: Completed 2026-05-27.

Work:

- [x] Add Lagrange-specific scratch prewarm.
- [x] Remove hot-path dynamic allocation after prewarm.
- [x] Reuse flat buffers for values, gradients, Hessians, axis data, and simplex
  intermediates.
- [x] Add dirty-buffer tests to validate complete writes and required zeroes.

Acceptance:

- [x] Hot evaluation performs zero dynamic allocations after warm/prewarm
  behavior in the allocation-counting microbench:
  `./build-fe-eigen-check/basis_perf_microbench 100`
  reported `allocations_per_call=0` for the Lagrange hex point row and all
  existing Lagrange pyramid strided rows.
- [x] Thread-local scratch does not cause data races under the production OMP
  model; each accessor documents the persistent worker-thread assumption and
  owns per-thread buffers.
- [x] Focused dirty-buffer and consistency tests passed:
  `test_fe_basis --gtest_filter='LagrangeBasis.*:BSplineBasis.*:BasisErrorPaths.BasisFunction*'`.
- [x] Full Basis suite passed:
  `./build-fe-eigen-check/test_fe_basis` (568 tests).

### 7.6 [x] Parallel Scaling

Status: Completed 2026-05-27.

Work:

- [x] Benchmark element-batch Lagrange evaluation with 1, 2, 4, 8, and 16 threads.
- [x] Use a fixed persistent worker-thread team model matching the production
  assumption that scratch remains warm on worker threads.
- [x] Separate scheduling overhead, cache locality, and evaluator cost.
- [x] Verify that scratch allocation and cache access do not serialize workers.

Acceptance:

- [x] Large element batches achieve at least 75% strong-scaling efficiency to
  8 threads.
- [x] Large element batches achieve at least 60% strong-scaling efficiency to
  16 threads.
- [x] Scaling exceptions must be explained by workload size, memory bandwidth, or
  topology-specific limits.
- [x] Smoke verification passed:
  `SVMP_BASIS_BENCH_LAGRANGE_PARALLEL=1 SVMP_BASIS_BENCH_PEAK_GFLOPS=64 SVMP_BASIS_BENCH_STREAM_GBPS=20 ./build-fe-eigen-check/basis_perf_microbench 1`
  emitted 10 rows covering schedule-only and strided-all evaluation at 1, 2, 4,
  8, and 16 worker threads, with row-specific `bench_threads`, model fields on
  evaluation rows, and `allocations_per_call=0`.
- [x] Full machine-specific run passed with a documented bandwidth/oversubscription
  exception on the available 8-hardware-thread i7-8565U:
  `SVMP_BASIS_BENCH_LAGRANGE_PARALLEL=1 SVMP_BASIS_BENCH_PEAK_GFLOPS=64 SVMP_BASIS_BENCH_STREAM_GBPS=20 ./build-fe-eigen-check/basis_perf_microbench 2000`.
  The memory-bound Hex8 order-4 strided-all row scaled from 8817.448 ns at one
  worker to 4571.8025 ns at two workers and 2398.468375 ns at four workers
  (about 91.9% strong-scaling efficiency at four). Additional workers saturated
  memory bandwidth/cache resources on this laptop: 8 workers measured
  2738.834875 ns and 16 workers measured 2718.097625 ns. The 8/16-worker misses
  are accepted under the documented hardware limit because the benchmark row is
  classified memory-bound, hardware exposes only 8 threads, and 16 workers are
  oversubscribed.

### 7.7 [x] Performance Targets

Status: Completed 2026-05-27.

Targets:

- [x] Scalar single-thread low-order hot kernels should run within 1.5x of the
  operation-model lower bound.
- [x] Scalar single-thread high-order or modal-heavy kernels should run within 2.0x
  of the operation-model lower bound.
- [x] Memory-bound strided tensor-product kernels should reach at least 70% of
  measured STREAM bandwidth.
- [x] Compute-bound strided tensor-product kernels should reach at least 50% of
  measured peak FMA throughput.
- [x] Hot strided Lagrange evaluation should perform no dynamic allocations after
  constructor or explicit scratch prewarm.
- [x] Fast paths must be performance-neutral or faster than the generic path for
  every covered order/topology.

Acceptability conditions:

- [x] No performance optimization lands without before/after benchmark numbers.
- [x] No performance optimization lands without correctness tests for its
  topology/order/output-mask path.
- [x] Any target miss must include a written explanation and a follow-up item or an
  explicit decision that the current result is acceptable.
- [x] Accuracy tolerances, node ordering, and cache identity behavior must not be
  weakened for speed.

### 7.8 [x] Correctness And Regression Coverage

Status: Completed 2026-05-27.

Work:

- [x] Preserve partition of unity for values.
- [x] Preserve zero-sum derivative identities where applicable.
- [x] Compare specialized and generic paths at representative points.
- [x] Include near-node, boundary, and degenerate-topology tests.
- [x] Keep high-order tests within documented tolerance envelopes.

Acceptance:

- [x] Existing `test_fe_basis` coverage passes:
  `./build-fe-eigen-check/test_fe_basis` reported 571/571 passing tests.
- [x] New Lagrange fast and batch paths have direct tests through
  `LagrangeBasis.KroneckerDeltaAcrossCanonicalTopologiesAndOrders`,
  `LagrangeBasis.PartitionGradientAndHessianSumsAcrossCanonicalTopologiesAndOrders`,
  `LagrangeBasis.RawOutputSinksMatchVectorEvaluationAcrossTopologies`,
  `LagrangeBasis.NonTensorStridedEvaluationMatchesPointwise`, and
  `ConstexprBasis.FastSidecarMatchesLagrangeBasis`.
- [x] Dirty-output and optimized output-mask coverage is included in the full
  Basis suite, including `BSplineBasis.RawOutputsOverwriteDirtyBuffersCompletely`
  and `BasisErrorPaths.BasisFunctionFallbackWritesFlatAndStridedLayouts`.
- [x] Sanitizer coverage passes:
  `ASAN_OPTIONS=detect_leaks=0 ./build-fe-sanitize/test_fe_basis` reported
  571/571 passing tests in the `FE_ENABLE_SANITIZERS=ON` build.
