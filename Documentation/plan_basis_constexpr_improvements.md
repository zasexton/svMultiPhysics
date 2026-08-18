# Plan: FE Basis `constexpr` Improvements

**Date**: 2026-05-21
**Scope**: `Code/Source/solver/FE/Basis/` and targeted basis unit tests
**Goal**: Use compile-time constants and constexpr-capable helpers to reduce
runtime setup work, remove duplicated topology logic, and make fixed-order
basis paths easier for the compiler to optimize without changing the existing
runtime-polymorphic `BasisFunction` API.

## Progress Checklist

- [x] Phase 0: Confirmed the FE basis target builds and kept runtime-only
  paths documented in the constraints and non-goals.
- [x] Phase 1: Added shared constexpr topology traits in `BasisTraits.h` and
  migrated duplicated local topology helpers.
- [x] Phase 2: Added constexpr low-order node tables, wired them through
  `NodeOrdering`, and covered fixed complete and serendipity aliases.
- [x] Phase 3: Added shared constexpr integer math in `BasisIntegerMath.h` and
  replaced safe integer-exponent `std::pow` uses in the Basis folder.
- [x] Phase 4: Made the current `LagrangeBasisFast` sidecar constexpr-capable,
  added Hessian coverage, added TRI6/TET10 sidecar specializations, and routed
  matching fixed-order `LagrangeBasis` paths through the sidecar.
- [x] Phase 5: Added constexpr-generated 1D Lagrange coefficient tables for
  axis sizes `N = 1..5` with the existing dynamic fallback preserved.
- [x] Phase 6: Made `ReferenceDerivativeJet` constexpr-capable and added
  compile-time coverage for representative expressions.
- [x] Validation: `cmake --build build-fe-check --target test_fe_basis -j2`
  completed, and `./build-fe-check/test_fe_basis` passed 474 tests.
- [ ] Phase 7: Optional compile-time reference tables remain deferred until a
  consuming assembly path and microbenchmark justify the added table surface.

## Guiding Constraints

- Keep `BasisFunction` as the canonical runtime API. Its virtual methods,
  `std::vector` outputs, cache identities, spline state, and quadrature-driven
  paths are runtime features and should not be forced into a constexpr shape.
- Add constexpr capability around data and pure helper logic: topology traits,
  fixed node tables, size formulas, integer powers, fixed-order Lagrange
  evaluators, and small derivative algebra.
- Preserve existing basis ordering, especially VTK-compatible node ordering.
- Keep arbitrary-order and dynamic spline/NURBS paths on runtime storage.
- Verify every constexpr table against the current runtime behavior before
  deleting existing generator logic.

## Cleanup Notes From Basis Review

These are the concrete cleanup targets found during the follow-up review of
`Code/Source/solver/FE/Basis/`. They align with the phases below and should be
handled before adding broader template or precomputed-table machinery.

- Shared topology traits are the highest-value cleanup. Local
  `is_line(...)`, `is_triangle(...)`, `is_wedge(...)`, and related helpers are
  duplicated in:
  - `BernsteinBasis.cpp`
  - `BubbleBasis.cpp`
  - `HierarchicalBasis.cpp`
  - `SpectralBasis.cpp`
  - `VectorBasis.cpp`
  - `LagrangeBasis.cpp`
  - `NodeOrderingConventions.cpp`
  Replacing these with `BasisTraits.h` should reduce drift and make subsequent
  constexpr work easier to audit.
- `NodeOrderingConventions.cpp` is the best first target for concise
  constexpr tables. Fixed complete-family aliases are currently generated into
  static vectors through `complete_lagrange_alias_coords(...)`, while
  serendipity coordinates are a long nested switch in
  `NodeOrdering::get_node_coords(...)`. Inline constexpr tables can make both
  `get_node_coords(...)` and `num_nodes(...)` shorter and less error-prone.
- Integer-power logic is scattered. `BernsteinBasis.cpp` has a local
  `pow_int(...)` wrapper that still calls `std::pow`, and other integer
  exponent calls appear in `LagrangeBasisPyramidDetail.h`,
  `SpectralBasis.cpp`, `SerendipityBasis.cpp`, and
  `OrthogonalPolynomials.cpp`. Move this to one shared constexpr scalar helper
  and replace only the safe non-negative integer exponent cases.
- `LagrangeBasisFast.h` already has the right sidecar shape but is underused.
  Equivalent order-1 formulas are repeated across `LagrangeBasis.cpp` values,
  gradients, Hessians, and `evaluate_all(...)`. Extending the fast sidecar to
  Hessians and using it to centralize those formulas would make the runtime
  implementation shorter while preserving the public API.
- `LagrangeBasis::compute_axis_monomial_coefficients()` rebuilds small 1D
  polynomial coefficient tables per instance. Constexpr coefficient tables for
  common axis sizes should shorten setup for orders `0..4` and keep the current
  dynamic fallback for larger orders.
- `detail/ReferenceDerivativeJet.h` is pure fixed-size arithmetic and should
  be made constexpr-capable before it is used by more fixed-order sidecar
  evaluators.

## Phase 0: Baseline And Inventory

- Confirm FE still builds as C++20 via `Code/Source/solver/FE/CMakeLists.txt`.
- Capture current targeted test status for:
  - `test_fe_basis`
  - `test_BasisCacheFactory`
  - `test_LagrangeBasis`
  - `test_BernsteinBasis`
  - `test_SerendipityTensorModal`
  - `test_VectorBases`
- Add a small inventory note in the final PR or commit message identifying
  which paths remain intentionally runtime-only.

Acceptance:

- No source changes yet.
- Known current test failures, if any, are documented before constexpr work.

## Phase 1: Shared Compile-Time Topology Traits

Create `Code/Source/solver/FE/Basis/BasisTraits.h` with constexpr helpers:

- `constexpr bool is_line(ElementType)`
- `constexpr bool is_triangle(ElementType)`
- `constexpr bool is_quadrilateral(ElementType)`
- `constexpr bool is_tetrahedron(ElementType)`
- `constexpr bool is_hexahedron(ElementType)`
- `constexpr bool is_wedge(ElementType)`
- `constexpr bool is_pyramid(ElementType)`
- `constexpr bool is_simplex(ElementType)`
- `constexpr bool is_tensor_product(ElementType)`
- `constexpr int reference_dimension(ElementType)`
- `constexpr ElementType canonical_lagrange_type(ElementType)`
- `constexpr int complete_lagrange_alias_order(ElementType)`
- constexpr scalar size formulas such as:
  - `line_lagrange_size(order)`
  - `triangle_lagrange_size(order)`
  - `quad_lagrange_size(order)`
  - `tetra_lagrange_size(order)`
  - `hex_lagrange_size(order)`
  - `wedge_lagrange_size(order)`
  - `pyramid_lagrange_size(order)`

Then replace duplicated local topology helpers in:

- `BernsteinBasis.cpp`
- `BubbleBasis.cpp`
- `HierarchicalBasis.cpp`
- `SpectralBasis.cpp`
- `VectorBasis.cpp`
- `NodeOrderingConventions.cpp`
- applicable detail headers

Acceptance:

- Add compile-time tests with `static_assert` coverage for representative
  element aliases.
- Runtime tests still pass.
- No behavior change in exception messages is required unless the old helper
  logic was inconsistent.

## Phase 2: Constexpr Low-Order Node Tables

Move fixed low-order node coordinate data out of runtime switch-generated
vectors and into `inline constexpr std::array` tables.

Targets:

- Complete-family aliases currently routed through
  `complete_lagrange_alias_coords(...)` in `NodeOrderingConventions.cpp`:
  - `Line2`, `Line3`
  - `Triangle3`, `Triangle6`
  - `Quad4`, `Quad9`
  - `Tetra4`, `Tetra10`
  - `Hex8`, `Hex27`
  - `Wedge6`, `Wedge18`
  - `Pyramid5`, `Pyramid14`
- Serendipity-only fixed node tables currently hard-coded in
  `NodeOrdering::get_node_coords(...)`:
  - `Quad8`
  - `Hex20`
  - `Wedge15`
  - `Pyramid13`

Implementation shape:

- Define tables using `std::array<std::array<Real, 3>, N>` or a small literal
  point struct if `math::Vector<Real, 3>` alignment makes constexpr table
  ergonomics awkward.
- Keep public return types unchanged:
  - `NodeOrdering::get_node_coords(...)` can construct `math::Vector<Real, 3>`
    from the constexpr table entry.
  - `NodeOrdering::get_lagrange_node_coords(...)` can keep returning
    `std::vector<math::Vector<Real, 3>>` for arbitrary order.
- Add an internal constexpr lookup for fixed aliases and use the current
  runtime generators only for order values outside the fixed table surface.

Acceptance:

- Add tests comparing every fixed table entry against the existing expected
  `NodeOrdering` coordinates.
- Add size checks for every fixed table.
- Preserve current arbitrary-order node generation behavior.

## Phase 3: Shared Constexpr Integer Math

Add a small header such as `BasisIntegerMath.h` or a `detail/` helper with:

- `constexpr Real pow_int(Real base, int exponent)`
- `constexpr Real binomial_real(int n, int k)`
- optional `constexpr std::size_t binomial_size(int n, int k)`

Replace runtime `std::pow` calls where the exponent is known to be a small
non-negative integer:

- `BernsteinBasis.cpp`
- `SerendipityBasis.cpp`
- `OrthogonalPolynomials.cpp` collapsed-coordinate factors where applicable
- `SpectralBasis.cpp` pyramid modal helpers where applicable
- `detail/LagrangeBasisPyramidDetail.h`

Do not replace `std::pow` for genuinely real exponents or where numerical
behavior depends on libm handling of non-integer cases.

Acceptance:

- Values, gradients, and Hessians are numerically unchanged within existing
  tolerances.
- Add unit coverage for `pow_int` edge cases:
  - exponent `0`
  - exponent `1`
  - exponent `2`
  - negative bases with odd/even exponents
- Add compile-time `static_assert` coverage for integer-only cases.

## Phase 4: Complete The `LagrangeBasisFast` Constexpr Sidecar

Extend `LagrangeBasisFast.h`, which already provides fixed-order static
specializations, into a true constexpr sidecar for fixed topology/order hot
paths.

Work items:

- Mark current `evaluate(...)` and `evaluate_gradients(...)` functions
  `static constexpr` where `math::Vector` and `math::Matrix` allow it.
- Add `evaluate_hessians(...)` for current specializations:
  - `LagrangeLineFast<1>`
  - `LagrangeLineFast<2>`
  - `LagrangeQuadFast<1>`
  - `LagrangeHexFast<1>`
  - `LagrangeTriFast<1>`
  - `LagrangeTetFast<1>`
- Add missing common fixed-order specializations only if they are used by real
  assembly paths or tests:
  - `LagrangeTriFast<2>`
  - `LagrangeTetFast<2>`
  - `LagrangeQuadFast<2>`
  - `LagrangeHexFast<2>`
- Keep this as an opt-in sidecar. Do not route generic `BasisFactory` creation
  through templates.

Acceptance:

- Add `static_assert` checks for representative points and derivative values.
- Add runtime parity tests comparing fast specializations against
  `LagrangeBasis` for values, gradients, and Hessians.
- No change to `LagrangeBasis` public behavior.

## Phase 5: Constexpr Lagrange Coefficient Tables For Common Orders

The runtime `LagrangeBasis` path currently builds 1D monomial coefficients per
instance in `compute_axis_monomial_coefficients()`. Replace common-order setup
with constexpr-generated tables while keeping the dynamic fallback.

Work items:

- Add a constexpr coefficient generator for equispaced 1D Lagrange nodes.
- Precompute or consteval-generate tables for axis sizes `N = 1..5`
  corresponding to orders `0..4`.
- In `compute_axis_monomial_coefficients()`, copy from constexpr tables for
  `N <= 5`; use the current runtime algorithm for larger `N`.
- Keep the existing templated Horner evaluators and runtime fallback.

Acceptance:

- Add tests comparing constexpr coefficients to the current generated
  coefficients indirectly through basis value/derivative parity.
- Benchmark or at least sanity-check construction cost for common orders.
- Preserve high-order behavior.

## Phase 6: Make `ReferenceDerivativeJet` Constexpr-Capable

`detail/ReferenceDerivativeJet.h` is pure fixed-size arithmetic. Convert its
constructors and operators to `constexpr` where possible:

- `constant_jet`
- `variable_jet`
- arithmetic operators
- `compose_univariate`
- `reciprocal`
- `pow_int`

Prerequisites:

- Confirm `math::Vector<Real, 3>` and `math::Matrix<Real, 3, 3>` operations
  used by `Jet3` are constexpr-capable. If assignment through expression
  templates blocks constexpr use, limit this phase to direct element-access
  code.

Acceptance:

- Add compile-time tests for simple jet expressions:
  - constant
  - variable
  - product
  - reciprocal
  - integer power
- Confirm Bernstein and pyramid Hessian tests still pass.

## Phase 7: Optional Compile-Time Reference Tables

Only after Phases 1-6 are stable, consider constexpr reference tables for
small fixed element/order pairs.

Candidate tables:

- Reference basis values/gradients/Hessians at standard low-order quadrature
  rules.
- Fixed modal-to-nodal transforms for retained low-order serendipity elements.
- Fixed vector-basis association tables for RT/ND low orders.

Constraints:

- Do this only when the consuming assembly path can use the table without
  adding indirection or cache layout churn.
- Do not precompute tables for custom quadrature, splines, NURBS, or
  arbitrary-order modal paths.

Acceptance:

- Require a microbenchmark or assembly timing before merging this phase.
- Reject any table path that improves construction time but slows warm
  assembly.

## Non-Goals

- Do not attempt to make `BasisFunction` virtual methods `constexpr`.
- Do not replace `BasisCache`; it solves runtime quadrature reuse, not compile
  time evaluation.
- Do not template the whole factory or public space construction stack.
- Do not convert dynamic spline/NURBS knot and weight handling to constexpr.
- Do not move arbitrary-order pyramid modal inversion into compile-time code
  unless the order is intentionally capped and tested.

## Validation Matrix

Unit tests:

- `test_fe_basis`
- `test_LagrangeBasis`
- `test_BernsteinBasis`
- `test_SerendipityTensorModal`
- `test_HigherOrderWedgePyramid`
- `test_VectorBases`
- `test_BasisCacheFactory`

Compile-time tests:

- New targeted test file, for example
  `Code/Source/solver/FE/Tests/Unit/Basis/test_ConstexprBasis.cpp`.
- Use `static_assert` for traits, low-order sizes, fixed node table sizes,
  integer math, and fast-basis evaluations.

Runtime parity tests:

- Fixed node tables match `NodeOrdering` expectations.
- `LagrangeBasisFast` matches `LagrangeBasis`.
- Integer-power replacements match current basis values and derivatives.

Performance checks:

- Construction-time microbenchmark for repeated common-order basis creation.
- Warm assembly smoke check to ensure no regressions in `prepareBasis`.

## Suggested Implementation Order

1. Add `BasisTraits.h` and tests.
2. Migrate duplicated topology helpers.
3. Add constexpr low-order node tables and table parity tests.
4. Add shared constexpr integer math and replace safe `std::pow` integer cases.
5. Extend `LagrangeBasisFast` with constexpr Hessian coverage and parity tests.
6. Add common-order constexpr Lagrange coefficient tables.
7. Convert `ReferenceDerivativeJet` helpers where constexpr-safe.
8. Consider optional reference tables only after profiling.

## Main Risks

- Table ordering drift: low-order node tables must exactly match current VTK
  ordering.
- Compile-time bloat: adding many template specializations can increase build
  time and binary size. Keep sidecar paths scoped to common orders.
- False performance wins: reducing construction cost may not affect warm
  assembly. Require timing before merging broad table work.
- Numerical drift from replacing `std::pow`: restrict replacements to integer
  exponents and compare against existing tolerances.
