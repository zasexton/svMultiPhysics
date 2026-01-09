# Implement General `grad`/`div`/`H` + `div(tensor)` (and Curved-Element Hessians)

This checklist tracks the remaining work to make FE/Forms derivative operators behave like the mathematical notation:

- `grad(·)`, `div(·)`, and `H(·)` work on general (composite) expressions, not only terminals.
- `div(vector) -> scalar` and `div(matrix) -> vector` (row-wise: \((\nabla \cdot A)_i = \partial_j A_{ij}\)).
- Nested derivatives (e.g. `div(grad(u))`) correctly trigger Hessian data requests in the compiler even without an explicit `H(...)` node.
- Second derivatives (basis + solution Hessians) work end-to-end on non-affine/isoparametric (curved) elements.

## Prerequisites

- [x] Rank-3 tensor (`Tensor3`) objects are supported in Forms, and `H(vector)` can return `Tensor3` (`IMPLEMENT_TENSOR3.md`).

---

## Plan: General `grad`/`div`/`H` + `div(tensor)`

### 1) Fix `RequiredData` / field-requirement inference (compiler)

- [x] Extend `forms::detail::analyzeRequiredData` (`Code/Source/solver/FE/Forms/FormCompiler.cpp`) to track *spatial derivative order* through nested `grad`/`div`/`curl`/`H` so implicit second derivatives request:
  - [x] `assembly::RequiredData::BasisHessians` whenever `∂²/∂x²` is implied (e.g. `div(grad(v))`).
  - [x] `assembly::RequiredData::SolutionHessians` for residual forms when the implied second derivative applies to `TrialFunction` (e.g. `div(grad(u))` in a residual).
- [x] Extend `forms::detail::analyzeFieldRequirements` (`Code/Source/solver/FE/Forms/FormCompiler.cpp`) so nested derivatives on `DiscreteField`/`StateField` request `SolutionGradients`/`SolutionHessians` even when the field is not the *immediate* child of `grad`/`div`/`curl`/`H`.
- [x] Add FormCompiler unit tests (`Code/Source/solver/FE/Tests/Unit/Forms/test_FormCompiler.cpp`) that assert required flags for nested-derivative patterns:
  - [x] `div(grad(v))`, `grad(div(v))` (test functions).
  - [x] Same expressions with `TrialFunction` in bilinear vs residual (solution-vs-basis flags).
  - [x] Same expressions with `DiscreteField`/`StateField`.

### 2) Implement a “spatial jet” evaluator (kernels)

- [x] Add a templated “jet” evaluator in `Code/Source/solver/FE/Forms/FormKernels.cpp` that can compute:
  - [x] Order-0: `{ value }`
  - [x] Order-1: `{ value, grad(value) }`
  - [x] Order-2: `{ value, grad(value), H(value) }`
  for arbitrary expression trees, parameterized on scalar type (`Real` vs AD `Dual`) so residual Jacobians still work.
- [x] Define shape-aware jet containers:
  - [x] Scalar jets: `value`, `grad[D]`, `hess[D][D]`.
  - [x] Vector jets: `value[m]`, `grad[m][D]`, `hess[m][D][D]` (returned as `Tensor3` for order-2).
  - [x] Matrix support for derivatives:
    - [x] Minimal support needed for `div(matrix)` (compute \(\partial_j A_{ij}\) without constructing `grad(matrix)` as a rank-3 object), and throw `NotImplemented` for unsupported derivative ranks (e.g. rank-4).
- [x] Implement jet rules for NS-relevant nodes/operators (all at least for order-1; order-2 where needed by `H` and nested derivatives):
  - [x] Terminals: `Constant`, `Coordinate`, `Identity`, `TestFunction`, `TrialFunction`, `DiscreteField`, `StateField`, `Coefficient` (FD for spatial derivatives when needed).
  - [x] Algebra: `+`, `-`, unary `-`, `*`, `/` (including scalar–tensor broadcast rules that exist today).
  - [x] Nonlinear scalars: `pow`, `sqrt`, `exp`, `log`, `abs`, `sign`.
  - [x] `min`/`max`/`conditional` (define consistent derivative behavior; emit clear error near non-differentiable points if needed).
  - [x] Structural: `component`/indexed access, `as_vector`, `as_tensor`, `as_tensor3` (tensor3-valued spatial derivatives intentionally `NotImplemented` since they require rank-4).
  - [x] Tensor ops: `inner`, `doubleContraction`, `trace`, `transpose`, `sym`, `skew`, `norm`, `normalize`.
  - [x] DG: `jump`, `avg`, `minus`, `plus` (and ensure derivative propagation matches existing value rules).
  - [x] Return a clear `NotImplemented` for all other nodes when order > 0 (to avoid silent wrong derivatives).

### 3) Route `grad`/`div`/`H` evaluation through the jet evaluator

- [x] Update `GradientNode` evaluation to remove “terminals-only” restrictions and use the jet evaluator (order-1).
- [x] Update `DivergenceNode` evaluation to:
  - [x] `div(vector) -> scalar` from the vector jet’s `grad` (trace).
  - [x] `div(matrix) -> vector` row-wise, without introducing `grad(matrix) -> Tensor3` as a required intermediate.
- [x] Update `HessianNode` evaluation to remove “terminals-only” restrictions and use the jet evaluator (order-2):
  - [x] `H(scalar) -> matrix`.
  - [x] `H(vector) -> Tensor3` (now possible thanks to `Tensor3` support), including composite expressions.
- [x] Ensure the Real and Dual execution paths share the same jet logic (templated helpers) to avoid divergence between residual and Jacobian evaluation.

### 4) Tests for composite derivatives

- [x] Add Forms unit tests (`Code/Source/solver/FE/Tests/Unit/Forms/`) that compare jet-based `grad`/`div`/`H` of composite expressions against finite differences in *physical space*:
  - [x] Scalar composite: `H( sqrt(1 + u*u) )`, `grad(exp(u))`, etc.
  - [x] Vector composite: `H( u / (1 + norm(u)) )` returning a `Tensor3`.
  - [x] `div(matrix)` cases where the matrix is built from composite expressions (e.g., `outer(u, v)` or `as_tensor(...)`).
- [x] Add tests that specifically validate nested derivative evaluation paths (e.g., `div(grad(u))` and `grad(div(u))`) do not crash and produce FD-consistent results.

---

## Plan: Extend Hessians for non-affine/isoparametric (curved) elements

### 1) Baseline audit (StandardAssembler)

- [x] Confirm the existing curved-element Hessian machinery is correct/complete in:
  - [x] `Code/Source/solver/FE/Assembly/StandardAssembler.cpp` (physical basis Hessians via mapping Hessians + \(d^2\xi/dx^2\) term).
  - [x] `Code/Source/solver/FE/Geometry/IsoparametricMapping.cpp` (`mapping_hessian()` implementation and usage).
- [x] Add or extend `StandardAssembler` unit tests (`Code/Source/solver/FE/Tests/Unit/Assembly/test_StandardAssembler.cpp`) to validate non-affine physical Hessians against finite differences on a curved element.

### 2) Close geometry coverage gaps

- [x] Audit `Geometry::MappingFactory` and geometry basis implementations for all curved element types required by svMultiPhysics:
  - [x] `mapping_hessian()` returns correct, nonzero values where appropriate.
  - [x] The geometry basis implements `evaluate_hessians()` (or equivalent) for those elements.
- [x] Implement any missing `mapping_hessian()` / geometry-basis Hessian paths for targeted element types (e.g., `Triangle6`, `Tetra10`).

### 3) Make it end-to-end usable from Forms

- [x] Ensure the improved FormCompiler inference (above) requests `BasisHessians`/`SolutionHessians` whenever second derivatives are implied, so `AssemblyContext` populates:
  - [x] `trialPhysicalHessian(...)`
  - [x] `solutionHessian(...)`
  on curved elements as well as affine ones.

### 4) Assembler parity decision

- [x] Decide and implement one of:
  - [ ] Add curved-element Hessian support to other assemblers (`MatrixFreeAssembler`, `FunctionalAssembler`, etc.), or
  - [x] Guard/document that `H(...)` (and implicit second derivatives) require `StandardAssembler` for now.

### 5) Curved-element validation

- [x] Add curved-element validation tests (Assembly/Geometry) on quadratic elements that compare:
  - [x] basis physical Hessians vs finite differences for a known polynomial field composed with a curved mapping
  - [x] solution Hessians vs finite differences after projecting/interpolating a known field

---

## Completion / Validation

- [x] `ctest -R FE_Forms_Tests --output-on-failure` passes.
- [x] If assembly tests are added/updated: `ctest -R FE_Assembly_Tests --output-on-failure` (or the repo’s assembly test target) passes.
