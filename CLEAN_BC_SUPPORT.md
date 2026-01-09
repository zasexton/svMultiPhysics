# Clean Coupled Boundary-Condition (BC) Support — Plan & Checklist

This document is a concrete, file-oriented plan for evolving the current coupled-BC infrastructure into a cleaner, more “symbolic” API suitable for reusable physics models (e.g., Windkessel RCR), while keeping the FE library physics-agnostic and avoiding assembly loops in physics code.

---

## 0) Baseline: What Already Exists (Current Repo State)

The FE library already has the core pieces needed for 0D–3D coupling:

- **Boundary-integrated scalars**: `FE/Forms/BoundaryFunctional.{h,cpp}`
  - `forms::BoundaryFunctional` (integrand + marker + name + reduction)
  - `forms::compileBoundaryFunctionalKernel(...)` → `assembly::FunctionalKernel` bridge
- **Auxiliary 0D state container**: `FE/Systems/AuxiliaryState.h`
  - named scalar variables + committed/work values + history
- **Coupled-BC orchestration**: `FE/Systems/CoupledBoundaryManager.{h,cpp}`
  - computes registered boundary functionals, evolves aux state, exposes `constraints::CoupledBCContext`
  - invoked automatically before PDE assembly (`FE/Systems/SystemAssembly.cpp`)
- **Coupled BC types**: `FE/Constraints/CoupledBCContext.h`, `CoupledNeumannBC.h`, `CoupledRobinBC.h`
- **Loop-free coupled BC helpers (v0)**: `FE/Systems/CoupledBoundaryConditions.h`
  - `systems::bc::applyCoupledNeumann/applyCoupledRobin` that accept pre-built `CoupledNeumannBC/CoupledRobinBC` objects
- **Example usage**: `Physics/Formulations/Poisson/PoissonModule.cpp`
  - demonstrates RCR-like coupling but still requires some boilerplate to construct BC objects and register functionals/state

The proposal below **refactors and extends** this baseline; it does not replace it.

---

## 1) Review of Proposed “Cleaner API” (High-Level)

The outline is directionally solid: it pushes physics-specific concepts (Windkessel, flow rate, pressure) into the Physics layer, while keeping FE generic:

- FE knows only:
  - “integrate this expression over boundary marker”
  - “evolve this auxiliary state variable with a generic ODE method”
  - “apply a flux term as a Neumann/Robin boundary integral”

Key technical tradeoffs to resolve:

1) **Implicit ODE methods (BackwardEuler/BDF2) need a solve**
   - With only `RHSFunction f(X, integrals, t)` a fully-generic implicit method requires a scalar nonlinear solve (Newton/secant) or a user-provided Jacobian.
   - Plan below assumes **scalar ODE** support (size==1) and uses **robust scalar Newton with finite-difference derivative** as default for implicit methods.

2) **Symbolic FormExpr terminals must be “resolved”**
   - Adding `FormExpr::boundaryIntegral(...)` and `FormExpr::auxiliaryState(...)` is feasible, but these symbols cannot be compiled into assembly kernels until they are mapped to runtime values.
   - To keep FE/Forms independent of FE/Systems and avoid plumbing `CoupledBCContext` through assemblers, the recommended approach is:
     - treat these as **symbolic placeholders**
     - **resolve** them inside the Systems helper into standard `FormExpr::coefficient(...)` nodes capturing `CoupledBCContext*`.

3) **Where to put the new helper**
   - Keep loop-free “physics authoring helpers” in **FE/Systems** (not FE/Forms) so they can legally access `FESystem` / `CoupledBoundaryManager`.

---

## 2) Target API (FE Library)

### 2.1 ODE Integration (New)

New generic integrator:

- `Code/Source/solver/FE/Systems/ODEIntegrator.h`
- `Code/Source/solver/FE/Systems/ODEIntegrator.cpp` (or header-only if preferred)

```cpp
namespace svmp::FE::systems {

enum class ODEMethod { ForwardEuler, BackwardEuler, RK4, BDF2 };

class ODEIntegrator {
public:
  using RHSFunction = std::function<Real(const AuxiliaryState& state,
                                         const forms::BoundaryFunctionalResults& integrals,
                                         Real t)>;

  static void advance(ODEMethod method,
                      std::string_view state_name,
                      AuxiliaryState& state,
                      const RHSFunction& rhs,
                      const forms::BoundaryFunctionalResults& integrals,
                      Real t, Real dt);
};

} // namespace svmp::FE::systems
```

### 2.2 Simplified AuxiliaryStateRegistration (Refactor)

Replace evolve-callback registrations with “RHS-only + integrator choice” registrations:

- **Move/define** `AuxiliaryStateRegistration` in:
  - preferred: `Code/Source/solver/FE/Systems/AuxiliaryState.h`
  - alternative: new `Systems/AuxiliaryStateRegistration.h` to keep headers small

Key fields:
- `spec`, `initial_values`, `required_integrals`
- `rhs` (only)
- `integrator` (default BackwardEuler)

### 2.3 Symbolic Terminals in Forms (New)

Add symbolic placeholders (not directly assemblable until resolved by Systems):

- `Code/Source/solver/FE/Forms/FormExpr.h` / `FormExpr.cpp`

```cpp
static FormExpr boundaryIntegral(FormExpr integrand, int marker, std::string name);
static FormExpr auxiliaryState(std::string name);
```

### 2.4 Expression-Aware applyCoupledNeumann/applyCoupledRobin (New Overloads)

Add new overload(s) to `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`:

```cpp
[[nodiscard]] forms::FormExpr applyCoupledNeumann(
    FESystem& system,
    FieldId primary_field,
    forms::FormExpr residual,
    const forms::FormExpr& v,
    int boundary_marker,
    const forms::FormExpr& flux,  // symbolic placeholders allowed
    std::span<const AuxiliaryStateRegistration> aux_states = {});
```

Recommended for Robin (flux-form convention consistent with existing Forms helpers):

```cpp
[[nodiscard]] forms::FormExpr applyCoupledRobin(
    FESystem& system,
    FieldId primary_field,
    forms::FormExpr residual,
    const forms::FormExpr& u,
    const forms::FormExpr& v,
    int boundary_marker,
    const forms::FormExpr& alpha, // may contain boundaryIntegral/auxiliaryState
    const forms::FormExpr& rhs,   // may contain boundaryIntegral/auxiliaryState
    std::span<const AuxiliaryStateRegistration> aux_states = {});
```

The helper must:
- traverse expressions to find placeholders
- register boundary functionals and auxiliary states with the manager
- resolve placeholders into normal coefficient nodes capturing `CoupledBCContext*`
- append boundary integral terms to the residual (no user loops)

### 2.5 Builder Pattern (Optional, but strongly recommended)

Add a convenience builder:

- `Code/Source/solver/FE/Systems/AuxiliaryStateBuilder.h` (+ optional `.cpp`)

```cpp
auto X_reg = systems::auxiliaryODE("X", X0)
  .requiresIntegral(Q)  // `requires` is a C++20 keyword
  .withRHS(rhs_fn)
  .withIntegrator(ODEMethod::BackwardEuler)
  .build();
```

---

## 3) Concrete Implementation Plan (Phased)

### Phase A — ODE Integrator + Registration Refactor (Systems)

Goal: CoupledBoundaryManager evolves auxiliary state using a generic integrator and a user-provided RHS (no custom per-physics integrator logic).

1) Add `ODEIntegrator` and `ODEMethod`.
2) Define the simplified `AuxiliaryStateRegistration` and keep the current registration temporarily as **legacy** for compatibility (see Compatibility section).
3) Update `CoupledBoundaryManager::prepareForAssembly` to:
   - reset aux work state to committed
   - compute all required functionals
   - for each auxiliary registration:
     - call `ODEIntegrator::advance(...)` for each scalar variable
4) Update existing call sites/tests to the new struct or provide adapter overloads.

### Phase B — Symbolic Coupled Terminals (Forms)

Goal: give physics authors a clean way to write coupled flux expressions as pure `FormExpr`.

1) Add `FormExprType::BoundaryFunctional` and `FormExprType::AuxiliaryState` (names can vary).
2) Implement nodes in `Forms/FormExpr.cpp`.
3) Ensure placeholders:
   - are printable (`toString`)
   - are discoverable (`type`, `boundaryMarker` for boundaryIntegral)
   - do **not** require test/trial functions
4) Add Forms unit tests for construction and formatting.

### Phase C — Expression-Aware Coupled BC Helpers (Systems)

Goal: implement `applyCoupledNeumann(..., flux_expr)` that “does the right thing” without user loops.

1) Add extraction utilities in `Systems/CoupledBoundaryConditions.h` (or a new internal header):
   - `collectBoundaryIntegrals(expr) -> vector<forms::BoundaryFunctional>`
   - `collectAuxStateRefs(expr) -> vector<string>`
2) Add “resolver” that converts placeholders into coefficients capturing a stable `CoupledBCContext*`.
   - Recommended approach (no assembler plumbing): deep-copy the expression tree and replace only the placeholder nodes with `FormExpr::coefficient(...)` nodes.
   - Restrict/validate supported node types in v1; throw with a clear diagnostic if unsupported types appear in the coupled expression.
3) Implement new overload(s) for Neumann/Robin using the extractor + resolver.
4) Keep the existing v0 helper overloads (span of `CoupledNeumannBC/CoupledRobinBC`) as a lower-level API.

### Phase D — Update Examples + Add Tests

1) Update PoissonModule (or a dedicated demo) to use:
   - `FormExpr::boundaryIntegral(...)`
   - `FormExpr::auxiliaryState(...)`
   - `systems::auxiliaryODE(...)` builder
   - expression-aware `systems::bc::applyCoupledNeumann(...)`
2) Add Systems tests that:
   - verify functional registration dedup
   - verify aux integration resets per assembly and commits correctly
   - verify the resolved flux yields correct assembled residual for a simple mesh

### Phase E — Physics Layer (Optional, separate module)

1) Add Physics base class `LumpedParameterModel`.
2) Add Windkessel RCR implementation using FE primitives.
3) Add a Physics test (or integration test via Poisson) verifying the model applies cleanly.

---

## 4) Detailed File-by-File Checklist (Actionable)

Each item includes: **file**, **change**, **dependencies**, **open decisions**.

### 4.1 FE/Systems: ODE Integration

- [ ] `Code/Source/solver/FE/Systems/ODEIntegrator.h`
  - Add `ODEMethod` enum and `ODEIntegrator` interface.
  - Dep: none.
  - Decision: scalar-only (v1) vs vector state support.

- [ ] `Code/Source/solver/FE/Systems/ODEIntegrator.cpp`
  - Implement:
    - Forward Euler (explicit)
    - RK4 (explicit)
    - Backward Euler (implicit): scalar Newton with finite-difference derivative
    - BDF2 (implicit): requires `AuxiliaryState::previous(1/2)`; scalar Newton
  - Dep: `Systems/AuxiliaryState.h`, `Forms/BoundaryFunctional.h`.
  - Decision: constant dt only (v1) vs variable dt (use `SystemStateView::dt_prev/dt_history`).

- [ ] `Code/Source/solver/FE/Tests/Unit/Systems/test_ODEIntegrator.cpp` (new)
  - Unit test explicit methods on `dX/dt = -X` and constant forcing.
  - Unit test implicit methods converge to expected (within tol).
  - Dep: ODEIntegrator.

### 4.2 FE/Systems: Registration Refactor

- [ ] `Code/Source/solver/FE/Systems/AuxiliaryState.h`
  - Add (or move) `AuxiliaryStateRegistration` struct with:
    - `spec`, `initial_values`, `required_integrals`
    - `rhs` and `integrator`
  - Dep: `ODEIntegrator.h` (for `ODEMethod` type).
  - Decision: keep existing `CoupledBoundaryManager::AuxiliaryStateRegistration` as legacy vs migrate in-place.

- [ ] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.h`
  - Switch to the new `AuxiliaryStateRegistration` type (or add overloads).
  - Dep: `Systems/AuxiliaryState.h`, `Systems/ODEIntegrator.h`.

- [ ] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
  - Replace `reg.evolve(...)` calls with `ODEIntegrator::advance(...)`.
  - Preserve “no Newton drift” semantics by re-integrating from committed state each `prepareForAssembly(...)`.
  - Dep: ODEIntegrator implementation.

- [ ] Compatibility bridge (recommended)
  - [ ] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.h`
    - Keep `addAuxiliaryStateLegacy(...)` (or keep old struct name) for existing code paths; implement internally by storing a legacy entry list.
  - Dep: existing call sites.
  - Decision: deprecation policy.

### 4.3 FE/Systems: Builder Pattern (Optional)

- [ ] `Code/Source/solver/FE/Systems/AuxiliaryStateBuilder.h`
  - Implement `AuxiliaryStateBuilder` + `auxiliaryODE(name, initial)`.
  - Dep: new `AuxiliaryStateRegistration`, `ODEIntegrator.h`.

- [ ] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateBuilder.cpp` (new)
  - Verify builder emits correct registration: name, initial value, integrals list, integrator selection.

### 4.4 FE/Forms: Symbolic Coupled Terminals

- [ ] `Code/Source/solver/FE/Forms/FormExpr.h`
  - Add `FormExpr::boundaryIntegral(FormExpr integrand, int marker, std::string name)`.
  - Add `FormExpr::auxiliaryState(std::string name)`.
  - Add new `FormExprType` entries (names TBD).
  - Dep: none.
  - Decision: allow `Reduction` selection at this level vs default `Sum` only.

- [ ] `Code/Source/solver/FE/Forms/FormExpr.cpp`
  - Implement corresponding node types:
    - boundaryIntegral node stores: name + marker + integrand (child)
    - auxiliaryState node stores: name
  - Ensure `hasTest/hasTrial` are false.

- [ ] `Code/Source/solver/FE/Tests/Unit/Forms/test_CoupledTerminals.cpp` (new)
  - Verify construction, markers, and `toString()` stability.

### 4.5 FE/Systems: Expression-Aware Coupled BC Helpers

- [ ] `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`
  - Add overload:
    - `applyCoupledNeumann(..., int boundary_marker, FormExpr flux, span<AuxRegistration>)`
  - Add overload for Robin in flux-form convention (recommended):
    - `applyCoupledRobin(..., int boundary_marker, FormExpr alpha, FormExpr rhs, span<AuxRegistration>)`
  - Keep existing v0 overloads (span of CoupledNeumannBC/CoupledRobinBC) unchanged.
  - Dep: new FormExpr terminals; new Aux registration type.

- [ ] Implement extraction utilities (internal to the header or new `.cpp` for compile-time reduction)
  - Traverse `FormExprNode::childrenShared()` and detect:
    - `FormExprType::BoundaryIntegralSymbol` → register `forms::BoundaryFunctional{integrand, marker, name}`
    - `FormExprType::AuxiliaryStateSymbol` → record referenced variable name
  - Dep: terminal node types.
  - Decision: allow cross-marker references in one flux expression (e.g., flux depends on an integral from a different boundary marker).

- [ ] Implement “resolver” (placeholder → coefficient)
  - Replace symbolic terminals with `FormExpr::coefficient(...)` nodes capturing `CoupledBCContext*`.
  - Ensure coefficient names are stable and diagnostic-friendly.
  - Dep: CoupledBoundaryManager context pointer.
  - Decision: supported expression subset for v1; provide clear error if unsupported nodes appear.

### 4.6 Examples + Migration

- [ ] `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`
  - Switch to the expression-aware helper:
    - define `Q_sym = FormExpr::boundaryIntegral(inner(u, normal()), marker, "Q_marker")` (or scalar field for Poisson demo)
    - define `X_sym = FormExpr::auxiliaryState("X_marker")`
    - define `flux = X_sym + Rp * Q_sym`
    - apply via `systems::bc::applyCoupledNeumann(...)` with `auxiliaryODE(...)` registration(s)
  - Dep: expression-aware helper + new terminals + new registration model.

### 4.7 Tests (Systems)

- [ ] `Code/Source/solver/FE/Tests/Unit/Systems/test_CoupledBoundaryConditionHelpers.cpp`
  - Add new test cases for expression-aware overload:
    - boundaryIntegral-only flux expression
    - auxiliaryState-only flux expression
    - combined expression `X + Rp*Q`
  - Dep: new terminals + resolver.

- [ ] `Code/Source/solver/FE/Tests/Unit/Systems/test_CoupledBoundaryManager.cpp`
  - Update/extend to use RHS-only registrations (if legacy evolve callback is removed).
  - Dep: new registration model.

### 4.8 Build Plumbing

- [ ] `Code/Source/solver/FE/CMakeLists.txt`
  - Add new headers/sources/tests:
    - `Systems/ODEIntegrator.*`
    - `Systems/AuxiliaryStateBuilder.h` (if added)
    - new unit test files
  - Dep: file additions.

---

## 5) Compatibility / Migration Strategy

To avoid breaking existing coupled-BC code:

- Keep the current (v0) coupled helper overloads that accept:
  - `std::span<const constraints::CoupledNeumannBC>`
  - `std::span<const constraints::CoupledRobinBC>`
- Keep the current evolve-callback registration path temporarily as “legacy”, either by:
  1) preserving the old `AuxiliaryStateRegistration` struct name and adding the new fields, or
  2) providing `addAuxiliaryStateLegacy(...)` alongside the new RHS-only registration.

Deprecation plan (suggested):
- Mark legacy APIs as deprecated in docs, not via compiler attributes initially.
- Convert internal examples (Poisson demo) to the new API first.

---

## 6) Physics Layer (Optional Follow-On Work)

If the repo should include the Physics-side clean API:

- [ ] `Code/Source/solver/Physics/BoundaryConditions/LumpedParameterModel.h` (new)
  - Defines the abstract base model (physics concept).
  - Uses FE primitives for functionals/state/flux expression.

- [ ] `Code/Source/solver/Physics/BoundaryConditions/Windkessel.h/.cpp` (new)
  - Implements RCR using:
    - required boundary functional(s)
    - auxiliary ODE registration(s)
    - symbolic flux expression using FE terminals

- [ ] `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`
  - Replace manual RCR wiring with `windkessel::RCR::applyNeumann(...)`.

---

## 7) Open Design Decisions (Track Explicitly)

1) **Implicit ODE solve details**
   - Newton tolerances/iterations, derivative approximation, failure behavior.
2) **Scalar-only vs vector auxiliary state**
   - v1: scalar-only is simplest and matches proposed signatures.
3) **BDF2 with variable dt**
   - v1: constant dt; later: variable-step BDF2.
4) **Expression subset allowed in coupled flux**
   - v1: arithmetic + coefficients + min/max/abs + boundaryIntegral + auxiliaryState.
   - Throw for unsupported constructs with actionable error text.
5) **Cross-boundary dependencies**
   - allow `flux(marker A)` to depend on `boundaryIntegral(..., marker B)`? (useful for coupled outlets) vs forbid for simplicity.

---

## 8) Success Criteria

This work is complete when:

1) Physics code can define coupled Neumann/Robin BCs **without explicit loops**, using only:
   - `BoundaryFunctional` declarations (or `FormExpr::boundaryIntegral` placeholders),
   - `auxiliaryODE(...).withRHS(...).withIntegrator(...)` registrations,
   - a symbolic flux expression (`FormExpr`) composed from `boundaryIntegral` and `auxiliaryState`.
2) Coupled evaluation is two-phase:
   - compute integrals once
   - evolve aux state once
   - assemble PDE with consistent context
3) Unit tests cover:
   - ODEIntegrator methods
   - placeholder extraction + resolution
   - assembly correctness for a minimal mesh
