# JIT Compatibility Review — FE/Forms & Boundary Conditions

This document reviews the current FE infrastructure for **Forms** and **boundary-condition application** and lists the **concrete changes needed** to make physics formulations and BCs compatible with the planned LLVM OrcJIT backend while preserving the existing “math-first” `FormExpr` EDSL.

Reference: `LLVM_JIT_IMPLEMENTATION_PLAN.md` (planned OrcJIT backend, kernel ABI, codegen mapping).

---

## 0. Scope / Non-scope

**In scope**
- `FE/Forms` expression vocabulary (`FormExpr`), compilation (`FormCompiler` → `FormIR`), and execution (`FormKernels` interpreter) as it relates to JIT compatibility.
- Forms-level BC helpers (`FE/Forms/BoundaryConditions.h`) and Systems-level coupled BC orchestration (`FE/Systems/CoupledBoundary*`).
- Boundary functionals (`FE/Forms/BoundaryFunctional*`) and auxiliary state / ODE coupling (`FE/Systems/AuxiliaryState*`, `FE/Systems/ODEIntegrator*`) insofar as they feed boundary-condition evaluation.

**Out of scope**
- Implementing the OrcJIT engine/codegen itself (this is explicitly deferred).
- Broad performance refactors unrelated to enabling JIT (though this document calls out hot-path anti-patterns that JIT should eliminate).

---

## 1. Current Architecture (as implemented today)

### 1.1 Forms pipeline
- Physics builds `forms::FormExpr` trees using the EDSL (`+`, `*`, `grad`, `.dx()`, `.ds(marker)`, etc.).
- `forms::FormCompiler` lowers a “sum of integrals” expression into a `forms::FormIR`:
  - decomposes into `IntegralTerm{domain, marker, integrand}` but **keeps `integrand` as a `FormExpr` tree** (`Code/Source/solver/FE/Forms/FormIR.h`).
- `forms::FormKernel` / `LinearFormKernel` / `NonlinearFormKernel` interpret the integrand AST via `evalReal`/`evalDual` in `Code/Source/solver/FE/Forms/FormKernels.cpp`.

### 1.2 Boundary conditions
- Non-coupled weak BCs are “FormExpr-native” (e.g., `forms::bc::applyNeumann`, `applyRobin`, `applyNitscheDirichletPoisson`) and simply add boundary integrals via `.ds(marker)` (`Code/Source/solver/FE/Forms/BoundaryConditions.h`).
- Strong Dirichlet is represented as a Forms declaration (`forms::bc::StrongDirichlet`) and lowered by Systems to constraints (not part of the assembly kernel hot path).

### 1.3 Coupled boundary conditions (non-local integrals + auxiliary state)
- Coupled BCs are orchestrated by `systems::CoupledBoundaryManager`:
  - evaluates boundary functionals (integrals) using `assembly::FunctionalAssembler` + `forms::FunctionalFormKernel`,
  - updates `systems::AuxiliaryState` via `systems::ODEIntegrator`,
  - exposes a stable `constraints::CoupledBCContext` (`integrals`, `aux_state`, `t`, `dt`).
- Coupled BC weak terms are expressed in Physics via helper APIs in `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`.

---

## 2. JIT Compatibility Findings (what currently blocks “clean” JIT kernels)

The current infrastructure is already “JIT-adjacent” (clear vocabulary, `FormIR` decomposition, planned `JITOptions`), but several patterns prevent hot kernels from being expressible as straight-line LLVM IR without opaque runtime calls.

### 2.1 Type-erasure in hot paths (critical)

**Coefficients are `std::function`**
- `forms::ScalarCoefficient`, `TimeScalarCoefficient`, `VectorCoefficient`, … are `std::function` aliases (`Code/Source/solver/FE/Forms/FormExpr.h`).
- `FormKernels` evaluates `FormExprType::Coefficient` by calling the stored `std::function` per quadrature point (`Code/Source/solver/FE/Forms/FormKernels.cpp`).
- `grad(coefficient)` / `div(coefficient)` / `hessian(coefficient)` are implemented via finite differences calling the coefficient repeatedly in the quadrature loop (`FormKernels.cpp`).

**Coupled symbols are resolved into `std::function` coefficients**
- `systems::bc::resolveCoupledSymbols` replaces coupled placeholders with `FormExpr::coefficient(...)` nodes that capture `(ctx_ptr, idx)` and return `ctx_ptr->integrals.all()[idx]` / `ctx_ptr->aux_state.values()[idx]`.
- This still becomes an opaque `std::function` call from the kernel’s perspective.

**Constitutive models are virtual**
- `forms::ConstitutiveModel` is a virtual interface. The interpreter calls it via a `const ConstitutiveModel*` per quadrature point.
- `ConstitutiveEvalContext` uses string-keyed parameter getters (pointers to `std::function`) for model parameters (`Code/Source/solver/FE/Forms/ConstitutiveModel.h`).

Impact:
- The planned LLVMGen can *technically* lower these as external calls (trampolines), but:
  - calls are opaque (no inlining/CSE across boundary),
  - they inhibit vectorization,
  - they complicate caching/hashing (captured lambdas, pointer identity).

### 2.2 Runtime string operations and name-based lookup (should be setup-only)

While most name usage is setup-time, several components still depend on string lookup patterns that would be unacceptable inside JIT hot kernels:
- `forms::BoundaryFunctionalResults` and `systems::AuxiliaryState` both support name-based access via `unordered_map<std::string, …>` (`get(name)`, `operator[](name)`).
- `constraints::CoupledBCContext` exposes these containers directly; user callbacks can call `integrals.get("Q")` inside flux evaluation, reintroducing string lookup.
- Parameter access in constitutive models goes through string-keyed getters (`ctx.realParam("mu")`, etc.).

Target behavior for JIT:
- **All names must be resolved to integer slots at setup time.**
- Kernels should only do `array[slot]` loads/stores.

### 2.3 Dynamic containers / allocations in evaluation

The interpreter uses patterns that are fundamentally not JIT-kernel compatible:
- Recursive AST evaluation with per-node `childrenShared()` returning `std::vector` (potential allocation/copies, pointer chasing).
- `unordered_map` caches inside evaluation (e.g., constitutive call cache in `FormKernels.cpp`).
- `Value<T>` includes dynamic payload vectors for shapes larger than inline storage (`Code/Source/solver/FE/Forms/Value.h`).

JIT will replace interpreter evaluation, but “JIT compatibility” requires a clean separation: **no dynamic containers in the generated kernel ABI**.

### 2.4 Control flow and exception paths in hot loops

- The interpreter throws exceptions on unsupported nodes and invalid states (fine for interpreter; not acceptable inside a JIT kernel).
- Many runtime checks/branches exist per evaluation to validate shapes and contexts.

Target behavior for JIT:
- Validation must be done pre-JIT (compile-time checks); the generated kernel should be branch-minimal and exception-free.

---

## 3. Target Invariants for JIT-Ready Forms & BCs

These are the invariants the infrastructure should enforce whenever `jit.enable == true` (and ideally for the fast interpreter path as well):

1. **No type-erased calls in the inner quadrature loops** for “JIT-fast” mode.
2. **No string operations or map lookups in kernels** (names resolve to slots before compilation).
3. **Fixed-layout kernel arguments** (packed POD struct with aligned pointers/spans and scalar constants).
4. **No heap allocation inside kernels** (no vectors, no `shared_ptr`, no `unordered_map`).
5. **No exceptions inside kernels** (unsupported features are rejected before compilation; fallback to interpreter is explicit).
6. **Expression nodes lower to LLVM primitives** (loads/stores, fadd/fmul, calls to known intrinsics).
7. **Derivatives are explicit**:
   - either symbolically lowered (preferred),
   - or provided as explicit derivative expressions/functions,
   - avoid finite-difference differentiation inside kernels.

---

## 4. Checklist of Required Changes

This section is the actionable “to-do list” to make the existing infrastructure JIT compatible.

### Phase A — Make the expression vocabulary lowerable without opaque callbacks

- [x] Add **slot-based terminals** to `forms::FormExprType` and `forms::FormExpr`:
  - [x] `ParameterRef` (runtime scalar parameter by slot/index).
  - [x] `BoundaryIntegralRef` (coupled integral by slot/index).
  - [x] `AuxiliaryStateRef` (aux state by slot/index).
  - [x] `PreviousSolutionRef(k)` (direct time-history loads).
- [x] Ensure these terminals carry only POD payload (e.g., `uint32_t slot`) and are printable for diagnostics.
- [x] Add a `forms::jit::canCompile(expr|ir)` pre-check:
  - [x] rejects unsupported node types,
  - [x] rejects `Coefficient` nodes in strict JIT mode,
  - [x] rejects derivatives of external-call coefficients/models (requires explicit derivatives).
- [x] Add a “lower-to-kernel-IR” pass that converts a `FormExpr` integrand tree into a **flat, deterministic op list** (SSA/DAG):
  - [x] used by JIT codegen,
  - [x] used to compute a stable hash key for caching,
  - [x] enables CSE/hoisting before LLVM emission.

### Phase B — Replace name-based coefficient/parameter access with slot resolution

- [x] Introduce a **Real-parameter slot registry** at Systems setup time:
  - [x] assigns stable integer slots for Real parameters used by a form,
  - [x] builds the runtime “constants array” bound via `AssemblyContext::jitConstants()`.
- [ ] (Deferred) Introduce an external-call **coefficient registry/trampoline table** for `AllowExternalCalls` JIT mode.
- [x] Add Forms API that preserves expressiveness but avoids `std::function` in “JIT-fast” mode:
  - [x] `FormExpr::parameter("mu")` becomes `ParameterRef(slot)` after setup-time resolution.
  - [x] Keep `FormExpr::coefficient(name, std::function)` for interpreter/fallback, and clearly mark it **non-JIT-fast**.
- [x] Define coefficient “kinds” for JIT:
  - [x] `Constant` (compile-time immediate),
  - [x] `ParameterRef(slot)` (runtime constants array load),
  - [x] `InlinedExpr` (coefficient expressed directly as `FormExpr`),
  - [x] `ExternalCall` (callback-based `FormExprType::Coefficient`; allowed only in relaxed mode).

### Phase C — Coupled BC placeholder resolution must produce slot refs (not lambdas)

- [x] Change `systems::bc::resolveCoupledSymbols(...)`:
  - [x] resolve `boundaryIntegral(...)` to `BoundaryIntegralRef(slot)` terminals,
  - [x] resolve `auxiliaryState(...)` to `AuxiliaryStateRef(slot)` terminals,
  - [x] eliminate creation of `FormExpr::coefficient(...)` getters that capture `(ctx_ptr, idx)`.
- [x] Make `constraints::CoupledBCContext` expose **flat spans**:
  - [x] `integralsValues()`,
  - [x] `auxValues()`,
  - [x] `Real t, dt`.
- [x] Deprecate/label callback-based coupled BC definitions as **non-JIT-fast** (supported for interpreter/fallback only).

### Phase D — BoundaryFunctional evaluation must be JIT-compatible

- [x] Ensure boundary functional integrands can be lowered using the same kernel IR:
  - [x] scalar-only integrand (already a constraint),
  - [x] uses `DiscreteField`/`StateField` loads, geometry terminals, and slot-based parameters/coupled refs.
- [ ] (Later) Provide a JIT path for `forms::FunctionalFormKernel` so coupled integrals can be computed without interpreter recursion.

### Phase E — Auxiliary state ODE specification must be JIT-lowerable

- [x] Replace callback-based auxiliary RHS with a **FormExpr-based ODE spec**:
  - [x] `rhs : FormExpr` with terminals `AuxiliaryStateRef(slot)`, `BoundaryIntegralRef(slot)`, `t`, `dt`.
  - [x] optional Jacobian expression `d_rhs_dX : FormExpr` for implicit methods.
- [x] Change ODE execution API from name-based to slot-based:
  - [x] `ODEIntegrator::advance(method, slot, state, rhs_expr, …)`.
- [x] Remove finite-difference derivatives inside implicit ODE stepping for JIT-fast mode:
  - [x] require analytic Jacobian (or generate it symbolically).

### Phase F — Kernel ABI and AssemblyContext “view” for JIT

Align with the planned ABI in `LLVM_JIT_IMPLEMENTATION_PLAN.md`:

- [x] Define versioned POD `KernelArgs` structs (cell, boundary-face, and interior-face variants):
  - [x] `n_qpts`,
  - [x] output pointers (element vector/matrix),
  - [x] two-side contexts + 4 output blocks for DG interior faces,
  - [x] geometry/basis pointers (weights, x, J, detJ, normals, basis, grad basis, etc.),
  - [x] constants/parameters pointer (slot array),
  - [x] coupled context pointers (integrals/aux arrays),
  - [x] `user_data` pointer (for slow-path trampolines).
- [x] Add assembler-side code to **pack** an `AssemblyContext` into `KernelArgs`.
- [ ] (Perf) Guarantee 32/64-byte SIMD alignment for all packed arrays (currently: natural alignment).

### Phase G — Constitutive model integration (JIT boundary)

- [x] Decide/encode an initial JIT lowering rule for `FormExprType::Constitutive`:
  - [x] **External-call only** in relaxed mode (`AllowExternalCalls`; non-cacheable),
  - [x] **Inlinable model** lowering.
- [ ] Eliminate string-keyed parameter lookup on the hot path:
  - [ ] resolve model parameters to slots and pass via `KernelArgs` constants,
  - [ ] keep string lookup only in setup/validation.

#### Phase G.1 — Concrete steps for “inlinable” constitutive models (JIT-fast)

Goal: allow `FormExpr::constitutive(model, input)` to disappear as a virtual call boundary in strict JIT mode by rewriting it into a lowerable, cacheable, slot-based representation.

- [x] Define an explicit **inlinable constitutive contract** (opt-in):
  - [x] new interface (e.g., `forms::InlinableConstitutiveModel`) or hook on `forms::ConstitutiveModel` that can:
    - [x] build output expressions from symbolic inputs (`FormExpr`),
    - [x] declare whether it is **pure** (no state writes) vs **stateful** (updates `state_work`),
    - [x] expose a stable **model kind id** for caching/hashing (not pointer identity).
- [x] Add a setup-time **constitutive lowering pass** (like parameter/coupled resolution):
  - [x] traverse each integrand and detect `FormExprType::Constitutive`,
  - [x] if the model is inlinable: rewrite the subtree into plain `FormExpr` (no `Constitutive` node remains),
  - [x] if not inlinable:
    - [x] strict JIT mode rejects it,
    - [x] relaxed mode keeps it as an external call (existing behavior).
- [x] Make model parameters **slot-based** in the inlined representation:
  - [x] require models to declare `parameterSpecs()` (already supported),
  - [x] resolve keys → slots at `systems::FESystem::setup()` using `systems::ParameterRegistry`,
  - [x] forbid `ConstitutiveEvalContext::param("key")` usage in JIT-fast mode (string lookup),
  - [x] ensure inlined laws reference parameters only through `FormExpr::parameterRef(slot)` (or compile-time constants).
- [x] Decide how to represent **material state** for inlinable models:
  - [x] Milestone 1 (recommended): only **pure/stateless** models are inlinable (no state reads/writes).
  - [x] Milestone 2: enable **state reads** via fixed-offset loads:
    - [x] require `stateLayout()` (preferred) or a fixed `StateSpec` with documented offsets,
    - [x] add explicit, lowerable terminals for state access (e.g., `MaterialStateOldRef(offset_bytes, kind, count)`).
  - [x] Milestone 3: enable **state writes** as explicit stores:
    - [x] define a “state update IR” (separate from `FormExpr`, since writes are side effects),
    - [x] lower/execute state updates inside the kernel using `KernelArgs.material_state_*` pointers (see `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`).
- [x] Support **multi-output constitutive calls** (required for realistic models):
  - [x] implement `outputCount() > 1` + `evaluateNaryOutputs` for models that need it,
  - [x] make `FormExprType::ConstitutiveOutput` fully lowerable by inlining outputs (no runtime dispatch).
- [x] Define the **derivative strategy** for nonlinear assembly:
  - [x] Option A (first implementation): JIT emits a Dual-valued kernel path (mirrors current interpreter AD).
  - [ ] Option B (performance): models provide explicit tangent/derivative expressions and JIT emits a Real-only kernel + tangent.
- [ ] Make inlinable models **cacheable**:
  - [x] update `forms::jit::KernelIR` hashing so inlined constitutive expansions hash structurally (no node address usage),
  - [ ] include model kind id + state layout signature + resolved parameter slots in the cache key.
- [x] Add validation + tests:
  - [x] extend `forms::jit::canCompile` strict mode to accept inlinable constitutive expansions,
  - [x] add unit tests comparing interpreter vs inlined expansion for at least one pure model (e.g., NeoHookean or linear elastic),
  - [x] add tests that non-inlinable models are rejected in strict mode and accepted (non-cacheable) in relaxed mode.

### Phase H — Validation, fallback, and tests

- [x] Add “strict JIT” validation mode:
  - [x] fail fast when a form contains non-lowerable nodes or opaque callbacks,
  - [x] provide a clear diagnostic listing the first offending node/subexpression.
- [ ] Add a graceful fallback path (requires JIT engine integration).
- [x] Add unit tests for **JIT compatibility validation** (even before implementing OrcJIT):
  - [x] strict accepts slot-based integrands,
  - [x] strict rejects `std::function` coefficients,
  - [x] strict rejects unresolved coupled/parameter symbols.

---

## 5. Guidance for Physics Authors (keep the API “math-first”, but JIT-friendly)

Preferred patterns (JIT-fast):
- Express coefficients as `FormExpr` composed of terminals (`x`, `t`, parameters, coupled refs).
- Use coupled placeholders (`boundaryIntegral(...)`, `auxiliaryState(...)`) only in expression-building; rely on Systems to resolve them to slot refs.
- Treat names as **setup-time identifiers**, not runtime keys.

Acceptable fallback (JIT-compatible but slower):
- External coefficients / callbacks lowered via trampolines (opaque calls).

Avoid in kernels:
- Name-based lookup (`integrals.get("Q")`, `aux["X"]`, `param("mu")` in a callback).
- Finite-difference differentiation of user callbacks in the quadrature loop.

---

## 6. “Red Flags” to Grep For (current codebase)

These patterns typically indicate JIT incompatibilities if they occur in per-qpt/per-dof evaluation paths:

### Type erasure
- `std::function` (coefficients, coupled evaluators, ODE RHS).

### String lookup / dynamic maps
- `unordered_map<std::string` in anything reachable from kernel evaluation.
- `.get(std::string_view` / `indexOf(std::string_view` in hot evaluators.

### Dynamic allocation in hot loops
- `std::vector<...>` resized in evaluation loops.
- `childrenShared()` returning vectors per node evaluation (interpreter-specific, should disappear in JIT).

---

## 7. Notes on Sequencing

For the smoothest path to OrcJIT (and minimal disruption to Physics authoring):

1. Implement **slot-based terminals** + coupled symbol resolution to slot refs.
2. Implement **setup-time slot resolution** (parameter/coefficient registries).
3. Implement **kernel IR flattening + validation** (independent of LLVM).
4. Only then implement the OrcJIT engine/codegen (per `LLVM_JIT_IMPLEMENTATION_PLAN.md`).
