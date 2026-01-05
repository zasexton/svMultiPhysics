# FE/Constitutive Subfolder — Design Plan

## Overview

`FE/Constitutive` provides the infrastructure for **local, point-wise models** used inside weak-form terms. A model is evaluated at integration points and maps **inputs** (field values/derivatives and optional parameters/state) to **outputs** consumed by `FE/Forms` expressions and/or hand-written `assembly::AssemblyKernel`s.

Within the `FE/` architecture this module is intentionally local and reusable:
- `FE/Systems` defines fields/operators and orchestrates assembly.
- `FE/Assembly` owns integration loops and local-to-global insertion.
- `FE/Forms` provides the user-facing EDSL and AD-backed Jacobian generation.
- `FE/Constitutive` provides model interfaces, adapters, and state/parameter plumbing so applications can define their own point-wise behavior without modifying core assembly code.

The guiding requirement is **domain-agnostic design**: APIs avoid assuming any particular equation set, variable meaning, or application interpretation.

---

## Core Philosophy: "Point-Wise Models with Explicit Contracts"

A constitutive model is treated as:
1. **A pure mapping** `input → output` at an integration point.
2. **Optionally stateful**, via an explicit state layout and state view.
3. **Linearizable**, either analytically or by evaluating with AD scalar types (as used by `FE/Forms`).

This makes models composable, testable, and usable from both the `Forms` and hand-written kernel paths.

---

## Scope of Responsibilities

1. **Model Call Boundary**
   - Define a stable, minimal evaluation interface callable at integration points.
   - Support both `Real` evaluation and AD-scalar evaluation (e.g., `forms::Dual`) so consistent Jacobians can be assembled.

2. **Input/Output Representation**
   - Use neutral containers/views for scalar/vector/matrix/tensor values.
   - Keep naming generic (`input`, `output`, `slot`, `kind`) rather than domain-specific labels.

3. **State Management (Optional)**
   - Provide a way for models to declare required per-integration-point state (`StateLayout`) and to access it via a `StateView`.
   - Keep allocation/ownership outside this module (owned by `FE/Systems` and surfaced through `FE/Assembly::AssemblyContext`).

4. **Parameter / Coefficient Access**
   - Provide a consistent way to read constant parameters and (future) spatially-varying coefficients, without tying the model to a particular storage backend.

---

## Explicit Non-Goals (Scope Boundaries)

- **No domain implementations:** no pre-packaged equation-specific models live here; application code provides models.
- **No mesh/DOF/constraint management:** nothing here distributes DOFs, owns constraints, or builds sparsity (belongs to `FE/Systems`, `FE/Dofs`, `FE/Constraints`, `FE/Sparsity`).
- **No integration loops:** no entity traversal or local-to-global insertion (belongs to `FE/Assembly`).
- **No nonlinear/time-stepping control:** no solver loops (belongs to solver layers / `FE/TimeStepping`).

---

## Core Components

### 1) Forms-Compatible Model Interface

`FE/Forms` already defines the type-erased call boundary `forms::ConstitutiveModel` used by `FormExpr::constitutive(model, input)`. `FE/Constitutive` should leverage this by providing **helper base classes and adapters** that make authoring models ergonomic and consistent.

Recommended helper pattern (CRTP) to avoid duplicating Real vs AD code:
- `constitutive::ModelCRTP<Derived>` implements the two virtual `forms::ConstitutiveModel::evaluate(...)` overloads.
- The derived type provides a single `template <class Scalar> evaluateImpl(...)` function.

### 2) Requirements Metadata

Models should be able to describe, in a lightweight way:
- expected input `Value::Kind` / shape,
- whether they require per-integration-point state,
- (future) any extra `assembly::RequiredData` beyond what `Forms` already requests through the expression tree.

This metadata is consumed by applications and by `FE/Systems` setup code, not by `FE/Assembly` at runtime.

### 3) State Layout and Views (Future)

Define:
- `StateLayout`: byte size, alignment, and optional named offsets for model-managed state.
- `StateView`: typed views into a state block at a single integration point.

`FE/Systems` is responsible for allocating `num_cells × num_qp × StateLayout` storage and making it accessible during assembly.

---

## Integration with Systems, Assembly, and Forms

**Design constraint (from `FE/Systems/PLAN.md`):** `FE/Systems` is the stable integration target. `FE/Constitutive` must be consumable via:
- `FE/Forms` (`FormExpr::constitutive(...)` lowered to kernels), and/or
- hand-written `assembly::AssemblyKernel` implementations.

1. **Ownership / Registration**
   - Model instances are owned by application terms (forms or kernels) as opaque dependencies.
   - `FE/Systems` registers only kernels/terms; it does not depend on model implementations.

2. **Information Flow**
   - `FE/Assembly::AssemblyContext` provides geometry and solution views requested by the kernel/form.
   - The kernel/form evaluates the model at each integration point and contributes to local residual/Jacobian blocks.

3. **State (future)**
   - During `systems::FESystem::setup()`, applications register required `StateLayout`s.
   - AssemblyContext surfaces a per-integration-point `StateView` when requested, so models can read/update state without global ownership.

---

## API Sketch (User-Facing Experience)

### Forms path (recommended)

```cpp
auto u = forms::FormExpr::trialFunction(V, "u");
auto v = forms::FormExpr::testFunction(V, "v");

auto model = std::make_shared<MyModel>(/* parameters */);
auto y = forms::FormExpr::constitutive(model, forms::grad(u));
auto form = forms::inner(y, forms::grad(v)).dx();
```

Then compile the form to a `forms::FormKernel` / `forms::NonlinearFormKernel` and register it with `FE/Systems` as an operator term (see `FE/Forms/SYSTEMS_INTEGRATION.md`).

### Hand-written kernel path

A kernel can call the same model interface directly at integration points, using `forms::Value<Real>` (and `forms::Value<Dual>` if it wants AD-driven linearization).

---

## Implementation Roadmap

### Milestone 1: Authoring ergonomics
- Add `constitutive::ModelCRTP` (or equivalent) to implement `forms::ConstitutiveModel` from a single scalar-templated implementation.
- Add small utilities for shape checking and error reporting.

### Milestone 2: State declarations and plumbing (cross-module)
- Define `StateLayout` / `StateView` in `FE/Constitutive`.
- Extend `FE/Systems` + `FE/Assembly::AssemblyContext` to allocate and expose per-integration-point state.

### Milestone 3: Parameter access conventions
- Define a minimal parameter access interface that can be backed by `systems::SystemStateView` and/or application-owned registries.

### Milestone 4: Performance hooks
- Optional batched evaluation APIs (vectorization-friendly).
- Optional caching policies (thread-safe and explicit).

---

## Testing Strategy

- Unit tests for the authoring helpers (`ModelCRTP`) and for toy models:
  - verify Real vs AD evaluation consistency,
  - verify derivative correctness via finite differences on small elements.
- Integration tests through `FE/Forms`:
  - assemble a bilinear form and a nonlinear residual/Jacobian that include `FormExpr::constitutive(...)`.

---

## References / Inspirations

### Form languages and integration-point hooks

- Alnæs et al. — “Unified Form Language.” *ACM Transactions on Mathematical Software* (2014). DOI: 10.1145/2566630.
- Rathgeber et al. — “Firedrake.” *ACM Transactions on Mathematical Software* (2016). DOI: 10.1145/2998441.
- Anderson et al. — “MFEM: A modular finite element methods library.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.06.009.
- Arndt et al. — “The deal.II finite element library: Design, features, and insights.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.02.022.
- Gaston et al. — “MOOSE: A parallel computational framework for coupled systems of nonlinear equations.” (2009). DOI: 10.1016/j.nucengdes.2009.05.021.

### Nonlinear solves and automatic differentiation

- Kelley — *Solving Nonlinear Equations with Newton’s Method* (2003).
- Deuflhard — *Newton Methods for Nonlinear Problems* (2004).
- Griewank & Walther — *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2008). DOI: 10.1137/1.9780898717761.
- Phipps, Pawlowski, Trott — “Automatic Differentiation of C++ Codes on Emerging Manycore Architectures with Sacado.” *ACM Transactions on Mathematical Software* (2022). DOI: 10.1145/3560262.
