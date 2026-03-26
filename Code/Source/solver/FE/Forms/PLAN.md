# FE/Forms Subfolder — Design Plan

## Overview

`FE/Forms` is the FE library’s **mathematical notation layer**: a C++ EDSL for expressing weak forms using a compact, math-like vocabulary (e.g., `inner(grad(u), grad(v)).dx()`).

`FE/Forms` owns the **form vocabulary** (terminals, operators, measures) and the compilation boundary from “math-like syntax” to an **assembly-consumable kernel**:
- `forms::FormExpr` (runtime AST handle)
- `forms::FormCompiler` → `forms::FormIR`
- `forms::FormKernel` / `forms::NonlinearFormKernel` (both implement `assembly::AssemblyKernel`)

Historically, similar code lived as a prototype in `FE/Assembly/SymbolicAssembler.{h,cpp}`. That assembler now acts as a convenience wrapper that compiles `FormExpr` and delegates execution to `FE/Assembly::StandardAssembler`, while `FE/Forms` remains the canonical home of the vocabulary and compiler.

Inspired by UFL/FEniCS-style form languages, but implemented in C++ to integrate directly with the existing `FE/Systems` + `FE/Assembly` architecture (and to keep an option open for optional JIT code generation later).

---

## Core Philosophy: "Zero-Cost Abstraction"

The goal is to allow code that looks like math:
```cpp
const spaces::FunctionSpace& V = /* ... */;
auto u = FormExpr::trialFunction(V, "u");
auto v = FormExpr::testFunction(V, "v");
auto a = inner(grad(u), grad(v)).dx();
```

...to compile down to the exact same machine code as a hand-written raw loop:
```cpp
for (int q = 0; q < num_quad_points; ++q) {
    matrix_elem += (dN_dx * dN_dy) * weight[q];
}
```

Practically, the **initial implementation** prioritizes a stable vocabulary and correctness. Performance is pursued via precompilation/caching and (optionally) JIT specialization, while keeping a single stable user-facing front-end:
- **UFL-like runtime AST/IR (implemented):** operators build a `FormExpr` tree at runtime, which is compiled to `FormIR` and executed via `FormKernel` / `NonlinearFormKernel`.

---

## Scope of Responsibilities

1.  **Expression Definition**:
    - Provide symbolic placeholders (`TestFunction`, `TrialFunction`, `Coefficient`, `Constant`).
    - Provide differential operators (`grad`, `div`, `curl`) and algebraic operators (`inner`, `outer`, scalar `+/-/*`).
    - Provide measure indicators (`dx` for cell integration, `ds(boundary_marker)` for boundary integration, `dS` for interior-face integration).

2.  **Automatic Differentiation (The "Magic")**:
    - Support *automatic* Jacobian generation from a residual statement using AD (currently forward-mode dual numbers).
    - Keep an escape hatch for analytic tangents and/or symbolic differentiation where performance or robustness demands it.

3.  **Kernel Generation**:
    - Compile the expression tree to an assembly-consumable representation (`FormIR`) and execute it via an `assembly::AssemblyKernel` implementation.
    - Bridge the gap between abstract math and concrete data views (DoF values, basis/gradient data, quadrature weights) via `assembly::AssemblyContext`.

---

## Explicit Non-Goals (Scope Boundaries)

- **No DOF or constraint management:** Forms does not distribute DOFs, build sparsity patterns, or enforce constraints (belongs to `FE/Systems`, `FE/Dofs`, `FE/Constraints`, `FE/Sparsity`).
- **No assembly loops:** Forms does not traverse cells/faces or insert into global matrices/vectors (belongs to `FE/Assembly`).
- **No solver/time-step control:** Forms does not own Newton/Krylov/time-integration loops (belongs to solver layers and/or `FE/TimeStepping`).

---

## Core Components

### 1. Terminal Symbols
These are the leaves of the expression tree.
- **`FormExpr::testFunction(V, name)`**: Represents the basis function $v$ (indexed by the assembler’s local test dof index) and is explicitly bound to a `spaces::FunctionSpace`.
- **`FormExpr::trialFunction(V, name)`**: Represents the unknown field $u$ (indexed by the assembler’s local trial dof index) and is explicitly bound to a `spaces::FunctionSpace`.
- **`FormExpr::coefficient()`**: Represents a known scalar or vector function (evaluated at quadrature points).
- **`FormExpr::constant()`**: Simple scalar constants.
- **`FormExpr::identity()`**, **`FormExpr::normal()`**: Common geometry terminals.

### 2. Operators (Expression Nodes)
These classes combine terminals into trees.
- Unary ops: `grad`, `div`, `curl`, `jump`, `avg`, unary `-`.
- Binary ops: `+`, `-`, `*`, `inner`, `outer`.
- **`FormExpr::constitutive(model, input)`**: Type-erased material-point operator hook for future `FE/Constitutive` integration.

### 3. Integral Measures
- `.dx()`: cell integral
- `.ds(boundary_marker)`: boundary-face integral (marker = `-1` means “all markers”)
- `.dS()`: interior-face integral (DG)

### 4. Automatic Differentiation (AD) Engine
- **`forms::Dual`**: forward-mode dual number storing `{ value, d(value)/d(U_j) }` for the current element dofs.
- **`forms::DualWorkspace`**: per-thread scratch allocator for derivative storage used during `NonlinearFormKernel` evaluation.
- When evaluating a residual for a Jacobian, `TrialFunction` values are represented as `Dual` numbers; the chain rule propagates derivatives through the expression tree.

---

## Mixed-Source / Block-IR Architecture

`FE/Forms` is the **source representation layer** for multiphysics weak forms. The
library adopts a split where:

- **First-class mixed expressions** in `FE/Forms` are the canonical user-facing
  representation for coupled multiphysics problems.
- **Block decomposition** in `FE/Systems` and `FE/Backends` remains the stable
  execution, sparsity, solver, and preconditioner IR.

This means users can write one coupled mixed weak form, and the `Forms` module
is responsible for:

1. Expressing mixed weak forms with multiple test/trial spaces in one source
   expression.
2. Validating mixed-space semantics.
3. Compiling mixed source into `MixedFormIR` via `FormCompiler::compileMixed()`.
4. Preserving source-level provenance and whole-form metadata.

Below `Forms`, every mixed operator is representable as active block kernels
indexed by `(test_field, trial_field, domain)`. The block layout produced by
first-class mixed forms matches the block layout produced by equivalent manual
block decomposition.

For the full cross-module plan, see `FE/Docs/MixedForms/PLAN.md`.

### Compilation pathways

| Input | Compiler entry point | Output |
|-------|---------------------|--------|
| Single test/trial space | `compileLinear()` / `compileBilinear()` / `compileResidual()` | `FormIR` |
| Manual block decomposition | `compileBilinear(BlockBilinearForm)` / `compileLinear(BlockLinearForm)` | `vector<optional<FormIR>>` |
| Mixed expression (multiple test/trial spaces) | `compileMixed()` | `MixedFormIR` |

`MixedFormIR` is the explicit compiler bridge between mixed source and block
execution. It carries:
- test/trial field descriptors,
- active block map (block-sparse `(test_idx, trial_idx)` structure),
- per-block `FormIR`,
- whole-form metadata for diagnostics and setup heuristics.

---

## Interaction with FE/Systems

The `Forms` module sits on top of `Systems` and targets it as the **stable compilation target** (as described in `FE/Systems/PLAN.md`).

### Single-field path

1.  **User Code**:
    ```cpp
    const spaces::FunctionSpace& V = /* ... */;
    auto u = FormExpr::trialFunction(V, "u");
    auto v = FormExpr::testFunction(V, "v");
    auto form = inner(grad(u), grad(v)).dx();
    ```

2.  **Compilation / Lowering**:
    - Build a `FormExpr` tree, compile it to `FormIR`, and execute it via `FormKernel` / `NonlinearFormKernel` (both `assembly::AssemblyKernel` implementations).

3.  **Execution (Systems)**:
    - `FE/Systems` registers the resulting term(s) (test/trial coupling + kernel + domain).
    - `FE/Assembly` runs the appropriate cell/face loops, using `assembly::AssemblyContext` as the data source.

### Mixed-field path

1.  **User Code**:
    ```cpp
    auto W = std::make_shared<MixedSpace>();
    W->add_component("u", velocity_space);
    W->add_component("p", pressure_space);
    auto [u, p] = TrialFunctions(*W, {"u", "p"});
    auto [v, q] = TestFunctions(*W, {"v", "q"});
    auto residual = (inner(grad(u), grad(v)) - p * div(v) + div(u) * q).dx();
    ```

2.  **Compilation / Lowering**:
    - `FormCompiler::compileMixed()` analyzes bound test/trial spaces, decomposes
      terms by `(test_name, trial_name)` pair, and compiles each active block
      independently into `MixedFormIR`.

3.  **Installation (Systems)**:
    - `FormsInstaller::installFormulation()` or `installMixedFormIR()` binds block
      indices to concrete `FieldId`s and installs each active block through the
      existing operator-registration pathways (`addCellKernel`, etc.).

4.  **Execution (Assembly/Backends)**:
    - Assembly and backends continue to operate on block IR only — they are
      agnostic to whether the operator was authored as one mixed form or manual
      block expressions.

---

## Interaction with FE/Constitutive

The `Constitutive` module handles complex point-wise physics. `Forms` acts as the "glue."

```cpp
const spaces::FunctionSpace& V = /* ... */;
auto u = FormExpr::trialFunction(V, "u");
auto v = FormExpr::testFunction(V, "v");

auto F = grad(u);
auto P = FormExpr::constitutive(material_model, F); // material_model: shared_ptr<const ConstitutiveModel>
auto form = inner(P, grad(v)).dx();
```

-   The `Forms` layer doesn't need to know *how* `NeoHookean` works.
-   It just passes the `Dual` number version of `F` into the constitutive model.
-   The constitutive model returns the `Dual` stress (stress + tangent).
-   `Forms` continues the chain rule to assemble the Jacobian.

---

## Relationship to `FE/Assembly/SymbolicAssembler` (current codebase)

The repository previously contained a prototype “symbolic forms” pipeline in `FE/Assembly/SymbolicAssembler.{h,cpp}`. That code has been migrated and consolidated into `FE/Forms`:
- `FormExpr` (value-semantic expression handle)
- `FormCompiler` → `FormIR`
- `FormKernel` / `NonlinearFormKernel` (`assembly::AssemblyKernel` implementations)

`SymbolicAssembler` remains as an assembly convenience layer (compile a form and delegate to `StandardAssembler`), but **does not** own the vocabulary or compiler anymore.

This separation aligns with the broader module boundaries:
- `FE/Forms` owns “what is the weak form?”
- `FE/Assembly` owns “how is it executed over mesh entities and inserted globally?”

---

## API Sketch (The "User Experience")

```cpp
namespace svmp::FE::forms {

const spaces::FunctionSpace& V = /* ... */;
auto u = FormExpr::trialFunction(V, "u");
auto v = FormExpr::testFunction(V, "v");

// Weak form: (grad u, grad v) * dx
auto bilinear = inner(grad(u), grad(v)).dx();

// Residual-only API (Jacobian via AD):
auto f = FormExpr::coefficient("f", [](Real x, Real y, Real z) { return x + y + z; });
auto residual = (inner(grad(u), grad(v)) - f * v).dx();

} // namespace
```

---

## Implementation Roadmap

### Milestone 0: Make forms a first-class module (code movement / ownership)
- [x] Define the `FE/Forms` public headers and namespaces.
- [x] Migrate/re-export the existing `FormExpr`/`FormIR`/`FormCompiler` API from `FE/Assembly/SymbolicAssembler` into `FE/Forms` (so Assembly does not “own” the math vocabulary).

### Milestone 1: Core vocabulary + cell integrals
- [x] `Test`, `Trial`, `Coefficient`, `Constant`, `Identity`, `Normal`
- [x] Geometry terminals: `x`, `X`, `J`, `Jinv`, `detJ`, and basic entity measures (`h`, `vol(K)`, `area(F)`)
- [x] `grad`, `div`, `curl` (see `FUTURE_FEATURES.md` for higher-derivative plans)
- [x] Algebra/tensor operators needed for a practical PDE vocabulary:
  - `/`, `pow`, `min/max`, comparisons + `conditional`,
  - `transpose`, `trace`, `det`, `inv`, `cofactor`, `sym`, `skew`, `dev`, `norm`, `normalize`, `cross`,
  - `component(i[,j])` indexing.
- [x] measures: `dx`, `ds(boundary_id)`, `dS` (DG interior faces)
- [x] Compile standard elliptic bilinear forms to a kernel and assemble via `FE/Assembly` (and, by extension, registerable as `assembly::AssemblyKernel` terms in `FE/Systems`).

### Milestone 2: AD-backed Jacobians for nonlinear residuals
- [x] Residual-only user API with derived Jacobian via AD (`NonlinearFormKernel`).
- [x] Validate Jacobians by finite differences on small problems (unit tests).

### Milestone 3: DG and boundary-term completeness
- [x] `jump`, `avg`, normal `n` and interior-face measure `dS`.
- [x] Explicit trace restrictions `expr.minus()` / `expr.plus()` for DG coupling terms.
- [x] Basic DG helper combinators in `FE/Forms/Vocabulary.h` (e.g., `upwindValue`, `interiorPenaltyCoefficient`).
- [x] Interior-face compilation to the 4-block DG coupling outputs in `assembly::AssemblyKernel::computeInteriorFace`.

### Milestone 4: Constitutive integration
- [x] Type-erased constitutive wrapper and a stable “material-point” call boundary (`forms::ConstitutiveModel`).
- [x] Ensure AD scalar types propagate through constitutive models cleanly (via `Value<Dual>` evaluation).

### Milestone 5: First-class mixed expressions
- [x] `MixedSpace`, `TrialFunctions(W)`, `TestFunctions(W)` helpers for mixed authoring.
- [x] `FormCompiler::compileMixed()` lowering mixed source to `MixedFormIR`.
- [x] `MixedFormIR` with block-sparse structure, field descriptors, per-block `FormIR`.
- [x] Zero-block elimination as a required property of `MixedFormIR`.
- [x] `FormsInstaller::installMixedFormIR()` installing active blocks into `FESystem`.
- [ ] Whole-form metadata and source provenance in `MixedFormIR`.
- [ ] Block-to-source mapping for diagnostics.
- [ ] Parity tests: mixed expression vs manual block decomposition.

---

## Testing Strategy

- Unit tests for the form vocabulary and compilation:
  - expression-tree construction and string dumps,
  - required-data inference (`RequiredData` flags) vs expected,
  - AD Jacobian correctness via finite differences.
- Integration tests with `FE/Systems` + `FE/Assembly`:
  - Linear elliptic, nonlinear elliptic, and a simple DG term smoke test.

---

## Vocabulary Roadmap

For an extensive “future PDE vocabulary” checklist (including cross-module concepts and scoping), see `FE/Forms/VOCABULARY_ROADMAP.md`.

## Systems Integration Notes

For a concrete mapping from `FE/Forms` vocabulary → `FE/Systems` operator registration and `FE/Assembly` execution (including DG 4-block structure and `RequiredData` flow), see `FE/Forms/SYSTEMS_INTEGRATION.md`.

---

## References / Inspirations

### Form languages and FE DSLs

- Alnæs et al. — “Unified form language.” *ACM Transactions on Mathematical Software* (2014). DOI: 10.1145/2566630.
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the Finite Element Method* (2012) (“The FEniCS Book”).
- Rathgeber et al. — “Firedrake.” *ACM Transactions on Mathematical Software* (2016). DOI: 10.1145/2998441.

### C++ EDSL patterns and AD

- Prud’homme et al. — “Feel++ : A computational framework for Galerkin Methods and Advanced Numerical Methods.” *ESAIM: Proceedings* (2012). DOI: 10.1051/proc/201238024. (C++ embedded DSL patterns for FE.)
- Janssens et al. — “Finite Element Assembly Using an Embedded Domain Specific Language.” *Scientific Programming* (2015). DOI: 10.1155/2015/797325. (EDSL-style FE assembly.)
- Griewank & Walther — *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2008). DOI: 10.1137/1.9780898717761.
- Phipps, Pawlowski, Trott — “Automatic Differentiation of C++ Codes on Emerging Manycore Architectures with Sacado.” *ACM Transactions on Mathematical Software* (2022). DOI: 10.1145/3560262.

### Well-established FE libraries (design precedents)

- MFEM — Anderson et al. — “MFEM: A modular finite element methods library.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.06.009.
- deal.II — Bangerth, Hartmann, Kanschat (2007) and Bangerth, Heister, Harten (2021) library papers.
- libMesh — Kirk, Peterson, Stogner, Carey — “libMesh: a C++ library for parallel adaptive mesh refinement/coarsening simulations.” *Engineering with Computers* (2006). DOI: 10.1007/s00366-006-0049-3.
