# FE/Forms → FE/Systems Integration

This note explains how the `FE/Forms` vocabulary (UFL-like weak-form expressions) translates into the concrete `FE/Systems` assembly choices and data flow in this codebase.

## 1. The Compilation and Execution Pipeline

`FE/Forms` is a **front-end** that builds a runtime expression tree and compiles it into an `assembly::AssemblyKernel` that `FE/Systems` can register and assemble:

1. **User vocabulary / EDSL**: build a weak form with `forms::FormExpr` (and helpers from `Forms/Vocabulary.h`).
2. **Compile**: `forms::FormCompiler` lowers the expression to `forms::FormIR`:
   - splits a sum into individual integral terms,
   - records the integration domain for each term (`dx`, `ds(marker)`, `dS`),
   - infers `assembly::RequiredData` flags for each term and the combined form.
3. **Execute**: `forms::FormKernel` (linear/bilinear) or `forms::NonlinearFormKernel` (residual + AD Jacobian) evaluates the `FormIR` term-by-term against an `assembly::AssemblyContext`.
4. **Assemble**: `FE/Assembly` (e.g., `StandardAssembler`) iterates mesh entities, prepares `AssemblyContext` according to `RequiredData`, calls the kernel, and inserts local contributions into backend-neutral global views.

`FE/Systems` is the **orchestrator**: it decides which kernels are assembled for which operator (residual/Jacobian/mass/etc.), which fields/spaces they apply to, and on which domains (cells, boundary markers, interior faces).

## 2. How “Forms Vocabulary” Maps to AssemblyContext Data

`forms::FormExpr` nodes are evaluated at quadrature points using `assembly::AssemblyContext`. The compiler propagates requirements upward so the assembler can prepare only what is needed.

### 2.1 Symbols and basis data

- `TrialFunction` / `TestFunction`
  - In this codebase these are **explicitly bound** to a `spaces::FunctionSpace` at construction time (`FormExpr::trialFunction(V, ...)`, `FormExpr::testFunction(V, ...)`) so there is no ambiguity about the intended discrete space.
  - Map to basis values at quadrature points (`AssemblyContext::trialBasisValue`, `basisValue`).
  - `grad(TrialFunction)` / `grad(TestFunction)` map to physical gradients (`trialPhysicalGradient`, `physicalGradient`).
  - For residual forms, `TrialFunction` is interpreted as the current solution field `u_h` at quadrature points (`AssemblyContext::solutionValue`, `solutionGradient`).

- `dt(TrialFunction, k)`
  - Represents the **k-th continuous-time derivative** of the unknown in the weak form (UFL-style vocabulary).
  - `dt(·,k)` is **purely symbolic** at the Forms layer and is **not directly assemblable** in a steady context.
    - If a kernel encounters `dt(...)` during evaluation without a transient time-integration context, it must fail with:
      - `dt(...) operator requires a transient time-integration context`
  - The compiler records:
    - per-term `IntegralTerm::time_derivative_order`, and
    - form-level `FormIR::maxTimeDerivativeOrder()`.
  - `forms::FormKernel` / `forms::NonlinearFormKernel` propagate this to `assembly::AssemblyKernel::maxTemporalDerivativeOrder()`, so `FE/Systems` can detect that a registered operator is part of a transient formulation.
  - During transient assembly, `FE/Systems` (via `systems::TransientSystem` + a `systems::TimeIntegrator`) supplies an `assembly::TimeIntegrationContext` through `systems::SystemStateView::time_integration`, and `FE/Assembly` attaches it to each `assembly::AssemblyContext`.
    - For bilinear forms, `dt(TrialFunction,k)` contributes only the **current** stencil coefficient (history belongs to Systems/right-hand-side management).
    - For residual forms, `dt(TrialFunction,k)` evaluates to a linear combination of the current and historical solution values using the integrator-provided stencil coefficients.

### 2.2 Geometry and mapping terminals

These are taken directly from `AssemblyContext` (prepared by the assembler’s geometry mapping):

- `x()` / `FormExpr::coordinate()` → `AssemblyContext::physicalPoint(q)`
- `X()` / `FormExpr::referenceCoordinate()` → `AssemblyContext::quadraturePoint(q)`
- `J()` / `FormExpr::jacobian()` → `AssemblyContext::jacobian(q)`
- `Jinv()` / `FormExpr::jacobianInverse()` → `AssemblyContext::inverseJacobian(q)`
- `detJ()` / `FormExpr::jacobianDeterminant()` → `AssemblyContext::jacobianDet(q)`
- `n` / `FormExpr::normal()` → `AssemblyContext::normal(q)` (face contexts)
- `h()` / `FormExpr::cellDiameter()` → `AssemblyContext::cellDiameter()`
- `vol()` / `FormExpr::cellVolume()` → `AssemblyContext::cellVolume()` (cell contexts)
- `area()` / `FormExpr::facetArea()` → `AssemblyContext::facetArea()` (face contexts)

### 2.3 Algebra/tensor operators

Operators like `transpose`, `trace`, `det`, `inv`, `cofactor`, `sym`, `skew`, `dev`, `norm`, `component(i[,j])`, etc. are **pure expression-level transforms** that operate on the values returned by terminals/operators at quadrature points. They do not introduce new assembly responsibilities; they only change which `RequiredData` flags are needed to evaluate their operands.

### 2.4 Measures and integration domains

The measure wrappers determine which assembler loop is used:

- `.dx()` → cell loop, kernel method `computeCell`
- `.ds(marker)` → boundary-face loop filtered by marker, kernel method `computeBoundaryFace`
- `.dS()` → interior-face loop (DG), kernel method `computeInteriorFace` (4 blocks: mm/pp/mp/pm)

The assembler multiplies the scalar integrand by `AssemblyContext::integrationWeight(q)` (already including the physical measure).

## 3. How Forms Become Systems Operators

In `FE/Systems`, an “operator” (e.g., `Jacobian`, `Residual`, `Mass`) is an aggregation of terms. Each term is an `assembly::AssemblyKernel` that contributes on one or more domains.

`FE/Forms` contributes by producing kernels that are already `assembly::AssemblyKernel` objects:

- **Bilinear form** `a(u,v)` → `forms::FormKernel` that fills a local matrix.
- **Linear form** `L(v)` → `forms::FormKernel` that fills a local vector.
- **Residual form** `F(u;v)` → `forms::NonlinearFormKernel` that fills a local residual vector and a consistent local Jacobian via AD.

Typical `FE/Systems` registration pattern:

1. Pick a `FieldId` and its `FunctionSpace` (via `Spaces` + `Dofs`).
2. Compile your `FormExpr` to `FormIR`.
3. Wrap `FormIR` into a kernel (`FormKernel` or `NonlinearFormKernel`).
4. Register the kernel as a term for an operator:
   - `addCellKernel(op, field, kernel)` for `.dx()` terms,
   - `addBoundaryKernel(op, boundary_id, field, kernel)` for `.ds(marker)` terms,
   - `addInteriorFaceKernel(op, field, kernel)` for `.dS()` terms.

At assembly time, `FESystem::assemble(...)` selects the right assembler calls:
- cells → `Assembler::assembleMatrix/assembleVector/assembleBoth`
- boundary faces → `Assembler::assembleBoundaryFaces(mesh, marker, ...)`
- interior faces → `Assembler::assembleInteriorFaces(...)`

### 3.1 Mixed / Multi-field via block decomposition

The core `FormCompiler` intentionally supports **exactly one** `TrialFunction` and **exactly one** `TestFunction` per compiled kernel. Multi-field weak forms are therefore expressed as a **block decomposition**:

- Construct component arguments from a `spaces::MixedSpace` using `forms::TrialFunctions(W)` / `forms::TestFunctions(W)`.
- Represent the coupled operator using `forms::BlockBilinearForm` / `forms::BlockLinearForm` (one `FormExpr` per block).
- Compile per-block using:
  - `FormCompiler::compileBilinear(const BlockBilinearForm&)`
  - `FormCompiler::compileLinear(const BlockLinearForm&)`
  - `FormCompiler::compileResidual(const BlockBilinearForm&)` (residual contributions split by (test,trial) block)
- Register each compiled block kernel into `FE/Systems` with the appropriate `(row_field, col_field)` pair.

## 4. DG Trace Selection and Block Structure

Interior-face assembly produces four coupling blocks. `FE/Forms` provides two mechanisms to express DG terms:

- `jump(expr)` and `avg(expr)` (standard DG notation).
- Explicit restrictions `expr.minus()` / `expr.plus()` to select a side directly (UFL-style `expr('-')` / `expr('+')`).

These map naturally onto the 4-block evaluation in `computeInteriorFace`:
- “mm” block: test = minus, trial = minus
- “pp” block: test = plus,  trial = plus
- “mp” block: test = minus, trial = plus
- “pm” block: test = plus,  trial = minus

This is how a single `FormExpr` can describe DG coupling, while `FE/Assembly` and `FE/Systems` handle the insertion into the correct global matrix sub-blocks (for single-field DG, this is just the global matrix; for future block/mixed systems, it becomes block placement).

## 5. Constitutive Interop Boundary (future-facing)

`FormExpr::constitutive(model, input)` is the planned bridge to the future `FE/Constitutive` module:
- The constitutive model is evaluated at quadrature points and must support both `Real` and AD scalars (currently forward-mode dual numbers).
- `FE/Forms` remains responsible only for wiring constitutive calls into weak-form expressions; the model itself belongs to `FE/Constitutive`.

## References (design precedents)

- Alnæs et al. — “Unified form language.” *ACM Transactions on Mathematical Software* (2014). DOI: 10.1145/2566630.
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the Finite Element Method* (2012).
- Rathgeber et al. — “Firedrake.” *ACM Transactions on Mathematical Software* (2016). DOI: 10.1145/2998441.
- MFEM — Anderson et al. — “MFEM: A modular finite element methods library.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.06.009.
- Arndt et al. — “The deal.II finite element library: Design, features, and insights.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.02.022.
