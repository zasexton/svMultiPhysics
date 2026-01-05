# FE/Forms — UFL Vocabulary Implementation Checklist

This checklist is a **concrete, code-tracked map** of the proposed UFL-like vocabulary (from the project notes) to what is currently:
- **Implemented** (`[x]`): constructible in the `forms::FormExpr` EDSL, survives compilation (`FormCompiler` → `FormIR`), and is evaluatable/assemblable in supported contexts.
- **Planned / Not implemented yet** (`[ ]`): not yet available or not yet evaluatable in kernels.

Notes:
- Items owned by other modules are listed here for completeness, but are **not Forms responsibilities** (e.g., time stepping in Systems, constitutive laws in Constitutive, constraints in Constraints).
- Where an operator exists as an AST node but is not evaluatable yet, it is marked `[ ]` with a **PARTIAL** note.

## 1. Primitive Mathematical Types

- [x] `Scalar` (Real scalar expressions)
- [x] `ComplexScalar` (via explicit real/imag splitting and 2×2 real block lifting; see `FE/Forms/Complex.h`)
- [x] `Vector` (3D vector-valued expressions; unused components are zero)
- [x] `Tensor` (3×3 matrix-valued expressions; used as rank-2 tensor)
- [x] `SymmetricTensor` (as a matrix sub-kind via `sym(A)` / `SymmetricTensor(A)`; tagged as `SymmetricMatrix` at evaluation)
- [x] `SkewTensor` (as a matrix sub-kind via `skew(A)` / `SkewTensor(A)`; tagged as `SkewMatrix` at evaluation)
- [x] `FourthOrderTensor` (as `Value::Kind::Tensor4` + tensor4 coefficients + `doubleContraction(Tensor4,Tensor)` → `Tensor`)
- [x] `MixedType` (supported via block containers: `BlockBilinearForm` / `BlockLinearForm`; each block compiles independently)
- [x] `BlockType` (supported via block containers + complex 2×2 lifting helpers; see `FE/Forms/BlockForm.h`, `FE/Forms/Complex.h`)

## 2. Indexing & Tensor Algebra

- [x] `Index`, `IndexSet`, `EinsteinSummation` (via `forms::Index`, `forms::IndexSet`, `FormExpr::operator()(Index[,Index])`, and `forms::einsum(expr)` lowering)
- [x] `Contraction` / `DoubleContraction` (supported for scalar/vector/matrix; and `doubleContraction(Tensor4,Tensor)` → `Tensor` via a dedicated AST node)
- [x] `Trace` (`trace(A)`)
- [x] `Determinant` (`det(A)`)
- [x] `Inverse` (`inv(A)`)
- [x] `Transpose` (`transpose(A)`)
- [x] `Cofactor` (`cofactor(A)`)
- [x] `Deviator` (`dev(A)`)
- [x] `SymmetricPart` (`sym(A)`)
- [x] `SkewPart` (`skew(A)`)
- [x] `OuterProduct` (`outer(a,b)` for vectors)
- [x] `InnerProduct` (`inner(a,b)` for scalars/vectors/matrices)
- [x] `FrobeniusNorm` (`norm(A)` for matrices; `norm(v)` for vectors)

## 3. Mesh & Geometry

- [ ] `Mesh`, `Cell`, `Facet`, `Edge`, `Vertex` (owned by Mesh/Assembly)
- [ ] `Subdomain`, `Boundary`, `Interface` (owned by Mesh/Systems)
- [ ] `ReferenceCell`, `PhysicalCell`, `CellType` (owned by Elements/Mesh)
- [x] `x` / `Coordinate` (physical coordinate at quadrature points)
- [x] `X` / `ReferenceCoordinate` (reference coordinate at quadrature points)
- [x] `Jacobian` (`J`)
- [x] `JacobianInverse` (`Jinv`)
- [x] `JacobianDeterminant` (`detJ`)
- [x] `NormalVector` (`n`) (face contexts)
- [x] `CellDiameter` (`h`)
- [x] `CellVolume` (`vol(K)`)
- [x] `FacetArea` (`area(F)`)
- [ ] `TangentVector`, `Curvature`, `MetricTensor`, `ChristoffelSymbols` (planned; requires Assembly/Geometry support)

## 4. Function Spaces

- [x] `FunctionSpace` (owned by `FE/Spaces`; Forms binds `TrialFunction`/`TestFunction` to a `spaces::FunctionSpace`)
- [x] `H1Space` (exists in `FE/Spaces`; used by current Forms tests/examples)
- [x] `L2Space` (exists in `FE/Spaces`)
- [x] `HdivSpace` / `HDivSpace` (exists in `FE/Spaces`)
- [x] `HcurlSpace` / `HCurlSpace` (exists in `FE/Spaces`)
- [x] `MixedSpace`, `EnrichedSpace`, `TraceSpace`, `MortarSpace` (Spaces provided in `FE/Spaces`; Forms provides `TrialFunctions/TestFunctions(MixedSpace)` + block-form compilation helpers; full mortar/interface assembly loops are deferred to Systems/Assembly)

## 5. Discrete Functions / Fields

- [x] `Function` (symbolic handle concept via `TrialFunction`/`Coefficient`)
- [x] `TrialFunction(V)`
- [x] `TestFunction(V)`
- [x] `Coefficient` (scalar + vector coefficients)
- [x] `Constant`
- [ ] `Parameter` (planned; currently use `Constant` as runtime parameter)
- [ ] `StateVariable`, `ControlVariable`, `AdjointVariable` (planned; Systems/Optimization integration)
- [ ] `HistoryVariable` (planned; requires material-point state plumbing)
- [x] `component(i[,j])` (explicit component selection)

## 6. Differential Operators (Spatial)

- [x] `Gradient` (`grad(u)`)
- [x] `Divergence` (`div(u)`) — vector `Coefficient` supported; vector trial/test fields supported (vector-valued `ProductSpace`)
- [x] `Curl` (`curl(u)`) — vector `Coefficient` supported; vector trial/test fields supported (vector-valued `ProductSpace`)
- [x] `Hessian` (`H(u)` / `u.hessian()`) — scalar terminals supported (trial/test/constant/scalar coefficient); Hessian of general expressions pending
- [x] `Laplacian` — available for scalar terminals via `trace(H(u))` / `laplacian(u)` (`FE/Forms/Vocabulary.h`)
- [ ] `Biharmonic` (planned; typically expressed via integration by parts and/or requires higher derivatives)
- [ ] `DirectionalDerivative`, `CovariantDerivative`, `LieDerivative`, differential-forms operators (planned)

## 7. Differential Operators (Temporal)

- [x] `dt(u, k)` symbolic time derivative (`k>=1`) — Systems-owned lowering; not directly assemblable without transient context
- [ ] Other temporal-derivative semantics (deferred; not part of core Forms vocabulary)

## 8. Integral & Measure Operators

- [x] `dx` (cell integral)
- [x] `ds(boundary_marker)` (boundary integral)
- [x] `dS` (interior facet integral)
- [ ] `InterfaceIntegral`, `TimeIntegral` (planned)
- [ ] `QuadratureRule` / `QuadratureWeight` user control (planned; currently internal)
- [ ] `DiracDelta` / point/line/surface sources (planned)

## 9. Discontinuous Galerkin Operators

- [x] `Jump` (`jump(u)` / `[[u]]`)
- [x] `Average` (`avg(u)` / `{u}`)
- [x] Trace restrictions: `u.minus()` / `u.plus()`
- [ ] `WeightedAverage` (planned)
- [x] `UpwindValue`, `DownwindValue` (implemented as helpers `upwindValue(u,beta)` / `downwindValue(u,beta)` in `FE/Forms/Vocabulary.h`)
- [x] Basic interior-penalty scaling helper `interiorPenaltyCoefficient(eta,p)` (in `FE/Forms/Vocabulary.h`)
- [ ] `Penalty`, `NumericalFlux`, `NitschePenalty` higher-level helpers (planned)

## 10. Algebraic Operators

- [x] `Add`, `Subtract`, `Multiply`, `Divide`
- [x] `Power` (`pow`)
- [x] `AbsoluteValue` (`abs`)
- [x] `Sign` (`sign`)
- [x] `Minimum`, `Maximum` (`min/max`)
- [x] `Conditional` (`conditional(cond, a, b)`)
- [x] `Heaviside`, `IndicatorFunction` (implemented as helpers in `FE/Forms/Vocabulary.h`)

## 11. Constitutive Hooks

- [x] Generic `Constitutive(model, input)` hook (type-erased boundary; owned jointly with future `FE/Constitutive`)

## 12. Variational Structure

- [x] `LinearForm` / `BilinearForm` / `Residual` workflows (via `FormCompiler::compileBilinear` and `compileResidual`)

## 13. Automatic Differentiation

- [x] AD-backed Jacobians for residuals (forward-mode `Dual`)
- [ ] Higher-order AD (`Reverse`, `Taylor`) (planned; enum exists but not implemented)

## 14. Constraints & Boundary Conditions

- [ ] `DirichletCondition`, `NeumannCondition`, `RobinCondition`, etc. (owned by Constraints/Systems; Forms only expresses weak terms)

## 15. Multiphysics Coupling

- [x] `BlockForm` containers (`BlockBilinearForm` / `BlockLinearForm`) for multi-field assembly (compile per-block; Systems registers block kernels)
- [ ] `CoupledForm`, interface conditions, and higher-level coupling semantics (planned; owned by Systems)

## 16. Stabilization & Multiscale

- [ ] Stabilization and multiscale helper terms (deferred; expressed as explicit weak-form terms and configured by Systems)
