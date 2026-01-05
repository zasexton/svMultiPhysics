# FE/Forms — Vocabulary Expansion Roadmap

This document enumerates an **extensive candidate vocabulary** for expressing weak forms for a wide range of future PDEs (elliptic/parabolic/hyperbolic, mixed/DG, multiphysics, nonlinear/constitutive, space–time, optimization, stochastic, nonlocal, etc.).

It is intentionally broader than what `FE/Forms` will implement immediately. Items are grouped by concept and annotated with a suggested **module owner**:
- **Forms**: should likely be a first-class `forms::FormExpr` terminal/operator, or a helper combinator that returns a `FormExpr`.
- **Assembly**: should come from `assembly::AssemblyContext` / geometry mapping and appear in Forms as terminals (`x`, `n`, `J`, …).
- **Spaces/Systems**: should be owned by `FE/Spaces` and `FE/Systems` but may influence operator semantics (`grad`, `div`, traces).
- **Constraints/Constitutive/TimeStepping/Backends/Solvers**: cross-module concepts that Forms may *reference* but should not own.

## 0. Implemented Today (baseline)

`FE/Forms` currently provides a minimal but working baseline:
- Terminals:
  - symbols: `TrialFunction`, `TestFunction`, scalar/vector `Coefficient`, `Constant`,
  - higher-rank coefficients: matrix `Coefficient` and rank-4 tensor coefficients,
  - geometry: `x` (physical coordinate), `X` (reference coordinate), `J`, `Jinv`, `detJ`, `n`,
  - measures: `h` (cell diameter), `vol(K)` (cell volume), `area(F)` (facet measure),
  - constitutive hook: `Constitutive(model, input)`.
- Ops:
  - algebra: unary `-`, `+/-/*//`, `pow`, `min/max`, comparisons (`<,<=,>,>=,==,!=`), `conditional`,
  - calculus: `grad` (trial/test/scalar coeff), `div`/`curl` (vector trial/test + vector coeff), `H(·)` / `hessian(·)` (scalar terminals), `laplacian(u)=trace(H(u))` (helper for scalar terminals),
  - temporal: `dt(u,k)` (continuous-time operator; symbolic-only and requires a transient time-integration context to assemble),
  - indexing: `component(i[,j])`, plus Einstein-style indexed access `A(i[,j])` + lowering via `Index` / `IndexSet` and `einsum(expr)` (fully-contracted scalar expressions only),
  - tensor ops: `transpose`, `trace`, `det`, `inv`, `cofactor`, `sym`, `skew`, `dev`, `norm`, `normalize`, `cross`, and `doubleContraction` (including rank-4 : rank-2 → rank-2),
  - DG: `jump`, `avg`, and explicit trace restrictions `expr.minus()` / `expr.plus()`.
- Measures: `.dx()`, `.ds(boundary_marker)`, `.dS()`.
- AD-backed Jacobians: residual form → consistent Jacobian via forward-mode dual numbers (`NonlinearFormKernel`), including coverage for `exp/log/sqrt/pow/div`.
- Mixed/block and complex helpers:
  - block containers: `BlockBilinearForm` / `BlockLinearForm` (`FE/Forms/BlockForm.h`)
  - complex split + 2×2 real lifting: `ComplexScalar`, `ComplexBilinearForm`, `toRealBlock2x2` (`FE/Forms/Complex.h`)

Convenience combinators (UFL-like shorthands) live in `FE/Forms/Vocabulary.h` (e.g., `heaviside`, `clamp`, `upwindValue`, `interiorPenaltyCoefficient`).

## 1. Primitive Mathematical Types (Shapes)

Owner: **Forms** (shape system) + **Math** (storage/ops)

- `Scalar` (Real)
- `ComplexScalar` (complex Real)
- `Vector` (fixed dimension, e.g. 2D/3D; and possibly runtime dimension)
- `Tensor` (rank-2)
- `SymmetricTensor` (rank-2 symmetric storage + ops)
- `SkewTensor` (rank-2 skew storage + ops)
- `FourthOrderTensor` (rank-4)
- `MixedType` (tuples/products of heterogeneous shapes)
- `BlockType` / `BlockVector` / `BlockTensor` (for multi-field/multi-component coupling)
- `Boolean` and `Integer` expression scalars (for `conditional`, `indicator`, discrete toggles)

## 2. Indexing & Tensor Algebra

Owner: **Forms** (index objects + expression nodes) + **Math**

- `Index`, `FreeIndex`, `BoundIndex`
- `IndexSet`, `Component(i)`, `Component(i,j,...)`, slicing helpers
- `EinsteinSummation` / implicit summation over bound indices
- `Contraction` (single and multiple index contraction)
- `DoubleContraction` (`A:B`)
- `InnerProduct` (generalized dot / Frobenius inner product)
- `OuterProduct` (dyadic / tensor product)
- `HadamardProduct` (componentwise multiplication; useful in some nonlinearities)
- `Trace`, `Determinant`, `Inverse`, `Transpose`, `Cofactor`
- `Deviator` / `dev(A)`, `VolumetricPart` / `spherical(A)`
- `SymmetricPart` / `sym(A)`, `SkewPart` / `skew(A)`
- `FrobeniusNorm`, `L2Norm`, `LpNorm(p)`, `MaxNorm`
- `Normalize(v)`, `Unit(v)`
- `CrossProduct` (3D), `Wedge` (exterior product; if supporting differential forms)
- `LeviCivita` / permutation tensor (for compact curls/cross)
- `IdentityTensor(dim)`, `Zero(shape)`

## 3. Mesh & Geometry (Terminals from Context)

Owner: **Assembly** (data) + **Forms** (terminals)

### 3.1 Topology-level identifiers (mostly not Expressions)

Owner: **Systems/Assembly/Mesh**

- `Mesh`, `Cell`, `Facet`, `Edge`, `Vertex`
- `Subdomain`, `Boundary`, `Interface`
- `Manifold`, `EmbeddedDomain`
- `ReferenceCell`, `PhysicalCell`
- `CellType` / `ReferenceElement`

### 3.2 Geometric fields available at quadrature points

Owner: **Assembly** → surfaced in **Forms**

- Coordinates: `x` (physical), `X` (reference), `t` (time, if space–time)
- `NormalVector` (`n`), `TangentVector` (`t1`, `t2`), `ProjectionNormal`, `ProjectionTangent`
- Element/face sizes: `CellDiameter` (`h`), `CellVolume`, `FacetArea`, `EdgeLength`
- Mapping/Jacobian: `Jacobian` (`J`), `JacobianInverse` (`Jinv`), `JacobianDeterminant` (`detJ`)
- Mapping families (metadata): `AffineMapping`, `IsoparametricMapping`, `PiolaMapping`, `ALEMapping`
- Curvature and surface geometry (for manifolds): `Curvature`, `MetricTensor`, `ChristoffelSymbols`

## 4. Function Spaces (Types/Metadata)

Owner: **Spaces/Systems** (definitions), referenced by **Forms**

- `FunctionSpace`
- `ScalarSpace`, `VectorSpace`, `TensorSpace`
- `MixedSpace`, `EnrichedSpace`
- `BrokenSpace`, `DiscontinuousSpace`
- `ConformingSpace`, `NonconformingSpace`
- `H1Space`, `L2Space`, `HdivSpace`, `HcurlSpace`
- `TraceSpace`, `MortarSpace`
- `SpaceTimeSpace` (for space–time FE)

Forms implications:
- Operator legality depends on space (e.g., `div` meaningful for H(div), tangential traces for H(curl)).
- Traces/restrictions to facets should be explicit in the vocabulary (`trace(u)`, `u.minus()`, `u.plus()`).
- Mixed/multi-field forms are represented via **block decomposition**: `BlockBilinearForm` / `BlockLinearForm` + `TrialFunctions/TestFunctions(MixedSpace)`; each block is compiled independently.
- `MortarSpace` exists as a semantic space type for future interface coupling; mesh/interface ownership and assembly loops live in Systems/Assembly.

## 5. Discrete Functions / Fields

Owner: **Forms** (symbolic handles) + **Systems** (field registration)

- `Function` (generic FE function handle)
- `TrialFunction`, `TestFunction`
- `Coefficient` (space-/time-dependent known field), including vector/tensor coefficients
- `Constant`, `Parameter` (runtime scalar parameters)
- `StateVariable`, `ControlVariable`, `AdjointVariable`
- `HistoryVariable` (requires material-point state plumbing in Assembly/Systems)
- `AuxiliaryVariable` (post-processing fields, diagnostics)
- `FieldComponent` selection (`u[i]`, `u(i)`, `u.component(k)`)

## 6. Differential Operators (Spatial)

Owner: **Forms** (operators) + **Assembly/Spaces** (data support)

- `Gradient`, `Divergence`, `Curl`
- `Hessian`, `Laplacian`, `Biharmonic`
- `DirectionalDerivative(v, u)` / `D_v u`
- `CovariantDerivative` (manifold/curvilinear)
- `LieDerivative`
- Differential forms: `ExteriorDerivative`, `Codifferential`, `DifferentialForm`
- Mappings: `Pullback`, `Pushforward` (e.g., Piola transforms, differential forms on manifolds)

## 7. Differential Operators (Temporal)

Owner: **TimeStepping/Systems** + referenced by **Forms**

- `TimeDerivative` (`dt(u,k)` with `k>=1`)

Forms implication: these operators typically depend on `SystemStateView` (time, dt, previous solutions).

## 8. Integral & Measure Operators

Owner: **Forms** (measures) + **Quadrature/Assembly** (execution)

- `Integral`, `DomainIntegral` (`dx`), `BoundaryIntegral` (`ds`), `InterfaceIntegral`
- `InteriorFacetIntegral` (`dS`) (DG)
- `TimeIntegral` (`dt`), `SpaceTimeIntegral`
- `Measure`, `SubdomainMeasure`, marker/tag-based measures
- `QuadratureRule`, `QuadratureWeight` (internal; surfaced only for advanced control/debug)
- Singular sources:
  - `DiracDelta`
  - `PointSource`, `LineSource`, `SurfaceSource`

## 9. Discontinuous Galerkin (DG) & Interface Operators

Owner: **Forms** (operators/helpers) + **Assembly** (face loops)

- `Jump`, `Average`, `WeightedAverage`
- `UpwindValue`, `DownwindValue`
- `Penalty`, `InteriorPenalty`
- `NumericalFlux`, `InterfaceFlux`
- `NitschePenalty` / Nitsche consistency/symmetry terms as helpers
- Optional trace selectors: `u.minus()`, `u.plus()` (more explicit than overloading)

## 10. Algebraic & Nonlinear Scalar Operators

Owner: **Forms**

- `Add`, `Subtract`, `Multiply`, `Divide`
- `Power`, `Sqrt`, `Exp`, `Log`
- `AbsoluteValue`, `Sign`
- `Minimum`, `Maximum`, `Clamp`
- `Conditional` / `if_then_else`, `Heaviside`, `IndicatorFunction`
- Complex ops (if supporting complex): `Conjugate`, `RealPart`, `ImagPart`, `Arg`

## 11. Constitutive Hooks

Owner: **Constitutive** (models) + **Forms** (call boundary)

- `ConstitutiveOperator(model, input)` (type-erased boundary; must support Real and AD scalars)
- Optional material-point state/history accessors (future; owned by `FE/Constitutive` + `FE/Systems`)

Non-goal: FE/Forms does not define domain-specific named quantities; those belong in `FE/Constitutive` or application code.

## 12. Variational Structure

Owner: **Forms** (syntax) + **Systems** (operator registration)

- `Residual`, `LinearForm`, `BilinearForm`, `MultilinearForm`
- `WeakForm` / `StrongForm` (strong form typically used for diagnostics or auto-derivation)
- Functionals: `EnergyFunctional`, `ActionFunctional`, `DissipationPotential`
- Constraints: `ConstraintFunctional`, `Lagrangian`, `AugmentedLagrangian`

## 13. Differentiation / AD Vocabulary

Owner: **Forms** (operators) + **Systems/Solvers** (workflows)

- `GateauxDerivative`, `FréchetDerivative`
- `Jacobian`, `HessianOperator`, `TangentOperator`
- `Linearization`, `Sensitivity`
- `Adjoint`, `AdjointResidual`, `AdjointJacobian`

## 14. Constraints & Boundary Conditions

Owner: **Constraints/Systems** (strong/algebraic) + **Forms** (weak terms)

- Strong/algebraic: `DirichletCondition`, `PeriodicConstraint`, `MultiPointConstraint`
- Classification helpers: `EssentialConstraint` (strong), `NaturalConstraint` (weak)
- Natural/weak terms: `NeumannCondition`, `RobinCondition`, `MixedCondition`, `WeakBoundaryCondition`
- Contact/inequalities: `ContactConstraint`, `InequalityConstraint`, `ComplementarityCondition`

## 15. Multiphysics Coupling & Block Structure

Owner: **Systems** (block bookkeeping) + **Forms** (syntax for coupling terms)

- `CoupledForm`, `BlockForm`
- `InterfaceCondition`, `ContinuityConstraint`
- `MortarCoupling`, `LagrangeMultiplier`, `PenaltyCoupling`
- `OperatorSplitting`, `MonolithicCoupling`, `PartitionedCoupling`

## 16. Stabilization & Multiscale

Owner: **Systems** (configuration) + **Forms** (optional helper terms; deferred)

This vocabulary is intentionally deferred; where needed, stabilization is expressed as explicit weak-form terms and configured by Systems-level options.

## 17. Time Discretization (Variational / Space–Time)

Owner: **TimeStepping** + referenced by **Forms**

- `TimeSlab`
- `TimeTestFunction`, `TimeTrialFunction`
- `DiscontinuousGalerkinTime`
- `SpaceTimeResidual`

## 18. Optimization & Inverse Problems

Owner: **Solvers/Optimization module** + referenced by **Forms**

- `ObjectiveFunctional`, `ConstraintSet`, `ControlSpace`
- `ReducedFunctional`
- `GradientOperator`, `HessianOperator`
- Regularization: `TikhonovTerm`, `SparsityPenalty`, TV-like penalties

## 19. Stochastic & Uncertainty Quantification

Owner: **UQ module** + referenced by **Forms**

- `RandomVariable`, `RandomField`
- `Expectation`, `Variance`, `Covariance`
- `ProbabilityMeasure`
- `MonteCarloOperator`
- `PolynomialChaosExpansion`, `StochasticGalerkin`

## 20. Nonlocal & Integral PDEs

Owner: **Forms** (operators) + **Assembly** (quadrature support)

- `IntegralKernel`, `NonlocalOperator`
- `FractionalLaplacian`
- `PeridynamicOperator`
- `Convolution`, `MemoryKernel`

## 21. Meta / Compilation-Level Constructs

Owner: **Forms**

- `Expression`, `Symbol`, `ASTNode`
- `Shape`, `Rank`, `FreeIndex`, `BoundIndex`
- `EvaluationContext`, `AssemblyContext` (read-only view)
- `LinearizationPoint`
- `CodegenHint`, `OptimizationPass`

## 22. Solver-Interaction Vocabulary

Owner: **Systems/Backends/Solvers**

- `LinearOperator`, `NonlinearOperator`, `BlockOperator`
- `Preconditioner`, `SchurComplement`
- `MatrixFreeOperator`
- `ResidualEvaluator`

## 23. Diagnostics & Analysis

Owner: **Systems/Analysis module** + referenced by **Forms**

- `Norm`, `ResidualNorm`, `EnergyNorm`
- `ErrorEstimator`, `APosterioriIndicator`
- `GoalFunctional`

## References / Inspirations (vocabulary design)

- Alnæs et al. — “Unified form language.” *ACM Transactions on Mathematical Software* (2014). DOI: 10.1145/2566630.
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the Finite Element Method* (2012) (“The FEniCS Book”).
- Rathgeber et al. — “Firedrake.” *ACM Transactions on Mathematical Software* (2016). DOI: 10.1145/2998441.
- Prud’homme et al. — “Feel++ : A computational framework for Galerkin Methods and Advanced Numerical Methods.” *ESAIM: Proceedings* (2012). DOI: 10.1051/proc/201238024.
- Janssens et al. — “Finite Element Assembly Using an Embedded Domain Specific Language.” *Scientific Programming* (2015). DOI: 10.1155/2015/797325.
- MFEM — Anderson et al. — “MFEM: A modular finite element methods library.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.06.009.
- deal.II — Arndt et al. — “The deal.II finite element library: Design, features, and insights.” *Computers & Mathematics with Applications* (2021). DOI: 10.1016/j.camwa.2020.02.022.
