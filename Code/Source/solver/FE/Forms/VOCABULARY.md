# FE/Forms Vocabulary

This is the canonical vocabulary document for `FE/Forms`. It replaces the
previous split between `VOCABULARY.md`, `VOCABULARY_ROADMAP.md`,
`UFL_IMPLEMENTATION_CHECKLIST.md`, and `FUTURE_FEATURES.md`.

The goal of this file is to keep the public form-authoring surface, expert
hooks, and planned vocabulary in one current place. It is scoped to the FE form
language and the adjacent FE modules it calls into; mesh ownership, DOF
management, solver configuration, and time-integration orchestration remain in
their owning modules.

## Status Legend

| Status | Meaning |
|--------|---------|
| `implemented/public` | Intended for formulation authors and covered by current headers/tests |
| `implemented/expert` | Supported but meant for specialized, manual, or lower-level workflows |
| `implemented/internal` | Setup, JIT, lowering, or byte/slot based hooks; not an ergonomic public DSL |
| `planned` | Not implemented or not public enough to document as author-facing vocabulary |
| `out-of-scope-for-now` | Deferred until there is a concrete product/module need |

## Design Principles

Forms vocabulary should expose mathematical primitives, not physics recipes or
opaque numerical policies. A formulation file should make the weak form,
interface terms, and local closure choices visible to a reviewer.

Practical rules:

- Keep Forms physics-agnostic: prefer `grad`, `jump`, `normalTrace`,
  `projectTangent`, side restrictions, and local pointwise model hooks over
  names tied to one PDE family.
- Prefer explicit expression of flux models in Forms terms. For conservation
  laws, authors should write the central, upwind, penalty, or dissipative flux
  expression directly from `minus`, `plus`, `normal`, `jump`, `avg`, and local
  helper expressions.
- Do not add a generic core `numericalFlux(...)` abstraction just to hide the
  boilerplate. That name is too broad to communicate side conventions,
  orientation, conservation, wave-speed choice, or boundary-state handling.
- Add reusable Forms helpers only when they clarify mathematical structure or
  prevent orientation/trace mistakes. Trace/orientation primitives belong in
  Forms; Roe, HLLC, farfield, wall, or other physics-specific policies belong
  in physics modules built on top of Forms.
- Concise aliases are welcome only when they remain unambiguous and preserve a
  clear canonical spelling.

## Header Map

| Header | Role |
|--------|------|
| `Forms/FormExpr.h` | Core symbolic AST and low-level constructors/operators |
| `Forms/Vocabulary.h` | High-level UFL-style helpers for normal formulation authoring |
| `Forms/BoundaryConditions.h` | Weak and strong scalar trace helper functions |
| `Forms/StandardBCs.h` | Public BC wrapper types used by Systems |
| `Forms/NitscheBC.h` | Boundary/interface Nitsche BC wrappers |
| `Forms/InterfaceConditions.h` | Interface-face helper terms over `.dI(marker)` |
| `Forms/ConstitutiveModel.h` | Constitutive callback and material-state interface |
| `Forms/BoundaryFunctional.h` | Boundary/domain functional compilation |
| `Forms/FiniteDeformationForms.h` | Finite-deformation kinematic vocabulary |
| `Forms/MovingFrameForms.h` | Moving-frame helper terminals |
| `Forms/CutCellForms.h` | Cut-cell helper terminals |
| `Forms/FormCompiler.h`, `Forms/FormKernels.h`, `Forms/FormIR.h` | Expert compilation, kernel, and IR layers |
| `Forms/BlockForm.h`, `Forms/Complex.h` | Manual block and real-lifted complex workflows |
| `Forms/Index.h`, `Forms/Einsum.h`, `Forms/Tensor/*` | Indexed/tensor-calculus infrastructure |

## Canonical Workflow

Most residual physics should be authored with field-bound state/test symbols and
installed through Systems:

```cpp
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/FormsInstaller.h"
#include "FE/Systems/BoundaryConditionManager.h"

auto u_id = system.addField({.name = "u", .space = V});
system.addOperator("equations");

auto u = StateField(u_id, *V, "u");
auto v = TestField(u_id, *V, "v");

auto residual = (k * inner(grad(u), grad(v)) - f * v).dx();

bc_manager.applyAll(system, residual, u, v, u_id);
installFormulation(system, "equations", {u_id}, residual);
```

Use unbound `TrialFunction`/`TestFunction` for explicit operator authoring
(`installMixedBilinear`, `installMixedLinear`) and `StateField`/`TestField` for
residual formulations (`installFormulation`).

## Module Boundaries

`FE/Forms` owns symbolic expression construction, form IR, symbolic
differentiation, form compilation, AD/JIT evaluation support, and small
formulation vocabulary helpers.

`FE/Systems`, `FE/Assembly`, `FE/Spaces`, `FE/Mesh`, and time-stepping code own
field registration, DOF layout, boundary-condition installation, assembly
loops, quadrature/basis data, mesh topology, and time-state storage. Forms may
expose symbolic terminals for that data, but those modules provide the data.

## Implemented Vocabulary

### Fields and Symbols

Status: `implemented/public`, unless noted.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `StateField(field_id, V, name)` | `Vocabulary.h` | Canonical residual unknown bound to a registered `FieldId` |
| `TestField(field_id, V, name)` | `Vocabulary.h` | Canonical residual test function bound to the same `FieldId` |
| `StateFields(...)`, `TestFields(...)` | `Vocabulary.h` | Multi-field residual helpers |
| `TrialFunction(V, name)`, `TestFunction(V, name)` | `Vocabulary.h` | Operator authoring without `FieldId` binding |
| `TrialFunctions(W, names)`, `TestFunctions(W, names)` | `Vocabulary.h` | Mixed-space operator authoring |
| `FormExpr::trialFunction`, `testFunction`, `discreteField`, `stateField` | `FormExpr.h` | Lower-level factories, including signature overloads |
| `FormExpr::coefficient(name, ...)` | `FormExpr.h` | Scalar, time-scalar, vector, matrix, rank-3, and rank-4 coefficient callbacks |
| `FormExpr::constant(value)` | `FormExpr.h` | Literal scalar expression |
| `FormExpr::parameter(name)` | `FormExpr.h` | Named runtime/JIT parameter |
| `FormExpr::parameterRef(slot)` | `FormExpr.h` | `implemented/internal`; post-resolution slot reference |
| `FormExpr::typedZero()` | `FormExpr.h` | Typed zero expression for lowering/simplification paths |

Supported FE space concepts live in `FE/Spaces` and are consumable by Forms:
`FunctionSpace`, `H1Space`, `L2Space`, `HDivSpace`, `HCurlSpace`,
`MixedSpace`, `ProductSpace`, `TraceSpace`, `MortarSpace`, `C1Space`,
`CompositeSpace`, `EnrichedSpace`, `AdaptiveSpace`, `GenericBasisSpace`, and
`SpaceFactory`.

### Geometry and Time

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `x()` | Active physical coordinate; preserves legacy active-coordinate behavior |
| `X()` | Reference-cell quadrature coordinate, not a physical material coordinate |
| `currentCoordinate()` | Current physical coordinate |
| `previousCoordinate()` | Previous physical coordinate when bound |
| `referenceCoordinatePhysical()` | Physical reference-configuration coordinate |
| `meshDisplacement()` | Mesh displacement at quadrature points |
| `meshVelocity()` | Mesh velocity at quadrature points |
| `meshAcceleration()` | Mesh acceleration when bound |
| `previousMeshVelocity()`, `predictedMeshVelocity()` | Time-level mesh velocity terminals |
| `t()`, `deltat()`, `deltat_eff()` | Time, time step, and effective time step |
| `J()`, `Jinv()`, `detJ()` | Active geometry Jacobian, inverse, and determinant |
| `currentJacobian()`, `referenceJacobian()` | Frame-explicit geometry Jacobians |
| `currentJacobianDeterminant()`, `referenceJacobianDeterminant()` | Signed determinants of the frame-explicit geometry Jacobians |
| `FormExpr::identity()`, `FormExpr::identity(dim)` | Identity tensors |
| `FormExpr::normal()`, `currentNormal()`, `referenceNormal()` | Active and frame-explicit normals |
| `currentMeasure()`, `referenceMeasure()`, `surfaceJacobian()` | Frame-explicit integration measures |
| `currentSurfaceVector()`, `referenceSurfaceVector()` | Measure-scaled normal vectors |
| `pullback(expr, from, to)` | Metadata-only frame marker; runtime evaluation fails for non-identity transforms |
| `pushforward(expr, from, to)` | Metadata-only frame marker; runtime evaluation fails for non-identity transforms |
| `nanson(expr)`, `nanson()` | Explicit Nanson surface-vector transform using `cofactor(currentJacobian() * inv(referenceJacobian()))` |
| `h()`, `vol()`, `area()`, `domainId()` | Cell diameter, cell volume, facet area, cell-domain marker |
| `hNormal()` | Facet-normal cell size `2 * volume / area` |

The moving-domain terminals are physics-neutral. They expose assembly-provided
mesh/domain motion and frame-explicit geometry; they do not encode a specific
ALE, FSI, solid-mechanics, contact, or free-surface model.
For ALE forms, spell out transport and moving-volume terms directly, such as
`u - meshVelocity()` and `rho * div(meshVelocity()) * inner(u, v)`, so the
mathematics remains visible in formulation modules.

### Measures

Status: `implemented/public`.

| Vocabulary | Integral domain |
|------------|-----------------|
| `.dx()` | Cell/domain integral |
| `.ds(marker)` | Boundary integral; `marker = -1` means default/all as supported by assembly |
| `.dS()` | Interior facet integral |
| `.dI(interface_marker)` | Registered interface-face integral |

The compiler/IR distinguishes `Cell`, `Boundary`, `InteriorFace`, and
`InterfaceFace` terms.

### Differential Operators

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `grad(u)` | Scalar-to-vector, vector-to-matrix, and supported intrinsic vector FE gradients |
| `div(u)`, `curl(u)` | Vector differential operators |
| `hessian(u)` | Spatial Hessian support for supported scalar/composite expressions and mappings |
| `laplacian(u)` | `trace(hessian(u))` helper |
| `dt(u, order)` | Symbolic time derivative; requires transient context for evaluation |
| `surfaceGradient(f, n)` | Projected surface gradient helper |
| `surfaceDivergence(u, n)` | Projected surface divergence helper |
| `surfaceLaplacian(f, n)` | Surface Laplacian helper |
| `unitNormalFromLevelSet(phi)`, `meanCurvatureFromLevelSet(phi)` | Level-set geometry helpers |
| `safeNorm(v)`, `safeNormalize(v)` | Regularized vector norm helpers |

Intrinsic `H(div)` and `H(curl)` vector-basis gradients are matrix-valued with
layout component-by-physical-coordinate derivative. Affine mappings are
evaluated analytically. Supported non-affine 3D curved volume mappings include
the analytic derivative terms for the relevant Piola transforms. Unsupported
element-family, geometry-order, or lower-dimensional curved-frame combinations
fail with diagnostics instead of silently using an affine approximation.

### Algebra, Predicates, and Scalar Functions

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `+`, `-`, `*`, `/`, unary `-` | Scalar/vector/matrix expression arithmetic where shape-compatible |
| `inner(a, b)`, `dot(a, b)` | Inner product |
| `outer(a, b)`, `cross(a, b)` | Vector/tensor products |
| `doubleContraction(A, B)` | Rank-2 Frobenius contraction and rank-4/rank-2 contraction |
| `pow(a, b)`, `sqrt(a)`, `exp(a)`, `log(a)` | Scalar math |
| `abs(a)`, `sign(a)` | Nonsmooth scalar operators |
| `min(a, b)`, `max(a, b)` | Nonsmooth min/max operators |
| `lt`, `le`, `gt`, `ge`, `eq`, `ne` | Predicate nodes returning scalar 0/1 values |
| `conditional(cond, then_expr, else_expr)` | Branching expression |
| `heaviside(a)`, `indicator(predicate)`, `clamp(a, lo, hi)` | Convenience combinators |
| `regionIndicator(domain_id)` | Domain-id predicate helper |

The exact nonsmooth operators are implemented. Smooth regularizations are
separate vocabulary:

| Vocabulary | Notes |
|------------|-------|
| `smoothAbs(a, eps)` | Regularized absolute value |
| `smoothSign(a, eps)` | Regularized sign |
| `smoothHeaviside(a, eps)` | Regularized Heaviside |
| `smoothMin(a, b, eps)` | Smooth minimum |
| `smoothMax(a, b, eps)` | Smooth maximum |

Additional trigonometric, hyperbolic, inverse-trigonometric, and special
functions are still planned; see `Planned Vocabulary`.

### Tensor Algebra and Constructors

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `as_vector({ ... })`, `FormExpr::asVector(...)` | Pack components into a vector expression |
| `zeroVector(dim)` | Vector of `dim` scalar zero constants |
| `as_tensor({ ... })`, `FormExpr::asTensor(...)` | Pack rows into a matrix/tensor expression |
| `component(a, i[, j])`, `a.component(i[, j])` | Component extraction |
| `transpose(A)`, `trace(A)`, `det(A)`, `inv(A)` | Matrix/tensor operations |
| `cofactor(A)`, `dev(A)`, `sym(A)`, `skew(A)` | Tensor decomposition helpers |
| `norm(a)`, `normalize(a)` | Vector/matrix norm helpers |
| `SymmetricTensor(A)`, `SkewTensor(A)` | Semantic aliases around `sym(A)` and `skew(A)` |
| `contraction(a, b)` | Shape-dependent product helper |
| `FormExpr::coefficient(..., Tensor3Coefficient)` | Rank-3 coefficient data |
| `FormExpr::coefficient(..., Tensor4Coefficient)` | Rank-4 coefficient data |

The public value model covers real scalars, 3D vectors, 3x3 matrices, symmetric
and skew matrix tags, rank-3 coefficients, and rank-4 coefficients with
dedicated contraction support. True tensor-valued unknown fields and broad
higher-rank public tensor calculus are not yet first-class authoring vocabulary.

### Indexed Notation

Status: `implemented/public` for the listed `Index`/`einsum` path; tensor
variance metadata is `implemented/expert`.

| Vocabulary | Notes |
|------------|-------|
| `Index`, `IndexSet` | Public symbolic index ids/extents |
| `A(i)`, `A(i, j)`, `A(i, j, k)`, `A(i, j, k, l)` | Indexed access nodes |
| `einsum(expr)` | Einstein lowering |
| `einsum(expr, auto_extent)` | Einstein lowering with auto-extent behavior |
| `tensor::TensorIndex`, `MultiIndex` | Expert tensor-calculus metadata path |
| `special::delta`, `special::levicivita`, `special::levicivita2d` | Expert/internal special-tensor infrastructure |

Current public lowering supports scalar outputs from fully contracted
expressions and vector/matrix outputs with up to two free indices. Each index id
may appear once as a free index or twice as a bound index in the supported
patterns.

### Matrix, Eigen, Spectral, and History Operators

Status: `implemented/public` for authoring; supported dimensions and
differentiability are validated by the relevant evaluators/JIT paths.

| Vocabulary | Notes |
|------------|-------|
| `expm(A)`, `A.matrixExp()` | Matrix exponential |
| `logm(A)`, `A.matrixLog()` | Matrix logarithm |
| `sqrtm(A)`, `A.matrixSqrt()` | Matrix square root |
| `powm(A, p)`, `A.matrixPow(p)` | Matrix power |
| `matrix*DirectionalDerivative(A, dA)` | Directional derivatives for matrix functions |
| `A.symmetricEigenvalue(which)` | Symmetric eigenvalue |
| `eigenvalue(A, which)`, `A.eigenvalue(which)` | General eigenvalue helper |
| `eigvec_sym(A, which)`, `A.symmetricEigenvector(which)` | Symmetric eigenvector |
| `spectralDecomp(A)`, `A.spectralDecomposition()` | Spectral decomposition/eigenvectors-as-columns |
| `symmetricEigen*DirectionalDerivative(...)` | Directional derivatives for supported eigen operations |
| `FormExpr::previousSolution(k)` | Previous solution value, `k >= 1` |
| `FormExpr::historyWeightedSum(weights)` | Weighted history sum |
| `FormExpr::historyConvolution(weights)` | History convolution helper |

### DG, Trace, and Interface Primitives

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `u.minus()`, `u.plus()`, `minus(u)`, `plus(u)` | Interior/interface side restrictions |
| `jump(u)`, `avg(u)` | DG jump and average |
| `weightedAverage(u, w_plus, w_minus)` | Weighted side average helper |
| `harmonicAverage(k)` | Harmonic side average helper |
| `upwindValue(u, beta)`, `downwindValue(u, beta)` | Basic DG upwind/downwind selection |
| `interiorPenaltyCoefficient(eta, p)` | SIPG-style penalty scaling helper |
| `normalComponent(u)` | Scalar normal trace `inner(u, n)` |
| `interfaceNormalComponent(u, reduction)` | Interface scalar normal trace helper |
| `applyInterfaceScalarTrace(...)` | Minus/plus/jump/average trace reduction |
| `bc::ScalarTraceOperator` | `Identity` or `NormalComponent` scalar trace selection |
| `bc::InterfaceTraceReduction` | `Minus`, `Plus`, `Jump`, or `Average` interface reduction |
| `.dI(marker)` | Interface-face integration measure |

This is enough for current weak trace loads, trace Robin terms, Nitsche terms,
interface exchange, and simple DG terms. First-class normal/tangential trace
families remain planned.

### Boundary Conditions and Constraints

Status: `implemented/public`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `bc::StrongDirichlet`, `strongDirichlet(...)` | `BoundaryConditions.h` | Strong scalar/vector value constraints |
| `applyNeumann(...)`, `applyNeumannValue(...)` | `BoundaryConditions.h` | Weak boundary load terms |
| `applyRobin(...)`, `applyRobinValue(...)` | `BoundaryConditions.h` | Weak Robin terms |
| `traceInequalityViolation(...)`, `applyTraceInequality(...)` | `BoundaryConditions.h` | One-sided scalar trace laws |
| `applyTraceNitsche(...)` | `BoundaryConditions.h` | Boundary Nitsche scalar trace imposition |
| `applyInterfaceTraceNitsche(...)` | `BoundaryConditions.h` | Interface Nitsche scalar trace imposition |
| `applyNitscheDirichletPoisson(...)` | `BoundaryConditions.h` | Legacy Poisson-specific Nitsche helper; prefer explicit `applyTraceNitsche(...)` |
| `buildTraceNitschePenalty(...)` | `BoundaryConditions.h` | Penalty helper using trace options |
| `bc::NitscheVariant`, `NitscheDirichletOptions` | `BoundaryConditions.h` | Symmetric/nonsymmetric/skew-symmetric Nitsche options |
| `TraceInequalitySense`, `TraceInequalityOptions` | `BoundaryConditions.h` | One-sided trace-law options |
| `EssentialBC`, `NaturalBC`, `RobinBC` | `StandardBCs.h` | Standard public BC wrappers |
| `NormalTraceEssentialBC` | `StandardBCs.h` | Strong `H(div)` normal-trace data |
| `TraceLoadBC`, `TraceRobinBC`, `TraceInequalityBC` | `StandardBCs.h` | Boundary scalar trace wrappers |
| `InterfaceTraceLoadBC`, `InterfaceTraceRobinBC`, `InterfaceTraceJumpPenaltyBC` | `StandardBCs.h` | Interface scalar trace wrappers |
| `make*BC(...)` helpers | `StandardBCs.h` | Factory helpers for the standard BC wrappers |
| `ScalarNitscheBC`, `TraceNitscheBC`, `InterfaceTraceNitscheBC` | `NitscheBC.h` | Nitsche wrapper types; `ScalarNitscheBC` is a legacy scalar-diffusion convenience |
| `interface::NitscheVariant`, `NitscheInterfaceOptions` | `InterfaceConditions.h` | Interface Poisson Nitsche options |
| `interface::nitschePoissonIntegrand(...)` | `InterfaceConditions.h` | Legacy interface Poisson integrand helper; prefer explicit interface trace terms |
| `interface::applyNitscheInterfacePoisson(...)` | `InterfaceConditions.h` | Legacy interface Poisson helper over `.dI(marker)` |

Use equality/Robin/Nitsche helpers for equality-style laws such as
`tau(u) = g` or weak continuity. Use `TraceInequalityBC` for one-sided laws
whose active set can change during nonlinear solve.

Advanced `H(div)` trace infrastructure also exposes periodic/MPC pairing
helpers and marker-scoped mortar/interface field workflows through the Spaces
and Systems layers. See `Docs/HDIV_ADVANCED_USAGE_GUIDE.md` for those flows.

### Installation and Compilation

Status: `implemented/public` for the Systems install API; compiler/IR APIs are
`implemented/expert`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `installFormulation(system, op, fields, residual)` | `Systems/FormsInstaller.h` | Canonical residual installation |
| `installStrongDirichlet(system, bcs)` | `Systems/FormsInstaller.h` | Strong constraint installation |
| `installMixedBilinear(...)` | `Systems/FormsInstaller.h` | Explicit bilinear block installation |
| `installMixedLinear(...)` | `Systems/FormsInstaller.h` | Explicit linear block installation |
| `installMixedFormIR(...)` | `Systems/FormsInstaller.h` | Expert precompiled IR installation |
| `BoundaryConditionManager::applyAll(...)` | `Systems/BoundaryConditionManager.h` | One-call BC workflow |
| `FormCompiler::compile*`, `compileMixed(...)` | `Forms/FormCompiler.h` | Expert compile-without-install workflows |
| `FormIR`, `MixedFormIR`, `FormKernels` | `Forms/*IR.h`, `Forms/FormKernels.h` | Expert IR/kernel surfaces |

### Symbolic Differentiation and AD

Status: `implemented/public` for normal residual-to-Jacobian workflows and
`implemented/expert` for direct symbolic differentiation helpers.

| Vocabulary | Notes |
|------------|-------|
| Residual form to Jacobian | Forward-mode dual evaluation in nonlinear kernels |
| `differentiateResidual(...)` | Symbolic residual differentiation helper |
| `directionalDerivativeWrtField(...)` | Directional derivative helper |
| `differentiateResidualHessianVector(...)` | Hessian-vector helper |
| `differentiateWrtAuxiliaryOutput(...)` | Auxiliary-output sensitivity helper |
| `simplify(...)`, `extractTermsReferencing(...)` | Symbolic utility helpers |
| `checkSymbolicDifferentiability(...)` | Validation helper |

### Auxiliary Coupling and Boundary Functionals

Status: `implemented/public` for named authoring helpers; slot references are
`implemented/internal`.

| Vocabulary | Notes |
|------------|-------|
| `AuxiliaryInput(name)` | Preferred named auxiliary input reference |
| `AuxiliaryOutput(name)` | Preferred named auxiliary model output reference |
| `AuxiliaryOutput(instance, name)` | Instance-qualified auxiliary model output |
| `AuxiliaryState(name)` | Raw auxiliary state access; advanced path |
| `AuxiliaryInputSlot(slot)`, `AuxiliaryOutputSlot(slot)` | Post-resolution slot references |
| `FormExpr::auxiliaryStateRef(slot)` | Internal slot reference |
| `BoundaryFunctional` | Boundary/domain functional definition |
| `BoundaryFunctionalResults` | Runtime functional output container |
| `compileBoundaryFunctionalKernel(...)` | Boundary/domain functional compilation |
| `FormExpr::boundaryIntegral(...)` | Expert/legacy boundary-functional symbol |
| `FormExpr::boundaryIntegralRef(slot)` | Internal/legacy slot reference |

Legacy `boundaryIntegralValue`/`boundaryIntegralRef` terminology should be
treated as superseded by `AuxiliaryInput`/`AuxiliaryInputSlot`.

### Constitutive Vocabulary

Status: `implemented/public` for calls and indexed outputs; state byte-offset
terminals are `implemented/internal`.

| Vocabulary | Notes |
|------------|-------|
| `constitutive(model, input)` | Unary constitutive call |
| `constitutive(model, {inputs...})` | N-ary constitutive call |
| `constitutive(model, a0, a1, ...)` | Variadic n-ary helper |
| `ConstitutiveCall::out(i)`, `output(i)` | Indexed multi-output selection |
| `FormExpr::constitutiveOutput(call, i)` | Lower-level multi-output selection |
| `ConstitutiveModel::evaluate*` | Real and dual constitutive evaluation hooks |
| `outputCount()`, `outputSpec(i)`, expected input metadata | Constitutive metadata |
| `stateLayout()`, `stateVariables()` | Constitutive state layout metadata |
| `parameterSpecs()` | Constitutive parameter metadata |
| `materialStateOldRef(offset_bytes)` | Internal byte-offset state terminal |
| `materialStateWorkRef(offset_bytes)` | Internal byte-offset state terminal |

Named material-state vocabulary is planned. Public form code should not need to
author raw byte offsets.

### Finite Deformation

Status: `implemented/public` helper header.

`Forms/FiniteDeformationForms.h` provides finite-deformation kinematic helpers:

| Vocabulary | Notes |
|------------|-------|
| `deformationGradient(u)` | `F` |
| `deformationGradientVariation(du)` | Variation of `F` |
| `finiteDeformationDimension(u[, dim])` | Dimension inference/helper |
| `jacobian(F)` | `det(F)` helper in finite-deformation namespace |
| `inverseDeformationGradient(F)` | `F^{-1}` |
| `inverseTransposeDeformationGradient(F)` | `F^{-T}` |
| `rightCauchyGreen(F)`, `leftCauchyGreen(F)` | `C` and `b` |
| `greenLagrangeStrain(F)`, `almansiStrain(F)` | Strain measures |
| `kinematics(u)` | Bundled finite-deformation expressions |
| `linearizeKinematics(u, du)` | Bundled variation expressions |
| `jacobianVariation(F, dF)` | Variation of `det(F)` |
| `inverseVariation(F, dF)` | Variation of `F^{-1}` |
| `rightCauchyGreenVariation(F, dF)` | Variation of `C` |
| `leftCauchyGreenVariation(F, dF)` | Variation of `b` |
| `greenLagrangeVariation(F, dF)` | Variation of Green-Lagrange strain |
| `almansiVariation(F, dF)` | Variation of Almansi strain |
| `scalarCurrentGradientFromReferenceGradient(...)` | Reference/current scalar gradient transforms |
| `vectorCurrentGradientFromReferenceGradient(...)` | Reference/current vector gradient transforms |
| `pushForwardVector(...)`, `pullBackVector(...)` | Vector frame transforms |
| `contravariantPiolaPushForward(...)`, `covariantPiolaPushForward(...)` | Piola transforms |
| `nansonMeasureVector(...)`, `nansonMeasureVectorVariation(...)` | Surface measure transforms |
| `pk1InternalVirtualWorkDensity(...)` | Legacy mechanics shortcut; prefer writing the density explicitly in physics modules |
| `initialStressGeometricStiffnessDensity(...)` | Legacy mechanics shortcut; prefer writing the density explicitly in physics modules |

### Moving Frame

Status: `implemented/public` helper header.

`Forms/MovingFrameForms.h` provides a small parameter-slot vocabulary for
prescribed moving reference frames:

| Vocabulary | Notes |
|------------|-------|
| `MovingFrameParameterSlots` | Slot layout constants |
| `frameVectorParameter(slot)` | Read a vector from scalar parameter slots |
| `frameOrigin()`, `frameLinearVelocity()` | Translational frame data |
| `frameAngularVelocity()` | Angular velocity |
| `frameLinearAcceleration()`, `frameAngularAcceleration()` | Acceleration data |
| `frameVelocityAtCurrentCoordinate()` | Rigid-frame velocity at `currentCoordinate()` |
| `frameAccelerationAtCurrentCoordinate()` | Rigid-frame acceleration at `currentCoordinate()` |
| `relativeMeshVelocity()` | `meshVelocity() - frameVelocityAtCurrentCoordinate()` |
| `MovingFrameFormTerminals`, `movingFrameTerminals()` | Bundled helper terminals |

### Cut Cell

Status: `implemented/public` helper header.

`Forms/CutCellForms.h` provides parameter-slot terminals for cut-cell and
embedded-boundary formulations:

| Vocabulary | Notes |
|------------|-------|
| `CutCellParameterSlots` | Slot layout constants |
| `cutVolumeFraction()` | Cut-cell volume fraction |
| `cutSideIndicator()` | Side/phase indicator |
| `cutEmbeddedNormal()` | Embedded-boundary normal |
| `cutStabilizationScale()` | Stabilization scale |
| `cutMeasureSensitivity()` | Measure sensitivity |
| `cutNormalSensitivity()` | Normal sensitivity vector |
| `cutQuadratureWeightSensitivity()` | Quadrature-weight sensitivity |
| `CutCellFormTerminals`, `cutCellTerminals()` | Bundled helper terminals |

### Manual Block and Complex Workflows

Status: `implemented/expert`.

| Vocabulary | Notes |
|------------|-------|
| `BlockBilinearForm`, `BlockLinearForm` | Manual block decomposition |
| `ComplexScalar`, `I()`, `conj(...)` | Real/imaginary complex expression adapter |
| `ComplexBilinearForm`, `ComplexLinearForm` | Complex form containers |
| `toRealBlock2x2(...)`, `toRealBlock2x1(...)` | Real block lifting for complex forms |

The runtime kernel model is real-valued. Complex PDEs are represented by
explicit real/imaginary splitting today.

## Current Capability Notes

- Forms is a real-valued symbolic DSL with scalar, vector, matrix, and selected
  higher-rank coefficient support. Broad tensor-valued unknowns are planned, not
  current public vocabulary.
- Spatial second derivatives are available in supported mappings and expression
  shapes. Non-affine higher-derivative coverage remains mapping- and element
  dependent, with validation preferred over silent approximations.
- Matrix, spectral, history, smooth regularization, and interface-measure
  vocabulary are implemented and should not be listed as future work.
- `FormExpr::parameter(name)` is implemented. `parameterRef(slot)` is an
  internal/lowering hook.
- Slot-indexed auxiliary references and byte-offset material-state references
  are implementation details, not the recommended formulation surface.
- `trace(A)` is tensor trace. Facet/interface trace vocabulary must use names
  such as `normalTrace` or `facetTrace`, not overload tensor `trace`.
- Quadrature-rule control, embedded/codimension measures, and mixed-dimensional
  transfer operators are not yet public form vocabulary.

## Planned Vocabulary

These items are not implemented or not public enough to document as current
authoring vocabulary. They are ordered by expected leverage.

### Priority 1: Trace and Interface Algebra

Status: `planned`.

Current public pieces include side restrictions, `jump`, `avg`,
`weightedAverage`, `normalComponent`, scalar trace BC helpers, and `.dI`.
Missing vocabulary:

| Planned vocabulary | Purpose |
|--------------------|---------|
| `facetTrace(expr, side)` | Explicit side/trace restriction distinct from tensor `trace` |
| `normalTrace(u)` | Normal scalar/vector trace with documented orientation |
| `tangentialTrace(u)` | Tangential trace for vector fields |
| `projectNormal(u)`, `projectTangent(u)` | Normal/tangential projection helpers |
| `normalJump(u)`, `tangentialJump(u)` | Orientation-aware trace jumps |
| Side-specific normals/orientation metadata | Robust interface semantics for H(div), H(curl), slip, contact, and Nitsche laws |

Acceptance criteria: common H(div), H(curl), slip, and Nitsche boundary/interface
terms can be written without raw component manipulation against side-specific
normals.

### Priority 2: Named History, Stage, and Material State

Status: `planned`.

Current public pieces include `dt`, `t`, `deltat`, `deltat_eff`,
`previousSolution(k)`, `historyWeightedSum`, `historyConvolution`, and named
auxiliary input/output/state terminals. Missing vocabulary:

| Planned vocabulary | Purpose |
|--------------------|---------|
| `prev(u, k)` or `u_prev(k)` | Field-oriented previous solution access |
| `stage(u, i)` | Time-integrator stage access |
| `stateOld(name)`, `stateWork(name)` | Named material-state access from `StateLayout` |
| Named coupled integral/state aliases where needed | Avoid authoring raw slots in public physics modules |

Acceptance criteria: transient and stateful constitutive forms can be written
with named field/state concepts instead of slot or byte-offset references.

### Priority 3: Generalized Measures and Quadrature Metadata

Status: `planned`.

Current public measures are `.dx()`, `.ds(marker)`, `.dS()`, and
`.dI(marker)`. Missing vocabulary:

| Planned vocabulary | Purpose |
|--------------------|---------|
| Subdomain-set and boundary-set measures | Integrate over marker sets without manual loops |
| Interface-pair/interface-set measures | Nontrivial interface grouping |
| Codimension-2 edge/curve measures | Fractures, fibers, vessel centerlines, embedded curves |
| Point measures / singular source measures | Point sources and probe-style weak terms |
| Embedded manifold measures | Immersed or lower-dimensional physics |
| Per-measure quadrature rule/order metadata | Local quadrature control attached to the measure |

Acceptance criteria: singular sources, embedded sources, and interface-heavy
problems can be expressed without handwritten kernels.

### Priority 4: Mixed-Dimensional Transfer Operators

Status: `planned`.

Spaces contain `TraceSpace` and `MortarSpace` concepts, and Systems can register
interface fields, but public bulk-to-lower-dimensional transfer vocabulary is
not complete.

| Planned vocabulary | Purpose |
|--------------------|---------|
| `restrictTo(manifold, u)` | Restrict a bulk field to a lower-dimensional manifold |
| `traceTo(manifold, u)` | Explicit trace from bulk to a target manifold/interface |
| `extendFrom(manifold, lambda)` | Extend lower-dimensional data back to the bulk |
| `lift(...)` | Lifting/prolongation helper |
| Nonmatching transfer operators | Mortar and embedded coupling |

Suggested sequencing: same-mesh codimension-1 coupling, nonmatching
codimension-1 mortar coupling, then codimension-2 and mixed 0D/1D/3D transfer.

### Priority 5: Scalar Math Expansion

Status: `planned`.

| Planned vocabulary | Notes |
|--------------------|-------|
| `sin`, `cos`, `tan` | Trigonometric functions |
| `sinh`, `cosh`, `tanh` | Hyperbolic functions |
| `atan`, `atan2` | Inverse tangent functions |
| `erf`, `erfc` | Special functions useful for regularized profiles |
| `positivePart`, `negativePart` | Exact or smooth part functions, with clear nonsmooth semantics |
| Complementarity/projector helpers | Contact/inequality formulations if needed |

New scalar functions should ship with AD, symbolic differentiation, interpreter,
JIT/lowering, validation, and tests.

### Priority 6: Constitutive Ergonomics

Status: `planned`.

Current support already includes n-ary constitutive inputs, multi-output calls,
indexed output selection, output metadata, parameter specs, and state layout
metadata. Missing vocabulary:

| Planned vocabulary | Purpose |
|--------------------|---------|
| `call.out("stress")` | Named output selection |
| Public output descriptors for docs/tooling | Discoverable constitutive outputs |
| Public named material state access | Pair `StateLayout` with form vocabulary |
| First-class adapter layer from FE constitutive laws | Reduce custom glue code |

### Priority 7: Conservation-Law Trace Support

Status: `planned`.

Current DG helpers cover side restrictions, jump/average, weighted/harmonic
averages, upwind/downwind values, and simple interior-penalty scaling.

The planned work here is not a core `numericalFlux(...)` wrapper. Flux models
should remain visible in formulation code or in physics-layer helper functions
that are themselves written from Forms expressions. Core Forms should instead
make those explicit flux expressions safer and clearer with better
trace/orientation vocabulary.

| Planned vocabulary | Purpose |
|--------------------|---------|
| Side-specific normal helpers | Make the normal used by `minus`/`plus` terms explicit |
| Vector-valued trace helpers | Avoid ad hoc component handling in face fluxes |
| Normal/tangential jump helpers | Make conservation-law and vector trace terms reviewable |
| Reusable penalty-scale primitives | Share element/face scaling without hiding flux policy |

SIPG/NIPG, Lax-Friedrichs/Rusanov, Roe, HLLC, entropy-stable fluxes, farfield
states, wall models, and similar policies should live in the relevant physics
module when needed. They should not become generic Forms vocabulary unless they
can be expressed as domain-neutral mathematical primitives.

### Priority 8: Richer Tensor and Complex Workflows

Status: `planned`.

| Planned vocabulary | Purpose |
|--------------------|---------|
| Tensor-valued trial/test/state fields | Public tensor-valued unknowns |
| Higher-rank component access and indexing | Rank > 2 public tensor calculus |
| More general Einstein contraction patterns | Beyond current implicit sum patterns |
| Complex trial/test handles and conjugation conventions | More ergonomic complex PDE authoring |
| True complex backend, if justified | Avoid real block lifting where needed |

## Out of Scope for Now

The following should remain deferred until there is a concrete product/module
need and a clear ownership plan:

- FEEC/manifold exterior calculus (`DifferentialForm`, `ExteriorDerivative`,
  `HodgeStar`, `Wedge`, broad manifold pullbacks) beyond existing
  `pullback`/`pushforward` metadata and finite-deformation transforms.
- Broad optimization, inverse-problem, adjoint, stochastic/UQ, and nonlocal PDE
  vocabulary.
- Large PDE-specific helper catalogs that duplicate cross-cutting primitives.
- Generic flux-policy wrappers such as core `numericalFlux(...)`, or
  PDE-specific flux catalogs, when explicit Forms expressions or physics-layer
  helpers make the formulation clearer.
- Solver-interaction vocabulary that belongs in Systems/Solvers rather than
  Forms.

## References

- Alnaes et al., "Unified form language." ACM Transactions on Mathematical
  Software, 2014. DOI: 10.1145/2566630.
- Logg, Mardal, Wells, eds. Automated Solution of Differential Equations by the
  Finite Element Method, 2012.
- Rathgeber et al., "Firedrake." ACM Transactions on Mathematical Software,
  2016. DOI: 10.1145/2998441.
- Anderson et al., "MFEM: a modular finite element methods library." Computers
  & Mathematics with Applications, 2021. DOI: 10.1016/j.camwa.2020.06.009.
- Arndt et al., "The deal.II finite element library: Design, features, and
  insights." Computers & Mathematics with Applications, 2021. DOI:
  10.1016/j.camwa.2020.02.022.
