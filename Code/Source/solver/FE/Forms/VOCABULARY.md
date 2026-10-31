# FE/Forms Vocabulary

This is the canonical vocabulary document for `FE/Forms`. It replaces the
previous split between `VOCABULARY.md`, `VOCABULARY_ROADMAP.md`,
`UFL_IMPLEMENTATION_CHECKLIST.md`, and `FUTURE_FEATURES.md`.

The goal of this file is to keep the public form-authoring surface, expert
hooks, and planned vocabulary in one current place. It is scoped to the FE form
language and the adjacent FE services a formulation calls (`FE/Systems`,
`FE/Constraints`, `FE/LevelSet`, `FE/Interfaces`, `FE/PostProcessing`,
`FE/Auxiliary`); mesh ownership, DOF management, solver configuration, and
time-integration orchestration remain in their owning modules.

Every entry below was checked against the headers and unit tests on
2026-10-23. Line-level details live in the headers; this file says what exists,
what it is for, and who should call it.

## Status Legend

| Status | Meaning |
|--------|---------|
| `implemented/public` | Intended for formulation authors and covered by current headers/tests |
| `implemented/expert` | Supported but meant for specialized, manual, or lower-level workflows |
| `implemented/internal` | Setup, JIT, lowering, or byte/slot based hooks; not an ergonomic public DSL |
| `planned` | Not implemented yet; tracked by a named step of milestone MR in `Documentation/free_surface_program_tracker.md` §5, or left for later by a phase (LM1-LM4) of decision D54 there |
| `out-of-scope-for-now` | Not implemented and not scheduled; deferred until there is a concrete product/module need |

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
- The expression tree is the identity of a term. Kernel grouping, JIT cache
  keys, the system manifest, and the floating-point evaluation order all follow
  the tree, so algebraically equal spellings are not interchangeable
  (`test_IntegrationDomainLowering.cpp` checks that `a * p` and `p * a`, or
  `inner(tangentialProjector(n, d), grad(v))` and `surfaceDivergence(v, n)`,
  hash differently). A helper that only renames must return the identical tree;
  typed measures and the `Vocabulary.h` helpers are tested for that.
- `Physics/Formulations` holds the forms-based governing equations; the
  supporting machinery (records, ledgers, certificates, logs, collectives) is
  posed as generic FE services that take forms and data (tracker decisions
  D42, D49, D50).

## Header Map

Form language (`FE/Forms`):

| Header | Role |
|--------|------|
| `Forms/FormExpr.h` | Core symbolic AST, low-level constructors, operators, and measures |
| `Forms/Vocabulary.h` | High-level UFL-style helpers for normal formulation authoring |
| `Forms/IntegrationDomain.h` | Typed integration domains (`VolumeDomain`, `InterfaceDomain`, `ContactLineDomain`, `BoundaryDomain`, `FacetDomain`) |
| `Forms/BoundaryConditions.h` | Value conversion (`bc::toScalarExpr`, `bc::toVectorExpr`), strong Dirichlet declarations, and weak trace helper functions |
| `Forms/BoundaryCondition.h` | `bc::BoundaryCondition` interface consumed by `BoundaryConditionManager` |
| `Forms/StandardBCs.h` | Standard BC wrapper types and `make*BC` factories |
| `Forms/NitscheBC.h` | Boundary/interface Nitsche BC wrappers |
| `Forms/ConstraintBCs.h` | Periodic and multi-point constraint BC wrappers |
| `Forms/InterfaceConditions.h` | Interface helper terms over `.dI(marker)` |
| `Forms/ConstitutiveModel.h` | Constitutive callback and material-state interface |
| `Forms/BoundaryFunctional.h` | Boundary/domain functional definition and compilation |
| `Forms/FiniteDeformationForms.h` | Finite-deformation kinematic vocabulary (`forms::finite_deformation`) |
| `Forms/MovingFrameForms.h` | Moving-frame helper terminals |
| `Forms/CutCellForms.h` | Cut-cell metadata terminals and cut-adjacent facet helpers |
| `Forms/SymbolicDifferentiation.h` | Expert symbolic differentiation helpers |
| `Forms/PointEvaluator.h`, `Forms/AffineAnalysis.h` | Expert point evaluation and affine residual splitting |
| `Forms/FormCompiler.h`, `Forms/FormKernels.h`, `Forms/FormIR.h`, `Forms/MixedFormIR.h` | Expert compilation, kernel, and IR layers |
| `Forms/BlockForm.h`, `Forms/Complex.h` | Manual block and real-lifted complex workflows |
| `Forms/Index.h`, `Forms/Einsum.h`, `Forms/Tensor/*` | Indexed/tensor-calculus infrastructure |
| `Forms/Authoring.h` | The one include of a formulation module: the form language and the FE services of the canonical workflow |
| `Forms/Forms.h` | Umbrella include of the core form language; a formulation module includes `Forms/Authoring.h` instead |
| `Forms/WeakForm.h` | Deprecated residual-plus-constraints bundle |

Adjacent FE services a formulation calls:

| Header | Role |
|--------|------|
| `Systems/FormsInstaller.h` | `installFormulation`, `FormInstallOptions`, strong Dirichlet and mixed-block installation |
| `Systems/FormulationBuilder.h` | Install options of one formulation, composed in call order, plus generated-marker registration |
| `Systems/FieldBinding.h` | Register-or-reuse of a module's fields with compatibility checks |
| `Systems/BoundaryConditionManager.h` | Validate, apply, and install a field's boundary conditions |
| `Systems/DiagnosticRegistry.h` | Opt-in diagnostic operators of a formulation, by set |
| `Systems/FunctionalForms.h`, `Systems/FunctionalFormEvaluator.h` | A formulation's functionals as measured 0-forms, and their rank-independent values |
| `Systems/ALEBinding.h`, `Systems/MeshDisplacementBinding.h` | Mesh-velocity/mesh-displacement field binding for moving-domain forms |
| `Systems/FESystem.h` | Fields, operators, constraints, auxiliary inputs, derived results, functional kernels, and the attachment store |
| `Systems/SystemCollectives.h`, `Systems/ExactAccumulator.h` | Collective reductions, including the exact rank-independent sum |
| `Constraints/ConstraintPlan.h` | Phase-ordered registration of a formulation's system constraints |
| `Constraints/InterfaceMeasureGaugeAnchor.h` | Gauge anchoring by a natural condition on a generated interface |
| `Constraints/ContactPointVelocityConstraint.h` | Strong contact-point velocity law (2D, planar walls) |
| `LevelSet/LevelSetGeometry.h` | Level-set geometry handles that return typed integration domains |
| `Interfaces/ContactLineGeometryCertificate.h` | Planar-wall and contact-rule certificates of a contact line |
| `PostProcessing/DerivedResultBuilder.h` | Derived output fields defined by a `FormExpr` |
| `Auxiliary/AuxiliaryModelDSL.h`, `Auxiliary/AuxiliaryBindings.h` | 0D auxiliary models coupled to forms |
| `Coupling/CouplingFormBuilder.h` | Participant-named forms for multi-module coupling contracts |

## Canonical Workflow

Every formulation module follows one skeleton: bind its fields with
`FieldBinding`, write the residual, add its boundary conditions, register its
constraints through a `ConstraintPlan`, declare its functionals and outputs,
and install through `FormulationBuilder`. One include, `FE/Forms/Authoring.h`,
declares all of it; `FE/Tests/Unit/Forms/test_Authoring.cpp` compiles this
skeleton with that include alone:

```cpp
#include "FE/Forms/Authoring.h"

namespace fe = svmp::FE;
using namespace fe::forms;

struct HeatOptions {
    fe::Real conductivity{1.0};
    fe::Real source{0.0};
    int wall{1};                    // T = wall_temperature
    fe::Real wall_temperature{0.0};
    int inlet{2};                   // k grad(T).n = inflow
    fe::Real inflow{0.0};
};

// -div(k grad T) = f in the domain, T = T_w on the wall, k grad(T).n = q on the inlet.
void registerHeat(fe::systems::FESystem& system,
                  std::shared_ptr<const fe::spaces::FunctionSpace> V,
                  const HeatOptions& o)
{
    fe::systems::FieldBinding fields(system, "Heat::registerOn");
    const auto [T_id, T, v] = fields.unknownSymbols({.name = "T", .space = V}, "temperature", "T", "v");

    const auto k = FormExpr::constant(o.conductivity);
    const auto f = FormExpr::constant(o.source);
    auto R = (k * inner(grad(T), grad(v)) - f * v).dx();

    fe::systems::BoundaryConditionManager bcs;
    bcs.add(std::make_unique<bc::EssentialBC>(o.wall, FormExpr::constant(o.wall_temperature), "T"));
    bcs.add(std::make_unique<bc::NaturalBC>(o.inlet, FormExpr::constant(o.inflow)));

    fe::constraints::ConstraintPlan plan(system);
    plan.essentials(bcs, R, T, v, T_id);
    plan.commit();

    fe::systems::FormulationBuilder F(system, "equations", {});
    F.functional("conduction_energy", fe::systems::FunctionalKind::Stored,
                 (0.5 * k * inner(grad(T), grad(T))).dx());
    F.symbolicTangent();
    (void)F.install({T_id}, R);

    system.addDerivedResult(fe::post::DerivedResultBuilder("heat_flux")
                                .scope(fe::post::DerivedResultScope::Cell)
                                .policy(fe::post::DerivedResultPolicy::CellAverage)
                                .shape(fe::systems::FEQuantityShape::vector(V->topological_dimension()))
                                .expression(-k * grad(T))
                                .referencedField(T_id)
                                .build());
}
```

`Forms/SYSTEMS_INTEGRATION.md` explains each step (section 2) and the path from
a deck key to a running equation (section 8). A module with several unknowns
binds each, sums their rows and installs them together,
`F.install({u_id, p_id}, R)`; `IncompressibleNavierStokesVMSModule::registerOn`
is the multi-field example.

Use unbound `TrialFunction`/`TestFunction` for explicit operator authoring
(`installMixedBilinear`, `installMixedLinear`) and `StateField`/`TestField` for
residual formulations (`installFormulation`). The residual form is
differentiated automatically (forward-mode dual numbers by default, or the
symbolic tangent when `use_symbolic_tangent` is set).

## Module Boundaries

`FE/Forms` owns symbolic expression construction, form IR, symbolic
differentiation, form compilation, AD/JIT evaluation support, and small
formulation vocabulary helpers.

`FE/Systems`, `FE/Assembly`, `FE/Spaces`, `FE/Mesh`, and time-stepping code own
field registration, DOF layout, boundary-condition installation, assembly
loops, quadrature/basis data, mesh topology, and time-state storage. Forms may
expose symbolic terminals for that data, but those modules provide the data.

`FE/Constraints` owns constraint objects and their registration order
(`ConstraintPlan`). `FE/LevelSet` owns level-set geometry, the generated cut
context, and its markers; `FE/Interfaces` owns generated interface domains and
their certificates. A formulation reaches the generated markers only through
`level_set::LevelSetGeometry` handles, which return typed integration domains.

## Implemented Vocabulary

### Fields and Symbols

Status: `implemented/public`, unless noted.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `StateField(field_id, V, name)` | `Vocabulary.h` | Canonical residual unknown bound to a registered `FieldId` |
| `TestField(field_id, V, name)` | `Vocabulary.h` | Canonical residual test function bound to the same `FieldId` |
| `StateFields(ids, spaces, names)`, `TestFields(ids, spaces, names)` | `Vocabulary.h` | Multi-field residual helpers; sizes must agree |
| `TrialFunction(V, name)`, `TestFunction(V, name)` | `Vocabulary.h` | Operator authoring without `FieldId` binding |
| `TrialFunctions(W, names)`, `TestFunctions(W, names)` | `Vocabulary.h` | One unbound symbol per `MixedSpace` component |
| `FormExpr::trialFunction`, `testFunction`, `discreteField`, `stateField` | `FormExpr.h` | Lower-level factories, including `FieldId` and `SpaceSignature` overloads |
| `FormExpr::coefficient(name, f)` | `FormExpr.h` | Scalar, time-scalar, vector, matrix, rank-3, and rank-4 callbacks; interpreted call boundary, not JIT-inlined |
| `FormExpr::constant(value)` | `FormExpr.h` | Literal scalar expression |
| `FormExpr::parameter(name)` | `FormExpr.h` | Named runtime parameter, resolved to a slot at setup |
| `FormExpr::parameterRef(slot)` | `FormExpr.h` | `implemented/internal`; post-resolution slot reference |
| `FormExpr::typedZero()` | `FormExpr.h` | `implemented/internal`; shape-preserving zero for lowering/simplification |
| `systems::FieldBinding::unknown(spec, role)`, `prescribed(spec, auto_register, role)` | `Systems/FieldBinding.h` | Register a field or reuse a compatible registered one (source kind; scope: volume, interface with its marker, or global, as the registry sets it from the space; components; space); incompatible reuse throws with `"<owner> <role>: "`, naming the first of these that differs (a volume field asked for as a global one: "is a volume field, expected a global field") |
| `systems::FieldBinding::lagged(U, Refresh::GeneratedState[, name])`, `previous(U, k = 1[, name])` | `Systems/FieldBinding.h` | An unknown held at an earlier value: its prescribed shadow field (`<field>_lagged`, `<field>_previous<k>`, or `name`) on the unknown's space, registered or reused, returned as its DiscreteField of the same name, which is data in the Jacobian (`u - lagged(U)` has the residual of `u - u_ref` and the Jacobian of `u`). `lagged` holds u at the last refresh of the generated state, `previous` its value k steps back in the time history (what the integrator keeps there). `shadowFields(system)` lists them; `timestepping::makeShadowFieldParticipant(system)` is the Reference participant of a `GeneratedStateRegistry` that refreshes them by a rank-local DOF copy. Not yet used by the Navier-Stokes semi-implicit term |
| `systems::FieldBinding::global(name, role)` | `Systems/FieldBinding.h` | A global multiplier: an unknown on a new `GlobalSpace` with the automatic quadrature, or the registered global field of that name; its space is `fieldRecord(id).space` |
| `systems::FieldBinding::check`, `requireDistinctNames`, `systems::spacesCompatible` | `Systems/FieldBinding.h` | Read-only checks before system mutation |
| `FESystem::addField(FieldSpec)`, `addInterfaceField(name, space, marker)` | `Systems/FESystem.h` | Volume fields and codimension-one interface fields |
| `spaces::GlobalSpace(n = 1, quadrature_order = 0)` | `Spaces/GlobalSpace.h` | One value for the whole domain: basis 1 and gradient 0 on every cell, so `lam * v` and `mu * u` integrate with the usual measures. One scalar per field (declare one global field per scalar). Its rows use the quadrature of the highest-order volume field, bound at setup, unless an order is given |
| `FieldSpec{.space = GlobalSpace, .scope = FieldScope::Global}`, `FESystem::hasGlobalFields()` | `Systems/FieldRegistry.h`, `Systems/FESystem.h` | A global field (an unknown; the scope follows a `GlobalSpace`). Every cell maps to its DOF, which follows the field DOFs in the monolithic numbering; the lowest rank owns it and the other ranks hold it as a ghost. It has no mesh entity, so the entity map and the VTK output leave it out. It is never a gauge candidate. Setup refuses a constraint on its DOF, and a global row (without `.dGlobal()` terms in global fields) whose every coupled DOF is constrained (the multiplier and an essential condition on the same DOFs). Refused with colored or work-stream assembly, with `GhostPolicy::OwnedRowsOnly` at several ranks, and with FSILS |
| `FESystem::traceMultiplierFields()`, `FESystem::unstabilizedTraceMultiplierFields()` | `Systems/FESystem.h` | The fields a `SupportRestrictionConstraint` restricts (trace multipliers, D54 T1), and those whose rows hold no term in themselves (a zero diagonal on the active rows): the Application refuses those under an FSILS solve other than its direct solve (scaling, ILU and the Krylov solvers need the diagonal) |

`TestFunction(FieldId, ...)` was removed; `TestField` is the only field-bound
test symbol. Supported FE space concepts live in `FE/Spaces` and are consumable
by Forms: `FunctionSpace`, `H1Space`, `L2Space`, `HDivSpace` (Raviart-Thomas,
BDM), `HCurlSpace` (Nedelec), `MixedSpace`, `ProductSpace`, `TraceSpace`,
`MortarSpace`, `C1Space`, `CompositeSpace`, `EnrichedSpace`, `AdaptiveSpace`,
`GenericBasisSpace`, `GlobalSpace`, and `SpaceFactory`.

### Geometry and Time

Status: `implemented/public`, unless noted.

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
| `t()`, `deltat()`, `deltat_eff()` | Time, time step, and effective time step (`FormExpr::effectiveTimeStep()`, the step implied by the `dt(.)` stencil: 1/\|a0\|, or `deltat()` without a stencil, in kernels, the JIT, derived results and point evaluation alike; 0.5333 `deltat()` under first-order generalized-alpha with rho_inf = 0.5) |
| `systems::timeFunction(system, name, f)` | `Systems/TimeSeries.h`; declares the parameter `name` with value f(t) and returns `FormExpr::parameter(name)`. In a weak form it is a slot of the parameter registry filled with f(state time), JIT-strict and cacheable, and agrees with a callback returning f(t) g(x) to round-off; in a strong value it is f at the constraint update time, and `Q * g` gives that callback's inhomogeneities bit for bit. The strong unsteady inflows of the Navier-Stokes module use it (`set_strong_unsteady_inflow` in `Physics/Options/BoundaryDataReaders.h`); weak inflows keep their callbacks |
| `systems::TimeSeries`, `readTimeSeries(path, end, components)` | `Systems/TimeSeries.h`; a tabulated function of time, linear between points, clamped or periodic outside them, interpolated as the solver's temporal-values files are (bit for bit); callable, so it is the f of `timeFunction` |
| `J()`, `Jinv()`, `detJ()` | Active geometry Jacobian, inverse, and determinant, shaped dim x dim |
| `currentJacobian()`, `referenceJacobian()` | Frame-explicit geometry Jacobians |
| `currentJacobianDeterminant()`, `referenceJacobianDeterminant()` | Signed determinants of the frame-explicit geometry Jacobians |
| `FormExpr::identity()`, `FormExpr::identity(dim)` | Identity tensors |
| `FormExpr::normal()`, `currentNormal()`, `referenceNormal()` | Active and frame-explicit normals |
| `FormExpr::boundaryNormal()` | Outward normal of the physical boundary under a generated rule (a contact line `dL`, a wetted boundary `ds(generatedActiveSubset(...))`), read from the rule; on a curved wall it follows the wall facet, on a flat wall it equals the configured wall normal; evaluation on any other rule fails; interpreted only (no JIT) |
| `currentMeasure()`, `referenceMeasure()`, `surfaceJacobian()` | Frame-explicit integration measures |
| `currentSurfaceVector()`, `referenceSurfaceVector()` | Measure-scaled normal vectors |
| `currentMeanCurvature()` | `implemented/expert`; assembly-provided mean curvature of a fitted boundary; no formulation uses it since the curvature-traction routes were retired |
| `pullback(expr, from, to)` | Metadata-only frame marker; runtime evaluation fails for non-identity transforms |
| `pushforward(expr, from, to)` | Metadata-only frame marker; runtime evaluation fails for non-identity transforms |
| `nanson(expr)`, `nanson()` | Explicit Nanson surface-vector transform using `cofactor(currentJacobian() * inv(referenceJacobian()))` |
| `h()`, `vol()`, `area()`, `domainId()` | Cell diameter, cell volume, facet area, cell-domain marker |
| `hNormal()` | Facet-normal cell size `2 * vol() / area()` |
| `FormExpr::geometryTrialVectorVariation()`, `meshVelocityVariation()`, `currentMeasureVariation()`, `currentNormalVariation()`, `surfaceJacobianVariation()`, `geometryTrialJacobianVariation()` | `implemented/internal`; produced by symbolic geometry differentiation |

The moving-domain terminals are physics-neutral. They expose assembly-provided
mesh/domain motion and frame-explicit geometry; they do not encode a specific
ALE, FSI, solid-mechanics, contact, or free-surface model.
For ALE forms, spell out transport and moving-volume terms directly, such as
`u - meshVelocity()` and `rho * div(meshVelocity()) * inner(u, v)`, so the
mathematics remains visible in formulation modules. `dt(u)` is the time
derivative at a fixed mesh node, `du/dt|_X`. Monolithic derivatives with
respect to a mesh-displacement unknown are an install option
(`GeometrySensitivityMode::MeshMotionUnknowns`, set by `ALEBinding`), not a
different vocabulary, and they are partial. Only the frame-explicit terminals
(`currentCoordinate()`, `meshDisplacement()`, `meshVelocity()`,
`currentJacobian()`, `currentNormal()`, `currentMeasure()`,
`surfaceJacobian()`) are differentiated, and only an integrand that contains
one of them gets mesh columns. `x()`, `J()`, `Jinv()`, `detJ()`, `h()`,
`vol()`, `area()`, `hNormal()`, `FormExpr::normal()` and the change of `grad`
and `div` of every field with the mesh are geometry constants, and the cell
blocks of the other fields' rows do not assemble the measure variation either.
The `MovingDomainShapeDerivativeTest` in
`Tests/Unit/Systems/test_JacobianVerification.cpp` records the missing part of
each operator, and `test_FittedFreeSurfaceALE.cpp` that of the production
coupled-ALE Jacobian.

### Trace and Projection Helpers

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `normalTrace(u, n)`, `normalTrace(u)` | Scalar normal component, using explicit `n` or the active facet normal |
| `normalProjection(u, n)`, `normalProjection(u)` | Normal vector projection |
| `projectNormal(u, n)`, `projectNormal(u)` | Alias for `normalProjection` |
| `tangentialProjection(u, n)`, `tangentialProjection(u)` | Tangential vector projection `u - (u.n) n` |
| `projectTangent(u, n)`, `projectTangent(u)` | Alias for `tangentialProjection` |
| `tangentialTrace(u, n)`, `tangentialTrace(u)` | Alias for `tangentialProjection` |
| `tangentialProjector(n, dim)` | Projector matrix `identity(dim) - outer(n, n)`, as in the surface-stress term `gamma (I - n x n) : grad(v)` |
| `conormal(n, wall_normal[, eps])` | Unit conormal of a contact line, `safeNormalize(n - (n.n_w) n_w)` |

The default overloads use `FormExpr::normal()`. Use explicit normals for
interface-side, level-set, or frame-specific laws where the orientation should
be visible in the formulation. `bc::normalComponent(u)` is the same tree as
`normalTrace(u)`.

### Measures

Status: `implemented/public`, unless noted.

Untyped measures (`FormExpr.h`):

| Vocabulary | Integral domain |
|------------|-----------------|
| `.dx()` | Cell/domain integral |
| `.ds(marker)` | Boundary integral; `marker = -1` means all boundary faces |
| `.dS()`, `.dS(marker)` | Interior facet integral; with a marker, over a marker-backed facet set |
| `.dS(marker, side)` | Interior facets of a set adjacent to one side of a cut |
| `.dI(marker)` | Registered or generated interface integral; `marker = -1` means all interfaces |
| `.dCutVolume(interface_marker, side)` | Generated cut-volume integral over the selected interface side |
| `.dExteriorBoundary(measure)` | Explicit `ExteriorBoundaryMeasure`: `fullPhysical(m)` or `generatedActiveSubset(m, g)` |
| `dInterfaceBoundary(expr, marker)` | Generated interface-boundary intersection; shares `.dI(marker)` dispatch, but the generated rule carries geometric dimension d-2 (a point in 2D, a line in 3D) and the codimension-two Jacobian |
| `.dGlobal()` | Unit point mass: a term of a global test function's row with no spatial integral, such as `-(mu * c).dGlobal()`, added once. Its integrand holds only global symbols, constants, parameters, time and the time step, and scalar algebra; anything else is refused at install |

Typed domains (`IntegrationDomain.h`) are sugar over the untyped measures: each
lowers to the tree of the untyped call (tested in
`test_IntegrationDomainLowering.cpp`).

| Vocabulary | Lowers to |
|------------|-----------|
| `.dx(VolumeDomain::full())`, `integrate(expr, VolumeDomain::full())` | `.dx()` |
| `.dx(VolumeDomain::cut(m, side))` | `.dCutVolume(m, side)` |
| `.dI(InterfaceDomain(m))` | `.dI(m)` |
| `.dL(ContactLineDomain(m, wall))`, `dInterfaceBoundary(expr, ContactLineDomain)` | `.dI(m)` with the codimension-two generated rule |
| `.ds(BoundaryDomain::fullPhysical(b))` | `.dExteriorBoundary(fullPhysical(b))`; not the same tree as `.ds(b)` |
| `.ds(BoundaryDomain::generatedActiveSubset(b, g))` | `.dExteriorBoundary(generatedActiveSubset(b, g))` |
| `.dS(FacetDomain::cutAdjacent(m[, side]))` | `.dS(m[, side])`, as `cutAdjacentFacetIntegral(expr, m[, side])` |

`VolumeDomain` and `BoundaryDomain` are aliases of `VolumeIntegrationDomain`
and `ExteriorBoundaryMeasure`.

Level-set geometry handles (`LevelSet/LevelSetGeometry.h`, namespace
`level_set`) return these domains with the stable generated markers, so a
formulation never computes a marker by hand:

| Vocabulary | Returns |
|------------|---------|
| `LevelSetGeometry::bind(system, {.level_set_field_name, .domain_id, .isovalue, .interface_marker})` | The geometry {phi = c}; throws for an unknown or non-scalar field |
| `G.interface()`, `L.interface()` | `InterfaceDomain` of Gamma |
| `G.side(CutVolumeSide::Negative \| Positive)` | `CutSide` L = {phi < c} or {phi > c} |
| `L.volume()` | `VolumeDomain::cut(marker, side)` |
| `L.outwardNormal()`, `outwardInterfaceNormal(side)` | The interface rule normal, negated on the positive side, so it points out of L |
| `L.boundary(b)`, `L.wetted(w)` | `BoundaryDomain::generatedActiveSubset(b, generated marker)` |
| `L.cutFacets()` | `FacetDomain::cutAdjacent(marker, side)` |
| `G.contactLine(wall)`, `L.contactLine(wall)` | `ContactLineDomain` of Gamma intersected with a wall |
| `G.contactLineKey(wall)`, `L.boundaryKey(b)` | `implemented/expert`; the generated marker keys |

Binding registers nothing. A formulation that integrates over a generated
contact line or a generated boundary subset registers that marker with
`FormulationBuilder::registered(domain)`; cut volumes are registered by the
installer. The cut context itself is generated by
`level_set::LevelSetGeometryService` from the Application's
`ActiveCutVolumeRequest`, not by the formulation.

The compiler/IR distinguishes `Cell`, `Boundary`, `InteriorFace`,
`InterfaceFace`, and `CutVolume` terms. Contact-line terms are `InterfaceFace`
terms; their dimension is carried by the generated rule. `.dGlobal()` terms
never reach the IR: `installFormulation` lowers them with the rest of their
global row (below), and the compiler refuses them.

### Differential Operators

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `grad(u)` | Scalar-to-vector, vector-to-matrix, and supported intrinsic vector FE gradients |
| `symgrad(u)` | Symmetric gradient of a vector field; exactly the expression tree `sym(grad(u))` |
| `advect(a, u)` | Vector advection `(a . grad) u`; exactly `grad(u) * a`, preserving all state dependencies in both operands. For a scalar field use `inner(grad(u), a)` |
| `div(u)`, `curl(u)` | Vector differential operators |
| `hessian(u)` | Spatial Hessian support for supported scalar/composite expressions and mappings |
| `laplacian(u)` | `trace(hessian(u))` helper |
| `dt(u, order)` | Symbolic time derivative; the stencil comes from the time integrator of the transient system. Its operand is a field (`StateField`, `TrialFunction` or `DiscreteField`); `dt()` of anything else is refused where it is built |
| `surfaceGradient(f, n)` | Projected surface gradient helper |
| `surfaceDivergence(u, n)` | Projected surface divergence `div(u) - (grad(u) n).n` |
| `surfaceLaplacian(f, n)` | Surface Laplacian helper |
| `meanCurvatureFromNormal(n)`, `curvatureVectorFromNormal(n)` | Curvature helpers from a supplied unit normal |
| `unitNormalFromLevelSet(phi)`, `meanCurvatureFromLevelSet(phi)` | Level-set geometry helpers |
| `safeNorm(v)`, `safeNormalize(v)` | Regularized vector norm helpers |

Intrinsic `H(div)` and `H(curl)` vector-basis gradients are matrix-valued with
layout component-by-physical-coordinate derivative. Affine mappings are
evaluated analytically. Supported non-affine 3D curved volume mappings include
the analytic derivative terms for the relevant Piola transforms. Unsupported
element-family, geometry-order, or lower-dimensional curved-frame combinations
fail with diagnostics instead of silently using an affine approximation.

The differential helpers only construct the spelled-out trees. For example,
`inner(advect(u - meshVelocity(), u), v) +
2.0 * mu * inner(symgrad(u), symgrad(v))` has the same ordered tree as
`inner(grad(u) * (u - meshVelocity()), v) +
2.0 * mu * inner(sym(grad(u)), sym(grad(v)))`, including when `mu` or the
advection direction depends on the unknown fields. They introduce no new
terminals or differentiation rules.

### Algebra, Predicates, and Scalar Functions

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `+`, `-`, `*`, `/`, unary `-` | Scalar/vector/matrix expression arithmetic where shape-compatible |
| `a + x`, `x + a`, `a - x`, `x - a`, `a * x`, `x * a`, `a / x`, `x / a` | Mixed `Real`/`FormExpr` arithmetic; the `Real` becomes `FormExpr::constant(a)` on the side it is written, except for `*`: `a * x` and `x * a` both build `constant(a) * x`, so rewriting `x * FormExpr::constant(a)` as `x * a` changes the tree, the kernels and the JIT keys (`FormExpr.h`, `test_FormExprOperators.cpp`) |
| `inner(a, b)`, `dot(a, b)` | Inner product (`dot` is an alias) |
| `outer(a, b)`, `cross(a, b)` | Vector/tensor products |
| `doubleContraction(A, B)` | Rank-2 Frobenius contraction and rank-4/rank-2 contraction |
| `pow(a, b)`, `sqrt(a)`, `exp(a)`, `log(a)` | Scalar math; `pow(x, Real)` builds `pow(x, constant(p))` |
| `abs(a)`, `sign(a)` | Nonsmooth scalar operators |
| `min(a, b)`, `max(a, b)` | Nonsmooth min/max operators; either argument may be a `Real` |
| `lt`, `le`, `gt`, `ge`, `eq`, `ne` | Predicate nodes returning scalar 0/1 values; either argument may be a `Real` |
| `conditional(cond, then_expr, else_expr)` | Branching expression |
| `heaviside(a)`, `indicator(predicate)`, `clamp(a, lo, hi)` | Convenience combinators |
| `regionIndicator(domain_id)` | Domain-id predicate helper |
| `sumOf(a, b)` | `a + b`, or `b` when `a` is empty: the left fold of a form built term by term |

An empty (default-constructed) `FormExpr` means "no term". Unary operations
and measures map it to an empty `FormExpr` (`grad(e)`, `-e`, `e.dx()`), so a
helper may return `{}` for an absent term and the caller tests `isValid()`.
Every operation with two or more `FormExpr` operands (`+ - * /`, `inner`,
`outer`, `cross`, `doubleContraction`, `pow`, `min`, `max`, the predicates,
`conditional`, the smooth and matrix functions) throws `std::invalid_argument`
when an operand is empty, naming the operation and the operand (for example
"the left operand of '+' is empty"); `sumOf` is the explicit way to start a
sum from an empty form. An empty residual is rejected at installation.

The exact nonsmooth operators are implemented. Smooth regularizations are
separate vocabulary:

| Vocabulary | Notes |
|------------|-------|
| `smoothAbs(a, eps)` | Regularized absolute value |
| `smoothSign(a, eps)` | Regularized sign |
| `smoothHeaviside(a, eps)` | Regularized Heaviside |
| `smoothMin(a, b, eps)` | Smooth minimum |
| `smoothMax(a, b, eps)` | Smooth maximum |

Trigonometric, hyperbolic, inverse-trigonometric, and special functions are not
implemented; see `Not Scheduled`.

### Tensor Algebra and Constructors

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `as_vector({ ... })`, `FormExpr::asVector(...)` | Pack components into a vector expression |
| `zeroVector(dim)` | Vector of `dim` scalar zero constants |
| `constantVector(values, dim)` | The first `dim` entries of a `Real` span as a constant vector |
| `as_tensor({ ... })`, `FormExpr::asTensor(...)` | Pack rows into a matrix/tensor expression |
| `component(a, i[, j])`, `a.component(i[, j])` | Component extraction |
| `transpose(A)`, `trace(A)`, `det(A)`, `inv(A)` | Matrix/tensor operations |
| `cofactor(A)`, `dev(A)`, `sym(A)`, `skew(A)` | Tensor decomposition helpers |
| `norm(a)`, `normalize(a)` | Vector/matrix norm helpers |
| `SymmetricTensor(A)`, `SkewTensor(A)` | Semantic aliases around `sym(A)` and `skew(A)` |
| `contraction(a, b)` | Shape-dependent product helper (`a * b`) |
| `FormExpr::coefficient(..., Tensor3Coefficient)` | Rank-3 coefficient data |
| `FormExpr::coefficient(..., Tensor4Coefficient)` | Rank-4 coefficient data |

The public value model covers real scalars, 3D vectors, 3x3 matrices, symmetric
and skew matrix tags, rank-3 coefficients, and rank-4 coefficients with
dedicated contraction support. True tensor-valued unknown fields and broad
higher-rank public tensor calculus are not first-class authoring vocabulary.

### Indexed Notation

Status: `implemented/public` for the listed `Index`/`einsum` path; tensor
variance metadata is `implemented/expert`.

| Vocabulary | Notes |
|------------|-------|
| `Index`, `IndexSet` | Public symbolic index ids/extents |
| `A(i)`, `A(i, j)`, `A(i, j, k)`, `A(i, j, k, l)` | Indexed access nodes |
| `einsum(expr)` | Einstein lowering; required before compilation |
| `einsum(expr, auto_extent)` | Einstein lowering with auto-extent behavior |
| `tensor::TensorIndex`, `MultiIndex` | Expert tensor-calculus metadata path |
| `tensor::special::delta`, `levicivita`, `levicivita2d` | `implemented/internal`; integer special-tensor helpers of the tensor lowering |

Current public lowering supports scalar outputs from fully contracted
expressions and vector/matrix outputs with up to two free indices. Each index id
may appear once as a free index or twice as a bound index in the supported
patterns. Compiling a form that still holds indexed access throws
`FormCompiler: detected indexed access (Einstein notation). Call
forms::einsum(expr) to lower it before compilation.`

### Matrix, Eigen, Spectral, and History Operators

Status: `implemented/public` for authoring; supported dimensions and
differentiability are validated by the relevant evaluators/JIT paths.

| Vocabulary | Notes |
|------------|-------|
| `expm(A)`, `A.matrixExp()` | Matrix exponential |
| `logm(A)`, `A.matrixLog()` | Matrix logarithm |
| `sqrtm(A)`, `A.matrixSqrt()` | Matrix square root |
| `powm(A, p)`, `A.matrixPow(p)` | Matrix power |
| `FormExpr::matrix*DirectionalDerivative(A, dA)` | Directional derivatives for matrix functions |
| `A.symmetricEigenvalue(which)` | Symmetric eigenvalue (2x2 and 3x3, sorted descending) |
| `eigenvalue(A, which)`, `A.eigenvalue(which)` | General eigenvalue helper |
| `eigvec_sym(A, which)`, `A.symmetricEigenvector(which)` | Symmetric eigenvector |
| `spectralDecomp(A)`, `A.spectralDecomposition()` | Spectral decomposition/eigenvectors-as-columns |
| `FormExpr::symmetricEigen*DirectionalDerivative(...)`, `spectralDecompositionDirectionalDerivative(...)` | Directional derivatives for supported eigen operations |
| `FormExpr::previousSolution(k)` | Previous solution value, `k >= 1`, of the trial field of the block being assembled (not field-bound); history slot k, refused at assembly where the integrator puts a rate or a scheme constant there (slot 2 of first-order generalized-alpha, slots 1 and 2 of second-order generalized-alpha and Newmark). In a multi-field install it (and `historyWeightedSum`, `historyConvolution`) may appear only in a row that depends on no unknown but its own field, where it is that field's history; elsewhere the install is refused, naming the term |
| `FormExpr::historyWeightedSum(weights)` | Weighted history sum |
| `FormExpr::historyConvolution(weights)` | History convolution helper |

### DG, Trace, and Interface Primitives

Status: `implemented/public`.

| Vocabulary | Notes |
|------------|-------|
| `u.minus()`, `u.plus()`, `minus(u)`, `plus(u)` | Interior/interface side restrictions |
| `jump(u)`, `avg(u)` | `jump(u) = u(-) - u(+)`, `avg(u) = (u(-) + u(+)) / 2` |
| `weightedAverage(u, w_plus, w_minus)` | `w_plus u(+) + w_minus u(-)` |
| `harmonicAverage(k)` | `2 k(+) k(-) / (k(+) + k(-))` |
| `upwindValue(u, beta)`, `downwindValue(u, beta)` | Basic DG upwind/downwind selection using `n(-)` |
| `interiorPenaltyCoefficient(eta, p)` | SIPG scaling `eta p^2 avg(1 / hNormal())` |
| `SidePair{neg, pos}` | One quantity held by two fields, one on each side of an interface (`neg` on `phi < c`, the minus side of `jump()`), e.g. `SidePair{u_neg, u_pos}` of StateFields or `SidePair{v_neg, v_pos}` of their TestFields |
| `jump(pair)`, `avg(pair, w_neg, w_pos)`, `avg(pair)` | `neg - pos` (the `jump()` convention), `w_neg neg + w_pos pos`, and `0.5 neg + 0.5 pos` |
| `map(f, A)`, `map(f, A, B)` | `{f(A.neg, B.neg), f(A.pos, B.pos)}`; `f` may take the side (`CutVolumeSide`) as its last argument for per-side parameters. `pair * e`, `e * pair`, `pair + pair`, `pair - pair` act side by side |
| `bc::normalComponent(u)` | Scalar normal trace `inner(u, n)` |
| `bc::interfaceNormalComponent(u, reduction)` | Interface scalar normal trace helper |
| `bc::applyScalarTrace(u, op)`, `bc::applyInterfaceScalarTrace(u, op, reduction)` | Scalar trace and side reduction |
| `bc::ScalarTraceOperator` | `Identity` or `NormalComponent` scalar trace selection |
| `bc::InterfaceTraceReduction` | `Minus`, `Plus`, `Jump`, or `Average` interface reduction |

`jump()` and `avg()` are only defined in interior-facet and interface
integrals. The form compiler rejects them in a cell, cut-volume or
exterior-boundary integral when the form is compiled (at installation), with
the operator, the integral and the term in the message, e.g. `FormCompiler:
jump() is defined only on interior-facet (dS) and interface (dI) integrals,
but a cell integral (dx) uses it: [[u]] in the integrand '...'`. The normal
of the minus side is `FormExpr::normal().minus()`; the DG helpers above write
fluxes against it. On a generated (level-set) interface the minus side is the
negative side `phi < c` and the plus side the positive one, so `jump(u)` is
`u(negative) - u(positive)`; in a two-sided term there `normal().minus()` is
`-n` and `normal().plus()` is `+n` with `n = grad(phi) / |grad(phi)|`, each
pointing into its own side, the opposite of interior facets
(`FE/Docs/LevelSet.md`, Two-Sided Generated Interfaces).

The BC trace helpers follow the core convention:
`bc::InterfaceTraceReduction::Jump` is `jump(u) = u(-) - u(+)` for
`ScalarTraceOperator::Identity` and `jump(u).n(-) = u(-).n(-) + u(+).n(+)`
for `NormalComponent`, so consistency fluxes passed to
`applyInterfaceTraceNitsche` or `InterfaceTraceNitscheBC` are written on
`n(-)`, and a prescribed jump `g` means `u(-) - u(+) = g`
(`test_InterfaceTraceConventions.cpp`). `Average` is `avg(u)` for the
Identity trace and the average flux `avg(u).n(-) = (u(-).n(-) - u(+).n(+)) / 2`
for `NormalComponent`.

A coupling of two fields is one integral: the symmetric Nitsche coupling of
two phases is `(-inner(avg(map(sigma, U, P) * n, w_neg, w_pos), jump(V)) -
inner(avg(map(sigma, V, Q) * n, w_neg, w_pos), jump(U)) + gamma *
inner(jump(U), jump(V))).dI(m)` with `U`, `P`, `V`, `Q` side pairs, and a
trace multiplier's is `(lam * v + mu * (u - g)).ds(m)`. `installFormulation`
splits an integral that holds the test functions of several fields into one
integral per test field (see Installation and Compilation).

This is enough for current weak trace loads, trace Robin terms, Nitsche terms,
interface exchange, and simple DG terms. First-class normal/tangential jump
families are not implemented.

### Boundary Conditions and Constraints

Status: `implemented/public`, unless noted. Free functions and BC classes are in
namespace `forms::bc`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `bc::ScalarValue` | `BoundaryConditions.h` | `std::variant<Real, ScalarCoefficient, TimeScalarCoefficient, FormExpr>` for option structs |
| `bc::toScalarExpr(value, name)` | `BoundaryConditions.h` | Converts a `ScalarValue` (or any alternative) to a `FormExpr`; callbacks become named coefficients |
| `bc::toVectorExpr(values, dim, prefix[, marker, style])` | `BoundaryConditions.h` | Per-component `std::vector<FormExpr>`; wrap with `FormExpr::asVector` for a vector expression |
| `bc::isConstantScalarValue`, `bc::isZeroConstantScalarValue` | `BoundaryConditions.h` | Literal tests on a `ScalarValue` |
| `bc::validateScalarLiteral(value, predicate, message)` | `BoundaryConditions.h` | Applies the numeric predicate only to a `Real` literal; spatial/time coefficients and form expressions are left untouched, including constant expressions. Does not sample callbacks |
| `bc::requireScalarLiteral(value, message)` | `BoundaryConditions.h` | Returns the `Real` alternative or throws the caller's exact message; imposes no sign or finiteness check |
| `numeric::isFinite`, `numeric::isNonnegative`, `numeric::isPositive`, `numeric::validate(value, predicate, message)` | `Core/NumericValidation.h` | Raw numeric checks in namespace `FE::numeric`; finiteness uses `std::isfinite`, sign comparisons alone accept positive infinity. The caller declares whether both checks are required and their diagnostic order |
| `bc::markerValueName`, `componentValueName`, `indexedComponentValueName` | `BoundaryConditions.h` | Coefficient naming helpers |
| `bc::boundaryMarkerOrThrow(bc, where)` | `BoundaryConditions.h` | The `boundary_marker` of an option struct; throws, naming `where`, when it is negative |
| `bc::hasForbiddenPrescribedValueDependency(node)` | `BoundaryConditions.h` | Whether a prescribed value reads a test/trial function, FE state, coupled or auxiliary data, material state, moving or variational geometry, a constitutive call or an integral |
| `bc::StrongDirichlet`, `strongDirichlet(...)` | `BoundaryConditions.h` | Strong scalar/component value constraints. The value is evaluated at each constrained node (active coordinates) and may read `x`, `t`, `deltat`, time functions (`systems::timeFunction`), prescribed fields that share the constrained field's scalar element, and `FormExpr::normal()`: the marker's facet normals averaged at the node as for `NormalComponentConstraint` (`constraints::boundaryPointNormals`, the same at any rank count), refused at a node where the facets meet at an edge or corner; it may not read an unknown |
| `bc::BoundaryData<Value>`, `bc::ComponentValues`, `bc::RobinValue` | `BoundaryConditions.h` | Typed data of one condition: a marker and a `ScalarValue`, a vector value with the components it prescribes, or a Robin pair (`alpha u - rhs`) |
| `bc::value(data, prefix)`, `bc::values(data, dim, prefix[, style])` | `BoundaryConditions.h` | The data as forms; a callback is named after the prefix and the marker (`markerValueName`, `toVectorExpr`) |
| `traceInequalityViolation(...)`, `applyTraceInequality(...)` | `BoundaryConditions.h` | One-sided scalar trace laws |
| `applyTraceNitsche(...)` | `BoundaryConditions.h` | Boundary Nitsche scalar trace imposition with explicit consistency flux, adjoint flux, and penalty weight |
| `applyInterfaceTraceNitsche(...)` | `BoundaryConditions.h` | Interface Nitsche scalar trace imposition |
| `buildTraceNitschePenalty(...)` | `BoundaryConditions.h` | `gamma p^2 * penalty_weight` |
| `bc::NitscheVariant`, `NitscheDirichletOptions` (`TraceNitscheOptions`) | `BoundaryConditions.h` | Symmetric/unsymmetric Nitsche options |
| `TraceInequalitySense`, `TraceInequalityLinearization`, `TraceInequalityOptions` | `BoundaryConditions.h` | One-sided trace-law options |
| `buildGeneratedBoundaryNitscheTraceTerms(u, v, g, form, ...)`, `symmetricGradientForm(k)`, `GeneratedBoundaryNitscheTraceFormBinding` | `BoundaryConditions.h`, `NitscheTraceBilinearForm.h` | `implemented/expert`; Nitsche terms on a generated boundary consistent with a declared bilinear form (`NitscheTraceBilinearForm::SymmetricGradient`, 2 k eps(u) : eps(v)), with an immutable binding for the installer's trace certificate, which bounds the form's flux by its energy |
| `EssentialBC`, `ComponentEssentialBC`, `vectorComponentEssentialBC(...)` | `StandardBCs.h` | Strong values: scalar, dense vector, or selected components |
| `bc::naturalBC(data, prefix)`, `robinBC(data, alpha_prefix, rhs_prefix)`, `essentialBC(data, prefix, symbol)`, `componentEssentialBC(data, dim, prefix, symbol, source)` | `StandardBCs.h` | The conditions of typed `BoundaryData`; the same objects, trees, manifests and JIT keys as the spelled-out `NaturalBC(m, toScalarExpr(g, markerValueName(prefix, m)))` and its siblings (Poisson builds its conditions so) |
| `NaturalBC`, `RobinBC` | `StandardBCs.h` | Weak flux and Robin terms on `ds(marker)` or an `ExteriorBoundaryMeasure` |
| `ReservedBC` | `StandardBCs.h` | Reserves a marker for conflict validation; contributes nothing |
| `NormalTraceEssentialBC` | `StandardBCs.h` | Strong `H(div)` normal-trace data (`HDivNormalConstraint`) |
| `NormalEssentialBC`, `makeNormalEssentialBC(marker, g)` | `StandardBCs.h` | Strong u.n = g of a vector Lagrange field on one marker (free slip for g = 0) through `constraints::NormalComponentConstraint`; the tangential components keep their natural condition |
| `constraints::NormalComponentConstraint(field, markers, g, {feature_angle_degrees})` | `Constraints/NormalComponentConstraint.h` | `implemented/expert`; the affine lines of u.n = g after deal.II's normal-flux constraints: facet normals averaged (by facet measure) within the feature angle, one slaved component per direction (the largest determinant), every component at a corner; the same lines at any rank count (contributions summed at the DOF owner in global facet order). Points an earlier constraint holds are left to it; walls that meet at an edge belong in one constraint. `constraints::boundaryPointNormals(system, field, markers)` gives the averaged normal of each owned point where the facets have one direction. No deck uses it yet |
| `TraceLoadBC`, `TraceRobinBC`, `TraceInequalityBC` | `StandardBCs.h` | Boundary scalar trace wrappers |
| `InterfaceTraceLoadBC`, `InterfaceTraceRobinBC`, `InterfaceTraceJumpPenaltyBC` | `StandardBCs.h` | Interface scalar trace wrappers |
| `makeNormalTraceEssentialBC`, `makeTraceLoadBC`, `makeTraceRobinBC`, `makeTraceInequalityBC`, `makeInterfaceTrace*BC` | `StandardBCs.h` | Factory helpers for the trace wrappers |
| `ScalarNitscheBC`, `TraceNitscheBC`, `InterfaceTraceNitscheBC` | `NitscheBC.h` | Nitsche wrapper types; `ScalarNitscheBC` is a legacy scalar-diffusion convenience |
| `LagrangeDirichletBC`, `makeLagrangeDirichletBC(marker, g, lambda, beta = 0, k = 1)` | `LagrangeDirichletBC.h` | u = g on a physical boundary through a multiplier field restricted to it (D54 T1): `(lambda * v).ds + (mu * (u - g)).ds`, lambda = -k du/dn; with beta > 0 the Barbosa-Hughes term `-(beta h (lambda + k du/dn)(mu + k dv/dn)).ds` (equal-order pairs). Install the residual with lambda in the field list; exclusive on its marker. A multiplier continuous around a corner cannot follow the jump of du/dn there: give each side its own multiplier |
| `PeriodicBC`, `MultiPointConstraintBC` | `ConstraintBCs.h` | Affine constraints installed through `BoundaryConditionManager` |
| `bc::BoundaryCondition` | `BoundaryCondition.h` | `implemented/expert`; interface for new BC wrappers. Each wrapper names its kind in `label()` (its class name), which the system manifest prints for the conditions a manager installs as affine constraints |
| `interface::NitscheVariant`, `NitscheInterfaceOptions` | `InterfaceConditions.h` | Interface Poisson Nitsche options |
| `interface::scalarContinuityPenaltyTerms(...)` | `InterfaceConditions.h` | Equal-and-opposite continuity penalty terms of two traced sides |
| `interface::nitschePoissonIntegrand(...)` | `InterfaceConditions.h` | Legacy interface Poisson integrand helper; prefer explicit interface trace terms |
| `interface::applyNitscheInterfacePoisson(...)` | `InterfaceConditions.h` | Legacy interface Poisson helper over `.dI(marker)`; the same terms as `nitschePoissonIntegrand`, both tested for consistency with a linear solution |
| `systems::BoundaryConditionManager::add`, `install(options, factory)` | `Systems/BoundaryConditionManager.h` | Collect BC objects, one factory call per option entry |
| `BoundaryConditionManager::applyAll(system, residual, u, v, field)` | `Systems/BoundaryConditionManager.h` | Validate markers, add weak terms to the residual, install strong and affine constraints |
| `BoundaryConditionManager::applyAll(system, field)` | `Systems/BoundaryConditionManager.h` | Strong/affine-only BCs of a field; throws before any mutation if a BC has weak terms |

Use equality/Robin/Nitsche helpers for equality-style laws such as
`tau(u) = g` or weak continuity. Use `TraceInequalityBC` for one-sided laws
whose active set can change during nonlinear solve. `H(curl)` tangential
essential conditions have no BC wrapper; the homogeneous
`constraints::HCurlTangentialConstraint(field, marker)` is added with
`FESystem::addSystemConstraint`.

Exterior weak-form helpers and wrappers accept
`ExteriorBoundaryMeasure::generatedActiveSubset(physical_marker,
generated_marker)` when a background exterior face is sharply clipped.
This preserves physical-boundary ownership while dispatching assembly through
the provenance-checked generated domain. A bare interface integral is not an
equivalent exterior-boundary selector. The selector establishes routing, not
a cut-stability certificate for a user-supplied Nitsche penalty.
Bare integer-marker overloads remain legacy routes and fail closed when the
same physical marker has generated-active provenance. Use
`ExteriorBoundaryMeasure::fullPhysical(physical_marker)` only for an
intentional complete-background-face operator.
The selector overloads of `CouplingFormBuilder::integrate(...)` additionally
verify that the selector's physical marker matches the coupling boundary
region.

Advanced `H(div)` trace infrastructure also exposes periodic/MPC pairing
helpers and marker-scoped mortar/interface field workflows through the Spaces
and Systems layers. See `Docs/HDIV_ADVANCED_USAGE_GUIDE.md` for those flows.

An interface-owned field (`addInterfaceField(name, space, marker)`, or a
`MortarSpace` field: `FieldScope::InterfaceFace`) may appear in forms on its
interface (D54 T2): `(lam * v.minus()).dI(m) - (lam * w.plus()).dI(m) +
(mu * (u.minus() - w.plus())).dI(m)`.  The installer splits such `dI(m)`
terms by side (`Systems/InterfaceSideSplit.h`): a product holding `u.minus()`
is evaluated on the minus cell of each interface face, one holding `u.plus()`
on the plus cell, one holding only the interface field on the minus cell;
the restrictions are dropped and dI's normal is re-signed on the plus side
(-n), and the interface field's basis is evaluated on the interface face's own
parameterization (also on triangular and quadrilateral faces, whichever way
the side's facet is turned).  The interface field appears in `dI(m)` terms
only, also on an interface built from a boundary label (not in `dx` or `ds`,
and not in the row of a global field); it takes no `.minus()`, `.plus()`,
`jump()` or `avg()` (it has one value); a volume field in such a term needs
its side; a product of the two sides, or of the interface-owned unknown with
another unknown, is refused, as is `dI()` without a marker.  The kernels are
interpreted AD kernels, also with the JIT on and with symbolic tangents.

### Constraint Plan and Gauge Anchors

Status: `implemented/public` for `ConstraintPlan`; the individual constraint
classes are `implemented/expert`. Namespace `constraints`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `ConstraintPlan(system)` | `Constraints/ConstraintPlan.h` | Holds declarations and registers them in the phase order `VertexValues < Essential < InactiveSupport < Aggregation < PointLaw`, then declaration order |
| `plan.vertexValues(field, values, mode)` | `Constraints/ConstraintPlan.h` | `VertexDirichletConstraint` (pressure pins) |
| `plan.essentials(install)` | `Constraints/ConstraintPlan.h` | Runs a callback that registers essential constraints, typically `BoundaryConditionManager::applyAll` or `installStrongDirichlet` |
| `plan.inactiveSide(fields, cut_side, value)` | `Constraints/ConstraintPlan.h` | Pins the inactive side of a cut (`LevelSetActiveSideVertexDirichletConstraint`) |
| `plan.restrictSupport(fields, domain)`, `plan.restrictSupport(fields, geometry)` | `Constraints/ConstraintPlan.h`, `Constraints/SupportRestrictionConstraint.h` | InactiveSupport: a trace multiplier written as a volume field (D54 T1). `SupportRestrictionConstraint` pins every DOF whose basis function vanishes on the domain: off the closure of a physical boundary (`BoundaryDomain::fullPhysical(b)`), of a fitted interface (an `InterfaceDomain` with an InterfaceMesh), or off the cells the generated interface of a `LevelSetGeometry` cuts; that support follows the installed cut integration context, so a new cut epoch re-pins without a new setup. H1 scalar or Product fields; DOFs another constraint fixes keep it. Setup never makes the restricted field a gauge candidate, takes its rows' dependence on a scalar field as that field's anchoring, and refuses a multiplier DOF whose rows couple only to fixed DOFs (an essential condition on the same DOFs) |
| `plan.aggregate(fields, cut_side, guards)` | `Constraints/ConstraintPlan.h` | Small-cut aggregation of the active side (`SmallCutAggregationConstraint`). It refuses, on every rank, an overlap that would make the lines or the Jacobian depend on the partition (a condensing rank without the masters of a slave's preferred root, or a row coupled to a slave without its masters; `InsufficientGhostHaloError` names the `<Ghost_layers>` to set), and a cut that would make every DOF of the active side a slave |
| `plan.contactPointVelocity(velocity, law)`, `plan.contactPointVelocity(velocity, geometry, wall, wall_normal, speed, name)` | `Constraints/ConstraintPlan.h` | Strong contact-point velocity law (`ContactPointVelocityConstraint`, 2D, axis-aligned planar walls) |
| `plan.declare(phase, constraint)` | `Constraints/ConstraintPlan.h` | Any `ISystemConstraint` in a phase |
| `plan.commitThrough(phase)`, `plan.commit()` | `Constraints/ConstraintPlan.h` | Registers held declarations; declaring into a passed phase throws `std::logic_error` |
| `plan.anchorGauge(operator_tag, anchor)` | `Constraints/ConstraintPlan.h` | Not a phase: records an `InterfaceMeasureGaugeAnchor` in the gauge registry and adds its measure checks |
| `InterfaceMeasureGaugeAnchor`, `InterfaceMeasureGaugeAnchorTexts`, `anchorGaugeOnInterfaceMeasure(...)` | `Constraints/InterfaceMeasureGaugeAnchor.h` | A natural condition on a generated interface anchors a constant mode while the interface has positive measure; the solve is rejected otherwise |
| `ContactPointVelocityLaw`, `ContactPointVelocitySample` | `Constraints/ContactPointVelocityConstraint.h` | Law data (`contact_speed(cos theta)`) and per-point samples |
| `GlobalConstraint::zeroMean(dofs)`, `fixedMean`, `pinDof` | `Constraints/GlobalConstraint.h` | `implemented/expert`; DOF-level global constraints. `GlobalConstraintStrategy` is `PinSingleDof`, `WeightedMean` or `NullspaceProjection`, and `NullspaceProjection` (so also `GlobalConstraint::nullspacePinning`) is a pin today: it pins one DOF like `PinSingleDof`. A constraint enforced by a multiplier is a global field in the form (`GlobalSpace`, `.dGlobal()`; D54), not a strategy or a gauge policy |
| `FESystem::gaugeRegistry()` | `Systems/FESystem.h` | `implemented/expert`; automatic constant-mode detection: a pure-Neumann scalar problem gets a zero-mean constraint without author code. A field that enters another field's equation undifferentiated (the pressure in `p div(v)`) is anchored by a boundary marker on which the other field has an unconstrained DOF, so the pressure of an enclosed flow gets a gauge. A global multiplier's row that a constant shift of a scalar field changes (a mean-value constraint such as `(mu * p).dx()`) anchors that field's constant, so no gauge is added; global fields are never candidates |
| `boundaryDofsByMarker(mesh, dof_handler, marker)`, `boundaryDofsByFaceSet(mesh, dof_handler, set)`, `makeDirichletConstantByMarker`, `makeDirichletConstantByFaceSet` | `Constraints/SystemConstraints.h` | `implemented/expert`; the DOFs of the vertices, edges and faces of a marker's or a face set's faces, with the edges numbered as the DofHandler numbers them (in 2D with every face stored, edge f is face f); the gauge reads natural boundaries from them |

### Installation and Compilation

Status: `implemented/public` for the Systems install API; compiler/IR APIs are
`implemented/expert`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `installFormulation(system, op, fields, residual[, options])` | `Systems/FormsInstaller.h` | Canonical residual installation; one or several test fields. A single-field residual affine in the unknown, without `dt()`, `dS` or `dI` terms, coupled terminals or geometry sensitivity, is installed through the linear kernel (`trySplitAffineResidual`). A term of the residual's sum (usually one integral) that holds the test functions of several fields is split into its part in each field's row: the other fields' test functions set to zero, with the zero terms dropped from sums, products and linear operators, and a subtree holding one field's test functions kept as it is (so `(A(v) + B(q)).dx()` installs the kernels of `A(v).dx() + B(q).dx()`). A product, quotient or other function of two fields' test functions is refused, naming the term |
| `installFormulationWithMetadata(...)` | `Systems/FormsInstaller.h` | Same, returning the analysis bridge metadata of the install |
| Global rows: `installFormulation(system, op, {u, lam}, R)` with `mu = TestField(lam, R1, "mu")` | `Systems/FormsInstaller.h`, `Systems/GlobalRowKernel.h` | The row of a global test function is lowered in two parts: its measured terms give the row's Jacobian through element kernels installed matrix-only (one kernel per block, no fused cell kernels), and a `GlobalRowKernel` adds its residual entry, the measured terms with `mu = 1` as a measured 0-form evaluated exactly over each rank's owned cells (`evaluateFunctionalFormsWithQuadratureOf`, the same at any rank count) plus the `.dGlobal()` terms, whose derivatives in the global fields it adds to the matrix. The entry is the residual at the state: assemble with `suppress_constraint_inhomogeneity` (as Newton does) or homogeneous constraints. A global field in the field list needs its row and must enter some row (its column; its own row counts), both checked per `installFormulation` call: a multiplier's `lam * v` and `u * mu` go in one call. The one-field installers refuse a global row |
| `FormInstallOptions` | `Systems/FormsInstaller.h` | `ad_mode` (`Forward`, the forward-mode dual numbers; `Reverse` and `Taylor` are not implemented and refused), `compiler_options` (`SymbolicOptions`), `extra_trial_fields`, `constitutive_laws`, `source_component_tag`, `contributes_to_problem_analysis`, generated-boundary Nitsche trace requests |
| Trial fields of an install | `Systems/FormsInstaller.h` | The fields of the list, the extra trial fields, and the fields of the geometry sensitivity (the mesh displacement, the level-set fields of a cut domain). In an install that contributes to the problem analysis, a `StateField` of an unknown that is none of them is refused, naming the term that holds it: its Jacobian columns would be dropped. A value held fixed during the solve is data (`frozen()`, `FormExpr::discreteField`); a `StateField` of a prescribed field is data too. Measurement installs (`contributes_to_problem_analysis = false`: residual-work channels, diagnostic operators) are assembled for their vectors and are not checked |
| Installer errors | `Systems/FormsInstaller.h` | Name the public entry (`installFormulation: ...`), the term and the fields concerned |
| `FormInstallOptions::recordConstitutiveLaw(law)` | `Systems/FormsInstaller.h` | `implemented/expert`; records a constitutive law (`analysis::ConstitutiveLawMetadata`, e.g. `analysis::dynamicViscosityMetadata(u, mu)`) for the coupling contracts |
| `installStrongDirichlet(system, bcs)` | `Systems/FormsInstaller.h` | Strong constraint installation |
| `installMixedBilinear(...)` | `Systems/FormsInstaller.h` | Explicit bilinear block installation |
| `installMixedLinear(...)` | `Systems/FormsInstaller.h` | Explicit linear block installation |
| `installMixedFormIR(...)` | `Systems/FormsInstaller.h` | Expert precompiled IR installation |
| `FormCompiler::compile*`, `compileMixed(...)` | `Forms/FormCompiler.h` | Expert compile-without-install workflows |
| `FormIR`, `MixedFormIR`, `FormKernels` | `Forms/*IR.h`, `Forms/FormKernels.h` | Expert IR/kernel surfaces |
| `trySplitAffineResidual(...)` | `Forms/AffineAnalysis.h` | `implemented/expert`; splits an affine residual into bilinear and linear parts |

### Formulation Services

Status: `implemented/public` for `FormulationBuilder` and `FieldBinding`;
the rest `implemented/expert`. Namespace `systems`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `FormulationBuilder(system, operator_tag, base_options)` | `Systems/FormulationBuilder.h` | One set of install options for a module; adds the operator when missing |
| `F.symbolicTangent()`, `F.configure(ale)`, `F.recordConstitutiveLaw(law)` | `Systems/FormulationBuilder.h` | Option changes, effective in call order |
| `F.multiplierLowering(MultiplierLowering::Auto \| InMatrix \| Bordered)`, `FESystem::requestMultiplierLowering(lowering, requester)`, `SetupOptions::multiplier_lowering`, `FESystem::multiplierLowering()` | `Systems/FormulationBuilder.h`, `Systems/MultiplierLowering.h`, `Systems/FESystem.h` | How global multipliers reach the linear algebra, chosen at setup; a formulation's request overrides `Auto`, and conflicting requests or options throw. `Auto` is `InMatrix` (every global row and column in the matrix). `Bordered` (D54, LM2), the route for backends with a fixed node layout and the one the Application takes with FSILS, keeps the global DOFs out of the matrix: see the bordered lowering row below |
| `MultiplierLowering::Bordered`, `FESystem::hasBorderedGlobalFields()`, `borderedGlobalValues()`, `setBorderedGlobalValues(values)`, `globalFieldValues(field, state)`, `BorderedGlobalSplitView`, `BorderedGlobalReadView`, `makeStateReadView` | `Systems/FESystem.h`, `Systems/BorderedGlobalViews.h` | The bordered lowering: the global fields stay out of `dofHandler()` (numbering, partition, field map, sparsity, backend vectors); their DOFs follow the field DOFs (`fieldDofOffset` of a global field is past `getNumDofs()`) and their values live in the system. Assembly sends the field entries to the backend and the global rows and columns to `borderedCoupling()` (B, Ct, D and g, summed over the ranks); Newton solves them with the explicit bordered Schur correction for any D, D = 0 or a stabilized multiplier's invertible D (one Krylov solve per global DOF and iteration). The values have a committed copy (`commitTimeStep`; restored by `beginTimeStep(reset)`, `rollbackAuxiliaryState` and the rollbacks of a rejected step) and travel in the auxiliary checkpoint (line search, rollbacks); the external-state outer loop measures their change but its dynamic relaxation is refused with them, and Newton's step-tolerance norm leaves them out, as it does the auxiliary unknowns. `globalFieldValues` reads a global field under either lowering. Setup refuses a gauge-type multiplier (its row the only anchoring of a constant mode, so the field block is singular: a mean-zero pressure, a pure-Neumann mean value), time derivatives in the system, and monolithic auxiliary unknowns, each for now |
| `F.install(fields, residual)` | `Systems/FormulationBuilder.h` | `installFormulation` with `F.options()` |
| `F.installGroup(fields, residual, FormulationGroup{...})` | `Systems/FormulationBuilder.h` | A separate kernel group with a copy of the options changed by `source_component_tag`, `extra_trial_fields`, `ale`, or dropped trace requests |
| `F.registered(ContactLineDomain)`, `F.registered(BoundaryDomain)` | `Systems/FormulationBuilder.h` | Registers a generated marker with the system and returns the domain |
| `F.certifyGeneratedBoundaryNitscheTraces(bindings, marker, ratio)` | `Systems/FormulationBuilder.h` | `implemented/expert`; trace-certificate requests on a cut volume |
| `F.functional(name, kind, form[, owner])` | `Systems/FormulationBuilder.h` | Declares a functional of the formulation as a measured 0-form (a sum of dx, ds, dI, dL or dCutVolume integrals with no test or trial function; dS is rejected); nothing is installed; the owner defaults to the operator |
| `declareFunctionalForm(system, FunctionalFormDeclaration)`, `functionalForms(system)`, `FunctionalKind` (`Stored`, `Power`, `DissipationRate`, `Measure`) | `Systems/FunctionalForms.h` | The system's declared functionals, in declaration order; declared before setup, names unique |
| `F.wallFunctional(WallFunctionalFormDeclaration)`, `declareWallFunctionalForm(system, ...)`, `wallFunctionalForms(system)`, `wallFunctionalForm(declaration, wall, contact_lines, wetted_parts)`, `WallMeasure` (`ContactLines`, `WettedPart`) | `Systems/FunctionalForms.h` | A functional of every wall an interface meets, stated by its integrand (no measure; one for every wall and/or one per wall marker) and integrated by the evaluator over the wall's contact lines (dL) or wetted part (ds) in the geometry snapshot (`interfaces::declaredWallFunctionalForms`); named `<prefix>wall.<b>.<quantity>` |
| `evaluateFunctionalForms(system, forms, state, comm[, cut_context])`, `evaluateDeclaredFunctionalForms(system, state, comm[, cut_context])`, `FunctionalFormValue` | `Systems/FunctionalFormEvaluator.h` | Collective values of measured 0-forms over `dx`, `ds`, `dCutVolume`, `dI`, `dL` and generated `ds` (cut rules from the cut context, for a free surface its authoritative snapshot), interpreted; every quadrature term is summed exactly over all ranks, so `value` and `absolute_mass` are the same at any rank count; `dS` is not evaluated |
| `F.workChannel(suffix, {WorkChannelPart{...}, ...})` | `Systems/FormulationBuilder.h` | Installs a residual-work channel on operator `<operator>_<suffix>`: one install per part with the builder's options, outside the problem analysis, trace requests dropped unless the part keeps them |
| `F.diagnostics()`, `F.diagnostic(set, operator, fields, form[, options])` | `Systems/FormulationBuilder.h` | The system's diagnostic registry; installs an opt-in diagnostic operator with the builder's options, as a measurement outside the problem analysis (`contributes_to_problem_analysis = false`) |
| `LogRecord(owner, message[, level])` with `.text`, `.quoted`, `.integer`, `.fixed`, `.general(key, value)`; `logRecord(record)`, `F.log(record)` | `Core/LogRecord.h`, `Systems/FormulationBuilder.h` | A log line as data: `<owner>: <message> key=value key='value' ...`, the keys in order and each value's format declared; printed by FE at the point that states it |
| `DiagnosticRegistry::declareSet`, `enabled`, `install`, `operators`; `diagnosticRegistry(system)`, `findDiagnosticRegistry(system)` | `Systems/DiagnosticRegistry.h` | Opt-in diagnostic operators, switched on by environment names, installed with the formulation's options; one registry per system, shared by its formulations (a set declared again with the same switches keeps its operators) |
| `resolveALEBinding(system, ALEBindingOptions)`, `ALEBinding::configureInstallOptions` | `Systems/ALEBinding.h` | Prescribed or coupled mesh velocity; sets geometry sensitivity and extra trial fields |
| `movingDomain(system, d[, MovingDomainOptions])`, `findMovingDomain(system)`, `MovingDomain::binding()` | `Systems/ALEBinding.h` | Declares once, before setup, that the domain moves with the vector unknown `d` (x = X + d): the current configuration becomes the assembly frame (`FESystem::setAssemblyConfiguration`), `d` and its derived mesh velocity are bound (`resolveALEBinding`, coupled displacement), and every later `installFormulation` on the system differentiates its geometry in `d` (MeshMotionUnknowns, the declaration's tangent path) with `d` added to its trial fields, as `configure(ale)` does for one builder. An install keeps its geometry constant with `FormInstallOptions::moving_domain_geometry = false` (`F.geometryConstant()`), e.g. a mesh-motion equation on the reference mesh; an install whose geometry sensitivity is set explicitly is left as it is. Not yet used by the Application, which still builds the frame from the deck |
| `FESystem::setAssemblyConfiguration(cfg)` | `Systems/FESystem.h` | The constructor's coordinate configuration, chosen before setup on a system with a native mesh |
| `resolveMeshDisplacementBinding(...)` | `Systems/MeshDisplacementBinding.h` | Mesh-displacement unknown of a mesh-motion module |
| `FESystem::attachment<T>()`, `findAttachment<T>()` | `Systems/FESystem.h` | Per-system storage of a component that extends the system, one per type |
| `collectiveExactSum(...)`, `collectiveExactSums(...)`, `ExactAccumulator` | `Systems/SystemCollectives.h`, `Systems/ExactAccumulator.h` | Sums that are bitwise identical at any rank count |
| `CollectiveCommunicator`, `collectiveSumDouble`, `collectiveMin/MaxDouble`, `agreeOnLocalPhase(...)` | `Systems/SystemCollectives.h` | Collective reductions and agreed failure of rank-local phases |

### Level-Set Interface Certificates

Status: `implemented/expert`. Namespace `interfaces`.

| Vocabulary | Header | Notes |
|------------|--------|-------|
| `planarWallEvidence(system, wall, normal)` | `Interfaces/ContactLineGeometryCertificate.h` | Every owned wall face is a first-order plane with the configured normal, reduced over the communicator |
| `certifyContactLineRules(system, context, check)` | `Interfaces/ContactLineGeometryCertificate.h` | Contact rules are codimension-two, aligned with the wall, and transverse to it |
| `certifyContactLineRulesCollectively(...)` | `Interfaces/ContactLineGeometryCertificate.h` | The same with the failure agreed over the ranks |
| `ContactLineGeometryCheck`, `kContactLineAlignmentTolerance`, `kContactLineMinimumTransverseSine` | `Interfaces/ContactLineGeometryCertificate.h` | Check data and tolerances |

### Symbolic Differentiation and AD

Status: `implemented/public` for normal residual-to-Jacobian workflows and
`implemented/expert` for direct symbolic differentiation helpers.

| Vocabulary | Notes |
|------------|-------|
| Residual form to Jacobian | Forward-mode dual evaluation in nonlinear kernels (`ADMode::Forward`), or the symbolic tangent with `use_symbolic_tangent` |
| `derivative(form, u, w)` | `implemented/public` (`Vocabulary.h`); the Gateaux derivative `d/de form(u + e w)` at `e = 0` in the field of the StateField `u`: with a measured 0-form (an energy `Pi`) and `w = v` the residual `dPi/du[v]`, with `w = u` the rate of a dissipation potential (`2 D` for a quadratic one). Every StateField of the field varies; its frozen values (DiscreteField, `frozen()`), test functions, other fields and the geometry are held fixed. Simplified (zero terms dropped); refused for a variable that is not a StateField, a Constitutive call (write the law as a FormExpr) and a form that does not depend on the field |
| `derivative(form, {{u, v}, {lam, mu}})` | The sum over the pairs, the stationarity (KKT) residual of a Lagrangian, with D54 multipliers and `.dGlobal()` terms; fields the form does not depend on drop out |
| `systems::derivative(form, U[, w])`, `systems::derivative(form, {U, Lambda})` | `Systems/FieldBinding.h`; the same for `BoundUnknown`s, `w` defaulting to `U.test` |
| `frozen(e)` | `implemented/public` (`Vocabulary.h`); `e` at the current iterate, held fixed in the Jacobian: every StateField becomes the DiscreteField of the same field, space and name (data to the dual numbers and the symbolic tangent), so `frozen(u)` is the tree `FormExpr::discreteField(u_id, V, "u")` and its kernels. Picard convection `advect(frozen(u), u)`, a frozen stabilization parameter `frozen(tau(u))`, or `(u - frozen(u)) * v` (a residual zero to round-off, the mass matrix as Jacobian). Refused for a TrialFunction, a test function, an integral and a current-configuration geometry terminal |
| `differentiateResidual(...)` | Symbolic residual differentiation helper |
| `directionalDerivativeWrtField(expr, field, w[, discrete_fields_vary])` | Directional derivative helper; field-bound test functions keep their binding |
| `directionalDerivativeWrtNormal(...)`, `directionalDerivativeWrtSpatialCoordinate(...)` | Geometry directional derivatives used by the symbolic geometry tangent |
| `differentiateResidualHessianVector(...)` | Hessian-vector helper |
| `differentiateWrtAuxiliaryOutput(...)` | Auxiliary-output sensitivity helper |
| `simplify(...)`, `extractTermsReferencing(...)` | Symbolic utility helpers |
| `checkSymbolicDifferentiability(...)`, `canDifferentiateSymbolically(...)` | Validation helpers |

What the Jacobian linearizes is stated by the symbols: it differentiates
every StateField of a trial field of the install (the field list, the extra
trial fields and the geometry-sensitivity fields), and treats DiscreteFields
and `frozen()` values as data. In the installed forms of the code base
(audit of 2026-10-30) every DiscreteField reads a prescribed field (the
prescribed level-set velocity, the semi-implicit free-surface reference
velocity) or sits in a functional; the StateFields of unknowns outside an
install's trial fields are in the free-surface residual-work channels and
the Navier-Stokes free-surface diagnostic operators, which are assembled for
their vectors only. `SymbolicOptions::simplify_expressions`,
`exploit_sparsity` and `cache_expressions` are read nowhere.

### Auxiliary Coupling, Functionals, and Derived Results

Status: `implemented/public` for named authoring helpers; slot references are
`implemented/internal`.

| Vocabulary | Notes |
|------------|-------|
| `AuxiliaryInput(name)` | Preferred named auxiliary input reference |
| `AuxiliaryOutput(name)` | Preferred named auxiliary model output reference |
| `AuxiliaryOutput(instance, name)` | Instance-qualified auxiliary model output |
| `AuxiliaryState(name)` | `implemented/expert`; raw auxiliary state access; `BoundaryConditionManager::apply` rejects it in a residual |
| `AuxiliaryInputSlot(slot)`, `AuxiliaryOutputSlot(slot)` | `implemented/internal`; post-resolution slot references |
| `FormExpr::auxiliaryStateRef(slot)` | `implemented/internal`; slot reference |
| `FESystem::boundaryIntegral(measured_form, schedule)` | Registers one exterior-boundary integral, `inner(u, n).ds(outlet)` or `integrand.ds(BoundaryDomain)`, as an auxiliary input: the functional `forms::boundaryFunctional(measured_form)` gives; the handle converts to a `FormExpr` |
| `FESystem::boundaryIntegral(BoundaryFunctional, schedule)`, `boundaryIntegral(integrand, marker, ...)` | Registers a boundary (or, with `is_domain_functional`, cell) integral as an auxiliary input; the handle converts to a `FormExpr` |
| `systems::aux::model(name, [](ModelFacade& m) { ... })` | 0D model DSL: `m.input`, `m.state`, `m.params`, `m << ddt(X) == rhs`, `m << alg(Z) == g`, `m << out("y") == e` |
| `FESystem::deploy(use(model).name(...).boundary(m).monolithic().params(...).bind(...))` | Deployed model; `handle.output(name)` is a `FormExpr` |
| `BoundaryFunctional`, `boundaryFunctional(measured_form, name)` | Boundary/domain functional definition (integrand plus marker; no measure); `boundaryFunctional` reads it from one exterior-boundary integral and throws for anything else |
| `BoundaryFunctionalResults` | Runtime functional output container |
| `compileBoundaryFunctionalKernel(...)` | `implemented/expert`; compiles a scalar integrand to a cell or boundary-face functional kernel |
| `FunctionalAssembler::assembleCutVolumeScalar(kernel, cut_context, marker, side)`, `assembleGeneratedInterfaceScalar(kernel, cut_context, marker)` | `implemented/expert`; a functional kernel on the cut context's volume rules, or on its interface, contact-line or active-boundary rules, of one marker, on owned parent cells; the caller reduces |
| `FESystem::addFunctionalKernel(tag, kernel)`, `evaluateFunctional(tag, state)` | `implemented/expert`; evaluates a compiled functional on a state |
| `post::DerivedResultBuilder(name).scope(...).policy(...).shape(...).expression(expr).build()` | Derived output field from a `FormExpr` (cell average, patch average, L2 projection, ...), added with `FESystem::addDerivedResult` |
| `FormExpr::boundaryIntegral(...)` | `implemented/internal`; legacy boundary-functional symbol, rejected by `BoundaryConditionManager::apply` |
| `FormExpr::boundaryIntegralRef(slot)` | `implemented/internal`; legacy slot reference |

Legacy `boundaryIntegralValue`/`boundaryIntegralRef` terminology is superseded
by `AuxiliaryInput`/`AuxiliaryInputSlot`. `BoundaryFunctional` takes an
integrand and a marker; a functional stated as a measured form
(`FunctionalForms.h`), with interface, contact-line, or cut-volume integrals,
is evaluated by `evaluateFunctionalForms` (`FunctionalFormEvaluator.h`).

### Constitutive Vocabulary

Status: `implemented/public` for calls and indexed outputs; state byte-offset
terminals are `implemented/internal`.

| Vocabulary | Notes |
|------------|-------|
| `constitutive(model, input)` | Unary constitutive call |
| `constitutive(model, {inputs...})` | N-ary constitutive call |
| `constitutive(model, a0, a1, ...)` | Variadic n-ary helper |
| `ConstitutiveCall::out(i)`, `output(i)`, `expr()` | Indexed multi-output selection; `expr()` or the implicit conversion gives output 0 |
| `FormExpr::constitutiveOutput(call, i)` | Lower-level multi-output selection |
| `ConstitutiveModel::evaluate*` | Real and dual constitutive evaluation hooks |
| `outputCount()`, `outputSpec(i)`, expected input metadata | Constitutive metadata |
| `stateLayout()`, `stateVariables()`, `stateSpec()` | Constitutive state layout metadata |
| `parameterSpecs()` | Constitutive parameter metadata |
| `inlinable()` | Inlinable models are required by the symbolic tangent |
| `law(name, input_kind, output_kind, body, metadata)`, `law(name, signature, body, metadata)` | `Constitutive/ExpressionLaw.h`; a law written as forms (`constitutive::FormLaw`): `body(inputs, dim)` returns the output expression, a call `mu(gamma)` is a constitutive node that setup expands into the body's tree, so the symbolic tangent and the strict JIT see the law; no pointwise evaluation; `kindId()` hashes `name`. `isochoricNeoHookeanPK1` and `newtonianViscosity` are written this way; `carreauYasudaViscosityLaw` is the opt-in forms twin of the Carreau-Yasuda class |
| `materialStateOldRef(offset_bytes)` | `implemented/internal`; byte-offset state terminal |
| `materialStateWorkRef(offset_bytes)` | `implemented/internal`; byte-offset state terminal |

Named material-state vocabulary is not implemented. Public form code should not
need to author raw byte offsets.

### Finite Deformation

Status: `implemented/public` helper header, namespace
`forms::finite_deformation`.

`Forms/FiniteDeformationForms.h` provides finite-deformation kinematic helpers:

| Vocabulary | Notes |
|------------|-------|
| `deformationGradient(u[, dim])` | `F = I + grad(u)` |
| `deformationGradientVariation(du)` | Variation of `F` |
| `finiteDeformationDimension(u[, dim])` | `dim` when given (1, 2 or 3), otherwise the topological dimension of `u`'s space; throws for another `dim` and, without one, when `u` is not a field symbol |
| `jacobian(F)` | `det(F)` helper in finite-deformation namespace |
| `inverseDeformationGradient(F)` | `F^{-1}` |
| `inverseTransposeDeformationGradient(F)` | `F^{-T}` |
| `rightCauchyGreen(F)`, `leftCauchyGreen(F)` | `C` and `b` |
| `greenLagrangeStrain(F)`, `almansiStrain(F)` | Strain measures |
| `kinematics(u[, dim])` | Bundled `F`, `J`, `Finv`, `FinvT`, `C`, `b`, `green_lagrange`, `almansi` |
| `linearizeKinematics(u, du)` | Bundled variation expressions |
| `jacobianVariation(F, dF)` | Variation of `det(F)` |
| `inverseVariation(F, dF)` | Variation of `F^{-1}` |
| `rightCauchyGreenVariation(F, dF)` | Variation of `C` |
| `leftCauchyGreenVariation(F, dF)` | Variation of `b` |
| `greenLagrangeVariation(F, dF)` | Variation of Green-Lagrange strain |
| `almansiVariation(F, dF)` | Variation of Almansi strain |
| `scalarCurrentGradientFromReferenceGradient(...)`, `scalarReferenceGradientFromCurrentGradient(...)` | Reference/current scalar gradient transforms |
| `vectorCurrentGradientFromReferenceGradient(...)`, `vectorReferenceGradientFromCurrentGradient(...)` | Reference/current vector gradient transforms |
| `pushForwardVector(...)`, `pullBackVector(...)` | Vector frame transforms |
| `contravariantPiolaPushForward(...)`, `covariantPiolaPushForward(...)` | Piola transforms |
| `nansonMeasureVector(...)`, `nansonMeasureVectorVariation(...)` | Surface measure transforms |
| `pk1InternalVirtualWorkDensity(...)` | Legacy mechanics shortcut; prefer writing the density explicitly in physics modules |
| `initialStressGeometricStiffnessDensity(...)` | Legacy mechanics shortcut; prefer writing the density explicitly in physics modules |

### Moving Frame

Status: `implemented/public` helper header.

`Forms/MovingFrameForms.h` builds the terminals of a named, prescribed moving
reference frame from named runtime parameters
`frame.<name>.<quantity>_<x|y|z>`, which the system resolves to their own
slots at setup like any `FormExpr::parameter`, so they never alias another
parameter (the fixed slots 0-14 of the old layout read whatever parameters
held them):

| Vocabulary | Notes |
|------------|-------|
| `movingFrameParameterNames(frame)`, `movingFrameParameterName(frame, quantity, axis)` | The 15 parameter names of a frame: `origin`, `linear_velocity`, `angular_velocity`, `linear_acceleration`, `angular_acceleration`, each `_x`, `_y`, `_z` |
| `frameVectorParameter(frame, quantity)` | The vector of a frame quantity's three parameters |
| `frameOrigin(frame)`, `frameLinearVelocity(frame)` | Translational frame data |
| `frameAngularVelocity(frame)` | Angular velocity |
| `frameLinearAcceleration(frame)`, `frameAngularAcceleration(frame)` | Acceleration data |
| `frameVelocityAtCurrentCoordinate(frame)` | Rigid-frame velocity at `currentCoordinate()` |
| `frameAccelerationAtCurrentCoordinate(frame)` | Rigid-frame acceleration at `currentCoordinate()` |
| `relativeMeshVelocity(frame)` | `meshVelocity() - frameVelocityAtCurrentCoordinate(frame)` |
| `MovingFrameFormTerminals`, `movingFrameTerminals(frame)` | Bundled helper terminals |

### Cut Cell

Status: `implemented/public` helper header, unless noted.

`Forms/CutCellForms.h` provides parameter-slot terminals for cut-cell and
embedded-boundary formulations, and the facet terms of cut-cell stabilization
(ghost penalty):

| Vocabulary | Notes |
|------------|-------|
| `CutCellParameterSlots` | Slot layout constants |
| `cutVolumeFraction()` | Cut-cell volume fraction |
| `cutSideIndicator()` | Side/phase indicator |
| `cutEmbeddedNormal()` | Embedded-boundary normal |
| `cutStabilizationScale()` | Stabilization scale of the cut cells next to a facet; zero away from cut cells |
| `cutMeasureSensitivity()` | Measure sensitivity |
| `cutNormalSensitivity()` | Normal sensitivity vector |
| `cutQuadratureWeightSensitivity()` | Quadrature-weight sensitivity |
| `CutCellFormTerminals`, `cutCellTerminals()` | Bundled helper terminals |
| `cutAdjacentFacetJump(e)`, `cutAdjacentFacetAverage(e)` | `jump(e)` and `avg(e)` |
| `cutAdjacentFacetGradientJump(e)`, `cutAdjacentFacetNormalGradientJump(e)` | `jump(grad(e))` and its `n(-)` component |
| `cutAdjacentFacetHessianJump(e)`, `cutAdjacentFacetSecondNormalDerivativeJump(e)` | `jump(hessian(e))` and its `n(-) n(-)` component |
| `cutAdjacentFacetIntegral(e, marker[, side])` | `.dS(marker[, side])`; the marker-less overload is deleted |
| `cutCellParameterCount(...)`, `cutCellParametersForRule(...)`, `cutSideIndicatorValue(...)` | `implemented/internal`; fill the parameter slots from a cut rule |

### Coupling Forms

Status: `implemented/expert`. Namespace `coupling`.

`Coupling/CouplingFormBuilder.h` builds forms from participant and field names
of a coupling contract (used by `Physics/Coupling/FSICouplingModule` and the
thermal interface coupling module): `state(participant, field, symbol)`,
`test(...)`, `timeDerivative(...)`, `previousSolution(...)`, geometry terminals
of a participant, `integrate(integrand, region[, boundary_measure])`,
`sharedInterface(name)` with per-side `state`, `test`, and `normal`, and
`equationSet(...)` to declare the coupled equations.

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
  higher-rank coefficient support. Tensor-valued unknowns are not public
  vocabulary.
- Spatial second derivatives are available in supported mappings and expression
  shapes. Non-affine higher-derivative coverage remains mapping- and element
  dependent, with validation preferred over silent approximations.
- Matrix, spectral, history, smooth regularization, interface, cut-volume, and
  generated contact-line measures are implemented and should not be listed as
  future work.
- `FormExpr::parameter(name)` is implemented. `parameterRef(slot)` is an
  internal/lowering hook.
- Slot-indexed auxiliary references and byte-offset material-state references
  are implementation details, not the recommended formulation surface.
- `trace(A)` is tensor trace. Facet/interface trace vocabulary must use names
  such as `normalTrace` or `facetTrace`, not overload tensor `trace`.
- Shapes are checked when a kernel evaluates, not when an expression is built:
  a size or shape mismatch of `+ - *`, `inner` or `doubleContraction` is
  reported at assembly, naming the term and the operand kinds (e.g. `with
  lhs=Vector[3] rhs=Vector[2]`). The form compiler checks earlier that a form
  is a sum of integrals, that it has the test and trial functions its kind
  needs, and that `jump()` and `avg()` appear only in `dS` and `dI` terms.
- The only codimension-two measure is the generated interface-boundary
  intersection (`.dL`, `dInterfaceBoundary`); general embedded-curve, point,
  and quadrature-rule-per-measure vocabulary is not public. Cut quadrature
  orders are inputs of the cut-volume request, not of the form.
- Interface-owned fields enter forms only linearly with one unknown per
  product on one side (see Mixed-Dimensional Transfer Operators above); their
  single-sided kernels are interpreted (D54 LM4).  Global scalar multipliers
  are forms (`GlobalSpace`, `FieldScope::Global`, `.dGlobal()`).

## Planned Vocabulary

### Planned in Milestone MR

Status: `planned`. These steps of milestone MR
(`Documentation/free_surface_program_tracker.md` §5, decisions D49 and D50)
change the authoring surface:

| Step | Planned vocabulary |
|------|--------------------|
| G4(a) | `FormulationBuilder` declarations of functionals, residual-work channels, diagnostic operators, and the external boundary energy |
| G4(b) | `NavierStokesPostProcessing.h`: residual work, diagnostic operators, and energy declarations stated as forms, as in `PoissonPostProcessing.h` and `UstructPostProcessing.h` |
| G1, G2 | Generic accepted-stage and functional records on the attachment store; an energy-balance ledger with named channels and declared identities |
| G5 | `ALEBinding` declarations, `LevelSetGeometry` markers and the contact-certificate hook, field binding, and vertex-field initialization as FE services |
| E3 | Energies, powers, works, and the logged measures as forms in `NavierStokesPostProcessing.h`, under the D49 tolerance gate |
| L1 | Log lines as FE data (`LogRecord`, above): the Navier-Stokes term and post-processing lines are records printed through `F.log`, and the free-surface observer is removed; the energy, record and maintenance lines of the FE services follow |

### Left for Later by D54 (multipliers as forms)

Status: `planned`. Decision D54
(`Documentation/free_surface_program_tracker.md`;
`agent-status/refactor/forms-multipliers-design.md`) is done through its last
phase: global multipliers as forms (LM1), their bordered lowering for fixed
node layouts such as FSILS (LM2), trace multipliers restricted to a boundary or
interface (LM3), interface-owned multipliers in forms (LM4), and the deletion of
the multiplier gauge policy and global-constraint strategy and the retirement
of `constraints::LagrangeMultiplier` (LM5; a multiplier is a field in the
form). What the phases left for later:

| Phase | Planned vocabulary |
|-------|--------------------|
| LM1 (later) | Several scalars per global field (`GlobalSpace(n)`, n > 1); the anchoring evidence of vector fields (componentwise constants); distributed sparsity graphs of global rows for the distributed matrix backends (PETSc, Trilinos) |
| LM2 (later) | The bordered lowering for transient problems and together with monolithic auxiliary unknowns |
| LM3 (later) | A support restriction to a union of boundaries (one multiplier on several markers), to a generated active boundary subset and to a contact line; the restriction of vector fields' rows (componentwise anchoring) |
| LM4 (later) | The JIT for single-sided interface kernels (not bitwise against the interpreted kernels in LM4: a form over two topological dimensions needs `inferAutoIndexExtent` to take the cells' dimension, and the symbolic residual kernels' JIT results then differ by round-off in a few entries); the state of an interface-owned field gathered by face, so a product of it with another unknown can be installed; an interface-owned field in the row of a global field (e.g. its zero mean as a gauge) |

Open limits, one line each (not scheduled):

- Analysis gives no inf-sup verdict for a trace multiplier pair: it classifies the pair (`TraceMultiplier`), but its Fortin construction needs trace DOF metadata it does not have, and `infSupEstimate` only forwards a supplied summary.
- A multiplier continuous around a corner of its boundary cannot follow the jump of du/dn there: it converges at O(h^1/2) and pollutes u; give each side its own multiplier.
- A mesh of several cell types cannot carry a field of order >= 1: `SpaceFactory` refuses such a mesh, and the DOF numbering of a Mesh refuses the variable-order space it would need (the assembler itself takes a space with an element per cell type).
- The material-state lookup of a kernel on an InterfaceMesh face still takes the face id the InterfaceMesh recorded, which goes stale once setup stores more faces (the facets are found by the face's vertices).

### Not Scheduled

Status: `out-of-scope-for-now`. None of the items below has a step in
milestone MR; they stay deferred until a module needs them. They are ordered by
expected leverage.

#### Trace and Interface Algebra

Current public pieces include side restrictions, `jump`, `avg`,
`weightedAverage`, `normalComponent`, normal/tangential trace-projection
helpers, scalar trace BC helpers, `.dI`, and the level-set handles with
`outwardNormal()`. Missing vocabulary:

| Vocabulary | Purpose |
|------------|---------|
| `facetTrace(expr, side)` | Explicit side/trace restriction distinct from tensor `trace` |
| `normalJump(u)`, `tangentialJump(u)` | Orientation-aware trace jumps |
| Vector trace laws | Nitsche and Robin helpers for vector traces (today scalar traces only) |
| `H(curl)` tangential essential BC wrapper | Counterpart of `NormalTraceEssentialBC`, with nonzero data |

Acceptance criteria: common H(div), H(curl), slip, and Nitsche boundary/interface
terms can be written without raw component manipulation against side-specific
normals.  Strong slip of a vector Lagrange field exists
(`NormalEssentialBC`, `NormalComponentConstraint`); weak (Nitsche) slip still
needs the vector trace laws.

#### Named History, Stage, and Material State

Current public pieces include `dt`, `t`, `deltat`, `deltat_eff`,
`previousSolution(k)`, `historyWeightedSum`, `historyConvolution`, `FieldBinding::previous` and `lagged` (above), and named
auxiliary input/output/state terminals. Missing vocabulary:

| Vocabulary | Purpose |
|------------|---------|
| `stage(u, i)` | Time-integrator stage access |
| `stateOld(name)`, `stateWork(name)` | Named material-state access from `StateLayout` |
| Named coupled integral/state aliases where needed | Avoid authoring raw slots in public physics modules |

Acceptance criteria: transient and stateful constitutive forms can be written
with named field/state concepts instead of slot or byte-offset references.

#### Generalized Measures and Quadrature Metadata

Current public measures are listed under `Measures`. Missing vocabulary:

| Vocabulary | Purpose |
|------------|---------|
| Subdomain-set and boundary-set measures | Integrate over marker sets without manual loops |
| Interface-pair/interface-set measures | Nontrivial interface grouping |
| Codimension-2 curve measures beyond generated contact lines | Fractures, fibers, vessel centerlines, embedded curves |
| Point measures / singular source measures | Point sources and probe-style weak terms |
| Embedded manifold measures | Immersed or lower-dimensional physics |
| Per-measure quadrature rule/order metadata | Local quadrature control attached to the measure |

Acceptance criteria: singular sources, embedded sources, and interface-heavy
problems can be expressed without handwritten kernels.

#### Mixed-Dimensional Transfer Operators and Multipliers

Spaces contain `TraceSpace` and `MortarSpace` concepts, and Systems can register
interface fields, but public bulk-to-lower-dimensional transfer vocabulary is
not complete.

| Vocabulary | Purpose |
|------------|---------|
| `restrictTo(manifold, u)` | Restrict a bulk field to a lower-dimensional manifold |
| `traceTo(manifold, u)` | Explicit trace from bulk to a target manifold/interface |
| `extendFrom(manifold, lambda)` | Extend lower-dimensional data back to the bulk |
| `lift(...)` | Lifting/prolongation helper |
| Nonmatching transfer operators | Mortar and embedded coupling |

Suggested sequencing: same-mesh codimension-1 coupling, nonmatching
codimension-1 mortar coupling, then codimension-2 and mixed 0D/1D/3D transfer.

#### Scalar Math Expansion

| Vocabulary | Notes |
|------------|-------|
| `sin`, `cos`, `tan` | Trigonometric functions |
| `sinh`, `cosh`, `tanh` | Hyperbolic functions |
| `atan`, `atan2` | Inverse tangent functions |
| `erf`, `erfc` | Special functions useful for regularized profiles |
| `positivePart`, `negativePart` | Exact or smooth part functions, with clear nonsmooth semantics |
| Complementarity/projector helpers | Contact/inequality formulations if needed |

New scalar functions should ship with AD, symbolic differentiation, interpreter,
JIT/lowering, validation, and tests.

#### Constitutive Ergonomics

Current support already includes n-ary constitutive inputs, multi-output calls,
indexed output selection, output metadata, parameter specs, and state layout
metadata. Missing vocabulary:

| Vocabulary | Purpose |
|------------|---------|
| `call.out("stress")` | Named output selection |
| Public output descriptors for docs/tooling | Discoverable constitutive outputs |
| Public named material state access | Pair `StateLayout` with form vocabulary |
| First-class adapter layer from FE constitutive laws | Reduce custom glue code |

#### Conservation-Law Trace Support

Current DG helpers cover side restrictions, jump/average, weighted/harmonic
averages, upwind/downwind values, and simple interior-penalty scaling.

This is not a core `numericalFlux(...)` wrapper. Flux models should remain
visible in formulation code or in physics-layer helper functions that are
themselves written from Forms expressions. Core Forms should instead make those
explicit flux expressions safer and clearer with better trace/orientation
vocabulary.

| Vocabulary | Purpose |
|------------|---------|
| Side-specific normal helpers | Make the normal used by `minus`/`plus` terms explicit |
| Vector-valued trace helpers | Avoid ad hoc component handling in face fluxes |
| Normal/tangential jump helpers | Make conservation-law and vector trace terms reviewable |
| Reusable penalty-scale primitives | Share element/face scaling without hiding flux policy |

SIPG/NIPG, Lax-Friedrichs/Rusanov, Roe, HLLC, entropy-stable fluxes, farfield
states, wall models, and similar policies should live in the relevant physics
module when needed. They should not become generic Forms vocabulary unless they
can be expressed as domain-neutral mathematical primitives.

#### Richer Tensor and Complex Workflows

| Vocabulary | Purpose |
|------------|---------|
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
