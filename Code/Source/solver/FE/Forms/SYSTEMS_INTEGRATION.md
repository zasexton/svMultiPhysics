# FE/Forms → FE/Systems Integration

How `FE/Forms` weak-form expressions become assembled operators in `FE/Systems`.

## Quick Reference: Which API to Use

| Task | API | Header |
|------|-----|--------|
| Owned unknown/prescribed symbols | `FieldBinding` | `FieldBinding.h` |
| Ordered residual rows and selected observations | `ResidualContributions` | `ResidualContributions.h` |
| Residual physics (single or multi-field) | `FormulationBuilder::install()`; direct `installFormulation()` | `FormulationBuilder.h` / `FormsInstaller.h` |
| Phase-ordered constraints and immediate BC application | `ConstraintPlan` | `ConstraintPlan.h` |
| Measured functionals and residual-work channels | `functional()` / `wallFunctional()` / `workChannel()` | `FormulationBuilder.h` / `FunctionalForms.h` |
| Mixed bilinear operator | `installMixedBilinear()` | `FormsInstaller.h` |
| Mixed linear operator | `installMixedLinear()` | `FormsInstaller.h` |
| Strong Dirichlet BCs | `installStrongDirichlet()` | `FormsInstaller.h` |
| H(div) normal-trace BCs | `BoundaryConditionManager` + `NormalTraceEssentialBC` | `StandardBCs.h` |
| Interface trace BCs | `BoundaryConditionManager` + `InterfaceTraceLoadBC` / `InterfaceTraceRobinBC` / `InterfaceTraceJumpPenaltyBC` | `StandardBCs.h` |
| Trace Nitsche BCs | `TraceNitscheBC` / `InterfaceTraceNitscheBC` | `NitscheBC.h` |
| One-sided trace laws | `TraceInequalityBC` | `StandardBCs.h` |
| H(div) periodic/MPC trace helpers | `makeHDivTracePeriodicBC*()` / `makeHDivTracePeriodicMPC*()` | `ConstraintTools.h` |
| Mixed-dimensional interface fields | `FESystem::addInterfaceField()`, coupled by `.dI(marker)` forms | `FESystem.h` |
| Mortar/hybrid facet fields | `MortarSpace`, coupled by `.dI(marker)` forms (`FESystem::addInterfaceFaceKernel()` for hand-written kernels) | `MortarSpace.h` / `FormsInstaller.h` |
| Global multipliers | A `GlobalSpace` field in the form (`(mu * p).dx()`, `.dGlobal()`) | `GlobalSpace.h` / `Vocabulary.h` |
| Trace multipliers | `plan.restrictSupport()` + `bc::LagrangeDirichletBC` | `ConstraintPlan.h` / `LagrangeDirichletBC.h` |
| Auto-detecting compilation | `FormCompiler::compile()` | `FormCompiler.h` |
| Explicit mixed compilation | `FormCompiler::compileMixed()` | `FormCompiler.h` |
| Pre-compiled IR installation | `installMixedFormIR()` | `FormsInstaller.h` |
| Auxiliary model definition | `AuxiliaryModelBuilder("name")` | `AuxiliaryModelBuilder.h` |
| Auxiliary model deployment | `use(model).scope(...).bind(...)` | `AuxiliaryBindings.h` |
| Auxiliary inputs | `AuxiliaryInputRegistry` | `AuxiliaryInputRegistry.h` |
| Generated Nitsche route agreement | `requireGeneratedBoundaryNitscheTraceBindings()` | `FormsInstaller.h` |
| Moving-boundary normal relations and measurements | `MeshNormalKinematicDeclaration`, `declareMeshNormalKinematics()` | `MeshBoundaryKinematics.h` |
| Vertex field targets and comparisons | `requireVertexField()`, `compareVertexValues()` | `VertexFieldInitialization.h` |

### Expert/Manual Paths

| Task | API | Notes |
|------|-----|-------|
| Manual block decomposition | `BlockBilinearForm` / `BlockLinearForm` | For explicit per-block control |
| Complex-valued 2x2 lifting | `ComplexScalar` / `toRealBlock2x2` | Adapter for complex PDEs |
| Direct kernel registration | `FESystem::addCellKernel()` etc. | Framework internals / handwritten kernels |

## 1. The Compilation and Execution Pipeline

`FE/Forms` is a **front-end** that builds a runtime expression tree and compiles it into an `assembly::AssemblyKernel` that `FE/Systems` can register and assemble:

1. **User vocabulary / EDSL**: build a weak form with `forms::FormExpr` (and helpers from `Forms/Vocabulary.h`).
2. **Compile**: `forms::FormCompiler` lowers the expression to `forms::FormIR` (single-field) or `forms::MixedFormIR` (multi-field):
   - splits a sum into individual integral terms,
   - records the integration domain for each term (`dx`, `ds(marker)`, `dS`, `dI`),
   - infers `assembly::RequiredData` flags for each term and the combined form.
3. **Install**: `FormsInstaller` wraps compiled IR in kernels and registers them into `FESystem`:
   - `installFormulation()` handles residual physics (auto-detects single vs mixed)
   - `installMixedBilinear()` / `installMixedLinear()` handle operator installation
   - mixed installation first builds a `MixedKernelPlan`, then selects either
     one `MixedBlockKernelSet` or one `MonolithicCellKernel` for cell-domain work
4. **Assemble**: `FE/Assembly` iterates mesh entities, prepares `AssemblyContext` according to `RequiredData`, calls the kernel, and inserts local contributions into backend-neutral global views.

## 2. Residual Physics (the main path)

A formulation can register or reuse fields, keep its equations visible, apply
ordinary boundary conditions, and install through a builder. Here `system` is
an `FESystem` and `V` is a shared scalar function space:

```cpp
#include "FE/Forms/Vocabulary.h"
#include "FE/Systems/FieldBinding.h"
#include "FE/Systems/FormulationBuilder.h"
#include "FE/Systems/BoundaryConditionManager.h"
#include "FE/Constraints/ConstraintPlan.h"

namespace fe = svmp::FE;
using namespace fe::forms;

fe::systems::FieldBinding fields(system, "Example::registerOn");
const auto U = fields.unknownSymbols(
    {.name = "u", .space = V, .components = 1}, "state", "u", "v");
fe::systems::FormulationBuilder formulation(system, "equations", {});

const auto source = FormExpr::constant(1.0);
auto residual = (inner(grad(U.state), grad(U.test)) - source * U.test).dx();

fe::systems::BoundaryConditionManager bc_manager;
// Populate with ordinary BoundaryCondition objects as required.
fe::constraints::ConstraintPlan constraints(system);
constraints.essentials(bc_manager, residual, U.state, U.test, U.id,
                       formulation.operatorTag());
constraints.commit();

formulation.symbolicTangent();
(void)formulation.install({U.id}, residual);
```

`bc_manager.applyAll(system, residual, U.state, U.test, U.id, operator_tag)`
remains the ordinary direct BC route when no constraint plan is needed.
`StateField`, `TestField`, `FESystem::addField`, and direct
`installFormulation()` also remain available. These authoring services do not
choose physical BCs, material laws, observation membership, or compiler policy.

### Owned field symbols

`FieldBinding` returns `BoundUnknown {id, state, test}` or
`BoundPrescribed {id, value}`. Select the registration policy explicitly:

- `unknownSymbols(spec, role, state_name, test_name)` registers a missing
  unknown or reuses a compatible one. Existing source kind, scope (including
  interface marker), component count, and space are checked in that order.
- `addUnknownSymbols(spec, state_name, test_name)` uses strict
  `FESystem::addField` insertion, including duplicate-name rejection.
- `prescribedSymbol(spec, auto_register, role, name)` reuses compatible
  prescribed data; a missing field is added only when `auto_register` is true.

Unknown bundles require `spec.source_kind == FieldSourceKind::Unknown`;
prescribed bundles require `PrescribedData`. Reuse compatibility errors precede
checking the supplied spec's source kind. Set `spec.components` explicitly for
vector fields. Registration follows call order and preserves earlier successful
registrations if a later call fails.

Bundle symbols use the **supplied** space's complete signature even on reuse;
the compatibility check is less restrictive than that signature. Form nodes own
the field IDs, names, and copied signatures, so registry growth and destruction
of the binding, caller strings, or supplied space do not invalidate those nodes.
The binding itself borrows the system. Creating symbols does not sample
coefficient callbacks.

### Immediate residual contributions

`forms::ResidualContributions` (`FE/Forms/ResidualContributions.h`) binds actual
`FieldId` test rows to borrowed production expressions and optional observation
expressions. Each `append(id, term)` immediately updates production, then each
non-null observation in list order, using `sumOf`'s left fold. Nothing is queued,
reordered, expanded, rescaled, or inferred from a symbol name. The sink copies
row bindings and observation lists; their expression targets must outlive it.
Use scoped sinks to select exactly which terms enter each observation.

`INVALID_FIELD_ID`, duplicate row IDs, and repeated non-null destinations across
production and observations are rejected at construction before mutation. Null
observations are skipped. An unknown append ID throws without undoing earlier
appends. The appended term is snapshotted before assignments, so the argument
may itself reference a destination. An invalid term is not silently skipped:
`sumOf(empty, invalid)` stays invalid, while adding it to a valid expression
throws. Assignments already completed in production or earlier observations
remain visible if a later update fails.

Observations only collect expressions. Declare measured 0-forms with
`FormulationBuilder::functional()` / `wallFunctional()` (or the corresponding
`FunctionalForms.h` functions), and install exact residual-work groups through
`workChannel()`. Those declarations and operator records remain authoritative;
a sink does not create functionals, work channels, certification, or applicability
metadata. Preserve explicit work formulas and subtractions where splitting a
production tree would change grouping or cancellation.

### Builder creation, options, and row spans

The three-argument `FormulationBuilder(system, tag, base)` uses
`OperatorCreation::Ensure`: it creates the operator only when absent. The
four-argument constructor accepts `OperatorCreation::Add` for unconditional
strict insertion and its original duplicate diagnostics. Construction acts
immediately, so keep it at the intended operator-registration point.

When compiler-option evaluation can fail, construct with `{}` and assign
`options()` at the original later evaluation point. `symbolicTangent(bool)`
changes the current options. Each `install()` uses those options immediately;
`installGroup()` uses a fresh copy with its group overrides, ALE configuration,
and missing extra trial fields appended in order. Group changes do not mutate
base options or earlier installations.

Both methods accept initializer lists or `std::span<const FieldId>` (including
a vector-backed span). The call consumes the view synchronously without
retaining, sorting, or deduplicating it; the caller's storage must remain alive
through the call. The existing installer remains responsible for validation,
row bindings, metadata, and partial progress on failure.

### Typed essential operations

`constraints::ConstraintPlan` retains the phase order
`VertexValues < Essential < InactiveSupport < Aggregation < PointLaw`, with
declaration order within each phase. Its typed `essentials` overloads accept a
manager plus residual/state/test/field/optional operator tag, a strong-only
manager plus field, or `std::span<const forms::bc::StrongDirichlet>`.

Every overload first commits held declarations through Essential and then calls
`applyAll` or `installStrongDirichlet` immediately. The manager route includes
its normal validation, weak terms, descriptors, and strong constraints; the
span route retains per-declaration progress. A failure leaves prior commits,
phase advancement, and any completed underlying side effects in place. There
is no rollback or deferred BC callback. Later calls into a phase already passed
throw; uncommitted held declarations are dropped on plan destruction. The
callback overload and direct BC APIs remain supported.

For `H(div)` fields, strong prescribed normal flux is not lowered through
pointwise `StrongDirichlet`. Use `forms::bc::NormalTraceEssentialBC`, which
installs an `H(div)` trace constraint through the BC manager. Weak trace loads
and Robin relations are available through `forms::bc::TraceLoadBC` and
`forms::bc::TraceRobinBC`. Trace-consistent weak imposition uses
`forms::bc::TraceNitscheBC` or `forms::bc::InterfaceTraceNitscheBC`. One-sided
boundary behavior uses `forms::bc::TraceInequalityBC`. FE intentionally does
not encode a special "passive outflow" primitive; that is represented by
leaving the normal trace unconstrained or by adding a generic weak trace term.

Example:

```cpp
auto q = StateField(q_field, *Vhdiv, "q");
auto w = TestField(q_field, *Vhdiv, "w");
auto residual = inner(k_inv * q, w).dx();

systems::BoundaryConditionManager bc_manager;
bc_manager.add(forms::bc::makeNormalTraceEssentialBC(4, Constant(1.0)));
bc_manager.add(forms::bc::makeTraceRobinBC(7, Constant(alpha), Constant(rhs)));
bc_manager.applyAll(system, residual, q, w, q_field);
```

For an exterior face clipped by an unfitted active domain, pass the typed
selector instead of a bare physical marker:

```cpp
const auto wet_wall =
    forms::ExteriorBoundaryMeasure::generatedActiveSubset(
        physical_wall_marker, generated_wet_wall_marker);
bc_manager.add(forms::bc::makeTraceRobinBC(
    wet_wall, Constant(alpha), Constant(rhs)));
```

The same selector overload is available for trace loads, trace inequalities,
generic trace Nitsche forms, scalar Nitsche wrappers, and the generic
Neumann/Robin helper families. The compiled term retains both markers so
system setup can verify generated-domain provenance before assembly.
An integer marker alone remains a legacy, unqualified boundary route; it is
rejected if the current cut context exposes a generated active subset for that
physical marker. A formulation that genuinely needs the complete background
face must say so with
`ExteriorBoundaryMeasure::fullPhysical(physical_marker)`.
`CouplingFormBuilder::integrate(...)` has corresponding selector overloads;
they require the selector's physical marker to match the declared coupling
boundary region before constructing the form.
Selection alone does not certify a cut-stable numerical penalty. Generated
Nitsche production routes that claim that property must also use the
form-bound trace-certificate workflow and its aggregation prerequisite.

Interface trace conditions use the same manager surface:

```cpp
bc_manager.add(forms::bc::makeInterfaceTraceLoadBC(marker, Constant(0.0)));
bc_manager.add(forms::bc::makeInterfaceTraceJumpPenaltyBC(marker,
                                                          Constant(beta),
                                                          Constant(0.0)));
bc_manager.applyAll(system, residual, q, w, q_field);
```

The canonical public trace-condition names are:

- `NormalTraceEssentialBC`
- `TraceLoadBC`
- `TraceRobinBC`
- `TraceNitscheBC`
- `TraceInequalityBC`
- `InterfaceTraceLoadBC`
- `InterfaceTraceRobinBC`
- `InterfaceTraceJumpPenaltyBC`
- `InterfaceTraceNitscheBC`

## 2.1 Advanced H(div) Paths

Three advanced `H(div)` capabilities are registered on the expert/system side:

- periodic and MPC trace relations through `Constraints/ConstraintTools.h`
- mortar and hybridized facet unknowns through `spaces::MortarSpace`
- mixed-dimensional codimension-1 unknowns through `FESystem::addInterfaceField()`

The periodic and MPC relations are constraints. The mortar and interface
fields are registered on the `FESystem` and coupled to the volume fields by
`.dI(marker)` terms that `installFormulation` installs (decision D54, LM4),
within the limits listed in `VOCABULARY.md`. The current public usage examples
are collected in `Docs/HDIV_ADVANCED_USAGE_GUIDE.md`.

For a coupled vector/scalar system, keep every measured integral in one test
row. Here `V` is vector-valued and `Q` scalar-valued; this is a separate example
using the same namespace setup and headers plus `FE/Forms/ResidualContributions.h`:

```cpp
fe::systems::FieldBinding fields(system, "CoupledExample::registerOn");
const auto U = fields.unknownSymbols(
    {.name = "u", .space = V, .components = V->value_dimension()},
    "vector", "u", "v");
const auto P = fields.unknownSymbols(
    {.name = "p", .space = Q, .components = 1}, "scalar", "p", "q");
fe::systems::FormulationBuilder formulation(system, "equations", {});
FormExpr u_row, p_row, observed_u_terms;
ResidualContributions rows({
    {U.id, u_row, {&observed_u_terms}},
    {P.id, p_row},
});
rows.append(U.id, inner(grad(U.state), grad(U.test)).dx());
rows.append(U.id, (-P.state * div(U.test)).dx());
rows.append(P.id, (div(U.state) * P.test).dx());

const auto residual = sumOf(u_row, p_row);
formulation.symbolicTangent();
(void)formulation.install({U.id, P.id}, residual);
```

`observed_u_terms` receives exactly the two appended vector-row terms. The
scalar row stays separate until the final measured sum. Supply row IDs from
registered bundles rather than assuming contiguous field numbering.

`installFormulation()` automatically:
- Decomposes the expression by test function
- Detects which fields each test row depends on
- Builds an explicit `MixedKernelPlan`
- Installs Jacobian blocks and residual kernels
- Selects either:
  - one exact `MixedBlockKernelSet`, or
  - one `MonolithicCellKernel` for mixed cell assembly

Optional colocation/text-layout optimization now lives inside
`MixedBlockKernelSet`; there is no separate helper kernel type in the
production semantic path.

### Mixed test-row separation

The canonical residual installer identifies bound test rows by `FieldId`, so
independent fields can share a function space and symbol name. Keep each
integral's integrand in one test row. Top-level sums and differences of such
integrals are separated without expanding contractions or nonlinear factors.
For example, an interface penalty on independently bound phase fields is
currently written as:

```cpp
auto penalty_terms =
    (penalty * inner(u_negative, v_negative)).dI(marker)
  + (-penalty * inner(u_negative, v_positive)).dI(marker)
  + (-penalty * inner(u_positive, v_negative)).dI(marker)
  + (penalty * inner(u_positive, v_positive)).dI(marker);
```

The compact `penalty * inner(u_negative - u_positive,
v_negative - v_positive)` under one `dI` contains two test rows and is rejected.
These differences concern independently bound fields; they are not the DG
`jump()` of one field. `CanonicalWorkflow` covers the same-space weighted
four-field interface, its twelve installed kernel couplings with reversed caller
field order, and rejection of its compact contraction before operator or
formulation-record installation.

The stored `FormulationRecord::block_couplings` currently pairs the test rows in
expression traversal order with the caller's field order. These recorded row
IDs can therefore differ from the installed kernel bindings when those orders
differ. Inspect the operator terms or returned `MixedKernelPlan` for the installed
bindings; the limitation test checks both. The records also feed coupling-graph
analysis and trace-multiplier checks, so correct kernel bindings do not establish
correctness of those metadata consumers. The test does not establish numerical
assembly equivalence for a compact expression, which is still rejected.

Automatic linear expansion remains a separate compiler task. Besides retaining
all test rows and nonlinear coefficient dependencies, adoption must preserve
the production trees' scalar placement, signs, term order, measures, and kernel
groups. Algebraic equivalence alone does not preserve the manifest or JIT key.
The expert `FormCompiler::compileMixed` path separately identifies symbols by
space signature and name; use distinct names there. It does not provide the
canonical residual installer's `FieldId` disambiguation.

### Generated Nitsche route agreement

`systems::requireGeneratedBoundaryNitscheTraceBindings(expected_routes,
bindings, mismatch_message)` compares spans of `forms::BoundaryDomain` and
`forms::bc::GeneratedBoundaryNitscheTraceFormBinding`. Counts must match, and
each expected domain must have generated-active-subset intent with the same
physical and generated markers as the corresponding binding. Order and
multiplicity matter: duplicate routes are retained; two empty spans agree.
A mismatch throws `std::logic_error` with the supplied message.

This check reads the supplied values without registering markers, changing
bindings, or constructing replacement forms. The caller owns both spans for
the duration of the call. Register required domains separately through
`FormulationBuilder::registered()` at the intended declaration point. Route
agreement does not authenticate a form's origin or certify its penalty:
`certifyGeneratedBoundaryNitscheTraces()` and installation still require the
original form anchors and prepare the live trace policies. Keep the original
production bindings through that workflow; a matching marker pair alone is
insufficient.

### Moving-boundary normal declarations

`systems::MeshNormalKinematicDeclaration` pairs a
`MeshNormalBoundaryConstraintDeclaration` with its
`MeshNormalKinematicBinding` (consumer source strings and optional prescribed
tangential measurement velocity). Pass an ordered vector to
`declareMeshNormalKinematics(system, operator_tag, declarations)`. For each
item the service declares the constraint, binds its consumer, then registers
its operator-stage measurement before advancing. Target expressions retain
their existing form nodes.

The operation does not roll back earlier items if a later item fails. The
callback overload resolves each binding immediately before declaring its
item. When caller-side value evaluation can throw and its timing matters,
evaluate and submit one paired item at a time at the required registration
point; eagerly evaluating the whole vector changes failure ordering. The
service consumes values during the call and stores the resulting declarations
and measurements in the system. It does not select physical boundary policies.

### Vertex initialization and comparison

`systems::requireVertexField()` resolves a named field and then its vertex DOF
map without mutation. Its `VertexFieldTarget` borrows that map: use it while
system numbering remains current and resolve it again after numbering changes.
`VertexValueSamples` likewise borrows its ordered value span.
`compareVertexValues()` samples each local, in-range input once, including
duplicates, and skips nonlocal or out-of-range inputs and their extrema. It
returns local counts, extrema, and maximum absolute error without enforcing
values or performing a collective reduction; empty extrema remain infinite.

For side-restricted vertex writes and admissibility checks, see
[Level-set vertex support and initialization](../Docs/LevelSet.md#vertex-support-admissibility-and-initialization).
The existing `initializeFieldFromMeshVertexField()` projects the same-name mesh
vertex field to all field coefficients. It has different coverage from the
vertex-only side initializer; choose explicitly according to the required DOFs.

## 3. Mixed Operators (bilinear / linear)

For assembling standalone operators (mass matrices, stiffness matrices) rather than residual physics:

```cpp
auto a = (inner(grad(u), grad(v))).dx() + (p * div(v)).dx() + (div(u) * q).dx();
installMixedBilinear(system, "stiffness", test_fields, trial_fields, a);

auto L = (f * v).dx() + (g * q).dx();
installMixedLinear(system, "load", test_fields, L);
```

When JIT cell fusion is enabled and the mixed cell blocks are compatible,
`installMixedBilinear()` / `installMixedFormIR()` register one
`MonolithicCellKernel` for the cell domain. Otherwise they register one
`MixedBlockKernelSet` for exact per-block cell work and keep boundary /
interior-face / interface-face terms as exact per-block kernels.

## 4. Modifiers

These modify the residual or operator expression, not the installation path:

- **Boundary terms**: `(h * u * v).ds(5)` — boundary integrals on marker 5
- **DG terms**: `(jump(u) * avg(grad(v))).dS()` — interior-face integrals
- **Interface terms**: `(u * v).dI(marker)` — interface-face integrals
- **Transient terms**: `(dt(u, 1) * v).dx()` — time derivative (requires `TransientSystem`)
- **Constitutive models**: `inner(constitutive(model, grad(u)), grad(v)).dx()`

## 5. Expert Paths

### Manual block decomposition

For explicit per-block control (heterogeneous compilation options, pre-split
expressions from external tooling). Compile each block individually and
install via `installMixedFormIR()`:

```cpp
FormCompiler compiler;
MixedFormIR mir(2, 2);
mir.setKind(FormKind::Bilinear);
mir.setBlock(0, 0, compiler.compileBilinear(inner(grad(u), grad(v)).dx()));
mir.setBlock(0, 1, compiler.compileBilinear((p * div(v)).dx()));
// PP block left empty (zero block)

installMixedFormIR(system, "op", test_fields, trial_fields, mir);
```

### Complex-valued PDEs

Uses `ComplexBilinearForm` and `toRealBlock2x2()` from `Forms/Complex.h`.
The result is a `BlockBilinearForm` which must be compiled per-block and
installed via `installMixedFormIR()`:

```cpp
ComplexBilinearForm a;
a.re = inner(grad(u), grad(v)).dx();  // real part
a.im = (k * u * v).dx();             // imaginary part
auto blocks = toRealBlock2x2(a);      // → 2×2 BlockBilinearForm

// Compile each block and build a MixedFormIR
FormCompiler compiler;
auto compiled = compiler.compileBilinear(blocks);  // vector<vector<optional<FormIR>>>
MixedFormIR mir(2, 2);
mir.setKind(FormKind::Bilinear);
for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
        if (compiled[i][j]) mir.setBlock(i, j, std::move(*compiled[i][j]));

installMixedFormIR(system, "op", test_fields, trial_fields, mir);
```

### Direct kernel registration

For handwritten `AssemblyKernel` implementations:

```cpp
system.addCellKernel("op", field, my_custom_kernel);
```

## 6. How "Forms Vocabulary" Maps to AssemblyContext Data

- `TrialFunction` / `TestFunction` → basis values at quadrature points
- `grad(·)` → physical gradients
- `dt(·, k)` → k-th time derivative (symbolic; needs transient context)
- `x()` → physical coordinate
- `n` → face normal
- `h()` → cell diameter
- `.dx()` → cell loop
- `.ds(marker)` → boundary-face loop
- `.dS()` → interior-face loop (DG 4-block)
- `.dI(marker)` → interface-face loop

## 6.1 Tracing The Active Kernel Path

Use `SVMP_FE_KERNEL_TRACE` to inspect mixed-kernel selection and execution:

```bash
SVMP_FE_KERNEL_TRACE=selection
SVMP_FE_KERNEL_TRACE=specialization
SVMP_FE_KERNEL_TRACE=assembly,capabilities
SVMP_FE_KERNEL_TRACE=all
```

Legacy trace env vars are still accepted:

- `SVMP_OOP_SOLVER_TRACE`
- `SVMP_JIT_TRACE_SPECIALIZATION`

## 7. Migrating from the Old Pattern

If your formulation module uses the previous pattern:

```cpp
// Old pattern:
auto u = FormExpr::stateField(u_id, *space, "u");        // ← verbose
auto v = FormExpr::testFunction(*space, "v");             // ← no field binding
...
auto strong = bc_manager.getStrongConstraints(u_id);      // ← manual 3-step
bc_manager.apply(system, residual, u, v, u_id);
installStrongDirichlet(system, strong);
```

Migrate to:

```cpp
// New pattern:
auto u = StateField(u_id, *space, "u");                   // ← clean helper
auto v = TestField(u_id, *space, "v");                    // ← field-bound
...
bc_manager.applyAll(system, residual, u, v, u_id);        // ← one call
```

Key changes:
- `FormExpr::stateField(...)` → `StateField(...)` (Vocabulary.h helper)
- `FormExpr::testFunction(...)` → `TestField(field_id, ...)` (field-bound, required for same-space multi-field)
- 3-step BC pattern → `applyAll()` (validate + apply + installStrongDirichlet in one call)
- `#include "Forms/Forms.h"` → `#include "Forms/Vocabulary.h"` (smaller surface)

The old APIs continue to work — this is a style migration, not a breaking change.

## References

- Alnæs et al. — "Unified form language." *ACM Trans. Math. Softw.* (2014).
- Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the FEM* (2012).
- Rathgeber et al. — "Firedrake." *ACM Trans. Math. Softw.* (2016).
