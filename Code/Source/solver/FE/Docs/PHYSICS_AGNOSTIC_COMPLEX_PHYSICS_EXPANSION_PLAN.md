# Physics-Agnostic FE Expansion Plan for Complex Future Physics

## Purpose

This document outlines the FE-library work needed to better support complex
future physics modules such as compressible Navier-Stokes, multispecies
transport, reactive flow, MHD, multiphase models, advanced transport, and other
coupled nonlinear systems.

The plan is intentionally physics-agnostic. The FE layer should provide
well-defined mathematical and computational infrastructure. Physics modules
should provide equations of state, flux laws, stabilization formulas, boundary
state construction, and validation cases.

The near-term compressible Navier-Stokes recommendation remains:

```text
rho : scalar field
m   : physical vector field, dimension = spatial dimension
E   : scalar field
```

That mixed-field formulation fits the current FE library and can be implemented
without first adding arbitrary n-component state-vector fields. The expansion
work below is still valuable because it makes that implementation cleaner and
creates a reusable foundation for broader systems of conservation laws.

## Goals

- Preserve the FE library as a physics-agnostic substrate.
- Reuse and extend existing FE infrastructure before adding new public
  subsystems.
- Make complex coupled systems easier to express, assemble, analyze, and solve.
- Keep physical vector calculus type-safe.
- Support both mixed-field formulations and eventual algebraic state-vector
  formulations.
- Add reusable extensions and metadata hooks for conservative fluxes,
  exterior-state boundary conditions, admissibility checks, stabilization,
  constitutive metadata, wave-speed metadata, and solver block semantics.
- Avoid compressible-flow-specific names in FE-core APIs.
- Keep existing physics modules working with a compatibility path.

## Non-Goals

- Do not implement a turnkey compressible Navier-Stokes solver in the FE layer.
- Do not add EOS, Riemann solvers, shock sensors, or characteristic boundary
  conditions as FE-core physics. Those belong in physics modules or reusable
  physics-side libraries.
- Do not replace the existing mixed-field residual workflow.
- Do not make every FE formulation use a conservation-law abstraction.
- Do not simply relax all vector field limits from `1..3` to `n`; that would
  blur physical vector semantics and make invalid expressions legal.

## Current FE Capabilities Relevant to This Plan

The current FE library already provides many required pieces:

- Multi-field residual installation and block Jacobian construction through
  `Systems/FormsInstaller.h`.
- Field registration metadata through `Systems/FieldRegistry.h`, including
  field names, FE spaces, component counts, scopes, interface markers, and
  time-dependence.
- FE-backed auxiliary quantity metadata through `Systems/FEQuantityDefinition.h`
  and `Systems/FEQuantityRegistry.h`, including scalar/vector/tensor shape
  descriptors and capability flags.
- Symbolic `Forms` algebra for `dt`, `grad`, `div`, tensor operations,
  cell/boundary/interior-face/interface integrals, and AD/symbolic tangents.
- Dynamic value payloads in the `Forms` value layer for scalar, vector, matrix,
  and tensor-like data. This means arbitrary algebraic payloads are not foreign
  to the expression layer, even though field evaluation paths still mostly
  assume scalar or physical vector fields.
- Existing nonlinear form kernels for cell, boundary face, interior face, and
  interface face assembly.
- `L2` and DG-oriented infrastructure including traces, jumps, averages, and
  interior face assembly.
- Boundary-condition infrastructure, Nitsche-style weak conditions, coupled
  auxiliary inputs, and boundary functional hooks.
- Auxiliary-input infrastructure for sampled state fields, coupled fields,
  boundary traces, coupled boundary traces, reductions, direct user data, and
  formulation callbacks.
- Time-integration lowering for symbolic `dt(...)`.
- Backend capability descriptors for monolithic and generic block solves.
- Analysis contribution descriptors, balance roles, preservation summaries,
  temporal stability summaries, invariant-domain summaries, flux-balance
  summaries, solver-compatibility summaries, and analyzer/reporting machinery.
- Constitutive state metadata through `Core/StateVariableMetadata.h`,
  constitutive state layouts through `Constitutive/StateLayout.h`, and material
  state declarations through `AssemblyKernel` metadata.
- A mature incompressible Navier-Stokes VMS module that demonstrates many of
  the required mixed nonlinear FE mechanisms.

The main missing pieces are not basic weak-form expressiveness. They are
semantic, ergonomic, and robustness infrastructure for large systems of
conservation laws and highly nonlinear multiphysics. The implementation should
therefore be mostly additive metadata, validation, adapters, and analysis
extensions around the existing FE services, not a replacement architecture.

## Architectural Principles

### 1. Separate Physical Vectors From Algebraic State Vectors

A physical vector field has geometric meaning. Its dimension is tied to the
spatial dimension, and operations like `div(u)`, `curl(u)`, `u dot n`,
Piola transforms, and ALE mesh velocity are meaningful.

An algebraic state vector is an ordered collection of components. Its dimension
may be 4, 5, 8, 20, or more. It is useful for conserved states, species
vectors, reaction networks, or multiphysics packs. It should support component
access, algebraic operations, componentwise `dt`, and gradients with shape
`n_components x spatial_dim`. It should not automatically support geometric
operators like `curl(U)` or `U dot n`.

### 2. Mixed Fields Are First-Class

Many physically meaningful systems are clearer as mixed fields:

```text
rho : scalar
m   : physical vector
E   : scalar
Y   : algebraic vector of species
```

The FE library should not force such systems into one packed field. Instead, it
should provide optional grouping and layout metadata so mixed fields can be
treated as one logical state when useful.

### 3. FE Provides Plumbing, Physics Provides Laws

The FE layer should provide:

- quadrature loops
- cell and face data access
- trace extraction
- normal and geometry data
- residual and Jacobian insertion
- state layout and metadata
- constraint and admissibility hooks
- diagnostics and analysis descriptors

Physics modules should provide:

- physical fluxes
- numerical fluxes
- EOS and transport models
- stabilization formulas
- boundary exterior states
- wave speed estimates
- admissibility definitions
- verification cases

### 4. Every New Public Concept Needs Analysis Metadata

Complex physics becomes hard to debug if the analysis subsystem cannot describe
what was installed. New FE concepts should emit descriptors for:

- field shapes and logical groups
- conservation variables
- flux balance surfaces
- admissibility constraints
- stabilization terms
- constitutive domains and output scales
- solver block structure
- wave-speed and CFL metadata

### 5. Keep Expert Escape Hatches

The `AssemblyKernel` path remains important. If the `Forms` EDSL cannot express
a specific flux or limiter cleanly, a custom kernel should still be able to
participate in the same field metadata, admissibility, analysis, and solver
infrastructure.

### 6. Extend Before Adding Parallel Infrastructure

Every workstream below should start by identifying the closest existing FE
owner. A new registry, analyzer, file, or service should be added only when the
existing owner cannot reasonably express the concept.

Default extension points:

- field identity and component metadata: `Systems/FieldRegistry.h`
- FE-backed auxiliary quantity shapes: `Systems/FEQuantityDefinition.h`
- system-owned services and registries: `Systems/FESystem.h`
- boundary, coupled, sampled, and callback data: `Auxiliary/`
- symbolic expression vocabulary and shape checking: `Forms/`
- custom residual/Jacobian paths: `Assembly/AssemblyKernel.h` and
  `Assembly/AssemblyContext.h`
- constitutive state and material metadata: `Core/StateVariableMetadata.h` and
  `Constitutive/`
- diagnostics and static summaries: `Analysis/`
- backend compatibility and block solver capabilities: `Backends/`

## Reuse-First Target Integration Map

Recommended placement by FE subsystem. Items marked "extend" should reuse
existing infrastructure. Items marked "new only if needed" should not be added
until a concrete implementation proves that extension is insufficient.

```text
FE/
├── Core/
│   ├── Types.h                      # extend FieldType or add shared shape tags
│   ├── StateVariableMetadata.h      # extend for generic state-domain metadata
│   └── FieldShape.h                 # new only if shape metadata cannot live cleanly
│                                      beside existing Core/Systems definitions
├── Spaces/
│   ├── FunctionSpace.h              # extend shape introspection, preserve API
│   ├── ProductSpace.h               # preserve physical vector/multi-component use
│   └── AlgebraicVectorSpace.h       # optional future arbitrary-component space
├── Forms/
│   ├── Value.h                      # reuse dynamic value payload support
│   ├── FormKernels.*                # extend shape validation and trace plumbing
│   ├── StateVocabulary.h            # optional helpers over FieldRegistry/StateGroups
│   ├── FluxForms.h                  # optional conservation-law form helpers
│   └── StabilizationForms.h         # optional terminals over Auxiliary/FE quantities
├── Assembly/
│   ├── AssemblyContext.h            # extend scalar/Vector3D access where needed
│   ├── AssemblyKernel.h             # extend metadata hooks and custom-kernel access
│   ├── StateTraceView.h             # new lightweight adapter only if useful
│   └── FluxKernel.h                 # new only if repeated custom kernels duplicate code
├── Constitutive/
│   ├── StateLayout.h                # reuse existing named state layout support
│   └── Existing model interfaces    # extend for named outputs/domains/scales
├── Auxiliary/
│   ├── AuxiliaryInputRegistry.h     # reuse for exterior-state dependencies
│   ├── AuxiliaryBindings.h          # reuse for form-visible auxiliary values
│   └── AuxiliaryStateManager        # reuse for sensor/stabilization storage
├── Systems/
│   ├── FieldRegistry.h              # extend with shape metadata
│   ├── FEQuantityRegistry.h         # reuse for derived/sensor quantities
│   ├── FESystem.h                   # own lightweight state groups and bindings
│   └── SolverBlockMetadata          # extend existing backend/block metadata
├── Backends/
│   └── BackendCapability.*          # extend generic block compatibility reporting
└── Analysis/
    ├── AnalysisSummaryTypes.h       # extend existing summaries first
    ├── ContributionDescriptor.h     # reuse balance/stabilization descriptors
    ├── Existing analyzers           # extend before adding new analyzer classes
    └── README.md                    # document metadata consumption
```

The exact file split can be adjusted during implementation. The important part
is to keep concepts separated while avoiding duplicate infrastructure. In
particular, do not add a second state-layout system, a second auxiliary data
registry, a second analysis pipeline, or a second backend capability model.

## Workstream 1: Field Shape and Value Semantics

### Problem

The current FE stack effectively treats vector fields as physical vectors with
dimension `1..3`. That is correct for velocity, displacement, momentum, flux,
and mesh velocity. It is not sufficient for a generic conserved state such as:

```text
U = [rho, rho*u_x, rho*u_y, rho*u_z, E]
```

Simply increasing the vector limit would be risky because geometric operations
would silently accept non-geometric vectors.

### Desired Capability

The FE layer should distinguish:

- scalar values
- physical vector values
- algebraic vector values
- tensor values
- grouped mixed-field states

### Proposed Types

First reconcile the existing shape concepts:

- `FieldSpec::components` in `Systems/FieldRegistry.h`
- `FEQuantityShape` in `Systems/FEQuantityDefinition.h`
- `FieldType` in `Core/Types.h`
- `SpaceSignature` in the `Forms` expression layer
- `FunctionSpace::value_dimension()` in `Spaces/FunctionSpace.h`

If these cannot be cleanly unified in place, add a small shared field-shape
descriptor and use it from those existing structures. The descriptor should be
the common vocabulary for field registration, FE quantities, forms, assembly,
and analysis.

Example:

```cpp
enum class FieldValueKind {
    Scalar,
    PhysicalVector,
    AlgebraicVector,
    Tensor
};

enum class ComponentSemantics {
    Unspecified,
    Spatial,
    ConservedState,
    PrimitiveState,
    Species,
    Auxiliary
};

struct FieldShape {
    FieldValueKind kind{FieldValueKind::Scalar};
    int rows{1};
    int cols{1};
    ComponentSemantics semantics{ComponentSemantics::Unspecified};

    bool isScalar() const noexcept;
    bool isPhysicalVector(int spatial_dim) const noexcept;
    bool isAlgebraicVector() const noexcept;
    int numComponents() const noexcept;
};
```

Extend existing metadata. Do not introduce a disconnected shape universe:

- `FE::systems::FieldSpec`
- `FE::systems::FEQuantityShape`
- `FE::forms::FormExprNode::SpaceSignature`
- `FE::spaces::FunctionSpace`
- field descriptors used by `Dofs`, `Systems`, and `Analysis`

### Public API Direction

Backward-compatible field registration:

```cpp
FieldSpec velocity;
velocity.name = "u";
velocity.space = velocity_space;
velocity.components = dim;
velocity.shape = FieldShape::physicalVector(dim);
```

Future algebraic state vector registration:

```cpp
FieldSpec U;
U.name = "U";
U.space = state_space;
U.components = 5;
U.shape = FieldShape::algebraicVector(5, ComponentSemantics::ConservedState);
```

Mixed-field logical grouping:

```cpp
StateGroupSpec conserved;
conserved.name = "compressible_state";
conserved.fields = {rho_id, m_id, E_id};
conserved.component_names = {"rho", "m_x", "m_y", "m_z", "E"};
conserved.semantics = ComponentSemantics::ConservedState;
```

### Required Changes

#### Core

- Prefer extending `Core/Types.h` and the existing `FEQuantityShape` concepts
  before adding a standalone `FieldShape.h`.
- If a standalone `FieldShape.h` is added, keep it small and make
  `FEQuantityShape` and `FieldSpec` consume it instead of maintaining parallel
  fields.
- Add helper constructors:
  - `FieldShape::scalar()`
  - `FieldShape::physicalVector(int dim)`
  - `FieldShape::algebraicVector(int n, ComponentSemantics semantics)`
  - `FieldShape::tensor(int rows, int cols)`
- Add validation utilities:
  - shape rank
  - component count
  - compatibility with spatial dimension
  - compatibility with `FunctionSpace::value_dimension()`

#### Spaces

- Preserve existing physical vector space behavior.
- Preserve `FunctionSpace::value_dimension()` as the legacy component-count
  API, and add shape introspection beside it.
- Add an optional `AlgebraicVectorSpace` only when true n-component field
  support is ready. This should not be the first step.
- Keep `ProductSpace` and `MixedSpace` semantics intact. Shape metadata should
  describe the space; it should not force existing spaces into a new hierarchy.

#### Systems

- Extend `FieldSpec` to include `FieldShape`.
- Keep `FieldSpec::components` for compatibility and derive it from shape where
  possible.
- Reuse `FEQuantityRegistry` shape metadata for derived FE quantities instead
  of adding a second quantity-shape table.
- Default missing shape from existing fields:
  - components = 1 -> scalar
  - components = spatial dimension and space says vector -> physical vector
  - otherwise require explicit shape
- Add setup-time validation so ambiguous fields produce clear diagnostics.

#### Forms

- Extend `SpaceSignature` with `FieldShape`.
- Reuse `Forms::Value` dynamic vector/matrix/tensor payloads where possible.
  The missing work is mostly expression typing and field evaluation, not basic
  value storage.
- Add shape checking rules:
  - `div(PhysicalVector)` allowed.
  - `curl(PhysicalVector)` allowed where dimension supports it.
  - `grad(Scalar)` gives physical vector or row vector.
  - `grad(PhysicalVector)` gives spatial tensor.
  - `grad(AlgebraicVector(n))` gives `n x dim` tensor-like object.
  - `div(AlgebraicVector)` rejected.
  - `normal dot AlgebraicVector` rejected unless an explicit projection or
    flux tensor is used.
- Make diagnostics identify the field name and shape.

#### Assembly

- Keep current `Vector3D` fast path for physical vectors.
- Add storage/evaluation path for algebraic vectors only when needed:
  - dynamic component array per quadrature point
  - `n_components x dim` gradients
  - history values for `dt`
- Avoid rewriting physical vector code paths prematurely.
- For the first mixed-field compressible implementation, prefer scalar and
  physical-vector accessors already exposed through `AssemblyContext`.

### Tests

- Unit tests for `FieldShape` validation.
- Field registration tests for scalar, physical vector, algebraic vector, and
  invalid combinations.
- Forms shape-checking tests:
  - `div(u_physical)` succeeds.
  - `div(U_algebraic)` fails.
  - `component(U_algebraic, i)` succeeds.
  - `grad(U_algebraic)` has the expected shape.
- Backward-compatibility tests for existing Poisson and incompressible
  Navier-Stokes modules.

### Definition of Done

- Existing modules continue to compile and run.
- New shape metadata is visible in `FESystem` introspection and analysis.
- Invalid physical/algebraic vector operations fail at setup or compile time
  with clear messages.

## Workstream 2: Logical State Groups for Mixed Fields

### Problem

Mixed fields are currently fully supported for residual assembly, but a physics
module has to manually remember that several fields form one logical state. This
becomes tedious for:

- conservation-law fluxes
- limiters
- positivity checks
- boundary exterior states
- solver block hints
- output and diagnostics

### Desired Capability

Allow multiple FE fields to be grouped into a logical state without changing
their underlying FE spaces.

Compressible Navier-Stokes example:

```text
compressible_conserved_state:
  rho : scalar
  m   : physical vector
  E   : scalar
```

This group can be flattened as `[rho, m_0, ..., m_dim-1, E]` when a numerical
flux, limiter, or diagnostic needs a packed state.

### Proposed Types

```cpp
struct StateComponentSpec {
    FieldId field{INVALID_FIELD_ID};
    int component{-1};              // -1 means whole scalar field or whole field
    std::string name;
    FieldShape shape;
};

struct StateGroupSpec {
    std::string name;
    std::vector<FieldId> fields;
    std::vector<std::string> component_names;
    ComponentSemantics semantics{ComponentSemantics::Unspecified};
    bool conservative{false};
};

class StateGroupRegistry {
public:
    StateGroupId addStateGroup(StateGroupSpec spec);
    const StateGroupSpec& stateGroup(StateGroupId id) const;
    StateGroupId findStateGroup(std::string_view name) const;
};
```

This registry should be a thin system-level grouping layer over already
registered fields. It should not replace `MixedSpace`, `ProductSpace`,
`FieldRegistry`, or the form installer. `MixedSpace` describes FE-space
composition; a state group describes logical physics/analysis layout across
already registered fields.

### Required Changes

#### Systems

- Add a lightweight `StateGroupRegistry` or equivalent state-group table owned
  by `FESystem` beside the existing `FieldRegistry`.
- Add public APIs:
  - `FESystem::addStateGroup(...)`
  - `FESystem::findStateGroup(...)`
  - `FESystem::stateGroups()`
- Validate that all fields in a group are registered.
- Validate component names and flattened component counts.
- Allow group metadata to be passed to analysis and assembly.
- Keep field ownership, DOF ownership, and residual installation in the
  existing `FieldRegistry`, `Dofs`, and `FormsInstaller` paths.

#### Forms

- Add optional vocabulary helpers for grouped states:
  - `StateGroup(system, "name")` or explicit helpers that return a vector of
    `FormExpr` components.
  - `component(group, "rho")`
  - `flatten(group)`
- Keep the current `StateField` and `TestField` workflow as the default for
  residual authoring.

#### Assembly

- Add lightweight `StateGroupValueView` for custom kernels:
  - current values
  - gradients
  - previous values
  - minus/plus traces
- Do not require all grouped fields to share the same finite element space.
- Implement this view as an adapter over existing scalar/vector field accessors
  first. Do not add a packed storage layer unless arbitrary-component fields
  become a concrete requirement.

#### Analysis

- Extend the existing analysis summary/reporting pipeline with logical-state
  descriptors:
  - name
  - field IDs
  - flattened component names
  - conservative vs primitive vs auxiliary semantics
  - expected admissibility constraints if registered

### Tests

- Register a group containing scalar, physical vector, scalar fields.
- Verify flattened component order in 1D, 2D, and 3D.
- Verify groups work when component fields use different spaces.
- Verify field deletion or invalid IDs are rejected.
- Verify analysis output includes group metadata.

### Definition of Done

- A compressible physics module can register `rho`, `m`, `E` as separate fields
  and one logical conserved state.
- Custom kernels and future flux infrastructure can request grouped state
  values without knowing the details of DOF storage.

## Workstream 3: Generic Conservation-Law Flux Infrastructure

### Problem

Conservation laws are often most naturally expressed through cell and face
fluxes:

```text
dt(U) + div(F(U, grad(U))) = S(U)
```

For DG or FV-like formulations, boundary and interior-face terms depend on a
numerical flux:

```text
Fhat(U_minus, U_plus, n)
```

The FE library has interior-face assembly, DG vocabulary, face kernels, balance
descriptors, and analysis summaries for flux-like contributions. It does not
yet provide a compact, reusable conservation-law authoring layer that ties those
pieces together.

### Desired Capability

Provide FE plumbing for:

- state trace extraction
- physical flux evaluation
- numerical flux evaluation
- boundary flux evaluation
- residual insertion
- Jacobian differentiation
- diagnostics for conservation balance

The FE layer should not provide a specific Riemann solver.

### Proposed Interfaces

#### Physics-Supplied Flux Concept

```cpp
class NumericalFluxModel {
public:
    virtual ~NumericalFluxModel() = default;

    virtual std::string name() const = 0;
    virtual int numEquations() const = 0;

    virtual void evaluate(
        const StateTraceView& minus,
        const StateTraceView& plus,
        const NormalView& normal,
        FluxOutputView& out) const = 0;
};
```

#### Forms-Friendly Flux Call

For fluxes expressible in the EDSL:

```cpp
auto U = conservedState({rho, m, E});
auto F = physicalFlux(U);                 // physics-side helper returns FormExpr tensor
auto cell = inner(F, grad(W)).dx();        // or helper for conservative residual
auto Fhat = numericalFlux(model, U.minus(), U.plus(), normal());
auto face = dot(Fhat, jump(W)).dS();
```

The exact syntax should be refined once `FieldShape` and `StateGroup` are in
place. The key is that the FE layer can support either symbolic expressions or
custom flux kernels.

### Required Changes

#### Assembly

- Prefer a lightweight `StateTraceView` adapter over existing face/cell
  assembly data:
  - field/group ID
  - side: cell, boundary interior, face minus, face plus, exterior
  - values
  - gradients if requested
  - previous values if requested
- Prefer a lightweight `FluxOutputView` adapter for repeated custom-kernel
  patterns:
  - flux vector per equation
  - optional wave speed
  - optional dissipation matrix or scalar
  - optional diagnostic components
- Add generic flux kernel adapters only after at least two flux implementations
  need the same boilerplate:
  - cell flux kernel
  - boundary flux kernel
  - interior-face flux kernel
- Ensure these adapters support residual-only and residual-plus-Jacobian modes.
- Preserve direct `AssemblyKernel` implementations as a first-class path.

#### Forms

- Add flux form helper vocabulary only after shape metadata exists and only as
  convenience vocabulary over existing `Forms` operations:
  - `normalFlux(F, n)`
  - `conservativeCellResidual(U, W, F)`
  - `conservativeInteriorFlux(W, Fhat)`
  - `conservativeBoundaryFlux(W, Fhat)`
- Add shape rules:
  - physical flux for `n_eq` equations in `dim` dimensions has shape
    `n_eq x dim`.
  - normal flux has shape `n_eq`.
  - test state must be compatible with `n_eq`.

#### Systems

- Consider installation helpers only as wrappers over existing form/kernel
  registration:
  - `installConservationLaw(...)`
  - `installFluxBoundary(...)`
  - `installInteriorNumericalFlux(...)`
- These should be optional convenience APIs over existing kernel registration.
- Do not require all conservation-law systems to use these helpers. The normal
  mixed-field residual workflow remains valid.

#### Analysis

- Extend existing `ContributionDescriptor`, balance roles, and flux-balance
  summaries with conservation-law descriptors:
  - conserved state group
  - cell flux terms
  - boundary flux terms
  - interior flux terms
  - source terms
  - conservative vs nonconservative form classification
- Avoid a separate conservation-analysis pipeline unless the existing
  `ConservationAnalyzer` and flux-balance summaries cannot express the needed
  metadata.

### Tests

- Scalar linear advection DG with upwind flux.
- Scalar Burgers residual/Jacobian finite-difference check.
- Two-equation toy system with custom numerical flux.
- Interior face conservation test: equal and opposite contributions on
  minus/plus cells.
- Boundary flux test with prescribed exterior state.
- MPI interior-face and boundary-face coverage.
- JIT and interpreter parity where symbolic fluxes are used.

### Definition of Done

- A physics module can implement a conservation-law system without hand-writing
  all trace-gathering and residual-insertion boilerplate.
- Conservation-law terms appear in analysis reports.
- Interior-face flux contributions are conservative by construction in tests.

## Workstream 4: Boundary Exterior-State Infrastructure

### Problem

For hyperbolic and mixed hyperbolic-parabolic systems, boundary conditions are
often expressed by constructing an exterior state and passing the interior and
exterior states to a boundary numerical flux:

```text
Fhat_boundary = Fhat(U_inside, U_outside, n)
```

The exterior state may depend on marker, time, coordinates, normal direction,
flow regime, coupled boundary models, or auxiliary state.

### Desired Capability

The FE library should provide a generic boundary-state mechanism:

- The FE layer identifies marker, quadrature point, normal, interior state, and
  auxiliary inputs.
- A physics-supplied provider constructs an exterior state or boundary flux
  inputs.
- The same infrastructure supports compressible flow, scalar transport,
  species transport, MHD, and multiphase systems.
- Existing `AuxiliaryInputRegistry`, boundary trace/reduction support,
  `FEQuantityRegistry`, and boundary face assembly should provide most of the
  data plumbing.

### Proposed Types

```cpp
struct BoundaryStateRequest {
    int marker{-1};
    Real time{};
    Real dt{};
    CoordinateView x;
    NormalView normal;
    StateTraceView interior;
    AuxiliaryInputView auxiliary;
};

class BoundaryStateProvider {
public:
    virtual ~BoundaryStateProvider() = default;
    virtual std::string name() const = 0;
    virtual StateShape stateShape() const = 0;
    virtual void evaluate(const BoundaryStateRequest& request,
                          StateValueView exterior) const = 0;
};
```

### Required Changes

#### Systems

- Start with boundary-state bindings owned by `FESystem` and backed by existing
  auxiliary/boundary infrastructure:
  - map marker plus state group to provider
  - validate marker coverage
  - expose provider metadata to analysis
- Integrate provider resolution into flux boundary installation.
- Do not add a standalone `BoundaryStateRegistry` unless marker/provider
  management grows beyond what `FESystem`, `AuxiliaryInputRegistry`, and
  boundary-condition infrastructure can express cleanly.

#### Forms

- Add optional boundary exterior-state terminals as wrappers over auxiliary
  bindings and state-group traces:
  - `ExteriorState(group, marker_policy)`
  - `boundaryState("name")`
- Keep direct `FormExpr` boundary terms available.

#### Assembly

- Provide interior trace values and normals to providers.
- Support Real and AD/Dual evaluation modes where possible.
- Provide fallback custom-kernel path for providers not expressible in Forms.
- Reuse existing boundary-face and interface-face context data. Add only the
  missing grouped-state/exterior-state view adapters.

#### Analysis

- Extend existing analysis summaries with boundary-state descriptors:
  - provider name
  - target state group
  - markers
  - dependency summary
  - whether provider is conservative, reflective, prescribed, coupled, or
    unknown if the provider declares that metadata

### Tests

- Scalar advection inflow exterior-state provider.
- Reflective boundary provider for a toy two-component state.
- Time-dependent prescribed boundary state.
- Coupled auxiliary input boundary state.
- Missing marker diagnostics.
- Residual/Jacobian tests when provider is differentiable.

### Definition of Done

- Physics modules can express boundary flux conditions without duplicating
  boundary trace plumbing.
- Boundary exterior-state metadata appears in analysis reports.

## Workstream 5: Admissibility and Invariant-Domain Infrastructure

### Problem

Complex nonlinear physics often has state domains:

- density must be positive
- internal energy must be positive
- temperature must be positive
- species mass fractions must be bounded
- saturation must lie in `[0, 1]`
- deformation Jacobian must be positive

These constraints are physics-defined, but the FE library needs generic hooks
to check, report, limit, project, or reject updates.

### Desired Capability

Provide generic hooks for state-domain constraints and runtime admissibility
handling.

This should be phased. Metadata and diagnostics should come first by extending
existing invariant-domain and preservation analysis. Runtime enforcement
through line searches, projections, limiters, or step rejection should be added
only when a solver/time-loop integration needs it.

### Proposed Types

```cpp
enum class AdmissibilityAction {
    DiagnoseOnly,
    RejectStep,
    LineSearch,
    Project,
    Limit
};

enum class ConstraintRelation {
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessEqual,
    Between,
    Custom
};

struct StateConstraintSpec {
    std::string name;
    StateGroupId group;
    std::string expression_name;
    ConstraintRelation relation;
    Real lower{};
    Real upper{};
    Real tolerance{};
    AdmissibilityAction action{AdmissibilityAction::DiagnoseOnly};
};

class AdmissibilityModel {
public:
    virtual ~AdmissibilityModel() = default;
    virtual std::string name() const = 0;
    virtual void evaluate(const StateValueView& state,
                          AdmissibilityResult& result) const = 0;
};
```

### Required Changes

#### Core

- Extend existing state-variable and preservation metadata with generic
  constraint descriptors where possible.
- Add result/status types:
  - satisfied
  - violated
  - minimum margin
  - maximum violation
  - first failing field/element/QP if available

#### Systems

- Add registration only as a thin `FESystem` metadata layer over state groups:
  - `FESystem::addAdmissibilityConstraint(...)`
  - `FESystem::addAdmissibilityModel(...)`
- Integrate checks at optional points:
  - before residual assembly
  - after nonlinear update proposal
  - before accepting a time step
  - during output diagnostics
- Keep policy configurable so existing workflows are unaffected.
- Keep diagnose-only behavior as the default. FE core should not silently modify
  solution updates unless an explicit solver/time-step policy requests it.

#### Assembly

- Add efficient quadrature-point state scans for registered groups.
- Provide optional nodal scans for nodal positivity or bounds.
- Provide hooks for limiters or projectors without implementing
  physics-specific limiting in FE core.
- Reuse state-group views and existing scalar/vector field accessors for the
  first implementation.

#### Nonlinear Solvers / Systems

- Expose a generic line-search admissibility callback:
  - proposed state
  - accepted alpha
  - violation diagnostics
- Step rejection should be available to time loops.

#### Analysis

- Extend existing invariant-domain and preservation summaries. Add a new
  analyzer only if the current preservation/analysis pipeline cannot consume
  the descriptors.
- Reports should include:
  - constraints registered
  - whether checks are active
  - last/known violation summaries
  - whether positivity preservation is claimed, checked, or unknown

### Tests

- Scalar positivity constraint on a toy field.
- Mixed-state constraint that depends on multiple fields.
- Diagnostic-only mode does not alter solves.
- Reject-step mode returns clear failure status.
- Line-search callback reduces update size in a controlled test.
- Analysis report includes constraints.

### Definition of Done

- Physics modules can register state-domain constraints once.
- Assembly/solvers/time loops can consume those constraints generically.
- FE analysis can report whether positivity or bounds are merely declared,
  checked, or enforced.

## Workstream 6: Stabilization, Sensors, and Artificial Diffusion Framework

### Problem

Stabilized methods and shock-capturing methods need reusable FE plumbing:

- element size metrics
- residual norms
- gradients and jumps
- sensors
- artificial diffusion coefficients
- storage for element or quadrature-point stabilization data
- analysis metadata

The actual formulas are physics-specific, but the infrastructure is not.

### Desired Capability

Provide a generic framework for element-wise and quadrature-point stabilization
data that can be consumed by forms or custom kernels.

This should reuse the existing auxiliary quantity/input/state machinery before
adding a stabilization-specific registry. Stabilization data is still FE data:
it should be exposed through the same mechanisms used for derived quantities,
sampled state fields, formulation callbacks, and auxiliary state whenever
possible.

### Proposed Types

```cpp
enum class StabilizationDataLocation {
    Element,
    QuadraturePoint,
    Face
};

struct StabilizationFieldSpec {
    std::string name;
    StabilizationDataLocation location;
    int components{1};
    bool time_dependent{false};
};

class SensorModel {
public:
    virtual ~SensorModel() = default;
    virtual std::string name() const = 0;
    virtual void evaluate(const AssemblyContext& ctx,
                          SensorOutputView& out) const = 0;
};
```

### Required Changes

#### Assembly

- Add optional element-sensor evaluation pass.
- Reuse `AuxiliaryStateManager`, FE quantities, or existing element/QP data
  channels for storage where they fit.
- Allow kernels to request stabilization data as `RequiredData`.
- Support serial and MPI ownership rules for element-local sensor data.

#### Forms

- Add terminals for registered stabilization fields as wrappers over existing
  auxiliary bindings:
  - `stabilizationField("name")`
  - `elementSensor("name")`
- Add helper vocabulary for common FE quantities:
  - element Peclet-like scales
  - residual norm access
  - face jump norms
  - directional mesh size
- Keep formulas physics-supplied.

#### Systems

- Start with stabilization metadata attached to `FESystem` and existing
  auxiliary/FE-quantity registries.
- Add a dedicated `StabilizationRegistry` only if repeated stabilization
  implementations need lifecycle management that the auxiliary infrastructure
  cannot provide.
- Configure when sensors update:
  - once per setup
  - every nonlinear iteration
  - every time step
  - on demand
- Allow physics modules to declare dependencies.

#### Analysis

- Extend existing contribution descriptors and stabilization analysis summaries
  with:
  - sensor name
  - active terms
  - artificial diffusion field
  - update cadence
  - whether term is conservative, dissipative, nonlinear, or unknown

### Tests

- Register an element sensor and consume it in a scalar diffusion form.
- Update sensor values as the solution changes.
- Verify quadrature-point stabilization terminal in interpreter and JIT modes.
- Verify MPI assembly uses correct local data.
- Verify analysis report lists stabilization fields.

### Definition of Done

- A physics module can register stabilization data and use it in `Forms`
  without hand-threading custom arrays through every kernel.
- Sensor update timing is explicit and tested.

## Workstream 7: Constitutive and EOS Metadata Improvements

### Problem

The existing constitutive call surface is useful, but complex physics needs
richer metadata:

- named inputs and outputs
- admissible input domains
- output scales and units
- derivative availability
- inlinability/JIT support
- multi-output caching
- Real and Dual evaluation consistency

Compressible flow, for example, may need:

```text
p, T, a, mu, kappa, cp, cv, entropy
```

from density, energy, species, and temperature-related inputs.

### Desired Capability

Make constitutive/EOS-like models easier to register, validate, call, cache,
differentiate, and analyze.

The implementation should extend the existing constitutive plan, state-layout
objects, state-variable metadata, material-state specs, and inlinable
constitutive lowering hooks. Do not add a parallel constitutive registry or a
second state-layout model unless the existing interfaces are first proven
insufficient.

### Proposed Types

```cpp
struct ConstitutiveInputSpec {
    std::string name;
    FieldShape shape;
    std::optional<Real> lower_bound;
    std::optional<Real> upper_bound;
    Real scale{1.0};
};

struct ConstitutiveOutputSpec {
    std::string name;
    FieldShape shape;
    Real scale{1.0};
    bool differentiable{true};
};

struct ConstitutiveModelMetadata {
    std::string model_name;
    std::vector<ConstitutiveInputSpec> inputs;
    std::vector<ConstitutiveOutputSpec> outputs;
    bool supports_dual{false};
    bool supports_jit_inline{false};
    bool pure_function{true};
};
```

### Required Changes

#### Constitutive

- Extend existing constitutive model interfaces, `Constitutive/StateLayout.h`,
  and material-state metadata with the missing named input/output/domain
  information.
- Add named output lookup:
  - `call.out("pressure")`
  - `call.out("temperature")`
- Add input-domain descriptors.
- Add derivative availability descriptors.
- Add optional caching when multiple outputs are requested from the same inputs.
- Preserve existing index-based and inlinable constitutive call paths.

#### Forms

- Support named constitutive outputs in vocabulary.
- Validate output shape.
- Use metadata to produce better diagnostics.
- Preserve index-based output access for compatibility.

#### Assembly/JIT

- Ensure metadata distinguishes:
  - interpreter-only opaque callbacks
  - AD-compatible callbacks
  - symbolically differentiable/inlinable models
  - JIT-fast models

#### Analysis

- Extend existing coefficient/constitutive analysis summaries:
  - positivity
  - bounds
  - scales
  - differentiability
  - parameter dependence
  - domain assumptions

### Tests

- Multi-output toy constitutive model with named outputs.
- Input-domain validation failure.
- AD finite-difference consistency.
- JIT/inlining metadata behavior.
- Analysis report includes output names and positivity/bounds.

### Definition of Done

- Physics modules can call multi-output EOS/transport models by name.
- Constitutive domain assumptions can feed admissibility and analysis.
- Existing index-based constitutive calls remain supported.

## Workstream 8: Hyperbolic Wave-Speed and CFL Metadata

### Problem

Complex transport and conservation-law systems often need local spectral-radius
or wave-speed estimates for:

- explicit time stepping
- IMEX splitting
- nonlinear damping
- stabilization parameters
- CFL diagnostics
- adaptive time-step control

The FE layer should not know the physical wave speeds, but it can provide the
registration, aggregation, and reporting infrastructure.

### Desired Capability

Allow physics modules to provide local wave-speed estimates and let FE systems
consume those estimates generically.

This should feed the existing temporal-stability and transport-character
analysis summaries first. Time-step controller integration can remain optional
and later.

### Proposed Types

```cpp
struct WaveSpeedRequest {
    StateGroupId group;
    ElementId element;
    QuadraturePointId qpt;
    NormalView normal;       // optional for directional estimates
    StateValueView state;
};

struct WaveSpeedResult {
    Real spectral_radius{0.0};
    Real directional_speed{0.0};
    Real recommended_dt{0.0};
};

class WaveSpeedEstimator {
public:
    virtual ~WaveSpeedEstimator() = default;
    virtual std::string name() const = 0;
    virtual void evaluate(const WaveSpeedRequest& request,
                          WaveSpeedResult& result) const = 0;
};
```

### Required Changes

#### TimeStepping

- Add registration for wave-speed estimators only if no existing time-step or
  analysis metadata path can own the estimator cleanly.
- Add utility to compute global minimum recommended `dt`.
- Integrate with step controllers as optional metadata.
- Keep implicit workflows able to ignore CFL suggestions.

#### Assembly

- Provide state values and mesh sizes to estimators.
- Support element-wise and face-normal directional estimates.

#### Systems

- Add setup and runtime APIs through `FESystem` or the existing analysis
  metadata path:
  - `FESystem::addWaveSpeedEstimator(...)`
  - `FESystem::computeCFLSummary(...)`
- Keep these APIs optional; no residual assembly should depend on wave-speed
  metadata unless a physics module explicitly wires it into stabilization or
  time stepping.

#### Analysis

- Extend existing temporal-stability and transport-character summaries with:
  - max spectral radius
  - min element dt
  - limiting element/marker
  - estimator name
  - whether estimate is directional or isotropic

### Tests

- Scalar advection constant wave speed.
- State-dependent wave speed in a nonlinear toy system.
- MPI global min/max reduction.
- Time-step controller consumes estimator output.
- Analysis report includes CFL summary.

### Definition of Done

- Physics modules can provide wave-speed estimates without modifying
  time-loop internals.
- FE can report CFL summaries independent of the specific physics.

## Workstream 9: Solver Block and Preconditioner Metadata

### Problem

Large mixed systems need solver and preconditioner guidance. The FE backends
already advertise generic block capabilities, but the system needs richer
metadata to describe logical block structure.

Compressible mixed fields may use:

```text
[rho] [m] [E]
```

or grouped:

```text
[conserved_state]
```

Future systems may need field splits such as:

```text
[fluid state] [temperature] [species] [structure] [auxiliary]
```

### Desired Capability

Let physics modules provide solver-block metadata without hard-coding backend
details.

### Proposed Types

```cpp
enum class SolverBlockRole {
    PrimaryState,
    Constraint,
    LagrangeMultiplier,
    Auxiliary,
    MeshMotion,
    PressureLike,
    TransportLike,
    ThermalLike
};

struct SolverBlockSpec {
    std::string name;
    std::vector<FieldId> fields;
    SolverBlockRole role{SolverBlockRole::PrimaryState};
    bool strongly_coupled{true};
};
```

### Required Changes

#### Systems

- Extend existing field/backend metadata with solver block intent. A small
  `SolverBlockSpec` table owned by `FESystem` is reasonable, but it should feed
  the current backend capability and analysis paths rather than create a new
  solver configuration system.
- Provide APIs:
  - `addSolverBlock(...)`
  - `solverBlocks()`
  - `defaultSolverBlocksFromFields()`
- Connect to backend compatibility checks.

#### Backends

- Use block metadata to configure existing generic field-split options where
  available.
- Clearly report when a backend only supports monolithic or two-field
  saddle-point solves.
- Do not add backend-specific physics assumptions.
- Preserve monolithic assembly and solve as the default fallback.

#### Analysis

- Extend existing solver compatibility summaries based on:
  - number of blocks
  - backend capabilities
  - matrix-free requirements
  - field split availability
  - distributed solve support

### Tests

- Three-block mixed system with PETSc/Trilinos capability metadata.
- FSILS compatibility diagnostic for unsupported generic block split.
- Monolithic fallback remains allowed when backend supports it.
- Analysis report identifies solver block layout.

### Definition of Done

- Solver block intent is explicit and visible.
- Backend compatibility diagnostics are clear before runtime failures.

## Workstream 10: Analysis and Diagnostics Integration

### Problem

New infrastructure is only useful if users can understand what was installed.
The FE analysis subsystem should describe complex systems in physics-agnostic
terms.

### Desired Capability

Extend analysis support for:

- field shapes
- logical state groups
- conservation-law fluxes
- boundary exterior states
- admissibility constraints
- stabilization fields
- constitutive domains and outputs
- wave-speed/CFL summaries
- solver block layout

### Required Changes

#### Analysis Types

Extend `AnalysisSummaryTypes.h`, `ContributionDescriptor.h`, and related
problem-analysis types where possible. Add new summary structs only for data
that does not fit an existing summary family.

Candidate descriptors/summaries:

```text
FieldShapeSummary
StateGroupSummary
ConservationLawSummary
BoundaryStateSummary
AdmissibilitySummary
StabilizationSummary
ConstitutiveMetadataSummary
WaveSpeedSummary
SolverBlockSummary
```

#### Analyzer Responsibilities

Prefer extending existing analyzers and report planners:

- field-shape reporting
- conservation-law and flux-balance reporting
- admissibility and invariant-domain reporting
- stabilization reporting
- constitutive metadata reporting
- wave-speed and temporal-stability reporting
- solver-compatibility reporting

These are conceptual responsibilities, not a requirement to add new classes. If
an existing analyzer already owns the responsibility, extend it.
For example, conservation metadata should feed existing conservation/flux
balance analysis, wave-speed metadata should feed temporal-stability analysis,
admissibility should feed invariant-domain/preservation analysis, and
stabilization should feed existing stabilization contribution summaries.

#### Reports

Reports should answer:

- What fields are present and what do their components mean?
- Which fields form logical states?
- Which residual terms are conservative?
- Which boundaries use exterior-state providers?
- Which constraints or admissibility checks are active?
- Which stabilization terms are active?
- Which constitutive models are used and what assumptions do they declare?
- What are the current CFL/wave-speed constraints?
- Can the selected backend handle the block structure?

### Tests

- Analysis snapshot tests for a toy conservation-law system.
- Analysis snapshot tests for a mixed-field compressible-like state.
- Missing metadata should be reported as unknown, not silently assumed.

### Definition of Done

- Analysis reports complex FE systems without knowing specific physics names.
- Missing information is visible as unknown or undeclared.

## Workstream 11: Documentation and User-Facing Workflow

### Problem

Complex infrastructure must have a clear workflow. Without documentation,
physics modules will bypass the infrastructure and reintroduce local patterns.

### Required Documentation

Update existing docs first, then add focused new docs only when a topic becomes
too large for the existing page. Documentation should clearly point to the
existing FE owner for each capability.

Cover:

- field shape semantics
- mixed-field state groups
- conservation-law flux authoring
- boundary exterior-state providers
- admissibility and invariant domains
- stabilization and sensors
- constitutive metadata
- wave-speed/CFL estimators
- solver block metadata

Recommended files:

```text
FE/Docs/FieldShapeSemantics.md
FE/Docs/StateGroups.md
FE/Docs/ConservationLawFluxes.md
FE/Docs/BoundaryExteriorStates.md
FE/Docs/AdmissibilityInfrastructure.md
FE/Docs/StabilizationInfrastructure.md
FE/Docs/ConstitutiveMetadata.md
FE/Docs/WaveSpeedAndCFL.md
FE/Docs/SolverBlockMetadata.md
```

These can be separate files or sections in existing docs depending on size.
Avoid creating a documentation page for a subsystem that remains only a small
extension to an existing API.

Update:

- `FE/README.md`
- `FE/Forms/VOCABULARY.md`
- `FE/Forms/SYSTEMS_INTEGRATION.md`
- `FE/Analysis/README.md`
- `FE/Backends/PLAN.md`
- `FE/TimeStepping/PLAN.md`

### Examples to Include

- Scalar advection with upwind DG flux.
- Two-equation nonlinear conservation-law toy model.
- Mixed-field compressible-state skeleton with `rho`, `m`, and `E`.
- Boundary exterior-state provider example.
- Positivity constraint registration example.
- Stabilization sensor example.
- Wave-speed estimator example.

### Definition of Done

- A new developer can identify the correct FE extension point for each problem.
- Docs clearly distinguish FE infrastructure from physics-module laws.

## Workstream 12: Test and Verification Strategy

Tests should prove both the new capability and the reuse contract. When a
feature extends an existing registry, analyzer, or assembly path, include a test
that exercises the existing path with the new metadata enabled.

### Unit Tests

Add focused unit tests for:

- field-shape validation
- shape metadata consistency between field specs, FE quantities, forms, and
  analysis summaries
- state-group flattening
- form shape rules
- conservation flux helpers
- exterior-state providers
- admissibility checks
- sensor evaluation
- constitutive metadata
- wave-speed estimators
- solver block metadata
- analysis descriptors
- compatibility behavior when metadata is omitted and legacy inference is used

### Integration Tests

Add small assembled systems:

- scalar advection DG
- scalar nonlinear Burgers equation
- two-component conservation-law toy problem
- mixed scalar/vector/scalar state problem
- boundary exterior-state flux problem
- positivity-check failure and recovery problem

### Jacobian Tests

For each new differentiable path:

- compare assembled Jacobian against finite differences
- test residual-only, matrix-only, and residual-plus-matrix requests
- test interpreter and JIT paths where applicable
- test custom-kernel path if AD or symbolic differentiation is available

### MPI Tests

Cover:

- field-shape metadata distribution
- state-group ownership and ghost values
- interior-face flux conservation across ranks
- boundary exterior-state providers on partition boundaries
- admissibility global reductions
- CFL global reductions

### Regression Tests

Existing tests that must continue passing:

- Poisson forms
- incompressible Navier-Stokes coupled forms
- boundary-condition tests
- mixed-form installer tests
- DG/interior-face tests
- backend capability tests
- analysis tests

### Performance Tests

Track:

- overhead of field-shape checks
- cost of state-group value extraction
- cost of generic flux adapters vs hand-written kernels
- JIT parity and compile-time impact
- sensor update cost
- admissibility scan cost

## Recommended Implementation Phases

### Phase 0: Baseline Inventory and Design Lock

Deliverables:

- Confirm current compressible NS can be implemented as mixed fields.
- Record the existing FE owners for each proposed capability:
  - `FieldRegistry` and `FEQuantityRegistry` for field/quantity shape metadata
  - `Forms::Value`, `SpaceSignature`, and form kernels for expression typing
  - `AssemblyContext` and `AssemblyKernel` for custom-kernel data access
  - `AuxiliaryInputRegistry` and auxiliary state for boundary/sensor data
  - `StateVariableMetadata` and `Constitutive/StateLayout` for constitutive
    state metadata
  - `AnalysisSummaryTypes` and `ContributionDescriptor` for diagnostics
  - backend capability descriptors for solver/block compatibility
- Write a short design note for field semantics and state groups.
- Identify exact current vector-field assumptions in Forms and Assembly.

Exit criteria:

- Team agrees that physical vectors and algebraic state vectors are distinct.
- Team agrees mixed fields remain first-class.
- Team agrees no new registry/analyzer/service will be added where an existing
  one can be extended cleanly.

### Phase 1: Field Shape Metadata

Deliverables:

- Unify existing shape concepts in `FieldSpec`, `FEQuantityShape`,
  `FieldType`, `SpaceSignature`, and `FunctionSpace` introspection.
- Add `FieldShape` and `FieldValueKind` only if a small shared type is the
  cleanest way to avoid duplicate shape metadata.
- Extend `FieldSpec`, `FEQuantityDefinition`, and `SpaceSignature`.
- Add setup-time validation.
- Add basic Forms shape checking.
- Preserve existing APIs and modules.

Exit criteria:

- Existing tests pass.
- New field-shape unit tests pass.
- Invalid algebraic/physical vector operations produce clear diagnostics.

### Phase 2: State Groups for Mixed Fields

Deliverables:

- Add `StateGroupSpec` as a thin `FESystem` layer over already registered
  fields.
- Add a lightweight `StateGroupRegistry` or equivalent table only if needed for
  ownership and lookup.
- Add group introspection.
- Extend existing analysis summaries with state-group metadata.
- Add custom-kernel state-group value views as adapters over existing
  scalar/vector accessors.

Exit criteria:

- A `rho/m/E` state group can be registered and flattened.
- State group metadata appears in analysis output.

### Phase 3: Boundary Exterior States

Deliverables:

- Add `BoundaryStateProvider`.
- Add marker/provider bindings through `FESystem` using existing
  `AuxiliaryInputRegistry`, boundary trace/reduction infrastructure, and
  boundary-face assembly data.
- Add only the missing boundary-state view/adaptor plumbing.
- Add tests with scalar transport and toy systems.

Exit criteria:

- Boundary fluxes can consume interior and exterior state values generically.
- No standalone boundary-state registry exists unless the existing auxiliary
  and boundary systems are insufficient.

### Phase 4: Conservation-Law Flux Plumbing

Deliverables:

- Add `StateTraceView` and flux output views as lightweight adapters over
  existing cell, boundary, and interior-face assembly data.
- Add `Forms` helpers where they are simple vocabulary over existing
  operations.
- Add generic cell and face flux kernel adapters only after repeated
  implementations duplicate the same trace/residual insertion code.
- Add scalar advection and toy-system tests.

Exit criteria:

- Generic flux infrastructure passes conservation and Jacobian tests.
- Direct `Forms` and custom `AssemblyKernel` implementations remain valid.

### Phase 5: Admissibility Infrastructure

Deliverables:

- Add state-domain constraint descriptors tied to state groups and existing
  preservation/invariant-domain analysis.
- Add diagnose-only runtime scan/evaluation hooks first.
- Defer solver/time-step policy integration, line search, projection, and
  limiting until a concrete solver path needs them.
- Extend existing analysis summaries.

Exit criteria:

- Positivity/bounds can be registered and checked for mixed states.
- Default behavior is diagnostic and non-mutating.

### Phase 6: Stabilization and Sensor Framework

Deliverables:

- Represent sensor/stabilization data through existing auxiliary state,
  FE quantities, or auxiliary bindings wherever possible.
- Add sensor metadata and update cadence controls to `FESystem`.
- Add `Forms` terminals for stabilization data as wrappers over existing
  auxiliary binding mechanisms.
- Extend existing stabilization/contribution analysis summaries.

Exit criteria:

- A physics module can register and consume an element sensor without custom
  side channels.
- A dedicated stabilization registry is not introduced unless lifecycle
  requirements exceed the auxiliary infrastructure.

### Phase 7: Constitutive Metadata

Deliverables:

- Extend existing constitutive model interfaces, `Constitutive/StateLayout.h`,
  `StateVariableMetadata`, and `AssemblyKernel` material-state specs with named
  inputs and outputs.
- Add domain, scale, derivative, and JIT/inlinability metadata to the existing
  constitutive metadata path.
- Add named output Forms helper.

Exit criteria:

- Multi-output EOS-like models can be called and analyzed by output name.
- Existing index-based constitutive calls still work.

### Phase 8: Wave-Speed and Solver Metadata

Deliverables:

- Add wave-speed estimator metadata that feeds existing temporal-stability and
  transport-character summaries.
- Add CFL reductions only when a physics module or time-step controller needs
  runtime estimates.
- Add solver block metadata that feeds existing backend capability and solver
  compatibility checks.
- Add MPI reductions and tests.

Exit criteria:

- Hyperbolic physics modules can provide CFL estimates generically.
- Backend diagnostics understand 3+ block systems.
- Monolithic solve paths remain the default fallback.

### Phase 9: Optional Algebraic Vector Field Support

Deliverables:

- Add true arbitrary-component algebraic vector field evaluation.
- Add dynamic component value and gradient storage.
- Add Forms operations for algebraic vectors.
- Add tests for `n > 3`.

Exit criteria:

- A packed 5-component conserved state can be represented as one field.
- Physical vector calculus remains type-safe.

This phase should come after mixed-field state groups unless a specific physics
module urgently requires packed state fields.

## Compressible Navier-Stokes Mapping

### Minimal Implementation With Current FE Infrastructure

The first compressible Navier-Stokes module can be implemented as:

```text
rho : scalar H1 or L2
m   : physical vector H1 or L2
E   : scalar H1 or L2
```

Derived quantities:

```text
u = m / rho
p = (gamma - 1) * (E - 0.5 * inner(m, m) / rho)
T = EOS(rho, E, m, species...)
tau = viscous_stress(u, T, ...)
q = heat_flux(T, ...)
```

Residuals:

```text
R_rho = dt(rho) + div(m)
R_m   = dt(m) + div(m tensor u + p I - tau) - source_m
R_E   = dt(E) + div((E + p) u - tau*u + q) - source_E
```

This can be written today using mixed fields and `installFormulation`, with
additional physics code for EOS, boundary conditions, stabilization, and
validation. The first implementation should use existing scalar and physical
vector field paths, existing mixed-form installation, existing face assembly,
existing auxiliary inputs, and existing custom-kernel escape hatches before
adding any optional conservation-law helper layer.

### Benefits From Expansion Work

The expansion work improves that implementation as follows:

- Extended field shapes make `m` a physical vector and keep `rho/E` scalar
  using the same metadata consumed by field registration, FE quantities, forms,
  assembly, and analysis.
- State groups let `rho/m/E` act as one conserved state for fluxes and
  limiters.
- Boundary exterior states reuse boundary traces, auxiliary inputs, and
  boundary-face assembly for inflow/outflow/wall flux conditions.
- Conservation-law flux plumbing builds on existing DG/interior-face assembly
  and contribution descriptors.
- Admissibility constraints extend invariant-domain/preservation diagnostics to
  track `rho > 0` and internal energy `> 0`.
- Stabilization infrastructure exposes residual-based sensors and artificial
  viscosity through auxiliary/FE quantity data channels.
- Constitutive metadata extends existing state layout and material metadata for
  EOS and transport models.
- Wave-speed estimators feed existing temporal-stability/CFL diagnostics.
- Solver block metadata extends backend compatibility reporting for `[rho] [m]
  [E]` or grouped-state block layouts.

### What Remains Physics-Specific

The following should not be implemented in FE core:

- ideal-gas or real-gas EOS
- viscosity and thermal-conductivity laws
- Roe/HLL/HLLC/Lax-Friedrichs flux formulas
- entropy-stable flux formulas
- wall/inflow/outflow characteristic logic
- shock sensor formulas
- positivity limiter formulas
- compressible-flow benchmark cases

Those belong in `Physics` or a physics-side reusable library.

## Backward Compatibility Plan

- Existing `components` and `value_dimension()` APIs remain available.
- Fields without explicit `FieldShape` infer legacy behavior.
- Physical vector paths continue using optimized `Vector3D` storage.
- Existing `Forms` expressions keep compiling.
- New shape checks should be introduced first as validation diagnostics, then
  hardened as errors once existing modules are migrated.
- Algebraic vector fields are optional and should not change mixed-field
  behavior.

## Risks and Mitigations

### Risk: Parallel Infrastructure Clutters the FE Library

Mitigation:

- For every proposed public type, first identify the existing FE owner.
- Extend `FieldRegistry`, `FEQuantityRegistry`, `AuxiliaryInputRegistry`,
  `AssemblyKernel` metadata, `Constitutive/StateLayout`, `AnalysisSummaryTypes`,
  `ContributionDescriptor`, and backend capability descriptors before adding
  new systems.
- Add a new registry/analyzer/service only when extension would make the
  existing owner incoherent.
- Require at least one concrete physics implementation or two repeated
  internal use cases before adding broad convenience APIs.

### Risk: Overengineering Before Compressible NS Exists

Mitigation:

- Implement compressible NS first with mixed fields.
- Promote generic infrastructure only when a concrete use case needs it.
- Prioritize field semantics and state groups because they clarify existing
  code without requiring full conservation-law infrastructure.

### Risk: Blurring Physical and Algebraic Vector Semantics

Mitigation:

- Add explicit `FieldValueKind`.
- Preserve physical vector checks.
- Reject invalid geometric operations on algebraic vectors.

### Risk: Generic Flux Interfaces Become Too Abstract

Mitigation:

- Start with scalar advection and a two-equation toy system.
- Keep direct custom kernel registration available.
- Keep flux adapters optional, not mandatory.

### Risk: AD/JIT Incompatibility for User-Supplied Models

Mitigation:

- Require models to declare differentiability and JIT support.
- Provide clear fallback paths.
- Add finite-difference tests for differentiable models.

### Risk: Solver Metadata Is Ignored by Backends

Mitigation:

- First use metadata for diagnostics and compatibility reports.
- Integrate backend configuration incrementally.

## Open Design Decisions

1. Can `FEQuantityShape`, `FieldSpec::components`, `FieldType`, and
   `SpaceSignature` share one shape model without adding a new standalone
   `FieldShape` type?
2. If a standalone `FieldShape` type is needed, should it live in `Core` or in
   `Systems` beside field/quantity registration?
3. Should algebraic vector fields use a new `FunctionSpace` subclass or a
   shape annotation on existing scalar spaces?
4. Should state groups be owned by `FESystem` only, or also be representable in
   standalone `Forms` compilation?
5. Should conservation-law fluxes be primarily `Forms` expressions, custom
   kernels, or both from the start?
6. How much solver/time-step policy should admissibility constraints control,
   and what remains diagnose-only metadata?
7. Can stabilization fields be stored through existing FE quantities,
   auxiliary state, or auxiliary inputs, or is a new element/QP data channel
   truly required?
8. How should named constitutive outputs interact with existing output-index
   calls?
9. What is the minimum backend integration needed for useful solver block
   metadata?
10. Which existing analysis summaries should own each new descriptor, and which
    descriptors genuinely require new summary structs?

## Recommended Near-Term Actions

1. Keep the first compressible Navier-Stokes implementation mixed-field based.
2. Inventory and consolidate existing shape metadata before adding a new
   `FieldShape` type.
3. Add `StateGroupSpec` for logical grouping of mixed fields as a thin
   `FESystem` extension over existing registered fields.
4. Build the first boundary exterior-state path on existing boundary traces and
   auxiliary-input infrastructure.
5. Add admissibility descriptors and analysis reporting before implementing
   positivity-sensitive nonlinear solve policies.
6. Add generic flux helpers only after one or two concrete flux forms expose
   repeated trace/residual boilerplate.
7. Extend existing analysis summaries and backend capability diagnostics before
   creating new analyzer or solver-metadata subsystems.

This sequence keeps the FE library moving toward a stronger general framework
without blocking immediate physics progress.
