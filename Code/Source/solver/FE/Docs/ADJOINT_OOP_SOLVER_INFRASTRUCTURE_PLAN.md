# Adjoint Infrastructure Plan for the OOP FE Solver

## Objective

Add the FE-library infrastructure required to support adjoint problems in the new OOP solver path.

The target is a physics-agnostic FE surface that can support:

- steady-state discrete adjoint solves
- transient discrete adjoint solves
- PDE-only and PDE-plus-auxiliary coupled adjoints
- objective or quantity-of-interest gradients with respect to field states
- later extension to parameter, control, and shape sensitivities
- goal-oriented error estimation and dual-weighted residual workflows

The FE layer must remain physics-agnostic. The target is not "a Navier-Stokes adjoint implementation" as a special case. The target is the reusable FE infrastructure that physics modules can build on.

The adjoint layer should be a consumer of existing Systems, Analysis, MovingMesh,
and current or planned FE/Coupling public metadata. It must not introduce a parallel
residual-analysis IR, coupling graph, transfer registry, geometry-transaction
model, or time-history model.

## Scope

### In Scope

- transpose-capable backend and operator interfaces
- first-class FE adjoint problem definitions and solve orchestration hooks
- generic objective or quantity-of-interest gradient assembly
- adjoint-consistent treatment of algebraic constraints
- adjoint support for coupled auxiliary-state infrastructure
- transient replay, checkpoint, and reverse-time infrastructure
- adjoint-ready metadata and preflight diagnostics using the public Analysis path
- transfer-adjoint contracts for interface, coupling, and marker-style operators
- a dedicated FE/Adjoint module boundary with clear dependency rules
- an adjoint-ready formulation contract for automatic support of new physics
- explicit policies for expert/custom kernels, nonsmooth behavior, and active
  sets
- verification tests for steady and transient adjoint paths
- documentation and usage guidance

### Out of Scope for the Initial Effort

- physics-specific adjoint PDE modules
- gradient-based optimization algorithms
- mesh adaptation algorithms beyond the FE hooks needed by DWR workflows
- full shape-calculus coverage for every geometry path in the first milestone
- a promise that every backend will support every adjoint feature on day one
- a new private Forms AST traversal or independent contribution-analysis model
- automatic support for arbitrary opaque C++ callbacks or custom kernels that do
  not provide public tangent, sensitivity, and transpose metadata

## Current State Summary

### What Already Exists

- Residual differentiation with respect to the active field already exists in `Forms/SymbolicDifferentiation.*`:
  - `differentiateResidual(...)`
  - `differentiateResidualHessianVector(...)`
- Auxiliary-output differentiation already exists in `Forms/SymbolicDifferentiation.*` via `differentiateWrtAuxiliaryOutput(...)`.
- Coupled residual sensitivity infrastructure already exists in `Forms/FormKernels.*` via `CoupledResidualSensitivityKernel`.
- Boundary and selected domain or region functional state gradients already
  exist for the coupled-boundary and FE-backed-quantity paths through:
  - `Systems/BoundaryReductionService.*`
  - `Systems/FESystem::assembleBoundaryGradient(...)`
  - `Forms/BoundaryFunctionalGradientKernel`
- FE-backed quantity metadata already exists in `Systems/FEQuantityDefinition.*`
  and `Systems/FEQuantityRegistry.*`, covering sampled fields, boundary/domain
  integrals, region integrals, FE expressions, shape metadata, referenced fields,
  and monolithic-linearization capability flags.
- Scalar functional and sampled-expression evaluation paths already exist in
  `Assembly/FunctionalAssembler.*`, `Systems/OperatorBackends.*`, and
  `Forms/PointEvaluator.*`. `PostProcessing/DerivedResultRegistry.*` is also an
  existing result-definition registry, but it should be treated as
  post-processing metadata unless a result is explicitly promoted to an
  adjoint-ready QoI with derivative support.
- Adjoint-consistency metadata already exists in `Analysis/ContributionDescriptor.*` and related analysis code.
- The Analysis infrastructure now carries or is planned to carry normalized
  contribution, dependency, domain, operator-class, error-estimator, and
  adjoint-consistency summaries. These summaries are readiness evidence, not by
  themselves authoritative solve-readiness contracts. `analysis::VariableKey` is
  the preferred identity for non-field graph variables.
- The constraints layer already contains local algebraic primitives for
  reduced-space transforms in `Constraints/ConstraintTransform.*`, including:
  - `P`
  - `P^T`
  - reduced operator application `P^T A P`
  - reduced RHS construction `P^T (b - A c)`
  These primitives still need backend-vector, DOF-layout, ghost-ownership,
  distributed-sparsity, and preconditioner/nullspace integration before they can
  be treated as a complete FE adjoint constraint path.
- Constraint dependency and moving-constraint metadata already exist in
  `Constraints/ConstraintDependency.*` and
  `Constraints/MovingConstraintComposition.*`, including geometry, time,
  topology, layout, active-state, and tangent-policy declarations.
- Moving-domain infrastructure now distinguishes prescribed mesh-motion data,
  derived moving-domain fields, and coupled mesh-displacement unknowns. Coupled
  ALE geometry-sensitive tangent paths are being built through Forms and
  Systems install options.
- Moving-domain and geometry-update lifecycle metadata already exists or is
  landing through `Systems/GeometryTransaction.*`,
  `Systems/GeometricNonlinearity.*`, `MovingMesh/MovingDomainOrchestrator.*`,
  `Systems/FEAdaptivityTransfer.*`, `Systems/ContactOperatorInvalidation.*`,
  and `Systems/CutIntegrationInvalidation.*`.
- `Systems/GlobalKernel.*` and `Systems/GlobalKernelStateProvider.*` already
  provide public extension points for non-element-local operator contributions,
  analysis metadata, sparsity augmentation, parameter requirements, and old/work
  global-kernel state. They do not yet provide a complete adjoint
  checkpoint/replay contract.
- Constitutive/material metadata already includes
  `Constitutive/StressTangentContract.*`, material-state specifications, and
  local derivative helpers such as `Forms/Dual.*` and
  `Constitutive/DualOps.*`.
- `FE/Coupling` is the current and planned owner of participant, endpoint,
  transfer, graph, temporal, geometry-terminal, and provenance metadata for
  monolithic and partitioned multiphysics workflows. The adjoint plan should
  extend those public records rather than create a sidecar transfer or coupling
  registry.
- The transient layer already has forward state/history containers:
  - `Systems/SystemState.h`
  - `TimeStepping/TimeHistory.*`
  - `Systems/TimeIntegrator.*`
  - `TimeStepping/TimeLoop.*`
- Auxiliary state already has pack or unpack support for checkpoint or restart in:
  - `Auxiliary/AuxiliaryStateManager.*`
  - `Systems/FESystem::checkpointAuxiliaryState()`

### What Is Missing

- There is no backend-neutral transpose solve contract.
- There is no backend-neutral transpose operator-apply contract.
- There is no first-class FE adjoint API at the `Systems` layer.
- There is no generic adjoint-ready objective/QoI adapter that unifies
  `FEQuantityRegistry`, `BoundaryReductionService`, `FunctionalAssembler`,
  `OperatorBackends`, and `PointEvaluator` while preserving derivative
  capability diagnostics.
- The constraint transpose machinery is not wired into the system or solver path,
  including backend-owned, distributed, ghosted, and block-structured vector
  paths.
- Constraint dependency/tangent policies are not yet checked by an adjoint
  readiness report.
- There is no public adjoint-ready metadata contract that ties Systems,
  Analysis, FE/Coupling, MovingMesh, and Forms install provenance
  together.
- There is no adjoint preflight report that can say why a formulation or
  coupled problem is not adjoint-ready.
- There is no end-to-end steady adjoint solve path.
- There is no end-to-end transient discrete adjoint path.
- There is no public derivative path with respect to FE parameter slots or design variables.
- There is no general transpose/adjoint contract for interface transfers,
  coupling transfer declarations, driver-owned transfers, or marker-style
  gather/spread operators.
- There is no complete moving-domain adjoint replay policy covering current
  coordinates, geometry revisions, interface maps, mesh-motion field sources,
  and geometry-sensitive tangent provenance.
- There is no global-kernel adjoint contract for transpose contributions,
  sparsity/block provenance, parameter derivatives, or replayable state snapshots.
- There is no constitutive/material adjoint-readiness check for consistent
  tangents, update-frame conventions, material-state history, or derivative
  support.
- Goal-oriented error estimation support is still placeholder-level in `Assembly/FunctionalAssembler.cpp`.
- There are no end-to-end adjoint regression tests in `FE/Tests`.

## Proposed Module Boundary

### Directory Layout

The adjoint implementation should live primarily in a dedicated FE module:

```text
Code/Source/solver/FE/Adjoint/
  AdjointTypes.h
  AdjointReadiness.h
  AdjointReadiness.cpp
  AdjointProblem.h
  AdjointProblem.cpp
  AdjointSolver.h
  AdjointSolver.cpp
  AdjointObjective.h
  AdjointObjective.cpp
  AdjointGradient.h
  AdjointGradient.cpp
  AdjointDiagnostics.h
  AdjointDiagnostics.cpp
  TransferAdjoint.h
  TransferAdjoint.cpp
  TransientAdjointReplay.h
  TransientAdjointReplay.cpp
```

`FE/Systems` should expose the high-level entry points needed by application and
physics code, but the reusable adjoint data structures, readiness checks,
diagnostics, objective-gradient helpers, transfer-adjoint utilities, and
transient replay helpers should live under `FE/Adjoint`.

### Dependency Direction Rules

- `FE/Adjoint` may consume public records and APIs from `FE/Systems`,
  `FE/Analysis`, `FE/Backends`, `FE/Constraints`, `FE/Assembly`, `FE/Forms`,
  `FE/MovingMesh`, and current or planned `FE/Coupling`.
- `FE/Systems` may expose adjoint entry points and metadata hooks, but should
  avoid owning most adjoint implementation details.
- `FE/Analysis` should not depend on `FE/Adjoint`; it should expose normalized
  metadata that `FE/Adjoint` can consume.
- `FE/Coupling` should not depend on `FE/Adjoint`; it should expose
  coupling graph, endpoint, transfer, temporal, geometry-terminal, and
  provenance metadata that `FE/Adjoint` can consume.
- Physics modules should not call backend-specific adjoint APIs. They should
  expose residuals, objectives, parameters, coupling contracts, and metadata
  through Forms, Systems, Analysis, and FE/Coupling public paths.
- Application code should select objectives, design variables, and solver
  options, then call public Systems/Adjoint APIs. It should not manually build
  backend transpose solves.

### FE README Follow-Up

Once a real `FE/Adjoint` skeleton exists, update `Code/Source/solver/FE/README.md`
to list `Adjoint/` in the module architecture table and add a short "which API
to use" row for adjoint readiness checks, adjoint solves, and total-gradient
queries.

## Design Principles

### 1. Discrete First, Not Continuous First

The implementation should target the discrete adjoint of the actual assembled nonlinear or transient system.

That means:

- the primal residual definition in FE is the source of truth
- the adjoint linearization must be consistent with the actual discrete Jacobian
- time-integration adjoints must match the chosen discrete scheme, not an idealized continuous-time derivation

### 2. Metadata First, Not Parallel Infrastructure

The adjoint implementation should consume the public metadata already owned by
the relevant subsystems:

- Forms and Systems own installed residuals, field uses, dependency provenance,
  geometry-sensitivity options, and operator tags.
- Systems owns FE-backed quantity definitions, auxiliary input/output metadata,
  parameter slots, global kernels, operator backends, and setup lifecycle state.
- Analysis owns normalized contribution, domain, dependency, block, stability,
  and adjoint-consistency summaries as diagnostics and evidence. Adjoint
  readiness must still be decided from hard capability contracts on the
  corresponding subsystem.
- FE/Coupling owns current and planned participant, endpoint, transfer, graph,
  temporal, geometry-terminal, and coupling-provenance metadata.
- MovingMesh, Mesh, and Systems geometry transaction APIs own coordinate
  transaction state, revision keys, interface-map provenance, restart payloads,
  invalidation decisions, and rollback/commit semantics.
- Constraints own full/reduced-space maps, value dependencies, structural
  dependencies, moving-constraint composition, and tangent-policy declarations.
- Constitutive owns material tangent contracts, update frames, material-state
  layouts, and local derivative support.

Adjoint code should add public adapters or summaries where a subsystem lacks a
needed hook. It should not recover semantics by walking private Forms AST nodes,
guessing from raw `FieldId`/marker integers, or maintaining an independent
coupling or moving-geometry model.

### 2A. Objective and QoI Support Must Reuse Existing FE Registries

The adjoint layer needs an objective-facing API, but it should not create a
second quantity registry that competes with the FE library. The first
objective/QoI implementation should provide `FE/Adjoint` adapters over existing
public infrastructure:

- `Systems/FEQuantityRegistry.*` and `Systems/FEQuantityDefinition.*` for
  FE-backed quantity kind, shape, field references, region/marker metadata, and
  monolithic-linearization capability, with an added adjoint capability overlay
  or explicit extension before any quantity is treated as adjoint-ready
- `Systems/BoundaryReductionService.*` for boundary and supported domain/region
  integral values and gradients
- `Assembly/FunctionalAssembler.*` and `Systems/OperatorBackends.*` for scalar
  functional evaluation
- `Forms/PointEvaluator.*` for sampled scalar expression values and only the
  limited auxiliary-seed derivative paths it currently supports
- `PostProcessing/DerivedResultRegistry.*` only as read-only post-processing
  metadata unless a derived result is explicitly promoted to an adjoint-ready
  QoI with `dQ/du`, `dQ/dp`, and geometry-derivative contracts

The adjoint objective API should answer whether a registered quantity is
value-only, state-gradient-capable, parameter-gradient-capable, or
geometry-gradient-capable. It should not silently reinterpret post-processing
results as differentiable objectives.

Point or sampled objectives should additionally declare interpolation/search
provenance, basis support, geometry derivative policy, and readiness diagnostics
before they are considered gradient-capable.

### 3. Backend-Neutral Public Contract

The public FE interfaces should talk about:

- operator application
- transpose operator application
- forward solve
- transpose solve

without exposing backend-specific PETSc, Trilinos, FSILS, or Eigen details.

### 4. Incremental Delivery

The recommended delivery order is:

0. adjoint-ready metadata and preflight diagnostics
1. steady assembled-matrix adjoint
2. constraint-consistent steady adjoint
3. transfer-adjoint contracts
4. monolithic coupled auxiliary or multiphysics adjoint
5. partitioned coupled adjoint graph replay
6. transient discrete adjoint
7. parameter, control, geometry, or design derivatives
8. goal-oriented estimators and matrix-free transpose support

### 5. Constraint Semantics Must Be Explicit

Adjoint support becomes fragile if the library leaves constraint semantics implicit.

The FE layer should explicitly define whether the adjoint lives in:

- the full constrained space with algebraic elimination already embedded, or
- a reduced unconstrained space mapped through `P` and `P^T`

The recommendation is to make reduced-space semantics first-class because the
constraints layer already has the local algebraic primitives. The adjoint plan
must still add backend-aware vector, ownership, block-layout, and sparsity
contracts before those primitives are sufficient for production FE solves.

### 6. Transfer and Coupling Adjointness Must Be Explicit

Coupled and partitioned workflows must represent transfers as operators with
declared transpose/adjoint behavior. Interface interpolation, conservative
projection, mortar transfer, driver-owned transfer operators, and marker
gather/spread operators need:

- source and target spaces, value rank, component layout, frame-transform, and
  measure/weight metadata
- accepted/trial interface-map and search provenance where geometry moves
- a normal apply and a transpose apply contract
- random-vector dot-product tests under the declared weights

The adjoint module should never infer transfer adjointness from a port name or
from the physical meaning of a coupling contract.

### 7. Transient Support Must Treat Replay as Infrastructure

Transient adjoints are not just "run the primal backward." They need:

- consistent checkpointing
- replay or recomputation policy
- auxiliary history restoration
- time-integrator-specific transpose actions
- moving-domain transaction replay for coordinates, mesh-motion field sources,
  geometry revisions, and interface maps
- coupling graph replay for partitioned endpoint temporal slots and transfers

This should be an FE infrastructure service, not ad hoc logic inside one physics module.

### 8. Automatic Support Requires an Adjoint-Ready Formulation Contract

The infrastructure can automatically support new physics only when those
physics enter through public, differentiable FE contracts. It should not promise
automatic adjoints for arbitrary opaque C++ code.

A physics formulation is adjoint-ready when:

- residuals are authored through Forms or installed through public Systems
  extension points that emit equivalent Analysis metadata
- the exact discrete tangent used by the primal solve is available as an
  assembled matrix, a normal/transpose operator, or a verified matrix-free
  operator pair
- all state fields, test fields, parameters, coefficients, material-state
  dependencies, auxiliary inputs, auxiliary outputs, boundary functionals,
  boundary integrals, geometry terminals, and coupling dependencies are
  declared through public metadata
- objectives and QoIs are exposed through adjoint objective adapters backed by
  existing FE quantity, functional, boundary-reduction, point-evaluation, or
  explicitly promoted derived-result metadata, and they provide `dQ/du` plus,
  where needed, direct `dQ/dp` or `dQ/dx_geometry` terms
- constraints declare the policy needed to map objective gradients and adjoint
  vectors between full and reduced spaces
- constraints with value, time, geometry, topology, auxiliary, or moving-map
  dependencies declare the tangent policy needed to differentiate those
  dependencies, or fail readiness
- coupling contracts expose endpoint, transfer, temporal-slot, geometry, and
  dependency provenance through current or planned `FE/Coupling`
- moving-domain formulations distinguish prescribed data, derived data, solved
  geometry fields, and design variables
- global kernels expose analysis metadata, sparsity/block provenance,
  parameter requirements, state replay requirements, and transpose/tangent hooks
  for any advertised adjoint path
- constitutive laws expose consistent tangent contracts, material-state
  metadata, history-update policy, update-frame convention, and derivative
  support for advertised adjoint paths
- nonsmooth switches, events, contact, remeshing, limiters, and active-set logic
  either provide a verified derivative policy or fail adjoint preflight

### 9. Expert and Custom Kernels Must Opt In Explicitly

Expert paths such as handwritten cell, boundary, interface, matrix-free, device,
global, constitutive, or backend-local kernels are not automatically
adjoint-ready. They must provide:

- contribution identity, operator tag, residual rows, dependency variables,
  domain kind, and matrix/vector contribution flags
- normal tangent contribution or normal operator apply
- transpose contribution or transpose operator apply where the adjoint path uses
  matrix-free or operator-only execution
- parameter, auxiliary, and geometry sensitivity hooks when those variables are
  advertised as differentiable
- state replay/checkpoint hooks when the contribution owns material, global, or
  auxiliary history state in transient adjoints
- Analysis metadata equivalent to Forms-installed contributions
- finite-difference or dot-product verification tests for the advertised
  support row

If any of those pieces are missing, the readiness report should reject the
custom contribution for adjoint use and identify the missing contract.

### 10. Inner Products and Weights Must Be Declared

Every transpose or adjointness claim must state the inner product under which it
is valid. This is especially important for distributed vectors, interface
transfers, mortar projections, conservative transfers, and marker gather/spread
operators.

Required metadata:

- source and target vector ownership and ghost/overlap policy
- source and target component layout
- source and target DOF/block layout, including `BlockDofMap`,
  `GhostDofManager`, and distributed sparsity row/column ownership conventions
  where applicable
- source and target measure, mass, marker, or quadrature weights
- frame-transform policy and any pass-through components
- MPI reduction semantics for dot-product tests
- whether the transpose is algebraic Euclidean transpose, mass-weighted adjoint,
  conservative-transfer adjoint, or another explicitly named convention

The standard verification pattern is:

```text
dot_target(B x, y; W_target) == dot_source(x, B^* y; W_source)
```

where `B^*` is the declared adjoint under the documented weights. A plain
matrix transpose is only valid when the declared source and target inner
products are Euclidean and the vector ownership semantics match.

### 11. Nonsmooth and Topology-Changing Behavior Fails Closed

Adjoint preflight must reject unsupported nonsmooth regimes by default. Examples
include:

- contact active-set changes
- limiter activation changes
- event-triggered auxiliary state changes
- nonsmooth auxiliary policies from `AuxiliaryNonsmoothPolicy` unless a scoped
  derivative policy is declared
- remeshing, repartitioning, or topology changes inside the differentiated
  window
- cut-cell topology changes without a qualified sensitivity policy
- discontinuous material laws or table lookups without derivative metadata
- opaque CAD or meshing callbacks without explicit derivative or finite-
  difference fallback declarations
- contact, cut-integration, moving-constraint, and adaptivity-transfer
  invalidation paths without replayable revision/provenance metadata

Support can be advertised only for a scoped regime where the module declares the
active-set, event, topology, or fallback policy and the gradient is verified for
that regime.

## Recommended Delivery Strategy

### Milestone 0: Adjoint-Ready Metadata and Preflight

Before exposing an adjoint API, add a public preflight layer that can inspect an
installed problem and report whether the requested adjoint path is supported.
This milestone should:

- consume Systems, Analysis, FE/Coupling, MovingMesh, and Forms-install
  metadata
- report missing transpose backends, missing objective gradients, unsupported
  constraints, unsupported transfer-adjoint paths, missing geometry-sensitive
  derivatives, and missing replay/checkpoint state
- report unsupported dynamic constraint dependencies, missing constraint tangent
  policies, unsupported global kernels, unsupported constitutive tangent/state
  contracts, and value-only QoI registrations that lack gradient support
- preserve contribution names, origins, operator tags, domain provenance,
  `analysis::VariableKey` identities, and owning-system provenance in diagnostics
- avoid private Forms AST traversal

### Milestone A: Minimum Viable Steady Adjoint

Deliver the smallest complete steady adjoint path first:

- assembled Jacobian already exists
- backend can solve `J^T lambda = rhs`
- objective gradient `dQ/du` can be assembled for an adjoint-ready FE
  objective/QoI adapter
- constraints are handled consistently
- the preflight report certifies the problem as steady assembled-adjoint ready
- tests verify the adjoint gradient against finite differences

This milestone should exclude:

- transient reverse-time logic
- shape derivatives
- partitioned coupling replay
- matrix-free transpose kernels unless they are cheap to add

### Milestone B: Production Steady Adjoint

Extend the minimum path with:

- PDE-plus-auxiliary couplings
- monolithic multiphysics couplings installed into one `FESystem`
- transfer-adjoint contracts for interface and coupling transfer operators
- more general objective types
- backend coverage beyond one reference backend
- better diagnostics and metadata

### Milestone C: Moving-Domain and Design Sensitivity Readiness

After the basic steady path is stable, add the infrastructure needed for
moving-domain and design workflows:

- mesh-displacement unknown adjoints for coupled monolithic ALE
- geometry-sensitive objective and residual derivative checks
- current-coordinate, current-measure, normal, face-measure, and
  mesh-velocity derivative provenance
- geometry revision, rollback, and interface-map provenance in adjoint
  diagnostics
- a separate shape/CAD design-sensitivity design note before broad
  CAD-surface support is advertised

### Milestone D: Transient and Partitioned Coupled Adjoint

Only after steady adjoint is stable should the library add:

- checkpoint or replay infrastructure
- reverse-time marching
- discrete adjoint support for BDF, generalized-alpha, Newmark, and collocation paths
- reverse traversal of partitioned coupling exchange graphs
- replay of endpoint temporal slots, accepted/trial interface maps, transfer
  state, and moving-domain transaction state

## Phase 0: Add an Adjoint-Ready Metadata and Preflight Contract

### Why

The adjoint module should only solve problems whose discrete residual,
objective, constraints, transfer operators, geometry state, coupling graph, and
time-history state are visible through public metadata. Without that preflight
contract, the adjoint path would need to guess about installed physics and would
eventually duplicate Systems, Analysis, FE/Coupling, and MovingMesh logic.

### Recommended Design

Add an adjoint preflight service at the `Systems` or `Analysis` boundary that
returns a structured readiness report.

Recommended public concepts:

- `AdjointReadinessRequest`
- `AdjointReadinessReport`
- `AdjointCapabilityIssue`
- `AdjointMetadataView`
- `AdjointProblemFingerprint`

`AdjointProblemFingerprint` should be mandatory for every positive readiness
report and every adjoint solve. It records the exact discrete problem being
differentiated, not just a loose operator name.

The report should include:

- requested objective/QoI tag and target fields
- residual tag, operator tag, linearization state, and accepted primal state
  identity
- state/time point, including `SystemStateView` time, `dt`, `effective_dt`,
  `dt_prev`, transient history identity, and time-integration context identity
- `FESystem::operatorRevisionSnapshot()`, `systemLayoutRevision()`,
  `FELayoutRevisionState`, constraint revision/layout state, geometry
  transaction state, mesh revision keys, and active coordinate configuration
- `SetupStoragePlan` requirements and the setup-storage revision or summary that
  prove required topology, GID, global-lookup, entity-DOF, point-search, and
  provenance storage was retained before `FESystem::setup()`
- `DofPermutation`, `BlockDofMap`, sparsity/distributed sparsity ownership, and
  backend vector-layout identity needed to interpret stored vectors
- matrix-free, partial-assembly, backend cache, native update, bordered update,
  nullspace, preconditioner, and constraint-transform state that affects the
  operator applied or solved
- contribution names, origins, owning systems, and domain provenance
- field and non-field dependencies using `analysis::VariableKey` where
  applicable
- objective-gradient availability
- backend normal/transpose apply and solve support
- backend transpose capability flags for assembled, matrix-free, partial
  assembly/device, preconditioner, and reduced/grouped update paths
- constraint policy and reduced/full-space compatibility
- constraint dependency and tangent-policy compatibility
- auxiliary and coupling block coverage
- FE-backed quantity capability coverage for objective/QoI requests
- global-kernel metadata, tangent/transpose, sparsity, and replay coverage
- constitutive tangent contract, material-state, update-frame, and history-policy
  coverage
- transfer normal/transpose availability and dot-product test status
- transfer adjointness summary coverage, including declared source/target inner
  products, weights, component layout, frame transform, ownership, and reduction
  semantics
- moving-domain coordinate configuration, geometry revision, and
  geometry-sensitive derivative availability
- moving-domain transaction, contact/cut/adaptivity invalidation, and
  moving-constraint replay provenance
- transient checkpoint/replay availability when requested

### Concrete Files to Modify

- `Code/Source/solver/FE/Analysis/*`
- `Code/Source/solver/FE/CMakeLists.txt`
- `Code/Source/solver/FE/Systems/FESystem.*`
- `Code/Source/solver/FE/Systems/SetupStoragePlan.*`
- `Code/Source/solver/FE/Systems/SystemSetup.*`
- `Code/Source/solver/FE/Systems/OperatorBackends.*`
- `Code/Source/solver/FE/Systems/FormsInstaller.*`
- `Code/Source/solver/FE/Systems/FEQuantityRegistry.*`
- `Code/Source/solver/FE/Systems/FEQuantityDefinition.h`
- `Code/Source/solver/FE/Systems/GlobalKernel.*`
- `Code/Source/solver/FE/Systems/GlobalKernelStateProvider.*`
- `Code/Source/solver/FE/Constraints/ConstraintDependency.*`
- `Code/Source/solver/FE/Constraints/MovingConstraintComposition.*`
- `Code/Source/solver/FE/Constitutive/*`
- `Code/Source/solver/FE/Coupling/*`
- `Code/Source/solver/FE/MovingMesh/*`
- `Code/Source/solver/FE/Dofs/BlockDofMap.*`
- `Code/Source/solver/FE/Dofs/GhostDofManager.*`
- `Code/Source/solver/FE/Sparsity/DistributedSparsityPattern.*`
- likely new files:
  - `Code/Source/solver/FE/Adjoint/AdjointTypes.h`
  - `Code/Source/solver/FE/Adjoint/AdjointReadiness.h`
  - `Code/Source/solver/FE/Adjoint/AdjointReadiness.cpp`
  - `Code/Source/solver/FE/Adjoint/AdjointDiagnostics.h`
  - `Code/Source/solver/FE/Adjoint/AdjointDiagnostics.cpp`
- tests under:
  - `Code/Source/solver/FE/Tests/Unit/Analysis/`
  - `Code/Source/solver/FE/Tests/Unit/Systems/`
  - `Code/Source/solver/FE/Tests/Unit/Adjoint/`

### Concrete Steps

1. Define the readiness request and report types.
2. Reuse or extend `ContributionDescriptor`, `FormulationRecord`,
   `FormStructureSummary`, and related Analysis summaries instead of creating a
   parallel adjoint contribution IR. Treat those summaries as diagnostics and
   evidence; do not make optional Analysis evidence the sole readiness contract
   for a solve.
3. Add a public Forms/Systems metadata bridge, or consume the one created for
   FE/Coupling, so installed residuals report:
   - field uses
   - non-field dependency provenance
   - temporal-symbol provenance
   - geometry-terminal provenance
   - geometry-sensitivity options and structured provenance
   - installed block/domain provenance
4. Define `AdjointProblemFingerprint` and require it to compare against current
   `FESystem::operatorRevisionSnapshot()`, `operatorInvalidationDecision()`,
   setup state, `SetupStoragePlan` revision or summary, geometry transaction
   state, layout revisions, constraint revisions, DOF permutation, and backend
   operator-cache revisions before any adjoint solve or gradient evaluation.
5. Define `AdjointStorageRequirements` or an equivalent request layer that can
   be merged into `SetupStoragePlan` before setup. Readiness must fail if an
   adjoint request needs topology, global lookup, entity-DOF maps, point-search
   provenance, transfer provenance, or replay data that was not retained by the
   completed setup.
6. Add capability checks for backend transpose support, objective-gradient
   support, constraint policy, transfer adjointness, moving-domain derivative
   support, and transient replay state.
7. Add capability checks for existing FE quantity/QoI registrations so
   value-only quantities, post-processing-only derived results, and unsupported
   point/global objectives fail before solve time.
8. Add checks for dynamic constraints using `ConstraintDependencyDeclaration`,
   `ConstraintTangentPolicy`, and moving-constraint composition metadata.
9. Add checks for global kernels and constitutive/material contracts, including
   tangent, transpose, state, history, and replay support.
10. Add checks for FE/Coupling transfer-adjoint contracts by extending or
    consuming public coupling transfer records and Analysis transfer summaries.
11. Add checks for adjoint-ready formulation contract violations, including
    missing custom-kernel metadata and unsupported nonsmooth or topology-changing
    behavior.
12. Make unsupported cases fail with actionable diagnostics before an adjoint
   solve is attempted.
13. Add tests for both positive and negative readiness reports.

### Acceptance Criteria

- A caller can ask whether a steady assembled adjoint is supported before
  calling `solveAdjoint(...)`.
- The readiness report identifies missing metadata, missing transpose support,
  missing objective gradients, unsupported constraints, unsupported transfers,
  and unsupported moving-domain replay paths.
- A positive readiness report includes a valid `AdjointProblemFingerprint`, and
  changing setup state, layout, geometry, constraint, DOF permutation, backend
  cache, or operator revision metadata invalidates that fingerprint before solve
  time.
- A positive readiness report proves that the completed setup retained every
  storage artifact required by the requested adjoint path, or it fails with a
  setup-storage diagnostic that names the missing `SetupStoragePlan` requirement.
- Diagnostics preserve contribution name, origin, operator tag, owning-system
  provenance, domain kind, and dependency identities.
- Diagnostics preserve FE quantity identity, shape, referenced fields,
  value/gradient capability, and any post-processing promotion status for
  objective/QoI requests.
- Dynamic constraints without a supported tangent policy are rejected with a
  stable capability issue.
- Global kernels and constitutive laws without required adjoint metadata or
  replay hooks are rejected with stable capability issues.
- Missing Analysis summaries may weaken diagnostics, but readiness is determined
  from explicit subsystem capability contracts unless a requested support row
  declares a particular Analysis summary as mandatory.
- Custom kernels without the required tangent, sensitivity, transpose, or
  Analysis metadata are rejected with a stable capability issue.
- Nonsmooth or topology-changing regimes without a declared derivative policy
  are rejected with a stable capability issue.
- No readiness check relies on private Forms AST traversal.

## Phase 1: Add Transpose-Capable Linear Algebra Contracts

### Why

This is the primary missing foundation. Without it, there is no clean way for the FE layer to request an adjoint solve from any backend.

Today:

- `Backends/Interfaces/LinearSolver.h` exposes only `solve(A, x, b)`
- `Backends/Interfaces/GenericMatrix.h` exposes only forward `mult(...)`
- `Assembly/MatrixFreeAssembler.h` exposes only forward `apply(...)`
- `Assembly/DeviceAssembler.h` exposes only forward partial-assembly/device
  `apply(...)`

### Recommended Design

Add explicit forward-vs-transpose operation modes to the backend-neutral interfaces.

Recommended public types:

- `enum class OperatorApplyMode { Normal, Transpose };`
- `enum class LinearSolveMode { Normal, Transpose };`

Recommended public interface changes:

- `GenericMatrix`
  - add `multTranspose(...)`
  - add `multAddTranspose(...)`
- `IMatrixFreeKernel`
  - add local transpose action hooks or a capability query that rejects
    transpose use when unavailable
- `MatrixFreeOperator`
  - add `applyTranspose(...)`
  - add `applyTransposeAdd(...)`
- `DeviceKernel` and partial-assembly paths
  - add transpose hooks only for kernels that explicitly advertise support
- `LinearSolver`
  - either add `solve(..., LinearSolveMode mode)`
  - or add explicit `solveTranspose(...)`

The preferred design is a mode enum, because it avoids duplicating every solve entry point.

Readiness diagnostics should distinguish:

- transpose matrix apply support
- transpose matrix-free apply support
- transpose device/partial-assembly apply support
- transpose solve support
- transpose preconditioner behavior
- transpose handling of rank-one, reduced-field, grouped-bordered, and
  constraint-transformed updates

### Concrete Files to Modify

- `Code/Source/solver/FE/Backends/Interfaces/GenericMatrix.h`
- `Code/Source/solver/FE/Backends/Interfaces/GenericVector.h`
- `Code/Source/solver/FE/Backends/Interfaces/LinearSolver.h`
- `Code/Source/solver/FE/Backends/Utils/BackendOptions.h`
- `Code/Source/solver/FE/Assembly/MatrixFreeAssembler.h`
- `Code/Source/solver/FE/Assembly/MatrixFreeAssembler.cpp`
- `Code/Source/solver/FE/Assembly/DeviceAssembler.h`
- `Code/Source/solver/FE/Assembly/DeviceAssembler.cpp`
- `Code/Source/solver/FE/Backends/Eigen/EigenLinearSolver.*`
- `Code/Source/solver/FE/Backends/PETSc/PetscLinearSolver.*`
- `Code/Source/solver/FE/Backends/Trilinos/TrilinosLinearSolver.*`
- `Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.*`
- backend matrix classes under `Backends/*/*Matrix.*`
- tests under `Code/Source/solver/FE/Tests/Unit/Backends/`

### Concrete Steps

1. Add the public mode enums in a backend-neutral header.
2. Extend `GenericMatrix` with transpose-apply methods.
3. Extend `IMatrixFreeKernel` and `MatrixFreeOperator` with transpose-apply
   methods or explicit unsupported capability diagnostics.
4. Extend `DeviceKernel` and partial-assembly paths with opt-in transpose
   contracts where matrix-free/device adjoints are advertised.
5. Extend `LinearSolver` with transpose-solve support.
6. Implement assembled transpose support in staged backend waves:
   - one reference assembled backend first, preferably the smallest backend with
     dense/sparse reference tests
   - PETSc and Trilinos only where their matrix, preconditioner, and solver
     wrappers explicitly advertise compatible transpose semantics
   - FSILS only after transpose semantics are defined for native rank-one
     updates, reduced-field updates, grouped-bordered updates, Dirichlet/native
     face handling, preconditioner behavior, and nullspace handling
7. For matrix-free and device operators:
   - add the interface now
   - allow runtime "not implemented" for kernels that do not yet supply transpose action
8. Extend backend options only if a backend needs separate transpose-specific defaults or diagnostics.
9. Add unit tests:
   - compare `A^T x` against explicit dense transpose on small systems
   - compare transpose solve result against solving an explicitly transposed dense matrix
   - verify advertised transpose-capable backends follow the same semantics
   - verify non-advertising backends fail readiness with stable diagnostics
   - verify unsupported native update, preconditioner, nullspace, or
     constraint-transformed transpose combinations fail readiness before solve
     time

### Acceptance Criteria

- FE can ask every backend that advertises assembled-adjoint support for
  `solve(..., Transpose)`.
- Backends that do not advertise transpose support are rejected by readiness
  checks before solve time.
- Backend support is feature-scoped: a backend can advertise plain assembled
  transpose support while still rejecting transpose solves that require
  unsupported native updates, preconditioner behavior, nullspace treatment, or
  constraint-transformed operators.
- `GenericMatrix` transpose apply is tested against dense reference matrices.
- transpose solve is covered in the backend conformance suite.

## Phase 2: Add a First-Class Adjoint API in FE/Systems

### Why

Even after backends support transpose solves, the FE layer still needs a public system-level way to define and solve adjoints.

Today the library has:

- forward operator assembly
- functional evaluation
- some local gradient pieces

but no object model for:

- the adjoint right-hand side
- the target field or fields
- the choice of forward vs transpose solve
- result packaging and diagnostic reporting

### Recommended Design

Add a small, explicit `Systems`-level adjoint surface.

Recommended new concepts:

- `AdjointObjectiveTag`
- `AdjointRequest`
- `AdjointResult`
- `AdjointProblemFingerprint`
- `AdjointLinearizationHandle`
- optional `AdjointWorkspace`
- optional `AdjointProblem` or `AdjointSystem` helper

Recommended responsibilities:

- identify and validate the exact accepted discrete Jacobian for the primal state
  being differentiated
- reject stale, lagged, or modified forward operators unless their semantics are
  explicitly captured in `AdjointProblemFingerprint`
- assemble the objective gradient `dQ/du`
- map through constraints if required
- call the backend transpose solve
- return the adjoint vector and diagnostic metadata

The first implementation should not silently reuse whatever matrix is currently
left in a Newton workspace. Modified Newton lagged Jacobians, pseudo-transient
continuation terms, line-search accepted states, Jacobian rebuild periods, native
backend updates, and cached constraint transforms all affect the actual discrete
operator. The adjoint API must either prove that the stored Jacobian corresponds
to the accepted primal residual state or reassemble the Jacobian for the recorded
fingerprint before solving.

### Concrete Files to Modify

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/SetupStoragePlan.*`
- `Code/Source/solver/FE/Systems/SystemAssembly.*`
- `Code/Source/solver/FE/Systems/OperatorRegistry.*`
- `Code/Source/solver/FE/Systems/OperatorBackends.h`
- `Code/Source/solver/FE/Systems/OperatorBackends.cpp`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.*`
- `Code/Source/solver/FE/CMakeLists.txt`
- likely a new file pair:
  - `Code/Source/solver/FE/Adjoint/AdjointProblem.h`
  - `Code/Source/solver/FE/Adjoint/AdjointProblem.cpp`
  - `Code/Source/solver/FE/Adjoint/AdjointSolver.h`
  - `Code/Source/solver/FE/Adjoint/AdjointSolver.cpp`
- optional Systems-facing bridge:
  - `Code/Source/solver/FE/Systems/AdjointSystem.h`
  - `Code/Source/solver/FE/Systems/AdjointSystem.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Adjoint/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`

### Concrete Steps

1. Define the minimal adjoint request structure:
   - required `AdjointProblemFingerprint`, or enough accepted-forward-solve
     metadata to construct one deterministically
   - primal residual tag
   - primal operator tag
   - linearization state
   - accepted state/time identity
   - objective tag resolved through the adjoint objective/QoI adapters
   - target field or fields
   - constraint policy
2. Define the adjoint result structure:
   - converged flag
   - linear solver report
   - adjoint vector
   - optional reduced-space diagnostics
3. Add an FE entry point such as:
   - `solveAdjoint(...)`
   - or `assembleAdjointRhs(...)` plus a separate solve helper
4. Add a linearization validation step that compares the request fingerprint with
   current `FESystem` setup, operator revision, geometry transaction, constraint,
   setup-storage summary, DOF layout, backend cache, and native-update metadata.
5. Require the steady MVP to solve against the exact discrete Jacobian for the
   accepted primal state. If the forward solve used a lagged Jacobian, PTC
   modification, line-search state change, or backend-local native update that is
   not represented in the fingerprint, readiness must fail or the operator must
   be reassembled.
6. Keep the first implementation matrix-based and steady-state only.
7. Ensure the API can later accept:
   - matrix-free operator handles
   - transient checkpoint context
   - parameter or control derivative requests
8. Add unit tests for:
   - Laplace or Poisson objective gradient
   - mixed Stokes-like system with known block structure
   - multi-field objective acting on one field only
   - stale fingerprint rejection after setup/layout/geometry/constraint changes
   - rejection or reassembly when a stored Newton Jacobian no longer represents
     the accepted primal state

### Acceptance Criteria

- A caller can request a steady adjoint solve through `FE/Systems`.
- The result includes both the adjoint vector and the solver report.
- The API is backend-neutral and does not leak backend types.
- The adjoint solve validates that the operator, state, constraints, geometry,
  setup storage, DOF layout, backend updates, and cache revisions match the recorded
  `AdjointProblemFingerprint`.
- Modified Newton, PTC, line-search, stale workspace, and backend-native-update
  cases either produce a matching fingerprint, trigger reassembly, or fail
  readiness before solving.
- Core adjoint objects live under `FE/Adjoint`; `FE/Systems` owns only the
  orchestration entry points and metadata hooks.

## Phase 3: Add Adjoint Objective and QoI Adapters

### Why

Current objective-gradient support is too narrow. The existing public path is
concentrated around boundary/domain reduction services, FE-backed auxiliary
quantities, scalar functional evaluation, and post-processing definitions. Those
pieces should be unified for adjoint use without creating a second FE quantity
registry.

Adjoint workflows need a generic FE service for:

- scalar domain objectives
- boundary objectives
- interface objectives
- point or sampled objectives
- global functional objectives
- multi-field objectives

### Recommended Design

Introduce `FE/Adjoint` objective/QoI adapters over existing Systems, Assembly,
Forms, and PostProcessing infrastructure.

Recommended public concepts:

- `AdjointObjectiveDefinition`
- `AdjointObjectiveTag`
- `AdjointObjectiveSource`
- `AdjointObjectiveCapability`
- `AdjointQuantityCapabilities`
- `AdjointObjectiveView`
- `assembleObjectiveGradient(...)`
- `evaluateObjective(...)`

This should reuse existing machinery where possible:

- reuse `FEQuantityRegistry` and `FEQuantityDefinition` for FE-backed quantity
  metadata, shape, referenced fields, region/marker metadata, and
  monolithic-linearization capability, but do not treat the current
  `FEQuantityCapabilities` flags as sufficient for adjoint readiness
- reuse `BoundaryReductionService` for boundary and domain integrals
- reuse `FunctionalKernel` and `OperatorBackends` for objective evaluation
- reuse `Forms/PointEvaluator.*` for sampled scalar expression values and only
  the limited auxiliary-seed derivative paths it actually supports
- allow `PostProcessing/DerivedResultRegistry.*` definitions to be referenced
  only when they are explicitly promoted to adjoint-ready QoIs with derivative
  contracts
- add gradient-capable objective kernels where a plain `FunctionalKernel` is not sufficient
- avoid adding a parallel `ObjectiveRegistry` that duplicates
  `FEQuantityRegistry`, `OperatorBackends`, or `DerivedResultRegistry`

The adjoint objective layer should add either an `AdjointQuantityCapabilities`
overlay or an explicit extension of `FEQuantityCapabilities`. Required adjoint
capabilities include value evaluation, state-gradient support, parameter-gradient
support, geometry-gradient support, target-field filtering, sampled/point
gradient support, and derivative ownership/provenance. The current FE quantity
flags are useful input metadata, not a complete adjoint contract.

Point and sampled-field objectives need their own gradient contract. Value
evaluation through `PointEvaluator` is not enough; an adjoint-ready point QoI
must also expose interpolation/search provenance, basis-value support for field
gradients, geometry derivative policy for moving domains, and readiness
diagnostics when any of those pieces are unavailable.

### Concrete Files to Modify

- `Code/Source/solver/FE/CMakeLists.txt`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/OperatorBackends.h`
- `Code/Source/solver/FE/Systems/OperatorBackends.cpp`
- `Code/Source/solver/FE/Systems/FEQuantityRegistry.*`
- `Code/Source/solver/FE/Systems/FEQuantityDefinition.h`
- `Code/Source/solver/FE/Systems/BoundaryReductionService.*`
- `Code/Source/solver/FE/Assembly/FunctionalAssembler.*`
- `Code/Source/solver/FE/Forms/FormKernels.*`
- `Code/Source/solver/FE/Forms/PointEvaluator.*`
- `Code/Source/solver/FE/PostProcessing/DerivedResultRegistry.*`
- likely new files:
  - `Code/Source/solver/FE/Adjoint/AdjointObjective.h`
  - `Code/Source/solver/FE/Adjoint/AdjointObjective.cpp`
  - `Code/Source/solver/FE/Adjoint/AdjointGradient.h`
  - `Code/Source/solver/FE/Adjoint/AdjointGradient.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Adjoint/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`
- tests under `Code/Source/solver/FE/Tests/Unit/Forms/`

### Concrete Steps

1. Define the objective source model:
   - FE-backed quantity from `FEQuantityRegistry`
   - registered scalar functional from `OperatorBackends`
   - boundary/domain/region functional from `BoundaryReductionService`
   - sampled point or expression from `PointEvaluator`
   - explicitly promoted derived result from `DerivedResultRegistry`
2. Define what kinds of objectives are first-class:
   - domain integral
   - boundary integral
   - interface integral
   - sampled point or sampled field value
   - global kernel objective
3. Add an adapter lookup keyed by objective tag that resolves to existing FE
   metadata rather than duplicating ownership.
4. Add objective capability queries:
   - value evaluation
   - state-gradient support
   - parameter-gradient support
   - geometry-gradient support
   - multi-field target-field filtering
   - sampled/point-gradient support and interpolation/search provenance
5. Add the adjoint capability overlay or extend `FEQuantityCapabilities` so
   readiness checks do not infer adjoint support from value-only or
   monolithic-linearization-only metadata.
6. Add a point-objective gradient contract that records:
   - physical and reference point provenance
   - element/search result identity and validity revision
   - interpolation basis and component layout
   - moving-domain geometry derivative policy
   - unsupported derivative diagnostics for value-only point expressions
7. Add a generic gradient assembly entry point:
   - `assembleObjectiveGradient(tag, field, state, options)`
8. Generalize the current boundary-gradient logic so it can serve:
   - arbitrary registered objectives
   - multiple target fields
   - multi-field objectives
9. Add a shared sparse gradient result type for:
   - `(GlobalIndex, value)` entries
   - optional block or field metadata
10. Add interface and point-objective support incrementally after domain and boundary objectives are stable.
11. Add tests:
   - objective value matches reference integral
   - objective gradient matches finite differences
   - multi-field objective gradients act only on the chosen field when requested
   - value-only quantities and post-processing-only derived results fail
     readiness unless explicitly promoted with derivative support
   - point/sampled objectives without interpolation/search derivative provenance
     fail readiness before solve time

### Acceptance Criteria

- `FE/Systems` can evaluate and differentiate an adjoint-ready objective
  through one public API.
- The objective/QoI surface reuses existing FE registries and services instead
  of introducing a parallel quantity registry.
- Boundary-reduction gradients are no longer a special-case-only path.
- Value-only and post-processing-only quantities produce actionable readiness
  diagnostics when requested as adjoint objectives.
- Existing FE quantity registrations are considered adjoint-ready only through an
  explicit adjoint capability overlay or extended capability contract.
- Point/sampled objectives are supported only when value evaluation and gradient
  provenance are both available; otherwise readiness fails before solve time.
- Objective gradients are verified by finite differences in unit tests.

## Phase 4: Make Constraint Handling Adjoint-Consistent

### Why

The constraints layer already has the local reduced-space transpose algebra, but
the forward solve path and the adjoint solve path need a common, explicit policy
that also covers backend-owned vectors, distributed ownership, block layouts,
DOF permutations, sparsity, preconditioners, and nullspaces.

Without that, adjoint behavior around:

- strong Dirichlet elimination
- periodic constraints
- multipoint constraints
- gauge conditions
- auxiliary-driven constraints
- time-dependent or geometry-dependent constraints
- moving/sliding/tied/contact-candidate constraints

will be ambiguous or wrong.

### Recommended Design

Adopt reduced-space adjoint semantics as a first-class option:

- primal reduced system: `P^T A P z = P^T (b - A c)`
- adjoint reduced system: `(P^T A P)^T lambda_r = dQ/dz`
- full adjoint recovery through `P lambda_r` when needed for post-processing

Even if the forward path continues using assembly-time elimination in some cases, the FE layer should expose a consistent constraint transform service for adjoints.

That service should not be limited to span-based local arrays. It should define
how `P`, `P^T`, and `P^T A P` are applied to the same vector and matrix
representations used by the selected FE backend, including owned/ghost entries,
global-to-local numbering, block offsets, reduced sparsity ownership, backend
vector layout revisions, and any nullspace or preconditioner metadata attached
to the constrained solve.

For constraints whose values or structure depend on time, geometry, topology,
auxiliary state, interface maps, or moving-domain transactions, the readiness
layer must also consume:

- `ConstraintDependencyDeclaration`
- `ConstraintRevisionSnapshot`
- `ConstraintTangentPolicy`
- `MovingConstraintCompositionResult`
- moving-constraint revision and conflict metadata

The adjoint path should reject constraints that affect the differentiated
problem but do not declare how their value derivatives enter the reduced
operator or objective-gradient mapping.

### Concrete Files to Modify

- `Code/Source/solver/FE/CMakeLists.txt`
- `Code/Source/solver/FE/Constraints/ConstraintTransform.*`
- `Code/Source/solver/FE/Constraints/AffineConstraints.*`
- `Code/Source/solver/FE/Constraints/ConstraintDependency.*`
- `Code/Source/solver/FE/Constraints/MovingConstraintComposition.*`
- `Code/Source/solver/FE/Constraints/AuxiliaryDrivenDirichletConstraint.*`
- `Code/Source/solver/FE/Constraints/SystemConstraints.*`
- `Code/Source/solver/FE/Systems/FESystem.*`
- `Code/Source/solver/FE/Backends/Interfaces/GenericVector.h`
- `Code/Source/solver/FE/Backends/Interfaces/DofPermutation.h`
- `Code/Source/solver/FE/Dofs/BlockDofMap.*`
- `Code/Source/solver/FE/Dofs/GhostDofManager.*`
- `Code/Source/solver/FE/Sparsity/DistributedSparsityPattern.*`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.*`
- possibly `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/Adjoint/`
- tests under `Code/Source/solver/FE/Tests/Unit/Constraints/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`

### Concrete Steps

1. Decide and document the official adjoint constraint policy.
2. Add a `Systems` helper that constructs or caches a `ConstraintTransform` when needed.
3. Add backend-aware constraint-transform adapters that can operate on the
   selected backend vector representation without losing ownership, ghost, block,
   permutation, or vector-layout metadata.
4. Add helpers to:
   - restrict objective gradients with `P^T`
   - solve in reduced coordinates
   - expand reduced adjoints back to full coordinates
5. Ensure inhomogeneous constraints are handled correctly through the `c` vector already represented by `ConstraintTransform`.
6. Define reduced sparsity, preconditioner, nullspace, and block-layout semantics
   for constrained adjoint solves, including ownership and MPI reduction rules
   for distributed runs.
7. Add readiness checks for constraints with nontrivial dependency declarations:
   - geometry-dependent values
   - time-dependent values
   - auxiliary-driven values
   - moving/sliding/tied interface maps
   - topology, ownership, numbering, label, and layout revisions
8. Require `ConstraintTangentPolicy::Analytic` or another explicit supported
   tangent/transpose policy before differentiating value-dependent constraints.
9. Add adjoint-specific tests for:
   - homogeneous Dirichlet elimination
   - inhomogeneous Dirichlet conditions
   - periodic constraints
   - multipoint constraints
   - gauge or nullspace handling
   - auxiliary-driven Dirichlet constraints
   - geometry-dependent moving constraints
   - unsupported constraint tangent policies failing readiness

### Acceptance Criteria

- The library has one documented adjoint policy for algebraic constraints.
- Reduced-space and full-space adjoint vectors are related in a tested, explicit way.
- Constraint transforms operate on the advertised backend vector path with
  tested ownership, ghost update, block-layout, DOF-permutation, reduced
  sparsity, nullspace, and preconditioner semantics.
- Dynamic constraints are either differentiated through a declared tangent
  policy or rejected by preflight.
- Constraint-heavy cases pass FD gradient verification.

## Phase 5: Add Coupled Auxiliary and Multiphysics Adjoint Infrastructure

### Why

The FE stack already has significant forward coupling infrastructure for
auxiliary states and direct coupled inputs, and the coupling-module plan adds a
current and planned FE/Coupling layer for monolithic and partitioned multiphysics
workflows. The adjoint plan must use and extend those public contracts instead
of inventing its own coupled-problem model.

This matters for:

- outlet or boundary models
- monolithic auxiliary states
- partitioned auxiliary states
- any objective that depends on auxiliary outputs or FE-coupled inputs
- monolithic multiphysics couplings installed into one shared `FESystem`
- partitioned multiphysics exchange graphs across multiple `FESystem` instances
- global kernels that represent non-element-local coupled terms, contact/search
  terms, or other globally coupled operator contributions

### Recommended Design

Build on the existing forward sensitivity artifacts rather than replacing them.

Key existing pieces worth reusing:

- `FESystem::BorderedCouplingData`
- `DirectCouplingRecord`
- `CoupledResidualSensitivityKernel`
- `differentiateWrtAuxiliaryOutput(...)`
- `BoundaryFunctionalGradientKernel`
- `GlobalKernel` and `GlobalKernelStateProvider`
- FE/Coupling participant, endpoint, graph, transfer, temporal, and geometry
  provenance records
- `Systems/InterfaceOperators.*` accepted/trial revision and interface-map
  provenance
- Analysis `ContributionDescriptor`, `analysis::VariableKey`, and block/domain
  metadata

Recommended new FE concepts:

- `AuxiliaryAdjointData`
- `CoupledAdjointAssemblyPlan`
- helpers for transpose bordered-system operations
- `MonolithicCoupledAdjointPlan`
- `PartitionedCoupledAdjointReplayPlan`
- `TransferAdjointDescriptor`
- `TransferInnerProductDescriptor`
- `WeightedAdjointnessReport`
- `TransferAdjointnessSummary`, or an equivalent extension to
  `Analysis::TransferOperatorSummary`

### Concrete Files to Modify

- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
- `Code/Source/solver/FE/Forms/FormKernels.*`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.*`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.*`
- `Code/Source/solver/FE/Systems/GlobalKernel.*`
- `Code/Source/solver/FE/Systems/GlobalKernelStateProvider.*`
- `Code/Source/solver/FE/Coupling/*`
- `Code/Source/solver/FE/Analysis/*`
- `Code/Source/solver/FE/Systems/InterfaceOperators.*`
- `Code/Source/solver/FE/Dofs/BlockDofMap.*`
- `Code/Source/solver/FE/Dofs/GhostDofManager.*`
- `Code/Source/solver/FE/Sparsity/DistributedSparsityPattern.*`
- `Code/Source/solver/FE/CMakeLists.txt`
- tests under `Code/Source/solver/FE/Tests/Unit/Adjoint/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`
- tests under `Code/Source/solver/FE/Tests/Unit/Auxiliary/`
- tests under `Code/Source/solver/FE/Tests/Unit/Coupling/`

### Concrete Steps

1. Document the block structure of the coupled adjoint system for:
   - PDE-only
   - PDE plus partitioned auxiliary states
   - PDE plus monolithic auxiliary states
   - monolithic multiphysics Forms contributions in one `FESystem`
   - partitioned exchange graphs across multiple `FESystem` instances
   - global-kernel coupled contributions
2. Add a data structure that can assemble the transpose of the existing coupled
   Jacobian contributions.
3. Reuse `differentiateWrtAuxiliaryOutput(...)` for `dR_pde/dx_aux` terms where possible.
4. Add assembly or operator hooks for the transpose bordered coupling terms.
5. Define how auxiliary-only adjoint unknowns are stored and updated.
6. For global kernels:
   - require `analysisContributions()` metadata for block/domain provenance
   - require tangent/transpose hooks for advertised adjoint paths
   - preserve state-provider replay requirements for transient adjoints
   - verify global-kernel transpose blocks against finite differences or
     dot-product tests
7. For monolithic FE/Coupling contributions:
   - consume installed dependency and block metadata from the public
     Forms/Systems coupling-analysis metadata bridge
   - preserve contribution name, origin, owning system, operator tag, domain,
     and `analysis::VariableKey` provenance
   - verify off-diagonal adjoint blocks against finite differences
8. For partitioned FE/Coupling plans:
   - reverse the exchange graph without choosing a partitioned driver algorithm
   - require explicit transpose/adjoint apply for every transfer operator
   - preserve endpoint temporal slots, interface-map provenance, source/target
     system identity, frame-transform metadata, and accepted/trial revision keys
   - attach adjoint descriptors to public coupling transfer records such as
     `ResolvedCouplingTransfer`, `CouplingDriverOwnedTransferDescriptor`, or
     their successors, rather than a parallel adjoint-only transfer registry
9. Define inner-product and weight conventions for each transfer-adjoint
   descriptor:
   - Euclidean algebraic transpose
   - mass-weighted adjoint
   - marker-measure-weighted adjoint
   - conservative-transfer adjoint
   - driver-owned declared convention
10. Define DOF/block/vector ownership conventions for transfer-adjoint tests:
   - `BlockDofMap` block identity and offsets
   - owned vs ghost vector semantics from `GhostDofManager`
   - distributed sparsity row/column ownership and reductions
11. Extend Analysis transfer evidence so transfer-adjoint diagnostics can report
   dot-product residuals, declared inner products, weights, ownership semantics,
   and missing transpose support in addition to conservation or
   constant-preservation evidence.
12. Add tests where:
   - the objective depends on a boundary integral input
   - the residual depends on an auxiliary output
   - a monolithic multiphysics coupling produces expected transpose blocks
   - a global kernel produces expected transpose contributions or fails
     readiness when it cannot
   - an interface transfer passes a weighted dot-product adjointness test
   - a partitioned exchange graph produces a valid reverse replay plan
   - the full coupled gradient matches finite differences

### Acceptance Criteria

- Coupled PDE-plus-auxiliary adjoint gradients are verified on at least one monolithic and one partitioned example.
- Existing direct-coupling records are reused instead of bypassed.
- Global kernels either provide adjoint-ready metadata/tangent/transpose/state
  hooks or fail readiness.
- Monolithic FE/Coupling adjoints consume installed Systems, Analysis, and
  FE/Coupling metadata rather than re-parsing Forms.
- Partitioned coupling adjoints use explicit transfer-adjoint descriptors and
  preserve endpoint, temporal, frame, and interface-map provenance.
- Every supported transfer-adjoint descriptor declares source and target inner
  products, weights, component layout, ownership, and MPI reduction semantics.
- Transfer-adjoint readiness diagnostics are attached to public coupling and
  Analysis records, not a private adjoint transfer registry.

## Phase 6: Add Parameter, Control, Moving-Domain, and Shape Derivative Infrastructure

### Why

An adjoint solve is only half the infrastructure. Many real use cases need:

- `dR/dp`
- `dQ/dp`
- total gradients `dQ/dp - lambda^T dR/dp`

Today `ParameterRegistry` validates and evaluates parameter values, but it does
not provide derivative infrastructure. Moving-domain work also adds solved mesh
displacement, prescribed mesh-motion data, derived mesh velocity, current
coordinates, and geometry-sensitive tangent paths that adjoint gradients must
distinguish from broader CAD shape derivatives. Constitutive and material models
also need explicit derivative contracts so the adjoint path can verify that the
assembled tangent is consistent with the primal residual and material-state
history policy.

### Recommended Design

Split this phase into three layers.

### Phase 6A: Parameter-Slot Sensitivities

Treat FE parameter slots as differentiable inputs.

Recommended capabilities:

- register which parameters are differentiable
- assemble `dR/dp_k`
- assemble `dQ/dp_k`
- build total gradients from the adjoint
- reuse parameter slot layout from `ParameterRegistry`

### Phase 6B: Moving-Domain Geometry Sensitivities

Treat moving-domain geometry sensitivities as the next step after parameter
slots because the ALE infrastructure already exposes solved mesh displacement,
derived mesh velocity, current-coordinate assembly, and geometry-sensitive
tangent metadata.

Recommended capabilities:

- geometry or mesh coordinate perturbation interface
- residual and objective derivatives with respect to solved mesh-displacement
  unknowns
- explicit consumption of `GeometryTransaction`, `GeometricNonlinearityPolicy`,
  `MovingDomainOrchestrator`, `FEAdaptivityTransfer`,
  `ContactOperatorInvalidation`, and `CutIntegrationInvalidation` metadata
- derivative provenance for current coordinates, current measure, inverse
  Jacobian, normal, face measure, surface Jacobian, mesh velocity, and mesh
  acceleration where supported
- geometry revision, coordinate-configuration, invalidation-decision,
  accepted/trial map, remesh/adaptivity-transfer, and rollback/commit metadata
  in the adjoint readiness report
- finite-difference checks for coupled ALE residual and objective gradients with
  respect to mesh displacement

### Phase 6C: Constitutive and Material Derivative Readiness

Treat constitutive/material derivative support as a separate contract from
generic parameter derivatives because material state, update frame, consistent
tangent, and history commit policy affect the exact discrete Jacobian.

Recommended capabilities:

- consume `StressTangentContract` for stress/tangent measure, input measure,
  update frame, and consistent-tangent availability
- consume `MaterialStateSpec` and state-variable metadata for replay/checkpoint
  requirements
- reuse `Forms/Dual` and `Constitutive/DualOps` for local derivative checks
  where they apply
- reject discontinuous material laws, table lookups, or history updates without
  scoped derivative policy and tests
- verify constitutive tangent blocks against finite differences for at least one
  representative material path

### Phase 6D: CAD or Design-Surface Shape Sensitivities

Treat CAD/design-surface shape differentiation as a later extension with its own
design note. It is broader than moving-domain mesh-displacement sensitivity
because it must define how design variables perturb geometry, mesh coordinates,
boundary measures, interface maps, remeshing, transfer, and objective
evaluation.

Recommended capabilities:

- design-variable registry for geometry or CAD parameters
- derivative hooks in geometry mappings, boundary measures, normals, and
  interface-transfer geometry where supported
- explicit opt-in finite-difference fallback for opaque CAD or meshing steps
- restart/replay metadata for design-surface and mesh-coordinate provenance

### Concrete Files to Modify

Phase 6A:

- `Code/Source/solver/FE/Systems/ParameterRegistry.*`
- `Code/Source/solver/FE/Forms/SymbolicDifferentiation.*`
- `Code/Source/solver/FE/Forms/FormKernels.*`
- `Code/Source/solver/FE/Systems/FESystem.*`
- `Code/Source/solver/FE/CMakeLists.txt`

Phase 6B:

- `Code/Source/solver/FE/Geometry/*`
- `Code/Source/solver/FE/Assembly/*`
- `Code/Source/solver/FE/Forms/*`
- `Code/Source/solver/FE/Systems/*`
- `Code/Source/solver/FE/MovingMesh/*`
- `Code/Source/solver/Mesh/*`

Phase 6C:

- `Code/Source/solver/FE/Constitutive/*`
- `Code/Source/solver/FE/Forms/Dual.*`
- `Code/Source/solver/FE/Constitutive/DualOps.*`
- `Code/Source/solver/FE/Assembly/AssemblyKernel.h`
- `Code/Source/solver/FE/Systems/GlobalKernel.*`

Phase 6D:

- `Code/Source/solver/Mesh/Geometry/*`
- `Code/Source/solver/Mesh/Motion/*`
- `Code/Source/solver/Mesh/Search/*`
- `Code/Source/solver/FE/Systems/InterfaceOperators.*`
- future CAD or Application-level design-variable binding code

### Concrete Steps

1. Add a derivative model for FE parameter slots:
   - slot metadata
   - differentiable vs nondifferentiable classification
2. Add symbolic or AD support for derivatives with respect to parameter terminals where feasible.
3. For coefficient callbacks that remain opaque, make the finite-difference fallback explicit and opt-in, not implicit.
4. Add `assembleParameterSensitivity(...)` and `assembleObjectiveParameterGradient(...)`.
5. Add moving-domain sensitivity APIs that consume existing ALE binding,
   geometry-sensitivity options, geometry-revision metadata,
   geometry-transaction state, invalidation decisions, and adaptivity-transfer
   provenance.
6. Add constitutive/material readiness checks for:
   - consistent tangent availability
   - material-state layout and replay requirements
   - update-frame compatibility
   - discontinuous or table-driven law derivative policy
7. Add steady adjoint tests for:
   - material coefficient parameter
   - source amplitude parameter
   - boundary control parameter
   - coupled mesh-displacement unknown
8. Add finite-difference tests for moving-domain residual and objective
   gradients with respect to mesh displacement.
9. Add finite-difference tests for at least one constitutive tangent path.
10. Treat CAD/design-surface shape differentiation as a second-stage extension
   with its own design note before implementation.

### Acceptance Criteria

- Parameter-slot total gradients can be produced through a public FE API.
- Gradient tests pass for at least scalar material, source, and boundary parameters.
- Moving-domain adjoint gradients distinguish prescribed mesh-motion data,
  derived moving-domain fields, and solved mesh-displacement unknowns.
- Coupled ALE geometry-sensitive residual and objective derivatives are verified
  against finite differences before moving-domain design gradients are advertised.
- Constitutive/material paths either provide consistent tangent and state-history
  derivative contracts or fail readiness.

## Phase 7: Add Transient Discrete-Adjoint Infrastructure

### Why

This is the largest missing subsystem.

The current transient stack is entirely forward-oriented:

- `TimeIntegrator` only builds forward `dt(...)` lowering metadata
- `TimeLoop` only marches forward
- `NewtonSolver` only performs forward solves
- `TimeHistory` only stores the short history needed by the primal stencil
- `TimeHistory` layout, ghost updates, backend repacking, variable-step history,
  and second-order state are not represented as replayable adjoint trajectory
  data
- auxiliary checkpoint packing does not preserve history
- global-kernel and material-state replay contracts are not part of an adjoint
  trajectory model

Transient adjoints require much more:

- a trajectory model
- checkpointing and replay
- scheme-specific reverse accumulation
- auxiliary history restoration
- clear semantics for variable-step methods
- backend-compatible vector layout restoration, including ghost state and
  backend repacking when the matrix factory changes vector ordering
- replay of `u_dot`, `u_ddot`, and `dt_history` for schemes that depend on
  second-order or variable-step state
- moving-domain transaction replay for current coordinates, mesh-motion data,
  mesh-displacement history, geometry revisions, and interface maps
- coupling replay for endpoint temporal slots, transfer state, accepted/trial
  interface-map state, and partitioned exchange graphs
- global-kernel and material-state old/work buffer replay where those states
  affect residuals, objectives, constraints, or transfers

### Recommended Design

Introduce a dedicated reverse-time infrastructure layer rather than trying to hide everything inside the forward `TimeLoop`.

Recommended concepts:

- `AdjointCheckpointPolicy`
- `AdjointCheckpointManager`
- `PrimalTrajectorySlice`
- `BackendVectorTrajectorySlice`
- `TimeHistoryReplayDescriptor`
- `MovingDomainTrajectorySlice`
- `CouplingTrajectorySlice`
- `GlobalKernelTrajectorySlice`
- `MaterialStateTrajectorySlice`
- `TransientAdjointSystem`
- `AdjointTimeIntegrator`
- `AdjointTimeLoop`

### Recommended Support Order

1. Backward Euler or BDF1
2. BDF2
3. generalized-alpha first-order
4. Newmark or structural generalized-alpha
5. multistage collocation methods
6. VSVO BDF

### Concrete Files to Modify

- `Code/Source/solver/FE/Systems/SystemState.h`
- `Code/Source/solver/FE/Systems/TimeIntegrator.*`
- `Code/Source/solver/FE/Systems/TransientSystem.h`
- `Code/Source/solver/FE/TimeStepping/TimeHistory.*`
- `Code/Source/solver/FE/TimeStepping/MovingMeshTimeIntegration.*`
- `Code/Source/solver/FE/TimeStepping/TimeLoop.*`
- `Code/Source/solver/FE/TimeStepping/NewtonSolver.*`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.*`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryService.*`
- `Code/Source/solver/FE/Systems/GlobalKernelStateProvider.*`
- `Code/Source/solver/FE/Constitutive/*`
- `Code/Source/solver/FE/MovingMesh/*`
- `Code/Source/solver/FE/Coupling/*`
- `Code/Source/solver/FE/Systems/InterfaceOperators.*`
- `Code/Source/solver/FE/CMakeLists.txt`
- likely new files:
  - `Code/Source/solver/FE/TimeStepping/AdjointTimeLoop.h`
  - `Code/Source/solver/FE/TimeStepping/AdjointTimeLoop.cpp`
  - `Code/Source/solver/FE/Systems/AdjointTimeIntegrator.h`
  - `Code/Source/solver/FE/Systems/AdjointTimeIntegrator.cpp`
- tests under `Code/Source/solver/FE/Tests/Unit/TimeStepping/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`

### Concrete Steps

1. Define the minimal stored primal state needed at each accepted time step:
   - time
   - dt
   - `dt_prev` and full `dt_history` required by the chosen scheme
   - accepted solution
   - accepted history vectors required by the chosen scheme
   - backend vector identity, vector-layout revision, owned/ghost policy, and
     enough metadata to restore `SystemStateView` backend vector pointers
   - ghost-update validity and `TimeHistory::repack(...)` requirements for
     backends whose vector layout depends on a primed matrix factory
   - `u_dot` and `u_ddot` for Newmark, structural generalized-alpha, and any
     other second-order path that allocates second-order state
   - auxiliary committed state
   - auxiliary history state where needed
   - global-kernel state and material-state old/work buffers where needed
   - committed current coordinates and geometry revision keys
   - geometry transaction state, coordinate configuration, and rollback/commit
     provenance
   - `MovingMeshTimeHistory` state, including current/previous coordinates,
     displacements, `dt` history, mesh velocity inputs, and acceleration inputs
     when moving-mesh time integration is active
   - mesh-motion field-source metadata and derived moving-domain field metadata
   - accepted/trial interface-map provenance when interface transfers are used
   - partitioned endpoint temporal-slot backing where partitioned replay is used
   - contact/cut/adaptivity invalidation and transfer provenance when those
     features are active
2. Add checkpoint policy options:
   - full storage
   - periodic checkpoints plus replay
   - recompute-on-demand
3. Extend time integrators with adjoint metadata:
   - transpose of time-stencil coefficients
   - per-stage reverse dependencies
   - variable-step sensitivity support
   - mesh-displacement-to-mesh-velocity derivative coefficients
4. Add a reverse-time driver that:
   - loads or reconstructs the primal state at step `n`
   - restores backend-compatible vector layout and ghost state before assembly
   - restores moving-domain and coupling trajectory slices before assembly
   - restores global-kernel and material-state trajectory slices before assembly
   - assembles the step-local adjoint system
   - accumulates contributions to step `n-1`
5. Start with one scheme only, preferably Backward Euler or BDF1.
6. Add one complete transient adjoint verification test:
   - objective at final time
   - compare adjoint gradient against finite differences in the initial condition or a parameter
7. Add a moving-domain transient replay test before advertising transient ALE
   adjoints.
8. Add variable-step and second-order replay tests before advertising BDF2,
   generalized-alpha, Newmark, or structural generalized-alpha adjoints.
9. Add a partitioned-coupling replay test before advertising partitioned
   transient adjoints.
10. Only after that should the library extend to BDF2 and generalized-alpha.

### Acceptance Criteria

- One transient scheme has a full discrete-adjoint implementation verified against finite differences.
- Checkpointing or replay policy is explicit and tested.
- Replay restores backend-compatible vectors, ghost validity, `dt_history`, and
  any required `u_dot`/`u_ddot` state before step-local assembly.
- Auxiliary states participate consistently in transient replay.
- Global-kernel and material states participate consistently in transient
  replay when advertised.
- Moving-domain trajectory replay restores coordinates, revisions, field-source
  metadata, geometry transaction state, invalidation provenance, and
  interface-map state consistently.
- Moving-mesh replay restores `MovingMeshTimeHistory` coordinates,
  displacements, `dt` history, and velocity/acceleration inputs for active
  moving-mesh time-integration paths.
- Partitioned coupling trajectory replay preserves endpoint temporal slots and
  transfer provenance.

## Phase 8: Replace Placeholder Goal-Oriented Indicator Support

### Why

`FunctionalAssembler::computeGoalOrientedIndicators(...)` is currently a placeholder implementation, not a production DWR workflow.

Adjoint-capable FE infrastructure should provide a real dual-weighted residual path once adjoint solves exist.

### Recommended Design

Make DWR support consume the same steady or transient adjoint solutions produced by the new adjoint APIs.

Recommended capabilities:

- residual-weighted local indicators
- cell-wise and face-wise contributions
- support for constrained spaces
- support for mixed systems
- optional aggregation by region or field block

### Concrete Files to Modify

- `Code/Source/solver/FE/Assembly/FunctionalAssembler.*`
- `Code/Source/solver/FE/Systems/FESystem.*`
- `Code/Source/solver/FE/Forms/FormKernels.*`
- tests under `Code/Source/solver/FE/Tests/Unit/Assembly/`
- tests under `Code/Source/solver/FE/Tests/Unit/Systems/`

### Concrete Steps

1. Replace the placeholder indicator computation with a real residual-times-dual evaluation path.
2. Add access to:
   - primal solution
   - adjoint solution
   - cell residual contributions
   - face residual contributions where required
3. Add verification on simple elliptic problems where indicator localization is known qualitatively.
4. Document that DWR is only considered production-ready after the underlying adjoint solve is production-ready.

### Acceptance Criteria

- The placeholder implementation is replaced or explicitly hidden behind a "not yet production" guard.
- DWR indicators use the actual adjoint solution instead of a surrogate.

## Phase 9: Verification, Testing, and Diagnostics

### Why

Adjoint infrastructure is especially easy to get almost right. The FE layer needs a stronger verification bar than ordinary forward assembly changes.

### Required Test Categories

#### Backend Tests

- transpose apply vs explicit transpose
- transpose solve vs explicit transpose solve
- advertised transpose-capability flags vs readiness diagnostics
- matrix-free/device transpose hooks either pass dot-product tests or fail
  readiness cleanly
- nullspace or gauge interaction where applicable

#### Systems Tests

- steady adjoint gradient vs finite differences
- mixed-system adjoint gradient vs finite differences
- constrained adjoint gradient vs finite differences
- coupled PDE-plus-auxiliary adjoint gradient vs finite differences
- monolithic multiphysics coupled adjoint gradient vs finite differences
- moving-domain geometry-sensitive adjoint gradient vs finite differences
- objective-gradient coverage for domain, boundary, interface, and selected
  multi-field objectives
- objective/QoI adapters reuse `FEQuantityRegistry`, `BoundaryReductionService`,
  `FunctionalAssembler`, `OperatorBackends`, and the value-supported
  `PointEvaluator` paths as intended
- value-only FE quantities and post-processing-only derived results fail
  readiness unless explicitly promoted to adjoint-ready QoIs
- point/sampled objectives without interpolation/search derivative provenance
  fail readiness before solve time
- global-kernel adjoint readiness and transpose contribution checks
- constitutive/material tangent and state-history readiness checks

#### Transfer and Coupling Tests

- interface interpolation transpose/adjointness dot-product tests
- conservative projection transpose/adjointness dot-product tests
- mortar transfer transpose/adjointness dot-product tests where supported
- driver-owned transfer rejects adjoint use unless an explicit transpose
  descriptor is registered
- partitioned exchange graph reverse-replay metadata is deterministic
- transfer tests include frame-transform, component-layout, and declared weight
  metadata
- transfer tests include owned/ghost vector semantics, block layout, distributed
  row/column ownership, and MPI reduction behavior

#### Time-Stepping Tests

- transient adjoint gradient vs finite differences for one-step schemes first
- variable-step regression after fixed-step support is stable
- backend vector layout, ghost validity, `dt_history`, and `u_dot`/`u_ddot`
  replay are covered before advertising schemes that need them
- global-kernel and material-state replay restores old/work state where active
- moving-domain trajectory replay restores coordinates, geometry revisions, and
  interface-map state
- partitioned coupling trajectory replay restores endpoint temporal slots and
  transfer provenance

#### Constraint Tests

- reduced-space objective-gradient mapping through `P^T`
- dynamic constraint dependency diagnostics for time, geometry, auxiliary, and
  moving-map dependencies
- unsupported `ConstraintTangentPolicy` values fail readiness

#### Analysis Tests

- adjoint-consistency metadata remains correct for:
  - symmetric Nitsche
  - unsymmetric Nitsche
  - mixed primal-dual block structures
- adjoint readiness reports supported and unsupported cases with stable,
  actionable diagnostics
- Analysis metadata is consumed through public records, not private Forms AST
  traversal

### Recommended Diagnostics

Add optional debug output for:

- adjoint-readiness capability issues
- objective gradient assembly statistics
- reduced vs full-space adjoint norms
- checkpoint load or replay counts
- transpose-solve backend path selection and unsupported-capability reasons
- objective/QoI source selection, FE quantity capability, and derived-result
  promotion status
- coupled adjoint block dimensions and conditioning hints
- global-kernel and constitutive/material readiness summaries
- dynamic constraint dependency and tangent-policy summaries
- transfer-adjoint operator path selection and dot-product residuals
- geometry coordinate configuration, revision, and replay state
- partitioned endpoint temporal-slot and interface-map provenance

### Concrete Files to Modify

- `Code/Source/solver/FE/CMakeLists.txt`
- `Code/Source/solver/FE/Tests/Unit/Backends/*`
- `Code/Source/solver/FE/Tests/Unit/Adjoint/*`
- `Code/Source/solver/FE/Tests/Unit/Systems/*`
- `Code/Source/solver/FE/Tests/Unit/TimeStepping/*`
- `Code/Source/solver/FE/Tests/Unit/Analysis/*`
- `Code/Source/solver/FE/Tests/Unit/Constitutive/*`
- `Code/Source/solver/FE/Tests/Unit/Constraints/*`
- `Code/Source/solver/FE/Tests/Unit/Coupling/*`
- `Code/Source/solver/FE/Tests/Unit/MovingMesh/*`
- optional trace or logging hooks in:
  - `TimeStepping/NewtonSolver.cpp`
  - `TimeStepping/TimeLoop.cpp`
  - `Systems/FESystem.cpp`
  - `FE/Coupling` implementation files
  - `Systems/InterfaceOperators.cpp`
  - backend solver implementations

## Recommended Concrete Implementation Order

1. Add the adjoint-ready metadata and preflight contract, including mandatory
   `AdjointProblemFingerprint` validation and adjoint storage requirements that
   merge into `SetupStoragePlan` before setup.
2. Add the `FE/Adjoint` module skeleton and dependency-boundary tests or
   include checks where practical.
3. Add the new `FE/Adjoint` files and `Tests/Unit/Adjoint` target entries to
   the centralized `Code/Source/solver/FE/CMakeLists.txt`.
4. Add transpose apply and transpose solve to the backend interfaces.
5. Add matrix-free/device transpose capability hooks and readiness failures for
   unsupported kernels.
6. Implement transpose support for one assembled backend and add conformance tests.
7. Add adjoint objective/QoI adapters backed by `FEQuantityRegistry`,
   `BoundaryReductionService`, `FunctionalAssembler`, `OperatorBackends`, and
   value-supported `PointEvaluator` paths, with explicit adjoint capability
   metadata for gradients.
8. Add a minimal steady adjoint solve API in `FE/Systems` backed by
   `FE/Adjoint` implementation types.
9. Wire `ConstraintTransform` into the adjoint path and test constrained adjoints.
10. Add dynamic-constraint readiness checks using dependency/tangent policies.
11. Add transfer-adjoint descriptors, inner-product metadata, and dot-product
    tests for the first supported interface transfer path.
12. Add custom-kernel, global-kernel, and constitutive/material readiness checks
    and reject unsupported opaque contributions.
13. Extend the steady path to PDE-plus-auxiliary coupling.
14. Extend the steady path to monolithic FE/Coupling multiphysics contributions
    after the required coupling metadata is stable and adjoint-capable.
15. Add end-to-end steady gradient verification tests.
16. Add parameter-slot derivative support.
17. Add moving-domain mesh-displacement sensitivity support and FD tests.
18. Add checkpoint or replay infrastructure for transient adjoints, including
    moving-domain trajectory slices.
19. Add global-kernel and material-state trajectory slices where needed.
20. Add one full discrete transient adjoint path, starting with Backward Euler or BDF1.
21. Add partitioned coupling reverse-replay metadata and transfer-adjoint checks.
22. Add nonsmooth/topology-changing regime fail-closed checks and scoped
    derivative-policy hooks.
23. Extend to higher-order or multistage schemes only after the first transient path is verified.
24. Replace the placeholder DWR implementation with a real adjoint-driven version.
25. Update `Code/Source/solver/FE/README.md` after the `FE/Adjoint` module
    skeleton exists.

## Recommended "Do Not Do This" List

- Do not make the first adjoint implementation physics-specific.
- Do not begin with generalized-alpha or VSVO as the first transient target.
- Do not hide constraint semantics inside backend-specific hacks.
- Do not silently approximate parameter derivatives unless the API makes that explicit.
- Do not call the current placeholder DWR path "adjoint support."
- Do not require every matrix-free kernel to support transpose apply before steady assembled adjoints can land.
- Do not require every backend to support transpose solves; require backends to
  advertise capabilities and fail readiness clearly when unsupported.
- Do not create a second coupling graph, transfer registry, moving-geometry
  transaction model, or Analysis contribution IR inside the adjoint module.
- Do not create a second FE quantity/objective registry that duplicates
  `FEQuantityRegistry`, `BoundaryReductionService`, `OperatorBackends`, or
  `DerivedResultRegistry`.
- Do not recover adjoint dependencies by walking private Forms AST internals.
- Do not infer transfer adjointness from physical port names or endpoint names.
- Do not treat post-processing-only derived results as differentiable
  objectives unless they are explicitly promoted with derivative contracts.
- Do not treat handwritten or backend-local kernels as adjoint-ready unless
  they provide the same public tangent, sensitivity, transpose, and Analysis
  metadata required of Forms-installed residuals.
- Do not ignore dynamic constraint dependencies, global-kernel state, material
  history state, or device/matrix-free capability limits in readiness checks.
- Do not advertise gradients through contact, limiter, event, remesh, cut-topology,
  or CAD/meshing discontinuities without a scoped derivative policy and tests.
- Do not update the FE README to advertise `FE/Adjoint` before the module
  skeleton and public API exist.

## Minimum Definition of Done

The FE library should not claim adjoint support until all of the following are true:

- one public adjoint-readiness/preflight report exists
- every positive readiness report carries a valid `AdjointProblemFingerprint`,
  and fingerprint validation fails after relevant setup, storage, layout,
  operator, geometry, constraint, DOF-permutation, backend-cache, or vector-layout
  changes
- the adjoint readiness path verifies required `SetupStoragePlan` artifacts were
  retained before setup, or fails with a named missing-storage diagnostic
- the core implementation lives under `FE/Adjoint` with `FE/Systems` acting as
  an orchestration surface
- the new module and tests are wired through the centralized FE build/test lists
- one public FE adjoint API exists
- one assembled backend advertises and supports transpose solve through the
  public backend contract, and unsupported backends fail readiness clearly
- objective/QoI requests use adapters over existing FE registries and services
  rather than a duplicate quantity registry
- one steady objective gradient is verified against finite differences
- constrained adjoint behavior is tested and documented
- constrained adjoint tests cover backend vector ownership, ghost state,
  block-layout, DOF-permutation, distributed-sparsity, nullspace, and
  preconditioner semantics for the advertised support row
- dynamic constraints either declare supported tangent/transpose policies or
  fail readiness
- at least one supported transfer operator has explicit transpose/adjoint apply
  and a dot-product adjointness test
- supported transfer adjointness tests declare source/target inner products,
  weights, ownership, component layout, frame transform, and MPI reduction
  semantics
- Analysis, Systems, and any coupling metadata consumed by the adjoint path use
  public records rather than private Forms AST traversal
- Analysis summaries are consumed as evidence and diagnostics; hard readiness is
  determined from explicit subsystem capability contracts unless a support row
  declares a specific summary mandatory
- custom kernels fail readiness unless they provide the required metadata and
  derivative/transpose hooks
- global kernels fail readiness unless they provide required metadata,
  derivative/transpose hooks, sparsity/block provenance, and snapshot/restore
  replay hooks when stateful
- constitutive/material paths fail readiness unless tangent, state-history,
  update-frame, and derivative contracts are available for the advertised support
  row
- unsupported nonsmooth or topology-changing regimes fail readiness unless a
  scoped derivative policy is declared and tested

The FE library should not claim coupled adjoint support until all of the following are true:

- one coupled PDE-plus-auxiliary adjoint case is verified if coupled auxiliary states are advertised as supported
- one monolithic FE/Coupling multiphysics adjoint case is verified after the
  required FE/Coupling metadata is stable if monolithic coupling is advertised as
  supported
- partitioned exchange graphs have reverse-replay metadata and every executed
  transfer has a supported transpose/adjoint descriptor if partitioned adjoints
  are advertised as supported

The FE library should not claim moving-domain or transient adjoint support until all of the following are true:

- one moving-domain mesh-displacement adjoint gradient is verified against
  finite differences if moving-domain adjoints are advertised as supported
- geometry revision, coordinate configuration, interface-map state, and
  mesh-motion field-source provenance are recorded in readiness/replay
  diagnostics for moving-domain adjoints
- moving-domain transaction state, invalidation decisions, and remesh/adaptivity
  transfer provenance are recorded when those features are active
- moving mesh time-integration replay restores `MovingMeshTimeHistory` data,
  including coordinate/displacement history, `dt` history, and velocity or
  acceleration inputs for advertised moving-mesh schemes
- one transient scheme is verified if transient adjoints are advertised as supported
- transient replay restores solution, auxiliary, global-kernel, material-state,
  moving-domain, and coupling trajectory slices for the advertised support row

## Recommended Follow-On Work After the Core Infrastructure Lands

- matrix-free transpose kernels for selected operators
- parameter and control optimization driver utilities outside FE core
- CAD/design-surface shape sensitivity design note and pilot implementation
- goal-oriented mesh-adaptation utilities driven by the new adjoint APIs
- application-layer examples for Poisson, Navier-Stokes, mixed systems,
  coupled ALE, and one coupling-module example
