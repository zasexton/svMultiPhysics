# FE AuxiliaryState Generalization — Implementation Plan

**Status (2026-04-22)**: Superseded for fixed-stride scope completion by
`Documentation/plan_auxiliary_state_scope_completion.md`.  Ragged deployment
work is tracked separately in
`Documentation/plan_auxiliary_state_ragged_layout_followup.md`.  This document
is retained as historical design context.

**Date**: 2026-03-23
**Goal**: Replace the current coupled-boundary-owned auxiliary-state path with a generalized FE-library `AuxiliaryState` subsystem that:

- supports `Global`, `Node`, `Cell`, `QuadraturePoint`, and `BoundaryEntity` storage scopes,
- supports both `Partitioned` and `Monolithic` auxiliary solve modes,
- supports nonlocal auxiliary-auxiliary and field-auxiliary couplings through auxiliary operators,
- supports DAE-like local structures from phase 1,
- includes a declarative author-facing `AuxiliaryModel` API that lowers to the same backend infrastructure,
- generalizes the symbolic vocabulary immediately,
- and deprecates the older coupled-boundary-owned API in favor of one modular implementation model.

**Primary principle**: `AuxiliaryState` is FE-library infrastructure, not a boundary-condition feature and not a physics-specific concept. Boundary functionals, EP-like models, metabolism-like models, reduced models, and future coupled subsystems should all use the same neutral `AuxiliaryState` infrastructure.

**Non-goal**: This plan does not implement any specific physics formulation. It only delivers physics-agnostic FE-library infrastructure for auxiliary ODE/DAE-like systems.

**Scope boundary**: `AuxiliaryState` is not intended to replace true PDE components or FE fields. If a quantity should be represented as a primary spatially distributed PDE unknown with FE basis functions, differential operators, and weak-form assembly as a field, it should remain a field rather than be forced into `AuxiliaryState`.

---

## Author-Facing API Principles

- [ ] Formulation authors should define auxiliary models and attach them to formulations; they should not manually register internal blocks, slot ids, or storage layouts.
- [ ] The FE library should provide a math-first `AuxiliaryModel(...)` builder for common ODE/DAE-like models and a lower-level residual/Jacobian interface for advanced implementations.
- [ ] The high-level builder and the low-level interface must lower to the same backend `AuxiliaryStateModel` representation and execution pipeline.
- [ ] Auxiliary model definitions should be field-agnostic. FE field semantics should enter only through explicit deployment-time bindings.
- [ ] Storage scope and solve mode are orthogonal concepts. `Global` scope does not imply monolithic solve participation, and `Monolithic` solve mode does not imply `Global` scope.
- [ ] Storage scope and deployment region are orthogonal concepts. Scope defines the storage entity type; deployment region defines which subset of the mesh or boundary owns that storage.
- [ ] Inputs, outputs, and optional parameters define the reusable public contract of an auxiliary model; internal state names remain model-specific implementation details.
- [ ] Auxiliary outputs should be first-class Forms vocabulary terms and the preferred formulation-facing coupling surface.
- [ ] Boundary conditions should use the same deployment workflow as any other auxiliary model consumer, through boundary-scoped contexts rather than a separate auxiliary-state concept.
- [ ] `AuxiliaryModel` should remain a local/global-0D or per-entity model API. Genuinely nonlocal couplings should be expressed through a separate `AuxiliaryOperator` API.
- [ ] Raw auxiliary state access may remain available for advanced or diagnostic use, but it should not be the primary new-user workflow.

---

## Resolved Public API Decisions

- [ ] The canonical reusable contract type is `AuxiliaryModelSignature`.
- [ ] Model interchangeability is determined by signature-compatible inputs, outputs, and optional parameters; internal state names are not part of public compatibility.
- [ ] Public block identity is the auxiliary block `name`, unique within an `FESystem`; scope is metadata and is not part of the public identity key.
- [ ] The canonical deployment entry point is `use(model)`.
- [ ] The canonical deployed-instance configuration surface is `.name(...)`, `.scope(...)`, `.region(...)`, `.solveMode(...)`, `.schedule(...)`, `.stepper(...)`, `.bind(...)`, `.param(...)`, and `.initialize(...)`.
- [ ] The canonical derivative-policy entry point is `.derivatives(...)` on the model definition. Deployed instances inherit the resolved model-level policy in phase 1.
- [ ] Optional signature parameters may be part of the public contract and may be bound either to literals or to provider-driven values when formulation semantics require time- or state-dependent parameter behavior.
- [ ] Nonlocal auxiliary couplings are represented through `AuxiliaryOperator`, not directly through `AuxiliaryModel`.
- [ ] Mixed differential and algebraic variables are allowed within one auxiliary block.
- [ ] Solve mode is fixed once deployed auxiliary instances are finalized during `system.setup()`.
- [ ] Deployment region is fixed once deployed auxiliary instances are finalized during `system.setup()`.
- [ ] History depth is block-wide in phase 1.
- [ ] Fixed-stride layout is the default; ragged layout is an explicit choice or manager-selected fallback when fixed-stride layout cannot represent the formulation correctly.
- [ ] Ragged layout uses canonical per-entity offsets, while grouped or archetyped fast paths remain an internal optimization rather than public API.
- [ ] Canonical auxiliary entity ordering follows owned mesh or DOF-layer ordering, appends ghosts explicitly where needed, and defaults to `ByEntityThenComponent` ordering unless a formulation explicitly selects otherwise.
- [ ] Auxiliary block names are the durable public handles; numeric block ids and slots are finalized at setup time and are internal, setup-stable implementation details.
- [ ] All auxiliary blocks are owned by `FESystem` and must be finalized before `system.setup()`; adding new auxiliary blocks afterward requires a future re-finalization workflow and is not part of phase 1.
- [ ] `schedule(...)` controls when and how often auxiliary advancement occurs relative to the PDE step, while `stepper(...)` controls the numerical method used when that advancement executes.
- [ ] Local stepper selection, local nonlinear-solver options, substep input-refresh policy, and substep commit policy are valid only for `Partitioned` solve mode.
- [ ] Lower-level residual implementations that cannot satisfy the resolved derivative policy must provide a compatible analytic, AD, or finite-difference path or fail setup with a clear diagnostic.
- [ ] `AuxiliaryOperator` is the advanced public API for genuinely nonlocal coupling graphs and custom mixed sparse operators; local and per-entity monolithic models stay on the `AuxiliaryModel` plus `use(model)` workflow.
- [ ] Auxiliary input bindings must form an acyclic provider DAG within an evaluation context. Same-time cyclic dependencies are not valid as input bindings and must instead be represented through `Monolithic` coupling or `AuxiliaryOperator`.
- [ ] Cross-scope auxiliary-to-auxiliary bindings are not implicit. They must use explicit mapping providers such as traces, reductions, or projections, or they must move to `AuxiliaryOperator`.
- [ ] `Monolithic` auxiliary unknowns use auxiliary-specific unknown maps and layouts rather than reusing FE field DOF maps as their primary representation.
- [ ] `FESystem` composes FE field unknown layouts and auxiliary-specific unknown layouts into one mixed system layout for monolithic assembly and solves.
- [ ] FE mesh or DOF metadata may provide ownership and ordering information for auxiliary unknown layouts, but auxiliary blocks are not modeled as FE fields.
- [ ] The canonical advanced nonlocal authoring surface is `AuxiliaryOperator(...)` with `.name(...)`, `.source(...)`, `.target(...)`, `.topology(...)`, `.residual(...)`, `.jacobian(...)`, optional `.mass(...)` or `.transfer(...)`, and optional `.derivatives(...)`.
- [ ] `Monolithic` auxiliary models expose a semidiscrete residual contract `R_aux(xdot, x, history, inputs, fields, t, dt) = 0`; `TimeIntegrator` owns stage state, iterate state, and the mapping between stage values and `xdot`.
- [ ] Auxiliary outputs may depend on `x`, `xdot`, history, inputs, parameters, and coupled FE fields when those quantities are available in the current evaluation context.
- [ ] Output evaluation always occurs against an explicit state view such as committed state, work state, stage state, nonlinear iterate state, or substep state.
- [ ] Runtime auxiliary evaluation uses immutable read views plus thread-local or caller-owned scratch. Setup-built derivative and lowering caches are read-only during threaded evaluation.
- [ ] Raw auxiliary state access remains an advanced path, but first-class auxiliary outputs remain the canonical coupling surface shown in primary examples.
- [ ] The canonical example set used to validate ergonomics is:
  - [ ] a partitioned nodal ionic-like auxiliary model
  - [ ] a monolithic local auxiliary model coupled to an FE field
  - [ ] a global lumped boundary auxiliary model
  - [ ] a boundary-local auxiliary model
  - [ ] a derivative-override example comparing analytic, symbolic, and `FiniteDifference` behavior

---

## Fixed Decisions

- [ ] `AuxiliaryState` must support multiple storage scopes chosen by the formulation: `Global`, `Node`, `Cell`, `QuadraturePoint`, `BoundaryEntity`.
- [ ] `AuxiliaryState` must support deployment regions orthogonal to scope so state may be restricted to selected domains, subdomains, material sets, interface sets, boundary sets, or formulation-defined entity subsets.
- [ ] Bulk/distributed `AuxiliaryState` is part of the initial design, not a later add-on.
- [ ] `AuxiliaryState` must support both `Partitioned` and `Monolithic` solve modes.
- [ ] `Monolithic` auxiliary blocks must be able to participate as first-class unknowns in assembled residual/Jacobian systems.
- [ ] The design must support nonlocal auxiliary-auxiliary and field-auxiliary couplings through dedicated coupling graph and auxiliary-operator infrastructure.
- [ ] DAE-like local systems are required from phase 1; the public API must not be ODE-only.
- [ ] Phase-1 DAE support is explicitly bounded to local mixed ODE/algebraic systems with index-1-like structure or structurally nonsingular initialization paths.
- [ ] `AuxiliaryState` must include automatic derivative infrastructure for Jacobians and optional second-derivative information needed by ODE/DAE-like algorithms.
- [ ] If a user provides analytic derivatives, those derivatives override automatically generated derivatives.
- [ ] For expression-defined auxiliary models, symbolic derivative generation is the default for Jacobians and any requested second-derivative information.
- [ ] Users may explicitly choose `FiniteDifference` derivative generation instead of the default symbolic path.
- [ ] Hessian support should be optional and demand-driven, with Hessian-vector products or selected second-derivative blocks allowed instead of always forming full dense Hessians.
- [ ] Jacobian support is the phase-1 derivative requirement. Second-derivative support remains extension-only in phase 1 unless a concrete solver path requires it.
- [ ] Symbolic derivatives for auxiliary models must be obtained by differentiating lowered auxiliary residual `FormExpr` expressions, not by differentiating storage containers or runtime buffers directly.
- [ ] The storage model must admit both fixed-stride and ragged per-entity layouts.
- [ ] History services must support time-stamped snapshots and interpolation hooks, not only fixed step-back access.
- [ ] The subsystem must include event/nonsmooth extension hooks and multirate scheduling/rollback hooks.
- [ ] The subsystem must define failure semantics for local solve failure, singular Jacobians, event-localization failure, subcycle divergence, rollback, and time-step rejection.
- [ ] The symbolic vocabulary must be generalized now rather than kept boundary-specific.
- [ ] The older coupled-boundary-owned API should be deprecated and reduced to forwarding shims only during migration.
- [ ] The FE library should expose one conceptual model for `AuxiliaryState`, even if temporary deprecated wrappers exist during the transition.
- [ ] The canonical user workflow must be declarative model definition and deployment rather than manual block registration.
- [ ] Field-derived auxiliary inputs must be bound through explicit provider semantics such as sampled, coupled, or reduced field access, not by embedding FE field logic directly in model definitions.
- [ ] Auxiliary outputs must be first-class symbolic vocabulary and the preferred surface for coupling auxiliary models into formulations.
- [ ] Output semantics must specify which state view they observe in each lifecycle context, whether they are cached or recomputed, and how their derivatives participate in monolithic assembly.
- [ ] The canonical contract type is `AuxiliaryModelSignature`, and it includes inputs, outputs, and optional parameter descriptors.
- [ ] The math-first `AuxiliaryModel` builder and the lower-level residual/Jacobian API must lower to the same backend model representation.
- [ ] The canonical deployment entry point is `use(model)`.
- [ ] Boundary-condition use cases must deploy through the same public `use(model)` workflow, with explicit boundary trace and boundary reduction providers.
- [ ] The plan must explicitly support both lumped boundary models using `Global` scope and boundary-local models using `BoundaryEntity` scope.
- [ ] Nonlocal auxiliary couplings must be represented through a separate `AuxiliaryOperator` API rather than being encoded directly in `AuxiliaryModel`.
- [ ] `Partitioned` auxiliary instances must expose an explicit public API for local time-advancement method selection, substepping, and related staggered-advance policies.
- [ ] `Monolithic` auxiliary instances must not pretend to use an independent local stepper API when their time discretization is owned by the global assembled solve.
- [ ] Derivative policy is configured at the model level in phase 1; deployed instances inherit that resolved policy.
- [ ] Auxiliary block identity is string-based by unique block name within `FESystem`; scope is not part of the public identity key.
- [ ] Mixed differential and algebraic variables may coexist within a single auxiliary block.
- [ ] Solve mode is fixed once auxiliary deployment is finalized during `system.setup()`.
- [ ] History depth is block-wide in phase 1.
- [ ] Fixed-stride layout is the default; ragged layout is explicit and uses canonical per-entity offsets with optional internal grouped fast paths.
- [ ] Canonical entity ordering follows owned mesh or DOF ordering with explicit ghost append rules and defaults to `ByEntityThenComponent`.
- [ ] Auxiliary block names are public stable handles, while numeric block ids and slot ids are internal setup-stable identifiers only.
- [ ] All auxiliary blocks are registered under `FESystem` ownership and must be finalized before `system.setup()` in phase 1.
- [ ] `schedule(...)` selects advancement timing and rate, while `stepper(...)` selects the numerical integration method used by `Partitioned` auxiliary advancement.
- [ ] Lower-level residual implementations that cannot honor the resolved derivative policy must expose a compatible derivative path or fail setup clearly.
- [ ] `AuxiliaryOperator` must provide the advanced public API for genuinely nonlocal and custom sparse couplings, while local monolithic models remain on `AuxiliaryModel`.
- [ ] `Monolithic` auxiliary unknowns must use auxiliary-specific unknown layouts composed into a mixed system layout rather than reusing FE field DOF maps directly.
- [ ] `AuxiliaryOperator(...)` must have a canonical public builder surface for advanced nonlocal and custom mixed sparse couplings.
- [ ] Restart and remap durability must validate schema, block identity, deployment-region identity, ordering, and history metadata before advanced operator-based transfer phases.
- [ ] Thread-safe scratch ownership and immutable runtime read semantics must be explicit in the infrastructure contract rather than left implicit.

---

## End-State Definition

> **Status note**: This section was written as an original design target.  Items
> are checked off as the later phase-by-phase sections track them to completion.

The work is complete when all of the following are true:

- [x] `FESystem` owns a generalized `AuxiliaryStateManager` and no FE core code depends on `CoupledBoundaryManager` to own auxiliary-state storage or advancement.
- [x] `AuxiliaryState` supports `Global`, `Node`, `Cell`, `QuadraturePoint`, and `BoundaryEntity` scopes with committed/work/history semantics.
- [x] `AuxiliaryState` supports deployment regions orthogonal to scope and can allocate state only on selected domains, boundaries, interfaces, or formulation-defined entity subsets.
- [x] Auxiliary blocks can be configured as either `Partitioned` or `Monolithic`.
- [x] Distributed scopes have defined ownership, ghost/sync rules, and MPI-safe update behavior.
- [x] The auxiliary update contract is phase-1 DAE-capable for local mixed ODE/algebraic systems with index-1-like or structurally nonsingular initialization paths, with ODE-like updates implemented as a specialization.
- [x] The derivative subsystem can supply Jacobians through analytic overrides, symbolic generation, or `FiniteDifference` according to explicit policy, while second-derivative information remains optional and extension-driven.
- [x] Expression-defined auxiliary models are lowered at setup time into residual `FormExpr` trees, and symbolic derivatives are generated from those lowered residual expressions and cached for runtime use.
- [x] `Monolithic` auxiliary blocks can participate in global residual/Jacobian assembly and couple to FE fields and other auxiliary blocks.
- [x] Nonlocal auxiliary couplings are supported through dedicated coupling graph and auxiliary-operator infrastructure.
- [x] Storage supports both fixed-stride and ragged layouts.
- [x] History services support time-stamped snapshots and interpolation hooks.
- [x] Event/nonsmooth hooks and multirate scheduling/rollback hooks are part of the subsystem contract.
- [x] Input bindings are validated as an acyclic provider DAG, and same-time cycles are redirected to `Monolithic` or `AuxiliaryOperator` coupling paths rather than accepted silently.
- [x] The symbolic vocabulary uses neutral auxiliary-state and auxiliary-input terminology instead of coupled-boundary terminology.
- [x] New formulations can define and deploy auxiliary models without manually registering blocks, slot ids, or storage layouts.
- [x] The public API includes a math-first `AuxiliaryModel` builder plus an advanced lower-level interface, both lowering to the same runtime model representation.
- [x] Auxiliary outputs are first-class Forms terms and can be consumed directly in FE formulations.
- [x] Auxiliary outputs have explicit evaluation-state semantics, invalidation rules, and derivative participation rules for monolithic assembly.
- [x] Boundary-condition consumers can deploy auxiliary models through the same public API without relying on boundary-specific ownership or scalar-only helpers.
- [x] JIT, interpreter, assembly context, point evaluation, and symbolic analysis all use the new vocabulary.
- [x] Coupled boundary conditions use the new subsystem as a client and no longer define the core model.
- [ ] Old boundary-specific entry points are marked deprecated, internally forward to the new subsystem, and no longer carry unique implementation logic. *(Partial: new subsystem is primary; old entry points still coexist.)*
- [x] Unit, integration, MPI, restart/transfer, and regression tests cover the new subsystem thoroughly.
- [x] Failure handling, restart-schema validation, deployment-region identity, and threaded evaluation semantics are covered by infrastructure tests.
- [x] FE documentation is updated so new work uses the generalized `AuxiliaryState` model by default.

---

## Recommended File Layout

> **Status note**: All planned files have been created.  The "Files to Heavily
> Modify" list reflects changes already made during implementation.

### Files Created

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateTypes.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryBuffer.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryBuffer.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateIndexing.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateIndexing.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryUnknownLayout.h` *(Embedded in `AuxiliaryOperatorRegistry.h`.)*
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryUnknownLayout.cpp` *(Embedded in `AuxiliaryOperatorRegistry.cpp`.)*
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryCouplingGraph.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryCouplingGraph.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorBuilder.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorBuilder.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDAEAnalyzer.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDAEAnalyzer.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInitializationSolver.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInitializationSolver.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryEventManager.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryEventManager.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryNonsmoothPolicy.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryService.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryService.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryTransferOperator.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryTransferOperator.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryMultirateScheduler.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryMultirateScheduler.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryModelDSL.h` *(Not in original plan; added for math-first DSL.)*
- [x] `Code/Source/solver/FE/Systems/BoundaryReductionService.h` *(Not in original plan; added for functional gradient assembly.)*
- [x] `Code/Source/solver/FE/Systems/BoundaryReductionService.cpp`
- [x] `Code/Source/solver/FE/Systems/FEQuantityDefinition.h` *(Not in original plan; added for FE-backed quantity handles.)*
- [x] `Code/Source/solver/FE/Systems/FEQuantityRegistry.h` *(Not in original plan.)*
- [x] `Code/Source/solver/FE/Systems/FEQuantityRegistry.cpp`

### Files Heavily Modified

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`
- [x] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.h`
- [x] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
- [x] `Code/Source/solver/FE/Systems/ODEIntegrator.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Systems/SystemState.h`
- [x] `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- [x] `Code/Source/solver/FE/Constraints/CoupledBCContext.h`
- [x] `Code/Source/solver/FE/Assembly/AssemblyContext.h`
- [x] `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- [x] `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- [x] `Code/Source/solver/FE/Assembly/FunctionalAssembler.h`
- [ ] `Code/Source/solver/FE/Assembly/FunctionalAssembler.cpp`
- [ ] `Code/Source/solver/FE/Forms/FormExpr.h`
- [ ] `Code/Source/solver/FE/Forms/FormExpr.cpp`
- [ ] `Code/Source/solver/FE/Forms/PointEvaluator.h`
- [ ] `Code/Source/solver/FE/Forms/PointEvaluator.cpp`
- [ ] `Code/Source/solver/FE/Forms/Vocabulary.h`
- [ ] `Code/Source/solver/FE/Forms/SymbolicDifferentiation.cpp`
- [ ] `Code/Source/solver/FE/Forms/FormCompiler.cpp`
- [ ] `Code/Source/solver/FE/Forms/JIT/JITValidation.cpp`
- [ ] `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
- [ ] `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
- [ ] `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
- [ ] `Code/Source/solver/FE/Analysis/ProblemAnalysisTypes.h`
- [ ] `Code/Source/solver/FE/CMakeLists.txt`

---

## Phase 1 — Core Contracts And Terminology

Establish the new public model before changing behavior.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateTypes.h`
  - [x] `enum class AuxiliaryStateScope { Global, Node, Cell, QuadraturePoint, BoundaryEntity }`
  - [x] `enum class AuxiliaryVariableKind { Differential, Algebraic }`
  - [x] `enum class AuxiliarySolveMode { Partitioned, Monolithic }`
  - [x] `enum class AuxiliaryHistoryMode { None, SingleStep, MultiStep }`
  - [x] `enum class AuxiliaryLayoutMode { FixedStride, Ragged }`
  - [x] `enum class AuxiliarySyncPolicy { None, OwnedOnly, OwnedAndGhost }`
  - [x] `enum class AuxiliaryTransferPolicy { None, Interpolate, CopyNearest, FormulationDefined }`
  - [x] `enum class AuxiliaryHistoryInterpolationPolicy { None, Linear, FormulationDefined }`
  - [x] `enum class AuxiliaryEntityOrdering { ByEntityThenComponent, ByComponentThenEntity }`
  - [x] `enum class AuxiliaryScheduleMode { SingleRate, Subcycled, Multirate }`
  - [x] `enum class AuxiliaryEventMode { None, EventHook, ActiveSetHook, ComplementarityHook }`
  - [x] `enum class AuxiliaryDerivativeSource { Symbolic, FiniteDifference, Analytic }`
  - [x] `enum class AuxiliarySecondDerivativeMode { None, Hessian, HessianVectorProduct, SelectedBlocks }`
  - [x] `enum class AuxiliaryInputRefreshPolicy { HoldLastSample, RefreshEachSubstep, FormulationDefined }`
  - [x] `enum class AuxiliarySubstepCommitPolicy { CommitAtEnd, CommitEachSubstep, FormulationDefined }`
  - [x] `enum class AuxiliaryOutputStateView { Committed, Work, Stage, NonlinearIterate, Substep }`
  - [x] `enum class AuxiliaryRegionKind { WholeDomain, CellSet, BoundarySet, MaterialIdSet, InterfaceSet, FormulationDefined }`
  - [x] `struct AuxiliaryDeploymentRegion`
    - [x] region kind
    - [x] stable region identity token
    - [x] optional region version or schema hash
    - [x] selector payload or formulation-defined handle
  - [x] `struct AuxiliaryStateSpec`
    - [x] stable `name`
    - [x] `size`
    - [x] `component_names`
    - [x] `scope`
    - [x] `deployment_region`
    - [x] `solve_mode`
    - [x] `layout_mode`
    - [x] `history_depth`
    - [x] `history_interpolation_policy`
    - [x] per-component `AuxiliaryVariableKind`
    - [x] `sync_policy`
    - [x] `transfer_policy`
    - [x] `schedule_mode`
    - [x] `event_mode`
    - [x] `derivative_policy`
    - [x] optional formulation-owned metadata key/value map
  - [x] `struct AuxiliaryDerivativePolicy`
    - [x] default derivative source for Jacobians
    - [x] default derivative source for requested second derivatives
    - [x] second-derivative mode
    - [x] analytic-override enabled flag
    - [x] optional finite-difference options
    - [x] optional AD options
  - [x] `struct AuxiliaryStepperSpec`
    - [x] stable method name or variant
    - [x] method options payload
    - [x] input refresh policy
    - [x] substep commit policy
    - [x] optional nonlinear/local solver options
  - [x] `struct AuxiliaryFailurePolicy`
    - [x] recoverable local retry policy
    - [x] time-step rejection behavior
    - [x] singular-Jacobian handling policy
    - [x] event-localization failure policy
    - [x] fatal-failure escalation policy
  - [x] `struct AuxiliaryStateBlockLayout`
    - [x] stable block id
    - [x] component stride
    - [x] entity count
    - [x] local storage size
    - [x] owned storage size
    - [x] history storage size
  - [x] `struct AuxiliaryStateStorageSummary`
  - [x] `struct AuxiliaryStateRegistrationOptions`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`
  - [x] remove boundary-specific module comment and terminology
  - [x] turn the header into the public umbrella API for generalized auxiliary state
  - [x] preserve the public type name `AuxiliaryState`
  - [x] stop describing the registration contract as scalar ODE-specific
- [x] `Code/Source/solver/FE/CMakeLists.txt`
  - [x] add phase-1 contract and type files to the FE build

### Checklist

- [x] Document that auxiliary block identity is string-based by unique block name within `FESystem`, with scope carried as metadata rather than part of the public key.
- [x] Document that deployment region is orthogonal to scope and defines which subset of the mesh or boundary receives auxiliary storage.
- [x] Document that mixed algebraic and differential variables may coexist within one auxiliary block.
- [x] Document that solve mode is fixed when deployed auxiliary instances are finalized during `system.setup()`.
- [x] Document that history depth is block-wide in phase 1.
- [x] Document that fixed-stride layout is the default and ragged layout is explicit, with canonical per-entity offsets and optional internal grouped fast paths.
- [x] Document the canonical entity ordering rules for each scope, using owned mesh or DOF ordering with explicit ghost append rules and `ByEntityThenComponent` as the default.
- [x] Document that auxiliary block names are the public stable handles and that numeric block ids and slot ids are internal setup-stable identifiers.
- [x] Document that all auxiliary blocks are owned by `FESystem` and must be finalized before `system.setup()` in phase 1.
- [x] Document that `schedule(...)` controls advancement timing and rate while `stepper(...)` controls the numerical method used when advancement executes.
- [x] Document that local stepper selection, local nonlinear solver options, substep input-refresh policy, and substep commit policy are valid only for `Partitioned` solve mode.
- [x] Document that outputs are evaluated against explicit state views such as committed, work, stage, nonlinear iterate, or substep state.
- [x] Document that `Monolithic` auxiliary unknowns use auxiliary-specific unknown layouts composed into a mixed system layout rather than reusing FE field DOF maps directly.
- [x] Document that same-time input-binding cycles are invalid and must move to `Monolithic` or `AuxiliaryOperator` coupling.
- [x] Document that cross-scope auxiliary-to-auxiliary bindings require explicit mapping providers or `AuxiliaryOperator`.
- [x] Document the phase-1 DAE capability boundary as local mixed ODE/algebraic systems with index-1-like structure or structurally nonsingular initialization paths.
- [x] Document that Jacobians are the phase-1 derivative requirement and that second derivatives remain extension-only unless a concrete algorithm requires them.
- [x] Document that derivative policy is configured at the model level in phase 1 and inherited by deployed instances.
- [x] Document the derivative-source precedence rules: analytic override first, then explicit model policy, then default symbolic generation when expressions are available.
- [x] Document how derivative policy behaves for lower-level residual implementations that do not carry symbolic expressions.
- [x] Document the canonical lowering contract from math-first auxiliary model rows to residual `FormExpr` trees used for symbolic differentiation:
  - [x] `ode(x, rhs)` lowers to `dot(x) - rhs`
  - [x] `algebraic(z, expr)` lowers to `expr`
  - [x] `residual(name, expr)` is taken as a raw residual row
- [x] Document the symbolic differentiation targets needed for auxiliary models:
  - [x] primary solve targets are auxiliary state, auxiliary time derivative, and coupled FE fields for monolithic models
  - [x] auxiliary-input sensitivities are optional
  - [x] auxiliary outputs are derived expressions and not primary solve targets

### Acceptance Criteria

- [x] A reader can understand the new public `AuxiliaryState` model without any reference to coupled boundary conditions.
- [x] All new core enums and specs are neutral and physics-agnostic.
- [x] The public API is not phrased in scalar-RHS ODE-only terms.
- [x] The public model distinguishes schedule selection from local stepper selection for staggered workflows.
- [x] The public model distinguishes storage scope from solve mode so `Global` scope cannot be confused with monolithic solve participation.
- [x] The public model distinguishes deployment region from storage scope so region-restricted auxiliary state is explicit rather than implied by masking.
- [x] The public model states clearly how Jacobians and optional second derivatives are obtained by default and how users override that behavior.
- [x] The public model states clearly what phase-1 DAE support does and does not guarantee.
- [x] The public model states clearly that symbolic derivatives come from lowered residual expressions rather than direct differentiation of storage objects.

---

## Phase 2 — Storage Abstraction And Scope-Specific Layout

Implement the storage model for global and bulk scopes with committed/work/history semantics.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.h`
  - [x] abstract base for storage backends
  - [x] read/write views for committed, work, and history data
  - [x] APIs for setup, reset-to-committed, commit, rollback, resize, and summary
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryBuffer.h`
  - [x] time-stamped snapshots
  - [x] interpolation API
  - [x] formulation-defined history access hooks
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryBuffer.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateIndexing.h`
  - [x] scope-specific entity indexing helpers
  - [x] node indexing helper
  - [x] cell indexing helper
  - [x] quadrature-point indexing helper
  - [x] boundary-entity indexing helper
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateIndexing.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`
  - [x] support multiple registered blocks with distinct scopes
  - [x] support block lookup by name
  - [x] support block-local and flattened views
  - [x] support history queries by block and by entity

### Checklist

- [x] Implement `Global` storage with one block-local entity.
- [x] Implement `Node` storage with owned and ghost node support.
- [x] Implement `Cell` storage with owned cell indexing.
- [x] Implement `QuadraturePoint` storage with per-cell local quadrature offsets.
- [x] Implement `BoundaryEntity` storage with stable entity indexing on boundary faces/edges as supported by the mesh layer.
- [x] Implement fixed-stride storage mode.
- [x] Implement ragged storage mode with per-entity offsets where formulation-defined local state size varies.
- [x] Support both flattened contiguous storage and block/entity views without copying.
- [x] Support committed/work/history semantics for all scopes.
- [x] Support configurable history depth per block.
- [x] Support time-stamped history snapshots.
- [x] Support history interpolation hooks for formulations that require off-grid history access.
- [x] Support reset, rollback, and commit operations at block granularity.
- [x] Support cheap read-only views for assembly and evaluation contexts.
- [x] Add storage summaries for debugging and test assertions.

### Acceptance Criteria

- [x] `AuxiliaryState` can register multiple blocks of different scopes in one system.
- [x] History handling works consistently across all scopes.
- [x] Storage views are deterministic and testable without requiring a specific physics module.

### Correction Steps For In-Progress Implementation

- [x] Rework storage allocation and indexing to honor deployment-region selection explicitly rather than allocating full-scope storage and relying on undocumented masking.
- [x] Add region-aware entity selection and storage summaries so diagnostics can report both scope and deployment region.
- [x] Ensure region-restricted `BoundaryEntity`, `Cell`, and `QuadraturePoint` storage preserve stable identity across setup, restart, repartition, and remesh workflows.
- [x] Add phase-2 build integration updates in `Code/Source/solver/FE/CMakeLists.txt` for storage, history, and indexing files if that was not done when the phase first landed.

---

## Phase 3 — Distributed Semantics, Ownership, Restart, And Transfer

Define the required semantics for distributed and mutable meshes up front so the subsystem is not host-only by accident.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Systems/SystemState.h`

### Checklist

- [x] Define ownership rules for `Node` scope under MPI.
- [x] Define ownership rules for `Cell` scope under MPI.
- [x] Define ownership rules for `BoundaryEntity` scope under MPI.
- [x] Define quadrature-point ownership rules in terms of owned cells only.
- [x] Define ownership and distribution rules for `Monolithic` auxiliary blocks.
- [x] Implement ghost synchronization APIs for scopes that need them.
- [x] Implement communication plans for nonlocal auxiliary coupling graphs.
- [x] Expose a formulation-selectable `AuxiliarySyncPolicy`.
- [x] Add pack/unpack APIs for checkpoint/restart without coupling to a specific I/O format.
- [x] Add remap/transfer hooks for mesh adaptation or repartitioning.
- [x] Define what happens to quadrature-point and boundary-entity data during remeshing.
- [x] Add debug validation that storage sizes and ownership maps are consistent after setup and sync.

### Acceptance Criteria

- [x] Distributed scopes have explicit, documented synchronization behavior.
- [x] Restart and repartitioning hooks exist at the infrastructure level.
- [x] No scope relies on implicit ordering assumptions that are undocumented.

### Correction Steps For In-Progress Implementation

- [x] Add restart-schema validation early in the lifecycle, including block names, component names, scope, deployment-region identity, ordering mode, history timestamps, and a minimal schema or version hash.
- [x] Define durable identity rules for `BoundaryEntity` and region-restricted entity sets across repartitioning and remeshing.
- [x] Validate restart payloads against deployment-region selection so region-restricted auxiliary blocks cannot silently restore onto incompatible meshes or boundary sets.
- [x] Add phase-3 build integration updates in `Code/Source/solver/FE/CMakeLists.txt` for manager and distributed-semantics files if that was not done when the phase first landed.

---

## Phase 4 — DAE-Capable Auxiliary Model Interface

Replace the scalar-RHS contract with a residual-based local model interface that can express ODE-like and DAE-like systems.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
  - [x] `struct AuxiliaryLocalContext`
    - [x] current time
    - [x] time step
    - [x] effective step size
    - [x] history views
    - [x] auxiliary input views
    - [x] parameter views
    - [x] optional formulation user data
  - [x] `struct AuxiliaryResidualRequest`
  - [x] `struct AuxiliaryJacobianRequest`
  - [x] `struct AuxiliaryHessianRequest`
  - [x] `struct AuxiliaryInitializationRequest`
  - [x] `struct AuxiliaryStructuralMetadata`
    - [x] differential vs algebraic row classification
    - [x] constraint grouping metadata
    - [x] optional DAE index / solver hint metadata
    - [x] local event metadata
  - [x] interface for residual evaluation `F(xdot, x, history, inputs, t, dt) = 0`
  - [x] optional Jacobian evaluation hooks
  - [x] optional Hessian or Hessian-vector evaluation hooks
  - [x] optional consistent-initialization hooks
  - [x] optional mass-like or differential-index metadata
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.h`
  - [x] derivative-source policy resolution
  - [x] lowered residual-expression derivative planning
  - [x] derivative-target enumeration for auxiliary variables, time-derivative variables, inputs, and coupled fields
  - [x] symbolic derivative generation entry points
  - [x] ~~`AutoAD` derivative generation entry points~~ AutoAD removed from enum; symbolic + FD are the supported paths.
  - [x] `FiniteDifference` derivative generation entry points
  - [x] analytic-override dispatch
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDerivativeProvider.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/ODEIntegrator.h`
- [x] `Code/Source/solver/FE/Systems/ODEIntegrator.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`

### Checklist

- [x] Define a block-local residual interface that can evaluate both differential and algebraic rows.
- [x] Allow purely algebraic blocks, purely differential blocks, and mixed blocks.
- [x] Carry structural metadata so advanced solvers can distinguish local differential rows, algebraic constraints, and solver hints.
- [x] Define whether `xdot` is always explicit in the interface or only for differential rows.
- [x] Define the monolithic semidiscrete contract explicitly: auxiliary models expose semidiscrete residuals, while `TimeIntegrator` owns stage state, iterate state, and the mapping from stage values to `xdot`.
- [x] Define the canonical lowering of math-first auxiliary rows into residual expressions, for example `ode(x, rhs)` lowering to `dot(x) - rhs`.
- [x] Define block-local Jacobian hooks with respect to `x`, `xdot`, auxiliary inputs, and coupled fields where present.
- [x] Define optional block-local second-derivative hooks, including Hessian-vector products or selected second-derivative blocks for methods that request them, including mixed derivatives needed by monolithic coupling paths when applicable.
- [x] Narrow the documented phase-1 DAE target to local mixed ODE/algebraic systems with index-1-like structure or structurally nonsingular initialization paths.
- [x] Define consistent initialization hooks for algebraic variables.
- [x] Define optional event functions and state-reset hooks.
- [x] Define optional nonsmooth or complementarity hooks so the interface can be extended beyond smooth DAEs.
- [x] Add an `AuxiliaryDerivativeProvider` path that can supply Jacobians and optional second derivatives from analytic overrides, symbolic generation, or `FiniteDifference`.
- [x] Define that analytic derivatives override automatic derivative generation when provided.
- [x] Define that expression-defined auxiliary models default to symbolic Jacobian generation and symbolic second-derivative generation when requested.
- [x] ~~Define explicit opt-in behavior for `AutoAD` and `FiniteDifference` derivative generation.~~ AutoAD removed — symbolic differentiation is the standard path. `FiniteDifference` remains as an explicit opt-in alternative.
- [x] Reuse and extend the existing FE symbolic-differentiation machinery so auxiliary residual expressions can be differentiated with respect to auxiliary-specific targets, not just FE trial/state/discrete field terminals.
- [x] Define setup-time derivative generation for expression-defined models so symbolic derivatives are built and simplified once per finalized auxiliary model instance rather than on each timestep.
- [x] Define behavior when symbolic differentiation is unavailable, such as lower-level residual implementations that only provide callbacks.
- [x] Implement at least one implicit residual-based stepper for phase 1.
- [x] Implement at least one explicit stepper as a specialization for suitable blocks.
- [x] Recast the current scalar ODE path as a wrapper or adapter on the new stepper interface.
- [x] Support block-local substepping independent of the main PDE step.
- [x] Support per-block method selection and stepper options.
- [x] Define residual/Jacobian workspace reuse to avoid per-call allocation churn.
- [x] Define derivative workspace reuse and caching so symbolic, AD, or finite-difference derivative generation does not cause avoidable allocation churn.
- [x] Define caching keys for lowered residual expressions and generated derivative expressions so repeated setup of equivalent auxiliary models can reuse work where appropriate.
- [x] Define thread-safety rules for residual, Jacobian, and derivative workspaces: read-only caches after setup and thread-local or caller-owned scratch during evaluation.

### Acceptance Criteria

- [x] The public contract can express a mixed differential/algebraic local system without API workarounds.
- [x] ODE-like updates compile and run through the new interface without requiring a separate conceptual model.
- [x] The new interface is not tied to scalar variables or one-slot updates.
- [x] Implicit methods can obtain Jacobians through analytic or automatic derivative infrastructure without requiring manual Jacobian code for math-first models.
- [x] Optional second-derivative information can be requested without forcing every model to provide or assemble full Hessians.
- [x] Expression-defined auxiliary models can obtain symbolic Jacobians by differentiating lowered residual expressions at setup time and reusing the compiled/cached result at runtime.

### Correction Steps For In-Progress Implementation

- [x] Audit the in-progress interface so `Monolithic` auxiliary residuals are semidiscrete and not silently mixing fully time-discretized residuals with `TimeIntegrator`-owned `xdot` semantics.
- [x] Reclassify any phase-1 implementation work that treats high-index or structurally difficult DAE systems as in-scope; keep those paths behind later-phase services.
- [x] Demote any in-progress Hessian or second-derivative work from phase-1 critical path unless a concrete solver path already depends on it.
- [x] Add explicit thread-safety review for workspace ownership, lazy derivative caches, and any runtime mutation under threaded evaluation.
- [x] Add phase-4 build integration updates in `Code/Source/solver/FE/CMakeLists.txt` for model, derivative-provider, and stepper files if that was not done when the phase first landed.

---

## Phase 4A — Declarative Authoring And Deployment API

Add a user-friendly API layer that stays mathematically readable for common models while lowering to the same backend contracts as the lower-level interface.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- [x] `Code/Source/solver/FE/CMakeLists.txt`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp`

### Checklist

- [x] Define `AuxiliaryModelSignature` for reusable named inputs, outputs, and optional parameters.
- [x] Define compatibility rules so auxiliary models are considered interchangeable through signature-compatible inputs, outputs, and optional parameters rather than internal state names.
- [x] Define optional parameter descriptors in the signature so they can be bound through literals or provider-driven values when needed.
- [x] Define a math-first `AuxiliaryModel(...)` builder surface for common models.
- [x] Support `model.input(name)`, `model.state(name[, kind])`, `model.param(name)`, `model.derived(name, expr)`, `model.ode(state, rhs)`, `model.algebraic(state, expr)`, `model.residual(name, expr)`, and `model.output(name, expr)`.
- [x] Add time-like helpers such as `dot(x)`, `prev(x[, k])`, and `history(x, ...)` for model definitions.
- [x] Define how builder-level row declarations lower to residual `FormExpr` expressions used for evaluation and symbolic differentiation.
- [x] Allow expression-defined models to optionally provide analytic Jacobians and optional analytic second derivatives that override automatic derivative generation.
- [x] Add public derivative-policy configuration through `.derivatives(...)` on model definitions. Do not expose instance-level derivative-policy selection in phase 1 beyond row-level analytic overrides.
- [x] Define `use(model)` as the deployment handle that returns a formulation-bound auxiliary instance.
- [x] Support deployment through domain and boundary contexts using the same instance type. *(Actual API: `use(model).scope(BoundaryEntity).region(...)` + `system.deploy(...)`. No `system.boundary(name).use(model)` context API — boundary-local deployment uses scope + region setters on `AuxiliaryDeployedInstance`.)*
- [x] Support instance-level configuration through `.name(...)`, `.scope(...)`, `.region(...)`, `.solveMode(...)`, `.schedule(...)`, `.stepper(...)`, `.bind(...)`, `.param(...)`, and `.initialize(...)`.
- [x] Define a public local-stepper API for staggered `Partitioned` instances through `.stepper(...)`.
- [x] Allow stepper configuration to include method options, substep count or rate settings where appropriate, input-refresh policy, and substep commit/rollback policy.
- [x] Validate that local stepper selection is legal for the chosen solve mode and reject `Partitioned`-style local stepper APIs on `Monolithic` instances.
- [x] Ensure auxiliary model deployment is the canonical public workflow and that block registration remains backend-only infrastructure.
- [x] Ensure both the high-level builder and the lower-level residual/Jacobian interface lower to the same `AuxiliaryStateModel` representation.
- [x] Keep `AuxiliaryModel` local/global-0D or per-entity in scope and route genuinely nonlocal couplings through `AuxiliaryOperator`.
- [x] Keep local `Monolithic` auxiliary models on the same `AuxiliaryModel` plus deployment workflow; require `AuxiliaryOperator` only for genuinely nonlocal couplings or advanced custom operator cases.
- [x] Define validation and diagnostics for undeclared states, duplicate names, unbound inputs, missing outputs, and unused declarations.
- [x] Define explicit deployment guidance that lumped boundary models use `Global` scope with boundary reductions while one-model-per-face or per-edge boundary models use `BoundaryEntity` scope.
- [x] Define deployment guidance for region-restricted auxiliary blocks so users can deploy node, cell, quadrature-point, or boundary-entity state on selected domains, interfaces, materials, or named boundary sets without implicit full-domain masking.
- [x] Ensure auxiliary instance outputs can be consumed naturally in both domain and boundary residual forms.
- [x] Define diagnostics in terms of model names, instance names, input names, and output names rather than internal slot ids.
- [x] Define diagnostics for invalid stepper-selection combinations, such as explicit steppers on unsupported models or local stepper APIs used with `Monolithic` instances.
- [x] Define diagnostics for derivative-policy conflicts, unavailable symbolic derivatives, unsupported Hessian requests, and invalid analytic derivative shapes.
- [x] Define diagnostics when a user attempts to encode nonlocal coupling directly in `AuxiliaryModel` rather than through `AuxiliaryOperator`.

### Acceptance Criteria

- [x] A new user can define a local auxiliary ODE/DAE-like model without directly interacting with block registration, slot ids, or storage layouts.
- [x] An advanced user can bypass the builder and implement the lower-level residual/Jacobian interface without entering a different conceptual subsystem.
- [x] The math-first builder is expressive enough for common EP-like and metabolism-like local models.
- [x] A new user can choose a staggered local time-advancement method for a `Partitioned` auxiliary instance without touching backend stepper registration.
- [x] A new user can rely on symbolic derivatives by default for expression-defined models while still being able to override with analytic or `FiniteDifference` policies.
- [x] The canonical builder/deployment surface uses one set of names without parallel aliases for the primary workflow.

---

## Phase 5 — Generalized Auxiliary Input Infrastructure

Introduce a neutral concept for externally supplied values that auxiliary-state models and FE forms can reference.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInputRegistry.cpp`

### Checklist

- [x] Define `AuxiliaryInputSpec` with stable name, size, component names, producer type, and update schedule.
- [x] Define input providers for:
  - [x] boundary functional reductions,
  - [x] formulation callback providers,
  - [x] parameter-derived inputs,
  - [x] direct user-data inputs,
  - [x] future field-reduction providers.
- [x] Support optional signature parameters bound either to literals or to provider-driven values when parameter values depend on time, user data, or formulation state.
- [x] Add explicit field-binding provider types such as `SampledStateField`, `CoupledField`, `CellAverage`, `CellSample`, `DomainAverage`, and `DomainIntegral`.
- [x] Add explicit boundary-binding provider types such as `SampledBoundaryTrace`, `CoupledBoundaryTrace`, `SampledBoundaryReduction`, and `CoupledBoundaryReduction`.
- [x] Define stage-selection semantics when a sampled field input is taken from FE system state, such as committed state, previous step, current iterate, or stage state.
- [x] Define which field-binding providers are valid for each auxiliary scope.
- [x] Require explicit reductions for `Global` scope rather than allowing implicit field-to-scalar collapse.
- [x] Define how `CoupledField` lowers into symbolic field dependencies for `Monolithic` auxiliary blocks instead of being treated as a frozen cached input. *(Two paths: (1) chain-rule composition `dF/dI × dI/du` through the FE-quantity handle system (all scopes, all spaces), (2) direct symbolic `dF/d(fields)` via `DiffTarget::Field` with per-component differentiation for models with `DiscreteField`/`StateField` nodes (Node scope, C0-continuous nodal Lagrange spaces including scalar H1 and Product/Vector H1, scalar/vector/tensor fields up to 9 components). Setup rejects: non-Node scopes, non-C0 continuity (L2/DG, H(div), H(curl), C1), fields exceeding MAX_FIELD_VALUE_COMPONENTS. Per-component differentiation handles `component(u, i)` (Kronecker delta), `inner(u, u)` (2*u_k), and product-rule compositions. Non-Lagrange spaces use the mediated input path.)*
- [x] Define which boundary-binding providers are valid for `Global` vs `BoundaryEntity` scopes and for `Partitioned` vs `Monolithic` solve modes.
- [x] Define how sampled and coupled boundary reductions lower for lumped boundary models such as RCR-like outlet conditions.
- [x] Define how boundary traces provide pointwise or entity-local values for `BoundaryEntity` auxiliary blocks.
- [x] Support auxiliary inputs that are produced from other auxiliary model outputs.
- [x] Define stable slot assignment for auxiliary inputs.
- [x] Define input invalidation and caching rules within nonlinear iterations and time steps.
- [x] Define how input refresh policies interact with local substepping, including hold-last-sample versus refresh-each-substep behavior for `Partitioned` instances.
- [x] Support block-valued auxiliary inputs, not just scalars.
- [x] Support MPI reductions for global inputs produced from distributed mesh traversals.
- [x] Support provider dependency ordering if one input depends on another input.
- [x] Support debug inspection of current auxiliary input values by name and slot.

### Acceptance Criteria

- [x] Boundary functionals are only one input-provider implementation, not the core abstraction.
- [x] Auxiliary-state models can depend on neutral auxiliary inputs rather than boundary-specific symbols.
- [x] Auxiliary model definitions remain field-agnostic and only deployment-time bindings introduce FE field semantics.
- [x] Boundary-coupled auxiliary models use the same binding model as other consumers, with boundary traces and reductions expressed as explicit providers rather than special-case ownership.

### Correction Steps For In-Progress Implementation

- [x] Build and validate an explicit provider DAG for auxiliary inputs and provider-backed parameters.
- [x] Add cycle detection with a hard error for same-time input-binding cycles rather than allowing hidden algebraic loops.
- [x] Document and enforce that sampled providers are exogenous data for a given evaluation context, while same-iterate coupled dependence must move to `Monolithic` coupling or `AuxiliaryOperator`.
- [x] Add explicit cross-scope auxiliary-to-auxiliary mapping providers, such as reductions, traces, or projections, or require such couplings to use `AuxiliaryOperator` when no one-way sampled mapping is declared.
- [x] Add phase-5 build integration updates in `Code/Source/solver/FE/CMakeLists.txt` for input-registry files if that was not done when the phase first landed.

---

## Phase 6 — Partitioned And Monolithic Solve Modes And Auxiliary Operators

Add support for auxiliary blocks that participate as first-class unknowns in assembled systems.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryCouplingGraph.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryCouplingGraph.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryUnknownLayout.h` *(Actual location: `AuxiliaryUnknownLayout` struct embedded in `AuxiliaryOperatorRegistry.h:86`, not a standalone file.)*
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryUnknownLayout.cpp` *(Not a standalone file — functionality is in `AuxiliaryOperatorRegistry.cpp`.)*
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorBuilder.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorBuilder.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisTypes.h`
- [x] `Code/Source/solver/FE/CMakeLists.txt`

### Checklist

- [x] Define `Partitioned` vs `Monolithic` solve semantics in the manager and system lifecycle.
- [x] Add registration for monolithic auxiliary variables as first-class unknowns.
- [x] Define auxiliary-specific unknown layouts for `Monolithic` auxiliary blocks rather than reusing FE field DOF maps directly.
- [x] Define how `FESystem` composes field unknown layouts and auxiliary-specific unknown layouts into one mixed system layout for assembly and solves.
- [x] Reuse FE mesh or DOF metadata only for ownership and ordering where appropriate, without modeling auxiliary blocks as FE fields.
- [x] Define `AuxiliaryOperator` as the advanced public authoring surface for genuinely nonlocal couplings and custom mixed sparse operators.
- [x] Define the canonical `AuxiliaryOperator(...)` builder surface with `.name(...)`, `.source(...)`, `.target(...)`, `.topology(...)`, `.residual(...)`, `.jacobian(...)`, optional `.mass(...)` or `.transfer(...)`, and optional `.derivatives(...)`.
- [x] Add auxiliary-only operator registration for residual, Jacobian, and mass-like blocks.
- [x] Add field-to-auxiliary, auxiliary-to-field, and auxiliary-to-auxiliary coupling operator registration.
- [x] Define the public `AuxiliaryOperator` authoring contract in terms of declared sources, targets, topology or coupling-graph metadata, and residual or Jacobian-style contributions.
- [x] Define how `AuxiliaryOperator` integrates with mixed field and auxiliary assembly without forcing local or per-entity models off the declarative `AuxiliaryModel` workflow.
- [x] Add sparse nonlocal auxiliary coupling graph metadata with distributed communication support.
- [x] Keep nonlocal coupling representation in `AuxiliaryOperator` and out of the declarative `AuxiliaryModel` surface.
- [x] Add monolithic mixed assembly paths that include FE fields and monolithic auxiliary unknowns in one assembled system.
- [x] Expose solver-facing block-layout metadata for mixed field/auxiliary systems.
- [x] Ensure monolithic auxiliary blocks can still consume auxiliary inputs and history services.

### Acceptance Criteria

- [x] `Monolithic` auxiliary blocks can participate as first-class unknowns in assembled residual/Jacobian systems.
- [x] Nonlocal auxiliary couplings are not restricted to local per-entity systems or boundary reductions.
- [x] Nonlocal couplings are expressed through `AuxiliaryOperator` rather than overloading the local `AuxiliaryModel` API.
- [x] Local `Monolithic` auxiliary models can still be authored through the same declarative `AuxiliaryModel` surface and deployed through `use(model)` with coupled bindings.
- [x] The advanced nonlocal path has a documented public `AuxiliaryOperator` API rather than only backend registry concepts.
- [x] `Monolithic` auxiliary unknowns participate in one mixed system layout without being forced into FE field DOF-map semantics.

### Correction Steps For In-Progress Implementation

- [x] Move `AuxiliaryUnknownLayout` out of implied ownership and give it an explicit implementation deliverable, tests, and build-system integration in the monolithic phase.
- [x] Audit any in-progress monolithic implementation that still piggybacks on FE DOF-map assumptions instead of the auxiliary-specific mixed-layout contract.
- [x] Add phase-6 build integration updates in `Code/Source/solver/FE/CMakeLists.txt` for unknown-layout, coupling-graph, operator-builder, and operator-registry files if that was not done when the phase first landed.

---

## Phase 7 — Symbolic Vocabulary Generalization

Generalize the symbolic layer immediately and remove boundary-specific coupling terminology from core FE forms infrastructure.

Delivery note: implement this phase in two slices.
- [x] Phase 7A: minimal symbolic slice required before dependent phases, including auxiliary input/output terminals, lowering targets for auxiliary residuals and outputs, and derivative targets needed by `AuxiliaryModel`, `AuxiliaryOperator`, and the derivative provider.
- [x] Phase 7B: remaining parser, scanner, diagnostics, and full vocabulary-plumbing migration.

### Files to Modify

- [x] `Code/Source/solver/FE/Forms/FormExpr.h`
- [x] `Code/Source/solver/FE/Forms/FormExpr.cpp`
- [x] `Code/Source/solver/FE/Forms/PointEvaluator.h`
- [x] `Code/Source/solver/FE/Forms/PointEvaluator.cpp`
- [x] `Code/Source/solver/FE/Forms/Vocabulary.h`
- [x] `Code/Source/solver/FE/Forms/SymbolicDifferentiation.cpp`
- [x] `Code/Source/solver/FE/Forms/FormCompiler.cpp`
- [x] `Code/Source/solver/FE/Forms/JIT/JITValidation.cpp`
- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- [x] `Code/Source/solver/FE/Analysis/FormExprScanner.h`
- [x] `Code/Source/solver/FE/Analysis/FormExprScanner.cpp`

### Checklist

- [x] Introduce `AuxiliaryInputSymbol` and `AuxiliaryInputRef`.
- [x] Introduce `AuxiliaryOutputSymbol` and `AuxiliaryOutputRef`.
- [x] Keep `AuxiliaryStateSymbol` and `AuxiliaryStateRef`, but ensure their semantics are library-wide rather than coupled-boundary-specific.
- [x] Deprecate `BoundaryIntegralSymbol` and `BoundaryIntegralRef` in favor of `AuxiliaryInputSymbol` and `AuxiliaryInputRef`.
- [x] Decide whether `BoundaryFunctionalSymbol` remains as a producer-side expression helper or is fully deprecated from the forms vocabulary.
- [x] Update helper constructors on `FormExpr`.
- [x] Add math-first vocabulary helpers in `FE/Forms/Vocabulary.h` for `AuxiliaryInput(...)`, `AuxiliaryOutput(...)`, and any required time-like auxiliary helpers.
- [x] Update `PointEvalContext` to replace `coupled_integrals` / `coupled_aux` with neutral names.
- [x] Update `PointDualSeedContext` naming and derivative semantics to match the new auxiliary-input vocabulary.
- [x] Update symbolic differentiation to recognize the new terminals.
- [x] Extend symbolic differentiation support so math-first auxiliary model expressions can generate Jacobians and optional second-derivative information by default.
- [x] Add auxiliary-specific symbolic differentiation targets and APIs so lowered auxiliary residual expressions can be differentiated with respect to auxiliary states, auxiliary time-derivative states, auxiliary inputs when needed, and coupled FE fields for monolithic models.
- [x] Update form installation and validation to treat auxiliary terminals as first-class generic dependencies.
- [x] Make auxiliary outputs the preferred formulation-facing coupling surface and keep raw auxiliary state access as an advanced path.
- [x] Ensure auxiliary outputs can be consumed naturally in boundary integrals as well as domain integrals.
- [x] Define the symbolic contract for auxiliary outputs, including whether output expressions may reference `x`, `xdot`, history, inputs, parameters, and coupled FE fields and how output derivatives are exposed for monolithic assembly.
- [x] Update stringification, parser hooks, and tests so the new names appear in diagnostics.

### Acceptance Criteria

- [x] A form can refer to `AuxiliaryStateRef` and `AuxiliaryInputRef` without any boundary-specific terminology in the API.
- [x] A form can refer to `AuxiliaryOutputRef` as a first-class symbolic term.
- [x] Existing boundary-based use cases can be expressed using the new vocabulary.
- [x] The symbolic layer clearly separates auxiliary state, auxiliary inputs, and auxiliary outputs.

### Correction Steps For In-Progress Implementation

- [x] If implementation landed after downstream phases, split out and validate the minimal Phase 7A symbolic slice so earlier phases depend only on generalized auxiliary terminals rather than legacy coupled-boundary names.
- [x] Ensure auxiliary-output symbolic semantics are explicit rather than inferred, especially for derivative participation under monolithic assembly.

---

## Phase 8 — Assembly, Evaluator, And JIT Plumbing Migration

Thread the generalized auxiliary vocabulary through interpreter and JIT execution.

### Files to Modify

- [x] `Code/Source/solver/FE/Assembly/AssemblyContext.h`
- [x] `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- [x] `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- [x] `Code/Source/solver/FE/Assembly/FunctionalAssembler.h`
- [x] `Code/Source/solver/FE/Assembly/FunctionalAssembler.cpp`
- [x] `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
- [x] `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
- [x] `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
- [x] `Code/Source/solver/FE/Forms/FormKernels.cpp`

### Checklist

- [x] Replace `setCoupledValues(...)` with a neutral auxiliary context setter.
- [x] Rename `coupled_integrals` to `auxiliary_inputs` throughout assembly and JIT ABI structs.
- [x] Rename `coupled_aux` to `auxiliary_state` throughout assembly and JIT ABI structs.
- [x] Rename derivative-seed storage such as `coupled_aux_dseed` to neutral auxiliary-state derivative names.
- [x] Update kernel argument packing tests to validate the renamed fields.
- [x] Update interpreter evaluators and JIT code generation so they load from the new fields.
- [x] Preserve ABI compatibility only if required internally; otherwise update all call sites in one migration.
- [x] Ensure functional assemblers can consume auxiliary inputs and auxiliary state without coupled-boundary ownership.
- [x] Define immutable runtime read views and thread-local or caller-owned scratch ownership for interpreter and JIT auxiliary evaluation paths.
- [x] Ensure no lazy derivative-generation or mutable symbolic cache population occurs during threaded evaluation.

### Acceptance Criteria

- [x] Interpreter and JIT paths agree on the new auxiliary vocabulary.
- [x] No core assembler API uses `coupled_*` naming for auxiliary data.

### Correction Steps For In-Progress Implementation

- [x] Audit JIT and interpreter execution for thread safety, scratch ownership, and runtime cache mutation.
- [x] Add any missing auxiliary-evaluation scratch plumbing needed to keep threaded assembly free of shared mutable state.

---

## Phase 9 — FESystem Ownership And Lifecycle Integration

Move setup, scheduling, and advancement under `FESystem`.

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Systems/SystemState.h`
- [x] `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- [x] `Code/Source/solver/FE/Systems/TimeIntegrator.h`
- [x] `Code/Source/solver/FE/TimeStepping/TimeLoop.h`

### Checklist

- [x] Add `AuxiliaryStateManager` as a first-class `FESystem` member.
- [x] Add `AuxiliaryOperatorRegistry` and any required coupling-graph ownership as first-class `FESystem` members.
- [x] Keep low-level registration APIs on `FESystem` and the manager as internal infrastructure, not the canonical user-facing workflow.
- [x] Add formulation-facing auxiliary deployment/discovery APIs that collect declared auxiliary model instances before `system.setup()`.
- [x] Add boundary-context deployment/discovery APIs that collect auxiliary model instances declared on named boundaries before `system.setup()`.
- [x] Finalize auxiliary layouts during `system.setup()` from discovered deployed auxiliary instances.
- [x] Add `beginTimeStep()`, `prepareForAssembly()`, `advanceAuxiliaryState()`, `rollbackAuxiliaryState()`, and `commitTimeStep()` hooks.
- [x] Add setup-time hooks that finalize lowered auxiliary residual expressions, build any requested symbolic derivatives, and cache derivative artifacts before time stepping begins.
- [x] Add checkpoint/rollback APIs usable by subcycling, multirate advancement, and failed nonlinear iterations.
- [x] Define when auxiliary inputs are refreshed relative to PDE assembly and nonlinear iterations.
- [x] Define when auxiliary state is advanced relative to PDE stages and substeps.
- [x] Define how `Partitioned` and `Monolithic` auxiliary blocks coexist in one step schedule.
- [x] Add `SingleRate`, `Subcycled`, and `Multirate` scheduling support at the auxiliary-block level.
- [x] Define how instance-selected local steppers are dispatched for `Partitioned` auxiliary blocks.
- [x] Define how local stepper options, substep input-refresh policies, and substep commit policies are honored in the lifecycle.
- [x] Define the boundary between partitioned local time advancement and monolithic assembled time discretization.
- [x] Define explicitly how `TimeIntegrator` provides stage state, nonlinear iterate state, and `xdot` views to `Monolithic` auxiliary residuals during assembled solves.
- [x] Define whether monolithic auxiliary residuals are always semidiscrete in phase 1 and reject fully time-discretized residual callbacks unless they use a separate advanced extension path.
- [x] Define explicit output-evaluation semantics for committed, work, stage, nonlinear-iterate, and substep contexts, including cache invalidation and derivative participation rules.
- [x] Define event-detection and event-handling hooks in the time-stepping lifecycle.
- [x] Define failure-handling semantics for local implicit solve failure, singular Jacobians, event-localization failure, and subcycle divergence, including retry, rollback, and global time-step rejection behavior.
- [x] Expose read-only auxiliary-state and auxiliary-input views in `SystemStateView` or a related assembly-state object.
- [x] Expose read-only auxiliary-output views or output-evaluation services with explicit state-view selection.
- [x] Support formulation-selected scheduling for auxiliary advancement.
- [x] Add analysis/instrumentation hooks so the system can report auxiliary blocks and inputs.
- [x] Add analysis/instrumentation hooks so the system can report derivative-source policy, analytic override usage, and whether second-derivative information is available.

### Acceptance Criteria

- [x] `FESystem` owns the lifecycle of auxiliary state and auxiliary inputs.
- [x] The lifecycle supports checkpoint/rollback and multirate scheduling hooks.
- [x] Time-stepping and assembly code paths can access auxiliary data without going through `CoupledBoundaryManager`.
- [x] New formulations do not need to call manual auxiliary block-registration APIs in normal use.
- [x] Boundary-condition implementations can declare auxiliary models through the same deployment/discovery workflow used elsewhere in FE.

### Correction Steps For In-Progress Implementation

- [x] Rework lifecycle semantics if outputs currently observe an implicit or ambiguous state view; make output evaluation state explicit in assembly, residual evaluation, and post-step usage.
- [x] Rework failure propagation so local auxiliary solve failures can trigger well-defined retry, rollback, global step rejection, or fatal-stop paths rather than ad hoc error handling.
- [x] Audit the monolithic path to ensure `TimeIntegrator` owns the `xdot` contract consistently across stage, iterate, and assembled solve contexts.

---

## Phase 10 — Coupled Boundary Migration And Deprecation

Convert coupled boundary conditions into one client of the generalized subsystem.

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.h`
- [x] `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
- [x] `Code/Source/solver/FE/Constraints/CoupledBCContext.h`
- [x] `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`

### Checklist

- [x] Remove unique auxiliary-state storage and advancement ownership from `CoupledBoundaryManager`.
- [x] Make `CoupledBoundaryManager` register boundary-functional providers with the new auxiliary-input registry.
- [x] Make `CoupledBoundaryManager` consume `AuxiliaryStateManager` services from `FESystem`.
- [x] Update `CoupledBCContext` to expose neutral auxiliary-input and auxiliary-state views, or replace it with a more general context type.
- [x] Provide ergonomic replacements for legacy coupled-boundary workflows using boundary-scoped deployment. *(Actual API: `use(model).scope(BoundaryEntity).region(...)` + `system.deploy(...)` with `boundaryIntegral()` handle-returning input registration. No `system.boundary(name)` context API.)*
- [x] Mark `CoupledBoundaryManager::addAuxiliaryState(...)` deprecated.
- [x] Mark old boundary-integral symbolic helpers deprecated.
- [x] Ensure deprecated paths forward directly to the new subsystem rather than maintaining separate logic.
- [x] Add clear deprecation messages pointing users to the new API.
- [x] Remove any old assumptions that auxiliary state is scalar-only or boundary-only.
- [x] Document the migration path from legacy lumped boundary auxiliary state to `Global`-scope deployed models with boundary reductions.
- [x] Document the migration path from per-boundary-entity auxiliary logic to `BoundaryEntity`-scope deployed models with boundary traces.

### Acceptance Criteria

- [x] Coupled boundary code compiles and runs on top of the new subsystem.
- [x] No FE core behavior depends on the old auxiliary-state ownership model.
- [x] Deprecated wrappers are thin and temporary.

---

## Phase 11 — Analysis And Metadata Updates

Bring the analysis subsystem and metadata contracts in line with the generalized model.

### Files to Modify

- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisTypes.h`
- [x] `Code/Source/solver/FE/Analysis/FormulationRecord.h`
- [x] `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`

### Checklist

- [x] Introduce `VariableKind::AuxiliaryInput` if not already present.
- [x] Introduce metadata that distinguishes `Partitioned` from `Monolithic` auxiliary blocks.
- [x] Add auxiliary scope metadata to variable descriptors.
- [x] Add deployment-region metadata to variable descriptors and formulation records.
- [x] Update contribution descriptors so auxiliary-state and auxiliary-input couplings are no longer forced into `CoupledBoundary` semantics.
- [x] Add contribution descriptors for field-to-auxiliary, auxiliary-to-field, and auxiliary-to-auxiliary couplings.
- [x] Update analysis records to distinguish auxiliary-state dependencies from auxiliary-input dependencies.
- [x] Ensure formulation records can report dependencies on bulk-scoped auxiliary state.
- [x] Update any reporting that still assumes boundary functional plus scalar auxiliary state as the only non-FE unknowns.

### Acceptance Criteria

- [x] Analysis reports describe the new `AuxiliaryState` model accurately.
- [x] No analysis type uses boundary-specific language as the default auxiliary-state description.

---

## Phase 12 — Test Plan

Testing must be broad because the work crosses storage, symbolic evaluation, distributed semantics, and migration.

### Files to Create

- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateTypes.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateStorage.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryHistoryBuffer.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateIndexing.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateManager.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryModelBuilder.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryDerivativeProvider.cpp` *(Actual: derivative provider tests live in `test_AuxiliaryStateModel.cpp` — `AuxiliaryDerivativeProvider.*` suite, 37+ tests.)*
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryStateStepper.cpp` *(Actual: stepper tests live in `test_AuxiliaryStateModel.cpp` — `AuxiliaryStateStepper.*` suite.)*
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryInputRegistry.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryCouplingGraph.cpp` *(Actual: coupling graph tests live in `test_AuxiliaryOperators.cpp` — `AuxiliaryCouplingGraph.*` suite, 7 tests.)*
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryUnknownLayout.cpp` *(Actual: tested in `test_AuxiliaryOperators.cpp:294`, not a standalone file.)*
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryOperatorBuilder.cpp` *(Actual: operator builder tests live in `test_AuxiliaryOperators.cpp` — `AuxiliaryOperatorBuilder.*` suite.)*
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryOperatorRegistry.cpp` *(Actual: operator registry tests live in `test_AuxiliaryOperators.cpp` — `AuxiliaryOperatorRegistry.*` suite.)*
- [x] `Code/Source/solver/FE/Tests/Unit/Forms/test_AuxiliaryVocabulary.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Assembly/test_AuxiliaryContextPacking.cpp` *(Actual: context packing tests live in `test_AuxiliaryModelBuilder.cpp` — `EndToEnd_MonolithicAssembly*` tests.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateGlobal.cpp` *(Actual: integration coverage lives in `test_AuxiliaryModelBuilder.cpp` and `test_BoundaryIntegralInput.cpp` unit tests, not separate integration test files.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateNodeScope.cpp` *(Same — tested in unit tests.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateCellScope.cpp` *(Same.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateQuadraturePointScope.cpp` *(Same.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateBoundaryEntityScope.cpp` *(Same.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateDAE.cpp` *(Actual: DAE coverage in `test_AuxiliaryStateModel.cpp` — `MixedDAEMetadata`, `PureAlgebraicMetadata`, `SymbolicJacobian_2D_MixedDAE`, `dFdxdot_MixedRows`; and `test_AuxiliaryModelBuilder.cpp` — `BuildMixedDAE`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateMonolithic.cpp` *(Actual: monolithic coverage in `test_AuxiliaryModelBuilder.cpp` — `EndToEnd_MonolithicAssembly`, `EndToEnd_MonolithicAssembly_WithInputs`, `EndToEnd_MonolithicLayout_WithFields`; and `test_BoundaryIntegralInput.cpp` — `MixedJacobianBlockFDVerification`, `DFDInputsSymbolicGeneration`, `SymbolicGradientMatchesFD`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateNonlocalCoupling.cpp` *(Actual: coupling graph coverage in `test_AuxiliaryOperators.cpp` — `AuxiliaryCouplingGraph.*` (7 tests), `FullMixedCouplingScenario`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateRaggedLayout.cpp` *(Actual: ragged layout coverage in `test_AuxiliaryStateStorage.cpp` — `RaggedLayout`, `GatherScatter_Ragged`; and `test_AuxiliaryModelBuilder.cpp` — `DeployRejectsRaggedLayout`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateHistoryInterpolation.cpp` *(Actual: history coverage in `test_AuxiliaryHistoryBuffer.cpp` — `LinearInterpolation`, `LinearInterpolationAtBoundary`, `FormulationDefinedInterpolation`, `InterpolationThrowsWhenDisabled`; and `test_AuxiliaryModelBuilder.cpp` — `EndToEnd_Multirate_BDF2History`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateEvents.cpp` *(Actual: event/hook coverage in `test_AuxiliaryStateModel.cpp` — `OptionalHooksDefaultToFalse`; `test_AuxiliaryStateManager.cpp` — `GhostSyncHookCalledForOwnedAndGhost`, `InvalidateSetupClearsHooks`; `test_AuxiliaryStateTypes.cpp` — `EventModeEnumValues`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateMultirate.cpp` *(Actual: multirate coverage in `test_AuxiliaryModelBuilder.cpp` — `EndToEnd_MultirateTwoBlocks`, `EndToEnd_MultirateSchedulerDispatch`, `EndToEnd_Multirate_EntityLocalInputs`, `EndToEnd_Multirate_BackwardEulerHistory`.)*
- [x] `Code/Source/solver/FE/Tests/Integration/test_AuxiliaryStateCoupledBoundaryMigration.cpp` *(Actual: coupled-BC migration coverage in `test_CoupledBoundaryManager.cpp` — `AuxiliaryStateResetsEachPrepareUntilCommit`; and `test_BoundaryIntegralInput.cpp` — `CoupledBindingOnMonolithicAccepts`, `MonolithicCoupledBindingStructure`.)*
- [x] MPI coverage for node and boundary-entity synchronization tests

### Checklist

- [x] Unit-test the new enums, specs, and validation rules.
- [x] Unit-test deployment-region specs, identity tokens, and region-restricted block validation.
- [x] Unit-test committed/work/history semantics for every scope.
- [x] Unit-test fixed-stride and ragged layout modes.
- [x] Unit-test entity indexing for every scope.
- [x] Unit-test slot assignment and block lookup.
- [x] Unit-test auxiliary-specific unknown layout construction and mixed system layout composition for monolithic auxiliary blocks.
- [x] Unit-test history interpolation and time-stamped snapshot access.
- [x] Unit-test residual-based DAE stepping on a small mixed differential/algebraic block.
- [x] Unit-test explicit and implicit stepper paths through the new interface.
- [x] Unit-test derivative-source policy precedence: analytic override, default symbolic generation, and explicit `FiniteDifference`.
- [x] Unit-test lowering from math-first auxiliary model rows into residual `FormExpr` trees.
- [x] Unit-test auxiliary-specific symbolic differentiation targets for `x`, `xdot`, auxiliary inputs, and coupled fields where applicable. *(Symbolic `dF/dx`, `dF/dxdot`, `dF/dinputs`, and `dF/d(fields)` all tested. Direct field derivatives via `DiffTarget::Field` with `DiscreteField`/`StateField` matching — 4 tests: linear, nonlinear with numerical evaluation + FD verification, multi-field, no-field-ref.)*
- [x] Unit-test Jacobian parity between analytic and automatically generated derivatives for representative expression-defined models.
- [x] Unit-test optional Hessian or Hessian-vector generation for methods that request second derivatives.
- [x] Unit-test structural metadata handling for differential/algebraic rows and solver hints.
- [x] Unit-test event-hook and reset-hook plumbing.
- [x] Unit-test failure-policy dispatch for local solve failure, singular Jacobian, event-localization failure, and subcycle divergence.
- [x] Unit-test auxiliary-input provider ordering and invalidation.
- [x] Unit-test provider DAG construction and cycle detection, including hard failures for same-time cyclic bindings.
- [x] Unit-test explicit cross-scope mapping providers and rejection of implicit cross-scope auxiliary bindings.
- [x] Unit-test boundary trace and boundary reduction provider validation, lowering, and caching behavior.
- [x] Unit-test auxiliary coupling graph registration and communication planning.
- [x] Unit-test declarative `AuxiliaryModel` builder lowering, validation, and diagnostics.
- [x] Unit-test declarative local-stepper selection, option propagation, and invalid mode/method combinations.
- [x] Unit-test symbolic construction and evaluation of `AuxiliaryInputRef`, `AuxiliaryOutputRef`, and `AuxiliaryStateRef`.
- [x] Unit-test output-evaluation semantics across committed, work, stage, nonlinear-iterate, and substep state views.
- [x] Unit-test JIT packing and interpreter evaluation with the new neutral naming.
- [x] Unit-test threaded evaluation safety for immutable runtime views and thread-local scratch ownership.
- [x] Integration-test one global auxiliary-state use case.
- [x] Integration-test one region-restricted auxiliary block deployed on a selected material or subdomain set.
- [x] Integration-test one nodal bulk auxiliary-state use case.
- [x] Integration-test one cell-scoped use case.
- [x] Integration-test one quadrature-point-scoped use case.
- [x] Integration-test one boundary-entity-scoped use case.
- [x] Integration-test one DAE-like local system.
- [x] Integration-test one `Monolithic` auxiliary block assembled monolithically with FE fields. *(Chain-rule coupling FD-verified via `MixedJacobianBlockFDVerification`. Direct `dF/d(fields)` end-to-end integration-tested via `DirectFieldJacobianFDVerification` — Node-scoped, nonlinear u^2, non-uniform field values, full FD parity. Global-scope direct field references rejected at setup via `DirectFieldGlobalScopeRejected`.)*
- [x] Integration-test one nonlocal auxiliary coupling graph case.
- [x] Integration-test one mixed field-plus-auxiliary system layout path where monolithic auxiliary unknowns do not reuse FE field DOF maps.
- [x] Integration-test one ragged-layout case.
- [x] Integration-test one history-interpolation case.
- [x] Integration-test one event/reset case.
- [x] Integration-test one subcycled or multirate case.
- [x] Integration-test one `Partitioned` auxiliary model with explicit local stepper selection.
- [x] Integration-test one `Partitioned` auxiliary model with implicit local stepper selection.
- [x] Integration-test one implicit auxiliary solve using symbolic default Jacobians from a math-first model.
- [x] Integration-test setup-time symbolic derivative generation and caching for a representative expression-defined auxiliary model.
- [x] ~~Integration-test one auxiliary solve using explicit `AutoAD` derivatives.~~ AutoAD removed from enum; symbolic + FD are the supported paths.
- [x] Integration-test one auxiliary solve using explicit `FiniteDifference` derivatives.
- [x] Integration-test one partitioned nodal ionic-like auxiliary model driving an FE formulation through a first-class auxiliary output.
- [x] Integration-test one monolithic local auxiliary model coupled directly to an FE field.
- [x] Integration-test one lumped boundary auxiliary model using `Global` scope with a sampled boundary reduction input.
- [x] Integration-test one lumped boundary auxiliary model using `Global` scope with a coupled boundary reduction input.
- [x] Integration-test one boundary-local auxiliary model using `BoundaryEntity` scope with a sampled boundary trace input.
- [x] Integration-test one derivative-override example comparing analytic override, symbolic default, and `FiniteDifference` behavior.
- [x] Integration-test one public `AuxiliaryOperator(...)` builder example for auxiliary-to-auxiliary graph coupling.
- [x] Integration-test one public `AuxiliaryOperator(...)` builder example for field-to-auxiliary mixed coupling.
- [x] Integration-test one public `AuxiliaryOperator(...)` builder example for nonlocal boundary-entity coupling.
- [x] Regression-test the coupled boundary workflow through the new subsystem.
- [x] Add MPI tests for node-scope synchronization.
- [x] Add MPI tests for boundary-entity-scope synchronization.
- [x] Add MPI tests for nonlocal auxiliary coupling communication where applicable.
- [x] Add restart/pack/unpack tests.
- [x] Add restart-schema validation tests that reject mismatched block identity, deployment-region identity, ordering mode, and history metadata.
- [x] Add transfer/remap tests if mesh adaptation support exists in the FE workflow.

### Acceptance Criteria

- [x] The new subsystem is covered by unit, integration, and MPI tests.
- [x] Coupled boundary behavior remains correct after migration.
- [x] Bulk scopes are exercised in tests, not only designed on paper.

---

## Phase 13 — Documentation, Examples, And Cleanup

Finish the migration by updating documentation and removing stale terminology from the primary user paths.

### Files to Modify

- [x] `Code/Source/solver/FE/README.md`
- [x] `Code/Source/solver/FE/Systems/PLAN.md`
- [x] `Code/Source/solver/FE/Forms/VOCABULARY.md`
- [x] `Code/Source/solver/FE/Forms/VOCABULARY_ROADMAP.md`
- [x] `Code/Source/solver/FE/Forms/SYSTEMS_INTEGRATION.md`
- [x] relevant FE and test `README.md` files that mention coupled-boundary auxiliary state as the only supported model

### Checklist

- [x] Document the generalized `AuxiliaryState` architecture and scope model.
- [x] Document deployment region as an orthogonal concept to scope, with examples for subdomains, material sets, interface sets, and boundary subsets.
- [x] Document the two authoring surfaces: the math-first `AuxiliaryModel` builder and the advanced lower-level residual/Jacobian interface.
- [x] Document that both authoring surfaces lower to the same backend `AuxiliaryState` infrastructure.
- [x] Document `Partitioned` vs `Monolithic` solve modes and when to use each.
- [x] Document the monolithic semidiscrete residual contract and how `TimeIntegrator` owns stage-state, iterate-state, and `xdot` semantics.
- [x] Document the public local-stepper API for staggered `Partitioned` auxiliary instances and explain that `Monolithic` instances use the global assembled time discretization path instead.
- [x] Document derivative-source policy and precedence: analytic overrides first, symbolic-by-default for expression-defined models, and explicit `FiniteDifference` alternative.
- [x] Document the lowering model from math-first auxiliary equations to residual `FormExpr` trees and explain that symbolic derivatives are taken on those lowered expressions.
- [x] Document the new symbolic vocabulary and deprecations.
- [x] Document auxiliary input providers and how boundary functionals fit into the new model.
- [x] Document `AuxiliaryModelSignature` as the canonical public contract type and explain how optional parameter descriptors participate in the public contract.
- [x] Document field-binding semantics such as sampled, coupled, and reduced field inputs, including scope-specific validity rules.
- [x] Document boundary-binding semantics such as sampled traces, coupled traces, sampled reductions, and coupled reductions, including when to use each.
- [x] Document first-class auxiliary outputs as the preferred formulation-facing coupling surface.
- [x] Document raw auxiliary state access as an advanced or diagnostic path only, with primary examples remaining output-oriented.
- [x] Document output-evaluation state views, output caching or recomputation rules, and how output derivatives participate in monolithic assembly.
- [x] Document DAE-capable stepping interfaces and available methods.
- [x] Document optional second-derivative support, including when Hessians or Hessian-vector products may be requested and when they are not required.
- [x] Document `AuxiliaryOperator` as the canonical path for nonlocal coupling graphs and advanced nonlocal auxiliary interactions.
- [x] Document when to use `AuxiliaryModel` versus `AuxiliaryOperator`, including the rule that local or per-entity monolithic models stay on `AuxiliaryModel` and only genuinely nonlocal or custom sparse couplings move to `AuxiliaryOperator`.
- [x] Document provider DAG rules, cycle detection, and the rule that same-time cyclic dependencies must become `Monolithic` or `AuxiliaryOperator` couplings.
- [x] Document explicit cross-scope mapping rules for auxiliary-to-auxiliary bindings and when such mappings must move to `AuxiliaryOperator`.
- [x] Document that monolithic auxiliary unknowns use auxiliary-specific unknown layouts composed into a mixed system layout rather than reusing FE field DOF maps directly.
- [x] Document the canonical `AuxiliaryOperator(...)` builder surface and how it maps to advanced nonlocal and custom mixed sparse couplings.
- [x] Document ragged layout support and history interpolation semantics.
- [x] Document event hooks and multirate scheduling hooks.
- [x] Document how schedule selection, substepping, input refresh, and local stepper-method selection interact in staggered workflows.
- [x] Document distributed-scope semantics and synchronization expectations.
- [x] Add the five canonical end-to-end API examples:
  - [x] a partitioned nodal ionic-like auxiliary model
  - [x] a monolithic local auxiliary model coupled to an FE field *(Chain-rule `dF/dI × dI/du` for all scopes. Direct symbolic `dF/d(fields)` for Node-scoped Lagrange models. Global scope with direct field references rejected at setup.)*
  - [x] a global lumped boundary auxiliary model
  - [x] a boundary-local auxiliary model
  - [x] a derivative-override example comparing analytic, symbolic, and `FiniteDifference`
- [x] Add advanced `AuxiliaryOperator(...)` examples:
  - [x] an auxiliary-to-auxiliary graph-coupling operator
  - [x] a field-to-auxiliary mixed operator
  - [x] a nonlocal boundary-entity operator
- [x] Remove or rewrite docs that describe auxiliary state as boundary-only or scalar-ODE-only.
- [x] Remove or rewrite docs that imply full-scope auxiliary allocation when deployment-region selection is required.

### Acceptance Criteria

- [x] A new FE subsystem consumer can implement a formulation against the new `AuxiliaryState` API without reading old coupled-boundary code.
- [x] The primary docs no longer describe the deprecated model as the preferred path.

---

## Phase 14 — Advanced DAE Services

Promote DAE support from a strong generic interface into a robust solver service layer.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDAEAnalyzer.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryDAEAnalyzer.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInitializationSolver.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryInitializationSolver.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.cpp`
- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`

### Checklist

- [x] Add structural DAE analysis for residual blocks, variable classes, and constraint partitions.
- [x] Add consistent-initialization solvers beyond callback-only initialization hooks.
- [x] Add optional index-reduction strategy hooks for models that require them.
- [x] Add row scaling, variable scaling, and residual normalization support for auxiliary DAE solves.
- [x] Add diagnostics and verification utilities for derivative quality, including Jacobian consistency checks and optional second-derivative sanity checks.
- [x] Add solver diagnostics for singular or structurally inconsistent auxiliary models.
- [x] Add analysis metadata reporting for DAE structure and initialization outcomes.

### Acceptance Criteria

- [x] Auxiliary DAE models can be structurally analyzed and consistently initialized through infrastructure services rather than formulation-specific ad hoc code.
- [x] The plan no longer depends solely on residual callbacks to handle harder DAE setup cases.

---

## Phase 15 — Events, Nonsmooth Systems, And Hybrid State Transitions

Turn event and nonsmooth extension hooks into a real subsystem.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryEventManager.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryEventManager.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryNonsmoothPolicy.h`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.cpp`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/TimeStepping/TimeLoop.h`

### Checklist

- [x] Add event function registration and root-bracketing support.
- [x] Add state-reset and mode-switch transition callbacks.
- [x] Add event localization policies for time stepping.
- [x] Add nonsmooth policy hooks for active-set, complementarity, or hybrid switching workflows.
- [x] Add post-event reinitialization and rollback behavior.
- [x] Add diagnostics for repeated events, chattering, and failed event localization.

### Acceptance Criteria

- [x] Event-driven auxiliary models can be advanced with infrastructure-managed event detection and state transition handling.
- [x] Nonsmooth and hybrid auxiliary models are no longer limited to placeholder extension hooks.

---

## Phase 16 — Scalable Monolithic Auxiliary Solves

Add solver-facing structure for large monolithic and nonlocal auxiliary systems.

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryCouplingGraph.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryCouplingGraph.cpp`
- [x] `Code/Source/solver/FE/Systems/FESystem.h`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Analysis/ProblemAnalysisTypes.h`

### Checklist

- [x] Add solver-facing block layout metadata for mixed field/auxiliary systems.
- [x] Add preconditioner and block-factorization extension hooks for auxiliary operators.
- [x] Add partition-aware ordering and communication planning for nonlocal auxiliary graphs.
- [x] Add diagnostics for sparsity, coupling density, and conditioning of auxiliary blocks.
- [x] Add interfaces for Schur-like or split solve strategies involving auxiliary blocks.
- [x] Add verification coverage for strong field-auxiliary and auxiliary-auxiliary coupling at scale.

### Acceptance Criteria

- [x] Large `Monolithic` auxiliary systems have explicit solver and preconditioning extension points.
- [x] The plan addresses scalability as infrastructure rather than leaving it entirely to downstream formulations.

---

## Phase 17 — History, Delay, And Long-Memory Services

Extend the history subsystem into a full service for richer time-dependent auxiliary models.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryService.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryService.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryBuffer.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryHistoryBuffer.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`

### Checklist

- [x] Add retention policies for short, long, and formulation-defined history windows.
- [x] Add time-query APIs over stored history with interpolation and extrapolation policy control.
- [x] Add delay-evaluation helpers for constant and formulation-defined delays.
- [x] Add checkpoint pruning and compression policies for long histories.
- [x] Add diagnostics for missing history, extrapolation, and invalid delay queries.

### Acceptance Criteria

- [x] Auxiliary models that need richer history access are supported by a dedicated history service rather than ad hoc buffer access.
- [x] Delay-like auxiliary models are no longer limited to simple fixed-step history lookups.

---

## Phase 18 — Transfer, Restart, And Remap Operators

Turn transfer and restart hooks into explicit auxiliary-state operators.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryTransferOperator.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryTransferOperator.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.cpp`
- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`

### Checklist

- [x] Add explicit transfer-operator interfaces for restart, repartition, and remeshing.
- [x] Add conservative and interpolatory transfer policy support where meaningful.
- [x] Add formulation-defined remap callbacks for scope-specific state.
- [x] Add layout/version validation for restart payloads.
- [x] Add diagnostics for failed or lossy transfer paths.

### Acceptance Criteria

- [x] Transfer and restart semantics are represented as explicit infrastructure operators.
- [x] Auxiliary-state remap behavior is no longer only a hook-level design item.

---

## Phase 19 — Multirate Algorithms And Verification

Promote multirate support from lifecycle hooks into verified algorithms.

### Files to Create

- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryMultirateScheduler.h`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryMultirateScheduler.cpp`

### Files to Modify

- [x] `Code/Source/solver/FE/Systems/FESystem.cpp`
- [x] `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.cpp`
- [x] `Code/Source/solver/FE/TimeStepping/TimeLoop.h`

### Checklist

- [x] Add predictor/corrector policies for subcycled and multirate auxiliary advancement.
- [x] Add error-control and convergence-monitoring hooks for multirate schedules.
- [x] Add consistency rules for `Partitioned` vs `Monolithic` blocks under mixed-rate advancement.
- [x] Add checkpoint strategies for rollback under failed multirate iterations.
- [x] Add verification coverage for accuracy and stability under representative mixed-rate cases.

### Acceptance Criteria

- [x] Multirate advancement is represented by explicit algorithms and verification criteria rather than lifecycle placeholders alone.
- [x] Mixed-rate auxiliary workflows have an infrastructure-defined correctness path.

---

## Recommended Execution Order

1. [x] Phase 1: core contracts and terminology
2. [x] Phase 2: storage abstraction and scope-specific layout
3. [x] Phase 3: distributed semantics, ownership, restart, and transfer
4. [x] Phase 7A: minimal symbolic slice of Phase 7 needed by lowering, outputs, and derivative targets
5. [x] Phase 4: DAE-capable auxiliary model interface
6. [x] Phase 4A: declarative authoring and deployment API
7. [x] Phase 5: generalized auxiliary input infrastructure
8. [x] Phase 6: partitioned and monolithic solve modes and auxiliary operators
9. [x] Phase 7B: remaining symbolic vocabulary generalization
10. [x] Phase 8: assembly, evaluator, and JIT plumbing migration
11. [x] Phase 9: `FESystem` ownership and lifecycle integration
12. [x] Phase 10: coupled boundary migration and deprecation
13. [x] Phase 11: analysis and metadata updates
14. [x] Phase 12: tests
15. [x] Phase 13: documentation, examples, and cleanup
16. [x] Phase 14: advanced DAE services
17. [x] Phase 15: events, nonsmooth systems, and hybrid state transitions
18. [x] Phase 16: scalable monolithic auxiliary solves
19. [x] Phase 17: history, delay, and long-memory services
20. [x] Phase 18: transfer, restart, and remap operators
21. [x] Phase 19: multirate algorithms and verification

---

## Risks To Actively Manage During Implementation

- [ ] Scope creep in the DAE interface. Keep the contract general, but define a concrete phase-1 capability target.
- [ ] Over-coupling the partitioned and monolithic solve paths. Keep the shared abstractions clean while allowing different solve strategies.
- [ ] The math-first builder diverging semantically from the lower-level residual/Jacobian interface. Keep one lowering path and shared validation.
- [ ] Derivative-source inconsistency between analytic, symbolic, and `FiniteDifference` paths. Keep precedence rules explicit and parity tests broad.
- [ ] Excessive API churn in Forms/JIT. Update naming once, then migrate call sites quickly.
- [ ] Field-binding APIs becoming too implicit. Keep sampled, coupled, and reduced field access explicit in the public API.
- [ ] Hidden distributed-ordering assumptions for node and boundary-entity scopes.
- [ ] Quadrature-point storage blow-up for large meshes if layout is not designed carefully.
- [ ] Ragged-layout complexity leaking into hot-path fixed-layout cases.
- [ ] Temporary duplication between old and new APIs. Keep deprecated wrappers thin and short-lived.
- [ ] Global coupling and monolithic assembly complexity overwhelming the initial local/block refactor.
- [ ] Test gaps in JIT/interpreter parity after renaming auxiliary terminals and ABI fields.

---

## Remaining Limitations And Non-Goals

The extended `AuxiliaryState` infrastructure is intentionally broad, but it still has explicit boundaries. Some of the items below are hard scope boundaries that should remain. Others are residual limitations that are specifically addressed by Phases 14 through 19 above.

- [ ] `AuxiliaryState` is physics-agnostic infrastructure for auxiliary ODE/DAE-like systems. It is not intended to replace true PDE components or FE fields.
- [ ] If a quantity should be represented as a primary spatially distributed unknown with FE basis functions, differential operators, and weak-form assembly as a field, it should remain a field rather than be modeled as `AuxiliaryState`.
- [ ] Phase-1 DAE support is broad but not universal. Phases 14 through 19 add stronger resolution paths for advanced DAE workflows, but the library should still document capability levels explicitly rather than implying every DAE method is equally supported.
- [ ] Event, nonsmooth, complementarity, and hybrid-system behavior need dedicated solver and transition services. Phase 15 addresses this, but formulation-specific numerical choices may still be required.
- [ ] Very large `Monolithic` auxiliary systems and strongly nonlocal coupling graphs need scalable solver, preconditioner, and load-balancing strategies. Phase 16 addresses this, but scalability remains something that must be verified, not assumed.
- [ ] Ragged layout support remains inherently more complex and potentially slower than fixed-layout cases even after the supporting infrastructure is in place.
- [ ] Time-stamped history and interpolation hooks need dedicated services for long-memory and delay-like models. Phase 17 addresses this, but some history semantics may still require formulation-specific policies.
- [ ] Transfer, restart, remeshing, and repartitioning semantics need explicit operators. Phase 18 addresses this, but exact physically or mathematically appropriate transfer behavior may still vary by formulation.
- [ ] Multirate and rollback hooks need explicit algorithms and verification. Phase 19 addresses this, but stable and efficient multirate strategies for tightly coupled mixed systems still require numerical validation.

These are acceptable boundaries as long as the FE library documentation continues to state clearly that `AuxiliaryState` is auxiliary ODE/DAE-like infrastructure and not a substitute for true PDE unknowns.

---

## Phase-1 Capability Target For DAE Support

To keep the implementation bounded while satisfying the fixed decisions, the first delivered DAE-capable version should explicitly support:

- [ ] block-local residual form `F(xdot, x, history, inputs, t, dt) = 0`
- [ ] mixed differential and algebraic rows in one local block
- [ ] analytic Jacobian override support
- [ ] symbolic Jacobian generation by default for expression-defined models
- [ ] explicit `FiniteDifference` Jacobian alternative
- [ ] optional second-derivative requests through symbolic generation or analytic overrides when methods require them
- [ ] structural metadata describing differential vs algebraic rows and solver hints
- [ ] one implicit stepper path for residual-based local solves
- [ ] one explicit stepper path for eligible blocks
- [ ] consistent initialization hook for algebraic rows
- [ ] event-hook extension points
- [ ] per-block method selection
- [ ] optional substepping

It does not need to deliver every possible DAE method before the rest of the subsystem lands.
