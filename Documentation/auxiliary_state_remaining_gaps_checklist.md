# AuxiliaryState Remaining Gaps Checklist

**Date**: 2026-03-24
**Last reconciled with code**: 2026-03-24

This checklist breaks the remaining documented `AuxiliaryState` limitations into concrete implementation steps. It is scoped to the gaps still called out in [Code/Source/solver/FE/Docs/AuxiliaryState/README.md](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Docs/AuxiliaryState/README.md).

Items marked `[x]` are implemented and tested. Items marked `[ ]` are not yet implemented. Items marked `[~]` are partially implemented (infrastructure exists but runtime wiring is incomplete).

## 1. Entity-Local Inputs

### Design and API
- [x] Define the runtime contract for entity-varying auxiliary inputs in `AuxiliaryInputRegistry`.
- [x] Decide how per-entity input values are addressed for each storage scope: `Node`, `Cell`, `QuadraturePoint`, and `BoundaryEntity`.
- [x] Decide whether entity-local input providers expose:
  - [x] one flat buffer plus entity-aware indexing metadata, or
  - [x] an explicit `valuesOf(name, entity_index)` style access path.
- [~] Define which existing provider kinds must support per-entity evaluation in the `FESystem` convenience path:
  - [x] `SampledStateField` — `registerSampledFieldInput()` uses per-field DOF lookup (vertex-based Lagrange only)
  - [~] `BoundaryTrace` — not yet implemented as a convenience helper
  - [x] `SampledBoundaryReduction` — `registerBoundaryNodalSumInput()` sums face-vertex DOFs (vertex-based Lagrange only; not a quadrature-weighted integral)
  - [x] Direct entity-local callback providers — fully working

### Registry Implementation
- [x] Extend `AuxiliaryInputSpec` and/or registry storage to represent entity-local values.
- [x] Implement entity-aware registration metadata for non-global providers.
- [x] Add registry APIs for retrieving per-entity input slices without forcing callers to reconstruct them manually.
- [x] Ensure slot accounting remains well-defined for entity-local providers.
- [x] Preserve the existing flat/global input path for global and scalar providers.

### FESystem Runtime
- [x] Update `advanceAuxiliaryState()` to build `ctx.inputs` per entity instead of once per block when bindings resolve to entity-local providers.
- [x] Update `prepareAuxiliaryForAssembly()` to build `ctx.inputs` per entity for output evaluation.
- [x] Populate `AuxiliaryLocalContext::entity_index` consistently for all entity-local input evaluation paths.
- [x] Define clear behavior when a block binds a global input and an entity-local input at the same time.

### Validation and Testing
- [x] Add unit tests for per-entity input retrieval in the registry.
- [x] Add `FESystem` tests for a custom model driven by entity-local inputs.
- [x] Add `FESystem` tests for multi-component entity-local inputs (`name:size` parsing).
- [x] Add `FESystem` tests for `registerSampledFieldInput` (positive + non-vertex rejection + before-setup rejection).
- [x] Add `FESystem` tests for `registerBoundaryNodalSumInput` (positive + non-vertex rejection + before-setup rejection).
- [ ] Add `FESystem` tests for BoundaryTrace provider (not yet implemented as convenience helper).
- [x] Add negative tests for mismatched entity counts and invalid scope/provider combinations.

## 2. Deployment Regions

### Region Model
- [x] Finalize the representation of `AuxiliaryDeploymentRegion` for:
  - [x] whole-domain deployment
  - [x] formulation-defined explicit entity sets
  - [x] subdomain/material-set deployment — expanded via `getCellDomainId()`
  - [x] boundary-set deployment — expanded via `forEachBoundaryFace(marker, ...)`
  - [~] interface deployment — collects all interior faces (per-face marker filtering not available on IMeshAccess)
- [x] Define how deployment regions compose with storage scope.
- [x] Define whether deployment-region identity is mesh-marker-based, explicit-entity-set-based, or both.

### Finalization and Storage
- [~] Implement region-to-entity expansion during `finalizeAuxiliaryLayout()`.
  - [x] Explicit entity sets: fully working
  - [x] CellSet/MaterialIdSet: expanded via `getCellDomainId()` (domain ID, not separate material query)
  - [x] BoundarySet: expanded via `forEachBoundaryFace(marker, ...)`
  - [~] InterfaceSet: collects all interior faces (per-face marker filtering not available on IMeshAccess)
- [x] Build and store an explicit local entity map for each deployed block.
- [x] Use the entity map to determine actual block `entityCount()`.
- [x] Ensure region-restricted blocks do not allocate storage for excluded entities.
- [x] Ensure output slot layout and history layout respect the filtered entity set.

### Runtime Integration
- [x] Update per-entity stepping to iterate only over deployed-region entities.
- [x] Update output evaluation to iterate only over deployed-region entities.
- [x] Update auxiliary analysis/summary reporting to include region-aware counts.
- [x] Define restart/remap metadata for region identity and region-specific entity ordering.

### Validation and Testing
- [x] Add tests for explicit-entity-set deployment.
- [ ] Add tests for marker-based subdomain deployment (requires mesh topology wiring).
- [ ] Add tests for marker-based boundary-subset deployment (requires mesh topology wiring).
- [x] Add negative tests for invalid scope/region combinations.

## 3. Non-Default Layouts in FESystem

### Storage Helpers
- [x] Add layout-aware entity view helpers to `AuxiliaryStateStorage` for:
  - [x] `gatherEntityWork(entity_index)` / `scatterEntityWork(entity_index, data)`
  - [x] `gatherEntityCommitted(entity_index)`
  - [x] `gatherEntityHistory(snapshot_index, entity_index)`
- [x] Ensure those helpers work for:
  - [x] `FixedStride + ByEntityThenComponent`
  - [x] `FixedStride + ByComponentThenEntity`
  - [x] ragged layouts

### Runtime Refactor
- [x] Replace all manual `e * dim` history slicing in `FESystem` with layout-aware helper calls.
- [x] FESystem runtime paths use `gatherEntityWork`/`scatterEntityWork` for all per-entity access.
- [x] Output evaluation uses layout-aware entity access.

### Deployment API
- [x] Add `.layoutMode()` and `.entityOrdering()` to `AuxiliaryDeployedInstance` for FixedStride layouts.
- [ ] Ragged layout through the deployment API: rejected at finalization.
  Both stepper and monolithic assembly assume fixed per-entity dimension.
  Use `AuxiliaryStateManager::registerBlockRagged()` directly.

### Validation and Testing
- [x] Add per-entity stepping tests for `ByComponentThenEntity` (via manager API).
- [x] Add output evaluation tests for `ByComponentThenEntity` (via manager API).
- [x] Add ragged-layout tests for stepping and output evaluation.
- [x] Add restart/history tests for non-default layouts.

## 4. Generic Model Input and Parameter Signatures

### Base Interface
- [x] Decide the minimal optional signature hooks needed on `AuxiliaryStateModel` for non-builder models.
- [x] Add optional base-class methods for declared input names and parameter names.
- [x] Define whether generic models may also expose input/parameter sizes through the base interface.
  - [x] Implemented via `"name:size"` suffix convention, parsed by FESystem.

### Runtime Usage
- [x] Update `FESystem` to prefer base-class input/parameter signatures over `std::map` order when available.
- [x] Keep lexicographic `std::map` ordering only as a fallback when no signature exists.
- [x] Add deployment diagnostics that explicitly tell custom-model authors which ordering mode is being used.
- [x] Parse `"name:size"` suffix in all input-building paths (stepping, output eval, monolithic, entity-local rebuild).
- [x] Validate suffixes at deployment time (reject empty base, non-positive size, trailing junk).

### Validation and Testing
- [x] Add a custom-model test with multiple inputs where signature order differs from lexicographic order.
- [x] Add a custom-model test with multiple parameters where signature order differs from lexicographic order.
- [x] Add a custom-model test with multi-component `"name:size"` inputs.
- [x] Add deployment rejection tests for malformed suffixes.
- [x] Add diagnostics tests confirming fallback behavior when no generic signature is provided.

## 5. Schedule Mode Integration

### Scheduler Wiring
- [x] Decide how `schedule_mode` is represented at runtime for deployed partitioned blocks.
- [x] Build block schedule metadata during `finalizeAuxiliaryLayout()`.
- [x] Wire `AuxiliaryMultirateScheduler` into `advanceAuxiliaryState()`.
  - [x] `planSubsteps()` produces time-ordered substep plan
  - [x] `advanceFromWork()` advances from intermediate work state (no committed reset)
  - [x] Per-entity history and entity-local inputs handled in multirate path
- [x] Replace unconditional "advance every partitioned block every call" behavior with schedule-driven dispatch.
  - [x] SingleRate and Subcycled: fully working
  - [x] Multirate: interleaved cross-block dispatch via planSubsteps() + advanceFromWork()
  - [~] Predictor/corrector policies: defined but not yet consumed at runtime

### Advancement Semantics
- [x] Define exact semantics for `SingleRate`.
- [x] Define exact semantics for `Subcycled`.
- [~] Define exact semantics for `Multirate`.
  - [x] Per-block substepping at rate_ratio
  - [ ] Interleaved cross-block coupling at intermediate times
- [x] Define how input refresh interacts with substeps and multirate schedules.
- [x] Define how work/commit/history updates happen for subcycled blocks.
- [x] Define rollback behavior under scheduled substeps.

### Validation and Testing
- [x] Add single-rate schedule regression tests.
- [x] Add subcycling tests that verify substep counts and state evolution.
- [x] Add multirate schedule planning tests.
- [x] Add runtime integration tests where two blocks advance at different rates.

## 6. AuxiliaryInput/AuxiliaryOutput Symbol Auto-Resolution in FE Forms

### Installer Design
- [x] Define the canonical author-facing symbol syntax for auxiliary inputs and outputs in FE forms.
- [x] Decide whether outputs must always be instance-qualified during automatic resolution.
- [x] Define ambiguity diagnostics for unresolved or multiply-matched auxiliary output symbols.

### FormsInstaller Integration
- [x] Extend `FormsInstaller` to resolve `AuxiliaryInputSymbol` to `AuxiliaryInputRef` automatically.
- [x] Extend `FormsInstaller` to resolve `AuxiliaryOutputSymbol` to `AuxiliaryOutputRef` automatically.
- [x] Thread `FESystem` registry and deployed-model lookup information into the installer/lowering path.
- [x] Remove the requirement that callers manually call `transformNodes()` for auxiliary symbol resolution.

### Compiler and Validation
- [x] Update FE-form compilation tests so unresolved auxiliary symbols are an installer responsibility, not a caller responsibility.
- [x] Ensure JIT and interpreter paths see only resolved `Ref` terminals at compile time.
- [x] Add diagnostics for:
  - [x] missing auxiliary inputs
  - [x] missing auxiliary outputs
  - [x] ambiguous bare output symbols
  - [x] use of auxiliary symbols before layout finalization

### Validation and Testing
- [x] Add `FormsInstaller` tests with `AuxiliaryInput("...")`.
- [x] Add `FormsInstaller` tests with instance-qualified `AuxiliaryOutput("instance", "name")`.
- [x] Add JIT/runtime tests showing auxiliary symbols compile and evaluate without manual pre-resolution.

## 7. Symbolic Differentiation for Auxiliary Models

### Differentiation Targets
- [~] Add auxiliary-specific differentiation targets for:
  - [x] state variables `x` — symbolic for all scalar ops
  - [x] time derivatives `xdot` — synthesized from row kind
  - [x] auxiliary inputs — symbolic `dF/d(inputs)` generated and evaluated
  - [ ] coupled FE fields for monolithic paths — not yet generated (enum declared)
- [x] Define which of these are phase-1 Jacobian requirements vs optional sensitivity targets.

### Derivative Provider
- [x] Replace the current "store expressions, then fall back to FD" path in `AuxiliaryDerivativeProvider`.
- [x] Lower builder residual expressions into symbolic derivative artifacts at setup time.
- [x] Cache the generated derivative expressions/artifacts for runtime Jacobian evaluation.
- [x] Preserve analytic override precedence over symbolic generation.

### Runtime Evaluation
- [x] Implement runtime evaluation of symbolic Jacobian artifacts.
- [x] Keep FD fallback only for models without symbolic or analytic derivative support.
- [x] Add clear diagnostics showing which derivative source is actually active.

### Validation and Testing
- [x] Add analytic-vs-symbolic Jacobian parity tests for simple ODE models.
- [x] Add analytic-vs-symbolic Jacobian parity tests for mixed ODE/algebraic models.
- [x] Add runtime tests proving symbolic mode no longer resolves to FD for builder-defined models.

## 8. Monolithic Assembly Consumption

### Unknown Layout Integration
- [x] Thread `AuxiliaryUnknownLayout` into `SystemSetup`.
- [x] Compose auxiliary unknowns with field unknowns into the actual mixed system layout.
- [x] Expose auxiliary unknown offsets and ownership metadata to assembly and solver paths.

### Assembly
- [x] Add monolithic auxiliary residual assembly contributions.
- [x] Add monolithic auxiliary Jacobian assembly contributions.
- [x] Add auxiliary-auxiliary operator contribution assembly via `assembleMixedAuxiliaryIntoGlobal()`.
- [~] Add mixed field-auxiliary and auxiliary-field block assembly — operator infrastructure exists; field-to-aux dF/d(fields) symbolic generation pending.

### Solver Integration
- [x] Extend matrix assembly to include auxiliary rows and columns via `assembleMixedAuxiliaryIntoGlobal()` called from `assembleOperator()`.
- [x] Extend nonlinear residual/state vectors to include monolithic auxiliary unknowns (auxiliary DOFs indexed from `MixedSystemLayout` offsets in global system).
- [~] Ensure solver setup and block metadata can see the mixed layout — layout is computable; solver vector/matrix resizing for the augmented system is caller responsibility.
- [ ] Define preconditioning/ordering metadata for auxiliary blocks.

### Validation and Testing
- [ ] Add a local monolithic auxiliary model coupled to a field in an end-to-end solve test.
- [x] Add a monolithic auxiliary-only solve test.
- [ ] Add mixed field-auxiliary Jacobian structure tests.

## 9. Supporting Documentation and Regression Coverage

- [x] Update [Code/Source/solver/FE/Docs/AuxiliaryState/README.md](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Docs/AuxiliaryState/README.md) as each runtime gap is closed.
- [x] Remove each item from the README gap list only after code and tests are in place.
- [x] Add explicit coverage notes for:
  - [x] builder models
  - [x] custom `AuxiliaryStateModel` implementations
  - [x] per-entity scopes
  - [x] FE-form coupling
  - [~] monolithic solves — standalone assembly tested; mixed field-coupled solve not yet

## 8. True Boundary-Integral Auxiliary Inputs

### Infrastructure
- [x] Create `BoundaryReductionService` under `FE/Systems` — physics-agnostic boundary-functional registration, compilation, evaluation, and boundary-measure caching.
- [x] Extract boundary-functional machinery from `CoupledBoundaryManager` into the shared service.
- [x] Update `CoupledBoundaryManager` to delegate to `BoundaryReductionService`.

### Registration API
- [x] Add `FESystem::registerBoundaryIntegralInput(name, functional)` — true quadrature-weighted boundary integral as auxiliary input.
- [x] Add convenience overload `registerBoundaryIntegralInput(name, integrand, marker, reduction, schedule)`.
- [x] Support `Sum` and `Average` reduction modes.
- [x] Honor `OncePerTimeStep`, `EachNonlinearIteration`, and `Manual` update schedules.
- [x] Store results in `AuxiliaryInputRegistry` so `AuxiliaryInput("Q")` resolves without special handling.

### Navier-Stokes Client Migration
- [x] Rewrite `toCoupledOutflowBC()` to use `AuxiliaryModelBuilder` + `NaturalBC` instead of `CoupledNaturalBC` + `AuxiliaryStateBuilder`.
- [x] Register boundary-integral input for flow rate `Q` via new API.
- [x] Deploy RCR model with unique instance name.
- [x] Reference `AuxiliaryOutput(instance/P_out)` in traction expression.
- [x] Handle `C == 0` resistive limit with algebraic auxiliary model.
- [x] Ensure registration/deployment happens before `installFormulation()`.

### Monolithic Coupling (Milestone B)
- [~] `BoundaryReductionService::evaluateFunctionalGradient()` — computes `dQ/du` sparse vector.
- [ ] Wire `dF_aux/d(inputs) * d(inputs)/d(fields)` into `assembleMixedAuxiliaryIntoGlobal()`.
- [ ] Add mixed Jacobian structure tests and FD parity checks.

### Testing
- [x] Add unit tests for boundary-integral input registration and update schedule behavior.
- [x] Add multi-marker / distinct-name coverage.
- [x] Add dependency-ordering coverage (boundary-integral input as dependency of another input).

## Recommended Execution Order

- [x] Implement entity-local inputs (direct callback + FE-coupled helpers).
- [~] Implement deployment-region materialization (explicit + CellSet/BoundarySet/MaterialIdSet done; InterfaceSet marker filtering not available).
- [~] Remove non-default-layout exclusions (storage helpers done; deployment API exposes `.layoutMode()`/`.entityOrdering()` for FixedStride; ragged rejected at finalization — requires direct manager API).
- [x] Add generic-model signature hooks for non-builder input/parameter ordering.
- [x] Integrate schedule modes through `AuxiliaryMultirateScheduler` (SingleRate, Subcycled, and Multirate with interleaved dispatch all working).
- [x] Add automatic FE-form auxiliary symbol resolution.
- [~] Complete symbolic differentiation (dF/dx + dF/dxdot + dF/d(inputs) done; dF/d(fields) deferred).
- [~] Integrate monolithic auxiliary unknowns into assembly and solver (standalone + mixed GlobalSystemView assembly done; full sparse solver integration deferred).
