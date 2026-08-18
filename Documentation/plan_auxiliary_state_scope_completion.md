# AuxiliaryState Scope Completion Plan

**Date**: 2026-04-20
**Status**: Fixed-stride scope deliverable complete; ragged layout follow-on complete
**Goal**: consolidate the remaining fixed-stride work required before `AuxiliaryState` can be described as general across `Global`, `Node`, `Cell`, `QuadraturePoint`, and `Region` scopes for both `Partitioned` and `Monolithic` smooth ODE/DAE subproblems.

## Plans Reviewed

- `Documentation/plan_auxiliary_state_generalization.md`
- `Documentation/plan_auxiliary_state_remaining_items.md`
- `Documentation/plan_auxiliary_state_api_cleanup_and_condensation_prereqs.md`
- `Code/Source/solver/FE/Docs/AuxiliaryState/QP_AUTO_LAYOUT_PLAN.md`
- `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`

## Why A New Plan Is Needed

- `plan_auxiliary_state_generalization.md` is still directionally useful, but its action list is too broad and partially stale.
- `plan_auxiliary_state_api_cleanup_and_condensation_prereqs.md` correctly captures completed `Cell` / `QuadraturePoint` / `Facet` local-condensation work, but intentionally defers a real `Node` monolithic strategy.
- `QP_AUTO_LAYOUT_PLAN.md` is largely implemented and should now be treated as baseline behavior, not future work.
- `plan_auxiliary_state_remaining_items.md` closes orthogonal cleanup items, but it does not complete scope generalizability.
- Recent implementation has addressed several earlier blockers:
  - first-class `Region` scope has been added to `AuxiliaryStateScope`, indexing, manager/storage, deployment bindings, restart schema naming, and basic tests
  - deployment-region expansion in `FESystem::finalizeAuxiliaryLayout()` is now scope-aware for fixed-stride `Node`, `Cell`, `QuadraturePoint`, `Facet`, and `Region` projections, with `Global`/`Boundary` treated as metadata-only
  - partitioned runtime now consumes `AuxiliaryStepResult` and applies `AuxiliaryFailurePolicy` retry/reject/restore behavior
- Current code has closed the fixed-stride scope-completion claim.  Remaining
  work is intentionally split out:
  - semismooth and complementarity monolithic hooks remain a separately gated follow-on
  - deployment-path ragged blocks are complete in `Documentation/plan_auxiliary_state_ragged_layout_followup.md`
- The plan must also conform to the newer explicit row-owned, PETSc-like FSILS backend path. Any monolithic auxiliary strategy that creates linear-system rows must define row ownership, distributed sparsity, and insertion semantics explicitly instead of relying on process-local dense bordered arrays or overlap accumulation.

## Scope Of This Plan

- Finish fixed-stride scope support for `Global`, `Node`, `Cell`, `QuadraturePoint`, and `Region`
- Cover both `Partitioned` and `Monolithic` solve modes
- Cover smooth ODEs and mixed ODE/algebraic local DAEs
- Cover serial and MPI ownership behavior where the scope requires it

## Non-Goals

- Do not redefine `Boundary` or `Facet` scope in this document except where shared infrastructure changes affect them.
- Do not overload `QuadraturePoint` to mean face quadrature. Boundary-face and interior-face quadrature should remain separate future scopes.
- Do not force nonlocal cross-entity coupling into `AuxiliaryModel`; those cases should stay on `AuxiliaryOperator`.
- Do not implement deployment-path ragged layout in this fixed-stride completion plan.  Ragged support is tracked by `Documentation/plan_auxiliary_state_ragged_layout_followup.md`.

## End-State Required Before Claiming Scope Completion

- [x] Every fixed-stride scope in this plan has a first-class storage/indexing contract and a documented entity identity.
- [x] `Partitioned` and `Monolithic` paths both work for fixed-size smooth ODEs and mixed ODE/algebraic DAEs at every supported scope.
- [x] Scope restriction via deployment regions works without requiring manual `explicit_entities` in normal cases.
- [x] `FESystem` runtime honors `AuxiliaryStepResult`, `AuxiliaryFailurePolicy`, consistent initialization, and smooth event hooks where the model exposes them; semismooth/complementarity hooks are explicitly gated or deferred.
- [x] Monolithic auxiliary `xdot` assembly uses the active FE time-integrator coefficients instead of a standalone hard-coded BDF1 assumption.
- [x] Monolithic auxiliary rows either participate in the backend through explicit row-owned mixed-layout metadata or are eliminated/reduced before backend assembly.
- [x] FSILS-compatible monolithic paths use the same row-owner map for distributed sparsity construction and numeric assembly.
- [x] FSILS full-matrix participation is used only when the mixed layout can be represented as complete nodal-interleaved row blocks for the configured `dof_per_node`; otherwise the plan must select local condensation, distributed low-rank correction, or a non-FSILS backend.
- [x] Restart/remap metadata is complete for restricted deployments and scope-specific entity maps.
- [x] Deployment-path ragged blocks are either fully supported for the scopes in this plan or explicitly carved out into a separate plan with scope-completion claims limited to fixed-stride blocks.
- [x] Serial, MPI, and FE-coupled regression tests exist for each fixed-stride scope in this plan.

## Primary Code Touchpoints

- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateTypes.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateIndexing.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateIndexing.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStepper.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInitializationSolver.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryInitializationSolver.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryEventManager.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryEventManager.cpp`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/FEQuantityDefinition.h`
- `Code/Source/solver/FE/Systems/BoundaryReductionService.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.cpp`
- `Code/Source/solver/FE/Backends/Utils/BackendOptions.h`
- `Code/Source/solver/FE/Backends/Utils/BackendOptions.cpp`
- `Code/Source/solver/FE/Backends/Interfaces/DofPermutation.h`
- `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.h`
- `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp`
- `Code/Source/solver/FE/Backends/FSILS/FsilsVector.h`
- `Code/Source/solver/FE/Backends/FSILS/FsilsVector.cpp`
- `Code/Source/solver/FE/Backends/FSILS/FsilsFactory.h`
- `Code/Source/solver/FE/Backends/FSILS/FsilsFactory.cpp`
- `Code/Source/solver/FE/Backends/FSILS/FsilsShared.h`
- `Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.h`
- `Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp`
- `Code/Source/solver/FE/Backends/FSILS/FSILS_BLOCKSCHUR_STRUCTURE.md`
- `Code/Source/solver/FE/Backends/FSILS/BLOCKSCHUR_BOUNDARY_COUPLING_PLAN.md`
- `Code/Source/solver/FE/Backends/FSILS/liner_solver/distributed_low_rank_correction.h`
- `Code/Source/solver/FE/Backends/FSILS/liner_solver/distributed_low_rank_correction.cpp`
- `Code/Source/solver/FE/Assembly/GhostContributionManager.h`
- `Code/Source/solver/FE/Assembly/GhostContributionManager.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Sparsity/DistributedSparsityPattern.h`
- `Code/Source/solver/FE/Sparsity/DistributedSparsityPattern.cpp`
- `Code/Source/solver/FE/Analysis/TopologyAnalysisContext.h`
- `Code/Source/solver/FE/Analysis/TopologyAnalysisContext.cpp`

## Cross-Cutting Workstreams

### 1. Freeze Scope Semantics And Scope-Aware Deployment Resolution

- [x] Introduce one centralized scope-resolution path in `FESystem` instead of open-coded scope checks inside `finalizeAuxiliaryLayout()`.
- [x] Make deployment-region expansion produce node ids for `Node`, cell ids for `Cell`, covered cell ids for `QuadraturePoint`, region ids for `Region`, and metadata-only attachment for `Global`.
- [x] Define and validate legal scope/region pairings explicitly.
- [x] Preserve stable entity ordering for restart, checkpointing, and monolithic output binding.
- [x] Ensure restricted deployments preserve ownership metadata for scopes that have owned/ghost semantics.

### 2. Add First-Class Region Scope Infrastructure

- [x] Add `AuxiliaryStateScope::Region` to the public vocabulary, helper factories, and docs.
- [x] Add `AuxiliaryBlockIndexing::createRegion()` and manager/storage support.
- [x] Define region entity identity as topology-region ids from `TopologyAnalysisContext`, not raw deployment markers.
- [x] Cache region-to-cells, region-to-nodes, region-to-boundary-markers, and region-to-interface-faces lookup tables in `FESystem`.
- [x] Extend restart/remap metadata to include region ids and region membership maps.

### 3. Conform To Explicit Row-Owned Backend Semantics

- [x] Require every auxiliary mixed block to state its backend assembly mode explicitly before FSILS solver-option normalization.
- [x] Extend `AuxiliaryBlockUnknownLayout` / `MixedSystemLayout` with enough ownership metadata to answer "which rank owns this auxiliary row?" for every monolithic auxiliary unknown.
- [x] Build a deterministic auxiliary row-owner policy for every scope: replicated/global rows have a single owner, node rows follow the node/backend owner, cell and QP rows follow the owning cell, and region rows follow an explicit region-owner rule.
- [x] Define deterministic `Region` auxiliary row ownership from the owner of the globally lowest owned cell in each topology region, with MPI-wide restricted-region identity preserved before row-owner expansion.
- [x] Ensure distributed sparsity construction uses the same owner map and the same DOF indexing/permutation as numeric assembly.
- [x] Route all auxiliary and mixed field-auxiliary matrix/vector insertions through `GlobalSystemView` using the selected backend ownership policy. For FSILS, ordinary matrix/RHS assembly uses explicit row-owner mapping and owned-row insertion; reverse-scatter routing is reserved for assembler paths that explicitly support it and still route to the same owner map.
- [x] For FSILS, validate whether a full mixed auxiliary system can be represented as complete nodal-interleaved blocks under one `dof_per_node`. If not, choose local condensation, distributed low-rank correction, or a PETSc/Trilinos-style backend that supports arbitrary row layouts.
- [x] If native FSILS full-matrix mixed rows are allowed, require their block descriptors to partition `[0, dof_per_node)` and require solver-role metadata to normalize through `backends::normalizeSolverOptionsForBackend()`.
- [x] Propagate auxiliary block roles into `backends::MixedBlockLayout` so FSILS `BlockSchur`, PETSc field-split, and fallback solvers see the same solver metadata.
- [x] Keep regular FSILS RHS and solution vectors owned-row authoritative. After solves or owner-row updates, synchronize ghosts through `GenericVector::updateGhosts()`; reserve `FsilsVector::accumulateRawContributionsAndUpdateGhosts()` for explicitly raw overlap contribution buffers.
- [x] Validate `FsilsMatrix::usesOwnedRowOperator()`, `FsilsMatrix::ownsFeDofRow()`, `FsilsVector::usesOwnedRowLayout()`, and `FsilsVector::ownsFeDof()` before accepting an FSILS scope-completion path.
- [x] Treat `FsilsMatrix::droppedEntryCount()` as a correctness failure for scope-completion tests, not as a diagnostic-only counter.

### 4. Finish Runtime DAE And Failure Plumbing

- [x] Consume `AuxiliaryStepResult` in both `advanceOneEntry()` and multirate dispatch.
- [x] Implement the configured `AuxiliaryFailurePolicy` behaviors instead of silently ignoring failed local solves.
- [x] Call `AuxiliaryInitializationSolver` when a model or deployment requires consistent initialization.
- [x] Integrate `AuxiliaryEventManager` into partitioned standard and multirate stepping.
- [x] Pass entity-local event context and keep event-manager state per materialized entity for partitioned scoped blocks.
- [x] Explicitly reject monolithic nonsmooth/complementarity hooks at finalize time until a semismooth or active-set policy is implemented.
- [x] Integrate `AuxiliaryEventManager` into monolithic smooth accepted-step lifecycle.
- [x] Defer monolithic semismooth/active-set handling for nonsmooth/complementarity hooks; fixed-stride scope completion rejects these hooks at finalize time until a concrete policy is implemented.
- [x] Surface model row-kind and event-mode metadata into the live registered `AuxiliaryStateSpec`, not only helper classes.
- [x] Surface mass-like metadata and remaining nonsmooth/event policy metadata consistently to the live runtime, not just to helper classes.
- [x] Thread FE time-integrator coefficients into monolithic auxiliary `xdot` evaluation.

### 5. Complete Monolithic Strategy Per Scope

- [x] Keep dense bordered assembly only as a serial/reference implementation or as an explicitly owned reduced system; production FSILS paths must use row-owned mixed rows, direct-only lowering, local condensation, or distributed low-rank correction.
- [x] Implement a sparse or block-sparse `Node` monolithic path; the current dense bordered path does not scale.
- [x] Keep `Cell` and `QuadraturePoint` local condensation for independent per-entity blocks and document when a model is not eligible.
- [x] Add a `Region` monolithic strategy: dense only for serial/reference, explicitly owned bordered/reduced systems, sparse owned rows, or distributed low-rank correction for production MPI.
- [x] State clearly when a formulation must move from `AuxiliaryModel` to `AuxiliaryOperator`.

### 6. Finish Deployment-Path Ragged And Entity-Local Provider Support

- [x] Carve deployment-path ragged layout out of this fixed-stride scope-completion pass with a setup-time diagnostic.
- [x] Replace the deployment-path rejection of ragged layout in a separate bounded follow-on plan: `Documentation/plan_auxiliary_state_ragged_layout_followup.md`.
- [x] Move removal of the current cell-generic ragged fallback in `AuxiliaryStateManager` to the ragged follow-on plan.
- [x] Document the fixed-stride entity-local FE-backed input contract; broader entity-local provider expansion remains follow-on provider work outside this fixed-stride deliverable.
- [x] Support topology-region-local FE-backed inputs for `Region` through `regionIntegral()` and `regionAverage()` quantity providers.
- [x] Move restart/serialization behavior for ragged restricted subsets to the ragged follow-on plan.

### 7. Verification And Documentation

- [x] Add focused scope-completion tests instead of only growing ad hoc coverage inside unrelated files.
- [x] Update `Docs/AuxiliaryState/README.md` to reflect the completed fixed-stride scope semantics.
- [x] Mark older planning docs as superseded for fixed-stride scope completion once this plan is implemented.

## Scope Checklists

### Global Scope

**Current baseline**

- Entity count `1`, `Partitioned` / `Monolithic` flows, runtime DAE lifecycle,
  failure handling, and direct-only algebraic lowering are implemented for the
  fixed-stride deliverable.

**Remaining implementation**

- [x] Treat `deployment_region` as metadata-only for `Global` and validate unsupported expansion attempts explicitly.
- [x] Support consistent initialization and event reset hooks for global partitioned DAEs.
- [x] Keep nonsmooth hooks for global partitioned DAEs explicitly deferred from this fixed-stride scope-completion pass; smooth event hooks remain supported.
- [x] Support global monolithic mixed differential/algebraic models with FE-backed inputs and exact time-integrator-consistent `xdot`.
- [x] For MPI monolithic global rows, choose one authoritative owner per row and make all residual/Jacobian contributions owner-routed or globally reduced before insertion.
- [x] For FSILS, prefer direct-only lowering, explicitly owned bordered/reduced coupling, or distributed low-rank correction for global auxiliary coupling unless the full mixed row layout is proven compatible with the backend's owned-row block layout.
- [x] Wire failure-policy behavior for implicit global local solves.
- [x] Preserve direct-only lowering for purely algebraic global models after the DAE/runtime refactor.

**Required tests**

- [x] No additional `test_AuxiliaryStateTypes.cpp` global-scope semantics extension was needed beyond the existing scope-vocabulary coverage and focused global lifecycle/assembly regressions.
- [x] Add a partitioned global mixed-DAE consistent-initialization test in `test_AuxiliaryStateModel.cpp` or a new lifecycle-focused auxiliary test file.
- [x] Add a partitioned global event-reset regression in `test_BoundaryIntegralInput.cpp`.
- [x] Add a monolithic global FE-coupled DAE test in a dedicated systems test file.
- [x] Add an MPI FSILS regression for global monolithic coupling that checks owner-row residuals, zero dropped matrix entries, and serial/parallel equivalence.
- [x] Add a failure-policy regression proving a nonconverged implicit global step does not silently succeed.

### Node Scope

**Current baseline**

- Node storage, owned/ghost indexing, ghost sync, scope-aware deployment
  filtering, and the owner-backed sparse/block-sparse monolithic path are
  implemented for the fixed-stride deliverable.
- Direct FE field references are limited to `Node` and `C0` nodal spaces, which
  is the correct baseline rule.

**Remaining implementation**

- [x] Make deployment-region expansion node-aware: cell/material filters to unique nodes, boundary markers to boundary nodes, interface markers to interface nodes, and topology-region filters to region-owned nodes.
- [x] Preserve owned/ghost partitioning correctly for restricted node subsets.
- [x] Implement a sparse or block-sparse monolithic `Node` layout and assembly path.
- [x] Define node auxiliary row ownership from the same backend permutation/owner map used by field rows when `use_backend_row_ownership_for_assembly` is active.
- [x] Gate native FSILS full-node auxiliary rows on complete nodal component blocks; otherwise use condensation/reduction or a backend that supports arbitrary mixed auxiliary rows.
- [x] Transfer deployment-path ragged node blocks to `Documentation/plan_auxiliary_state_ragged_layout_followup.md`.
- [x] Keep direct-field references restricted to `C0` nodal spaces and require mediated handles for all other spaces.

**Required tests**

- [x] Extend `test_AuxiliaryStateIndexing.cpp` for restricted node subsets and owned/ghost indexing.
- [x] Extend `test_AuxiliaryStateManager.cpp` for restricted node registration.
- [x] Transfer ragged node registration coverage to the ragged follow-on plan.
- [x] Extend `test_AuxiliaryStateManagerMPI.cpp` for restricted-node ghost sync, rollback, and restart behavior.
- [x] Add a monolithic node sparse-Jacobian parity test against a dense reference on a small mesh.
- [x] Add an MPI FSILS node-scope monolithic assembly test that verifies backend row ownership, no off-owner row writes, and no dropped entries.
- [x] Add a Node-scope solve/update regression proving correct ghost update after owner-row solution updates.
- [x] Keep the existing non-`C0` direct-field negative tests and add restricted-node coverage.

### Cell Scope

**Current baseline**

- Partitioned per-cell stepping, runtime DAE lifecycle support, scope-aware
  deployment filtering, and independent-cell monolithic local condensation are
  implemented for the fixed-stride deliverable.
- Deployment-path ragged support is transferred to the separate ragged follow-on plan.

**Remaining implementation**

- [x] Route `Cell` deployment filtering through the centralized scope-aware resolver.
- [x] Support consistent initialization, failure policies, and event hooks for cell-local partitioned and monolithic DAEs.
- [x] Keep cell-scope monolithic local condensation backend-facing as field-owned row updates; never append one row per cell to FSILS unless an explicit owned-row sparse path is implemented.
- [x] Transfer deployment-path ragged cell blocks to `Documentation/plan_auxiliary_state_ragged_layout_followup.md`.
- [x] Add stable restart/remap metadata for restricted cell subsets.
- [x] Explicitly reject or redirect cell-to-cell coupled models that do not satisfy local-condensation assumptions.

**Required tests**

- [x] Keep and extend the existing cell local-condensation equivalence tests.
- [x] Add a partitioned cell mixed-DAE test with algebraic rows.
- [x] Add a partitioned cell event-context regression proving event functions and resets see the correct entity index.
- [x] Add restricted `cellSet` / `materialIdSet` deployment tests.
- [x] Transfer ragged cell deployment coverage to the ragged follow-on plan.
- [x] Add an MPI FSILS regression proving condensed cell updates affect only owned field rows and produce zero dropped FSILS entries.
- [x] Add a restricted-cell restart serialization test.

### QuadraturePoint Scope

**Current baseline**

- Cell-volume QP auto-layout exists.
- Explicit `qpOffsets()` and inferred layout parity already have regression coverage.
- Monolithic QP local condensation exists for independent per-QP models.

**Remaining implementation**

- [x] Keep the scope definition as cell-volume quadrature only for this plan.
- [x] Route covered-cell discovery through the centralized scope-aware resolver instead of custom ad hoc handling.
- [x] Support consistent initialization, failure policies, and event hooks for QP-local partitioned and monolithic DAEs.
- [x] Keep QP monolithic local condensation backend-facing as field-owned row updates; do not materialize one backend row per quadrature point in FSILS.
- [x] Transfer deployment-path ragged QP blocks to `Documentation/plan_auxiliary_state_ragged_layout_followup.md`.
- [x] Finalize restart/remap metadata as covered-cell ids plus `qpOffsets`.
- [x] Preserve strict diagnostics for incompatible consumer quadrature layouts, unsupported non-cell consumers, and dormant-but-selected deployments.

**Required tests**

- [x] Keep the existing auto-layout vs explicit-`qpOffsets()` parity test.
- [x] Add a partitioned QP mixed-DAE test.
- [x] Add a region-restricted QP covered-cell test.
- [x] Add dormant unused deployment and incompatible-consumer negative tests.
- [x] Add an MPI FSILS regression proving condensed QP updates preserve owned-row assembly and produce zero dropped FSILS entries.
- [x] Add a restart serialization test for `qpOffsets` plus restricted covered cells.

### Region Scope

**Current baseline**

- `Region` is now a first-class fixed-stride storage/indexing scope.
- The codebase has region-aware FE quantities, topology-region metadata, lookup caches, restart/remap metadata, region-local FE quantity providers, deterministic MPI row ownership, partitioned DAE lifecycle coverage, and an owner-routed Region monolithic bordered/reduced strategy for MPI/FSILS.

**Remaining implementation**

- [x] Add `AuxiliaryStateScope::Region` to `AuxiliaryStateTypes.h`, convenience factories, string conversion, and documentation.
- [x] Add `AuxiliaryBlockIndexing::createRegion()` plus manager/storage support.
- [x] Build region lookup caches from `TopologyAnalysisContext`.
- [x] Implement scope-aware deployment for one-model-per-region materialization on disconnected or multi-region meshes.
- [x] Add region-local FE quantity providers so each region-scoped auxiliary entity can evaluate only its own region-restricted quantities.
- [x] Support partitioned region-local DAEs with consistent initialization, failure policy, event context, and restart.
- [x] Support monolithic region-local DAEs with lifecycle coverage once the Region monolithic assembly strategy exists.
- [x] Define deterministic region-row ownership, such as owner of the globally lowest cell in the region after MPI minloc tie-breaking for shared/distributed regions.
- [x] Add a `Region` monolithic strategy that stays dense only for serial/reference or explicitly owned bordered/reduced systems and uses the explicit Region row-owner map for MPI/FSILS.
- [x] Reject FSILS full-region rows unless the row layout is represented in the backend-owned sparsity and solver metadata.

**Required tests**

- [x] Extend `test_AuxiliaryStateTypes.cpp` and `test_AuxiliaryStateIndexing.cpp` for `Region`.
- [x] Extend `test_AuxiliaryStateManager.cpp` and auxiliary scope-resolution coverage for region block registration, owner metadata propagation, and restricted region subsets.
- [x] Add serial `FESystem` tests for one-model-per-region deployment on disconnected meshes.
- [x] Add FE-coupled region-average input tests proving each region sees only its own cells.
- [x] Add a monolithic region coupling test against a hand-built dense reference.
- [x] Add an MPI test proving region identity and restricted deployment are rank-stable on partitioned meshes.
- [x] Add an MPI FSILS region-scope test proving exactly one owner contributes each region row and zero FSILS entries are dropped.

## Recommended Implementation Order

1. [x] Add `Region` vocabulary and indexing support, and centralize scope-aware deployment resolution.
2. [x] Add auxiliary row-owner metadata and backend compatibility gates before adding new monolithic rows.
3. [x] Finish runtime DAE, failure-policy, and time-integrator plumbing.
4. [x] Implement the real `Node` monolithic sparse path.
5. [x] Transfer deployment-path ragged support and broader entity-local provider follow-ons out of this fixed-stride plan.
6. [x] Complete `Region` monolithic strategy after the partitioned runtime and ownership contract.
7. [x] Add focused serial, MPI, restart, FSILS, and FE-coupling tests for each fixed-stride scope.
8. [x] Update docs and mark older scope-planning docs as superseded.

## Test Plan

### Extend Existing Tests

- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateTypes.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateIndexing.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateManager.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateStorage.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryOperators.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryModelBuilder.cpp`
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateModel.cpp`; lifecycle coverage landed in focused systems/auxiliary tests.
- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_BoundaryIntegralInput.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp`
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackendMPI.cpp`; MPI owner-map coverage landed in FSILS parity and auxiliary scope-completion tests.
- [x] `Code/Source/solver/FE/Tests/Unit/Backends/test_BackendOptions.cpp`
- [x] `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsAssemblyParityMPI.cpp`
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp`; solver metadata coverage is exercised through backend option normalization and FSILS assembly tests.
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Sparsity/test_DistributedSparsityPattern.cpp`; owner-map sparsity behavior is covered by FSILS assembly parity tests.
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Assembly/test_GhostContributionManagerMPI.cpp`; raw reverse accumulation remains isolated from ordinary owned-row FSILS assembly.
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Assembly/test_FsilsSolutionViewGhostUpdateMPI.cpp`; owner-to-ghost solution refresh is covered by focused FSILS vector semantics and Node owner-update regressions.
- [x] No fixed-stride scope-completion changes required in `Code/Source/solver/FE/Tests/Unit/Assembly/test_FESystemSerialParallelEquivalenceMPI.cpp`; serial/parallel auxiliary equivalence is covered by focused auxiliary scope-completion MPI regressions.

### Add New Focused Tests

- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryScopeResolution.cpp`
- [x] No separate `Code/Source/solver/FE/Tests/Unit/Systems/test_AuxiliaryScopeCompletion.cpp` file was needed; fixed-stride systems coverage landed in `test_AuxiliaryScopeResolution.cpp`, `test_BoundaryIntegralInput.cpp`, and MPI scope-completion tests.
- [x] `Code/Source/solver/FE/Tests/Unit/Assembly/test_AuxiliaryScopeCompletionMPI.cpp`
- [x] No separate `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsAuxiliaryRowOwnershipMPI.cpp` file was needed; owner-row FSILS coverage landed in `test_FsilsAssemblyParityMPI.cpp` and `test_AuxiliaryScopeCompletionMPI.cpp`.

### Scope-Level Coverage Matrix

- [x] `Global`: partitioned DAE, monolithic DAE, failure-policy handling
- [x] `Node`: restricted deployment, MPI owned/ghost lifecycle, sparse monolithic parity
- [x] `Cell`: restricted deployment, local-condensation parity, partitioned DAE
- [x] `QuadraturePoint`: auto-layout parity, restricted covered cells, partitioned DAE, restart metadata
- [x] `Region`: region indexing, one-model-per-region deployment, FE-coupled region-local input, partitioned and monolithic DAE lifecycle, deterministic owner metadata, MPI-stable restricted identity, and owner-routed monolithic FSILS bordered/reduced coupling
- [x] `FSILS row-owned backend`: `usesOwnedRowOperator()` / `usesOwnedRowLayout()` true where expected, `ownsFeDofRow()` / `ownsFeDof()` owner queries match the auxiliary owner map, no non-owned row writes, no dropped entries, serial/parallel equivalence, correct post-solve ghost sync
- [x] `FSILS compatibility gates`: native mixed auxiliary layouts that are not complete nodal-interleaved `[0, dof_per_node)` block partitions fail before `FsilsMatrix` construction; eligible layouts normalize through `BackendOptions` and preserve BlockSchur metadata
- [x] `FSILS vector semantics`: ordinary RHS/solution vectors call owner-to-ghost `updateGhosts()` after solve/update, while raw overlap accumulation is covered only by explicit contribution-buffer tests

## Progress Log

- [x] 2026-04-21: Added `AuxiliaryStateScope::Region` across public types, bindings, block indexing, manager/storage registration, restart schema scope naming, and assembly output-scope vocabulary.
- [x] 2026-04-21: Added scope-aware deployment expansion in `FESystem::finalizeAuxiliaryLayout()` for fixed-stride `Node`, `Cell`, `QuadraturePoint`, `Facet`, and `Region` projections; `Global` and `Boundary` deployments are metadata-only.
- [x] 2026-04-21: Added partitioned retry/reject/restore handling for `AuxiliaryStepResult` and `AuxiliaryFailurePolicy` in standard and multirate stepping paths.
- [x] 2026-04-21: Verified with `cmake --build build-fe-check --target test_fe_auxiliary -j2`, `ctest --test-dir build-fe-check -R "^FE_Auxiliary_Tests$" --output-on-failure`, `cmake --build build-fe-check --target test_fe_systems -j2`, and `ctest --test-dir build-fe-check -R "^FE_Systems_Tests$" --output-on-failure`.
- [x] 2026-04-21: Added explicit auxiliary mixed-block assembly-mode and row-ownership policy metadata, propagated it through `FESystem` solver-option augmentation, and added FSILS gates for ambiguous auxiliary blocks and native nodal-component partitions.
- [x] 2026-04-21: Verified the row-owned mixed-layout contract with `cmake --build build-fe-check --target test_fe_backends -j2`, `ctest --test-dir build-fe-check -R "^(FE_Backends_Tests|FE_Auxiliary_Tests)$" --output-on-failure`, `cmake --build build-fe-check --target test_fe_systems -j2`, and `ctest --test-dir build-fe-check -R "^FE_Systems_Tests$" --output-on-failure`.
- [x] 2026-04-21: Added concrete auxiliary row-owner map storage, scope-owner expansion helpers for `Global`, `Node`, `Cell`, `QuadraturePoint`, and `Region`, mixed-layout owner queries, FSILS native-row map validation, and registry-to-`FESystem` propagation for explicit auxiliary row-owner maps.
- [x] 2026-04-21: Verified concrete row-owner map support with `cmake --build build-fe-check --target test_fe_auxiliary -j2`, `cmake --build build-fe-check --target test_fe_backends -j2`, `cmake --build build-fe-check --target test_fe_systems -j2`, and `ctest --test-dir build-fe-check -R "^(FE_Auxiliary_Tests|FE_Backends_Tests|FE_Systems_Tests)$" --output-on-failure`.
- [x] 2026-04-21: Wired the mixed-layout owner query into FSILS backend-level distributed sparsity and numeric assembly coverage, added off-owner matrix-write diagnostics/rejection in FSILS insertion paths, and added an MPI mixed field+auxiliary owner-map regression that requires zero off-owner writes and zero dropped entries.
- [x] 2026-04-21: Verified FSILS owner-map coverage with `cmake --build build-fe-check --target test_fe_backends_mpi -j2`, `mpiexec -n 2 ./build-fe-check/test_fe_backends_mpi --gtest_filter=FsilsAssemblyParityMPI.MixedAuxOwnerMapDrivesSparsityAndNumericAssembly`, `ctest --test-dir build-fe-check -R "test_fe_backends_mpi" --output-on-failure`, and `ctest --test-dir build-fe-check -R "^FE_Backends_Tests$" --output-on-failure`.
- [x] 2026-04-21: Wired one-time consistent initialization through live `FESystem` runtime paths using `AuxiliaryInitializationSolver`; structural algebraic-row auto-initialization is limited to partitioned DAEs, while monolithic blocks remain Newton-owned unless a model provides an explicit initialization hook. Added a global partitioned mixed-DAE regression in `test_BoundaryIntegralInput.cpp` proving committed state and output values are initialized before runtime use.
- [x] 2026-04-21: Wired `AuxiliaryEventManager` into partitioned standard and multirate stepping after successful local advances, applying `resetAfterEvent()` transitions before scattering work state. Added a global partitioned event-reset regression in `test_BoundaryIntegralInput.cpp`.
- [x] 2026-04-21: Made partitioned event handling entity-aware by allocating event-manager state per materialized entity and passing entity index, history, and effective step size into event detection/reset contexts. Added a cell-scope event-context regression in `test_BoundaryIntegralInput.cpp`.
- [x] 2026-04-21: Propagated model structural row kinds and event mode into registered `AuxiliaryStateSpec` metadata, with lifecycle regressions asserting mixed DAE row-kind and event-mode visibility through `AuxiliaryStateManager`.
- [x] 2026-04-21: Verified existing monolithic auxiliary `xdot` evaluation uses active FE time-integration stencils through `AuxiliaryModelBuilder.EndToEnd_MonolithicAssembly_UsesGeneralizedAlphaStencil`.
- [x] 2026-04-22: Centralized `FESystem` auxiliary scope/entity resolution into `resolveAuxiliaryDeploymentScope_()`, preserving marker-based projections for `Cell`, `Node`, `QuadraturePoint`, `Facet`, and `Region`. Added focused `test_AuxiliaryScopeResolution.cpp` coverage for material-id projection and monolithic nonsmooth rejection.
- [x] 2026-04-22: Adopted the fixed-stride scope-completion decision for ragged deployment paths: deployment-path ragged blocks remain rejected with an explicit setup diagnostic and are tracked as follow-on ragged-layout work.
- [x] 2026-04-22: Added a monolithic nonsmooth/complementarity finalize-time gate so these models fail before mixed-layout/backend setup until semismooth or active-set policy support exists.
- [x] 2026-04-22: Added an MPI FSILS vector-semantics regression proving ordinary `updateGhosts()` is owner-to-ghost only and raw overlap reverse accumulation is available only through `FsilsVector::accumulateRawContributionsAndUpdateGhosts()`.
- [x] 2026-04-22: Preserved owned/ghost metadata for restricted `Node` deployments by reordering materialized node maps into an owned prefix plus ghost suffix and adding focused scope-resolution coverage.
- [x] 2026-04-22: Added focused material-restricted `QuadraturePoint` coverage proving covered-cell discovery is resolved before explicit `qpOffsets()` validation and block registration.
- [x] 2026-04-22: Added serial `Region` scope coverage proving disconnected topology components materialize as one auxiliary entity per topology region.
- [x] 2026-04-22: Integrated smooth monolithic event/reset hooks into the accepted-step finalization path, including `alpha_f == 1` and generalized-alpha stage transforms, while keeping nonsmooth/complementarity monolithic hooks rejected.
- [x] 2026-04-22: Surfaced constraint-group, DAE-index, mass-diagonal, event-count, nonsmooth flag, and nonsmooth-policy metadata into live `AuxiliaryStateSpec` registration; added regressions for direct-only global algebraic lowering and monolithic global mixed-DAE FE-backed input assembly.
- [x] 2026-04-22: Closed the global monolithic MPI/FSILS contract by keeping `Global`/`Boundary` monolithic auxiliary unknowns in explicitly single-owned bordered/reduced storage, rejecting native global scalar rows that cannot prove a common nodal-interleaved FSILS layout, and adding a 2-rank `FESystem` regression for zero off-owner writes, zero dropped entries, field matrix/vector parity, and exact analytic bordered `B`/`Ct` shared-node contributions.
- [x] 2026-04-22: Closed the Cell and QuadraturePoint MPI/FSILS local-condensation row-owned contract by globally ordering condensed slots with block/entity/component keys, reducing condensed factors before owner-row filtering, avoiding per-entity output-gradient allreduces for local-condensed records, allowing zero-local-cell QP blocks to register empty `{0}` offsets when the deployment is globally non-empty, and adding 2-rank regressions for zero off-owner writes, zero dropped FSILS entries, zero-owned-rank participation, and row-owned dense/FSILS effective assembly parity.
- [x] 2026-04-22: Closed the Cell and QuadraturePoint runtime DAE lifecycle gap by passing entity indices through consistent-initialization residuals and model hooks, adding Cell/QP partitioned mixed-DAE initialization regressions, verifying Cell/QP partitioned failure-policy reject/restore behavior, and covering QP partitioned plus Cell/QP monolithic event/reset hooks.
- [x] 2026-04-22: Closed the remaining runtime/contract bookkeeping by adding a focused global implicit-step failure-policy reject/restore regression, marking smooth/event-hook lifecycle plumbing complete for fixed-stride scope completion, and explicitly deferring semismooth/complementarity handling to the nonsmooth follow-on path.
- [x] 2026-04-22: Closed stable entity identity and restart/remap metadata for fixed-stride restricted deployments by adding explicit entity-remap metadata to `AuxiliaryStateManager`, restart-schema validation for entity ids/QP covered cells/region memberships, `FESystem` region lookup caches keyed by `TopologyAnalysisContext` region ids, and focused Region/Cell/Node/QP scope-resolution regressions.
- [x] 2026-04-22: Added topology-region-local FE-backed `regionIntegral()` and `regionAverage()` providers as entity-local inputs, routed explicit domain functional evaluation through owned cell subsets, fixed material-marker region filtering to use the same subset path, and added a Region-scope FE average regression proving each topology region consumes only its own cells.
- [x] 2026-04-22: Closed the Region ownership and partitioned runtime contract before Region monolithic assembly by adding MPI-stable restricted Region entity-id union, deterministic globally-lowest-owned-cell Region row owners, Region row-owner propagation into monolithic layout metadata, FSILS native full-region-row rejection, Region partitioned DAE/failure/event/restart regressions, serial owner-map coverage, and a 2-rank MPI Region identity/owner-map regression.
- [x] 2026-04-22: Implemented the Region monolithic bordered/reduced strategy by owner-routing Region auxiliary rows through deterministic row-owner metadata, reducing owner-routed D/g/Ct/dF_dxdot rows for replicated bordered solves, enabling topology-region FE input gradients over explicit cell sets, and adding serial dense-reference, monolithic Region lifecycle, and 2-rank FSILS no-off-owner/no-dropped-entry regressions.
- [x] 2026-04-22: Verified Region monolithic scope coverage with `cmake --build build-fe-check --target test_fe_auxiliary test_fe_assembly_mpi test_fe_systems -j2`, focused Region serial/lifecycle/MPI regressions, `ctest --test-dir build-fe-check -R '^(FE_Auxiliary_Tests|FE_Systems_Tests)$' --output-on-failure`, and `ctest --test-dir build-fe-check -R '^test_fe_assembly_mpi_mpi_2$' --output-on-failure`.
- [x] 2026-04-22: Implemented the Node owner-backed sparse/block-sparse monolithic path by deriving node auxiliary row ownership from backend DOF permutation owners, selecting local condensation for FSILS owner-backed Node blocks, keeping native full-node FSILS rows gated on complete nodal component blocks, reducing separated B and D^-1C factors for distributed sparse updates, and adding boundary-node projection plus 2-rank dense-reference/FSILS no-off-owner/no-dropped-entry regressions.
- [x] 2026-04-22: Verified Node owner-backed scope coverage with `cmake --build build-fe-check --target test_fe_auxiliary test_fe_assembly_mpi -j2`, `ctest --test-dir build-fe-check -R '^FE_Auxiliary_Tests$' --output-on-failure`, focused 2-rank `AuxiliaryScopeCompletionMPI.NodeLocalCondensationUsesOwnerBackedSparseFsilsRowsWithoutDroppedEntries`, focused 2-rank `AuxiliaryScopeCompletionMPI.*LocalCondensation*`, and `ctest --test-dir build-fe-check -R '^test_fe_assembly_mpi_mpi_2$' --output-on-failure`.
- [x] 2026-04-22: Closed the remaining non-ragged Node scope contract by adding explicit `TopologyRegion` deployment expansion to region-local nodes/cells/regions/facets, fixed-stride restricted-node indexing and restart metadata regressions, MPI restricted-node ghost/rollback/restart coverage, an owner-update ghost-refresh regression, and restricted-node non-`C0` direct-field rejection coverage.
- [x] 2026-04-22: Verified non-ragged Node scope completion with `cmake --build build-fe-check --target test_fe_auxiliary test_fe_systems test_fe_assembly_mpi -j2`, `ctest --test-dir build-fe-check -R '^(FE_Auxiliary_Tests|FE_Systems_Tests)$' --output-on-failure`, and `ctest --test-dir build-fe-check -R '^test_fe_assembly_mpi_mpi_2$' --output-on-failure`.
- [x] 2026-04-22: Closed the remaining fixed-stride Cell and QuadraturePoint contract items by adding finalize-time diagnostics for entity-local bindings that cannot be addressed by stable original entity ids, rejecting entity-local auxiliary-output coupling for local-condensed Cell/QP models, and adding restricted Cell/QP restart-schema validation for entity ids, QP covered-cell ids, and `qpOffsets`.
- [x] 2026-04-22: Verified fixed-stride Cell and QuadraturePoint contract closure with `cmake --build build-fe-check --target test_fe_auxiliary test_fe_systems test_fe_assembly_mpi -j2`, focused Cell/QP scope-resolution regressions, `ctest --test-dir build-fe-check -R '^(FE_Auxiliary_Tests|FE_Systems_Tests)$' --output-on-failure`, `ctest --test-dir build-fe-check -R '^test_fe_assembly_mpi_mpi_2$' --output-on-failure`, and `git diff --check`.
- [x] 2026-04-22: Closed the fixed-stride documentation deliverable by updating `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`, marking older scope-related plans superseded, and moving deployment-path ragged work into `Documentation/plan_auxiliary_state_ragged_layout_followup.md`.

## Definition Of Done

- [x] `Region` is a first-class scope in the public API and runtime.
- [x] `Global`, `Node`, `Cell`, `QuadraturePoint`, and `Region` all have explicit scope semantics, deployment rules, and restart metadata.
- [x] `Partitioned` runtime respects local solve failure and DAE lifecycle hooks.
- [x] `Monolithic` runtime has a scope-appropriate strategy for every scope in this plan.
- [x] Any monolithic auxiliary rows assembled into FSILS have explicit owner metadata, compatible distributed sparsity, and zero dropped-entry diagnostics.
- [x] FSILS auxiliary rows and vectors remain owned-row authoritative, with owner queries matching the backend row map and post-solve ghost synchronization verified in MPI.
- [x] Native FSILS mixed auxiliary rows are accepted only when their solver block descriptors form a complete nodal-interleaved `dof_per_node` layout and their solver options are normalized through backend metadata.
- [x] Any monolithic auxiliary rows not compatible with FSILS full-row storage are condensed, lowered, or routed through a backend that supports their row layout.
- [x] Scope-completion tests are green in serial and MPI suites.
- [x] `Docs/AuxiliaryState/README.md` matches the implemented fixed-stride behavior.
