# AuxiliaryState Remaining Items Implementation Plan

**Status (2026-04-22)**: Superseded for fixed-stride scope completion by
`Documentation/plan_auxiliary_state_scope_completion.md`.  This document is
retained as historical cleanup context.

**Date**: 2026-04-09
**Status**: Implemented with follow-on items noted
**Scope**: Concrete implementation plan for the remaining AuxiliaryState review items that are still unresolved after the recent MPI ghost-sync, owned/ghost layout, and spec-ergonomics fixes.

## Summary

This plan resolves the remaining AuxiliaryState review items in five ordered workstreams:

1. Migrate off the legacy flat AuxiliaryState API.
2. Add auxiliary-to-constraint lowering for strong BCs and time-varying constraint RHS updates.
3. Finish owned-only storage semantics for ghosted auxiliary blocks.
4. Make monolithic auxiliary solver metadata feed mixed-layout and backend decisions.
5. Port coupled-BC convenience wiring into the modern auxiliary/input/output path.

The workstreams are intentionally ordered to minimize churn:

- Workstream 1 removes the last production dependency on the flat API before deprecation.
- Workstream 2 addresses the highest-value functional gap using the existing constraint-update infrastructure.
- Workstream 3 finishes the ghost-storage correctness work after the registration and sync wiring changes.
- Workstream 4 builds on the new separation between FE constraints and solver-structure metadata.
- Workstream 5 adds usability sugar on the modern path after the core runtime model is stable.

## Non-Goals

- Do not delete the legacy flat API in this series.
- Do not implement general MPC or global auxiliary-driven constraints in the first auxiliary-lowering patch.
- Do not add full sparse Schur/static-condensation algorithms in the same patch that introduces solver-role metadata propagation.
- Do not expand the deprecated `CoupledBoundaryManager` feature surface unless required for migration compatibility.

## Current Code Touchpoints

- Legacy flat auxiliary API:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`
  - `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
  - `Code/Source/solver/FE/Constraints/CoupledBCContext.h`
- Constraint build and update path:
  - `Code/Source/solver/FE/Constraints/SystemConstraint.h`
  - `Code/Source/solver/FE/Constraints/StrongDirichletConstraint.h`
  - `Code/Source/solver/FE/Constraints/StrongDirichletConstraint.cpp`
  - `Code/Source/solver/FE/Systems/SystemSetup.cpp`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
- Auxiliary solver metadata path:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.h`
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.cpp`
  - `Code/Source/solver/FE/Backends/Utils/BackendOptions.h`
- Modern deployed auxiliary path:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
  - `Code/Source/solver/FE/Systems/FESystem.h`
  - `Code/Source/solver/FE/Systems/FESystem.cpp`
  - `Code/Source/solver/FE/Systems/FormsInstaller.cpp`

## Ordered Workstreams

### 1. Migrate Off The Legacy Flat Auxiliary API

**Goal**

Remove the flat `AuxiliaryState` API as a production dependency before compiler deprecation is enabled.

**Primary files**

- `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`
- `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`
- `Code/Source/solver/FE/Constraints/CoupledBCContext.h`
- `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`

**Implementation checklist**

- [x] Add a manager-level scalar compatibility helper on top of `AuxiliaryStateManager` for legacy scalar/global/boundary usage.
- [x] Define a clear internal replacement for `registerState()` that maps one legacy scalar variable to one named auxiliary block.
- [x] Replace `CoupledBoundaryManager::addAuxiliaryState()` storage registration so it no longer calls the flat `registerState()` path.
- [x] Replace flat value access in `CoupledBoundaryManager` with named block access and explicit gathers/scatters.
- [x] Replace temporary-copy logic that assumes `AuxiliaryState::values()` is the authoritative representation.
- [x] Update `CoupledBCContext` neutral accessors so they can read from the compatibility layer without requiring the flat buffer.
- [x] Keep external helper behavior unchanged for deprecated coupled-BC callers.
- [x] Add `[[deprecated]]` to the flat API only after all in-tree production consumers are removed.
- [x] Add deprecation comments that point users to `AuxiliaryStateManager` / `registerBlock()` / `getBlock(name)`.

**Validation checklist**

- [x] Existing `CoupledBoundaryManager` tests still pass with no behavior change.
- [x] Legacy coupled Neumann/Robin helper tests still pass.
- [x] No in-tree production code reads `AuxiliaryState::values()` or `registerState()` except compatibility tests.
- [x] Warning policy is checked so the new deprecations do not break current builds unintentionally.

**Definition of done**

- `CoupledBoundaryManager` no longer depends on the flat auxiliary storage path.
- The flat API remains available but is clearly deprecated and non-authoritative.

### 2. Add Auxiliary-To-Constraint Lowering

**Goal**

Allow auxiliary state or auxiliary outputs to drive FE strong Dirichlet constraints through the existing `ISystemConstraint` and `updateConstraints()` lifecycle.

**Primary files**

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- `Code/Source/solver/FE/Auxiliary/` new auxiliary-constraint descriptor files
- `Code/Source/solver/FE/Constraints/SystemConstraint.h`
- `Code/Source/solver/FE/Constraints/StrongDirichletConstraint.h`
- `Code/Source/solver/FE/Constraints/StrongDirichletConstraint.cpp`
- `Code/Source/solver/FE/Constraints/` new auxiliary-driven system-constraint files
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`

**Implementation checklist**

- [x] Introduce a new descriptor family for auxiliary-driven FE constraints separate from `AuxiliaryBlockRole`.
- [x] Define at least these descriptor fields:
  - [x] target field
  - [x] target component or all-components
  - [x] target region or boundary marker
  - [x] enforcement kind
  - [x] source value selector: auxiliary state vs auxiliary output
  - [x] state view selector when relevant
- [x] Extend the deployed auxiliary DSL with one explicit strong-Dirichlet binding API.
- [x] Store unresolved auxiliary-constraint bindings on deployed instances before setup.
- [x] During setup, lower those bindings into concrete `ISystemConstraint` objects before `AffineConstraints::close()`.
- [x] Implement a new `AuxiliaryDrivenDirichletConstraint` that reuses the DOF discovery logic from `StrongDirichletConstraint`.
- [x] In `apply(...)`, evaluate the initial auxiliary-driven value and register the constraint line(s).
- [x] In `updateValues(...)`, update only the inhomogeneity via `AffineConstraints::updateInhomogeneity(...)`.
- [x] Ensure value lookup can read from `AuxiliaryStateManager` and from resolved auxiliary outputs.
- [x] Decide and document whether the source for strong BC updates is committed state, work state, or evaluated output metadata.
- [x] Reject unsupported combinations with explicit diagnostics:
  - [x] missing block or output
  - [x] missing target field
  - [x] unsupported space/continuity
  - [x] unsupported scope/region pairing

**Validation checklist**

- [x] Add a system test where a partitioned auxiliary model drives a time-varying strong Dirichlet BC.
- [x] Add a test showing the set of constrained DOFs is fixed after setup while only inhomogeneity changes over time.
- [x] Add a negative test for invalid auxiliary output or block references.
- [x] Add an MPI test proving the auxiliary-driven Dirichlet values remain rank-consistent after synchronization.
- [x] Confirm `updateConstraints()` is sufficient and no full constraint rebuild is needed during time stepping.

**Definition of done**

- Auxiliary state can drive a time-varying strong Dirichlet BC without rebuilding the constraint graph.
- The lowering path uses the existing `ISystemConstraint` lifecycle rather than a one-off callback path.

### 3. Finish Owned-Only Storage Semantics For Ghosted Blocks

**Goal**

Make commit/reset/rollback semantics authoritative only on the owned prefix for ghosted node-scoped auxiliary blocks.

**Primary files**

- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateStorage.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`

**Implementation checklist**

- [x] Decide exact semantics for ghost entries during:
  - [x] `commitTimeStep()`
  - [x] `resetToCommitted()`
  - [x] `rollback()`
  - [x] checkpoint restore
- [x] Update fixed-stride storage lifecycle methods so only owned entries are committed/reset for `OwnedAndGhost` layouts.
- [x] Preserve current behavior for non-ghosted and owned-only layouts.
- [x] Ensure ghost entries are either left untouched or explicitly marked stale after owned-only lifecycle operations.
- [x] Trigger `syncGhosts()` at the right recovery points after reset/rollback/restore.
- [x] Tighten validation so storage owned-prefix assumptions and indexing metadata cannot diverge silently.
- [x] Audit pack/unpack semantics so owned/ghost expectations remain clear across restart.
- [x] Document the authoritative-state rule: owned entries are locally advanced and committed; ghost entries are communication-populated.

**Validation checklist**

- [x] Add a unit test proving commit only copies the owned prefix for ghosted node blocks.
- [x] Add a unit test proving reset only restores the owned prefix.
- [x] Add an MPI regression test for ghost freshness after advance.
- [x] Add an MPI regression test for ghost freshness after rollback/reset.
- [x] Add a checkpoint/restart regression for ghosted node blocks.

**Definition of done**

- Ghost entries are no longer treated as locally committed truth.
- All distributed lifecycle paths end with a correct owned/ghost state model.

### 4. Make Monolithic Auxiliary Solver Metadata Real

**Goal**

Feed auxiliary solver metadata into mixed-layout diagnostics and backend block-role decisions instead of storing it unused in `AuxiliaryOperatorRegistry`.

**Primary files**

- `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryOperatorRegistry.cpp`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryStateModel.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Backends/Utils/BackendOptions.h`

**Implementation checklist**

- [x] Extend the monolithic auxiliary block layout to carry solver metadata alongside name/count/stride/scope.
- [x] Propagate deployed-instance solver metadata into `AuxiliaryOperatorRegistry` during monolithic unknown registration.
- [x] Map auxiliary solver roles onto backend-facing block roles where applicable.
- [x] Add mixed-layout queries for:
  - [x] constraint-like auxiliary blocks
  - [x] Schur-eliminable auxiliary blocks
  - [x] special-precondition auxiliary blocks
- [x] Thread `constraint_groups` and related structural metadata into the runtime metadata model where useful.
- [x] Surface this metadata in analysis/diagnostics output so it can be inspected without backend-specific logging.
- [x] Keep the first patch metadata-only from a solver-behavior perspective.
- [x] In a follow-on patch, use the propagated metadata to guide backend block-role selection and preconditioner setup.
  Completed scope: backend normalization now materializes stable role mappings, drives PETSc field-split policy from mixed-layout metadata, and resolves FSILS BlockSchur row/column scaling plus Schur-side preconditioner choices from auxiliary block metadata.
- [x] Defer full static condensation / sparse Schur algorithms until after metadata propagation is stable.

**Validation checklist**

- [x] Add unit tests for solver metadata registration and lookup.
- [x] Add unit tests proving metadata is attached to mixed auxiliary layout blocks.
- [x] Add backend-option tests for role mapping from auxiliary blocks into generic backend block-role names.
- [x] Confirm no regression in current monolithic auxiliary solve behavior when metadata is absent.

**Definition of done**

- Auxiliary solver metadata is visible in mixed layout and backend-facing role resolution.
- `AuxiliaryBlockRole` is no longer aspirational metadata with zero consumers.

### 5. Port Coupled-BC Convenience Wiring To The Modern Path

**Goal**

Make common RCR / Windkessel / resistance setups ergonomic on the modern auxiliary/input/output path without extending the deprecated `CoupledBoundaryManager`.

**Primary files**

- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
- `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
- `Code/Source/solver/FE/Systems/FESystem.h`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp`
- `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`

**Implementation checklist**

- [x] Define an explicit modern convenience API for FE-backed auxiliary-input binding.
- [x] Keep the convenience explicit; do not infer cross-boundary or cross-instance wiring implicitly.
- [x] Support boundary-reduction handle binding with marker/region consistency checks.
- [x] Support common by-name binding when handle names already match model input names.
- [x] Reuse existing FE-quantity metadata instead of parsing legacy placeholder terminals.
- [x] Ensure deployed boundary-scoped instances can declare their bindings without manual registry plumbing.
- [x] Update deprecated coupled helper docs so they point to the modern replacement flow.
- [x] Where safe, make deprecated helper implementations forward through the modern path internally.
- [x] Preserve disambiguation for multi-outlet cases so bindings remain instance-local and marker-correct.

**Validation checklist**

- [x] Add a system test for a partitioned RCR-like model bound through the modern convenience API.
- [x] Add a monolithic auxiliary coupling test proving handle-backed bindings still preserve FE-coupling metadata.
- [x] Add a multi-outlet test proving auto-bind sugar does not cross-wire outlets with similar forms.
- [x] Add a regression test showing deprecated helper usage still functions during the migration window.

**Definition of done**

- A standard boundary-coupled auxiliary setup is concise on the modern path.
- Deprecated coupled-BC helpers are no longer the only ergonomic authoring route.


## Cross-Cutting Verification

- [x] Rebuild the touched FE targets for each workstream.
- [x] Run focused unit tests for the modified subsystem after each workstream
- [x] Add MPI coverage for any change affecting ghost state, constraints, or distributed boundary coupling.
- [x] Update auxiliary-state documentation and examples after each externally visible API change.
- [x] Keep deprecated APIs functional until all in-tree call sites have migrated.

## Exit Criteria For The Full Plan

- [x] No production in-tree code depends on the flat auxiliary-state API.
- [x] Auxiliary state can drive time-varying FE strong BC values through the standard constraint lifecycle.
- [x] Ghosted node-scoped auxiliary blocks have owned-only authoritative commit/reset semantics.
- [x] Monolithic auxiliary solver metadata is visible to mixed-layout and backend role selection.
- [x] The modern auxiliary/input/output path is the preferred ergonomic route for boundary-coupled models.
