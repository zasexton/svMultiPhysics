# AuxiliaryState Legacy Removal Checklist

**Date**: 2026-04-10
**Status**: In Progress
**Purpose**: strict checklist for deleting the remaining legacy `AuxiliaryState` / coupled-boundary compatibility surface after all in-tree callers are migrated.

## Scope

This checklist covers removal of the old scalar coupled-boundary AuxiliaryState path, including:

- legacy scalar authoring and registration:
  - `AuxiliaryStateBuilder`
  - `auxiliaryODE(...)`
  - `AuxiliaryStateRegistration`
  - `ODEMethod`
  - `ODEIntegrator`
  - `LegacyScalarAuxiliaryState`
- legacy coupled-boundary runtime and BC authoring:
  - `CoupledBoundaryManager`
  - `FESystem::coupledBoundaryManager(...)`
  - `CoupledBCContext`
  - `CoupledNeumannBC`
  - `CoupledRobinBC`
  - `applyCoupledNeumann(...)`
  - `applyCoupledRobin(...)`
  - `forms::bc::CoupledNaturalBC`
  - `forms::bc::CoupledRobinBC`
- legacy symbolic / convenience surface:
  - `FormExpr::boundaryIntegralValue(...)`
  - public legacy use of `FormExpr::boundaryIntegral(integrand, marker, name)`
  - `AuxiliaryDeployedInstance::bindCoupled(...)`
- legacy flat `AuxiliaryState` compatibility API:
  - `size()`
  - `values()`
  - `previous()`
  - `has()`
  - `tryIndexOf()`
  - `indexOf()`
  - `hasHistory()`
  - `previousValue()`
  - `operator[]`
  - `registerState()`
  - `resetToCommitted()`
  - `commitTimeStep()`

## Shared Primitives Not To Delete Blindly

These are not legacy-removal targets in the same sense as the compatibility APIs above. They are still used by the modern subsystem or by internal lowering paths:

- Keep `FormExpr::auxiliaryState(std::string)` for now.
  - It still backs the modern `forms::AuxiliaryState(...)` vocabulary and helper layers in:
    - `Code/Source/solver/FE/Auxiliary/AuxiliaryModelBuilder.h`
    - `Code/Source/solver/FE/Auxiliary/AuxiliaryModelDSL.h`
    - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
    - `Code/Source/solver/FE/Forms/Vocabulary.h`
- Do not delete `FormExpr::boundaryIntegral(FormExpr, int, std::string)` until the internal lowering sites are rewritten.
  - It is still synthesized internally in:
    - `Code/Source/solver/FE/Systems/FESystem.cpp:8118`
    - `Code/Source/solver/FE/Systems/FESystem.cpp:8307`
    - `Code/Source/solver/FE/Forms/SymbolicDifferentiation.cpp:946`
  - Public legacy users of the named placeholder form should still be migrated away.

## Exit Criteria

The legacy path is considered fully removed only when all of the following are true:

- [x] No non-test production code includes or calls the legacy scalar builder / registration / coupled-boundary APIs.
- [x] No `FESystem` runtime code branches on `coupled_boundary_` for setup, assembly, parameter registration, or Jacobian/sensitivity plumbing.
- [x] No default unit or physics tests rely on the legacy API surface.
- [ ] Public docs no longer teach, exemplify, or compare against the legacy path except in explicit archival notes.
- [x] `FE/CMakeLists.txt` no longer lists legacy-only headers, sources, or tests.
- [x] The flat compatibility API on `AuxiliaryState` is removed, not just deprecated.
- [x] `bindCoupled(...)` is deleted rather than left as a deprecated alias.

## Current Blocker Snapshot

As of 2026-04-10, the hard blockers are narrower than the full removal surface.
The previously identified hard migration blockers are now resolved:

- [x] Legacy Navier-Stokes outlet factory path was removed from
  `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`.
  - The legacy overload returning `CoupledNaturalBC` was deleted.
  - The factory now only exposes the modern deployed-auxiliary outlet path.
- [x] Legacy Poisson Windkessel / coupled-natural path was removed from
  `Code/Source/solver/Physics/Formulations/Poisson/PoissonBCFactories.h`, and
  the old compiler-options hook was removed from
  `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`.
  - The Poisson Windkessel path now deploys an auxiliary model and returns a
    standard natural BC.
- [x] FE runtime `coupled_boundary_` setup / assembly plumbing was removed from
  `Code/Source/solver/FE/Systems/FESystem.h`,
  `Code/Source/solver/FE/Systems/FESystem.cpp`,
  `Code/Source/solver/FE/Systems/SystemSetup.cpp`, and
  `Code/Source/solver/FE/Systems/SystemAssembly.cpp`.
  - Legacy coupled-boundary setup/assembly branches are gone.
  - `BoundaryConditionManager` now lowers auxiliary outputs through the modern
    generalized auxiliary path.

What remains now is mostly cleanup outside the core runtime:

- public documentation still teaches or references the legacy path
- a small number of analysis tests still use legacy provenance strings such as
  `"CoupledBoundaryManager"` in expected metadata
- public notes still need to clarify the status of the named
  `boundaryIntegral(..., marker, name)` placeholder form

## Strict Production Blockers

These are the in-tree non-test callers and runtime dependencies that still
prevent deleting the legacy surface.

### 1. Physics Modules Still Using The Legacy Path

- [x] Migrate `Code/Source/solver/Physics/Formulations/NavierStokes/NavierStokesBCFactories.h`.
  - The legacy overload and legacy includes were removed.
  - Only the modern deployed-auxiliary outlet path remains.

- [x] Migrate `Code/Source/solver/Physics/Formulations/Poisson/PoissonBCFactories.h`.
  - The legacy coupled-natural implementation was replaced with a deployed
    auxiliary model plus a standard natural BC.

- [x] Remove the legacy compiler-option hook from `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.cpp`.
  - `system.coupledBoundaryManager(...).setCompilerOptions(...)` is gone.

### 2. FE Runtime Still Carries Legacy Coupled-Boundary Plumbing

- [x] Remove `CoupledBoundaryManager` ownership and accessors from `Code/Source/solver/FE/Systems/FESystem.h` and `Code/Source/solver/FE/Systems/FESystem.cpp`.
  - `FESystem::coupledBoundaryManager(...)` overloads and `coupled_boundary_`
    are gone.

- [x] Remove coupled-boundary parameter-registration plumbing from `Code/Source/solver/FE/Systems/SystemSetup.cpp`.
  - The `parameter_registry_.addAll(coupled_boundary_->parameterSpecs(), ...)`
    path is gone.

- [x] Remove coupled-boundary assembly plumbing from `Code/Source/solver/FE/Systems/SystemAssembly.cpp`.
  - The legacy `setCoupledValues(...)` / coupled sensitivity chain-rule path is
    gone from assembly.

- [x] Delete `Code/Source/solver/FE/Systems/CoupledBoundaryManager.h` and `Code/Source/solver/FE/Systems/CoupledBoundaryManager.cpp`.
  - The files were deleted from the tree and removed from the FE build lists.

### 3. Legacy Surface Still Present In Production Headers / Sources

These are mostly not independent hard blockers. They are the production-side
deletion targets that remain after the physics/runtime blocker clusters are
resolved.

- [x] Delete `Code/Source/solver/FE/Auxiliary/AuxiliaryStateBuilder.h`.

- [x] Delete the legacy-only pieces from `Code/Source/solver/FE/Auxiliary/AuxiliaryState.h`.
  - Removal targets:
    - `AuxiliaryStateRegistration`
    - flat compatibility methods listed in the Scope section above

- [x] Delete `LegacyScalarAuxiliaryState` from:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.h`
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryStateManager.cpp`

- [x] Delete the legacy scalar integrator from:
  - `Code/Source/solver/FE/Systems/ODEIntegrator.h`
  - `Code/Source/solver/FE/Systems/ODEIntegrator.cpp`

- [x] Delete the legacy coupled-BC context and evaluator aliases from `Code/Source/solver/FE/Constraints/CoupledBCContext.h`.

- [x] Delete the legacy coupled BC descriptor types:
  - `Code/Source/solver/FE/Constraints/CoupledNeumannBC.h`
  - `Code/Source/solver/FE/Constraints/CoupledRobinBC.h`

- [x] Delete the legacy coupled-BC helper layer:
  - `Code/Source/solver/FE/Systems/CoupledBoundaryConditions.h`

- [x] Delete the legacy coupled-BC boundary-condition wrappers:
  - `Code/Source/solver/FE/Forms/CoupledBCs.h`

- [x] Delete the deprecated binding sugar from:
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.h`
  - `Code/Source/solver/FE/Auxiliary/AuxiliaryBindings.cpp`
  - Removal target:
    - `AuxiliaryDeployedInstance::bindCoupled(...)`
  - The deprecated alias was removed from the public API.

- [x] Delete `FormExpr::boundaryIntegralValue(std::string)` from:
  - `Code/Source/solver/FE/Forms/FormExpr.h`
  - `Code/Source/solver/FE/Forms/FormExpr.cpp`
  - Code and tests were migrated; only docs still reference the old name.

## Strict Test Blockers

These tests still directly exercise the legacy surface and must be deleted, rewritten, or moved to an archival / compatibility-only target before final removal.

### 1. Physics Tests

- [x] `Code/Source/solver/Physics/Tests/Unit/test_NavierStokesLegacyBCs.cpp`
  - Updated to assert the modern auxiliary input/output surface instead of
    `system.coupledBoundaryManager()`.

- [x] `Code/Source/solver/Physics/Tests/Unit/test_NavierStokesOutletFactory.cpp`
  - Updated to stop checking for `CoupledNaturalBC` and
    `sys.coupledBoundaryManager()`.

### 2. FE Unit Tests For Legacy Authoring / Registration

- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateBuilder.cpp`
  - The legacy-builder-only test file was deleted.

- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateTypes.cpp`
  - The `AuxiliaryStateRegistration` backward-compatibility coverage was removed.

- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_ODEIntegrator.cpp`
  - The legacy-integrator-only test file was deleted.

- [x] `Code/Source/solver/FE/Tests/Unit/Auxiliary/test_AuxiliaryStateStorage.cpp`
  - Legacy flat-storage coverage was removed; the file now tests block-only behavior.

- [x] `Code/Source/solver/FE/Tests/Unit/Constraints/test_NeumannBC.cpp`
  - Legacy `CoupledNeumannBC` and flat-storage coverage were removed.

- [x] `Code/Source/solver/FE/Tests/Unit/Constraints/test_RobinBC.cpp`
  - Legacy `CoupledRobinBC` and flat-storage coverage were removed.

### 3. FE Unit Tests For Coupled-Boundary Runtime / Helpers

- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_CoupledBoundaryManager.cpp`
  - Removed from the default `test_fe_systems` target.

- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_CoupledBoundaryConditionHelpers.cpp`
  - Removed from the default `test_fe_systems` target.

- [x] `Code/Source/solver/FE/Tests/Unit/Constraints/test_GlobalConstraintConsistencyMPI.cpp`
  - The deprecated coupled-helper comparison test was removed.
  - The MPI target now verifies only the modern auxiliary-driven path.

- [x] `Code/Source/solver/FE/Tests/Unit/Analysis/test_BoundaryConditionDescriptor.cpp`
  - Legacy coupled-BC descriptor tests and includes were removed.

### 4. FE Unit Tests For Legacy Symbolic Terminals

- [x] `Code/Source/solver/FE/Tests/Unit/Analysis/test_FormStructureAnalyzer.cpp`
  - Legacy `boundaryIntegralValue(...)` coverage was removed.

- [x] `Code/Source/solver/FE/Tests/Unit/Forms/test_JITValidation.cpp`
  - The strict-mode coverage now targets the supported unresolved auxiliary-state symbol path.

- [ ] `Code/Source/solver/FE/Tests/Unit/Forms/test_CoupledTerminals.cpp`
  - Uses the public named `FormExpr::boundaryIntegral(..., marker, name)` placeholder syntax.
  - Uses `FormExpr::auxiliaryState(...)` as part of the coupled-terminal test surface.
  - This test must be rewritten carefully, not blindly deleted, because `FormExpr::auxiliaryState(...)` still has a modern meaning.

### 5. FE Unit Tests With Legacy Includes / Metadata Expectations

- [x] `Code/Source/solver/FE/Tests/Unit/Assembly/test_FESystemSerialParallelEquivalenceMPI.cpp`
  - Stale `Systems/CoupledBoundaryManager.h` include was removed.

- [x] `Code/Source/solver/FE/Tests/Unit/Systems/test_FESystem.cpp`
  - Stale `Systems/CoupledBoundaryManager.h` include was removed.

- [ ] `Code/Source/solver/FE/Tests/Unit/Analysis/test_ContributionDescriptor.cpp`
  - Still hard-codes `"CoupledBoundaryManager"` as a contribution origin string.

- [ ] `Code/Source/solver/FE/Tests/Unit/Analysis/test_Phase19InterfaceAware.cpp`
  - Still simulates / asserts `CoupledBoundaryManager` contribution records.

## Strict Documentation And Build Blockers

- [ ] Remove legacy examples and migration-window language from `Code/Source/solver/FE/Docs/AuxiliaryState/README.md`.
  - Current blockers:
    - legacy compatibility comparison table
    - migration-window text saying old helpers remain functional
    - explicit old example using `auxiliaryODE(...)` and `CoupledNaturalBC`
    - explicit note that the legacy `CoupledBoundaryManager` path remains functional

- [ ] Remove stale public references from `Code/Source/solver/FE/README.md`.
  - Current blocker:
    - table entry still advertising `CoupledBoundaryManager.h`

- [ ] Remove legacy terminal mention from `Code/Source/solver/FE/Forms/VOCABULARY.md`.
  - Current blocker:
    - `boundaryIntegralValue("Q")`

- [ ] Update `Code/Source/solver/FE/Systems/PLAN.md` and any remaining FE design docs that still mention the legacy path as an existing implementation option.

- [x] Update `Code/Source/solver/FE/CMakeLists.txt`.
  - Completed in this pass:
    - removed `Systems/CoupledBoundaryManager.cpp` from active sources
    - removed `Systems/CoupledBoundaryManager.h` from active public header lists
    - removed `Systems/CoupledBoundaryConditions.h` from active public header lists
    - removed `Forms/CoupledBCs.h` from active public header lists
    - removed legacy system test sources from `test_fe_systems`
    - removed legacy auxiliary/system header and test entries such as
      `Constraints/CoupledNeumannBC.h`,
      `Constraints/CoupledRobinBC.h`,
      `Auxiliary/AuxiliaryStateBuilder.h`,
      `Systems/ODEIntegrator.*`, and their dedicated tests

## Symbol-By-Symbol Removal Checklist

### A. Legacy Scalar Authoring And Registration

- [x] Delete `systems::AuxiliaryStateBuilder`.
- [x] Delete `systems::auxiliaryODE(...)`.
- [x] Delete `systems::AuxiliaryStateRegistration`.
- [x] Delete `systems::ODEMethod`.
- [x] Delete `systems::ODEIntegrator`.
- [x] Delete `systems::LegacyScalarAuxiliaryState`.

### B. Legacy Coupled-Boundary Runtime

- [x] Delete `systems::CoupledBoundaryManager`.
- [x] Delete `FESystem::coupledBoundaryManager(...)` overloads and `coupled_boundary_`.
- [x] Delete `constraints::CoupledBCContext`.
- [x] Delete `constraints::CoupledBCEvaluator`.
- [x] Delete `constraints::CoupledVectorBCEvaluator`.
- [x] Delete `constraints::CoupledNeumannBC`.
- [x] Delete `constraints::CoupledRobinBC`.
- [x] Delete `systems::bc::applyCoupledNeumann(...)`.
- [x] Delete `systems::bc::applyCoupledRobin(...)`.
- [x] Delete `forms::bc::CoupledNaturalBC`.
- [x] Delete `forms::bc::CoupledRobinBC`.

### C. Legacy Symbolic / Convenience Surface

- [x] Delete `AuxiliaryDeployedInstance::bindCoupled(...)`.
- [x] Delete `FormExpr::boundaryIntegralValue(std::string)`.
- [ ] Remove public user-facing use of `FormExpr::boundaryIntegral(integrand, marker, name)`.
- [ ] Decide whether the named `boundaryIntegral(...)` primitive remains as an internal lowering-only node or is replaced entirely.

### D. Flat Compatibility Surface On `AuxiliaryState`

- [x] Delete `AuxiliaryState::size()` flat-compat meaning.
- [x] Delete `AuxiliaryState::values()`.
- [x] Delete `AuxiliaryState::previous()`.
- [x] Delete `AuxiliaryState::has()`.
- [x] Delete `AuxiliaryState::tryIndexOf()`.
- [x] Delete `AuxiliaryState::indexOf()`.
- [x] Delete `AuxiliaryState::hasHistory()`.
- [x] Delete `AuxiliaryState::previousValue()`.
- [x] Delete `AuxiliaryState::operator[]`.
- [x] Delete `AuxiliaryState::registerState()`.
- [x] Delete `AuxiliaryState::resetToCommitted()`.
- [x] Delete `AuxiliaryState::commitTimeStep()`.
- [x] Delete the legacy flat buffers and maps from `AuxiliaryState`.

## Recommended Deletion Order

Use this order. Do not try to delete the legacy surface in one patch.

### Phase 1: Remove Zero-Caller Deprecated Sugar

- [x] Delete `bindCoupled(...)`.
- [ ] Remove any remaining docs / plans that still recommend `bindCoupled(...)`.
- [x] Delete `FormExpr::boundaryIntegralValue(std::string)` in code.
  - Docs still reference the old helper and need cleanup.

### Phase 2: Migrate Remaining Physics Callers

- [x] Rewrite the remaining legacy outlet overload in Navier-Stokes factories.
- [x] Rewrite the remaining legacy Windkessel / coupled natural BC path in Poisson factories.
- [x] Add a modern replacement for the Poisson compiler-options hook.

### Phase 3: Remove FE Runtime Dependence On `CoupledBoundaryManager`

- [x] Remove `coupled_boundary_` parameter-registration plumbing from setup.
- [x] Remove `setCoupledValues(...)` and coupled auxiliary sensitivity plumbing from assembly.
- [x] Delete `CoupledBoundaryManager` ownership from `FESystem`.

### Phase 4: Delete Legacy BC Authoring Surface

- [x] Delete `CoupledBoundaryConditions.h`.
- [x] Delete `CoupledBCs.h`.
- [x] Delete `CoupledBCContext.h`.
- [x] Delete `CoupledNeumannBC.h`.
- [x] Delete `CoupledRobinBC.h`.

### Phase 5: Delete Legacy Scalar Auxiliary Authoring Surface

- [x] Delete `AuxiliaryStateBuilder.h`.
- [x] Delete `AuxiliaryStateRegistration`.
- [x] Delete `ODEMethod`.
- [x] Delete `ODEIntegrator`.
- [x] Delete `LegacyScalarAuxiliaryState`.

### Phase 6: Delete Flat `AuxiliaryState` Compatibility API

- [x] Remove the deprecated flat methods from `AuxiliaryState`.
- [x] Remove the legacy flat storage members from `AuxiliaryState`.
- [x] Delete or rewrite tests that still exercise flat registration and flat slot access.

### Phase 7: Clean Docs, Tests, And Build Lists

- [ ] Remove all remaining legacy examples from public docs.
- [x] Remove legacy test targets from default FE / physics test suites.
- [x] Remove legacy file entries from `FE/CMakeLists.txt`.

## Final Verification Checklist

- [ ] `rg` for `AuxiliaryStateBuilder`, `auxiliaryODE(`, `AuxiliaryStateRegistration`, `CoupledBoundaryManager`, `coupledBoundaryManager(`, `applyCoupledNeumann(`, `applyCoupledRobin(`, `CoupledNaturalBC`, `CoupledRobinBC`, `boundaryIntegralValue(`, and `bindCoupled(` returns no production hits.
- [ ] `rg` for the same symbols returns no default-test hits.
- [ ] `rg` for the same symbols returns no public-doc hits outside archival migration notes.
- [x] `FE/CMakeLists.txt` no longer references removed files or tests.
- [ ] Full FE and physics test suites pass without any legacy compatibility shims.
