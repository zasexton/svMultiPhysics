# Level-Set FE Library Migration Plan

## Purpose

Move level-set infrastructure out of the Physics library and into the FE library so
level-set methods are reusable by any physics module. The final design should make
level-set field transport, generated interface construction, cut-cell volume
diagnostics, signed-distance repair, volume correction, diagnostics, and restart
metadata physics-agnostic.

Physics modules should own only input translation and equation-specific coupling.
FE should own reusable level-set algorithms and FE-system integration services.

## Target Architecture

New FE location:

- `Code/Source/solver/FE/LevelSet/`

Target namespace:

- `svmp::FE::level_set`

Allowed dependencies for `FE/LevelSet`:

- `FE/Core`
- `FE/Forms`
- `FE/Systems`
- `FE/Spaces`
- `FE/Dofs`
- `FE/Assembly`
- `FE/Interfaces`
- `FE/Geometry`
- `Mesh`
- standard library dependencies

Forbidden dependencies for `FE/LevelSet`:

- `Physics/Core`
- `Physics/Formulations`
- any other `Physics/...` include

Physics should retain a thin adapter that:

- Parses equation-module input.
- Infers or constructs FE function spaces.
- Translates input parameters into `FE::level_set` option structs.
- Translates Physics runtime policy into FE form-install options.
- Calls FE-level installation and diagnostic APIs.

## Non-Goals

- Do not change user-facing level-set input names unless a compatibility alias is
  kept.
- Do not move FE interface geometry down into `Mesh/Geometry` when it depends on
  FE fields, DOFs, quadrature, or `FESystem`.
- Do not embed Navier-Stokes, free-surface, material, contact-angle, or fluid
  assumptions in the FE level-set library.

## Phase 1 - Establish FE/LevelSet API Skeleton

Goal: create the destination package and shared option surface without changing
behavior.

Files to create:

- [x] `Code/Source/solver/FE/LevelSet/LevelSetOptions.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetTransport.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetTransport.cpp`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetInterfaceLifecycle.cpp`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetVolume.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetVolume.cpp`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetReinitialization.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetReinitialization.cpp`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetDiagnostics.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetDiagnostics.cpp`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetRestart.h`
- [x] `Code/Source/solver/FE/LevelSet/LevelSetRestart.cpp`
- [x] Optional: `Code/Source/solver/FE/LevelSet/LevelSet.h` umbrella header

Checklist:

- [x] Add `FE_LEVELSET_HEADERS` to `Code/Source/solver/FE/CMakeLists.txt`.
- [x] Add `FE_LEVELSET_SOURCES` to `Code/Source/solver/FE/CMakeLists.txt`.
- [x] Add the new headers and sources to `svfe`.
- [x] Install `FE/LevelSet` headers with the rest of the FE public headers.
- [x] Add a minimal compile-only FE test that includes each public
  `FE/LevelSet` header.

Done when:

- `svfe` builds with the new empty or forwarding LevelSet package.
- The new package has no `Physics/...` includes.

## Phase 2 - Move Shared Level-Set Options And Utilities

Goal: move physics-neutral option types and cadence helpers into FE.

Current source:

- `Code/Source/solver/Physics/Formulations/LevelSet/LevelSetTransportModule.h`

Target content:

- `FE::level_set::LevelSetFieldSource`
- `FE::level_set::LevelSetVelocitySource`
- `FE::level_set::LevelSetFieldOptions`
- `FE::level_set::LevelSetVelocityOptions`
- `FE::level_set::LevelSetSUPGOptions`
- `FE::level_set::LevelSetReinitializationMethod`
- `FE::level_set::LevelSetReinitializationOptions`
- `FE::level_set::LevelSetVolumeCorrectionOptions`
- `FE::level_set::LevelSetInflowBoundary`
- `FE::level_set::LevelSetOutflowBoundary`
- `FE::level_set::LevelSetBoundaryOptions`
- `FE::level_set::LevelSetTransportOptions`
- `FE::level_set::shouldReinitializeLevelSet`
- `FE::level_set::shouldApplyLevelSetVolumeCorrection`

Checklist:

- [x] Move option structs and enums into `FE/LevelSet/LevelSetOptions.h`.
- [x] Remove `PhysicsJITPolicy` from `LevelSetTransportOptions`.
- [x] Represent JIT/compiler behavior through `FE::systems::FormInstallOptions`.
- [x] Move cadence helper declarations into `FE/LevelSet/LevelSetOptions.h`.
- [x] Move cadence helper definitions into `FE/LevelSet/LevelSetTransport.cpp` or
  a dedicated utility source.
- [x] Update all users to refer to `svmp::FE::level_set`.
- [x] Keep any required compatibility aliases temporary and clearly marked for
  removal.

Done when:

- Level-set option types compile without including Physics headers.
- Cadence helper tests pass from FE tests.

## Phase 3 - Move Generated Interface Lifecycle

Goal: make generated interface construction a reusable FE service.

Current files:

- `Physics/Formulations/LevelSet/LevelSetInterfaceLifecycle.h`
- `Physics/Formulations/LevelSet/LevelSetInterfaceLifecycle.cpp`

Target files:

- `FE/LevelSet/LevelSetInterfaceLifecycle.h`
- `FE/LevelSet/LevelSetInterfaceLifecycle.cpp`

Checklist:

- [x] Move `LevelSetGeneratedInterfaceOptions` into `FE::level_set`.
- [x] Move `LevelSetGeneratedInterfaceResult` into `FE::level_set`.
- [x] Move `LevelSetGeneratedInterfaceLifecycle` into `FE::level_set`.
- [x] Keep dependence on `FE::interfaces::LevelSetInterfaceDomain`.
- [x] Keep generated marker assignment in FE.
- [x] Ensure scalar field resolution is based only on `FE::systems::FESystem`.
- [x] Preserve mesh geometry, topology, ownership, and value revision metadata.
- [x] Move generated-interface lifecycle unit tests from Physics to FE where
  practical.

Done when:

- Any physics module can build a generated level-set interface domain through
  `FE::level_set` without linking to Physics.

## Phase 4 - Move Cut-Cell Volume And Volume Correction

Goal: expose volume measurement and global shift correction as generic FE
level-set operations.

Current files:

- `Physics/Formulations/LevelSet/LevelSetVolume.h`
- `Physics/Formulations/LevelSet/LevelSetVolume.cpp`

Target files:

- `FE/LevelSet/LevelSetVolume.h`
- `FE/LevelSet/LevelSetVolume.cpp`

Checklist:

- [x] Move `LevelSetVolumeOptions` into `FE::level_set`.
- [x] Move `LevelSetVolumeResult` into `FE::level_set`.
- [x] Move `LevelSetGlobalShiftCorrectionOptions` into `FE::level_set`.
- [x] Move `LevelSetGlobalShiftCorrectionResult` into `FE::level_set`.
- [x] Move `computeLevelSetCutCellVolume` overloads into FE.
- [x] Move `applyGlobalLevelSetShiftCorrection` overloads into FE.
- [x] Keep APIs accepting `FE::assembly::IMeshAccess` and
  `FE::dofs::DofHandler`.
- [x] Keep convenience APIs accepting `FE::systems::FESystem` and
  `FE::FieldId`.
- [x] Move cut-cell volume tests from Physics to FE where practical.

Done when:

- Volume diagnostics and shift correction are available from FE without any
  Physics include.

## Phase 5 - Move Signed-Distance Repair And Reinitialization Utilities

Goal: make signed-distance repair a reusable FE level-set utility.

Current files:

- `Physics/Formulations/LevelSet/LevelSetReinitialization.h`
- `Physics/Formulations/LevelSet/LevelSetReinitialization.cpp`

Target files:

- `FE/LevelSet/LevelSetReinitialization.h`
- `FE/LevelSet/LevelSetReinitialization.cpp`

Checklist:

- [ ] Move `LevelSetSignedDistanceRepairResult` into `FE::level_set`.
- [ ] Move `repairLevelSetSignedDistanceByProjection` overloads into FE.
- [ ] Keep repair implementation independent of physics equations.
- [ ] Keep support for `IMeshAccess` plus scalar `DofHandler`.
- [ ] Keep support for `FESystem` plus `FieldId`.
- [ ] Move signed-distance repair tests from Physics to FE where practical.
- [ ] Document unsupported element types and current projection limitations.

Done when:

- Signed-distance repair can be called from FE tests without linking to Physics.

## Phase 6 - Move Level-Set Output Diagnostics

Goal: make scalar level-set diagnostics available to all physics modules.

Current files:

- `Physics/Formulations/LevelSet/LevelSetDiagnostics.h`
- `Physics/Formulations/LevelSet/LevelSetDiagnostics.cpp`

Target files:

- `FE/LevelSet/LevelSetDiagnostics.h`
- `FE/LevelSet/LevelSetDiagnostics.cpp`

Checklist:

- [ ] Move `LevelSetScalarDiagnostic` into `FE::level_set`.
- [ ] Move `LevelSetOutputDiagnosticsOptions` into `FE::level_set`.
- [ ] Move `LevelSetOutputDiagnostics` into `FE::level_set`.
- [ ] Move `computeLevelSetOutputDiagnostics` overloads into FE.
- [ ] Keep diagnostic scalar names stable unless explicitly versioned.
- [ ] Move diagnostics tests from Physics to FE where practical.

Done when:

- Volume loss and signed-distance error diagnostics are FE-level services.

## Phase 7 - Move Restart Records

Goal: keep restart metadata for level-set fields and generated interfaces
physics-agnostic.

Current files:

- `Physics/Formulations/LevelSet/LevelSetRestart.h`
- `Physics/Formulations/LevelSet/LevelSetRestart.cpp`

Target files:

- `FE/LevelSet/LevelSetRestart.h`
- `FE/LevelSet/LevelSetRestart.cpp`

Checklist:

- [ ] Move `LevelSetFieldRestartRecord` into `FE::level_set`.
- [ ] Move `LevelSetGeneratedInterfaceRestartRecord` into `FE::level_set`.
- [ ] Move `LevelSetRestartSnapshot` into `FE::level_set`.
- [ ] Move `captureLevelSetFieldRestartRecord` into FE.
- [ ] Move `captureLevelSetGeneratedInterfaceRestartRecord` into FE.
- [ ] Move `optionsFromLevelSetGeneratedInterfaceRestartRecord` into FE.
- [ ] Move `levelSetGeneratedInterfaceRestartRecordMatches` into FE.
- [ ] Ensure restart records use FE-level option enums.
- [ ] Move restart tests from Physics to FE where practical.

Done when:

- Level-set restart metadata can be captured and validated without Physics.

## Phase 8 - Split Transport Form Installation From Physics Module Wrapping

Goal: make the level-set advection formulation itself FE-owned while preserving
Physics input compatibility.

Current files:

- `Physics/Formulations/LevelSet/LevelSetTransportModule.h`
- `Physics/Formulations/LevelSet/LevelSetTransportModule.cpp`

Target FE API:

```cpp
FE::systems::CoupledResidualKernels installLevelSetTransport(
    FE::systems::FESystem& system,
    std::shared_ptr<const FE::spaces::FunctionSpace> level_set_space,
    const FE::level_set::LevelSetTransportOptions& options,
    const FE::systems::FormInstallOptions& install_options = {});
```

Checklist:

- [ ] Move level-set field validation into FE.
- [ ] Move velocity field validation into FE.
- [ ] Move boundary option validation into FE.
- [ ] Move reinitialization option validation into FE.
- [ ] Move volume correction option validation into FE.
- [ ] Move field auto-registration helpers into FE.
- [ ] Move residual construction into `FE::level_set::installLevelSetTransport`.
- [ ] Move SUPG residual construction into FE.
- [ ] Move inflow boundary residual construction into FE.
- [ ] Keep operator tag stable as `"level_set"` unless a compatibility plan
  changes it.
- [ ] Return FE kernel installation metadata from the FE API.
- [ ] Update tests for residual form structure and SUPG behavior to call the FE
  API directly.

Done when:

- Level-set transport residual assembly can be installed from FE without a
  `PhysicsModule`.

## Phase 9 - Reduce Physics Level-Set Code To Input Adapter

Goal: leave Physics with only equation-module factory glue.

Current file:

- `Physics/Formulations/LevelSet/LevelSetRegister.cpp`

Checklist:

- [ ] Replace direct construction of `LevelSetTransportModule` with a thin
  Physics adapter that calls `FE::level_set::installLevelSetTransport`.
- [ ] Keep XML/input parameter parsing in Physics.
- [ ] Keep element type and space inference in Physics if it depends on
  equation-module input.
- [ ] Translate parsed values into `FE::level_set::LevelSetTransportOptions`.
- [ ] Translate Physics JIT runtime policy into `FE::systems::FormInstallOptions`.
- [ ] Preserve accepted parameter names:
  - [ ] `Level_set_field_name`
  - [ ] `Level_set_source`
  - [ ] `Auto_register_level_set_field`
  - [ ] `Velocity_field_name`
  - [ ] `Velocity_source`
  - [ ] `Constant_velocity`
  - [ ] `SUPG_enabled`
  - [ ] `Reinitialization_*`
  - [ ] `Volume_correction_*`
- [ ] Preserve supported boundary condition names:
  - [ ] `LevelSetInflow`
  - [ ] `LevelSetOutflow`
- [ ] Keep Physics tests for input translation and module registration.
- [ ] Move pure algorithm and FE-system behavior tests out of Physics.

Done when:

- Physics contains no reusable level-set algorithms.
- Physics remains backward compatible for existing level-set equation inputs.

## Phase 10 - Update Navier-Stokes And Free-Surface Consumers

Goal: update other physics modules to use FE-level level-set services directly.

Checklist:

- [ ] Update Navier-Stokes free-surface includes from
  `Physics/Formulations/LevelSet/...` to `FE/LevelSet/...`.
- [ ] Keep `UnfittedLevelSet` as a Navier-Stokes free-surface implementation
  name.
- [ ] Ensure free-surface generated interface markers still bind to `.dI(marker)`.
- [ ] Ensure curvature helpers remain FE Forms helpers, not Physics helpers.
- [ ] Verify fitted ALE paths are unaffected.
- [ ] Verify unfitted level-set paths still assemble pressure jump and surface
  tension terms.

Done when:

- Navier-Stokes consumes FE level-set interfaces without depending on
  Physics-level level-set implementation files.

## Phase 11 - Move And Reorganize Tests

Goal: align test ownership with the new architecture.

FE tests to add or move:

- [ ] Option defaults and cadence helpers.
- [ ] Field auto-registration and validation.
- [ ] Level-set transport residual structure.
- [ ] SUPG stabilization terms.
- [ ] Inflow and outflow boundary validation.
- [ ] Generated interface lifecycle.
- [ ] Cut-cell volume calculation.
- [ ] Global shift volume correction.
- [ ] Signed-distance repair by projection.
- [ ] Output diagnostics.
- [ ] Restart record capture and validation.
- [ ] Header compile tests for `FE/LevelSet`.

Physics tests to keep:

- [ ] Equation-module factory registration.
- [ ] Input parameter translation.
- [ ] Boundary condition translation.
- [ ] Constant velocity input translation.
- [ ] JIT option translation into FE install options.
- [ ] Compatibility of existing open-vessel example input files.

Done when:

- FE tests cover reusable level-set behavior.
- Physics tests cover only Physics input glue and downstream compatibility.

## Phase 12 - Remove Old Physics Level-Set Implementation

Goal: eliminate duplicate implementation and stale includes.

Checklist:

- [ ] Remove moved headers from `Physics/Formulations/LevelSet`.
- [ ] Remove moved sources from `Physics/Formulations/LevelSet`.
- [ ] Remove moved files from `Code/Source/solver/Physics/CMakeLists.txt`.
- [ ] Keep only the registration adapter if Physics still needs a
  level-set equation module entry point.
- [ ] Update all include paths across the repository.
- [ ] Confirm no stale include references remain:
  - [ ] `rg 'Physics/Formulations/LevelSet' Code Documentation tests`
- [ ] Confirm FE has no forbidden Physics dependency:
  - [ ] `rg '#include "Physics/' Code/Source/solver/FE/LevelSet`
- [ ] Confirm the old namespace is gone from migrated code:
  - [ ] `rg 'Physics::formulations::level_set' Code/Source/solver`

Done when:

- There is a single FE-owned implementation of level-set infrastructure.

## Phase 13 - Documentation And Public API Cleanup

Goal: document the new ownership model and migration contract.

Checklist:

- [ ] Update FE documentation to describe `FE::level_set` services.
- [ ] Update free-surface documentation to say Navier-Stokes consumes FE
  level-set interfaces.
- [ ] Document which level-set APIs are stable public FE APIs.
- [ ] Document which compatibility aliases, if any, are temporary.
- [ ] Update Doxygen grouping for `FE/LevelSet`.
- [ ] Add a short module-authoring note explaining how other Physics modules
  can use FE level-set services.

Done when:

- Developers can identify whether new level-set functionality belongs in FE or
  in a physics adapter.

## Phase 14 - Verification Matrix

Goal: prove the migration preserved behavior and improved dependency boundaries.

Checklist:

- [ ] Build `svfe`.
- [ ] Build `svphysics`.
- [ ] Run FE geometry tests.
- [ ] Run FE systems tests that cover form installation.
- [ ] Run new FE level-set tests.
- [ ] Run Physics unit tests.
- [ ] Run application input translation tests.
- [ ] Run open-vessel example coverage tests.
- [ ] Run moving free-surface verification tests.
- [ ] Run dependency guard checks:
  - [ ] `rg '#include "Physics/' Code/Source/solver/FE/LevelSet`
  - [ ] `rg 'Physics/Formulations/LevelSet' Code Documentation tests`

Done when:

- All migrated behavior passes in FE and Physics test suites.
- Dependency guard checks show no forbidden FE-to-Physics includes.

## Completion Criteria

- [ ] `Code/Source/solver/FE/LevelSet` contains all reusable level-set
  implementation.
- [ ] `Code/Source/solver/Physics/Formulations/LevelSet` contains no reusable
  algorithms.
- [ ] `FE::level_set` exposes generated interface, transport, volume,
  reinitialization, diagnostics, and restart APIs.
- [ ] Physics level-set registration is an adapter over `FE::level_set`.
- [ ] Navier-Stokes free-surface code uses FE level-set services directly.
- [ ] Existing level-set input files remain compatible.
- [ ] FE tests own reusable level-set behavior.
- [ ] Physics tests own only input and registration behavior.
- [ ] No `Physics/...` include appears under `FE/LevelSet`.
- [ ] No stale `Physics/Formulations/LevelSet` include remains after removal.
