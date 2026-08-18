# ALE Mesh-Motion Data and Coupled Mesh-Displacement Gap Closure

This document collates the implementation outlines for closing the ALE moving-mesh data path gap and adding true coupled monolithic mesh-displacement support in the new OOP solver.

The two supported paths must be distinct:

- Prescribed or FE-smoothed ALE: mesh motion is prescribed/derived data consumed by physics.
- Coupled monolithic ALE: mesh displacement is a solved unknown; mesh velocity and acceleration are derived from that unknown and the time-integration scheme.

Relevant current code anchors:

- `Code/Source/solver/Physics/Formulations/NavierStokes/IncompressibleNavierStokesVMSModule.cpp`
- `Code/Source/solver/FE/Systems/FESystem.cpp`
- `Code/Source/solver/FE/Systems/FieldRegistry.h`
- `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- `Code/Source/solver/FE/Assembly/Assembler.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/MovingMesh/MovingDomainOrchestrator.cpp`
- `Code/Source/solver/Application/Core/ApplicationDriver.cpp`
- `Code/Source/solver/Application/Core/SimulationBuilder.cpp`

## Implementation Status

Status after the core implementation pass:

- [x] Prescribed ALE mesh velocity is now represented as FE prescribed data, not as an unconstrained solve unknown.
- [x] FE setup excludes prescribed and derived moving-domain fields from the global unknown vector, block layout, and solve DOF count while preserving field-level DOF metadata for interpolation.
- [x] FE assembly can gather prescribed field coefficients and derived mesh-velocity coefficients through `FieldSolutionAccess`.
- [x] Moving-domain orchestration syncs standard Mesh motion fields into FE prescribed buffers after geometry advancement.
- [x] Navier-Stokes ALE defaults to prescribed mesh-velocity data and supports coupled-displacement ALE through a simple source option.
- [x] ALE field/source binding is centralized in `FE::systems::ALEBinding` so physics formulations keep moving-domain setup out of residual code.
- [x] Coupled ALE registers `mesh_displacement` as an unknown and `mesh_velocity` as a first time derivative derived from that unknown.
- [x] Forms/AD geometry sensitivity treats `meshVelocity()` as a mesh-motion-sensitive terminal and seeds its value and gradient derivatives from the active time-integration coefficient.
- [x] Forms installation supports extra trial fields so fluid residual rows can receive tangent columns for coupled mesh displacement.
- [x] A baseline harmonic mesh-motion physics module exists for solved mesh-displacement rows.
- [x] Mesh-displacement binding for mesh-motion equations is centralized in `FE::systems::MeshDisplacementBinding`, keeping physics modules math-first.
- [x] Harmonic and pseudoelastic mesh-motion modules now write their weak forms directly in Forms primitives.
- [x] Targeted FE systems, FE moving-mesh, and physics tests verify prescribed layout, prescribed sync, ALE residual use of prescribed data, coupled ALE field registration, and coupled metadata.

Remaining items are qualification and application-integration work rather than the core Mesh/FE/Forms/Navier-Stokes data path:

- [ ] Add an end-to-end Application/SimulationBuilder configuration path for XML-driven prescribed and coupled ALE examples.
- [ ] Add restart persistence for prescribed buffers, current coordinates, mesh-displacement history, and derived velocity metadata.
- [ ] Add dedicated finite-difference tests for full Navier-Stokes coupled-ALE mesh-displacement tangents.
- [ ] Add serial coupled-solve smoke tests with the harmonic mesh-motion module.
- [ ] Add MPI qualification for prescribed buffers, coupled displacement layout, ghost updates, and derived velocity consistency.
- [ ] Record backend/preconditioner support and fail-closed behavior for each coupled ALE block layout.
- [ ] Add optional derived mesh acceleration for schemes/modules that require second derivatives of mesh displacement.

## Design Invariants

- [x] `meshVelocity()` must not silently create an unconstrained Newton unknown.
- [x] Prescribed/FE-smoothed ALE must read mesh velocity from mesh-motion data, not from an unsolved field block.
- [x] Coupled monolithic ALE must solve mesh displacement as a real unknown with residual rows and tangent columns.
- [x] Coupled monolithic ALE must derive mesh velocity from mesh displacement and the active time-integration stencil.
- [x] Geometry-sensitive physics tangents must include derivatives with respect to mesh displacement when coupled mode is enabled.
- [x] Unsupported combinations must fail closed with diagnostics before setup or first assembly.

## Field Source Semantics

Add explicit field-source semantics so FE can distinguish solved unknowns from prescribed data and derived moving-domain quantities.

- [x] Add a source/participation enum to `FieldSpec` and `FieldRecord`, for example:

```cpp
enum class FieldSourceKind {
  Unknown,
  PrescribedData,
  DerivedFromUnknown
};
```

- [x] Preserve existing behavior by defaulting all ordinary fields to `Unknown`.
- [x] Add metadata for `DerivedFromUnknown`, including source field, derivative role, time level, and derivative stencil.
- [x] Add diagnostics that reject invalid combinations, such as a derived mesh velocity without a source mesh-displacement field.
- [ ] Update field summary/debug output to show each field source kind.
- [x] Add tests proving legacy/static fields remain `Unknown` by default.

## Prescribed or FE-Smoothed ALE Data Path

This path is for motion supplied by `MeshMotion`, `MovingDomainOrchestrator`, prescribed maps, or FE-smoothed mesh-motion solves where the fluid solve consumes mesh motion but does not solve for it.

### FE Field Layout

- [x] Register prescribed mesh-motion fields as `PrescribedData`, not as `Unknown`.
- [x] Exclude `PrescribedData` fields from global `dof_handler_`.
- [x] Exclude `PrescribedData` fields from `field_map_`, block layout, backend vector size, sparsity patterns, Newton unknown count, and solver block metadata.
- [x] Still build enough interpolation/evaluation metadata for prescribed fields to evaluate values and gradients at quadrature points.
- [x] Add a setup diagnostic if a prescribed field lacks a supported interpolation layout.

### Prescribed Data Storage

- [x] Add FE-owned buffers for prescribed field coefficients.
- [x] Add version/revision metadata to prescribed buffers so stale data can be diagnosed.
- [x] Add APIs to set, resize, clear, and query prescribed field buffers.
- [ ] Ensure prescribed buffers participate in rollback/restart where needed.
- [ ] Add MPI ghost/update handling for prescribed field buffers or require sync from Mesh fields after mesh ghost exchange.

### Field Access and Assembly

- [x] Extend `FieldSolutionAccess` so a field source can be either global solution-vector data or prescribed coefficient data.
- [x] Update `StandardAssembler::populateFieldSolutionData()` to gather prescribed field coefficients without reading `state.u`.
- [x] Keep `StandardAssembler::populateMovingDomainFieldData()` using the same moving-domain terminal path, but allow the backing field to be prescribed data.
- [x] Ensure `meshVelocity()`, `meshAcceleration()`, `meshDisplacement()`, previous mesh velocity, and predicted mesh velocity all fail clearly if their backing source is missing.
- [x] Add value and gradient tests for prescribed vector fields used through moving-domain terminals.

### FE Mesh-Motion APIs

- [x] Add `FESystem` APIs such as `addMeshMotionDataField(...)` for prescribed moving-domain fields.
- [x] Add `FESystem` APIs to bind prescribed data fields to `MeshMotionFieldRole`.
- [x] Add a sync path that copies standard Mesh motion fields into FE prescribed buffers without writing into the Newton state vector.
- [x] Keep the existing `syncBoundMeshMotionFieldsToState()` path for true unknown-based workflows.
- [x] Add diagnostics that distinguish "synced prescribed mesh-motion data" from "solved mesh-motion unknown".

### Navier-Stokes ALE Registration

- [x] Change ALE auto-registration so `mesh_velocity` is not registered as a normal unknown in prescribed/FE-smoothed mode.
- [x] Centralize ALE binding in `FE::systems::resolveALEBinding(...)`, not in the Navier-Stokes residual implementation.
- [x] Find an existing mesh-velocity binding first.
- [x] If no binding exists and prescribed mode is active, register a prescribed mesh-velocity data field.
- [x] If no binding exists and coupled mode is active, require a derived mesh-velocity binding from mesh displacement.
- [x] If ALE is enabled but no valid mesh-motion source exists, fail before setup or first assembly with a clear diagnostic.
- [x] Keep ALE-disabled behavior unchanged.

### Application Orchestration

- [ ] Parse or construct moving-domain configuration before FE setup when ALE/moving mesh is requested.
- [ ] In prescribed/FE-smoothed mode, create moving-domain data fields before `FESystem::setup()`.
- [ ] Wire `MovingDomainOrchestrator::makeBeforePhysicsSolveCallback()` into the OOP time loop.
- [x] After each successful mesh-motion advance, sync Mesh motion fields into FE prescribed buffers.
- [ ] Ensure sync occurs after current-coordinate and ghost-coordinate exchange and before physics assembly.
- [ ] Emit diagnostics with geometry revision, field-buffer revision, motion mode, and mesh-velocity source.

### Prescribed/FE-Smoothed Tests

- [x] FE setup test: prescribed `mesh_velocity` does not increase global DOFs.
- [x] FE setup test: prescribed `mesh_velocity` does not appear in block layout.
- [x] FE assembly test: `meshVelocity()` reads prescribed values at quadrature points.
- [x] FE assembly test: `div(meshVelocity())` or mesh-velocity gradient uses prescribed field gradients.
- [x] Physics test: ALE Navier-Stokes DOF count is `u + p`, not `u + p + w_mesh`.
- [x] Physics test: ALE residual changes when prescribed mesh velocity changes.
- [ ] Application smoke test: prescribed moving mesh advances, syncs mesh velocity, and assembles ALE Navier-Stokes.
- [ ] Negative test: ALE enabled with no mesh-motion source fails with actionable diagnostics.

## Coupled Monolithic Mesh-Displacement Support

This path is for true coupled ALE/FSI-style workflows where mesh displacement is part of the nonlinear unknown vector.

### Unknown and Derived Field Model

- [x] Register `mesh_displacement` as an `Unknown`.
- [x] Register `mesh_velocity` as `DerivedFromUnknown(mesh_displacement)`, not as a separate unknown.
- [ ] Register `mesh_acceleration` as derived where second-order schemes need it.
- [x] Ensure derived mesh velocity uses the active time-integration stencil:

```text
w_mesh = d/dt(d_mesh)
```

- [x] Expose the current-step derivative coefficient, for example:

```text
d(w_mesh) / d(d_mesh_current) = active_stencil_current_coefficient
```

- [x] Add diagnostics if coupled mode tries to use a prescribed mesh velocity independent of the solved mesh displacement.

### Mesh-Motion Equation Module

Add a Physics/Application-owned module for the mesh-motion equation. Mesh and FE should provide infrastructure, but they should not encode the governing mesh-motion model.

- [x] Add a mesh-motion equation module or module factory path.
- [x] Support at least one baseline equation, such as harmonic smoothing.
- [x] Support pseudoelastic mesh smoothing as a separate math-first module.
- [x] Register residual rows for `mesh_displacement`.
- [x] Register tangent blocks for `dR_mesh / d(mesh_displacement)`.
- [x] Add Dirichlet constraints for fixed boundaries and moving/interface boundaries.
- [ ] Add optional coupling constraints tying mesh displacement to structural/interface displacement.
- [x] Keep mesh-motion material/model choices out of FE core.

### Trial Geometry Lifecycle

Coupled mode is complete only when the nonlinear solve assembles on trial current geometry.

- [x] Enable `GeometricNonlinearityPolicy` automatically in coupled monolithic mode.
- [x] At each nonlinear trial, update `X_cur` from the trial `mesh_displacement` state.
- [ ] Invalidate FE geometry caches, matrix-free geometry data, search structures, interface maps, and geometry-dependent operators after trial updates.
- [x] Roll back coordinates and mesh-motion fields after rejected line-search steps.
- [x] Commit geometry after accepted nonlinear states and accepted time steps.
- [ ] Keep accepted nonlinear state and accepted time-step state distinct where the time integrator requires it.
- [ ] Add diagnostics showing whether assembly used reference, committed current, or trial current geometry.

### Geometry-Sensitive Fluid Tangents

- [x] Require geometry sensitivity to be enabled when Navier-Stokes ALE is coupled to solved mesh displacement.
- [x] Configure coupled-ALE geometry sensitivity through the FE ALE binding helper and `FormInstallOptions`.
- [x] Ensure ALE fluid residuals include tangent blocks with respect to `mesh_displacement`.
- [x] Ensure `meshVelocity()` contributes the time-derivative coefficient to `dR_fluid / d(mesh_displacement)`.
- [x] Ensure current-coordinate, current-measure, inverse-Jacobian, normal, and face-measure dependencies contribute geometry-sensitive tangent terms.
- [x] Add fail-closed diagnostics if a coupled ALE form requests current geometry but no mesh-displacement sensitivity field is configured.

### Application Coupled Mode

- [x] Stop reporting `CoupledMonolithic` as unsupported once the following items are implemented.
- [ ] Add `mesh_motion.mode=coupled_monolithic` application configuration.
- [ ] Create `mesh_displacement` before `FESystem::setup()`.
- [ ] Register the mesh-motion equation module before setup.
- [x] Bind `mesh_velocity` as a derived moving-domain field.
- [ ] Bind optional `mesh_acceleration` as a derived moving-domain field where needed.
- [x] Enable Navier-Stokes ALE with the derived mesh-velocity binding.
- [x] Configure FE coordinate configuration as current geometry.
- [x] Configure block layout for fluid velocity, pressure, and mesh displacement.
- [ ] Reject unsupported backend/solver combinations before solve.
- [ ] Add clear diagnostics for field sources and coupled mesh-motion configuration.

### Solver and Backend Behavior

- [x] Ensure global unknown layout includes `mesh_displacement`.
- [x] Ensure global unknown layout excludes derived `mesh_velocity`.
- [x] Ensure Jacobian sparsity includes fluid-mesh and mesh-fluid coupling blocks where physics modules register them.
- [x] Add block-layout metadata for `mesh_displacement`.
- [ ] Confirm FSILS, Eigen, PETSc, and any selected backend either support the coupled block layout or fail closed.
- [ ] Ensure preconditioner and matrix-reuse policies invalidate on geometry and mesh-displacement changes.

### Time History, Restart, and Transfer

- [x] Store previous `mesh_displacement` states needed by the active time scheme.
- [ ] Derive previous/current/predicted mesh velocity consistently from displacement history.
- [ ] Persist `mesh_displacement`, derived velocity metadata, current coordinates, and reference coordinates in restart.
- [ ] Restore current coordinates before FE geometry caches are built after restart.
- [ ] Preserve or transfer mesh displacement and required history through remesh/adaptivity.
- [ ] Recompute derived velocity after restart, remesh, or rebase rather than treating it as an independent unknown.

### Coupled Monolithic Tests

- [x] FE setup test: coupled mode includes `mesh_displacement` as an unknown.
- [x] FE setup test: coupled mode does not include separate `mesh_velocity` unknowns.
- [ ] Time integration test: `meshVelocity()` equals the time derivative of `mesh_displacement`.
- [x] FE assembly test: current geometry updates from trial `mesh_displacement` before assembly.
- [x] FE rollback test: rejected line search restores coordinates and motion fields.
- [ ] Forms/AD test: geometry-sensitive residual has nonzero derivative with respect to mesh displacement.
- [ ] Physics test: Navier-Stokes ALE residual has finite-difference-verified tangent with respect to mesh displacement.
- [x] Mesh-motion module test: harmonic and pseudoelastic residuals and tangents pass finite-difference checks.
- [ ] Coupled solve test: small manufactured coupled ALE case converges with Newton.
- [ ] Application smoke test: XML-configured coupled monolithic ALE runs through setup and at least one time step.
- [ ] Restart test: coupled ALE restart restores displacement history, current coordinates, and derived mesh velocity.
- [ ] MPI test: coupled ALE field layout, ghosted displacement, derived velocity, and geometry update are consistent across ranks.

## Support Matrix Updates

- [ ] Mark prescribed/FE-smoothed ALE as supported only after app-level sync and smoke tests pass.
- [ ] Mark coupled monolithic ALE as unsupported until mesh-displacement equations, derived mesh velocity, trial geometry lifecycle, and tangent tests pass.
- [ ] Add release-note rows distinguishing:
  - prescribed ALE with mesh-motion data fields
  - FE-smoothed ALE with mesh-motion data fields
  - coupled monolithic ALE with solved mesh displacement
  - coupled monolithic ALE plus FSI/interface coupling
- [ ] Record exact solver/backend combinations that are qualified.
- [ ] Record MPI qualification separately from serial qualification.

## Definition of Done

- [x] Prescribed ALE can use `meshVelocity()` without adding mesh velocity to the nonlinear unknown vector.
- [x] Coupled monolithic ALE solves `mesh_displacement` as a real unknown.
- [x] Coupled monolithic ALE derives `meshVelocity()` from `mesh_displacement` and time history.
- [ ] Navier-Stokes ALE residual and tangent are finite-difference-verified for both prescribed and coupled modes.
- [ ] OOP application configuration can run at least one prescribed ALE smoke test and one coupled monolithic ALE smoke test.
- [ ] Unsupported configurations fail before solve with diagnostics that identify the missing field source, binding, geometry sensitivity, or solver support.

## Related Math-First Formulation Work

- [x] `Documentation/plan_mesh_motion_math_first_formulations.md` records the completed math-first formulation work.
- [x] `Documentation/mesh_motion_math_first_formulation_guide.md` documents the intended Forms style for harmonic, pseudoelastic, boundary, and coupled-ALE mesh-motion terms.
