# Application Layer Implementation Plan (Checklist)

Goal: Add a new `Code/Source/solver/Application/` layer that allows running the new OOP Mesh/FE/Physics solver using existing `solver.xml` files unchanged, with the only XML addition being:

```xml
<Use_new_OOP_solver>true</Use_new_OOP_solver>
```

inside `<GeneralSimulationParameters>`.

This document is a **detailed checklist** of work items required to implement and validate that plan while keeping the **legacy solver almost completely unchanged** (only the minimal `Use_new_OOP_solver` parameter + a minor early-dispatch change in `main.cpp`, plus build system wiring to compile/link the new code).

---

## Non‑Negotiable Constraints (Must Hold)

- [x] **Identical XML schema:** Existing XML files run unchanged on both solvers; the only new element is `<Use_new_OOP_solver>`.
- [x] **Early dispatch:** Selection happens immediately after parsing `<GeneralSimulationParameters>` and **before** any legacy infrastructure is built.
- [ ] **Complete separation when enabled (`Use_new_OOP_solver=true`):**
  - [x] `ComMod` is never created/populated.
  - [x] Legacy mesh reading is never called (no `load_msh`, no legacy mesh readers).
  - [ ] New Mesh/FE/Physics stack handles mesh, FE system creation, physics modules, solvers, and output.
- [x] **Minimal external changes:** Legacy code path remains logically identical. Only minimal edits are allowed:
  - [x] `Parameters` gets a minimal boolean flag (if required by parser behavior) that defaults to `false` and does not affect legacy execution.
  - [x] `Code/Source/solver/main.cpp` gets an early-dispatch check + call into `ApplicationDriver`.
  - [x] Build-system wiring needed to compile/link `Application` (must not change runtime behavior unless flag is enabled).
- [x] **No duplicate XML parsing:** The Application layer reuses the existing `Parameters` class for full parsing (schema remains the legacy schema + one optional element).
- [ ] **Graceful unsupported-feature handling:** New solver throws/prints a clear error identifying the unsupported feature, lists supported features, and suggests `Use_new_OOP_solver=false` for the legacy solver.

---

## Definition of Done (Acceptance Criteria)

- [ ] Running an existing `solver.xml` **without** `<Use_new_OOP_solver>`:
  - [ ] Executes the legacy solver unchanged (same code path, no behavior changes).
- [ ] Running an existing `solver.xml` **with** `<Use_new_OOP_solver>true</Use_new_OOP_solver>`:
  - [ ] Executes the new Application/OOP solver path.
  - [ ] Loads mesh via new Mesh IO using `<Mesh_file_path>`.
  - [ ] Creates Physics modules from `<Add_equation>` definitions (at minimum: heat/Poisson first; fluid next).
  - [ ] Applies BCs from `<Add_BC>` definitions (at minimum: scalar BCs first; then fluid BCs).
  - [ ] Solves using FE infrastructure and outputs VTK results.
  - [ ] `ComMod` is not constructed/used anywhere.
- [ ] Same XML input produces comparable results between legacy and new solver for supported cases.

---

## Phase 0 — Recon / Read‑Only Study (Before Writing New Code)

- [x] Inspect XML root + schema expectations in `Code/Source/solver/Parameters.h`:
  - [x] Confirm root element name (e.g. `svMultiPhysicsFile`).
  - [x] Locate `GeneralSimulationParameters` struct and confirm how options are stored (types, defaults).
  - [x] Identify the canonical field names needed for the first end-to-end run: time stepping, VTK output toggles, restart/continuation.
- [x] Inspect XML parsing behavior in `Code/Source/solver/Parameters.cpp`:
  - [x] Determine whether unknown XML elements are ignored or cause failure.
  - [x] Decide whether `<Use_new_OOP_solver>` requires a minimal addition to `Parameters` parsing (preferred: tolerant/optional).
- [x] Identify the earliest point in `Code/Source/solver/main.cpp` where:
  - [x] `xml_file` is known.
  - [x] Legacy objects would otherwise be created (e.g. `Simulation`, `ComMod`, `read_files()` calls).
- [x] Review new stack entry points and APIs:
  - [x] Mesh IO: `Code/Source/solver/Mesh/IO/MeshIO.h` (+ readers).
  - [x] FE system/time stepping/solvers: `Code/Source/solver/FE/Systems/FESystem.h`, `FE/TimeStepping/TimeLoop.h`, `FE/TimeStepping/NewtonSolver.h`.
  - [x] Backend selection: `Code/Source/solver/FE/Backends/Interfaces/BackendFactory.h`.
  - [x] Physics modules: `Code/Source/solver/Physics/Formulations/Poisson/PoissonModule.h`, `.../NavierStokes/IncompressibleNavierStokesVMSModule.h`.
- [x] Decide and document (in code/logging) the initial supported subset:
  - [x] Start: `heatS`/`heatF` (Poisson-style) steady-state.
  - [x] Next: transient heat.
  - [x] Next: `fluid`/`stokes` (Navier–Stokes) steady-state and transient.

---

## Phase 1 — Build System Scaffolding (New Application Library)

### 1.1 Create directory skeleton

- [x] Create `Code/Source/solver/Application/` directory structure:
  - [x] `Code/Source/solver/Application/CMakeLists.txt`
  - [x] `Code/Source/solver/Application/Application.h`
  - [x] `Code/Source/solver/Application/Core/`
    - [x] `ApplicationDriver.h/.cpp`
    - [x] `SimulationBuilder.h/.cpp`
  - [x] `Code/Source/solver/Application/Translators/`
    - [x] `MeshTranslator.h/.cpp`
    - [x] `EquationTranslator.h/.cpp`
    - [x] `BoundaryConditionTranslator.h/.cpp`
    - [x] `MaterialTranslator.h/.cpp`
  - [ ] `Code/Source/solver/Application/Tests/Integration/` (optional gated tests)
    - [ ] `test_ApplicationDriver.cpp`

### 1.2 Define the `svapplication` target

- [x] Implement `Code/Source/solver/Application/CMakeLists.txt`:
  - [x] Add `svapplication` library target (static/shared consistent with existing project conventions).
  - [x] Link dependencies:
    - [x] `svphysics`
    - [x] `svfe`
    - [x] `svmesh`
    - [x] TinyXML2 library target/variable used by `Parameters` parsing.
  - [x] Public include path so consumers can `#include "Application/..."` (consistent with FE/Mesh/Physics include layout).
  - [x] Add `APPLICATION_BUILD_TESTS` option and `add_subdirectory(Tests)` when enabled.
- [x] Ensure build-order correctness:
  - [x] If the main solver build currently does not add FE/Mesh/Physics subdirectories, add the minimal build wiring required so `svmesh`, `svfe`, and `svphysics` targets exist before linking `svapplication`.
  - [x] Keep any such additions purely build-system changes (no legacy runtime behavior changes).

### 1.3 Wire `svapplication` into the main executable

- [x] Update `Code/Source/solver/CMakeLists.txt`:
  - [x] Add `add_subdirectory(Application)` (and any prerequisites needed so the target exists).
  - [x] Link `svapplication` into `${SV_MULTIPHYSICS_EXE}` (or the appropriate solver executable target).
  - [x] Ensure the legacy executable still builds/links without enabling the new solver at runtime.

---

## Phase 2 — Minimal Dispatch Flag (Parameters + Early Main Dispatch)

### 2.1 Add/verify XML flag support

- [x] Decide how `<Use_new_OOP_solver>` is handled:
  - [ ] **Preferred:** unknown XML elements are ignored by `Parameters`, so no changes required.
  - [x] If not ignored:
    - [x] Add `Use_new_OOP_solver` as an optional boolean parameter under `GeneralSimulationParameters` in `Code/Source/solver/Parameters.h`.
    - [x] Parse it in `Code/Source/solver/Parameters.cpp` with default `false`.
    - [x] Confirm the legacy solver ignores this flag (i.e., no branching inside legacy code based on it).
- [x] Add at least one sample XML snippet to internal docs (or a test) demonstrating the flag placement:
  ```xml
  <svMultiPhysicsFile>
    <GeneralSimulationParameters>
      <Use_new_OOP_solver>true</Use_new_OOP_solver>
    </GeneralSimulationParameters>
    <!-- ... legacy schema unchanged ... -->
  </svMultiPhysicsFile>
  ```

### 2.2 Implement `ApplicationDriver::shouldUseNewSolver()`

- [x] Implement `Application/Core/ApplicationDriver.h/.cpp`:
  - [x] `shouldUseNewSolver(xml_file)` performs **minimal parsing only**:
    - [x] `LoadFile(xml_file)`
    - [x] Navigate `svMultiPhysicsFile -> GeneralSimulationParameters -> Use_new_OOP_solver`
    - [x] Robust boolean parsing: accept `true/false`, `1/0`, `yes/no` (case-insensitive, whitespace tolerant).
    - [x] Return `false` on missing flag or parse errors.
  - [x] Ensure `shouldUseNewSolver()` does **not** instantiate `Parameters`, `Simulation`, or `ComMod`.

### 2.3 Add early dispatch in `main.cpp` (minor change)

- [x] Update `Code/Source/solver/main.cpp`:
  - [x] Include `Application/Core/ApplicationDriver.h`.
  - [x] After locating `xml_file` (and after MPI init if required), add:
    - [x] `if (ApplicationDriver::shouldUseNewSolver(xml_file)) { ApplicationDriver::run(xml_file); finalize MPI if needed; return 0; }`
  - [x] Confirm the dispatch occurs **before** any legacy objects are created or legacy file reading begins.
  - [x] Confirm legacy path remains unchanged when flag is absent/false.

---

## Phase 3 — Application Core (Build + Orchestration)

### 3.1 Create stable public include façade

- [x] Implement `Code/Source/solver/Application/Application.h`:
  - [x] Provide convenient includes for Application users (driver/builder/translators).
  - [x] Avoid pulling legacy headers or unnecessary heavy includes.

### 3.2 Define SimulationComponents + SimulationBuilder skeleton

- [x] Implement `Application/Core/SimulationBuilder.h`:
  - [x] Define `SimulationComponents` struct:
    - [x] Mesh map keyed by `<Add_mesh name="...">`.
    - [x] `primary_mesh`.
    - [x] FE system.
    - [x] Physics modules collection.
    - [x] Backend factory + linear solver.
    - [x] Time history.
  - [x] Define `SimulationBuilder` class and method skeletons:
    - [x] `loadMeshes()`
    - [x] `createFESystem()`
    - [x] `createPhysicsModules()`
    - [x] `setupSystem()`
    - [x] `createSolvers()`
    - [x] `allocateHistory()`
- [x] Implement `Application/Core/SimulationBuilder.cpp` with minimal compile-ready stubs:
  - [x] Constructor stores `const Parameters&`.
  - [x] `build()` calls the internal steps in a deterministic order.
  - [x] Add structured logging (at least step-by-step messages) to aid debugging.

### 3.3 Implement `ApplicationDriver::run()` skeleton

- [x] Implement `Application/Core/ApplicationDriver.cpp`:
  - [x] `run(xml_file)`:
    - [x] Parse full XML with existing `Parameters` (`params.read_xml(xml_file)`).
    - [x] Call `runWithParameters(params)`.
  - [x] `runWithParameters(params)`:
    - [x] Create `SimulationBuilder`, build components.
    - [x] Determine steady vs transient using `number_of_time_steps`.
    - [x] Call `runSteadyState()` or `runTransient()`.
  - [x] Add clear “new solver enabled” banner logging + echo supported features.

---

## Phase 4 — Mesh Translation (New Mesh Stack Only)

### 4.1 Implement `MeshTranslator::loadMesh()`

- [x] Implement `Application/Translators/MeshTranslator.h/.cpp`:
  - [x] Read `mesh_file_path` from `MeshParameters`.
  - [x] Detect format from extension (`.vtu`, `.vtk`, etc.) if the Mesh IO API needs it.
  - [x] Load mesh through the new Mesh IO layer (not legacy mesh readers).
  - [x] Store/propagate mesh name and domain ID metadata as needed by Physics/FE layers.

### 4.2 Apply face/boundary labels (from `<Add_face>`)

- [x] Define the labeling strategy:
  - [x] Determine how the new Mesh representation stores boundary markers.
  - [x] Determine how face meshes (e.g. VTP) map onto volume mesh boundary entities.
  - [x] Decide how “face name” becomes an integer marker used by FE/Physics BC application.
- [x] Implement `applyFaceLabels(mesh, face_params)`:
  - [x] For each face: load the face mesh using Mesh IO.
  - [x] Compute/set boundary entity markers in the volume mesh.
  - [x] Create a stable mapping `face_name -> marker_id` accessible by BC translation.
  - [x] Add clear errors if a face mesh cannot be matched/applied.

### 4.3 Apply domain labels (optional domain ID / domain files)

- [x] Implement `applyDomainLabels(mesh, params)`:
  - [x] Handle `domain_id` if present.
  - [x] Fail clearly if `domain_file_path` is provided (not supported yet).
  - [x] Define behavior when multiple domains are present in one volume mesh.

---

## Phase 5 — Equation Translation (Start with Heat/Poisson)

### 5.1 Implement EquationTranslator dispatch and supported-type checks

- [x] Implement `Application/Translators/EquationTranslator.h/.cpp`:
  - [x] `createModule(eq_params, system, meshes)`:
    - [x] Resolve the associated mesh (currently supports single-mesh cases; errors clearly otherwise).
    - [x] Dispatch by `eq_params.type` to supported creators.
    - [x] On unsupported equation type, throw a `runtime_error` that:
      - [x] names the unsupported type,
      - [x] lists supported types,
      - [x] suggests `Use_new_OOP_solver=false` to use the legacy solver.

### 5.2 Implement heat equation path (PoissonModule)

- [x] Implement `createHeatModule()`:
  - [x] Create scalar function space from mesh cell family + polynomial order.
  - [x] Populate Poisson options:
    - [x] Diffusion coefficient from conductivity-like fields.
    - [x] Source term from domain parameters.
  - [x] Apply scalar BCs via `BoundaryConditionTranslator::applyScalarBCs()`.
  - [x] Construct `Physics::PoissonModule` and return.
- [x] Confirm FE system registration:
  - [x] `module->registerOn(system)` (or equivalent) registers unknowns/forms/Jacobians.

---

## Phase 6 — Boundary Condition Translation (Start with Scalar BCs)

### 6.1 Implement scalar BC support for Poisson/heat

- [x] Implement `Application/Translators/BoundaryConditionTranslator.h/.cpp`:
  - [x] `applyScalarBCs(bc_params, PoissonOptions&)`:
    - [x] Map legacy BC types to new Poisson options (Dirichlet/Neumann/Robin).
    - [x] Resolve boundary marker from `<Add_BC name="...">` (face name).
    - [x] Parse constant scalar value from `<Value>`.
    - [x] Handle optional `Temporal_values_file_path` / `Spatial_values_file_path`:
      - [ ] If supported in new stack: configure corresponding time/space field.
      - [x] If not supported yet: fail with clear error and guidance.
    - [ ] Handle ramping or profile flags if present:
      - [ ] Implement if feasible; otherwise fail clearly.

### 6.2 Define “face name -> marker” resolution

- [x] Implement `getBoundaryMarker(face_name)` with a documented strategy:
  - [x] Use the mapping produced by `MeshTranslator::applyFaceLabels()` (preferred).
  - [x] Fail clearly if a BC references an unknown face name.

---

## Phase 7 — Material Translation (Start with Thermal, Then Fluid)

### 7.1 Thermal properties for Poisson

- [x] Implement `Application/Translators/MaterialTranslator.h/.cpp`:
  - [x] `applyThermalProperties(domain_params, PoissonOptions&)`:
    - [x] Conductivity -> diffusion coefficient.
    - [x] Source term if present.
    - [x] Domain selection rules (single vs multiple domains).

### 7.2 Fluid properties + viscosity models (later milestone)

- [ ] Implement `applyFluidProperties(domain_params, NavierStokesOptions&)`:
  - [ ] Density mapping.
  - [ ] Viscosity mapping:
    - [ ] Constant/Newtonian viscosity.
    - [ ] Carreau–Yasuda (and any other supported models).
  - [ ] Optional body force mapping.
- [ ] Implement `createViscosityModel(domain_param)` returning the correct constitutive model instance.

### 7.3 Solid mechanics properties (later milestone)

- [ ] Implement `applySolidProperties(domain_params, ElasticityOptions&)` (if/when elasticity is supported).
- [ ] Implement constitutive model mapping (neo-Hookean, linear elastic, etc.) as required by existing XML schema.

---

## Phase 8 — FE System + Solvers + Time Integration

### 8.1 Create FE system consistent with equation(s)

- [ ] Implement `SimulationBuilder::createFESystem()`:
  - [ ] Instantiate `FE::systems::FESystem` with the chosen mesh.
  - [ ] Define unknowns/spaces as required by the first supported module (scalar field for Poisson).
  - [ ] Ensure consistent MPI communicator usage (if applicable).

### 8.2 Backend + linear solver selection

- [ ] Implement `SimulationBuilder::createSolvers()`:
  - [ ] Choose backend kind (PETSc/Trilinos) based on build availability and/or XML settings.
  - [ ] Translate legacy linear solver parameters into new backend options:
    - [ ] solver type, preconditioner, tolerances, max iterations, etc.
  - [ ] Create `BackendFactory` + `LinearSolver`.

### 8.3 System setup + dof allocation

- [ ] Implement `SimulationBuilder::setupSystem()`:
  - [ ] Finalize FE system assembly structures (dof maps, sparsity, etc.).
  - [ ] Ensure all modules have registered contributions before finalization.
- [ ] Implement `allocateHistory()`:
  - [ ] Instantiate `TimeHistory` consistent with transient/steady use.
  - [ ] Allocate vectors for solution/state/history as required by the time integrator.

---

## Phase 9 — Run Loop + Output (Steady First, Then Transient)

### 9.1 Steady-state execution

- [ ] Implement `ApplicationDriver::runSteadyState(sim, params)`:
  - [ ] Configure Newton/nonlinear solve parameters from XML (max iterations, tolerance, etc.).
  - [ ] Assemble residual/Jacobian via FE system + registered modules.
  - [ ] Solve for steady state.
  - [ ] Call `outputResults()` at final state (and optionally intermediate iterations if desired).

### 9.2 Transient execution

- [ ] Implement `ApplicationDriver::runTransient(sim, params)`:
  - [ ] Configure time stepping:
    - [ ] `number_of_time_steps`, `time_step_size`
    - [ ] generalized-α parameters (`spectral_radius_of_infinite_time_step`) if used by new time integrator
  - [ ] Implement step loop:
    - [ ] update time/history,
    - [ ] assemble and solve each step,
    - [ ] output at requested frequency.
  - [ ] Handle continuation/restart settings if/when supported:
    - [ ] If not supported initially: fail clearly with guidance.

### 9.3 VTK output

- [ ] Implement `outputResults(sim, params, step, time)`:
  - [ ] Respect `Save_results_to_VTK_format` (and any legacy output toggles relevant to VTK).
  - [ ] Define output directory + naming consistent enough for comparison with legacy.
  - [ ] Ensure the new output includes at least primary solution fields for supported equations.

---

## Phase 10 — Incremental Support Expansion (Fluid, Then More)

### 10.1 Fluid (Navier–Stokes) module support

- [ ] Extend `EquationTranslator`:
  - [ ] Support `fluid` and/or `stokes` types mapping to `IncompressibleNavierStokesVMSModule`.
  - [ ] Create mixed velocity/pressure spaces (`createMixedSpaces()`).
  - [ ] Translate fluid material properties via `MaterialTranslator::applyFluidProperties()`.
  - [ ] Translate velocity/traction/pressure BCs via `BoundaryConditionTranslator`.
- [ ] Extend `BoundaryConditionTranslator` for fluid:
  - [ ] Velocity Dirichlet (including optional profiles).
  - [ ] Traction/Neumann-like BCs.
  - [ ] Pressure outlet/inlet if represented in legacy schema.
  - [ ] Time-/space-varying BC handling (implement or fail clearly).

### 10.2 Additional equation types (explicitly staged)

- [ ] Linear elasticity (`lElas`) support.
- [ ] Multi-physics coupling (`FSI`, `CMM`, etc.) — keep explicitly unsupported until designed.
- [ ] Mesh motion / ALE / shell / ustruct — explicitly unsupported until designed.
- [ ] For every unsupported type:
  - [ ] Provide a clear runtime error listing supported types and fallback guidance.

---

## Phase 11 — Tests (Integration-Focused, Legacy-Safety Focused)

### 11.1 Unit/integration tests for early dispatch

- [ ] Add `Application/Tests/Integration/test_ApplicationDriver.cpp`:
  - [ ] Test `shouldUseNewSolver()` parsing for:
    - [ ] missing flag -> false
    - [ ] true variants (`true`, `1`, `yes`, case-insensitive) -> true
    - [ ] false variants (`false`, `0`, `no`) -> false
    - [ ] whitespace handling
  - [ ] If feasible, add a test that verifies `ApplicationDriver::run()` can parse a minimal XML without touching legacy.

### 11.2 End-to-end solver smoke tests (first: Poisson)

- [ ] Create or reuse a minimal heat/Poisson XML input and minimal mesh files for CI/local testing.
- [ ] Add a smoke test that:
  - [ ] runs the new solver path,
  - [ ] produces VTK output,
  - [ ] returns success.

### 11.3 Legacy regression safety checks

- [ ] Add at least one test/verification that:
  - [ ] legacy solver still runs when the flag is absent/false,
  - [ ] output/regression for legacy is unchanged (as much as possible).
- [ ] Add a “separation” guard:
  - [ ] Optional: build-time checks (include dependency audits) to ensure Application code does not include legacy headers like `ComMod.h`, `Simulation.h`, `read_files.h`.
  - [ ] Optional: runtime logging to confirm early dispatch avoids legacy initialization.

---

## Phase 12 — Validation Against Legacy Solver

- [ ] Select a canonical heat/Poisson case used by the legacy solver.
- [ ] Run legacy solver and record:
  - [ ] residual history / convergence behavior,
  - [ ] key field statistics (min/max/norm),
  - [ ] output files generated.
- [ ] Run new solver with the same XML + `Use_new_OOP_solver=true` and compare:
  - [ ] solution agreement within expected tolerance,
  - [ ] stable convergence behavior,
  - [ ] comparable boundary condition enforcement.
- [ ] Repeat the same validation workflow after enabling fluid support.

---

## Phase 13 — Documentation / User Guidance

- [ ] Update user-facing docs (as appropriate) to describe the new flag:
  - [ ] Where to place `<Use_new_OOP_solver>`.
  - [ ] Supported equation types/features in the new solver.
  - [ ] Clear instruction to disable the flag to run the legacy solver.
- [ ] Document known gaps/unsupported XML features and the expected error messages.

---

## Notes for Implementation Hygiene (Checklist)

- [ ] Keep `Application/` self-contained and OOP-stack-only (no legacy includes).
- [ ] Prefer clear, structured runtime errors over silent fallbacks.
- [ ] Add logging at each major build/translate/solve stage to ease debugging.
- [ ] Keep behavior gated strictly behind `<Use_new_OOP_solver>` so legacy users are unaffected.
- [ ] Implement incrementally (Poisson steady -> Poisson transient -> Fluid steady -> Fluid transient) with a runnable state at each milestone.
