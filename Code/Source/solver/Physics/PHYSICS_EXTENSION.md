# Physics Extension Checklist (New OOP Solver)

Goal: after the one-time scaffolding is in place, **adding a new physics requires only adding files under**
`Code/Source/solver/Physics/Formulations/<NewPhysics>/` (and optionally `Physics/Materials/*`), with **no edits** to:
- `Code/Source/solver/Application/*`
- `Code/Source/solver/Physics/CMakeLists.txt`
- `Code/Source/solver/Application/CMakeLists.txt`
- `Code/Source/solver/CMakeLists.txt`

This keeps the Application layer from “knowing” about specific formulations and prevents `if (eq_type == "...")`
growth over time.

---

## Definition Of Done (DoD)

- [ ] Adding `<Add_equation type="...">` support for a new physics requires **no code changes outside `Physics/`**.
- [ ] `Application` does not include any formulation headers (no `#include "Physics/Formulations/...`).
- [ ] `Application` contains **zero** `if/else` chains or `switch` statements over equation type strings.
- [ ] Unsupported type errors list `registeredTypes()` and include a “how to enable” hint.
- [ ] New physics can define its own options without editing `Code/Source/solver/Parameters.h`.

---

## One-Time Scaffolding (done once; enables “drop-in” physics)

### 1) Physics-side registry/factory API
- [ ] Add `Code/Source/solver/Physics/Core/EquationModuleRegistry.h` (+ `.cpp`) with:
  - `registerFactory(type_string, factory_fn)`
  - `create(type_string, input) -> std::unique_ptr<svmp::Physics::PhysicsModule>`
  - `registeredTypes() -> std::vector<std::string>`
- [ ] Define a stable factory signature (no Application types), e.g.:
  - input struct owned by Physics (see next section)
  - returns a `PhysicsModule` ready to `registerOn(system)`

### 2) Physics-owned “generic input” contract (decouples Application)
- [ ] Add `Code/Source/solver/Physics/Core/EquationModuleInput.h` that is *physics-neutral* and can represent:
  - mesh selection (mesh name + `MeshBase*` or direct `MeshBase&`)
  - domain/material properties (generic key/value map per domain, plus domain id/label if needed)
  - boundary conditions (face name/marker + kind + value(s) + flags like weak/strong)
  - module-specific options as either:
    - `std::map<std::string, std::string> options`, or
    - `std::string options_string` (JSON/YAML), or
    - `options_file_path`
- [ ] Keep this contract stable; new physics-specific needs should be encoded as data, not new fields that force
  Application edits.

### 3) Refactor Application translation to be data-driven (no per-physics code)
- [ ] Update `Code/Source/solver/Application/Translators/EquationTranslator.cpp` to:
  - build `EquationModuleInput` from `EquationParameters`/`DomainParameters`/`BoundaryConditionParameters`
  - call `EquationModuleRegistry::create(eq_type, input)` (and then `module->registerOn(system)` if not done inside)
  - on registry miss: throw with `registeredTypes()` listed
- [ ] Remove direct includes of formulation headers from Application translators (e.g. stop including
  `"Physics/Formulations/Poisson/PoissonModule.h"` in Application).

### 4) Move per-physics “legacy XML → options/BC/material” translation into Physics
- [ ] For each currently-supported physics (Poisson, Navier–Stokes, etc.), create an adapter TU inside that physics
  folder, e.g. `Code/Source/solver/Physics/Formulations/Poisson/PoissonLegacyAdapter.cpp`, that:
  - translates `EquationModuleInput` → `PoissonOptions`
  - constructs `PoissonModule`
  - registers its factory with the registry (see registration pattern below)
- [ ] De-physics the Application translators:
  - `BoundaryConditionTranslator` and `MaterialTranslator` should not contain Poisson/Navier–Stokes specifics; those
    specifics live in the formulation folder adapters.

### 5) Registration mechanism (link-time registrars)
- [ ] Add a lightweight registration macro in Physics core, e.g. `SVMP_REGISTER_EQUATION("heatS", FactoryFn)`.
- [ ] Ensure registrars are not dropped by static linking:
  - choose one approach and document it (e.g. `WHOLE_ARCHIVE` for the registrar-containing target, or build shared,
    or an object library aggregated into the executable).
  - current implementation: the solver executable links `svphysics` with `-Wl,--whole-archive/--no-whole-archive` (GNU/Clang)

### 6) Build system auto-discovers new formulation folders/sources
- [ ] Change `Code/Source/solver/Physics/CMakeLists.txt` once to avoid hard-coded source lists for formulations:
  - auto-add all `Physics/Formulations/*/*.cpp` (or `add_subdirectory()` every formulation folder)
- [ ] Ensure `svapplication` links the target that contains the registrars (so the registry is populated).
- [ ] (Optional) Add auto-discovery for formulation-local tests.

### 7) Single extensibility hook for new physics options (avoid `Parameters.h` edits)
- [ ] Add exactly one generic way to pass arbitrary per-equation options through existing XML parsing, e.g.:
  - `<Module_options>{"kappa":1.2,"scheme":"SUPG"}</Module_options>` (string)
  - or `<Module_options_file>foo.json</Module_options_file>` (path)
  - or `<Option key="kappa" value="1.2"/>` repeated entries (key/value list)
- [ ] Store this data in `EquationParameters` without throwing on unknown physics-specific keys/tags.
  - Note: today `ParameterLists::set_parameter_value()` throws on unknown tags, so without a generic hook, every new
    physics option forces edits outside `Physics/`.

---

## Per-New-Physics Checklist (only touches `Physics/Formulations/<NewPhysics>/`)

- [ ] Create `Code/Source/solver/Physics/Formulations/<NewPhysics>/`.
- [ ] Add docs:
  - [ ] `FORMULATION.md` (unknowns, strong/weak form, signs, BCs, parameters/units, linearization)
  - [ ] `README.md` (how the module registers into `FESystem`, required mesh labels/markers, expected spaces)
- [ ] Implement the module:
  - [ ] `<NewPhysics>Module.h/.cpp` deriving from `svmp::Physics::PhysicsModule`
  - [ ] `registerOn(system)` installs all fields/operators/kernels needed before `system.setup()`
- [ ] Define module options and parsing:
  - [ ] `<NewPhysics>Options` struct
  - [ ] parse from `EquationModuleInput.options` / `options_string` / `options_file_path`
  - [ ] validate required keys locally (errors point to this module’s docs)
- [ ] Implement module-local translation helpers (no Application edits):
  - [ ] map generic BC specs → module BC options
  - [ ] map generic material/domain specs → module material/options
- [ ] Register the equation type(s):
  - [ ] Add `<NewPhysics>Register.cpp` that calls `SVMP_REGISTER_EQUATION("<type>", factory_fn)` for each supported
    `EquationParameters::type` string
- [ ] Add tests (local to the formulation folder if supported by the repo’s testing):
  - [ ] unit test for option parsing/validation
  - [ ] smoke test: can register + assemble residual/Jacobian on a tiny mesh
- [ ] Add an example input file (optional but recommended):
  - [ ] `Physics/Formulations/<NewPhysics>/Examples/<type>.xml` (or a minimal snippet in `README.md`)

---

## Guardrails (keep the architecture clean)

- [ ] Never add new `if (eq_type == "...")` logic in `Code/Source/solver/Application/*`.
- [ ] Never add formulation-specific logic to shared Application translators (materials/BCs/etc.).
- [ ] Keep equation-type strings owned by Physics (registry is the single source of truth).
- [ ] If you must extend the generic input contract, do it in a backward-compatible, physics-neutral way.
