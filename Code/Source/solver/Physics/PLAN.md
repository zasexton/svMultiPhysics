# Physics Library — Design Plan

## Overview

`Physics/` is the **PDE + constitutive model** layer of svMultiPhysics. It is where we implement simulation-ready physics in a way that is:

- **Readable:** each physics module documents the strong form, weak form, unknowns, and conventions next to the code.
- **FE-native:** formulations are expressed primarily using the `FE/Forms` vocabulary and installed into `FE/Systems` (with `AssemblyKernel` as the “escape hatch”).
- **Reusable:** physics modules are backend-agnostic and rely on `FE/Systems` + `FE/Assembly` to assemble into whatever linear algebra backend is selected.

The dependency direction is intentionally one-way:

```
Physics  ──►  FE  ──►  Mesh
  │            │
  └────►  (optional: solver/app drivers outside this folder)
```

---

## Scope Boundaries

### Physics **DOES**

- Implement **equation sets** (fluid, solid, transport, etc.) as weak forms and/or `assembly::AssemblyKernel`s.
- Provide **simulation-ready constitutive models** (domain-specific material laws) that plug into `FE/Forms` / kernels.
- Provide **boundary/load terms** (Neumann/Robin/Nitsche/DG fluxes) as boundary-face forms/kernels.
- Provide reusable **coupling terms** for multiphysics (e.g., FSI interface coupling) expressed as block forms/kernels.
- Provide **validation tests** (patch tests, manufactured solutions, Jacobian checks).

### Physics **DOES NOT**

- Own DOF distribution, constraints, sparsity construction, or assembly loops (belongs to `FE/*`).
- Own linear algebra storage or solver algorithms (belongs to `FE/Backends` and solver/application layers).
- Own mesh generation/refinement/partitioning (belongs to `Mesh/*` and external tooling).

---

## “Documentation First” Contract (what must exist for each physics)

Every physics submodule must be understandable without reading unrelated code. Concretely, each physics should ship with:

1. `FORMULATION.md`
   - Unknowns, parameters, and units conventions.
   - Strong form + weak form (with sign conventions explicitly stated).
   - Domain/boundary notation: what `dx`, `ds(id)`, `dS` represent and how boundary IDs are assigned (via Mesh labels).
   - Linearization strategy (analytic tangent vs AD via `FE/Forms`).
   - Stabilization/discretization choices (SUPG/PSPG, DG fluxes, etc.) when applicable.

2. `README.md`
   - Minimal “how to use” snippet showing how the module registers fields/operators into `FE::systems::FESystem`.
   - Required mesh labels/markers and expected spaces (e.g., `P2/P1` Taylor–Hood).

3. Implementation that mirrors the math
   - Prefer `FE/Forms` expressions for the weak form and use `FE::systems::install*` helpers to register kernels.
   - Use hand-written `assembly::AssemblyKernel` only when required (performance, complex geometry data, custom quadrature, etc.).

---

## Proposed Directory Structure

```
Physics/
  CMakeLists.txt
  PLAN.md
  Physics.h                  # umbrella include (optional)

  Core/                      # common infrastructure (no equation-specific code)
    PhysicsModule.h          # interface: “register into FESystem”
    Domain.h                 # boundary IDs, subdomain filters, marker utilities
    ParameterSchema.h        # declare/validate parameter sets per physics

  Materials/                 # simulation-ready constitutive models
    Fluid/                   # viscosity models, turbulence closures (future)
    Solid/                   # hyperelasticity, viscoelasticity, growth/remodeling (future)
    Common/                  # shared helpers (invariants, tensor utilities, etc.)

  Formulations/              # equation sets (weak forms + kernels)
    Poisson/
    Heat/
    LinearElasticity/
    IncompressibleStokes/
    IncompressibleNavierStokes/

  Coupling/                  # multiphysics couplings (block terms)
    FSI/
    MultiDomain/             # (optional) domain-decomposition / mortar / constraints

  BCs/                       # reusable BC term builders (weak) + constraint helpers (strong)
  QoI/                       # quantities of interest (functionals / post-processing)
  Tests/                     # unit + integration tests (gtest)
  Examples/                  # small executable examples (optional)
  cmake/                     # standalone build helpers + package config
```

Notes:
- `FE/Constitutive` remains **domain-agnostic infrastructure**; `Physics/Materials` is where *specific* models live.
- Keep submodules cohesive: a physics folder should not “reach across” into another physics’ internals; share via `Physics/Core` or `Physics/Materials/Common`.

---

## Core API Contract (PhysicsModule)

Physics modules must be installable uniformly by an application/driver layer. The minimal contract is a pure-virtual interface located at `Physics/Core/PhysicsModule.h`:

```cpp
namespace svmp::Physics {

class PhysicsModule {
public:
    virtual ~PhysicsModule() = default;

    // The core contract: Physics tells the System what fields, operators,
    // and kernels it needs.
    virtual void registerOn(svmp::FE::systems::FESystem& system) const = 0;

    // Optional: initial condition helpers (application owns the actual state vectors).
    // Default implementation is a no-op.
    virtual void applyInitialConditions(const svmp::FE::systems::FESystem& system,
                                        std::span<svmp::FE::Real> u0) const {}

    // Optional: quantities of interest (QoIs) / post-processing registration.
    // Default implementation is a no-op.
    virtual void registerFunctionals(svmp::FE::systems::FESystem& system) const {}
};

} // namespace svmp::Physics
```

Conventions:
- `registerOn(...)` is called during the **definition phase**, before `system.setup()`.
- Physics modules should register the standard operator tags used by `FESystem` entry points:
  - `"residual"` (assembled by `FESystem::assembleResidual`)
  - `"jacobian"` (assembled by `FESystem::assembleJacobian`)
  - `"mass"` (assembled by `FESystem::assembleMass`, if transient)

---

## Integration Pattern with FE (recommended default)

### 1) Forms-first (preferred)

- Express residual/Jacobian terms with `FE/Forms` (including AD-backed Jacobians where helpful).
- Install into a `FE::systems::FESystem` using `FE::systems::installResidualForm`, `installResidualBlocks`, or `installCoupledResidual`.

Conceptually:

```cpp
// inside a Physics module
using namespace svmp::FE;
using namespace svmp::FE::forms;

auto u = FormExpr::trialFunction(V, "u");
auto v = FormExpr::testFunction(V, "v");
auto R = inner(grad(u), grad(v)).dx();
systems::installResidualForm(system, op, u_id, u_id, R);
```

### 2) Custom kernels (escape hatch)

- Implement `FE::assembly::AssemblyKernel` directly when the term:
  - requires specialized data layouts, custom quadrature, or non-standard stabilization,
  - is performance critical and benefits from hand-optimized loops,
  - needs features not yet expressible in the forms vocabulary.

Kernels must correctly declare `getRequiredData()` and rely on `AssemblyContext` for all per-entity data.

---

## Transient Problems & Time Integration

Time discretization is **solver-owned**. Physics should avoid hard-coding a particular scheme (Backward Euler, Crank–Nicolson, etc.).

Recommended starting point (time-integrator neutral):
- Register **spatial** contributions in `"residual"` / `"jacobian"` (e.g., diffusion, convection, pressure coupling).
- Register a separate `"mass"` operator for the time-dependent fields.
- The solver/time-integrator combines operators into an effective transient residual, e.g.:
  - Backward Euler: `R(u) = (1/dt) M (u - u_prev) + K(u)`
  - BDF2: `R(u) = M (a0*u + a1*u_prev + a2*u_prev2) + K(u)`

If/when it is more convenient, physics may alternatively express time derivatives symbolically using `FE/Forms` `dt(·)` nodes and rely on `FE::systems::TimeIntegrator` (via `SystemStateView::time_integration`) to lower them, but the ownership remains the same: the solver supplies the time-integration context.

---

## Boundary Conditions (strong vs weak)

- **Strong/algebraic BCs (Dirichlet/periodic/MPC):** defined and managed via `FE/Constraints/*` and installed onto the `FESystem` by the solver/application layer or physics module convenience helpers.
- **Weak BCs (Neumann/Robin/Nitsche/traction/flux):** implemented as boundary-face terms via `ds(boundary_id)` forms or boundary kernels.

Physics code should treat boundary IDs as data (Mesh labels), not hard-coded integers.

---

## Initial Conditions (ICs)

Initial conditions are applied by the **application/driver** because they act on the solver-owned state vectors (`u`, `u_prev`, …). Physics can still standardize ICs via optional helpers:

- `PhysicsModule::applyInitialConditions(system, u0)` fills the **initial global vector** in the system’s DOF ordering (using `system.fieldMap()` / `system.dofHandler()` as needed).
- For multi-physics, the driver remains responsible for allocating/owning `u0` and deciding which fields are initialized (and how) at `t=0`.

---

## Outputs & Quantities of Interest (QoI)

Physics should provide standard derived quantities (when applicable) as scalar functionals registered on the system:

- Use `FE::assembly::FunctionalKernel` + `FESystem::addFunctionalKernel(tag, kernel)`.
- Evaluate via `FESystem::evaluateFunctional(...)` / `FESystem::evaluateBoundaryFunctional(...)`.

This supports common QoIs such as fluxes, energies, norms, drag/lift, and boundary integrals without coupling to file I/O.

---

## Build & Packaging Plan (shared library like Mesh/FE)

`Physics/` should build as a standalone library and integrate into the parent project, mirroring `Mesh/` and `FE/`:

- CMake target: `svphysics` (static or shared) with alias `svphysics::svphysics`.
- Dependencies: link against `svfe::svfe` (and `svmesh` where needed).
- Standalone build mode when `Physics/CMakeLists.txt` is the top-level project.
- Export/install:
  - headers installed under `include/svphysics/` (with `Physics/...` include layout),
  - `svphysicsConfig.cmake` + `svphysicsTargets.cmake` for downstream `find_package(svphysics)`.

Suggested options (patterned after Mesh/FE):
- `PHYSICS_BUILD_SHARED` (default: OFF)
- `PHYSICS_BUILD_TESTS` (default: standalone=ON, integrated=OFF)
- `PHYSICS_WITH_MESH` (default: ON; allow OFF for purely FE-based unit tests)
- `PHYSICS_ENABLE_MPI` (default: inherit from FE/Mesh capabilities)

Milestone 0 deliverable: add the missing `Physics/CMakeLists.txt` that defines `svphysics`, wires include paths consistent with `Mesh/` and `FE/`, and links the appropriate targets (`svfe::svfe`, plus `svmesh` when used).

---

## Roadmap (implementation milestones)

### Milestone 0 — Scaffolding
- Add the missing `Physics/CMakeLists.txt` (and `Physics/cmake/*` as needed) patterned after Mesh/FE:
  - `add_library(svphysics ...)` + `add_library(svphysics::svphysics ALIAS svphysics)`
  - `target_link_libraries(svphysics PUBLIC svfe::svfe)` and optionally `svmesh`
  - install/export package config for downstream `find_package(svphysics)`
- Add `Physics/Core` skeleton (`PhysicsModule` interface, domain/marker utilities).
- Add at least one “hello world” physics that assembles a scalar Poisson operator via `FE/Forms`.

### Milestone 1 — Materials library
- Implement a small set of concrete constitutive models using `FE/Forms`’ constitutive call boundary:
  - Fluid: Newtonian viscosity (and a generalized-Newtonian example).
  - Solid: linear elastic and Neo-Hookean stress (as a first hyperelastic model).
- Unit tests: Real vs AD consistency + finite-difference Jacobian checks on single elements.

### Milestone 2 — Core PDE formulations (reference implementations)
- Poisson/Heat as reference scalar problems (CG).
- Linear elasticity (small-strain) with standard BCs and body forces.
- Stokes (mixed) as the foundation for Navier–Stokes.

### Milestone 3 — Incompressible Navier–Stokes (SVMP baseline)
- Incremental addition of:
  - transient term via a registered `"mass"` operator (time discretization solver-owned),
  - convection, diffusion, pressure-velocity coupling,
  - stabilization (PSPG/SUPG) as optional terms,
  - weak traction/outflow variants.
- Validation against manufactured solutions and regression tests.

### Milestone 4 — Multiphysics coupling
- FSI coupling terms and interface conditions (kinematic + dynamic).
- Block-structured operators suited for preconditioning (e.g., Schur complement-friendly assembly).

---

## Testing & Verification Strategy

- **Unit tests:** constitutive models and small kernels (single element, known tensors).
- **Assembly integration tests:** assemble residual/Jacobian blocks and compare against finite differences.
- **Method of manufactured solutions (MMS):** Poisson/heat/linear elasticity/Stokes baselines.
- **Patch tests:** solid mechanics (rigid body motion, constant strain) and fluid (constant velocity/pressure).

The guiding rule is: every new physics module ships with at least one test that validates both residual and Jacobian behavior.
