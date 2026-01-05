# FE/TimeStepping Subfolder — Design Plan

## Overview

`FE/TimeStepping` owns the solver-facing control flow for transient problems:
- advancing a discrete state in time (history management, step acceptance, adaptivity hooks),
- orchestrating implicit solves (Newton + linear solves),
- selecting and configuring time advancement schemes.

It sits above:
- `FE/Systems` (fields, operators, constraints, assembly orchestration),
- `FE/Backends` (matrix/vector storage and linear solvers),
and relies on:
- `FE/Forms` (weak-form vocabulary, including symbolic `dt(·,k)`),
- `FE/Assembly` (mesh loops and kernel execution).

### Existing transient plumbing (already in this repo)

The codebase already has a cross-module design boundary for `dt(·,k)`:

- `FE/Forms` keeps `dt(·,k)` symbolic; it is not assemblable without a transient context.
- `FE/Assembly` defines `assembly::TimeIntegrationContext` stencils (`Assembly/TimeIntegrationContext.h`).
- `FE/Systems` defines:
  - `systems::TimeIntegrator` to build a `TimeIntegrationContext` (`Systems/TimeIntegrator.h`),
  - `systems::TransientSystem` to attach that context and assemble transient operators (`Systems/TransientSystem.h`),
  - `systems::SystemStateView` to pass `time`, `dt`, and `{u,u_prev,u_prev2}` spans (`Systems/SystemState.h`).

Design implication:
- TimeStepping should **use** `systems::TransientSystem` (not reimplement dt lowering),
- TimeStepping must manage the solution history vectors expected by `SystemStateView`.

---

## Scope of Responsibilities

1. **Time loop driver**
   - Own `(t, dt, step_index)` and step acceptance/rejection.
   - Manage history vectors (`u`, `u_prev`, `u_prev2`) and ghost updates.
   - Provide callbacks for:
     - time-dependent BC/parameter updates,
     - output/checkpoint hooks (no file formats here),
     - adaptive dt policies.

2. **Implicit solve orchestration**
   - Provide a reusable Newton driver for systems assembled through `FE/Systems`.
   - Assemble residual/Jacobian via `systems::TransientSystem::assemble(...)`.
   - Solve linearized systems via `backends::LinearSolver`.

3. **Scheme selection/config**
   - Choose a `systems::TimeIntegrator` (dt stencils) and scheme-level parameters.
   - Advertise requirements (history depth, supported derivative orders, explicit vs implicit).

### Explicit Non-Goals

- No weak-form vocabulary or dt syntax (belongs to `FE/Forms`).
- No element integration loops (belongs to `FE/Assembly`).
- No DOF/constraint construction or operator registration (belongs to `FE/Systems`).
- No backend implementations or solver packages (belongs to `FE/Backends`).
- No IO formats / restart files (application responsibility; TimeStepping only offers hooks).

---

## Time Advancement Schemes

`TimeStepping/SCHEMES.md` is the catalog of candidate methods. This PLAN defines the target *implementation set* and the required plumbing for each.

### Tier 1 (works with current dt-stencil infrastructure)

These methods evaluate all terms at the “current” assembled state and are directly compatible with `dt(·,k)` lowering:

- **Backward Euler / BDF1** (already supported via `systems::BackwardDifferenceIntegrator` for `dt(u)`).
- **BDF2** (planned; requires `SystemStateView::u_prev2`).
- **Backward difference for `dt(u,2)`** (already in `BackwardDifferenceIntegrator`; reference-quality for dynamics, not the final structural integrator).

This tier supports the most ergonomic workflow:
- write continuous-time forms with `dt(u)` / `dt(u,2)`,
- run them with a chosen `systems::TimeIntegrator`,
- assemble via `systems::TransientSystem`.

### Tier 2 (requires additional plumbing or operator splitting)

- **θ-method / Crank–Nicolson**: needs spatial terms evaluated at a mixture of current and previous states.
  - Near-term path: split operators into dt-containing terms vs dt-free terms using `forms::FormIR::IntegralTerm::time_derivative_order`, assemble separately, and combine in the time stepper.
  - Longer-term path: extend `assembly::TimeIntegrationContext` to support multiple evaluation states (stage states).

- **TR-BDF2 (Trapezoidal + BDF2 composite)**: stiff-friendly 2nd order method that avoids CN “ringing” on stiff transients.
  - Requires the same dt-free term mixing as CN/θ for the trapezoidal substep, plus an intermediate stage state.
  - Fits the existing dt-stencil model for the BDF2 substep by treating `{u^{n+1}, u^{n+γ}, u^n}` as `{u, u_prev, u_prev2}`.

- **Generalized-α (1st/2nd order)**: needs stage evaluation (`α_m`, `α_f`) and optional algorithmic damping (`ρ_∞`).
  - Requires extending `TimeIntegrationContext` beyond simple fixed stencils, or introducing a “stage state view” concept in `AssemblyContext`.

- **Newmark-β (2nd order)**: requires velocity/acceleration state (or an equivalent first-order reformulation).
  - Recommended representation: treat `(u, v)` (and optionally `a`) as mixed fields so Forms can express `dt(u)=v` and `dt(v)=a` cleanly.

- **DIRK / SDIRK / ESDIRK**: higher-order stiff one-step methods (multiple implicit stages).
  - Requires stage state support (multiple assembly contexts per step); can reuse Jacobians/preconditioners across stages.

- **Rosenbrock / ROW (linearly implicit)**: avoids Newton by taking several linear solves per step.
  - Requires Jacobian + residual assembly as a reusable primitive (shared with Newton), plus scheme-specific stage bookkeeping.

- **Explicit RK / SSPRK**: requires efficient application of `M^{-1}` (lumping or mass solves per stage).
- **IMEX / splitting**: requires tagging residual contributions as implicit vs explicit operators in `FE/Systems`.
- **VSVO-BDF (variable-step / variable-order BDF)**: production-grade stiff integrators (BDF1–5) that can be more efficient than RK for large stiff PDE/DAE systems.
  - Requires deeper history than `{u_prev,u_prev2}` (or a Nordsieck representation), an error estimator, and a step/order controller.
- **cG/dG in time (variational time stepping)**: space–time methods with strong adaptivity/error-estimation hooks (useful for multiphysics, goal-oriented error control, and space–time coupling).
  - Requires a dedicated time-discretization layer (beyond fixed stencils) and per-step local solve/assembly control.
- **Structural dynamics variants (HHT-α / Bossak-α / WBZ-α / Bathe)**: commonly used second-order families that tune high-frequency dissipation and dispersion.
  - Natural follow-ons once `Newmark-β` / `generalized-α` are implemented and verified.

---

## Core Components (Flat Structure)

- `TimeLoop.*`: time loop driver + callback interfaces.
- `NewtonSolver.*`: Newton-Raphson driver that:
  - assembles via `systems::TransientSystem`,
  - solves via `backends::LinearSolver`,
  - applies `systems::FESystem::constraints()` distribution as needed.
- `StepController.*`: adaptive dt policies (future).
- `TimeSteppingUtils.h`: small helpers (e.g., parameter transforms like `ρ_∞ → (α_m, α_f)`; logging helpers).

---

## API Sketch (Target User Experience)

**Namespace:** `svmp::FE::timestepping`

The typical transient workflow is:

```cpp
using namespace svmp::FE;

// 1) Define and set up the spatial system (fields/operators/kernels).
systems::FESystem system(mesh_access);
system.setup();

// 2) Choose a dt(·,k) lowering strategy and wrap the system for transient assembly.
auto integrator = std::make_shared<const systems::BackwardDifferenceIntegrator>();
systems::TransientSystem transient(system, integrator);

// 3) Allocate backend storage and linear solver.
auto factory = backends::BackendFactory::create("eigen");
auto A = factory->createMatrix(system.sparsity("jacobian"));
auto r = factory->createVector(system.dofHandler().getNumDofs());
auto du = factory->createVector(system.dofHandler().getNumDofs());
auto linear = factory->createLinearSolver(backends::SolverOptions{});

// 4) Time loop driver manages history and calls transient assembly inside Newton.
timestepping::TimeHistory history{/* u, u_prev, u_prev2 as GenericVector-backed storage */};
timestepping::TimeLoop loop(timestepping::TimeLoopOptions{/* t0, t_end, dt, ... */});
loop.run(transient, *linear, history);
```

Notes:
- `systems::SystemStateView` should be populated from `backends::GenericVector::localSpan()` and passed through `systems::TransientSystem::assemble(...)`.
- `system.beginTimeStep()` / `system.commitTimeStep()` should bracket accepted steps (material-state history).

---

## Integration Details (How Modules Connect)

### Assembly and `dt(·,k)`

- Always assemble transient operators through `systems::TransientSystem::assemble(...)` so `SystemStateView::time_integration` is set.
- Keep `SystemStateView::{u,u_prev,u_prev2}` backed by `backends::GenericVector::localSpan()` (or host vectors in serial tests).
- Call `backends::GenericVector::updateGhosts()` on all history vectors before assembly in MPI backends.

### Constraints and Material State

- For strong constraints, TimeStepping is responsible for:
  - updating time-dependent inhomogeneities (via user callback; stored in `constraints::AffineConstraints`),
  - calling `constraints::AffineConstraints::distribute(...)` on the solution after Newton updates and on accepted steps when needed.
- Bracket accepted steps with `systems::FESystem::beginTimeStep()` / `commitTimeStep()` to maintain material-state history.

### Backends

- Use `backends::BackendFactory` to create:
  - matrices matching `system.sparsity(op_tag)`,
  - vectors of size `system.dofHandler().getNumDofs()`,
  - a `backends::LinearSolver` with user options.

---

## Milestones

### Milestone 1: Minimal Implicit Transient Solve (BDF1)

- [x] Implement `TimeLoop` + history container for `{u,u_prev,u_prev2}`.
- [x] Implement `NewtonSolver` using `systems::TransientSystem` + `backends::LinearSolver`.
- [x] Verify with a transient heat equation (manufactured solution) using `dt(u)` forms.

### Milestone 2: Higher Accuracy without New Assembly/Forms Features (BDF2)

- [x] Add a `systems::TimeIntegrator` implementation for BDF2 stencils.
- [x] Add variable-`dt` safety rules (history reset or variable-step formulas).
- [x] Add temporal convergence tests.

### Milestone 3: Advanced Schemes

- [x] Decide CN/θ implementation route (operator splitting vs richer `TimeIntegrationContext`).
- [x] Add TR-BDF2 as a robust stiff 2nd order default once CN/θ plumbing exists.
- [x] Extend `TimeIntegrationContext` if needed (stage state support).
- [x] Add generalized-α (FSI) and Newmark-β (structures) verification problems.
