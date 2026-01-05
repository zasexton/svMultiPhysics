# TimeStepping Notes

This folder implements several time-integration schemes on top of the FE library’s
Systems layer (`Systems::TimeIntegrator`, `Systems::TransientSystem`) and an adaptive
step-size/order interface (`TimeStepping::StepController`).

## Implemented

### Multi-stage scheme framework

- `TimeStepping/MultiStageScheme.h` and `TimeStepping/MultiStageScheme.cpp`
  - `MultiStageSolver`: solves implicit “stages” using `NewtonSolver` and Systems transient assembly.
  - `WeightedIntegrator`: wraps a Systems `TimeIntegrator` and scales `dt(·)` terms vs non-`dt` terms.
  - `ResidualAdditionSpec`: supports stage formulas that require adding a residual evaluated at a
    different state (e.g., θ-method).
- `TimeStepping/TimeLoop.cpp` uses this framework for:
  - `θ-method` (including Crank–Nicolson as `θ=0.5`)
  - TRBDF2 stage-1 (trapezoidal substep)

### Generalized-α (first-order / JWH-style)

- `TimeStepping/GeneralizedAlpha.h` and `TimeStepping/GeneralizedAlpha.cpp`
  - Implements the first-order generalized-α family parameterized by `ρ_∞`.
  - Uses solution history to approximate rates (no separate stored `u̇` vector).
- `TimeStepping/TimeLoop.cpp`:
  - Solves for the stage value at `t_{n+α_f}` and maps it back to `u_{n+1}`.

### Generalized-α (second-order / structural dynamics)

- `TimeStepping/GeneralizedAlpha.h` and `TimeStepping/GeneralizedAlpha.cpp`
  - `GeneralizedAlphaSecondOrderIntegrator`: provides `dt(u)` and `dt(u,2)` stencils for a stage solve at
    `t_{n+α_f}` using the Chung–Hulbert parameterization (`ρ_∞ → α_m, α_f, β, γ`).
- `TimeStepping/TimeLoop.cpp`
  - For systems with `temporalOrder()==2`, `SchemeKind::GeneralizedAlpha` solves for `u_{n+α_f}` and maps
    back to `u_{n+1}`, updating `TimeHistory::{uDot,uDDot}` on acceptance.

### Newmark-β (structural dynamics)

- `TimeStepping/NewmarkBeta.h` and `TimeStepping/NewmarkBeta.cpp`
  - `NewmarkBetaIntegrator`: provides `dt(u)` and `dt(u,2)` stencils for a displacement-only Newmark-β solve
    at `t_{n+1}`.
- `TimeStepping/TimeLoop.cpp`
  - For systems with `temporalOrder()==2`, `SchemeKind::Newmark` runs true Newmark-β and updates
    `TimeHistory::{uDot,uDDot}` on acceptance.
  - For systems with `temporalOrder()<=1`, `SchemeKind::Newmark` remains an alias of Crank–Nicolson (`θ=0.5`)
    for compatibility with first-order mixed formulations.

### VSVO-BDF (variable-step/variable-order BDF)

- `Math/FiniteDifference.h`
  - Fornberg finite-difference weights and Lagrange interpolation weights used by variable-step stencils.
- `Systems/TimeIntegrator.{h,cpp}`
  - `systems::BDFIntegrator` (orders 1..5): computes `dt(u)` and `dt(u,2)` stencils from actual step history.
- `TimeStepping/VSVO_BDF_Controller.{h,cpp}`
  - Adaptive controller proposing `(dt, order)` updates and supporting error-based rejection of
    *converged* nonlinear solves.
  - Uses PI control for accepted-step adaptation.
  - Includes a “starter ramp” that increases order as history becomes available up to `max_order`.
- `TimeStepping/TimeLoop.cpp`
  - Computes a predictor via Lagrange extrapolation and uses the predictor–corrector difference as a
    weighted RMS error measure.
  - Bootstrap for the first step uses an embedded pair:
    - `temporalOrder()<=1`: Backward Euler vs Crank–Nicolson
    - `temporalOrder()==2`: Backward Euler vs Newmark-β (O(dt²) reference)

### cG/dG-in-time

- `SchemeKind::DG0` is an alias of Backward Euler.
- `SchemeKind::CG1` is an alias of Crank–Nicolson (`θ=0.5`).
- `SchemeKind::DG` implements `dG(k)` via its collocation-equivalent Radau IIA Runge–Kutta method:
  - degree `k` ⇒ `k+1` stages, order `2k+1` (stiffly accurate)
- `SchemeKind::CG` implements `cG(k)` via its collocation-equivalent Gauss Runge–Kutta method:
  - degree `k` ⇒ `k` stages, order `2k` (not stiffly accurate)
- `SchemeKind::DG1` and `SchemeKind::CG2` are fixed-degree convenience aliases (`dG(1)`, `cG(2)`).
- Collocation supports both `temporalOrder()==1` (`dt(u)`) and `temporalOrder()==2` (`dt(u,2)`) with a
  monolithic stage solve in `TimeStepping/TimeLoop.cpp`.

## Known limitations / expectations

- **Structural dynamics (`temporalOrder()==2`) expects consistent `(u̇,ü)` initial conditions.**
  - `TimeHistory` stores `uDot` and `uDDot` for Newmark-β, structural generalized-α, and second-order
    collocation runs.
  - If `(u̇,ü)` storage is missing, `TimeLoop` allocates it and initializes it from displacement history
    (finite differences on `dtHistory()`), enforcing homogeneous constraints on `(u̇,ü)`.
  - If `ü` is required and cannot be inferred from history, `TimeLoop` falls back to a residual-based
    acceleration initialization (`M ü = -other`) and throws if the acceleration is still undefined.
- **High-stage collocation can be expensive, but a cheaper solve strategy is available.**
  - The default monolithic solve couples all stages, producing a block matrix with dense stage coupling;
    cost grows quickly with degree/stage count.
  - `TimeLoopOptions::collocation_solve = StageGaussSeidel` performs a nonlinear block Gauss–Seidel sweep
    over stages (supported for `temporalOrder()==1` and `==2`) and avoids assembling/solving the block
    Jacobian. Convergence is controlled by `collocation_max_outer_iterations` and
    `collocation_outer_tolerance`.
  - Degree is clamped to `<= 10` by `TimeLoopOptions::{dg_degree,cg_degree}`.
- **State providers follow end-of-step commits.**
  - `TimeLoop` calls `FESystem::beginTimeStep()` before each transient assemble to reset work buffers.
  - For methods whose nonlinear solve occurs at an intermediate time (e.g., generalized-α stage solve,
    Gauss collocation), `TimeLoop` performs an end-of-step assembly at `t_{n+1}` before calling
    `commitTimeStep()` so material/global-kernel state reflects the accepted end state.
- **VSVO-BDF uses an LTE-based estimate and cost-model order selection.**
  - The step error estimate is based on the BDF local truncation error (divided differences / Nordsieck
    equivalent) rather than a raw predictor–corrector difference; the first step still uses an embedded
    reference solve to bootstrap the estimate.
  - `VSVO_BDF_Controller` consumes the current-order estimate plus optional adjacent-order estimates to
    choose `(dt,order)` via a simple efficiency model, with PI-based dt control when history is available.
- **Restart requires consistent histories (now validated for VSVO-BDF).**
  - Variable-step schemes assume the provided displacement history and `dtHistory()` match the last
    accepted step(s).
  - `TimeHistory::setDtHistory()` enables injecting a restart `dtHistory()`. For `SchemeKind::VSVO_BDF`
    restarts (`history.stepIndex()>0`), `TimeLoop` validates that the required `dtHistory()` prefix is
    present rather than silently fabricating missing entries.

## Pointers

- Schemes and orchestration: `TimeStepping/TimeLoop.{h,cpp}`
- Multi-stage utilities: `TimeStepping/MultiStageScheme.{h,cpp}`
- Generalized-α: `TimeStepping/GeneralizedAlpha.{h,cpp}`
- VSVO controller: `TimeStepping/VSVO_BDF_Controller.{h,cpp}`
- Variable-order BDF integrator: `Systems/TimeIntegrator.{h,cpp}` (`systems::BDFIntegrator`)
- History storage: `TimeStepping/TimeHistory.{h,cpp}`
- Unit tests: `Tests/Unit/TimeStepping/` (convergence, stability, history/utils, controllers, multistage, integrators)
