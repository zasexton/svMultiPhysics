# Boundary Conditions (BC) Refactor Checklist

Goal: allow Physics formulations to be written **mostly as FE/Forms**, while FE/Systems lowers boundary-condition declarations to the correct **kernels** (Neumann/Robin via `.ds(...)`) and **constraints** (strong Dirichlet), with minimal DOF boilerplate in formulations.

## Phase 1 — FE/Forms extensions

- [x] Add Forms terminals for time (`t`) and timestep (`Δt`) usable in expressions.
- [x] Add time-aware coefficient support (e.g. `g(x,y,z,t)`) in `FormExpr::coefficient(...)`.
- [x] Add a small “point evaluator” for scalar `FormExpr` (value-only) to evaluate BC value expressions at `(x,y,z,t)` when lowering strong Dirichlet.
- [x] Add a Forms-level strong Dirichlet declaration type (prints like math) that can be carried alongside weak forms (not assembled directly).

## Phase 2 — FE/Systems lowering + constraints wiring

- [x] Introduce a Systems-side “system-aware constraint” interface so constraints can query mesh access + DOF handlers without capturing raw `FESystem*` pointers in Physics.
- [x] Implement boundary DOF + (optional) boundary DOF coordinate extraction using `assembly::IMeshAccess` + `spaces::FaceRestriction` + `geometry::MappingFactory`.
- [x] Implement a Systems installer that lowers Forms strong-Dirichlet declarations to constraints during `FESystem::setup()`.
- [x] Add an overload that installs residual forms **and** boundary-condition declarations together (so Physics modules call one installer).
- [x] Add `FESystem::updateConstraints(time)` to update time-dependent strong Dirichlet inhomogeneities without a full `setup()` rebuild (serial-first).

## Phase 3 — Tests

- [x] Add FE/Forms unit tests for `t`, `Δt`, time-aware coefficients, and scalar point evaluation.
- [x] Add FE/Systems unit tests for strong Dirichlet-by-marker installed before `setup()` (no “double setup” required).
- [x] Add FE/Systems unit tests for time-dependent strong Dirichlet updated via `FESystem::updateConstraints(time)`.

## Phase 4 — Physics cleanup

- [x] Refactor `Physics/Formulations/Poisson/PoissonModule.*` to:
  - keep Neumann/Robin as explicit `.ds(marker)` boundary integrals in the weak residual, and
  - express strong Dirichlet via the new Forms BC declaration (no custom DOF extraction in Physics).
