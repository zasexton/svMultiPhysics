# RIS / Coupled-BC (Physics-Agnostic) Support Checklist

This checklist tracks FE library + Forms infrastructure needed for legacy-parity features used by RIS and 0D-coupled (resistance-style) boundary contributions, while keeping `Code/Source/solver/FE` physics-agnostic.

## A) Forms: Trigonometric Ops (URIS delta term — Option A)

- [x] Add `sin`/`cos` unary expression kinds to the Forms AST (`Code/Source/solver/FE/Forms/FormExpr.h`, `Code/Source/solver/FE/Forms/FormExpr.cpp`).
- [x] Extend forward-mode AD scalar (`Dual`) with `sin`/`cos` (`Code/Source/solver/FE/Forms/Dual.h`).
- [x] Implement kernel evaluation + spatial-jet derivatives for `sin`/`cos` (value/grad/H) (`Code/Source/solver/FE/Forms/FormKernels.cpp`).
- [x] Unit tests:
  - [x] `Dual` trig derivative sanity (`Code/Source/solver/FE/Tests/Unit/Forms/test_Dual.cpp`).
  - [x] Composite spatial derivatives through `cos(...)` match finite differences (`Code/Source/solver/FE/Tests/Unit/Forms/test_FormSpatialDerivatives.cpp`).

## B) URIS delta mollifier expressibility/validation (physics-agnostic)

Legacy URIS distance uses **interpolated absolute nodal SDF**: `dist(x) = Σ_a N_a(x) * |sdf_node(a)|` (note: abs before interpolation).

- [x] Add a Forms unit test that evaluates the cosine mollifier
  `DDir(dist) = (dist <= eps) ? (1 + cos(pi*dist/eps)) / (2*eps*eps) : 0`
  and matches a reference implementation using the same `dist(x)` semantics (`Code/Source/solver/FE/Tests/Unit/Forms/test_FormSpatialDerivatives.cpp` or a new Forms unit test).

## C) Backend: Generic low-rank boundary coupling hook (FSILS)

Expose FSILS’ existing low-rank face coupling (`add_bc_mul`: `Y += res * v * (vᵀ X)`) via a backend-facing, physics-agnostic API.

- [x] Add an FE-level data model/API to configure “coupled faces” (nodes + `val` moments + BC type) on `FsilsMatrix` without introducing any “resistance BC” operator in Forms (`Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.h`, `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp`).
- [x] Store per-face `incL` + `res` arrays on `FsilsMatrix` and pass them through to `fsils_solve()` (`Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp`).
- [x] Ensure `FsilsMatrix::mult()` includes the configured low-rank coupling term in its mat-vec (so `A->mult()` reflects the true operator when coupling is active) (`Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp`).
- [x] Unit test: configure a single coupled face and verify mat-vec/solve behavior changes by the expected low-rank update (`Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp`).

## D) Assembly: Generic DOF remap/alias hook (RIS-style remap)

Legacy `doassem_ris()` adds additional element contributions into a *mapped* DOF row/col set (dynamic “row/col remap + duplicate assembly”).

- [x] Implement an `assembly::GlobalSystemView` decorator that applies a user-provided DOF map:
  - [x] For each local row DOF with a mapping, add an additional assembled row into the mapped row, mapping columns when possible (legacy `doassem_ris()` semantics).
  - [x] Apply the same duplication/remap rule for vector insertion.
  (`Code/Source/solver/FE/Assembly/RemappedSystemView.h`)
- [x] Unit test: dense-system assembly with a synthetic mapping reproduces expected duplicated/remapped matrix/vector entries (`Code/Source/solver/FE/Tests/Unit/Assembly/test_GlobalSystemView.cpp`).

## E) Validation

- [x] `ctest -R FE_Forms_Tests --output-on-failure` passes.
- [x] `ctest -R FE_Backends_Tests --output-on-failure` (or the repo’s backend test target) passes.
- [x] `ctest -R FE_Assembly_Tests --output-on-failure` passes.
