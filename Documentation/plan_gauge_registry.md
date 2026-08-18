# Gauge Registry: Automatic Nullspace Detection and Enforcement

## Overview

Introduces a systems-side `GaugeRegistry` that automatically detects nullspace
modes from Forms expressions and BC declarations, then auto-creates the
appropriate `GlobalConstraint` to enforce gauge conditions (e.g., mean-zero
pressure in incompressible flow with pure Neumann BCs).

The core abstraction is unchanged: `A u = f`, known/suspected nullspace basis
`Z`, side conditions `C u = d`, and enforcement policies (pinning, elimination,
multiplier augmentation, solver-side nullspace handling).

---

## Architecture

```
 Forms residual ─► NullspaceAnalyzer ─► GaugeCandidate[]
                                              │
 BoundaryCondition::gaugeAnchoring() ─────────┤
                                              │
 AssemblyKernel::gaugeMetadata()     ─────────┤  (Path B: non-Forms)
 GlobalKernel::gaugeMetadata()       ─────────┘
                                              │
                                     GaugeRegistry
                                      │ resolve()
                                      ▼
                              ResolvedMode[]
                                      │
                              auto-create GlobalConstraint
                              or pass nullspace to LinearSolver
```

### Dual-Path Detection

- **Path A — automatic inference**: `NullspaceAnalyzer` walks the `FormExpr`
  DAG for each field and determines which canonical transformations leave the
  weak form invariant.
- **Path B — explicit declarations**: Hand-written `AssemblyKernel`,
  `GlobalKernel`, and custom BC classes override optional `gaugeMetadata()`
  hooks to declare semantics when symbolic inference is unavailable.

Both paths produce the same `GaugeCandidate` objects.

---

## Implementation Checklist

### Phase 1: Scalar Constant Modes (gradient-annihilated fields)

Covers: pure-Neumann Poisson, incompressible pressure gauge, any scalar field
that appears only through `grad(field)` in the weak form.

- [x] **1.1 GaugeRegistry data structures** (`Constraints/GaugeRegistry.h`)
  - [x] `NullspaceModeFamily` enum (ScalarConstant, ComponentwiseConstant, RigidBody)
  - [x] `Confidence` enum (High, Medium, Low)
  - [x] `AnchoringVerdict` enum (Anchored, PartiallyAnchored, Preserved, Unknown)
  - [x] `GaugeStatus` enum (Anchored, ExactNullspace, NearNullspace, Unknown)
  - [x] `EnforcementPolicy` enum (None, PinDof, MeanZeroElimination, LagrangeMultiplier, SolverNullspace)
  - [x] `GaugeCandidate` struct (field, component, family, confidence, reason, source)
  - [x] `AnchoringEvidence` struct (verdict, source description)
  - [x] `ResolvedMode` struct (candidate + status + policy + anchoring evidence)
  - [x] `GaugeRegistry` class (addCandidate, addAnchoring, resolve, resolvedModes, diagnosticReport)

- [x] **1.2 GaugeRegistry implementation** (`Constraints/GaugeRegistry.cpp`)
  - [x] `resolve()` logic: merge candidates, apply anchoring rules, choose enforcement policy
  - [x] Auto-create `GlobalConstraint` for exact nullspace modes
  - [x] Warn + conservative fallback for medium-confidence modes
  - [x] `diagnosticReport()` for human-readable output

- [x] **1.3 NullspaceAnalyzer** (`Forms/NullspaceAnalyzer.h`, `Forms/NullspaceAnalyzer.cpp`)
  - [x] `analyze(residual, fields)` → `vector<GaugeCandidate>`
  - [x] Recursive DAG walk classifying how each field appears
  - [x] Detect scalar fields appearing only through `Gradient` → ScalarConstant nullspace
  - [x] Detect fields with absolute-value terms (mass, Robin) → anchored
  - [x] Detect stabilization patterns (CellDiameter-scaled penalties) → near-nullspace flag
  - [x] Stubs/documentation for future families (sym_grad, rigid body, per-component)

- [x] **1.4 BoundaryCondition anchoring hook** (`Forms/BoundaryCondition.h`)
  - [x] Add virtual `gaugeAnchoring(FieldId, NullspaceModeFamily)` with default `Unknown`

- [x] **1.5 Built-in BC anchoring overrides** — deferred to Phase 2
  (Phase 1 relies on StrongDirichlet detection in SystemSetup instead)

- [x] **1.6 FESystem integration** (`Systems/FESystem.h`)
  - [x] Add `std::unique_ptr<GaugeRegistry> gauge_registry_` member
  - [x] Public accessor: `gaugeRegistry()` (creates on first access)
  - [x] `hasGaugeRegistry()` query

- [x] **1.7 FormsInstaller integration** (`Systems/FormsInstaller.cpp`)
  - [x] Call `NullspaceAnalyzer::analyze()` after residual parsing
  - [x] Store candidates in `system.gaugeRegistry()`

- [x] **1.8 SystemSetup integration** (`Systems/SystemSetup.cpp`)
  - [x] After constraints are applied but before `close()`:
    detect StrongDirichlet constraints as anchoring evidence
  - [x] Call `gaugeRegistry().resolve()`
  - [x] Apply resolved constraints (auto-create `GlobalConstraint` entries)

- [x] **1.9 CMakeLists.txt updates**
  - [x] Add `GaugeRegistry.cpp` to `FE_CONSTRAINTS_SOURCES`
  - [x] Add `NullspaceAnalyzer.cpp` to `FE_FORMS_SOURCES`

- [x] **1.10 Unit tests**
  - [x] `test_GaugeRegistry.cpp`: registry CRUD, resolve logic, enforcement selection (15 tests)
  - [x] `test_NullspaceAnalyzer.cpp`: scalar Poisson, Robin anchoring, vector fields (15 tests)
  - [x] `test_GaugeIntegration.cpp`: end-to-end with FESystem setup + BC verdicts (9 tests)

### Phase 2: Extended Mode Families

- [x] **2.1 Componentwise vector constant modes**
  - [x] Extend `NullspaceAnalyzer` to detect vector fields appearing only through `grad`
  - [x] `field_value_dimension` tracking from `SpaceSignature` to distinguish scalar vs vector
  - [x] Vector fields with grad-only → `ComponentwiseConstant` family
  - [x] GaugeRegistry resolver uses `MeanZeroElimination` for `ComponentwiseConstant`

- [x] **2.2 Rigid-body modes for H1 vector fields**
  - [x] Detect `sym(grad(u))` patterns → `RigidBody` family
  - [x] Track `only_through_sym_grad` and `has_plain_grad` flags per field
  - [x] Mixed sym(grad)+grad falls back to `ComponentwiseConstant`
  - [ ] Compute RB mode vectors from mesh vertex coordinates (deferred — requires mesh access at resolution time)

- [ ] **2.3 Per-connected-component scope**
  - [ ] `ConnectedComponents` utility (`Mesh/ConnectedComponents.h/.cpp`)
  - [ ] BFS/DFS on cell adjacency from `IMeshAccess::cell2vertex`
  - [ ] Replicate candidates per connected component with scoped DOF lists

- [x] **2.4 Built-in BC anchoring overrides**
  - [x] `EssentialBC`: override → `Anchored` for all families
  - [x] `NaturalBC`: override → `Preserved` for all families
  - [x] `RobinBC`: override → `Anchored` (scalar/componentwise), `PartiallyAnchored` (rigid body)
  - [x] `ReservedBC`: override → `Preserved` for all families
  - [x] `CoupledNaturalBC`: inherits `Preserved` from `NaturalBC`
  - [x] `CoupledRobinBC`: inherits anchoring semantics from `RobinBC`
  - [x] `PeriodicBC`: override → `Anchored` (scalar/componentwise), `PartiallyAnchored` (rigid body)
  - [x] `MultiPointConstraintBC`: inherits default `Unknown` (too general to classify)

### Phase 3: Non-Forms Kernel Hooks (Path B)

- [x] **3.1 AssemblyKernel::gaugeMetadata()** — optional virtual, default empty
  - [x] Added to `Assembly/AssemblyKernel.h` with `GaugeRegistry.h` include
  - [x] Returns `vector<GaugeCandidate>`, default empty
- [x] **3.2 GlobalKernel::gaugeMetadata()** — optional virtual, default empty
  - [x] Added to `Systems/GlobalKernel.h` with `GaugeRegistry.h` include
  - [x] Returns `vector<GaugeCandidate>`, default empty
- [x] **3.3 Collect kernel metadata during setup**
  - [x] In `SystemSetup.cpp`, iterate all operator definitions (cell, boundary,
        interior face, interface face, global kernels)
  - [x] Call `gaugeMetadata()` on each, add candidates to `gaugeRegistry()`
  - [x] Runs before gauge resolution, after user constraints
  - [x] 5 new tests: `CellKernel_GaugeMetadata_CollectedDuringSetup`,
        `GlobalKernel_GaugeMetadata_CollectedDuringSetup`,
        `CellKernel_GaugeMetadata_AnchoredByDirichlet`,
        `DefaultKernel_EmptyGaugeMetadata`,
        `DefaultGlobalKernel_EmptyGaugeMetadata`

### Phase 4: Solver-Side Nullspace Support

- [x] **4.1 LinearSolver::supportsNullspace()** — default false
  - [x] Added to `Backends/Interfaces/LinearSolver.h`
- [x] **4.2 LinearSolver::setNullspaceBasis()** — default no-op
  - [x] Accepts `std::span<const std::vector<double>>` (dense basis vectors)
  - [x] Default no-op in base class
- [x] **4.3 GaugeRegistry::buildNullspaceBasis()**
  - [x] Constructs dense orthonormalized basis vectors for SolverNullspace modes
  - [x] ScalarConstant: 1 vector with 1/sqrt(n) at field DOFs
  - [x] ComponentwiseConstant: n_comp vectors (one per component block)
  - [x] RigidBody: translation modes only (rotation requires mesh coords, deferred)
- [x] **4.4 PETSc `MatSetNullSpace` integration**
  - [x] `PetscLinearSolver::supportsNullspace()` returns true
  - [x] `PetscLinearSolver::setNullspaceBasis()` creates `MatNullSpace` from basis vectors
  - [x] Nullspace attached to matrix via `MatSetNullSpace()` before each `KSPSolve()`
  - [x] Proper cleanup in destructor and move operators
- [x] **4.5 GaugeRegistry resolver integration**
  - [x] `resolve()` accepts `solver_supports_nullspace` flag
  - [x] When true, prefers `EnforcementPolicy::SolverNullspace` over algebraic enforcement
  - [x] `applyEnforcement()` skips constraint creation for SolverNullspace modes
- [x] **4.6 NewtonSolver bridging**
  - [x] Before linear solve: checks `linear.supportsNullspace()`
  - [x] Builds nullspace basis from `gaugeRegistryIfPresent()->buildNullspaceBasis()`
  - [x] Calls `linear.setNullspaceBasis(basis)` to pass to solver
- [x] **4.7 SetupOptions integration**
  - [x] `SetupOptions::prefer_solver_nullspace` flag (default false)
  - [x] Passed to `resolve()` during `FESystem::setup()`
- [ ] **4.8 FSILS nullspace handling** — deferred (FSILS has no native nullspace API)
- [x] **4.9 Unit tests** (8 tests)
  - [x] `LinearSolver_DefaultSupportsNullspace_IsFalse`
  - [x] `Resolve_SolverNullspace_WhenSolverSupports`
  - [x] `Resolve_FallbackToAlgebraic_WhenSolverDoesNotSupport`
  - [x] `BuildNullspaceBasis_ScalarConstant`
  - [x] `BuildNullspaceBasis_ComponentwiseConstant_3D`
  - [x] `BuildNullspaceBasis_Empty_WhenNoSolverNullspacePolicy`
  - [x] `ApplyEnforcement_SolverNullspace_SkipsAlgebraicConstraint`
  - [x] `PreferSolverNullspace_SetupOption`

### Phase 5: Diagnostics and Validation

- [x] **5.1 Numerical validation pass** (`Constraints/GaugeDiagnostics.h/.cpp`)
  - [x] `validateNullspaceBasis()`: computes `y = A*z` via `matrix.mult()`,
        then `||A*z|| / (||A||_est * ||z||)` for each basis vector
  - [x] `||A||` estimated via random SpMV power iteration (configurable iterations)
  - [x] Configurable tolerance (default 1e-8) via `ValidationOptions`
  - [x] Gate behind `SVMP_GAUGE_VALIDATE` env var via `isNullspaceValidationEnabled()`
  - [x] Called from `NewtonSolver` after first Jacobian assembly (`it == 0`)
  - [x] `formatValidationReport()` produces human-readable `[PASS]/[FAIL]` report
  - [x] Backend-agnostic: uses only `GenericMatrix::mult()` and `GenericVector::norm()`
  - [x] 7 unit tests: Laplacian nullspace passes, SPD non-nullspace fails,
        empty basis, report formatting, env var check, strict tolerance, multiple vectors

- [x] **5.2 Diagnostic logging**
  - [x] `GaugeRegistry::diagnosticReport()` — implemented in Phase 1
  - [x] `SetupOptions::gauge_diagnostics` flag for explicit opt-in at setup time
  - [x] Also triggered by `SVMP_GAUGE_VALIDATE` env var
  - [x] Logged to stderr after gauge resolution in `SystemSetup.cpp`

---

## NullspaceAnalyzer — Operator Classification Table

The analyzer walks from each field leaf (DiscreteField/StateField/TrialFunction)
toward the root, tracking which differential operators are applied before the
field's value reaches a test-function bilinear pairing.

| Operator path                       | Effect on constant mode | Effect on RB mode       |
|-------------------------------------|-------------------------|-------------------------|
| `Gradient`                          | Annihilates             | Annihilates (constant)  |
| `SymmetricPart(Gradient(...))`      | Annihilates             | Annihilates all 6 modes |
| `Divergence`                        | Annihilates             | Annihilates (constant)  |
| `Curl`                              | Annihilates             | Partially annihilates   |
| `Hessian`                           | Annihilates             | Annihilates             |
| `TimeDerivative`                    | Annihilates             | Annihilates             |
| `InnerProduct(field, test)` (no op) | Preserves → anchors     | Preserves → anchors     |
| `field * test` (no differential op) | Preserves → anchors     | Preserves → anchors     |
| Stabilization (h-scaled penalty)    | Near-nullspace          | Near-nullspace          |

### Future analysis families (Phase 2+)

1. **Componentwise vector constant**: Each component of a vector field analyzed
   independently. If component `i` appears only through `Gradient[i]`, it has a
   constant-mode nullspace in that component.

2. **Rigid-body modes**: When a vector field appears only through
   `sym(grad(u))`, the nullspace includes 3 translations + 3 rotations (in 3D).
   Mode vectors computed from mesh vertex coordinates.

3. **Disconnected components**: For meshes with multiple disconnected regions,
   each candidate is replicated per connected component with DOFs scoped to
   that component's vertices.

4. **Stabilization detection**: Patterns like `h * (∇p·∇q)` (PSPG) or
   `h * div(u) * div(v)` (LSIC) weakly break the nullspace. The analyzer
   should flag these as `NearNullspace` with medium confidence, meaning the
   resolver will warn but not enforce.

---

## Enforcement Policy Selection

```
resolve(solver_ptr):
  for each candidate:
    evidence = all anchoring evidence for (field, component)

    if ANY evidence is Anchored:
      status = Anchored,  policy = None

    elif ALL evidence is Preserved or empty:
      if confidence == High:
        status = ExactNullspace
      else:
        status = NearNullspace  (warn)

    elif ANY evidence is PartiallyAnchored:
      status = NearNullspace  (warn)

    // Choose enforcement for exact nullspace
    if status == ExactNullspace:
      if solver_ptr && solver_ptr->supportsNullspace():
        policy = SolverNullspace
      elif family == ScalarConstant:
        policy = MeanZeroElimination  // → GlobalConstraint::zeroMean()
      else:
        policy = PinDof               // deterministic fallback
```

---

## File Inventory

| Action   | File                                      | What                                         |
|----------|-------------------------------------------|----------------------------------------------|
| **New**  | `FE/Constraints/GaugeRegistry.h`          | Data structures + GaugeRegistry class        |
| **New**  | `FE/Constraints/GaugeRegistry.cpp`        | Resolver + diagnostic report                 |
| **New**  | `FE/Forms/NullspaceAnalyzer.h`            | FormExpr DAG nullspace analyzer              |
| **New**  | `FE/Forms/NullspaceAnalyzer.cpp`          | DAG walk implementation                      |
| **Edit** | `FE/Systems/FESystem.h`                   | Add GaugeRegistry member + accessor          |
| **Edit** | `FE/Systems/FormsInstaller.cpp`           | Call NullspaceAnalyzer after form parse       |
| **Edit** | `FE/Systems/SystemSetup.cpp`              | Collect anchoring, call resolve, apply        |
| **Edit** | `FE/Forms/BoundaryCondition.h`            | Add virtual gaugeAnchoring()                 |
| **Edit** | `FE/CMakeLists.txt`                       | Add new source files + test files            |
| **Edit** | `FE/Forms/StandardBCs.h`                  | gaugeAnchoring() overrides (Phase 2)         |
| **Edit** | `FE/Forms/ConstraintBCs.h`                | gaugeAnchoring() override for PeriodicBC     |
| **Edit** | `FE/Assembly/AssemblyKernel.h`            | Add gaugeMetadata() hook (Phase 3)           |
| **Edit** | `FE/Systems/GlobalKernel.h`               | Add gaugeMetadata() hook (Phase 3)           |
| **New**  | `Tests/Unit/Constraints/test_GaugeRegistry.cpp` | 15 tests for registry CRUD + resolve   |
| **New**  | `Tests/Unit/Forms/test_NullspaceAnalyzer.cpp`   | 15 tests for DAG analysis              |
| **New**  | `Tests/Unit/Systems/test_GaugeIntegration.cpp`  | 22 tests end-to-end (Phases 1-4)       |
| **New**  | `FE/Constraints/GaugeDiagnostics.h`       | Numerical validation interface (Phase 5)     |
| **New**  | `FE/Constraints/GaugeDiagnostics.cpp`     | SpMV validation + report formatting          |
| **New**  | `Tests/Unit/Backends/test_GaugeDiagnostics.cpp` | 7 tests for numerical validation       |
| **Edit** | `FE/Backends/Interfaces/LinearSolver.h`   | Add supportsNullspace() + setNullspaceBasis() |
| **Edit** | `FE/Backends/PETSc/PetscLinearSolver.h`   | Override supportsNullspace/setNullspaceBasis  |
| **Edit** | `FE/Backends/PETSc/PetscLinearSolver.cpp` | MatSetNullSpace implementation               |
| **Edit** | `FE/TimeStepping/NewtonSolver.cpp`        | Bridge nullspace + validation after assembly |
| Future   | `FE/Mesh/ConnectedComponents.h/.cpp`      | Cell adjacency BFS (Phase 2.3)               |
