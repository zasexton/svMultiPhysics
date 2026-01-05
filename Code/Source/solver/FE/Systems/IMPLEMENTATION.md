# FE/Systems Subfolder — Implementation Plan

This document turns `Code/Source/solver/FE/Systems/PLAN.md` into a concrete, buildable implementation roadmap for the `FE/Systems` subfolder.

The goal is to deliver a solver-facing, physics-agnostic “system container” that *wires together* existing FE infrastructure (Dofs/Constraints/Sparsity/Assembly/Spaces) without re-implementing it.

When the Mesh library is available (`SVMP_FE_WITH_MESH`), the **preferred Systems integration path** uses the unified runtime mesh type `svmp::Mesh` and its **zero-copy view/export accessors** (CSR connectivity arrays, boundary labels, named sets) to drive DOF distribution and boundary DOF extraction.

---

## Guiding Constraints

- **Physics-agnostic:** no PDE-specific operators; Systems only manages fields, operator tags, and term registries.
- **Leverage existing FE modules:** prefer composing `FE/*` APIs over creating new parallel abstractions.
- **Incremental delivery:** start with single-field square operators, then expand to multi-field, DG faces, MPI, and operator backends.
- **Be honest about prerequisites:** some capabilities require upstream extensions (notably: per-cell solution injection into `AssemblyContext`, and true rectangular multi-field DOF maps).

---

## Target Deliverables (Milestones)

### Milestone 1 — Systems Core (registries + types, no assembly)

**Outcome:** `systems::FESystem` can be constructed, fields/operators/terms can be registered, and internal registries can be queried.

**Checklist**
- [x] Add `Systems/FESystem.h` with the public API skeleton from `PLAN.md`
- [x] Add `Systems/OperatorRegistry.h` with term storage types and lookup
- [x] Add `Systems/FieldRegistry.h` mapping `FieldId` ↔ field metadata
- [x] Add `Systems/SystemState.h` (`SystemStateView` + parameter access helpers)
- [x] Add `Systems/SystemsExceptions.h` (thin wrappers around `Core/FEException.h` patterns)
- [x] Add minimal unit tests for registry behavior in `FE/Tests/Unit/Systems/` (if this test tree exists/allowed)

---

### Milestone 2 — Setup Pipeline (DOFs + constraints + sparsity + storage)

**Outcome:** `FESystem::setup()` produces:

- finalized DOFs (`dofs::DofHandler`)
- closed constraints (`constraints::AffineConstraints`)
- (optional) MPI-consistent constraints (`constraints::ParallelConstraints`)
- sparsity patterns (`sparsity::*`)
- allocated operator storage (initially dense/test storage, later backend-specific)
- a configured assembler instance (`assembly::StandardAssembler` initially)

**Checklist**
- [x] Add `Systems/SystemSetup.h/.cpp` implementing `FESystem::setup()`
- [x] Define `systems::SetupInputs` (or equivalent) that supplies Mesh-driven topology/boundary data (details below)
- [x] Integrate `dofs::DofHandler` DOF distribution from `svmp::Mesh` (preferred), using mesh view/export helpers (serial + MPI-safe plumbing)
- [x] Build `constraints::AffineConstraints` from registered `constraints::Constraint` objects
- [x] If MPI: run `constraints::ParallelConstraints` consistency/import before `AffineConstraints::close()`
- [x] Build operator sparsity patterns using `sparsity::SparsityFactory` or `sparsity::SparsityBuilder`
- [x] If constraints are active: augment sparsity using `sparsity::ConstraintSparsityAugmenter` (pattern must allow constraint fill)
- [x] Validate with dense/test operator storage via `assembly::DenseMatrixView`/`assembly::DenseVectorView` in unit tests (storage is caller-owned in Phase 1)
- [x] Configure `assembly::StandardAssembler`:
  - [x] `Assembler::setDofHandler(dof_handler)`
  - [x] `Assembler::setConstraints(&constraints)` (if `use_constraints_in_assembly`)
  - [x] `Assembler::setSparsityPattern(&pattern)` (per operator when assembling matrices)
  - [x] `Assembler::setOptions(assembly_options)`

---

### Milestone 3 — Assembly Dispatch (cells + boundary + DG interior faces)

**Outcome:** `FESystem::assemble(req, state, ...)` executes a full assembly pass for a tagged operator by dispatching terms to `FE/Assembly`.

**Checklist**
- [x] Add `Systems/SystemAssembly.h/.cpp`
- [x] Implement `FESystem::assemble(...)`:
  - [x] Validate setup state and operator existence
  - [x] Zero outputs if requested (`GlobalSystemView::zero()` or equivalent)
  - [x] Prefer one-pass `Assembler::assembleBoth` when both matrix+vector are requested
  - [x] Loop over cell terms, boundary terms, interior-face terms in a deterministic order
  - [x] Call `Assembler::finalize(matrix_out, vector_out)` once per request
- [x] Support boundary-marker dispatch via `Assembler::assembleBoundaryFaces(mesh, marker, ...)`
- [x] Support DG interior-face dispatch via `Assembler::assembleInteriorFaces(...)`
- [x] Add unit tests assembling with `assembly::DenseMatrixView/DenseVectorView` and simple kernels (e.g., `assembly::MassKernel`)

**Known prerequisite (outside Systems):**
- `assembly::AssemblyContext::solutionValue()` currently requires `setSolutionCoefficients()`, but `StandardAssembler` does not populate it. To support nonlinear terms generically:
  - [x] Extend `FE/Assembly` so assemblers can accept a read-only global solution vector and set per-cell coefficients in `AssemblyContext` before `kernel.computeCell(...)`.
  - (Concrete option) Add optional `std::span<const Real> solution` to `Assembler::assemble*` APIs, and implement in `StandardAssembler` by gathering cell DOFs → local coefficient vector → `context_.setSolutionCoefficients(local)`.

---

### Milestone 4 — “Make it easy” Conveniences (constraints/BC wiring)

**Outcome:** Users can specify common constraints without hand-building DOF lists.

**Checklist**
- [x] Add `Systems/SystemConstraints.h/.cpp` convenience entry points that leverage:
  - `constraints::ConstraintTools` for building `constraints::AffineConstraints`
  - `dofs::DofTools` / `dofs::EntityDofMap` for boundary/subdomain DOF extraction
- [x] Decide and implement a concrete boundary-selection strategy:
  - [x] **Option B (chosen):** derive boundary DOFs from `svmp::Mesh` boundary labels and/or explicit named face sets (zero-copy view/export), consistent with `DofHandler::EntityDofMap`
  - [ ] **Option D (chosen fallback):** support geometry-based DOF selection via predicates when explicit tags/sets are unavailable
- [x] Add “constraint by marker” helpers (constant values first; function-based when DOF coordinates are available)

---

### Milestone 5 — Operator Backends (matrix-free + functionals)

**Outcome:** Systems can expose additional FE infrastructure without changing the solver-facing system definition.

**Checklist**
- [x] Add `Systems/OperatorBackends.h/.cpp`
- [x] Matrix-free support:
  - [x] Allow registering a matrix-free operator backend per operator tag
  - [x] Wire `assembly::MatrixFreeAssembler` setup/apply for iterative solvers
- [x] Functional/QoI support:
  - [x] Allow registering `assembly::FunctionalKernel` objects under tags (e.g., `"qoi:<name>"`)
  - [x] Wire `assembly::FunctionalAssembler` execution using the same mesh/space/dofs

---

### Milestone 6 — Full Multi-field + Rectangular Assembly

This is the point where Systems becomes “multiphysics-general” rather than “single-space general”.

**Checklist**
- [x] Define a `MultiFieldDofLayout` strategy:
  - [x] monolithic numbering (single global DOF space with sub-blocks)
  - [x] per-field DOF maps with a block-assembly insertion plan
- [x] Extend `FE/Assembly` to support rectangular assembly with truly different row/col DOF maps
  - (Today `StandardAssembler` notes this is required; it currently assumes a single DOF map.)
- [ ] Extend `SystemStateView` / `AssemblyContext` to support vector/mixed field values robustly (not only scalar `solutionValue()`)
  - [x] Evaluate solution in the trial space for rectangular terms
  - [ ] Provide vector-valued field evaluation at quadrature points
  - [ ] Provide multi-field state access in `AssemblyContext` (cross-field coupling kernels)
- [x] Integrate `dofs::FieldDofMap` / `dofs::BlockDofMap` as the authoritative block metadata exposed to solvers

**Notes for Multi-field Implementation**

1.  **Block-Aware Assembler API**
    - The underlying `Assembler` API (and `AssemblyContext`) must be updated to accept `(row_space, col_space)` pairs instead of a single `space`.
    - `assembleMatrix` should evolve to `assembleMatrix(mesh, test_space, trial_space, kernel, ...)`.
    - This is critical for problems where fields use different spaces (e.g., Quadratic vs Linear), requiring rectangular element matrices.

2.  **Interface/Boundary Rectangular Coupling**
    - Volume coupling is insufficient for surface-coupled multiphysics. The assembly layer must support `assembleBoundaryFaces` with distinct test/trial spaces to handle interface coupling conditions.
    - **Optimization:** Implement a "Coupling Domain" iterator (or intersection iterator) to avoid iterating over the entire boundary when fields only overlap on a specific interface.

3.  **Global Block Insertion & Offsets**
    - When assembling a cross-coupling term (e.g., $K_{uv}$), the system must calculate global row/column offsets correctly.
    - `dofs::BlockDofMap` should provide the authoritative lookup: `global_row_index = block_row_offset + local_row_index`.
    - Backend views (`GlobalSystemView`) should ideally accept block-relative indices + block IDs to avoid manual offset arithmetic in the assembly loop.

4.  **Preconditioner Support**
    - Monolithic solvers often require block-specific preconditioners (e.g., Schur complement on a specific physics block).
    - The `FESystem` should expose individual diagonal blocks (e.g., $K_{uu}$) as standalone operator views to the linear solver, even if they are stored in a monolithic matrix.

5.  **Rectangular Sparsity Generation**
    - `SparsityBuilder` and `SparsityPattern` must be extended to support distinct row and column DOF maps.
    - The connectivity logic must intersect the support of the test space (rows) with the trial space (cols) to determine the non-zero pattern for off-diagonal coupling blocks.

6.  **Cross-Field State Access**
    - Kernels often depend on fields that are not part of the current assembly term (e.g., a Momentum kernel depending on Temperature).
    - The `AssemblyContext` population logic in `StandardAssembler` must be able to gather and map solution coefficients for *all* registered fields on the current element, ensuring kernels have access to the full multiphysics state at quadrature points.

7.  **Rectangular Constraint Application**
    - Standard `AffineConstraints` distributors often assume square local matrices.
    - For rectangular coupling blocks ($K_{uv}$), a specialized distributor is needed. It must handle zeroing rows (if the test field DOF is constrained) and distributing columns (if the trial field DOF is constrained) correctly, respecting the distinct row/column index spaces.

8.  **DOF Renumbering / Block Permutations**
    - Assembly is most efficient with node-major ordering (interleaved fields) for cache locality.
    - Many advanced block preconditioners (e.g., PETSc FieldSplit) require field-major ordering (blocked fields).
    - Systems must provide a renumbering or permutation layer to map between "Assembly Indexing" and "Solver Indexing" seamlessly.

---

## Concrete File/Type Outline (what to create)

All files live in `Code/Source/solver/FE/Systems/` (flat layout).

### `FESystem.h/.cpp`

**Public class:** `svmp::FE::systems::FESystem`

Core functions (minimum set):

- Construction / definition:
  - `explicit FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh);`
  - (Preferred when Mesh is available) `explicit FESystem(std::shared_ptr<const svmp::Mesh> mesh);` which internally:
    - stores the `svmp::Mesh` pointer for topology/labels/sets,
    - constructs an `assembly::MeshAccess` adapter for assembly iteration.
  - `FieldId addField(FieldSpec spec);`
  - `void addOperator(OperatorTag name);`
  - `void addConstraint(std::unique_ptr<constraints::Constraint> c);`
  - `void addCellKernel(OperatorTag op, FieldId test, FieldId trial, std::shared_ptr<assembly::AssemblyKernel> k);`
  - `void addBoundaryKernel(OperatorTag op, BoundaryId marker, FieldId test, FieldId trial, std::shared_ptr<assembly::AssemblyKernel> k);`
  - `void addInteriorFaceKernel(OperatorTag op, FieldId test, FieldId trial, std::shared_ptr<assembly::AssemblyKernel> k);`

- Setup:
  - `void setup(const SetupOptions& opts = {}, const SetupInputs& inputs = {});`

- Assembly:
  - `assembly::AssemblyResult assemble(const AssemblyRequest&, const SystemStateView&, GlobalSystemView* A, GlobalSystemView* b);`
  - `assembleResidual`, `assembleJacobian`, `assembleMass` convenience wrappers

- Accessors:
  - `const dofs::DofHandler& dofHandler() const;`
  - `const dofs::FieldDofMap& fieldMap() const;`
  - `const dofs::BlockDofMap* blockMap() const;`
  - `const constraints::AffineConstraints& constraints() const;`
  - `const sparsity::SparsityPattern& sparsity(const OperatorTag& op) const;`

**Dependencies leveraged**

- `FE/Core`: `Types.h`, `FEException.h`
- `FE/Spaces`: `FunctionSpace`
- `FE/Assembly`: `IMeshAccess`, `Assembler`, `AssemblyKernel`, `GlobalSystemView`
- `FE/Dofs`: `DofHandler`, `FieldDofMap`, `BlockDofMap`
- `FE/Constraints`: `Constraint`, `AffineConstraints`, `ParallelConstraints`
- `FE/Sparsity`: `SparsityFactory` (or `SparsityBuilder`)

---

### `OperatorRegistry.h/.cpp`

**Types**

- `struct CellTerm { FieldId test, trial; std::shared_ptr<AssemblyKernel> kernel; };`
- `struct BoundaryTerm { int marker; FieldId test, trial; std::shared_ptr<AssemblyKernel> kernel; };`
- `struct InteriorFaceTerm { FieldId test, trial; std::shared_ptr<AssemblyKernel> kernel; };`
- `struct OperatorDefinition { std::vector<CellTerm> cells; std::vector<BoundaryTerm> boundary; std::vector<InteriorFaceTerm> interior; /* metadata */ };`

**Functions**

- `void addOperator(std::string tag);`
- `OperatorDefinition& get(const std::string& tag);`
- `const OperatorDefinition& get(const std::string& tag) const;`
- validation helpers:
  - `validateOperatorTerms(tag, field_registry)`
  - deterministic ordering utilities

**Dependencies leveraged**

- `FE/Assembly::AssemblyKernel` for the physics extension point.
- `FE/Core::Types` for `FieldId`.

---

### `FieldRegistry.h/.cpp`

**Types**

- `struct FieldRecord { FieldId id; std::string name; std::shared_ptr<const spaces::FunctionSpace> space; int components; };`

**Functions**

- `FieldId add(FieldSpec spec);`
- `const FieldRecord& get(FieldId) const;`
- `FieldId findByName(std::string_view) const;`
- helpers to build:
  - `dofs::FieldDofMap` (block metadata only at first)
  - optional `dofs::BlockDofMap`

**Dependencies leveraged**

- `FE/Spaces::FunctionSpace`
- `FE/Dofs::FieldDofMap`, `FE/Dofs::BlockDofMap`

---

### `SystemState.h`

**Types**

- `SystemStateView` (from `PLAN.md`)
- Optional helper `ParameterStore` implementing `getRealParam`

**Dependencies leveraged**

- `FE/Core::Types` (`Real`)

---

### `SystemSetup.h/.cpp`

**Types**

- `struct SetupInputs` (fills gaps not covered by `IMeshAccess`):
  - `std::shared_ptr<const svmp::Mesh> mesh;` (preferred; drives DOFs + boundary sets via Mesh view/export helpers)
  - `svmp::Configuration coord_cfg{svmp::Configuration::Reference};` (which coordinates to use when evaluating boundary functions)
  - Optional escape hatches (tests / non-Mesh builds):
    - `std::optional<dofs::MeshTopologyInfo> topology_override;`
    - `std::function<std::optional<std::vector<GlobalIndex>>(int marker, FieldId field)> boundary_dofs_override;`

**Functions**

- `void buildDofs(...)` → uses `dofs::DofHandler`
- `void buildConstraints(...)` → applies `constraints::Constraint` objects into `constraints::AffineConstraints`
- `void makeConstraintsParallelConsistent(...)` → uses `constraints::ParallelConstraints`
- `void buildSparsity(...)` → uses `sparsity::SparsityFactory` (and `ConstraintSparsityAugmenter` when needed)
- `void allocateStorage(...)` → uses `assembly::createDense*View` initially, later `FE/Backends`
- `void configureAssembler(...)` → uses `assembly::StandardAssembler`

---

### `SystemAssembly.h/.cpp`

**Functions**

- `AssemblyResult assembleOperator(...)` (internal core)
- `assembleCellTerms(...)`
- `assembleBoundaryTerms(...)`
- `assembleInteriorFaceTerms(...)`

**Dependencies leveraged**

- `FE/Assembly::{Assembler, StandardAssembler, GlobalSystemView}`
- `FE/Spaces::FunctionSpace` for test/trial lookups
- `FE/Dofs::DofHandler` (for DOF map)

---

### Build system integration

When code files are added:

- Update `Code/Source/solver/FE/CMakeLists.txt` to include the new `Systems/*.h` and `Systems/*.cpp` files.
- Ensure include paths match existing patterns (most FE code includes headers as `"Dofs/..."`, `"Assembly/..."`, etc.).

**Checklist**
- [x] Add Systems headers/sources to `Code/Source/solver/FE/CMakeLists.txt`
- [x] Add a small unit-test CMake entry if Systems tests are introduced

---

## How Systems Leverages Other FE Subfolders (Concrete Wiring)

### DOFs (`FE/Dofs`)

- `dofs::DofHandler` owns distribution/finalization and provides:
  - `getDofMap()` for assembly
  - `getPartition()` for MPI constraint exchange
  - `getEntityDofMap()` when available for boundary DOF extraction
- `dofs::FieldDofMap` / `dofs::BlockDofMap` provide solver-facing block metadata.

### Constraints (`FE/Constraints`)

- Store constraints as `constraints::AffineConstraints` in Systems.
- Accept individual constraints as `constraints::Constraint` objects and apply them during setup.
- For MPI:
  - use `constraints::ParallelConstraints` to make constraint sets consistent and import ghost constraint lines before closing.
- For convenience:
  - use `constraints::ConstraintTools` to generate constraints from boundary DOF sets.

### Sparsity (`FE/Sparsity`)

- Use `sparsity::SparsityFactory` (or `SparsityBuilder`) to build per-operator patterns from DOF maps.
- If constraints are applied during assembly, ensure the matrix pattern supports constraint fill via `ConstraintSparsityAugmenter`.

### Assembly (`FE/Assembly`)

- Kernels are `assembly::AssemblyKernel` (cell/boundary/interior-face).
- Dispatch uses:
  - `Assembler::assembleMatrix/assembleVector/assembleBoth` for cell terms
  - `Assembler::assembleBoundaryFaces` for marker-based boundary terms
  - `Assembler::assembleInteriorFaces` for DG interface terms
- Output uses `assembly::GlobalSystemView` (dense/test views first; backend views later).

### Spaces (`FE/Spaces`)

- Each field references a `spaces::FunctionSpace` used to:
  - select `elements::Element` prototypes (via `FunctionSpace::getElement`)
  - communicate DOFs-per-element and basis type to assembly contexts

---

## Notes on a “Math Notation” Front-End (Future)

If a future `FE/Forms` or `FE/DSL` layer is added, it should compile down to Systems-level constructs:

- fields (`FieldSpec`)
- operators (`OperatorTag`)
- terms (`CellTerm`, `BoundaryTerm`, `InteriorFaceTerm`) with `(test_field, trial_field, kernel)`
- constraints (strong constraints and/or weak enforcement terms)
- parameter names required at assembly time

Systems remains the stable compilation target that knows how to:

- distribute DOFs
- close constraints (including MPI consistency)
- build sparsity patterns
- allocate storage / wire backends
- assemble via `FE/Assembly`

---

## Open Design Decisions (track explicitly)

**Checklist**
- [x] Decide how Systems receives mesh topology for DOF distribution (chosen: Mesh adapter via `svmp::Mesh` + view/export)
- [x] Decide how Systems receives boundary DOF sets (chosen: Mesh-derived via entity maps + explicit boundary sets, with geometric fallback)
- [x] Decide how solution/state is passed into assembly contexts (extend `FE/Assembly` APIs vs alternative kernel interface)
- [x] Decide the first supported “multi-field” definition:
  - [x] single field (scalar)
  - [ ] single field (multi-component)
  - [ ] mixed space as single `FunctionSpace`
  - [ ] true multiple independent fields/spaces (requires rectangular DOF maps)

### Options & tradeoffs

#### 1) How Systems receives mesh topology for DOF distribution

**Option A — Caller provides `dofs::MeshTopologyInfo` / `MeshTopologyView` to `setup()`**

Pros:
- Most **mesh-library-independent**: matches `dofs::DofHandler`’s primary API.
- Keeps Systems simple and avoids “hidden” expensive conversions.
- Parallel-ready if the caller can supply `cell_owner_ranks`, `vertex_gids`, neighbors, etc.

Cons:
- Pushes responsibility to applications to build/maintain topology arrays (and keep them consistent).
- Requires additional plumbing for boundary labeling/regions (if Systems needs them too).

**Option B — Systems builds `MeshTopologyInfo` via a new mesh adapter interface**

Example: `systems::IMeshTopologyAccess` that can produce the arrays required by `DofHandler`.

Pros:
- Better user experience: one mesh object can feed both assembly iteration and DOF distribution.
- Systems can enforce consistent conventions (IDs, canonical ordering expectations).

Cons:
- New interface + maintenance burden; easy to end up with “half a mesh library”.
- `assembly::IMeshAccess` does not currently expose enough for robust topology (`vertex_gids`, edge/face tables, ownership).
- Risk of accidentally coupling Systems to a specific mesh backend (or duplicating mesh infrastructure).

**Option C — Caller provides a finalized `dofs::DofHandler` (Systems does not distribute DOFs)**

Pros:
- Fastest path for applications that already own DOF distribution.
- Avoids additional mesh topology needs in Systems entirely.

Cons:
- Weakens the Systems lifecycle (“define → setup → assemble” is no longer fully owned).
- Harder to evolve into true multi-field DOF management because Systems isn’t the authority.

**Decision (chosen for Systems)**
- Use **Option B**: require/provide `svmp::Mesh` during `setup()` and derive DOF topology from the Mesh library (zero-copy view/export), using `dofs::DofHandler::distributeDofs(const Mesh&, ...)` where available.
- Keep **Option A/C** as optional overrides for tests and mesh-independent build modes (`topology_override`, pre-finalized `DofHandler`).

---

#### 2) How Systems receives boundary DOF sets (for constraints / convenience APIs)

**Option A — Application provides boundary DOF sets (and optional DOF coordinates)**

Pros:
- Robust to mesh indexing mismatches (Systems doesn’t assume how faces/regions map to DOFs).
- Works with arbitrary boundary definitions (named sets, imported groups, geometric selections).
- Minimal Systems dependencies; easiest to get correct in parallel.

Cons:
- More work for applications; risk of inconsistent boundary extraction logic across projects.

**Option B — Systems derives boundary DOFs using `dofs::EntityDofMap` + `dofs::DofTools`**

Requires boundary facet labels and facet connectivity arrays that match the entity numbering used by `DofHandler`.

Pros:
- Centralizes a “standard” boundary→DOF extraction workflow.
- Reuses existing FE utilities (`dofs::DofTools`, `constraints::ConstraintTools`).

Cons:
- Needs additional topology inputs (facet labels + facet→vertex connectivity, possibly edge→vertex).
- Requires consistent entity numbering between the mesh boundary representation and `EntityDofMap`.
- In practice this tends to be mesh-backend-specific unless a unified mesh topology convention exists.

**Option C — Extend/augment mesh access to provide boundary topology directly**

Example: add a separate Systems-side boundary topology interface rather than expanding `assembly::IMeshAccess`.

Pros:
- Can be clean if the project already has a unified mesh boundary representation.

Cons:
- Still needs stable IDs and explicit ownership rules for parallel; easy to drift into “mesh re-implementation”.

**Option D — Geometry-based selection via predicates (when coordinates exist)**

Pros:
- Boundary selection independent of mesh boundary tagging quality.
- Useful for quick prototyping and for meshes without reliable labels.

Cons:
- Requires DOF coordinates; sensitive to tolerances; can be ambiguous for curved/complex boundaries.

**Decision (chosen for Systems)**
- Use **Option B**: derive boundary DOFs from the Mesh library’s boundary representation and explicit named sets:
  - prefer Mesh **explicit face sets** (e.g., `mesh.base().get_set(svmp::EntityKind::Face, name)`) to avoid copies and to support “boundary by name” workflows,
  - fall back to boundary labels (`face_boundary_ids()`) when sets are not provided.
- Implementation note: avoid allocation-heavy queries like `faces_with_label()` inside hot paths; if “faces-by-label” views are needed, add a Mesh-side cached/view API that returns `std::span<const index_t>` over a stable face-id list.
- Also implement **Option D** as a fallback: geometry-based selection via predicates when explicit tags/sets are unavailable.
- Keep `boundary_dofs_override` as an escape hatch for tests and unusual meshes.

**Mesh interface expansion notes**
- Option B/B/D is simplest when Mesh provides stable, zero-copy “views” for: entity connectivity (cell/face→vertex CSR), boundary markers, and named entity sets.
- If the Mesh API does not already offer an allocation-free marker→faces lookup, add a cached view helper (returning `std::span<const index_t>`) to avoid per-assembly allocations.
- For function-based boundary conditions and geometry predicates, Systems benefits from Mesh exposing coordinates by configuration (reference/current) via a view/export accessor with well-defined lifetime.

---

#### 3) How solution/state is passed into assembly contexts (nonlinear/time-dependent kernels)

**Option A — Extend `FE/Assembly` assembler APIs to accept a global solution vector**

Concrete approach:
- Add `Assembler::setSolution(std::span<const Real>)` (or an optional `solution` parameter on assemble calls).
- In `StandardAssembler`, when `kernel.getRequiredData()` requests solution data, gather cell coefficients using `DofMap::getCellDofs()` and call `AssemblyContext::setSolutionCoefficients(local)`.

Pros:
- Preserves the existing physics extension point (`assembly::AssemblyKernel`).
- Aligns with `RequiredData::{SolutionValues,SolutionGradients,...}` already in `FE/Assembly`.
- Centralizes the “how to build solution-at-quadrature” logic in Assembly (where contexts live).

Cons:
- Requires changing `FE/Assembly` public APIs and updating assembler implementations.
- Parallel requires a ghosted solution (or explicit ghost synchronization) before assembly.
- Current `AssemblyContext` solution access is scalar-centric; vector/mixed will require further extension.

**Option B — Introduce a new kernel interface that receives `SystemStateView` directly**

Pros:
- Kernels can access time/parameters/multiple fields directly.

Cons:
- Splits the kernel ecosystem (two kernel types) or forces a breaking change.
- Duplicates responsibilities already represented by `AssemblyContext` + `RequiredData`.

**Option C — Systems bypasses `FE/Assembly` and runs its own element loop**

Pros:
- Maximum control for Systems; can inject arbitrary state.

Cons:
- Violates the “leverage existing FE subfolders” goal by re-implementing assembly orchestration.
- Risks divergence from the optimized/validated assembly infrastructure.

**Option D — Phase 1: restrict Systems to linear kernels that do not require solution data**

Pros:
- Very fast to deliver an end-to-end assembly path.

Cons:
- Defers a foundational capability and may force API changes later anyway.

**Suggested path**
- Implement **Option D** for Milestone 3 validation (linear assembly).
- Plan and implement **Option A** next, starting with `StandardAssembler` only, and require callers to provide a solution vector with the needed ghost values in parallel.

---

#### 4) First supported “multi-field” definition

**Option A — Single field (scalar)**

Pros:
- Minimal surface area; matches current AssemblyContext scalar assumptions.
- Best for proving Systems lifecycle and operator registry/dispatch correctness.

Cons:
- Does not exercise coupling/blocks; multiphysics design remains unvalidated until later.

**Option B — Single field (multi-component) with a documented DOF ordering convention**

Pros:
- Unlocks common “vector unknown” use cases without immediately requiring rectangular DOF maps.
- Constraints already have `ComponentMask`, so BC UX can be good once DOF/component mapping is standardized.

Cons:
- Requires a clear convention (component-major vs node-major ordering) exposed to kernels.
- `AssemblyContext` does not currently expose component mapping; kernels must infer it or Systems/Assembly must add helpers.

**Option C — Mixed space as a single `spaces::FunctionSpace`**

Pros:
- Conceptually attractive: “one field” that is internally block-structured.

Cons:
- Current `spaces::MixedSpace` is not assembly-ready (element/basis queries are not sufficient for combined DOFs).
- Risks introducing fragile special-casing in Assembly/Spaces early.

**Option D — True multiple independent fields/spaces**

Pros:
- Most general and matches multiphysics expectations cleanly.

Cons:
- Requires true rectangular/block assembly: different row/col DOF maps and consistent insertion into global blocks.
- Current `StandardAssembler` explicitly notes it assumes a single DOF map for rectangular assembly.

**Suggested path**
- Start with **Option A**.
- Add **Option B** once a single, explicit component/DOF ordering is committed and (ideally) exposed via lightweight helpers.
- Defer **Option C** until MixedSpace is made assembly-capable (or replaced).
- Implement **Option D** only after the assembly layer supports distinct row/col DOF maps and Systems has a multi-field DOF construction story.
