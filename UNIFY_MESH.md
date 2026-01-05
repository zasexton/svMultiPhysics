# UNIFY_MESH: Make `DistributedMesh` the Default `Mesh`

## Objective

Unify the Mesh library so that:

- Users/developers write against a single public mesh type (`svmp::Mesh`) in both serial and MPI builds.
- When MPI is enabled, `svmp::Mesh` behaves as a distributed mesh (partition/ghost/exchange/migrate as needed).
- When MPI is not enabled, `svmp::Mesh` behaves like a normal serial mesh while keeping the same API surface.
- Code no longer relies on inheritance differences or `#ifdef MESH_HAS_MPI` for basic mesh usage.

## Scope / Non-goals

- This document focuses on **API unification and migration**. It does **not** require “perfect” parallel algorithms for every operation on day 1, but it does require consistent, documented semantics.
- We will not silently change numerical meaning of existing routines without a clear API/semantic decision (owned vs ghost vs global, etc.).

## Baseline (Already Implemented)

- [x] `Code/Source/solver/Mesh/Core/DistributedMesh.h`: serial `DistributedMesh` refactored to composition (local mesh is always `MeshBase`).
- [x] `Code/Source/solver/Mesh/CMakeLists.txt`: always compile `Core/DistributedMesh.h/.cpp` (serial `.cpp` becomes empty TU).
- [x] `Code/Source/solver/Mesh/Mesh.h`: added factories returning `std::shared_ptr<DistributedMesh>` and `load_mesh(...)`.
- [x] `Code/Source/solver/Mesh/Tests/Unit/Observer/test_DistributedMeshStubEvents.cpp`: updated for composition (`dm.local_mesh()`).

---

## Phase Checklist (High-level)

- [x] **Phase 1** — Lock the public `Mesh` surface (serial + MPI verified)
- [x] **Phase 2** — API parity (`MeshBase`-like surface on `Mesh`) (serial + MPI verified)
- [x] **Phase 3** — Define distributed semantics (owned/ghost/global) (serial + MPI verified)
- [x] **Phase 4** — Unify construction, I/O, partition lifecycle (serial + MPI verified)
- [x] **Phase 5** — Migrate the codebase + deprecations (serial + MPI verified)
- [ ] **Phase 6** — Guardrails: tests + CI to prevent regressions (serial + MPI verified)

---

## Phase 1 — Lock The Public `Mesh` Surface

### Outcomes

- [x] `svmp::Mesh` is the single public runtime mesh type for users.
- [x] The existing template `svmp::Mesh<Dim>` no longer conflicts with the runtime `svmp::Mesh` name.
- [x] User-facing code can construct a mesh (and pass a communicator) without `#ifdef MESH_HAS_MPI`.
- [x] `Mesh.h` becomes the recommended include for all user code.

### File & Code Changes

- [ ] `Code/Source/solver/Mesh/Mesh.h`
  - [ ] Pick the public type strategy:
    - [x] **Option A (alias)**: `using Mesh = DistributedMesh;` (fastest path, fewest moving parts).
    - [ ] **Option B (façade)**: `class Mesh { ... }` wrapping `DistributedMesh` (more control; helps with deprecations and semantic defaults).
  - [x] Rename the existing template `svmp::Mesh<Dim>` to avoid naming conflict (recommended: `MeshView<Dim>` or `LocalMesh<Dim>`).
    - [x] Update aliases: `Mesh1D/2D/3D` to match the new name.
  - [ ] Introduce typed distributed wrapper aliases:
    - [x] `template<int Dim> using Mesh_t = DistributedMesh_t<Dim>;` (or equivalent).
  - [x] Change factories to return `std::shared_ptr<Mesh>` (and keep the current `create_mesh()` returning `DistributedMesh` as deprecated wrappers if needed).
  - [x] Move MPI-only factory overloads behind an always-available communicator abstraction (see below).

- [ ] `Code/Source/solver/Mesh/Core/MeshComm.h` (new)
  - [x] Add a lightweight communicator wrapper that exists in all builds (serial + MPI):
    - [x] Stores `MPI_Comm` in MPI builds; stores a trivial `{rank=0,size=1}` in serial.
    - [x] Provides `rank()`, `size()`, `is_parallel()`.
    - [x] Provides `native()` accessor only when MPI is enabled (or a safe “no-MPI” equivalent).
  - [x] Add convenience constructors: `MeshComm::world()`, `MeshComm::self()`, and a default `MeshComm{}` that means serial/self.

- [ ] `Code/Source/solver/Mesh/Core/DistributedMesh.h`
  - [x] Replace “generic placeholder CommT” constructors with `MeshComm` (or ensure `Mesh.h` factories accept `MeshComm` and delegate).
  - [x] Ensure MPI-specific surface is not required in headers that are included by serial-only consumers.

- [ ] `Code/Source/solver/Mesh/CMakeLists.txt`
  - [x] Export/install the new header `Core/MeshComm.h` as part of `svmesh` public headers.
  - [x] If Option B (façade) is selected and requires a `.cpp`, add it here.

- [ ] `Code/Source/solver/Mesh/README.md`
  - [x] Update documentation to present `Mesh.h` + `svmp::Mesh` as the default entry point and explain communicator usage.

### Verification (must be checked before Phase 1 is “done”)

- [x] Serial build: `MESH_ENABLE_MPI=OFF` builds `svmesh` and all enabled Mesh unit tests.
- [x] MPI build: `MESH_ENABLE_MPI=ON` builds `svmesh` and all enabled Mesh unit tests (including `Unit/Core` MPI tests).
- [x] A minimal example (single TU) can:
  - [x] `#include "Mesh/Mesh.h"` (or include path used by the project)
  - [x] `svmp::Mesh mesh;`
  - [x] `auto m = svmp::create_mesh(/* optional comm */);`
  - [x] Compile unchanged in both serial and MPI builds.

Notes:
- Full `ctest` currently reports failures outside Phase 1 scope; Phase 1 verification focuses on compilation + representative smoke tests.
- In this sandboxed environment, OpenMPI `mpiexec` requires escalated permissions (local socket/PMIx); outside the sandbox, MPI tests run normally.

---

## Phase 2 — API Parity (Stop Forcing `local_mesh()` in Common Code)

### Outcomes

- [x] `svmp::Mesh` supports the *common* `MeshBase` surface directly (builders, queries, topology/labels/fields access, search helpers, etc.).
- [x] Downstream code does not need to call `mesh.local_mesh()` for typical operations.
- [x] The API stays consistent between serial and MPI builds.

### File & Code Changes

- [x] `Code/Source/solver/Mesh/Core/DistributedMesh.h`
  - [x] Add forwarding methods (thin wrappers) matching `MeshBase` names/signatures where semantics are “local mesh”.
  - [x] Re-export any nested/alias types that downstream templates expect (kept minimal: `LoadFn`, `SaveFn`).
  - [x] Add `base()` synonym(s) (`base()` == `local_mesh()`).

- [ ] `Code/Source/solver/Mesh/Core/DistributedMesh.cpp`
  - [ ] If any forwarding wrappers need non-trivial logic or would bloat headers, move definitions to `.cpp`.

- [ ] Update call sites currently reaching into `local_mesh()` for basic operations:
  - [x] `Code/Source/solver/Mesh/Tests/Unit/Observer/test_DistributedMeshStubEvents.cpp`: switched back to `dm.build_from_arrays(...)`.
  - [ ] Search/Geometry/Validation tests and utilities: prefer `Mesh`/`DistributedMesh` surface once available.

### Verification

- [x] Added a compile-only “API parity” test TU calling a representative subset of the `MeshBase` surface on `svmp::Mesh`:
  - `Code/Source/solver/Mesh/Tests/Unit/PublicApi/test_MeshApiParity.cpp`
- [x] Serial (no MPI): `test_MeshApiParity` + `test_DistributedMeshStubEvents` pass.
- [x] MPI: all `Unit/Core` MPI tests pass (2 + 4 ranks), and `test_MeshApiParity` passes without requiring `MPI_Init`.
- [ ] Full Mesh unit suite passes (currently failing Search/Topology tests unrelated to this unification effort).

---

## Phase 3 — Define Distributed Semantics (Owned vs Ghost vs Global)

### Outcomes

- [x] Every `MeshBase`-like method on `Mesh` has a documented meaning in parallel (or is explicitly marked local-only).
- [x] Ambiguous operations expose explicit variants instead of guessing (owned/ghost/local/global).
- [x] Iteration and counts become predictable and correct.

### Decisions To Make (explicitly record in code/docs)

- [x] Default meaning of `n_cells()/n_vertices()/...` on `Mesh` in MPI:
  - [x] **Option A**: “local including ghosts”
  - [ ] **Option B**: “owned only”
  - [ ] **Option C**: “owned only, and provide `n_local_*` for owned+ghost”
- [x] Indexing expectations for methods that take `index_t` (local index vs global ID).
- [x] What “global” means for geometry/search (true global reductions vs best-effort/local).

### Semantics (Implemented)

- `index_t` is always a **local index** into the rank-local arrays (`0..n_local-1`).
- `gid_t` is a **global identifier** used to match entities across ranks (see `global_to_local_*`).
- `n_vertices()/n_cells()/n_faces()/n_edges()` are **local counts** and include ghosts if a ghost layer is present.
- Ownership states:
  - `Ownership::Owned`: this rank is the canonical owner and contributes to `global_n_*`.
  - `Ownership::Shared`: this rank has a local copy, but another rank is the owner (typical partition interface).
  - `Ownership::Ghost`: this rank has an imported ghost copy; **ghost copies never “win” ownership**.
- Global reductions:
  - `global_n_vertices/cells/faces()` return **owned-only** global totals (MPI reduction).
  - In MPI builds, these reductions are safe to call even before `MPI_Init` (they fall back to local counts).
- Search:
  - `locate_point(...)` is **local-only** (operates on the local mesh, including ghosts if present).
  - `locate_point_global(...)` is the **distributed** variant.

### Implementation Notes

- Face GIDs in MPI builds are canonicalized (derived from sorted vertex GIDs) so shared-face detection and face exchange patterns are well-defined.
- Shared-entity detection keys by `(EntityKind, gid_t)` and selects owners from the lowest-rank **non-ghost** copy.
- Ghost faces are defined as faces whose **incident cells are all ghost** (owned/ghost interface faces remain owned/shared).

### File & Code Changes

- [x] `Code/Source/solver/Mesh/Core/DistributedMesh.h`
  - [x] Add explicit count/query methods with clear semantics:
    - [x] `n_owned_vertices/cells/faces/edges`
    - [x] `n_shared_vertices/cells/faces/edges`
    - [x] `n_ghost_vertices/cells/faces/edges`
    - [x] `n_local_*` is not needed under Option A.
    - [x] Keep/clarify `global_n_*` methods and define when they are valid.
  - [x] Add explicit iteration/view helpers (simple filtered vectors):
    - [x] `owned_cells()`, `ghost_cells()`, `owned_vertices()`, etc.
  - [x] Clarify and enforce ownership invariants after ghost build/migration:
    - [x] `*_owner_` arrays are maintained by the `DistributedMesh` partition/ghost lifecycle (builders reset to owned; ghost build/migrate recompute via shared detection).
    - [x] `MeshEvent::PartitionChanged` remains the event boundary for partition/ghost mutations.

- [x] `Code/Source/solver/Mesh/Geometry/*` and `Code/Source/solver/Mesh/Validation/*`
  - [x] Default geometry/validation remains **local including ghosts** (matches `MeshBase` loops and ghosted local mesh representation).
  - [x] Owned-only variants are deferred to Phase 4+ where partition lifecycle + policies are formalized.

- [x] `Code/Source/solver/Mesh/Search/*`
  - [x] `locate_point` is local; `locate_point_global` is distributed.
  - [x] `build_search_structure` builds on the local mesh (includes ghosts when present).

### Verification

- [x] Add unit tests covering counts/iteration semantics under MPI:
  - [x] owned/shared/ghost counts after `build_exchange_patterns` + `build_ghost_layer`
  - [x] stability after `migrate` (global counts + invariants)
  - [x] `global_n_*` correctness (vertices/cells/faces)
- [x] Serial: semantics tests reduce to trivial (owned==local, ghost==0).

---

## Phase 4 — Unify Construction, I/O, And Partition Lifecycle

### Outcomes

- [x] `Mesh` construction/loading/saving is “single-call” from user code in both builds.
- [x] Partitioning/ghosting lifecycle is explicit and consistent (with events).
- [x] I/O policies are defined and implemented (rank-0 read + distribute vs parallel read/write).

### File & Code Changes

- [x] `Code/Source/solver/Mesh/Mesh.h`
  - [x] Provide a single set of factories that work in both builds:
    - [x] `create_mesh()` (default comm)
    - [x] `create_mesh(MeshComm comm)`
    - [x] `load_mesh(opts, MeshComm comm)` or `Mesh::load(...)`
  - [x] Provide `save_mesh(...)` or `Mesh::save(...)` that dispatches to `save_parallel` when MPI is enabled.
  - [x] Remove or deprecate direct `MPI_Comm` overloads from the public API (keep internal bridge if needed).

- [x] `Code/Source/solver/Mesh/Core/DistributedMesh.h` / `Code/Source/solver/Mesh/Core/DistributedMesh.cpp`
  - [x] Define build/partition semantics for “builder” entry points:
    - [x] If `build_from_arrays` is called in MPI, is it “local build” (each rank supplies its part) or “global build” (rank 0 supplies all)?
    - [x] If both are needed, add explicit APIs: `build_from_arrays_local(...)` and `build_from_arrays_global_and_partition(...)`.
  - [x] Align `load_parallel` and `load_mesh`:
    - [x] Decide default: rank-0 load + partition scatter vs reading pre-partitioned formats.
    - [x] Implement/clarify both if required.
  - [x] Ensure coordinate/field ghost exchange APIs have consistent naming and preconditions:
    - [x] `update_exchange_ghost_coordinates`
    - [x] `update_exchange_ghost_fields` / `update_ghosts(fields)`
  - [x] Ensure event emission is correct and consistent (`MeshEvent::PartitionChanged`, `MeshEvent::GeometryChanged`, `MeshEvent::FieldsChanged`, etc.).

- [x] `Code/Source/solver/Mesh/IO/*` (if present) and/or `MeshBase` I/O registry
  - [x] Decide whether distributed I/O should reuse the `MeshBase` registry or have a `DistributedMesh` registry.
  - [x] Document file formats supported in parallel and expected file layout (single vs per-rank files).

### Verification

- [x] Serial: `load_mesh` + `save` roundtrip works.
- [x] MPI: `load_mesh` policy works (rank-0 load + distribute, or parallel) and matches documented behavior.
- [x] MPI: ghost exchange and partition events are validated by unit tests.

---

## Phase 5 — Migrate The Codebase + Deprecations

### Outcomes

- [x] Internal solver modules consume `svmp::Mesh` instead of `MeshBase` or `DistributedMesh` directly (except where explicitly intended).
- [x] Minimal `#ifdef MESH_HAS_MPI` remains in user-facing codepaths.
- [x] Backward compatibility is maintained via deprecations for a release window (as appropriate).

### File & Code Changes (expected hotspots)

- [x] `Code/Source/solver/FE/*`
  - [x] Update signatures and overload sets to take `const svmp::Mesh&` / `svmp::Mesh&` where possible.
  - [x] Consolidate duplicate `MeshBase`/`DistributedMesh` overloads into:
    - [x] one `Mesh` overload, or
    - [x] a templated `MeshLike` implementation used by both.
  - [x] Remove conditional includes like `__has_include("../../Mesh/Core/DistributedMesh.h")` in favor of a stable include (`Mesh/Mesh.h`).

- [x] `Code/Source/solver/Mesh/Motion/*`
  - [x] Replace mixed `MeshBase` vs `DistributedMesh` entry points with `Mesh` entry points.
  - [x] Keep distributed-specialized paths internally where needed, but hide behind `Mesh` APIs/queries (`is_parallel()`, ownership filters).

- [x] `Code/Source/solver/Mesh/Adaptivity/*`
  - [x] Update API to accept `Mesh` and use explicit owned/ghost semantics (Phase 3).

- [x] Deprecation layer (choose locations based on strategy)
  - [x] Keep `MeshBase` accessible for low-level local operations, but discourage as the *default* public type.
  - [x] Provide deprecated typedefs/factories where applicable (no external `create_distributed_mesh`-style factories were found in this repo).

### Verification

- [x] Serial: full solver build (or the relevant subset) compiles with `MESH_ENABLE_MPI=OFF`.
- [x] MPI: full solver build (or the relevant subset) compiles with `MESH_ENABLE_MPI=ON`.
- [x] Key integration tests (if any) still pass.

---

## Phase 6 — Guardrails: Tests + CI

### Outcomes

- [ ] Regressions in serial/MPI API parity are caught automatically (CI integration deferred).
- [x] Semantics regressions (owned/ghost/global) are tested.
- [x] Documentation/examples stay aligned with the public API.

### File & Code Changes

- [x] Add a dedicated “API surface” compile test:
  - [x] `Code/Source/solver/Mesh/Tests/Unit/Core/test_MeshPublicApi.cpp` (new)
    - [x] Includes only `Mesh.h`
    - [x] Exercises constructors/factories + a representative subset of MeshBase-like calls on `svmp::Mesh`
    - [x] Builds in both serial and MPI configurations

- [x] Add MPI-focused semantics tests (owned/ghost/global):
  - [x] `Code/Source/solver/Mesh/Tests/Unit/Core/test_DistributedSemantics.cpp` (new)

- [x] `Code/Source/solver/Mesh/Tests/CMakeLists.txt`
  - [x] Ensure new tests are discovered and categorized correctly.
  - [x] Ensure MPI tests run under `ctest` with `mpiexec` (already supported for `Unit/Core`).

- [ ] CI hooks (wherever CI is configured in this repo)
  - [ ] Add a serial Mesh test job (`MESH_ENABLE_MPI=OFF`).
  - [ ] Add an MPI Mesh test job (`MESH_ENABLE_MPI=ON`, run `ctest` so `Unit/Core` MPI tests execute).
  - [ ] Prefer “minimal dependencies” configs (e.g., `MESH_ENABLE_VTK=OFF`) unless I/O coverage is required.

### Verification

- [ ] Serial CI job runs and passes.
- [ ] MPI CI job runs and passes.
- [ ] A deliberate API break (e.g., removing a forwarder) causes the API compile test to fail.
