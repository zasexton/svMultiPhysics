# FE/Backends — Implementation Notes

## Build / feature flags

- `FE_ENABLE_ASSEMBLY=ON` is required to compile `FE/Backends` (the backends expose `assembly::GlobalSystemView` insertion adapters).
- `FE_ENABLE_EIGEN=ON` enables the Eigen backend and defines `FE_HAS_EIGEN=1` on the `svfe` target.
- `FE_ENABLE_PETSC=ON` enables the PETSc backend and defines `FE_HAS_PETSC=1` on the `svfe` target.
  - PETSc is discovered via `find_package(PETSc CONFIG)` when available, or by providing `SV_PETSC_DIR` (legacy; points at `PETSC_DIR/PETSC_ARCH`).
- `FE_ENABLE_TRILINOS=ON` enables the Trilinos backend and defines `FE_HAS_TRILINOS=1` on the `svfe` target.
- The FSILS backend is always compiled when `FE_ENABLE_ASSEMBLY=ON` and defines `FE_HAS_FSILS=1` on the `svfe` target.
  - The FSILS linear-solver sources are vendored under `Backends/FSILS/liner_solver` (a copy of `Code/Source/liner_solver`).
  - FSILS depends on MPI (even for `MPI_COMM_SELF`), so FE will link MPI for `svfe` when Assembly/Backends are enabled.
  - The vendored FSILS sources still include legacy solver headers (`CmMod.h`, `Array.h`, `Vector.h`, etc.). FE provides minimal static definitions for those templates in `Backends/FSILS/FsilsLegacyStatics.cpp`.
  - If you build the full svMultiPhysics stack that also links the legacy `Code/Source/liner_solver` library, avoid linking both FSILS implementations into the same binary (duplicate symbol risk).

## Current backend surface

- Core interfaces live in `Backends/Interfaces/`:
  - `GenericMatrix`, `GenericVector`, `LinearSolver`
  - `BackendFactory` + `BackendKind`
- Options/diagnostics live in `Backends/Utils/BackendOptions.h`.

## Eigen backend

- Storage is `Eigen::SparseMatrix<double, RowMajor, int>` with a fixed sparsity pattern created from `sparsity::SparsityPattern`.
- Assembly insertion is done through an `assembly::GlobalSystemView` wrapper; updates are **structure-preserving** (entries not present in the sparsity pattern are ignored).
- `EigenLinearSolver` supports:
  - Direct: `Eigen::SparseLU` (factorization uses a column-major copy)
  - Iterative: `Eigen::ConjugateGradient`, `Eigen::BiCGSTAB`, `Eigen::GMRES` (from `unsupported/Eigen/IterativeSolvers`)
    - `SolverMethod::FGMRES` is mapped to Eigen `GMRES` (note: Eigen's implementation is not a true "flexible" GMRES variant).
    - `SolverMethod::BlockSchur` is treated as a `GMRES` solve on the monolithic operator (no explicit Schur complement / saddle-point preconditioning).
    - `PreconditionerType::ILU` is supported for `BiCGSTAB`/`GMRES` via `Eigen::IncompleteLUT` (AMG is not supported).

## FSILS backend (optional)

- Intended as an in-tree, swappable backend.
- Current FE integration is intentionally conservative:
  - Matrix setup translates FE CSR sparsity into an FSILS `FSILS_lhsType` compatible structure.
  - Solve path uses a **work copy** of the matrix values because FSILS preconditioning / solver routines may modify the `Val` array in-place.
  - `FsilsFactory(dof_per_node)` selects the FSILS block size (default `dof_per_node=1`).
    - `BackendFactory::create("fsils", BackendFactory::CreateOptions{.dof_per_node=...})` provides the same knob via the generic factory.
    - The FE view uses **interleaved DOF ordering** per node: global DOF `gid = node*dof + component`.
    - `FsilsMatrix` builds a node-level sparsity pattern (nnz blocks) and stores dense `dof×dof` blocks in FSILS column-major layout.
  - `SolverMethod::BlockSchur` maps to the FSILS NS solver (`LS_TYPE_NS`) and requires `dof=3` (2D) or `dof=4` (3D) with the per-node ordering `(u,v[,w],p)`.
    - The NS solver uses `max_iter` to size `O(nNo * max_iter)` workspace; for safety, very large values are treated as unset (fallback to the FSILS default).
  - FSILS preconditioning notes:
    - The upstream `fsils_solve()` path always applies a post-solve diagonal scaling step (`Wc ⊙ R`); if no preconditioner routine runs, `Wc` is undefined.
    - For correctness, the FE FSILS backend treats `PreconditionerType::None` (and unsupported ILU/AMG requests) as the built-in diagonal preconditioner (`PREC_FSILS`), unless `RowColumnScaling`/`fsils_use_rcs` is requested.
    - `FsilsLinearSolver` detects numerical breakdowns (NaN/Inf residuals, corrupted iteration counts, non-finite solution values) and returns a safe `SolverReport` with `converged=false` and a zeroed solution vector.
  - MPI is supported via FSILS’ native **overlap/shared-node** communication model:
    - `FsilsMatrix` supports `sparsity::DistributedSparsityPattern` and uses stored **ghost rows** to include overlap nodes locally (owned nodes + ghost nodes).
    - Ghost row requirements (per rank):
      - Ghost rows must include **all components** of each ghost node (i.e., if `dof_per_node=k`, then every ghost node must provide `k` dof-rows).
      - Ghost row columns must reference only nodes present in the local overlap set (owned nodes + ghost nodes).
    - `FsilsLinearSolver` applies an FSILS `COMMU` to its working RHS vector before calling `fsils_solve()` (FSILS assumes overlap contributions have been communicated before norm/dot operations).
  - Vector ghost synchronization:
    - `FsilsVector::localSpan()` exposes the full overlap storage (`owned nodes + ghost nodes`, interleaved by `dof_per_node`).
    - `FsilsVector::updateGhosts()` performs an **owner → ghost** update (copies owned values into ghost slots) using FSILS overlap communication (`fsils_commuv`) with local ghost slots zeroed to avoid double-counting.

## PETSc backend (optional)

- `PetscVector`/`PetscMatrix` wrap PETSc `Vec`/`Mat`, with `GlobalSystemView` insertion implemented via `VecSetValues` / `MatSetValues`.
- Vector ghost synchronization:
  - When a `PetscMatrix` is created from `sparsity::DistributedSparsityPattern`, subsequent vectors created by the same `PetscFactory` use `VecCreateGhost()` with the pattern’s ghost column map.
  - `PetscVector::localSpan()` exposes PETSc’s local ghosted form (`owned entries` followed by `ghost entries`).
  - `PetscVector::updateGhosts()` calls `VecGhostUpdateBegin/End()` to refresh ghost entries from the owning ranks.
- Matrix allocation:
  - Serial `sparsity::SparsityPattern` is supported only when `MPI_Comm_size(PETSC_COMM_WORLD) == 1` (otherwise use `sparsity::DistributedSparsityPattern`).
  - `sparsity::DistributedSparsityPattern` uses PETSc `MatCreateAIJ` preallocation (diag/offdiag nnz per owned row).
- `PetscLinearSolver` wraps `KSP`/`PC` and supports:
  - `SolverOptions::petsc_options_prefix` + `SolverOptions::passthrough` to inject PETSc options before `KSPSetFromOptions()`.
  - `PreconditionerType::FieldSplit` for `BlockMatrix`/`BlockVector` systems using `MatNest`/`VecNest` and `PCFIELDSPLIT` with stride `IS` splits derived from block offsets.
  - `SolverMethod::GMRES`/`FGMRES` and `PreconditionerType::AMG`/`ILU` mappings, with best-effort override via `KSPSetFromOptions()`.
  - `SolverMethod::BlockSchur` as a 2×2 `PCFIELDSPLIT` Schur setup for `BlockMatrix`/`BlockVector` saddle-point systems.

## Trilinos backend (optional)

- `TrilinosVector`/`TrilinosMatrix` wrap `Tpetra::Vector` / `Tpetra::CrsMatrix`, and `TrilinosLinearSolver` uses Belos iterative solvers.
- Current implementation choices/limitations:
  - Direct solvers are not wired yet (Amesos2 would be the natural next step).
  - Field-split preconditioning is not implemented.
  - Serial `sparsity::SparsityPattern` matrices are supported only when `Tpetra::getDefaultComm()->getSize() == 1` (otherwise use `sparsity::DistributedSparsityPattern`).
  - `SolverMethod::GMRES` is mapped to Belos `PseudoBlockGmres`.
  - `PreconditionerType::ILU` is a best-effort Ifpack2 ILU-style preconditioner (depends on the Trilinos build).
  - `PreconditionerType::AMG` uses MueLu when available in the Trilinos build (guarded by header detection).
  - `SolverOptions::trilinos_xml_file` is applied via `Teuchos::updateParametersFromXmlFile()` for solver factory configuration.
  - Assembly is **owned-row insertion only**: attempts to insert into non-owned rows throw (a future improvement would use an Exporter/FE-style assembly path, or Tpetra FE objects if available).
  - Vector ghost synchronization:
    - When a `TrilinosMatrix` is created from `sparsity::DistributedSparsityPattern`, subsequent vectors created by the same `TrilinosFactory` create an **overlap vector** (owned + ghost) and a `Tpetra::Import` from the owned map.
    - `TrilinosVector::localSpan()` exposes the overlap layout (owned entries first, then ghosts in the pattern’s ghost-column order).
    - `TrilinosVector::updateGhosts()` performs an Import (`INSERT`) to refresh ghost entries from owners.

## Block systems

- `BlockVector` and `BlockMatrix` provide backend-agnostic block structure for multi-field systems.
  - They support assembly insertion through a composite `GlobalSystemView` that routes each `(row,col)` to the appropriate sub-block view.
  - `BlockVector::localSpan()` is only available for the single-block case (for multi-block, use `block(i).localSpan()`).

## Unit tests

- `Tests/Unit/Backends/` covers: backend kind parsing, factory behavior, Eigen vector/matrix assembly views (including add modes), `A*x` multiply, direct + iterative solves, and option-string helpers.
- Additional solver verification coverage:
  - `Tests/Unit/Backends/test_LinearSolverConformance.cpp` runs a backend-parameterized conformance suite (options validation, classic matrices, Poisson stencils, scaling edge cases, nonconvergence, and assembly invariants).
  - `Tests/Unit/Backends/test_LinearSolverMPI.cpp` extends distributed verification (FSILS overlap solves and dot/norm reductions; PETSc/Trilinos MPI tests when enabled).
    - Includes explicit MPI tests for `GenericVector::updateGhosts()` owner→ghost propagation for FSILS/PETSc/Trilinos.
- PETSc/Trilinos unit tests are built conditionally:
  - PETSc tests run when `FE_HAS_PETSC` is defined.
  - Trilinos tests are a separate executable with `Tpetra::ScopeGuard` and are built when `FE_ENABLE_TRILINOS=ON` (and `FE_HAS_TRILINOS` is defined).

## Known follow-ups

- FSILS `GlobalSystemView` insertion currently uses per-entry binary search within each CSR row. A faster precomputed (row,col)->nnz-index map could be added if assembly profiling shows this is a bottleneck.
- FSILS block assembly currently assumes interleaved per-node ordering (`gid = node*dof + component`); alternative DOF layouts would require an explicit mapping layer.
- Trilinos MPI-safe FE assembly (handling non-owned row contributions) will require a different insertion strategy than the current owned-only `replaceLocalValues`/`sumIntoLocalValues` path.
