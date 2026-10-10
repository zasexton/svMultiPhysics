# FE/Backends — Implementation Notes

## Build / feature flags

- `FE_ENABLE_ASSEMBLY=ON` is required to compile `FE/Backends` (the backends expose `assembly::GlobalSystemView` insertion adapters).
- `FE_ENABLE_EIGEN=ON` enables the Eigen backend and defines `FE_HAS_EIGEN=1` on the `svfe` target.
- `FE_ENABLE_PETSC=ON` enables the PETSc backend and defines `FE_HAS_PETSC=1` on the `svfe` target.
  - PETSc is discovered via `find_package(PETSc CONFIG)` when available, or by providing `SV_PETSC_DIR` (legacy; points at `PETSC_DIR/PETSC_ARCH`).
- `FE_ENABLE_TRILINOS=ON` enables the Trilinos backend and defines `FE_HAS_TRILINOS=1` on the `svfe` target.
- `FE_ENABLE_MUMPS=ON` (default OFF, requires `FE_ENABLE_MPI=ON`) links an external parallel double-precision MUMPS build and defines `FE_HAS_MUMPS=1` on the `svfe` target (`cmake/EnableMUMPS.cmake`). MUMPS is never vendored.
  - `SV_MUMPS_DIR=<prefix>` must contain `include/dmumps_c.h` and `lib/libdmumps.a`, `libmumps_common.a`, `libpord.a`.
  - MUMPS's own dependencies are searched with `SV_MUMPS_SCALAPACK_DIR`, `SV_MUMPS_BLAS_DIR` (OpenBLAS, or reference LAPACK + BLAS), `SV_MUMPS_METIS_DIR` (empty when MUMPS was built without METIS), the MPI Fortran libraries next to the MPI C library and the compiler's `libgfortran`/`libquadmath`; `SV_MUMPS_EXTRA_LIBRARIES` (libraries or linker flags) replaces that search.
  - Without the option the MUMPS entry points are stubs (`mumpsAvailable()` is false) and every default path is unchanged.
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
  - MPI is supported through FSILS owned-row operators with explicit halo storage:
    - `FsilsMatrix` supports `sparsity::DistributedSparsityPattern` and stores owned rows with ghost nodes available as local columns and vector halo entries.
    - Ghost row requirements (per rank):
      - Ghost rows must include **all components** of each ghost node (i.e., if `dof_per_node=k`, then every ghost node must provide `k` dof-rows).
      - Ghost row columns must reference only nodes present in the local overlap set (owned nodes + ghost nodes).
    - FE assembly routes off-owner row contributions to the owning rank; solver vectors use explicit owner-to-ghost synchronization only when ghost values are required as operator input.
  - Vector ghost synchronization:
    - `FsilsVector::localSpan()` exposes the full overlap storage (`owned nodes + ghost nodes`, interleaved by `dof_per_node`).
    - `FsilsVector::updateGhosts()` performs an explicit **owner → ghost** update that copies owned values into ghost slots.
  - Gathered sparse direct solve (`SolverMethod::Direct`, deck `<LS type="Direct">` with `<Linear_algebra type="fsils">`; opt-in, the Krylov path is unchanged):
    - `FsilsGatheredDirectSolver` (`FsilsDirectSolver.{h,cpp}`, requires `FE_ENABLE_EIGEN=ON`) gathers every rank's owned rows (global node numbering, columns sorted), right-hand side and Dirichlet DOFs on rank 0, factors with Eigen `SparseLU` and scatters the owned solution; ghosts are refreshed with `updateGhosts()`.
    - Dirichlet DOFs get `x = 0` with unit rows/columns (the operator the Krylov path solves through zero RCS/diagonal weights on Dirichlet faces). No RCS is applied.
    - Stored structure: diagonals plus every entry that has been nonzero since the node pattern or the Dirichlet set changed; exact zeros of the node blocks are not stored (about half of the block entries in the free-surface systems). Ordering (approximate minimum degree on `A + A^T` of the stored structure) and symbolic analysis are kept until that structure changes. On the free-surface systems scalar AMD gives about a third of the factorization time of AMD on the node graph and a fifth of COLAMD with partial pivoting.
    - Numerics: exact power-of-two row/column equilibration, threshold partial pivoting with diagonal preference (0.01) in SparseLU's symmetric mode, iterative refinement (at most 3 steps) to `max(abs_tol, rel_tol ||b||)`. `SolverReport::iterations` is `1 + refinement steps`.
    - Only rank 0 factors, so the result does not depend on the row distribution; the rank count enters only through assembly round-off and the global node numbering.
    - Singular or non-finite factorizations are reported on every rank as `numerical_breakdown` with a zero correction. Rank-one, reduced and grouped bordered operator updates are refused (`NotImplementedException`).
    - Rank 0 logs `diagnostic=fsils_direct_analysis` at every new analysis and `diagnostic=fsils_direct_summary` every 200 solves and at destruction (counts, times, `nnz(L+U)`, factor memory, peak RSS).
  - Aggregation multigrid right preconditioner (`RightPreconditionerType::Amg`, deck `<Right_preconditioner>amg</Right_preconditioner>` inside `<LS type="GMRES">`; opt-in, the default path is unchanged):
    - `FsilsAmgHierarchy` (`FsilsAmg.{h,cpp}`) runs one V-cycle per GMRES iteration through the FSILS right-preconditioner hook (`FsilsKrylovPreconditioner`, kind `amg`) on the nodal blocks of the RCS-scaled operator, so it acts on the whole coupled system (no physics-specific splitting).
    - Partition independence: aggregates are a distance-2 maximal independent set computed in synchronous rounds with priorities hashed from `DofPermutation::node_key` (hash of the vertex coordinates, filled by `FESystem` setup); roots take their distance-1 neighbours and the remaining nodes join the neighbouring aggregate of highest priority. Unknowns without off-diagonal couplings (Dirichlet, condensed constraints) stay out of the aggregates and are solved by the smoother. Smoother: Chebyshev iteration on the block-Jacobi operator `D^{-1} A` with the bound `||D^{-1} A||_inf` (a maximum reduction, exact on every partition). Coarse operators are Galerkin products; the coarsest one (at most `AMG_coarse_nodes` nodes) is gathered on rank 0 in key order and factored by `SparseLU`. All couplings between ranks are kept, so the rank count enters only through the summation order of reductions and Galerkin sums (round-off). Without node keys (other callers) the backend node ids are used, which makes the aggregates partition dependent.
    - Settings (`SolverOptions::amg_*`, deck `<AMG_*>`): `smoother_degree` (3), `prolongator` (`plain` default, or `smoothed` with one damped block-Jacobi step), `coarse_nodes` (600), `max_levels` (10), `lambda_iterations` (0: use the bound; >0 power iterations capped by it), `strength_threshold` (0: every nonzero block couples). `<Preconditioner_reuse>true</Preconditioner_reuse>` reuses the hierarchy across solves under `PreconditionerReusePolicy` (the finest values are then copied).
    - Each solve logs `diagnostic=fsils_right_preconditioner kind=amg ...` with the hierarchy summary (levels, nodes, blocks, setup, aggregation, Galerkin and coarse-factor times, MIS rounds).

## MUMPS distributed direct solves (optional, `FE_ENABLE_MUMPS=ON`)

- `MumpsDistributedSolver` (`Backends/MUMPS/MumpsDistributedSolver.{h,cpp}`) is a small exact-solve utility, not a `GenericMatrix` backend: every rank passes any subset of the global triplets (0-based global indices, duplicates are summed, MUMPS distributed assembled input `ICNTL(18)=3`), `factorize()` is collective, and `solveReplicated()` takes the right-hand side on rank 0 and returns the full solution on every rank.
  - Symmetry: `SymmetricPositiveDefinite` (LDL^T without pivoting, pass the lower or the upper triangle), `GeneralSymmetric` (LDL^T with pivoting) or `Unsymmetric` (LU with threshold partial pivoting).
  - The analysis (sequential METIS ordering by default, `ICNTL(28)=1`, `ICNTL(7)=5`) is kept while the global pattern is unchanged (same triplet positions on every rank), so a refactorization runs only the numerical phase. Workspace shortfalls are retried with more relaxation (`ICNTL(14)`), at most four times.
  - Errors (bad indices, singular or non-finite factorizations, MUMPS failures) are reported on every rank (`factorize()`/`solveReplicated()` return false, `lastError()` holds the message); the factorization is then dropped.
  - The factorization uses all ranks of the communicator, so the result can differ with the rank count at round-off (pivot order and summation order). Repeated factorizations of the same matrix on the same ranks are reproducible in the tests (`test_MumpsDistributedSolverMPI.cpp` prints the bitwise status).
  - Destruction is collective (MUMPS `JOB=-2`); keep instances on the same ranks for their whole life.
- Used by the PDE velocity extension (`Application/Core/LevelSetPdeVelocityExtension.cpp`) when `SVMP_PDE_EXTENSION_FACTORIZATION=mumps`: the replicated dry-region system is factorized once over all ranks (each rank passes the lower triangle of the rows `r % size == rank`; harmonic operator SPD, `pde_normal` general symmetric) and the three velocity components are solved collectively; components with the same unknowns (the same wall masks) have the same matrix and share one factorization. `SVMP_PDE_EXTENSION_MUMPS_DIAGNOSTICS=1` (or `SVMP_TRACE_LEVEL_SET_ADVECTION=1`) makes rank 0 log every factorization (`diagnostic=pde_extension_mumps`: size, entries, factor entries, analysis and factorization seconds, MUMPS peak memory of the largest rank and of all ranks). The 3-entry factorization cache, its collective votes and the `SVMP_PDE_EXTENSION_SELF_CHECK` comparison are unchanged. A failed MUMPS factorization falls back to the backend direct solve. The variable is rejected at startup when the binary was built without `FE_ENABLE_MUMPS`. Default (`lu_colamd`, unset) is unchanged.

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
