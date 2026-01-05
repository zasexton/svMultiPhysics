# FE/Backends Subfolder — Design Plan

## Overview

`FE/Backends` is the FE library’s **linear algebra + linear-solver abstraction layer**. Its purpose is to decouple finite element mechanics (assembly, DOF management, constraints) from the specific packages used to **store matrices/vectors** and **solve linear systems** (Eigen, PETSc, Trilinos, etc.).

This module ensures that `FE/Systems` and `FE/Assembly` can remain agnostic to whether code is running on a laptop (Serial/Eigen) or a supercomputer (MPI/PETSc or MPI/FSILS).

## Scope of Responsibilities

1.  **Uniform Data Storage:**
    - Provide generic wrappers for **Vectors** (distributed arrays) and **Sparse Matrices** (CSR/CSC/BSR as required by the backend/solver).
    - Own the memory allocation for global system objects.

2.  **Solver Interfaces:**
    - Provide a uniform API for **Linear Solvers** (KSP/Krylov methods) and **Preconditioners**.
    - Abstract away library-specific configuration (e.g., mapping generic options like "gmres" + "ilu" to PETSc/Trilinos equivalents).
    - Provide a way to **pass through** package-native option strings when needed (important for PETSc/Trilinos, which expose many third-party solvers through their own configuration layers).

3.  **Assembly Integration:**
    - Provide the mechanism to convert opaque backend matrices into `assembly::GlobalSystemView` objects (the writable interface used by kernels).

4.  **Parallel Data Transfer:**
    - Handle vector updates, ghost value exchange, and norms/dot-products in a backend-agnostic way.

---

## Architecture

The architecture uses a **Factory Pattern** combined with **polymorphic wrappers** (similar in spirit to the linear algebra adapter layers in deal.II and MFEM). While virtual function overhead is avoided in inner assembly loops (via `assembly::GlobalSystemView` writing directly into backend-specific storage), high-level management (setup, solve, diagnostics) uses virtual interfaces.

### Backend model: “storage” vs “solver package”

Many “solver packages” are most effectively supported **through** PETSc/Trilinos rather than by adding a bespoke backend for each solver:

- PETSc can expose Krylov methods + preconditioners (and several external packages) behind `KSP`/`PC` (e.g., algebraic multigrid via hypre, or sparse direct factorization via MUMPS / SuperLU_DIST / STRUMPACK / PaStiX, when available in the PETSc build).
- Trilinos can expose iterative solvers (Belos), preconditioners (Ifpack2, MueLu), and direct solvers (Amesos2), while keeping a single “Trilinos backend” surface.

**Design implication:** `FE/Backends` should treat “backend selection” as selecting a *backend stack* (matrix/vector storage + solver provider), while allowing package-native configuration pass-through to avoid over-constraining what PETSc/Trilinos can do.

### 1. Core Interfaces (`Interfaces/`)

-   **`BackendKind`:** Enumerates supported backend families and provides string conversion helpers.
-   **`GenericVector`:** Abstract base for vector operations:
    -   `add(val)`, `set(val)`, `scale(alpha)`
    -   `norm()`, `dot(other)`
    -   `updateGhosts()` (for parallel consistency)
-   **`GenericMatrix`:** Abstract base for matrix operations:
    -   `zero()`
    -   `finalizeAssembly()`
    -   `mult(x, y)` (y = Ax)
-   **`LinearSolver`:** Abstract base for solving $Ax = b$.
    -   `setOptions(...)` + `solve(A, x, b)` returning a `SolverReport`
-   **`BackendFactory`:** Static entry point to create objects based on the selected backend type.

### 2. Implementations

#### A. Eigen (Serial / Reference)
*Designed for unit testing, debugging, and small-scale serial problems.*
-   **Vector:** Wraps `Eigen::VectorXd`.
-   **Matrix:** Wraps `Eigen::SparseMatrix<double, Eigen::RowMajor>`.
-   **Solver:** Wraps `Eigen::SparseLU` (Direct) and `Eigen::ConjugateGradient` / `Eigen::BiCGSTAB` (Iterative).

#### B. FSILS (Native / In-tree Optional Backend)
*Designed for an in-tree, MPI-capable iterative solver backend that can be swapped in like any other.*

FSILS already exists in this repository (see `Code/Source/solver/liner_solver/`) and provides distributed sparse matrix storage (`fsi_linear_solver::FSILS_lhsType`) plus Krylov solvers (CG/GMRES/BiCGS/NS) and simple preconditioning (diagonal/Jacobi, row-column scaling).

**FE integration strategy:** FSILS is vendored into `FE/Backends/FSILS/liner_solver` so the `"fsils"` backend is always available when `FE/Backends` is built (no external dependency or separate CMake gate).

The FE backend integration should be optional and follow the same factory pathway as other backends (e.g., `backend_type="fsils"`):

- **Matrix:** wraps `FSILS_lhsType` plus the FSILS “values array” (block-structured by DOF, as FSILS expects).
- **Vector:** wraps a contiguous vector type (backend-owned), with MPI exchange implemented either through FSILS comm utilities or FE’s MPI wrapper.
- **Solver:** wraps `fsi_linear_solver::fsils_ls_create()` and `fsi_linear_solver::fsils_solve()` (or higher-level FSILS entry points, if exposed).
- **Assembly view:** provide a `GlobalSystemView` implementation that inserts element contributions into FSILS storage without per-entry searches (requires that the FE sparsity builder can provide a stable CSR-like pattern and/or a (row,col)→nnz-index map during setup).

This backend is particularly valuable as:
- a “native” option that does not require PETSc/Trilinos to be present,
- a compatibility bridge for existing SimVascular solver workflows,
- a baseline MPI path while PETSc/Trilinos features are staged in.

Limitations should be explicit in the API/plan: FSILS does not aim to replicate the breadth of PETSc/Trilinos preconditioning and external solver integrations.

#### C. PETSc (Parallel / Production)
*Designed for scalable MPI simulations.*
-   **Vector:** Wraps `Vec` (PETSc object).
-   **Matrix:** Wraps `Mat` (MPIAIJ format).
-   **Solver:** Wraps `KSP` and `PC`, with support for pass-through PETSc options to access PETSc-native and PETSc-enabled third-party solvers/preconditioners.

#### D. Trilinos (Future Work)
*Designed for advanced block preconditioners and GPU support.*
-   **Vector:** Wraps `Tpetra::Vector`.
-   **Matrix:** Wraps `Tpetra::CrsMatrix`.

---

## Third-Party Solver Packages to Consider (and how to support them)

The goal is to **avoid** a proliferation of one-off backends. Prefer these integration paths:

1. **Through PETSc** (recommended when PETSc is enabled)
   - AMG / multigrid: hypre (BoomerAMG) and PETSc GAMG.
   - Sparse direct solvers (when PETSc is configured with them): MUMPS, SuperLU_DIST, STRUMPACK, PaStiX, PARDISO, etc.
   - Practical implication for `FE/Backends`: expose a minimal stable `SolverOptions` API plus a string pass-through channel for PETSc option prefixes.

2. **Through Trilinos** (recommended when Trilinos is enabled)
   - Iterative: Belos; preconditioning: Ifpack2, MueLu; direct: Amesos2.
   - Practical implication: the FE solver option schema should not assume PETSc-only concepts (e.g., keep a generic “preconditioner type + parameters” model, plus pass-through).

3. **Through Eigen (serial-only convenience)**
   - Eigen can optionally interoperate with SuiteSparse (UMFPACK/CHOLMOD), SuperLU, PaStiX, MKL PARDISO, etc. (when those libs are available).
   - Practical implication: treat these as “Eigen solver choices” rather than separate FE backends.

4. **Direct integration (only when necessary)**
   - Header-only or very lightweight libraries (e.g., AMGCL) could be considered later, but should not be the default path.

---

## Mapping Strategy: “Generic Intent” → Backend Implementation

The FE library exposes a **small, stable** set of generic solver/preconditioner choices and then provides “escape hatches” so advanced users can access backend-native configuration (PETSc options database, Trilinos parameter lists, etc.).

| Generic FE option | FSILS backend | PETSc backend | Trilinos backend |
|---|---|---|---|
| Method: `GMRES` | `LS_TYPE_GMRES` | `KSPGMRES` | Belos `PseudoBlockGmres` |
| Method: `FGMRES` | (maps to GMRES) | `KSPFGMRES` | (maps to GMRES) |
| Method: `CG` | `LS_TYPE_CG` | `KSPCG` | Belos `CG` |
| Method: `BiCGSTAB` | `LS_TYPE_BICGS` | `KSPBCGS` | Belos `BiCGStab` |
| Method: `BlockSchur` (saddle point) | `LS_TYPE_NS` | `KSP*` + `PCFIELDSPLIT` (Schur) | Belos (+ Thyra) (future) |
| Prec: `Diagonal` | `PREC_FSILS` | `PCJACOBI` | Ifpack2 Relaxation (Jacobi) |
| Prec: `ILU` | (fallback) | `PCILU` | Ifpack2 ILU/ILUT |
| Prec: `AMG` | (fallback) | `PCGAMG` (or `PCHYPRE` when available) | MueLu (when available) |
| Prec: `RowColumnScaling` | `PREC_RCS` | diagonal scaling (best-effort) | (best-effort) |

**Notes**
- “BlockSchur / saddle point” is the generic notion of solving a 2×2 block system (Navier–Stokes, incompressible elasticity, FSI, mixed methods). It is not fluid-specific.
- PETSc/Trilinos can expose many additional solvers/preconditioners (hypre, MUMPS, SuperLU_DIST, STRUMPACK, PaStiX, Amesos2, etc.) and should be accessed through their native configuration interfaces when needed.

## Directory Structure

Target (planned) layout:

```text
Code/Source/solver/FE/Backends/
├── CMakeLists.txt
├── Interfaces/
│   ├── BackendFactory.h      # Factory for creating matrices/vectors
│   ├── GenericVector.h       # Vector abstract base class
│   ├── GenericMatrix.h       # Matrix abstract base class
│   └── LinearSolver.h        # Solver abstract base class
├── Eigen/
│   ├── EigenFactory.h
│   ├── EigenVector.h
│   ├── EigenMatrix.h
│   └── EigenLinearSolver.h
├── FSILS/
│   ├── FsilsFactory.h
│   ├── FsilsVector.h
│   ├── FsilsMatrix.h
│   └── FsilsLinearSolver.h
├── PETSc/
│   ├── PetscFactory.h
│   ├── PetscVector.h
│   ├── PetscMatrix.h
│   └── PetscLinearSolver.h
└── Utils/
    └── BackendOptions.h      # Structs for configuring solvers (tol, max_iter)
```

---

## Implementation Plan

### Milestone 1: The Abstraction Layer & Eigen Backend
**Goal:** Run a serial simulation using Eigen without direct Eigen calls in `Systems`.

-   [x] Define `GenericVector` and `GenericMatrix` interfaces.
-   [x] Implement `EigenVector` and `EigenMatrix`.
-   [x] Implement `BackendFactory` to instantiate Eigen objects.
-   [x] Create a `createAssemblyView()` API (or helper) that returns the `assembly::GlobalSystemView` required by the Assembler.

### Milestone 2: Linear Solver Interface
**Goal:** Solve $Ax=b$ generically.

-   [x] Define `LinearSolver` interface (setup, solve, statistics).
-   [x] Implement `EigenLinearSolver` (wrapping `Eigen::SparseLU`, `Eigen::ConjugateGradient`, and `Eigen::BiCGSTAB`).
-   [x] Add `SolverOptions` struct (rel_tol, abs_tol, max_iter).

### Milestone 3: FSILS Backend Integration (Native Optional)
**Goal:** Enable an in-tree MPI-capable backend that can be swapped in like third-party packages.

-   [x] Add `FSILS` backend wrappers (matrix/vector/solver).
-   [x] Implement setup translation from FE sparsity patterns to FSILS `FSILS_lhsType` (structure) and value storage.
-   [x] Provide a `GlobalSystemView` insertion path (CSR lookup; index-map optimization optional).
-   [x] Mirror solver diagnostics into a backend-agnostic `SolverReport` (iterations, norms, success flag).
-   [x] Implement `FsilsVector::updateGhosts()` for owner→ghost synchronization.

### Milestone 4: PETSc Integration (Parallel / External Solver Hub)
**Goal:** Enable MPI runs and access to PETSc-enabled third-party solvers.

-   [x] Add `PETSc` backend class wrappers.
-   [x] Implement `PetscVector` (wrapping `Vec`) and `PetscMatrix` (wrapping `Mat`).
-   [x] Use ghosted vectors + `updateGhosts()` (`VecCreateGhost` + `VecGhostUpdate`) when a distributed sparsity pattern is available.
-   [x] Ensure `GlobalSystemView` works with PETSc's `MatSetValues`.
-   [x] Implement `PetscLinearSolver` (wrapping `KSP`).
-   [x] Add a safe “options pass-through” mechanism (prefix + key/value list) for advanced PETSc configuration (including PETSc-enabled external solvers).
-   [x] Add `GMRES`/`FGMRES` method mapping and `AMG`/`ILU` preconditioners.
-   [x] Support `BlockSchur` (saddle point) using `PCFIELDSPLIT` (Schur) on `BlockMatrix` / `BlockVector`.
-   [x] Add pass-through via `petsc_options_prefix` + `KSPSetFromOptions()` so command-line options can override FE defaults.

### Milestone 5: Trilinos Integration
**Goal:** Provide an alternative scalable stack (Belos/Ifpack2/MueLu/Amesos2), including potential GPU-focused paths.

-   [x] Add `Trilinos` backend wrappers.
-   [x] Implement solver/preconditioner mapping plus pass-through options.
-   [x] Implement `GMRES` mapping (Belos `PseudoBlockGmres`) and `ILU`/`AMG` preconditioners (Ifpack2 / MueLu when available).
-   [x] Implement overlap vectors + `updateGhosts()` via `Tpetra::Import` when a distributed sparsity pattern is available.
-   [x] Add `trilinos_xml_file` pass-through for user-provided `Teuchos::ParameterList` configuration.

### Milestone 6: Advanced Features (Block Systems)
**Goal:** Support the "Multi-field + Rectangular" requirement from Systems.

-   [x] Add `BlockMatrix` and `BlockVector` abstractions (collections of generic objects).
-   [x] Implement "Nest" matrix support in PETSc (`MatCreateNest`) for monolithic fields.
-   [x] Expose field-split preconditioner options (PETSc `PCFIELDSPLIT`).
-   [x] Rename “NS solver” concept to **Block Schur / Saddle Point** in docs and options.
-   [x] FSILS: support `dof>1` and map generic `BlockSchur` to `LS_TYPE_NS` (FSILS NS solver).

---

## Integration with FE/Systems

The `Systems` module will own `std::unique_ptr<Backends::GenericMatrix>` and request storage allocation during setup:

```cpp
// Systems/SystemSetup.cpp
void FESystem::allocateStorage() {
    auto factory = Backends::BackendFactory::createInstance(opts.backend_type);
    
    // Allocate matrix using the sparsity pattern
    matrix_ = factory->createMatrix(sparsity_pattern_);
    rhs_ = factory->createVector(dof_handler_.getNumDofs());
    solution_ = factory->createVector(dof_handler_.getNumDofs());
}

// Systems/SystemAssembly.cpp
void FESystem::assemble(...) {
    // Create the lightweight view for the assembler
    auto view = matrix_->createAssemblyView(); 
    assembler_->assemble(..., &view, ...);
    matrix_->finalizeAssembly();
}
```

---

## References (design + solver ecosystem)

- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- S. Balay et al., *PETSc Users Manual*, Argonne National Laboratory, ANL-95/11.
- M. A. Heroux et al., “An overview of the Trilinos project,” *ACM Trans. Math. Softw.*, 31(3), 2005. DOI: 10.1145/1089014.1089021.
- R. D. Falgout and U. M. Yang, “hypre: A Library of High Performance Preconditioners,” in *Computational Science — ICCS 2002*, LNCS 2331, 2002. DOI: 10.1007/3-540-47789-6_66.
- D. Arndt et al., “The deal.II finite element library: Design, features, and insights,” *Computers & Mathematics with Applications*, 81, 2021. DOI: 10.1016/j.camwa.2020.02.022.
- R. Anderson et al., “MFEM: A Modular Finite Element Methods Library,” *Computers & Mathematics with Applications*, 81, 2021. DOI: 10.1016/j.camwa.2020.06.009.
- A. Logg, K.-A. Mardal, G. Wells (eds.), *Automated Solution of Differential Equations by the Finite Element Method: The FEniCS Book*, Springer, 2012.
- B. S. Kirk, J. W. Peterson, R. H. Stogner, and G. F. Carey, “libMesh: a C++ library for parallel adaptive mesh refinement/coarsening simulations,” *Engineering with Computers*, 22, 2006. DOI: 10.1007/s00366-006-0049-3.
