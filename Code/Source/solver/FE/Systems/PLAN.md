# FE/Systems Subfolder — Design Plan

## Overview

`FE/Systems` is the **physics-agnostic problem-definition and assembly-orchestration** layer of the FE library. It binds together:

- Mesh traversal (`FE/Assembly::IMeshAccess`)
- Spaces/elements (`FE/Spaces::FunctionSpace`)
- DOF distribution and multi-field bookkeeping (`FE/Dofs::*`)
- Constraints and BC bookkeeping (`FE/Constraints::*`)
- Assembly engines (`FE/Assembly::{Assembler, AssemblyKernel}`)
- Backend-neutral insertion (`FE/Assembly::GlobalSystemView`), with storage owned by `FE/Backends` (or test/dense views)

**Key philosophy:** `Systems` **defines what to assemble and when**; it does not **solve** or implement any specific PDE.

**Note on organization:** All implementation files reside directly within the `Systems/` folder (no subdirectories) to match the module layout of the rest of `FE/`.

---

## Scope of Responsibilities

### What Systems **DOES** (the intended narrow scope)

1. **System Definition Lifecycle**
   - Own the “define → setup → assemble” lifecycle (and invalidation rules when the definition changes).
   - Provide a solver-facing object that represents “the discrete FE system”.

2. **Multi-field Bookkeeping (Physics-agnostic)**
   - Register unknown and auxiliary fields by name/`FieldId`.
   - Maintain field/block metadata for multiphysics (via `dofs::FieldDofMap` / `dofs::BlockDofMap`).
   - Expose subspace/block views for solvers and preconditioners (no solver logic in Systems).

3. **Operator/Residual Assembly Orchestration**
   - Maintain registries of **operators** (residual, Jacobian, mass, preconditioner/auxiliary operators, etc.).
   - Each operator is a collection of `assembly::AssemblyKernel` terms that may contribute on:
     - cells (volume integrals),
     - boundary faces (marker-based integrals),
     - interior faces (DG coupling terms).
   - Provide solver-friendly assembly entry points:
     - `assembleResidual()`, `assembleJacobian()`, `assembleMass()`,
     - plus a general `assemble(operator_tag, outputs, state)` API for extensibility.

4. **Constraints / BC Integration (without “owning physics”)**
   - Own the constraint set (`constraints::AffineConstraints`) and keep it up to date.
   - Accept algebraic constraints via `constraints::Constraint` objects (Dirichlet, periodic, MPC, global pinning).
   - Treat Neumann/Robin/Nitsche-style BCs as **weak boundary terms** assembled through boundary-face kernels (not as algebraic constraints).

5. **State Plumbing for Nonlinear/Transient Problems**
   - Maintain and expose a minimal `SystemState` (time, dt, current/previous solution vectors, parameter registry) so kernels can evaluate nonlinear/time-dependent integrands.
   - Systems does not time-step; it just makes state available for assembly.

### What Systems **DOES NOT DO** (hard boundaries)

- **Solving:** no Newton/GMRES/time integration loops (belongs in a solver layer or application code).
- **Mesh management:** no refinement decisions, partitioning, or mesh generation (belongs in mesh/adaptivity tooling).
- **I/O / visualization:** no file output (belongs in IO tooling).
- **Physics kernels:** no Navier–Stokes/elasticity/etc. implementations (belongs in physics modules that implement `assembly::AssemblyKernel`).

---

## Relationship to Other FE Modules (current codebase alignment)

```
USER / PHYSICS MODULES
  - implement assembly::AssemblyKernel (cell/boundary/interior-face terms)
  - provide material/BC data and kernel parameters
  - own solver loops (Newton, time stepping, coupling strategies)
             │
             ▼
FE/Systems
  - defines fields/spaces, constraints, operator registries
  - builds dofs + sparsity + backend storage
  - calls FE/Assembly to fill matrices/vectors
             │
             ├── FE/Spaces      (FunctionSpace, MixedSpace, DG spaces, ...)
             ├── FE/Dofs        (DofHandler, DofMap, FieldDofMap, BlockDofMap)
             ├── FE/Constraints (AffineConstraints, DirichletBC, PeriodicBC, ...)
             ├── FE/Sparsity    (SparsityBuilder, DGSparsityBuilder, ...)
             └── FE/Assembly    (Assembler, AssemblyKernel, GlobalSystemView)
```

---

## Core Concepts (API vocabulary)

- **Field:** A named unknown/auxiliary FE function with a `FunctionSpace` and (optionally) multiple components.
- **Operator:** A named assembled object (matrix and/or vector) built from one or more **terms**.
- **Term:** One `assembly::AssemblyKernel` applied over a particular integration domain **and associated with the equation and unknown blocks it couples**:
  - `test_field` (which equation block this contributes to),
  - optional `trial_field` (which unknown block this contributes with respect to; required for Jacobian/coupling terms).

  Terms are integrated on one of:
  - **Cell** term (volume integrals),
  - **Boundary** term (boundary faces with a specific marker),
  - **Interior face** term (DG coupling, jump/penalty terms).
  - (Optional) **Subdomain restriction** via a Systems-side filtered mesh view (e.g., a predicate on `cell_id` or material IDs supplied by the application).
- **Constraints:** Algebraic relations among DOFs stored as `constraints::AffineConstraints`.
  - Dirichlet/periodic/MPC are handled here.
  - Neumann/Robin/Nitsche are handled as boundary terms (kernels).
- **SystemState:** Read-only view used by kernels during assembly: time, dt, solution vectors, parameter registry.

---

## Lifecycle (typical solver usage)

1. **Define**
   - Register fields/spaces.
   - Register operators and their kernels (cell/boundary/interior-face).
   - Add algebraic constraints (Dirichlet/periodic/etc.) and weak BC terms (boundary kernels).

2. **Setup**
   - Distribute DOFs (`dofs::DofHandler`) and finalize.
   - Build multi-field maps (`dofs::FieldDofMap` / optional `dofs::BlockDofMap`).
   - Build sparsity patterns per operator (`sparsity::SparsityBuilder`, plus DG/constraint augmentation when needed).
   - Build and close constraints (`constraints::AffineConstraints`); in MPI, ensure consistency with `constraints::ParallelConstraints` before `close()`.
   - Allocate backend storage (matrix/vector objects) and wrap them in `assembly::GlobalSystemView`.
   - Configure chosen assembler(s): `Assembler::setDofHandler`, `setConstraints`, `setSparsityPattern`, `setOptions`.

3. **Assemble / Solve Loop (owned by solver layer)**
   - Update `SystemState` (time, dt, solution, parameters).
   - Call `assembleResidual()`, `assembleJacobian()`, etc.
   - External solver solves and updates solution.
   - Enforce constraints on the solution vector if required (`AffineConstraints::distribute`).

---

## API Plan (comprehensive but minimal)

### Namespaces and naming

- Use `namespace svmp::FE::systems`.
- Prefer explicit “operator tags” for extensibility (`"residual"`, `"jacobian"`, `"mass"`, `"preconditioner"`, ...).

### 1) `systems::FESystem` — solver-facing system container

```cpp
namespace svmp::FE::systems {

using OperatorTag = std::string;
using BoundaryId  = int; // marker id from assembly::IMeshAccess

struct FieldSpec {
    std::string name;   // e.g. "u", "pressure", "temperature"
    std::shared_ptr<const spaces::FunctionSpace> space;
    int components{1};  // metadata for FieldDofMap/BlockDofMap
};

struct SetupOptions {
    dofs::DofDistributionOptions dof_options{};
    assembly::AssemblyOptions assembly_options{};
    sparsity::SparsityBuildOptions sparsity_options{};

    // Assembly strategy selection (default: StandardAssembler).
    // Phase 1: enum/string; later: factory injection for custom assemblers.
    std::string assembler_name{"StandardAssembler"};

    // Multi-field coupling hints for sparsity/preallocation
    sparsity::CouplingMode coupling_mode{sparsity::CouplingMode::Full};
    std::vector<sparsity::FieldCoupling> custom_couplings{};

    // If true, Systems supplies AffineConstraints to the assembler so
    // constraints are handled during local-to-global distribution.
    bool use_constraints_in_assembly{true};
};

struct SystemStateView {
    double time{0.0};
    double dt{0.0};

    // Current (and optional historical) solution vectors; storage is backend-defined.
    // Phase 1: use spans over host vectors; later: backend vector views.
    std::span<const Real> u;
    std::span<const Real> u_prev;
    std::span<const Real> u_prev2;

    // Parameter registry (minimal; extensible)
    std::function<std::optional<Real>(std::string_view)> getRealParam;
};

struct AssemblyRequest {
    OperatorTag op;                // which operator to assemble
    bool want_matrix{false};       // assemble matrix contributions
    bool want_vector{false};       // assemble vector contributions
    bool zero_outputs{true};       // clear output views before assembly
    bool assemble_boundary_terms{true};
    bool assemble_interior_face_terms{true};
};

class FESystem {
public:
    explicit FESystem(std::shared_ptr<const assembly::IMeshAccess> mesh);

    // ---- Definition phase ----
    FieldId addField(FieldSpec spec);
    void addConstraint(std::unique_ptr<constraints::Constraint> c);

    // Operator registry
    void addOperator(OperatorTag name);

    // Register operator terms
    void addCellKernel(OperatorTag op, FieldId field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel); // test=trial=field
    void addCellKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                       std::shared_ptr<assembly::AssemblyKernel> kernel); // coupling/rectangular

    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel); // test=trial=field
    void addBoundaryKernel(OperatorTag op, BoundaryId boundary, FieldId test_field, FieldId trial_field,
                           std::shared_ptr<assembly::AssemblyKernel> kernel);

    void addInteriorFaceKernel(OperatorTag op, FieldId field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel); // test=trial=field
    void addInteriorFaceKernel(OperatorTag op, FieldId test_field, FieldId trial_field,
                               std::shared_ptr<assembly::AssemblyKernel> kernel);

    // ---- Setup phase ----
    void setup(const SetupOptions& opts = {});

    // ---- Assembly phase (solver calls) ----
    assembly::AssemblyResult assemble(
        const AssemblyRequest& req,
        const SystemStateView& state,
        assembly::GlobalSystemView* matrix_out,
        assembly::GlobalSystemView* vector_out);

    // Convenience wrappers (common FEM solver calls)
    assembly::AssemblyResult assembleResidual(
        const SystemStateView& state,
        assembly::GlobalSystemView& rhs_out);
    assembly::AssemblyResult assembleJacobian(
        const SystemStateView& state,
        assembly::GlobalSystemView& jac_out);
    assembly::AssemblyResult assembleMass(
        const SystemStateView& state,
        assembly::GlobalSystemView& mass_out);

    // ---- Accessors for solvers/preconditioners ----
    const dofs::DofHandler& dofHandler() const;
    const dofs::FieldDofMap& fieldMap() const;
    const dofs::BlockDofMap* blockMap() const; // optional (nullptr if unused)
    const constraints::AffineConstraints& constraints() const;
    const sparsity::SparsityPattern& sparsity(const OperatorTag& op) const;
};

} // namespace svmp::FE::systems
```

**Notes for implementers**

- Kernels remain in **physics code** and implement `assembly::AssemblyKernel` (`computeCell`, `computeBoundaryFace`, `computeInteriorFace`).
- Systems is responsible for mapping “operator terms” onto the appropriate `Assembler` calls:
  - cells: `Assembler::assembleMatrix/assembleVector/assembleBoth`
  - boundary faces: `Assembler::assembleBoundaryFaces(mesh, marker, ...)`
  - interior faces: `Assembler::assembleInteriorFaces(...)`
- In multiphysics settings, each term must carry `(test_field, trial_field)` so Systems can:
  - choose the right test/trial spaces,
  - choose the right row/col DOF maps (or a future multi-field DOF handler),
  - and place the result into the correct global blocks.
- For nonlinear/transient kernels, Systems must ensure per-cell solution coefficients are made available to kernels (likely via a small extension in Assembly so the context can be populated from `state.u` and the cell DOF list).
- When both matrix and vector are requested for the same operator, Systems should prefer a single pass (`Assembler::assembleBoth`) to avoid duplicated geometry/basis work.
- Systems should combine the request (`want_matrix`, `want_vector`) with kernel capabilities (`isMatrixOnly()`, `isVectorOnly()`) to pick the most efficient assembly path.

### 2) Operator / Term registry (enables multiphysics flexibility)

Minimum internal representation:

- `OperatorDefinition`
  - `std::vector<CellTerm>` where `CellTerm = {test_field, trial_field, kernel}`
  - `std::vector<BoundaryTerm>` where `BoundaryTerm = {boundary_id, test_field, trial_field, kernel}`
  - `std::vector<InteriorFaceTerm>` where `InteriorFaceTerm = {test_field, trial_field, kernel}`
  - optional metadata: symmetry hints, “matrix-only/vector-only” expectations, block coupling hints

This supports:

- **Multi-physics** by registering multiple kernels per operator (each kernel may represent one coupling term).
- **DG methods** via interior face kernels (jump/penalty/upwind, etc.).
- **Multiple operators** for advanced solvers (mass, preconditioner operators, and other solver-specific operators).

#### Why the mass operator is a first-class citizen

Even in a completely physics-agnostic Systems layer, it is valuable to treat the **mass matrix** as a named operator from day one:

- It is the **discrete inner product / Gram matrix** induced by the chosen space and basis:
  `M_ij = ∫_Ω φ_i φ_j dx` (and analogs for vector/mixed spaces).
- It is required by broad classes of FE workflows that Systems should support generically:
  - **Time-dependent semi-discretizations** where the time derivative term produces `M * u_t` (leading to ODE/DAE forms like `M u' = F(u,t)`).
  - **Projection and transfer operations** (e.g., L² projection for initial conditions, remapping between spaces, computing norms/energies consistently).
  - **Generalized eigenproblems** of the form `K x = λ M x` (modal analysis / stability analysis).
- It is **space- and discretization-defined**, not PDE-defined, so it is a clean Systems responsibility and a good “baseline operator” to validate assembly/backends/constraints integration.

### 3) Constraints and boundary conditions

- Algebraic constraints: stored as `constraints::AffineConstraints` constructed from `constraints::Constraint` objects.
  - supports Dirichlet, periodic, MPC, global pinning/zero-mean (depending on chosen strategy).
  - supports efficient time-dependent Dirichlet updates via `AffineConstraints::updateInhomogeneity(...)` without rebuilding the full constraint structure.
  - Systems should provide *convenience entry points* that leverage `constraints::ConstraintTools` + `dofs::DofTools` so users can specify constraints by boundary/subdomain markers (instead of manually building DOF lists).
- Weak BCs:
  - Neumann/Robin/Nitsche are integrated as boundary-face kernels registered on boundary markers.
  - `constraints::{NeumannBC, RobinBC}` may be used as *data containers*, but the actual assembly contribution is a kernel.
  - Strong vs weak enforcement is an explicit user choice: either add a Dirichlet constraint (strong) or register a Nitsche kernel (weak), but not both for the same DOFs.

### 4) “Agnostic but practical” state and parameter handling

To stay physics-agnostic but multiphysics-ready, Systems should:

- provide time/dt and solution vectors to kernels,
- provide a minimal parameter registry (scalar parameters by name),
- avoid prescribing how materials are represented (kernels can capture arbitrary read-only shared data).

### 5) Operator Backends (extensible; leverage existing FE infrastructure)

The primary Systems path is **assembled operators** (matrix/vector) via `assembly::Assembler` + `assembly::GlobalSystemView`. To “fully support FE infrastructure”, Systems should also be able to register and expose:

- **Matrix-free operators** via `assembly::MatrixFreeAssembler` and `assembly::IMatrixFreeKernel` (iterative solvers, high-order, GPU-oriented paths).
- **Scalar functionals / QoIs** via `assembly::FunctionalAssembler` and `assembly::FunctionalKernel` (norms, fluxes, boundary integrals, goal functionals).

---

## Examples (solver-facing usage)

### Example A: Single-field linear system (system matrix + RHS + strong Dirichlet)

1. Define one field `u` and add a cell kernel (a custom `assembly::AssemblyKernel` providing local matrix/vector contributions).
2. Add a `constraints::DirichletBC` (algebraic) for a boundary marker.
3. `setup()`, then assemble:
   - system matrix (Jacobian for nonlinear problems) into `matrix_out`
   - residual/RHS into `rhs_out`

### Example B: Mixed/multiphysics (block structure for solvers)

1. Define fields `u` and `p` (potentially different spaces/orders).
2. Register separate kernels for each coupling term (e.g., `A_uu`, `B_up`, `B_pu`).
3. Build a `dofs::BlockDofMap` so solvers can extract blocks for Schur complements.

### Example C: DG method (interior face + boundary terms)

1. Use a DG space for the field(s).
2. Register interior-face kernels for flux/jump terms.
3. Register boundary-face kernels for weak boundary/interface terms and penalty terms.

---

## File Structure (planned; flat directory)

All files reside directly in `Code/Source/solver/FE/Systems/`.

- `FESystem.h/cpp` — solver-facing container + lifecycle.
- `OperatorRegistry.h/cpp` — operator/tag management and term storage.
- `SystemState.h` — `SystemStateView` and parameter registry helpers.
- `SystemSetup.h/cpp` — wiring for dofs + sparsity + backend allocation.
- `SystemAssembly.h/cpp` — assembly dispatch (cells/boundary/interior-face) and bookkeeping.

**Deliberate non-goals for this folder**

- No new “BilinearForm/LinearForm” hierarchy: physics extension point is `assembly::AssemblyKernel`.
- No boundary “manager” that re-implements constraints: use `FE/Constraints`.

---

## References / Inspirations

### Gold-standard FEM texts (selected)

- Ciarlet — *The Finite Element Method for Elliptic Problems*.
- Strang & Fix — *An Analysis of the Finite Element Method*.
- Zienkiewicz & Taylor — *The Finite Element Method* (multiple volumes/editions).
- Hughes — *The Finite Element Method: Linear Static and Dynamic Finite Element Analysis*.
- Johnson — *Numerical Solution of Partial Differential Equations by the Finite Element Method*.
- Brenner & Scott — *The Mathematical Theory of Finite Element Methods*.
- Ern & Guermond — *Theory and Practice of Finite Elements*.
- Boffi, Brezzi, Fortin — *Mixed Finite Element Methods and Applications* (saddle-point systems).
- Cockburn, Karniadakis, Shu (eds.) — *Discontinuous Galerkin Methods: Theory, Computation and Applications* (DG flux/interface terms).
- Hesthaven & Warburton — *Nodal Discontinuous Galerkin Methods* (high-order DG practice).

### Key method papers (selected)

- Nitsche — weak enforcement of Dirichlet conditions (Nitsche’s method): “Über ein Variationsprinzip zur Lösung von Dirichlet-Problemen …” (1971).
- Brooks & Hughes — SUPG stabilization: “Streamline Upwind/Petrov-Galerkin formulations for convection dominated flows …” (1982).
- Arnold — interior penalty methods: “An interior penalty finite element method with discontinuous elements.” (1982).
- Arnold, Brezzi, Cockburn, Marini — unified DG analysis: “Unified analysis of discontinuous Galerkin methods for elliptic problems.” (2002).

### Well-established FE libraries (selected)

- deal.II — Bangerth, Hartmann, Kanschat — “deal.II — a general-purpose object oriented finite element library.” (2007); and Bangerth, Heister, Harten — “deal.II: A general-purpose object-oriented finite element library.” (2021).
- MFEM — Anderson et al. — “MFEM: A modular finite element methods library.” (Computers & Mathematics with Applications).
- FEniCS — Logg, Mardal, Wells (eds.) — *Automated Solution of Differential Equations by the Finite Element Method* (Springer; “The FEniCS Book”).
- libMesh — Kirk, Peterson, Stogner, Carey — “libMesh: a C++ library for parallel adaptive mesh refinement/coarsening simulations.” (Engineering with Computers).
- Firedrake — Rathgeber et al. — “Firedrake: automating the finite element method by composing abstractions.” (ACM Transactions on Mathematical Software).
- DUNE — Bastian et al. — “DUNE: A modular toolbox for solving PDEs with grid-based methods.” (Computing).

### Solver/backends commonly paired with FEM

- PETSc — Balay et al. — PETSc users manual / PETSc library papers (KSP/SNES, Mat/Vec, assembly backends).
- Trilinos — Heroux et al. — “An overview of the Trilinos project.” (ACM Transactions on Mathematical Software) and related Trilinos papers.

---

## Accommodating a Future “Mathematical Notation” Layer

It is expected that a future subfolder (e.g., `FE/Forms` or `FE/DSL`) will provide a more mathematical, weak-form-oriented way to specify problems (UFL-like, MFEM-style integrator composition, or a small C++ EDSL).

To accommodate that cleanly, `FE/Systems` should remain the **stable compilation target**:

- The math layer compiles to a **Systems-level intermediate representation**:
  - fields (name, space, components),
  - operators (tags),
  - terms (domain + boundary marker + `(test_field, trial_field)` + kernel object),
  - constraints specifications (strong constraints and/or weak enforcement kernels),
  - parameter names expected at assembly time.
- Systems then handles:
  - DOF distribution + constraints closure (and parallel consistency),
  - sparsity construction + backend allocation,
  - dispatch to `FE/Assembly` for assembly and/or to other operator backends (future).

This separation keeps `FE/Systems` physics-agnostic while enabling a higher-level “math-first” front-end without leaking PDE-specific logic into Systems.
