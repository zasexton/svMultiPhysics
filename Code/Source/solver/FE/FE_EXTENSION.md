# FE Library Extension Checklist

This document outlines physics-agnostic extensions needed in the FE library to support advanced multi-field formulations including interface-coupled problems, multi-material domains, and discontinuous coefficient handling.

---

## 1. Interface Assembly Integration

The `InterfaceMesh` infrastructure exists but is not fully connected to the Forms assembly pipeline.

- [x] **InterfaceMesh-to-Forms Integration**
  - [x] Create `InterfaceAssembler` that iterates over `InterfaceMesh` faces and assembles interface terms
  - [x] Support assembly of jump/average expressions from `DGOperators` on extracted interface surfaces
  - [x] Provide face-local quadrature rules for interface integration (consistent with volume element order)
  - [x] Handle orientation-aware normal vectors for consistent flux direction

- [x] **Interface Kernel Templates**
  - [x] Assemble `[u]`/`{u}` terms across interfaces via Forms: `jump(u)`, `avg(u)` + `.dI(marker)` (compiled by `forms::FormKernel`)
  - [x] Assemble flux-based interface couplings via Forms on `.dI(marker)` (e.g., `-∫ ⟨{σ}·n⟩ [[v]] ds`)
  - [x] Support weighted averages for discontinuous coefficients (harmonic/arithmetic) (`forms::weightedAverage`, `forms::harmonicAverage`, `avg(...)`)
  - [x] Template for Nitsche-type interface conditions with automatic penalty computation (`Forms/InterfaceConditions.h`)

- [x] **Parent Cell Access**
  - [x] Ensure `InterfaceMesh` provides efficient access to both parent cells (+ and - sides) (`InterfaceMesh::volume_cell_minus/plus`)
  - [x] Support evaluation of volume-based fields at interface quadrature points from both sides
  - [x] Handle boundary faces (single parent) vs interior faces (two parents) uniformly (`InterfaceMesh::is_boundary_face`)

---

## 2. Discontinuous Galerkin Completion

DG operators exist in the vocabulary but end-to-end DG assembly needs completion.

- [x] **Interior Face Loop Infrastructure**
  - [x] Implement interior-face iteration (`assembly::IMeshAccess::forEachInteriorFace`)
  - [x] Integrate interior-face couplings into operator sparsity generation (implemented in `systems::FESystem::setup`)
  - [x] Support MPI-parallel interior face assembly (ghost cell handling)

- [x] **DG Form Integration**
  - [x] Interior-face integrals use `.dS()` and trigger interior face assembly
  - [x] DG forms compile into kernels that assemble cell + boundary + interior-face contributions (`forms::FormKernel`)
  - [x] Support generic upwind flux assembly for advective terms (`forms::upwindValue`)

- [x] **Penalty Parameter Automation**
  - [x] Compute interior penalty coefficient from element size and polynomial order (`forms::interiorPenaltyCoefficient`)
  - [x] Support user-specified penalty scaling factor (gamma multiplier via `eta`)
  - [x] Handle anisotropic meshes with directional h estimation (`forms::hNormal`, `2|K|/|F|`)

---

## 3. Coupled Boundary Condition Jacobians

The coupled BC framework uses lagged/explicit evaluation. Full implicit coupling requires Jacobian contributions.

- [x] **Boundary Functional Derivatives**
  - [x] Implement `dQ/du` computation for `BoundaryFunctional` (e.g., `Q = ∫_Γ u·n ds`)
  - [x] Support rank-1 Jacobian updates: `J += ∂R/∂Q ⊗ ∂Q/∂u`
  - [x] Handle multiple boundary functionals with correct assembly ordering

- [x] **Auxiliary State Jacobian Coupling**
- [x] Account for `∂X/∂Q` in coupled Jacobians via AD-based auxiliary sensitivities (`systems::CoupledBoundaryManager::computeAuxiliarySensitivityForIntegrals`)
  - [x] Support full Newton iteration for eliminated 0D state (reduced Jacobian with chain rule in `systems::assembleOperator`)
  - [x] Use implicit block elimination (Schur complement) by keeping auxiliary state out of the global DOF vector and injecting the reduced coupling into `J`

- [x] **Time-Dependent Coupled BCs**
  - [x] Support implicit time integration of auxiliary ODEs (`systems::ODEIntegrator`)
  - [x] Provide time-stepping hooks for auxiliary state update within nonlinear solve (`systems::CoupledBoundaryManager::prepareForAssembly`)
  - [x] Handle multi-stage time integration methods consistently

---

## 4. Coefficient Discontinuity Support

Multi-material and multi-region problems require discontinuous coefficients.

- [x] **Region-Dependent Coefficients**
  - [x] Support coefficient expressions of the form `k(x) = k₁·χ₁(x) + k₂·χ₂(x)`
  - [x] Implement region indicator concept in Forms vocabulary (`domainId()`, `regionIndicator(domain_id)`)
  - [x] Allow smooth (regularized) and sharp (discontinuous) indicator functions (via `clamp`, `heaviside`, `indicator`)

- [ ] **Per-Region Material Assignment**
  - [ ] Extend `MaterialStateProvider` to support region-based material lookup
  - [ ] Handle element-wise constant properties efficiently (no quadrature-point variation)
  - [ ] Support property evaluation from auxiliary field values

- [x] **Interface Coefficient Averaging**
  - [x] Implement harmonic averaging for coefficients at interfaces (`forms::harmonicAverage`)
  - [x] Implement arithmetic averaging as alternative (`avg(k)`)
  - [x] Support custom weighting functions (`forms::weightedAverage`)

---

## 5. Geometric Differential Operators

Interface and surface problems require geometric quantities computed from fields.

- [ ] **Curvature Computation**
  - [x] Implement mean curvature from scalar field: `κ = ∇·(∇φ/|∇φ|)` (`forms::meanCurvatureFromLevelSet`)
  - [x] Support smoothed/regularized gradient magnitude to avoid division by zero (`forms::safeNormalize`)
  - [ ] Provide L2 projection of curvature to finite element space

- [ ] **Unit Normal from Scalar Field**
  - [x] Compute unit normal from gradient: `n = ∇φ/|∇φ|` (`forms::unitNormalFromLevelSet`)
  - [ ] Support extension of normal field away from zero-level set
  - [x] Handle degenerate cases (vanishing gradient) gracefully (`forms::safeNormalize` with `eps`)

- [x] **Surface Differential Operators**
  - [x] Implement surface gradient: `∇_s f = ∇f - (∇f·n)n` (`forms::surfaceGradient`)
  - [x] Implement surface divergence for tangential vector fields (`forms::surfaceDivergence`)
  - [x] Support surface Laplacian (Laplace-Beltrami operator) (`forms::surfaceLaplacian`)

---

## 6. Cut-Cell and Embedded Interface Methods

Sharp interface treatment for problems with embedded boundaries.

- [ ] **Ghost DOF Framework**
  - [ ] Implement ghost DOF identification near interfaces
  - [ ] Support extrapolation of fields across interface for ghost values
  - [ ] Handle generic jump conditions: `[u] = g`, `[∂u/∂n] = h` at interface

- [ ] **Cut-Cell Integration**
  - [ ] Support integration over partial elements cut by interface
  - [ ] Implement sub-triangulation or moment-fitting for cut-cell quadrature
  - [ ] Handle small cut-cell stability (cell merging or special treatment)

- [ ] **Regularized Delta Functions**
  - [ ] Implement regularized delta function kernels (1D, 2D, 3D)
  - [ ] Support spreading operations from interface to volume mesh
  - [ ] Support interpolation operations from volume mesh to interface

---

## 7. Mass Matrix and Assembly Options

Flexible assembly options for various numerical schemes.

- [ ] **Mass Matrix Variants**
  - [ ] Support lumped vs consistent mass matrix options
  - [ ] Implement row-sum lumping with optional correction
  - [ ] Handle mixed continuous/discontinuous spaces consistently

- [ ] **Flux Conservation at Interfaces**
  - [ ] Ensure numerical flux is single-valued at interfaces (no flux double-counting)
  - [ ] Support conservative flux reconstruction for post-processing
  - [ ] Implement local conservation diagnostics per element/face

- [x] **Assembly Mode Selection**
  - [x] Support element-by-element assembly for matrix-free methods (`assembly::MatrixFreeAssembler`)
  - [x] Support pre-assembled sparse matrix mode (`assembly::StandardAssembler`)
  - [x] Allow hybrid approaches (some operators assembled, some matrix-free) (`systems::OperatorBackends`)

---

## 8. Time Integration Framework Extensions

Generic time integration infrastructure for multi-field problems.

- [ ] **Operator Splitting Framework**
  - [ ] Framework for sequential operator splitting (Lie-Trotter, Strang)
  - [ ] Support sequential solve of subproblems with data exchange
  - [ ] Handle splitting error estimation for adaptive time stepping

- [ ] **Multi-Rate Time Stepping**
  - [ ] Support different time step sizes for different fields
  - [ ] Implement subcycling infrastructure for fast components
  - [ ] Ensure data synchronization at coarse time levels

- [ ] **IMEX Time Integration**
  - [ ] Support implicit-explicit splitting of operators
  - [ ] Template for IMEX Runge-Kutta schemes (ARK methods)
  - [ ] Allow user specification of implicit vs explicit operator treatment

---

## 9. Parallel and Performance

Scalability for large-scale interface-coupled problems.

- [ ] **Interface-Aware Load Balancing**
  - [ ] Weight partitioning to account for interface assembly cost
  - [ ] Support dynamic repartitioning as interface evolves
  - [ ] Minimize interface crossings across MPI boundaries

- [ ] **Ghost Exchange Optimization**
  - [ ] Efficient halo exchange for fields with interface-localized updates
  - [ ] Support narrow-band updates (only exchange near interface region)
  - [ ] Overlap communication with computation where possible

- [ ] **Assembly Optimization**
  - [ ] Cache interface quadrature points and weights
  - [ ] Support vectorized/batched interface kernel evaluation
  - [ ] Minimize redundant geometric computations at interfaces

---

## 10. Forms + LLVM JIT Infrastructure for New Physics

This section tracks physics-agnostic extensions needed so the Forms IR, symbolic differentiation, and
LLVM JIT backend can cover advanced constitutive models and frequency-/time-domain formulations
(structural, fluids, electromagnetics, acoustics).

### 10.1 FormExprType Vocabulary Extensions

- [x] **Matrix functions (non element-wise)**
  - [x] Add `FormExprType::{MatrixExponential,MatrixLogarithm,MatrixSqrt,MatrixPower}` in `Forms/FormExpr.h`
  - [x] Add node constructors / printing / invariants in `Forms/FormExpr.cpp`
  - [x] Lower to deterministic IR in `Forms/JIT/KernelIR.cpp` (and TensorIR lowering if applicable)
  - [x] Emit LLVM via versioned ExternalCalls helpers in `Forms/JIT/LLVMGen.cpp`
  - [x] Add differentiation rules in `Forms/SymbolicDifferentiation.cpp` (and `Forms/Tensor/TensorDifferentiation.cpp`)
  - [x] Add JIT validation rules (shape + domain constraints, SPD checks where required) in `Forms/JIT/JITValidation.cpp`
  - [x] Decide initial support scope (recommended: 2×2 and 3×3; `log/sqrt` restricted to SPD matrices)

- [x] **Regularized / smooth approximations**
  - [x] Add `FormExprType::{SmoothHeaviside,SmoothAbsoluteValue,SmoothMin,SmoothMax,SmoothSign}` in `Forms/FormExpr.h`
  - [x] Decide whether these are first-class nodes vs `Forms/Vocabulary.h` combinators
  - [x] If combinators: add JIT peephole patterns in `Forms/JIT/KernelIR.cpp` and/or `Forms/JIT/LLVMGen.cpp`
        (not applicable; implemented as first-class nodes)
  - [x] `clamp(x,lo,hi)` exists as a combinator in `Forms/Vocabulary.h` (currently `min(max(x,lo),hi)`)

- [x] **General eigendecomposition operators**
  - [x] Add `FormExprType::{Eigenvalue,SymmetricEigenvector,SpectralDecomposition}` in `Forms/FormExpr.h`
  - [x] Add lowering rules in `Forms/JIT/KernelIR.cpp` (or force ExternalCalls lowering for determinism)
  - [x] Add LLVM emission in `Forms/JIT/LLVMGen.cpp` via ExternalCalls eigensolver helpers
  - [x] Add symbolic derivatives:
    - [x] extend eigenvalue rules (beyond existing symmetric eigenvalue directional derivatives)
    - [x] add eigenvector/spectral decomposition derivatives (including repeated-eigenvalue handling strategy)
  - [x] Add validation rules (symmetric vs general, conditioning limits, repeated eigenvalues) in `Forms/JIT/JITValidation.cpp`

- [x] **Convolution / history integral operators**
  - [x] Add `FormExprType::{HistoryWeightedSum,HistoryConvolution}` in `Forms/FormExpr.h`
  - [x] Define semantics for history terminals (weights slot, history depth, which solution stream)
  - [x] Extend JIT ABI to pass history weights and history coefficient pointers (see 10.2) and lower in `Forms/JIT/KernelIR.cpp`
  - [x] Add LLVM emission loops over history in `Forms/JIT/LLVMGen.cpp`

- [x] **Higher-order time derivatives**
  - [x] Generalize `TimeDerivative(order)` to order `N` (order > 2) in `Forms/FormExpr.*`
  - [x] Extend ABI packing for additional time discretization coefficients (see 10.2)
  - [x] Extend LLVM emission (`termWeight`, dt coefficient loads) in `Forms/JIT/LLVMGen.cpp` for order > 2

### 10.2 KernelArgs ABI Extensions (JIT)

- [x] **History access ABI**
  - [x] Extend `Assembly/JIT/KernelArgs.h` side args with:
    - [x] `history_weights` pointer (or a general pointer+stride descriptor)
    - [x] `num_history_steps`
    - [x] `history_solution_coefficients` (array-of-pointers or packed 2D layout)
  - [x] Update packers (`Assembly/JIT/KernelArgs.*`) and bump the KernelArgs ABI version constant
  - [x] Update `Forms/JIT/LLVMGen.cpp` to load new fields and preserve strict alignment checks

- [x] **Higher-order dt coefficients ABI**
  - [x] Extend `Assembly/JIT/KernelArgs.h` to provide dt coefficients for order ≥ 3
        (either `dt3_coeff0`, etc., or a general array)
  - [x] Update packers and bump ABI version (if layout changes)
  - [x] Update `Forms/JIT/LLVMGen.cpp` to consume the generalized coefficients

- [x] **Complex-valued field support**
  - [x] Decide strategy:
    - [x] keep current real/imag block splitting and add LLVMGen fusions (10.3), or
    - [ ] add native complex pointers/flags to KernelArgs and emit complex arithmetic directly
  - [x] Add JIT test coverage for real/imag block splitting (`FE/Tests/Unit/Forms/test_Complex.cpp`)
  - [ ] If ABI extension (future; only needed for native complex ABI):
    - [ ] add complex solution and basis pointers in `Assembly/JIT/KernelArgs.h`
    - [ ] update packers + bump ABI version
    - [ ] extend `Forms/JIT/LLVMGen.cpp` to load/compute complex values (including conjugation and Hermitian products)

- [ ] **Material state descriptors (optional)**
  - [ ] Decide whether advanced structured material state should remain ExternalCalls-based
        vs adding a descriptor pointer/counts to `Assembly/JIT/KernelArgs.h`
  - [ ] If ABI extension: update packers + bump ABI version, and update `Forms/JIT/LLVMGen.cpp`

### 10.3 LLVMGen / Codegen Extensions

- [x] **Matrix function emission**
  - [x] Add `emitMatrix{Exp,Log,Sqrt,Pow}` helpers in `Forms/JIT/LLVMGen.cpp`
  - [x] Route emission through versioned ExternalCalls helpers (10.4) for numerical robustness

- [x] **Efficient complex arithmetic emission**
  - [x] If keeping real/imag block splitting: add peephole fusions in `Forms/JIT/LLVMGen.cpp`
        (complex multiply/add, conjugate transpose patterns, Hermitian inner products)
  - [x] If native complex ABI: introduce a `ComplexCodeValue`-style representation in `Forms/JIT/LLVMGen.cpp`
        (not applicable; this project uses real/imag block splitting)

- [x] **Local iteration framework (material-point Newton)**
  - [x] Decide strategy: keep local solves in ExternalCalls only (current approach)
  - [x] If in-kernel: (not applicable; ExternalCalls-only)
    - [x] add while-loop emission + convergence checks in `Forms/JIT/LLVMGen.cpp` (not applicable)
    - [x] add bounded iteration + failure handling (fallback path) in `Forms/JIT/JITKernelWrapper.cpp` (not applicable)
    - [x] ensure temporary storage strategy is safe and performant (stack/reuse limits) (not applicable)

### 10.4 ExternalCalls ABI Extensions (C-ABI helpers)

- [x] **Matrix function helpers (versioned symbols)**
  - [x] Add signatures in `Forms/JIT/ExternalCalls.h`:
    - [x] `svmp_fe_jit_matrix_exp_3x3_v1(const double* A, double* expA)`
    - [x] `svmp_fe_jit_matrix_log_3x3_v1(const double* A, double* logA)`
    - [x] `svmp_fe_jit_matrix_sqrt_3x3_v1(const double* A, double* sqrtA)`
    - [x] `svmp_fe_jit_matrix_pow_3x3_v1(const double* A, double p, double* Ap)`
  - [x] Implement in `Forms/JIT/ExternalCalls.cpp` using a robust algorithm (spectral maps for 2×2/3×3)
  - [x] Add tests validating correctness and failure modes (SPD checks, conditioning) in `FE/Tests/Unit/Forms`

- [x] **Eigendecomposition helpers**
  - [x] Add `svmp_fe_jit_eig_sym_3x3_v1(const double* A, double* eigvals, double* eigvecs)`
  - [x] Add `svmp_fe_jit_eig_general_3x3_v1(...)` with explicit real/imag output ABI
  - [x] Implement in `Forms/JIT/ExternalCalls.cpp` and validate edge cases (repeated eigenvalues)

- [x] **Local Newton helper (optional)**
  - [x] Keep local Newton solves model-owned (ExternalCalls-based constitutive boundary); no generic helper (not applicable)
  - [x] Tests for generic local Newton helper (not applicable)

### 10.5 Symbolic Differentiation Extensions

- [x] **Matrix function derivatives**
  - [x] Add derivative rules in `Forms/SymbolicDifferentiation.cpp` (Fréchet derivatives or an approved approximation)
  - [x] Add tensor-index equivalents in `Forms/Tensor/TensorDifferentiation.cpp`

- [x] **Eigenvalue/eigenvector derivatives**
  - [x] Extend eigenvalue derivatives beyond the current directional derivative support
  - [x] Add eigenvector derivatives and define behavior for repeated eigenvalues (fallback/regularization)

- [x] **Local solve differentiation**
  - [x] If `LocalSolve` becomes a first-class node, add rules to differentiate its residual and form a local Jacobian
        (not applicable; LocalSolve remains ExternalCalls-based and is not a first-class node)

### 10.6 JIT Validation Extensions

- [x] **Matrix input validation**
  - [x] SPD checks for `MatrixLogarithm`/`MatrixSqrt` and shape checks for all matrix functions

- [x] **Complex expression validation**
  - [x] Ensure consistent real/imag pairing and supported operations when complex mode is enabled
        (complex is implemented via real/imag splitting and 2×2 block lifting; JIT validates each real-valued block)

- [x] **Local solve validation**
  - [x] Check well-formedness of `LocalSolve` nodes (tolerance, max iters, residual shape)
        (not applicable; LocalSolve is not a first-class node in Forms IR)

---

## Priority Summary

### High Priority (Core Functionality)
1. InterfaceMesh-to-Forms integration (Section 1)
2. Interior face loop infrastructure (Section 2)
3. Coupled BC Jacobians (Section 3)
4. DG form integration (Section 2)
5. Forms + LLVM JIT new-physics coverage (Section 10)

### Medium Priority (Common Use Cases)
6. Region-dependent coefficients (Section 4)
7. Geometric differential operators (Section 5)
8. Mass matrix variants (Section 7)
9. Penalty parameter automation (Section 2)

### Lower Priority (Advanced Features)
10. Ghost DOF framework (Section 6)
11. Cut-cell integration (Section 6)
12. Operator splitting framework (Section 8)
13. Multi-rate time stepping (Section 8)

---

## References

- `Code/Source/solver/FE/Forms/Vocabulary.h` - Expression builders and DG operators
- `Code/Source/solver/FE/Forms/SymbolicDifferentiation.cpp` - Symbolic derivatives for Forms
- `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h` - KernelArgs ABI (packing + offsets)
- `Code/Source/solver/FE/Forms/JIT/ExternalCalls.h` - JIT-resolved helper ABI
- `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp` - LLVM IR generation backend
- `Code/Source/solver/FE/Forms/BlockForm.h` - Block assembly infrastructure
- `Code/Source/solver/Mesh/Core/InterfaceMesh.h` - Interface mesh representation
- `Code/Source/solver/FE/Spaces/DGOperators.h` - Discontinuous Galerkin operators
- `Code/Source/solver/FE/Sparsity/DGSparsityBuilder.h` - DG sparsity patterns
