# FE/Forms/JIT — LLVM OrcJIT Implementation Checklist (Assembly Acceleration)

This is a **concrete, code-tracked checklist** for completing the LLVM OrcJIT backend described in `LLVM_JIT_IMPLEMENTATION_PLAN.md`, targeting fast element assembly for Forms (`FormExpr` → `FormIR` → `AssemblyKernel`).

Legend:
- `[x]` already implemented in this repo
- `[ ]` remaining work

## 0. “Done” Definition (what “complete JIT backend” means)

- [ ] `SymbolicOptions::jit.enable = true` produces JIT-compiled `assembly::AssemblyKernel` instances (cell, boundary-face, interior-face, interface-face) with interpreter fallback.
- [ ] JIT kernels accept the **stable POD ABI** in `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h` (or its next version) and never depend on C++ object layouts.
- [ ] JIT kernels match interpreter results (within tolerance) for supported Forms vocabulary, including time-derivative weights, coupled slots, and material state.
- [ ] Kernel caching works (in-memory; optional on-disk) and is safe under MPI + OpenMP usage.
- [ ] Debuggability exists (optional IR/asm dump, perf/gdb listeners) and failures never crash simulations (graceful fallback).

## 1. Existing Groundwork (already in tree)

- [x] JIT compatibility pre-checks: `Code/Source/solver/FE/Forms/JIT/JITValidation.h` / `.cpp`
- [x] Deterministic, hashable pre-LLVM IR: `Code/Source/solver/FE/Forms/JIT/KernelIR.h` / `.cpp`
- [x] Versioned kernel ABI + packers: `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
- [x] Slot-based terminals for JIT-friendly parameter/coupled access: `FormExprType::{ParameterRef,BoundaryIntegralRef,AuxiliaryStateRef,...}` (`Code/Source/solver/FE/Forms/FormExpr.h`)

## 2. Build / Dependency Work

- [x] Decide the supported LLVM version range and platform matrix (Linux/macOS/Windows; Clang/GCC/MSVC).
- [x] Add a build option (recommended) e.g. `FE_ENABLE_LLVM_JIT` to allow building FE without LLVM.
- [x] Update `Code/Source/solver/FE/CMakeLists.txt` to:
  - [x] `find_package(LLVM CONFIG REQUIRED)` when `FE_ENABLE_LLVM_JIT=ON`
  - [x] link required LLVM components (at minimum OrcJIT + native target + support + passes)
  - [x] add include dirs + definitions (`LLVM_ENABLE_ASSERTIONS` handling, RTTI if needed)
  - [x] compile+link the new JIT sources (Engine/Gen/Compiler/Kernel wrappers)
- [x] Ensure CI/build docs mention LLVM requirement and how to enable the JIT backend.

## 3. Kernel ABI Finalization (critical path)

The JIT backend should treat `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h` as the canonical ABI. The current `*V1` structs are close, but still missing data needed for full Forms coverage.

- [x] Audit Forms terminals/operators vs `KernelSideArgsV1` fields; add missing fields and bump ABI version if layout changes:
  - [x] `CellDomainId` support (needs `cell_domain_id` in side args)
  - [x] interface-face marker support (needs `interface_marker` somewhere in args)
  - [x] H(curl)/H(div) basis table support (vector basis values + curl/div tables)
  - [x] Multi-field discrete/state field support (values/derivatives for `DiscreteField`/`StateField`, plus optional history)
- [x] Add explicit alignment guarantees for packed pointers (perf-critical):
  - [x] decide required alignment (32/64 bytes)
  - [x] enforce/validate in assembler-side allocation paths (basis/geometry scratch, material state, etc.)
  - [x] add debug assertions in packers when `jit.enable` is on
- [x] Define (and implement) “kernel output contract”:
  - [x] outputs are always zeroed by caller vs kernel must write/overwrite (pick one; document in ABI)
  - [x] matrix storage order is row-major and consistent everywhere (already the convention)

## 4. JIT Engine Wrapper (OrcJIT runtime)

Create a small wrapper around LLVM OrcJIT to isolate LLVM headers from most of FE/Forms.

- [x] Add `Code/Source/solver/FE/Forms/JIT/JITEngine.h`
- [x] Add `Code/Source/solver/FE/Forms/JIT/JITEngine.cpp`
- [x] Implement `JITEngine` responsibilities:
  - [x] initialize native target (`InitializeNativeTarget*`) exactly once per process
  - [x] create/configure `llvm::orc::LLJIT` (target triple, data layout)
  - [x] support module submission (`addModule(ThreadSafeModule)`)
  - [x] support symbol lookup (`lookup(name)` → function pointer)
  - [x] configure optimization pipeline (O0/O2/O3 + optional vectorization)
  - [x] hook up process symbol resolution (e.g., `DynamicLibrarySearchGenerator`)
  - [x] optional object cache hook (in-memory first; on-disk later)
  - [x] optional perf/gdb event listeners (Linux `perf`, GDB JIT interface)
  - [x] thread safety for compilation + caching (mutex or Orc facilities)

## 5. JIT Compiler Facade (FormIR → compiled kernels)

This is the orchestration layer that:
1) validates, 2) lowers to `KernelIR`, 3) emits LLVM, 4) JIT-compiles, 5) returns an `AssemblyKernel`.

- [x] Add `Code/Source/solver/FE/Forms/JIT/JITCompiler.h`
- [x] Add `Code/Source/solver/FE/Forms/JIT/JITCompiler.cpp`
- [ ] Implement compilation pipeline:
  - [ ] `forms::jit::canCompile(FormIR, strictness)` gate
  - [ ] `lowerToKernelIR(term.integrand)` per term (or per fused domain kernel)
  - [ ] decide kernel fusion strategy:
    - [ ] **preferred**: generate one kernel per `(domain, marker, form-kind)` that accumulates all matching terms
    - [ ] alternative: one kernel per term (simpler, but more overhead)
  - [ ] decide specialization strategy:
    - [ ] generic loops using `n_qpts/n_dofs` from args
    - [ ] specialized variants for common `n_qpts` (and/or dof counts)
  - [ ] produce stable cache key (see section 8)
  - [ ] compile via `JITEngine` and return a configured kernel wrapper
  - [ ] handle compilation failures with clear diagnostics and fallback (no hard crash)

## 6. LLVM IR Generator (`KernelIR`/`FormExpr` → LLVM)

The generator should ideally consume `KernelIR` (deterministic op list), not raw AST recursion.

- [ ] Add `Code/Source/solver/FE/Forms/JIT/LLVMGen.h`
- [ ] Add `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
- [ ] Implement a shape/type system for codegen (scalar/vector/matrix/tensor3/tensor4):
  - [ ] decide supported max ranks/sizes in JIT-fast mode (recommend: vectors ≤3, matrices ≤3×3, tensor3 ≤3×3×3, tensor4 3×3×3×3)
  - [ ] implement compile-time shape inference for `KernelIR` ops (or reuse existing metadata in nodes)
  - [ ] validate shapes at compile-time; reject (strict) or lower slower (relaxed) for dynamic shapes
- [ ] Implement ABI bindings: helpers to load from `assembly::jit::{Cell,BoundaryFace,InteriorFace}KernelArgs*`
  - [ ] fast path loads for basis tables / geometry / weights / constants / coupled slots / time coefficients
  - [ ] material state load/store helpers for `MaterialState*Ref`
- [ ] Implement kernel skeleton emission:
  - [ ] function prototypes:
    - [ ] `void kernel_cell_vN(const assembly::jit::CellKernelArgsVn* __restrict)`
    - [ ] `void kernel_bdry_vN(const assembly::jit::BoundaryFaceKernelArgsVn* __restrict)`
    - [ ] `void kernel_iface_vN(const assembly::jit::InteriorFaceKernelArgsVn* __restrict)`
  - [ ] loop structure matching interpreter semantics:
    - [ ] term loop (or fused term evaluation)
    - [ ] quadrature loop
    - [ ] test/trial dof loops
    - [ ] interior/interface face: 4 blocks (mm, pp, mp, pm) with correct restriction semantics
  - [ ] correct scaling:
    - [ ] `integration_weights[q]` multiplier
    - [ ] time-derivative term weights (from `KernelSideArgs`)
    - [ ] marker filtering semantics (compile-time filtered kernels preferred)
- [ ] Implement lowering for Forms vocabulary (emit LLVM instructions/calls):
  - [ ] Terminals / loads:
    - [ ] `Constant`, `Identity(dim)`
    - [ ] `ParameterRef`, `BoundaryIntegralRef`, `AuxiliaryStateRef`
    - [ ] `Coordinate`, `ReferenceCoordinate`, `Normal`
    - [ ] `Jacobian`, `JacobianInverse`, `JacobianDeterminant`
    - [ ] `CellDiameter`, `CellVolume`, `FacetArea`, `CellDomainId`
    - [ ] `Time`, `TimeStep`, `EffectiveTimeStep`
    - [ ] `MaterialStateOldRef`, `MaterialStateWorkRef`
    - [ ] `PreviousSolutionRef(k)`
    - [ ] `TestFunction`, `TrialFunction` (FormKind-dependent semantics)
    - [ ] `DiscreteField`, `StateField` (requires ABI decision in section 3)
  - [ ] Differential operators:
    - [ ] `Gradient`, `Divergence`, `Curl`, `Hessian`, `TimeDerivative`
  - [ ] DG operators:
    - [ ] `RestrictMinus`, `RestrictPlus`
    - [ ] `Jump`, `Average`
  - [ ] Algebra:
    - [ ] `Negate`, `Add`, `Subtract`, `Multiply`, `Divide`
    - [ ] `InnerProduct`, `DoubleContraction`, `OuterProduct`, `CrossProduct`
    - [ ] `Power`, `Minimum`, `Maximum`
  - [ ] Comparisons / control:
    - [ ] `Less`, `LessEqual`, `Greater`, `GreaterEqual`, `Equal`, `NotEqual`
    - [ ] `Conditional` (prefer `select` to avoid branches for SIMD)
  - [ ] Constructors / indexing:
    - [ ] `AsVector`, `AsTensor`
    - [ ] `Component`
    - [x] `IndexedAccess` (Einstein notation) supported for fully-contracted scalar contractions (canonicalized in `KernelIR` hashing and lowered to index loops in `LLVMGen`)
  - [ ] Tensor ops / scalar functions:
    - [ ] `Transpose`, `Trace`, `Determinant`, `Inverse`, `Cofactor`
    - [ ] `Deviator`, `SymmetricPart`, `SkewPart`
    - [ ] `Norm`, `Normalize`
    - [ ] `AbsoluteValue`, `Sign`, `Sqrt`, `Exp`, `Log`
  - [ ] Constitutive hooks:
    - [ ] `ConstitutiveOutput` (inlined constitutive expansions)
    - [ ] `Constitutive` (either rejected in strict mode or lowered via external call in relaxed mode)
- [ ] Implement loop/graph optimizations in codegen:
  - [ ] use `KernelIR` CSE result to avoid recomputation
  - [ ] hoist loop invariants (per-term weights, geometry invariants, basis-independent constants)
  - [ ] cache per-qpt solution quantities (e.g., `u(q)`, `grad(u)(q)`) when FormKind requires it

## 7. External Calls / Trampolines (relaxed mode)

For `Strictness::AllowExternalCalls`, JIT kernels must be able to call out to C/C++ helpers for opaque operations.

- [ ] Define a stable C ABI for coefficient evaluation:
  - [ ] scalar and vector/matrix/tensor coefficients
  - [ ] time-dependent coefficient variants
  - [ ] calling convention and how to identify which coefficient to call (slot/id)
  - [ ] how `KernelSideArgs::user_data` participates (context pointer, vtable, etc.)
- [ ] Define a stable C ABI for non-inlinable constitutive evaluation:
  - [ ] evaluate outputs for one qpt given inputs
  - [ ] (optional) evaluate vectorized batch of qpts for reduced overhead
- [ ] Register these symbols with OrcJIT so `LLVMGen` can emit `call` instructions to them.
- [ ] Mark any kernel using external calls as **non-cacheable** (or cacheable only within-process with pointer-identity keys).

## 8. Kernel Caching (in-memory + optional on-disk)

- [ ] Define the cache key precisely (must cover all semantic + codegen knobs):
  - [ ] `KernelIR::stableHash64()` of each fused kernel (or term list) + ABI version
  - [ ] FormKind + IntegralDomain (+ marker id for boundary/interface)
  - [ ] element/space signature inputs that affect basis sizes and semantics
  - [ ] JITOptions knobs (opt level, vectorize, debug_info, specialization policy)
  - [ ] target triple + CPU features (+ LLVM version if on-disk)
  - [ ] include inlinable constitutive kind ids / state layout signatures when relevant
- [ ] Implement in-memory cache:
  - [ ] `unordered_map<CacheKey, CompiledKernelHandle>` on the engine or compiler layer
  - [ ] concurrency control (compile-once semantics)
- [ ] Optional: implement LLVM `ObjectCache`-based on-disk cache (directory = `JITOptions::cache_directory`).

## 9. JIT Kernel Wrappers (`AssemblyKernel` implementations)

Implement concrete `assembly::AssemblyKernel` classes that invoke compiled function pointers.

- [x] Add `Code/Source/solver/FE/Forms/JIT/JITFormKernel.h` (and `.cpp` if needed)
- [x] Implement wrappers for:
  - [x] linear/bilinear `forms::FormKernel` equivalent (cell + boundary + face domains)
	  - [x] residual/Jacobian kernel strategy:
	    - [x] Option A (affine residual): compile `forms::LinearFormKernel` as real bilinear (+ optional linear) kernels and form residual via `K*u`
	    - [ ] Option B: emit Dual-capable kernels (larger IR; mirrors `NonlinearFormKernel`)
	    - [x] Option C: symbolic tangent decomposition for nonlinear residuals (derive bilinear tangent form and assemble Jacobian without Dual)
- [x] Implement `computeCell/computeBoundaryFace/computeInteriorFace/computeInterfaceFace`:
  - [x] pack args using `assembly::jit::pack*KernelArgs*`
  - [x] call the compiled function pointer
  - [x] set `KernelOutput::{has_matrix,has_vector}` correctly
  - [x] ensure material-state updates are applied (either inside JIT kernel or via separate compiled pre-pass)
- [x] Preserve setup-time rewrites already used by interpreter kernels:
  - [x] `resolveParameterSlots(...)`
  - [x] `resolveInlinableConstitutives()`

## 10. Integration Points (how users enable the JIT)

- [x] Update `Code/Source/solver/FE/Assembly/SymbolicAssembler.cpp` to honor `SymbolicOptions::jit.enable`:
  - [x] when enabled and `canCompile(...)` passes → return/use JIT kernel wrappers
  - [x] otherwise → use existing interpreter kernels
- [x] Ensure Systems integration paths that build kernels (e.g., `FE/Systems/FormsInstaller`) can request JIT kernels where appropriate.
- [x] (Later) Add a JIT path for functional kernels (`forms::FunctionalFormKernel`) used for coupled boundary integrals.

## 11. Robustness / Diagnostics

- [ ] Never crash simulations:
  - [ ] wrap JIT compilation in error handling and fall back to interpreter kernels
  - [ ] include diagnostic context: form dump, offending node (`JITValidationIssue`), LLVM error message
- [ ] Add debug options (driven by `JITOptions`):.vtu
  - [ ] dump lowered `KernelIR`
  - [ ] dump generated LLVM IR before/after optimization
  - [ ] optional emission of DWARF for stepping through JIT code
- [ ] Profiling hooks:
  - [ ] perf listener (Linux)
  - [ ] GDB registration listener

## 12. Testing + Benchmarking

- [ ] Add unit tests for end-to-end JIT correctness (new suite recommended: `Code/Source/solver/FE/Tests/Unit/JIT/`):
  - [ ] Poisson (cell bilinear): `inner(grad(u), grad(v))`
  - [ ] Mass matrix (cell bilinear): `u*v`
  - [ ] Neumann/Robin (boundary): `g*v`
  - [ ] Interior penalty / jump/avg (interior face)
  - [ ] Time-derivative weighted forms (`dt(u)*v`) with transient context
  - [ ] ParameterRef / coupled slot loads
  - [ ] Material state loads/stores (inlined updates)
  - [ ] Vector-valued trial/test (ProductSpace) at least for H1-like bases
- [ ] Add regression tests that run assembly with `jit.enable=true` and compare to interpreter outputs on small meshes.
- [ ] Add benchmarks:
  - [ ] micro-bench: kernel invocation throughput (per element)
  - [ ] macro-bench: `assembleMatrix/assembleVector/assembleBoth` scaling vs interpreter
  - [ ] cache effectiveness metrics (first-call compile time vs steady-state)

## 13. Symbolic Differentiation + Tangent Decomposition (Nonlinear JIT path)

Goal: avoid element-level Dual/AD for nonlinear residual Jacobians by symbolically differentiating
`R(u; v)` into a bilinear tangent form `a(δu, v) = dR/du[δu, v]`, enabling fast scalar LLVM JIT
for both residual and Jacobian assembly.

### 13.1 Public API (Forms)

- [x] Add a clean API in `svmp::FE::forms`:
  - [x] `FormExpr differentiateResidual(const FormExpr& residual)` (differentiate w.r.t. active `TrialFunction`)
  - [x] `bool canDifferentiateSymbolically(const FormExpr& expr)`
  - [x] `FormExpr simplify(const FormExpr& expr)`
  - [x] (optional) overloads to differentiate w.r.t. a specific TrialFunction/FieldId (multi-field support)
- [x] Decide differentiation staging:
  - [x] support `IndexedAccess` in the differentiator (d/dx commutes with indexing); interpreter fallback lowers via `forms::einsum` when needed
  - [x] ensure coupled placeholders (`ParameterSymbol`, `BoundaryIntegralSymbol`, `AuxiliaryStateSymbol`) are either:
    - [x] treated as constants and preserved for later slot resolution, or
    - [ ] rejected early with a clear diagnostic (policy decision)

### 13.2 Core Differentiation Engine (AST → AST)

- [x] Implement recursive differentiation on `FormExprNode` trees:
  - [x] structural recursion with sharing (avoid expression blow-up where possible)
  - [x] preserve space signatures for `TestFunction`/`TrialFunction` nodes
  - [x] propagate marker/domain wrappers: `dx/ds/dS/dI` should wrap the differentiated integrand
  - [x] deterministic output (stable hashing/stringification for caching)
- [x] Define and implement a “symbolic differentiability” contract:
  - [x] identify nodes that must trigger fallback (e.g., non-inlinable constitutive calls without tangents)
  - [x] provide first-issue diagnostics (subexpr + reason) similar to `Forms/JIT/JITValidation`

### 13.3 Differentiation Rules — Coverage for Current Forms Vocabulary

Checklist below is intended to cover **all currently implemented** `FormExprType` vocabulary in
`Code/Source/solver/FE/Forms/FormExpr.h` as it relates to `d/d(TrialFunction)`.

- [x] Terminals (treat as constants; derivative = 0):
  - [x] `Constant`
  - [x] `Coefficient`
  - [x] `ParameterSymbol`, `ParameterRef`
  - [x] `BoundaryFunctionalSymbol`, `BoundaryIntegralSymbol`, `BoundaryIntegralRef`
  - [x] `AuxiliaryStateSymbol`, `AuxiliaryStateRef`
  - [x] `MaterialStateOldRef`, `MaterialStateWorkRef`
  - [x] `PreviousSolutionRef`
  - [x] `Coordinate`, `ReferenceCoordinate`
  - [x] `Time`, `TimeStep`, `EffectiveTimeStep`
  - [x] `Identity(dim)`, `Jacobian`, `JacobianInverse`, `JacobianDeterminant`, `Normal`
  - [x] `CellDiameter`, `CellVolume`, `FacetArea`, `CellDomainId`
- [x] Field/test/trial semantics:
  - [x] `TestFunction` (derivative = 0)
  - [x] `TrialFunction` (derivative = `TrialFunction(δu)` / perturbation)
  - [x] `DiscreteField` (derivative = 0)
  - [x] `StateField` (policy: derivative = 0 unless lowered to TrialFunction; decide and implement)
- [x] Differential operators (commute with differentiation):
  - [x] `Gradient`, `Divergence`, `Curl`, `Hessian`, `TimeDerivative`
- [x] DG restrictions/operators (commute; preserve side semantics):
  - [x] `RestrictMinus`, `RestrictPlus`
  - [x] `Jump`, `Average`
- [x] Algebraic operators:
  - [x] `Negate`
  - [x] `Add`, `Subtract`
  - [x] `Multiply` (product rule)
  - [x] `Divide` (quotient rule)
  - [x] `InnerProduct`, `DoubleContraction`, `OuterProduct`, `CrossProduct` (product rules)
  - [x] `Power` (support at least constant exponent; define behavior for general exponent)
  - [x] `Minimum`, `Maximum` (define subgradient policy or reject; non-smooth)
- [x] Comparisons / predicates:
  - [x] `Less`, `LessEqual`, `Greater`, `GreaterEqual`, `Equal`, `NotEqual` (define derivative policy: usually 0; non-smooth)
  - [x] `Conditional(cond, a, b)` → `Conditional(cond, da, db)` (do not differentiate `cond`)
- [x] Constructors / indexing:
  - [x] `AsVector`, `AsTensor` (componentwise)
  - [x] `Component` (extract from derivative)
  - [x] `IndexedAccess` (Einstein notation): propagate indices through differentiation (`d(A(i,...))/du = dA/du (i,...)`)
- [x] Tensor ops / scalar functions:
  - [x] `Transpose`, `Trace` (linearity)
  - [x] `Determinant`, `Inverse`, `Cofactor` (matrix calculus rules)
  - [x] `Deviator`, `SymmetricPart`, `SkewPart` (linearity)
  - [x] `Norm`, `Normalize` (chain rule; define behavior at zero)
  - [x] `AbsoluteValue`, `Sign` (non-smooth; define policy or reject)
  - [x] `Sqrt`, `Exp`, `Log` (chain rule)
- [x] Constitutive hooks:
  - [x] `ConstitutiveOutput`
  - [x] `Constitutive`
- [x] Integral wrappers (d/d(u) commutes with integration):
  - [x] `CellIntegral`
  - [x] `BoundaryIntegral`
  - [x] `InteriorFaceIntegral`
  - [x] `InterfaceIntegral`

### 13.4 Constitutive Tangent Integration (Symbolic)

- [x] Decide supported path(s) for differentiating constitutive calls:
  - [x] **Preferred**: require `resolveInlinableConstitutives()` to eliminate `FormExprType::Constitutive` and symbolically differentiate the expanded `FormExpr`.
- [ ] Optional: add an explicit tangent interface for non-inlinable `forms::ConstitutiveModel` calls and represent it in the AST:
    - [ ] define new `FormExprType` node(s) for tangent evaluation and shape metadata
    - [ ] implement interpreter evaluation for tangent nodes (Real)
    - [ ] implement JIT lowering for tangent nodes (strict or relaxed)
    - [ ] update `JITValidation` to gate/mark cacheability for tangent nodes
- [x] Validate tangent correctness for at least one nonlinear constitutive model used in-tree (e.g., Neo-Hookean if present).

### 13.5 Simplification / Normalization (post-diff)

- [x] Implement core simplification rules:
  - [x] `Add(a, 0) → a`, `Add(0, a) → a`
  - [x] `Multiply(a, 0) → 0`, `Multiply(0, a) → 0`
  - [x] `Multiply(a, 1) → a`, `Multiply(1, a) → a`
  - [x] `Negate(Negate(a)) → a`
  - [x] `Add(a, Negate(a)) → 0` (and commuted form)
- [x] Add constant folding for purely constant subtrees (including scalar math functions where possible).
- [x] Ensure simplification preserves legality:
  - [x] no measure nodes inside integrands
  - [x] `IndexedAccess` may remain post-simplify; interpreter fallback lowers via `forms::einsum`, and the JIT path lowers to index loops (fully-contracted only)
- [x] Ensure the output remains friendly to `KernelIR` CSE (stable structure and deterministic traversal order).

### 13.6 New Kernel: Symbolic Nonlinear Residual/Jacobian (no Dual)

- [x] Add `SymbolicNonlinearFormKernel` (or extend `NonlinearFormKernel`) so that:
  - [x] takes residual `FormExpr` (or compiled residual `FormIR`)
  - [x] builds a symbolic tangent `FormExpr` via `differentiateResidual(...)`
  - [x] compiles:
    - [x] residual → `FormIR` with `FormKind::Residual` (for vector assembly)
    - [x] tangent → `FormIR` with `FormKind::Bilinear` (for matrix assembly)
  - [x] assembles residual + Jacobian via scalar Real evaluation (no Dual arrays)
  - [x] supports `NonlinearKernelOutput::{Both,MatrixOnly,VectorOnly}`
  - [x] supports all domains present in the residual (`dx/ds/dS/dI`)
  - [x] applies the same setup-time rewrites as existing kernels:
    - [x] `resolveParameterSlots(...)`
    - [x] `resolveInlinableConstitutives()`
- [x] Decide caching strategy:
  - [x] cache the compiled tangent `FormIR` in-memory (LRU) keyed by a stable residual hash (includes coefficient/callback identity) + cache version

### 13.7 LLVM JIT Enablement for Nonlinear (Residual + Tangent)

- [x] Extend `LLVMGen` to support `FormKind::Residual` codegen:
  - [x] interpret `TrialFunction` and its derivatives as **current solution** values computed from `solution_coefficients` + basis tables
  - [x] preserve existing bilinear/linear semantics and keep compatibility with `KernelArgs` ABI
- [x] Update `JITKernelWrapper` to JIT-accelerate the new symbolic nonlinear kernel:
  - [x] compile the residual (FormKind::Residual) kernel when vector output is requested
  - [x] compile the tangent bilinear kernel when matrix output is requested
  - [x] preserve interpreter fallback on any JIT unavailability or validation failure
- [x] Update integration points so users can enable symbolic tangent mode:
  - [x] `Assembly/SymbolicAssembler.cpp` option to prefer symbolic tangent over AD for nonlinear residuals
  - [x] `Systems/FormsInstaller` option to request symbolic tangent for nonlinear kernels

### 13.8 Verification (Symbolic vs AD vs Finite Difference)

- [x] Add unit test suite: `Code/Source/solver/FE/Tests/Unit/Forms/test_SymbolicDifferentiation.cpp`
- [x] Rule-level tests (each operator rule compared to `NonlinearFormKernel` AD output).
- [ ] End-to-end form tests (compare symbolic tangent vs AD Jacobian):
  - [x] Poisson residual (should be linear; tangent equals original bilinear)
  - [x] nonlinear diffusion `inner(k(u) * grad(u), grad(v))`
  - [x] hyperelasticity-like form (matrix ops `det`, `inv`, `cof`, `trace`, etc.)
  - [x] Navier–Stokes convection `inner((u · grad(u)), v)` (if available in vocabulary tests)
- [x] Finite difference verification harness:
  - [x] `a(δu, v) ≈ [R(u + εδu; v) - R(u; v)] / ε`
- [x] JIT-vs-interpreter regression tests for symbolic nonlinear kernels with `jit.enable=true` (once residual codegen is supported).

## 14. Tensor Calculus System Roadmap (Detailed Checklist)

This section extends the core LLVM JIT plan with a detailed, incremental checklist for adding a **tensor-calculus / index-notation** subsystem that preserves tensor structure (contractions, symmetry, special tensors) long enough to generate compact, loop-based LLVM IR. The primary goal is to avoid the current `einsum()` strategy of expanding contractions into many scalar terms, which inflates `FormExpr`/`KernelIR` size and slows JIT compilation.

### 14.1 Decisions, Scope, and Compatibility
- [x] Decide the representation strategy: reuse existing `forms::Index`/`FormExprType::IndexedAccess` (Einstein notation) and lower to loop-based codegen in the LLVM JIT backend; interpreter fallback lowers via `forms::einsum`.
- [x] Define supported dimensions/ranks and MVP operator subset: 3D (dim=3), rank ≤ 4, and JIT loop lowering supports **fully-contracted** Einstein sums only (each index id appears exactly twice); operators are the existing FE/Forms vocabulary already supported by `LLVMGen` (incl. grad/div/curl/hess, det/inv/cof/tr, inner/double contraction, and DG ops in face kernels).
- [x] Define compatibility contract: tensor-calculus path is **opt-in** (via `SymbolicOptions::jit.enable`) and preserves existing behavior; interpreter fallback lowers via `einsum()` for scalar-expanded evaluation.
- [x] Define a performance acceptance target: reduce explicit scalar-term expansion for common contractions and keep LLVM IR size/compile time bounded (proxy metric: KernelIR op-count; unit test: `Tests/Unit/Forms/Tensor/test_TensorContraction.cpp` `IndexedAccessAvoidsScalarExpansionIRBloat`).

### 14.2 Phase 1 — Tensor Types & Metadata (Foundation)
- [x] Create `Code/Source/solver/FE/Forms/Tensor/` and wire it into the build (CMake + includes).
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorIndex.h/.cpp`:
  - [x] `IndexVariance` (Lower/Upper/None) and `IndexRole` (Free/Dummy/Fixed).
  - [x] `TensorIndex` (id/name/variance/dimension/fixed_value) with `raised()`/`lowered()` helpers.
  - [x] `MultiIndex` with `freeIndices()`, `contractionPairs()`, and `isFullyContracted()`.
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorSymmetry.h/.cpp`:
  - [x] `SymmetryType`, `SymmetryPair`, and `TensorSymmetry` queries (`isSymmetricIn`, `isAntisymmetricIn`).
  - [x] Independent-component enumeration (`numIndependentComponents`, `independentComponents`) for common symmetries (2nd-order symmetric/antisymmetric, elasticity major/minor/full).
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/SpecialTensors.h/.cpp`:
  - [x] `SpecialTensorKind` and constexpr helpers for Kronecker delta and Levi-Civita (2D/3D).
  - [x] Define how “metric tensors” are represented (identity metric default; hooks for curved/moving meshes).
- [x] Add unit tests under `Code/Source/solver/FE/Tests/Unit/Forms/Tensor/`:
  - [x] `test_TensorIndex.cpp` (free/dummy/fixed behavior, variance operations).
  - [x] `test_TensorSymmetry.cpp` (independent component counts and mapping).
- [x] Support free indices in tensor expressions:
  - [x] `A_{ij} B^j` → vector output (one free index `i`)
  - [x] `A_{ik} B^{kj}` → matrix output (two free indices `i`, `j`)
  - [x] Validate result shape matches expected test/trial function shape
- [x] Support tensor structures for H(curl)/H(div) spaces:
  - [x] Piola transformations: covariant vs contravariant mapping
  - [x] Curl/div of tensor-valued fields
  - [x] Integration with existing `FormExprType::Curl`/`Divergence`
  
### 14.3 Phase 2 — FormExpr Integration (Index Notation + New Vocabulary)
- [x] Extend the `FormExpr` layer to carry tensor-index metadata and symmetry tags where needed (stable printing, hashing, structural equality).
- [x] Add tensor-calculus vocabulary to `Code/Source/solver/FE/Forms/FormExpr.h`/`.cpp` (choose one approach):
  - [ ] **Option A (new node types):** add `FormExprType` entries for:
    - [ ] `TensorLiteral`, `TensorContraction`, `TensorProduct`
    - [ ] `KroneckerDelta`, `LeviCivita`, `MetricTensor`
    - [ ] `IndexRaise`, `IndexLower`, `IndexSwap`, `IndexTrace`
    - [ ] `Symmetrize`, `Antisymmetrize`
  - [x] **Option B (reuse existing):** represent tensor calculus using existing `IndexedAccess` + existing algebraic nodes, and attach tensor metadata via node payloads.
- [x] Add EDSL helpers to build tensor/index expressions ergonomically (without requiring explicit component expansion).
- [x] Update all relevant switch-based infrastructure to recognize the new vocabulary (as applicable):
  - [x] `Code/Source/solver/FE/Forms/FormCompiler.*` (opt-in compilation of `IndexedAccess` when `SymbolicOptions::jit.enable` is set).
  - [x] `Code/Source/solver/FE/Forms/FormKernels.cpp` interpreter evaluation (Real + Dual): ensure the tensor path lowers via `forms::einsum` before evaluation.
  - [x] `Code/Source/solver/FE/Forms/SymbolicDifferentiation.*`: allow `IndexedAccess` through symbolic differentiation and propagate indices through derivatives.
- [x] Implement tensor expression validation with clear diagnostics (MVP: fully-contracted Einstein sums only):
  - [x] Mismatched index dimensions: "Index 'i' has extent 3 in A but extent 2 in B"
  - [x] Invalid contractions: "Index 'i' appears 3 times; must appear exactly twice for contraction"
  - [x] Free index mismatch: "Result has free indices {i,j} but expected scalar output"
  - [x] Variance errors: "Cannot contract covariant index with covariant index without metric"
- [x] Implement tensor-aware pretty-printing:
  - [x] Print indexed access in subscript form (e.g., `A_{ij} * B_{jk}`) instead of expanded form (variance-aware `^`/`_` notation is future work)
  - [x] Show symmetry tags: `A_{(ij)}` for symmetric, `A_{[ij]}` for antisymmetric
  - [x] Special tensors: print `δ_{ij}` and `ε_{ijk}` symbolically
  - [x] Canonical form display for debugging canonicalization

### 14.4 Phase 3 — Tensor Algebra Layer (Contraction, Simplify, Canonicalize)
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorContraction.h/.cpp`:
  - [x] `analyzeContractions(expr)` (identify free vs bound indices; validate extent consistency for `IndexedAccess`).
  - [x] `optimalContractionOrder(...)` with a cost model (dynamic programming parenthesization for chains).
  - [x] `contractIndices(expr, a, b)` as a primitive contraction transform.
- [x] Define contraction cost model:
  - [x] FLOP-based cost: O(∏ extents of all indices involved)
  - [x] Memory access cost: consider tensor sizes and cache effects
  - [x] Special-tensor shortcuts: δ-contractions are O(n) not O(n²)
  - [x] Symmetry-aware cost: symmetric tensors have fewer unique entries
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorSimplify.h/.cpp`:
  - [x] δ-contraction rules (substitution, trace-to-dimension, composition).
  - [x] ε-contraction identities (ε·ε → δδ − δδ; symmetry annihilation).
  - [x] Metric contraction/raise/lower simplifications (`g_ij g^{jk} → δ^k_i`, etc.).
  - [x] Symmetry-aware simplification (symmetric vs antisymmetric contractions → 0).
  - [x] Fixed-point iteration with termination guarantees and debug counters (for profiling).
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorCanonicalize.h/.cpp`:
  - [x] Canonical index renaming so structurally identical contractions hash identically (implemented in `KernelIR` lowering; also available as `computeCanonicalIndexRenaming` helper).
  - [x] Canonical term ordering for sums/products (where mathematically valid).
- [x] Add unit tests:
  - [x] `test_TensorSimplify.cpp` (δ/ε/metric/symmetry rules).
  - [x] `test_TensorCanonicalize.cpp` (index renaming and ordering invariants).

### 14.5 Phase 4 — Tensor-Aware Symbolic Differentiation
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorDifferentiation.h/.cpp`:
  - [x] `TensorDiffContext` (diff variable + indices; include multi-field identification such as `FieldId` where relevant).
  - [x] Product and chain rules for tensor products/contractions.
  - [x] Determinant/inverse/trace/cofactor derivatives in index form (keep contractions symbolic).
  - [x] Optional: spectral/eigen derivatives.
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/SpecialTensorDerivatives.h/.cpp`:
  - [x] Kronecker delta and Levi-Civita derivatives (0).
  - [x] Metric tensor derivatives (identity metric default; hook for mesh-motion/curved metrics).
  - [x] Deformation-gradient derivatives (for hyperelasticity chains) in index form.
- [x] Provide a public entry point (and capability query) for tensor differentiation:
  - [x] `differentiateTensorResidual(residual, ctx)` and `checkTensorDifferentiability(expr)`.
  - [x] Integrate with existing `forms::differentiateResidual(...)` (dispatch when tensor calculus nodes are present).
- [x] Verification tests:
  - [x] Compare tensor-derived tangents against existing AD (Dual) for representative nonlinear forms (hyperelasticity, nonlinear diffusion, convection).
  - [x] Add finite-difference verification harness for complex tensor forms (optional but recommended).
- [x] (Future) Higher-order derivative support:
  - [x] Second derivatives for Hessian computation (optimization, stability analysis)
  - [x] Define index structure for d²A/du²
  - [x] Integration with Newton-type solvers requiring Hessian-vector products
  
### 14.6 Phase 5 — Optimal Lowering (Symmetry, CSE, Loop-Nest IR)
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/SymmetryOptimizer.h/.cpp`:
  - [x] Compute independent component sets for common symmetries and provide canonical mapping/sign rules.
  - [x] Lower tensor expressions using only independent components (`lowerWithSymmetry`).
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorCSE.h/.cpp`:
  - [x] Tensor-aware CSE that recognizes repeated contractions and expensive subexpressions (e.g., `det(F)`, `inv(F)`).
  - [x] Temporary-introduction strategy compatible with `KernelIR`/LLVM emission.
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/LoopStructure.h/.cpp`:
  - [x] `generateLoopNest(expr)` for contractions/products (generate loops instead of scalar-term expansion).
  - [x] `fuseLoops(...)` and `optimizeLoopOrder(...)` with a cache-locality heuristic.
  - [x] Vectorization hints in the loop metadata (candidate inner loops, vector widths).
- [x] Implement sparse/structured tensor handling:
  - [x] Recognize zero blocks in tensors with known structure (e.g., elasticity minor symmetry)
  - [x] Skip computation of structurally zero entries in generated loops
  - [x] Optional: compressed storage for Voigt notation (6×6 for 3×3×3×3 symmetric)
- [x] Define an incremental lowering strategy:
  - [x] Lower tensor calculus → loop-based IR when profitable; otherwise fall back to `einsum()` expansion.
  - [x] Ensure canonicalization runs before hashing/caching so equivalent expressions share compiled kernels.
- [x] Define temporary allocation strategy for tensor evaluation (`Code/Source/solver/FE/Forms/Tensor/TensorAllocation.h/.cpp`):
  - [x] Small tensors (≤ 3×3×3×3): stack allocation with fixed-size arrays
  - [x] Large/dynamic tensors: arena allocator per quadrature point batch
  - [x] Temporary reuse: lifetime analysis (`LoopNestProgram::Contribution::available_after_op`) to minimize peak memory
  - [x] Alignment: ensure 32/64-byte alignment for SIMD

### 14.7 Phase 6 — LLVM JIT Integration (TensorIR + Loop Emission)
- [x] Implement `Code/Source/solver/FE/Forms/Tensor/TensorIR.h/.cpp`:
  - [x] `TensorIRNode` kinds for tensor ops, loop nests, loads/stores, and scalar ops.
  - [x] `lowerToTensorIR(expr)` that preserves tensor structure until final emission.
- [x] Implement `Code/Source/solver/FE/Forms/JIT/LLVMTensorGen.h/.cpp`:
  - [x] Emit loop nests (`emitLoopNest`) with correct PHI nodes and reduction patterns.
  - [x] Emit contractions as loops (`emitContraction`) and apply vectorization when enabled.
  - [x] Emit symmetry-aware loops (e.g., `j >= i` for symmetric tensors) when safe.
- [x] Integrate TensorIR into the existing JIT pipeline:
  - [x] Extend `Code/Source/solver/FE/Forms/JIT/LLVMGen.*` to delegate tensor-structured nodes/IR to `LLVMTensorGen`.
  - [x] Extend `Code/Source/solver/FE/Forms/JIT/JITValidation.*` to validate tensor/loop IR (strict vs allow-external-calls).
  - [x] Ensure kernel caching keys include tensor-lowering options and canonicalized TensorIR hashes.
- [x] Optional: Polly integration (behind a build/runtime flag):
  - [x] Add loop metadata suitable for Polly tiling/vectorization.
  - [x] Validate behavior across LLVM versions used by CI.
- [x] Ensure thread safety in tensor calculus pipeline:
  - [x] Canonicalization: no global mutable state; use local index renaming
  - [x] TensorIR caching: thread-safe cache with compile-once semantics
  - [x] Loop emission: no shared mutable state between threads

### 14.8 Testing, Benchmarks, and Rollout
- [ ] Add end-to-end tests that compare:
  - [ ] Tensor-calculus interpreter path vs existing scalar-expanded interpreter path (correctness).
  - [x] Tensor-calculus JIT path vs interpreter (bitwise/tolerance) for representative kernels (`Code/Source/solver/FE/Tests/Unit/Forms/Tensor/test_IndexedAccessJIT.cpp`).
- [ ] Add benchmarks to measure:
  - [ ] Expression/IR size (FormExpr nodes, KernelIR nodes, LLVM IR instruction count).
  - [ ] JIT compile time (first compile + cache hit).
  - [ ] Runtime assembly throughput (elements/sec) vs current `einsum`-expanded JIT and interpreter.
- [ ] Rollout strategy:
  - [ ] Keep tensor calculus disabled by default until tests/benchmarks meet targets.
  - [ ] Add a runtime toggle (e.g., `jit.tensor_calculus = on/off/auto`) with “auto” gating on expression complexity.
- [ ] Implement auto-detection heuristics for tensor calculus benefit:
  - [ ] Threshold: use tensor path when scalar expansion would exceed N terms (e.g., N=50)
  - [ ] Pattern recognition: detect matrix-matrix products, tensor contractions
  - [ ] Complexity estimation: estimate scalar term count vs loop iteration count
  - [ ] Feedback mechanism: log when tensor path is chosen and why  
