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

- [ ] Add `Code/Source/solver/FE/Forms/JIT/JITEngine.h`
- [ ] Add `Code/Source/solver/FE/Forms/JIT/JITEngine.cpp`
- [ ] Implement `JITEngine` responsibilities:
  - [ ] initialize native target (`InitializeNativeTarget*`) exactly once per process
  - [ ] create/configure `llvm::orc::LLJIT` (target triple, data layout)
  - [ ] support module submission (`addModule(ThreadSafeModule)`)
  - [ ] support symbol lookup (`lookup(name)` → function pointer)
  - [ ] configure optimization pipeline (O0/O2/O3 + optional vectorization)
  - [ ] hook up process symbol resolution (e.g., `DynamicLibrarySearchGenerator`)
  - [ ] optional object cache hook (in-memory first; on-disk later)
  - [ ] optional perf/gdb event listeners (Linux `perf`, GDB JIT interface)
  - [ ] thread safety for compilation + caching (mutex or Orc facilities)

## 5. JIT Compiler Facade (FormIR → compiled kernels)

This is the orchestration layer that:
1) validates, 2) lowers to `KernelIR`, 3) emits LLVM, 4) JIT-compiles, 5) returns an `AssemblyKernel`.

- [ ] Add `Code/Source/solver/FE/Forms/JIT/JITCompiler.h`
- [ ] Add `Code/Source/solver/FE/Forms/JIT/JITCompiler.cpp`
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
    - [ ] `IndexedAccess` must not appear (ensure `forms::einsum` lowering happens pre-JIT)
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

- [ ] Add `Code/Source/solver/FE/Forms/JIT/JITFormKernel.h` (and `.cpp` if needed)
- [ ] Implement wrappers for:
  - [ ] linear/bilinear `forms::FormKernel` equivalent (cell + boundary + face domains)
  - [ ] residual/Jacobian kernel strategy:
    - [ ] Option A: symbolic differentiation → compile residual + Jacobian as Real kernels
    - [ ] Option B: emit Dual-capable kernels (larger IR; mirrors `NonlinearFormKernel`)
- [ ] Implement `computeCell/computeBoundaryFace/computeInteriorFace/computeInterfaceFace`:
  - [ ] pack args using `assembly::jit::pack*KernelArgs*`
  - [ ] call the compiled function pointer
  - [ ] set `KernelOutput::{has_matrix,has_vector}` correctly
  - [ ] ensure material-state updates are applied (either inside JIT kernel or via separate compiled pre-pass)
- [ ] Preserve setup-time rewrites already used by interpreter kernels:
  - [ ] `resolveParameterSlots(...)`
  - [ ] `resolveInlinableConstitutives()`

## 10. Integration Points (how users enable the JIT)

- [x] Update `Code/Source/solver/FE/Assembly/SymbolicAssembler.cpp` to honor `SymbolicOptions::jit.enable`:
  - [x] when enabled and `canCompile(...)` passes → return/use JIT kernel wrappers
  - [x] otherwise → use existing interpreter kernels
- [x] Ensure Systems integration paths that build kernels (e.g., `FE/Systems/FormsInstaller`) can request JIT kernels where appropriate.
- [ ] (Later) Add a JIT path for functional kernels (`forms::FunctionalFormKernel`) used for coupled boundary integrals.

## 11. Robustness / Diagnostics

- [ ] Never crash simulations:
  - [ ] wrap JIT compilation in error handling and fall back to interpreter kernels
  - [ ] include diagnostic context: form dump, offending node (`JITValidationIssue`), LLVM error message
- [ ] Add debug options (driven by `JITOptions`):
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
