# Accelerating FE Assembly (OOP solver + LLVM JIT)

This document is a **concrete, code-oriented checklist** for making the new OOP solver’s FE assembly (Forms + LLVM JIT) **outperform svSolver’s element assembly** on typical fluid workloads (3D incompressible Navier–Stokes with stabilization).

Scope (what this targets first):
- Cell (volume) assembly of coupled velocity/pressure residual + Jacobian.
- Boundary-face terms used by fluid (traction, outflow/backflow, Nitsche).
- The full “assembly time” includes: **context prep + kernel eval + constraint distribution + global insertion**.

Non-scope (still important, but not assembly):
- Linear solver / preconditioner performance.
- MPI communication patterns outside the assembly phase boundaries.

References already in-tree:
- `Code/Source/solver/FE/Forms/JIT/LLVM_JIT_IMPLEMENTATION_CHECKLIST.md`
- `JIT_COMPATIBILITY_REVIEW.md`
- `LLVM_JIT_IMPLEMENTATION_PLAN.md`

---

## 0) Define “win” + build a repeatable benchmark harness

**Goal:** Make assembly throughput (elements/sec) and/or time-per-Newton-step better than svSolver for comparable discretizations.

Concrete steps:
- [ ] Pick 2–3 representative meshes/cases and freeze them (Tet4 P1/P1 or P2/P1, Tet10, a boundary-heavy case).
- [ ] Add wall-clock timers around the three dominant phases in `StandardAssembler`:
  - `prepareContextCell(...)` / `prepareContextFace(...)` (geometry+basis+packing).
  - `kernel.computeCell/computeBoundaryFace(...)` (JIT/interpreter kernel eval).
  - constraint distribution + `GlobalSystemView::addMatrixEntries/addVectorEntries(...)`.
- [ ] Report at least:
  - elements/sec (cell), boundary faces/sec,
  - % time in prep vs kernel vs scatter,
  - per-element bytes read/written (rough estimate) for memory-bound diagnosis.
- [ ] Add a micro-benchmark mode that assembles on a single rank/thread with “null” backends (dense or no-op GlobalSystemView) to isolate kernel+prep costs.

Why: svSolver’s performance comes largely from **tight, vectorizable inner loops** and **low overhead**. If we don’t isolate overhead sources, it’s easy to “optimize the wrong thing”.

---

## 1) Ensure the OOP solver is actually using the LLVM JIT fast path

Today, LLVM JIT acceleration in `Forms::JITKernelWrapper` only covers kernels backed by Forms IR, and it **does not JIT-accelerate** `NonlinearFormKernel` (dual-number path). See:
- `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp` (`maybeCompile()` and `WrappedKind` handling)
- `Code/Source/solver/FE/Systems/FormsInstaller.cpp` (`use_symbolic_tangent` choice)

Concrete steps:
- [ ] For nonlinear residual/Jacobian assembly, ensure the solver uses **symbolic tangents**:
  - route through `forms::SymbolicNonlinearFormKernel` (not `forms::NonlinearFormKernel`),
  - because `JITKernelWrapper` compiles `residualIR()` and `tangentIR()` only for `SymbolicNonlinearFormKernel`.
- [ ] Enable `jit.enable=true`, use `optimization_level=3`, `vectorize=true` in `forms::JITOptions` (`Code/Source/solver/FE/Forms/FormExpr.h`).
- [ ] Add a “JIT-fast compliance” check for the fluid formulation at setup time:
  - fail fast (or loudly warn) if a form contains `Coefficient` nodes, non-inlinable constitutives, or other nodes that force external calls.
  - target **Strict** mode for performance (see `JIT_COMPATIBILITY_REVIEW.md`); avoid `Strictness::AllowExternalCalls` in production-performance runs.
- [ ] Precompile/warm-up kernels at setup (avoid lazy first-element compilation in the timestep loop).

---

## 2) Remove the biggest current cost: `StandardAssembler` context prep overhead

In many FE codes, once the per-qpt integrand is JIT-optimized, the bottleneck becomes:
1) building per-element “context” arrays (basis/geometry/solution), and
2) scattering dense element blocks to global storage.

### 2.1 Eliminate redundant copies into `AssemblyContext`

Current hot-path pattern (cell and face):
- compute into `StandardAssembler` scratch vectors (`scratch_*`)
- then copy into `AssemblyContext` using `set*()` methods, which call `.assign()` and copy again

See:
- scratch arrays: `Code/Source/solver/FE/Assembly/StandardAssembler.h` (e.g., `scratch_basis_values_`)
- copies: `Code/Source/solver/FE/Assembly/StandardAssembler.cpp` calls like `context.setQuadratureData(...)`
- `.assign()` copies: `Code/Source/solver/FE/Assembly/AssemblyContext.cpp` (`setQuadratureData`, `setJacobianData`, `setTestBasisData`, ...)

Concrete steps (pick one strategy and commit to it):
- [ ] **Strategy A (preferred): write directly into `AssemblyContext` aligned storage**
  - add “mutable raw buffer” accessors in `AssemblyContext` for each packed array,
  - resize once, then fill in-place in `StandardAssembler` (no intermediate `scratch_*` vectors).
- [ ] **Strategy B: make `AssemblyContext` support view-mode**
  - store `std::span<const T>` views to assembler-owned scratch arrays for the duration of the kernel call,
  - keep current owning mode for other users.
- [ ] **Strategy C: bypass `AssemblyContext` entirely for JIT kernels**
  - pack `KernelArgsV3` directly from `StandardAssembler` scratch buffers and call the JIT function pointer,
  - keep `AssemblyContext` only for interpreter kernels.

Acceptance criteria:
- No `.assign()`/copy of basis/geometry arrays in the per-element hot loop for the JIT path.
- Alignment remains compatible with `KernelArgsV3` assertions (`Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`).

### 2.2 Stop evaluating basis polynomials per element/qpt (use caching)

`StandardAssembler` currently calls `BasisFunction::evaluate_values/gradients` at each quadrature point.
This is expensive and defeats the purpose of JIT-fast assembly.

You already have caches:
- basis-at-quadrature cache: `Code/Source/solver/FE/Basis/BasisCache.h`
- combined element cache: `Code/Source/solver/FE/Elements/ElementCache.cpp`

Concrete steps:
- [ ] In `StandardAssembler::prepareContextCell/Face`, fetch basis tables from `BasisCache` once:
  - values and reference gradients (and optionally Hessians),
  - for the exact `(basis type, element type, order, quadrature rule)` tuple.
- [ ] Store cached basis tables in a layout directly consumable by kernels:
  - today kernels expect `[i * n_qpts + q]` layout (see packers in `KernelArgs.h`),
  - restructure `BasisCacheEntry` (or introduce a new cache) so it can return that layout without per-element reshaping.
- [ ] For face assembly, cache face-restricted basis tables per `(face type, orientation, quad rule)` and apply any needed orientation transforms once per face-orientation.

Acceptance criteria:
- For H1 Lagrange on Tet/Hex, **no calls** to `BasisFunction::evaluate_*` inside the element loop.

### 2.3 Add affine-geometry fast paths (common in fluid meshes)

For linear simplices (e.g., Tet4) with affine mappings, Jacobians and physical gradients are constant.
`StandardAssembler` still computes mapping/J/invJ/detJ per quadrature point.

Concrete steps:
- [ ] Detect affine mapping (`LinearMapping` or mapping Hessian is identically zero) and:
  - compute `J`, `invJ`, `detJ` once,
  - replicate across qpts,
  - compute physical gradients once for P1 and reuse across qpts.
- [ ] Avoid per-element allocation/conversion when creating mappings:
  - remove per-element `std::vector<math::Vector<...>> node_coords` allocations/copies,
  - reuse buffers or change `MappingFactory` to accept spans of POD coords.

Acceptance criteria:
- For Tet4: one Jacobian/invJ computation per element (not per qpt).

### 2.4 Compute only what the kernel actually needs (`RequiredData`)

Concrete steps:
- [ ] Audit `RequiredData` emitted by the Forms compiler for Navier–Stokes (cell + boundary terms).
- [ ] In `prepareContextCell/Face`, branch early based on `RequiredData`:
  - don’t compute `physical_points` unless needed,
  - don’t compute normals unless boundary/face and needed,
  - don’t compute Hessians unless requested.
- [ ] Ensure `RequiredData::None` is not treated as “compute everything” for performance-critical kernels; make it explicit.

---

## 3) Make the JIT kernel do “svSolver-style” work: one pass, fused, vectorized

### 3.1 Fuse residual + tangent (Jacobian) into one JIT kernel

svSolver computes residual and tangent contributions in a single integration loop.
The OOP path currently compiles and calls residual and tangent separately for `SymbolicNonlinearFormKernel`.

Concrete steps:
- [ ] Add a “fused output” compilation mode in `JITCompiler`/`LLVMGen`:
  - one kernel writes **both** `output.element_matrix` and `output.element_vector`,
  - share intermediate results (gradients, stresses, stabilization parameters, etc.) in registers.
- [ ] Update `JITKernelWrapper` so that when `want_matrix && want_vector` it calls the fused kernel once.

Acceptance criteria:
- One JIT call per element for residual+Jacobian assembly (cell).

### 3.2 Specialize kernels by element/space/qpt counts and unroll hot loops

Concrete steps:
- [ ] Extend the cache key and grouping so kernels specialize on:
  - element type, polynomial order, value dims, `n_qpts`, and domain (cell/boundary/face).
- [ ] Generate specialized kernels for the most common `n_qpts` values and unroll the qpt loop.
- [ ] Add a specialization policy knob in `JITOptions` (e.g., “specialize <= 27 qpts”).

### 3.3 Ensure LLVM sees alignment, restrict/noalias, and stable loop structure

Concrete steps:
- [ ] In `LLVMGen`, emit:
  - `noalias`/`nonnull` on pointer arguments where valid,
  - alignment assumptions from `KernelArgsV3.pointer_alignment_bytes`,
  - `llvm.assume` where needed,
  - vectorization metadata for qpt loops.
- [ ] Keep the kernel’s memory access patterns SoA-friendly (contiguous qpt-major or dof-major as appropriate).

### 3.4 Avoid external calls in performance mode

Concrete steps:
- [ ] Add a strict JIT mode that rejects:
  - `FormExprType::Coefficient` (callback),
  - non-inlinable constitutive models,
  - derivatives of opaque functions.
- [ ] Provide explicit inlinable constitutive implementations for common fluid models:
  - Newtonian (constant `mu`),
  - Carreau–Yasuda (if needed), etc.

---

## 4) Reduce per-element invocation overhead and unlock batching (npro-style)

Even a perfect per-element kernel can lose to svSolver if we pay too much overhead per call.

Concrete steps:
- [ ] Remove repeated RTTI/dynamic_cast in `JITKernelWrapper::compute*`:
  - store typed pointers at construction time based on `kind_`.
- [ ] Make alignment checks configurable:
  - keep `validate_alignment=true` for debug/testing,
  - allow `false` in optimized production runs once validated.
- [ ] Implement **batched element kernels**:
  - new ABI (e.g., `CellKernelArgsBatchV1`) that processes `B` elements per call,
  - pack SoA arrays for basis/geometry/solution to enable vectorization across elements,
  - choose `B` as a multiple of SIMD width (and/or cache-friendly chunk size).
- [ ] Update assemblers to assemble in blocks (like svSolver’s `npro`):
  - prefetch the next block’s geometry/DOFs,
  - amortize constraint lookups and backend insertion.

Acceptance criteria:
- Measurable reduction in overhead per element (fewer function calls, fewer branches, better CPI).

---

## 5) Speed up gather/scatter and constraints (often the real bottleneck)

### 5.1 Faster local solution gathers

Concrete steps:
- [ ] Provide a fast gather path when DOFs are contiguous (memcpy) and fall back otherwise.
- [ ] For common vector-valued H1 spaces, store component-wise DOF ordering that improves locality.

### 5.2 Use constraint distribution batch APIs and fast paths

`constraints::ConstraintDistributor` supports batch distribution, but typical loops distribute per element.

Concrete steps:
- [ ] Add a “no-constraints fast path” when the element’s DOF set has no constrained DOFs.
- [ ] Use `ConstraintDistributor` batch APIs to distribute multiple elements together.
- [ ] Cache constraint expansions per element “shape” (same pattern repeats across mesh for H1).

### 5.3 Optimize global insertion (`GlobalSystemView`)

Concrete steps:
- [ ] Ensure backends implement `addMatrixEntries(row_dofs, col_dofs, block)` efficiently (single call per element).
- [ ] For threaded assembly, prefer thread-local accumulation + bulk flush when the backend is not thread-safe.
- [ ] Favor block-structured sparse formats (BSR) for vector-valued systems (3×3 velocity blocks + pressure coupling).

Longer-term (if global assembly dominates):
- [ ] Switch to matrix-free Jacobian application (`MatrixFreeAssembler` / `MatrixFreeOperator`) to avoid global matrix construction for Krylov methods.

---

## 6) Boundary/functional kernels: don’t let faces become the new bottleneck

Concrete steps:
- [ ] Add JIT acceleration for functional kernels used by coupled BCs (boundary integrals / auxiliary ODEs).
- [ ] Apply the same caching + zero-copy context strategy to `prepareContextFace`.
- [ ] Batch boundary faces by marker and element type to improve locality.

---

## 7) Correctness, stability, and guardrails (so performance work is safe)

Concrete steps:
- [ ] Add unit tests that compare interpreter vs JIT for:
  - cell bilinear/linear forms,
  - residual+tangent (symbolic) for a nonlinear form,
  - boundary integrals,
  - coupled slot terminals (`ParameterRef`, `BoundaryIntegralRef`, `AuxiliaryStateRef`),
  - material state loads/stores.
- [ ] Add “JIT-fast compliance” diagnostics that print the first offending subexpression when strict compilation fails.
- [ ] Ensure fallback-to-interpreter is safe, but make it **impossible to silently fall back** in performance benchmarking mode.

---

## 8) Recommended implementation order (maximize speedups early)

1. **Get onto the real JIT path** for fluid: symbolic tangents + strict JIT + precompile.
2. **Kill the big overhead:** remove scratch→context copies and basis-per-qpt evaluation.
3. **Affine fast paths:** Tet4/linear mapping reuse J/invJ and constant gradients.
4. **Fuse residual+tangent** into one JIT kernel invocation.
5. **Batch elements** (npro-style) + optimize scatter/constraints.
6. Add boundary/functional JIT and batch boundary faces.

