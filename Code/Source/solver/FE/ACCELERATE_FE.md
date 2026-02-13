# FE Library Performance & Memory Optimization Catalog

Comprehensive catalog of concrete changes to improve speed and memory efficiency
across the FE library (~234K LOC, 15 modules). Organized by impact tier; each item
includes file:line references, estimated impact, and implementation effort.

---

## Tier 1 — Critical Hot-Path Optimizations

These changes affect code executed per-quadrature-point or per-element inside the
assembly loop. Even small constant-factor improvements here multiply across
millions of element evaluations.

### 1.1 Enable LLVM JIT for Coupled Forms ✅

**Status:** DONE. `FE_ENABLE_LLVM_JIT=ON` set in both `build-fe-tests` and `build/svMultiPhysics-build` CMake caches. LLVM 14 linked. JIT initializes successfully at runtime. Note: boundary form JIT has a known bug in `ExternalCalls.cpp:44` (null external call table); NonlinearFormKernel falls back to interpreter as expected.

The entire JIT pipeline (FormExpr -> FormIR -> KernelIR -> LLVM IR -> native code)
is wired up for `SymbolicNonlinearFormKernel`, including coupled multi-field forms.
The `maybeWrapForJIT` call exists at `FormsInstaller.cpp:519`, and all commonly used
FormExprTypes (including `JacobianInverse`, `EffectiveTimeStep`,
`DoubleContraction`, etc.) pass JIT validation in `AllowExternalCalls` mode.

**Change:** Set `FE_ENABLE_LLVM_JIT=ON` in the CMake build (requires LLVM as a
build dependency). No FE library code changes needed for forms that use only
constant-valued parameters (no non-inlinable constitutive models).

**Files:**
- `CMakeLists.txt:63` — `option(FE_ENABLE_LLVM_JIT ... OFF)` -> ON
- `cmake/EnableLLVMJIT.cmake` — finds and links LLVM

**Impact:** 3-10x speedup on assembly kernels (eliminates interpreter switch
dispatch, enables LLVM O3 optimizations, SIMD vectorization). Assembly is typically
25-50% of total solve time, so expect significant end-to-end speedup.

**Effort:** Low (build config change + verification).

---

### 1.2 Eliminate Interpreter Switch Dispatch Overhead ✅

**Status:** DONE. Added 7 category dispatch functions (`evalRealDispatchParams`, `evalRealDispatchFields`, `evalRealDispatchDG`, `evalRealDispatchUnaryOps`, `evalRealDispatchConstructors`, `evalRealDispatchBinaryAlgebra`, `evalRealDispatchDifferential`) covering 76 of ~106 FormExprTypes. Dispatch table updated with 62 new registrations (was 14). Remaining advanced ops (smooth approximations, matrix functions, eigenvalue/spectral, history, integral symbols) still fall through to `evalRealDispatchFallback`.

**Previous state:** `evalReal()` in `FormKernels.cpp:4704-7788` was a monolithic
~3000-line, 261-case switch statement. A dispatch table existed at lines 4672-4694
but only 14 of 261 operations were directly dispatched — the remaining 247 all hit
`evalRealDispatchFallback` (line 4669), which called the full switch anyway.

**Issues:**
- Branch prediction catastrophe: 261 cold/hot cases interleaved
- No spatial locality of related operations
- Dispatch table is dead weight (99.5% fallback rate)

**Changes:**
1. **Populate the dispatch table fully** — add function pointers for all 261 ops
   instead of routing through `evalRealDispatchFallback`. This eliminates the switch
   entirely when all ops are covered.
2. **Group hot cases** — profile to identify the 20-30 most common FormExprTypes in
   typical forms (Constant, Add, Mul, Gradient, TestFunction, TrialFunction,
   InnerProduct, etc.) and ensure their function pointers are contiguous in memory
   for i-cache locality.
3. **Consider computed goto** — replace the switch with `goto *dispatch[type]` for
   compilers that support it (GCC/Clang), eliminating comparison chains.

**Files:**
- `Forms/FormKernels.cpp:4664-4694` — dispatch table definition
- `Forms/FormKernels.cpp:4704-7788` — the switch implementation

**Impact:** 15-30% speedup on interpreted assembly (removes branch misprediction
penalty of ~15 cycles per node evaluation, hundreds of nodes per element).

**Effort:** Medium. Requires extracting each case into a standalone function.

---

### 1.3 Eliminate Virtual `childrenShared()` in Interpreter Hot Path ✅

**Status:** DONE. Added `childCount()` and `child(index)` virtual methods to `FormExprNode` base class, with overrides in 15 node subclasses. Replaced 80 of 81 `childrenShared()` call sites in `FormKernels.cpp` with the non-allocating `child()`/`childCount()` pattern. One remaining `childrenShared()` call at the constitutive inlining path (~line 626) requires shared ownership and was left as-is.

**Previous state:** The interpreter called `node.childrenShared()` 81 times across the eval switch. Each
call was virtual (`FormExprNode::childrenShared()` at `FormExpr.h:378`) and returned a
newly allocated `std::vector<std::shared_ptr<FormExprNode>>` — a heap allocation per
call.

For a depth-5 expression tree with 10 quadrature points: ~500 virtual calls + 500
vector allocations per element.

**Changes:**
1. **Add non-virtual `children()` returning `std::span`** — store child pointers in
   a flat array inside each node (or use small-buffer optimization for 1-3 children).
   Expose a `std::span<const FormExprNode* const>` accessor that avoids allocation.
2. **Cache children in FormIR** — when compiling FormExpr to FormIR, flatten the tree
   into a linear array of `(type, child_offsets[])` tuples. The interpreter then
   indexes into this array instead of chasing pointers.
3. **Pre-resolve child pointers** per integrand at compile time so the interpreter
   never calls `childrenShared()` at all.

**Files:**
- `Forms/FormExpr.h:378-380` — virtual `childrenShared()`
- `Forms/FormExpr.cpp:548-551` — UnaryNode returns `{child_}` (vector allocation)
- `Forms/FormExpr.cpp:670-684` — BinaryNode returns `{left_, right_}`
- `Forms/FormKernels.cpp` — 81 call sites (search `childrenShared`)

**Impact:** 10-20% speedup on interpreted assembly by removing virtual dispatch +
heap allocation in the innermost loop.

**Effort:** Medium-High. Requires refactoring the FormExprNode hierarchy.

---

### 1.4 Optimize Dual Number Arithmetic for SIMD ✅

**Status:** DONE (partial — workspace pre-allocation). Added `DualWorkspace::reserve(num_slots)` method that pre-allocates capacity to avoid dynamic growth during evaluation. Alignment was already correct (64-byte via `AlignedAllocator`). SoA layout and explicit SIMD intrinsics deferred as higher-effort follow-up.

**Previous state:** `SymbolicNonlinearFormKernel` uses dual numbers for automatic differentiation. Each
arithmetic operation (add, mul, div) loops over `n_trial_dofs` derivatives:

```cpp
// Dual.h:160-168
for (std::size_t k = 0; k < n; ++k) {
    out.deriv[k] = a.deriv[k] * b.value + a.value * b.deriv[k];
}
```

**Issues:**
- `SVMP_DUAL_VECTORIZE` pragma hint exists but derivative arrays are not guaranteed
  aligned to SIMD boundaries
- Per-operation workspace allocation via `env.ws->alloc()` (70+ call sites in
  `FormKernels.cpp:1059-1102`) creates ~11-16 KB of temporaries per quadrature point
- No batch evaluation across multiple quadrature points

**Changes:**
1. **Align derivative storage to `kFEPreferredAlignmentBytes` (64 bytes)** — ensure
   the workspace allocator returns 64-byte-aligned spans. Update `Dual.h` to use
   `AlignedAllocator`.
2. **Pre-allocate fixed workspace per element** — compute `max_dual_temps` at IR
   compile time and allocate once per element instead of per-operation.
3. **Batch quadrature points** — evaluate dual operations for all qpts simultaneously
   using SoA layout: `deriv[k][qpt]` instead of `deriv[qpt][k]`. This turns the
   inner `k` loop into a SIMD-friendly stride-1 access pattern.
4. **Explicit SIMD intrinsics for mul/add** — for `n_trial >= 4`, use AVX2/AVX-512
   vector operations directly instead of relying on auto-vectorization.

**Files:**
- `Forms/Dual.h:59-109` — Dual struct and workspace allocator
- `Forms/Dual.h:140-200` — arithmetic operations (add, sub, mul, div)
- `Forms/FormKernels.cpp:1059-1102` — `makeDualConstant` + alloc call sites
- `Forms/FormKernels.cpp:2126-2230` — per-trial-DOF nested loops

**Impact:** 2-4x speedup on dual-number evaluation (interpreter path). For 10-DOF
elements, SIMD can process 4-8 derivatives per cycle vs 1 scalar.

**Effort:** Medium. Workspace pre-allocation is easy; SoA layout is harder.

---

### 1.5 Fuse Coupled Block Assembly into Single Mesh Traversal ✅

**Status:** DONE. Added `CellBlockSpec` struct and `assembleCellBlocksFused()` virtual method to `Assembler.h` with a default fallback (N² passes). Implemented an optimized single-pass override in `StandardAssembler.cpp` (~200 lines) that processes all N² block kernels per cell in one mesh traversal. Modified `SystemAssembly.cpp` to build a `vector<CellBlockSpec>` from all cell terms and call the fused method instead of looping over terms individually. Verified: 698/698 assembly tests pass, all MPI tests pass (np=2, np=4), vortex shedding 2D coupled Navier-Stokes runs correctly with 2 MPI ranks. Performance: OOP solver 42.4s vs legacy 54.1s wall clock (1.28x faster) on 3-step vortex shedding with 2 MPI ranks.

Currently, a coupled N×N block system (e.g., a 2-field problem) produces N²
separate kernels, each traversing the mesh independently. With the
unified operator optimization already done, this is N² passes per Newton iteration
(e.g., 4 passes for a 2-field system).

**Change:** Implemented `assembleCellBlocksFused()` in `Assembler.h` (base with fallback) and `StandardAssembler.cpp` (optimized single-pass). Modified `SystemAssembly.cpp` to use the fused path for all cell terms.

**Files:**
- `Assembly/Assembler.h` — `CellBlockSpec` struct + `assembleCellBlocksFused()` virtual with default fallback
- `Assembly/StandardAssembler.h` — override declaration
- `Assembly/StandardAssembler.cpp` — optimized single-pass fused assembly implementation
- `Systems/SystemAssembly.cpp` — builds `CellBlockSpec` vector and calls fused method

**Impact:** ~(N²-1)/N² speedup on assembly phase (N² passes -> 1 pass). Assembly context
setup (Jacobian cache lookup, basis evaluation, DOF gathering) is ~40% of per-cell
cost, so eliminating 3 redundant setups is significant.

**Effort:** High. Requires new assembler infrastructure.

---

### 1.6 Cache Affine Element Jacobians in IsoparametricMapping ✅

**Status:** DONE. Added `isAffine()` override and `detectAffine()` private method to `IsoparametricMapping`. Detection evaluates the Jacobian at the reference center plus strategically chosen offset points (2 total for 2D, 4 for 3D) with sign-flipped coordinates to rigorously determine if the Jacobian is constant. The result is cached as `bool is_affine_` at construction time. For bilinear Quad4 on parallelogram geometry and trilinear Hex8 on parallelepiped geometry, `isAffine()` now returns `true`, triggering the existing single-evaluation optimization in `JacobianCache::compute()`. Added 8 unit tests covering: identity/parallelogram/trapezoid Quad4, identity/parallelepiped/distorted Hex8, curved Quad8, and JacobianCache integration. All 84 geometry tests pass.

`JacobianCache` correctly detects affine elements via `mapping.isAffine()` and
computes the Jacobian once. However, `IsoparametricMapping` previously lacked an
`isAffine()` override — it always returned `false` (default in `GeometryMapping.h:71`).

This meant Quad4/Hex8 elements with affine geometry (parallelogram/parallelepiped)
still computed Jacobians at every quadrature point.

**Changes:**
1. **Added `isAffine()` override to `IsoparametricMapping`** — detects affine geometry
   by comparing Jacobians at center and offset reference points.
2. **Cached detection result** — computed once at construction, stored as `bool is_affine_`.

**Files:**
- `Geometry/IsoparametricMapping.h` — added `bool isAffine() const noexcept override`, `bool detectAffine() const`, `bool is_affine_`
- `Geometry/IsoparametricMapping.cpp` — implemented `detectAffine()` with dimension-aware test points
- `Geometry/JacobianCache.cpp:56-66` — already handles affine case correctly (no changes needed)

**Impact:** 20-30% speedup on geometry computation for affine Quad4/Hex8 elements
(common in structured meshes). Reduces per-element Jacobian evaluations from
n_qpts to 1.

**Effort:** Low.

---

## Tier 2 — Memory Layout & Cache Efficiency

### 2.1 Unify AssemblyContext Data Layout

AssemblyContext stores data in inconsistent layouts:

| Data | Layout | Location |
|------|--------|----------|
| Test basis values | `[dof * n_qpts + qpt]` (dof-major) | `AssemblyContext.h:1416` |
| Physical gradients | `[qpt * n_dofs + dof]` (qpt-major) | `AssemblyContext.h:548` |
| Solution coefficients | `[i * n_qpts + qpt]` (dof-major) | `AssemblyContext.h` |

This stride mismatch causes cache misses when kernels access both basis values and
gradients at the same quadrature point.

**Change:** Standardize all per-element data to a single layout. The JIT ABI uses
`[qpt * n_dofs + dof]` (qpt-major); adopt this consistently. Add a transpose step
in context setup for data that arrives in dof-major order.

**Files:**
- `Assembly/AssemblyContext.h:1416-1448` — storage declarations
- `Assembly/AssemblyContext.cpp:65-100` — arena allocation

**Impact:** 10-20% speedup on kernel evaluation by eliminating stride conflicts.

**Effort:** Medium. Requires updating all accessors and JIT kernel argument packing.

---

### 2.2 Pack Jacobian Data Per-Quadrature-Point

JacobianCache stores J, J_inv, and detJ in three separate vectors:

```cpp
// JacobianCache.h:27-29
std::vector<Matrix<Real,3,3>> J;      // [qpt]
std::vector<Matrix<Real,3,3>> J_inv;  // [qpt]
std::vector<Real> detJ;               // [qpt]
```

Accessing all three for the same quadrature point requires three non-contiguous
memory reads.

**Change:** Pack into a single struct-per-qpt array:
```cpp
struct JacobianQPData { Matrix<3,3> J; Matrix<3,3> J_inv; Real detJ; };
std::vector<JacobianQPData> data;  // [qpt]
```

Also add `J_invT` (inverse transpose) to avoid recomputing the transpose in
`PushForward::gradient()` at every call (`GeometryMapping.h:119`).

**Files:**
- `Geometry/JacobianCache.h:27-29` — storage
- `Geometry/JacobianCache.cpp:56-98` — population
- `Geometry/PushForward.cpp:14-24` — uses J_invT

**Impact:** 10-15% memory bandwidth improvement for Jacobian access in assembly.

**Effort:** Low-Medium.

---

### 2.3 Fix Alignment Mismatch: Vector/Matrix vs kFEPreferredAlignmentBytes

`Core/Alignment.h:15` defines `kFEPreferredAlignmentBytes = 64` (correct for
AVX-512). But `Math/Vector.h:41` and `Math/Matrix.h:44` both use `alignas(32)`.

On AVX-512 targets, this forces unaligned loads with up to 4x penalty.

**Change:** Replace `alignas(32)` with `alignas(kFEPreferredAlignmentBytes)` in
Vector and Matrix.

**Files:**
- `Math/Vector.h:41` — `alignas(32)` -> `alignas(kFEPreferredAlignmentBytes)`
- `Math/Matrix.h:44` — same
- `Math/Matrix.h:777,1013` — 2x2/3x3 specializations

**Impact:** Up to 2x speedup for SIMD operations on AVX-512 targets.

**Effort:** Trivial.

---

### 2.4 Consolidate Multi-Field Solution Data

AssemblyContext allocates 8-10 separate vectors per field for solution data:

```cpp
// AssemblyContext.h:1451-1468
struct FieldSolutionData {
    JITAlignedVector<Real> values{};
    JITAlignedVector<Vector3D> vector_values{};
    JITAlignedVector<Vector3D> gradients{};
    // ... 8+ separate allocations
};
```

For 3+ fields this creates 24+ separate heap allocations per context.

**Change:** Use a single packed buffer with stride offsets for all field data.
Allocate one contiguous block sized for `n_fields * max_qpts * max_components` and
index into it.

**Files:**
- `Assembly/AssemblyContext.h:1451-1468` — FieldSolutionData struct
- `Assembly/AssemblyContext.cpp` — allocation

**Impact:** 50-70% reduction in field data allocation overhead for multi-field
systems.

**Effort:** Medium.

---

### 2.5 Eliminate SpaceCache -> BasisCache Vector Copy

`SpaceCache.cpp:54` copies the entire basis values vector from BasisCache:

```cpp
data.basis_values = entry.scalar_values;  // Full vector copy
```

**Change:** Return a `std::span` or `const&` reference to BasisCache data instead
of copying.

**Files:**
- `Spaces/SpaceCache.cpp:54` — the copy
- `Spaces/SpaceCache.h:61-62` — storage declaration

**Impact:** Eliminates one O(n_dofs * n_qpts) allocation + copy per cache miss.

**Effort:** Low.

---

## Tier 3 — Algorithmic Improvements

### 3.1 Pre-Compute Assembly (row,col) -> NNZ Index Map

FSILS matrix assembly uses per-entry binary search within each CSR row:

```cpp
// FsilsMatrix.cpp:1049
const auto it = std::lower_bound(begin, finish, col_internal);
```

For dense element matrices (e.g., 27-node hex with 3 DOFs = 81 DOFs/element),
this is 81*81 = 6561 binary searches per element.

**Change:** Pre-compute a `(row, col) -> nnz_index` lookup table during sparsity
pattern finalization. Use it for O(1) scatter during assembly.

**Files:**
- `Backends/FSILS/FsilsMatrix.cpp:1002-1077` — `addValue()` with binary search
- `Backends/FSILS/FsilsMatrix.cpp:262-298` — assembly view loop

**Impact:** 2-5x speedup on matrix scatter (currently ~15% of assembly time).

**Effort:** Medium.

---

### 3.2 Cache FSILS Work Matrix Across Solves

`FsilsLinearSolver.cpp:160` copies the entire matrix on every `solve()` call because
FSILS preconditioning modifies values in-place:

```cpp
std::vector<Real> values_work(value_count);  // nnz * dof * dof
std::copy(A->fsilsValuesPtr(), A->fsilsValuesPtr() + value_count, values_work.data());
```

For nnz=1M, dof=3: copies 72 MB per solve.

**Change:** Allocate the work buffer once and reuse across solves. Only re-copy when
the matrix values have been modified (track a "dirty" flag).

**Files:**
- `Backends/FSILS/FsilsLinearSolver.cpp:160-170` — work buffer allocation + copy

**Impact:** Eliminates 72+ MB memcpy per linear solve (typically 5-25 solves per
time step).

**Effort:** Low.

---

### 3.3 Eliminate Redundant Residual Norm Copies (FSILS)

`NewtonSolver.cpp:150-174` copies the entire residual vector into a scratch buffer
before computing the norm, because FSILS requires overlap accumulation:

```cpp
std::copy(src.begin(), src.end(), dst.begin());  // O(n_dofs) copy
scratch_fs->accumulateOverlap();                  // MPI reduction
return scratch_fs->norm();
```

This happens at every Newton iteration (including line search trials).

**Change:** Add a `normWithOverlap()` method to the FSILS vector that accumulates
and computes the norm in-place without requiring a separate scratch copy.

**Files:**
- `TimeStepping/NewtonSolver.cpp:150-174` — `residualNormForConvergence()`
- `Backends/FSILS/FsilsVector.cpp` — add `normWithOverlap()`

**Impact:** Eliminates O(n_dofs) copy per Newton iteration.

**Effort:** Low.

---

### 3.4 Use Cuthill-McKee DOF Numbering by Default

Default DOF numbering is sequential (`DofHandler.h:125`), which produces matrix
bandwidth ~O(n_dofs^(2/d)). Cuthill-McKee BFS reordering reduces bandwidth by
30-50%, improving:
- Sparse factorization fill-in
- Iterative solver convergence
- Cache hit rate during matrix-vector products

**Change:** Set `DofNumberingStrategy::CuthillMcKee` as the default.

**Files:**
- `Dofs/DofHandler.h:125` — default strategy
- `Dofs/DofNumbering.cpp:199+` — CM implementation (already exists)

**Impact:** 10-30% improvement in linear solver time.

**Effort:** Trivial (one-line default change + verification).

---

### 3.5 Replace History Vector Repacking with Raw Memory Copy

`TimeHistory::repack()` uses per-DOF assembly view loops:

```cpp
// TimeHistory.cpp:37-41
for (GlobalIndex dof = 0; dof < dst.size(); ++dof) {
    dst_view->addVectorEntry(dof, src_view->getVectorEntry(dof), AddMode::Insert);
}
```

For FSILS backend, each `addVectorEntry` involves index mapping overhead.

**Change:** Add a `copyFrom()` method to `GenericVector` that performs raw memory
copy (or `memcpy` for same-backend vectors), bypassing the assembly view.

**Files:**
- `TimeStepping/TimeHistory.cpp:319-361` — `repack()`
- `Backends/*/Vector.cpp` — add `copyFrom()` override

**Impact:** 10-100x speedup on history repacking (from O(n_dofs * view_overhead) to
O(n_dofs)).

**Effort:** Low.

---

### 3.6 Use Ring Buffer for Time History

`TimeHistory::acceptStep()` shifts all history vectors by copying:

```cpp
// TimeHistory.cpp:371-374
for (std::size_t i = history_.size(); i-- > 1;) {
    copySpan(history_[i]->localSpan(), history_[i-1]->localSpan());
}
```

For history depth 3 with 100K DOFs: 300K doubles copied per time step.

**Change:** Use a ring buffer (circular index into fixed-size array of vectors)
instead of shifting. `acceptStep()` becomes O(1) — just advance the ring index.

**Files:**
- `TimeStepping/TimeHistory.cpp:363-386` — `acceptStep()`
- `TimeStepping/TimeHistory.h` — storage

**Impact:** Eliminates O(history_depth * n_dofs) copy per time step.

**Effort:** Low-Medium.

---

### 3.7 Matrix-Matrix Multiply Loop Order

`Matrix.h:1330-1342` uses ijk loop order for matrix-matrix multiplication:

```cpp
for (i) for (j) { sum = 0; for (k) sum += A(i,k) * B(k,j); }
```

Inner loop accesses `B(k,j)` with stride `N` on a row-major matrix, causing L1
cache misses.

**Change:** Use ikj loop order:
```cpp
for (i) for (k) { a_ik = A(i,k); for (j) C(i,j) += a_ik * B(k,j); }
```

Inner loop now accesses `B(k,j)` with stride 1 (row-wise) and `C(i,j)` with
stride 1.

**Files:**
- `Math/Matrix.h:1330-1342` — `operator*(Matrix, Matrix)`

**Impact:** 2-3x speedup for matrix-matrix products (mostly affects 4x4+ matrices;
2x2 and 3x3 have specialized implementations).

**Effort:** Trivial.

---

## Tier 4 — Memory Allocation & Lifetime

### 4.1 Pre-Allocate Previous Solution History Vectors

`StandardAssembler.cpp:698-720` resizes `previous_solutions_` on each call to
`setPreviousSolutionK()`:

```cpp
if (previous_solutions_.size() < static_cast<std::size_t>(k)) {
    previous_solutions_.resize(static_cast<std::size_t>(k));
}
```

**Change:** Pre-allocate in `initialize()` to the maximum history depth.

**Files:**
- `Assembly/StandardAssembler.cpp:698-720` — resize
- `Assembly/StandardAssembler.h` — add `setMaxHistoryDepth()`

**Impact:** Eliminates potential reallocation during assembly.

**Effort:** Trivial.

---

### 4.2 Pre-Size Constraint Scratch Vectors

`StandardAssembler.h:544-550` declares scratch vectors for constraint distribution
that grow during assembly:

```cpp
std::vector<GlobalIndex> scratch_rows_;
std::vector<GlobalIndex> scratch_cols_;
std::vector<Real> scratch_matrix_;
```

**Change:** Pre-allocate to `max_dofs_per_element^2` in `initialize()`.

**Files:**
- `Assembly/StandardAssembler.h:544-550` — declarations
- `Assembly/StandardAssembler.cpp` — `initialize()`

**Impact:** Eliminates reallocation stalls in early assembly iterations.

**Effort:** Trivial.

---

### 4.3 Reduce FormExpr AST shared_ptr Overhead

Every non-terminal FormExprNode uses `std::shared_ptr` for children:

```cpp
// FormExpr.cpp:553
std::shared_ptr<FormExprNode> child_;  // UnaryNode: 16 bytes overhead
// FormExpr.cpp:670-684
std::shared_ptr<FormExprNode> left_, right_;  // BinaryNode: 32 bytes overhead
```

For a 100-node expression tree: ~1600 bytes of reference-counting overhead +
fragmented heap allocations.

**Change:** Use an arena allocator for FormExprNode instances. Allocate nodes from a
contiguous pool; use `FormExprNode*` (raw pointer) for children since the arena
owns all nodes. This eliminates per-node shared_ptr overhead and improves cache
locality during tree traversal.

**Files:**
- `Forms/FormExpr.h:320-385` — FormExprNode base class
- `Forms/FormExpr.cpp:28-870` — all node subclasses

**Impact:** 30-50% reduction in AST memory + faster tree traversal from cache
locality. Primarily affects compile time, not assembly time.

**Effort:** High. Requires ownership model redesign.

---

### 4.4 Eliminate FSILS Ghost Sync Temporary Allocations

`FsilsVector.cpp:265-315` allocates a temporary vector on every `updateGhosts()`:

```cpp
std::vector<double> u_internal(dof * nNo, 0.0);  // O(n_dofs) allocation
```

**Change:** Store the reorder buffer as a member variable, allocated once.

**Files:**
- `Backends/FSILS/FsilsVector.cpp:265-315` — `updateGhosts()`
- `Backends/FSILS/FsilsVector.cpp:317-361` — `accumulateOverlap()` (same pattern)

**Impact:** Eliminates O(n_dofs) allocation per ghost sync (called many times per
solve).

**Effort:** Low.

---

### 4.5 Bound JIT In-Memory Kernel Cache

`FormKernels.h:128` specifies `max_in_memory_kernels{0}` (unbounded):

```cpp
struct JITOptions {
    std::size_t max_in_memory_kernels{0};  // 0 = unlimited
};
```

For applications with many different element types or quadrature rules, the cache
can grow without bound.

**Change:** Set a reasonable default (e.g., 1024) and implement LRU eviction.

**Files:**
- `Forms/FormKernels.h:128` — default value
- `Forms/JIT/JITKernelWrapper.cpp` — cache management

**Impact:** Prevents unbounded memory growth in long-running simulations.

**Effort:** Low-Medium.

---

### 4.6 Optimize AssemblyContext Arena Resize

`AssemblyContext.cpp:164-261` snapshots all ~25 data arrays, reallocates the arena,
then restores:

```cpp
const auto quad_points = snapshot(quad_points_);  // vector<> copy
// ... 20+ more snapshots ...
reserve(...);
restore(quad_points_, quad_points);  // copy back
```

**Change:** Use `realloc`-style in-place growth or move semantics to avoid the
double-copy.

**Files:**
- `Assembly/AssemblyContext.cpp:164-261` — `ensureArenaCapacity()`

**Impact:** Eliminates ~50 vector copies on arena resize (rare but expensive when
hit).

**Effort:** Medium.

---

## Tier 5 — Backend-Specific Optimizations

### 5.1 Batch Element Matrix Insertion (All Backends)

Constraint distribution (`ConstraintDistributor.cpp:156-242`) inserts matrix entries
one-at-a-time via `addValue()`:

```cpp
for (i) for (j) {
    for (r_entry : row_constraint)
        for (c_entry : col_constraint)
            matrix->addValue(r_master, c_master, weight * value);
}
```

**Change:** Accumulate the expanded element matrix locally, then insert as a single
batch via a new `addElementMatrix(rows[], cols[], values[])` backend method.

**Files:**
- `Constraints/ConstraintDistributor.cpp:156-242` — per-entry insertion
- `Backends/*/Matrix.cpp` — add batch insert method

**Impact:** 2-3x speedup on constrained assembly.

**Effort:** Medium.

---

### 5.2 PETSc Dual-Buffer Elimination

PETSc vectors maintain both a `Vec` and a `std::vector<Real>` cache:

```cpp
// PetscVector.h:64-71
mutable Vec vec_{nullptr};
mutable std::vector<Real> local_cache_{};
mutable bool local_cache_valid_{false};
mutable bool local_cache_dirty_{false};
```

Every state change requires synchronization between the two buffers.

**Change:** Eliminate the local cache; use `VecGetArray`/`VecRestoreArray` directly
for all access. If read-only span access is needed, use `VecGetArrayRead`.

**Files:**
- `Backends/PETSc/PetscVector.h:64-71` — dual buffer
- `Backends/PETSc/PetscVector.cpp:140-150` — sync methods

**Impact:** 2x memory reduction for PETSc vectors + eliminates sync overhead.

**Effort:** Medium.

---

### 5.3 FSILS Node Mapping Optimization

For non-contiguous owned nodes, `globalNodeToOld()` uses binary search:

```cpp
// FsilsShared.h:61-77
const auto it = std::lower_bound(owned_nodes.begin(), owned_nodes.end(), global_node);
```

**Change:** Use a hash map for non-contiguous cases, or ensure contiguous node
ownership during DOF distribution.

**Files:**
- `Backends/FSILS/FsilsShared.h:61-77` — `globalNodeToOld()`

**Impact:** O(1) instead of O(log n) per node lookup in assembly.

**Effort:** Low.

---

## Tier 6 — Setup-Time Optimizations

These affect initialization cost (once per simulation), not per-step cost.

### 6.1 Cache Constraint Queries During Assembly

`ConstraintDistributor.cpp:159` performs hash table lookups inside an O(n_dofs^2)
loop:

```cpp
for (i in n_rows) {
    const auto row_constraint = constraints_->getConstraint(row_dofs[i]);
    for (j in n_cols) {
        const auto col_constraint = constraints_->getConstraint(col_dofs[j]);
```

**Change:** Pre-scan element DOFs for constraints before entering the nested loop.
Store results in a small local array.

**Files:**
- `Constraints/ConstraintDistributor.cpp:156-242`
- `Constraints/AffineConstraints.h:672` — `slave_to_index_` hash map

**Impact:** Reduces hash lookups from O(n_dofs^2) to O(n_dofs) per element.

**Effort:** Low.

---

### 6.2 Replace SparsityFactory String-Keyed Cache

`SparsityFactory.h:651` uses `std::unordered_map<std::string, ...>` for caching
sparsity patterns, with O(n_cells * dofs/cell) key computation cost:

```cpp
mutable std::unordered_map<std::string, std::shared_ptr<SparsityPattern>> cache_;
```

**Change:** Use a `std::size_t` hash key computed from a fast fingerprint (pointer +
size + first/last few entries) instead of hashing the entire connectivity.

**Files:**
- `Sparsity/SparsityFactory.h:651-653` — cache maps
- `Sparsity/SparsityFactory.cpp:122-139` — hash computation

**Impact:** Orders of magnitude faster cache lookup.

**Effort:** Low.

---

### 6.3 Use Direct Vector Indexing for FieldRegistry

`FieldRegistry::get(FieldId)` uses O(n) linear search:

```cpp
// FieldRegistry.cpp:40-49
for (const auto& f : fields_) {
    if (f.id == id) return f;
}
```

FieldId is already a sequential integer suitable for direct indexing.

**Change:** Store fields in a `std::vector` indexed by `FieldId` for O(1) lookup.

**Files:**
- `Systems/FieldRegistry.cpp:40-49` — `get()`
- `Systems/FieldRegistry.cpp:77-88` — `has()`
- `Systems/SystemAssembly.cpp:438-439` — hot-path callers

**Impact:** Eliminates O(n_fields) scan on every term in assembly.

**Effort:** Trivial.

---

### 6.4 Cache Operator Definition Reference in Assembly Loop

`SystemAssembly.cpp:431` looks up the operator definition via string key on every
assembly call:

```cpp
const auto& def = system.operator_registry_.get(request.op);  // string hash
```

**Change:** Cache the `OperatorDefinition&` reference in the `AssemblyRequest` or
resolve it once at the start of the Newton loop.

**Files:**
- `Systems/SystemAssembly.cpp:431` — lookup
- `Systems/OperatorRegistry.h:75` — `std::unordered_map<OperatorTag, ...>`

**Impact:** Eliminates string hashing per assembly call.

**Effort:** Trivial.

---

## Tier 7 — MPI / Parallel Optimizations

### 7.1 Overlap Communication with Computation

FSILS requires at least 2 MPI calls per Krylov iteration (dot product Allreduce +
ghost update COMMU). These are synchronous.

**Change:** Use non-blocking MPI (`MPI_Iallreduce`, `MPI_Isend`/`MPI_Irecv`) and
overlap communication with local computation in the next iteration.

**Files:**
- `Backends/FSILS/FsilsVector.cpp:223-250` — dot product with Allreduce
- `Backends/FSILS/FsilsVector.cpp:265-315` — ghost update

**Impact:** Hides MPI latency; ~10-20% speedup at high rank counts.

**Effort:** High. Requires restructuring FSILS solver internals.

---

### 7.2 Pre-Allocate Ghost Contribution Buffers

`GhostContributionManager.h:502` declares ghost buffers without pre-reservation:

```cpp
std::vector<GhostBuffer> send_buffers_;
std::vector<GhostContribution> received_matrix_;
```

**Change:** Call `reserveBuffers()` during initialization based on DOF ownership
analysis.

**Files:**
- `Assembly/GhostContributionManager.h:495-506` — buffer declarations
- `Assembly/GhostContributionManager.h:424` — `reserveBuffers()` exists

**Impact:** Eliminates reallocation during first assembly.

**Effort:** Trivial.

---

### 7.3 Batch DOF Constraint Distribution for MPI

In MPI assembly, constraint distribution and ghost contribution management interact.
Currently, constraints are applied per-element, and ghost contributions are
accumulated separately.

**Change:** Batch constraint expansion at the element level before ghost contribution
management, reducing the number of ghost messages.

**Files:**
- `Assembly/StandardAssembler.cpp` — constraint + ghost interaction
- `Assembly/GhostContributionManager.h` — ghost accumulation

**Impact:** Fewer, larger MPI messages; reduced latency overhead.

**Effort:** Medium.

---

## Tier 8 — Quadrature & Basis Optimizations

### 8.1 Template Quadrature Rules by Dimension

Quadrature points are stored as 3D vectors regardless of actual dimension:

```cpp
// QuadratureRule.h:108
std::vector<QuadPoint> points_;  // QuadPoint = Vector<Real, 3>
```

For 1D rules, this wastes 16 bytes per point (2 unused coordinates).

**Change:** Template `QuadratureRule<Dim>` to store only needed coordinates.

**Files:**
- `Quadrature/QuadratureRule.h:108` — point storage

**Impact:** 33-67% memory reduction for 1D/2D quadrature rules.

**Effort:** Medium (template refactoring).

---

### 8.2 Cache Facet Normals for Affine Elements

Facet normals are recomputed per-quadrature-point even for affine elements where
they are constant.

**Change:** Detect affine facets and cache the normal vector, reusing it across all
facet quadrature points.

**Files:**
- `Geometry/ElementTransform.cpp` — facet frame computation

**Impact:** 20-40% speedup on boundary integrals for affine elements.

**Effort:** Low.

---

### 8.3 Batch Basis Evaluation

`BasisCache.cpp:85-101` evaluates basis functions one quadrature point at a time:

```cpp
for (qp = 0; qp < points.size(); ++qp) {
    basis.evaluate_values(points[qp], scalar_values);
    // copy to cache...
}
```

**Change:** Add a batch evaluation interface that evaluates all quadrature points
simultaneously, enabling SIMD across points for tensor-product bases.

**Files:**
- `Basis/BasisCache.cpp:85-101` — per-qpt evaluation
- `Basis/BasisFunction.h` — add `evaluate_values_batch()`

**Impact:** 2-4x speedup for basis evaluation on tensor-product elements.

**Effort:** Medium.

---

## Tier 9 — Interpreter Micro-Optimizations

### 9.1 Reduce Constitutive Cache Double-Hashing

`FormKernels.cpp:4309,4481` computes the constitutive cache key hash twice per call
(once for dependency mask lookup, once for result lookup).

**Change:** Compute hash once and reuse for both lookups.

**Files:**
- `Forms/FormKernels.cpp:4208-4222` — `ConstitutiveCallKeyHash`
- `Forms/FormKernels.cpp:4309,4481` — double lookup

**Impact:** 50% reduction in hashing cost for constitutive calls.

**Effort:** Trivial.

---

### 9.2 Add __builtin_expect Hints for Common FormExprTypes

In the interpreter switch, the most common operations (Constant, Add, Mul, Gradient,
TestFunction, TrialFunction, InnerProduct) should be marked as likely.

**Change:** Add `[[likely]]` attributes (C++20) to the most common cases.

**Files:**
- `Forms/FormKernels.cpp:4704-7788` — switch cases

**Impact:** Minor (5-10%) improvement in branch prediction.

**Effort:** Trivial.

---

### 9.3 Specialize SIMD Reciprocal for Double Precision

`SIMD.h:575` falls back to slow division for double reciprocal:

```cpp
static inline vec_type reciprocal(vec_type v) {
    // double: no fast path, reverts to 1.0 / x
}
```

**Change:** Implement Newton-Raphson refinement from single-precision approximation:
```cpp
// Initial approximation (float)
__m256 approx = _mm256_rcp_ps(_mm256_cvtpd_ps(v));
// Refine to double precision
__m256d x = _mm256_cvtps_pd(approx);
x = x * (2.0 - v * x);  // Newton step
x = x * (2.0 - v * x);  // Second Newton step (52-bit accuracy)
```

**Files:**
- `Math/SIMD.h:563-577` — reciprocal implementation

**Impact:** 2-3x speedup for division-heavy kernels.

**Effort:** Low.

---

## Summary Table

| ID | Change | Tier | Impact | Effort | Subsystem |
|----|--------|------|--------|--------|-----------|
| 1.1 | Enable LLVM JIT | 1 | 3-10x assembly | Low | Forms/JIT |
| 1.2 | Full dispatch table | 1 | 15-30% interp | Medium | Forms |
| 1.3 | Eliminate childrenShared() | 1 | 10-20% interp | Med-High | Forms |
| 1.4 | SIMD dual numbers | 1 | 2-4x dual eval | Medium | Forms |
| 1.5 | Fused block assembly | 1 | ~3x assembly | High | Assembly |
| 1.6 | Affine isoparametric | 1 | 20-30% geometry | Low | Geometry |
| 2.1 | Unified data layout | 2 | 10-20% kernels | Medium | Assembly |
| 2.2 | Packed Jacobian data | 2 | 10-15% bandwidth | Low-Med | Geometry |
| 2.3 | AVX-512 alignment | 2 | Up to 2x SIMD | Trivial | Math |
| 2.4 | Packed field data | 2 | 50-70% field alloc | Medium | Assembly |
| 2.5 | SpaceCache span | 2 | Eliminates copy | Low | Spaces |
| 3.1 | Pre-computed NNZ map | 3 | 2-5x scatter | Medium | Backends |
| 3.2 | Cached work matrix | 3 | -72MB/solve | Low | Backends |
| 3.3 | Residual norm no-copy | 3 | -O(n) per iter | Low | Newton |
| 3.4 | Cuthill-McKee default | 3 | 10-30% solver | Trivial | Dofs |
| 3.5 | Raw history copy | 3 | 10-100x repack | Low | TimeStepping |
| 3.6 | Ring buffer history | 3 | -O(depth*n) copy | Low-Med | TimeStepping |
| 3.7 | gemm loop order | 3 | 2-3x matmul | Trivial | Math |
| 4.1 | Pre-alloc history | 4 | Avoids realloc | Trivial | Assembly |
| 4.2 | Pre-size scratch | 4 | Avoids realloc | Trivial | Assembly |
| 4.3 | Arena FormExpr nodes | 4 | 30-50% AST mem | High | Forms |
| 4.4 | FSILS ghost buffer | 4 | -O(n) per sync | Low | Backends |
| 4.5 | Bound JIT cache | 4 | Prevents OOM | Low-Med | Forms/JIT |
| 4.6 | Arena resize opt | 4 | Rare but large | Medium | Assembly |
| 5.1 | Batch matrix insert | 5 | 2-3x constrained | Medium | Backends |
| 5.2 | PETSc no dual buffer | 5 | 2x vector mem | Medium | Backends |
| 5.3 | FSILS hash node map | 5 | O(1) vs O(log n) | Low | Backends |
| 6.1 | Cache constraint queries | 6 | O(n) vs O(n^2) | Low | Constraints |
| 6.2 | Fast sparsity cache key | 6 | Orders of mag | Low | Sparsity |
| 6.3 | FieldRegistry indexing | 6 | O(1) vs O(n) | Trivial | Systems |
| 6.4 | Cache operator def | 6 | -string hash | Trivial | Systems |
| 7.1 | Async MPI | 7 | 10-20% at scale | High | Backends |
| 7.2 | Pre-alloc ghost bufs | 7 | Avoids realloc | Trivial | Assembly |
| 7.3 | Batch constraint+ghost | 7 | Fewer MPI msgs | Medium | Assembly |
| 8.1 | Dim-templated quad | 8 | 33-67% quad mem | Medium | Quadrature |
| 8.2 | Cache facet normals | 8 | 20-40% boundary | Low | Geometry |
| 8.3 | Batch basis eval | 8 | 2-4x basis eval | Medium | Basis |
| 9.1 | Single constitutive hash | 9 | 50% hash cost | Trivial | Forms |
| 9.2 | [[likely]] hints | 9 | 5-10% branch | Trivial | Forms |
| 9.3 | Double reciprocal SIMD | 9 | 2-3x division | Low | Math |

---

## Quick Wins (Trivial Effort, Measurable Impact)

1. **3.4** — Change default DOF numbering to Cuthill-McKee
2. **2.3** — Fix Vector/Matrix alignment to 64 bytes
3. **3.7** — Fix matrix-matrix multiply loop order
4. **6.3** — Direct-index FieldRegistry by FieldId
5. **6.4** — Cache operator definition reference
6. **4.1** — Pre-allocate history vectors
7. **4.2** — Pre-size constraint scratch vectors
8. **7.2** — Pre-allocate ghost contribution buffers
9. **9.1** — Eliminate double constitutive hash
10. **9.2** — Add [[likely]] to hot switch cases
