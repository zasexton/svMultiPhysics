# Assembly Subfolder Unit Test Checklist

This document tracks missing unit tests for the FE/Assembly module. Each item includes the rationale for why the test is important for ensuring correctness, robustness, and performance.

**Last Updated:** 2026-01-08
**Status Legend:**
- [ ] Not started
- [~] In progress
- [x] Completed

---

## Table of Contents

1. [TimeIntegrationContext Tests](#1-timeintegrationcontext-tests)
2. [RemappedSystemView Tests](#2-remappedsystemview-tests)
3. [AssemblyContext Tests](#3-assemblycontext-tests)
4. [GlobalSystemView Tests](#4-globalsystemview-tests)
5. [AssemblyKernel Tests](#5-assemblykernel-tests)
6. [MatrixFreeAssembler Tests](#6-matrixfreeassembler-tests)
7. [AssemblyLoop Tests](#7-assemblyloop-tests)
8. [AssemblyStatistics Tests](#8-assemblystatistics-tests)
9. [CachedAssembler Tests](#9-cachedassembler-tests)
10. [DeviceAssembler Tests](#10-deviceassembler-tests)
11. [GhostContributionManager Tests](#11-ghostcontributionmanager-tests)
12. [StandardAssembler Edge Cases](#12-standardassembler-edge-cases)
13. [Performance and Stress Tests](#13-performance-and-stress-tests)

---

## 1. TimeIntegrationContext Tests

**File to create:** `Tests/Unit/Assembly/test_TimeIntegrationContext.cpp`

**Current status:** No dedicated test file exists. This is a critical gap for transient simulations.

### TimeDerivativeStencil Tests

- [x] **Construction and defaults**
  - Verify `order` defaults to 0
  - Verify `a` vector is empty by default
  - **Why:** Ensures safe default state for uninitialized stencils

- [x] **requiredHistoryStates() with empty coefficients**
  - Input: `a = {}`
  - Expected: Returns 0
  - **Why:** Prevents out-of-bounds access when no history is needed

- [x] **requiredHistoryStates() with single coefficient**
  - Input: `a = {1.0}` (only current state)
  - Expected: Returns 0
  - **Why:** BDF1/backward Euler only needs current state

- [x] **requiredHistoryStates() with two coefficients**
  - Input: `a = {1.5, -0.5}` (BDF1 style)
  - Expected: Returns 1
  - **Why:** Verifies single history state detection

- [x] **requiredHistoryStates() with three coefficients**
  - Input: `a = {1.5, -2.0, 0.5}` (BDF2 style)
  - Expected: Returns 2
  - **Why:** Verifies multi-step method support

- [x] **requiredHistoryStates() with trailing zeros**
  - Input: `a = {1.0, 0.5, 0.0, 0.0}`
  - Expected: Returns 1 (ignores trailing zeros)
  - **Why:** Ensures sparse stencils work correctly

- [x] **requiredHistoryStates() with all-zero coefficients**
  - Input: `a = {0.0, 0.0, 0.0}`
  - Expected: Returns 0
  - **Why:** Edge case for degenerate/disabled time derivative

- [x] **coeff() with valid index**
  - Input: `a = {1.0, -2.0, 1.0}`, `history_index = 1`
  - Expected: Returns -2.0
  - **Why:** Basic coefficient access

- [x] **coeff() with negative index**
  - Input: `history_index = -1`
  - Expected: Returns 0.0
  - **Why:** Bounds checking for invalid input

- [x] **coeff() with out-of-range index**
  - Input: `a = {1.0, 2.0}`, `history_index = 5`
  - Expected: Returns 0.0
  - **Why:** Bounds checking prevents crashes

- [x] **coeff() with zero index**
  - Input: `a = {3.14}`, `history_index = 0`
  - Expected: Returns 3.14
  - **Why:** Current state coefficient access

### TimeIntegrationContext Tests

- [x] **Construction and defaults**
  - Verify `integrator_name` defaults to `"<unset>"`
  - Verify `dt1` and `dt2` are nullopt
  - Verify all weights default to 1.0
  - **Why:** Safe uninitialized state

- [x] **stencil() with order 1**
  - Set `dt1` to valid stencil
  - Call `stencil(1)`
  - Expected: Returns pointer to dt1
  - **Why:** First-order time derivative access

- [x] **stencil() with order 2**
  - Set `dt2` to valid stencil
  - Call `stencil(2)`
  - Expected: Returns pointer to dt2
  - **Why:** Second-order time derivative access (acceleration)

- [x] **stencil() with order 0**
  - Call `stencil(0)`
  - Expected: Returns nullptr
  - **Why:** Order 0 is not a time derivative

- [x] **stencil() with invalid order (3+)**
  - Call `stencil(3)`, `stencil(-1)`
  - Expected: Returns nullptr
  - **Why:** Only orders 1 and 2 are supported

- [x] **stencil() with unset optional**
  - Do not set `dt1`
  - Call `stencil(1)`
  - Expected: Returns nullptr
  - **Why:** Safe handling of uninitialized stencil

- [x] **Weight multipliers**
  - Set `time_derivative_term_weight = 0.5`
  - Set `non_time_derivative_term_weight = 2.0`
  - Verify values are stored correctly
  - **Why:** Theta-method and other schemes need term weighting

- [x] **Per-derivative weights**
  - Set `dt1_term_weight = 0.25`
  - Set `dt2_term_weight = 0.75`
  - Verify independent storage
  - **Why:** Newmark-beta and other schemes weight dt/dt2 differently

---

## 2. RemappedSystemView Tests

**File to update:** `Tests/Unit/Assembly/test_GlobalSystemView.cpp` (add new fixture)

**Current status:** Only 1 test exists. Critical for RIS (Resistive Immersed Surface) support.

### DofRemapper Tests

- [x] **DofRemapTable::set() and map() basic**
  - Set mapping 5 -> 10
  - Verify `map(5)` returns 10
  - Verify `map(3)` returns nullopt
  - **Why:** Basic remapping functionality

- [x] **DofRemapTable with multiple mappings**
  - Set mappings: 0->10, 1->11, 2->12
  - Verify all mappings work
  - **Why:** Real use cases have many mapped DOFs

- [x] **DofRemapTable with self-mapping**
  - Set mapping 5 -> 5
  - Verify `map(5)` returns 5
  - **Why:** Edge case that should be handled (no-op)

- [x] **DofRemapTable with negative target**
  - Set mapping 5 -> -1
  - Verify behavior (should store -1)
  - **Why:** Negative indices may indicate special handling

- [x] **DofRemapTable overwrite existing mapping**
  - Set 5 -> 10, then 5 -> 20
  - Verify `map(5)` returns 20
  - **Why:** Remapping may be updated during setup

### RemappedSystemView Tests

- [x] **Construction with valid base and remapper**
  - Verify no exceptions
  - Verify `hasMatrix()`, `hasVector()` delegate correctly
  - **Why:** Basic construction validation

- [x] **addMatrixEntry() with unmapped DOF**
  - Add entry at (0, 0) where 0 is not mapped
  - Verify only original entry added
  - **Why:** Unmapped DOFs should pass through unchanged

- [x] **addMatrixEntry() with mapped row DOF**
  - Map row 1 -> 5
  - Add entry at (1, 2)
  - Verify entries at (1, 2) AND (5, 2)
  - **Why:** Core RIS duplication behavior

- [x] **addMatrixEntry() with mapped row and column**
  - Map 1 -> 5 and 2 -> 6
  - Add entry at (1, 2)
  - Verify entries at (1, 2), (5, 6)
  - **Why:** Full row+column remapping

- [x] **addMatrixEntry() where mapped_row == original_row**
  - Map 5 -> 5 (self-map)
  - Add entry at (5, 3)
  - Verify only one entry (no duplicate)
  - **Why:** Self-mapping should not duplicate

- [x] **addMatrixEntry() with negative mapped row**
  - Map 1 -> -1
  - Add entry at (1, 2)
  - Verify only original entry (skip invalid remap)
  - **Why:** Negative mapped DOF should be skipped

- [x] **addMatrixEntries() batch with mixed mappings**
  - Map some DOFs, not others
  - Add 3x3 block
  - Verify correct duplication pattern
  - **Why:** Real assembly uses batch operations

- [x] **addVectorEntry() with mapped DOF**
  - Map 2 -> 7
  - Add vector entry at 2
  - Verify entries at 2 AND 7
  - **Why:** Vector duplication for RHS

- [x] **addVectorEntries() batch**
  - Add batch with some mapped DOFs
  - Verify correct duplication
  - **Why:** Batch vector operations

- [x] **setDiagonal() passthrough**
  - Call `setDiagonal()`
  - Verify delegated to base (no duplication)
  - **Why:** Diagonal setting is not duplicated

- [x] **zeroRows() passthrough**
  - Call `zeroRows()`
  - Verify delegated to base
  - **Why:** Row zeroing delegates directly

- [x] **beginAssemblyPhase/endAssemblyPhase/finalizeAssembly**
  - Call lifecycle methods
  - Verify delegated to base
  - Verify `getPhase()` reflects base state
  - **Why:** Lifecycle must be consistent

- [x] **Property queries with null base**
  - Construct with valid base, then somehow have null
  - Verify `hasMatrix()` returns false, `numRows()` returns 0
  - **Why:** Null safety checks

- [x] **backendName()**
  - Verify returns base backend name or "RemappedSystem"
  - **Why:** Debugging and logging

- [x] **zero() delegation**
  - Call `zero()`
  - Verify base is zeroed
  - **Why:** Reset functionality

- [x] **getMatrixEntry/getVectorEntry delegation**
  - Get entries after adding
  - Verify reads from base correctly
  - **Why:** Debug/verification access

---

## 3. AssemblyContext Tests

**File to update:** `Tests/Unit/Assembly/test_AssemblyContext.cpp`

**Current status:** ~338 lines, missing multi-field and material state tests.

### Multi-Field Access Tests

- [x] **fieldValue() single field**
  - Set up context with one field
  - Query `fieldValue(field_index=0, qpt)`
  - **Why:** Multi-physics problems need field-by-field access

- [x] **fieldValue() multiple fields**
  - Set up context with 3 fields (velocity, pressure, temperature)
  - Query each field independently
  - **Why:** Coupled problems need independent field access

- [x] **fieldGradient() single field**
  - Query gradient of specific field
  - **Why:** Advection terms need velocity gradient

- [x] **fieldGradient() out-of-range field index**
  - Query with invalid field index
  - Expected: Exception or zero
  - **Why:** Bounds checking

- [x] **fieldJacobian() for vector field**
  - Set up vector-valued field
  - Query Jacobian (gradient of vector = tensor)
  - **Why:** Deformation gradient in solid mechanics

- [x] **fieldHessian() for scalar field**
  - Query second derivatives of scalar field
  - **Why:** Fourth-order PDEs, stabilization terms

- [x] **fieldLaplacian() for scalar field**
  - Query trace of Hessian
  - **Why:** Biharmonic equations, SUPG stabilization

- [x] **fieldValue() component access for vector field**
  - Query specific component of vector field
  - **Why:** Component-wise operations in Navier-Stokes

### Historical Solution Access Tests

- [x] **previousSolutionValue() with history available**
  - Set previous solution in context
  - Query `previousSolutionValue(qpt)`
  - **Why:** Time-stepping methods need u^{n-1}

- [x] **previousSolutionValue() without history**
  - Do not set previous solution
  - Query should return zero or throw
  - **Why:** Graceful handling of missing history

- [x] **previousSolutionGradient()**
  - Query gradient of previous solution
  - **Why:** Some schemes need grad(u^{n-1})

- [x] **previousSolutionValue(k)** for k=1,2
  - Set multiple history levels
  - Query u^{n-1} and u^{n-2}
  - **Why:** Multi-step methods (BDF2, etc.)

### Material State Tests

- [x] **materialState() basic access**
  - Reserve material state storage
  - Write and read state at quadrature point
  - **Why:** History-dependent materials (plasticity)

- [x] **materialStateOld() access**
  - Query previous time step state
  - **Why:** State update algorithms need old state

- [x] **materialStateWork() scratch access**
  - Use work buffer during computation
  - Verify isolated from stored state
  - **Why:** Temporary storage for Newton iterations

- [x] **materialState() multiple quadrature points**
  - Access state at different qpts
  - Verify independent storage
  - **Why:** State is per-integration-point

- [x] **materialState() multi-component state**
  - Store vector-valued state (e.g., plastic strain tensor)
  - **Why:** Real material models have complex state

### Geometric Measure Tests

- [x] **cellDiameter()**
  - Query element diameter
  - Verify against known geometry
  - **Why:** Stabilization parameters, CFL conditions

- [x] **cellVolume()**
  - Query element volume
  - Verify against analytical value
  - **Why:** Error estimators, mesh quality

- [x] **facetArea()**
  - Query face area in face context
  - **Why:** Flux computations, DG methods

- [x] **cellDiameter() for different element types** (N/A: element-type-specific measure computation is handled by assemblers; AssemblyContext stores provided measures)
  - Test Tetra4, Hex8, Triangle3, Quad4
  - **Why:** Element-specific diameter definitions

### Context Type Tests

- [x] **isCell() in cell context** (via `contextType() == ContextType::Cell`)
  - Configure for cell
  - Verify `isCell()` returns true
  - **Why:** Kernels may branch on context type

- [x] **isBoundaryFace() in face context** (via `contextType() == ContextType::BoundaryFace`)
  - Configure for boundary face
  - Verify `isBoundaryFace()` returns true
  - **Why:** Boundary condition application

- [x] **isInteriorFace() in DG context** (via `contextType() == ContextType::InteriorFace`)
  - Configure for interior face
  - Verify `isInteriorFace()` returns true
  - **Why:** DG flux computations

- [x] **normal() throws in cell context**
  - Configure for cell (not face)
  - Query `normal()`
  - Expected: Exception
  - **Why:** Normals only exist on faces

### Petrov-Galerkin Tests

- [x] **Non-square test/trial basis**
  - Set up with different test and trial spaces
  - Verify `numTestDofs() != numTrialDofs()`
  - **Why:** Mixed methods, stabilized formulations

- [x] **trialBasisValue() distinct from test**
  - Query trial and test basis separately
  - **Why:** Petrov-Galerkin requires both

---

## 4. GlobalSystemView Tests

**File to update:** `Tests/Unit/Assembly/test_GlobalSystemView.cpp`

**Current status:** ~420 lines, missing AddMode variants and stress tests.

### AddMode Tests

- [x] **AddMode::Max**
  - Add entry with value 5
  - Add again with value 3 using Max
  - Verify stored value is 5
  - **Why:** Some assembly patterns need maximum

- [x] **AddMode::Min**
  - Add entry with value 5
  - Add again with value 3 using Min
  - Verify stored value is 3
  - **Why:** Some assembly patterns need minimum

- [x] **AddMode transitions**
  - Add with Add, then Insert, then Add
  - Verify correct accumulation
  - **Why:** Mixed mode assembly

### Thread Safety Tests

- [x] **Concurrent addMatrixEntry() to different locations**
  - Multiple threads adding to non-overlapping regions
  - Verify no data races
  - **Why:** Colored/parallel assembly safety

- [x] **Concurrent addMatrixEntry() to same location** (N/A: DenseMatrixView does not provide atomic/thread-safe same-entry updates)
  - Multiple threads adding to same entry with atomics
  - Verify correct sum
  - **Why:** Atomic assembly correctness

### Edge Case Tests

- [x] **Very large matrix (10000x10000)** (optional; gated by `SVMP_FE_RUN_STRESS_TESTS=1`)
  - Create and populate large dense matrix
  - Verify memory allocation and access
  - **Why:** Stress test for memory handling

- [x] **Empty span arguments**
  - Call `addMatrixEntries()` with empty spans
  - Verify no crash, no-op behavior
  - **Why:** Edge case in assembly loops

- [x] **Mismatched span sizes**
  - Call `addMatrixEntries(row_dofs, col_dofs, matrix)` with wrong sizes
  - Expected: Exception
  - **Why:** Input validation

- [x] **isDistributed() for serial views**
  - Query on DenseMatrixView
  - Expected: false
  - **Why:** Serial/parallel distinction

---

## 5. AssemblyKernel Tests

**File to update:** `Tests/Unit/Assembly/test_AssemblyKernel.cpp`

**Current status:** ~472 lines, missing composite and face kernel tests.

### CompositeKernel Tests

- [x] **CompositeKernel with 3+ sub-kernels**
  - Compose Mass + Stiffness + Source
  - Verify combined output
  - **Why:** Complex physics often combines many terms

- [x] **CompositeKernel required data union**
  - Compose kernels with different RequiredData
  - Verify union of all requirements
  - **Why:** Context must provide all needed data

- [x] **CompositeKernel with zero scaling**
  - Add kernel with scale=0
  - Verify kernel is effectively disabled
  - **Why:** Conditional term inclusion

- [x] **CompositeKernel with negative scaling**
  - Add kernel with scale=-1
  - Verify subtraction behavior
  - **Why:** Residual formulations may subtract terms

### Face Kernel Tests

- [x] **Kernel with both boundary and interior face support**
  - Create kernel implementing both
  - Verify `hasBoundaryFace()` and `hasInteriorFace()` return true
  - **Why:** Some kernels apply to all faces

- [x] **BilinearFormKernel base class**
  - Derive simple kernel from BilinearFormKernel
  - Verify default implementations
  - **Why:** Convenience base class validation

- [x] **LinearFormKernel base class**
  - Derive simple kernel from LinearFormKernel
  - Verify matrix output is disabled
  - **Why:** RHS-only kernels

### Material State Tests

- [x] **materialStateSpec() with non-trivial size**
  - Return spec with `state_size = 6` (plastic strain)
  - Verify context reserves correct storage
  - **Why:** History-dependent materials

- [x] **materialStateSpec() with alignment requirement**
  - Return spec with `alignment = 32` (SIMD)
  - Verify aligned allocation
  - **Why:** Performance optimization

### Query Method Tests

- [x] **maxTemporalDerivativeOrder()**
  - Kernel with dt(u) returns 1
  - Kernel with dt(u,2) returns 2
  - Kernel without dt returns 0
  - **Why:** Time integrator needs to know derivative orders

- [x] **isSymmetric()**
  - Mass kernel returns true
  - Advection kernel returns false
  - **Why:** Solver selection optimization

- [x] **isMatrixOnly() / isVectorOnly()**
  - Stiffness kernel: isMatrixOnly=true
  - Source kernel: isVectorOnly=true
  - **Why:** Assembly optimization

---

## 6. MatrixFreeAssembler Tests

**File to update:** `Tests/Unit/Assembly/test_MatrixFreeAssembler.cpp`

**Current status:** ~530 lines, but ALL tests are configuration/mock-based. No actual matvec tests.

### Actual Operation Tests

- [x] **apply() with identity operator**
  - Configure with identity kernel
  - Call `apply(x, y)`
  - Verify `y == x`
  - **Why:** Basic operator application

- [x] **apply() with scaling operator**
  - Configure with kernel that scales by 2
  - Call `apply(x, y)`
  - Verify `y == 2*x`
  - **Why:** Non-trivial operator

- [x] **apply() with Laplacian kernel**
  - Configure with stiffness kernel
  - Apply to known vector
  - Verify against explicit matrix result
  - **Why:** Real PDE operator validation

- [x] **applyResidual()** (N/A: `MatrixFreeAssembler` currently exposes `assembleResidual()` with placeholder implementation)
  - Configure nonlinear kernel
  - Compute `r = f(u)`
  - **Why:** Newton method residual evaluation

- [x] **getDiagonal()**
  - Extract diagonal of implicit operator
  - Compare to explicit matrix diagonal
  - **Why:** Jacobi preconditioner needs diagonal

- [x] **computePreconditioner()** (N/A: not implemented in `MatrixFreeAssembler`)
  - Compute approximate inverse
  - Verify reduces condition number
  - **Why:** Iterative solver acceleration

### Caching Tests

- [x] **Geometry caching** (N/A: geometry caching is not implemented beyond setup-time element metadata)
  - Enable `cache_geometry = true`
  - Call apply() twice
  - Verify geometry computed only once
  - **Why:** Performance optimization

- [x] **Basis caching** (N/A: basis caching is not implemented beyond setup-time element metadata)
  - Enable `cache_basis = true`
  - Call apply() twice
  - Verify basis evaluated only once
  - **Why:** Performance optimization

- [x] **Cache invalidation**
  - Modify solution
  - Verify cache invalidated appropriately
  - **Why:** Correctness after state change

### Performance Comparison Tests

- [x] **Matrix-free vs explicit assembly time** (out of scope for unit tests; benchmark suite recommended)
  - Measure apply() time vs Ax time
  - Log comparison
  - **Why:** Understand tradeoffs

- [x] **Matrix-free vs explicit memory** (out of scope for unit tests; benchmark suite recommended)
  - Measure memory usage
  - Matrix-free should use much less
  - **Why:** Main advantage of matrix-free

### Vectorization Tests

- [x] **Vectorized kernel execution** (N/A: vectorized execution path not implemented)
  - Enable `vectorize = true`
  - Verify SIMD code path taken
  - **Why:** Performance optimization

- [x] **Batched element processing** (N/A: batched execution path not implemented)
  - Set `batch_size = 8`
  - Verify batch processing
  - **Why:** Amortize kernel launch overhead

---

## 7. AssemblyLoop Tests

**File to update:** `Tests/Unit/Assembly/test_AssemblyLoop.cpp`

**Current status:** ~350 lines, missing coloring and parallel correctness tests.

### Coloring Tests

- [x] **computeElementColoring() with real DofMap**
  - Create mesh with shared DOFs
  - Compute coloring
  - Verify no two adjacent elements have same color
  - **Why:** Race-free parallel assembly

- [x] **computeElementColoring() color count**
  - Verify color count is reasonable (typically < 20)
  - **Why:** Too many colors hurts parallelism

- [x] **computeOptimizedColoring()**
  - Compare to greedy coloring
  - Verify fewer or equal colors
  - **Why:** Better load balance

- [x] **Coloring quality metric**
  - Measure max elements per color / min elements per color
  - **Why:** Load balance assessment

### Parallel Correctness Tests

- [x] **cellLoopOpenMP() produces same result as sequential** (skips when OpenMP is unavailable)
  - Run same assembly with Sequential and OpenMP
  - Compare results
  - **Why:** Parallel correctness

- [x] **cellLoopColored() produces same result as sequential** (skips when OpenMP is unavailable)
  - Run same assembly with Sequential and Colored
  - Compare results
  - **Why:** Coloring correctness

- [x] **Deterministic parallel assembly**
  - Set `deterministic = true`
  - Run multiple times
  - Verify bitwise identical results
  - **Why:** Reproducibility for debugging

### Unified Loop Tests

- [x] **unifiedLoop() cell + boundary**
  - Process cells and boundary faces in one loop
  - Verify all contributions assembled
  - **Why:** Efficiency for combined assembly

- [x] **unifiedLoop() cell + boundary + interior**
  - Process all entity types
  - Verify DG assembly completeness
  - **Why:** Full DG support

### Prefetch Tests

- [x] **Prefetch hint generation**
  - Enable `prefetch_next = true`
  - Verify prefetch hints generated
  - **Why:** Cache optimization

---

## 8. AssemblyStatistics Tests

**File to update:** `Tests/Unit/Assembly/test_AssemblyStatistics.cpp`

**Current status:** ~100 lines, missing FLOP, memory, and export tests.

### FLOP Tracking Tests

- [x] **recordFLOPs() accumulation**
  - Record FLOPs from multiple operations
  - Verify total FLOP count
  - **Why:** Performance analysis

- [x] **getFLOPRate()**
  - Record FLOPs and time
  - Compute GFLOP/s
  - **Why:** Roofline model analysis

- [x] **FLOP breakdown by category**
  - Record kernel vs scatter FLOPs
  - Verify per-category tracking
  - **Why:** Identify bottlenecks

### Memory Tracking Tests

- [x] **trackAllocation()**
  - Track allocation events
  - Verify byte count
  - **Why:** Memory profiling

- [x] **trackDeallocation()**
  - Track deallocation
  - Verify net memory tracking
  - **Why:** Leak detection

- [x] **getMemoryStats() peak tracking**
  - Allocate, deallocate, allocate more
  - Verify peak memory recorded
  - **Why:** High-water mark analysis

### Load Balance Tests

- [x] **recordThreadStats()**
  - Record per-thread work
  - Verify thread-local accumulation
  - **Why:** Parallel efficiency

- [x] **computeLoadBalance()**
  - Simulate uneven work distribution
  - Compute imbalance metric
  - **Why:** Identify parallel bottlenecks

### Export Tests

- [x] **exportJSON() format**
  - Export statistics
  - Parse JSON and verify structure
  - **Why:** Programmatic analysis

- [x] **exportCSV() format**
  - Export statistics
  - Verify CSV headers and data
  - **Why:** Spreadsheet analysis

- [x] **compareWith() relative performance**
  - Compare two statistics objects
  - Verify percentage differences
  - **Why:** Regression detection

### Suggestion Generation Tests

- [x] **OptimizationSuggestion generation**
  - Simulate poor cache utilization
  - Verify suggestion generated
  - **Why:** Automated performance hints

---

## 9. CachedAssembler Tests

**File to update:** `Tests/Unit/Assembly/test_CachedAssembler.cpp`

**Current status:** ~100 lines, configuration only. No actual caching tests.

### Cache Behavior Tests

- [x] **Cache hit behavior**
  - Assemble element once
  - Assemble same element again
  - Verify second assembly uses cached matrix
  - **Why:** Core caching functionality

- [x] **Cache miss behavior**
  - Assemble new element
  - Verify fresh computation
  - **Why:** Cache miss path

- [x] **Hit/miss rate tracking**
  - Assemble mix of cached/uncached
  - Verify `CacheStats` accuracy
  - **Why:** Performance monitoring

### Eviction Policy Tests

- [x] **LRU eviction**
  - Fill cache to capacity
  - Access some elements
  - Add new element
  - Verify least-recently-used evicted
  - **Why:** LRU policy correctness

- [x] **FIFO eviction**
  - Fill cache to capacity
  - Add new element
  - Verify first-in evicted
  - **Why:** FIFO policy correctness

### Memory Limit Tests

- [x] **Memory limit enforcement**
  - Set small memory limit
  - Attempt to cache many elements
  - Verify limit respected
  - **Why:** Memory budget compliance

- [x] **Cache clear on limit change**
  - Reduce memory limit
  - Verify excess entries evicted
  - **Why:** Dynamic limit adjustment

### Invalidation Tests

- [x] **Cache invalidation on solution change** (N/A: no solution-change hook; cache invalidation is explicit via `invalidateCache()` / per-element invalidation)
  - Cache element matrices
  - Change solution (nonlinear)
  - Verify cache invalidated
  - **Why:** Correctness for nonlinear problems

- [x] **Selective invalidation**
  - Invalidate specific elements
  - Verify others remain cached
  - **Why:** Efficient partial updates

### Strategy Tests

- [x] **FullMatrix caching strategy**
  - Cache full element matrices
  - Verify full matrix retrieval
  - **Why:** Maximum reuse

- [x] **ReferenceElement caching strategy**
  - Cache reference element data
  - Transform to physical element
  - **Why:** Memory-efficient for affine elements

- [x] **GeometricFactors caching strategy**
  - Cache Jacobians, determinants
  - Recompute basis on-the-fly
  - **Why:** Balance memory/compute

---

## 10. DeviceAssembler Tests

**File to update:** `Tests/Unit/Assembly/test_DeviceAssembler.cpp`

**Current status:** ~100 lines, enum tests only. No GPU execution tests.

**Note:** GPU tests should be conditional on device availability.

### Conditional GPU Tests

- [x] **CUDA kernel launch (if CUDA available)** (N/A: GPU backend not implemented; `DeviceAssembler::isGPUAvailable()` is false in CPU fallback builds)
  - Simple element kernel on GPU
  - Verify correct results
  - **Why:** CUDA backend validation

- [x] **HIP kernel launch (if HIP available)** (N/A: GPU backend not implemented; `DeviceAssembler::isGPUAvailable()` is false in CPU fallback builds)
  - Same test for AMD GPUs
  - **Why:** HIP backend validation

- [x] **SYCL kernel launch (if SYCL available)** (N/A: GPU backend not implemented; `DeviceAssembler::isGPUAvailable()` is false in CPU fallback builds)
  - Same test for SYCL
  - **Why:** SYCL backend validation

### Memory Transfer Tests

- [x] **Host to device transfer**
  - Transfer mesh data to device
  - Verify data integrity
  - **Why:** Input data transfer

- [x] **Device to host transfer**
  - Compute on device
  - Transfer results to host
  - Verify correctness
  - **Why:** Output data transfer

- [x] **Unified memory mode** (N/A: unified memory is not implemented in the CPU fallback backend)
  - Use unified memory
  - Verify automatic migration
  - **Why:** Simplified programming model

### Async Tests

- [x] **Async transfer overlap** (N/A: async transfers/overlap not implemented in the CPU fallback backend)
  - Overlap transfer with compute
  - Verify correctness
  - **Why:** Performance optimization

- [x] **Multi-stream execution** (N/A: multi-stream execution not implemented in the CPU fallback backend)
  - Use multiple CUDA streams
  - Verify concurrent execution
  - **Why:** GPU utilization

### Partial Assembly Tests

- [x] **Partial assembly on GPU** (CPU fallback partial-assembly interface tested with a mock `DeviceKernel`)
  - Assemble element contributions on GPU
  - Scatter to host
  - **Why:** Hybrid CPU/GPU assembly

---

## 11. GhostContributionManager Tests

**File to update:** `Tests/Unit/Assembly/test_GhostContributionManager.cpp`

**Current status:** ~100 lines, single-rank only. No MPI tests.

**Note:** MPI tests require test infrastructure with multiple ranks.

### MPI Tests (Require MPI Test Harness)

- [x] **Two-rank ghost exchange** (conditional: requires `FE_HAS_MPI` and running the test with 2 MPI ranks)
  - Rank 0 owns DOFs 0-9, ghosts 10-14
  - Rank 1 owns DOFs 10-19, ghosts 5-9
  - Exchange and verify
  - **Why:** Basic distributed assembly

- [x] **Multi-rank ghost exchange** (conditional: requires `FE_HAS_MPI` and running the test with 4+ MPI ranks)
  - 4+ ranks with complex ghost patterns
  - Verify all contributions accumulated
  - **Why:** Scalability

- [x] **Non-blocking exchange** (N/A: `startExchange()` / `waitExchange()` are currently stubs; state machine is covered)
  - Overlap communication with computation
  - Verify correctness
  - **Why:** Performance optimization

- [x] **Large message handling** (conditional: requires `FE_HAS_MPI`, 2+ MPI ranks, and `SVMP_FE_RUN_STRESS_TESTS=1`)
  - Exchange very large ghost buffers
  - Verify no truncation
  - **Why:** Stress test

### Buffer Management Tests

- [x] **Buffer sizing**
  - Verify buffer sized correctly for ghost DOF count
  - **Why:** Memory allocation

- [x] **Buffer reuse**
  - Multiple exchanges without reallocation
  - **Why:** Performance

- [x] **Non-contiguous DOF ranges**
  - Ghosts at scattered DOF indices
  - Verify correct pack/unpack
  - **Why:** General DOF numbering

---

## 12. StandardAssembler Edge Cases

**File to update:** `Tests/Unit/Assembly/test_StandardAssembler.cpp`

**Current status:** ~1758 lines, good coverage but missing edge cases.

### Empty/Minimal Mesh Tests

- [x] **Empty mesh (0 cells)**
  - Create mesh with no cells
  - Assemble matrix/vector
  - Verify empty result, no crash
  - **Why:** Edge case handling

- [x] **Single element mesh**
  - One Tetra4 element
  - Verify correct assembly
  - **Why:** Minimal configuration

- [x] **Single node element** (Point1) (conditional: test skips if `Point1` path is not supported)
  - If supported, test point element
  - **Why:** Source term at point

### Degenerate Geometry Tests

- [x] **Near-zero Jacobian determinant**
  - Create nearly-degenerate element
  - Verify warning or handling
  - **Why:** Ill-conditioned elements

- [x] **Inverted element**
  - Create element with negative Jacobian
  - Verify detection/handling
  - **Why:** Mesh quality issues

### Rectangular Assembly Tests

- [x] **Rectangular with row_offset**
  - Assemble into subblock of larger matrix
  - Verify correct insertion locations
  - **Why:** Block assembly

- [x] **Rectangular with col_offset**
  - Assemble into off-diagonal block
  - Verify correct insertion
  - **Why:** Coupling terms

- [x] **Rectangular with both offsets**
  - Assemble into arbitrary subblock
  - **Why:** General block assembly

### Kernel Output Tests

- [x] **Kernel returns empty output**
  - Kernel that produces no contributions
  - Verify no crash, no insertions
  - **Why:** Conditional assembly

- [x] **Kernel returns NaN**
  - Detect NaN in assembled values
  - Verify error or warning
  - **Why:** Numerical debugging

- [x] **Kernel returns Inf**
  - Detect Inf in assembled values
  - **Why:** Overflow detection

### High-Order Element Tests

- [x] **p=3 element (Tetra20)** (N/A: `ElementType::Tetra20` not present in this FE library)
  - 20 DOFs per element
  - Verify large local matrix handling
  - **Why:** High-order methods

- [x] **p=5 element** (N/A: p=5 element support not present in this FE library)
  - Very high order
  - Stress test local matrix size
  - **Why:** Spectral element methods

### Constraint Chain Tests

- [x] **DOF constrained to constrained DOF**
  - DOF A depends on DOF B, DOF B depends on DOF C
  - Verify chain resolved correctly
  - **Why:** Hanging node hierarchies

---

## 13. Performance and Stress Tests

**File to create:** `Tests/Unit/Assembly/test_AssemblyPerformance.cpp`

**Note:** Performance tests should be separate from unit tests, possibly in a benchmark suite.

### Throughput Tests

- [x] **Elements per second** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`)
  - Measure assembly throughput
  - Report elements/second
  - **Why:** Performance baseline

- [x] **DOFs per second** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`)
  - Measure DOF throughput
  - **Why:** Alternative metric

- [x] **FLOP/s achieved** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`, estimate based on kernel FLOP count)
  - Measure floating point throughput
  - Compare to peak
  - **Why:** Roofline analysis

### Scaling Tests

- [x] **Strong scaling (OpenMP)** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`)
  - Fixed problem size, 1-16 threads
  - Measure speedup
  - **Why:** Parallel efficiency

- [x] **Weak scaling (OpenMP)** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`)
  - Problem size proportional to threads
  - Measure efficiency
  - **Why:** Scalability

### Memory Tests

- [x] **Peak memory usage** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`, uses `getrusage` when available)
  - Track high-water mark during assembly
  - **Why:** Memory requirements

- [x] **Memory bandwidth utilization** (optional; gated by `SVMP_FE_RUN_PERF_TESTS=1`, estimate based on bytes written per element)
  - Measure bytes/FLOP
  - Compare to machine bandwidth
  - **Why:** Memory-bound analysis

### Scheduler Comparison Tests

- [x] **Natural vs Hilbert ordering** (N/A: ordering correctness is unit-tested; performance comparison is benchmark-level)
  - Compare cache miss rates
  - **Why:** Ordering effectiveness

- [x] **Hilbert vs Morton ordering** (N/A: ordering correctness is unit-tested; performance comparison is benchmark-level)
  - Compare performance
  - **Why:** Space-filling curve comparison

- [x] **RCM ordering benefit** (N/A: current RCM implementation is a simplified placeholder; benefit should be validated with a connectivity-based implementation)
  - Compare matrix bandwidth before/after
  - **Why:** Solver performance impact

### Large Scale Tests

- [x] **1M DOF assembly** (N/A: benchmark/stress test; not suitable for unit test runs)
  - Assemble large problem
  - Measure time and memory
  - **Why:** Realistic problem size

- [x] **10M DOF assembly (if feasible)** (N/A: benchmark/stress test; not suitable for unit test runs)
  - Very large problem
  - **Why:** HPC scale

---

## Implementation Notes

### Test File Naming Convention
- Unit tests: `test_<ClassName>.cpp`
- Performance tests: `test_<Module>Performance.cpp`
- Integration tests: `test_<Feature>Integration.cpp`

### Test Fixture Pattern
```cpp
class TimeIntegrationContextTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper methods
};
```

### Mock Class Guidelines
- Place in anonymous namespace within test file
- Implement only methods needed for specific tests
- Document what the mock represents

### Assertion Guidelines
- Use `EXPECT_*` for non-fatal checks
- Use `ASSERT_*` when subsequent tests depend on result
- Use `EXPECT_NEAR` for floating-point comparisons with appropriate tolerance
- Use `EXPECT_THROW` for exception testing

---

## Priority Order for Implementation

### Phase 1: Critical Gaps (High Priority)
1. `test_TimeIntegrationContext.cpp` - Transient simulation support
2. RemappedSystemView expansion - RIS support
3. MatrixFreeAssembler actual operations - Matrix-free methods
4. Multi-field AssemblyContext tests - Coupled physics

### Phase 2: Robustness (Medium Priority)
5. Edge case tests for empty/degenerate cases
6. Coloring correctness tests
7. CachedAssembler functionality tests
8. AssemblyStatistics export tests

### Phase 3: Performance (Lower Priority)
9. Performance benchmark suite
10. MPI integration tests
11. GPU tests (conditional)
12. Large-scale stress tests

---

## Maintenance

This checklist should be updated when:
- New classes/functions are added to Assembly/
- Existing tests are expanded
- Bugs reveal missing test coverage
- Performance requirements change

**Review Schedule:** Quarterly or after major feature additions
