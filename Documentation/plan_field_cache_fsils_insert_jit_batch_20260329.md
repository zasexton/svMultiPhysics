# Field Cache, FSILS Insert, And Exact JIT Batch Plan

**Date**: 2026-03-29
**Scope**: Next exact-path FE assembly optimization pass after `20260328_monolithic_compiled_batch_phase2`
**Goal**: Reduce the remaining exact-path assembly hot buckets by targeting:

1. shared field value/jacobian caching
2. FSILS resolved block-insert fast path
3. a narrower exact-path JIT batch runner

## Why This Phase

The current exact-path timing probes show the remaining warm monolithic bottlenecks are still concentrated in:

- `kernel`
- `shared fields`
- `insert`

Representative exact-path bucket mix from `2026-03-29` timing probes:

- `pipe_simple`: `kernel ~63%`, `shared fields ~15%`, `insert ~10%`
- `iliac_artery`: `kernel ~62%`, `shared fields ~12%`, `insert ~11%`
- `Channel2D`: `kernel ~49%`, `shared fields ~14%`, `insert ~8.5%`

That makes the next optimization order clear:

1. cut repeated field reconstruction work
2. reduce resolved-entry insertion overhead
3. narrow the remaining exact-path JIT batch overhead without reopening the broad batch-patching regression

## Guardrails

- [x] Keep the exact production path as the qualification target.
- [x] Leave legacy-solver files untouched except for comparison-only instrumentation if absolutely required.
- [x] Do not promote the compiled monolithic path; it remains opt-in until it is both compare-clean and faster than the exact path.
- [x] Preserve the current readable structure in `StandardAssembler`: shared helpers, capability-gated fast paths, no ad hoc solver-specific branches.
- [x] Require targeted FE parity coverage and real-case nonlinear convergence checks before keeping any optimization.

## Workstream 1: Shared Field Value/Jacobian Caching

### Objective

Avoid re-evaluating the same field values and field gradients/jacobians across multiple monolithic blocks on the same cell when the field access pattern is identical.

### Target Files

- `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- `Code/Source/solver/FE/Assembly/AssemblyContext.h`
- `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`

### Implementation Checklist

- [x] Audit `populateFieldSolutionDataFast()` in `StandardAssembler.cpp` and identify the exact cache key needed to reuse evaluated field results safely.
- [x] Introduce a per-cell field-evaluation cache entry type that can represent:
  - scalar values
  - scalar gradients
  - vector values
  - vector jacobians
  - previous-step scalar values
  - previous-step vector values
- [x] Key the cache by the minimum stable identity needed for correctness:
  - field id
  - cell id
  - history index
  - required-data mask subset actually evaluated
  - access recipe identity / space identity when needed
- [x] Keep coefficient gather reuse and evaluated-field reuse separate so coefficient caching stays reusable even when field-evaluation caching misses.
- [x] Rework `populateFieldSolutionDataFast()` to:
  - check the evaluated-field cache first
  - populate `AssemblyContext` directly from cached spans when present
  - fill the cache after first evaluation when absent
- [x] Reuse the same cache for previous-solution field replay so repeated `dt` blocks do not re-evaluate prior states.
- [x] Avoid storing unnecessary data:
  - do not cache jacobians if only values were requested
  - do not cache previous states beyond the actually-required history depth
- [x] Keep fallback behavior intact when recipes are invalid, unsupported, or require data not handled by the fast path.
- [x] Add or extend unit coverage for:
  - current-state field reuse
  - previous-state field reuse
  - mixed scalar/vector field requests
  - value-only vs value-plus-gradient cache compatibility rules

### Success Signal

- [ ] `shared fields` time drops materially on `pipe_simple`, `iliac_artery`, and `Channel2D` with unchanged nonlinear traces.
  Result: not met on the `2026-03-29` exact timing probe; the new cache is functionally correct but the measured hotspot windows regressed.

## Workstream 2: FSILS Resolved Block-Insert Fast Path

### Objective

Reduce the remaining scalar resolved-slot scatter overhead in FSILS matrix insertion for common dense local block shapes.

### Target Files

- `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp`
- `Code/Source/solver/FE/Assembly/GlobalSystemView.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`

### Implementation Checklist

- [x] Inspect the resolved-slot layout emitted by `resolveMatrixEntrySlotsCached()` and classify the common contiguous/block-structured cases seen in the fluid systems.
- [x] Add a narrow fast-path predicate in `FsilsMatrix::addResolvedMatrixEntries()` for resolved matrices that correspond to dense local node blocks or other contiguous slot regions.
- [x] Implement a block update path that:
  - writes contiguous slot runs with tight pointer arithmetic
  - keeps the current scalar fallback for irregular or partially invalid slot maps
  - preserves `Add` vs `Insert` semantics exactly
- [x] Keep the fast path backend-local; do not add FE-layer special casing for specific equations.
- [x] Determine that no new `GlobalSystemView` capability or layout hint is needed; keep slot-run detection entirely backend-local.
- [x] Reuse the existing resolved-entry insertion machinery in `StandardAssembler` without changing the higher-level assembler flow.
- [x] Verify behavior on:
  - constrained rows / invalid slots
  - mixed rectangular local matrices
  - pure insert vs additive accumulation
- [x] Add or extend backend unit coverage for:
  - contiguous resolved block insertion
  - irregular fallback insertion
  - exact equivalence to the previous scalar loop

### Success Signal

- [ ] `insert` time drops on the 3D monolithic cases without any parity or convergence change.
  Result: not met on the representative exact timing probe; correctness held, but the measured insert bucket did not improve.

## Workstream 3: Narrower Exact-Path JIT Batch Runner

### Objective

Cut remaining exact-path wrapper overhead in batched cell execution without reintroducing the broader template-patching regression.

### Target Files

- `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
- `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`

### Implementation Checklist

- [x] Re-profile the exact batch path in `JITKernelWrapper::computeCellBatch()` and isolate the still-repeated per-element work that is truly batch-invariant.
- [x] Keep the scope narrow:
  - do not redesign the generic batch ABI
  - do not reopen the rejected broad template-patching shortcut
  - do not couple the exact path to the compiled monolithic dispatcher
- [x] Introduce a prepared exact-batch helper that owns:
  - one specialization lookup per batch
  - one packed template argument block per batch
  - one scratch-layout preparation per batch
- [x] Restrict per-element patching to the minimal fields that actually vary:
  - geometry/field pointers
  - output pointers
  - per-element material-state pointers when present
- [x] Keep the per-element inlined state update loop only where required; do not pay that cost for kernels with no updates.
- [x] Prefer stack-like contiguous scratch buffers and simple pointer patching over new heap-managed structures.
- [x] Revisit the exact non-compiled batch path first; do not assume the compiled monolithic path is the right vehicle for this work.
- [ ] Add or extend unit coverage for:
  - `FormKernel` batch path
  - `LinearFormKernel` batch path
  - nonlinear batch path
  - matrix-only, vector-only, and matrix-plus-vector modes

### Success Signal

- [ ] `kernel` time drops measurably on `pipe_simple`, `iliac_artery`, and `Channel2D` while all targeted FE tests stay green.
  Result: not met on the `2026-03-29` exact timing probe; parity stayed green, but the warm kernel bucket regressed.

## Execution Order

- [x] Implement Workstream 1 first.
- [x] Rebuild FE targets and run targeted FE/unit parity coverage.
- [x] Run a small real-case timing probe with `SVMP_ASSEMBLY_TIMING=1` on:
  - `pipe_simple`
  - `iliac_artery`
  - `Channel2D`
- [ ] Only proceed to Workstream 2 if Workstream 1 is correct and beneficial.
- [ ] Repeat rebuild, targeted tests, and the same timing probe after Workstream 2.
- [ ] Only proceed to Workstream 3 if Workstreams 1 and 2 are stable.
- [x] After Workstream 3, rerun:
  - FE parity/unit coverage
  - the qualified exact-path fluid matrix
- [x] Record kept vs rejected changes in `Documentation/assembly_optimization_log.md`.

## Validation Checklist

- [x] `cmake --build build/svMultiPhysics-build --target svmultiphysics test_fe_systems test_fe_assembly -j8`
- [x] `./build/svMultiPhysics-build/bin/test_fe_systems --gtest_filter='BackendParity.ResidualPath_JitMonolithicSparsityMatchesFallback:MixedFormPerformance.InstallFormulation_MonolithicJITParity_VersusPerBlockFallback:MixedFormPerformance.InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientVMSResidual:MonolithicCoupling.MixedJacobianBlockFDVerification'`
- [x] `./build/svMultiPhysics-build/bin/test_fe_assembly --gtest_filter='StandardAssembler*'`
- [x] Representative exact-path timing probe with `SVMP_ASSEMBLY_TIMING=1` on:
  - `pipe_simple`
  - `iliac_artery`
  - `Channel2D`
- [x] Full qualified exact-path fluid matrix rerun under:
  - `OMP_NUM_THREADS=1`
  - `SVMP_FSILS_GMRES_REORTH=off`

## Acceptance Criteria

- [x] The exact production path remains compare-clean against the targeted FE parity/unit coverage.
- [x] All accepted steps in the qualified fluid matrix still report `converged=1`.
- [x] Newton iteration sequences remain unchanged or improve, with no unexplained nonlinear deterioration.
- [ ] At least one of the three targeted buckets (`shared fields`, `insert`, `kernel`) drops materially on the representative timing probe.
- [x] The full exact-path qualified matrix improves FE assembly time on the main monolithic fluid cases relative to `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/final_matrix_exact/`.
