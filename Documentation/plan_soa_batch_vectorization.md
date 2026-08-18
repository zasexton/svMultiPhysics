# Plan: SoA Batch Vectorization Across Elements

## Summary

Transform the JIT kernel's batch loop from processing one element at a time (AoS, scalar
arithmetic) to processing `SIMD_WIDTH` elements simultaneously (SoA, LLVM vector arithmetic).
On SSE2, `SIMD_WIDTH = 2`, so every `fadd`/`fmul`/`fsqrt` operates on 2 elements in a single
instruction. The batch loop iterates `batch_size / 2` times instead of `batch_size` times.

## Key Design Decision: Keep Existing ABI

Do NOT create a new `CellKernelBatchArgsV2` with pre-transposed SoA data. Instead, keep the
existing `CellKernelBatchArgsV1` ABI and have the JIT codegen perform AoS-to-SoA gather at
load time. For SSE2, gathering 2 doubles is just 2 scalar loads + 1 insertelement. Gather
overhead is ~0.25% of compute cost.

## Broadcast vs Gather Classification

**Broadcast (same for all elements):**
- Reference basis values: `test_basis_values`, `trial_basis_values`
- Loop bounds: `n_qpts`, `n_test_dofs`, `n_trial_dofs`
- Time integration: `time`, `dt`, stencil coefficients, term weights
- JIT constants/parameters

**Gather (different per element):**
- Physical gradients/hessians (depend on Jacobian)
- Integration weights (depend on Jacobian determinant)
- Solution/previous solution coefficients
- Cell diameter, cell volume
- Jacobians, inverse Jacobians
- Output: element_matrix, element_vector pointers

## Implementation Phases

### Phase A: Foundation (~5 lines)
- `HardwareProfile::simdDoubles()` convenience method
- `JITOptions::simd_batch` field (default true)
- `SVMP_SIMD_BATCH=0` env var to disable

### Phase B: LLVMGen SIMD Infrastructure (~300 lines)
- `vf64` = `FixedVectorType::get(f64, simd_w)` for `<2 x double>`
- `vf64c(v)` splat constant helper
- `loadRealPtrAtSIMD()` gather lambda: 2 scalar loads + insertelement
- `emitMatrixAccumSIMD()`/`emitVectorAccumSIMD()`: extract lanes, accumulate per-element
- `loadSideViewSIMD()`: load from SIMD_WIDTH adjacent KernelSideArgsV6 structs
- Batch loop stride: `b_idx += simd_w` instead of `b_idx += 1`

### Phase C: evalKernelIRSingleValue Vectorization (~200 lines)
- All `CodeValue.elems[i]` become `<SIMD_W x double>`
- Arithmetic ops (`FAdd`, `FMul`, etc.) naturally work on vector types
- Unary intrinsics (`llvm.sqrt`, `llvm.fabs`) are overloaded for vectors
- `CreateFCmpOEQ/OLT` on vectors → `<SIMD_W x i1>`, `CreateSelect` works correctly
- Constants become splat vectors
- Basis value loads: broadcast for ref basis, gather for phys gradients
- Solution coefficient reduce-sums produce vector sums
- Component flags remain scalar (batch-invariant)

### Phase D: JITKernelWrapper Padding (~30 lines)
- Round batch_size up to next multiple of SIMD_WIDTH
- Padding slots: clone last element's side args (valid reads), point output to thread_local scratch

### Phase E: Testing (~150 lines)
- Parity test: scalar vs SIMD batch results match to machine epsilon
- Remainder handling: odd batch sizes
- NS-VMS convergence matches scalar path

## Interactions with Existing Optimizations

- **Trial-only caching**: Cached values become `<2 x double>`. Alloca, store/load work identically. Still profitable (eliminates 22-49 recomputations).
- **Three-level unrolling**: Orthogonal — vectorizes across batch dimension, unrolling across QP/DOF dimensions. Text budget unchanged (SSE2 packed ops same encoding size as scalar).
- **QP shared cache**: Entries become `SIMD_W * 8` bytes each. 5 patterns × ~9 doubles × 16 bytes = 720 bytes, well within 4KB budget.
- **Colored parallel assembly**: Orthogonal — SoA operates within one thread's batch.

## Expected Impact

- ~40-50% reduction in kernel compute time
- Channel2D: assembly ~170ms/call → ~100-120ms/call
- iliac_artery: assembly ~1.3s/call → ~0.8-1.0s/call
- L1i neutral (SSE2 packed ops same encoding size as scalar)

## Risks

1. LLVM 14 vector codegen quality → gated by `SVMP_SIMD_BATCH` env var
2. External coefficients (C++ trampolines) can't vectorize → fall back to per-lane scalar calls
3. QP cache budget halved → still sufficient for the 5 critical patterns
