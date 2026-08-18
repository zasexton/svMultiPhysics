# Plan: Explicit SIMD for Rolled QP Inner Loops

## Summary

When QP loops are rolled (three-level policy level 2 or 3), emit LLVM vector types
(`<2 x double>` for SSE2) for the QP accumulation dimension. Process 2 QPs per iteration
instead of 1, halving the QP loop trip count.

## Approach

### QP-Vectorized Loop Structure

Current rolled QP loop:
```
for q = 0..n_qpts-1:
    w = integration_weights[q]
    basis_i = test_basis[i*nq + q]
    basis_j = trial_basis[j*nq + q]
    grad_i = test_grad[i*nq + q]  // 3 components
    ... evaluate expression ...
    matrix[i,j] += w * expr
```

SIMD-vectorized (2-wide):
```
for q = 0..n_qpts-1 step 2:
    w = <integration_weights[q], integration_weights[q+1]>
    basis_i = <test_basis[i*nq+q], test_basis[i*nq+q+1]>
    ... evaluate expression as <2 x double> ...
    partial = w * expr   // <2 x double>
    matrix[i,j] += partial[0] + partial[1]  // horizontal sum
```

### Key Difference from SoA Batch Vectorization

- **SoA batch**: same QP, same DOF, different elements → no horizontal sum needed
- **QP SIMD**: same element, different QPs → needs horizontal sum at accumulation
- These are orthogonal and can be combined (process 2 QPs × 2 elements = 4-wide)

### Data Layout Requirements

Current basis data is stored as `basis[i*nq + q]` (dof-major). For QP-vectorized loads,
adjacent QPs for the same DOF must be contiguous. Current layout IS contiguous for QP
within a DOF: `basis[i*nq + q]` and `basis[i*nq + q+1]` are adjacent. So aligned
`<2 x double>` loads work directly.

### Horizontal Sum at Accumulation

The QP accumulator needs a horizontal sum before writing to the element matrix:
```llvm
%sum = call double @llvm.vector.reduce.fadd(double 0.0, <2 x double> %accum)
```
Or for SSE2: `hadd` + extract. This adds ~2 instructions per matrix entry.

## Implementation

### LLVMGen Changes
- When QP loop is rolled AND `n_qpts % 2 == 0`: use `<2 x double>` QP vectorization
- Load basis/gradient/solution values as contiguous pairs
- All QP-interior arithmetic uses vector types
- Accumulator is `<2 x double>`, horizontally summed before matrix write
- Remainder QP (odd count): scalar epilogue

### Interaction with SoA Batch
- If both active: `<2 x double>` from QP × `<2 x double>` from batch
- Would need `<4 x double>` or separate treatment
- Recommend implementing independently, combining later

## Expected Impact

- ~30-40% reduction in rolled QP loop iterations
- Horizontal sum overhead partially offsets gains
- Most beneficial when QP loop is the innermost rolled loop (policy level 2)
- Less impactful for Tet4 (only 4 QPs → 2 iterations)

## Risk

- LLVM 14 vector reduction intrinsics may not be available
- Horizontal sum overhead for short QP loops (4 QPs → 2 iters, 1 hadd each)
- Interaction with trial-only caching loop reordering
