# Kernel Batch And Basis Optimization Plan

**Date**: 2026-03-28
**Scope**: Exact production monolithic path in the new OOP FE solver
**Primary goal**: Reduce FE assembly time by cutting the current dominant exact-path costs without changing assembly semantics or nonlinear behavior.

## Why This Phase

The latest exact-path timings show that insertion is no longer the dominant cost. The remaining warm monolithic bottlenecks are:

- `kernel`
- `prepareBasis`
- shared field setup

The next safe performance work therefore needs to target:

1. JIT wrapper batch overhead inside exact per-block execution
2. dense `K*u` work inside `LinearFormKernel` JIT dispatch
3. the common scalar `prepareBasis` fast path

This phase explicitly does **not** attempt to qualify compiled monolithic dispatch. That remains a separate follow-up once the real-case compiled residual mismatch is fixed.

## Implementation Checklist

- [ ] Add a narrower batch patching path in `JITKernelWrapper` that preserves template-invariant ABI fields and only updates per-cell pointers/scalars.
  Result: attempted, then reverted. It regressed the 3D monolithic hot path.
- [x] Rework the batched `LinearFormKernel` JIT path so vector-only mode can execute the bilinear kernel in batch into scratch matrix storage instead of falling back to one scalar JIT call per element.
- [x] Replace the vector-only `K*u` fallback with contiguous scratch-matrix accumulation while leaving the hot matrix-plus-vector Newton path on the validated direct accumulation loops.
- [ ] Rework the scalar `prepareBasis` fast path to transform cached qpt-major reference gradients/hessians directly into qpt-major physical storage, improving cache locality on the dominant affine/same-topology path.
  Result: attempted, then reverted. It regressed the 3D monolithic hot path.
- [x] Preserve exact fallback behavior and existing parity coverage for all touched paths.
- [x] Rebuild `svmultiphysics` and FE test binaries after the code changes.
- [x] Re-run targeted FE parity/unit tests for monolithic exact-path correctness.
- [x] Re-run the qualified OOP fluid performance cases and summarize:
  - FE assembly time
  - total runtime
  - Newton iterations per accepted step
  - final nonlinear residual behavior

## Acceptance Criteria

- The modified exact path remains the default qualified production path.
- No targeted FE parity/unit test regresses.
- The qualified fluid cases continue to converge cleanly with `converged=1` on accepted steps.
- At least the main 3D monolithic cases show lower FE assembly time than the current exact-path baseline from the most recent post-refactor logs.

## Notes

- If one of the planned changes regresses real-case timing or nonlinear behavior, it should be removed before final qualification.
- If a change helps only the compiled monolithic path, it is out of scope for this phase and should be documented separately instead of merged into this exact-path optimization pass.

## Final Outcome

Kept production change:

- `JITKernelWrapper` now has a true batched vector-only `LinearFormKernel` bilinear path that evaluates the bilinear part into batch scratch matrix storage instead of dropping back to one scalar JIT call per element.

Rejected after validation:

- the narrower template-invariant batch patching shortcut in `JITKernelWrapper`
- the qpt-major scalar `prepareBasis` fast-path rewrite

Reason for rejection:

- both experimental paths preserved correctness but slowed the real 3D monolithic assembly cases.

Final qualification status against the archived `20260328_serial_suite_rerun` logs:

- all accepted steps still reported `converged=1`
- targeted FE parity/unit coverage passed
- the final measured warm matrix under `OMP_NUM_THREADS=1` and `SVMP_FSILS_GMRES_REORTH=off` improved substantially versus that archived baseline
