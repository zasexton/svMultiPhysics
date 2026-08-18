# Monolithic Kernel Follow-Up Plan

**Date**: 2026-03-28
**Scope**: FE/OOP kernel assembly follow-up after the first monolithic exact-path qualification pass
**Goal**: Reduce the remaining FE assembly hot buckets while preserving qualified nonlinear behavior on the fluid case matrix.

## Targeted Work

- [x] Implement true batched monolithic compiled dispatch in `StandardAssembler` using the existing coupled-kernel batch ABI instead of forcing `batch_size = 1`.
- [x] Make monolithic compiled dispatch matrix-only in practice: skip compiled residual writes when the exact residual fallback is still the qualified path.
- [x] Reuse current/previous solution coefficient gathers across shared field reconstruction and later monolithic block setup for the same cell.
- [x] Extend the monolithic coupled scalar-cache path so affine batched runs can precompute scalar physical gradients/hessians once per slot and reuse them for every block, including block 0.
- [x] Use `setSolutionCoefficientsOnly` / `setPreviousSolutionCoefficientsOnlyK` on monolithic blocks whose exact fallback kernel is a ready JIT wrapper, avoiding unnecessary FE-side QP solution reconstruction.
- [x] Trim the remaining `JITKernelWrapper` `K*u` accumulation overhead with raw contiguous accumulation helpers instead of repeated `matrixEntry()` / `vectorEntry()` access in the hot loops.
- [x] Leave legacy-solver code untouched except for any comparison-only instrumentation if required.
- [x] Rebuild `svmultiphysics` and FE test binaries.
- [x] Re-run targeted monolithic FE parity/unit coverage.
- [x] Re-run the qualified fluid matrix and summarize:
  - FE assembly time
  - total runtime
  - Newton iterations per accepted step
  - final nonlinear residual behavior

## Acceptance Criteria

- The exact monolithic default path remains correct and converges cleanly on the qualified fluid matrix.
- Any compiled monolithic acceleration kept from this pass remains opt-in unless the real-case compare evidence is clean.
- The new code keeps the monolithic assembly flow readable: shared helpers, narrow fast paths, no solver-layer special casing.
- The final qualified runs show lower FE assembly time on the main monolithic fluid cases than the current `20260328_kernel_batch_basis_phase1` matrix.

## Final Outcome

Artifacts:

- Exact-path qualification matrix: `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/final_matrix_exact/`
- Compiled compare matrix: `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/compare/`
- Compiled performance probe matrix: `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/final_matrix/`

Kept production changes:

- Batched monolithic compiled dispatch is implemented and the shared `CoupledBlockView` packing now includes Hessian pointers, fixing the transient compiled-path crash.
- The exact monolithic path now shares per-cell coefficient gathers with field reconstruction and monolithic block setup.
- Affine coupled scalar-cache batches reuse scalar physical gradients and Hessians for every block, including block 0.
- JIT-ready fallback blocks now use coefficient-only solution/history handoff where possible.
- `JITKernelWrapper` now uses raw contiguous helpers for the remaining dense `K*u` accumulation loops.

Qualification result:

- Targeted FE parity coverage passed on both the default exact path and the opt-in compiled compare path.
- The real-case compiled compare matrix was clean with `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`, `SVMP_FE_COMPARE_MONOLITHIC_COMPILED=1`, and `SVMP_FE_COMPARE_MONOLITHIC_MAX_CELLS=4`.
- The opt-in compiled monolithic path is still not production-worthy for performance. On the qualified fluid matrix it was dramatically slower than the exact path, so it remains opt-in.
- The production exact path improved on every qualified case versus `20260328_kernel_batch_basis_phase1`.

Exact-path qualified comparison versus `20260328_kernel_batch_basis_phase1/final_matrix`:

- `channel2d`: FE assembly `8.094485 -> 6.470769 s`, time loop `36.279080 -> 34.465194 s`
- `channel2d_simple`: FE assembly `0.414273 -> 0.301716 s`, time loop `0.574375 -> 0.456485 s`
- `vortex_shedding`: FE assembly `1.145903 -> 0.995596 s`, time loop `23.706173 -> 23.327462 s`
- `pipe_rcr_3d`: FE assembly `3.856757 -> 3.307784 s`, time loop `5.323722 -> 4.662353 s`
- `pipe_rcr_3d_rcrcr`: FE assembly `3.545970 -> 2.818439 s`, time loop `11.065862 -> 9.911781 s`
- `pipe_simple`: FE assembly `4.590686 -> 3.785898 s`, time loop `7.258278 -> 6.149500 s`
- `iliac_artery`: FE assembly `32.078952 -> 26.070436 s`, time loop `69.632548 -> 62.527526 s`

Nonlinear behavior on the exact production path remained excellent:

- all accepted steps reported `converged=1`
- Newton iteration sequences matched the prior qualified matrix on every case
