# Assembly Optimization Log

## Target

- Goal: reduce new OOP solver FE assembly time to at most half of the measured legacy assembly time.
- Current direct legacy reference: `Channel2D` serial legacy assembly total `5.019001 s` from `/tmp/legacy_assembly_timer_channel2d.log`.
- Current OOP reference: `Channel2D` clean-JIT-cache total FE assembly `23.789129 s` from `/tmp/svmp_target2x_pass2/channel2d.log`.
- Practical target for `Channel2D`: about `2.51 s` total FE assembly.

## Validated Baseline

- Artifact: `/tmp/svmp_target2x_pass2`
- `Channel2D_Simple`: wall `25.88 s`, FE assembly `1.103462 s`
- `Channel2D`: wall `135.11 s`, FE assembly `23.789129 s`
- `vortex_shedding`: wall `331.44 s`, FE assembly `14.877579 s`
- `Channel2D` warm cell-loop split:
  - `prepareBasis` `0.037750 s`
  - `field solutions` `0.027333 s`
  - `dof lookup` `0.002115 s`
  - `sol gather` `0.036907 s`
  - `kernel` `0.089943 s`
  - `insert` `0.031187 s`

## Experiments

### 2026-03-05: JIT pass tuning

- Change: reduced LLVM loop interleave/unroll work while keeping vectorization enabled.
- Files: `Code/Source/solver/FE/Forms/JIT/JITEngine.cpp`
- Result:
  - helped cold `Channel2D_Simple`
  - mixed or negative on larger cases
- Verdict: partially useful, not sufficient for large-case assembly.

### 2026-03-05: setup-time JIT priming for cell kernels

- Change: eager cell-specialization priming before the time loop.
- Files: `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.*`, `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- Result:
  - initial implementation appeared ineffective because stale binaries were still being exercised
  - after clean rebuild and tracing, cell priming was confirmed to work
- Verdict: keepable.

### 2026-03-05: specialization tracing

- Change: env-gated tracing for JIT specialization compile/hit/skip behavior.
- Files: `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.*`
- Result:
  - showed `COUPLED-FB` cell paths were already reusing primed kernels
  - identified remaining lazy work on boundary variants
- Verdict: useful diagnostic, keepable.

### 2026-03-05: boundary JIT priming

- Change: extended setup-time priming from cell to boundary variants.
- Files: `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.*`, `Code/Source/solver/FE/Systems/SystemSetup.cpp`
- Result:
  - removed step-0 runtime JIT compiles after the start of the solve
  - strong cold-start win on `Channel2D_Simple`, modest on `Channel2D`, negligible on `vortex_shedding`
- Verdict: keepable.

### 2026-03-05: FE gather/scatter locality pass

- Change: cached backend-local vector gathers, direct resolved-entry reads, node-level cache in FSILS matrix insertion.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`, `Code/Source/solver/FE/Assembly/GlobalSystemView.h`, `Code/Source/solver/FE/Backends/FSILS/FsilsVector.cpp`, `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp`
- Result:
  - modest FE-local improvement
  - `Channel2D` wall improved about `3.8%`
- Verdict: keepable baseline for later work.

### 2026-03-05: generic connectivity caches for vector entries and matrix slots

- Change: per-entity connectivity-cache designs in assembler and FSILS backend.
- Files: `Code/Source/solver/FE/Assembly/GlobalSystemView.h`, `Code/Source/solver/FE/Assembly/StandardAssembler.*`, `Code/Source/solver/FE/Backends/FSILS/FsilsVector.*`, `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.*`
- Result:
  - preserved nonlinear convergence
  - regressed wall time and drove insert cost sharply upward
- Verdict: rejected.

### 2026-03-05: phase 1-4 setup-time tables and FSILS matrix cache

- Change: setup-time per-cell tables for rows, cols, and field accesses plus an FSILS matrix cache redesign.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.*`, `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.*`
- Result:
  - phases 1-3 mildly helped lookup work
  - phase 4 matrix cache dominated runtime and memory
- Verdict: reject phase 4, keep only setup-side lessons.

### 2026-03-05: FSILS one-time row/block-pointer tables

- Change: reverted the bad hot-path matrix cache and rebuilt insertion around setup-time block-base lookup tables.
- Files: `Code/Source/solver/FE/Backends/FSILS/FsilsShared.h`, `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.*`
- Result:
  - recovered memory and part of the regression
  - still slower than the last good locality baseline
- Verdict: not the final answer.

### 2026-03-05 to 2026-03-06: serial FSILS kernel fast paths

- Change: single-thread fast paths for hottest FSILS GMRES and SpMV helpers.
- Files: `Code/Source/solver/FE/Backends/FSILS/liner_solver/gmres.cpp`, `Code/Source/solver/FE/Backends/FSILS/liner_solver/spar_mul.cpp`
- Result:
  - keepable modest end-to-end wins
  - not an FE assembly fix, but reduced total wall time
- Verdict: keepable.

### 2026-03-06: reference-space field accumulation

- Change: moved scalar/vector field accumulation to reference space and transformed once per quadrature point.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/svmp_fe_field_transform_20260306`
- Result:
  - reduced the `field solutions` slice
  - total assembly barely moved on `Channel2D`
  - `vortex_shedding` regressed
  - nonlinear traces stopped matching exactly
- Verdict: rejected and reverted.

### 2026-03-06: resolved FSILS matrix-slot insertion via generic views

- Change: precomputed resolved matrix insert entries in the assembler and inserted through generic resolved-entry hooks.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/svmp_resolved_insert_20260306`
- Result:
  - exact nonlinear convergence on all three cases
  - slower on all three cases
- Verdict: rejected and reverted.

### 2026-03-06: batched trial-group gather in coupled batch assembly

- Change: batched resolved-entry fetches across active slots in the non-monolithic `COUPLED-FB` path.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/svmp_trialgroup_batch_channel2d`
- Result:
  - exact nonlinear convergence
  - `Channel2D` assembly and wall time regressed
- Verdict: rejected and reverted.

### 2026-03-06: narrow field interpolation transform

- Change: reduced `populateFieldSolutionData()` arithmetic by accumulating reference-space gradients once per quadrature point and avoiding some basis-cache copies on value-history reconstruction.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/channel2d_fieldgrad_opt_20260306.time`
- Result:
  - preserved iteration counts and linear iteration counts
  - residuals stayed very close, but not bit-for-bit identical
  - `Channel2D` total FE assembly regressed to `25.936884 s`
  - wall time regressed to `2:19.12`
- Verdict: rejected and reverted.

### 2026-03-06: reuse active trial coefficients as field coefficients

- Change: when a requested field matched the active trial space / DOF map / offset / cell, `populateFieldSolutionData()` reused the already-gathered trial coefficients instead of gathering again.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.h`, `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/channel2d_aliasreuse_20260306.time`
- Result:
  - exact nonlinear history on `Channel2D`
  - `Channel2D` total FE assembly still regressed to `24.812552 s`
  - wall time regressed to `2:17.22`
- Verdict: rejected and reverted.

## Additional Findings

### 2026-03-28: exact-path wrapper/basis follow-up

- Change set explored:
  - narrower template-invariant batch patching in `JITKernelWrapper`
  - batched vector-only `LinearFormKernel` bilinear evaluation into scratch matrix storage
  - contiguous helper-based `K*u` cleanup
  - qpt-major scalar `prepareBasis` fast-path rewrite
- Kept:
  - vector-only `LinearFormKernel` batch scratch-matrix path in `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
- Rejected:
  - template-invariant batch patching shortcut in `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
  - qpt-major scalar `prepareBasis` rewrite in `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Reason for rejection:
  - both rejected paths preserved parity but regressed the real 3D monolithic hot path
- Final validation artifacts:
  - `Documentation/qualification_logs/20260328_kernel_batch_basis_phase1/final_matrix/`
- Final measured comparison versus `Documentation/qualification_logs/20260328_serial_suite_rerun/`:
  - `Channel2D`: FE assembly `15.320182 -> 8.292141 s`, time loop `40.276851 -> 36.279080 s`
  - `Channel2D_Simple`: FE assembly `0.685674 -> 0.356148 s`, time loop `0.854352 -> 0.574375 s`
  - `vortex_shedding`: FE assembly `2.139435 -> 1.144139 s`, time loop `21.983361 -> 23.706173 s`
  - `pipe_RCR_3d`: FE assembly `10.352840 -> 4.286416 s`, time loop `11.355194 -> 5.323722 s`
  - `pipe_RCR_3d_RCRCR`: FE assembly `8.761214 -> 3.768851 s`, time loop `15.262578 -> 11.065862 s`
  - `pipe_simple`: FE assembly `13.746440 -> 5.723541 s`, time loop `15.147560 -> 7.258278 s`
  - `iliac_artery`: FE assembly `92.053131 -> 36.429237 s`, time loop `127.533840 -> 69.632548 s`
- Nonlinear result:
  - all accepted steps in the final measured matrix reported `converged=1`

- The next viable architectural step is a constrained fused-JIT prototype, not a wholesale assembler rewrite.
- Current JIT residual kernels already reconstruct current-state values from local coefficients; the missing fused work is mainly basis push-forward and related FE-side handoff overhead.
- The first safe prototype scope is cell-domain scalar-basis gradients with the current physical-gradient path preserved as the outer FE fallback.

### 2026-03-28: monolithic follow-up qualification

- Change set:
  - true batched monolithic compiled dispatch in `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
  - matrix-only compiled monolithic residual handling with exact vector fallback
  - shared per-cell coefficient/history cache reuse between field reconstruction and block setup
  - affine coupled scalar-cache reuse for block-0 as well as follower blocks
  - coefficient-only handoff for JIT-ready fallback blocks
  - raw contiguous `K*u` helpers in `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
  - Hessian packing fix in `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h` for the batched compiled coupled ABI
- Validation artifacts:
  - exact-path matrix: `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/final_matrix_exact/`
  - compiled compare matrix: `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/compare/`
  - compiled performance probe: `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/final_matrix/`
- Result:
  - targeted FE parity coverage passed on the default exact path and on the opt-in compiled compare path
  - the real-case compiled compare matrix was clean with `SVMP_FE_ENABLE_MONOLITHIC_COMPILED_DISPATCH=1`, `SVMP_FE_COMPARE_MONOLITHIC_COMPILED=1`, `SVMP_FE_COMPARE_MONOLITHIC_MAX_CELLS=4`
  - the opt-in compiled monolithic path remained far slower than the exact path on every qualified fluid case, so it was not promoted to the production default
  - the exact production path improved on every qualified fluid case versus `Documentation/qualification_logs/20260328_kernel_batch_basis_phase1/final_matrix/`
- Exact-path comparison versus `20260328_kernel_batch_basis_phase1/final_matrix`:
  - `channel2d`: FE assembly `8.094485 -> 6.470769 s`, time loop `36.279080 -> 34.465194 s`
  - `channel2d_simple`: FE assembly `0.414273 -> 0.301716 s`, time loop `0.574375 -> 0.456485 s`
  - `vortex_shedding`: FE assembly `1.145903 -> 0.995596 s`, time loop `23.706173 -> 23.327462 s`
  - `pipe_rcr_3d`: FE assembly `3.856757 -> 3.307784 s`, time loop `5.323722 -> 4.662353 s`
  - `pipe_rcr_3d_rcrcr`: FE assembly `3.545970 -> 2.818439 s`, time loop `11.065862 -> 9.911781 s`
  - `pipe_simple`: FE assembly `4.590686 -> 3.785898 s`, time loop `7.258278 -> 6.149500 s`
  - `iliac_artery`: FE assembly `32.078952 -> 26.070436 s`, time loop `69.632548 -> 62.527526 s`
- Nonlinear result:
  - all accepted steps still reported `converged=1`
  - Newton iteration sequences matched the prior qualified exact-path matrix on every case

### 2026-03-06: fused-JIT prototype plan narrowing

- Change: narrowed the broader fused-kernel plan to a cell-only prototype that moves scalar-basis physical gradient evaluation into the JIT from raw reference gradients plus `J^{-T}`.
- Scope:
  - keep the existing FE `AssemblyContext` / physical-gradient path available outside the JIT
  - preserve physics agnosticism by adding generic raw-reference-basis ABI data rather than hard-coding any Navier-Stokes logic
  - defer batching / SoA codegen and Hessian fusion until the scalar gradient path is validated
- Intended verification:
  - exact nonlinear convergence check on `Channel2D`, `Channel2D_Simple`, and `vortex_shedding`
  - `perf stat` comparison after clearing `~/.cache/svMultiPhysics/jit_cache`
- Verdict: plan only. The implementation attempt below was rejected.

### 2026-03-06: fused-JIT reference-gradient prototype

- Change:
  - extended the V6 JIT ABI with raw reference-gradient pointers
  - passed those raw gradients through `JITKernelWrapper`
  - changed the LLVM lowering for scalar-basis `grad`, `div`, and `curl` to reconstruct physical gradients inside the JIT via `J^{-T} * grad_ref`
- Files:
  - `Code/Source/solver/FE/Assembly/AssemblyContext.h`
  - `Code/Source/solver/FE/Assembly/JIT/KernelArgs.h`
  - `Code/Source/solver/FE/Forms/JIT/JITCompiler.cpp`
  - `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
  - `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Assembly/test_JITKernelArgsPacking.cpp`
- Artifact: `/tmp/svmp_fused_refgrad_perf_20260306`
- Result:
  - the prototype did shift work out of the FE-side handoff and into the kernel
  - warm cell-loop shares moved roughly to:
    - `Channel2D`: kernel `~65%`, `prepareBasis ~10%`, `sol gather ~9%`, `field solutions ~6%`, `insert ~8%`
    - `vortex_shedding`: kernel `~63%`, `prepareBasis ~9.5%`, `sol gather ~9%`, `field solutions ~6%`, `insert ~10%`
  - but total runtime regressed on all three cases:
    - `Channel2D_Simple`: `Total time loop 2.019822 s`
    - `Channel2D`: `Total time loop 126.836202 s`
    - `vortex_shedding`: `Total time loop 336.281294 s`
  - exact nonlinear traces were not preserved
    - all three cases still converged cleanly in 20 accepted steps
    - Newton iteration counts stayed aligned
    - residuals and some linear iteration counts drifted slightly versus the verified baseline
- Read:
  - this confirms the architectural direction has real leverage, but the naive implementation is too expensive
  - the current lowering recomputes `J^{-T} * grad_ref` repeatedly at every basis access, which increases arithmetic cost and register pressure enough to outweigh the FE-side savings
- Verdict: rejected and reverted. Any future fused-JIT follow-up needs q-point-level hoisting/reuse inside the generated kernel, not repeated inline reconstruction at each access.

- The `Channel2D` OOP assembly count is not only Newton `J+r`.
- Trace + source inspection show substantial FE assembly outside `NewtonSolver`.
- The main extra source on `Channel2D` is not stage residual-addition for generalized-alpha.
  The actual dominant extra pass is the accepted-step finalization path in `TimeLoop`, which did a full residual-only `transient_finalize.assemble(...)` once per accepted step and discarded the result.
- That final-state residual assembly explains `20` of the extra OOP `assembleOperator` calls on `Channel2D`.
- The remaining `2` extra calls on `Channel2D` come from first-step generalized-alpha `uDot` initialization (one residual-only and one matrix-only assembly).

### 2026-03-06: remove dead accepted-step final residual assembly

- Change: removed the accepted-step `transient_finalize.assemble(...)` residual-only pass that populated `scratch_vec0` and was never consumed afterward.
- Files: `Code/Source/solver/FE/TimeStepping/TimeLoop.cpp`
- Artifacts:
  - `/tmp/channel2d_remove_final_resid_20260306.log`
  - `/tmp/channel2d_simple_remove_final_resid_20260306.log`
  - `/tmp/channel2d_simple_remove_final_resid_rerun_20260306.log`
  - `/tmp/vortex_remove_final_resid_20260306.log`
- Result:
  - exact nonlinear trace match on `Channel2D`, `Channel2D_Simple`, and `vortex_shedding`
  - `Channel2D`: assembly calls `92 -> 72`, FE assembly `23.789129 -> 22.374438 s`, wall `135.11 -> 134.82 s`
  - `Channel2D_Simple`: FE assembly `1.103462 -> 1.067306 s`; first wall run was a noisy `27.34 s` outlier, rerun dropped to `23.78 s` with the same nonlinear history
- `vortex_shedding`: FE assembly `14.877579 -> 13.718386 s`, wall `331.44 -> 323.57 s`
- Verdict: keepable.

### 2026-03-06: skip first-step generalized-alpha PDE-consistent `uDot` solve

- Change: replaced the one-time first-step generalized-alpha `uDot` initialization solve with the existing displacement-history initializer, removing the residual-only and matrix-only startup assemblies.
- Files: `Code/Source/solver/FE/TimeStepping/TimeLoop.cpp`
- Artifacts:
  - `/tmp/channel2d_uDot_histinit_20260306.log`
  - `/tmp/channel2d_simple_uDot_histinit_20260306.log`
  - `/tmp/vortex_uDot_histinit_20260306.log`
- Result:
  - all three cases still converged in 20 accepted steps
  - `Channel2D` wall improved sharply to `76.31 s`, but the nonlinear trace changed materially from the verified baseline starting at step 0
  - `Channel2D_Simple` wall regressed slightly to `26.44 s` and the nonlinear trace also changed materially
- `vortex_shedding` wall improved slightly to `327.11 s`, with a similarly shifted nonlinear trace
- Verdict: rejected and reverted. This changes the generalized-alpha initial-rate state, so it is not a safe “assembly-only” optimization.

### 2026-03-06: check reuse of first-step generalized-alpha startup assemblies in Newton

- Change: investigated whether the one-time `uDot` initialization residual/matrix work could be reused by the first Newton assembly without changing the time integration.
- Files inspected: `Code/Source/solver/FE/TimeStepping/TimeLoop.cpp`, `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`, `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- Result:
  - direct reuse into Newton is not valid with the current design
  - the `uDot` initialization residual and dt-only Jacobian are assembled at `u_n`
  - the first Newton solve starts from the predicted stage state `u_n + alpha_f * dt * uDot_n`
  - because the states differ, reusing those startup outputs in Newton would change the nonlinear path unless the FE library gains a true same-op, split-time-integration fused assembly feature
  - clean baseline `Channel2D` logs show the two startup assemblies are about `0.135 s` and `0.266 s`, so the total exact-behavior upside is only about `0.40 s`
- Verdict: no keepable code change landed from this check. The exact-behavior path would require a larger FE-infrastructure feature for split-context same-op fused assembly, and the payoff is modest relative to the remaining assembly gap.

### 2026-03-06: setup-time operator assembly plans

- Change: built setup-time operator assembly plans so `assembleOperator()` reuses pre-resolved term metadata for cell, boundary, interior-face, interface-face, and global terms instead of rebuilding that dispatch data every call.
- Files: `Code/Source/solver/FE/Systems/FESystem.h`, `Code/Source/solver/FE/Systems/FESystem.cpp`, `Code/Source/solver/FE/Systems/SystemSetup.cpp`, `Code/Source/solver/FE/Systems/SystemAssembly.cpp`
- Artifact: `/tmp/svmp_plan_pack_phase`
- Result:
  - validated together with the packed-slab change below
  - exact nonlinear trace match on `Channel2D`, `Channel2D_Simple`, and `vortex_shedding`
  - contributes to a combined assembly drop on all three cases without changing assembly ordering or formulation behavior
- Verdict: keepable as FE-infrastructure cleanup.

### 2026-03-06: packed per-assemble coefficient and history slabs

- Change: packed solution/history coefficients once per `(source, DOF map, offset)` table and reused the packed cell slabs for repeated cell/face gathers and field-solution reconstruction.
- Files: `Code/Source/solver/FE/Assembly/StandardAssembler.h`, `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/svmp_plan_pack_phase`
- Result:
  - first implementation preferred raw spans over backend views when both existed
  - that broke `Channel2D` immediately (`NaN` residuals, 2 linear iterations/solve), so it was rejected
  - final implementation kept backend views authoritative when available and only used raw spans when no view existed
  - final combined results versus the current keepable post-`TimeLoop` baseline:
    - `Channel2D`: exact nonlinear trace, FE assembly `22.374438 -> 20.508209 s`, wall `134.82 -> 132.74 s`
    - `Channel2D_Simple`: exact nonlinear trace, FE assembly `1.067306 -> 0.972165 s`, wall `23.78 -> 26.22 s`
    - `vortex_shedding`: exact nonlinear trace, FE assembly `13.718386 -> 12.947305 s`, wall `323.57 -> 332.05 s`
- Verdict: keepable for the assembly-first goal. It lowers FE assembly on all three cases and improves the most FE-sensitive case (`Channel2D`) end-to-end, but wall-time impact is mixed where assembly is only a small share of total runtime.

### 2026-03-06: borrowed AssemblyContext coefficient spans plus chunked packed gathers

- Change:
  - `AssemblyContext` stopped eagerly copying active and history coefficient spans into owned arrays and instead borrowed the incoming spans while rebuilding values/gradients.
  - `StandardAssembler` replaced the full-table packed gather cache with a smaller 64-cell chunked packed cache keyed by `(source, DOF map, offset)` and reused it from cell/face gather paths.
- Files:
  - `Code/Source/solver/FE/Assembly/AssemblyContext.h`
  - `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`
  - `Code/Source/solver/FE/Assembly/StandardAssembler.h`
  - `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/svmp_coeff_chunk_opt_perf`
- Result:
  - exact nonlinear trace match on `Channel2D`, `Channel2D_Simple`, and `vortex_shedding`
  - `perf stat` vs `/tmp/svmp_perf_20260306_current`:
    - `Channel2D_Simple`: FE assembly `0.944443 -> 0.962841 s`, time loop `1.339483 -> 1.355967 s`, LLC miss `9.67% -> 9.68%`
    - `Channel2D`: FE assembly `20.319212 -> 21.187136 s`, time loop `112.800694 -> 114.264197 s`, cache-miss rate `41.93% -> 42.67%`, LLC miss `46.60% -> 47.27%`
    - `vortex_shedding`: FE assembly `13.566002 -> 13.301815 s`, time loop `310.394994 -> 313.063265 s`, cache-miss rate `39.60% -> 38.41%`, LLC miss `46.69% -> 44.91%`
  - `perf record` on `Channel2D` showed modest FE-local sample improvements:
    - `prepareBasis`: `3.10% -> 2.78%`
    - `populateFieldSolutionData`: `2.30% -> 2.05%`
    - `AssemblyContext::setSolutionCoefficients`: `1.28% -> 1.16%`
    - `__memmove/__memcpy`: `1.21% -> 0.91%`
  - but `AssemblyContext::setPreviousSolutionCoefficientsK` rose slightly (`1.44% -> 1.53%`) and the larger FE-local sample improvements did not translate into a net end-to-end win on the larger cases
- Verdict: rejected and reverted. The reduced-copy path helped a few FE-local samples, but the chunked cache still increased overall runtime on `Channel2D` and `Channel2D_Simple`, and `vortex_shedding` only improved assembly while slowing the full loop.

### 2026-03-07: enable shared q-point JIT cache for ordinary cell dispatch

- Change:
  - reused the existing `LLVMGen` q-point cache outside coupled-block kernels by enabling it for the ordinary non-face, non-coupled cell dispatch path as well
  - this keeps the exact same math and only memoizes already-computed current-solution / gradient / history reductions across terms
- File:
  - `Code/Source/solver/FE/Forms/JIT/LLVMGen.cpp`
- Artifact: `/tmp/svmp_qpcache_terms_perf_20260306`
- Result:
  - exact nonlinear trace match on `Channel2D`, `Channel2D_Simple`, and `vortex_shedding`
  - warm fused cell-loop splits stayed essentially unchanged:
    - `Channel2D`: `prepareBasis 17.0%`, `field solutions 10.9%`, `sol gather 16.5%`, `kernel 37.4%`, `insert 14.0%`
    - `vortex_shedding`: `prepareBasis 16.4%`, `field solutions 10.7%`, `sol gather 15.8%`, `kernel 36.1%`, `insert 16.9%`
  - cold `perf stat` totals versus `/tmp/svmp_perf_20260306_current`:
    - `Channel2D_Simple`: FE assembly `0.944443 -> 0.998910 s`, time loop `1.339483 -> 1.415559 s`, LLC miss `9.67% -> 9.76%`
    - `Channel2D`: FE assembly `20.319212 -> 20.511714 s`, time loop `112.800694 -> 110.822681 s`, LLC miss `46.60% -> 47.00%`
    - `vortex_shedding`: FE assembly `13.566002 -> 13.017350 s`, time loop `310.394994 -> 309.526380 s`, LLC miss `46.69% -> 45.26%`
- Read:
  - the idea is numerically safe, but it does not produce a clear FE-assembly win on the main target case
  - `Channel2D` and `vortex_shedding` moved slightly in total loop time, but the FE assembly totals stayed flat-to-worse, and `Channel2D_Simple` clearly regressed
- Verdict: rejected and reverted. This memoization is too small a lever for the remaining assembly gap.

### 2026-03-07: setup-time resolved FSILS scatter path for cell insertion

- Change:
  - enabled the existing pre-resolved matrix-slot tables in `StandardAssembler` for cell assembly when the backend exposes a reusable matrix layout handle
  - guarded the path with `SVMP_FE_RESOLVED_MATRIX_INSERT=1`
  - kept the optimization backend-local by only changing the cell insertion handoff, not the FE kernel ABI
- File:
  - `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifact: `/tmp/svmp_resolved_insert_env_20260307`
- Result:
  - exact nonlinear trace match on `Channel2D`, `Channel2D_Simple`, and `vortex_shedding`
  - cold `perf stat` versus `/tmp/svmp_perf_20260306_current`:
    - `Channel2D_Simple`: FE assembly `0.944443 -> 1.040015 s`, time loop `1.339483 -> 1.453098 s`, LLC miss `9.67% -> 11.12%`
    - `Channel2D`: FE assembly `20.319212 -> 21.479095 s`, time loop `112.800694 -> 112.326276 s`, LLC miss `46.60% -> 48.38%`
    - `vortex_shedding`: FE assembly `13.566002 -> 14.004854 s`, time loop `310.394994 -> 310.500440 s`, LLC miss `46.69% -> 47.30%`
  - warm cell-loop insert slices did not improve materially:
    - `Channel2D`: insert `13.9%`
    - `vortex_shedding`: insert `16.1%`
- Read:
  - the resolved slot tables increase working-set size enough to hurt cache behavior
  - the existing FSILS block-scatter path is already good enough that this extra table layer is net negative
- Verdict: rejected and reverted.

## Current Read

- The JITed kernel is no longer the dominant part of assembly on the larger cases.
- `Channel2D` OOP assembly is still far above legacy because:
  - the FE infrastructure around the kernel is expensive
  - the OOP path assembles more often than legacy, including stage-residual addition passes outside Newton itself
- The next workstream is to reduce assembly frequency where the Newton driver can safely reuse work, while keeping the FE layer general and physics-agnostic.

### 2026-03-28: monolithic exact path combined insertion

- Change:
  - finished the existing monolithic-batch fused-insert path in `StandardAssembler` instead of adding a new assembler-side mechanism
  - zeroed and reused the preallocated combined scratch buffers per batch, scattered each block into one combined local matrix/vector per cell, and performed one final combined insertion after all block kernels ran
  - kept the optimization conservative by leaving owned-row filtering on the existing per-block insertion path
- File:
  - `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
- Artifacts:
  - `/tmp/pipe_simple_monolithic_fused_insert.log`
  - `/tmp/pipe_RCR_3d_monolithic_fused_insert.log`
  - `/tmp/channel2d_simple_monolithic_fused_insert.log`
  - `/tmp/perf_pipe_simple_monolithic_fused_insert.txt`
  - `/tmp/perf_pipe_simple_monolithic_fused_insert_notiming.txt`
- Result:
  - monolithic parity/system tests stayed green:
    - `BackendParity.ResidualPath_JitMonolithicSparsityMatchesFallback`
    - `MixedFormPerformance.InstallFormulation_MonolithicJITParity_VersusPerBlockFallback`
    - `MixedFormPerformance.InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientVMSResidual`
    - `MonolithicCoupling.MixedJacobianBlockFDVerification`
  - exact nonlinear traces remained stable on the measured cases:
    - `pipe_simple`: `5,5` Newton iterations, converged residuals `1.14e-09`, `1.27e-08`
    - `pipe_RCR_3d`: `4,3` Newton iterations, converged residuals `5.98e-14`, `7.90e-11`
    - `Channel2D_Simple`: `3` Newton iterations on every step
  - versus the previous OOP monolithic-batch baseline:
    - `pipe_simple`: FE assembly `5.224178 -> 3.924692 s` (`1.33x` faster, `-24.9%`), total `7.596161 -> 6.046571 s`
    - `pipe_RCR_3d`: FE assembly `4.660291 -> 3.479592 s` (`1.34x` faster, `-25.3%`), total `6.054419 -> 4.703307 s`
    - `Channel2D_Simple`: FE assembly `0.483791 -> 0.356590 s` (`1.36x` faster, `-26.3%`), total `0.634266 -> 0.469737 s`
  - warm monolithic cell-loop timing on `pipe_simple` shifted materially:
    - steady-state total `~0.46-0.49 s -> ~0.35-0.36 s`
    - `insert` `~0.10-0.11 s -> ~0.02-0.03 s`
    - `kernel` also fell modestly because the path now avoids repeated per-block insertion handoff work
  - `perf stat` on `pipe_simple` without assembly timing showed fewer instructions and fewer cache misses than `/tmp/perf_pipe_simple_oop_final.txt`, but a noisier single-sample elapsed time, so the in-app FE assembly timings remain the main performance measure
- Verdict: keep. This is a contained FE-side optimization with clear code ownership, preserved parity, and a meaningful assembly reduction on the monolithic 3D cases.

### 2026-03-29: field-evaluation cache, resolved FSILS slot runs, and prepared exact JIT batch packing

- Change:
  - added a per-cell evaluated-field cache in `StandardAssembler` that is separate from the existing coefficient cache and reuses current/history field evaluations when the same cell is populated multiple times
  - added a backend-local contiguous-run fast path in `FsilsMatrix::addResolvedMatrixEntries()` so resolved slot regions are updated with tight pointer walks instead of one scalar scatter at a time
  - replaced full per-element `CellKernelArgsV6` template copies in `JITKernelWrapper::computeCellBatch()` with a narrower prepared batch template that reuses one packed side template per batch and only patches the varying pointers
  - added direct regression coverage for previous field history binding/copy in `AssemblyContext` and for contiguous/irregular FSILS resolved-entry insertion
- Files:
  - `Code/Source/solver/FE/Assembly/StandardAssembler.h`
  - `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
  - `Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp`
  - `Code/Source/solver/FE/Forms/JIT/JITKernelWrapper.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Assembly/test_AssemblyContext.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp`
- Artifacts:
  - `Documentation/qualification_logs/20260329_field_cache_fsils_jit_batch/hotspots/`
  - `Documentation/qualification_logs/20260329_field_cache_fsils_jit_batch/final_matrix_exact/`
- Validation:
  - build passed:
    - `cmake --build build/svMultiPhysics-build --target svmultiphysics test_fe_assembly test_fe_backends test_fe_systems -j8`
  - FE/unit coverage passed:
    - `./build/svMultiPhysics-build/bin/test_fe_assembly --gtest_filter='AssemblyContextMultiField.*:StandardAssembler*'`
    - `./build/svMultiPhysics-build/bin/test_fe_backends --gtest_filter='FsilsBackend.ResolvedMatrixEntries*'`
    - `./build/svMultiPhysics-build/bin/test_fe_systems --gtest_filter='BackendParity.ResidualPath_JitMonolithicSparsityMatchesFallback:MixedFormPerformance.InstallFormulation_MonolithicJITParity_VersusPerBlockFallback:MixedFormPerformance.InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientVMSResidual:MonolithicCoupling.MixedJacobianBlockFDVerification'`
  - note:
    - the broader `./build/svMultiPhysics-build/bin/test_fe_backends --gtest_filter='FsilsBackend.*'` run still has the pre-existing `SolveBlockSchurDof3SingleNode` failure (`BlockSchur requires block_layout with saddle-point annotation`), which is unrelated to the resolved-entry change
- Representative exact-path timing probe versus `/tmp/svmp_exact_hotspots_20260329`:
  - `pipe_simple`: warm monolithic cell-loop `0.217673 -> 0.230103 s`; `shared fields 0.033497 -> 0.037975 s`; `kernel 0.138093 -> 0.145729 s`; `insert 0.021388 -> 0.020608 s`
  - `Channel2D`: warm monolithic cell-loop `0.149370 -> 0.165388 s`; `shared fields 0.020990 -> 0.023703 s`; `kernel 0.073699 -> 0.078796 s`; `insert 0.012652 -> 0.016475 s`
  - `iliac_artery`: warm monolithic cell-loop `1.959046 -> 2.285217 s`; `shared fields 0.241429 -> 0.291312 s`; `kernel 1.221882 -> 1.379931 s`; `insert 0.213769 -> 0.289296 s`
- Full exact-path qualification matrix versus `Documentation/qualification_logs/20260328_monolithic_compiled_batch_phase2/final_matrix_exact/`:
  - `Channel2D`: exact Newton trace preserved; FE assembly `6.470769 -> 6.999178 s` (`+8.2%`), time loop `34.465194 -> 35.230505 s` (`+2.2%`)
  - `Channel2D_Simple`: exact Newton trace preserved; FE assembly `0.301716 -> 0.323491 s` (`+7.2%`), time loop `0.456485 -> 0.484028 s` (`+6.0%`)
  - `vortex_shedding`: exact Newton trace preserved; FE assembly `0.995596 -> 0.969203 s` (`-2.7%`), time loop `23.327462 -> 21.535602 s` (`-7.7%`)
  - `pipe_RCR_3d`: exact Newton trace preserved; FE assembly `3.307784 -> 3.300638 s` (`-0.2%`), time loop `4.662353 -> 4.607360 s` (`-1.2%`)
  - `pipe_RCR_3d_RCRCR`: exact Newton trace preserved; FE assembly `2.818439 -> 2.918460 s` (`+3.5%`), time loop `9.911781 -> 9.668843 s` (`-2.5%`)
  - `pipe_simple`: exact Newton trace preserved; FE assembly `3.785898 -> 3.765205 s` (`-0.5%`), time loop `6.149500 -> 6.021070 s` (`-2.1%`)
  - `iliac_artery`: exact Newton trace preserved; FE assembly `26.070436 -> 25.823774 s` (`-0.9%`), time loop `62.527526 -> 59.406323 s` (`-5.0%`)
- Read:
  - correctness is solid: the targeted FE parity slice stayed green, the qualified matrix kept `converged=1` on all accepted steps, and the Newton iteration sequences were unchanged
  - the performance signal is mixed:
    - the instrumentation-heavy warm exact-path hotspot probe regressed on all three representative cases
    - the full qualification matrix improved the main 3D monolithic cases slightly (`pipe_simple`, `pipe_RCR_3d`, `iliac_artery`) and improved total loop time on five of seven cases
    - the 2D cases regressed measurably, so this phase did not deliver the clean bucket win the plan targeted
- Verdict: keep with caution. The code changes are contained and correctness-clean, and they do help several main 3D exact-path cases, but the hotspot regressions and the 2D slowdowns mean the next pass should focus on why the new cache/packing layers increased per-cell exact-path cost in the representative probe.

### 2026-03-29: geometry-copy helper, field subset copy, row-group caching, and FSILS resolved-vector runs

- Change:
  - added `AssemblyContext::copyGeometryDataFrom()` so prepared quadrature/geometry state can be cloned without repeating the setter pipeline
  - added `AssemblyContext::copyFieldSolutionDataSubsetFrom()` so monolithic block contexts only clone the field/history data they actually consume
  - replaced the monolithic exact-path geometry-copy lambda in `StandardAssembler` with the shared helper and hoisted the reused batch-slot geometry copy out of the per-block exact loop
  - added row-group DOF caches alongside the existing trial-group cache in the monolithic batch path and reused the same grouped coefficient/history gathers in the non-batched exact path
  - added a contiguous-run fast path to `FsilsVector` resolved insertion and covered both contiguous and irregular resolved-vector cases in unit tests
- Files:
  - `Code/Source/solver/FE/Assembly/AssemblyContext.h`
  - `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`
  - `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`
  - `Code/Source/solver/FE/Backends/FSILS/FsilsVector.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Assembly/test_AssemblyContext.cpp`
  - `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp`
  - `Documentation/plan_basis_field_insert_orchestration_20260329.md`
- Artifacts:
  - `Documentation/qualification_logs/20260329_basis_field_insert_orchestration/final_matrix_exact/`
- Validation:
  - build passed:
    - `cmake --build build/svMultiPhysics-build --target test_fe_assembly test_fe_backends test_fe_systems svmultiphysics -j8`
  - FE/unit coverage passed:
    - `./build/svMultiPhysics-build/bin/test_fe_assembly --gtest_filter='AssemblyContextMultiField.*:StandardAssembler*'`
    - `./build/svMultiPhysics-build/bin/test_fe_backends --gtest_filter='FsilsBackend.Resolved*'`
    - `./build/svMultiPhysics-build/bin/test_fe_systems --gtest_filter='BackendParity.ResidualPath_JitMonolithicSparsityMatchesFallback:MixedFormPerformance.InstallFormulation_MonolithicJITParity_VersusPerBlockFallback:MixedFormPerformance.InstallFormulation_MonolithicJITParity_GeneralizedAlphaTransientVMSResidual:MonolithicCoupling.MixedJacobianBlockFDVerification'`
- Full exact-path qualification matrix versus `Documentation/qualification_logs/20260329_field_cache_fsils_jit_batch/final_matrix_exact/`:
  - comparison metric:
    - summed per-operator `Cell terms` from the fresh logs
  - `Channel2D`: exact Newton trace preserved; cell assembly `7.133677 -> 6.302098 s` (`-11.7%`), time loop `35.230505 -> 31.362240 s` (`-11.0%`)
  - `Channel2D_Simple`: exact Newton trace preserved; cell assembly `0.303246 -> 0.275675 s` (`-9.1%`), time loop `0.484028 -> 0.448744 s` (`-7.3%`)
  - `vortex_shedding`: exact Newton trace preserved; cell assembly `0.967379 -> 0.940665 s` (`-2.8%`), time loop `21.535602 -> 21.467017 s` (`-0.3%`)
  - `pipe_RCR_3d`: exact Newton trace preserved; cell assembly `3.603168 -> 3.719754 s` (`+3.2%`), time loop `4.607360 -> 4.760073 s` (`+3.3%`)
  - `pipe_RCR_3d_RCRCR`: exact Newton trace preserved; cell assembly `3.072390 -> 3.012271 s` (`-2.0%`), time loop `9.668843 -> 9.489056 s` (`-1.9%`)
  - `pipe_simple`: exact Newton trace preserved; cell assembly `4.581268 -> 4.611105 s` (`+0.7%`), time loop `6.021070 -> 6.124453 s` (`+1.7%`)
  - `iliac_artery`: exact Newton trace preserved; cell assembly `29.439458 -> 29.462328 s` (`+0.1%`), time loop `59.406323 -> 60.180746 s` (`+1.3%`)
- Read:
  - correctness is strong:
    - all accepted steps still reported `converged=1`
    - Newton iteration sequences and final residuals were unchanged on all seven cases
  - performance is mixed:
    - the 2D cases improved cleanly
    - `vortex_shedding` and `pipe_RCR_3d_RCRCR` improved modestly
    - the main 3D monolithic set did not all improve; `pipe_RCR_3d`, `pipe_simple`, and `iliac_artery` regressed slightly
- Verdict: keep the code, but do not claim this phase as a net 3D assembly-speed win. The shared helper structure is cleaner and the nonlinear behavior stayed exact, yet the current orchestration changes did not improve the main 3D monolithic cases as a group.
