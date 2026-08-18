# FE Assembly Follow-Up Plan: Basis, Field, Insert, Orchestration

Date: 2026-03-29

Scope: production exact-path FE assembly only. This pass targets non-kernel assembly overhead in the OOP solver without changing legacy solver behavior or FE semantics.

## Goals

- Reduce repeated geometry and field-state copying in monolithic exact assembly.
- Reduce DOF lookup and coefficient-gather orchestration overhead in fused monolithic batches.
- Improve backend-local vector scatter performance for resolved FSILS insertion.
- Keep nonlinear behavior unchanged on the qualified FE matrix.

## Workstream 1: Basis / Geometry Preparation

- [x] Add an `AssemblyContext` geometry-copy helper so geometry transfer is one shared API instead of repeated setter sequences in `StandardAssembler`.
- [x] Replace local geometry-copy lambdas in `Code/Source/solver/FE/Assembly/StandardAssembler.cpp` with the shared helper.
- [x] Hoist monolithic exact-path batch geometry copies out of the per-block inner loop where the same slot context is reused across blocks for one cell.
- [x] Keep basis preparation exact and unchanged mathematically; only remove redundant context setup/copy work.

Implementation targets:

- `Code/Source/solver/FE/Assembly/AssemblyContext.h`
- `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`

## Workstream 2: Field / History Reconstruction

- [x] Add an `AssemblyContext` field-solution subset copy helper that preserves current and history field data for only the requested field IDs.
- [x] Build per-block field-copy plans for monolithic fused assembly so block contexts copy only the field state they actually consume.
- [x] Use the subset helper in the monolithic batched and non-batched exact paths instead of always cloning the full shared union field state.
- [x] Preserve fallback behavior for blocks that require field data not present in the shared union context.

Implementation targets:

- `Code/Source/solver/FE/Assembly/AssemblyContext.h`
- `Code/Source/solver/FE/Assembly/AssemblyContext.cpp`
- `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`

## Workstream 3: Insertion / Backend Scatter

- [x] Add a contiguous-run fast path to FSILS resolved-vector insertion.
- [x] Apply the same fast path when `FsilsVectorView::addVectorEntries()` first resolves entries and then scatters them.
- [x] Add focused unit coverage for resolved-vector contiguous and irregular insertion.
- [x] Leave matrix semantics unchanged; this pass is limited to vector scatter because that path is still purely scalar today.

Implementation targets:

- `Code/Source/solver/FE/Backends/FSILS/FsilsVector.cpp`
- `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp`

## Workstream 4: Assembly Frequency / Orchestration

- [x] Add row-group DOF caches to the monolithic batched exact path, matching the existing trial-group cache pattern for shared column spaces.
- [x] Reuse cached row and column DOF spans across blocks that share the same `(dof_map, dof_offset, space)` tuple.
- [x] Route block field-state setup through the new subset-copy helper so blocks with no field needs avoid copying union field state.
- [x] Keep timing and result collection unchanged so the new runs remain comparable to the archived exact-path baselines.

Implementation targets:

- `Code/Source/solver/FE/Assembly/StandardAssembler.h`
- `Code/Source/solver/FE/Assembly/StandardAssembler.cpp`

## Validation

- [x] Rebuild `svmultiphysics`, `test_fe_assembly`, `test_fe_backends`, and `test_fe_systems`.
- [x] Run `test_fe_assembly --gtest_filter='AssemblyContextMultiField.*:StandardAssembler*'`.
- [x] Run `test_fe_backends --gtest_filter='FsilsBackend.Resolved*'`.
- [x] Run the monolithic parity subset in `test_fe_systems`.
- [x] Re-run the qualified exact-path FE case matrix and compare FE assembly timing and Newton traces against the latest exact-path baseline.

## Acceptance Criteria

- [x] All new helper/unit tests pass.
- [x] Monolithic parity tests remain green.
- [x] Every accepted step in the exact-path case matrix reports `converged=1`.
- [x] Newton iteration sequences remain unchanged or improve.
- [ ] FE assembly time improves on the main 3D monolithic cases (`pipe_simple`, `pipe_RCR_3d`, `pipe_RCR_3d_RCRCR`, `iliac_artery`) relative to the latest exact-path baseline.
- [x] Any 2D regressions are either eliminated or explicitly documented if the net matrix result is still favorable.

## Outcome

- The implementation is correctness-clean and the nonlinear traces stayed exact on the full qualified matrix.
- Using the fresh `20260329_basis_field_insert_orchestration/final_matrix_exact` logs and comparing the summed per-operator `Cell terms` bucket against `20260329_field_cache_fsils_jit_batch/final_matrix_exact`, the result is mixed:
  - `Channel2D`: `7.133677 -> 6.302098 s`
  - `Channel2D_Simple`: `0.303246 -> 0.275675 s`
  - `vortex_shedding`: `0.967379 -> 0.940665 s`
  - `pipe_RCR_3d`: `3.603168 -> 3.719754 s`
  - `pipe_RCR_3d_RCRCR`: `3.072390 -> 3.012271 s`
  - `pipe_simple`: `4.581268 -> 4.611105 s`
  - `iliac_artery`: `29.439458 -> 29.462328 s`
- Because the main 3D monolithic set did not all improve, the final performance acceptance item remains intentionally unchecked.
