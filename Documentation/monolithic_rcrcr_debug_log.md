# Monolithic RCRCR Debug Log

This file tracks experiments, hypotheses, and outcomes while debugging poor nonlinear convergence of the monolithic `AuxiliaryState` path for the `RCRCR` outlet model on `tests/cases/fluid/pipe_RCR_3d`.

## Scope

- Target case: `tests/cases/fluid/pipe_RCR_3d/solver_perf_oop_rcrcr.xml`
- Solver path: new OOP solver only
- Focus: monolithic `AuxiliaryState` coupling for the `RCRCR` outlet
- Non-goals: legacy solver changes, line-search tuning

## Established Constraints

- Monolithic auxiliary blocks should use the same PDE time-integration scheme and stencil as the coupled PDE solve.
- We are not using line-search changes as the main fix path.

## Previously Confirmed Items

### Auxiliary sensitivity blocks

- `B = dR_PDE/dx_aux` was checked against the expected reduced expression and matched to machine precision.
- The reduced bordered solve algebra `du = K^{-1}(r - B dx)` with `dx` from the Schur complement matched dense reference solves in unit coverage.

### Previously fixed infrastructure issues

- Missing direct coupling `dP_out/dQ = Rp` path was added.
- FSILS destructive matrix modification around bordered `K^{-1}B` solves was handled with Jacobian snapshot/restore.
- Bordered storage reset issues were fixed.
- Boundary-scope entity count handling was fixed.
- Qualified-name and symbolic differentiation plumbing bugs were fixed.
- `D` received the missing `1/dt` contribution.
- Monolithic auxiliary trial state resets across Newton assembly were fixed.
- Auxiliary generalized-alpha staging/history support was added to the monolithic path.
- Nonlinear stagnation no longer overrides requested tolerances.
- Exact outer-product fallback was added when direct coupling is not representable as a symmetric rank-1 update.

## Experiments And Outcomes

### Native direct-coupling rank-1 update vs explicit matrix insertion

- Explicit direct-coupling insertion into the PDE matrix was much less stable for monolithic `RCRCR`.
- Preferring native FSILS rank-1 updates when available stopped the catastrophic blow-up seen with explicit insertion.
- Even after that, monolithic `RCRCR` still converged poorly compared with partitioned.

### Disabling direct coupling

- With direct coupling disabled, monolithic `RCRCR` became less unstable but converged poorly and slowly.
- Conclusion: direct coupling is involved, but disabling it is not the answer.

### Final residual mismatch during Newton

- Earlier traces showed low in-loop residuals followed by large `final_check` residuals.
- That was partially explained by the no-line-search path checking residuals before the update inside the loop and only evaluating the updated state in `final_check`.
- This did not explain the whole convergence problem.

### Current traced behavior in the live tree

- One-step traced monolithic `RCRCR` run still shows poor nonlinear behavior.
- Residual sequence in the current tree:
  - `it=0`: `0.609051`
  - `it=1`: `0.4635`
  - `it=2`: `0.109492`
  - `it=3`: `0.398524`
  - `it=4`: `0.12239`
  - `it=5`: `72.4935`
  - `it=6`: `70.8335`
  - later iterations drop again but do not settle cleanly
- The bordered correction norm `||dx_aux||` is usually tiny in these later iterations.
- The bordered `K^{-1}B` column for the first auxiliary state is stable at about `55.7405`.
- The second `K^{-1}B` column is zero, which is expected because `P_out = P1 + Rp * Q` does not depend directly on `P2`.

### Current strongest live clue

- In the current traced run, the auxiliary residual norm reported through `borderedCoupling().g` is essentially zero at every Newton iteration.
- For a monolithic two-state outlet under nonzero outlet flux, that is suspicious and is now a primary investigation target.

## Active Hypotheses

1. `borderedCoupling().g` is being zeroed, skipped, or otherwise not populated correctly in the full pipe case even though unit coverage says it should be nonzero.
2. The FSILS-native direct-coupling path still differs from the exact assembled Jacobian in a way that matters for the bordered monolithic solve.
3. The monolithic PDE solve and the bordered column solves may not be using exactly the same effective operator in FSILS after native rank-1 update bridging.

## Next Checks

- Trace where `borderedCoupling().g` is populated in the full case and compare it against direct evaluation of the outlet model residual.
- Verify whether FSILS-native rank-1 updates are exact for the main solve and bordered column solves in the monolithic path.
- Compare the auxiliary residual assembled in the pipe case against a smaller controlled reproduction if needed.

### 2026-03-27 14:39:06 PDT
- Added trace-only logging for monolithic auxiliary residual assembly in FESystem to inspect x, xdot, inputs, and residual in the live pipe case.
- Fixed missing Logger include after adding monolithic residual trace logging.
- Added debug env switch SVMP_FORCE_EXPLICIT_RANK_ONE to compare FSILS-native direct coupling against exact explicit matrix insertion.

### 2026-03-27 14:58:00 PDT
- Finished the backend-isolation check with a temporary `GMRES` linear-solver variant of the monolithic `RCRCR` pipe case.
- Result: `GMRES` still blows up badly, so the remaining bug is not just BlockSchur-specific.
- Added runtime Jacobian-vs-finite-difference checks on the real pipe case:
  - monolithic `RCRCR` with explicit direct coupling: relative `||Jv-FD||` about `6.7e-3`
  - monolithic `RCRCR` with direct coupling disabled: relative `||Jv-FD||` about `6.3e-3`
  - monolithic `RCR` on the same pipe case: relative `||Jv-FD||` about `1.76e-2`
- Conclusion: the runtime Jacobian check is useful as a coarse diagnostic but it is not isolating the `RCRCR` bug by itself because the simpler `RCR` case also shows a nontrivial PDE-side mismatch while still converging well.
- New live clue:
  - native monolithic `RCR` and native monolithic `RCRCR` are producing materially different direct-coupling vectors on the same pipe mesh (`RCR`: `sigma≈382.432`, `nnz=108`; `RCRCR`: `sigma=121`, `nnz=143`).
  - Since both models are driven by the same outlet flux `Q`, this difference should not exist unless the direct-coupling assembly is following different code paths or reconstructing `dQ/du` differently.
- Added focused trace hook `SVMP_MONO_DIRECT_TRACE` in `FESystem` to log:
  - whether each monolithic direct-coupling term uses the exact FE gradient or Ct-based reconstruction
  - `dO_dI`, gradient nnz/norm, and whether the path collapses to rank-one or falls back to exact outer-product assembly

### 2026-03-27 15:14:00 PDT
- Inspected the new monolithic exact-gradient path against the legacy coupled-boundary path.
- Found a concrete mismatch:
  - `FESystem::assembleBoundaryGradient(...)` explicitly disabled assembler constraints before forming exact `dQ/du`
  - the monolithic `dR/d(output)` vector build also explicitly disabled assembler constraints
  - the legacy `CoupledBoundaryManager` path does not do this
- Working hypothesis:
  - on the real pipe outlet, the outlet face shares edge DOFs with Dirichlet-constrained wall/intersection DOFs
  - the unconstrained exact-gradient path therefore injects constrained columns into the monolithic direct feedthrough term
  - this explains the traced `grad_nnz=143` for monolithic `RCRCR` versus `dQ/du: 108 nonzero entries` on the legacy `RCR` pipe path
- Action taken:
  - changed both monolithic assembly sites to keep the active affine constraints instead of switching them off
  - added a new FE-system regression that constrains one outlet-face velocity DOF and checks the monolithic direct-coupling Jacobian against finite differences only on the free subspace

- Rebuilt test_fe_systems after relaxing the stale unconstrained-support assertion in DirectCouplingRankOneUsesActualOutputSensitivity.

### 2026-03-27 15:31:00 PDT
- Traced the live pipe runs again after the constrained-gradient fix.
- Key observation:
  - monolithic `RCRCR` step 1 now ends with auxiliary residual essentially zero but PDE residual still around `7.9e-2`
  - monolithic `RCR` still converges through the old `SystemAssembly` coupled-Jacobian path, not the new bordered monolithic path
- This means the remaining `RCRCR` problem is not the local auxiliary residual; it is the PDE-side reduced coupling seen by Newton.
- New hypothesis:
  - for the simple one-input / one-output boundary block, the bordered `B D^{-1} Ct` state-mediated outlet gain should be folded into the same PDE rank-one update that already carries the direct `Rp dQ/du` feedthrough
  - then the bordered solve should skip those condensed `B` columns to avoid double counting
- Action taken:
  - added aux-index metadata to `BorderedCouplingData::DirectCouplingRecord`
  - added a Newton-side condensation plan that augments the PDE rank-one updates by the reduced state-mediated gain for the eligible monolithic block
  - zeroes the corresponding `B` columns only in the bordered solve path so the state-mediated coupling is handled once, through the PDE operator

### 2026-03-27 15:37:00 PDT
- First condensation attempt made the pipe case worse (`step 0 ~1.33e-1`, `step 1 ~1.06e-1`).
- Root cause identified immediately after the run:
  - the reduced PDE operator is `K - B D^{-1} Ct`
  - the first condensation patch used `+ dO_dx * D^{-1} * dF_dI`
  - that sign is wrong; the condensed state-mediated rank-one gain must carry the leading minus
- Patched the condensation formula to use the correct reduced-operator sign before rerunning.

### 2026-03-27 15:39:00 PDT
- The sign fix alone still made the pipe case worse (`step 0 ~1.46e-1`, `step 1 ~2.40e-1`).
- Missing term identified:
  - the reduced monolithic PDE solve is not just `(K - B D^{-1} Ct) du = r`
  - it is `(K - B D^{-1} Ct) du = r - B D^{-1} g`
- The first two condensation attempts modified the Jacobian but left the Newton RHS as the uncondensed PDE residual.
- Action taken:
  - added a condensed RHS shift along the same outlet gradient vector with coefficient `B D^{-1} g`
  - apply that shift only to the linear-solve RHS, after residual assembly and before the main PDE solve

### 2026-03-27 15:58:00 PDT
- Added a temporary env-gated switch `SVMP_DISABLE_COUPLED_BLOCK_KERNEL` in `FormsInstaller.cpp` to bypass fused `CoupledBlockKernel` installation for mixed residuals and mixed FormIR.
- Purpose: isolate whether the missing mixed Jacobian blocks come from the fused coupled-block assembly path rather than from residual lowering.
- Result:
  - `NavierStokesOutletFactory.MonolithicRCRCR_MixedFieldJacobianMatchesFD` passed with the fused coupled-block path disabled.
  - `NavierStokesOutletFactory.MonolithicRCRCR_GeneralizedAlphaMixedFieldJacobianMatchesFD` also passed with the fused path disabled.
  - The real `pipe_RCR_3d/solver_perf_oop_rcrcr.xml` monolithic run converged excellently with the fused path disabled:
    - step 0: `3` Newton iterations, `||r|| ≈ 1.22e-13`
    - step 1: `3` Newton iterations, `||r|| ≈ 2.32e-13`
- Conclusion:
  - the bad monolithic `RCRCR` behavior is strongly tied to the `CoupledBlockKernel` fused mixed-residual optimization used by `installFormulation()`'s `coupled_residual_from_jacobian_block` path.
  - the safer fix is to stop using `CoupledBlockKernel` for that specific path, while leaving the optimization available elsewhere.

### 2026-03-27 15:59:20 PDT
- Added temporary env-gated switch `SVMP_DISABLE_COUPLED_BLOCK_KERNEL` in `FormsInstaller.cpp` to bypass fused `CoupledBlockKernel` installation for mixed residuals and mixed FormIR.
- Purpose: isolate whether the missing mixed Jacobian blocks come from the fused coupled-block assembly path rather than from residual lowering.
- Made the fix permanent by disabling `CoupledBlockKernel` creation specifically when `coupled_residual_from_jacobian_block` is active in `installCoupledResidual()`. This is the mixed residual path used by monolithic NS+AuxiliaryState coupling.

### 2026-03-27 16:04:00 PDT
- Re-verified the permanent fix in the current tree without any debug env switches.
- Focused regression checks:
  - `./build/svMultiPhysics-build/bin/test_physics --gtest_filter='NavierStokesOutletFactory.MonolithicRCRCR_MixedFieldJacobianMatchesFD:NavierStokesOutletFactory.MonolithicRCRCR_GeneralizedAlphaMixedFieldJacobianMatchesFD'`
  - both tests passed
- Full case check:
  - `/home/zack/Downloads/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics solver_perf_oop_rcrcr.xml`
  - monolithic `RCRCR` again converged excellently on `pipe_RCR_3d`
    - step 0: `3` Newton iterations, `||r|| = 1.2161978462290207e-13`
    - step 1: `3` Newton iterations, `||r|| = 2.3242511996940827e-13`
- Current diagnosis stands:
  - the remaining monolithic `RCRCR` convergence failure was caused by the fused `CoupledBlockKernel` optimization on the `coupled_residual_from_jacobian_block` path, not by the auxiliary bordered algebra itself.

### 2026-03-27 22:28:00 PDT
- Ran a direct partitioned `RCRCR` A/B on `pipe_RCR_3d` to check whether the same fusion disable also helps the partitioned outlet path.
- Temporary source setup for the experiment:
  - switched `NavierStokesBCFactories.h` `RCRCR` deployment to `.partitioned("BackwardEuler").bind("Q", Q)`
  - compared current installer fix against a temporary revert that re-enabled `CoupledBlockKernel` on the `coupled_residual_from_jacobian_block` path
- Results:
  - partitioned with fusion disabled:
    - step 0: `8` Newton iterations, `||r|| = 2.0866861533040620e-10`
    - step 1: `8` Newton iterations, `||r|| = 6.0140565100865324e-10`
    - total time loop: about `10.73 s`
  - partitioned with fusion enabled:
    - step 0: `8` Newton iterations, `||r|| = 2.0866861533040620e-10`
    - step 1: `8` Newton iterations, `||r|| = 6.0140565100865324e-10`
    - total time loop: about `7.33 s`
- Conclusion:
  - disabling the fused mixed-residual path does not improve partitioned `RCRCR` nonlinear convergence on this case
  - it only slows the partitioned run down
  - the fusion disable should therefore remain targeted to the monolithic-sensitive `coupled_residual_from_jacobian_block` path

### 2026-03-27 22:31:00 PDT
- Reviewed the fused mixed assembly path itself to decide whether monolithic fusion can be repaired instead of disabled.
- Key code observations:
  - `installFormulation()` drives the multi-field residual path with `coupled_residual_from_jacobian_block = true`.
  - `CoupledBlockKernel` is only a colocation / shared-geometry wrapper around per-block fallback kernels; it does not itself evaluate a true coupled residual+tangent kernel.
  - The assembler fused path then loops block-by-block and scatters matrix/vector pieces, including mixed `MatrixOnly` and `Both` blocks, into one combined insertion buffer.
  - There is already an unused JIT entry point `JITCompiler::compileMonolithic(...)` for a true per-block tangent+residual coupled kernel.
- Current recommendation:
  - short-term safe repair: for `coupled_residual_from_jacobian_block`, fuse only pure `MatrixOnly` blocks and leave `Both` blocks on the standalone path
  - long-term correct repair: replace the current colocation shim for this path with a real monolithic coupled JIT kernel built through `compileMonolithic(...)`
- Rationale:
  - the monolithic-sensitive failure appears tied to the current fused path handling blocks that carry residual vectors (`NonlinearKernelOutput::Both`), not to matrix-only mixed block fusion in general
  - partitioned runs do not benefit from disabling fusion, so the bug boundary is specific to the monolithic residual-from-jacobian lowering
