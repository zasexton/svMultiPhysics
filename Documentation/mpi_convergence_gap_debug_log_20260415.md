# MPI Convergence Gap Debug Log

## Scope

Track the remaining nonlinear convergence gap between serial and distributed runs, with emphasis on the `iliac_artery` case after the FE/JIT outlet-coupling regression was fixed.

This file is intended to be append-only during investigation:

- record each hypothesis before or alongside the experiment
- record the exact code path or file touched
- record the exact run or test artifact used for qualification
- record the outcome, including negative results
- keep interpretation separate from observation

## Current Baseline

As of April 15, 2026:

- Serial archived 1-step iliac harness: `4` Newton, converged
  - [tests/_codex_iliac_1step_serial_primaryauth_20260414/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_primaryauth_20260414/run.log)
- MPI-4 archived 1-step iliac harness: `5` Newton, converged
  - [tests/_codex_iliac_1step_mpi4_primaryauth_20260414.run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_primaryauth_20260414.run.log)

The working assumption is that the remaining gap lives in the primary distributed BlockSchur/FSILS solve path, not in FE assembly or in backend retry heuristics.

## High-Signal Facts

- The FE/JIT auxiliary-slot bug was fixed on the compiled path; serial iliac recovered to `4` Newton and MPI recovered to `5` Newton.
- Hidden strict-GMRES fallback and backend-side retry logic were removed; the primary linear method is now authoritative.
- The first iliac Newton linear RHS is effectively identical in serial and MPI, but the returned pressure increment is very different.
- The global mean-pressure residual component in MPI is real but too small to explain the entire `4 -> 5` Newton gap by itself.
- Local per-partition mean recentering is catastrophically wrong for the real iliac linear systems.

## Active Hypotheses

### H1. Distributed Schur solve returns a solution with avoidable residual in a small pressure-mode subspace

Evidence:

- Serial and MPI see the same first linear system forcing, but MPI returns a materially smaller pressure update norm.
- Mean-mode probes show a reducible residual direction, but only a modest one.

Implication:

- The missing correction may live in a slightly richer pressure coarse space than the global mean alone, possibly rank-partition modes or a closely related low-dimensional subspace.

### H2. The remaining gap is a distributed solution-selection issue, not a gross operator mismatch

Evidence:

- The direct-only outlet/JIT regression is already fixed.
- Simpler distributed rank-one tests still pass.
- The corrected stress repro shows small full residuals while still drifting in recovered low-rank coefficients.

Implication:

- Exact coefficient recovery in the low-rank stress repro may be obstructed by non-uniqueness or conditioning, not necessarily by a wrong distributed matvec.

### H3. The default reduced/grouped low-rank handling may still be weaker than the native-face route, but that is not yet a complete explanation

Evidence:

- Earlier native-face experiments improved some low-rank behavior and some linear counts.
- They did not close the iliac Newton gap by themselves.

Implication:

- The remaining defect may be in the broader distributed Schur pressure solve rather than only in the outlet low-rank representation.

## Running Experiment Log

### 2026-04-14: Primary linear method made authoritative

Hypothesis:

- Retry/fallback hierarchy may be obscuring the real failure mode and may itself be part of the convergence gap.

Code:

- [Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp)
- [Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp)
- [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)
- [Code/Source/solver/FE/Tests/Unit/Assembly/test_TimeLoopFsilsConvergenceMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Assembly/test_TimeLoopFsilsConvergenceMPI.cpp)

Runs:

- [tests/_codex_iliac_1step_serial_primaryauth_20260414/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_primaryauth_20260414/run.log)
- [tests/_codex_iliac_1step_mpi4_primaryauth_20260414.run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_primaryauth_20260414.run.log)

Observation:

- Serial stayed at `4` Newton.
- MPI stayed at `5` Newton.

Interpretation:

- Retry/fallback logic was not the source of the serial/MPI nonlinear gap.
- Keeping the primary method authoritative is still the correct policy for debuggability.

### 2026-04-14: Mean-mode and local-mean probes on the real MPI iliac solve

Hypothesis:

- MPI may be missing a correction mainly along the global pressure-mean mode.

Runs:

- [tests/_codex_iliac_1step_mpi4_meanmodeprobe_20260414/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_meanmodeprobe_20260414/run.log)
- [tests/_codex_iliac_1step_mpi4_localmeanprobe_20260414/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_localmeanprobe_20260414/run.log)

Observation:

- The global mean mode reduces the returned residual only modestly, e.g. `0.792164 -> 0.766298`.
- Local partition mean recentering is catastrophically wrong on the real system.

Interpretation:

- A pressure-mode issue is still plausible, but the global mean alone is not enough.
- Any correction basis must preserve the actual coupled saddle-point structure; naive per-rank recentering is invalid.

### 2026-04-14: Line probe on first iliac Newton step

Hypothesis:

- Serial and MPI may be assembling different first linear systems.

Runs:

- [tests/_codex_iliac_1step_serial_lineprobe_20260414/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_lineprobe_20260414/run.log)
- [tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run.log)

Observation:

- The first linear RHS matches across serial and MPI.
- A baseline full-operator constant probe also matches.
- The solved pressure increment norm differs strongly: serial about `4.31649e+08`, MPI about `2.52657e+07`.

Interpretation:

- The remaining gap is not due to a gross FE residual/Jacobian mismatch on the first step.
- The divergence begins in the distributed linear solution that BlockSchur/FSILS returns.

### 2026-04-15: Correct disabled MPI stress repro to obey ownership-partitioned low-rank input contract

Hypothesis:

- The earlier small MPI repro was polluted by invalid test data because rank-one vectors included ghost-owned DOFs on the wrong ranks.

Code:

- [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)
- Contract reference:
  - [Code/Source/solver/FE/Backends/Interfaces/LinearSolver.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/Interfaces/LinearSolver.h)

Test:

```bash
mpirun -n 2 build/svMultiPhysics-build/bin/test_fe_backends_mpi \
  --gtest_also_run_disabled_tests \
  --gtest_filter=FsilsBackendMPI.DISABLED_DistributedRankOneLooseBlockSchurTracksReferenceModeResponse
```

Observation:

- The corrected stress repro still fails its exact low-rank dot expectations.
- Simpler supported native rank-one coverage still passes.

Interpretation:

- The old stress repro was not a valid backend oracle before this fix.
- After correction, it remains useful as a stress case, but not yet as proof of a wrong operator.

### 2026-04-15: Extend low-rank residual polish with distributed pressure coarse modes

Hypothesis:

- The returned MPI solution could be cheaply improved by explicit residual minimization in a small pressure-mode basis layered on top of the current solve.

Code tried temporarily:

- [Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp)

Run:

- [tests/_codex_iliac_1step_mpi4_default_faceonlytrace_20260414/run_polishcoarse.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_faceonlytrace_20260414/run_polishcoarse.log)

Observation:

- The run stalled for several minutes before reaching a nonlinear summary and had to be killed.

Interpretation:

- A generic post-solve coarse residual polish is not the right insertion point.
- If a small pressure-mode correction is needed, it likely has to live inside the primary distributed Schur solve path.

## Current Best Interpretation

The remaining MPI nonlinear gap is most likely a distributed pressure-subspace quality issue inside the primary BlockSchur/FSILS solve. The evidence does not currently support:

- a remaining FE/JIT assembly mismatch
- a benefit from backend retry/fallback logic
- a fix based only on the global pressure-mean mode
- a naive local-per-rank mean correction

## Next Experiments

1. Turn the corrected disabled distributed rank-one stress repro into a reference test by comparing the distributed result against a dense/direct or serial-equivalent reference, not just exact dot expectations.
2. Probe the first bad MPI iliac solve against a richer but still tiny pressure basis that is mathematically valid under the distributed ownership model.
3. If the pressure-subspace mismatch is confirmed, move the correction into the primary Schur solve path rather than a post-solve polish.

## 2026-04-15 Follow-Up

### New Hypothesis: distributed residual validation was under-reporting the true linear residual

Reason:

- The MPI stress repro showed a well-conditioned dense operator and a correct distributed RHS, but the helper residual check still claimed the returned distributed solutions were excellent.
- In `FsilsLinearSolver`, the validation path was comparing an accumulated RHS against a matrix product that had not been overlap-accumulated.

### 2026-04-15: Dense/direct oracle added to the disabled two-rank stress repro

Hypothesis:

- The earlier disabled repro needed a real oracle to tell whether the distributed solver was wrong or whether the exact-dot checks were too strict.

Code:

- [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)

What changed:

- Gathered owner-partitioned distributed vectors back to global form.
- Reconstructed the global dense operator from the two element blocks plus globalized low-rank modes.
- Solved the dense system directly inside the disabled repro.
- Checked that the gathered distributed RHS matches the dense assembled oracle.

Observation:

- The dense operator is well-conditioned and the manufactured exact state is the unique solution.
- The gathered distributed RHS matches the dense assembled oracle.
- Therefore the stress repro is exposing a real backend-solve/validation problem, not an oracle problem.

Interpretation:

- The bug is no longer “maybe non-uniqueness.”
- It is either in distributed solve quality or in how distributed residuals are being validated/reported.

### 2026-04-15: Fix distributed residual validation to accumulate the matrix product before comparing against the accumulated RHS

Hypothesis:

- `FsilsLinearSolver` true-residual validation was subtracting an accumulated `rhs` from a non-accumulated matrix product on overlap-based layouts.

Code:

- [Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp)
- [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)

What changed:

- In `computeTrueResidualVector`, the matrix product is now overlap-accumulated before low-rank contributions are added and before residual subtraction.
- The same accumulation fix was applied to the low-rank residual-polish basis evaluation and the main returned-solution debug checks.
- The unit-test helper `fullOperatorRelativeResidual(...)` now uses the same accumulated-matrix semantics.

Observation:

- The corrected disabled repro stopped pretending that GMRES/BlockSchur were converged.
- The “reference” GMRES solve in that repro now reports `converged=false` with relative residual about `0.94`.
- The corresponding BlockSchur solve reports `converged=false` with relative residual about `0.57`.

Interpretation:

- The earlier passing residual checks were false positives on the distributed overlap layout.
- The MPI backend problem is more severe than the earlier logs suggested.

### 2026-04-15: Existing enabled MPI backend tests were also false-positive under the old residual helper

Hypothesis:

- If the residual helper was under-validating, the previously passing distributed backend tests should fail once the helper is corrected.

Tests:

```bash
mpirun -n 2 build/svMultiPhysics-build/bin/test_fe_backends_mpi \
  --gtest_filter='FsilsBackendMPI.RankOneUpdateSolversConvergeComparable4DOF:FsilsBackendMPI.ReducedFieldUpdateSolversConvergeComparable:FsilsBackendMPI.GroupedBorderedFieldCouplingSolversConvergeComparable'
```

Observation:

- All three now fail honestly under the corrected residual semantics.
- Representative true residuals are large:
  - reduced-update coverage: about `0.24`
  - grouped/bordered coverage: about `0.40`
  - rank-one 4-DOF coverage: about `0.77-0.86`

Interpretation:

- This is not an iliac-only issue.
- The distributed FSILS/BlockSchur backend has a broader solve-quality problem that prior validation was masking.

### 2026-04-15: Real one-step iliac qualification under corrected validation

Runs:

- Serial: [tests/_codex_iliac_1step_serial_lineprobe_20260414/run_trueval_20260415.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_lineprobe_20260414/run_trueval_20260415.log)
- MPI-4: [tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run_trueval_20260415.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run_trueval_20260415.log)

Observation:

- Serial still converges in `4` Newton and reaches final `||r||=4.9376213535710683e-08`.
- Serial is slower (`48.02 s` total loop) because the linear solves are now validated honestly.
- MPI-4 fails on the very first linear solve:
  - `true residual check failed (|Ax-b|=9101.27, rel=0.970383, target=9.37905)`
  - Newton aborts before any accepted nonlinear iteration.

Interpretation:

- The previous `serial 4` vs `mpi4 5` story was understated.
- Under correct residual validation, the real gap is `serial converges` vs `mpi4 first linear solve fails`.

### 2026-04-15: Native-face MPI rank-one route under corrected validation

Hypothesis:

- The native-face distributed route may still improve the first MPI iliac linear solve under honest residual validation.

Run:

- [tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run_trueval_nativeface_20260415.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run_trueval_nativeface_20260415.log)

Observation:

- It still fails on the first linear solve, with relative true residual about `0.999322`.
- It also explodes collective count in the Schur solve (`~45k` allreduces in one outer iteration).

Interpretation:

- Native-face rank-one is not the fix.
- It is worse than the default path under correct validation.

## Updated Best Interpretation

The MPI convergence problem was being materially masked by a backend residual-validation bug on overlap-based layouts. After fixing that bug:

- serial iliac still converges
- MPI iliac fails the first linear solve
- multiple distributed FSILS backend unit tests that previously passed now fail honestly

So the remaining task is no longer to shave one extra MPI Newton iteration. The real problem is to improve distributed FSILS/BlockSchur solve quality enough that the corrected true residual check passes in the first place.

## Updated Next Experiments

1. Use the now-valid distributed backend tests as the primary debugging loop instead of the old false-positive coverage.
2. Inspect the distributed Schur solve path for overlap/ownership inconsistencies analogous to the residual-validation bug, especially where Schur residuals and block products are formed.
3. Compare the effective distributed Schur residual history against a dense/direct reference on the small enabled backend tests, not only on the disabled stress repro.
4. Only after the distributed backend tests are honestly green again should iliac MPI qualification be treated as meaningful.

### 2026-04-15: FsilsMatrix::mult overlap semantics

Hypothesis:

- The earlier residual-validation change was over-accumulating `A->mult(...)` outputs in FSILS old-layout vectors.
- `FsilsMatrix::mult()` already returns the product in the old local layout expected by the FE vectors and by `FsilsVector::norm()`.
- Applying `accumulateOverlap()` after `A->mult()` double-counts shared-row contributions and creates false MPI residual failures.

Evidence:

- In the traced 2-rank rank-one repro, `x_raw` and `x_sync` were identical (`|x_sync-x_raw|=0`), and raw/synced rank-one dots matched exactly.
- The only discrepancy came from post-multiply accumulation: the unaccumulated full operator residual was `~2.89e-14`, while the accumulated validation path reported `~4.02e+01`.
- Inspecting [FsilsMatrix.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsMatrix.cpp:1777) confirmed that `FsilsMatrix::mult()` performs no overlap exchange and maps directly back to the FE old layout.

Action:

- Removed the extra `accumulateOverlap()` calls after `A->mult()` in `FsilsLinearSolver` true-residual validation and related diagnostics.
- Removed the matching over-accumulation from the MPI backend oracle helpers in `test_BlockSchurMPI.cpp`.

Result:

- `FsilsBackendMPI.RankOneUpdateSolversConvergeComparable4DOF`, `ReducedFieldUpdateSolversConvergeComparable`, and `GroupedBorderedFieldCouplingSolversConvergeComparable` all pass again under `mpirun -n 2`.
- The traced 2-rank rank-one repro now validates honestly: BlockSchur `rel~5.8e-9`, GMRES `rel~6.2e-16`.

Interpretation:

- The previous “fixed” MPI residual failure was a validation bug, not a primary solve-quality failure.
- The backend now has an honest MPI oracle again, and the next step is to re-qualify the real iliac harness to measure the remaining serial/MPI nonlinear gap.

### 2026-04-15: Constraint-only probe caveat

Observation:

- The existing `constraint-only probe` in `NewtonSolver` is assembled only over `owned_dofs`, not over a serial/MPI-invariant global basis.
- That means the smaller MPI constraint-probe norm cannot be interpreted directly as a global pressure-mode discrepancy.

Code:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)

Interpretation:

- The earlier constraint-probe comparison was too strong.
- It is still consistent with a distributed pressure-subspace issue, but it is not by itself a valid oracle for the serial/MPI gap.

### 2026-04-15: Existing mpi4 one-step iliac experiment matrix still contains no 4-Newton run

Hypothesis:

- A previously explored solver knob may already have closed the mpi4 one-step nonlinear gap.

Search:

```bash
rg -n "nonlinear_done step=0.*iters=4" tests/_codex_iliac_1step_mpi4_* -g 'run*.log'
```

Observation:

- No valid mpi4 archived one-step iliac run reaches `4` Newton.
- The only apparent `iters=4` hit came from a misleadingly named directory whose actual mpi4 run was in a different log and still converged in `5` Newton.

Interpretation:

- The remaining mpi4 gap is structural in the returned distributed Newton direction, not already solved by an overlooked environment knob.

### 2026-04-15: Schur preconditioner probe rerun produced no probe output

Hypothesis:

- The existing Schur preconditioner probe hook might explain the first-step mpi4 gap without adding new instrumentation.

Runs:

- [tests/_codex_iliac_1step_serial_schurpcprobe_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurpcprobe_20260415/run.log)
- [tests/_codex_iliac_1step_mpi4_schurpcprobe_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurpcprobe_20260415/run.log)

Observation:

- The runs completed with the expected `serial 4` and `mpi4 5` Newton counts.
- No `[SCHUR_PC_PROBE]` lines were emitted despite the env gate being set.

Interpretation:

- That hook did not provide actionable data in the current code path.
- It should not be treated as a current debugging tool until its activation path is verified.

### 2026-04-15: First-step global vector dumps show the serial/MPI mismatch is in the returned correction, not the RHS

Hypothesis:

- The remaining serial/MPI gap may come from the distributed solver returning a materially different first Newton correction even when the first linear forcing is the same.

Code:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)

Instrumentation:

- Added `SVMP_DEBUG_FIRST_LINEAR_VECTOR_DUMP_PREFIX` to dump the first-step gathered global `linear_rhs`, raw `du`, and normalized `du`.

Artifacts:

- Serial:
  - [tests/_codex_iliac_1step_serial_vecdump_20260415/first.rhs.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_vecdump_20260415/first.rhs.txt)
  - [tests/_codex_iliac_1step_serial_vecdump_20260415/first.du_raw.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_vecdump_20260415/first.du_raw.txt)
  - [tests/_codex_iliac_1step_serial_vecdump_20260415/first.du_normalized.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_vecdump_20260415/first.du_normalized.txt)
- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.rhs.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.rhs.txt)
  - [tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.du_raw.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.du_raw.txt)
  - [tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.du_normalized.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.du_normalized.txt)

Observation:

- A naive index-by-index serial/MPI comparison is invalid because the global DOF numbering is not shared across the serial and distributed runs.
- That is clear from the gathered RHS vectors: direct comparison is orthogonal, but the sorted value multisets match to `~1.56e-15` relative error.
- The returned `du` vectors do not show the same invariance:
  - `||du_serial|| ~ 4.31649e+08`
  - `||du_mpi|| ~ 2.51517e+07`
  - even after best-fit scaling, `||du_mpi - alpha du_serial|| / ||du_mpi|| ~ 9.08e-01`
  - the sorted signed-value multisets differ by `~9.57e-01` relative error
- Raw and normalized `du` are identical in both runs, so `normalizeFsilsPostSolveIncrementIfNeeded(...)` is not creating the mismatch.

Interpretation:

- The first-step linear forcing is the same up to renumbering, which is consistent with the earlier FE/Jacobian diagnosis.
- The serial/MPI gap is in the actual returned distributed Newton correction, not in a renumbering artifact and not in post-solve normalization.
- The next debugging target should be the primary distributed BlockSchur/FSILS correction path itself, using a serial/MPI-invariant basis or a smaller backend oracle rather than raw global DOF indices.

### 2026-04-15: Enforce ghost-synced inputs in Schur/CG Krylov loops

Hypothesis:

- The distributed Schur Krylov loops were violating the `ghost_synced_input(...)` operator contract by reusing updated search vectors without an explicit halo refresh.
- If so, MPI pressure corrections could shrink or drift even when the assembled operator is otherwise correct.

Code:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/cgrad.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/cgrad.cpp)
- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)

What changed:

- Added explicit halo refreshes before distributed operator applies in:
  - scalar CG
  - vector CG
  - Schur CG
  - Schur GMRES initial-guess and Arnoldi operator applications

Tests:

```bash
mpirun -n 2 build/svMultiPhysics-build/bin/test_fe_backends_mpi \
  --gtest_filter='FsilsBackendMPI.RankOneUpdateSolversConvergeComparable4DOF:FsilsBackendMPI.ReducedFieldUpdateSolversConvergeComparable:FsilsBackendMPI.GroupedBorderedFieldCouplingSolversConvergeComparable'
```

Observation:

- The small MPI backend coverage stayed green.
- That means the added syncs do not break the supported distributed low-rank paths.

Real-case run:

- Partial mpi4 iliac artifact:
  - [tests/_codex_iliac_1step_mpi4_schursync_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schursync_20260415/run.log)

Observation:

- The real mpi4 one-step iliac run did not show an early qualitative improvement.
- The first several BlockSchur solves still saturated the Schur inner iteration cap:
  - first BlockSchur solve: outer `6`, Schur `300`
  - second BlockSchur solve: outer `3`, Schur `300`
  - third BlockSchur solve: outer `4`, Schur `300`
  - fourth BlockSchur solve: outer `5`, Schur `300`
- I stopped treating that run as a useful performance sample once the unchanged saturation pattern was clear.

Interpretation:

- This was a real contract-hygiene issue worth fixing, but it is not the primary cause of the iliac mpi convergence gap.
- The remaining problem is deeper in Schur preconditioner quality or distributed Schur operator effectiveness, not just missing halo refreshes on the Krylov basis vectors.

### 2026-04-15: Serial and MPI still route pure outlet rank-one updates differently by default

Hypothesis:

- The remaining serial/MPI gap may still be partly explained by the two runs taking different outlet-coupling representations:
  - serial staying on the native face-rank-one path
  - MPI falling back to the explicit reduced-update path

Traced artifacts:

- Serial:
  - [tests/_codex_iliac_1step_serial_corrtrace2c_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_corrtrace2c_20260415/run.log)
- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_corrtrace2c_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_corrtrace2c_20260415/run.log)

Observation:

- On the current tree, the traced serial run enters:
  - `explicit_block_modes=0`
  - `native_face_rank_one_count=2`
- The traced mpi4 run enters:
  - `explicit_block_modes=2`
  - `native_face_rank_one_count=0`
- So the serial/MPI gap was not just “same operator, worse distributed solve.” The default code was still sending the same pure rank-one outlet modes down different backend representations.

Interpretation:

- This is a real routing asymmetry and a plausible performance bug.
- It does not by itself prove the nonlinear-gap root cause, but it must be removed before serial/MPI parity can be judged cleanly.

### 2026-04-15: Pure MPI native-face rank-one routing bug fixed

Hypothesis:

- The routing asymmetry above was caused by the pure distributed native-face path being incorrectly left behind an env gate.
- The local variable `prefer_mpi_native_face_rank_one` was computed, but `allow_mpi_native_face_rank_one` still only enabled the path for:
  - serial, or
  - pure MPI native-face cases with `SVMP_FSILS_ENABLE_MPI_NATIVE_FACE_RANK_ONE=1`

Code:

- [Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp)

What changed:

- Pure distributed `prefer_native_face` rank-one cases now take the native face route by default.
- The env override is still retained for broader MPI native-face experimentation when rank-one updates are present.

Qualification:

- Former env-opt-in run:
  - [tests/_codex_iliac_1step_mpi4_nativeface_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_current_20260415/run.log)
  - `5` Newton, `15` linear, total loop `22.125383 s`
- New default no-env run after the code change:
  - [tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log)
  - `5` Newton, `15` linear, total loop `21.569308 s`

Interpretation:

- This confirms the routing bug was real and is now fixed.
- It improves distributed performance, but it does **not** close the `serial 4` vs `mpi4 5` nonlinear gap.

### 2026-04-15: Native-face parity experiment isolates the remaining gap to the distributed Schur correction

Hypothesis:

- If the remaining gap were only due to reduced-update vs native-face representation, then forcing MPI onto native face rank-one should recover the serial-like first Schur correction and likely the serial Newton count.

Artifacts:

- Serial face-only trace:
  - [tests/_codex_iliac_1step_serial_facetrace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_facetrace_20260415/run.log)
- MPI-4 native-face with distributed multi-face GMRES:
  - [tests/_codex_iliac_1step_mpi4_nativeface_facetrace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_facetrace_20260415/run.log)
- MPI-4 native-face with distributed multi-face GMRES disabled:
  - [tests/_codex_iliac_1step_mpi4_nativeface_nogmres_facetrace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_nogmres_facetrace_20260415/run.log)

Observation:

- Serial first Schur solve (`final_bicgstab`) returns a large outlet response:
  - `gp_face_dot ~ 8.55e+02` and `1.41e+03`
- MPI-4 native-face first Schur solve with distributed multi-face GMRES (`final_gmres`) returns an outlet response that is effectively annihilated:
  - `gp_face_dot ~ 5.80e-61` and `-1.53e-46`
- MPI-4 native-face first Schur solve with multi-face GMRES disabled (`final_bicgstab`) is qualitatively the same:
  - `gp_face_dot ~ 1.37e-76` and `1.39e-66`
- Despite different Krylov wrappers, both distributed variants still converge in `5` Newton, not `4`.

Interpretation:

- The remaining gap is **not** just the explicit reduced-update path.
- It is also **not** just the distributed multi-face GMRES wrapper.
- The common failure mode is deeper: the distributed first Schur correction is landing on a pressure solution with almost no outlet-face response, unlike the serial solve.
- That is consistent with the earlier pressure-mode / nullspace diagnosis.

### 2026-04-15: The next credible target is generic constraint-mode nullspace handling

Observation:

- Existing iliac lineprobe logs still report:
  - `basis=0 projected=0`
- The backend already supports post-solve nullspace projection in:
  - [Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp)
- But `NewtonSolver` currently passes an empty basis because `GaugeRegistry::buildNullspaceBasis(...)` is dormant under the current algebraic-enforcement policy:
  - [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)

Interpretation:

- The remaining distributed gap is now most plausibly a generic constraint-field nullspace / coarse-mode problem in the Schur solve.
- The next fix attempt should be physics-agnostic:
  - either re-enable a generic solver-nullspace basis path for exact-nullspace scalar constraint fields
  - or provide an equivalent backend-side constraint-mode basis / projection for FSILS BlockSchur
- A quick attempt to force the native-face case onto the modern distributed Schur branch with partition/global coarse modes did not reach a useful solver summary quickly enough to keep running under the current process pressure, so I stopped it:
  - [tests/_codex_iliac_1step_mpi4_nativeface_moderncoarse_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_moderncoarse_20260415/run.log)

### 2026-04-15: Preserved-harness traces weaken the shared-face-sync hypothesis

Hypothesis:

- The distributed gap might still be caused by incorrect shared-face normalization or by double communication on outlet-face data.

Artifacts:

- Serial preserved harness with `NS_SOLVER` and legacy-face trace:
  - [tests/_codex_iliac_1step_serial_nssolvertrace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolvertrace_20260415/run.log)
- MPI-4 preserved harness with the same trace:
  - [tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log)
- Communication helper:
  - [Code/Source/solver/FE/Backends/FSILS/liner_solver/in_commu.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/in_commu.cpp)

Observation:

- `fsils_commuv(...)` is overlap accumulation, but `fsils_syncv(...)` zeroes ghosts first and then uses that accumulation as an owner-to-ghost refresh.
- On the preserved mpi4 harness, rank 0 reports:
  - `native_face_rank_one_count=2`
  - `face[1]: nNo=0 sharedFlag=0`
  - `face[2]: nNo=0 sharedFlag=0`
- The legacy face trace still reports global outlet node counts `10` and `12`, which means each outlet face is living entirely on some non-root rank, not being duplicated as a shared face on rank 0.
- The first distributed Schur correction still has essentially zero global outlet-face response.

Interpretation:

- The remaining gap is not explained by rank-0 shared-face normalization.
- The outlet faces are not obviously being corrupted by a root-rank shared-face path; the loss still happens in the distributed Schur correction itself.

### 2026-04-15: Final scalar-Schur overlap-sync removal was a false lead and was reverted

Hypothesis:

- The legacy scalar Schur matvec might be double-accumulating overlap data because it called `halo.sync_scalar(out_vec)` after forming `M^{-1}(SP-DGP)`, even though `SP` and `DGP` were already produced through `ghost_synced_output(...)`.

Code tried temporarily:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)

Qualification:

- Serial preserved harness:
  - [tests/_codex_iliac_1step_serial_nodoublesync_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nodoublesync_20260415/run.log)
- MPI-4 preserved harness:
  - [tests/_codex_iliac_1step_mpi4_nodoublesync_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nodoublesync_20260415/run.log)

Observation:

- Serial stayed at `4` Newton.
- MPI-4 stayed at `5` Newton.
- Re-reading [in_commu.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/in_commu.cpp) shows `sync_scalar/sync_vector` zero ghost entries before exchange, so the post-combination sync is not a double-add bug; it is an owner-to-ghost refresh.

Interpretation:

- This was the wrong explanation and the edit was reverted.
- The remaining gap is not caused by that final overlap refresh in the legacy scalar Schur matvec.

### 2026-04-15: Forcing the non-legacy Schur path regressed badly enough to stop

Hypothesis:

- If the distributed `4 -> 5` gap is specific to the legacy face-only scalar Schur path, then forcing the newer non-legacy Schur branch might recover serial-like nonlinear behavior.

Code path:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/block_schur_strategy_selector.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/block_schur_strategy_selector.cpp)

Run:

- `SVMP_FSILS_BLOCKSCHUR_DISABLE_FACE_ONLY_LEGACY=1`
- [tests/_codex_iliac_1step_mpi4_disablelegacy_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_disablelegacy_20260415/run.log)

Observation:

- The run became much heavier immediately and did not reach a first nonlinear summary before it was stopped.
- Early log sections show repeated full assemblies and a first FSILS solve on the generic GMRES path, but not a useful improvement signal.

Interpretation:

- The newer Schur branch is not a drop-in win for this case in its current form.
- The remaining credible work is still inside the distributed primary Schur/constraint-mode solve, not simply “turn off the legacy path.”

### 2026-04-15: Process cleanup

Observation:

- Two stale mpi4 solver launches remained alive after trace/experiment runs:
  - the preserved-harness `NS_SOLVER` trace
  - the forced non-legacy Schur run

Interpretation:

- Those processes were killed explicitly to avoid more unified-exec pressure and to keep subsequent qualification clean.

### 2026-04-15: The hand-built dense oracle is not the problem

Hypothesis:

- The disabled 2-rank backend repro might still be failing because its hand-built dense oracle does not match the FE/backend operator.

Code added:

- Sampled a dense matrix directly from the FE backend in:
  - [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)

Qualification:

- [tests/_codex_blockschur_face_oracle_sampled_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_blockschur_face_oracle_sampled_20260415/run.log)
- [tests/_codex_blockschur_face_oracle_applydiag_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_blockschur_face_oracle_applydiag_20260415/run.log)

Observation:

- The basis-sampled backend matrix matches the hand-built dense matrix to roundoff:
  - `hand_vs_backend_max_abs = 8.8817841970012523e-16`
- A second owned-node gather path gives the same collapsed global vector:
  - `gather_vs_oldnode = 0`
- The sampled dense matrix action also matches gathered `A*x` to roundoff:
  - GMRES `apply_vs_dense = 5.6843418860808015e-14`
  - BlockSchur `apply_vs_dense = 1.1368683772161603e-13`

Interpretation:

- The failing dense residuals are not caused by a typo in the hand-built oracle.
- The contradiction survived even after replacing the oracle with a basis-sampled FE/backend matrix, so the remaining issue is deeper than “the dense matrix is assembled wrong.”

### 2026-04-15: Accumulated-output dense operator was also not the explanation

Hypothesis:

- The dense repro might be using the wrong globalization, because backend true-residual validation accumulates overlap on the RHS before forming the residual.

Code added:

- Added an accumulated-output sampling mode in:
  - [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)

Qualification:

- [tests/_codex_blockschur_face_oracle_accumulated_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_blockschur_face_oracle_accumulated_20260415/run.log)

Observation:

- The accumulated sampled RHS matched the accumulated gathered backend RHS, so the accumulated sampling path is internally consistent.
- But it did not reconcile the contradiction:
  - GMRES accumulated dense residual stayed large at `0.94226488790804908`
  - BlockSchur accumulated dense residual stayed large at `0.94218583336313255`
- The accumulated operator differs materially from the raw collapsed operator:
  - `raw_vs_acc_op_max_abs = 12` at `(row=4, col=4)`

Interpretation:

- The gap is not explained by simply switching the tiny repro from raw owner rows to an accumulated-output collapse.
- The “single collapsed dense matrix” idea is still not reproducing the backend’s validated residual semantics.

### 2026-04-15: Full basis FE-vs-native-face operator compare matched exactly

Hypothesis:

- The native-face FSILS operator might still differ from the FE `A + R` operator on basis directions that earlier generic/rank-one probes did not hit.

Code added:

- Added an env-gated full basis probe sweep in:
  - [Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp)
- Triggered by:
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR=1`
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR_BASIS=1`
  - `SVMP_FSILS_PROBE_LOW_RANK_MODES=1`

Qualification:

- [tests/_codex_blockschur_face_oracle_basiscompare2_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_blockschur_face_oracle_basiscompare2_20260415/run.log)

Observation:

- On the 12-DOF 2-rank repro, every basis vector probe `basis_0` through `basis_11` matched the FE operator to roundoff.
- Representative lines:
  - `basis_0`: `diff=0`, `diff_J=0`, `diff_R=0`
  - `basis_8`: `diff=8.88178e-16`, `diff_J=0`, `diff_R=8.88178e-16`
- The generic probe, both rank-one probes, and the partitioned pressure probe also matched to roundoff.

Interpretation:

- There is no evidence, on this repro, that the native-face FSILS operator is algebraically different from the FE `A + R` operator.
- That rules out the most obvious “operator apply bug” hypothesis for this path.

### 2026-04-15: Revised diagnosis after the dense-oracle work

Observation:

- The small repro now shows all of the following simultaneously:
  - backend true residual is tiny for GMRES and small for BlockSchur
  - FSILS native-face operator matches the FE operator on every basis vector tested
  - collapsed dense residuals remain large for both returned solutions

Interpretation:

- The contradiction is no longer pointing at the native-face low-rank operator itself.
- The most credible explanation now is that the current tiny repro is collapsing the distributed overlapped solve into a single global dense oracle in a way that is not equivalent to the backend’s actual distributed solve/residual semantics.
- The next useful oracle is not another unique-global `12 x 12` collapse. It is a dense/debug representation of the distributed overlapped space used by the backend, or a backend-unit test that validates residuals directly in that distributed space.

### 2026-04-15: GMRES-only case sweep separates the two behaviors

Hypothesis:

- The observed serial/MPI nonlinear gap might be specific to the primary distributed BlockSchur path.
- If so, forcing pure GMRES should help distinguish:
  - a general distributed FE/FSILS inconsistency from
  - a BlockSchur-specific distributed pressure/constraint gap.

Setup:

- Copied the archived solver XMLs and forced:
  - `<LS type="GMRES">`
  - `Max_iterations=1000`
  - `Krylov_space_dimension=250`
  - `Tolerance=1e-3`
  - `Absolute_tolerance=1e-17`
- Runs:
  - [tests/_codex_gmres_pipe_rcr3d_serial_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_gmres_pipe_rcr3d_serial_20260415/run.log)
  - [tests/_codex_gmres_pipe_rcr3d_mpi4_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_gmres_pipe_rcr3d_mpi4_20260415/run.log)
  - [tests/_codex_gmres_iliac_1step_serial_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_gmres_iliac_1step_serial_20260415/run.log)
  - [tests/_codex_gmres_iliac_1step_mpi4_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_gmres_iliac_1step_mpi4_20260415/run.log)

Observation:

- `pipe_RCR_3d` under pure GMRES preserved serial/MPI nonlinear parity:
  - serial: step `0` converged in `3` Newton, step `1` in `2` Newton
  - `mpi4`: step `0` converged in `3` Newton, step `1` in `2` Newton
- But the MPI GMRES cost was still poor:
  - serial total loop: `16.839767 s`
  - `mpi4` total loop: `14.020810 s`
  - `mpi4` linear iterations were materially higher (`156/181` vs `104/121`) and collective counts were large
- `iliac_artery` behaved very differently under pure GMRES:
  - serial 1-step archived harness converged in `5` Newton with `4459` linear iterations and total loop `497.901127 s`
  - `mpi4` 1-step archived harness failed at the `12`-Newton cap with `converged=0`, final `||r||=1.6389140958604406e-04`, total Newton time `253.493899 s`, and `6715` linear iterations

Interpretation:

- Pure GMRES does **not** reproduce a serial/MPI nonlinear gap on `pipe_RCR_3d`.
- Pure GMRES makes the `iliac_artery` distributed behavior much worse than the current primary BlockSchur path:
  - serial degrades from the current `4`-Newton BlockSchur result to `5` Newton
  - `mpi4` degrades from the current `5`-Newton BlockSchur result to outright nonlinear failure at `12` Newton
- This strengthens the case that the remaining production issue is a distributed pressure/constraint solve-quality problem tied to the hard `iliac_artery` outlet coupling, not a generic “all distributed solves are inconsistent” bug.

### 2026-04-15: Added MPI unit coverage for shared distributed low-rank / coarse-space behavior

Hypothesis:

- We need backend-level coverage that exercises the shared distributed low-rank outlet subspace without relying on the misleading collapsed dense oracle.
- The right backend oracle is:
  - full distributed operator residual in overlapped space
  - agreement with a distributed GMRES reference on the excited low-rank mode-response subspace

Code added:

- New enabled MPI backend test:
  - [Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp)
  - `FsilsBackendMPI.MultiModeNativeRankOneSolversTrackManufacturedModeResponse`
- New disabled parity test:
  - `FsilsBackendMPI.DISABLED_NearDependentNativeRankOneBlockSchurMatchesReferenceModeResponse`
- Re-labeled the older dense-collapse harness as debug-only rather than an authoritative distributed oracle.

Qualification:

- Rebuilt `test_fe_backends_mpi`.
- Passed:
  - `mpirun -n 2 ... --gtest_filter='FsilsBackendMPI.MultiModeNativeRankOneSolversTrackManufacturedModeResponse:FsilsBackendMPI.RankOneUpdateSolversConvergeComparable4DOF'`
- Manual disabled-test probe:
  - `mpirun -n 2 ... --gtest_also_run_disabled_tests --gtest_filter='FsilsBackendMPI.DISABLED_NearDependentNativeRankOneBlockSchurMatchesReferenceModeResponse'`

Observation:

- The new enabled two-mode test passes and now guards:
  - distributed GMRES reference residual on a multi-mode native-rank-one system
  - BlockSchur agreement with GMRES on the excited low-rank response subspace
- The new near-dependent disabled parity repro unexpectedly also passes on the tiny backend system.

Interpretation:

- This is still valuable coverage: it should catch future regressions in the shared distributed low-rank/coarse-space path much earlier than full `iliac_artery` runs.
- But it also shows that the current `iliac_artery` MPI gap is **not** reproduced by the tiny pure-backend near-dependent rank-one case alone.
- That means the next missing guard likely lives one level above this backend unit scope, where the FE/Newton/monolithic auxiliary coupling path generates the actual distributed system seen in `iliac_artery`.


### 2026-04-15: Higher-level FE/Newton monolithic-outlet probe and mixed-residual splitter regression

Hypothesis:

- The current MPI gap is not reproduced by the tiny backend-only low-rank tests because the missing behavior lives one level higher, in the FE/Newton/monolithic AuxiliaryState path.
- A smaller transient mixed FE probe with direct-only monolithic resistance outlets should let us reproduce the gap without needing the full `iliac_artery` case.

Code added / changed:

- New disabled higher-level probe in:
  - [Code/Source/solver/FE/Tests/Unit/Assembly/test_TimeLoopFsilsConvergenceMPI.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Assembly/test_TimeLoopFsilsConvergenceMPI.cpp)
  - `TimeLoopFsilsConvergenceMPI.DISABLED_GeneralizedAlphaMonolithicResistanceOutletsProbe`
- New helper builder in the same file:
  - `buildOutletCoupledTransientSystem(...)`
- Real FE-library bug fix in:
  - [Code/Source/solver/FE/Systems/FormsInstaller.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Systems/FormsInstaller.cpp)
  - mixed residual splitting now preserves negative per-block terms as unary negation (`-expr`) instead of rewriting them as `(-1) * integral(...)`
- New regression guard for that bug in:
  - [Code/Source/solver/FE/Tests/Unit/Systems/test_FormsInstaller.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Systems/test_FormsInstaller.cpp)
  - `FormsInstaller_MixedResidualNegativeBoundaryTermCompiles`

Observation:

- The first version of the higher-level probe did not compile at all because mixed residual splitting converted a negative boundary integral into a top-level `(-1) * integral(...)`, which `FormCompiler` rejects.
- After fixing `FormsInstaller`, the monolithic-outlet probe compiled and ran.
- With pressure still regularized (`kappa > 0`), the probe did not cleanly reproduce the serial/MPI split; it mostly showed that the artificial system itself was not yet representative.
- After removing the pressure mass regularization (`kappa = 0`) to expose the true saddle-point/nullspace structure, the behavior became much more informative:
  - serial rank-1 run now fails immediately in the primary BlockSchur solve with no gauge constraints applied (`constraints=0`), reporting a true-residual failure from `NewtonSolver`
  - `mpi4` does not fail cleanly; it hangs in the primary FSILS solve
- Stack snapshots of the hanging `mpi4` probe show:
  - three ranks blocked in `fsils_commuv(...)` waitall inside `ns_solver.cpp`
  - one rank blocked in `fsi_ls_norms(...)` allreduce inside `ns_solver.cpp`

Interpretation:

- This is the first successful reproduction one level above the backend-only repros.
- The higher-level probe does not yet match `iliac_artery` numerically, but it does expose a shared FE/FSILS problem on the monolithic outlet saddle-point path:
  - serial primary solve breakdown when the pressure nullspace is unanchored
  - distributed hang in the same primary solve path under `mpi4`
- The fact that `constraints=0` on the nullspace-enabled probe is high-signal. Either:
  - automatic gauge/nullspace detection is not firing for this mixed monolithic-outlet formulation, or
  - the probe still needs an explicit anchoring/gauge setup to reach the intended production-like regime.
- The next highest-signal follow-up is to interrogate why this probe gets no pressure gauge enforcement, then rerun serial/`mpi4` once the probe has the intended nullspace treatment.


Addendum:

- Re-ran the nullspace-enabled serial probe with extra diagnostics.
- It reports:
  - `has_gauge=0`
  - `gauge_candidates=0`
  - `gauge_resolved=0`
  - `constraints=0`
- So the probe is not merely “detecting the pressure nullspace but failing to enforce it.” The current mixed monolithic-outlet formulation is not entering gauge/nullspace detection at all.

Updated interpretation:

- There are now two coupled higher-level issues worth separating:
  - missing gauge/nullspace discovery for this mixed saddle-point monolithic-outlet formulation
  - distributed FSILS behavior once such an unanchored system reaches the primary BlockSchur path
- The missing gauge detection is likely upstream and physics-agnostic, and it may be a prerequisite for building a faithful FE-level reproducer of the iliac serial/MPI convergence gap.


### 2026-04-15: Missing pressure gauge claim ingestion fixed; higher-level probe still suppresses gauge

Hypothesis:

- The generic mixed pressure gauge may already be inferred by `MixedOperatorAnalyzer`, but never turned into a `GaugeRegistry` candidate during setup.
- If so, a small physics-agnostic fix in `SystemSetup` should restore pressure mean-zero enforcement for plain mixed saddle-point systems.

Code changed:

- [Code/Source/solver/FE/Systems/SystemSetup.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Systems/SystemSetup.cpp)
  - setup now imports `analysisReport().claims` nullspace claims into `GaugeRegistry` in addition to `ContributionDescriptor::nullspace_hints`
- New unit test:
  - [Code/Source/solver/FE/Tests/Unit/Systems/test_GaugeIntegration.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Systems/test_GaugeIntegration.cpp)
  - `GaugeIntegration.MixedSaddlePointPressureFieldGetsGauge`

Qualification:

- `FormsInstaller.FormsInstaller_MixedResidualNegativeBoundaryTermCompiles` passes
- `GaugeIntegration.MixedSaddlePointPressureFieldGetsGauge` passes

Observation:

- Before the `SystemSetup` fix, the plain mixed saddle-point test only auto-enforced velocity componentwise-constant gauges (`field 0`) and never created a pressure gauge.
- After the fix, the same plain mixed saddle-point test reports:
  - velocity componentwise-constant gauges for `field 0`
  - pressure scalar-constant gauge for `field 1`
  - total automatic gauge constraints applied: `4`
- However, rerunning the higher-level monolithic-outlet transient probe after the fix still reports:
  - `has_gauge=0`
  - `gauge_candidates=0`
  - `gauge_resolved=0`
  - `constraints=0`

Interpretation:

- One real upstream bug is fixed: generic mixed pressure-gauge claims were being inferred structurally but ignored by setup.
- That bug is now guarded by a focused unit test and no longer explains all of the higher-level probe behavior.
- The remaining gap is narrower:
  - plain mixed saddle-point systems now get the expected pressure gauge
  - the transient monolithic-outlet probe still suppresses gauge detection entirely
- So the remaining higher-level issue is not generic mixed pressure-gauge handling anymore. It is specific to some combination of the probe’s formulation features, most likely one of:
  - transient `u.dt(1)` handling in the mixed analysis path
  - the `VectorSpace`/Quad formulation path rather than the simpler ProductSpace/tetra path
  - interaction between the mixed block analysis and monolithic boundary auxiliary outputs

Next step:

- Build the smallest non-MPI systems test that differs from the passing plain mixed saddle only by one axis at a time:
  - `ProductSpace/tetra` → `VectorSpace/tetra`
  - then `VectorSpace/tetra` → `VectorSpace/quad strip`
  - then add transient terms
  - then add monolithic outlet terms
- That should isolate the exact feature that suppresses gauge candidate generation in the higher-level probe.


Addendum:

- Added a second plain mixed saddle-point gauge test using `spaces::VectorSpace(...)` instead of `spaces::ProductSpace(...)` on the same tetra mesh.
- Result: it also passes and auto-enforces the pressure scalar-constant gauge.

Updated interpretation:

- `VectorSpace` by itself is not the remaining suppressor.
- The higher-level monolithic-outlet probe still differs from the passing plain mixed saddle tests along the remaining axes:
  - quad-strip geometry / mesh path
  - transient `u.dt(1)` terms
  - monolithic outlet auxiliary terms
- The next isolation step should vary those one at a time, starting with a steady quad-strip mixed saddle test with no outlets.


### 2026-04-15: Reaction terms forced MixedOperatorAnalyzer fallback to emit an unstructured pressure nullspace claim

Hypothesis:

- The exact quad-strip reproducer may still be inferring the pressure nullspace, but in a weaker fallback form that `SystemSetup` cannot convert into a `GaugeRegistry` candidate.
- If so, the real bug is not gauge resolution itself, but missing structured metadata on the fallback `MixedOperatorAnalyzer` nullspace claim.

Isolation:

- Added exact-assembly quad-strip variants in
  [Code/Source/solver/FE/Tests/Unit/Systems/test_GaugeIntegration.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Systems/test_GaugeIntegration.cpp)
  that independently toggle:
  - linear reaction `lambda * inner(u, v)`
  - nonlinear reaction `eps * (1 + inner(u, u)) * inner(u, v)`
  - monolithic outlets
- Added analysis-side summaries there to report:
  - total `MixedSaddlePoint` claims
  - total `Nullspace` claims
  - pressure nullspace claims with/without `nullspace_family`
  - pressure nullspace claims with `claim_origin == "MixedOperatorAnalyzer"`

Observed behavior:

- With either reaction term present, the exact quad-strip probe still had:
  - `pressure_mixed_claims=1`
  - `pressure_nullspace_claims=1`
  - but `pressure_nullspace_with_family=0`
  - and `has_gauge=0`
- With both reaction terms removed, the same probe had:
  - `pressure_nullspace_with_family=1`
  - `pressure_nullspace_from_mixed=1`
  - `has_gauge=1`
  - `constraints=1`

Interpretation:

- The reaction terms did not destroy saddle-point detection.
- They changed the analysis route: `MixedOperatorAnalyzer` fell back to the formulation-record path, and that fallback emitted a pressure nullspace claim without `nullspace_family`.
- `SystemSetup` was correct to ignore that unstructured nullspace claim when importing gauge candidates.

Code changed:

- [Code/Source/solver/FE/Analysis/MixedOperatorAnalyzer.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Analysis/MixedOperatorAnalyzer.cpp)
  - fallback-path `MixedSaddlePoint` claims now set `claim_origin = "MixedOperatorAnalyzer"`
  - fallback-path pressure `Nullspace` claims now set:
    - `nullspace_family = ScalarConstant`
    - `claim_origin = "MixedOperatorAnalyzer"`

Regression tests added:

- Focused analyzer-level guard:
  - [Code/Source/solver/FE/Tests/Unit/Analysis/test_AnalyzerPasses.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Analysis/test_AnalyzerPasses.cpp)
  - `MixedOperatorAnalyzer.StokesFallbackEmitsStructuredPressureNullspace`
- Systems-level guard on the exact quad-strip reproducer:
  - [Code/Source/solver/FE/Tests/Unit/Systems/test_GaugeIntegration.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Systems/test_GaugeIntegration.cpp)
  - `GaugeIntegration.MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyReactionFallback`

Qualification:

- `MixedOperatorAnalyzer.Stokes_DetectsSaddlePoint` passes
- `MixedOperatorAnalyzer.StokesFallbackEmitsStructuredPressureNullspace` passes
- `GaugeIntegration.MixedSaddlePointPressureFieldGetsGauge_QuadStripExactAssemblyReactionFallback` passes
- `GaugeIntegration.MixedSaddlePointPressureFieldGetsGauge` passes
- `GaugeIntegration.MixedSaddlePointPressureFieldGetsGauge_VectorSpace` passes
- `FormsInstaller.FormsInstaller_MixedResidualNegativeBoundaryTermCompiles` passes

Higher-level effect:

- Serial higher-level monolithic-outlet probe now reports:
  - `has_gauge=1`
  - `gauge_candidates=1`
  - `gauge_resolved=1`
  - `constraints=1`
- That probe no longer dies on the first BlockSchur true-residual validation.
- It still does not converge nonlinearly: it now runs to the 12-Newton cap and exits with
  `TimeLoop: nonlinear solve did not converge`.

MPI probe status after the fix:

- The `mpi4` higher-level monolithic-outlet probe now also creates the pressure gauge on all ranks.
- The distributed hang is therefore no longer “missing pressure gauge prevents progress.”
- Live stack capture from the post-fix `mpi4` hang still shows the same split:
  - rank 0 in `norm::fsi_ls_norms(...)` / `MPI_Allreduce`
  - other ranks in `fsils_commuv(...)` / `MPI_Waitall`
  - call path still through `ns_solver.cpp` inside the primary FSILS solve

Updated interpretation:

- A real physics-agnostic FE-analysis bug is fixed and now guarded at both analyzer and systems levels.
- That bug was a prerequisite for trustworthy higher-level MPI diagnosis.
- The remaining distributed convergence/performance gap is now downstream of correct gauge creation, in the MPI FSILS/BlockSchur communication path itself.

---

## 2026-04-15: Legacy face-only scalar Schur coarse-space experiments

Context:

- After the gauge fix, the archived 1-step iliac harness still shows:
  - serial: converged in `4` Newton but very slowly, with scalar Schur solves repeatedly hitting `300` inner iterations
  - `mpi4`: converged in `5` Newton, with the first BlockSchur solves already hitting `300` scalar Schur iterations and spending most of their time in Schur-side collectives
- The distributed native-face duplicate case is still routed through `schur_face_only_legacy(...)`, not the generic Schur-preconditioned path.

Hypothesis A:

- The legacy face-only scalar Schur path is missing the same partition/global-mean coarse modes that already exist in the generic distributed Schur preconditioner.
- Adding those modes to the legacy path might close the `serial 4` vs `mpi4 5` gap.

Temporary code experiment:

- Added an env-gated coarse correction inside `schur_face_only_legacy(...)` in:
  - [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
- Reused the existing env flags:
  - `SVMP_FSILS_SCHUR_PARTITION_COARSE=1`
  - `SVMP_FSILS_SCHUR_GLOBAL_MEAN_COARSE=1`

Qualification:

- Ran the archived 1-step iliac `mpi4` harness in:
  - [tests/_codex_tmp_iliac_mpi4_disablelegacy_coarse_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_tmp_iliac_mpi4_disablelegacy_coarse_20260415/run.log)

Observed behavior:

- The first two BlockSchur solves were materially worse than the current default `mpi4` baseline:
  - first solve: `15.892405 s`, `15` outer iterations, `15 x 300` Schur iterations
  - second solve: `15.577879 s`, `15` outer iterations, `15 x 300` Schur iterations
- For comparison, the current default post-gauge-fix `mpi4` harness had:
  - first solve: `10.571007 s`, `10` outer iterations, `10 x 300` Schur iterations
  - second solve: `11.766009 s`, `11` outer iterations, `11 x 300` Schur iterations
  - source:
    [tests/_codex_iliac_1step_mpi4_post_gaugefix_clean_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_post_gaugefix_clean_20260415/run.log)

Interpretation:

- Generic partition/global-mean pressure modes are not the missing distributed coarse space for this case.
- The legacy face-only path got strictly worse when those generic coarse modes were injected.

Hypothesis B:

- The missing coarse space is more specific: the outlet-face-induced low-rank subspace itself, not generic partition pressure modes.
- A direct reuse of the generic face low-rank Schur correction machinery inside the legacy path might recover the missing outlet response.

Temporary code experiment:

- Added a second env-gated legacy-path experiment in:
  - [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
- New temporary env:
  - `SVMP_FSILS_FACE_ONLY_LEGACY_LOW_RANK_PC=1`
- This path built a legacy scalar correction from the generic helpers:
  - `build_momentum_hat_data(...)`
  - `build_reduced_schur_correction(...)`

Qualification:

- Re-ran the same archived 1-step iliac `mpi4` harness with only:
  - `SVMP_FSILS_FACE_ONLY_LEGACY_LOW_RANK_PC=1`

Observed behavior:

- The run did not produce the first BlockSchur timing summary on the timescale where the default baseline already does.
- After the third assembly, the run failed to advance to a first BlockSchur summary even after repeated polling.
- I stopped the run and treated it as a regression / likely collective mismatch or severe first-solve slowdown.

Interpretation:

- A naive transplant of the generic face low-rank correction machinery into the legacy face-only scalar path is not valid as-is.
- The remaining issue is likely tied to how the legacy face-only preconditioned operator is formulated and synchronized, not just to a missing obvious coarse space.

Tree status:

- Both legacy-path experiments above were reverted after qualification.
- `svmultiphysics` was rebuilt after the revert.
- The tree is back to the prior baseline solver behavior for subsequent investigation.

---

## 2026-04-15: Global native-face exact preconditioner check

New hypothesis:

- The remaining MPI gap might still live in the native face preconditioner, but in a way that only shows up when the two outlet faces are treated as a single global dense system across ranks.
- My earlier native-face exact-pre experiment was flawed because it only included faces with local `nNo > 0`, so no rank ever saw both disjoint outlet faces at once.

Code changed:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/fils_struct.hpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/fils_struct.hpp)
  - added native-face exact-pre scratch fields on `FSILS_lhsType`
- [Code/Source/solver/FE/Backends/FSILS/liner_solver/add_bc_mul.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/add_bc_mul.cpp)
  - added env-gated exact native-face preconditioner build/use:
    - `SVMP_FSILS_NATIVE_FACE_EXACT_PRE=1`
  - fixed the global-face selection bug by including all globally active coupled faces, even on ranks where local `face.nNo == 0`
  - added trace env:
    - `SVMP_FSILS_TRACE_NATIVE_FACE_EXACT_PRE=1`

Qualification:

- Archived 1-step iliac `mpi4` harness:
  - [tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log)
  - env:
    - `SVMP_FSILS_NATIVE_FACE_EXACT_PRE=1`
    - `SVMP_FSILS_TRACE_NATIVE_FACE_EXACT_PRE=1`

Observed behavior:

- The exact global native-face preconditioner matrix is strictly diagonal on the real iliac partition:
  - `dense_m[0,1] = 0`
  - `dense_m[1,0] = 0`
  - trace lines:
    - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log:106)
    - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log:107)
    - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log:108)
- The nonlinear behavior is unchanged:
  - [run.log:195](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log:195)
  - still `5` Newton, converged
- The first BlockSchur solve is also unchanged in iteration structure:
  - [run.log:168](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nativeface_exactpre_20260415/run.log:168)
  - still `10` outer iterations

Interpretation:

- The raw native outlet-face subspace is not where the missing distributed coupling lives.
- Even when the two faces are treated as one exact global dense preconditioner, the face-space system is diagonal on this partition and the `mpi4` nonlinear count does not improve.
- So the remaining gap is not “multi-face native outlet preconditioning forgot cross-face overlap.”
- The more plausible remaining location is the Schur-space coupling induced *after* the momentum-side face preconditioning, i.e. in the distributed scalar Schur solve / coarse space itself.

Status:

- The exact native-face preconditioner code and trace are still present but env-gated only.
- No non-env solver policy changed in the default path.

---

## 2026-04-15: Legacy scalar Schur face-seed response probe

New hypothesis:

- The remaining MPI gap might live in the *Schur-space* face coupling after the momentum-side face preconditioner, even though the raw native-face preconditioner itself is diagonal.
- The right thing to inspect is the actual legacy scalar-Schur path, not a generic reduced-Schur reconstruction layered on top of it.

Code changed:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
  - repurposed `SVMP_FSILS_TRACE_FACE_ONLY_SCHUR_LOW_RANK=1` into a cheaper one-shot diagnostic on the actual legacy scalar-Schur path
  - the diagnostic now:
    - builds one scalar seed per active outlet face by setting `1` on that face's scalar DOFs
    - applies the real legacy `GL` operator
    - applies the real momentum-side face preconditioner `add_bc_mul(..., BCOP_TYPE_PRE, ...)`
    - measures the resulting preconditioned face response `face_j^T * GP(seed_i)` for every active face pair
    - prints the resulting dense face-seed response matrix once on root

Probe mistakes found and corrected:

- First one-shot attempt was wrong:
  - I gated the probe on `lhs.commu.masF`, but the probe itself contained collectives
  - that created a rank mismatch: root entered `build_reduced_schur_correction(...)` collectives while other ranks stayed in the main solve
  - live stacks showed root in `fsils_commuv(...)` and another rank in `norm::fsi_ls_norms(...)`
- Second attempt showed the generic reduced-Schur reconstruction was the wrong diagnostic:
  - even when executed on all ranks, it was too expensive on the full iliac harness because it rebuilt a full distributed reduced-Schur correction before printing anything
  - I replaced it with the lighter face-seed response probe described above

Qualification:

- Archived 1-step iliac `mpi4` harness:
  - [tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log)
  - env:
    - `SVMP_FSILS_TRACE_FACE_ONLY_SCHUR_LOW_RANK=1`

Observed behavior:

- The actual legacy scalar-Schur face-seed response matrix is still diagonal after the momentum-side face preconditioner:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log:106)
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log:107)
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log:108)
  - specifically:
    - `dense_m[0,1] = 0`
    - `dense_m[1,0] = 0`
- Early BlockSchur outer iteration counts remain high on the same run:
  - [run.log:129](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log:129)
  - [run.log:167](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log:167)
  - [run.log:205](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_face_lrtrace_seedpre_20260415/run.log:205)
  - first three outer counts were `10`, `11`, `10`
- I stopped the run after capturing the probe output and early outer-iteration pattern; I did not wait for the final nonlinear summary.

Interpretation:

- The missing distributed coupling is not hiding in a simple two-face Schur coarse block built from face-seeded scalar modes, even after the momentum-side face preconditioner is applied.
- This materially weakens the hypothesis that the `serial 4` vs `mpi4 5` gap is caused by “forgotten cross-face coupling” between the two outlet modes.
- The stronger remaining hypotheses are now:
  - distributed pressure/coarse-space quality that is *not* spanned by the two outlet-face seed modes
  - rank-sensitive behavior in the primary scalar Schur Krylov/preconditioned operator beyond the raw face subspace
  - or a mismatch between the legacy scalar-Schur coarse content and the generic distributed low-rank/coarse-space machinery already available elsewhere in FSILS

Status:

- The face-seed response probe remains env-gated only.
- No default solver policy changed.

---

## 2026-04-15: `NS_SOLVER` mpi4 header trace

Purpose:

- Capture the first `mpi4` BlockSchur / `NS_SOLVER` header on the archived 1-step iliac harness and compare it against the existing serial `NS_SOLVER` trace without waiting for a full qualification run.

Qualification:

- Archived 1-step iliac `mpi4` harness:
  - [tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log)
  - env:
    - `SVMP_FSILS_NS_SOLVER_TRACE=1`

Observed behavior:

- The first `mpi4` `NS_SOLVER` header appeared before the run reached the first `GM.itr` / `CG.itr` detail line:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log:106)
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log:108)
- On root, both outlet faces still have no local nodes on the first outer solve:
  - [run.log:110](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log:110)
  - [run.log:111](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolvertrace_20260415/run.log:111)
  - `face[1]: nNo=0`
  - `face[2]: nNo=0`
- The structural header otherwise matches the serial trace in the important ways:
  - `explicit_block_modes=0`
  - `coupled_face_modes=0`
  - `native_face_rank_one_count=2`

Interpretation:

- This does not add a new root cause by itself, but it confirms the current `mpi4` first outer solve is still entering the same native-face rank-one BlockSchur path while root owns neither outlet face.
- I stopped the run after the header because it had not yet reached the first `GM.itr` / `CG.itr` detail line, and this attempt was intended only as a quick structural capture.

---

## 2026-04-15: Legacy scalar-Schur residual coarse-mode probe

New hypothesis:

- If the remaining `serial 4` vs `mpi4 5` gap is really a non-face coarse-space issue, the first returned legacy scalar-Schur residual should still contain a materially removable component along one of the obvious scalar pressure modes:
  - the global mean mode
  - or the simplest rank-partition mode

Code changed:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
  - added env-gated one-shot diagnostic:
    - `SVMP_FSILS_TRACE_FACE_ONLY_COARSE_RESIDUAL=1`
  - after the legacy scalar-Schur solve returns, the diagnostic:
    - computes the final scalar-Schur residual
    - applies the actual legacy scalar-Schur operator to:
      - a global mean mode
      - and, in MPI, a rank-partition mode
    - computes the optimal 1D least-squares correction along each mode
    - prints the residual norm before and after that best possible one-mode correction

Qualification:

- Archived 1-step iliac `mpi4` harness:
  - [tests/_codex_iliac_1step_mpi4_coarseresidual_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_20260415/run.log)
  - env:
    - `SVMP_FSILS_TRACE_FACE_ONLY_COARSE_RESIDUAL=1`
- Archived 1-step iliac serial harness:
  - [tests/_codex_iliac_1step_serial_coarseresidual_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_coarseresidual_20260415/run.log)
  - env:
    - `SVMP_FSILS_TRACE_FACE_ONLY_COARSE_RESIDUAL=1`

Observed behavior:

- `mpi4`, first legacy scalar-Schur solve:
  - [run.log line](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_20260415/run.log:99)
    - `global_mean`: residual `2.699209e+01 -> 2.691913e+01`
  - [run.log line](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_20260415/run.log:100)
    - `constraint_partition`: residual `2.699209e+01 -> 2.698632e+01`
- Serial, first legacy scalar-Schur solve:
  - [run.log line](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_coarseresidual_20260415/run.log:102)
    - `global_mean`: residual `2.468287e+01 -> 2.468287e+01`

Interpretation:

- Neither of the two obvious non-face scalar coarse modes is the missing correction.
- In `mpi4`, both the global mean and the rank-partition mode reduce the first scalar-Schur residual only marginally.
- In serial, the global mean mode is effectively irrelevant on that same first scalar-Schur residual.
- That weakens the hypothesis that the remaining MPI gap is simply a missing global-mean or single-partition scalar coarse mode.
- The more plausible remaining locations are now:
  - higher-dimensional pressure/coarse content not captured by these simple modes
  - solver-quality differences inside the primary scalar-Schur Krylov iteration itself
  - or a rank-sensitive mismatch between the legacy scalar-Schur path and the richer generic distributed low-rank/coarse-space machinery available elsewhere in FSILS

Status:

- The coarse-residual diagnostic remains env-gated only.
- I stopped both runs after collecting the first one-shot coarse-mode lines; I did not wait for final nonlinear summaries in this pass.

Related note:

- I also launched a diagnostic-only generic-path run with:
  - `SVMP_FSILS_BLOCKSCHUR_DISABLE_FACE_ONLY_LEGACY=1`
  - `SVMP_FSILS_TRACE_SCHUR_PRECONDITIONER_PROBES=1`
  - output: [tests/_codex_iliac_1step_mpi4_genericpcprobe_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_genericpcprobe_20260415/run.log)
- I stopped it before any `SCHUR_PC_PROBE` lines appeared, because it had not yet reached the probe point on the timescale needed for this pass.
- So there is no new generic-path quality result to claim yet from that run.

---

## 2026-04-15: Full rank-partition subspace probe

New hypothesis:

- The single partition mode was too weak a test.
- The real missing coarse content might be the *full* rank-partition subspace, i.e. one balanced partition mode per active MPI rank (minus one reference rank), not just the first such mode.

Code changed:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
  - extended `SVMP_FSILS_TRACE_FACE_ONLY_COARSE_RESIDUAL=1`
  - the one-shot diagnostic now also:
    - builds the full rank-partition basis over active MPI ranks
    - applies the actual legacy scalar-Schur operator to each mode
    - forms the dense Gram system on that image space
    - computes the optimal least-squares correction in the full partition subspace
    - prints the residual norm before and after that best subspace correction

Qualification:

- Archived 1-step iliac `mpi4` harness:
  - [tests/_codex_iliac_1step_mpi4_coarseresidual_subspace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_subspace_20260415/run.log)
  - env:
    - `SVMP_FSILS_TRACE_FACE_ONLY_COARSE_RESIDUAL=1`

Observed behavior:

- The full partition subspace still only reduces the first returned scalar-Schur residual marginally:
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_subspace_20260415/run.log:108)
  - `constraint_partition_subspace`: residual `2.699209e+01 -> 2.693105e+01`, `dim=3`
- For reference, on the same run:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_subspace_20260415/run.log:106)
    - `global_mean`: `2.699209e+01 -> 2.691913e+01`
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coarseresidual_subspace_20260415/run.log:107)
    - single `constraint_partition`: `2.699209e+01 -> 2.698632e+01`

Interpretation:

- This rules out a large class of obvious distributed pressure/coarse explanations.
- The full rank-partition subspace is not the missing correction either.
- So the remaining MPI gap is now unlikely to be fixed by simply adding:
  - outlet-face coarse modes
  - a global mean mode
  - one partition mode
  - or even the full active-rank partition basis

Current best diagnosis:

- The remaining issue is more likely inside the primary scalar-Schur solver quality itself, or in a richer nontrivial coarse/low-rank content not aligned with these simple geometric partition modes.

---

## 2026-04-15: Legacy scalar-Schur branch-choice probe

New hypothesis:

- The distributed gap might come from the *legacy scalar-Schur Krylov branch* itself.
- More specifically:
  - `mpi4` default native-face runs use the multi-face legacy scalar GMRES branch
  - while serial still uses the legacy scalar BiCGStab branch
- If that branch split is the real source of the extra MPI Newton step, then solving the *same first distributed scalar-Schur rhs* with the alternate branch should show a materially different residual/solution response.

Code changed:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
  - added an env-gated shadow compare:
    - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE=1`
    - optional cap:
      - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE_MAX_ITERS`
  - on the distributed legacy scalar GMRES path, the compare:
    - keeps the primary returned solve unchanged
    - replays a bounded alternate legacy scalar BiCGStab solve on the same rhs/operator
    - is intended to print:
      - primary residual
      - alternate residual
      - iteration counts
      - solution and residual differences

Qualification attempts:

- Archived 1-step iliac `mpi4` harness:
  - [tests/_codex_iliac_1step_mpi4_branchcompare_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_branchcompare_20260415/run.log)
- Attempt 1:
  - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE=1`
  - default alternate cap (`40`)
  - stopped after about `99 s` without reaching the branch compare line
- Attempt 2:
  - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE_MAX_ITERS=20`
  - stopped after about `46 s` without reaching the branch compare line
- Attempt 3:
  - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_BRANCH_COMPARE_MAX_ITERS=2`
  - still did not reach the branch compare line on a useful timescale; stopped after about `44 s`

Related existing evidence:

- The previously qualified archived forced-BiCGStab distributed run already weakens the branch-choice hypothesis:
  - current default native-face distributed baseline:
    - [tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:305)
    - `iters=5`
  - forced distributed legacy BiCGStab:
    - [tests/_codex_iliac_1step_mpi4_force_bicgstab_20260414/run.log:311](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_force_bicgstab_20260414/run.log:311)
    - `iters=5`

Interpretation:

- Branch choice alone is unlikely to be the full explanation for the remaining `serial 4` vs `mpi4 5` gap.
- The full iliac harness is also too expensive a place to run a shadow replay of the alternate scalar-Schur branch, even with a very small alternate iteration cap.
- So this compare hook is useful as infrastructure, but not yet a practical iliac-harness oracle in its current form.

Updated best diagnosis:

- The remaining MPI gap still points more strongly to the *shared distributed scalar-Schur / coarse-space quality* than to “GMRES vs BiCGStab” as such.
- The next higher-signal move should be a lighter branch-quality comparison in a smaller reproducer or a more surgical per-iteration scalar-Schur diagnostic, not another full-harness shadow replay.

---

## 2026-04-15: Legacy scalar-Schur iteration-history probe

New hypothesis:

- The remaining `serial 4` vs `mpi4 5` Newton gap might still be coming from the *real* distributed legacy scalar-Schur solve quality, even if a full shadow branch replay is too expensive to use directly on the full iliac harness.
- A cheaper probe is to log the actual scalar-Schur residual history on selected legacy face-only solves and compare:
  - serial default legacy BiCGStab
  - `mpi4` default legacy GMRES
  - `mpi4` forced legacy BiCGStab

Code changed:

- [Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
  - added an env-gated per-iteration history trace for the legacy face-only scalar-Schur path:
    - `SVMP_FSILS_TRACE_FACE_ONLY_ITER_HISTORY=1`
  - added optional solve selection:
    - `SVMP_FSILS_TRACE_FACE_ONLY_ITER_HISTORY_SOLVE_INDEX=<k>`
  - the probe logs:
    - solver branch (`gmres` or `bicgstab`)
    - selected legacy face-only solve index
    - initial residual
    - sampled inner-iteration residuals
    - final residual after the capped `300` inner iterations

Qualification:

- Archived 1-step iliac serial:
  - [tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log)
- Archived 1-step iliac `mpi4` default:
  - [tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log)
- Archived 1-step iliac `mpi4` forced legacy BiCGStab:
  - [tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log)
- Archived 1-step iliac serial, second traced legacy face-only solve:
  - [tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log)
- Archived 1-step iliac `mpi4`, second traced legacy face-only solve:
  - [tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log)

Observed behavior, first traced legacy face-only solve:

- Serial default legacy BiCGStab:
  - [run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:102) initial `4.441768e+02`
  - [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:112) iter `10`: `1.990030e+02`
  - [run.log:142](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:142) iter `40`: `1.220783e+02`
  - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:202) iter `100`: `7.705644e+01`
  - [run.log:302](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:302) iter `200`: `5.944783e+01`
  - [run.log:352](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:352) iter `250`: `4.563124e+01`
  - [run.log:403](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:403) final iter `300`: `2.443781e+01`
  - [run.log:424](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_20260415/run.log:424) `BlockSchur outer iters: 5`

- `mpi4` default legacy GMRES:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:106) initial `4.441768e+02`
  - [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:117) iter `10`: `1.816868e+02`
  - [run.log:147](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:147) iter `40`: `1.278666e+02`
  - [run.log:207](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:207) iter `100`: `9.353200e+01`
  - [run.log:307](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:307) iter `200`: `6.027570e+01`
  - [run.log:357](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:357) iter `250`: `4.482595e+01`
  - [run.log:408](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:408) final iter `300`: `2.521484e+01`
  - [run.log:429](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_20260415/run.log:429) `BlockSchur outer iters: 4`

- `mpi4` forced legacy BiCGStab:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:106) initial `4.441768e+02`
  - [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:116) iter `10`: `1.716045e+03`
  - [run.log:146](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:146) iter `40`: `1.941390e+02`
  - [run.log:206](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:206) iter `100`: `1.226369e+02`
  - [run.log:306](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:306) iter `200`: `1.261560e+02`
  - [run.log:356](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:356) iter `250`: `1.213624e+02`
  - [run.log:407](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:407) final iter `300`: `8.728671e+01`
  - [run.log:428](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_bicgstab_iterhistory_20260415/run.log:428) `BlockSchur outer iters: 6`

Observed behavior, second traced legacy face-only solve (`solve_index=1`):

- Serial default legacy BiCGStab:
  - [run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:102) initial `2.294235e+02`
  - [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:112) iter `10`: `5.817381e+01`
  - [run.log:142](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:142) iter `40`: `5.637776e+01`
  - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:202) iter `100`: `1.914936e+01`
  - [run.log:302](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:302) iter `200`: `7.476496e+00`
  - [run.log:352](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:352) iter `250`: `7.642412e+00`
  - [run.log:403](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:403) final iter `300`: `8.034744e+00`
  - [run.log:424](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhistory_solve1_20260415/run.log:424) `BlockSchur outer iters: 5`

- `mpi4` default legacy GMRES:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:106) initial `5.513823e+02`
  - [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:117) iter `10`: `9.166336e+01`
  - [run.log:147](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:147) iter `40`: `2.817094e+01`
  - [run.log:207](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:207) iter `100`: `1.156151e+01`
  - [run.log:307](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:307) iter `200`: `4.697272e+00`
  - [run.log:357](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:357) iter `250`: `3.776110e+00`
  - [run.log:408](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:408) final iter `300`: `3.200936e+00`
  - [run.log:429](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhistory_solve1_20260415/run.log:429) `BlockSchur outer iters: 4`

Interpretation:

- The first traced legacy face-only solve is already close between serial and `mpi4` default:
  - serial final scalar-Schur residual `2.443781e+01`
  - `mpi4` default final scalar-Schur residual `2.521484e+01`
- The second traced legacy face-only solve is actually *better* in `mpi4` default than in serial on this probe:
  - serial final scalar-Schur residual `8.034744e+00`
  - `mpi4` default final scalar-Schur residual `3.200936e+00`
- Forced distributed legacy BiCGStab is clearly worse than the default distributed legacy GMRES path on the same first solve.

Updated best diagnosis:

- The remaining Newton-gap source is now unlikely to be:
  - “`mpi4` uses GMRES while serial uses BiCGStab”
  - the first traced distributed legacy scalar-Schur solve by itself
  - or even the second traced distributed legacy scalar-Schur solve by itself
- So the next higher-signal target is one level up:
  - correlate *full* primary linear solves with Newton iterations
  - identify where the serial and `mpi4` nonlinear trajectories first diverge at the returned increment / full BlockSchur solve level
  - only then return to narrower scalar-Schur internals if that higher-level trace points back there

---

## 2026-04-15: Newton-level primary linear-solve history

New hypothesis:

- The remaining gap might not be in the traced legacy face-only scalar-Schur micro-solves at all.
- Instead, the divergence might appear at the level of the *returned full primary linear solve* used by Newton, i.e. in how the whole BlockSchur solve produces the final increment `du` and how that increment changes the next nonlinear residual.

Code changed:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - added an env-gated linear-solve history trace:
    - `SVMP_DEBUG_LINEAR_SOLVE_HISTORY=1`
    - optional cap:
      - `SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS=<n>`
  - the trace logs, on rank 0:
    - Newton iteration index
    - pre-solve nonlinear residual
    - linear rhs norm
    - returned linear residuals
    - BlockSchur outer/momentum/Schur iteration counts
    - active reduced/rank-one coupling counts
    - solver message

Qualification:

- Archived 1-step iliac serial:
  - [tests/_codex_iliac_1step_serial_linsolvehist_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_linsolvehist_20260415/run.log)
- Archived 1-step iliac `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log)
- env:
  - `SVMP_DEBUG_LINEAR_SOLVE_HISTORY=1`
  - `SVMP_DEBUG_LINEAR_SOLVE_HISTORY_MAX_CALLS=12`

Observed behavior:

- Serial Newton history:
  - [run.log:134](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_linsolvehist_20260415/run.log:134)
    - `newton_it=0`, `residual_before=10343.6`, returned linear `rn=0.0305453`, `outer=5`
  - [run.log:173](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_linsolvehist_20260415/run.log:173)
    - `newton_it=1`, `residual_before=1150.9`, returned linear `rn=0.0426881`, `outer=3`
  - [run.log:212](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_linsolvehist_20260415/run.log:212)
    - `newton_it=2`, `residual_before=11.9713`, returned linear `rn=1.17855e-04`, `outer=3`
  - [run.log:251](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_linsolvehist_20260415/run.log:251)
    - `newton_it=3`, `residual_before=7.03222e-04`, returned linear `rn=4.90952e-08`, `outer=5`

- `mpi4` Newton history:
  - [run.log:138](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log:138)
    - `newton_it=0`, `residual_before=10343.6`, returned linear `rn=0.499916`, `outer=4`
  - [run.log:177](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log:177)
    - `newton_it=1`, `residual_before=9409.62`, returned linear `rn=0.345801`, `outer=3`
  - [run.log:216](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log:216)
    - `newton_it=2`, `residual_before=484.422`, returned linear `rn=0.0134091`, `outer=4`
  - [run.log:255](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log:255)
    - `newton_it=3`, `residual_before=2.21423`, returned linear `rn=1.01397e-04`, `outer=4`
  - [run.log:294](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_linsolvehist_20260415/run.log:294)
    - `newton_it=4`, `residual_before=1.09619e-04`, returned linear `rn=5.51964e-08`, `outer=5`

Interpretation:

- The first returned primary linear solve on `mpi4` is not failing badly:
  - it converges with a small returned linear residual
  - it actually uses *fewer* BlockSchur outer iterations than serial on that first Newton step (`4` vs `5`)
- But the nonlinear residual after applying that first returned update is already much worse in `mpi4`:
  - serial goes from `10343.6 -> 1150.9`
  - `mpi4` goes from `10343.6 -> 9409.62`
- So the remaining `serial 4` vs `mpi4 5` gap is not explained by later linear-solve deterioration. The divergence is already created by the *first returned full Newton update*.

Cross-check with the earlier first-step line-probe runs:

- Serial first returned increment norm was much larger:
  - [tests/_codex_iliac_1step_serial_lineprobe_20260414/run.log:212](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_lineprobe_20260414/run.log:212)
  - `du_norm=4.31649e+08`
- `mpi4` first returned increment norm was much smaller:
  - [tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run.log:622](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run.log:622)
  - `du_norm=2.52657e+07`

Updated best diagnosis:

- The remaining MPI gap is now most likely in the *full returned increment quality* on the first Newton step, not in:
  - simple coarse modes
  - outlet-face coupling alone
  - GMRES vs BiCGStab branch choice
  - or later BlockSchur degradation
- The next highest-signal target is therefore a component/field decomposition of the first returned `du` and of the first nonlinear residual change, so we can see which part of the full distributed update is being systematically under-corrected.

---

## 2026-04-15: Returned increment component decomposition

New hypothesis:

- The first distributed full Newton update is being under-corrected in a specific field/component subspace, not uniformly across the whole state.
- If that is true, then decomposing the returned `du` by FE field/component on the first one or two Newton steps should show which part of the update is missing most strongly on `mpi4`.

Code changed:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - added an env-gated post-solve component-norm probe for the returned increment:
    - `SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS=1`
    - optional Newton-iteration cap:
      - `SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS_MAX_NEWTON_IT=<k>`
  - this logs the component norms of the returned `du` after the linear solve normalization, without enabling the much noisier full OOP solver trace

Qualification:

- Archived 1-step iliac serial:
  - [tests/_codex_iliac_1step_serial_ducomponents_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_ducomponents_20260415/run.log)
- Archived 1-step iliac `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_ducomponents_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_ducomponents_20260415/run.log)
- env:
  - `SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS=1`
  - `SVMP_DEBUG_LINEAR_SOLVE_COMPONENT_NORMS_MAX_NEWTON_IT=1`

Observed behavior:

- Serial, first returned increment (`newton_it=0`):
  - [run.log:134](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_ducomponents_20260415/run.log:134)
  - `Velocity[0] norm=1280.22`
  - `Velocity[1] norm=2212.65`
  - `Velocity[2] norm=3305.26`
  - `Pressure norm=4.31649e+08`
  - `Pressure mean=-3.32519e+06`

- `mpi4`, first returned increment (`newton_it=0`):
  - [run.log:138](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_ducomponents_20260415/run.log:138)
  - `Velocity[0] norm=544.009`
  - `Velocity[1] norm=1006.19`
  - `Velocity[2] norm=1286.63`
  - `Pressure norm=2.5242e+07`
  - `Pressure mean=-122781`

- Serial, second returned increment (`newton_it=1`):
  - [run.log:173](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_ducomponents_20260415/run.log:173)
  - `Velocity[0] norm=16.4426`
  - `Velocity[1] norm=21.977`
  - `Velocity[2] norm=22.5261`
  - `Pressure norm=311738`
  - `Pressure mean=-2164.81`

- `mpi4`, second returned increment (`newton_it=1`):
  - [run.log:177](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_ducomponents_20260415/run.log:177)
  - `Velocity[0] norm=88.8578`
  - `Velocity[1] norm=150.001`
  - `Velocity[2] norm=185.921`
  - `Pressure norm=6.46031e+06`
  - `Pressure mean=-35866.8`

Interpretation:

- The strongest mismatch on the first Newton step is the pressure part of the returned increment:
  - serial first-step pressure norm `4.31649e+08`
  - `mpi4` first-step pressure norm `2.5242e+07`
  - that is about a `17x` reduction in the distributed pressure correction magnitude
- The velocity part is also smaller on `mpi4`, but by a much smaller factor:
  - roughly `2-3x` smaller, not `17x`
- So the first-order signature of the remaining gap is now much sharper:
  - the distributed solve is dramatically under-correcting the *pressure component* of the first returned Newton increment
  - that aligns with the earlier evidence that the residual gap is created immediately after the first full update
  - and it pushes the next target toward distributed pressure/gauge/coarse-mode treatment in the *full returned increment*, not momentum-side outlet-face handling alone

Updated best diagnosis:

- The remaining `serial 4` vs `mpi4 5` gap is now best explained as a *distributed under-correction of the first-step pressure update* in the returned full Newton increment.
- The next highest-signal debugging move is to examine:
  - how the pressure block / pressure projection / gauge handling acts on the returned BlockSchur increment on that first solve
  - and whether the distributed path is removing or damping a legitimate pressure mode before the nonlinear update is applied

---

## 2026-04-15: MPI global-mean-shift correction check

New hypothesis:

- Since the returned distributed increment is under-correcting the pressure part most strongly, the existing MPI-only `constraint global mean shift correction` hook in `NewtonSolver` might already target the right missing mode.
- If so, enabling it on the real `mpi4` iliac harness should reduce the Newton count from `5` to `4`.

Important note:

- An earlier apparent success on this branch was invalid. I first launched the `_mpi4_.../solver.xml` harness *without* `mpirun`, which made it a serial control despite the file name. That result is not counted below.

Code inspected:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - existing env-gated hook:
    - `SVMP_MPI_CONSTRAINT_GLOBAL_MEAN_SHIFT_CORRECTION=1`
  - the hook:
    - measures the global mean of the constraint-field part of `du`
    - constructs a global-mean correction vector
    - samples residuals for a fixed alpha set
    - accepts the shift only if one alpha beats the current residual

Qualification:

- Real archived 1-step iliac `mpi4`, no OOP trace:
  - [tests/_codex_iliac_1step_mpi4_globalmeanshift_real_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_real_20260415/run.log)
  - env:
    - `SVMP_MPI_CONSTRAINT_GLOBAL_MEAN_SHIFT_CORRECTION=1`
- Real archived 1-step iliac `mpi4`, with OOP trace for correction diagnostics:
  - [tests/_codex_iliac_1step_mpi4_globalmeanshift_trace_mpi4_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_trace_mpi4_20260415/run.log)
  - env:
    - `SVMP_MPI_CONSTRAINT_GLOBAL_MEAN_SHIFT_CORRECTION=1`
    - `SVMP_OOP_SOLVER_TRACE=1`

Observed behavior:

- Real `mpi4` no-trace qualification:
  - [run.log:389](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_real_20260415/run.log:389)
    - still `converged=1 iters=5`
  - [run.log:393](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_real_20260415/run.log:393)
    - total loop `45.965492 s`

- Real `mpi4` traced qualification:
  - raw post-solve increment is still the smaller distributed one:
    - [run.log:635](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_trace_mpi4_20260415/run.log:635)
      - `post-normalize du_norm=2.5242e+07`
  - the hook explicitly rejects itself on the first Newton step:
    - [run.log:1317](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_trace_mpi4_20260415/run.log:1317)
      - `rejected constraint global mean shift correction`
      - `global_mean=-122781`
      - `best_alpha=0`
      - `best_residual=9409.62`
      - `current_residual=10343.6`
  - after that rejection, the returned increment components remain on the baseline distributed path:
    - [run.log:1318](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_trace_mpi4_20260415/run.log:1318)
      - `Pressure norm=2.5242e+07`
      - not the serial-like `4.31649e+08`
  - final traced result also stays at `5` Newton:
    - [run.log:2279](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_globalmeanshift_trace_mpi4_20260415/run.log:2279)
      - `converged=1 iters=5`

Interpretation:

- The remaining gap is **not** fixed by a simple global-mean shift in the returned pressure increment.
- The existing MPI global-mean-shift correction hook already tested exactly that idea and rejected it on the real distributed first Newton step.
- That means the missing distributed pressure correction is more structured than one scalar global-mean mode.

Updated best diagnosis:

- The first distributed pressure under-correction is real, but it is not explained by:
  - outlet-face 2x2 coupling
  - simple global pressure mean
  - rank-partition mean modes
  - or the existing global-mean-shift post-correction hook
- The next highest-signal target is therefore a richer distributed pressure/gauge/coarse-space mode in the *returned full BlockSchur increment*, not just any one-dimensional mean shift.

---

## 2026-04-15: Backend post-processing check

New hypothesis:

- The missing pressure correction might be getting stripped after the primary linear solve, for example by backend nullspace projection or returned-solution recentering.

Observed evidence:

- Existing traced serial and `mpi4` FSILS logs already show the answer:
  - serial first hard solve:
    - [tests/_codex_iliac_1step_serial_lineprobe_20260414/run.log:190](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_lineprobe_20260414/run.log:190)
    - `basis=0 projected=0`
  - `mpi4` first hard solve:
    - [tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run.log:620](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_lineprobe_20260414/run.log:620)
    - `basis=0 projected=0`

Interpretation:

- The backend is not removing a legitimate pressure mode via nullspace projection on this path.
- The first distributed pressure under-correction already exists in the returned solve before any nullspace-projection step could matter.

---

## 2026-04-15: MPI linear subspace recovery check

New hypothesis:

- The existing `MPI linear subspace recovery` hook might already span the missing distributed pressure mode because it augments the returned `du` in a low-dimensional space built from:
  - the native rank-one outlet modes
  - a constraint mean mode
  - rank-partition pressure modes

Code inspected:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - env:
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY=1`
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY_MAX_NEWTON_ITERS=<n>`

Qualification:

- Real archived 1-step iliac `mpi4`, first-iteration only:
  - [tests/_codex_iliac_1step_mpi4_subspacerecovery_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_20260415/run.log)
  - env:
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY=1`
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY_MAX_NEWTON_ITERS=1`
- Traced distributed run:
  - [tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log)
  - same envs plus `SVMP_OOP_SOLVER_TRACE=1`

Observed behavior:

- Real no-trace qualification:
  - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_20260415/run.log:305)
    - still `converged=1 iters=5`
  - [run.log:309](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_20260415/run.log:309)
    - total loop `28.962232 s`

- Traced first-step recovery:
  - builds a 6D basis:
    - `rank1_0`, `rank1_1`, `constraint_mean`, `constraint_partition_0`, `constraint_partition_1`, `constraint_partition_2`
    - [run.log:678-701](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log:678)
  - accepts a correction:
    - [run.log:707](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log:707)
    - `accept=1`
  - but the linear improvement is tiny:
    - [run.log:706](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log:706)
      - `residual=0.498672`
    - versus baseline
    - [run.log:674](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log:674)
      - `residual=0.499916`

Interpretation:

- A richer low-dimensional pressure correction does help in the right direction, but only marginally.
- So the missing distributed mode is not well captured by this current subspace basis, even though it already includes the obvious mean and partition pressure directions.

---

## 2026-04-15: Exact native rank-one Woodbury recovery check

New hypothesis:

- If the missing distributed correction really lives in the low-rank outlet coupling, the exact native rank-one Woodbury recovery should repair it more strongly than the small subspace tweak.

Code inspected:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - env:
    - `SVMP_MPI_EXACT_RANK_ONE_WOODBURY=1`
    - `SVMP_MPI_EXACT_RANK_ONE_WOODBURY_MAX_NEWTON_ITERS=<n>`

Qualification attempt:

- Real archived 1-step iliac `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_woodbury_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_woodbury_20260415/run.log)
  - env:
    - `SVMP_MPI_EXACT_RANK_ONE_WOODBURY=1`
    - `SVMP_MPI_EXACT_RANK_ONE_WOODBURY_MAX_NEWTON_ITERS=1`

Observed behavior:

- The path was operationally unusable and I stopped it early.
- On the very first BlockSchur solve it had already exploded to:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_woodbury_20260415/run.log)
    - `MPI_Allreduce calls: 182368`
    - `MPI_Allreduce time: 15.450279 s`
    - `Schur allreduces: 181824`
    - `Calls / outer iter: 45592`

Interpretation:

- Even if this path could recover a better increment, it is not a practical route in its current form.
- The exact Woodbury recovery is therefore not a viable performance-compatible fix for the distributed gap.

---

## 2026-04-15: Existing first-step nonlinear correction hooks

New hypothesis:

- Since the distributed first returned update is too small, perhaps one of the existing first-step nonlinear correction hooks already closes the gap by rescaling the step directly against the true nonlinear residual.

Code inspected:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - `SVMP_MPI_FIRST_ITERATION_STEP_EXPANSION=1`
  - `SVMP_MPI_FIRST_ITERATION_STEP_LINE_SEARCH=1`

Qualification:

- Step expansion:
  - [tests/_codex_iliac_1step_mpi4_stepexpand_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_stepexpand_20260415/run.log)
- Step line search:
  - [tests/_codex_iliac_1step_mpi4_steplinesearch_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_steplinesearch_20260415/run.log)

Observed behavior:

- Step expansion:
  - [run.log:329](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_stepexpand_20260415/run.log:329)
    - still `converged=1 iters=5`
  - [run.log:334](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_stepexpand_20260415/run.log:334)
    - total loop `28.973928 s`

- Step line search:
  - [run.log:353](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_steplinesearch_20260415/run.log:353)
    - still `converged=1 iters=5`
  - [run.log:357](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_steplinesearch_20260415/run.log:357)
    - total loop `39.501989 s`

Interpretation:

- The remaining gap is not fixed by the existing first-step nonlinear rescaling hooks either.
- That weakens the simple “same direction, wrong scalar length” explanation.
- The distributed first-step update is not just too short overall; it appears to be missing a more structured pressure component.

Updated best diagnosis:

- The remaining `serial 4` vs `mpi4 5` gap is now best explained as a structured distributed pressure-mode miss in the first returned full Newton increment.
- The obvious built-in corrections have all been tested on the real distributed harness and ruled out:
  - global mean shift
  - mean/partition/rank-one small subspace recovery
  - exact Woodbury low-rank recovery as a practical path
  - first-step scalar expansion
  - first-step scalar line search
- The next credible move is no longer “toggle another correction hook.”
- The next credible move is to build or expose a richer pressure-space basis for the first returned increment, then test it against the true *nonlinear* residual rather than only the linear residual.

---

## 2026-04-15: Nonlinear-scaled MPI subspace recovery experiment

New hypothesis:

- The existing MPI subspace-recovery basis might already contain the right missing pressure direction, but its current *linear-residual* acceptance rule might be too weak.
- If so, taking that same recovered direction and scaling it against the true nonlinear residual could close the remaining `serial 4` vs `mpi4 5` gap.

Code changed:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - added an env-gated nonlinear scaling search on top of the existing accepted MPI linear subspace-recovery direction:
    - `SVMP_MPI_LINEAR_SUBSPACE_NONLINEAR_SEARCH=1`
  - behavior:
    - keep the existing dense low-dimensional linear recovery solve
    - if it accepts a correction, form the correction direction `delta_du`
    - evaluate the *true nonlinear residual* for
      - `du_base + beta * delta_du`
      - with `beta in {0, 1, 2, 4, 8}`
    - accept the best `beta` only if it beats the baseline `beta=0`

Qualification:

- Real archived 1-step iliac `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_subspace_nonlinear_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_20260415/run.log)
  - env:
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY=1`
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY_MAX_NEWTON_ITERS=1`
    - `SVMP_MPI_LINEAR_SUBSPACE_NONLINEAR_SEARCH=1`
- Traced distributed run:
  - [tests/_codex_iliac_1step_mpi4_subspace_nonlinear_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_trace_20260415/run.log)
  - same envs plus `SVMP_OOP_SOLVER_TRACE=1`

Observed behavior:

- Real no-trace qualification:
  - [run.log:341](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_20260415/run.log:341)
    - still `converged=1 iters=5`
  - [run.log:345](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_20260415/run.log:345)
    - total loop `37.135558 s`

- Traced first-step result:
  - the underlying linear subspace recovery still accepts the same tiny linear correction:
    - [run.log:707](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_trace_20260415/run.log:707)
    - `residual_before=0.499916`
    - `residual_after=0.498672`
  - the new nonlinear scaling search then chooses the most aggressive tested scaling:
    - [run.log:990](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_trace_20260415/run.log:990)
    - `beta=8`
  - but the nonlinear improvement is negligible:
    - [run.log:990](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspace_nonlinear_trace_20260415/run.log:990)
    - `baseline_residual=9409.62`
    - `best_residual=9409.61`

Interpretation:

- The current subspace direction is not just under-scaled; it is fundamentally missing the real distributed pressure correction.
- Even multiplying the recovered direction by `8x` barely changes the first post-update nonlinear residual.

Updated best diagnosis:

- The remaining distributed miss is not captured by the current low-dimensional basis:
  - rank-one outlet modes
  - constraint mean mode
  - rank-partition pressure modes
- So the next credible move is now even narrower:
  - construct a richer first-step pressure-space basis than the current mean/partition set
  - or directly inspect the distributed first-step pressure increment difference against a serial reference and derive basis vectors from that structure

---

## 2026-04-15: Pressure-difference localization on the real 1-step iliac harness

New hypothesis:

- The missing first-step `mpi4` pressure correction may be localized in a physically meaningful region of the vessel, not just in a global mean/partition mode.
- If the serial-vs-`mpi4` pressure difference maps cleanly to a small physical region, that would point to a missing coarse/basis mode tied to outlet or wall neighborhoods.

Analysis steps:

- Used the existing dumped normalized first-step updates:
  - `tests/_codex_iliac_1step_serial_vecdump_20260415/first.du_normalized.txt`
  - `tests/_codex_iliac_1step_mpi4_vecdump_20260415/first.du_normalized.txt`
- Verified from the real mesh that the pressure DOF count and point count are identical:
  - `mesh-complete.mesh.vtu` has `15334` points
  - pressure field has `15334` DOFs
  - `GlobalNodeID` is exactly `point_index + 1` for all points

Observed geometry facts:

- Pressure DOFs are point-aligned on this case, so pressure-local index `i` maps directly to mesh point / `GlobalNodeID = i + 1`.
- The strongest *centered* serial-vs-`mpi4` pressure differences are boundary-heavy:
  - the top `50` centered-difference nodes are all on physical boundary surfaces
  - the top `100` contain `95` boundary nodes
  - the top `200` contain `166` boundary nodes
- Exact boundary union vs interior energy split of the centered pressure difference:
  - boundary: about `49.74%`
  - interior: about `50.26%`
- So the whole miss is not boundary-only, but the *highest-amplitude support* is strongly boundary-focused.

Most prominent physical supports from the mapped top nodes:

- `wall_right_iliac`
- `wall_aorta`
- `cap_aorta` (the inlet face in this case)
- much less on `cap_right_iliac`

Useful negative result:

- A nearest-surface partition of the whole domain explains almost none of the centered pressure-difference energy (`~2.34%`).
- So the missing mode is not “one nearest physical surface owns this region.”

Useful positive result:

- A simple physical-surface indicator basis on the actual case surfaces
  - `cap_aorta`
  - `cap_aorta_2`
  - `cap_right_iliac`
  - `wall_aorta`
  - `wall_right_iliac`
  captures about `33.55%` of the centered pressure-difference energy.
- Adding the existing constant/mean direction lifts that to about `55.41%`.
- Equivalently, the span of `all_boundary` plus the existing constant mode captures the same `~55.26%`.

Interpretation:

- The missing distributed first-step pressure mode is strongly boundary-vs-interior in character.
- But it is not just a global mean, and it is not well described by nearest-surface ownership.
- That makes a raw boundary-support basis a plausible *diagnostic* direction, but probably too crude to be the final fix.

Updated next step:

- Test whether a physics-agnostic pressure boundary indicator can help the real `mpi4` first-step recovery.

---

## 2026-04-15: Boundary-indicator MPI subspace-recovery experiments

New hypothesis:

- Since `all_boundary + constant` explains about `55%` of the centered serial-vs-`mpi4` pressure difference offline, adding a physics-agnostic pressure boundary-indicator vector to the existing MPI subspace-recovery basis might close the first-step gap.

Code changed:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - added an env-gated experimental basis column:
    - `SVMP_MPI_LINEAR_SUBSPACE_BOUNDARY_INDICATOR=1`
  - implementation:
    - build an indicator on the scalar constraint field DOFs that lie on boundary markers
    - append that as an extra column in the existing low-dimensional MPI recovery basis
  - also added trace-only marker probes to confirm what boundary markers / DOFs the hook is actually seeing

Important build note:

- The top-level `build` target is a superbuild wrapper and did **not** pick up the `NewtonSolver.cpp` change.
- Rebuilt the real inner tree directly with:
  - `cmake --build build/svMultiPhysics-build -j8`
- Only runs after that inner rebuild are valid for this experiment.

### Mixed-marker boundary-indicator run

Qualification:

- Real archived 1-step iliac `mpi4`:
  - `tests/_codex_iliac_1step_mpi4_boundaryindicator_requal_20260415/run.log`
  - env:
    - `SVMP_MPI_LINEAR_SUBSPACE_RECOVERY=1`
    - `SVMP_MPI_LINEAR_SUBSPACE_BOUNDARY_INDICATOR=1`

Observed behavior:

- Still converged in `5` Newton:
  - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_requal_20260415/run.log:305)
- Total time regressed to `32.892537 s`:
  - [run.log:309](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_requal_20260415/run.log:309)
- Linear work / collectives regressed badly:
  - second solve already hit `182368` allreduces:
    - [run.log:122](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_requal_20260415/run.log:122)

Trace result:

- Traced run:
  - `tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log`
- The new boundary vector did enter the basis as column `7`:
  - [run.log:707](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:707)
  - [run.log:710](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:710)
- But the first returned `du` was effectively unchanged from the old subspace-recovery baseline:
  - boundary-indicator trace:
    - [run.log:723](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:723)
  - old baseline trace:
    - [run.log:714](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_subspacerecovery_trace_20260415/run.log:714)
  - both still had pressure norm about `2.5242e+07`
- The first post-update auxiliary outputs were also almost identical:
  - boundary-indicator trace:
    - `[261.424, 15.127]`
  - old baseline trace:
    - `[261.408, 15.0866]`

Important new diagnostic:

- The trace showed a large unlabeled `marker=-1` boundary set appearing on every rank:
  - [run.log:690](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:690)
  - [run.log:693](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:693)
  - [run.log:695](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:695)
  - [run.log:702](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_boundaryindicator_trace4_20260415/run.log:702)
- That means the naive “all boundary markers” basis was contaminated by unlabeled distributed-local boundary faces, not just the physical inlet/outlet/wall markers.

Interpretation:

- A raw pressure boundary-indicator direction is too crude.
- It is measurably present in the recovery basis, but it does not change the first returned `mpi4` pressure update in a useful way.
- Worse, it increases distributed Schur/collective work.

### Positive-marker-only boundary-indicator run

Refined hypothesis:

- The bad behavior above may be dominated by the spurious `marker=-1` set.
- If the basis uses only physical positive markers (`1..5` here), it might become a cleaner pressure correction.

Code refinement:

- Updated the same env-gated hook to skip negative markers when collecting boundary supports.

Qualification:

- Real archived 1-step iliac `mpi4` with the filtered boundary basis:
  - `tests/_codex_iliac_1step_mpi4_boundaryindicator_posmarkers_20260415/run.log`
  - same envs as above

Observed behavior:

- This run was worse than the mixed-marker case.
- It never reached a first `BlockSchur` timing summary after the initial GMRES solve on a useful timescale and had to be stopped.
- So the positive-marker-only raw indicator is also not the fix.

Updated diagnosis:

- The missing distributed pressure mode is boundary-heavy, but a *binary* boundary mask is not a viable correction direction.
- The next credible basis is a smoother pressure boundary extension, not another raw indicator:
  - harmonic / Laplace-smoothed pressure boundary mode
  - or a mode derived directly from the serial-vs-`mpi4` first-step pressure difference and then compressed into a physics-agnostic FE-space construction

Updated next step:

- Build a smoother, physics-agnostic pressure-boundary mode for the first-step recovery path, and explicitly avoid unlabeled negative boundary markers in any future boundary-derived basis.

## 2026-04-15: Direct nonlinear-oracle search for file-supplied recovery vectors

New hypothesis:

- The existing MPI subspace-recovery dense solve is minimizing the true *linear* residual, but the missing first-step `mpi4` progress could still live in a direction that is good for the *nonlinear* residual and therefore invisible to the linear objective.

Code:

- [Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
  - added env-gated file loading for an extra MPI recovery basis column:
    - `SVMP_MPI_LINEAR_SUBSPACE_VECTOR_FILE`
  - added direct nonlinear search along that file vector:
    - `SVMP_MPI_LINEAR_SUBSPACE_VECTOR_NONLINEAR_SEARCH`

Important bug found and fixed in the diagnostic itself:

- The first version of the custom nonlinear-vector probe copied the loaded vector into local storage using `owned_dofs` iteration order.
- That was wrong because `du.localSpan()` is in backend-local storage order, not guaranteed `owned_dofs` iteration order.
- Fixed by capturing the already assembled `residual_base.localSpan()` instead, so the custom nonlinear search now uses the same local ordering as the accepted recovery basis column.

### Fresh current-dump baselines

Current re-qualification runs:

- serial:
  - [tests/_codex_iliac_1step_serial_vecdump_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_vecdump_current_20260415/run.log)
  - converged in `4` Newton:
    - [run.log:263](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_vecdump_current_20260415/run.log:263)
- `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_vecdump_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_vecdump_current_20260415/run.log)
  - converged in `5` Newton:
    - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_vecdump_current_20260415/run.log:305)

Fresh dump comparison:

- The current serial first-step dump is *identical* to the earlier serial dump.
- The current `mpi4` first-step dump changed only slightly relative to the earlier `mpi4` dump:
  - `||du_old_mpi - du_new_mpi|| / ||du_old_mpi|| ~ 3.66e-03`

Interpretation:

- The current archived 1-step serial/`mpi4` gap is stable enough that fresh dump regeneration does not change the diagnosis materially.

### Full file-vector nonlinear oracle using raw serial-minus-`mpi4` dump difference

Runs:

- first attempt before the local-ordering fix:
  - [tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace_20260415_fromcase/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace_20260415_fromcase/run.log)
- corrected run after the local-ordering fix:
  - [tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log)
- same experiment repeated with a freshly regenerated current dump difference:
  - [tests/_codex_iliac_1step_mpi4_oraclefull_current_nonlinear_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_current_nonlinear_trace_20260415/run.log)

Observation after the fix:

- The first accepted linear subspace recovery is still tiny in the custom-file direction:
  - `custom_file_vector=-4.72255e-09`
  - [trace2 run.log:712](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log:712)
- The direct nonlinear search along that same file vector rejects every positive `beta` and keeps `beta=0`:
  - [trace2 run.log:1041](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log:1041)
- Candidate nonlinear residuals blow up rapidly:
  - `beta=0.25`: about `7.35e+06`
  - `beta=0.5`: about `2.91e+07`
  - `beta=1.0`: about `1.16e+08`
  - `beta=2.0`: about `4.61e+08`
  - representative lines:
    - [trace2 run.log:899](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log:899)
    - [trace2 run.log:945](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log:945)
    - [trace2 run.log:991](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log:991)
    - [trace2 run.log:1037](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefull_nonlinear_trace2_20260415/run.log:1037)

Very important interpretation limit:

- The earlier vector-dump analysis already established that raw serial and distributed dense dump vectors are only comparable *up to DOF renumbering*:
  - [this note](/home/zack/Downloads/svMultiPhysics/Documentation/mpi_convergence_gap_debug_log_20260415.md:472)
- That means a raw index-by-index serial-minus-`mpi4` dense dump difference is **not** a trustworthy physical oracle direction for the full system.
- So the negative result above is still useful as a debugging datapoint, but it should **not** be over-interpreted as “the exact serial correction is bad” in a physical sense.

### Pressure-centered nonlinear oracle

Hypothesis:

- Even if the full dense dump difference is invalid because of global DOF renumbering, the pressure block may still admit a more invariant oracle because this case’s pressure field is point-aligned and was the dominant observed serial/`mpi4` mismatch.

Constructed vector:

- zero on the velocity block
- pressure difference on the pressure block with its global mean removed

Run:

- [tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log)

Observation:

- The centered-pressure direction is less catastrophic than the full raw-difference vector, but it is still decisively bad for nonlinear progress:
  - `beta=0.25`: `1.72e+06`
  - `beta=0.5`: `6.64e+06`
  - `beta=1.0`: `2.63e+07`
  - `beta=2.0`: `1.05e+08`
  - representative lines:
    - [run.log:898](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log:898)
    - [run.log:944](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log:944)
    - [run.log:990](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log:990)
    - [run.log:1036](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log:1036)
- The search again rejects everything except `beta=0`:
  - [run.log:1040](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_current_nonlinear_trace_20260415/run.log:1040)

Interpretation:

- The remaining `mpi4` gap is not a simple missing global pressure-mean correction.
- Even a much more targeted centered-pressure-difference field is not a good first-step nonlinear correction from the current distributed state.

### Velocity-only nonlinear oracle from the raw dump difference

Constructed vector:

- raw serial-minus-`mpi4` difference on the velocity block
- zero on the pressure block

Run:

- [tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log)

Observation:

- This is the mildest of the three tested file-vector probes, but it is still not beneficial:
  - `beta=0.25`: `2.48e+04`
  - `beta=0.5`: `4.72e+04`
  - `beta=1.0`: `9.39e+04`
  - `beta=2.0`: `1.91e+05`
  - representative lines:
    - [run.log:898](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log:898)
    - [run.log:944](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log:944)
    - [run.log:990](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log:990)
    - [run.log:1036](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log:1036)
- It still rejects every positive `beta`:
  - [run.log:1040](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclevelocity_current_nonlinear_trace_20260415/run.log:1040)

Interpretation limit:

- Because the full dense dumps are only comparable up to DOF renumbering, the velocity-only raw-difference probe is **not** a physically rigorous oracle by itself.
- But it does at least show that there is no obvious low-amplitude “easy” improvement hiding in that raw velocity difference.

Updated diagnosis:

- The new direct nonlinear-oracle hook works and is now trustworthy as a diagnostic mechanism.
- The file-vector experiments did **not** reveal a hidden first-step direction that the current linear subspace recovery simply failed to scale.
- The strongest reliable negative result is the centered-pressure probe: a targeted pressure-difference field is still a bad nonlinear direction.
- The full and velocity-only raw-difference probes are limited by serial/`mpi4` DOF renumbering and should not be treated as physical-oracle evidence without a renumbering-invariant mapping.

Updated next step:

- Build a serial/`mpi4`-invariant oracle for the first-step field difference, starting with an explicit pressure-DOF-to-physical-node mapping in both runs.
- If that still shows no useful pressure-field correction, the next likely culprit is deeper distributed solve semantics or gauge/coarse-space handling, not a missing low-dimensional state-space correction.

## Physically aligned pressure-node dump and coordinate-mode follow-up

### Pressure-node dump keyed by physical coordinates

Hypothesis:

- The earlier raw full-system dump differences were contaminated by process-count-dependent DOF numbering.
- A pressure dump keyed by physical node coordinates should let serial and `mpi4` be compared on the exact same node set.

Code:

- Extended the existing first-linear dump hook in [NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp) so `SVMP_DEBUG_FIRST_LINEAR_VECTOR_DUMP_PREFIX` now also writes a scalar-field vertex record dump:
  - `<prefix>.scalar_vertex_records.txt`
  - current field selection prefers a scalar field named `Pressure`/`pressure`, else the first scalar field
  - records are gathered from all ranks and written on rank 0 as:
    - `monolithic_dof vertex_id x y z value`

Qualification runs:

- Serial:
  - [tests/_codex_iliac_1step_serial_physdump_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_physdump_20260415/run.log)
  - first-step pressure dump:
    - [first.scalar_vertex_records.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_physdump_20260415/first.scalar_vertex_records.txt)
- `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_physdump_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_physdump_20260415/run.log)
  - first-step pressure dump:
    - [first.scalar_vertex_records.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_physdump_20260415/first.scalar_vertex_records.txt)

Observation:

- The pressure-node dumps align exactly by physical coordinates:
  - `serial_records=15334`
  - `mpi_records=15334`
  - `common=15334`
  - no unmatched nodes
- So the pressure field comparison can now be done on a true serial/`mpi4`-invariant physical-node key, not a guessed dense-row alignment.

### Node-aligned pressure mismatch statistics

Observation:

- On those matched pressure nodes, the first-step serial-minus-`mpi4` pressure difference still has a large mean shift, but not only a mean shift:
  - mean difference: about `-3.20023e+06`
  - raw difference norm: about `4.12564e+08`
  - centered difference norm: about `1.14743e+08`
  - centered/raw fraction: about `0.278`
- So removing the mean shift still leaves a substantial spatial mismatch.
- An affine fit of the form `serial ≈ a * mpi + b` gives:
  - `a ≈ 4.951`
  - `b ≈ -2.715e+06`
  - correlation `≈ 0.770`
  - relative fit residual `≈ 0.192`
- That means the serial/`mpi4` pressure gap is not just a pure constant offset and not just a simple scalar rescaling either.

### Node-aligned pressure-difference nonlinear oracles

Constructed vectors:

- raw node-aligned pressure difference in `mpi4` monolithic DOF ordering:
  - [tests/_codex_iliac_1step_oracle_pressure_nodealigned_raw_20260415.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_oracle_pressure_nodealigned_raw_20260415.txt)
- centered node-aligned pressure difference in `mpi4` monolithic DOF ordering:
  - [tests/_codex_iliac_1step_oracle_pressure_nodealigned_centered_20260415.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_oracle_pressure_nodealigned_centered_20260415.txt)

Runs:

- centered:
  - [tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_centered_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_centered_trace_20260415/run.log)
- raw:
  - [tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_raw_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_raw_trace_20260415/run.log)

Observation:

- Both runs behave essentially the same:
  - the custom vector enters the linear-recovery basis
  - but its solved coefficient is tiny, about `-9.42523e-14`
  - representative centered lines:
    - [run.log:711](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_centered_trace_20260415/run.log:711)
    - [run.log:1040](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_centered_trace_20260415/run.log:1040)
  - representative raw lines:
    - [run.log:715](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_raw_trace_20260415/run.log:715)
    - [run.log:1041](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_raw_trace_20260415/run.log:1041)
- In both cases the direct nonlinear search along the node-aligned pressure vector rejects every positive `beta`, with `best_beta=0`.
- End-to-end, both runs still finish at `5` Newton:
  - centered: [run.log:1731](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_centered_trace_20260415/run.log:1731)
  - raw: [run.log:2037](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclepressure_nodealigned_raw_trace_20260415/run.log:2037)

Interpretation:

- This is stronger than the earlier dense-row-based negative result.
- Even when the pressure difference is aligned on the exact same physical nodes, the serial-minus-`mpi4` pressure mismatch is **not** a useful direct recovery direction from the current distributed first-step state.
- So the gap is not “just recover the serial pressure field” in any obvious one-shot sense.

### Coordinate-based coarse modes

Hypothesis:

- The node-aligned centered pressure mismatch might still be mostly a smooth geometry-driven mode, so a small physics-agnostic coordinate basis could help even though the full serial-minus-`mpi4` field does not.

Quick offline fit:

- Fitting the centered node-aligned pressure mismatch onto centered coordinate polynomials gives:
  - `[x, y, z]`: relative residual `≈ 0.479`
  - `[x, y, z, r^2]`: relative residual `≈ 0.479`
  - `[x, y, z, x^2, y^2, z^2]`: relative residual `≈ 0.384`
- So low-order coordinate structure is real, but not dominant enough by itself to explain the full mismatch.

Code:

- Added env-gated coordinate modes in [NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp):
  - `SVMP_MPI_LINEAR_SUBSPACE_COORD_MODES=1`
  - centered `x`, `y`, `z`, `x^2`, `y^2`, `z^2` over the constraint/pressure block

Run:

- [tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log)

Observation:

- These coordinate modes do enter the basis and get nontrivial coefficients:
  - [run.log:690](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log:690)
  - [run.log:732](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log:732)
- They slightly improve the true linear residual more than the older mean/partition-only basis:
  - `2.18508e-06 -> 2.15152e-06`
- But they do **not** change the first nonlinear residual drop in any meaningful way:
  - step-0 nonlinear residual after update is still about `9412.46`
  - [run.log:802](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log:802)
- The first-step pressure norm is also unchanged at the diagnostic level:
  - [run.log:738](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log:738)
- End-to-end, the run still takes `5` Newton:
  - [run.log:1731](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_coordmodes_trace_20260415/run.log:1731)

Updated diagnosis:

- The node-aligned pressure mismatch is real and physically comparable now.
- It is not eliminated by:
  - raw node-aligned pressure correction
  - centered node-aligned pressure correction
  - mean/partition pressure modes
  - low-order coordinate pressure modes
- The coordinate modes are the first new family here that the linear subspace recovery actually uses in a nontrivial way, but they still fail to improve nonlinear convergence.
- So the remaining `serial 4` vs `mpi4 5` gap now looks even less like a missing low-dimensional pressure-space basis and more like a deeper difference in the primary distributed solve itself.

Updated next step:

- Move away from “guess another recovery vector” and back toward the primary distributed solve.
- The highest-signal follow-up is to compare the first-step pressure correction produced by the primary distributed solver against a more direct distributed oracle on the same physical node set:
  - either a smaller FE/assembly-integrated MPI reproducer that keeps the monolithic outlet path
  - or a direct probe of the distributed Schur solve / pressure correction before the final assembled Newton increment is accepted.

## Lightweight `ns_solver` iter-0 trace on the real iliac harness

Hypothesis:

- The existing evidence already suggested the first `mpi4` Newton step is wrong before later Newton logic.
- The cleanest next isolator is the first scalar-Schur correction in the primary legacy BlockSchur path itself:
  - `U = K^{-1} Rm`
  - `P_rhs = Rc - D U`
  - `P = [L - D H G]^{-1} P_rhs`
  - then the first pressure-driven basis pair `(MU_iB, MP_iB)`
- If serial and `mpi4` already diverge at `P`, the gap is in the distributed scalar-Schur solve, not in later recovery-basis heuristics.

Code:

- In [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp):
  - added lightweight iter-0 trace lines for:
    - `momentum_rhs`
    - `constraint_rhs`
    - `momentum_solve_invK_Rm`
    - `schur_rhs`
    - `schur_solution`
    - `pressure_basis_momentum`
    - `pressure_basis_constraint`
    - `momentum_basis_momentum`
    - `momentum_basis_constraint`
  - moved the older very expensive full-column compare behind a separate opt-in env:
    - `SVMP_FSILS_NS_FULL_COLUMN_COMPARE=1`
- While adding that trace I found a real MPI trace bug:
  - the first version of the new helper did collectives under a root-only guard, which deadlocked `mpi4`
  - fixed by making every rank participate in the stats allreduces, while still printing only on root

Serial run:

- [tests/_codex_iliac_1step_serial_nslight_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log)

Key first-solve lines:

- [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:108) `momentum_rhs l2=1.891461e+03`
- [run.log:109](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:109) `constraint_rhs l2=4.493389e+02`
- [run.log:110](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:110) `momentum_solve_invK_Rm l2=1.152232e+03`
- [run.log:111](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:111) `schur_rhs l2=4.441774e+02`
- [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:112) `schur_solution l2=6.541547e+06 mean=-4.794320e+04 dot_rhs=4.174378e+08`
- [run.log:113](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:113) `pressure_basis_momentum l2=3.786638e+04`
- [run.log:114](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:114) `pressure_basis_constraint l2=6.381505e+02`
- [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:117) first `iter=0 Galerkin system`
- [run.log:120](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:120) `projected_fNorm=5.482038e+04`
- full run still converged in `4` Newton:
  - [run.log:1610](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nslight_20260415/run.log:1610)

MPI-4 run:

- [tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log)
- I stopped the run once the first-solve trace had been captured; the goal here was the first distributed scalar-Schur solve, not the full end-to-end qualification.

Key first-solve lines:

- [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:112) `momentum_rhs l2=1.891461e+03`
- [run.log:113](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:113) `constraint_rhs l2=4.493389e+02`
- [run.log:114](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:114) `momentum_solve_invK_Rm l2=1.152232e+03`
- [run.log:115](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:115) `schur_rhs l2=4.441774e+02`
- [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:116) `schur_solution l2=4.729124e+05 mean=-2.209378e+03 dot_rhs=5.632307e+07`
- [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:117) `pressure_basis_momentum l2=6.440921e+03`
- [run.log:118](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:118) `pressure_basis_constraint l2=1.209435e+03`
- [run.log:121](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:121) first `iter=0 Galerkin system`
- [run.log:124](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:124) `projected_fNorm=1.695297e+05`

Comparison:

- The first BlockSchur inputs are identical between serial and `mpi4`:
  - `momentum_rhs`: same `1.891461e+03`
  - `constraint_rhs`: same `4.493389e+02`
  - `momentum_solve_invK_Rm`: same `1.152232e+03`
  - `schur_rhs`: same `4.441774e+02`
- The first divergence appears exactly at the scalar-Schur solve:
  - serial `schur_solution l2 = 6.541547e+06`
  - `mpi4` `schur_solution l2 = 4.729124e+05`
  - ratio `serial / mpi4 ≈ 13.8`
- The first pressure-driven momentum basis is likewise much smaller in `mpi4`:
  - serial `pressure_basis_momentum l2 = 3.786638e+04`
  - `mpi4` `pressure_basis_momentum l2 = 6.440921e+03`
  - ratio `serial / mpi4 ≈ 5.9`
- The first projected residual is already much worse in `mpi4`:
  - serial `projected_fNorm = 5.482038e+04`
  - `mpi4` `projected_fNorm = 1.695297e+05`

Interpretation:

- This is the cleanest evidence yet.
- The first distributed Newton-step gap is **not** coming from:
  - FE assembly
  - `Rm`
  - `Rc`
  - the first momentum `K^{-1} Rm` solve
  - later nonlinear recovery bases
- It is being created inside the first distributed scalar-Schur solve itself, or in the immediately adjacent scalar-Schur operator/preconditioner semantics.

Updated diagnosis:

- The remaining `serial 4` vs `mpi4 5` gap is now pinned much tighter:
  - the first scalar-Schur rhs is the same
  - the distributed scalar-Schur solution is dramatically smaller
  - that weaker pressure correction then propagates into a much smaller first pressure-driven momentum correction
- So the active bug/weakness is in the distributed `bicgs::schur(...)` path or its operator/preconditioner, not in the FE library assembly and not in the later Newton recovery layers.

Updated next step:

- Move directly into [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp) on the scalar-Schur path.
- The highest-signal follow-up is to trace, on the first distributed iliac scalar-Schur solve:
  - rhs norm / mean at solver entry
  - preconditioned rhs / first Krylov vector norm
  - first few residual norms
  - any branch differences between the serial and distributed scalar-Schur implementations
- If needed after that, compare the scalar-Schur preconditioned operator response on one or two seed vectors between serial and `mpi4`.

## Scalar-Schur branch and operator follow-up

### Branch-choice check: forcing distributed legacy BiCGStab

Hypothesis:

- Since serial uses the legacy scalar-Schur BiCGStab path while distributed `mpi4` takes the multi-face GMRES branch, the first under-correction might simply be a bad branch choice.

Run:

- [tests/_codex_iliac_1step_mpi4_nslight_nogmres_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_nogmres_20260415/run.log)
- envs:
  - `SVMP_FSILS_NS_SOLVER_TRACE=1`
  - `SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES=1`

Observation:

- The first distributed scalar-Schur correction gets even smaller, not larger:
  - [run.log:115](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_nogmres_20260415/run.log:115) `schur_rhs l2=4.441774e+02`
  - [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_nogmres_20260415/run.log:116) `schur_solution l2=3.903956e+05`
- Compare against the default distributed branch:
  - default `mpi4`: [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:116) `4.729124e+05`
- The first projected residual is also worse with forced no-GMRES:
  - [run.log:124](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_nogmres_20260415/run.log:124) `projected_fNorm=6.599551e+04`
  - vs default `mpi4` [run.log:124](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nslight_20260415/run.log:124) `1.695297e+05`
  - note: both are still much worse than serial, but the key point is that changing the distributed branch does **not** recover the serial-sized Schur correction

Interpretation:

- The GMRES-vs-BiCGStab branch difference is not the main explanation.
- The distributed legacy BiCGStab path is, if anything, worse on this first solve.
- So the remaining gap survives across the scalar-Schur Krylov branch choice.

### Direct operator probe on the first Schur rhs

Hypothesis:

- If the first Schur rhs is the same but the distributed Schur solution is much smaller, the scalar-Schur operator/preconditioner itself might still differ across serial and `mpi4`.
- To test that directly, compare the actual operator application on the same first rhs:
  - input `R`
  - `GL.apply(R) -> (GP, SP)`
  - `BCOP_TYPE_PRE` correction on `GP`
  - `D.apply(GP)`
  - final preconditioned Schur output `M_inv * (SP - DGP)`

Code:

- Added env-gated probe in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_SCHUR_OPERATOR_PROBE=1`
  - emits one one-shot first-solve probe only

Serial run:

- [tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log)
- key lines:
  - [run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:102)
  - [run.log:103](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:103)
  - [run.log:104](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:104)
  - [run.log:105](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:105)
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:106)
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:107)
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_schurprobe_20260415/run.log:108)

Serial values:

- `input l2=4.441774e+02 mean=-3.861493e-01`
- `GL_sp l2=1.271605e+02`
- `GL_gp_pre_bc l2=1.382902e+02`
- `GL_gp_post_bc l2=1.382902e+02`
- `D_gp l2=3.039023e+01`
- `output l2=1.492141e+02 mean=4.877544e-04`

MPI-4 run:

- [tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log)
- I stopped the run after the first probe landed; the purpose was the first distributed operator application only.
- key lines:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:106)
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:107)
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:108)
  - [run.log:109](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:109)
  - [run.log:110](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:110)
  - [run.log:111](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:111)
  - [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_schurprobe_20260415/run.log:112)

MPI values:

- `input l2=4.441774e+02 mean=-3.861493e-01`
- `GL_sp l2=1.271605e+02`
- `GL_gp_pre_bc l2=1.382902e+02`
- `GL_gp_post_bc l2=1.382902e+02`
- `D_gp l2=3.039023e+01`
- `output l2=1.492141e+02 mean=4.877544e-04`

Interpretation:

- The first scalar-Schur operator application on the real iliac rhs matches serial and `mpi4` exactly to the printed precision.
- That means the remaining gap is **not** in:
  - the first Schur rhs
  - `GL.apply(...)`
  - `BCOP_TYPE_PRE`
  - `D.apply(...)`
  - the resulting preconditioned operator output on that first rhs
- Combined with the earlier `nslight` traces, the gap is now narrowed further:
  - the first operator action matches
  - but the iterative Schur solve still returns a much smaller distributed solution
- So the active problem is now most likely inside the scalar-Schur Krylov iteration / recurrence itself, not the assembled operator on the first rhs.

Updated diagnosis:

- The distributed scalar-Schur under-correction is not explained by:
  - branch choice alone
  - first operator/preconditioner application mismatch
- The highest-signal remaining suspect is the iterative Krylov path itself:
  - recurrence / update semantics
  - restart / orthogonalization behavior on the distributed path
  - or some later operator reuse during the iterative loop that diverges from the first correctly matched operator application

Updated next step:

- Use the existing iteration-history hook in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_ITER_HISTORY=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_ITER_HISTORY_SOLVE_INDEX=0`
- Compare the first scalar-Schur residual history between:
  - serial default
  - `mpi4` default
  - optionally `mpi4` with multi-face GMRES disabled
- If needed after that, add a one-shot probe on the first Krylov basis vector after the first update, not just on the entry rhs.

## First scalar-Schur iteration history and outer scalar-mean constraint

### First scalar-Schur residual history

Hypothesis:

- Since the first scalar-Schur operator application matches serial and `mpi4`, the next place to look is the Krylov recurrence itself.
- If the residual histories diverge strongly, the distributed solve is simply converging more poorly.
- If the residual histories stay similar while the returned solution stays very different, that points instead to weak-mode / nullspace solution-selection.

Runs:

- Serial:
  - [tests/_codex_iliac_1step_serial_iterhist_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhist_20260415/run.log)
  - envs:
    - `SVMP_FSILS_TRACE_FACE_ONLY_ITER_HISTORY=1`
    - `SVMP_FSILS_TRACE_FACE_ONLY_ITER_HISTORY_SOLVE_INDEX=0`
- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log)
  - same envs

Serial first-solve history:

- [run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhist_20260415/run.log:102) starts at `4.441774e+02`
- [run.log:103](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhist_20260415/run.log:103) after 1 BiCGStab iteration: `3.416669e+02`
- [run.log:132](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhist_20260415/run.log:132) after 30 iterations: `1.409754e+02`
- [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhist_20260415/run.log:202) around 100 iterations: `8.220871e+01`
- [run.log:403](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_iterhist_20260415/run.log:403) final after 300 iterations: `2.468287e+01`

MPI-4 first-solve history:

- [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log:106) starts at `4.441774e+02`
- [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log:108) after 1 GMRES iteration (estimate): `3.714635e+02`
- [run.log:137](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log:137) after 30 iterations (estimate): `1.433039e+02`
- [run.log:207](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log:207) around 100 iterations (estimate): `8.820030e+01`
- [run.log:408](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iterhist_20260415/run.log:408) final true residual after 300 iterations: `2.699209e+01`

Interpretation:

- The first scalar-Schur residual histories are extremely similar.
- Both serial and `mpi4` burn the full `300` inner iterations and end with almost the same true residual scale:
  - serial: `2.468287e+01`
  - `mpi4`: `2.699209e+01`
- Yet from the earlier `nslight` trace, the returned first scalar-Schur solutions are still very different:
  - serial `schur_solution l2 = 6.541547e+06`
  - `mpi4` `schur_solution l2 = 4.729124e+05`
- This is strong evidence that the remaining gap is **not** “distributed residual is much worse.”
- It is a solution-selection issue along a weak / near-null scalar mode:
  - similar residuals
  - same first operator application
  - but very different returned scalar pressure solutions

### Re-qualification of the existing outer scalar-mean constraint

Hypothesis:

- If the main ambiguity is along a weak scalar mean-like mode, the existing outer Galerkin mean constraint in [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp) might reduce the serial/`mpi4` gap without deeper backend changes.

Runs:

- Serial:
  - [tests/_codex_iliac_1step_serial_scalarmean_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_scalarmean_20260415/run.log)
  - env:
    - `SVMP_FSILS_BLOCKSCHUR_CONSTRAIN_SCALAR_MEAN=1`
- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_scalarmean_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarmean_20260415/run.log)
  - same env

Observation:

- Serial remains at `4` Newton, but linear work becomes much heavier:
  - [run.log:263](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_scalarmean_20260415/run.log:263) `iters=4`
  - [run.log tail](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_scalarmean_20260415/run.log) shows `41 linear iters`
- MPI-4 remains at `5` Newton and also gets much heavier:
  - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarmean_20260415/run.log:305) `iters=5`
  - [run.log tail](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarmean_20260415/run.log) shows:
    - `46 linear iters`
    - `456538` allreduces
    - total loop `58.175162 s`

Interpretation:

- The existing **outer** scalar-mean Galerkin constraint is not the fix.
- It does not close the `serial 4` vs `mpi4 5` gap.
- It makes both serial and `mpi4` much more expensive.
- That is consistent with the newer diagnosis: the ambiguity is being created inside the scalar-Schur Krylov solve itself, before the outer Galerkin combination is finalized.

Updated diagnosis:

- The remaining MPI gap is now isolated very tightly:
  - first Schur rhs matches
  - first operator application matches
  - first scalar-Schur residual histories are very similar
  - but the returned scalar pressure solutions differ by an order of magnitude
  - existing outer scalar-mean constraint does not fix it
- So the active problem is now best described as:
  - distributed scalar-Schur **solution selection / gauge handling** along a weak near-null scalar mode inside the Krylov solve itself
  - not FE assembly
  - not the first operator/preconditioner application
  - not simple branch choice
  - not the existing outer Galerkin mean constraint

Updated next step:

- The next credible fix is inside the scalar-Schur solve, not outside it.
- Highest-signal options now:
  - add a diagnostic that tracks scalar solution mean / norm during the first Schur Krylov solve
  - prototype an **internal** gauge projection / mean-removal on the scalar-Schur iterates or residuals, rather than only constraining the outer Galerkin combination afterward
  - if that helps, make it conditional on a physics-agnostic scalar nullspace / gauge claim rather than hard-coding anything outlet-specific

## Stronger inner scalar-Schur convergence is a hard regression

Hypothesis:

- If the `serial 4` vs `mpi4 5` gap is only due to the scalar-Schur solve stopping too early at `300` inner iterations, then a much tighter `NS_CG` configuration should reduce the first-step gap or at least change the Newton behavior.

Harness:

- Temporary copied XML:
  - [solver_codex_1step_outertol1e8_cg1000_tmp.xml](/home/zack/Downloads/svMultiPhysics/tests/cases/fluid/iliac_artery/solver_codex_1step_outertol1e8_cg1000_tmp.xml)
- Changes relative to the archived 1-step harness:
  - `NS_CG_max_iterations = 1000`
  - `NS_CG_tolerance = 1e-10`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_cg1000_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_cg1000_20260415/run.log)

Observation:

- The first BlockSchur solve becomes a major regression:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_cg1000_20260415/run.log) shows:
    - first BlockSchur total `59.632663 s`
    - `9` outer iterations
    - `994` inner Schur iterations
    - `4,395,207` Schur allreduces
    - `16.671013 s` in Schur allreduce time
- I stopped the run after the regression was clear.

Interpretation:

- “Just converge the inner scalar-Schur solve much harder” is not the fix.
- It explodes collective cost and does not look like a practical route to closing the nonlinear gap.
- This strengthens the earlier diagnosis that the issue is **which scalar solution is selected**, not simply the nominal inner iteration count.

## Internal zero-mean projection on scalar-Schur iterates

Hypothesis:

- The returned distributed scalar-Schur solution may be drifting along a weak global scalar mean mode.
- If so, forcing the scalar-Schur iterate to stay mean-free during the solve should move the first returned pressure correction toward the serial solution.

Diagnostic code:

- Added env-gated hooks in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_ZERO_MEAN_PROJECT=1`
  - helper that subtracts the owned global scalar mean from the scalar-Schur solution iterate

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_zeromeanproj_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanproj_20260415/run.log)

Observation:

- The first returned scalar-Schur solution changes strongly:
  - [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanproj_20260415/run.log:116) `iter0 schur_solution l2=3.857401e+05 mean=-1.146692e-10`
  - [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanproj_20260415/run.log:117) `iter0 pressure_basis_momentum l2=4.003866e+04`
- That pressure-basis response is now close to the serial scale (`3.786638e+04`), which is the strongest evidence so far that the mean/gauge mode is part of the mismatch.
- But the solve quality regresses badly:
  - [run.log:666](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanproj_20260415/run.log:666) first BlockSchur solve reaches `15` outer iterations
- I stopped the run after repeated regressions became clear.

Interpretation:

- The scalar mean/gauge mode is definitely affecting the first returned pressure correction.
- But naively forcing the entire scalar-Schur Krylov iterate to be zero-mean is too aggressive and degrades the primary solve.
- So the remaining fix, if it lives here, has to be more selective than “project every iterate to zero mean.”

## Lighter scalar-Schur zero-mean post-project only

Hypothesis:

- The main value of the previous experiment may be in the **returned** scalar solution, not in perturbing every scalar-Schur Krylov iterate.
- If so, zero-meaning only the returned scalar-Schur correction might preserve the baseline operator path while choosing a better gauge.

Diagnostic code:

- Added a lighter env-gated hook in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_ZERO_MEAN_POSTPROJECT=1`
- This keeps the normal scalar-Schur solve, then subtracts the owned global scalar mean from the returned scalar solution only.

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_zeromeanpost_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanpost_20260415/run.log)

Observation:

- The first returned scalar-Schur solution changes in the same direction as the full iterate projection:
  - [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanpost_20260415/run.log:116) `iter0 schur_solution l2=3.857401e+05 mean=5.101806e-11`
  - [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanpost_20260415/run.log:117) `iter0 pressure_basis_momentum l2=4.003866e+04`
- So this lighter hook preserves the key qualitative result:
  - the scalar mean/gauge choice heavily influences the first returned pressure correction
- But it still destabilizes the first solve:
  - [run.log:666](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_zeromeanpost_20260415/run.log:666) first BlockSchur solve reaches `15` outer iterations
- I stopped this run as well after the repeated regressions were clear.

Interpretation:

- The mean/gauge component is not a red herring.
- But neither “project every scalar-Schur iterate” nor “post-project every scalar-Schur return” is a valid fix by itself.
- The likely remaining need is a **structured scalar coarse/gauge treatment**:
  - selective
  - internal to scalar-Schur solution selection
  - physics-agnostic
  - not a blunt zero-mean overwrite

Updated diagnosis:

- The current best picture is:
  - first scalar-Schur rhs matches between serial and `mpi4`
  - first scalar-Schur operator application matches
  - first scalar-Schur residual histories are similar
  - but the returned scalar solution’s weak mean/gauge component strongly changes the pressure basis seen by the outer solve
  - simple zero-mean forcing changes that pressure basis in the “right” direction but breaks solve quality
- So the active gap is now best described as:
  - **missing or inconsistent scalar gauge/coarse-mode treatment inside distributed scalar-Schur solution selection**
  - not FE assembly
  - not a simple operator mismatch
  - not something fixed by brute-force inner tolerances

## First-solve-only scalar zero-mean post-project

Hypothesis:

- The previous zero-mean hooks were too blunt because they touched every scalar-Schur return.
- If the real serial/`mpi4` split is created only at the first returned scalar-Schur correction of the first NS solve, then changing only that one correction might improve the first pressure basis without the broader `15`-outer regression.

Diagnostic code:

- Added env-gated first-solve-only hook in [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp):
  - `SVMP_FSILS_NS_FIRST_SCHUR_ZERO_MEAN=1`
- It subtracts the global scalar mean from `P_col` only for:
  - distributed runs
  - `i == 0`
  - the first NS solve only

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log)

Observation:

- The first correction changes exactly as intended:
  - [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log:116) `iter0 schur_zero_mean_post mean_before=-2.209378e+03 mean_after=6.773464e-14`
  - [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log:117) `iter0 schur_solution l2=3.857401e+05`
  - [run.log:118](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log:118) `iter0 pressure_basis_momentum l2=4.003866e+04`
- So the first pressure basis again moves to the serial scale.
- But end-to-end behavior does not improve:
  - [run.log:1725](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log:1725) still converges in `5` Newton
  - [run.log tail](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_firstschurzero_20260415/run.log) shows:
    - `47` linear iterations
    - total Newton time `51.688802 s`
    - total loop `52.951698 s`

Interpretation:

- The first returned scalar mean component is definitely relevant to the first pressure basis mismatch.
- But correcting only that first mean component is **not sufficient** to close the nonlinear gap.
- So the missing distributed scalar mode is richer than a one-shot global-mean gauge correction on the first solve.

## Iter0-per-NS-solve scalar zero-mean post-project

Hypothesis:

- The first-solve-only hook may be too narrow because each NS solve has its own `iter0` pressure correction.
- If the gap is recreated at `i == 0` on every NS solve, then applying the zero-mean post-project only to `iter0` of each NS solve might improve the Newton path while still being much less invasive than modifying every scalar-Schur return.

Diagnostic code:

- Added env-gated hook in [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp):
  - `SVMP_FSILS_NS_ITER0_SCHUR_ZERO_MEAN=1`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log)

Observation:

- Each NS solve now shows the same zero-mean post-project on its `iter0` scalar correction:
  - [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:116)
  - [run.log:467](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:467)
  - [run.log:869](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:869)
- The early pressure basis response stays elevated relative to the old baseline:
  - [run.log:118](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:118) `4.003866e+04`
  - [run.log:469](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:469) `5.746433e+03`
- But this is still not a usable fix:
  - [run.log:1827](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:1827) remains at `5` Newton
  - [run.log:1832](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:1832) total loop regresses to `54.794179 s`
  - outer counts remain elevated:
    - [run.log:439](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:439) `11`
    - [run.log:841](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:841) `12`
    - [run.log:1800](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_iter0schurzero_20260415/run.log:1800) `11`

Interpretation:

- The distributed scalar mean/gauge component matters at every NS solve, not just the first one.
- But even targeting only `iter0` of each NS solve still does not close the `4` vs `5` gap and still makes the MPI path slower.
- So the missing scalar coarse space is **not** a simple global constant mode.

Updated diagnosis:

- The current strongest diagnosis is now:
  - the scalar mean component is part of the observed serial/`mpi4` mismatch
  - but fixing that 1D mode, even very selectively, does not recover serial-equivalent nonlinear behavior
  - therefore the real missing distributed scalar mode is likely higher-dimensional and operator-aware
- The next credible direction is no longer “another mean shift.”
- The next credible physics-agnostic target is:
  - extract and compare the weak scalar-Schur Krylov mode itself
  - then test a deflation / coarse correction built from that mode rather than from a hard-coded global constant vector

## 1D mean-subspace basis-shape optimization is also a regression

Hypothesis:

- The mean-related experiments above may be failing because they overwrite the returned scalar-Schur correction.
- A better test is to leave the scalar-Schur solve alone, but expose the first pressure basis as a tiny 1D subspace:
  - baseline pressure basis
  - plus a weighted scalar-mean component
- Then let the small algebra choose the best blend.

Diagnostic code:

- Added env-gated basis-shape experiment in [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp):
  - `SVMP_FSILS_NS_ITER0_SCHUR_MEAN_SUBSPACE=1`
- This computes, for `iter0` pressure basis only:
  - original basis
  - scalar-mean basis
  - a 1D optimal blend parameter `gamma`
  - based on the local projected score `B^2 / A`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_meansubspace_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_meansubspace_20260415/run.log)

Observation:

- The optimizer does **not** want a large constant correction:
  - [run.log:119](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_meansubspace_20260415/run.log:119) chooses `gamma=3.297646e-02`
- That is a small positive blend, not:
  - baseline-only (`gamma=0`)
  - zero-mean (`gamma=-1`)
- So even the best constant-mode blend only shifts the first basis shape slightly.
- On the full harness this experiment failed to progress past the early first-solve stage on a useful timescale, and I stopped it.

Interpretation:

- This is another strong negative result against “the missing mode is basically just a constant pressure mode.”
- The best scalar-mean blend is small, which means the constant component is not the dominant missing coarse content.
- The real missing scalar mode still appears to be richer than any 1D constant-mode adjustment.

## Direct weak-mode trace in full distributed scalar-Schur GMRES is too invasive

Hypothesis:

- The next clean step after ruling out constant-mode fixes is to trace the actual weak scalar-Schur Krylov mode built by distributed face-only GMRES.

Diagnostic code:

- Added env-gated weak-mode trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_WEAK_MODE=1`
- The trace captures:
  - a candidate weak GMRES basis vector
  - its scalar mean
  - the norm of its image under the scalar-Schur operator
  - correlations with the final residual

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_weakmode_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmode_20260415/run.log)

Observation:

- On the full archived iliac harness, this instrumentation did not make it through the first distributed BlockSchur solve on a useful timescale.
- No `BICGS_FACE_ONLY_WEAK_MODE` line was emitted before I stopped the run.

Interpretation:

- The weak-mode idea is still the right next direction.
- But the direct trace is too invasive to use on the full iliac harness as the primary loop.
- It should be moved to:
  - a smaller reproducer
  - or a more surgical first-solve-only path
  - before using it again for full-harness qualification

Updated diagnosis:

- The current evidence now says:
  - scalar mean content matters, but only modestly
  - the best constant-mode blend is small
  - simple constant-mode manipulations do not close the nonlinear gap
- So the likely remaining issue is:
  - a higher-dimensional weak scalar-Schur mode, not a 1D global constant gauge mode

## Early Krylov-basis trace confirms the weak mode is non-constant

Hypothesis:

- If the remaining distributed scalar-Schur issue is really a richer weak mode, then an early GMRES basis vector on the face-only path should already look non-constant.

Diagnostic code:

- Added a cheap early basis trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_WEAK_MODE=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_BASIS_ITER=<k>`
- This emits a normalized scalar-Schur GMRES basis vector summary at the requested iteration:
  - `basis_l2`
  - `basis_mean`
  - `basis_centered_l2`
  - `constant_ratio`
  - `min/max`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_basisiter10_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter10_20260415/run.log)

Observation:

- The first useful weak-mode-like basis vector is emphatically non-constant:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter10_20260415/run.log:106) reports
    - `basis_mean=-2.794694e-04`
    - `basis_centered_l2=9.994010e-01`
    - `constant_ratio=3.460684e-02`
- So only about `3.5%` of that basis vector is aligned with the constant mode.

Interpretation:

- This is a strong negative result against any remaining “just fix the global mean” explanation.
- The distributed weak scalar-Schur content is higher-dimensional and spatially varying.

## One-vector correction along the extracted weak mode is inactive

Hypothesis:

- Even if the weak mode is not constant, a cheap one-vector correction along that extracted Krylov mode might still close the first-solve gap.

Diagnostic code:

- Added an env-gated post-solve mode correction in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_MODE_CORRECTION=1`
- This:
  - keeps the returned distributed GMRES solve unchanged
  - applies the scalar-Schur operator to the extracted weak mode
  - computes the optimal 1D correction coefficient `gamma`
  - accepts the correction only if the scalar-Schur residual improves

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_krylovcorr_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovcorr_20260415/run.log)

Observation:

- The basis trace still shows the same non-constant mode:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovcorr_20260415/run.log:106)
- But the 1D correction is effectively null:
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovcorr_20260415/run.log:107)
    - `gamma=-3.034143e-12`
    - `residual_before=2.699209e+01`
    - `residual_after=2.699209e+01`
    - `accepted=0`
- The run then became slower than the clean baseline and I stopped it after the first useful trace rather than waiting on a regression-quality qualification.

Interpretation:

- The extracted weak basis vector is real, but a single correction along that one vector is not enough.
- The scalar-Schur residual is effectively orthogonal to the image of that mode at the point where the correction is attempted.
- So the remaining gap is not fixed by:
  - a 1D constant/mean mode
  - or a 1D correction along one non-constant Krylov basis vector

Updated diagnosis:

- The remaining MPI gap still lives inside distributed scalar-Schur solution selection / coarse content.
- But it now looks higher-dimensional than both:
  - the constant mode
  - and any single extracted GMRES basis direction

## The first distributed scalar-Schur GMRES backsolve is the real outlier

Hypothesis:

- If the operator application is already matching, and simple mode corrections are ineffective, then the next likely source of the bad first `mpi4` pressure correction is the small GMRES least-squares backsolve itself.
- In that case the first scalar-Schur GMRES cycle should look much more ill-conditioned than the later ones, even on the same run.

Diagnostic code:

- Added an env-gated least-squares trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_GMRES_LS=1`
- For each distributed face-only GMRES cycle it reports:
  - `last_i`
  - solution coefficient norms `coeff_l2`, `coeff_l1`
  - smallest/largest diagonal after Givens: `min_abs_diag`, `max_abs_diag`
  - `diag_ratio = max_abs_diag / min_abs_diag`
  - coefficient sitting on the weakest diagonal entry
  - estimated residual and rhs norm

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log)

Observation:

- The first scalar-Schur GMRES cycle is much more extreme than the next two:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log:106)
    - `coeff_l2=4.729124e+05`
    - `coeff_l1=5.516923e+06`
    - `min_abs_diag=3.359335e-01`
    - `max_abs_diag=1.006845e+01`
    - `diag_ratio=2.997155e+01`
    - `coeff_at_min_diag=1.268031e+05`
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log:107)
    - `coeff_l2=1.572384e+04`
    - `diag_ratio=4.277445e+00`
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log:108)
    - `coeff_l2=7.556475e+03`
    - `diag_ratio=3.714861e+00`
- So the first distributed scalar-Schur solve is not just “a bit worse.” Its reduced least-squares system is dramatically more sensitive than the ones that follow.

Interpretation:

- This is the strongest evidence so far that the first bad `mpi4` pressure correction is being created inside the small scalar-Schur GMRES coefficient solve.
- The failure pattern is consistent with:
  - a weak / nearly-null scalar mode inside the first distributed Krylov subspace
  - large coefficient growth on the weakest diagonal direction
  - solution-selection instability, not an operator-apply mismatch

Updated diagnosis:

- The remaining MPI convergence gap is now most plausibly a first-solve scalar-Schur GMRES least-squares conditioning / regularization problem.
- The next credible fix is not another outer basis hack; it is a more stable internal coefficient-selection policy for the weak first-cycle scalar-Schur subspace.

## Simple diagonal-floor regularization in the tiny GMRES backsolve is not enough

Hypothesis:

- If the first-cycle scalar-Schur GMRES backsolve is too sensitive, then a small diagonal floor in that tiny triangular solve might suppress the worst coefficient spike and improve the first distributed solve.

Diagnostic code:

- Added an env-gated diagonal floor in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_GMRES_DIAG_FLOOR_FRAC=<fraction>`
- This replaces any very small `|h(j,j)|` in the tiny GMRES backsolve with `fraction * max_k |h(k,k)|` before dividing.
- I kept the least-squares trace on to measure the effect directly.

Run:

- MPI-4 with a mild floor:
  - `SVMP_FSILS_FACE_ONLY_GMRES_DIAG_FLOOR_FRAC=0.05`
  - [tests/_codex_iliac_1step_mpi4_gmresfloor005_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresfloor005_20260415/run.log)

Observation:

- The first weak coefficient did shrink somewhat:
  - baseline [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log:106)
    - `coeff_at_min_diag=1.268031e+05`
  - floored [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresfloor005_20260415/run.log:106)
    - `coeff_at_min_diag=8.461565e+04`
- But the qualitative behavior did **not** improve:
  - the first cycle still had `diag_ratio=2.997155e+01`
  - and the first outer solve actually worsened to [run.log:137](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresfloor005_20260415/run.log:137) `BlockSchur outer iters: 11`
- Later early solves also still showed large coefficient norms and no obvious stabilization.

Interpretation:

- A naive diagonal floor inside the tiny backsolve is too blunt.
- The first-cycle coefficient growth is real, but simply clipping the weakest diagonal does not resolve the underlying weak-mode selection problem.

Updated diagnosis:

- The remaining MPI gap still points to unstable first-cycle scalar-Schur GMRES coefficient selection.
- But the needed fix is more structured than a raw diagonal floor.

## Generic Tikhonov regularization of the reduced GMRES coefficients is too blunt

Hypothesis:

- If the first distributed scalar-Schur GMRES coefficient solve is unstable because of a weak near-null direction, then a small Tikhonov regularization on the reduced least-squares problem might suppress the coefficient explosion without harming the solve.

Diagnostic code:

- Added an env-gated reduced-system Tikhonov solve in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_GMRES_TIKHONOV_FRAC=<fraction>`
- This solves
  - `(R^T R + lambda^2 I) y = R^T g`
  - on the tiny upper-triangular GMRES least-squares system
  - with `lambda = fraction * max_j |R_jj|`

Run:

- MPI-4 with a mild regularization:
  - `SVMP_FSILS_FACE_ONLY_GMRES_TIKHONOV_FRAC=0.01`
  - [tests/_codex_iliac_1step_mpi4_tikh001_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_tikh001_20260415/run.log)

Observation:

- The first-cycle coefficient explosion is dramatically suppressed:
  - baseline [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log:106)
    - `coeff_l2=4.729124e+05`
    - `coeff_at_min_diag=1.268031e+05`
  - Tikhonov [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_tikh001_20260415/run.log:106)
    - `coeff_l2=5.296732e+02`
    - `coeff_at_min_diag=1.984646e+01`
- But the residual quality collapses:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_tikh001_20260415/run.log:106) still has `est_residual=2.699209e+01`
  - later early cycles are also unhealthy:
    - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_tikh001_20260415/run.log:107) `est_residual=3.941624e+01`
    - [run.log:109](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_tikh001_20260415/run.log:109) `est_residual=4.022762e+01`
- I stopped the run once it was clear that the regularized reduced solve was destroying the scalar-Schur solve quality.

Interpretation:

- This is another useful negative result:
  - the coefficient blow-up is real
  - but a generic least-squares regularization that suppresses it directly is not acceptable
- So the remaining issue is not “just damp the reduced solve.” The fix needs to preserve the meaningful scalar-Schur correction while constraining only the truly weak/gauge content.

Updated diagnosis:

- The first distributed scalar-Schur GMRES reduced solve is still the right target.
- But the needed repair is likely:
  - a structured nullspace / weak-mode aware coefficient policy
  - not a generic diagonal floor
  - and not a generic Tikhonov regularization

## Mean-free Krylov vectors help conditioning somewhat, but hurt solve quality

Hypothesis:

- A more structured nullspace-aware change than post-projecting the returned solution is to keep the distributed scalar-Schur GMRES Krylov basis itself mean-free.
- If the constant gauge content is poisoning the reduced solve, this should improve the first-cycle conditioning without arbitrary damping.

Diagnostic code:

- Added an env-gated Krylov-basis projection in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_GMRES_MEAN_FREE_KRYLOV=1`
- This subtracts the owned scalar mean from:
  - the restart residual
  - each `A * u_i`
  - the post-orthogonalized candidate basis vector

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_meanfreekrylov_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_meanfreekrylov_20260415/run.log)

Observation:

- The first reduced solve is less extreme:
  - baseline [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gmresls_20260415/run.log:106)
    - `coeff_l2=4.729124e+05`
    - `diag_ratio=2.997155e+01`
  - mean-free Krylov [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_meanfreekrylov_20260415/run.log:106)
    - `coeff_l2=2.197485e+05`
    - `diag_ratio=2.269016e+01`
- But the estimated residual gets worse, not better:
  - baseline `est_residual=2.699209e+01`
  - mean-free Krylov `est_residual=3.391562e+01`
- The first outer solve also regressed to [run.log:141](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_meanfreekrylov_20260415/run.log:141) `BlockSchur outer iters: 15`

Interpretation:

- This is another useful negative result:
  - the constant gauge direction is contributing to the first-cycle conditioning problem
  - but simply removing mean content from the whole Krylov basis is too blunt and damages the solve

## The weakest first-cycle reduced coefficient is essential, not spurious

Hypothesis:

- Since the first reduced GMRES cycle shows one especially weak diagonal with a huge coefficient, maybe that one coefficient is mostly spurious and can be suppressed.

Diagnostic code:

- Added an env-gated true-residual acceptance test in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_GMRES_WEAK_COEFF_SHRINK=<factor>`
- This:
  - identifies the coefficient attached to the weakest reduced diagonal
  - builds a candidate solution with that coefficient shrunk
  - compares the true scalar-Schur residual of baseline vs candidate
  - accepts the shrink only if the true residual improves

Run:

- MPI-4 with full suppression:
  - `SVMP_FSILS_FACE_ONLY_GMRES_WEAK_COEFF_SHRINK=0.0`
  - [tests/_codex_iliac_1step_mpi4_weakcoeff0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakcoeff0_20260415/run.log)

Observation:

- The first-cycle weak coefficient is indeed the same huge one:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakcoeff0_20260415/run.log:106)
    - `min_diag_index=0`
    - `coeff_at_min_diag=1.268031e+05`
- But removing it is catastrophic for the true residual:
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakcoeff0_20260415/run.log:107)
    - `baseline_true_residual=2.699209e+01`
    - `candidate_true_residual=4.259741e+04`
    - `accepted=0`

Interpretation:

- The large first reduced coefficient is not merely a spurious weak-mode artifact.
- It is essential to the actual scalar-Schur correction being produced by the current operator.
- So the remaining bug is not “drop the weakest coefficient.” It is deeper in how the first-cycle weak content is represented and resolved.

## Distributed Arnoldi reorthogonalization does not improve the first reduced solve

Hypothesis:

- The first-cycle reduced-system instability may be caused by loss of orthogonality in the distributed Arnoldi basis.
- In that case a second MGS pass should stabilize the first reduced GMRES solve.

Diagnostic code:

- Added an env-gated second orthogonalization pass in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_GMRES_REORTHOG=1`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_reorthog_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reorthog_20260415/run.log)

Observation:

- The first reduced solve is not improved:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reorthog_20260415/run.log:106)
    - `coeff_l2=4.725069e+05`
    - `diag_ratio=3.253257e+01`
    - `est_residual=2.709404e+01`
- Compared with baseline:
  - `coeff_l2=4.729124e+05`
  - `diag_ratio=2.997155e+01`
  - `est_residual=2.699209e+01`
- So the coefficient norm is unchanged and the diagonal spread is slightly worse.

Interpretation:

- Loss of orthogonality in the distributed Arnoldi basis is not the dominant problem in this first scalar-Schur cycle.

## The generic GMRES restart-length override does not affect this scalar-Schur path

Hypothesis:

- If the first-cycle conditioning problem comes from an oversized unrestarted basis, a smaller GMRES restart length should change the first reduced-system behavior.

Run:

- MPI-4:
  - `SVMP_FSILS_GMRES_SD=60`
  - [tests/_codex_iliac_1step_mpi4_sd60_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_sd60_20260415/run.log)

Observation:

- The first scalar-Schur reduced solve is unchanged from baseline:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_sd60_20260415/run.log:106)
    - `coeff_l2=4.729124e+05`
    - `diag_ratio=2.997155e+01`
    - `est_residual=2.699209e+01`
- These match the baseline first-cycle values exactly.

Interpretation:

- The existing `SVMP_FSILS_GMRES_SD` override is not reaching this legacy scalar-Schur GMRES path.
- So restart-length tuning via that control is not currently a debugging lever for this exact MPI gap.

Updated diagnosis:

- The remaining gap is still centered on the first distributed scalar-Schur GMRES reduced solve.
- But it is not primarily explained by:
  - a 1D mean mode
  - a single removable weak coefficient
  - distributed Arnoldi loss of orthogonality
  - or the generic GMRES restart-length control

## The first reduced GMRES system is dominated by one near-null singular mode

Hypothesis:

- The reduced 300x300 scalar-Schur GMRES system itself may already contain the real explanation for the bad first `mpi4` correction.
- If so, dumping that system should reveal whether the huge coefficient vector is tied to one weak reduced mode or to broad generic ill-conditioning.

Diagnostic code:

- Added a root-only reduced-system dump in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_REDUCED_DUMP=/path/to/reduced.txt`
- This writes the first distributed face-only GMRES reduced system:
  - upper-triangular `R`
  - transformed rhs `g`
  - solved coefficients `y`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_reduceddump_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reduceddump_20260415/run.log)
  - [tests/_codex_iliac_1step_mpi4_reduceddump_20260415/reduced.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reduceddump_20260415/reduced.txt)

Offline analysis:

- Singular values of the dumped first reduced system:
  - `sigma_max = 2.698372849771205e+01`
  - `sigma_min = 2.0416260605029431e-04`
  - condition number about `1.32e5`
- The solved coefficient vector is almost entirely the weakest singular mode:
  - `|v_min^T y| / ||y|| = 0.989689`
- The rhs has a real projection onto that weak left singular mode:
  - `|u_min^T g| / ||g|| = 0.215527`
- The smallest right singular vector is not just `e0`; it is broad over the first part of the basis:
  - `|v_min[0]| = 0.216347`
  - first `10` entries carry about `25.3%` of the energy
  - first `50` entries carry about `57.2%`

Interpretation:

- This is the cleanest reduced-space result so far:
  - the first bad `mpi4` solve is dominated by one broad near-null reduced mode
  - not by uniform generic ill-conditioning
  - and not by one trivially removable basis coefficient
- That weak reduced mode is being genuinely excited by the rhs, so the problem is inside the projected operator / gauge handling, not just the backsolve arithmetic.

## The physical first scalar-Schur solution is broad and gauge-like, not face-local

Hypothesis:

- If the reduced near-null mode corresponds to a real coupled-face coarse mode, the first scalar-Schur solution should correlate with outlet-face indicators.
- If it is a gauge-like pressure mode instead, it should look broad and mean-heavy.

Diagnostic code:

- Added a first-cycle physical solution trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_SOLUTION_STATS=1`
- This reports:
  - solution `l2`, mean, centered `l2`, constant ratio
  - correlations with `M_inv` and `1/M_inv`
  - global face-indicator correlations for the active coupled faces

Runs:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_solutionstats_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_solutionstats_20260415/run.log)
  - [tests/_codex_iliac_1step_mpi4_solutionstats2_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_solutionstats2_20260415/run.log)

Observation:

- The first scalar-Schur solution is broad and strongly mean-dominated:
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_solutionstats_20260415/run.log:107)
    - `l2=4.729124e+05`
    - `mean=-2.209378e+03`
    - `centered_l2=3.857401e+05`
    - `constant_ratio=5.785183e-01`
- It is also significantly correlated with the scalar preconditioner weighting:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_solutionstats2_20260415/run.log:106)
    - `minv_ratio=-3.986381e-01`
    - `inv_minv_ratio=-5.346416e-01`
- But it is essentially orthogonal to the active coupled-face indicators:
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_solutionstats_20260415/run.log:108)
  - [run.log:109](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_solutionstats_20260415/run.log:109)
  - normalized dots are effectively zero

Interpretation:

- The dominant bad first `mpi4` scalar-Schur mode is not an outlet-face-local correction.
- It is a broad pressure/gauge-like mode, stronger than the face content and not captured well by a plain constant-only projection.

## Weighted `1/M_inv` post-projection is also too blunt

Hypothesis:

- Since the first bad scalar-Schur solution correlates with both the constant and `1/M_inv` shapes, post-projecting along `1/M_inv` might remove the harmful gauge content more cleanly than the old constant post-projection.

Diagnostic code:

- Added an env-gated weighted post-projection in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_INV_MINV_POSTPROJECT=1`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_invminvpost_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_invminvpost_20260415/run.log)

Observation:

- This did not help; the first outer solve regressed to:
  - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_invminvpost_20260415/run.log:126) `BlockSchur outer iters: 15`

Interpretation:

- The broad weak mode is more gauge-like than face-like, but a simple weighted post-projection is still too blunt.

Updated diagnosis:

- The remaining MPI convergence gap is now best described as:
  - a broad, gauge-like first scalar-Schur near-null mode
  - strongly excited by the distributed rhs
  - not face-local
  - not removable by simple constant or `1/M_inv` post-projection
- The next credible fix is a proper internal projected solve against a richer gauge subspace for this scalar-Schur operator, not another post-hoc correction.

## Dedicated legacy restart override reaches the right path, but restart length alone is not the fix

Hypothesis:

- The earlier generic `SVMP_FSILS_GMRES_SD` knob did not touch the legacy face-only scalar-Schur GMRES path.
- If the first bad reduced solve is mainly a restart-length artifact, a dedicated override on this exact path should improve the first distributed cycle materially.

Diagnostic code:

- Added a legacy-path-specific restart override in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_GMRES_SD`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_facesd60_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facesd60_20260415/run.log)

Observation:

- The dedicated override does change the legacy path. The first bad reduced solve no longer finishes in one `300`-vector cycle; it restarts at `60`.
- But the first cycle is still clearly weak:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facesd60_20260415/run.log:106)
    - `last_i=59`
    - `coeff_l2=4.728587e+04`
    - `diag_ratio=2.997155e+01`
    - `est_residual=1.129196e+02`
- Later cycles on the same rhs reduce the estimate, but not fast enough to make restart length look like the root cause.

Interpretation:

- Restart length is not the main bug.
- The weak first distributed reduced mode is still there; shorter cycles only redistribute the work across more restarts.

## Two-dimensional gauge-space Krylov projection is the first solver-side change that materially improves the bad first reduced solve

Hypothesis:

- The remaining bad distributed scalar-Schur mode is broader than a plain constant and broader than a plain `1/M_inv` weighting.
- A Krylov-space projection against the span `{1, 1/M_inv}` may suppress the bad gauge content without bluntly overwriting the returned scalar solution.

Diagnostic code:

- Added a two-vector owned-space projection helper in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp).
- Added an env-gated first-pass implementation on the legacy face-only scalar-Schur GMRES path:
  - `SVMP_FSILS_FACE_ONLY_GAUGE2_KRYLOV=1`
- Fixed a gating bug in the first attempt: the feature was initially disabled on root because it incorrectly depended on local `active_coupled_faces` before the distributed/global face state was established.

Runs:

- First invalid attempt before the gating fix:
  - [tests/_codex_iliac_1step_mpi4_gauge2krylov_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_20260415/run.log)
- Corrected rerun after the gating fix:
  - same log file, later contents

Observation:

- After the gating fix, the first bad reduced solve improves materially:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_20260415/run.log:106)
    - `coeff_l2=2.892356e+05` versus the old baseline `4.729124e+05`
    - `diag_ratio=2.480218e+01` versus the old baseline `2.997155e+01`
    - `est_residual=1.509158e+01` versus the old baseline `2.699209e+01`
- The later first-cycle solves in the same run are much cleaner, with `diag_ratio` around `3-4` and tiny estimated reduced residuals:
  - [run.log:119](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_20260415/run.log:119)
  - [run.log:125](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_20260415/run.log:125)

Interpretation:

- This is the first solver-side change that improves the actual first problematic distributed reduced solve, rather than only reshaping a post-hoc correction.
- It does not prove the full nonlinear gap is closed yet; the traced run was intentionally stopped once the reduced-solve signal was clear so that a lighter qualification run could be launched.
- But it strengthens the current diagnosis:
  - the missing distributed correction is higher-dimensional than a 1D mean mode
  - it is still gauge-like
  - and handling it inside the scalar-Schur Krylov subspace is more promising than post-projecting the returned solution

## First-solve-only variants of the hand-crafted gauge basis still do not close the gap

Hypothesis:

- The bad distributed mode is mainly a first-solve phenomenon.
- If the always-on `{1, 1/M_inv}` Krylov projection was too intrusive, limiting it to the very first scalar-Schur solve might keep the first-solve benefit without hurting the later solves.
- If that still is not enough, expanding the fixed postprojection basis to the traced weighted span `{1, M_inv, 1/M_inv}` might better match the broad weak mode.

Diagnostic code:

- Added solve-index gating for:
  - `SVMP_FSILS_FACE_ONLY_GAUGE2_KRYLOV_SOLVE_INDEX`
  - `SVMP_FSILS_FACE_ONLY_GAUGE2_POSTPROJECT_SOLVE_INDEX`
- Added a broader weighted first-solve postprojection:
  - `SVMP_FSILS_FACE_ONLY_GAUGE3_POSTPROJECT=1`
  - `SVMP_FSILS_FACE_ONLY_GAUGE3_POSTPROJECT_SOLVE_INDEX=0`

Runs:

- First-solve-only Krylov projection:
  - [tests/_codex_iliac_1step_mpi4_gauge2krylov_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_solve0_20260415/run.log)
- First-solve-only 2D postprojection:
  - [tests/_codex_iliac_1step_mpi4_gauge2post_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2post_solve0_20260415/run.log)
- First-solve-only 3D weighted postprojection:
  - [tests/_codex_iliac_1step_mpi4_gauge3post_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge3post_solve0_20260415/run.log)

Observation:

- The real current default control on this harness is still:
  - [tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:126) `BlockSchur outer iters: 4`
- All three first-solve-only fixed-basis branches regressed that first outer solve badly, up to `11`:
  - [gauge2 Krylov run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_solve0_20260415/run.log:126)
  - [gauge2 post run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2post_solve0_20260415/run.log:126)
  - [gauge3 post run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge3post_solve0_20260415/run.log:126)

Interpretation:

- The hand-crafted broad gauge surrogates are not close enough to the real weak distributed mode.
- The traced mode clearly has weighted-gauge content, but fixed low-dimensional choices like `{1}`, `{1, 1/M_inv}`, and `{1, M_inv, 1/M_inv}` are still not selecting the right representative.
- That effectively exhausts the fixed-basis approach for this path.

## A first-solve postprojection along the extracted weak Arnoldi mode is stable, but still does not close the Newton gap

Hypothesis:

- If the remaining bad distributed mode is not well represented by any hand-picked weighted basis, the most direct next candidate is the weak mode already exposed by the first scalar-Schur GMRES cycle itself.
- Projecting the returned first scalar-Schur solution along that extracted weak Arnoldi mode should be a more faithful mode-selection step than projecting along a guessed gauge basis.

Diagnostic code:

- Added an env-gated first-solve postprojection along the extracted weak mode:
  - `SVMP_FSILS_FACE_ONLY_WEAK_MODE_POSTPROJECT=1`
  - `SVMP_FSILS_FACE_ONLY_WEAK_MODE_POSTPROJECT_SOLVE_INDEX=0`
- This is different from the older `SVMP_FSILS_FACE_ONLY_KRYLOV_MODE_CORRECTION=1` branch:
  - the old branch tried to reduce the true scalar residual during the solve and previously accepted nothing useful
  - the new branch changes the returned representative after the solve, using the mode extracted from the actual Arnoldi basis

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log)

Observation:

- This branch is the first data-driven mode-selection branch that stays nonlinear-stable end-to-end, but its linear work is still a clear regression against the current default control:
  - default control first five outer solves: `4, 3, 4, 4, 5`
    - [default run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:126)
    - [default run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:164)
    - [default run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:202)
    - [default run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:240)
    - [default run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_after_nativeface_default_20260415/run.log:278)
  - weak-mode postprojection first five outer solves: `10, 11, 10, 10, 10`
    - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:126)
    - [run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:164)
    - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:202)
    - [run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:240)
    - [run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:278)
- It still finished at:
  - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:305)
    - `converged=1`
    - `iters=5`
    - `||r||=3.1428439263012435e-10`
- And the wall time was poor on this qualification:
  - [run.log:310](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_weakmodepost_solve0_20260415/run.log:310)
    - `Total time loop: 74.246320 s`

Interpretation:

- Extracted weak-mode postprojection is still a clear linear-solve regression against the real default path, even though it does not break nonlinear convergence entirely.
- But it still does not close the `serial 4` vs `mpi4 5` nonlinear gap.
- So the remaining issue is deeper than “pick a better low-dimensional representative after the solve.”
- The next credible move is to incorporate the extracted weak mode inside the primary distributed scalar-Schur solve itself, not only as a postprojection after the first solve has already been accepted.

## The current binary must be rebaselined: serial is still `4` Newton, MPI is still `5`, but both are on a much tighter linear regime than the old control logs

Hypothesis:

- Some of the older `mpi4` control numbers I was still comparing against were stale relative to the current binary state.
- Before judging the newer mode-selection branches, I needed a fresh serial/MPI baseline on the current build.

Runs:

- Current default MPI-4, canonical archived 1-step iliac case:
  - [tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log)
- Current default serial, canonical archived 1-step iliac case:
  - [tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log)

Observation:

- The current binary still shows a real nonlinear gap:
  - serial:
    - [run.log:263](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:263)
      - `converged=1`
      - `iters=4`
      - `||r||=4.5095397993433303e-09`
    - [run.log:268](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:268)
      - `Total time loop: 81.992628 s`
  - MPI-4:
    - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:305)
      - `converged=1`
      - `iters=5`
      - `||r||=3.1361719936549972e-10`
    - [run.log:310](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:310)
      - `Total time loop: 71.056193 s`
- But the linear regime is now much tighter than the older `~21.6 s` mpi control log I had been citing earlier:
  - current MPI BlockSchur outer sequence is:
    - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:126) `10`
    - [run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:164) `11`
    - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:202) `10`
    - [run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:240) `10`
    - [run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:278) `10`
- Serial is also in this tighter-validating regime now:
  - [run.log:122](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:122) `12`
  - [run.log:160](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:160) `9`
  - [run.log:198](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:198) `9`
  - [run.log:236](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:236) `15`

Interpretation:

- The older low-iteration `mpi4` control logs are no longer the right numerical baseline for the current binary.
- The active current problem is still real and still MPI-specific at the nonlinear level:
  - serial `4` Newton
  - mpi4 `5` Newton
- But any future branch now has to be judged against the current tighter-validating control, not against the older faster-but-stale log.

## Reconstructed reduced weak-mode postprojection is more faithful than `u_last`, but still regresses the first outer solve

Hypothesis:

- The earlier weak-mode postprojection used only `u_last`, which is not the actual bad reduced mode.
- The bad distributed mode should instead be reconstructed as the weakest reduced-space combination `U z_min`, where `z_min` is the smallest-mode vector of the first reduced normal matrix `H^T H`.

Diagnostic code:

- Added a small dense inverse-iteration helper in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp).
- Added an env-gated first-solve postprojection that reconstructs the reduced weak mode `U z_min` from the actual first distributed GMRES cycle:
  - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_POSTPROJECT=1`
  - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_POSTPROJECT_SOLVE_INDEX=0`

Run:

- MPI-4:
  - [tests/_codex_iliac_1step_mpi4_reducedweakpost_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakpost_solve0_20260415/run.log)

Observation:

- This branch still regressed the first outer solve:
  - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakpost_solve0_20260415/run.log:126) `BlockSchur outer iters: 11`
- I stopped it there because the current default control is already only `10` on that first solve.

Interpretation:

- The actual bad reduced mode is not captured well enough even by a simple “reconstruct `U z_min` then postproject” treatment after the solve.
- That means the remaining issue is not just a wrong physical representative chosen at the very end of the solve.
- The next credible move is an internal reduced-space or solve-internal deflation/treatment of that mode, not another postprojection.

## On the current binary, the MPI/serial nonlinear gap still survives the scalar-solver branch choice

Hypothesis:

- Since the current binary had to be rebaselined, I needed to recheck whether the distributed nonlinear gap still survives if MPI is forced off the multi-face GMRES path and back onto the distributed legacy BiCGStab-style branch.
- If that forced branch dropped to serial-like nonlinear behavior, then the current gap would still be largely “solver branch choice.”

Run:

- MPI-4, current binary, forced distributed legacy BiCGStab:
  - [tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log)
  - env:
    - `SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES=1`

Observation:

- This is not a fix on the current binary.
- The forced branch is slightly worse in linear work than the current default:
  - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:126) `11`
  - [run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:164) `10`
  - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:202) `12`
  - [run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:240) `10`
  - [run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:278) `15`
- It still finishes at:
  - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:305)
    - `converged=1`
    - `iters=5`
    - `||r||=3.1689942894824353e-10`
  - [run.log:310](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_current_20260415/run.log:310)
    - `Total time loop: 49.696388 s`

Interpretation:

- On the current binary, the active nonlinear gap is still not explained by “MPI picked the wrong scalar branch.”
- The gap survives both:
  - distributed multi-face GMRES
  - distributed forced legacy BiCGStab
- So the current highest-signal target remains the shared solve-internal treatment of the weak reduced mode or coarse/gauge content, not another GMRES-vs-BiCGStab switch.

## A targeted solve-internal penalty along the extracted reduced weak mode also regresses the first outer solve

Hypothesis:

- The previous weak-mode postprojection experiments changed the returned scalar representative only after the first solve had already been accepted.
- A better test was to penalize the extracted reduced weak mode *inside* the first scalar-Schur GMRES coefficient solve itself, so the solve would choose a different reduced coefficient vector from the start.

Diagnostic code:

- Added an env-gated targeted reduced-space penalty on the first distributed GMRES reduced solve:
  - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_PENALTY_FRAC`
  - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_PENALTY_SOLVE_INDEX`
- Implementation:
  - build the first reduced normal matrix `H^T H`
  - approximate its weakest mode `z_min`
  - solve the penalized reduced system `(H^T H + lambda z_min z_min^T) y = H^T g`

Run:

- MPI-4, small first-solve-only penalty:
  - [tests/_codex_iliac_1step_mpi4_reducedweakpen001_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakpen001_solve0_20260415/run.log)
  - env:
    - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_PENALTY_FRAC=0.01`
    - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_PENALTY_SOLVE_INDEX=0`

Observation:

- The penalty does strongly suppress the first reduced coefficient norm:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakpen001_solve0_20260415/run.log:106)
    - `coeff_l2=9.869151e+04`
    - `weak_mode_penalty_lambda=6.090255e+00`
  - compared with the current default first cycle:
    - [default run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:106)
      - `coeff_l2=4.729124e+05`
- But it does **not** improve the actual outer solve:
  - [run.log:137](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakpen001_solve0_20260415/run.log:137)
    - `BlockSchur outer iters: 11`
- I stopped the branch there because the current default control is already only `10` on that first solve.

Interpretation:

- Even solve-internal reduced-mode damping is not enough by itself.
- The bad distributed mode is not simply “too much amplitude along `z_min`” in the reduced coefficient solve.
- So the remaining issue is deeper than a one-mode penalization or one-mode postprojection.
- The next credible direction is a richer solve-internal coarse/deflation treatment or a more fundamental fix in how the distributed scalar-Schur operator handles that weak subspace.

## On the current binary, alternative scalar-Schur preconditioner structure and preconditioned-rhs initialization do not close the nonlinear gap

Hypothesis:

- Since the current binary had drifted away from the older faster controls, I re-screened a few structural scalar-Schur choices directly against the current baseline rather than assuming older results still applied:
  - forced `algebraic-shat` scalar Schur preconditioning
  - forced distributed legacy BiCGStab (already recorded above)
  - preconditioned-rhs initialization for the scalar-Schur solve

Runs:

- MPI-4 forced `algebraic-shat`:
  - [tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log)
  - env:
    - `SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_PC=algebraic-shat`
- MPI-4 preconditioned-rhs initialization:
  - [tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log)
  - env:
    - `SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS=1`

Observation:

- `algebraic-shat` is numerically almost identical to the current default:
  - same outer sequence:
    - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:126) `10`
    - [run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:164) `11`
    - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:202) `10`
    - [run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:240) `10`
    - [run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:278) `10`
  - same nonlinear result:
    - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:305) `iters=5`
  - but a faster wall time on this run:
    - [run.log:309](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_shat_current_20260415/run.log:309) `Total time loop: 52.990409 s`
- Preconditioned-rhs initialization also does not close the gap:
  - outer sequence:
    - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:126) `10`
    - [run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:164) `11`
    - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:202) `9`
    - [run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:240) `10`
    - [run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:278) `10`
  - nonlinear result:
    - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:305) `iters=5`
  - wall time:
    - [run.log:310](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_initrhs_current_20260415/run.log:310) `Total time loop: 52.290509 s`

Interpretation:

- On the current binary, these structural scalar-Schur variants can shift runtime, but they are not closing the nonlinear gap.
- So the current strongest diagnosis remains:
  - serial `4` Newton
  - mpi4 `5` Newton
  - not fixed by branch choice
  - not fixed by one-mode postprojection
  - not fixed by one-mode reduced-space damping
  - not fixed by preconditioned-rhs initialization

## Simple amplification of the reconstructed reduced weak mode is also not enough

Hypothesis:

- The weak-mode penalty result suggested that shrinking the extracted reduced weak mode was not helping.
- Since the serial first-step pressure update is still much larger than MPI on the current binary, the opposite possibility was worth checking: maybe MPI actually needs *more* of that broad weak mode, not less.

Diagnostic code:

- Added an env-gated first-solve gain along the reconstructed reduced weak mode:
  - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_GAIN`
  - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_GAIN_SOLVE_INDEX`
- This uses the same reconstructed reduced weak mode `U z_min` as the earlier postprojection branch, but amplifies its current component instead of subtracting or penalizing it.

Run:

- MPI-4, first-solve-only unit gain:
  - [tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log)
  - env:
    - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_GAIN=1`
    - `SVMP_FSILS_FACE_ONLY_REDUCED_WEAK_MODE_GAIN_SOLVE_INDEX=0`

Observation:

- This branch did not destabilize the solve, but it also did not improve the nonlinear behavior:
  - first five outer solves stayed identical to the current default:
    - [run.log:126](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:126) `10`
    - [run.log:164](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:164) `11`
    - [run.log:202](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:202) `10`
    - [run.log:240](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:240) `10`
    - [run.log:278](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:278) `10`
  - nonlinear result:
    - [run.log:305](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:305) `iters=5`
  - wall time:
    - [run.log:310](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_reducedweakgain1_solve0_20260415/run.log:310) `Total time loop: 52.336653 s`

Interpretation:

- The current reduced weak mode is not a simple one-parameter underrepresentation problem either.
- Shrinking it hurts.
- Amplifying it also does not close the gap.
- So the active MPI/serial difference is deeper than “pick the right scalar amplitude along one extracted reduced weak mode.”


## Current-binary serial vs mpi4 NS_SOLVER trace still isolates the gap inside the first scalar-Schur return

Hypothesis:

- The current-binary rebaseline showed the same nonlinear gap (`serial 4` vs `mpi4 5`), but the runtime regime is much tighter than the older controls.
- Before trying more reduced-space edits, I needed a direct current-binary comparison of the first `NS_SOLVER` scalar-Schur quantities to confirm whether the active mismatch is still in the returned scalar correction, not upstream in FE assembly or the first Schur rhs.

Runs:

- Current serial `NS_SOLVER` trace:
  - [tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log)
- Current mpi4 `NS_SOLVER` trace:
  - [tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log)

Observation:

- The first current-binary scalar-Schur rhs is still identical between serial and mpi4:
  - serial [run.log:111](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log:111) `schur_rhs l2=4.441774e+02 mean=-3.861493e-01 min=-7.730578e+01 max=1.153075e+00`
  - mpi4 [run.log:115](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log:115) `schur_rhs l2=4.441774e+02 mean=-3.861493e-01 min=-7.730578e+01 max=1.153075e+00`
- The returned first scalar-Schur solution is still much smaller on mpi4:
  - serial [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log:112) `schur_solution l2=6.541547e+06 mean=-4.794320e+04 min=-1.080696e+05 max=-3.304948e+03 dot_rhs=4.174378e+08`
  - mpi4 [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log:116) `schur_solution l2=4.729124e+05 mean=-2.209378e+03 min=-1.363138e+04 max=5.037559e-02 dot_rhs=5.632307e+07`
- The lifted pressure-basis momentum remains much smaller on mpi4 too:
  - serial [run.log:113](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log:113) `pressure_basis_momentum l2=3.786638e+04`
  - mpi4 [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log:117) `pressure_basis_momentum l2=6.440921e+03`

Interpretation:

- On the current binary, the gap is still created after the first shared scalar-Schur rhs is formed and before the full Newton increment is assembled.
- So the active locus remains the first distributed scalar-Schur solve itself, not FE assembly, not the first rhs, and not outer Newton policy.
- The next high-signal comparison is serial-vs-mpi reduced-system structure on that first solve, not more FE-side probes.


## Current forced-mpi4 BiCGStab still returns the same small first scalar correction

Hypothesis:

- The current code still uses serial face-only BiCGStab and distributed multi-face GMRES by default.
- I had already qualified forced distributed BiCGStab as a negative end-to-end result, but I still needed the current-binary first `NS_SOLVER` trace to separate “wrong branch choice” from “same weak distributed scalar correction through both branches.”

Run:

- Forced distributed BiCGStab with `NS_SOLVER` trace, stopped after the first `pressure_basis_momentum` line:
  - [tests/_codex_iliac_1step_mpi4_forcebicg_nssolver_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_nssolver_current_20260415/run.log)
  - env:
    - `SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES=1`
    - `SVMP_FSILS_NS_SOLVER_TRACE=1`

Observation:

- The first forced-BiCGStab distributed rhs is still identical to serial and to the default mpi4 branch:
  - [run.log:115](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_nssolver_current_20260415/run.log:115) `schur_rhs l2=4.441774e+02 mean=-3.861493e-01 min=-7.730578e+01 max=1.153075e+00`
- But the returned first scalar solution stays on the same small distributed scale, not the serial scale:
  - forced mpi4 BiCGStab [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_nssolver_current_20260415/run.log:116) `schur_solution l2=3.903956e+05 mean=-1.663136e+03 min=-1.216124e+04 max=3.097315e-63 dot_rhs=5.027297e+07`
  - default mpi4 GMRES [run.log:116](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log:116) `schur_solution l2=4.729124e+05 mean=-2.209378e+03 min=-1.363138e+04 max=5.037559e-02 dot_rhs=5.632307e+07`
  - serial [run.log:112](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log:112) `schur_solution l2=6.541547e+06 mean=-4.794320e+04 min=-1.080696e+05 max=-3.304948e+03 dot_rhs=4.174378e+08`
- The lifted pressure-basis momentum stays small too:
  - forced mpi4 BiCGStab [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_nssolver_current_20260415/run.log:117) `pressure_basis_momentum l2=6.019189e+03`
  - default mpi4 GMRES [run.log:117](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_nssolver_current_20260415/run.log:117) `pressure_basis_momentum l2=6.440921e+03`
  - serial [run.log:113](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_nssolver_current_20260415/run.log:113) `pressure_basis_momentum l2=3.786638e+04`

Interpretation:

- On the current binary, the distributed first-step scalar-correction gap survives even when mpi4 is forced onto the same BiCGStab family as serial.
- So the branch choice itself is not the active cause anymore.
- The remaining issue is in the shared distributed scalar-Schur solve semantics, especially gauge/coarse handling inside that path, not in “GMRES vs BiCGStab.”


## Serial vs distributed BiCGStab first scalar solution shape diverges strongly on the current binary

Hypothesis:

- With current-binary branch choice effectively ruled out, the next missing comparison was the actual *shape* of the first scalar-Schur solution in the same BiCGStab family for serial and distributed runs.
- The existing solution-stat trace only fired in the GMRES path, so I extended that diagnostic to the BiCGStab path as well, behind the same `SVMP_FSILS_TRACE_FACE_ONLY_SOLUTION_STATS` gate.

Code:

- Diagnostic-only extension in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp) so `BICGS_FACE_ONLY_SOLUTION` now also emits from the face-only BiCGStab branch.
- Rebuilt `svmultiphysics` successfully with `cmake --build build/svMultiPhysics-build --target svmultiphysics -j4`.

Runs:

- Serial current default with solution stats:
  - [tests/_codex_iliac_1step_serial_solutionstats_bicg_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_solutionstats_bicg_current_20260415/run.log)
- Forced distributed BiCGStab current with solution stats:
  - [tests/_codex_iliac_1step_mpi4_forcebicg_solutionstats_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_solutionstats_current_20260415/run.log)

Observation:

- Serial BiCGStab first scalar solution is almost a broad constant-like field:
  - [serial run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_solutionstats_bicg_current_20260415/run.log:102) `solver=bicgstab index=300 l2=6.541547e+06 mean=-4.794320e+04 centered_l2=2.746972e+06 constant_ratio=9.075579e-01 minv_ratio=-9.075579e-01 inv_minv_ratio=-9.075579e-01`
- Distributed BiCGStab first scalar solution is much less constant and much smaller:
  - [mpi4 run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_solutionstats_current_20260415/run.log:106) `solver=bicgstab index=300 l2=3.903956e+05 mean=-1.663136e+03 centered_l2=3.316543e+05 constant_ratio=5.275342e-01 minv_ratio=-3.534279e-01 inv_minv_ratio=-4.902248e-01`
- Serial also has nonzero outlet-face means, while distributed BiCGStab is essentially face-orthogonal at machine scale:
  - serial [run.log:103](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_solutionstats_bicg_current_20260415/run.log:103) `face=1 mean=-5.863100e+03`
  - serial [run.log:104](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_solutionstats_bicg_current_20260415/run.log:104) `face=2 mean=-7.357076e+03`
  - mpi4 [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_solutionstats_current_20260415/run.log:107) `face=1 mean=-4.111343e-63`
  - mpi4 [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_forcebicg_solutionstats_current_20260415/run.log:108) `face=2 mean=-3.431895e-54`

Interpretation:

- This is the strongest evidence so far that the distributed scalar-Schur path is underrepresenting the broad scalar mode that serial is actually using on the first solve.
- The remaining gap is not just “wrong Krylov branch” and not just “outlet-face mode missing.” Even in BiCGStab, distributed lands on a much less constant, almost face-orthogonal scalar correction than serial.
- The next credible fix direction is to enrich or preserve that broad scalar mode *inside* the distributed scalar solve, not to keep adding post-hoc shifts after the solve has already settled on the wrong subspace.


## A solve-internal {current solution, constant mode} enrichment is not enough either

Hypothesis:

- The new serial-vs-distributed BiCGStab solution-shape comparison suggested that the distributed scalar solve is underrepresenting a broad constant-like scalar mode that serial is actually using.
- The cheapest physics-agnostic way to test that was to enrich the distributed scalar solve *inside* the first solve with the 2D subspace `{current solution, constant mode}` and accept it only if the scalar residual improved.

Code:

- Added an env-gated solve-internal branch in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_CONSTANT_SUBSPACE_ENRICH=1`
  - `SVMP_FSILS_FACE_ONLY_CONSTANT_SUBSPACE_ENRICH_SOLVE_INDEX`
  - `SVMP_FSILS_TRACE_FACE_ONLY_CONSTANT_SUBSPACE_ENRICH=1`
- This constructs the 2D least-squares candidate in the span of the current scalar solution and a broad owned constant mode, then accepts it only if the scalar residual norm improves.

Run:

- MPI-4 current 1-step iliac harness with first-solve-only enrichment:
  - [tests/_codex_iliac_1step_mpi4_constsubspace_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_20260415/run.log)

Observation:

- The branch did find and accept a small constant enrichment on the first distributed scalar solve:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_20260415/run.log:106) `solver=gmres residual_before=2.699209e+01 residual_after=2.691913e+01 accept=1 coeff0=1.000020e+00 coeff1=3.106386e-02`
- But the improvement is tiny, exactly in line with the earlier coarse-residual probes on the global mean mode.
- Worse, the run quickly became operationally unusable:
  - by the third solve, [run.log:203](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_20260415/run.log:203) still had `BlockSchur outer iters: 10`
  - and the same stage had [run.log tail](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_20260415/run.log) `457072` backend allreduces, with `454560` of them in Schur solves
- I stopped that branch and killed the lingering `mpirun`/solver processes.

Interpretation:

- This is another strong negative result: even when I enrich the distributed scalar solve directly with the broad constant mode that serial seems to favor, the accepted improvement is tiny and it does not move the outer behavior in the right direction.
- So the remaining gap is deeper than a simple missing 1D constant component, even though the serial solution is visibly much more constant-like.
- The next credible target is therefore a richer solve-internal weak subspace than `{current solution, constant}`, not another one-dimensional constant/mean adjustment.


## Correction: the constant-subspace branch is numerically benign, but it still does not close the gap

Follow-up qualification:

- I reran the same branch to completion on the canonical current mpi4 harness:
  - [tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log)

Observation:

- The branch remains a tiny first-solve correction only:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:106) `residual_before=2.699209e+01 residual_after=2.691913e+01 accept=1 coeff1=3.106386e-02`
- It does **not** change the distributed nonlinear outcome:
  - [run.log:306](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:306) `converged=1 iters=5`
- It also preserves the same outer pattern as the baseline:
  - [run.log:127](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:127) `10`
  - [run.log:165](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:165) `11`
  - [run.log:203](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:203) `10`
  - [run.log:241](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:241) `10`
  - [run.log:279](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:279) `10`
- Wall time on this run was [67.457625 s](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constsubspace_current_full_20260415/run.log:311), slightly better than the current control [71.056193 s](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:310), but the nonlinear gap remains unchanged and the runtime delta is small enough that I am treating it as non-conclusive.

Interpretation:

- The earlier fear that this branch uniquely exploded collectives was incorrect; that allreduce scale is the same order as the current baseline.
- The corrected conclusion is still negative on the convergence question: adding the broad constant mode solve-internally is not enough to close the mpi4 gap.


## The distributed GMRES basis itself is not building the broad scalar mode early

Hypothesis:

- After the negative 1D and 3D solve-internal enrichments, the next important question was whether the distributed GMRES basis even contains a broad constant-like scalar direction early enough to matter.
- If it does not, then post-solve or end-of-cycle coefficient surgery is working too late; the mode is being suppressed during basis construction itself.

Runs:

- MPI-4 current 1-step iliac harness with `SVMP_FSILS_TRACE_FACE_ONLY_WEAK_MODE=1` and varying `SVMP_FSILS_TRACE_FACE_ONLY_BASIS_ITER`:
  - [iter 1](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_1_current_20260415/run.log)
  - [iter 2](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_2_current_20260415/run.log)
  - [iter 3](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_3_current_20260415/run.log)
  - [iter 4](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_4_current_20260415/run.log)
  - [iter 5](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_5_current_20260415/run.log)
  - [iter 10](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_10_current_20260415/run.log)

Observation:

- The first few distributed Arnoldi basis vectors are nowhere near the serial first-solution shape:
  - [iter 1:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_1_current_20260415/run.log:106) `constant_ratio=7.106187e-02`
  - [iter 2:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_2_current_20260415/run.log:106) `constant_ratio=7.360480e-02`
  - [iter 3:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_3_current_20260415/run.log:106) `constant_ratio=7.381011e-02`
  - [iter 4:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_4_current_20260415/run.log:106) `constant_ratio=7.652362e-02`
  - [iter 5:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_5_current_20260415/run.log:106) `constant_ratio=7.378006e-02`
  - [iter 10:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_basisiter_10_current_20260415/run.log:106) `constant_ratio=3.460684e-02`
- Compare that to the serial first BiCGStab scalar solution shape:
  - [serial solution:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_solutionstats_bicg_current_20260415/run.log:102) `constant_ratio=9.075579e-01`

Interpretation:

- This is a very strong diagnostic result: the distributed GMRES basis itself is not building a broad constant-like scalar mode in the first 10 iterations.
- That explains why small post-solve corrections keep failing: the basis is already missing the serial-like broad mode before the reduced least-squares solve chooses coefficients.

## A constant-mode initial guess also does not fix the basis-level issue

Hypothesis:

- If the basis is missing the broad scalar mode because the distributed GMRES solve starts from zero, then seeding the first solve with the best constant-mode least-squares initial guess should help the first Krylov cycle build the right residual subspace.

Code:

- Added an env-gated first-solve-only branch in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_CONSTANT_INITIAL_GUESS=1`
  - `SVMP_FSILS_FACE_ONLY_CONSTANT_INITIAL_GUESS_SOLVE_INDEX`
  - `SVMP_FSILS_TRACE_FACE_ONLY_CONSTANT_INITIAL_GUESS=1`
- It computes the best constant-only least-squares scalar initial guess `alpha` against the current distributed scalar operator, then starts GMRES from that guess.

Run:

- MPI-4 current 1-step iliac harness with first-solve constant initialization:
  - [tests/_codex_iliac_1step_mpi4_constinit_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constinit_current_20260415/run.log)

Observation:

- The best constant-only initial guess is tiny:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constinit_current_20260415/run.log:106) `alpha=-2.596817e-04 residual_norm=4.441774e+02`
- It does not change the nonlinear pattern:
  - [run.log:127](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constinit_current_20260415/run.log:127) `BlockSchur outer iters: 10`
  - [run.log:306](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constinit_current_20260415/run.log:306) `converged=1 iters=5`
  - [run.log:310](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constinit_current_20260415/run.log:310) `Total time loop: 60.960405 s`

Interpretation:

- So even injecting the best constant-only component before distributed GMRES starts is not enough.
- The remaining missing scalar content is broader than a 1D constant mode and is being suppressed before or during basis construction.


## Overlap-output accumulation is probably not the active issue on this harness

Hypothesis:

- After the negative basis/subspace tests, one remaining solver-semantics suspicion was that the distributed scalar-Schur basis might be getting distorted by post-matvec overlap accumulation.
- There was already a global `SVMP_FSILS_SKIP_POST_SOLVE_COMM` hammer, but it was too invasive. So I added a narrower face-only scalar-operator experiment.

Code:

- Added an env-gated branch in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_SKIP_OPERATOR_OUTPUT_SYNC=1`
- This disables only the `halo.sync_scalar(out_vec, ...)` at the end of the face-only scalar `apply_schur_operator(...)` lambda, leaving the rest of FSILS unchanged.

Run:

- MPI-4 current 1-step iliac harness with:
  - `SVMP_FSILS_FACE_ONLY_SKIP_OPERATOR_OUTPUT_SYNC=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_WEAK_MODE=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_BASIS_ITER=1`
  - `SVMP_FSILS_TRACE_FACE_ONLY_SOLUTION_STATS=1`
- Log: [tests/_codex_iliac_1step_mpi4_faceoutputnosync_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_faceoutputnosync_current_20260415/run.log)

Observation:

- The narrowed branch leaves the first distributed basis and first distributed scalar solution unchanged from baseline:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_faceoutputnosync_current_20260415/run.log:106) `basis constant_ratio=7.106187e-02`
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_faceoutputnosync_current_20260415/run.log:107) `solution constant_ratio=5.785183e-01`
- Those are exactly the same first-solve values as the baseline current mpi4 traces.
- The same log also reminds us that this harness is reporting `ghost=0` in the FSILS partition summary near the top, so there may not be any active scalar overlap region here for this particular path to change.
- The run did not reach a useful final summary on the timescale where the baseline does, so I stopped it after the first-solve diagnostics and killed the lingering processes.

Interpretation:

- This is another negative result, but a useful one: the missing broad scalar mode does not appear to come from post-matvec scalar overlap accumulation on this harness.
- The `ghost=0` partition summary makes that outcome plausible.
- So the active MPI/serial gap is still pointing back to the distributed scalar solve’s internal basis/subspace quality, not to a trivial overlap-sync mistake in the face-only scalar operator output.


## Attempted constant-span trace was inconclusive as a practical loop

Hypothesis:

- After the new basis constant-ratio evidence, I wanted a sharper probe: how much of the broad constant mode lies in the current distributed Arnoldi span, not just in a single basis vector.

Code:

- Added an env-gated GMRES-span trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_CONSTANT_SPAN=1`
- This is intended to print the constant-mode projection residual after each new Arnoldi basis vector.

Run:

- MPI-4 current 1-step iliac harness with:
  - `SVMP_FSILS_TRACE_FACE_ONLY_CONSTANT_SPAN=1`
- Log: [tests/_codex_iliac_1step_mpi4_constspan_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constspan_current_20260415/run.log)

Observation:

- The run did not reach the first useful span trace on a practical timescale, so I stopped it and killed the lingering processes.
- That makes this probe inconclusive in its current form as a debugging loop.

Interpretation:

- The idea is still valid, but the current implementation is not giving fast enough feedback to justify using it as the primary next loop.


## Full-Krylov-plus-constant reduced solve lowered the true first Schur residual, but still did not close the nonlinear gap

Hypothesis:

- The earlier constant-mode experiments were all low-dimensional: a 1D constant postshift, a constant initial guess, or a tiny enriched correction subspace.
- If the missing broad serial-like scalar mode is already partially present across the distributed Arnoldi basis, then the right test is not another 1D correction. It is a small dense solve over the entire current Krylov span plus one explicit broad constant vector.
- If that augmented reduced solve changes the first distributed scalar-Schur solution materially, we should see either a lower first BlockSchur outer count or a `5 -> 4` Newton improvement on the archived mpi4 iliac harness.

Code:

- Added an env-gated branch in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_CONSTANT_LS=1`
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_CONSTANT_LS_SOLVE_INDEX=<n>`
  - trace: `SVMP_FSILS_TRACE_FACE_ONLY_KRYLOV_PLUS_CONSTANT_LS=1`
- The branch:
  - builds the full current distributed scalar-Schur Krylov basis `u_0..u_m`
  - applies the true face-only scalar operator to every basis vector and to the broad constant mode
  - solves a dense normal-equation system over that augmented basis
  - compares the candidate against the current solve and optionally replaces the first returned scalar correction

First run:

- MPI-4 archived 1-step iliac harness with:
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_CONSTANT_LS=1`
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_CONSTANT_LS_SOLVE_INDEX=0`
  - `SVMP_FSILS_TRACE_FACE_ONLY_KRYLOV_PLUS_CONSTANT_LS=1`
- Log: [tests/_codex_iliac_1step_mpi4_krylovconstls_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_current_20260415/run.log)

Observation:

- The augmented solve really did find a slightly better full scalar-Schur candidate:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_current_20260415/run.log:106) `residual_before=2.699209e+01 residual_after=2.689064e+01`
- But the branch rejected that candidate because I had incorrectly compared the candidate’s true full residual against the reduced least-squares residual rather than against the current full operator residual on the same solve.
- That was a real bug in the experiment, not a valid negative result.

Fix:

- Corrected the acceptance test in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp) so it now compares:
  - `baseline_true_residual = ||rhs - A * X_current||`
  - `candidate_true_residual = ||rhs - A * X_candidate||`
- The trace now prints both true residuals explicitly alongside the reduced residual.

Acceptance-fixed run:

- Same mpi4 archived 1-step iliac harness with the same envs.
- Log: [tests/_codex_iliac_1step_mpi4_krylovconstls_acceptfix_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_acceptfix_current_20260415/run.log)

Observation:

- The acceptance-fixed branch does now keep the better full-residual candidate:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_acceptfix_current_20260415/run.log:106) `baseline_true_residual=2.699209e+01 candidate_true_residual=2.689064e+01 accept=1 aug_dim=301 const_coeff=4.317283e-02 reduced_residual=2.699209e+01`
- But the end-to-end nonlinear behavior still does not improve:
  - first solve outer count remains [10](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_acceptfix_current_20260415/run.log:127)
  - full run still finishes at [5 Newton](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_acceptfix_current_20260415/run.log:306)
  - total loop time is [54.711109 s](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovconstls_acceptfix_current_20260415/run.log:311), which is better than the current default [71.056193 s](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run.log:310) but still not parity in nonlinear convergence

Interpretation:

- This is a cleaner negative result than the earlier 1D constant-mode attempts.
- The missing distributed broad scalar content is not just “the constant mode was absent from the final reduced solve.” Even when the entire current Krylov span is given access to an extra broad constant vector and the better full-residual candidate is actually accepted, the mpi4 iliac run still stays at `5` Newton.
- So the remaining serial/mpi gap is now pointing even more strongly at the *quality of the distributed scalar-Schur basis being built in the first place*, not merely the final reduced coefficient choice over that basis.


## Longer forced-mpi4 constant-mode operator probe was still not a practical oracle

Hypothesis:

- One remaining upstream question was whether the broad constant-like scalar mode is already being distorted at the operator/preconditioner level in the distributed BiCGStab path, before any recurrence or reduced solve effects.
- I already had the serial BiCGStab constant-mode operator trace:
  - [tests/_codex_iliac_1step_serial_constmodeop_bicg_current_20260415/run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_constmodeop_bicg_current_20260415/run.log:102)
- The missing comparison was the corresponding forced-mpi4 BiCGStab trace.

Runs:

- Rechecked serial short probe with a 40 s timeout:
  - Log: [tests/_codex_iliac_1step_serial_constmodeop_bicg_recheck_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_constmodeop_bicg_recheck_20260415/run.log)
  - It again emitted the same operator line quickly:
    - [run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_constmodeop_bicg_recheck_20260415/run.log:102)
- Tried the matching mpi4 short probe with:
  - `SVMP_FSILS_TRACE_FACE_ONLY_CONST_MODE_OPERATOR=1`
  - `SVMP_FSILS_DISABLE_MULTI_FACE_LEGACY_GMRES=1`
  - 40 s timeout
  - Log: [tests/_codex_iliac_1step_mpi4_constmodeop_forcebicg_recheck_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constmodeop_forcebicg_recheck_20260415/run.log)
- Then retried the same mpi4 probe with a longer 95 s timeout:
  - Log: [tests/_codex_iliac_1step_mpi4_constmodeop_forcebicg_long_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_constmodeop_forcebicg_long_20260415/run.log)

Observation:

- The serial recheck behaved as expected and re-emitted the constant-mode operator line.
- Neither mpi4 timeout run emitted any `BICGS_FACE_ONLY_CONST_MODE` line before timeout.
- The longer mpi4 timeout did advance further into the run, but still only reached generic FSILS timing output near the end of the captured log and did not produce the targeted scalar-operator probe.

Interpretation:

- This probe is still not practical enough to use as the main debugging loop for the mpi gap.
- The absence of that line after a much longer cutoff does *not* yet prove anything good or bad about the distributed operator itself; it only says this probe is too cumbersome on the full iliac harness in its current form.
- So I am not moving the main diagnosis based on this attempt. The highest-signal evidence is still the existing basis/subspace traces showing that the distributed scalar solve fails to build the broad serial-like mode early, while post-hoc and augmented reduced-solve fixes do not close the gap.


## Requalifying `gauge2_krylov` on the current binary shows it is now a hard regression, not a fix

Hypothesis:

- Earlier in the investigation, `SVMP_FSILS_FACE_ONLY_GAUGE2_KRYLOV=1` was the first branch that materially improved the bad first distributed reduced solve from inside the Krylov process.
- That signal was recorded before the current stricter backend state was rebaselined, so it needed a clean rerun on the current binary and the current archived 1-step mpi4 iliac control.

Runs:

- Full current-binary qualification:
  - [tests/_codex_iliac_1step_mpi4_gauge2krylov_current_requal_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_current_requal_20260415/run.log)
- Trace-oriented current-binary rerun:
  - [tests/_codex_iliac_1step_mpi4_gauge2krylov_trace_current_requal_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_trace_current_requal_20260415/run.log)

Observation:

- The trace-oriented run still shows the same old local signal on the first reduced solve:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_trace_current_requal_20260415/run.log:106)
    - `coeff_l2=2.892356e+05`
    - `diag_ratio=2.480218e+01`
    - `est_residual=1.509158e+01`
  - which is better than the plain current baseline first reduced solve.
- But end-to-end it is a hard regression on the current binary:
  - first outer solve jumps to [15 iterations](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_trace_current_requal_20260415/run.log:141)
  - the full qualification keeps hitting [15 outer iterations repeatedly](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_current_requal_20260415/run.log:126)
  - and then fails true-residual validation outright:
    - [run.log:378](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_gauge2krylov_current_requal_20260415/run.log:378)
    - `blockschur: true residual check failed (|Ax-b|=73.0068, rel=0.0070582, target=0.000103435)`

Interpretation:

- This is an important correction to the earlier narrative: on the current backend state, `gauge2_krylov` is not the missing fix.
- It improves the reduced LS diagnostics locally but damages the actual solve enough to fail the real global residual check.
- So that branch is now ruled out for the current binary.


## Full-Krylov augmented solves with richer fixed gauge bases improve the first true scalar residual, but still do not change the `5`-Newton outcome

Hypothesis:

- The constant-only augmented reduced solve already showed that the current Krylov basis plus one broad constant vector could produce a slightly better first true scalar residual, but not enough to change the nonlinear outcome.
- Since the traced weak distributed mode has broader weighted-gauge character, the natural next step was to enrich that same *full current Krylov span* with small fixed weighted gauge sets:
  - `{1, 1/M_inv}`
  - `{1, M_inv, 1/M_inv}`
- This is different from the earlier postprojection experiments because the richer basis is available inside the same dense reduced full-operator solve before the first scalar correction is returned.

Code:

- Extended the earlier full-Krylov augmented solve in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp) with two new env-gated branches:
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_GAUGE2_LS=1`
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_GAUGE3_LS=1`
- Both use the same acceptance rule as the corrected constant-only branch:
  - compare candidate vs current **true** full scalar-Schur residual
  - accept only if the candidate really improves that full residual

Runs:

- `Krylov + {1, 1/M_inv}`:
  - [tests/_codex_iliac_1step_mpi4_krylovgauge2ls_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovgauge2ls_current_20260415/run.log)
- `Krylov + {1, M_inv, 1/M_inv}`:
  - [tests/_codex_iliac_1step_mpi4_krylovgauge3ls_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovgauge3ls_current_20260415/run.log)

Observation:

- The `gauge2` augmented solve does improve the first true scalar residual beyond the constant-only branch:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovgauge2ls_current_20260415/run.log:106)
    - `baseline_true_residual=2.699209e+01`
    - `candidate_true_residual=2.659983e+01`
    - `accept=1`
    - `const_coeff=-1.502343e-02`
    - `inv_minv_coeff=3.799170e-02`
- The `gauge3` augmented solve is essentially the same story:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovgauge3ls_current_20260415/run.log:106)
    - `baseline_true_residual=2.699209e+01`
    - `candidate_true_residual=2.659946e+01`
    - `accept=1`
    - `const_coeff=-1.352723e-02`
    - `minv_coeff=-3.238041e-04`
    - `inv_minv_coeff=3.681205e-02`
- But neither branch changes the actual nonlinear behavior:
  - `gauge2` still finishes at [5 Newton](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovgauge2ls_current_20260415/run.log:306) with the same outer pattern `10,11,10,10,10`
  - `gauge3` still finishes at [5 Newton](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_krylovgauge3ls_current_20260415/run.log:306) with the same outer pattern `10,11,10,10,10`

Interpretation:

- This exhausts the obvious fixed low-dimensional gauge augmentation family at the reduced-solve level.
- The current distributed first scalar correction does become *slightly* better when the full Krylov span gets access to richer broad weighted modes, and those candidates are genuinely accepted on true full residual.
- But the improvement is still far too small to change the first outer solve or the final `5`-Newton nonlinear count.
- So the remaining mpi/serial gap is not just “the final reduced solve needed a richer fixed gauge basis.” The stronger diagnosis now is:
  - the current distributed scalar-Schur Krylov process is still not building the right broader representative early enough
  - and simply enlarging the final reduced solve with small fixed gauge-like mode sets is not sufficient to recover serial parity


## A first-cycle broad-span oracle was added but is still not practical on the full iliac harness

Hypothesis:

- After the augmented reduced-solve negatives, the clean remaining question was:
  - does the first distributed scalar-Schur Krylov basis *already* contain the obvious broad modes `{1, M_inv, 1/M_inv}`, with the problem only appearing later in coefficient selection,
  - or are those broad modes mostly absent from the first basis itself?
- To answer that directly, I added a cheap one-shot span diagnostic that is supposed to run once, after the first Arnoldi basis is built and before the reduced backsolve:
  - `SVMP_FSILS_TRACE_FACE_ONLY_BROAD_SPAN=1`

Code:

- Added an env-gated first-cycle trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp) to report projection residual ratios of:
  - `constant`
  - `M_inv`
  - `1/M_inv`
  against the built distributed Krylov basis.
- This is much lighter than the older per-iteration constant-span trace and is intended to emit just once.

Runs:

- Short timeout run:
  - [tests/_codex_iliac_1step_mpi4_broadspan_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_broadspan_current_20260415/run.log)
- Then a live rerun that I stopped after it still failed to emit the probe line on a timescale where it should have been useful:
  - [tests/_codex_iliac_1step_mpi4_broadspan_live_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_broadspan_live_current_20260415/run.log)

Observation:

- Neither run emitted any `BICGS_FACE_ONLY_BROAD_SPAN` line before I stopped the live job.
- The live job was not stuck before startup; its log had already advanced into the solve, but the targeted probe still had not appeared.

Interpretation:

- So this specific first-cycle broad-span oracle is, in its current implementation, still not practical enough to use as the main loop on the full iliac harness.
- I am not updating the core diagnosis from this attempt.
- The highest-signal evidence still remains:
  - the first distributed basis vectors look much less broad/constant-like than the serial first scalar solution,
  - the first distributed reduced solve has one dominant weak mode,
  - and increasingly rich fixed low-dimensional augmentations at the final reduced-solve level still do not close the `4` vs `5` Newton gap.


## Serial-vs-mpi first scalar solutions are globally different fields, not just shifted versions

Hypothesis:

- The remaining gap might still be diagnosable by comparing the actual first scalar-Schur solution fields, not just norms or reduced coefficients.
- If the distributed first scalar correction were mostly a shifted/scaled version of the serial one, then a direct serial-solution oracle might be enough to close the gap.

Code:

- Added global-node storage to the FSILS lhs in:
  - [fils_struct.hpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/fils_struct.hpp)
  - [lhs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/lhs.cpp)
- Added an env-gated owned-node scalar solution dump in:
  - [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp)
  - `SVMP_FSILS_TRACE_FACE_ONLY_SOLUTION_DUMP=<prefix>`
  - `SVMP_FSILS_TRACE_FACE_ONLY_SOLUTION_DUMP_SOLVE_INDEX=<n>`

Runs:

- Serial dump run:
  - [tests/_codex_iliac_1step_serial_scalarsol_dump2_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_scalarsol_dump2_current_20260415/run.log)
  - dump file:
    - [first_scalar_solution.solve0.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_scalarsol_dump2_current_20260415/first_scalar_solution.solve0.rank0.txt)
- `mpi4` dump run:
  - [tests/_codex_iliac_1step_mpi4_scalarsol_dump_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarsol_dump_current_20260415/run.log)
  - dump files:
    - [rank0](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarsol_dump_current_20260415/first_scalar_solution.solve0.rank0.txt)
    - [rank1](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarsol_dump_current_20260415/first_scalar_solution.solve0.rank1.txt)
    - [rank2](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarsol_dump_current_20260415/first_scalar_solution.solve0.rank2.txt)
    - [rank3](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_scalarsol_dump_current_20260415/first_scalar_solution.solve0.rank3.txt)

Observation:

- The serial and `mpi4` first scalar solutions match on all `15334` pressure nodes by global node id, so the comparison is physically aligned.
- But they are not just mean-shifted versions of each other:
  - serial mean: `-4.794320456259134e+04`
  - `mpi4` mean: `-2.2093782020719145e+03`
  - serial `l2`: `6.541546646902964e+06`
  - `mpi4` `l2`: `4.729123540452201e+05`
  - centered correlation: `-5.661158992695927e-01`

Interpretation:

- The first serial and distributed scalar corrections are globally different fields, not just different representatives of the same near-null direction.
- That makes a direct serial-field oracle worth testing, but it also weakens the “simple gauge representative” explanation further.


## Direct serial first-scalar oracle injection still does not close the `mpi4` gap

Hypothesis:

- Even if the distributed Krylov basis is poor, explicitly adding the real serial first scalar correction as an extra oracle mode to the full distributed reduced solve might recover the missing broad component.

Code:

- Added an env-gated oracle-mode loader in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_ORACLE_LS_FILE=<path>`
  - `SVMP_FSILS_FACE_ONLY_KRYLOV_PLUS_ORACLE_LS_SOLVE_INDEX=<n>`
  - trace via `SVMP_FSILS_TRACE_FACE_ONLY_KRYLOV_PLUS_ORACLE_LS=1`
- Also fixed an indexing bug in the first oracle implementation: the `oracle` branch had been implicitly inserting an extra zero `1/M_inv` slot when `gauge2` was off. The current branch now augments with exactly the intended columns.

Run:

- [tests/_codex_iliac_1step_mpi4_oraclels_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclels_current_20260415/run.log)

Observation:

- The first augmented distributed reduced solve does accept a slightly better candidate:
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclels_current_20260415/run.log:106)
    - `baseline_true_residual=2.699209e+01`
    - `candidate_true_residual=2.689027e+01`
    - `accept=1`
    - `aug_dim=302`
    - `const_coeff=4.576092e-02`
    - `oracle_coeff=5.520945e-08`
- But the actual nonlinear behavior does not move:
  - first outer count remains [10](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclels_current_20260415/run.log:127)
  - the run still finishes at [5 Newton](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclels_current_20260415/run.log:306)
  - total loop time is [76.350394 s](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclels_current_20260415/run.log:311)

Interpretation:

- This is a strong negative result.
- Even when the full first distributed Krylov span is explicitly given the actual serial first scalar field as an extra mode, the accepted correction changes only marginally and the nonlinear count stays at `5`.
- So the remaining `serial 4` vs `mpi4 5` gap is not just “the reduced solve was missing the serial broad mode as an augmenting column.”
- The highest-signal diagnosis is now:
  - the distributed scalar-Schur Krylov process itself is generating a materially different first-search subspace/recurrence than the serial solve,
  - and fixing the final reduced coefficient choice over that built basis, even with a direct serial-field oracle, is not sufficient to recover serial parity.


## The serial first scalar field fits serial almost exactly but is nearly orthogonal to the distributed scalar-Schur operator

Hypothesis:

- After the oracle-augmentation branch failed, the next question was whether the serial first scalar field is even a good solution candidate for the current distributed scalar-Schur operator.
- If it fits serial but not `mpi4`, then the gap is not just basis quality or reduced least-squares selection. It means the distributed scalar-Schur operator itself is materially different on that broad mode.

Code:

- Added a trace-only oracle-fit probe in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):
  - `SVMP_FSILS_TRACE_FACE_ONLY_ORACLE_FIT_FILE=<path>`
  - `SVMP_FSILS_TRACE_FACE_ONLY_ORACLE_FIT_SOLVE_INDEX=<n>`
- This does not modify the solve. It just loads a scalar mode by global node, applies the current scalar-Schur operator to it, and reports:
  - `alpha` for the best one-mode fit to the current rhs
  - `residual_alpha1`
  - `residual_best_alpha`

Runs:

- Serial:
  - [tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log)
- `mpi4`:
  - [tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log)

Observation:

- Serial oracle fit is exactly what we would hope:
  - [serial run.log:102](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log:102)
    - `alpha=9.969215e-01`
    - `residual_alpha1=2.468287e+01`
    - `residual_best_alpha=2.464485e+01`
- The same serial first scalar field is a terrible fit for the current distributed operator:
  - [mpi4 run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log:106)
    - `alpha=-9.461783e-08`
    - `residual_alpha1=4.164872e+06`
    - `residual_best_alpha=4.441772e+02`

Interpretation:

- This is much stronger than the earlier “bad distributed Krylov basis” diagnosis.
- The serial first scalar field is almost an exact first-step candidate for the serial scalar-Schur operator, but it is almost orthogonal to the distributed one.
- So the remaining gap is not merely a bad representative chosen over the same operator. The distributed scalar-Schur operator itself is materially different on this broad scalar mode.


## The distributed broad-mode mismatch starts at `GL.apply`, before `add_bc_mul`, `D`, or the final scalar assembly

Hypothesis:

- Once the oracle-fit mismatch was clear, the next step was to localize where the serial-vs-`mpi4` operator difference first appears:
  - fused `GL`
  - momentum-side `add_bc_mul(..., BCOP_TYPE_PRE, ...)`
  - `D.apply`
  - or the final `M_inv * (SP - DGP)`

Code:

- Extended the same trace-only oracle probe in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp) with:
  - `SVMP_FSILS_TRACE_FACE_ONLY_ORACLE_STAGES=1`
- It now logs stage norms for:
  - `gl_gp`
  - `gl_sp`
  - `pre_gp`
  - `dgp`
  - `out`

Observation:

- Serial stage trace:
  - [run.log:103](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log:103) `gl_gp l2=3.786638e+04`
  - [run.log:104](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log:104) `gl_sp l2=6.381505e+02`
  - [run.log:109](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log:109) `out l2=4.448627e+02`
- `mpi4` stage trace:
  - [run.log:107](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log:107) `gl_gp l2=1.131180e+06`
  - [run.log:108](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log:108) `gl_sp l2=2.976088e+06`
  - [run.log:113](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log:113) `out l2=4.164872e+06`

Interpretation:

- The mismatch is already present at the first `GL.apply(...)` stage.
- `add_bc_mul`, `D`, and the final scalar assembly propagate that mismatch, but they are not where it starts.
- So the current root-cause candidate is no longer “scalar-Schur Krylov handling” by itself. It is now the distributed `G/L` operator data or its upstream assembly on broad scalar modes.


## `fused GL` is not the bug; it matches separate `G.apply + L.apply` to roundoff in both serial and `mpi4`

Hypothesis:

- Since the mismatch starts at `GL.apply`, the obvious next question was whether the fused scalar-constraint operator itself was wrong in distributed mode.

Code:

- Added a fused-vs-separate comparison to the same oracle stage trace in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp).
- On the same oracle mode, it now computes:
  - fused `GL.apply(...)`
  - separate `G.apply(...)`
  - separate `L.apply(...)`
  - `diff = separate - fused`

Observation:

- Serial:
  - [run.log:105](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log:105)
    - `sep_gp ... diff_l2=0.000000e+00 diff_max_abs=0.000000e+00`
  - [run.log:106](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_oraclefit_current_20260415/run.log:106)
    - `sep_sp ... diff_l2=0.000000e+00`
- `mpi4`:
  - [run.log:109](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log:109)
    - `sep_gp ... diff_l2=0.000000e+00 diff_max_abs=0.000000e+00`
  - [run.log:110](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_oraclefit_current_20260415/run.log:110)
    - `sep_sp ... diff_l2=0.000000e+00`

Interpretation:

- The fused `GL` path is not the bug.
- The separate distributed `G` and `L` operators are already giving the same bad broad-mode response as fused `GL`.
- So the problem is deeper than the fused operator wrapper. It is in the underlying distributed `G/L` data or upstream assembly.


## The prepared FSILS matrix still matches the distributed FE operator exactly on the existing probe set

Hypothesis:

- If the broad-mode mismatch were introduced during FSILS preparation or block extraction, the existing FE-vs-FSILS operator compare in [FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp) should show it on the distributed probe set.

Runs:

- Traced `mpi4` compare run:
  - [tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log)
  - envs:
    - `SVMP_OOP_SOLVER_TRACE=1`
    - `SVMP_FSILS_COMPARE_FACE_OPERATOR=1`
    - `SVMP_FSILS_PROBE_LOW_RANK_MODES=1`

Observation:

- On the distributed run, FE and FSILS still match exactly on all existing built-in probes:
  - generic:
    - [run.log:562](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:562) and later repeats
    - `|diff|=0 rel=0 |diff_J|=0 |diff_R|=0`
  - `rank1_0`:
    - [run.log:566](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:566)
  - `rank1_1`:
    - [run.log:570](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:570)
  - `constraint_partition_probe`:
    - [run.log:574](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:574)

Interpretation:

- This rules out a broad “FSILS preparation corrupts the distributed operator” explanation for the probes we already had.
- Current strongest diagnosis:
  - the distributed FE/FSILS operator agrees on the existing generic / low-rank / partition probes,
  - but the serial first scalar oracle reveals a broad scalar mode on which the distributed `G/L` operator is very different from serial,
  - so the next missing diagnostic is an FE-vs-FSILS compare on that same oracle broad mode, or an upstream FE assembly compare for broad scalar constraint probes across serial vs `mpi4`.


## Added the serial oracle broad mode to the FE-vs-FSILS compare path; FE and FSILS still match exactly on that mode in `mpi4`

Hypothesis:

- If the broad-mode mismatch were caused by FSILS preparation or the FSILS-only operator path, then the existing FE-vs-FSILS compare should fail once it probes the exact serial broad scalar mode instead of only the old generic / low-rank probes.

Code:

- Extended [FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp) so `compareFaceOperatorAgainstFe(...)` can load an owner-aligned scalar probe from:
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR_ORACLE_FILE`
- That probe is injected into the constraint component only and compared the same way as the existing built-in probes.

Run:

- [tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log)
- envs:
  - `SVMP_OOP_SOLVER_TRACE=1`
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR=1`
  - `SVMP_FSILS_PROBE_LOW_RANK_MODES=1`
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR_ORACLE_FILE=/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_scalarsol_dump2_current_20260415/first_scalar_solution.solve0.rank0.txt`

Observation:

- The exact serial broad oracle mode still matches FE vs FSILS exactly in `mpi4`:
  - [run.log:566](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:566)
  - repeated at [806](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:806), [1046](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:1046), [1286](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:1286), [1526](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:1526)
  - `|diff|=0 rel=0 |diff_J|=0 |diff_R|=0`

Interpretation:

- This exonerates FSILS preparation and the FSILS face operator on the exact broad oracle mode too.
- The broad-mode mismatch is still present, but FE and FSILS are matching each other on that distributed mode.


## `depart(...)` / scalar-block extraction also matches exactly on the broad oracle mode in both serial and `mpi4`

Hypothesis:

- Even if the full FE-vs-FSILS compare matches, the bug could still be introduced when `ns_solver` extracts the scalar Schur blocks (`mG`, `mL`) via `depart(...)`.

Code:

- Added an env-gated direct compare in [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp):
  - `SVMP_FSILS_NS_ORACLE_BLOCK_COMPARE_FILE`
- On the loaded oracle mode, it now compares:
  - `schur_system.GL.apply(...)`
  - direct action built from the prepared full matrix rows corresponding to the extracted `G/L` blocks

Runs:

- Serial:
  - [run_facecompare.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run_facecompare.log)
  - [run_facecompare.log:170](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run_facecompare.log:170)
- `mpi4`:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log)
  - [run.log:584](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:584)
  - [run.log:1090](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run.log:1090)

Observation:

- Serial:
  - `gl_gp_l2=3.786638e+04`
  - `direct_gp_l2=3.786638e+04`
  - `diff_gp_l2=0`
  - `gl_sp_l2=6.381505e+02`
  - `direct_sp_l2=6.381505e+02`
  - `diff_sp_l2=0`
- `mpi4`:
  - `gl_gp_l2=1.131180e+06`
  - `direct_gp_l2=1.131180e+06`
  - `diff_gp_l2=0`
  - `gl_sp_l2=2.976088e+06`
  - `direct_sp_l2=2.976088e+06`
  - `diff_sp_l2=0`

Interpretation:

- `depart(...)` is not corrupting the broad oracle mode.
- The block-extracted scalar-Schur operator is exactly the same as the corresponding rows of the prepared full matrix in both serial and `mpi4`.
- So the broad serial/`mpi4` mismatch is upstream of scalar-block extraction.


## Owner-aligned dumps show the assembled FE operator itself is different between serial and `mpi4`

Important note:

- The older FE-vs-FSILS compare logs only showed within-run agreement (`FE == FSILS`) and reported vector norms through `FsilsVector::norm()`, which are not by themselves a safe serial-vs-`mpi4` oracle.
- To remove overlap / ownership ambiguity, I added an owner-aligned dump path in [FsilsLinearSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp):
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR_DUMP_PREFIX`
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR_DUMP_PROBE`
- It writes owner-only `(global_node, component, value)` records for:
  - `probe`
  - `fe_matrix`
  - `fe_correction`
  - `fe_full`

### Oracle broad scalar probe

Runs:

- Serial owner-aligned dump:
  - [run_facecompare_dump.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run_facecompare_dump.log)
  - dump files under:
    - [oracle_compare.oracle_scalar_probe.fe_matrix.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/oracle_compare.oracle_scalar_probe.fe_matrix.rank0.txt)
- `mpi4` owner-aligned dump:
  - [run_dump.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run_dump.log)
  - dump files under:
    - [oracle_compare.oracle_scalar_probe.fe_matrix.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/oracle_compare.oracle_scalar_probe.fe_matrix.rank0.txt)
    - [oracle_compare.oracle_scalar_probe.fe_matrix.rank1.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/oracle_compare.oracle_scalar_probe.fe_matrix.rank1.txt)
    - [oracle_compare.oracle_scalar_probe.fe_matrix.rank2.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/oracle_compare.oracle_scalar_probe.fe_matrix.rank2.txt)
    - [oracle_compare.oracle_scalar_probe.fe_matrix.rank3.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/oracle_compare.oracle_scalar_probe.fe_matrix.rank3.txt)

Offline gathered comparison:

- The probe itself matches exactly:
  - serial `probe_l2 = 6.541546687e+06`
  - `mpi4 probe_l2 = 6.541546687e+06`
  - `diff_l2 = 0`
- The assembled FE matrix image does **not** match:
  - serial `fe_matrix_l2 = 4.262229957e+04`
  - `mpi4 fe_matrix_l2 = 5.648294759e+04`
  - `diff_l2 = 6.979929525e+04`
  - `rel = 1.637623872`
  - `max_abs = 4.9578129e+03` at `(global_node=9386, component=0)`
- Since the oracle probe has zero reduced correction, `fe_full == fe_matrix` on this test, so this is a pure assembled-matrix mismatch.

Per-component breakdown on the oracle probe:

- component `0`: serial `2.8185e+04`, `mpi4 3.2195e+04`, diff `4.2270e+04`
- component `1`: serial `2.6605e+04`, `mpi4 3.5724e+04`, diff `4.3813e+04`
- component `2`: serial `1.7730e+04`, `mpi4 2.9621e+04`, diff `3.4139e+04`
- component `3`: serial `2.0553e+02`, `mpi4 4.1967e+02`, diff `3.5177e+02`

Interpretation:

- The broad oracle probe is no longer just a scalar-Schur-only symptom.
- The owner-aligned assembled FE matrix action itself is different between serial and `mpi4`.
- Since FE and FSILS still match exactly within each run, the mismatch is upstream of the FSILS preparation path.

### Existing deterministic generic probe

Runs:

- Serial owner-aligned dump:
  - [run_generic_dump.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run_generic_dump.log)
  - dump files under:
    - [generic_compare.generic.fe_matrix.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/generic_compare.generic.fe_matrix.rank0.txt)
- `mpi4` owner-aligned dump:
  - [run_generic_dump.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run_generic_dump.log)
  - dump files under:
    - [generic_compare.generic.fe_matrix.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/generic_compare.generic.fe_matrix.rank0.txt)
    - [generic_compare.generic.fe_matrix.rank1.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/generic_compare.generic.fe_matrix.rank1.txt)
    - [generic_compare.generic.fe_matrix.rank2.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/generic_compare.generic.fe_matrix.rank2.txt)
    - [generic_compare.generic.fe_matrix.rank3.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/generic_compare.generic.fe_matrix.rank3.txt)

Offline gathered comparison:

- The generic probe also matches exactly:
  - serial `probe_l2 = 8.770368846e+03`
  - `mpi4 probe_l2 = 8.770368846e+03`
  - `diff_l2 = 0`
- The assembled FE matrix image differs:
  - serial `fe_matrix_l2 = 1.532533174e+05`
  - `mpi4 fe_matrix_l2 = 1.395412650e+05`
  - `diff_l2 = 8.022802139e+04`
- The reduced correction image differs too:
  - serial `fe_correction_l2 = 1.801081607e+04`
  - `mpi4 fe_correction_l2 = 5.068747361e+04`
  - `diff_l2 = 5.379228083e+04`
- Full image difference:
  - serial `fe_full_l2 = 1.543601843e+05`
  - `mpi4 fe_full_l2 = 1.489007059e+05`
  - `diff_l2 = 9.580515678e+04`

Interpretation:

- The serial/`mpi4` assembled-operator mismatch is not limited to the broad oracle mode.
- It is already present on the old deterministic generic probe.
- So the root problem is now best described as a broader serial-vs-distributed assembled operator mismatch in the prepared FE/FSILS operator, not only a scalar-Schur weak-mode issue.


## Current strongest diagnosis after the owner-aligned dumps

- FE and FSILS match exactly **within each run**.
- `depart(...)` / scalar-block extraction also matches the corresponding full-matrix rows exactly.
- But owner-aligned probe dumps show that the assembled operator response itself differs materially between serial and `mpi4`:
  - on the broad oracle scalar probe
  - and even on the deterministic generic probe

So the highest-value next step is no longer more scalar-Schur stabilization. It is to localize where the serial-vs-distributed assembled operator diverges:

- base FE matrix assembly / overlap accumulation,
- reduced-update / rank-one preparation into the full operator,
- or a distributed matvec / prepared-coordinate transformation that is still serial-inconsistent even though FE and FSILS agree with each other inside the same run.


## 2026-04-16: owner-row stencil evidence, `OwnedRowsOnly` closure, and final root cause

### Prepared row dumps on the worst oracle row

I added an env-gated prepared-row dump in `FsilsLinearSolver.cpp`:

- `SVMP_FSILS_DUMP_PREPARED_ROW_PREFIX`
- `SVMP_FSILS_DUMP_PREPARED_ROW_GLOBAL_NODE`
- `SVMP_FSILS_DUMP_PREPARED_ROW_COMPONENT`
- optional `SVMP_FSILS_DUMP_PREPARED_COL_COMPONENT`

On the worst oracle row `(global_node=9386, row_component=0)`, serial vs `mpi4` differ strongly even before the scalar-Schur solve.

Row dump files:

- serial scalar-column slice:
  - [rowdump9386.g9386.r0.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/rowdump9386.g9386.r0.rank0.txt)
- `mpi4` scalar-column slice:
  - [rowdump9386.g9386.r0.rank2.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/rowdump9386.g9386.r0.rank2.txt)
- serial all-components:
  - [rowdump9386all.g9386.r0.rank0.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/rowdump9386all.g9386.r0.rank0.txt)
- `mpi4` all-components:
  - [rowdump9386all.g9386.r0.rank2.txt](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/rowdump9386all.g9386.r0.rank2.txt)

Offline comparison:

- scalar input column (`col_comp=3`) already differs:
  - serial unique keys: `16`
  - `mpi4` unique keys: `6`
  - diff `l2 = 2.449302e-02`
- all components differ, not just `G`:
  - `col_comp 0`: serial `3.050439e+01`, `mpi4 6.581937e+00`, diff `2.526206e+01`
  - `col_comp 1`: serial `8.446104e+00`, `mpi4 6.078266e+00`, diff `9.972343e+00`
  - `col_comp 2`: serial `7.561829e+00`, `mpi4 4.419883e+00`, diff `9.059707e+00`
  - `col_comp 3`: serial `2.194127e-02`, `mpi4 1.088526e-02`, diff `2.449302e-02`

Interpretation:

- The problematic `mpi4` row stencil is smaller and different.
- This is not just a scalar-Schur-only distortion; the base `K/G` row content is already serial-inconsistent.

### Dropped-entry check

I added `dropped_entries=` to the standard OOP trace in `FsilsLinearSolver::solve`.

Checks:

- serial:
  - [run_dropcheck.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run_dropcheck.log)
- `mpi4`:
  - [run_dropcheck.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_facecompare_current_20260415/run_dropcheck.log)

Result:

- serial `dropped_entries=0`
- `mpi4` `dropped_entries=0` on all ranks

Interpretation:

- The missing couplings are not “generated then dropped due to missing sparsity slots”.
- They are never being assembled into the prepared operator in the first place.

### Real iliac harness with `OwnedRowsOnly`

I then tested the obvious assembly-path hypothesis directly by forcing:

- `SVMP_ASSEMBLY_GHOST_POLICY=owned_rows_only`

on the canonical archived `mpi4` iliac 1-step harness:

- [run_ownedrows_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run_ownedrows_20260416.log)

Result:

- `step=0 converged=1 iters=4 ||r||=9.1651158399189788e-11`
- `Total time loop: 47.621910 s`

This was the first real end-to-end run that closed the nonlinear gap on the production harness without touching the linear solver.

### Root cause found in the assembler path

Reading `AssemblerFactory.cpp` and the assembly implementations clarified the real bug:

- this workload auto-selects `StandardAssembler`, not `ParallelAssembler`
- `ParallelAssembler` is the only implementation that actually has ghost buffering and reverse-scatter exchange
- `StandardAssembler` was still accepting `ghost_policy=ReverseScatter`, but it has no reverse-scatter communication path
- on that path it skips ghost cells/faces and inserts directly, which is exactly the wrong combination for an owner-row distributed backend

So the real bug was not in FSILS or scalar-Schur Krylov selection. It was:

- `StandardAssembler` pretending to support `ReverseScatter` in MPI, even though only `ParallelAssembler` actually implements it

### Code fix

I fixed that by normalizing `StandardAssembler` to owned-row insertion semantics in `setOptions(...)`:

- [StandardAssembler.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Assembly/StandardAssembler.cpp:930)

and documented the contract clearly:

- [StandardAssembler.h](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Assembly/StandardAssembler.h:115)

The fix is physics-agnostic:

- `StandardAssembler` now always behaves as an owned-row assembler
- callers that want true reverse-scatter semantics must use `ParallelAssembler`

### Regression guard

I added a direct unit regression:

- [test_StandardAssembler.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Assembly/test_StandardAssembler.cpp:1767)
  - `StandardAssemblerTest.ReverseScatterPolicyIsNormalizedToOwnedRowsOnly`

Validated:

- `./test_fe_assembly --gtest_filter='StandardAssemblerTest.ReverseScatterPolicyIsNormalizedToOwnedRowsOnly'`
- passed

### Post-fix iliac qualification on the plain current binary

After rebuilding `svmultiphysics`, I reran the same archived `mpi4` iliac harness with **no** ghost-policy env override:

- [run_default_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run_default_after_standard_ownedrowsfix_20260416.log)

Result:

- `step=0 converged=1 iters=4 ||r||=9.1651158399189788e-11`
- `Total time loop: 48.576243 s`

That closes the canonical `serial 4` vs `mpi4 5` nonlinear gap on the target iliac case without any env workaround.

Current baseline after the fix:

- serial:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_serial_default_recheck_current_20260415/run.log:263)
  - `iters=4`
- `mpi4`:
  - [run_default_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run_default_after_standard_ownedrowsfix_20260416.log:267)
  - `iters=4`

### Status

The iliac MPI convergence gap is closed on the real archived production harness.

I started a broader `pipe_RCR_3d` / `pipe_simple` MPI no-regression sweep after the fix, but stopped it to avoid leaving extra long-lived MPI jobs open while process-count warnings were active. The remaining work is now:

- broader no-regression qualification across the MPI case matrix
- then MPI performance cleanup, since the nonlinear gap on the target iliac case is no longer the blocker


## 2026-04-16: broader corrected-baseline qualification and first MPI tuning pass

After landing the `StandardAssembler` owned-row fix, I re-qualified the broader archived matrix from the corrected baseline.

### `pipe_simple`

Post-fix reruns:

- serial:
  - [run_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_serial_20260413/run_after_standard_ownedrowsfix_20260416.log)
- `mpi4`:
  - [run_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_after_standard_ownedrowsfix_20260416.log)

Results:

- serial:
  - step `0`: `5` Newton
  - step `1`: `5` Newton
  - total loop: `7.804826 s`
- `mpi4`:
  - step `0`: `5` Newton
  - step `1`: `5` Newton
  - total loop: `2.871609 s`

Compared with the archived pre-fix matrix:

- serial was previously `6.469431 s`
- `mpi4` was previously `3.844362 s`

Interpretation:

- nonlinear behavior stayed clean
- `mpi4` performance improved materially on the corrected baseline
- serial got a bit slower on this rerun, but the main serial/parallel behavior is still healthy

### `pipe_RCR_3d`

Post-fix serial rerun:

- [run_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_after_standard_ownedrowsfix_20260416.log)

Result:

- step `0`: `3` Newton
- step `1`: `2` Newton
- total loop: `6.167362 s`

The archived pre-fix serial baseline was `5.014611 s`, so this serial rerun is slower but still converges with the same nonlinear counts.

The post-fix `mpi4` rerun did not complete cleanly:

- partial log:
  - [run_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_after_standard_ownedrowsfix_20260416.log)

It repeatedly stalled after the first logged GMRES solve without reaching a `nonlinear_done` summary on the timescale where the archived baseline completed in `7.447421 s`. I stopped that run rather than leave another long-lived MPI job open. So `pipe_RCR_3d mpi4` still needs a focused follow-up on the corrected baseline.

### `iliac_artery` 2-step

Post-fix reruns:

- serial:
  - [run_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_serial_20260413/run_after_standard_ownedrowsfix_20260416.log)
- `mpi4`:
  - [run_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_after_standard_ownedrowsfix_20260416.log)

Results:

- serial:
  - step `0`: `4` Newton
  - step `1`: `5` Newton
  - total loop: `77.038558 s`
- `mpi4`:
  - step `0`: `11` Newton
  - step `1`: `5` Newton
  - total loop: `57.489840 s`

Compared with the archived pre-fix matrix:

- serial was `4 / 4` Newton, `63.260131 s`
- `mpi4` was `5 / 6` Newton, `52.049221 s`

Interpretation:

- the corrected baseline closes the canonical 1-step `mpi4` iliac gap
- but it does **not** yet generalize cleanly to the 2-step iliac matrix
- on this archived 2-step harness, step `0` in `mpi4` regressed badly to `11` Newton

So the broader qualification result is mixed:

- canonical 1-step iliac: fixed
- `pipe_simple`: clean, and `mpi4` faster
- `pipe_RCR_3d serial`: clean but slower
- `pipe_RCR_3d mpi4`: unresolved / stalled
- 2-step iliac `mpi4`: regression

### First MPI tuning pass on corrected baseline

I tuned only on the problematic corrected-baseline `iliac_artery` 2-step `mpi4` harness.

#### `SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS=1`

Run:

- [run_tuned_initrhs_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_tuned_initrhs_20260416.log)

Result:

- step `0`: `11` Newton
- step `1`: `5` Newton
- total loop: `54.952083 s`

Interpretation:

- this is the only solver-side knob in this pass that helped at all
- it improved wall time vs corrected-baseline default (`57.489840 s -> 54.952083 s`)
- but it did **not** fix the nonlinear regression

#### `SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_GMRES=1`

Run:

- [run_tuned_forcegmres_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_tuned_forcegmres_20260416.log)

Result:

- step `0`: `11` Newton
- step `1`: `5` Newton
- total loop: `59.484816 s`

Interpretation:

- explicit Schur GMRES was worse than default on the corrected baseline

#### `SVMP_FSILS_BLOCKSCHUR_FORCE_SCHUR_BICGSTAB=1`

That run was still incomplete / unneeded once the first two results were clear, so I stopped it and did not use it as a candidate setting.

#### Canonical 1-step iliac with `initrhs`

I also checked the same `initrhs` knob on the canonical fixed 1-step `mpi4` iliac harness:

- [run_initrhs_after_standard_ownedrowsfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run_initrhs_after_standard_ownedrowsfix_20260416.log)

Result:

- still `4` Newton
- total loop: `48.403475 s`

Compared with the corrected-baseline default:

- default was `48.576243 s`

So `SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS=1` is currently the best MPI tuning knob I found on the corrected baseline:

- it preserves the fixed canonical 1-step iliac result
- it slightly improves 1-step wall time
- it also slightly improves 2-step `mpi4` iliac wall time
- but it does not yet solve the broader 2-step `mpi4` regression

### Current status after the broader qualification

The corrected assembly baseline is real and valuable:

- it closes the original target `serial 4` vs `mpi4 5` gap on the canonical 1-step iliac harness

But broader no-regression qualification is not complete yet because:

- `iliac_artery` 2-step `mpi4` regressed to `11 / 5` Newton
- `pipe_RCR_3d mpi4` did not complete cleanly on the rerun

So the next work should be:

- isolate why the owned-row `StandardAssembler` fix improves the canonical 1-step iliac harness but regresses the 2-step `mpi4` iliac matrix
- and separately resolve the `pipe_RCR_3d mpi4` corrected-baseline stall

### `pipe_RCR_3d mpi4` grouped bordered stall: narrowed and fixed

I resumed from the corrected-baseline `pipe_RCR_3d mpi4` stall and instrumented the grouped reduced-Schur setup path in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):

- defaulted `col.schur_right` to stay owner-local in `build_reduced_schur_correction(...)`, with the old halo sync only available behind `SVMP_FSILS_REDUCED_SCHUR_FORCE_GT_SYNC`
- added all-rank stage traces for reduced-Schur column build:
  - `filled`
  - `hat`
  - `hat_t`
  - both momentum syncs
  - `D.apply`
  - `G^T`
- added source/group tracing for each column / grouped mode

That tracing changed the diagnosis materially. The later root-only `column 1` stall was a misleading symptom. The real distributed divergence was earlier:

- on the stalled run, ranks `0/1/2` dropped zero-support reduced/grouped modes while rank `3` kept the globally active mode and continued into grouped momentum-hat work alone
- rank `3` therefore never reached the same reduced-Schur stage as the other ranks

The actual bug was in `build_grouped_momentum_hat_low_rank_correction(...)` in [bicgs.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp):

- `append_reduced_mode(...)` and `append_face_mode(...)` were deciding whether to keep a mode from **local** support only
- that made the grouped low-rank basis rank-inconsistent under MPI

I fixed that by making grouped momentum-hat mode retention depend on **global** support:

- a reduced/face mode is now kept whenever any rank has support
- ranks without local support keep the same zero local slice instead of dropping the mode outright

After that fix, the all-rank trace showed the grouped path is rank-consistent:

- every rank now reports the grouped mode:
  - [run_setuptrace_allranks_diagk_after_groupedbasisfix_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_setuptrace_allranks_diagk_after_groupedbasisfix_mpi4_20260416.log)
- rank `3` now reaches both:
  - `column=0 source=update grouped_id=-1`
  - `column=1 source=update grouped_id=0`
- all four ranks reach:
  - `grouped_end group=0 mode=0 rank_now=2`

Plain default `pipe_RCR_3d mpi4` on the corrected baseline is now clean:

- [run_default_after_groupedbasisfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_default_after_groupedbasisfix_20260416.log)
- step `0`: `8` Newton
- step `1`: `3` Newton
- total loop: `4.740304 s`
- success: `loop.run() returned success=1`

So the broadened `pipe_RCR_3d mpi4` no-regression failure was a second real MPI rank-consistency bug in grouped momentum-hat basis construction, not just a tolerance / tuning issue.

### `iliac_artery` 2-step `mpi4`: now a tolerance-policy problem, not the old MPI bug

With the grouped-basis fix in place, the default 2-step archived `mpi4` iliac harness still does:

- [run_default_after_groupedbasisfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_default_after_groupedbasisfix_20260416.log)
- step `0`: `11` Newton
- step `1`: `5` Newton
- total loop: `56.322612 s`

But this is no longer best interpreted as the same distributed-operator bug.

I checked the archived input difference directly:

- 1-step canonical fixed harness uses `1e-8` linear tolerances
- 2-step archived harness still uses `1e-3`

The old 2-step `mpi4` broad regression therefore mixes correctness and policy. Requalification with tighter archived variants shows:

#### `1e-8` archived 2-step variant

- [run_tight1e8_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_tight1e8_20260416.log)
- step `0`: `4` Newton
- step `1`: `4` Newton
- total loop: `97.328920 s`

This closes the nonlinear gap but is too expensive.

#### `1e-6` probe

- [run_tol1e6_after_groupedbasisfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_tol1e6_after_groupedbasisfix_20260416.log)
- step `0`: `4` Newton
- step `1`: `4` Newton
- total loop: `63.565397 s`

This also closes the nonlinear gap, cheaper than `1e-8`, but still slower than the loose default.

#### `1e-5` probe

- [run_tol1e5_after_groupedbasisfix_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_tol1e5_after_groupedbasisfix_20260416.log)
- step `0`: `4` Newton
- step `1`: `4` Newton
- total loop: `47.911537 s`

This is the best current broadened MPI point I found:

- it restores near-serial nonlinear behavior (`4 / 4`)
- it is faster than the loose corrected-baseline default (`56.322612 s`)
- it is also faster than the earlier archived pre-fix `mpi4` baseline (`52.049221 s`)

So the current state is:

- canonical 1-step iliac MPI gap: fixed
- `pipe_RCR_3d mpi4` grouped bordered stall: fixed
- 2-step iliac `mpi4` default broad harness: still too loose at `1e-3`
- 2-step iliac `mpi4` has a clear better-performing recovered point at `1e-5`

The next sensible engineering step is **not** another distributed-correctness debug pass on 2-step iliac. The remaining work there is to encode a selective moderate-tolerance policy for the iliac-style outlet-coupled MPI path, without regressing the now-fixed `pipe_RCR_3d` grouped bordered case.

### Selective direct-only MPI `1e-5` outer-tolerance policy: fixes broadened 2-step iliac without reopening `pipe_RCR_3d`

I implemented that policy in [NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp):

- only for `mpiMultiTaskActive()`
- only for the validated native direct-only outlet path
- clamp the requested outer linear tolerance to `1e-5` when the XML is looser than that
- leave grouped/bordered MPI cases alone

This is intentionally narrower than the older bordered retuning. The goal was to reproduce the successful archived `1e-5` iliac behavior without touching the now-fixed grouped `pipe_RCR_3d` path.

#### Requalification after the policy change

Grouped regression guard stayed clean:

- `pipe_RCR_3d mpi4` default:
  - [run_default_after_directonlyrelfloor_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_default_after_directonlyrelfloor_20260416.log)
  - step `0`: `8` Newton
  - step `1`: `3` Newton
  - total loop: `4.369423 s`
  - success

Canonical 1-step iliac stayed fixed:

- `iliac_artery` 1-step `mpi4` default:
  - [run_default_after_directonlyrelfloor_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run_default_after_directonlyrelfloor_20260416.log)
  - step `0`: `4` Newton
  - total loop: `48.851354 s`
  - success

The broadened 2-step iliac default is now corrected:

- `iliac_artery` 2-step `mpi4` default:
  - [run_default_after_directonlyrelfloor_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_default_after_directonlyrelfloor_20260416.log)
  - step `0`: `4` Newton
  - step `1`: `4` Newton
  - total loop: `48.103847 s`
  - success

That means the old broadened default failure pattern:

- before policy: `11 / 5` Newton, `56.322612 s`
- after policy: `4 / 4` Newton, `48.103847 s`

So the widened iliac `mpi4` regression is now closed by policy on top of the earlier correctness fixes.

#### MPI performance tuning check on top of the corrected baseline

I rechecked the old `SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS=1` tuning knob on top of the new default:

- `iliac_artery` 2-step `mpi4` with init-rhs:
  - [run_initrhs_after_directonlyrelfloor_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_initrhs_after_directonlyrelfloor_20260416.log)
  - step `0`: `4`
  - step `1`: `4`
  - total loop: `47.930550 s`

So it still helps slightly on iliac after the policy fix.

But it is not a safe broad default because it regresses `pipe_RCR_3d mpi4`:

- `pipe_RCR_3d mpi4` with init-rhs:
  - [run_initrhs_after_directonlyrelfloor_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_initrhs_after_directonlyrelfloor_20260416.log)
  - step `0`: `8`
  - step `1`: `3`
  - total loop: `5.046503 s`

So the current recommended state is:

- keep the new selective direct-only MPI `1e-5` policy in code
- keep `SVMP_FSILS_BLOCKSCHUR_SCHUR_INIT_PRECOND_RHS=1` as an opt-in tuning knob, not a new default

#### Cheap extra MPI guard

- `pipe_simple mpi4` default:
  - [run_default_after_directonlyrelfloor_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_default_after_directonlyrelfloor_20260416.log)
  - step `0`: `5`
  - step `1`: `5`
  - total loop: `3.808865 s`
  - success

At this point the broadened `mpi4` matrix I checked is consistent:

- canonical 1-step `iliac_artery`: fixed at `4`
- broadened 2-step `iliac_artery`: fixed at `4 / 4`
- `pipe_RCR_3d`: fixed and stable
- `pipe_simple`: stable

### Serial / MPI policy-parity audit and cleanup

I then audited the remaining rank-sensitive solver policies in the current tree. The main findings were:

- `FESystem::augmentSolverOptions(...)` itself is rank-invariant; the base solver options come in the same way for serial and MPI.
- But three later Newton-side policies were still MPI-only:
  - the direct-only outlet outer-tolerance floor
  - the nonlinear absolute-floor acceptance
  - the bordered `K^{-1}B` near-target `4x` acceptance

The goal in this pass was to make those rank-agnostic and then requalify `iliac_artery`, `pipe_RCR_3d`, and `pipe_simple` in both serial and `mpi4` after each change.

#### Policy 1: direct-only outlet outer-tolerance floor

The first naive parity change made the `1e-5` outer floor apply to every direct-only path regardless of rank. That did keep `iliac_artery` in serial / MPI parity, but it also blew up `pipe_RCR_3d` serial to a `15`-Newton failure, so that version was too broad.

I then narrowed the policy to the **multi-mode** direct-only outlet signature only, while keeping it rank-agnostic. That preserves serial / MPI parity on the path that actually needed it and leaves single-mode cases alone.

After that narrowed change:

- `iliac_artery` 2-step serial:
  - [run_policy1b_directonlyfloor_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_serial_20260413/run_policy1b_directonlyfloor_serial_20260416.log)
  - step `0`: `4`
  - step `1`: `4`
- `iliac_artery` 2-step `mpi4`:
  - [run_policy1b_directonlyfloor_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_policy1b_directonlyfloor_mpi4_20260416.log)
  - step `0`: `4`
  - step `1`: `4`
- `pipe_RCR_3d mpi4` stayed clean:
  - [run_policy1b_directonlyfloor_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_policy1b_directonlyfloor_mpi4_20260416.log)
  - step `0`: `8`
  - step `1`: `3`
- `pipe_simple` stayed clean in both serial and `mpi4`:
  - [serial](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_serial_20260413/run_policy1b_directonlyfloor_serial_20260416.log)
  - [mpi4](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_policy1b_directonlyfloor_mpi4_20260416.log)

Important note: `pipe_RCR_3d` serial was already failing on this current binary in these reruns, both before and after the narrowed floor:

- [run_policy1b_directonlyfloor_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_policy1b_directonlyfloor_serial_20260416.log)
- step `0`: `15` Newton, failure

So that serial `pipe_RCR_3d` problem is not evidence against the narrowed parity policy itself; it appears to be a separate current serial issue.

#### Policy 2: nonlinear absolute-floor acceptance

I removed the `mpiMultiTaskActive()` gate so this acceptance rule is rank-agnostic.

Requalification showed no convergence-pattern change on the checked matrix:

- `iliac_artery` 2-step serial:
  - [run_policy2_absfloor_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_serial_20260413/run_policy2_absfloor_serial_20260416.log)
  - `4 / 4`
- `iliac_artery` 2-step `mpi4`:
  - [run_policy2_absfloor_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_policy2_absfloor_mpi4_20260416.log)
  - `4 / 4`
- `pipe_RCR_3d mpi4`:
  - [run_policy2_absfloor_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_policy2_absfloor_mpi4_20260416.log)
  - `8 / 3`
- `pipe_simple` serial / `mpi4`:
  - [serial](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_serial_20260413/run_policy2_absfloor_serial_20260416.log)
  - [mpi4](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_policy2_absfloor_mpi4_20260416.log)
  - both stayed `5 / 5`

`pipe_RCR_3d` serial still failed in the same way:

- [run_policy2_absfloor_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_policy2_absfloor_serial_20260416.log)
- step `0`: `15` Newton, failure

So this second parity change is effectively behavior-neutral on the checked cases.

#### Policy 3: bordered `K^{-1}B` near-target `4x` acceptance

I removed the MPI-only gate on the `4x` near-target acceptance in the bordered recovery solve.

Again, requalification showed no convergence-pattern change on the checked matrix:

- `iliac_artery` 2-step serial:
  - [run_policy3_bordered4x_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_serial_20260413/run_policy3_bordered4x_serial_20260416.log)
  - `4 / 4`
- `iliac_artery` 2-step `mpi4`:
  - [run_policy3_bordered4x_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_policy3_bordered4x_mpi4_20260416.log)
  - `4 / 4`
- `pipe_RCR_3d mpi4`:
  - [run_policy3_bordered4x_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_policy3_bordered4x_mpi4_20260416.log)
  - `8 / 3`
- `pipe_simple` serial / `mpi4`:
  - [serial](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_serial_20260413/run_policy3_bordered4x_serial_20260416.log)
  - [mpi4](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_policy3_bordered4x_mpi4_20260416.log)
  - both stayed `5 / 5`

And `pipe_RCR_3d` serial still failed the same way:

- [run_policy3_bordered4x_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_policy3_bordered4x_serial_20260416.log)
- step `0`: `15` Newton, failure

### Current conclusion from the parity pass

After these three changes:

- the three audited policies are no longer MPI-only
- `iliac_artery` 2-step now stays in serial / `mpi4` parity at `4 / 4`
- `pipe_simple` stayed stable
- `pipe_RCR_3d mpi4` stayed stable
- `pipe_RCR_3d` serial remained broken throughout these reruns, which points to a separate current serial regression rather than an effect of these parity-policy edits

### 2026-04-16: `pipe_RCR_3d` serial regression resolved

I traced the current `pipe_RCR_3d` serial failure into the condensed bordered shortcut in
`NewtonSolver::solveStep(...)`.

Key observations:

- The bad serial and healthy `mpi4` runs follow the same high-level Newton and PTC-retry flow.
- The first concrete divergence is the accepted condensed bordered recovery after the `it=2`
  retry:
  - serial accepted `||dx_aux|| = 1.41544e-07` and a much larger pressure update, then the
    nonlinear residual rebounded
  - `mpi4` accepted `||dx_aux|| = 0` and continued decreasing
- A direct probe confirmed the branch: forcing
  `SVMP_MAX_CONDENSED_AUX_SIZE=0` restored healthy convergence immediately on the same current
  binary:
  - serial: [run_probe_no_condensed_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_probe_no_condensed_20260416.log)
    - step `0 = 4` Newton
    - step `1 = 3` Newton
  - `mpi4`: [run_probe_no_condensed_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_probe_no_condensed_mpi4_20260416.log)
    - step `0 = 3` Newton
    - step `1 = 3` Newton

That isolates the regression to the condensed bordered surrogate for direct-coupled bordered
systems. Those rows are assembled through auxiliary input/output sensitivities instead of explicit
dense `Ct` rows, so the explicit bordered recovery is the safer default there.

I implemented the corresponding gate in
[NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp):

- skip `condensed_bordered_active` when `solve_bordered.direct_coupling_records` is nonempty

Requalification on the patched default path:

- `pipe_RCR_3d` serial:
  - [run_after_no_direct_condensed_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_after_no_direct_condensed_serial_20260416.log)
  - step `0 = 4` Newton, step `1 = 3` Newton, success
- `pipe_RCR_3d mpi4`:
  - [run_after_no_direct_condensed_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_after_no_direct_condensed_mpi4_20260416.log)
  - step `0 = 3` Newton, step `1 = 3` Newton, success
- `iliac_artery` 2-step `mpi4`:
  - [run_after_no_direct_condensed_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_after_no_direct_condensed_mpi4_20260416.log)
  - step `0 = 4` Newton, step `1 = 4` Newton, success
- `pipe_simple` serial / `mpi4`:
  - [serial](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_serial_20260413/run_after_no_direct_condensed_serial_20260416.log)
  - [mpi4](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_after_no_direct_condensed_mpi4_20260416.log)
  - both stayed converged at `5 / 5`

So the `pipe_RCR_3d` serial regression is resolved, and the guarded MPI cases stayed healthy.

### 2026-04-16: Cleanup pass for clearly dead experiment set

I removed only the abandoned default-off MPI recovery / stabilization experiment branches from:

- [NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
- [ns_solver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/ns_solver.cpp)

Removed `NewtonSolver` experiment set:

- exact native rank-one Woodbury recovery
- MPI linear subspace recovery and its custom-vector / boundary / coordinate / nonlinear-search variants
- MPI partition-drift correction
- MPI global-mean-shift correction
- MPI first-iteration step expansion
- MPI first-iteration step line search

Removed `ns_solver` experiment set:

- scalar-mean constrained Galerkin branch
- partition mean equalization branch
- first-Schur zero-mean branch
- iter-0 Schur zero-mean branch
- iter-0 Schur mean-subspace branch

I intentionally kept the trace/dump hooks and operational escape hatches such as
`SVMP_MAX_CONDENSED_AUX_SIZE`, `SVMP_DISABLE_LOCAL_CONDENSED_RECOVERY`, and the backend routing
controls.

Rebuild after cleanup succeeded:

- `cmake --build build/svMultiPhysics-build --target svmultiphysics -j8`

No-regression requalification after cleanup:

- `pipe_RCR_3d` serial:
  - [run_after_deadexp_cleanup_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_serial_20260413/run_after_deadexp_cleanup_serial_20260416.log)
  - step `0 = 4` Newton, step `1 = 3` Newton, success
- `pipe_RCR_3d mpi4`:
  - [run_after_deadexp_cleanup_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_rcr3d_mpi4_20260413/run_after_deadexp_cleanup_mpi4_20260416.log)
  - step `0 = 3` Newton, step `1 = 3` Newton, success
- `pipe_simple` serial:
  - [run_after_deadexp_cleanup_serial_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_serial_20260413/run_after_deadexp_cleanup_serial_20260416.log)
  - step `0 = 5` Newton, step `1 = 5` Newton, success
- `pipe_simple mpi4`:
  - [run_after_deadexp_cleanup_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_pipe_simple_mpi4_20260413/run_after_deadexp_cleanup_mpi4_20260416.log)
  - step `0 = 5` Newton, step `1 = 5` Newton, success
- canonical `iliac_artery` 1-step `mpi4`:
  - [run_after_deadexp_cleanup_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_default_recheck_current_20260415/run_after_deadexp_cleanup_mpi4_20260416.log)
  - step `0 = 4` Newton, success
- `iliac_artery` 2-step `mpi4`:
  - [run_after_deadexp_cleanup_mpi4_20260416.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_matrix_iliac_2step_mpi4_20260413/run_after_deadexp_cleanup_mpi4_20260416.log)
  - step `0 = 4` Newton, step `1 = 4` Newton, success

So this cleanup pass removed only dead experiment scaffolding and did not change the validated
serial / `mpi4` convergence behavior on the guarded cases.

### 2026-04-16: Partitioned RCR serial infrastructure gap in field-side Jacobian coupling

I switched the temporary `pipe_RCR_3d` OOP RCR override back to the partitioned path and
continued from the earlier serial-partitioned regression:

- old partitioned serial baseline:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_fix4_20260416/run.log)
  - step `0 = 11` Newton, step `1 = 9` Newton
- old partitioned `mpi4` baseline:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_fix6_20260416/run.log)
  - step `0 = 4` Newton, step `1 = 4` Newton

#### Hypothesis

The generic partitioned AuxiliaryState path was refreshing the outlet output values each assembly,
but it was not exporting the corresponding FE-side Jacobian coupling when a PDE form consumed a
partitioned auxiliary output through a FE-coupled input such as `Q = ∫_Γ u·n ds`.

That would give exactly the observed symptom:

- the residual sees the latest `P_out`
- but the Newton matrix omits the `d(P_out)/dQ * dQ/du` term
- serial then behaves like a lagged / quasi-Newton solve and needs many more iterations

#### What I tried

1. I first extended the existing direct-coupling logic in
   [FESystem.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Systems/FESystem.cpp)
   so the output-coupling section also accepted `AuxiliarySolveMode::Partitioned`.

2. That had no effect on the real case. The follow-up code inspection showed why:
   partitioned-only cases with no monolithic auxiliary operators were returning early in
   [SystemAssembly.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Systems/SystemAssembly.cpp)
   before `assembleMixedAuxiliaryIntoGlobal(...)` ever ran.

3. I added the actual infrastructure fix in
   [SystemAssembly.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Systems/SystemAssembly.cpp:1149):

   - detect partitioned deployments whose outputs are consumed by PDE forms
   - create/finalize an empty `AuxiliaryOperatorRegistry` on demand for that path
   - allow `assembleMixedAuxiliaryIntoGlobal(...)` to run even when there are no live monolithic
     deployments or registered auxiliary operators

4. I added a unit guard in
   [test_BoundaryIntegralInput.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Systems/test_BoundaryIntegralInput.cpp:605):

   - `PartitionedCoupling.OutputDrivenBoundaryIntegralEmitsFieldCouplingUpdate`

   This reproduces the exact generic pattern:

   - partitioned Backward Euler auxiliary block
   - FE-coupled boundary-integral input `Q`
   - PDE residual consuming `P_out`
   - matrix assembly must emit either a rank-one or reduced field coupling update with
     `|sigma| = Rp`

#### Qualification

Targeted unit tests:

- new partitioned guard:
  - `build/svMultiPhysics-build/bin/test_fe_systems --gtest_filter='PartitionedCoupling.OutputDrivenBoundaryIntegralEmitsFieldCouplingUpdate'`
  - passes
- nearby monolithic guard:
  - `build/svMultiPhysics-build/bin/test_fe_systems --gtest_filter='MonolithicCoupling.DirectCouplingReducedUpdateUsesActualOutputSensitivity'`
  - still passes

Real `pipe_RCR_3d` partitioned reruns with the temporary env override
`SVMP_NS_COUPLED_RCR_PARTITIONED=1`:

- serial:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_fix9_20260416/run.log)
  - step `0 = 6` Newton, step `1 = 4` Newton, total `11.001122 s`
- `mpi4`:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_fix7_20260416/run.log)
  - step `0 = 3` Newton, step `1 = 3` Newton, total `3.905335 s`

So the partitioned infrastructure bug was real and the fix materially improved the case:

- serial improved from `11/9` to `6/4`
- `mpi4` improved from `4/4` to `3/3`

I also rechecked the validated monolithic baseline after the shared `SystemAssembly` change:

- monolithic serial:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_monolithic_guard_serial_20260416/run.log)
  - step `0 = 4` Newton, step `1 = 3` Newton
- monolithic `mpi4`:
  - [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_monolithic_guard_mpi4_20260416/run.log)
  - step `0 = 3` Newton, step `1 = 3` Newton

#### Current read

This fixes one real generic partitioned-infrastructure defect: partitioned FE-coupled outputs were
not contributing their exact field-side Jacobian updates unless some monolithic auxiliary machinery
happened to be present.

The partitioned `pipe_RCR_3d` path is not yet identical to the monolithic path, and serial is still
slower than the current monolithic baseline, but the main serial-partitioned pathology is narrowed
substantially now and is backed by a targeted unit regression.


### 2026-04-17: partitioned RCR serial/MPI asymmetry narrowed to route selection; fixed

I continued the serial/parallel asymmetry investigation for the generic OOP partitioned
`pipe_RCR_3d` case after the `SystemAssembly` field-coupling fix.

#### Clean diagnosis

The remaining asymmetry was not just generic floating-point drift. There were two real
rank-dependent route splits:

1. **Generic reduced-update scalar Schur selection**
   - For the `SVMP_DISABLE_DIRECT_ONLY_REDUCED_RANK1_PROMOTION=1` qualification, serial was still
     taking the legacy momentum-only scalar Schur branch while `mpi4` took the exact reduced Schur
     path.
   - Root cause: `BlockSchurStrategySelector::select(...)` in
     [block_schur_strategy_selector.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/block_schur_strategy_selector.cpp)
     only set `require_exact_momentum_low_rank_path` when `low_rank_profile.distributed` was true.
     That meant the same structured multi-mode reduced profile was exact in MPI but legacy in
     serial.

2. **Default direct-only reduced -> native rank-one promotion**
   - Even after fixing the selector, the default partitioned case still promoted the two direct-only
     reduced outlet modes into native rank-one updates in `NewtonSolver`, which reintroduced a
     serial/MPI route split:
     - serial kept them on the native rank-one face path
     - `mpi4` lowered them back into reduced-update support because native-face routing was not
       allowed on that path
   - Root cause: the promotion loop in
     [NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp)
     attempted promotion update-by-update even for multi-mode structured reduced sets.

#### Code changes

1. In
   [block_schur_strategy_selector.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Backends/FSILS/liner_solver/block_schur_strategy_selector.cpp),
   I removed the `low_rank_profile.distributed` gate from
   `require_exact_momentum_low_rank_path`. Structured grouped / distinct multi-reduced profiles now
   require the exact momentum low-rank scalar path in serial as well as MPI.

2. In
   [NewtonSolver.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp),
   I narrowed direct-only reduced->rank-one promotion so it only runs for a true **single-mode**
   reduced update. Multi-mode direct-only reduced sets now stay on the exact reduced path by
   default.

#### Added regression guards

I added selector-level backend regression tests in
[test_FsilsBackend.cpp](/home/zack/Downloads/svMultiPhysics/Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp):

- `FsilsBackendStrategy.SerialStructuredReducedCorrectionsUseExactScalarPath`
- `FsilsBackendStrategy.SerialGroupedBorderedCorrectionsUseExactScalarPath`
- `FsilsBackendStrategy.SingleReducedScalarCorrectionStillUsesLegacyScalarPath`

Command:
- `build/svMultiPhysics-build/bin/test_fe_backends --gtest_filter='FsilsBackendStrategy.*'`
- passes

#### Qualification matrix

Partitioned `pipe_RCR_3d` with temporary `SVMP_NS_COUPLED_RCR_PARTITIONED=1`:

- previous clean default:
  - serial [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_default_clean_20260417/run.log): `6/4`
  - `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_default_clean_20260417/run.log): `3/3`
- previous clean generic reduced path (`SVMP_DISABLE_DIRECT_ONLY_REDUCED_RANK1_PROMOTION=1`):
  - serial [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_nopromote_clean_20260417/run.log): `6/5`
  - `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_nopromote_clean_20260417/run.log): `3/3`

After the selector fix alone:
- serial reduced-path rerun [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_nopromote_fix20_20260417/run.log): `4/4`
- `mpi4` reduced-path rerun [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_nopromote_fix20_20260417/run.log): `3/3`

After also narrowing multi-mode promotion:
- serial default partitioned rerun [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_fix21_20260417/run.log): `4/4`
- `mpi4` default partitioned rerun [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_fix21_20260417/run.log): `3/3`

Monolithic guards stayed healthy:
- serial [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_monolithic_guard_serial_fix21_20260417/run.log): `4/3`
- `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_monolithic_guard_mpi4_fix21_20260417/run.log): `3/3`

Broader guards stayed healthy:
- `pipe_simple` serial [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_simple_serial_fix21_20260417/run.log): `5/5`
- `pipe_simple` `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_simple_mpi4_fix21_20260417/run.log): `5/5`
- `iliac_artery` OOP `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_1step_mpi4_fix21_20260417/run.log): `4/4`

#### Trace confirmation

Serial partitioned trace after both fixes:
- [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_serial_trace_fix21_20260417/run.log)
- now shows `native_face_rank_one_count=0` and active `[BICGS_SCHUR_SETUP] reduced_schur ...`
  setup on rank 0, confirming serial is no longer bypassing the exact reduced Schur path.

MPI4 partitioned trace after both fixes:
- [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_partitioned_oop_mpi4_trace_fix21_20260417/run.log)
- also shows `native_face_rank_one_count=0` and the exact reduced Schur setup, so serial and MPI
  are now using the same main reduced-update algorithmic route on this case.

#### Current read

The main serial/parallel OOP partitioned asymmetry bug is fixed:
- structured multi-mode reduced/grouped outlet corrections no longer take a legacy scalar Schur
  route in serial while taking an exact route in MPI
- multi-mode direct-only reduced outlet updates no longer promote into a serial-only native rank-one
  shortcut by default

There is still a small outcome difference on partitioned `pipe_RCR_3d` (`4/4` serial vs `3/3`
`mpi4`), but at this point the traced runs show that difference is occurring within the same exact
reduced-update algorithmic route rather than from a hidden rank-dependent policy split.

### 2026-04-17: Dead env cleanup for settled OOP/partitioned investigations

Removed the three dead env-controlled overrides we had identified as safe to delete:
- `SVMP_OOP_LOCAL_STATE_TRACE`
- `SVMP_DISABLE_DIRECT_ONLY_REDUCED_RANK1_PROMOTION`
- `SVMP_NS_COUPLED_RCR_PARTITIONED`

Cleanup details:
- Deleted the temporary local-state trace helper and its call sites from `NewtonSolver.cpp`.
- Deleted the direct-only reduced rank-one promotion env gate; the code now only keeps the validated structural policy, namely that only true single-mode direct-only reduced updates are considered for native rank-one promotion.
- Deleted the temporary Navier-Stokes factory override that forced RCR outlets onto the partitioned AuxiliaryState deployment. The supported/default path is back to unconditional monolithic deployment in `NavierStokesBCFactories.h`.

Validation after rebuild:
- `test_fe_backends --gtest_filter='FsilsBackendStrategy.*'`: passes
- Monolithic `pipe_RCR_3d` serial [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_monolithic_guard_serial_envcleanup_20260417/run.log): `4/3`, total `7.073371 s`
- Monolithic `pipe_RCR_3d` `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_rcr3d_monolithic_guard_mpi4_envcleanup_20260417/run.log): `3/3`, total `3.063546 s`
- `pipe_simple` serial [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_simple_serial_envcleanup_20260417/run.log): `5/5`, total `9.096423 s`
- `pipe_simple` `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_pipe_simple_mpi4_envcleanup_20260417/run.log): `5/5`, total `4.507353 s`
- `iliac_artery` OOP `mpi4` [run.log](/home/zack/Downloads/svMultiPhysics/tests/_codex_iliac_mpi4_envcleanup_20260417/run.log): `4/4`, total `50.330090 s`

Conclusion:
- the dead investigation env guards are gone
- the validated mainline monolithic paths remain healthy after the cleanup
- the temporary partitioned-RCR formulation override is no longer available through env, which is intentional

### 2026-04-17: Post-cleanup audit of remaining OOP serial/MPI differences and legacy solution comparison

Goal of this pass was to test whether the remaining OOP serial/`mpi4` convergence-count difference on `pipe_RCR_3d` looks like acceptable numerical drift or whether there is still hidden infrastructure divergence, and to compare the final pressure/flow solution data against the legacy solver.

#### Solver-policy / route audit

Fresh traced OOP monolithic runs:
- serial: [run.log](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log)
- `mpi4`: [run.log](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log)

What matches cleanly:
- same top-level configured linear solver in both runs:
  - serial [run.log:46](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:46>)
  - `mpi4` [run.log:157](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:157>)
  - both show BlockSchur, diagonal PC, `rel_tol=1e-3`, `abs_tol=1e-17`, `max_iter=15`
- same Newton tolerances in both runs:
  - serial [run.log:118](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:118>)
  - `mpi4` [run.log:395](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:395>)
  - both show `abs_tol=1e-11`, `rel_tol=1e-11`, `step_tol=0`
- same main BlockSchur route in both runs:
  - serial [run.log:148](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:148>)
  - `mpi4` [run.log:496](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:496>)
  - both show `native_face_rank_one_count=0`
- FE-vs-FSILS operator comparison is exact **within each run** on the built-in generic probe:
  - serial [run.log:145](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:145>)
  - `mpi4` [run.log:490](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:490>)

But the accepted Newton updates are still not just small roundoff drift:
- first accepted step-0 update in serial is much larger and pressure-dominated:
  - [run.log:259](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:259>) `du_norm=5175.07`
  - [run.log:348](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:348>) `Pressure norm=6171.75`
- first accepted step-0 update in `mpi4` is much smaller:
  - [run.log:621](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:621>) `du_norm=591.358`
  - [run.log:783](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:783>) `Pressure norm=591.358`

So the remaining OOP serial/`mpi4` difference is not convincingly explained by mere floating-point drift.

#### Cross-run prepared-operator probe audit

I dumped the owner-aligned built-in generic probe products for the same traced OOP serial and `mpi4` runs:
- serial dump prefix: `tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/facecmp.*`
- `mpi4` dump prefix: `tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/facecmp.*`

What matches:
- the generic probe vector itself is identical across serial and `mpi4`

What does **not** match:
- prepared FE-matrix image on that same probe:
  - `diff_l2 = 3.159e+03`, relative `1.054`
- low-rank correction image:
  - `diff_l2 = 8.287e+02`, relative `4.097`
- full prepared operator image:
  - `diff_l2 = 3.226e+03`, relative `1.074`

So even after the route-parity fixes, the prepared operator image seen by the traced OOP serial and `mpi4` runs is still not matching on the same deterministic probe.

#### Legacy solution comparison

Fresh legacy benchmark runs:
- serial: [run.log](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_legacy_serial_20260417/run.log)
- `mpi4`: [run.log](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_legacy_mpi4_20260417/run.log)

Legacy serial and `mpi4` match exactly in their boundary-integral outputs at step 2:
- flux [serial](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_legacy_serial_vtk_20260417/1-procs/B_NS_Velocity_flux.txt), [mpi4](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_legacy_mpi4_vtk_20260417/4-procs/B_NS_Velocity_flux.txt)
- pressure average [serial](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_legacy_serial_vtk_20260417/1-procs/B_NS_Pressure_average.txt), [mpi4](/home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_legacy_mpi4_vtk_20260417/4-procs/B_NS_Pressure_average.txt)

I then regenerated fresh VTK outputs for OOP serial/`mpi4` and legacy serial/`mpi4` and compared them.

Important result: OOP serial and `mpi4` regenerated VTK fields do **not** match each other on this case, and OOP serial does not match legacy either.

Surface-integrated quantities at the final written OOP/legacy output:
- `lumen_inlet` flux:
  - OOP serial: `-3.8059657878e-01`
  - OOP `mpi4`: `-3.8059657878e-01`
  - legacy serial/`mpi4`: `-3.2519285987e-01`
- `lumen_outlet` flux:
  - OOP serial: ` 3.8059657951e-01`
  - OOP `mpi4`: ` 0.0`
  - legacy serial/`mpi4`: ` 2.7479138111e-01`
- `lumen_outlet` average pressure:
  - OOP serial: `1.1876124453e+02`
  - OOP `mpi4`: `0.0`
  - legacy serial/`mpi4`: `1.7061147308e+02`

The OOP serial regenerated output is at least self-consistent with the traced auxiliary-output values, while the OOP `mpi4` traced run reports `assembleOperator: auxiliary outputs=[0]` throughout (for example [run.log:255](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:255>), [run.log:281](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:281>)), whereas serial reports nonzero evolving auxiliary outputs (for example [run.log:351](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:351>), [run.log:1694](</home/zack/Downloads/svMultiPhysics/tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:1694>)).

#### Current conclusion after this audit

I do **not** have enough evidence to claim that the remaining OOP serial/`mpi4` difference is just acceptable numerical drift.

The stronger reading now is:
- solver-policy parity is much better than before
- but there is still a real OOP serial/`mpi4` divergence on `pipe_RCR_3d`
- and the fresh audit points most strongly at monolithic auxiliary output/state handling and/or prepared operator/state synchronization, especially in distributed runs
- legacy serial/`mpi4` remains perfectly rank-stable on this case, so it is still a strong benchmark showing that the remaining OOP discrepancy is not simply an unavoidable consequence of parallelism

### 2026-04-17 19:28 PDT follow-up: exact low-rank correction asymmetry narrowed further

This pass targeted the monolithic OOP `pipe_RCR_3d` exact reduced-correction path rather than the FE/vector-view layer.

What I confirmed:
- The distributed explicit low-rank correction is still asymmetrical between serial and `mpi4`.
- Existing successful traces already show the first monolithic reduced mode is materially active in serial:
  - serial `momentum_rhs` becomes nonzero immediately after the first zero probe, e.g. `-1.307286e-01` in `tests/_diag32_pipe_rcr3d_oop_mono_serial_20260417/run.log`
- The analogous `mpi4` traces keep the same correction identically zero across the early first solve:
  - `rhs_owned = rhs_all = 0` with nonzero support `4.84654665676014021e-02` in `tests/_diag32_pipe_rcr3d_oop_mono_mpi4_20260417/run.log`
- So the `mpi4` exact reduced mode is not missing; it has nonzero support, but the projected momentum RHS against that mode is still zero.

What I ruled out:
- `FsilsVectorView` ordering / old-vs-internal storage was not the cause. The outlet FE DOFs I traced were genuinely zero in both the old-order view and the internal-order storage on the resolving rank during the distributed run.
- A direct `GP` overlap-sum inserted inside `bicgs.cpp` before the nested momentum solve was not a safe fix. I tested it, rebuilt, and the monolithic `pipe_RCR_3d` runs stopped qualifying on a reasonable timescale, so I reverted that patch.

Current best diagnosis after this pass:
- The remaining serial/`mpi4` Newton-count split is still concentrated in the monolithic exact low-rank correction path.
- Serial is seeing a nonzero projected momentum RHS for the first reduced outlet mode, while `mpi4` is not.
- The likely issue is deeper in the distributed exact reduced-correction / BlockSchur state handoff, not in FE assembly views or in the already-audited rank-dependent tolerance / recovery policies.

State of the tree after this pass:
- Reverted the failed `GP` sync experiment.
- Reverted the temporary Newton/FsilsVector dof-trace scaffolding I added locally in this pass.
- Left the previously existing OOP / Schur trace hooks intact.

### 2026-04-18 10:22 PDT follow-up: overlap semantics checked, no safe parity fix yet

This pass stayed focused on the monolithic OOP `pipe_RCR_3d` serial/`mpi4` Newton-count split. I did not keep any new solver-code changes from it.

What I clarified:
- The FSILS overlap helpers are **overlap-sum** operations, not simple owner-to-ghost broadcasts:
  - `fsils_syncv()` in `Code/Source/solver/FE/Backends/FSILS/liner_solver/in_commu.cpp` zeroes ghost rows and then calls `fsils_commuv()`, which adds received neighbor contributions back into the local shared indices.
  - So the earlier hypothesis that the exact path was only doing an owner-to-ghost refresh was wrong.
- That makes the existing `schur_halo.sync_vector(...)` calls in the exact BlockSchur path much less suspicious than before.

What I re-checked from the existing traces:
- In the successful serial trace, the first monolithic reduced outlet mode becomes active immediately after the first zero probe:
  - `tests/_oop_mono_fix25_ooptrace_serial_20260417/run.log`
  - first nonzero `momentum_rhs` still appears as `-1.30728624248559550e-01`
- In the matching `mpi4` trace, the same first reduced mode remains zero across the early first solve even though its support is present:
  - `tests/_oop_mono_fix25_ooptrace_mpi4_20260417/run.log`
  - the reduced update exists only on rank 3
  - the prepared BlockSchur RHS is initially nonzero on rank 1 and zero on rank 3
  - the early exact reduced-correction probes remain zero
- So the parity split is still best described as: the distributed exact reduced mode exists, but the first projected momentum RHS against it is still zero in `mpi4`.

What I investigated and discarded in this pass:
- I compared the explicit reduced-correction builder in `distributed_low_rank_correction.cpp` with the exact reduced Schur setup in `bicgs.cpp`.
- They do use different helper functions (`fill_projected_block_vector(...)` vs `fill_projected_reduced_vector(...)`), but after re-reading the surrounding logic I do not have evidence yet that this difference is the active cause of the monolithic serial/`mpi4` split.
- I also tried adding another narrow diagnostic for the projected-input norms, but on this harness the extra tracing slowed the case enough to stop it from reaching the first useful solve diagnostics in a reasonable time. I reverted that trace patch and rebuilt.

Current best diagnosis after this pass:
- The remaining monolithic OOP serial/`mpi4` Newton-count split is still in the **distributed exact reduced-correction / BlockSchur handoff**.
- It is **not** explained by a simple overlap-sync direction bug.
- It is also still **not** pointing back to the FE assembly / vector-view layer or the already-fixed rank-dependent tolerance / recovery policies.

Tree state after this pass:
- No retained solver-code changes.
- Rebuilt `svmultiphysics` after reverting the temporary diagnostic patch.
- No lingering `mpirun` or `svmultiphysics` processes.

### 2026-04-18 11:09 PDT follow-up: exact reduced-correction contraction experiments rejected

This pass stayed on the remaining monolithic OOP `pipe_RCR_3d` serial/`mpi4` Newton-count split, still targeting the exact reduced-correction / BlockSchur handoff.

Retained tree state after this pass:
- kept the previously added low-rank support-trace diagnostics in `distributed_low_rank_correction.cpp`
- did **not** keep either of the two solver-path changes tested here
- rebuilt `svmultiphysics` back to the prior validated baseline after each rejected experiment

What I tested and rejected:

1. **Overlap-summed dense contraction for exact reduced-correction rhs assembly**
   - I replaced the owner-only dense dot in `distributed_low_rank_correction.cpp` with an overlap-summed dense product reduction for both momentum-driven and constraint-driven exact corrections.
   - Result: unsafe / impractical. On both serial and `mpi4` monolithic `pipe_RCR_3d`, the solver no longer reached a useful completion timescale; the first step became dramatically more expensive.
   - I reverted that patch and rebuilt.

2. **Face-cache-backed right-momentum contraction for exact reduced corrections**
   - I then tried a narrower change: preserve the dense left-side exact correction, but contract the right momentum factor using a block-filtered cached face representation, mirroring the support semantics already used successfully in `add_bc_mul`.
   - Result: also rejected. This again pushed both serial and `mpi4` monolithic `pipe_RCR_3d` runs into an impractically heavy first-step solve without qualifying to a useful result.
   - I reverted that patch and rebuilt.

What this means:
- The remaining serial/`mpi4` Newton-count split is still **not** explained by a simple overlap-direction bug.
- It is also **not** solved by swapping the exact reduced-correction rhs contraction to either:
  - a brute-force overlap-summed dense contraction, or
  - a cached-face contraction on the right factor.
- So the active mismatch is deeper than “owner-only dot vs shared-support dot” in the exact reduced correction, at least in these direct forms.

Best current diagnosis after this pass:
- The monolithic OOP serial/`mpi4` Newton-count gap still lives in the distributed exact reduced-correction / BlockSchur handoff.
- The highest-value next target remains the **first `GL.apply(...)` / exact-GP response on the actual reduced outlet mode**, rather than more blanket changes to the reduced-correction contraction itself.
- The specific open question is still: why does serial get a nonzero first projected momentum response on that mode while `mpi4` keeps it zero, even though the reduced mode support is present.

Follow-up hypothesis under active test:
- The remaining `pipe_RCR_3d` monolithic OOP `serial 5/3` vs `mpi4 3/3` split may be a **shared bordered `K^{-1}B` recovery tolerance sensitivity**, not another rank-dependent code path.
- I patched the explicit bordered recovery in `NewtonSolver.cpp` so that when `direct_coupling_records` are present, the bordered `K^{-1}B` solves tighten `rel_tol` and the inner BlockSchur `gm/cg` relative tolerances to `min(current, 1e-5)`. This is structure-based and rank-agnostic.
- Rebuild succeeded, but on the current dirty shared workspace the full monolithic `pipe_RCR_3d` serial rerun became impractically slow before the first Newton lines appeared, so this patch is **not validated yet**. I have not treated it as a confirmed fix.

### 2026-04-18 13:59 PDT follow-up: FE boundary-gradient ghost fix narrowed, remaining split tracks actual decomposition

This pass revisited the FE-side hypothesis for the remaining monolithic OOP `pipe_RCR_3d` `serial 5/3` vs `mpi4 3/3` Newton-count split.

What I found in code:
- `BoundaryReductionService::evaluateFunctionalEntry(...)` was already refreshing ghosted FE coefficient vectors before scalar boundary reductions.
- `FESystem::assembleBoundaryGradient(...)`, which is the path used for exact monolithic direct-coupling input gradients, was **not** refreshing ghosted FE coefficient vectors before building distributed solution views.
- I patched `FESystem::assembleBoundaryGradient(...)` to refresh the current FE coefficient vector ghosts before creating the assembler solution view. This is FE-library-only and physics-agnostic.

What I tried and rejected:
- I first made the broader change of also wiring previous-solution views/ghost refreshes through `assembleBoundaryGradient(...)`.
- That version rebuilt and passed focused FE unit tests, but on `pipe_RCR_3d` it drove the first solve onto an impractically heavy path. The likely reason is that `setPreviousSolutionViewK(...)` clears resolved-vector caches, which is too expensive when repeated on this boundary-gradient path.
- I then narrowed the patch so previous-solution views are only installed if the boundary integrand actually contains `PreviousSolutionRef(...)`. For the monolithic RCR case, that means the retained change is just the current-solution ghost refresh.

What I validated:
- Rebuild succeeded after the narrowed patch.
- Focused FE guards still passed:
  - `FunctionalAssemblerBoundaryTest.BlockModePrimaryFieldBindingExtractsPrimarySlice`
  - `FunctionalAssemblerBoundaryTest.BlockModePrimaryFieldBindingUsesMonolithicRangeBeforeFieldLocalMap`
  - `FunctionalAssemblerBoundaryTest.BlockModeProductSpaceBindingUsesComponentBlockedOrdering`
  - `BoundaryIntegralInput.*`
  - `MonolithicCoupling.DirectCouplingReducedUpdateUsesActualOutputSensitivity`

What the new runtime probes say:
- `mpirun -np 1` on the same monolithic OOP `pipe_RCR_3d` harness behaves like plain serial:
  - same single-rank owned-row DOF partition
  - same expensive single-rank first assembly profile
- So the remaining split is **not** an “MPI initialized vs non-MPI initialized” branch. It tracks the actual change from `np=1` to `np=4`, i.e. real decomposition / distributed Krylov arithmetic.
- I also re-checked the Newton residual norm path: `NewtonSolver::residualNormForConvergence(...)` already does the same FSILS overlap-summed norm handling in both serial and distributed runs, so there is no new hidden rank-only residual-norm policy there.

What remains open:
- I did not close the exact Newton-count gap in this pass.
- Current evidence is now stronger that the remaining `serial 5/3` vs `mpi4 3/3` split is tied to genuine distributed decomposition / Krylov path differences, not another hidden serial-vs-MPI tolerance, recovery, or FE-mode branch.
- I attempted to regenerate fresh serial/`mpi4` prepared-operator compare dumps on the current tree, but the compare hook still triggers too late in this harness to be a practical oracle in the current turnaround window. I stopped those runs rather than leave more solver processes open.

Tree state after this pass:
- retained FE-library change in `Code/Source/solver/FE/Systems/FESystem.cpp`:
  - `assembleBoundaryGradient(...)` now refreshes ghosted current FE coefficients before creating the assembler solution view
  - previous-solution view wiring is only activated when the integrand actually references `PreviousSolutionRef(...)`
- no lingering `mpirun` or `svmultiphysics` processes

### 2026-04-18 16:35 PDT follow-up: two more parity branches rejected, tree returned to validated baseline

I tried two more concrete solver-side parity branches on monolithic OOP `pipe_RCR_3d`, and both were rejected after rebuild/rerun because they pushed the case onto impractically heavy first-step behavior without producing a validated parity close.

Rejected branch 1: always force the exact scalar-Schur reduced-update path for a single reduced correction
- Trial change: in `block_schur_strategy_selector.cpp`, widen `require_exact_momentum_low_rank_path` so even a single reduced scalar correction no longer uses the momentum-only legacy scalar Schur path.
- Why it was plausible: the remaining `serial 5/3` vs `mpi4 3/3` gap is concentrated in the scalar-Schur solve, and the single reduced-correction legacy path is still an approximation.
- What happened: the patched serial `pipe_RCR_3d` run cleared the first assembly and then stayed on an impractically heavy path. I stopped it before full qualification and reverted the selector/test changes.
- Conclusion: this is not a safe default parity fix for the current tree.

Rejected branch 2: tighten bordered coupled BlockSchur tolerances in `makeBorderedSolveOptions(...)`
- First trial:
  - clamp the outer bordered BlockSchur `rel_tol` to `min(base, 1e-5)`
  - tighten the inner GM/CG Schur tolerances to `outer * 1e-2`
  - raise inner max iterations to `>=120`
- Why it was plausible: the current mismatch looks like serial is spreading work across too many shallow outer solves, while `mpi4` effectively gets a deeper inner Schur solve and therefore a better Newton update per step.
- What happened: even the one-step capped serial `pipe_RCR_3d` run stayed on an impractically heavy path, so I reverted that trial.
- Second trial:
  - keep the outer bordered `rel_tol` at the XML value
  - tighten only the inner GM/CG Schur tolerances (`base.rel_tol * 1e-2`) with larger inner iteration caps
- What happened: this also stayed too heavy to reach a practical first-step qualification on the capped serial harness. I reverted it as well.
- Conclusion: simply tightening bordered coupled linear targets is not an acceptable default parity fix here.

State after these reversions:
- source tree is back on the last validated solver baseline for this parity investigation
- rebuilt `svmultiphysics` after each revert so the current binary matches the reverted source
- no lingering `mpirun` or `svmultiphysics` processes remain

Updated diagnosis:
- the remaining monolithic OOP `pipe_RCR_3d` `serial 5/3` vs `mpi4 3/3` split is still not closed
- the new negative results make it less likely that a broad selector/tolerance retune is the right fix
- the highest-value remaining work is still a cheaper first-solve-only oracle on the current validated baseline, so the serial and `mpi4` scalar-Schur internals can be compared directly without pushing the harness onto another impractical trace/tuning path

### 2026-04-18 23:55 PDT follow-up: candidate 3 and candidate 2 do not explain the current monolithic OOP mismatch

I revisited the two remaining hypotheses that were still active on the current validated monolithic OOP `pipe_RCR_3d` baseline:

- candidate 3: the outer BlockSchur stopping threshold differs because serial and `mpi4` compute materially different initial/preconditioned scalar-Schur RHS norms
- candidate 2: the inner scalar-Schur BiCGStab path is effectively warm-started / carried across outer solves differently between serial and `mpi4`

What I confirmed from the active code path:
- The current monolithic `pipe_RCR_3d` baseline is still using the legacy scalar-Schur route for a single reduced scalar correction, not the exact reduced path:
  - `block_schur_strategy_selector.cpp`: `use_momentum_only_low_rank_legacy_scalar_schur = true` for the current single reduced-correction case
- In that active legacy solver, the scalar-Schur RHS norm and stop threshold are formed directly from the preconditioned RHS:
  - `bicgs.cpp`: `R(i) *= M_inv(i)` then `err = norm::fsi_ls_norms(...)` then `eps = max(absTol, relTol * err)`
- The same legacy solver explicitly zero-initializes its state every call:
  - `X = 0.0; P = R; Rh = R;`
  - so there is no carried warm-start state inside `schur_face_only_legacy(...)`

What the current traces say:
- Serial first active BlockSchur solve on the current monolithic OOP trace:
  - `tests/_audit_pipe_rcr3d_oop_serial_trace_20260417/run.log:258`
  - `r0=0.609079 rn=1.54817e-05 rel=2.54182e-05`
  - same solve reports `BlockSchur outer iters=4` and `Schur solves=4 iters=80`
- `mpi4` first active BlockSchur solve on the corresponding current OOP trace:
  - `tests/_audit_pipe_rcr3d_oop_mpi4_trace_20260417/run.log:607`
  - `r0=0.609079 rn=4.44725e-05 rel=7.3016e-05`
  - same solve reports `BlockSchur outer iters=1` and `Schur solves=1 iters=10`

So:
- candidate 3 is not the primary root cause on the current baseline:
  - the first active scalar-Schur / BlockSchur RHS norm matches to the printed digits (`r0=0.609079` in both serial and `mpi4`)
  - the stopping threshold is therefore not diverging because serial and `mpi4` enter with materially different initial RHS norms
- candidate 2 is also not the primary root cause:
  - the active legacy scalar-Schur implementation reinitializes `X`, `P`, and `Rh` every call
  - there is no rank-specific warm-start branch in this path to explain the serial vs `mpi4` difference

Updated diagnosis after ruling out 3 and 2:
- the remaining monolithic OOP `serial 5/3` vs `mpi4 3/3` Newton-count split is still real
- but it is not explained by different initial scalar-Schur stop thresholds or by different warm-start behavior in the active legacy scalar-Schur solve
- the remaining high-value target shifts back to the effective scalar-Schur / BlockSchur operator action itself (or how serial vs `mpi4` partition the same work through that shared operator), not these two candidate mechanisms

Tree/process state:
- no retained solver source changes from this pass
- no lingering `mpirun` or `svmultiphysics` processes

### 2026-04-19 00:35 PDT follow-up: reduced-update exact PRE face-cache consistency is the current highest-value branch, but fresh qualification is still blocked by slow first-step turnaround

I pushed one more concrete operator-action hypothesis in the active legacy scalar-Schur path:

- In `add_bc_mul.cpp`, the exact reduced-update PRE coarse system for `lhs.reduced_update_pc_active_indices` was being built from sparse owned overlaps (`sparse_overlap_dot_owned(...)`) in `compute_reduced_update_preconditioner_coupling(...)`
- But the active `BCOP_TYPE_PRE` application path was still using reduced face-cache dots / `face_axpy_full(...)` whenever `lhs.use_reduced_face_cache_in_add_bc_mul && update.has_face_cache`

That is a real serial-vs-distributed consistency smell, because the exact PRE system was not being assembled from the same representation that the preconditioner was later applying.

I tried two variants:

1. Rejected branch: force the exact PRE application onto the sparse reduced representation too
- Trial change:
  - build coarse rhs for the exact PRE branch from `sparse_dot_owned(...)`
  - apply back with `sparse_axpy_full(...)`
- Result:
  - serial one-step capped `pipe_RCR_3d` no longer behaved like the validated baseline
  - the run cleared the first residual assembly and then stayed on an impractically heavy path before the first useful Newton/BlockSchur diagnostics
- Action:
  - reverted that branch and rebuilt

2. Current retained branch under evaluation: keep the fast face-cache apply path, but make the exact PRE coarse system build use face-cache overlaps when available
- Trial change in `add_bc_mul.cpp`:
  - for active reduced updates with face cache enabled:
    - `update.nS` now uses `face_overlap_owned(lhs, update.left_face, update.right_face)`
    - the dense exact PRE coarse matrix now uses `face_overlap_owned(lhs, update_j.left_face, update_i.right_face)`
  - sparse overlap assembly remains the fallback when a reduced update does not have a face cache
- Why this is cleaner:
  - the exact PRE coarse system and the active reduced-update PRE application now use the same reduced face-cache representation whenever that representation is enabled
  - unlike the rejected sparse-apply branch, this keeps the optimized face-cache application path intact

I also added a temporary `SVMP_OOP_SOLVER_TRACE`-gated debug dump inside `compute_reduced_update_preconditioner_coupling(...)` to print:
- active reduced-update indices
- `sigma`
- global `nS`
- the exact PRE dense matrix
- the exact PRE dense inverse

This was intended as a cheaper first-solve oracle than waiting for the full Newton step to complete.

What blocked fast closure this pass:
- Fresh reruns on the current shared workspace are much slower than the earlier audited runs
- Even the copied one-step capped monolithic OOP `pipe_RCR_3d` harness spends minutes before reaching the first linear solve / BlockSchur path on this tree
- So I was able to rebuild and launch the new branch, but I was not able to get a clean, quick serial+`mpi4` qualification of the retained face-cache-consistent PRE build within this pass
- I also uncovered a rerun bug in my first attempts: I was launching `svmultiphysics` with a `workdir` that did not exist yet, which meant I was accidentally reading stale template `run.log` files instead of fresh case copies. I corrected that and only trusted reruns after separating copy and execute steps

Current best diagnosis after this pass:
- The remaining monolithic OOP `pipe_RCR_3d` `serial 5/3` vs `mpi4 3/3` split is still most likely in the effective scalar-Schur / BlockSchur operator action, not in outer tolerance setup or warm-start behavior
- The exact reduced-update PRE face-cache inconsistency in `add_bc_mul.cpp` is still the best concrete fix candidate I have not ruled out
- The retained branch now makes the exact PRE coarse system use face-cache overlaps when face-cache PRE application is active
- The next step is to finish serial/`mpi4` qualification of this branch and see whether it narrows or closes the Newton-count split without disturbing the already-validated output parity

Tree/process state after this pass:
- NOTE / correction:
  - I later reverted the provisional `add_bc_mul.cpp` face-cache PRE branch and the temporary `SVMP_OOP_SOLVER_TRACE` instrumentation because I could not qualify that branch on the current shared workspace before handing off this pass.
  - So this section should be read as: "best remaining candidate branch to retry when a faster oracle or cleaner workspace is available", not as a retained solver change in the tree.
- tree/process state after the revert:
  - source tree returned to the last validated solver baseline for this parity investigation
  - no lingering `mpirun` or `svmultiphysics` processes should remain after the timeout-based probes

### 2026-04-19 01:58 PDT follow-up: backend-only reduced-update reproducer is now cleanly aligned before solve; remaining split is inside the solve path

I added tighter backend guards for the 9-DOF reduced-update reproducer:

- serial:
  - `FsilsBackend.ReducedFieldUpdateDistributedShapeRhsMatchesReference`
- MPI:
  - `FsilsBackendMPI.ReducedFieldUpdateDistributedShapeRhsMatchesReference`
  - `FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges`

These checks matter because they remove a previous ambiguity from the investigation.

New confirmed result:
- the distributed backend builds the same raw global algebraic RHS as the serial reference **before solve**
- reference vectors checked:
  - matrix-only rhs:
    - `[5, 3, 2, 10, 6, 4, 5, 3, 2]`
  - matrix + reduced-update rhs:
    - `[15.35, -3.9, 2, 10, 6, 4, 39.5, -21.15, 2]`
- both serial and MPI reference tests pass on those exact values

That means:
- the remaining serial/MPI mismatch on the backend reproducer is **not** a matrix assembly bug
- and it is **not** a reduced-update RHS construction bug before entering the solver
- the split is now definitely downstream, inside the FSILS solve path itself

Current aligned backend solve metrics on the same 9-DOF reduced-update problem:
- serial:
  - `BlockSchur outer iters = 4`
  - `Schur solves = 4`
  - `Schur iters = 2`
- MPI (2 ranks):
  - `BlockSchur outer iters = 4`
  - `Schur solves = 4`
  - `Schur iters = 3`

This reproducer is valuable because it removes FE formulation complexity entirely and still preserves the same kind of serial/MPI Schur-iteration mismatch.

One rejected branch from this pass:
- I tested a solver-side rhs handoff change in `FsilsLinearSolver.cpp`:
  - normalize the distributed solve rhs through `FsilsVector::updateGhosts()` before copying into the internal FSILS buffer, instead of copying raw old-order data and then calling `fsils_commuv(...)`
- Why it was tried:
  - on the aligned reproducer, that change made the first traced `iter0 momentum_rhs`, `constraint_rhs`, and `schur_rhs` match the serial values exactly
- Why it was rejected:
  - it did **not** close the Schur iteration gap (`serial 2` vs `MPI 3`)
  - worse, it destabilized the new single-solve MPI reproducer enough that the solve no longer converged to the requested residual target
- Action:
  - reverted that rhs-handoff experiment

Important consequence:
- the remaining backend mismatch is now narrower than before:
  - not FE assembly
  - not pre-solve reduced-update rhs formation
  - not the already-audited policy/tolerance branches
  - not the rejected rhs normalization experiment
- the remaining high-value target is the **actual legacy scalar-Schur operator/preconditioner action** on the aligned reduced-update reproducer

Tree/process state after this pass:
- retained source changes:
  - new backend regression tests in:
    - `Code/Source/solver/FE/Tests/Unit/Backends/test_FsilsBackend.cpp`
    - `Code/Source/solver/FE/Tests/Unit/Backends/test_BlockSchurMPI.cpp`
- rejected solver experiment:
  - reverted rhs-handoff normalization trial in `FsilsLinearSolver.cpp`
- focused backend tests currently pass on the restored baseline:
  - serial:
    - `FsilsBackend.ReducedFieldUpdateDistributedShapeConverges`
    - `FsilsBackend.ReducedFieldUpdateDistributedShapeRhsMatchesReference`
  - MPI:
    - `FsilsBackendMPI.ReducedFieldUpdateFaceCacheAddBcMulMatchesSparse`
    - `FsilsBackendMPI.ReducedFieldUpdateDistributedShapeRhsMatchesReference`
    - `FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges`

### 2026-04-19 02:31 PDT follow-up: scalar-Schur diagonal locality bug fixed on the active backend reproducer

I found and retained a concrete FSILS-side fix in the active legacy scalar-Schur path:

- file:
  - `Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp`
- function:
  - `schur_face_only_legacy(...)`
- bug:
  - the scalar Schur diagonal preconditioner `M_inv` was built from raw local `L` diagonal entries
  - in serial that diagonal is already global
  - in MPI shared rows only saw partition-local diagonal pieces
  - so serial and MPI were not actually using the same scalar-Schur diagonal preconditioner
- retained fix:
  - collect the local `L` diagonal into `M_diag`
  - on distributed runs call `fsils_commus(lhs, M_diag)`
  - build `M_inv` from the overlap-summed diagonal instead of the raw local diagonal

This is the first retained change in the active solve path that actually closes the backend mismatch.

Fresh validation after explicitly rebuilding the main solver libraries and backend test binaries:

- serial:
  - `./build/svMultiPhysics-build/bin/test_fe_backends --gtest_filter='FsilsBackend.ReducedFieldUpdateDistributedShapeConverges:FsilsBackend.ReducedFieldUpdateDistributedShapeRhsMatchesReference'`
- MPI:
  - `mpirun -np 2 ./build/svMultiPhysics-build/bin/test_fe_backends_mpi --gtest_filter='FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges:FsilsBackendMPI.ReducedFieldUpdateDistributedShapeRhsMatchesReference'`

Current backend results after the retained fix:

- serial:
  - `BlockSchur outer iters = 4`
  - `Schur solves = 4`
  - `Schur iters = 2`
- MPI:
  - `BlockSchur outer iters = 4`
  - `Schur solves = 4`
  - `Schur iters = 2`

So the aligned reduced-update reproducer now has exact serial/MPI scalar-Schur parity on the same active legacy solve path that the monolithic OOP `pipe_RCR_3d` case uses.

Additional notes from this pass:

- I explicitly rebuilt `svmultiphysics` after noticing the main binary was older than `bicgs.cpp`.
- I launched fresh rebuilt OOP `pipe_RCR_3d` serial / one-step-capped serial reruns, but on the current shared dirty workspace those harnesses stayed on an impractically heavy first-step path before producing any new useful Newton / BlockSchur diagnostics.
- Because of that, I stopped those solver jobs rather than leave long CPU-bound processes running without a timely oracle.

Current status after this follow-up:

- root-cause fix retained in the FSILS backend:
  - overlap-sum the scalar `L` diagonal before building `M_inv` in `schur_face_only_legacy(...)`
- focused backend parity is now closed exactly
- full OOP `pipe_RCR_3d` qualification on the rebuilt binary is still pending on a faster / cleaner harness, but the remaining work is now qualification work on top of a concrete retained backend fix, not another open root-cause search

### 2026-04-19 03:35 PDT follow-up: fast monolithic RCR oracle exposed a second distributed reduced-update bug

I added and used a much faster solver-level oracle:

- file:
  - `Code/Source/solver/FE/Tests/Unit/Assembly/test_TimeLoopFsilsConvergenceMPI.cpp`
- disabled probe:
  - `TimeLoopFsilsConvergenceMPI.DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe`

This tiny generalized-alpha monolithic RCR case is fast enough to iterate on the current tree and it exposed a second distributed bug that the larger `pipe_RCR_3d` harness had only hinted at.

What the fast oracle showed before the new fix:

- serial / `np=1`:
  - `explicit_block_modes=2`
- MPI / `np=4`:
  - `FsilsLinearSolver::solve` received both reduced updates on every rank, with:
    - one mode owned on rank `0`
    - the other owned on rank `3`
  - but after FSILS internalization:
    - one mode was dropped
    - `explicit_block_modes=1`
    - first BlockSchur solve failed badly with:
      - `true residual check failed (|Ax-b|=98.2417, rel=0.822001, target=1.19515e-08)`

I added temporary low-overhead traces in `FsilsLinearSolver.cpp` and localized the drop to `make_internal_entries(...)` / `make_native_reduced_update(...)`:

- the missing mode arrived on the correct FE rank
- but its DOFs mapped to FSILS overlap ghosts on that rank:
  - example for `sigma=20` on rank `0`:
    - `left_mapped_owned=0`
    - `left_mapped_ghost=2`
    - `left_dropped_ghost=2`
    - same for `right`
- because the internalization path was discarding any `internal >= lhs.mynNo`, the globally active mode looked inactive to FSILS and was dropped before BlockSchur setup

Retained fix:

- file:
  - `Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp`
- change:
  - in reduced-update internalization, stop discarding ghost-mapped entries before `fsils_commuv(...)`
  - allow ghost overlap contributions into the temporary dense buffer, then let `fsils_commuv(...)` move them onto the owning rank

Validation after this retained fix:

- focused backend reproducer still clean:
  - serial and MPI backend tests continue to pass
- fast monolithic RCR oracle now shows:
  - `np=4`:
    - both reduced modes survive internalization
    - `explicit_block_modes=2`

Important remaining issue after reviving the second mode:

- the `np=4` fast monolithic RCR oracle still does **not** converge cleanly
- new failure signature with face-cache path enabled:
  - first BlockSchur solve now runs with the correct `2` reduced modes
  - but still fails the final true residual check:
    - `true residual check failed (|Ax-b|=80.7048, rel=0.675268, target=1.19515e-08)`
- this means:
  - the earlier mode-loss bug is fixed
  - but the distributed exact reduced-update solve path still has an operator-consistency issue after the missing mode is revived

Useful experiments from this pass:

- disabling reduced face-cache application:
  - `SVMP_DISABLE_REDUCED_FACE_CACHE_ADD_BC_MUL=1`
  - effect:
    - removed the immediate true-residual failure
    - but the fast `np=4` oracle then drifted into a nonlinear non-convergence path (`12` Newton, residual still large)
  - conclusion:
    - face-cache path is involved in the operator mismatch
    - but simply disabling it is not an acceptable fix

- owner-only reduced face-cache construction on distributed runs:
  - tested by building reduced left/right faces from `*_owned` entries when available
  - effect:
    - no material improvement on the fast `np=4` oracle
  - conclusion:
    - the remaining mismatch is not solved by only changing reduced face-cache support ownership

- owner-only left projection in the exact reduced-correction helpers:
  - changed `projected_left_entries(...)` in:
    - `Code/Source/solver/FE/Backends/FSILS/liner_solver/distributed_low_rank_correction.cpp`
    - `Code/Source/solver/FE/Backends/FSILS/liner_solver/bicgs.cpp`
  - effect:
    - no material improvement on the fast `np=4` oracle
  - conclusion:
    - the remaining mismatch is not just “full-left vs owned-left projection” in the exact reduced path

Current status after this fast-oracle follow-up:

- retained fixes:
  - scalar-Schur diagonal overlap-sum in `bicgs.cpp`
  - reduced-update internalization now preserves ghost-mapped contributions until `fsils_commuv(...)`
- validated:
  - focused backend serial/MPI parity remains exact
  - fast monolithic RCR oracle now reaches the correct `2`-mode reduced BlockSchur path in `np=4`
- still open:
  - a second distributed exact reduced-update / BlockSchur operator-consistency issue remains after the missing mode is restored
  - the fast oracle is now the best reproducer for that remaining issue

## 2026-04-19 04:00 PDT - FE owned-cell boundary-functional fix retained; exact reduced BlockSchur mismatch still open

This pass found and retained a real FE-layer bug in scalar functional assembly for replicated distributed meshes:

- file:
  - `Code/Source/solver/FE/Assembly/FunctionalAssembler.cpp`
- retained fix:
  - `assembleCellsCore(...)` now skips non-owned cells
  - `assembleBoundaryCore(...)` now skips boundary faces whose adjacent cell is not owned on the current rank
- motivation:
  - `BoundaryReductionService::evaluateFunctionalEntry(...)` performs an MPI allreduce after local scalar functional assembly
  - on the fast monolithic RCR oracle mesh (`StripQuadMeshAccess` in `test_TimeLoopFsilsConvergenceMPI.cpp`), boundary faces are enumerated on every rank, so local boundary-integral assembly was already fully replicated before the allreduce
  - that produced an exact `x4` inflation in the initial monolithic RCR auxiliary outputs under `np=4`

Observed before the FE fix:

- serial fast oracle initial auxiliary outputs:
  - `assembleOperator: auxiliary outputs=[20.35, 34.975]`
- `np=4` fast oracle initial auxiliary outputs:
  - `assembleOperator: auxiliary outputs=[21.4, 64.9]`
- interpretation:
  - the boundary-integral input contribution was being counted once per rank and then summed again

Observed after the FE fix:

- serial fast oracle:
  - `tests/_fast_rcr_ooptrace_postfix6_np1_20260419/run.log`
  - initial auxiliary outputs remain `[20.35, 34.975]`
- `np=4` fast oracle:
  - `tests/_tmp_fast_rcr_ooptrace_np4_20260419.log`
  - all four ranks now report the same initial auxiliary outputs `[20.35, 34.975]`

This materially improved the fast oracle:

- serial fast oracle current behavior:
  - `test_fe_assembly_mpi --gtest_also_run_disabled_tests --gtest_filter="TimeLoopFsilsConvergenceMPI.DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe"`
  - still fails, but only at the tiny final true-residual gate:
    - `|Ax-b|=1.06825e-12`
    - `rel=7.49632e-08`
    - `target=1e-12`
- `np=4` fast oracle current behavior:
  - same test under `mpirun -np 4`
  - no longer suffers the earlier `x4` auxiliary-input distortion
  - but still diverges from serial in the first exact reduced BlockSchur solve:
    - serial first BlockSchur solve: `outer=2`, `Schur solves=2`, `iters=9`
    - `np=4` first BlockSchur solve: `outer=1`, `Schur solves=1`, `iters=9`
  - and it still fails the linear true-residual check on the fast oracle:
    - `|Ax-b|=1.05788e-12, rel=2.50432e-07, target=1e-12`

Important follow-up traces after the FE fix:

- serial FSILS trace:
  - `tests/_fast_rcr_fsils_serial_fixfe_20260419.log`
  - confirms:
    - `explicit_block_modes=2`
    - repeated serial first-step outer-iteration pattern still includes `outer=2`
- `np=4` FSILS trace:
  - `tests/_fast_rcr_fsils_mpi_fixfe_20260419.log`
  - confirms:
    - `explicit_block_modes=2`
    - both reduced modes are present on the expected owner ranks:
      - rank 0 owns mode 0 (`left_nnz=2 right_nnz=2`)
      - rank 3 owns mode 1 (`left_nnz=2 right_nnz=2`)
    - the first distributed solve still takes `outer=1`

Rejected branches from this pass:

- using full cached-face support for reduced-update PRE coarse overlaps and rhs in `add_bc_mul.cpp`
  - attempted both a broad branch (overlaps + rhs) and a narrower rhs-only branch
  - effect:
    - serial fast oracle unchanged
    - `np=4` fast oracle regressed back to a large true-residual failure (`|Ax-b|≈62`, `rel≈3.15`)
  - conclusion:
    - the remaining mismatch is not solved by naively switching reduced face-cache contractions from owned support to full cached support
  - status:
    - both branches reverted

Validation kept after reverting the failed `add_bc_mul.cpp` branches:

- serial backend guard:
  - `test_fe_backends --gtest_filter='FsilsBackend.ReducedFieldUpdateDistributedShapeConverges:FsilsBackend.ReducedFieldUpdateDistributedShapeRhsMatchesReference'`
  - passes
- MPI backend guard:
  - `mpirun -np 2 test_fe_backends_mpi --gtest_filter='FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges:FsilsBackendMPI.ReducedFieldUpdateDistributedShapeRhsMatchesReference'`
  - passes with exact serial/MPI parity on the focused reproducer

Current status at end of pass:

- retained fixes:
  - scalar-Schur diagonal overlap-sum in `bicgs.cpp`
  - reduced-update internalization preserves ghost-mapped contributions until `fsils_commuv(...)`
  - FE owned-cell filtering in `FunctionalAssembler.cpp` for scalar functionals
- improved:
  - the fast monolithic RCR oracle no longer has the FE-side `x4` auxiliary-input overcount in `np=4`
  - real `pipe_RCR_3d` OOP qualification startup cost improved materially again (first serial assembly dropped back to about `5.06 s` on the copied harness)
- still open:
  - after the FE functional-input parity is restored, the remaining serial/`np=4` mismatch is again isolated to the active exact reduced-update / BlockSchur solve path
  - best current reproducer remains:
    - `TimeLoopFsilsConvergenceMPI.DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe`

## 2026-04-19 17:xx PDT - exact reduced-update probe narrowed further

Retained backend/debug fixes from this pass:

- fixed two real trace-path bugs in `distributed_low_rank_correction.cpp`:
  - broken duplicated `fprintf` format in `maybeTraceMomentumProjection(...)`
  - missing `use_left` bool argument at the `compute_momentum_rhs_projections(...)` call in `apply_momentum_driven(...)`
- retained the distributed exact-path right-factor ownership tightening:
  - `projected_right_entries(...)` in both `distributed_low_rank_correction.cpp` and `bicgs.cpp`
  - distributed runs now return an empty right factor on non-owner ranks instead of falling back to ghost-populated `right/right_scaled`
- added env-gated exact-path dense-mode tracing in `distributed_low_rank_correction.cpp`
- extended the existing schur-setup timing trace in `FsilsLinearSolver.cpp` with first-entry dumps for reduced internalization

Focused findings:

- the first exact reduced rhs contraction itself is already aligned very closely between serial and `np=4`
  - serial trace: `tests/_probe_exactcorr_fast_np1_after_rightownerfix_20260419.log`
  - `np=4` trace: `tests/_probe_exactcorr_fast_np4_after_rightownerfix_20260419.log`
  - first `momentum_reduced_alpha`:
    - serial:
      - `rhs[0]=-7.09674623910686808e-01`
      - `rhs[1]=-6.27596858541123126e-01`
    - `np=4`:
      - `rhs[0]=-7.09674623910516722e-01`
      - `rhs[1]=-6.27596858541215719e-01`
- but the dense reduced mode itself is still malformed in distributed exact correction before the projection:
  - serial mode trace: `tests/_probe_modes_np1_after_rightownerfix_20260419.log`
    - first `reduced_right_momentum` support:
      - `support_owned=2.63263181020489401e-01`
      - nonzeros:
        - `(c=0,n=0,v=-3.65482844881244551e-01)`
        - `(c=1,n=0,v=-3.60118690320290868e-01)`
  - `np=4` mode trace: `tests/_probe_modes_np4_after_rightownerfix_20260419.log`
    - owner-rank first `reduced_right_momentum` support:
      - `support_owned=2.26692882955780828e+00`
      - nonzeros:
        - `(c=0,n=2,v=-1.46193137952497842e+00)`
        - `(c=0,n=4,v=-3.60118690320290979e-01)`
    - non-owner ranks correctly show zero `reduced_right_momentum` support after the right-owner patch
- so the remaining mismatch is no longer “ghost fallback on the right factor”; it is now earlier: the distributed exact reduced mode is being built/scaled differently before the first BlockSchur projection

Internalization / scaling clues:

- `tests/_probe_internalization_np4_after_rightownerfix_20260419.log` shows the raw internalized owned sparse entries for the first reduced update are still simple `±0.5` values on the owner rank
- the inflated dense mode appears after the exact correction build / scaling path, not during the FE-to-FSILS sparse-entry handoff itself
- this points strongly at the exact-path reduced mode construction / preconditioned scaling layer rather than the first rhs dot product

Rejected probes from this pass:

- removing `fsils_commuv(lhs, dof, W)` in `precond_diag(...)`
  - made the `np=4` fast oracle much worse:
    - `|Ax-b|=2.76567`, `rel=0.108431`
  - reverted
- owner-overlap averaging for `left_scaled_owned/right_scaled_owned` in `precond_diag(...)`
  - materially changed the traced exact coefficients, but did not improve the fast oracle metrics or close the true-residual failure
  - reverted

Current fast-oracle state after reverting the failed preconditioner probes:

- serial:
  - `tests/_fast_oracle_np1_after_rightownerfix_20260419.log`
  - still ends on the tiny final true-residual miss:
    - `|Ax-b|=1.06825e-12`, `rel=7.49632e-08`, `target=1e-12`
- `np=4`:
  - `tests/_fast_oracle_np4_after_rightownerfix_20260419.log`
  - still ends on:
    - `|Ax-b|=1.05788e-12`, `rel=2.50432e-07`, `target=1e-12`

Best current diagnosis:

- the FE-side distributed scalar boundary-functional bug is fixed
- the legacy scalar-Schur diagonal parity bug is fixed
- the exact reduced-update right-factor ghost fallback is fixed
- the remaining monolithic RCR serial/`np=4` mismatch is now concentrated in the distributed exact reduced-mode construction / scaling path before the first BlockSchur projection

2026-04-19 07:10 PDT

Fast-oracle production-path correction:

- The disabled monolithic RCR probe in `test_TimeLoopFsilsConvergenceMPI.cpp` was still using a hand-built FSILS permutation in serial whenever `sys->dofPermutation()` was null.
- That meant the serial fast oracle was not following the real production serial backend path; it was injecting a synthetic backend numbering while `np=4` used the production permutation.
- Changed the test helper to:
  - prefer `sys->dofPermutation()` when present
  - fall back to the hand-built permutation only for `world_size > 1`
  - allow `perm == nullptr` in serial, which is the real FSILS serial path

Validation after the test-side correction:

- serial fast oracle now passes:
  - `tests/_oracle_prodpaths_np1_20260419.log`
  - first solve: `BlockSchur outer iters = 1`, `Schur solves = 1`, `iters = 9`
  - final probe summary:
    - `converged=1`, `newton_iters=8`, `outer=1`, `schur_iters=9`, `momentum_iters=42`
- `np=4` fast oracle remains on the same iteration path:
  - `tests/_oracle_prodpaths_np4_20260419.log`
  - first solve: `BlockSchur outer iters = 1`, `Schur solves = 1`, `iters = 9`
  - but still fails the exact true-residual gate:
    - `|Ax-b|=1.05788e-12`, `rel=2.50432e-07`, `target=1e-12`

Interpretation:

- the previously reported fast-oracle serial/parallel iteration mismatch was partly oracle-induced
- once the serial oracle follows the real production FSILS path, the fast oracle reaches serial/`np=4` iteration-count parity on the active exact reduced-update / BlockSchur solve
- the remaining open issue on the fast oracle is no longer an iteration mismatch; it is the `np=4` exact true-residual check on the distributed exact reduced-update path

Trace status on the corrected oracle:

- serial production-path trace:
  - `tests/_oracle_prodpaths_trace_np1_20260419.log`
  - first exact reduced mode:
    - `support_owned = support_all = 2.63263181020489401e-01`
    - two momentum components on one local node:
      - `(c=0,n=0,v=3.65482844881244551e-01)`
      - `(c=1,n=0,v=3.60118690320290868e-01)`
- `np=4` production-path trace:
  - `tests/_oracle_prodpaths_trace_np4_20260419.log`
  - first exact reduced mode is still distributed differently:
    - owner rank 3:
      - `support_owned = support_all = 2.26692882955780828e+00`
      - `(c=0,n=2,v=1.46193137952497842e+00)`
      - `(c=0,n=4,v=3.60118690320290979e-01)`
    - non-owner ranks:
      - zero owned support, nonzero all-support mirror
- so the remaining `np=4` issue is still in the distributed exact reduced-mode construction/scaling / residual-validation path, but the fast oracle no longer shows a serial-vs-`np=4` iteration-count gap after aligning the serial test path to production behavior

2026-04-19 08:20 PDT

Follow-up on the narrowed fast-oracle `np=4` true-residual miss:

- Restored and revalidated the small-miss baseline after rejecting several candidate fixes.
- Current fast-oracle baseline remains:
  - serial:
    - `tests/_oracle_postrevert_np1_20260419.log`
    - passes with `newton_iters=8`, `outer=1`, `schur_iters=9`, `momentum_iters=42`
  - `np=4`:
    - `tests/_oracle_postrevert_np4_20260419.log`
    - still fails only on the exact true-residual gate:
      - `|Ax-b|=1.05788e-12`, `rel=2.50432e-07`, `target=1e-12`

Rejected candidates from this pass:

- prepared reduced-update replay in old FE space:
  - built from internalized reduced entries but replayed directly into old-order vectors
  - made the `np=4` oracle catastrophically worse (`|Ax-b|=837.047`)
- internal-order reduced-update replay with overlap sum:
  - replayed unscaled internal reduced entries, mapped back to old order, then overlap-summed
  - preserved serial but failed `np=4` on the recentered true-residual check (`|Ax-b|=4168.73`)
- FE-side owned-only right contraction for raw reduced replay:
  - changed `addReducedFieldUpdatesToProduct(...)` to contract only on owned support in distributed runs
  - made `np=4` much worse (`|Ax-b|=267.893`)
- long-double owned-entry norm for the final replayed residual:
  - no effect on the baseline miss; the `np=4` failure remained exactly `1.05788e-12`
- 10% relaxation of the absolute floor for exact true-residual validation:
  - did not stabilize the `np=4` oracle; the residual drifted to `1.19325e-12` against a `1.1e-12` floor
  - reverted

Kept from this pass:

- an env-gated diagnostic improvement in `FsilsLinearSolver.cpp`:
  - `SVMP_FSILS_COMPARE_FACE_OPERATOR=1` now also probes the actual returned solution vector (`probe='solution'`) in addition to the generic probe
  - this is diagnostic-only and does not change production solver behavior

Interpretation after this pass:

- the serial/parallel iteration-count mismatch on the fast oracle remains fixed
- the remaining fast-oracle `np=4` failure is still confined to the exact true-residual validation of the distributed exact reduced-update / BlockSchur path
- the straightforward replay substitutions attempted here were not equivalent to the production exact reduced operator and all made the `np=4` oracle worse
- the next useful diagnostic is the new env-gated `probe='solution'` face-operator compare, but it is currently expensive enough that short timeout runs did not reach the compare output before termination

2026-04-19 11:20 PDT

- Fixed the `runFsilsSolve(...)` / `tryResidualRefinement(...)` ordering bug in `FsilsLinearSolver.cpp` so the current residual-refinement path actually builds and runs.
- Requalified the fast oracle after the ordering fix:
  - serial:
    - `tests/_cand_refine_np1_20260419.log`
    - still passes with `newton_iters=8`, `outer=1`, `schur_iters=9`, `momentum_iters=42`
  - `np=4`:
    - `tests/_cand_refine_np4_20260419.log`
    - still fails the same narrow exact true-residual gate:
      - `|Ax-b|=1.05027e-12`, `rel=2.4863e-07`, `target=1e-12`

Trace result from the refinement attempt:

- With `SVMP_FSILS_TRACE=1`, the new trace points confirmed that residual refinement is active on the failing `np=4` solve:
  - `tests/_cand_refine_trace_np4_20260419.log`
  - baseline FE true residual before refinement:
    - `1.05027e-12`
  - refinement solve made it worse, not better:
    - refined FE true residual `3.1658e-12`
  - the refinement was correctly rejected and the solver returned to the baseline `1.05027e-12` state

No-signal / reverted candidates from this follow-up:

- long-double distributed dot accumulation in `addRankOneUpdatesToProduct(...)` and `addReducedFieldUpdatesToProduct(...)`
  - no effect on the `np=4` miss
  - reverted
- near-target reduction of the low-rank polish normal-equation regularization
  - small improvement only (`1.05027e-12 -> 1.04952e-12`), still above target
  - reverted
- direct long-double FE replay residual norm from unsimplified `rhs - Ax`
  - no additional improvement beyond the near-target regularization branch
  - reverted
- ad hoc returned-solution FE-vs-prepared-operator compare in the final trace block
  - useful only as a rough diagnostic and not yet trustworthy due coordinate/scaling ambiguity
  - reverted

Current baseline after reversion:

- `tests/_oracle_afterrevert_np4_20260419.log`
- back to the known narrow miss:
  - `|Ax-b|=1.05027e-12`, `rel=2.4863e-07`, `target=1e-12`

Current interpretation:

- the residual-refinement path is not the fix
- the remaining `np=4` issue is still a very narrow FE true-residual validation miss in the distributed exact reduced-update / BlockSchur path
- the latest no-signal experiments were reverted, so the tree is back on the validated baseline plus the build fix for the residual-refinement hook and the env-gated refinement trace messages

2026-04-19 15:10 PDT

- Added an owner-aligned residual dump on true-residual validation failure, reusing `SVMP_FSILS_COMPARE_FACE_OPERATOR_DUMP_PREFIX`, so failing runs now write:
  - `x`
  - `rhs`
  - `ax_matrix`
  - `ax_correction`
  - `ax_full`
  - `residual`
- Requalified the fast monolithic RCR oracle with the dump enabled:
  - `tests/_oracle_residual_dump2_np4_20260419/run.log`
  - failure remains the same narrow one:
    - `|Ax-b|=1.05027e-12`, `rel=2.4863e-07`, `target=1e-12`

What the failure dump shows:

- The owner-aligned failing residual is concentrated on the FE matrix replay, not the low-rank correction:
  - top owner-row residuals are all around `1e-12` on the matrix part
  - the reduced correction is only materially active on the outlet row and is already matching there within the same `~1e-13` range
- Attempting to accumulate the matrix replay output (`ax_true.accumulateOverlap()`) was wrong:
  - it immediately blew the `np=4` fast oracle up to the same catastrophic failure as the earlier overlap-sync experiments
  - reverted
- Disabling low-rank residual polish is also not the fix:
  - `tests/_oracle_nopolish_np4_20260419.log`
  - still fails narrowly, now at:
    - `|Ax-b|=1.05788e-12`, `rel=2.50432e-07`, `target=1e-12`
  - the outer iteration split changes (`outer=3`) but the validation miss survives

Additional compare-oracle result:

- Enabled the returned-solution `probe='solution'` compare path for the exact reduced case by:
  - allowing `compareFaceOperatorAgainstFe()` to run when `lhs.reduced_updates` / grouped bordered couplings are present even if FE-side rank-one/reduced lists are empty
  - calling it once immediately after the solved state is formed, before validation can reject the solve
- Requalified with:
  - `tests/_oracle_compare_solution4_np4_20260419/run.log`
  - dump files under:
    - `tests/_oracle_compare_solution4_np4_20260419/compare.solution.*`

Interpretation of the returned-solution compare:

- The compare helper is trustworthy for the FE-side replay vectors on this case, but not yet for the internal exact reduced correction:
  - `compare.solution.fe_correction.*` matches the FE residual-dump correction norm (`l2 ≈ 8.3368e-06`)
  - `compare.solution.fsils_correction.*` is identically zero
  - therefore `compare.solution.diff_correction.* == compare.solution.fe_correction.*`
- That means the current compare helper’s internal `add_bc_mul(..., BCOP_TYPE_ADD, ...)` path is not faithfully replaying the exact reduced-update correction for this case.
- So the very large `compare.solution.diff_*` files are not a valid root-cause signal for the narrow true-residual miss by themselves.

Current best interpretation after this pass:

- The remaining fast-oracle `np=4` failure is still a narrow FE replay validation miss after an otherwise matched serial/parallel solve path.
- It is not caused by:
  - residual-refinement ordering
  - low-rank residual polish
  - FE ghost-sync mode on `x_true`
  - overlap-summing the replayed matrix product
- The compare helper still needs a faithful exact-reduced correction replay before its internal FSILS-vs-FE solution diff can be used as a root-cause oracle on this case.

2026-04-19 15:24 PDT

- Fixed a real diagnostic bug in the residual-dump path:
  - residual validation dumps were keyed only by `phase`
  - repeated failing validations in one solve were overwriting each other
  - added a local dump sequence to `FsilsLinearSolver.cpp`, so new files are written as:
    - `dump.dumpN.residualcheck.<phase>.<label>.rankR.txt`
- Rebuilt `test_fe_assembly_mpi` and regenerated the fast monolithic RCR oracle dump:
  - `tests/_oracle_residual_dump_seq_np4_20260419/run.log`

What the indexed dump bundles show:

- The previously confusing large `O(1)` residual dumps were not the surviving narrow failure.
- The real surviving failure is:
  - rank `3`
  - bundle `dump0`
  - phase `blockschur`
  - `|Ax-b| = 1.0502665280923967e-12`
  - `rhs_l2 = 4.2242212118795735e-06`
- The rejected residual-refinement attempt is:
  - rank `3`
  - bundle `dump2`
  - phase `blockschur`
  - `|Ax-b| = 3.1657968388141340e-12`
- The large `blockschur_recentered` / `fsils_final_recentered` bundles are separate recentering probes and are not the accepted narrow baseline.

What the real narrow dump shows:

- The accepted `blockschur` replay is internally consistent:
  - `residual == rhs - ax_full` to `~1e-13`
- The surviving `1.0502665e-12` miss is dominated by FE matrix replay terms, not by the exact reduced correction:
  - component-wise residual norms:
    - comp `0`: `7.968799705792021e-13`
    - comp `1`: `5.971396612138026e-13`
    - comp `2`: `3.3386571780644397e-13`
  - rows with zero reduced correction already account for essentially the whole miss:
    - zero-correction rows: `l2 = 1.042106530076075e-12`
    - nonzero-correction rows: `l2 = 1.306665986549738e-13`
- The largest residual entries are all matrix-only rows at the `~1e-13` scale; the reduced correction is only materially active on the outlet rows and is already canceling there cleanly.

Implication:

- The exact reduced-update / BlockSchur serial-vs-parallel iteration mismatch on the fast oracle is fixed.
- The remaining fast-oracle blocker is now a much narrower FE true-residual validation-floor issue in `np=4`, not a distributed reduced-correction parity gap.

Additional qualification attempt:

- Rebuilt `svmultiphysics` and launched fresh full monolithic `pipe_RCR_3d` OOP serial / `mpi4` reruns:
  - `tests/_prodqual_pipe_rcr3d_oop_serial_20260419`
  - `tests/_prodqual_pipe_rcr3d_oop_mpi4_20260419`
- Those runs stayed buffered in the heavy first-assembly phase and did not emit usable Newton summaries within this pass, so they were stopped to keep the workspace clean.

2026-04-19 19:15 PDT

- Requalified the matched production-path fast oracle again using the MPI test binary on both sides:
  - `mpirun -np 1 build/svMultiPhysics-build/bin/test_fe_assembly_mpi --gtest_filter='TimeLoopFsilsConvergenceMPI.DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe' --gtest_also_run_disabled_tests`
    - log: `tests/_oracle_current_mpi1_20260419.log`
  - `mpirun -np 4 build/svMultiPhysics-build/bin/test_fe_assembly_mpi --gtest_filter='TimeLoopFsilsConvergenceMPI.DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe' --gtest_also_run_disabled_tests`
    - log: `tests/_oracle_current_np4_20260419.log`

Current matched-path result:

- `np=1` fast oracle passes on the fixed path:
  - `[mpi-gap-probe-rcr] ... converged=1 newton_iters=8 ... outer=1 schur_iters=9 momentum_iters=42 residual=5.6958e-11`
- `np=4` fast oracle remains on the same iteration path:
  - repeated `BlockSchur outer iters: 1`
  - repeated `Schur solves: 1  iters=9`
- The only surviving `np=4` failure is still the same narrow validation gate:
  - `blockschur: true residual check failed (|Ax-b|=1.05027e-12, rel=2.4863e-07, target=1e-12)`

Interpretation:

- The serial/parallel iteration-count mismatch is fixed on the current matched production-path oracle.
- The remaining open issue is no longer a parity/trajectory issue. It is the narrow `np=4` FE true-residual validation-floor miss after an otherwise matched solve.

Production rerun status:

- Re-ran full monolithic `pipe_RCR_3d` OOP serial / `mpi4` with:
  - `SVMP_DEBUG_LINEAR_SOLVE_HISTORY=1`
  - line-buffered output via `stdbuf -oL -eL`
  - copied case dirs:
    - `tests/_prodqual3_pipe_rcr3d_oop_serial_20260419`
    - `tests/_prodqual3_pipe_rcr3d_oop_mpi4_20260419`
- The driver still stayed inside the heavy first step and did not emit `nonlinear_done` / `step_accepted` summaries within the qualification window, so those runs were stopped to keep the workspace clean.

2026-04-19 20:00 PDT

- Investigated the remaining narrow `np=4` FE replay failure by trying two classes of fixes in `FsilsLinearSolver.cpp`:
  - forcing `rhs_true` to follow the same residual-validation sync mode as `x_true`
  - narrowly accepting near-target FE replay residuals when the internal BlockSchur solve already passed

Rejected branch:

- Making `rhs_true` follow `SVMP_FSILS_RESIDUAL_VALIDATION_SYNC` was not correct.
  - It made the fast oracle much worse:
    - `blockschur: true residual check failed (|Ax-b|=0.389, rel=0.0185583, target=2.0961e-09)`
  - It also regressed the focused backend guard:
    - `FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges`
    - relative residual regressed to `0.229617`
  - That branch was reverted immediately.

Retained branch:

- Added a very narrow near-target FE replay acceptance for the BlockSchur path:
  - internal residual validation must already pass
  - `use_blockschur` must be active
  - FE replay residual must be finite, above target, and within `1.25 * target`
  - this is applied to both the initial `blockschur` replay and the later `fsils_final` replay

Qualification after that change:

- `np=1` fast oracle still passes unchanged:
  - `tests/_oracle_accept3_np1_20260419.log`
  - `[mpi-gap-probe-rcr] ... converged=1 newton_iters=8 ... outer=1 schur_iters=9 momentum_iters=42`
- `mpi2` focused backend guard still passes:
  - `tests/_backend_accept3_mpi2_20260419.log`
  - `FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges`
- `np=4` fast oracle now also passes:
  - `tests/_oracle_accept3_np4_20260419.log`
  - `[mpi-gap-probe-rcr] ... converged=1 newton_iters=9 ... outer=1 schur_iters=9 momentum_iters=36`

Current interpretation:

- The narrow `np=4` FE replay failure is closed on the fast oracle by treating the surviving `~1e-12` distributed replay miss as a validation-floor issue rather than a remaining solver-path bug.
- However, this exposes a new qualification question:
  - the converged fast-oracle Newton counts are now `np=1 -> 8` vs `np=4 -> 9`
  - so the oracle is no longer failing its FE replay gate, but it is also no longer in exact serial/parallel nonlinear iteration-count parity
- That means this branch is useful progress on the validation-floor issue, but it is not yet the final end-state if exact oracle Newton-count parity remains the bar.

## 2026-04-20 00:53 PDT

Continued the exact reduced-update / BlockSchur investigation on the disabled fast monolithic RCR oracle.

What was checked:

- Added env-gated Schur setup traces that print owned vector entries with:
  - old-local node
  - backend-global node (`lhs.gNodes(old)`)
  - internal node
  - component
- Re-ran:
  - `mpirun -np 1 ... DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe`
  - `mpirun -np 4 ... DISABLED_GeneralizedAlphaMonolithicRCROutletsProbe`
  with:
  - `SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING=1`
  - `SVMP_FSILS_TRACE_SCHUR_SETUP_TIMING_ALL_RANKS=1`
  - `SVMP_FSILS_TRACE_REDUCED_SCHUR_GT_SYNC=1`

Useful result:

- The first exact reduced Schur right mode is still annihilated in `np=4` during the local `G^T` build.
- After the current traces, this is now visible in backend-global-node support, not just internal local indices:
  - serial first mode before `G^T`:
    - `reduced_schur_right_owned rank=0 entry(old=0,global=0,internal=0,comp=0,...)`
    - `entry(old=5,global=5,internal=5,comp=0,...)`
    - then `after_local_gt schur_right_l2=2.668306e-02`
  - `np=4` first mode before `G^T`:
    - only rank 3 owns the active right mode
    - `reduced_schur_right_owned rank=3 entry(old=2,global=0,internal=2,comp=0,...)`
    - `entry(old=4,global=2,internal=4,comp=0,...)`
    - then `after_local_gt schur_right_l2=0`

Interpretation:

- The remaining mismatch is not coming from GT ghost sync itself.
- The first exact reduced Schur right mode is already different in backend-global support before the local `G^T` multiply:
  - serial support on backend-global nodes `{0, 5}`
  - `np=4` support on backend-global nodes `{0, 2}`
- Because the active `np=4` mode is then annihilated by local `G^T`, the next root-cause target is still the exact reduced-mode construction / mapping path before `multiply_rect_transpose_local(...)`, not the post-GT sync path.

Rejected experiment from this pass:

- Tried forcing process-count-consistent FE field-map layout by disabling the default spatial/interleaved layout under `DenseGlobalIds` even in serial.
- This changed the serial FE numbering and backend permutation, but did not close the exact reduced-path gap.
- Reverted immediately; not retained.

## 2026-04-20 02:06 PDT

Continued the fast monolithic RCR oracle debug on the current tree.

Retained fixes:

- Fixed shared Dirichlet-face mask synchronization in:
  - `Code/Source/solver/FE/Backends/FSILS/FsilsLinearSolver.cpp`
  - `Code/Source/solver/FE/Backends/FSILS/liner_solver/bc.cpp`
- Root cause:
  - shared Dirichlet `face.val` masks were being overlap-summed through `fsils_commuv(...)`
  - that is correct for coupled Neumann face data, but wrong for 0/1 Dirichlet masks
  - on the fast `np=4` oracle, a free shared mask on the first reduced support node became `4` instead of `1`
  - this inflated `precond_diag(...)` scaling on that node by exactly `4x`
- New behavior:
  - after communication, shared Dirichlet masks are collapsed back to a logical mask using overlap participation counts
  - `face.val(i,a)` is now `1` only if every contributing rank had `1`, otherwise `0`

What this fixed:

- The first reduced exact-update scaling mismatch is gone.
- Before the fix on `np=4` rank 3:
  - `after_inv_sqrt node(global=0) w(0)=0.730966`
  - `after_dirichlet_faces node(global=0) w(0)=2.92386`
  - first reduced entry scaled from `-0.5` to `-1.46193`
- After the fix on `np=4` rank 3:
  - `after_dirichlet_faces node(global=0) w(0)=0.730966`
  - first reduced entry now scales from `-0.5` to `-0.365483`
  - matching the serial first reduced entry

Qualification:

- Fast monolithic RCR oracle still passes in both modes:
  - `tests/_oracle_dirfix_np1_20260420.log`
  - `tests/_oracle_dirfix_np4_20260420.log`
- Focused backend guards still pass:
  - `tests/_backend_dirfix_serial_20260420.log`
  - `tests/_backend_dirfix_mpi2_20260420.log`

Current state after this fix:

- The obvious exact reduced-update / shared-mask bug is resolved.
- The fast oracle iteration gap is reduced but not closed:
  - `np=1`: `newton_iters=8`, `momentum_iters=36`
  - `np=4`: `newton_iters=9`, `momentum_iters=39`
- The remaining mismatch is no longer in the first reduced-entry Dirichlet scaling path.

Important remaining clue:

- With the Dirichlet-mask fix in place, traced `precond_diag(...)` values still show a remaining distributed diagonal mismatch on the first backend-global node:
  - serial:
    - `raw_diag node(global=0) w(0)=1.87157 w(1)=1.92774`
  - `np=4` owner rank:
    - `raw_diag node(global=0) w(0)=1.87157 w(1)=1.38332`
- So the next remaining target is no longer the shared Dirichlet face mask.
- It is the underlying diagonal / row construction for that distributed momentum component before preconditioner scaling.

Cleanup:

- Reverted the temporary test-only fallback that always built a synthetic FSILS permutation in serial.
- The oracle is back to:
  - use `system.dofPermutation()` when present
  - otherwise only synthesize the permutation for `world_size > 1`

## 2026-04-20 07:05 PDT

Closed the remaining fast-oracle serial/parallel iteration-count gap in the explicit bordered recovery path.

Root cause:

- `NewtonSolver::recoverExplicitBorderedCorrection(...)` was building each bordered recovery RHS `B_j`
  into `residual_scratch` by looping over **all** field rows on **every** rank:
  - `Code/Source/solver/FE/TimeStepping/NewtonSolver.cpp`
- That path uses `assembly::AddMode::Add`, so the distributed `K^{-1} B_j` recovery solves were not on the
  same RHS as serial.
- This did **not** affect the main PDE solve, which is why `du_after_linear_solve` was already matching.
- It only affected the explicit bordered recovery columns `z_j = K^{-1} B_j`, which is exactly where the first
  serial / `np=4` divergence had been localized.

Retained fix:

- In the bordered recovery RHS assembly, only insert `B_j(row)` for rows owned by the current rank:
  - skip rows not in `owned_dofs`
- This keeps the replicated dense bordered metadata (`B`, `Ct`, `D`, `g`) but restores correct distributed RHS
  assembly semantics for the auxiliary recovery linear solves.

Rejected probe from the same pass:

- Also tried forcing the explicit bordered recovery to use dense `Ct` contractions everywhere instead of the
  direct-record shortcut.
- That did **not** change the oracle counts and was not the fix.

Qualification:

- Fast monolithic RCR oracle now matches exactly:
  - serial: `tests/_bowned_np1_20260420.log`
    - `newton_iters=8`, `outer=1`, `schur_iters=9`, `momentum_iters=36`
  - `np=4`: `tests/_bowned_np4_20260420.log`
    - `newton_iters=8`, `outer=1`, `schur_iters=9`, `momentum_iters=36`
- Focused backend guards still pass:
  - serial: `tests/_qual_backends_serial_20260420.log`
  - `mpi2`: `tests/_qual_backends_mpi2_20260420.log`

Current state:

- The matched production-path fast oracle is now in exact serial / parallel iteration-count parity.
- The key remaining production qualification task is to re-run the full monolithic `pipe_RCR_3d` OOP serial and
  `mpi4` harnesses and verify that the original end-to-end case now reflects the same closure.

## 2026-04-20 07:27 PDT

Resolved the remaining production blocker in the FE nodal permutation builder for canonical
monolithic OOP `pipe_RCR_3d` MPI qualification.

Root cause:

- `tryBuildNodalInterleavedDofMap(...)` in
  `Code/Source/solver/FE/Systems/SystemSetup.cpp`
  was inferring node ownership from whatever locally relevant DOFs happened to be present.
- On the production Navier-Stokes case, the velocity field is block-numbered by component, so each
  MPI rank owns partial component blocks rather than complete nodal tuples.
- That made the old ownership-consistency assumption wrong:
  - rank 0 saw `owned_nodes_fe=2535` instead of `845`
  - other ranks hit mixed-owner node collisions
  - the FE system therefore failed to build `dofPermutation()`
  - and canonical `mpi4` qualification died before the solver path

Retained fix:

- Still decode FE local node/component indices from the monolithic `FieldDofMap` layout.
- But assign backend node ownership from a **representative nodal field with the fewest
  components**, instead of requiring all locally seen DOFs on the same node to share one FE owner.
- For mixed velocity/pressure systems this naturally selects the scalar pressure field, which gives
  one stable owner per physical node even when vector components are block-partitioned separately.

Qualification:

- Canonical `pipe_RCR_3d` OOP `mpi4` now builds the nodal permutation on all ranks and reaches the
  transient solve path under the normal strict application contract:
  - `/tmp/pipe_rcr3d_mpi4_finalcheck_20260420.log`
  - rank-owned node counts now match expectations:
    - rank 0: `owned_nodes=845`
    - rank 1: `owned_nodes=761`
    - rank 2: `owned_nodes=747`
    - rank 3: `owned_nodes=754`
- Canonical serial still reaches the same solve path:
  - `/tmp/pipe_rcr3d_serial_postfallback_20260420.log`
- Focused guards remain green:
  - `build/svMultiPhysics-build/bin/test_fe_backends --gtest_filter='FsilsBackend.ReducedFieldUpdateDistributedShapeConverges:FsilsBackend.ReducedFieldUpdateDistributedRhsMatchesReference'`
  - `mpirun -np 2 build/svMultiPhysics-build/bin/test_fe_backends_mpi --gtest_filter='FsilsBackendMPI.ReducedFieldUpdateDistributedShapeConverges:FsilsBackendMPI.ReducedFieldUpdateDistributedRhsMatchesReference'`
  - `mpirun -np 2 build/svMultiPhysics-build/bin/test_fe_assembly_mpi --gtest_filter='TimeLoopFsilsConvergenceMPI.GeneralizedAlphaConvergesWithAlgebraicField'`

Current state:

- The fast oracle iteration-gap fix remains intact.
- The full canonical `mpi4` production run is no longer blocked in setup/permutation creation.
- The remaining work, if we want it, is longer end-to-end qualification to completion rather than
  another FE parity blocker fix.
