# Task 4 bounded minimizer failure evidence report

## Scope and starting state

- Worktree: `/scratch/users/zsexton/wp4-application-regression-fixes-20260902`.
- Supplied and verified clean base:
  `d1e16ea99d8ac3ce50fd5c3e3054882252c7f036`.
- Branch: `wp4-application-regression-fixes-20260902`.
- Target remote branch: `origin/issue-449-modern-mesh-core`.
- No scheduler action or delegated worker was used.

Git 1.8.3 from the initial login path could not interpret the linked-worktree
administrative directory. The intact metadata was verified directly and all
repository operations use the available `git/2.45.1` module.

## Investigation and bounded design

The cap-128 physical run stopped after 104 accepted transitions with terminal
reason
`capillary_merit_line_search_failed:candidate_cut_topology_changed`; it did
not exhaust the transition budget. The minimizer already computes the current
merit, trial merit, Armijo bound, predicted decrease, trial volume error and
both epoch keys inside the existing line-search branch, but returns only its
last textual disposition. The deferred transition reproduction also computes
its own functional values in the existing required evaluation.

The implementation under test retains only a chronological suffix of 64
scalar records owned by the result. It resets the trace before every new
direction/search, including a projected-gradient retry after a failed
limited-memory search. Appending a record does not evaluate the objective,
perform a collective, or retain coefficient and gradient arrays. Uncomputed
trial measurements remain NaN behind explicit unavailability.

## TDD ledger

The public enum and passive default record/result fields were introduced first
only so the behavioral tests could compile. The production recording path was
still absent at that point.

The standalone RED build used GCC 12.4, C++20, one process, a 2 GiB virtual
memory limit, the FE include directory, and the retained read-only GoogleTest
headers and archives named in the brief. The binary was written under
`/scratch/users/zsexton/wp4-minimizer-trace-20260904`.

```text
module load gcc/12.4.0
ulimit -v 2097152
CPLUS_INCLUDE_PATH= LIBRARY_PATH= CPATH= g++ -std=c++20 -O0 -g0 -pthread -ICode/Source/solver/FE -I/scratch/users/zsexton/svMultiPhysics-free-surface-head/build-physics-gcc12/_deps/googletest-src/googletest/include Code/Source/solver/FE/LevelSet/LevelSetStaticCapillaryEquilibrium.cpp Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetStaticCapillaryEquilibrium.cpp /scratch/users/zsexton/wp4-balanced-capillary-v3-eb5dde58-20260903/build-fe-41945918/lib/libgtest_main.a /scratch/users/zsexton/wp4-balanced-capillary-v3-eb5dde58-20260903/build-fe-41945918/lib/libgtest.a -o /scratch/users/zsexton/wp4-minimizer-trace-20260904/test_LevelSetStaticCapillaryEquilibrium_red
/scratch/users/zsexton/wp4-minimizer-trace-20260904/test_LevelSetStaticCapillaryEquilibrium_red --gtest_filter='LevelSetStaticCapillaryEquilibrium.RecordsCrossTopologyDecreaseThatMissesTheArmijoBound:LevelSetStaticCapillaryEquilibrium.RecordsCrossTopologyMeritIncrease:LevelSetStaticCapillaryEquilibrium.RetainsTheLatestSixtyFourLineSearchTrialsInOrder:LevelSetStaticCapillaryEquilibrium.MarksUnavailableTrialMeasurementsWithoutFiniteSubstitutes:LevelSetStaticCapillaryEquilibrium.RecordsTheActualRejectedTopologyReproductionWithoutAnotherProbe'
```

The build succeeded and the behavioral run reported `0 passed, 5 failed`.
Each failure was the expected empty/default result trace while the original
failure reasons, evaluator counts and rollback sentinels had already matched.

The first focused GREEN attempt reported `4 passed, 1 failed`. The remaining
assertion had expected a finite Armijo bound for an unavailable trial, but the
unchanged minimizer branch computes that bound only after evaluation succeeds.
The assertion was corrected to require NaN, consistent with passive recording
and the explicit unavailable state; no calculation was moved across the
evaluator call.

The corrected focused run reported `5 passed, 0 failed`. A sixth focused test
then exercised a failed limited-memory search followed by the existing
projected-gradient retry. It reported `1 passed, 0 failed`, exactly six
functional evaluations, and only the two records from the fallback attempt.

For the reset mutation check, the single call that clears result evidence
before a new search was temporarily removed. The fallback test then failed as
required with `5` accumulated records instead of the expected `2`. The reset
was restored before the final build.

The final standalone command rebuilt the complete translation unit from the
restored source and ran the binary without a filter:

```text
module load gcc/12.4.0
ulimit -v 2097152
CPLUS_INCLUDE_PATH= LIBRARY_PATH= CPATH= g++ -std=c++20 -O0 -g0 -pthread -ICode/Source/solver/FE -I/scratch/users/zsexton/svMultiPhysics-free-surface-head/build-physics-gcc12/_deps/googletest-src/googletest/include Code/Source/solver/FE/LevelSet/LevelSetStaticCapillaryEquilibrium.cpp Code/Source/solver/FE/Tests/Unit/LevelSet/test_LevelSetStaticCapillaryEquilibrium.cpp /scratch/users/zsexton/wp4-balanced-capillary-v3-eb5dde58-20260903/build-fe-41945918/lib/libgtest_main.a /scratch/users/zsexton/wp4-balanced-capillary-v3-eb5dde58-20260903/build-fe-41945918/lib/libgtest.a -o /scratch/users/zsexton/wp4-minimizer-trace-20260904/test_LevelSetStaticCapillaryEquilibrium_full
/scratch/users/zsexton/wp4-minimizer-trace-20260904/test_LevelSetStaticCapillaryEquilibrium_full
```

Result: `29 passed, 0 failed` in one suite. This is the complete standalone
minimizer test file, not a fresh full FE or Application build.

## Application serialization and pending verification

Only the existing prepublication failure exception appends the recognizable
`line_search_trace_summary` and one `line_search_trace_record` per retained
sample. Numeric stream precision is 17 before these fields. The original
failure prefix and summary fields remain in place; no normal-run output path
was added.

Fresh full FE and Application builds and the focused Application workflow test
remain pending for the coordinator because this task was explicitly permitted
to commit after the complete standalone minimizer test passed. The subsequent
bounded physical diagnostic also remains coordinator-owned.

## Concerns

- Application compilation and linked workflow serialization are not claimed
  by the standalone minimizer executable and remain pending as described.
- No physical interpretation is selected by this evidence-only change.
- WP4/Q2 remain open.

Immediately before staging, `git fetch origin issue-449-modern-mesh-core`
resolved both the local and remote heads to
`d1e16ea99d8ac3ce50fd5c3e3054882252c7f036`; ancestry was identical.
