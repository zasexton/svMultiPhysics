# Free-surface architecture refactoring job ledger

**Plan:** [Architecture refactoring implementation](free_surface_architecture_refactoring_plan_20260904.md).

**Owner scope:** Jobs submitted for this refactoring execution only. An existing job under the same Unix account is not owned by this ledger. No cancel, requeue, resize, priority, time-limit, or other mutation is permitted for an unlisted job.

**Resource ceiling:** `amarsden`; at most four nodes and 80 GB aggregate requested memory across this ledger's nonterminal jobs, counting pending jobs as reserved. Submissions use explicit nodes, tasks, CPUs, memory and wall-time. Resources of unrelated jobs are observed only; their allocations are not reused.

**Scratch namespace:** `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/`.

**Submission procedure:** Record a prepared request, source/build identity, script hash and requested resources before submission; submit with `sbatch --parsable`; immediately record the returned ID. Query `squeue` and `sacct` for that exact ID. Record terminal state, exit code, elapsed time, maximum resident memory and output paths. A missing live queue entry alone is not evidence of success.

**Resumption rule:** Reconcile every listed nonterminal job with Slurm before submitting a replacement. A tool timeout does not establish that the job stopped. Any owned cancellation or resubmission must retain its reason and original job ID.

## Jobs

Five owned jobs have been submitted; three are terminal failures. Their live state and final results are recorded below.

| Job ID | Request | Source/build identity | Nodes | CPUs | Memory | State | Evidence |
|---|---|---|---:|---:|---:|---|---|
| 42068077 | Integrated build; FE/Physics/Application serial and selected MPI checks | `0d77e6cd`; `baseline/core-build`; JIT/Eigen OFF | 1 | 8 (build uses 6) | 38,000 MiB | FAILED (tests, 1:0) | `jobs/baseline-core-request.json`; script SHA-256 `1dc106604f0e5931f38eccdec41a58cfa2df8cf2f0cb79aafc60a6b2ed0704c4` |
| 42068499 | Standalone FE Forms/Assembly/Systems with explicit LLVM JIT | `0d77e6cd`; `baseline/jit-build`; LLVM 17.0.6, Eigen OFF | 1 | 6 (build uses 4) | 38,000 MiB | FAILED (configure, 1:0) | `jobs/baseline-jit-request.json`; script SHA-256 `b10ebd78700766067914bdff009315dc6e530e46a2ed0949ace07ba0cc4dc2cf` |
| 42070565 | JIT baseline with explicit Terminfo dependency | `0d77e6cd`; `baseline/jit-r1-build`; LLVM 17.0.6, ncurses 6.4 | 1 | 6 (build uses 4) | 38,000 MiB | FAILED (build, 2:0) | `jobs/baseline-jit-r1-request.json`; script SHA-256 `86f433a1add39e87e9f5e75ea94dfb9eada0d9d49d105933433a475144bd6709` |

| 42080271 | baseline-jit-r2 | `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/source`; `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/baseline/jit-r2-build` | 1 | 2 (build uses 1) | 6,000 MiB | RUNNING | `jobs/baseline-jit-r2-request.json`; script SHA-256 `62c79a8f6a3f3ec60e6c30301cf538f092176b198ae97629bd093074b407de23` |

| 42080275 | candidate-r1-options | `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/candidate/r1-options-source`; `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/candidate/r1-options-build` | 1 | 6 (build uses 4) | 32,000 MiB | RUNNING | `jobs/candidate-r1-options-request.json`; script SHA-256 `f7f3e74c3c823362f62ec648487b1d9fab3295c533a4c3d3b1989f76e46176ea` |

## Events

- 2026-09-04: Created this ledger before the first refactoring allocation. Existing account jobs were inspected without mutation. No allocation or output directory belonging to another workstream was changed.

- 2026-09-04: Prepared baseline-core, 8-hour wall limit. Reserved total: one node and 38,000 MiB. Preserved the original 52-file source diff and three untracked inputs separately; they are not inputs to this baseline.

- 2026-09-04: `sbatch --parsable` returned owned job `42068077`; receipt retained in `jobs/baseline-core-submission.txt`. No other job ID is authorized for mutation by this execution.

- 2026-09-04: Prepared baseline-jit, 4-hour wall limit. Combined reservation with baseline-core is two nodes and 76,000 MiB (79.69 GB), within the four-node/80 GB ceiling.

- 2026-09-04: Submitted owned JIT-baseline job `42068499`; its receipt is `jobs/baseline-jit-submission.txt`. Only this ID and `42068077` are currently owned.

- 2026-09-04: Exact-ID Slurm inspection confirms `42068077` running and compiling on `sh02-07n60`; `42068499` remains pending for priority. Core CMake configuration completed. No build or test pass is claimed yet.

- 2026-09-04: `sacct` confirms `42068499` FAILED with exit `1:0` after 27 seconds, peak batch RSS 90,292 KiB. LLVM could not discover Terminfo. Retained the failed configure logs and terminal record, then prepared a new directory/profile with the installed ncurses 6.4 Terminfo library explicitly selected. Reserved total remains two nodes / 76,000 MiB including the core run.

- 2026-09-04: Submitted owned replacement JIT-baseline job `42070565`; immutable submission receipt retained. Original failed job `42068499` and its logs remain intact.

- 2026-09-04: Prepared a C++ syntax-verification step inside owned allocation `42068077`: one CPU, 4,000 MiB step limit, at most ten minutes, concurrent with the six-worker baseline build in its eight-CPU/38,000-MiB allocation. This is not a new node/memory reservation. It reads the candidate source and baseline compile commands and writes only `r1-options-check/` plus its own step logs. Shell script SHA-256 `3e580af6a701a1a3365abbe2410a5fe5b1bde3bdda83ba1c2bc8c56f444248c0`; checker SHA-256 `cffe764fe8854becba93891398f90114ced692ba1130c9548eb058f522aa319d`.

- 2026-09-04: The prepared syntax step did not launch: explicit `srun --jobid=42068077` failed allocation confirmation. Exact-ID scheduler records still showed the core job running. The tool shell inherited another workstream's allocation; no check ran there and no unowned allocation was used.
- 2026-09-04: `sacct` confirms `42070565` FAILED with exit `2:0` after 10:35, peak batch RSS 4,463,548 KiB. Compilation mixed LLVM 17 and LLVM 4 headers because loading X11 adds LLVM 4 to CPATH. The failed source, request and logs remain intact.
- 2026-09-04: Prepared `baseline-jit-r2`: one node, two CPUs, 6,000 MiB, one build worker, four hours; removes conflicting compiler include environment after module loading. Also prepared `candidate-r1-options`: one node, six CPUs, 32,000 MiB, four build workers, eight hours, frozen source and five syntax checks followed by focused Physics/Application checks. Including running `42068077`, the reservation is three nodes / 76,000 MiB (79.69 GB). Both prepared requests and script hashes are retained under `jobs/` before submission.

- 2026-09-04: Submitted owned `baseline-jit-r2` job `42080271`; receipt retained in `jobs/baseline-jit-r2-submission.txt`. Only recorded job IDs are owned; failed attempts remain unchanged.

- 2026-09-04: Submitted owned `candidate-r1-options` job `42080275`; receipt retained in `jobs/candidate-r1-options-submission.txt`. Only recorded job IDs are owned; failed attempts remain unchanged.

- 2026-09-04: Prepared a bounded routing probe in owned allocation `42068077` (one CPU, 512 MiB step limit, two minutes). The inherited `SLURM_STEPMGR` points to an unowned allocation; clear inherited step/job routing values and retain explicit `--jobid=42068077`. The probe only prints its job/step/host identity and verifies the owned ID.

- 2026-09-04: Routing probe succeeded in owned job `42068077`, step `0`, host `sh02-07n60`. Clearing inherited `SLURM_STEPMGR` and related routing values restored explicit owned-job steps. Probe receipt and cleared variable names are retained in `jobs/owned-step-routing-probe.json`.

- 2026-09-04: Core build completed and FE serial results report seven passing CTest suites and one failed suite. The sole GoogleTest failure requests `isJITReady()` in the explicit JIT-OFF profile; retained as a failed verification result, with JIT-enabled coverage still required. The core job remains active on later groups.

- 2026-09-04: Prepared the R1 resolver red-stage check inside owned candidate allocation `42080275`: one CPU, 6,000 MiB step limit, fifteen minutes. It compiles only the changed translator and test into a new verification directory, copies the immutable baseline Application archive before replacing its translator object, and links a separate focused test executable. The baseline build is read-only; test working directory is the frozen baseline source. Helper SHA-256 `c0d704431691fbb9061fee9aaec35d5d7f4a24875274aab0f06ad4251b94603b`; shell SHA-256 `2c5f8fe0088e9aa768d557879c35d2a2ed6f1e27851db297f2385d2ca9e32ea2`.

- 2026-09-04: Core job `42068077` reached phase complete with exit `1:0` after 1:31:52, peak batch RSS 6,328,436 KiB. All 491 Physics cases passed. Application reports 17 fixture-root lookup failures; MPI groups failed before tests because the one-task allocation exposed too few slots. These setup failures remain preserved and require corrected reruns. Current jobs `42080271` and `42080275` reserve two nodes / 38,000 MiB.

- 2026-09-04: Candidate job `42080275` compiled the extracted public header and four direct consumers successfully. Its focused baseline checks passed 52 Physics and 84 Application cases; the candidate build and before/after comparison are still in progress.

- 2026-09-04: R1 resolver step `42080275.0` compiled both files but stopped before linking; the helper expected a library entry, while this target compiles Application sources directly. Preserve the attempt as a harness failure, not the intended red test. Prepared a corrected direct-object overlay using the actual test-target compile flags and a new output directory, same owned step limits. Helper SHA-256 `e78fc36550b5e77df06592772dbecc888e4190edddfb2295b3ee85f2d28a5a3f`.
