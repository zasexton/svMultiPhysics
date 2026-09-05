# Free-surface architecture refactoring job ledger

**Plan:** [Architecture refactoring implementation](free_surface_architecture_refactoring_plan_20260904.md).

**Owner scope:** Jobs submitted for this refactoring execution only. An existing job under the same Unix account is not owned by this ledger. No cancel, requeue, resize, priority, time-limit, or other mutation is permitted for an unlisted job.

**Resource ceiling:** `amarsden`; at most four nodes and 80 GB aggregate requested memory across this ledger's nonterminal jobs, counting pending jobs as reserved. Submissions use explicit nodes, tasks, CPUs, memory and wall-time. Resources of unrelated jobs are observed only; their allocations are not reused.

**Scratch namespace:** `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/`.

**Submission procedure:** Record a prepared request, source/build identity, script hash and requested resources before submission; submit with `sbatch --parsable`; immediately record the returned ID. Query `squeue` and `sacct` for that exact ID. Record terminal state, exit code, elapsed time, maximum resident memory and output paths. A missing live queue entry alone is not evidence of success.

**Resumption rule:** Reconcile every listed nonterminal job with Slurm before submitting a replacement. A tool timeout does not establish that the job stopped. Any owned cancellation or resubmission must retain its reason and original job ID.

## Jobs

Ten owned jobs have been submitted: four terminal failures, four completed successfully, and two nonterminal. Their combined reservation is two nodes / 64,000 MiB.

| Job ID | Request | Source/build identity | Nodes | CPUs | Memory | State | Evidence |
|---|---|---|---:|---:|---:|---|---|
| 42068077 | Integrated build; FE/Physics/Application serial and selected MPI checks | `0d77e6cd`; `baseline/core-build`; JIT/Eigen OFF | 1 | 8 (build uses 6) | 38,000 MiB | FAILED (tests, 1:0) | `jobs/baseline-core-request.json`; script SHA-256 `1dc106604f0e5931f38eccdec41a58cfa2df8cf2f0cb79aafc60a6b2ed0704c4` |
| 42068499 | Standalone FE Forms/Assembly/Systems with explicit LLVM JIT | `0d77e6cd`; `baseline/jit-build`; LLVM 17.0.6, Eigen OFF | 1 | 6 (build uses 4) | 38,000 MiB | FAILED (configure, 1:0) | `jobs/baseline-jit-request.json`; script SHA-256 `b10ebd78700766067914bdff009315dc6e530e46a2ed0949ace07ba0cc4dc2cf` |
| 42070565 | JIT baseline with explicit Terminfo dependency | `0d77e6cd`; `baseline/jit-r1-build`; LLVM 17.0.6, ncurses 6.4 | 1 | 6 (build uses 4) | 38,000 MiB | FAILED (build, 2:0) | `jobs/baseline-jit-r1-request.json`; script SHA-256 `86f433a1add39e87e9f5e75ea94dfb9eada0d9d49d105933433a475144bd6709` |
| 42080271 | baseline-jit-r2 | `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/source`; `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/baseline/jit-r2-build` | 1 | 2 (build uses 1) | 6,000 MiB | COMPLETED (0:0) | `jobs/baseline-jit-r2-request.json`; script SHA-256 `62c79a8f6a3f3ec60e6c30301cf538f092176b198ae97629bd093074b407de23` |
| 42080275 | candidate-r1-options | `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/candidate/r1-options-source`; `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/candidate/r1-options-build` | 1 | 6 (build uses 4) | 32,000 MiB | COMPLETED (0:0) | `jobs/candidate-r1-options-request.json`; script SHA-256 `f7f3e74c3c823362f62ec648487b1d9fab3295c533a4c3d3b1989f76e46176ea` |
| 42082703 | Corrected core fixture/MPI execution | `0d77e6cd`; unchanged `baseline/core-build` | 1 | 4 tasks, 1 CPU each | 16,000 MiB | FAILED (1:0) | `jobs/baseline-core-rerun-request.json`; script SHA-256 `1372bdb9edec3874c3c4b31dd3c857a172bb2c2b06b3de59ba999b5b1a103e1b` |
| 42083671 | Resolved level-set green and integrated candidate | `6f5f56d6` plus seven frozen source files; `candidate/r1-resolved-build` | 1 | 6 (build uses 4) | 32,000 MiB | COMPLETED (0:0) | `jobs/candidate-r1-resolved-request.json`; script SHA-256 `9a7f7d5f5aff802f15b53609c8c8a065b2b83e71f05009f3009f3a2ebecee814` |
| 42086668 | Explicit serial MPI launch and remaining baseline groups | `0d77e6cd`; unchanged `baseline/core-build` | 1 | 4 tasks, 1 CPU each | 16,000 MiB | COMPLETED (0:0) | `jobs/baseline-core-rerun-r1-request.json`; dependent on `42082703` |
| 42090895 | Enabled-feature numerical baseline | `0d77e6cd`; `baseline/enabled-features-build`; Eigen/JIT ON | 1 | 8 requested; 4 allocated (build uses 4) | 32,000 MiB | RUNNING | `jobs/baseline-enabled-features-request.json`; script SHA-256 `93bbce1338efab6c422f41982fcb6ab73f633e3f019c8a830fe9c483e18f5fd4` |

| 42093152 | Sequential candidate verification allocation | `candidate/validation-cache-source`; per-step immutable Git/input snapshots; new build cache | 1 | 6 tasks/CPUs (build uses 4) | 32,000 MiB | RUNNING | `jobs/candidate-validation-cache-request.json`; script SHA-256 `0d67b34b093241484f40a42ca40303336c8cdc626506350e280dd48fe0d5782c` |

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

- 2026-09-04: Prepared `baseline-core-rerun`: one node, four MPI tasks, 16,000 MiB, two hours. Reuses unchanged core binaries, reruns only the 17 fixture failures from the frozen source directory and the seven MPI groups that never launched, and requires per-rank XML results. Combined reservation with `42080271` and `42080275`: three nodes / 54,000 MiB. Request, selection, helpers and script hashes recorded before submission.

- 2026-09-04: Resolver red step `42080275.1` compiled/linked successfully and all three new tests failed at their intended missing behavior. The extraction is now staged for green verification. Prepared one CPU / 6,000 MiB / fifteen-minute step in the same owned allocation, using a new result directory and the full LevelSetEquationTranslator fixture; shell SHA-256 `f3f2ebefd8f1ce5b865eeee37d701b58e7f1ed8a2acdb296ac4ee8644ea11398`.

- 2026-09-04: Submitted owned baseline correction job `42082703`; immutable receipt retained in `jobs/baseline-core-rerun-submission.txt`. Reserved total: three nodes / 54,000 MiB including the two active builds.

- 2026-09-04: Option candidate `42080275` completed successfully (0:0); five syntax checks, 52 Physics and 84 Application cases passed for both baseline and candidate. The prepared green step was not launched because that allocation ended. Prepared a fresh `candidate-r1-resolved` job: one node, six CPUs, 32,000 MiB, four build workers; frozen source at `6f5f56d6d6ee36b448ab28a06aa9d45afc42ae1d` plus seven recorded files, focused green followed by a configured build and full serial Application checks. With owned jobs `42080271` and `42082703`, reserved total will be three nodes / 54,000 MiB. Script SHA-256 `9a7f7d5f5aff802f15b53609c8c8a065b2b83e71f05009f3009f3a2ebecee814`.

- 2026-09-04: Submitted owned resolver validation job `42083671`; receipt retained under `jobs/`. Two existing owned jobs plus this reservation: three nodes / 54,000 MiB.

- 2026-09-04: Parsed individual test outcomes before accepting the option extraction: 135 passed and one explicit JIT-only skip in each baseline/candidate run, with identical selections and skip identity. Earlier counts of 136 referred to selected cases. The core Physics profile likewise has 466 passed, 25 skipped and zero failed among 491 selected. Updated the live manifest; retained all raw results. Option candidate source hashes match all three reviewed files; batch peak RSS was 4,741,896 KiB and elapsed time 18:42. Current reservation remains three nodes / 54,000 MiB including pending `42083671`.

- 2026-09-04: Prepared a read-only stack inspection of owned Application process 170058 in job `42082703`: one CPU, 512 MiB, two-minute step limit, 45-second debugger timeout. The serial fixture is stalled after entering its MPI initialization helper; no solver source or test input will change. Step resources remain inside the four-CPU / 16,000-MiB parent allocation. Script SHA-256 `a8d333a4bc91a25414bd13d035c9315651e86056a5147da47016f82ef516c5c2`.

- 2026-09-04: Owned inspection step in `42082703` confirmed its serial fixture is waiting inside `ompi_mpi_init`; debugger detached normally. Prepared `baseline-core-rerun-r1` with an explicit one-rank MPI launcher for serial Application fixtures and unchanged declared two/four-rank commands. It is dependent on termination of `42082703`, preventing overlapping fixture writers; requested one node / four tasks / 16,000 MiB / two hours. If submitted while all three current jobs remain nonterminal, the reservation is four nodes / 70,000 MiB (73.4 GB). Helper records actual passes/skips on every rank. Script SHA-256 `d2661be0ae6ae9bfecb56b674935867488f638fe6ba614a4b976382d1b2f8693`.

- 2026-09-04: Submitted owned dependent baseline correction `42086668`; request and receipt retained. Maximum reservation including the prior job is four nodes / 70,000 MiB. No unowned job was changed.

- 2026-09-04: Prepared the serial wet-block capture overlay inside owned candidate job `42083671`: one CPU, 6,000 MiB, twenty-minute step limit, within its six-CPU / 32,000-MiB allocation. Only the frozen overlaid Physics test translation unit is compiled; all headers, remaining objects and numerical libraries come from immutable baseline `0d77e6cd`. The two existing serial cases run with capture disabled and in separate reference/repeat output roots. Compiler/link/test commands and digests are recorded; independent artifact validation remains required. Script SHA-256 `d5cd3284e7321e9e271d0507b5a0aa0b3dd46a37396285549bc747cf17e296c0`; helper SHA-256 `f55d1e2bfb3c447541d0230949a1b6172d83d78aa6ec67fd327b8e82d1897376`.

- 2026-09-04: Serial capture step `42083671.0` failed at a matrix span/view type mismatch. Prepared a new frozen overlay with the parameter using the existing `DenseMatrixView` contract; previous source and logs remain immutable. Retry is one CPU / 6,000 MiB inside `42083671`, same twenty-minute limit. Script SHA-256 `24b5e27882dbf27c1ae7e19729874cc5b9840a627ab1a4ac8eefe90fdbaeb862`; overlay SHA-256 `e05861857a80e5b7515fcdb44c3029fd53322213268821c60b5aa2e00ed2f037`.

- 2026-09-04: Prepared an independent `baseline-enabled-features` profile at immutable `0d77e6cd`: LLVM 17 JIT and Eigen ON, four allocated MPI tasks, four build workers, 32,000 MiB, eight hours. It covers Eigen-dependent capillary history and the JIT-only transport case, retains the existing serial matrix exclusion, records per-rank outcomes, and adds the specific one/four-rank FE cases skipped in the two-rank run. With remaining JIT job `42080271`, maximum reservation is two nodes / 38,000 MiB. Script SHA-256 `93bbce1338efab6c422f41982fcb6ab73f633e3f019c8a830fe9c483e18f5fd4`; four tasks with two CPUs each reserve room for owned overlay checks while compilation uses four workers.

- 2026-09-04: Submitted owned enabled-feature baseline `42090895`; immutable source and receipt recorded. With `42080271`, maximum reservation is two nodes / 38,000 MiB.

- 2026-09-04: Prepared post-review serial capture validation in owned `42090895`: one CPU / 6,000 MiB / twenty minutes within its eight-CPU / 32,000-MiB allocation. Frozen overlay `1dcabf85f1e3eee5e822c039142a800790bac891d23d4701089042deb38b75e9` addresses unavailable fractions, exclusive publication, identity validation and missing gates. Besides disabled/reference/repeat runs, validation rejects existing destinations, existing temporary files and invalid provenance. Script SHA-256 `e8322b86daa63e52ab99e6990997d32b0ba3cc3143f8e3803ed589054fad6fae`.

- 2026-09-04: Reconciled exact owned IDs with scheduler accounting. `42082703` FAILED (1:0, 20:13, 118,332 KiB batch peak RSS) after a 1,200-second serial MPI initialization timeout. Its replacement `42086668` COMPLETED (0:0, 00:45, 668,084 KiB) with eight successful groups: 127 logical passes/four rank-layout skips; 239 per-rank passes/eight skips. All per-rank XML is present. Failed attempts remain intact.

- 2026-09-04: `42083671` COMPLETED (0:0, 23:10, 5,236,452 KiB batch peak RSS). The focused resolver check passed 23 tests; the configured candidate passed 309 Application tests/four Eigen skips and 51 Physics tests/one JIT skip. Baseline options rerun from frozen-source cwd passed 84 Application and 51 Physics tests with the same JIT skip. All seven candidate source hashes match the implementation.

- 2026-09-04: `42080271` COMPLETED (0:0, 1:06:14, 2,166,004 KiB batch peak RSS). Parsed individual XML: 2,145 passed and 27 skipped among 2,172 selected Forms/Assembly/Systems cases. This profile does not build LevelSet; its JIT-only case remains assigned to `42090895`.

- 2026-09-04: `42090895` is RUNNING. Scheduler records eight requested CPUs but four allocated CPUs; the prepared overlay check `42090895.0` shares one bound CPU with the build inside the allocated four CPUs / 32,000 MiB. The earlier spare-CPU description was based on the request, not the observed allocation. Only this job remains nonterminal: one node / 32,000 MiB. The serial capture remains unaccepted pending new execution, artifact validation and scoped review.

- 2026-09-04: Prepared a reusable candidate validation allocation: one node, six MPI tasks/CPUs, 32,000 MiB, eight hours, four build workers. Each verification step freezes a retained Git source snapshot and materialized-input manifest, then records commands, binary identity and results; the cache changes only between terminal steps. All prior frozen baseline/candidate paths remain intact. With running 42090895, total reservation is two nodes / 64,000 MiB (67.11 GB). Script SHA-256 `0d67b34b093241484f40a42ca40303336c8cdc626506350e280dd48fe0d5782c`.

- 2026-09-04: Submitted owned candidate validation allocation `42093152`; receipt and exact request retained. Together with `42090895`, reservation is two nodes / 64,000 MiB. Each test/build step will be recorded separately; the holding batch exit is not numerical validation.

- 2026-09-04: Prepared R1 builder runtime-red step inside owned `42093152`: one CPU / 6,000 MiB / twenty minutes. Source snapshot `9a8c51e9` retains three test/scaffolding files. Recompiles all six direct Application objects dependent on the changed headers/tests, using the frozen resolved candidate numerical libraries; no mixed `SimulationComponents` layout. The six-case filter must establish runtime missing-retention failures before implementation. Script SHA-256 `53d21d19baf6b32fb281d9defd6de5a42dd9eaccf36ed075ff68fcf6c9399c8e`.

- 2026-09-04: Prepared a configured candidate build step in `42093152`: four CPUs / 24,000 MiB / ninety minutes, LLVM 17 JIT and Eigen ON, complete Application test target at retained red source `9a8c51e9`. Together with its one-CPU / 6,000-MiB red overlay step, this stays within six allocated CPUs / 32,000 MiB. Source cache remains frozen until both reader steps terminate. Script SHA-256 `8ae0b76e41cdd5d9a58f31078c7a925986a6df16c31936d3bc6697923e83c052`.

- 2026-09-04: Red step `42093152.0` stopped before compilation when its inventory guard also detected the intentionally updated live baseline manifest. Preserve this as a harness failure. Prepared a new helper/result directory recording that exact metadata-only difference while still requiring the three intended changed source files and identical remaining input hashes/membership. Same one CPU / 6,000 MiB / twenty-minute limits. Script SHA-256 `baad53b5e76366172314c7d4fe1e49c379c10a86e8070854677e79ccb6467a67`.

- 2026-09-04: Red retry `42093152.2` stopped before compilation because the system Git cannot resolve the linked worktree. The configured build already uses Git 2.45.1 successfully; pin the same executable in a new overlay retry, retain both failed harness attempts, and keep the source/tests unchanged. Same one CPU / 6,000 MiB / twenty-minute limits. Script SHA-256 `00b1dd1fc295cbe2d67703aed38ce2679f170edfa7fb9810a47915fe40813365`.

- 2026-09-04: Capture step `42090895.0` COMPLETED (0:0, 1:42, 1,409,016 KiB). Six expected success/rejection groups have complete individual XML and meaningful outcomes. Independent artifact validation passed and was archived under `verification/r0-wet-block-serial-r2-acceptance/`. Accepted reference SHA-256 values: physical `139491d55b566a87fadc3b1868de528766c327272b84f5da7069c72d404203d2`; islands `faddf56ff28241269a598dd0432779eb40cd634a28d0baa17bab71d094c361e9`. Both repeats are byte-identical. Prior rejected captures remain unaccepted and intact; full R0 is incomplete.
