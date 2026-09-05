# Free-surface architecture refactoring job ledger

**Plan:** [Architecture refactoring implementation](free_surface_architecture_refactoring_plan_20260904.md).

**Owner scope:** Jobs submitted for this refactoring execution only. An existing job under the same Unix account is not owned by this ledger. No cancel, requeue, resize, priority, time-limit, or other mutation is permitted for an unlisted job.

**Resource ceiling:** `amarsden`; at most four nodes and 80 GB aggregate requested memory across this ledger's nonterminal jobs, counting pending jobs as reserved. Submissions use explicit nodes, tasks, CPUs, memory and wall-time. Resources of unrelated jobs are observed only; their allocations are not reused.

**Scratch namespace:** `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/`.

**Submission procedure:** Record a prepared request, source/build identity, script hash and requested resources before submission; submit with `sbatch --parsable`; immediately record the returned ID. Query `squeue` and `sacct` for that exact ID. Record terminal state, exit code, elapsed time, maximum resident memory and output paths. A missing live queue entry alone is not evidence of success.

**Resumption rule:** Reconcile every listed nonterminal job with Slurm before submitting a replacement. A tool timeout does not establish that the job stopped. Any owned cancellation or resubmission must retain its reason and original job ID.

## Jobs

Two owned jobs have been submitted. Their live state and final results are recorded below.

| Job ID | Request | Source/build identity | Nodes | CPUs | Memory | State | Evidence |
|---|---|---|---:|---:|---:|---|---|
| 42068077 | Integrated build; FE/Physics/Application serial and selected MPI checks | `0d77e6cd`; `baseline/core-build`; JIT/Eigen OFF | 1 | 8 (build uses 6) | 38,000 MiB | RUNNING | `jobs/baseline-core-request.json`; script SHA-256 `1dc106604f0e5931f38eccdec41a58cfa2df8cf2f0cb79aafc60a6b2ed0704c4` |
| 42068499 | Standalone FE Forms/Assembly/Systems with explicit LLVM JIT | `0d77e6cd`; `baseline/jit-build`; LLVM 17.0.6, Eigen OFF | 1 | 6 (build uses 4) | 38,000 MiB | PENDING | `jobs/baseline-jit-request.json`; script SHA-256 `b10ebd78700766067914bdff009315dc6e530e46a2ed0949ace07ba0cc4dc2cf` |

## Events

- 2026-09-04: Created this ledger before the first refactoring allocation. Existing account jobs were inspected without mutation. No allocation or output directory belonging to another workstream was changed.

- 2026-09-04: Prepared baseline-core, 8-hour wall limit. Reserved total: one node and 38,000 MiB. Preserved the original 52-file source diff and three untracked inputs separately; they are not inputs to this baseline.

- 2026-09-04: `sbatch --parsable` returned owned job `42068077`; receipt retained in `jobs/baseline-core-submission.txt`. No other job ID is authorized for mutation by this execution.

- 2026-09-04: Prepared baseline-jit, 4-hour wall limit. Combined reservation with baseline-core is two nodes and 76,000 MiB (79.69 GB), within the four-node/80 GB ceiling.

- 2026-09-04: Submitted owned JIT-baseline job `42068499`; its receipt is `jobs/baseline-jit-submission.txt`. Only this ID and `42068077` are currently owned.

- 2026-09-04: Exact-ID Slurm inspection confirms `42068077` running and compiling on `sh02-07n60`; `42068499` remains pending for priority. Core CMake configuration completed. No build or test pass is claimed yet.
