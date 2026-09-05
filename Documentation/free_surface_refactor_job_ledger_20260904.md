# Free-surface architecture refactoring job ledger

**Plan:** [Architecture refactoring implementation](free_surface_architecture_refactoring_plan_20260904.md).

**Owner scope:** Jobs submitted for this refactoring execution only. An existing job under the same Unix account is not owned by this ledger. No cancel, requeue, resize, priority, time-limit, or other mutation is permitted for an unlisted job.

**Resource ceiling:** `amarsden`; at most four nodes and 80 GB aggregate requested memory across this ledger's nonterminal jobs, counting pending jobs as reserved. Submissions use explicit nodes, tasks, CPUs, memory and wall-time. Resources of unrelated jobs are observed only; their allocations are not reused.

**Scratch namespace:** `/scratch/users/zsexton/free-surface-refactor-20260904-905239de/`.

**Submission procedure:** Record a prepared request, source/build identity, script hash and requested resources before submission; submit with `sbatch --parsable`; immediately record the returned ID. Query `squeue` and `sacct` for that exact ID. Record terminal state, exit code, elapsed time, maximum resident memory and output paths. A missing live queue entry alone is not evidence of success.

**Resumption rule:** Reconcile every listed nonterminal job with Slurm before submitting a replacement. A tool timeout does not establish that the job stopped. Any owned cancellation or resubmission must retain its reason and original job ID.

## Jobs

No jobs submitted yet.

| Job ID | Request | Source/build identity | Nodes | CPUs | Memory | State | Evidence |
|---|---|---|---:|---:|---:|---|---|

## Events

- 2026-09-04: Created this ledger before the first refactoring allocation. Existing account jobs were inspected without mutation. No allocation or output directory belonging to another workstream was changed.
