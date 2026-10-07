# Incremental and threaded cut-geometry rebuilds

A free-surface time step rebuilds the generated cut geometry several times
(outer fixed-point passes, the endpoint candidate, maintenance): the
generated interface domain (`LevelSetGeneratedInterfaceLifecycle`), the
authoritative geometry snapshot (`buildFreeSurfaceGeometrySnapshot`), the
cut integration context and the constraints.  Successive passes move the level
set only slightly.  This note describes how a rebuild reuses the work of the
previous one and how the per-cell work runs on several threads, both without
changing any result: the outputs are bitwise identical to a rebuild from
scratch on one thread.

## Reuse between rebuilds

**Generated domain.**  The lifecycle keeps every cell's fragments and regions.
A cell is recomputed only when its level-set coefficients changed, and a
linear cell that stays full on one side keeps its region when its values
change (its signature records only the side).  Cut cells are recomputed.

**Snapshot records.**  `FreeSurfaceGeometrySnapshotReuseCache` (one per
generated domain, held by the driver) keeps the previous snapshot through a
weak pointer and, for every record, the content-only part of its validation:
the ledger counts and maxima of its rule checks and the digest state of its
identity-free content.  A full-cell volume record depends only on the rule's
classification fields, the parent cell and the snapshot policy; its source
identities embed the source value revision.  A new build under the same mesh
(object and revisions), communicator and policy copies the record of a
full-cell volume rule whose inputs compare equal bit for bit (every rule field
except the revision stamps, plus the region's source topology key and
construction observation), re-stamps its identities, replays the content-only
ledger contributions and evaluates the level-set dependent phase checks again.
Every other record is built as before.  The rule content digest covers the
identity-free content followed by the identities, so the kept digest state
completes the digest of a copied record.

**Context and measures.**  A classification-only full-cell rule is imported
into the cut integration context without its points (it released them right
after import before), and its physical measure is taken from the positions
and weights its source region holds.

The assembler, its cut-volume epoch cache and the sparsity pattern persist
across passes and steps: `FESystem::setup()` runs only when the constraint
structure dependencies change, not once per step.

## Threads

The per-cell work of a rebuild runs on `N` threads of one MPI rank:

- the generated-domain cells to (re)compute (cut construction, LinearCorner
  geometry), including every cell of a full rebuild;
- the snapshot records to build (moment certificates, mapping, identity),
  their rule checks and their content digests.

`N` is the threaded-assembly setting: `SVMP_ASSEMBLY_THREADS` when it is set,
otherwise `AssemblyOptions::num_threads` (`FESystem::assemblyThreadCount()`).
The contract is the one of the threaded assembly (`ThreadedAssembly.md`):

- every item (cell, rule, record) is computed by the same code as in the
  serial loop into its own slot, and the slots are gathered in item order;
- items are split into fixed blocks, block `b` on participant `b mod N`;
- a failing item keeps its exception, which is raised where the serial loop
  would have raised it;
- `N = 1`, or few items, runs the serial loop; workers keep OpenMP at one
  thread.

The level-set evaluator caches the last cell's coefficients, so each thread
uses its own copy (`LevelSetCellEvaluator` is copied per participant; the
snapshot's scalar evaluator provides `make_concurrent_copy`).  Counts and
maxima of the validation ledger are merged from per-thread scratch ledgers,
which is exact; floating-point sums stay in the ordered loop.  The helper is
`FE/Core/DeterministicParallel.h`; its `ParallelTeam` interface lets the
assembly thread pool provide the threads.

## Checks and switches

| Setting | Effect |
|---|---|
| `SVMP_DISABLE_INCREMENTAL_REBUILD=1` | build every snapshot record |
| `SVMP_INCREMENTAL_REBUILD_SELF_CHECK=1` | also build every snapshot without the cache on one thread; fail unless identical (`compareFreeSurfaceGeometrySnapshots`) |
| `SVMP_ASSEMBLY_THREADS=N` | `N` geometry (and assembly) threads |
| `SVMP_GEOMETRY_THREADS_SELF_CHECK=1` | recompute every threaded generated-domain cell serially; fail unless identical |
