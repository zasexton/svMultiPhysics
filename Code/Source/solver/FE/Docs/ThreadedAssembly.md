# Threaded Assembly

This note describes the opt-in multithreaded assembly in `StandardAssembler`.
Threads work inside one MPI rank, so that a rank can use several cores of a
node in addition to MPI.  The motivation is the 3D free-surface cases, where a
rank's assembly is too slow and 2D meshes are too small to scale past about 8
ranks.

## Contract: results do not depend on the thread count

With `N` assembly threads the assembled matrix and vector are bitwise
identical to the serial assembly, for every `N`.  The default (`N = 1`) runs
exactly the serial code.  This is a hard requirement (it matches the
preference for results that do not depend on the number of ranks or
threads), and it rules out the usual shortcuts:

- no graph colouring of elements, which changes the order in which a global
  entry receives its contributions;
- no atomic or locked accumulation into the global system;
- no per-thread copies of the global system reduced at the end.

Instead every global entry receives its contributions in the original item
order, as in the serial loop.

## Scheme: parallel local work, ordered global insertion

Each threaded loop runs in two phases.

1. **Setup (serial).**  The unchanged preamble of the loop runs on the
   calling thread: DOF tables, resolved insertion tables, field access plans,
   owned-row views, kernel metadata.  The threaded path additionally builds
   the tables the serial loop would build lazily on first use: the resolved
   solution-gather tables of the history solution views and the resolved
   insertion tables of the loop's matrix and vector views.  Nothing builds a
   shared table while the threads run (guarded).
2. **Compute and insert (concurrent).**  The items (cut-volume rules,
   interior faces, cell batches, ...) are split into fixed-size blocks.
   Block `b` is computed by compute thread `b mod N`, so the assignment is
   fixed for a given item list and thread count; this keeps each thread's
   caches warm across Newton iterations.  Each compute thread runs the same
   per-item code as the serial loop on its own worker copy of the assembler
   (below).  Where the serial loop calls an insertion routine
   (`insertLocalForCell`, `insertLocal`, `insertLocalConstrained`,
   `addMatrixEntries`), the thread records the call, with a copy of the local
   output and DOF lists, in the block's `DeferredInsertBuffer`
   (`FE/Assembly/DeferredInsertBuffer.h`).  Meanwhile the calling thread
   replays the blocks strictly in item order, each as soon as it is complete,
   through the same insertion routines.  The global system therefore receives
   exactly the serial sequence of calls with identical values; constraint
   distribution, owned-row filtering and the reverse scatter of off-rank rows
   (MPI) are unchanged.

The record buffers form a ring of 16 blocks per compute thread, so compute
threads run at most that far ahead of the insertion and the recorded outputs
need a few MB per thread, independently of the mesh size.  With `N`
assembly threads a rank runs `N` compute threads plus the calling thread,
which only inserts and waits otherwise; insertion overlaps with computation,
so a loop takes about (compute + insert) / `N` instead of compute / `N` +
insert.  Counters (`AssemblyResult`) are summed per block; they are integers.

Per-item results are bitwise identical to the serial ones because each item's
local computation depends only on the item and on read-only shared data.
The assembler's caches that carry state between items (basis scratch,
geometry mapping, the cut-volume epoch cache, the basis-cache handles) are
exact: a hit returns the arrays a miss would compute, and the state after a
hit equals the state after a miss.  The per-thread caches therefore change
hit rates but not values.  The batched cell path keeps its batch boundaries,
because block boundaries are multiples of the batch size.

## What is threaded

| Loop | Function | Item | Status |
|---|---|---|---|
| Cut volumes (fused terms; full and partial rules) | `assembleCutVolumesFused` | rule | threaded |
| Interior faces (ghost penalty, DG) | `assembleInteriorFaces` | face | threaded |
| Cut interfaces | `assembleCutInterfaces` | interface rule | threaded |
| Cells, monolithic kernel, matrix and vector | `assembleCellsFused` | batch of 32 cells | threaded with the coupled scalar-basis cache (affine simplices) |
| Cells, monolithic kernel, residual only | `assembleCellsFused` | cell | threaded after the first cell |
| Cells, other `assembleCellsFused` paths (mixed blocks, fused batches) | | | serial |
| Boundary faces, interface faces, single-kernel cut volumes | | | serial (small share) |

The free-surface runs use the first five rows; together they are the
assembly time of the profiles (cut volumes about half of it).

Two cell paths carry state from cell to cell that affects values: whether
field values are evaluated from cached recipes or by the general routine
(the two round differently, mathematically equal) depends on prepareBasis
calls earlier in the loop.  The batched path is threaded only when it never
calls prepareBasis (coupled scalar cache with affine cells; a thread that
meets another cell defers to the serial loop).  In the residual-only path the
state is settled within the first cell and then the same for every cell, so
the first cell runs serially and every worker starts from the resulting
state.

A loop runs serially when the thread count is 1, when it has fewer items
than a small threshold, when it is called from an assembly thread, or when a
feature that is not thread-safe or that writes ordered diagnostics is
active: per-item diagnostic logging (`SVMP_FE_CUT_VOLUME_*` diagnostics and
the direct-PSPG topology policy), kernels that need material state, the
opt-in cut-volume basis cache.  Cut rebuilds, snapshots, constraint builds and
the linear solver are outside assembly and stay serial.

## Per-thread state

Each thread uses a worker `StandardAssembler` owned by the calling assembler
(created on first use, reconfigured on every threaded call).  A worker shares
the configuration (DOF maps, constraints, solution views and spans, history,
time, parameters, cut integration context, field access, JIT constants) by
pointer, and owns all mutable per-item state: `AssemblyContext`s, kernel
outputs, geometry mapping and scratch, basis-cache handles, the cut-volume
epoch cache, the field-evaluation caches.  The large read-only tables built in
the setup phase stay with the calling assembler and are read by the workers:

- cell DOF tables, resolved solution-gather tables, field access plans.

A worker never builds or modifies these.  If an item needs a table that the
setup did not build, its thread stops at that block; the blocks before it are
inserted as usual, the calling thread runs the stopped block serially
(building the table as the serial loop would), and the threads resume after
it.  The insertion sequence is then still the serial one.  After three passes
in a row that stop at their first block, or when the serial block rebuilt the
field access plans the workers' state refers to, the calling thread finishes
the loop serially.  Invalidation calls on the calling assembler (`reset`,
`invalidateGeometryCaches`, ...) are forwarded to the workers.

Shared objects are only read during the compute phase: the mesh, the cut
integration context, function spaces and elements, DOF maps, the solution
vectors (`getVectorEntriesResolved`), the constraints (queries only) and the
kernels.

## JIT kernels and other lazy one-time work

Kernel objects are shared by the threads.  Their compute entry points are
safe for concurrent calls once the kernel is compiled: scratch is
`thread_local` or local, compiled dispatch tables are immutable, and mutable
bookkeeping is under the wrapper's mutex.  What is not safe, and not
deterministic, is lazy one-time work: generic and specialized JIT compiles
(the generated code depends on a global calibration fed by earlier compiles,
and a second thread asking for a specialization while it is compiled would
silently use the generic kernel), the interpreter's first-use lowering of
indexed access in the form IR, and the switch to the interpreter after a JIT
runtime failure.

Assembly threads therefore run in a *no lazy work* mode
(`assembly::ConcurrentComputeScope`).  When a kernel would compile, lower its
IR or record a runtime failure in this mode, it throws
`assembly::DeferredSerialWork` instead; the calling thread runs that block
serially, where the compile happens, and the threads resume after it (other
errors on a thread are handled the same way, so the serial run of the block
raises them at the same item).  All lazy work thus happens in the serial
order, the same as without threads.  Only the first assemblies of a run meet
such work (new kernel shapes); later calls run threaded throughout.  The composite
cut-volume kernel's scratch output, previously a shared mutable member, is
thread-local.  The thread pool also sets the OpenMP thread count of
its threads to one, so OpenMP regions inside kernels stay serial.

Every kernel call used to take the wrapper's mutex three or four times
(compile check, primed-shape lookups, specialization lookup) and copy a
`shared_ptr` to the dispatch table.  With several threads calling the same
kernels these became the bottleneck for small elements (2D), so the hot path
is now lock-free: the compile check reads an atomic "settled" flag, and the
specialization lookup is answered from a thread-local memo keyed by (wrapper,
dispatch epoch, role, domain, context sizes).  The epoch is incremented under
the mutex by every change of the state the lookup depends on (revision,
compiler, specialized and attempted variants, primed shapes), so a memo entry
is used only while the locked lookup would return the same dispatch object.
The default (one thread) runs the same kernels with the same dispatch tables.

## Choosing the thread count

- Deck: `<Assembly_threads>N</Assembly_threads>` in
  `<GeneralSimulationParameters>` (default 1).
- Environment: `SVMP_ASSEMBLY_THREADS=N` overrides the deck value.

Both set `AssemblyOptions::num_threads`; values below 1 are ignored, and
the solver log reports the value and its source.  Assembly threads are used
only with one OpenMP thread.  OpenMP threads change the results of the FSILS
OpenMP reductions, enable the existing coloured cell path in
`assembleCellsFused` (another summation order) and split cell batches of the
JIT batch kernels across threads, so with OpenMP threads the results would
depend on the assembly thread count.  With more than one assembly thread and
`OMP_NUM_THREADS` unset the application therefore keeps OpenMP at one thread
(instead of cores per rank); if OpenMP has more than one thread anyway
(`OMP_NUM_THREADS` > 1), assembly runs serially and a message says so.  If
the pool cannot start its threads, the loops also run serially.

A rank with `N` assembly threads runs `N` compute threads and its main
thread, which inserts in order during the threaded loops and otherwise does
the serial work.

## MPI ranks and threads

Ranks and threads combine: each rank assembles its own cells with `N`
threads.  On Slurm, request ranks with `--ntasks` and threads with
`--cpus-per-task`, and give each rank its cores:

```bash
#SBATCH --ntasks=4 --cpus-per-task=4
export OMP_NUM_THREADS=1 SVMP_ASSEMBLY_THREADS=$SLURM_CPUS_PER_TASK
mpiexec -n $SLURM_NTASKS --map-by slot:PE=$SLURM_CPUS_PER_TASK --bind-to core svmultiphysics solver.xml
# or, without binding: mpiexec -n $SLURM_NTASKS --bind-to none ...
```

Only assembly is threaded; the linear solve, cut rebuild, geometry snapshots
and constraint build use one core per rank.  `ranks x threads` should
normally not exceed the cores of the node, and ranks remain the first choice
where they scale; threads add speed where ranks no longer do (small 2D
meshes beyond about 8 ranks) or where memory per rank limits the rank count
(3D).

## Known limitations

- Kernels with mutable member scratch are not safe for concurrent calls.  In
  the solver this is `CompositeKernel`, used only to wrap matrix-free
  operators (not assembled by these loops); the cut-volume composite kernel
  was changed to thread-local scratch.  Kernels that need material state run
  serially (the provider has one slot per kernel and cell).
- `prepareBasis` computes physical trial gradients by two routines (a fast
  path reused when the previous call had the same spaces, rule and cell type,
  and the general path).  They give identical values except possibly for the
  sign of an exact zero (the general path adds to `+0.0`).  Which one runs
  depends on the previous item, which differs at block starts in two cases
  (a single fused cut-volume term with different test and trial spaces on
  full-side cells, and the residual-only cell loop with one vector block).
  Global entries accumulate from `+0.0`, so the sign of a zero local entry
  cannot change the assembled system; it could only matter inside a kernel
  that is sensitive to the sign of zero (for example `copysign` or `atan2`
  of a basis gradient); no such use is known, and all full-run comparisons
  were bitwise identical.  Making both
  routines start from `+0.0` would remove this; it changes no value but the
  sign of such zeros in the default path.
- `assembleCellsFused` can build a new DOF table during its setup, which
  clears the field access plans after they were built (the serial loop has
  the same order).  Workers then cannot find their plans and the call runs
  serially; results are unaffected.

## Diagnostics

`SVMP_ASSEMBLY_THREAD_TIMING=1` prints to stderr, per call of a
threaded-capable loop, `[ASSEMBLY_LOOP] loop=... items=... threads=...
threaded_items=... time=...` (also with one thread; `threaded_items` counts
up to where the call finished serially), and per threaded pass
`[ASSEMBLY_THREADS] loop=... begin=<first item> compute=<wall of the pass>
insert=<busy time of the inserting thread> serial_from=<item> [stop=<reason>]`;
a stop names the deferred work (for example a JIT specialization compile) or
the error at whose block the pass ended.  Summing
`serial_from - begin` over the passes gives the number of items computed
by the threads.

## Verification

- Bitwise comparison of full runs for 1, 2, 4 and 8 threads against the
  default build: the nine-case reference set, the 3D sphere proxy and the 3D
  tank, serial and on 2 and 4 ranks.
- Unit tests in `FE/Tests/Unit/Assembly/test_ThreadedAssembly.cpp` compare
  threaded and serial assembly of the same system bitwise.
- ThreadSanitizer build of the assembly unit tests.

Results and timings are recorded in `Documentation/free_surface_program_tracker.md`.
