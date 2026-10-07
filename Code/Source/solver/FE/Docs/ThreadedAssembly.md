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

Each threaded loop runs in three phases.

1. **Setup (serial).**  The unchanged preamble of the loop runs on the
   calling thread: DOF tables, resolved insertion tables, field access plans,
   owned-row views, kernel metadata.  The threaded path additionally builds
   the resolved solution-gather tables for the history solution views, which
   the serial loop would otherwise build lazily on first use.
2. **Compute (parallel).**  The items (cut-volume rules, interior faces, ...)
   are split into fixed-size blocks.  Block `b` is processed by thread
   `b mod N`, so the assignment is fixed for a given item list and thread
   count; this keeps each thread's caches warm across Newton iterations.
   Each thread runs the same per-item code as the serial loop on its own
   worker copy of the assembler (below).  Where the serial loop calls an
   insertion routine (`insertLocalForCell`, `insertLocal`,
   `insertLocalConstrained`, `addMatrixEntries`), the thread records the call,
   with a copy of the local output and DOF lists, in the block's
   `DeferredInsertBuffer` (`FE/Assembly/DeferredInsertBuffer.h`).
3. **Insertion (serial, ordered).**  The calling thread replays the blocks in
   item order through the same insertion routines.  The global system
   therefore receives exactly the serial sequence of calls with identical
   values; constraint distribution, owned-row filtering and the reverse
   scatter of off-rank rows (MPI) are unchanged.

Blocks are processed in waves of a bounded number of blocks, so the recorded
outputs need bounded memory (a few MB per thread) independently of the mesh
size.  Counters (`AssemblyResult`) are summed over threads; they are integers.

Per-item results are bitwise identical to the serial ones because each item's
local computation depends only on the item and on read-only shared data.
The assembler's caches that carry state between items (basis scratch,
geometry mapping, the cut-volume epoch cache, the basis-cache handles) are
exact: a hit returns the arrays a miss would compute, and the state after a
hit equals the state after a miss.  The per-thread caches therefore change
hit rates but not values.  The batched cell path keeps its batch boundaries,
because block boundaries are multiples of the batch size.

## What is threaded

| Loop | Function | Status |
|---|---|---|
| Cut volumes (fused terms) | `assembleCutVolumesFused` | threaded |
| Interior faces (ghost penalty, DG) | `assembleInteriorFaces` | threaded |
| Cells (fused terms) | `assembleCellsFused` | see status in the tracker |
| Cut interfaces | `assembleCutInterfaces` | see status in the tracker |
| Boundary faces, interface faces, single-kernel cut volumes | | serial (small share) |

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
setup did not build, the worker stops, the threaded attempt is discarded
(nothing has been inserted yet) and the loop is run again serially; this
gives the serial result by construction.  Invalidation calls on the calling
assembler (`reset`, `invalidateGeometryCaches`, ...) are forwarded to the
workers.

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
`assembly::DeferredSerialWork` instead, and the loop is redone serially.  All
lazy work thus happens in the serial order, the same as without threads, and
calls after the first assemblies of a run (when every kernel shape has been
compiled) run threaded.  The thread pool also sets the OpenMP thread count of
its threads to one, so OpenMP regions inside kernels stay serial.

## Choosing the thread count

- Deck: `<Assembly_threads>N</Assembly_threads>` in
  `<GeneralSimulationParameters>` (default 1).
- Environment: `SVMP_ASSEMBLY_THREADS=N` overrides the deck value.

Both set `AssemblyOptions::num_threads`.  The thread count is independent of
`OMP_NUM_THREADS`.  Keep `OMP_NUM_THREADS=1`: OpenMP threads change the
results of the FSILS OpenMP reductions and enable the existing coloured cell
path in `assembleCellsFused`, which changes the summation order.  The
application sets OpenMP threads to cores per rank when `OMP_NUM_THREADS` is
unset, so set it explicitly.

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

Only assembly is threaded; the linear solve, cut rebuild and constraint
build use one core per rank, so `ranks x threads` should normally not exceed
the cores of the node and ranks remain the first choice where they scale.

## Diagnostics

`SVMP_ASSEMBLY_THREAD_TIMING=1` prints, per threaded call, the compute and
insertion times and whether the call fell back to the serial loop (and why).

## Verification

- Bitwise comparison of full runs for 1, 2, 4 and 8 threads against the
  default build: the nine-case reference set, the 3D sphere proxy and the 3D
  tank, serial and on 2 and 4 ranks.
- Unit tests in `FE/Tests/Unit/Assembly/test_ThreadedAssembly.cpp` compare
  threaded and serial assembly of the same system bitwise.
- ThreadSanitizer build of the assembly unit tests.

Results and timings are recorded in `Documentation/free_surface_program_tracker.md`.
