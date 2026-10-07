/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_DETERMINISTICPARALLEL_H
#define SVMP_FE_CORE_DETERMINISTICPARALLEL_H

/**
 * @file DeterministicParallel.h
 * @brief Thread-count independent parallel loops for per-item geometry work.
 *
 * The cut-geometry rebuild computes many independent per-cell (per-rule,
 * per-face) results and then gathers them in a fixed order.  The helpers
 * here run the per-item computations on several threads of one MPI rank
 * under the contract of the threaded assembly (FE/Docs/ThreadedAssembly.md):
 *
 *  - each item is computed by the same code as in the serial loop and
 *    writes only its own result slot; the caller gathers the slots in item
 *    order afterwards, so results do not depend on the thread count;
 *  - items are split into fixed-size blocks and block b runs on participant
 *    b mod N, a fixed assignment for a given item count and thread count;
 *  - when items throw, the exception of the lowest throwing item is
 *    rethrown, the one the serial loop would have raised (items after it
 *    may have run as well; their results are discarded);
 *  - with one thread, or fewer items than a threshold, the loop is the plain
 *    serial loop.
 *
 * The thread count is the threaded-assembly setting: SVMP_ASSEMBLY_THREADS
 * when it is set, otherwise the caller's AssemblyOptions::num_threads.
 * Threads are a team of std::thread workers created on first use
 * (ParallelTeam); each worker keeps OpenMP at one thread.  The team is an
 * interface so that a shared assembly thread pool can provide it.
 */

#include <cstddef>
#include <functional>

namespace svmp {
namespace FE {

/**
 * Fork-join team: run(n, task) calls task(p) for every participant
 * p in [0, n), participant 0 on the calling thread, and returns when all
 * have finished.  An exception of a participant is rethrown on the caller
 * (the lowest-numbered failing participant's).  A run() issued from inside
 * a participant runs its participants in order on that thread.
 */
class ParallelTeam {
public:
    virtual ~ParallelTeam() = default;
    virtual void run(int n_participants,
                     const std::function<void(int)>& task) = 0;
};

/// Process-wide team of std::thread workers, created on first use.
[[nodiscard]] ParallelTeam& defaultParallelTeam();

/**
 * Threads for per-item geometry work: SVMP_ASSEMBLY_THREADS when it holds
 * an integer >= 1, otherwise `assembly_threads` (AssemblyOptions::num_threads
 * of the caller), at least one.
 */
[[nodiscard]] int geometryThreadCount(int assembly_threads = 1) noexcept;

/**
 * Whether SVMP_GEOMETRY_THREADS_SELF_CHECK is set: callers that thread
 * geometry work then also run it serially and require identical results.
 */
[[nodiscard]] bool geometryThreadsSelfCheckEnabled() noexcept;

/**
 * Calls body(item, participant) for every item in [0, n_items) as described
 * in the file comment.  participant is in [0, threads) and identifies the
 * thread's private state (for example a copy of a non-thread-safe
 * evaluator); it is 0 in the serial loop.  The serial loop runs when
 * threads <= 1 or n_items < min_parallel_items, or when called from inside
 * a participant.
 */
void deterministicParallelFor(
    std::size_t n_items,
    int threads,
    const std::function<void(std::size_t, int)>& body,
    std::size_t block_size = 16u,
    std::size_t min_parallel_items = 64u,
    ParallelTeam* team = nullptr);

/// True on a team worker or on a caller while it runs participant 0.
[[nodiscard]] bool insideDeterministicParallelRegion() noexcept;

} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_DETERMINISTICPARALLEL_H
