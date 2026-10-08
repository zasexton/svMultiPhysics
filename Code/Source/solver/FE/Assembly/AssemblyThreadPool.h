/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_ASSEMBLY_ASSEMBLYTHREADPOOL_H
#define SVMP_FE_ASSEMBLY_ASSEMBLYTHREADPOOL_H

/**
 * @file AssemblyThreadPool.h
 * @brief Fork-join thread team of the threaded assembly path.
 *
 * StandardAssembler runs the element/face/cut-volume compute phase of an
 * assembly on several threads when AssemblyOptions::num_threads > 1 (see
 * FE/Docs/ThreadedAssembly.md). The threads are the process-wide team of
 * FE/Core/DeterministicParallel.h (defaultParallelTeam()), which the
 * threaded cut-geometry rebuild uses as well, so a rank has one set of
 * worker threads:
 *  - workers are created on first use and grown on demand; the calling
 *    thread is always participant 0;
 *  - run(n, task) calls task(p) once for every participant p in [0, n) and
 *    returns after all of them finished, so the caller sees every write the
 *    participants made (mutex/condition-variable hand-off);
 *  - an exception thrown by a participant is captured and the one of the
 *    lowest-numbered failing participant is rethrown on the caller;
 *  - a run() issued from inside a participant (of an assembly or a geometry
 *    loop) executes its participants one after another on that thread;
 *  - every worker sets its OpenMP thread count to one, so OpenMP regions that
 *    kernels open internally stay serial on worker threads.
 *
 * The team uses standard mutexes and condition variables (transparent to
 * ThreadSanitizer). Optionally the pipelined assembly loop polls for a while
 * before a thread blocks (spinWait, SVMP_ASSEMBLY_SPIN_US, off by default);
 * this saves context switches but did not shorten the measured loops.
 * Polling changes only timing. The team is independent of OMP_NUM_THREADS.
 */

#include "Core/DeterministicParallel.h"

#include <chrono>
#include <functional>
#include <thread>

namespace svmp {
namespace FE {
namespace assembly {

class AssemblyThreadPool final : public ParallelTeam {
public:
    /// The process-wide team (forwards to FE::defaultParallelTeam()).
    [[nodiscard]] static AssemblyThreadPool& global();

    AssemblyThreadPool() = default;
    AssemblyThreadPool(const AssemblyThreadPool&) = delete;
    AssemblyThreadPool& operator=(const AssemblyThreadPool&) = delete;

    /**
     * @brief Run task(p) for p = 0..n_participants-1 and wait for all of them.
     *
     * Participant 0 runs on the calling thread. With n_participants <= 1, or
     * when called from inside a participant, the participants run in order on
     * the calling thread. Callers whose participants wait for each other
     * must reserveWorkers(n_participants - 1) first.
     */
    void run(int n_participants, const std::function<void(int)>& task) override;

    /// True on a team worker and on a caller while it executes participant 0.
    [[nodiscard]] static bool insideParallelRegion() noexcept;

    /// Number of worker threads currently alive (excluding callers).
    [[nodiscard]] int workerCount() const;

    /// Starts worker threads until there are at least `n_workers`; false if
    /// a thread could not be created (the caller should then run serially).
    [[nodiscard]] bool reserveWorkers(int n_workers) noexcept;

    /// Polling time before a thread of the threaded assembly blocks:
    /// SVMP_ASSEMBLY_SPIN_US microseconds (default 0: block at once).
    [[nodiscard]] static std::chrono::nanoseconds spinBudget() noexcept;

    /// Polls ready() (a cheap, thread-safe check) for up to spinBudget();
    /// returns its last value. The caller then blocks if it is still false.
    template <class Ready>
    static bool spinWait(Ready&& ready)
    {
        if (ready()) {
            return true;
        }
        const auto budget = spinBudget();
        if (budget.count() <= 0) {
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        for (unsigned i = 1;; ++i) {
            cpuRelax();
            if (ready()) {
                return true;
            }
            if ((i & 63u) == 0u && std::chrono::steady_clock::now() - start >= budget) {
                return ready();
            }
        }
    }

private:
    static void cpuRelax() noexcept
    {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLYTHREADPOOL_H
