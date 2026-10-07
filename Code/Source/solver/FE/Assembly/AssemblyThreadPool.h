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
 * @brief Persistent fork-join thread team for the threaded assembly path.
 *
 * StandardAssembler runs the element/face/cut-volume compute phase of an
 * assembly on several threads when AssemblyOptions::num_threads > 1 (see
 * FE/Docs/ThreadedAssembly.md). This pool provides the threads:
 *  - one process-wide team of std::thread workers, created on first use and
 *    grown on demand; the calling thread is always participant 0;
 *  - run(n, task) calls task(p) once for every participant p in [0, n) and
 *    returns after all of them finished, so the caller sees every write the
 *    participants made (mutex/condition-variable hand-off);
 *  - an exception thrown by a participant is captured and the one of the
 *    lowest-numbered failing participant is rethrown on the caller;
 *  - a run() issued from inside a participant executes its participants one
 *    after another on that thread (no nested teams);
 *  - every worker sets its OpenMP thread count to one, so OpenMP regions that
 *    kernels open internally stay serial on assembly threads.
 *
 * The pool uses only standard mutexes and condition variables, which keeps it
 * transparent to ThreadSanitizer. It is independent of OMP_NUM_THREADS.
 */

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace svmp {
namespace FE {
namespace assembly {

class AssemblyThreadPool {
public:
    /// Process-wide pool shared by every assembler.
    [[nodiscard]] static AssemblyThreadPool& global();

    AssemblyThreadPool() = default;
    ~AssemblyThreadPool();

    AssemblyThreadPool(const AssemblyThreadPool&) = delete;
    AssemblyThreadPool& operator=(const AssemblyThreadPool&) = delete;

    /**
     * @brief Run task(p) for p = 0..n_participants-1 and wait for all of them.
     *
     * Participant 0 runs on the calling thread. With n_participants <= 1, or
     * when called from inside a participant, the participants run in order on
     * the calling thread.
     */
    void run(int n_participants, const std::function<void(int)>& task);

    /// True on a pool worker and on a caller while it executes participant 0.
    [[nodiscard]] static bool insideParallelRegion() noexcept;

    /// Number of worker threads currently alive (excluding callers).
    [[nodiscard]] int workerCount() const;

private:
    void ensureWorkersLocked(int n_workers);
    void workerLoop(int worker_index, std::uint64_t start_generation);

    mutable std::mutex mutex_{};
    std::condition_variable start_cv_{};
    std::condition_variable done_cv_{};
    std::vector<std::thread> threads_{};
    std::uint64_t generation_{0};
    int active_participants_{0};
    int remaining_{0};
    const std::function<void(int)>* task_{nullptr};
    std::vector<std::exception_ptr> errors_{};
    bool stop_{false};

    /// Serializes run() calls issued concurrently by different caller threads.
    std::mutex run_mutex_{};
};

} // namespace assembly
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_ASSEMBLY_ASSEMBLYTHREADPOOL_H
