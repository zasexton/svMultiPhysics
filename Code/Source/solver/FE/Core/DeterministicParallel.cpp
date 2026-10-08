/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/DeterministicParallel.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace {

thread_local bool t_inside_region = false;

class InsideRegionScope {
public:
    InsideRegionScope() noexcept : previous_(t_inside_region)
    {
        t_inside_region = true;
    }
    ~InsideRegionScope() { t_inside_region = previous_; }
    InsideRegionScope(const InsideRegionScope&) = delete;
    InsideRegionScope& operator=(const InsideRegionScope&) = delete;

private:
    bool previous_;
};

// Runs participants 0..n-1 in order on the calling thread and rethrows the
// exception of the lowest failing one.
void runParticipantsInOrder(int n_participants,
                            const std::function<void(int)>& task)
{
    std::exception_ptr first_error;
    for (int participant = 0; participant < n_participants; ++participant) {
        try {
            InsideRegionScope inside;
            task(participant);
        } catch (...) {
            if (!first_error) {
                first_error = std::current_exception();
            }
        }
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

class ThreadTeam final : public ParallelTeam {
public:
    ThreadTeam() = default;

    ~ThreadTeam() override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        start_cv_.notify_all();
        for (auto& thread : threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    ThreadTeam(const ThreadTeam&) = delete;
    ThreadTeam& operator=(const ThreadTeam&) = delete;

    void run(int n_participants,
             const std::function<void(int)>& task) override
    {
        if (n_participants <= 1 || t_inside_region) {
            runParticipantsInOrder(n_participants, task);
            return;
        }
        std::lock_guard<std::mutex> run_lock(run_mutex_);
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!ensureWorkersLocked(n_participants - 1)) {
                lock.unlock();
                runParticipantsInOrder(n_participants, task);
                return;
            }
            task_ = &task;
            errors_.assign(static_cast<std::size_t>(n_participants), nullptr);
            active_participants_ = n_participants;
            remaining_ = n_participants - 1;
            ++generation_;
        }
        start_cv_.notify_all();
        try {
            InsideRegionScope inside;
            task(0);
        } catch (...) {
            errors_[0] = std::current_exception();
        }
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [this] { return remaining_ == 0; });
            task_ = nullptr;
            active_participants_ = 0;
        }
        for (const auto& error : errors_) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }

    bool reserveWorkers(int n_workers) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            return ensureWorkersLocked(n_workers);
        } catch (...) {
            return false;
        }
    }

    int workerCount()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(threads_.size());
    }

private:
    // Called with mutex_ held.
    bool ensureWorkersLocked(int n_workers)
    {
        while (static_cast<int>(threads_.size()) < n_workers) {
            const int index = static_cast<int>(threads_.size());
            try {
                threads_.emplace_back(
                    [this, index, generation = generation_] {
                        workerLoop(index, generation);
                    });
            } catch (const std::system_error&) {
                return false;
            }
        }
        return true;
    }

    void workerLoop(int worker_index, std::uint64_t seen_generation)
    {
#if defined(_OPENMP)
        omp_set_num_threads(1);
#endif
        t_inside_region = true;
        for (;;) {
            const std::function<void(int)>* task = nullptr;
            int participant = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&] {
                    return stop_ || generation_ != seen_generation;
                });
                if (stop_) {
                    return;
                }
                seen_generation = generation_;
                participant = worker_index + 1;
                if (participant >= active_participants_) {
                    continue;
                }
                task = task_;
            }
            std::exception_ptr error;
            try {
                (*task)(participant);
            } catch (...) {
                error = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                errors_[static_cast<std::size_t>(participant)] = error;
                --remaining_;
                if (remaining_ == 0) {
                    done_cv_.notify_all();
                }
            }
        }
    }

    std::mutex run_mutex_{};
    std::mutex mutex_{};
    std::condition_variable start_cv_{};
    std::condition_variable done_cv_{};
    std::vector<std::thread> threads_{};
    std::uint64_t generation_{0};
    int active_participants_{0};
    int remaining_{0};
    const std::function<void(int)>* task_{nullptr};
    std::vector<std::exception_ptr> errors_{};
    bool stop_{false};
};

[[nodiscard]] bool positiveEnvInteger(const char* name, int& value) noexcept
{
    const char* text = std::getenv(name);
    if (text == nullptr || text[0] == '\0') {
        return false;
    }
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 1 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

ThreadTeam& processThreadTeam()
{
    static ThreadTeam team;
    return team;
}

} // namespace

ParallelTeam& defaultParallelTeam()
{
    return processThreadTeam();
}

bool reserveParallelTeamWorkers(int n_workers) noexcept
{
    return processThreadTeam().reserveWorkers(n_workers);
}

int parallelTeamWorkerCount()
{
    return processThreadTeam().workerCount();
}

int geometryThreadCount(int assembly_threads) noexcept
{
    int threads = 0;
    if (positiveEnvInteger("SVMP_ASSEMBLY_THREADS", threads)) {
        return threads;
    }
    return std::max(1, assembly_threads);
}

bool geometryThreadsSelfCheckEnabled() noexcept
{
    const char* text = std::getenv("SVMP_GEOMETRY_THREADS_SELF_CHECK");
    return text != nullptr && text[0] != '\0' && text[0] != '0';
}

bool insideDeterministicParallelRegion() noexcept
{
    return t_inside_region;
}

void deterministicParallelFor(std::size_t n_items,
                              int threads,
                              const std::function<void(std::size_t, int)>& body,
                              std::size_t block_size,
                              std::size_t min_parallel_items,
                              ParallelTeam* team)
{
    block_size = std::max<std::size_t>(block_size, 1u);
    if (threads <= 1 || n_items < std::max<std::size_t>(min_parallel_items, 2u) ||
        t_inside_region) {
        for (std::size_t item = 0; item < n_items; ++item) {
            body(item, 0);
        }
        return;
    }
    const std::size_t n_blocks = (n_items + block_size - 1u) / block_size;
    const int participants = static_cast<int>(
        std::min<std::size_t>(static_cast<std::size_t>(threads), n_blocks));
    if (participants <= 1) {
        for (std::size_t item = 0; item < n_items; ++item) {
            body(item, 0);
        }
        return;
    }
    constexpr auto no_failure = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> failed_item(static_cast<std::size_t>(participants),
                                         no_failure);
    std::vector<std::exception_ptr> failure(
        static_cast<std::size_t>(participants));
    const auto task = [&](int participant) {
        // A participant's blocks are visited in increasing order, so after
        // its first failure none of its later items can fail earlier.
        for (std::size_t block = static_cast<std::size_t>(participant);
             block < n_blocks;
             block += static_cast<std::size_t>(participants)) {
            const std::size_t begin = block * block_size;
            const std::size_t end = std::min(n_items, begin + block_size);
            for (std::size_t item = begin; item < end; ++item) {
                try {
                    body(item, participant);
                } catch (...) {
                    failed_item[static_cast<std::size_t>(participant)] = item;
                    failure[static_cast<std::size_t>(participant)] =
                        std::current_exception();
                    return;
                }
            }
        }
    };
    (team != nullptr ? *team : defaultParallelTeam()).run(participants, task);
    std::size_t first = no_failure;
    std::exception_ptr first_failure;
    for (std::size_t participant = 0; participant < failed_item.size();
         ++participant) {
        if (failed_item[participant] < first) {
            first = failed_item[participant];
            first_failure = failure[participant];
        }
    }
    if (first_failure) {
        std::rethrow_exception(first_failure);
    }
}

} // namespace FE
} // namespace svmp
