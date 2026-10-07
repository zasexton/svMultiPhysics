/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/AssemblyThreadPool.h"
#include "Assembly/ConcurrentCompute.h"

#include <algorithm>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace svmp {
namespace FE {
namespace assembly {

namespace {

thread_local bool tl_inside_parallel_region = false;
thread_local bool tl_concurrent_compute = false;

class InsideRegionGuard {
public:
    InsideRegionGuard() noexcept : previous_(tl_inside_parallel_region)
    {
        tl_inside_parallel_region = true;
    }
    ~InsideRegionGuard() { tl_inside_parallel_region = previous_; }
    InsideRegionGuard(const InsideRegionGuard&) = delete;
    InsideRegionGuard& operator=(const InsideRegionGuard&) = delete;

private:
    bool previous_;
};

} // namespace

bool concurrentComputeActive() noexcept
{
    return tl_concurrent_compute;
}

void requireSerial(const char* what)
{
    if (tl_concurrent_compute) {
        throw DeferredSerialWork(what != nullptr ? what : "lazy one-time work");
    }
}

ConcurrentComputeScope::ConcurrentComputeScope() noexcept
    : previous_(tl_concurrent_compute)
{
    tl_concurrent_compute = true;
}

ConcurrentComputeScope::~ConcurrentComputeScope()
{
    tl_concurrent_compute = previous_;
}

AssemblyThreadPool& AssemblyThreadPool::global()
{
    static AssemblyThreadPool pool;
    return pool;
}

AssemblyThreadPool::~AssemblyThreadPool()
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    start_cv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
}

bool AssemblyThreadPool::insideParallelRegion() noexcept
{
    return tl_inside_parallel_region;
}

int AssemblyThreadPool::workerCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(threads_.size());
}

bool AssemblyThreadPool::reserveWorkers(int n_workers) noexcept
{
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        ensureWorkersLocked(n_workers);
        return true;
    } catch (...) {
        return false;
    }
}

void AssemblyThreadPool::ensureWorkersLocked(int n_workers)
{
    while (static_cast<int>(threads_.size()) < n_workers) {
        // The worker starts from the generation current at creation, so a run()
        // that creates it and then publishes a new generation is not missed.
        const int index = static_cast<int>(threads_.size());
        const std::uint64_t start_generation = generation_;
        threads_.emplace_back([this, index, start_generation]() {
            workerLoop(index, start_generation);
        });
    }
}

void AssemblyThreadPool::workerLoop(int worker_index, std::uint64_t start_generation)
{
    tl_inside_parallel_region = true;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    std::uint64_t seen_generation = start_generation;
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        start_cv_.wait(lock, [&]() { return stop_ || generation_ != seen_generation; });
        if (stop_) {
            return;
        }
        seen_generation = generation_;
        const int participant = worker_index + 1;
        if (participant >= active_participants_) {
            continue;
        }
        const auto* task = task_;
        lock.unlock();
        std::exception_ptr error;
        try {
            (*task)(participant);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        if (error) {
            errors_[static_cast<std::size_t>(participant)] = std::move(error);
        }
        if (--remaining_ == 0) {
            done_cv_.notify_all();
        }
    }
}

void AssemblyThreadPool::run(int n_participants, const std::function<void(int)>& task)
{
    if (n_participants <= 1 || tl_inside_parallel_region) {
        for (int p = 0; p < std::max(1, n_participants); ++p) {
            task(p);
        }
        return;
    }

    std::lock_guard<std::mutex> run_lock(run_mutex_);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ensureWorkersLocked(n_participants - 1);
        task_ = &task;
        active_participants_ = n_participants;
        remaining_ = n_participants - 1;
        errors_.assign(static_cast<std::size_t>(n_participants), nullptr);
        ++generation_;
    }
    start_cv_.notify_all();

    std::exception_ptr caller_error;
    {
        InsideRegionGuard guard;
        try {
            task(0);
        } catch (...) {
            caller_error = std::current_exception();
        }
    }

    std::vector<std::exception_ptr> errors;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [&]() { return remaining_ == 0; });
        task_ = nullptr;
        active_participants_ = 0;
        errors.swap(errors_);
    }
    if (caller_error) {
        std::rethrow_exception(caller_error);
    }
    for (auto& e : errors) {
        if (e) {
            std::rethrow_exception(e);
        }
    }
}

} // namespace assembly
} // namespace FE
} // namespace svmp
