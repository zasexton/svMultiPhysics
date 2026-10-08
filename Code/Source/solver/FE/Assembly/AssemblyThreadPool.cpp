/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Assembly/AssemblyThreadPool.h"
#include "Assembly/ConcurrentCompute.h"

#include <cstdlib>


namespace svmp {
namespace FE {
namespace assembly {

namespace {

thread_local bool tl_concurrent_compute = false;

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

void AssemblyThreadPool::run(int n_participants, const std::function<void(int)>& task)
{
    defaultParallelTeam().run(n_participants, task);
}

bool AssemblyThreadPool::insideParallelRegion() noexcept
{
    return insideDeterministicParallelRegion();
}

int AssemblyThreadPool::workerCount() const
{
    return parallelTeamWorkerCount();
}

bool AssemblyThreadPool::reserveWorkers(int n_workers) noexcept
{
    return reserveParallelTeamWorkers(n_workers);
}

std::chrono::nanoseconds AssemblyThreadPool::spinBudget() noexcept
{
    static const std::chrono::nanoseconds budget = []() {
        long long us = 0;
        if (const char* value = std::getenv("SVMP_ASSEMBLY_SPIN_US")) {
            char* end = nullptr;
            const long long parsed = std::strtoll(value, &end, 10);
            if (end != value && parsed >= 0) {
                us = parsed;
            }
        }
        return std::chrono::nanoseconds(us * 1000);
    }();
    return budget;
}

} // namespace assembly
} // namespace FE
} // namespace svmp
