/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/AggregationGuardDiagnostics.h"

#include <algorithm>
#include <mutex>
#include <sstream>

namespace svmp {
namespace FE {
namespace diagnostics {

namespace {

std::mutex& totalsMutex()
{
    static std::mutex mutex;
    return mutex;
}

AggregationGuardRootlessTotals& totals()
{
    static AggregationGuardRootlessTotals value;
    return value;
}

} // namespace

void recordAggregationGuardRootless(std::uint64_t root_path_guard_candidates,
                                    std::uint64_t proposal_guard_candidates)
{
    const auto refresh_total = root_path_guard_candidates + proposal_guard_candidates;
    if (refresh_total == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(totalsMutex());
    auto& t = totals();
    ++t.refreshes_with_cases;
    t.root_path_guard_candidates_total += root_path_guard_candidates;
    t.proposal_guard_candidates_total += proposal_guard_candidates;
    t.candidates_per_refresh_max = std::max(t.candidates_per_refresh_max, refresh_total);
}

void noteAggregationGuardRootlessFallbackEnabled()
{
    std::lock_guard<std::mutex> lock(totalsMutex());
    totals().fallback_enabled = true;
}

AggregationGuardRootlessTotals aggregationGuardRootlessTotals()
{
    std::lock_guard<std::mutex> lock(totalsMutex());
    return totals();
}

std::string aggregationGuardRootlessSummary()
{
    const auto t = aggregationGuardRootlessTotals();
    std::ostringstream oss;
    oss << "diagnostic=aggregation_guard_rootless_summary fallback_enabled="
        << (t.fallback_enabled ? 1 : 0) << " candidates_total="
        << (t.root_path_guard_candidates_total + t.proposal_guard_candidates_total)
        << " root_path_guard_candidates_total=" << t.root_path_guard_candidates_total
        << " proposal_guard_candidates_total=" << t.proposal_guard_candidates_total
        << " refreshes_with_cases=" << t.refreshes_with_cases
        << " candidates_per_refresh_max=" << t.candidates_per_refresh_max;
    return oss.str();
}

} // namespace diagnostics
} // namespace FE
} // namespace svmp
