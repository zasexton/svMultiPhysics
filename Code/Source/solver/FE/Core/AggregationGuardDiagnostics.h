/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_AGGREGATION_GUARD_DIAGNOSTICS_H
#define SVMP_FE_CORE_AGGREGATION_GUARD_DIAGNOSTICS_H

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace diagnostics {

/**
 * @brief Run totals of small-cut aggregation candidates that had no root
 *        inside the aggregation guards and received the rootless-island
 *        policy instead of an extension (opt-in fallback,
 *        Small_cut_aggregation_rootless_fallback; without it such a
 *        candidate stops the run).
 *
 * A candidate whose cut feature holds full cells only beyond the root-path
 * guard (`root_path_guard`), or whose every root proposal fails the
 * extrapolation or coefficient guards (`proposal_guard`), is treated like a
 * candidate of a feature without full cells.  Results that rely on this are
 * flagged by a nonzero total.  The counts are canonical (identical on every
 * rank), so any rank can print the summary.
 */
struct AggregationGuardRootlessTotals {
    bool fallback_enabled{false};
    std::uint64_t refreshes_with_cases{0};
    std::uint64_t root_path_guard_candidates_total{0};
    std::uint64_t proposal_guard_candidates_total{0};
    std::uint64_t candidates_per_refresh_max{0};
};

/// One aggregation refresh: the candidates it resolved by the rootless-island
/// policy because no root lay inside the guards (nothing is recorded when both
/// counts are zero).
void recordAggregationGuardRootless(std::uint64_t root_path_guard_candidates,
                                    std::uint64_t proposal_guard_candidates);

/// An aggregation refresh ran with the opt-in fallback enabled.
void noteAggregationGuardRootlessFallbackEnabled();

[[nodiscard]] AggregationGuardRootlessTotals aggregationGuardRootlessTotals();

/// "diagnostic=aggregation_guard_rootless_summary ..." with every total.
[[nodiscard]] std::string aggregationGuardRootlessSummary();

} // namespace diagnostics
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_AGGREGATION_GUARD_DIAGNOSTICS_H
