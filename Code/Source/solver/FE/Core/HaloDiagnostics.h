/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CORE_HALO_DIAGNOSTICS_H
#define SVMP_FE_CORE_HALO_DIAGNOSTICS_H

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace diagnostics {

/**
 * @brief Run totals of the diagnostics that show whether the ghost halo was
 *        deep enough for a partition-independent discretization.
 *
 * Every recorded value is communicator-global (identical on all ranks), so any
 * rank can print the summary.  Nonzero values mean the result depends on the
 * partition: increase <Ghost_layers>.
 */
struct HaloCorrectnessTotals {
    std::uint64_t aggregation_refreshes{0};
    std::uint64_t halo_limited_root_choices_total{0};
    std::uint64_t halo_limited_root_choices_max{0};
    std::uint64_t row_coupled_slaves_beyond_halo_total{0};
    std::uint64_t row_coupled_slaves_beyond_halo_max{0};
    std::uint64_t fill_refreshes{0};
    std::uint64_t fill_refreshes_with_rejections{0};
    std::uint64_t fill_rejected_columns_total{0};
    std::uint64_t fill_rejected_columns_max{0};
    std::uint64_t boundary_owner_completions{0};
    std::uint64_t boundary_owner_completion_added_total{0};
    std::uint64_t boundary_owner_completion_added_max{0};
};

/// One distributed small-cut aggregation rebuild (canonical counts).
void recordAggregationHalo(std::uint64_t halo_limited_root_choices,
                           std::uint64_t row_coupled_slaves_beyond_halo);

/// One distributed constraint-sparsity refresh (communicator sum of the
/// constraint-fill columns rejected because they lie beyond the ghost layers).
void recordConstraintFillRejections(std::uint64_t rejected_columns);

/// One strong boundary constraint collection (communicator sum of the owned
/// boundary DOFs that had to come from the ranks owning their faces' cells).
void recordBoundaryDofOwnerCompletion(std::uint64_t added_dofs);

[[nodiscard]] HaloCorrectnessTotals haloCorrectnessTotals();

/// "diagnostic=halo_correctness_summary ..." with every total and
/// `exact_halo=1` when the aggregation and fill totals are zero (the boundary
/// owner completion totals are informational: completed DOFs are exact).
[[nodiscard]] std::string haloCorrectnessSummary();

} // namespace diagnostics
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CORE_HALO_DIAGNOSTICS_H
