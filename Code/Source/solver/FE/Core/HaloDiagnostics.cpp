/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Core/HaloDiagnostics.h"

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

HaloCorrectnessTotals& totals()
{
    static HaloCorrectnessTotals value;
    return value;
}

} // namespace

void recordAggregationHalo(std::uint64_t halo_limited_root_choices,
                           std::uint64_t row_coupled_slaves_beyond_halo)
{
    std::lock_guard<std::mutex> lock(totalsMutex());
    auto& t = totals();
    ++t.aggregation_refreshes;
    t.halo_limited_root_choices_total += halo_limited_root_choices;
    t.halo_limited_root_choices_max =
        std::max(t.halo_limited_root_choices_max, halo_limited_root_choices);
    t.row_coupled_slaves_beyond_halo_total += row_coupled_slaves_beyond_halo;
    t.row_coupled_slaves_beyond_halo_max =
        std::max(t.row_coupled_slaves_beyond_halo_max, row_coupled_slaves_beyond_halo);
}

void recordConstraintFillRejections(std::uint64_t rejected_columns)
{
    std::lock_guard<std::mutex> lock(totalsMutex());
    auto& t = totals();
    ++t.fill_refreshes;
    if (rejected_columns > 0) {
        ++t.fill_refreshes_with_rejections;
    }
    t.fill_rejected_columns_total += rejected_columns;
    t.fill_rejected_columns_max = std::max(t.fill_rejected_columns_max, rejected_columns);
}

void recordBoundaryDofOwnerCompletion(std::uint64_t added_dofs)
{
    std::lock_guard<std::mutex> lock(totalsMutex());
    auto& t = totals();
    ++t.boundary_owner_completions;
    t.boundary_owner_completion_added_total += added_dofs;
    t.boundary_owner_completion_added_max =
        std::max(t.boundary_owner_completion_added_max, added_dofs);
}

HaloCorrectnessTotals haloCorrectnessTotals()
{
    std::lock_guard<std::mutex> lock(totalsMutex());
    return totals();
}

std::string haloCorrectnessSummary()
{
    const auto t = haloCorrectnessTotals();
    const bool exact = t.halo_limited_root_choices_total == 0 &&
                       t.row_coupled_slaves_beyond_halo_total == 0 &&
                       t.fill_rejected_columns_total == 0;
    std::ostringstream oss;
    oss << "diagnostic=halo_correctness_summary exact_halo=" << (exact ? 1 : 0)
        << " aggregation_refreshes=" << t.aggregation_refreshes
        << " canonical_halo_limited_root_choices_total=" << t.halo_limited_root_choices_total
        << " canonical_halo_limited_root_choices_max=" << t.halo_limited_root_choices_max
        << " canonical_row_coupled_slaves_beyond_halo_total=" << t.row_coupled_slaves_beyond_halo_total
        << " canonical_row_coupled_slaves_beyond_halo_max=" << t.row_coupled_slaves_beyond_halo_max
        << " constraint_fill_refreshes=" << t.fill_refreshes
        << " off_rank_constraint_fill_refreshes_with_rejections=" << t.fill_refreshes_with_rejections
        << " off_rank_constraint_fill_rejected_columns_total=" << t.fill_rejected_columns_total
        << " off_rank_constraint_fill_rejected_columns_max=" << t.fill_rejected_columns_max
        << " boundary_dof_owner_completions=" << t.boundary_owner_completions
        << " boundary_dof_owner_completion_added_total=" << t.boundary_owner_completion_added_total
        << " boundary_dof_owner_completion_added_max=" << t.boundary_owner_completion_added_max;
    return oss.str();
}

} // namespace diagnostics
} // namespace FE
} // namespace svmp
