/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_CUTINTEGRATIONINVALIDATION_H
#define SVMP_FE_SYSTEMS_CUTINTEGRATIONINVALIDATION_H

/**
 * @file CutIntegrationInvalidation.h
 * @brief Invalidation and conditioning policy for unfitted/cut integration data.
 */

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
#include "Mesh/Search/CutCell.h"
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace systems {

struct CutIntegrationRevisionSnapshot {
    bool valid{false};
    std::uint64_t cut_revision_key{0};
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t label_revision{0};
    std::uint64_t active_configuration_epoch{0};
    std::uint64_t embedded_geometry_epoch{0};
    std::uint64_t embedded_constraint_epoch{0};
    std::uint64_t fe_space_revision{0};
    std::uint64_t fe_dof_layout_revision{0};
    std::uint64_t fe_constraint_layout_revision{0};
    std::uint64_t fe_block_layout_revision{0};
    std::uint64_t restart_layout_revision{0};
    std::size_t cut_cell_count{0};
    std::size_t cut_face_count{0};

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
    [[nodiscard]] static CutIntegrationRevisionSnapshot capture(
        const svmp::search::CutClassificationMap& map,
        std::uint64_t fe_space_revision = 0,
        std::uint64_t fe_dof_layout_revision = 0,
        std::uint64_t fe_constraint_layout_revision = 0,
        std::uint64_t fe_block_layout_revision = 0,
        std::uint64_t restart_layout_revision = 0);
#endif
};

struct CutIntegrationRefreshDecision {
    bool rebuild_cut_classification{false};
    bool rebuild_quadrature{false};
    bool rebuild_matrix{false};
    bool rebuild_matrix_free_data{false};
    bool refresh_preconditioner{false};
    bool refresh_restart_metadata{false};
    bool update_stabilization_hooks{false};
    std::string reason{};

    [[nodiscard]] bool any() const noexcept {
        return rebuild_cut_classification || rebuild_quadrature || rebuild_matrix ||
               rebuild_matrix_free_data || refresh_preconditioner ||
               refresh_restart_metadata || update_stabilization_hooks;
    }
};

struct CutConditioningDiagnostic {
    bool ok{true};
    std::size_t small_cut_cell_count{0};
    std::size_t degenerate_cut_count{0};
    std::vector<std::string> messages{};
};

[[nodiscard]] CutIntegrationRefreshDecision classifyCutIntegrationRefresh(
    const CutIntegrationRevisionSnapshot& cached,
    const CutIntegrationRevisionSnapshot& current) noexcept;

[[nodiscard]] CutConditioningDiagnostic diagnoseCutConditioning(
    const std::vector<Real>& volume_fractions,
    Real small_fraction_threshold = 1.0e-8,
    Real degenerate_threshold = 1.0e-14);

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_SYSTEMS_CUTINTEGRATIONINVALIDATION_H
