/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/CutIntegrationInvalidation.h"

#include <algorithm>

namespace svmp {
namespace FE {
namespace systems {

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
CutIntegrationRevisionSnapshot CutIntegrationRevisionSnapshot::capture(
    const svmp::search::CutClassificationMap& map,
    std::uint64_t fe_space_revision,
    std::uint64_t fe_dof_layout_revision,
    std::uint64_t fe_constraint_layout_revision,
    std::uint64_t fe_block_layout_revision,
    std::uint64_t restart_layout_revision) {
    CutIntegrationRevisionSnapshot snapshot;
    snapshot.valid = true;
    snapshot.cut_revision_key = map.revision_key();
    snapshot.geometry_revision = map.revision.geometry_revision;
    snapshot.topology_revision = map.revision.topology_revision;
    snapshot.ownership_revision = map.revision.ownership_revision;
    snapshot.numbering_revision = map.revision.numbering_revision;
    snapshot.label_revision = map.revision.label_revision;
    snapshot.active_configuration_epoch = map.revision.active_configuration_epoch;
    snapshot.embedded_geometry_epoch = map.revision.embedded_geometry_epoch;
    snapshot.embedded_constraint_epoch = map.revision.embedded_constraint_epoch;
    snapshot.fe_space_revision = fe_space_revision;
    snapshot.fe_dof_layout_revision = fe_dof_layout_revision;
    snapshot.fe_constraint_layout_revision = fe_constraint_layout_revision;
    snapshot.fe_block_layout_revision = fe_block_layout_revision;
    snapshot.restart_layout_revision = restart_layout_revision;
    snapshot.cut_cell_count = static_cast<std::size_t>(std::count_if(
        map.cells.begin(), map.cells.end(), [](const auto& r) {
            return r.classification == svmp::search::CutClassification::Cut;
        }));
    snapshot.cut_face_count = static_cast<std::size_t>(std::count_if(
        map.faces.begin(), map.faces.end(), [](const auto& r) {
            return r.classification == svmp::search::CutClassification::Cut;
        }));
    return snapshot;
}
#endif

CutIntegrationRefreshDecision classifyCutIntegrationRefresh(
    const CutIntegrationRevisionSnapshot& cached,
    const CutIntegrationRevisionSnapshot& current) noexcept {
    CutIntegrationRefreshDecision decision;
    if (!cached.valid || !current.valid) {
        decision.rebuild_cut_classification = true;
        decision.rebuild_quadrature = true;
        decision.rebuild_matrix = true;
        decision.rebuild_matrix_free_data = true;
        decision.refresh_preconditioner = true;
        decision.refresh_restart_metadata = true;
        decision.update_stabilization_hooks = true;
        decision.reason = "missing cut integration revision snapshot";
        return decision;
    }

    const bool mesh_or_embedded_changed =
        cached.cut_revision_key != current.cut_revision_key ||
        cached.geometry_revision != current.geometry_revision ||
        cached.topology_revision != current.topology_revision ||
        cached.ownership_revision != current.ownership_revision ||
        cached.numbering_revision != current.numbering_revision ||
        cached.label_revision != current.label_revision ||
        cached.active_configuration_epoch != current.active_configuration_epoch ||
        cached.embedded_geometry_epoch != current.embedded_geometry_epoch ||
        cached.embedded_constraint_epoch != current.embedded_constraint_epoch ||
        cached.cut_cell_count != current.cut_cell_count ||
        cached.cut_face_count != current.cut_face_count;

    const bool fe_layout_changed =
        cached.fe_space_revision != current.fe_space_revision ||
        cached.fe_dof_layout_revision != current.fe_dof_layout_revision ||
        cached.fe_constraint_layout_revision != current.fe_constraint_layout_revision ||
        cached.fe_block_layout_revision != current.fe_block_layout_revision;

    if (mesh_or_embedded_changed) {
        decision.rebuild_cut_classification = true;
        decision.rebuild_quadrature = true;
        decision.rebuild_matrix = true;
        decision.rebuild_matrix_free_data = true;
        decision.refresh_preconditioner = true;
        decision.refresh_restart_metadata = true;
        decision.update_stabilization_hooks = true;
        decision.reason = "mesh, embedded geometry, or cut topology changed";
    } else if (fe_layout_changed) {
        decision.rebuild_matrix = true;
        decision.refresh_preconditioner = true;
        decision.refresh_restart_metadata = true;
        decision.reason = "FE cut integration layout changed";
    } else if (cached.restart_layout_revision != current.restart_layout_revision) {
        decision.refresh_restart_metadata = true;
        decision.reason = "restart layout changed";
    }

    return decision;
}

CutConditioningDiagnostic diagnoseCutConditioning(
    const std::vector<Real>& volume_fractions,
    Real small_fraction_threshold,
    Real degenerate_threshold) {
    CutConditioningDiagnostic diagnostic;
    for (const Real fraction : volume_fractions) {
        if (fraction <= degenerate_threshold) {
            ++diagnostic.degenerate_cut_count;
        } else if (fraction < small_fraction_threshold) {
            ++diagnostic.small_cut_cell_count;
        }
    }
    if (diagnostic.degenerate_cut_count > 0) {
        diagnostic.ok = false;
        diagnostic.messages.push_back("degenerate cut cells require aggregation or stabilization");
    }
    if (diagnostic.small_cut_cell_count > 0) {
        diagnostic.messages.push_back("small cut cells may require conditioning stabilization");
    }
    return diagnostic;
}

} // namespace systems
} // namespace FE
} // namespace svmp
