/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Systems/ContactOperatorInvalidation.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

namespace svmp {
namespace FE {
namespace systems {

ContactOperatorRevisionSnapshot ContactOperatorRevisionSnapshot::capture(
    const svmp::search::ContactProximityMap& contact_map,
    svmp::search::ContactExternalRevisions external) {
    ContactOperatorRevisionSnapshot snapshot;
    snapshot.valid = true;
    snapshot.contact_revision_key = contact_map.revision_key();
    snapshot.candidate_generation_epoch = contact_map.candidate_generation_epoch;
    snapshot.active_set_epoch = contact_map.active_set_epoch;
    snapshot.fe_space_revision = external.fe_space_revision;
    snapshot.fe_dof_layout_revision = external.fe_dof_layout_revision;
    snapshot.fe_constraint_layout_revision = external.fe_constraint_layout_revision;
    snapshot.fe_block_layout_revision = external.fe_block_layout_revision;
    snapshot.restart_layout_revision = external.restart_layout_revision;
    snapshot.pair_count = contact_map.pairs.size();
    snapshot.active_pair_count = contact_map.active_pair_count();
    return snapshot;
}

ContactOperatorRefreshDecision classifyContactOperatorRefresh(
    const ContactOperatorRevisionSnapshot& cached,
    const ContactOperatorRevisionSnapshot& current) noexcept {
    ContactOperatorRefreshDecision decision;
    if (!cached.valid || !current.valid) {
        decision.structural_rebuild = true;
        decision.matrix_rebuild = true;
        decision.matrix_free_rebuild = true;
        decision.preconditioner_refresh = true;
        decision.restart_metadata_update = true;
        decision.reason = "missing contact operator revision snapshot";
        return decision;
    }

    if (cached.fe_space_revision != current.fe_space_revision ||
        cached.fe_dof_layout_revision != current.fe_dof_layout_revision ||
        cached.fe_constraint_layout_revision != current.fe_constraint_layout_revision ||
        cached.fe_block_layout_revision != current.fe_block_layout_revision) {
        decision.structural_rebuild = true;
        decision.matrix_rebuild = true;
        decision.matrix_free_rebuild = true;
        decision.preconditioner_refresh = true;
        decision.restart_metadata_update = true;
        decision.reason = "FE contact operator layout changed";
        return decision;
    }

    if (cached.candidate_generation_epoch != current.candidate_generation_epoch ||
        cached.pair_count != current.pair_count) {
        decision.structural_rebuild = true;
        decision.matrix_rebuild = true;
        decision.matrix_free_rebuild = true;
        decision.preconditioner_refresh = true;
        decision.restart_metadata_update = true;
        decision.reason = "contact pair structure changed";
        return decision;
    }

    if (cached.active_set_epoch != current.active_set_epoch ||
        cached.active_pair_count != current.active_pair_count) {
        decision.value_update = true;
        decision.matrix_rebuild = true;
        decision.matrix_free_rebuild = true;
        decision.preconditioner_refresh = true;
        decision.restart_metadata_update = true;
        decision.reason = "contact active set changed";
        return decision;
    }

    if (cached.contact_revision_key != current.contact_revision_key) {
        decision.value_update = true;
        decision.matrix_rebuild = true;
        decision.matrix_free_rebuild = true;
        decision.preconditioner_refresh = true;
        decision.restart_metadata_update = true;
        decision.reason = "contact geometry or provenance changed";
        return decision;
    }

    if (cached.restart_layout_revision != current.restart_layout_revision) {
        decision.restart_metadata_update = true;
        decision.reason = "contact restart layout changed";
        return decision;
    }

    decision.reason = "contact operator state is reusable";
    return decision;
}

bool contactOperatorStructuralRebuildRequired(
    const ContactOperatorRevisionSnapshot& cached,
    const ContactOperatorRevisionSnapshot& current) noexcept {
    return classifyContactOperatorRefresh(cached, current).structural_rebuild;
}

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH
