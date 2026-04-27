/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_SYSTEMS_CONTACTOPERATORINVALIDATION_H
#define SVMP_FE_SYSTEMS_CONTACTOPERATORINVALIDATION_H

#include "Core/Types.h"

#if defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#include "Mesh/Search/ContactProximity.h"

#include <cstdint>
#include <string>

namespace svmp {
namespace FE {
namespace systems {

struct ContactOperatorRevisionSnapshot {
    bool valid = false;
    std::uint64_t contact_revision_key = 0;
    std::uint64_t candidate_generation_epoch = 0;
    std::uint64_t active_set_epoch = 0;
    std::uint64_t fe_space_revision = 0;
    std::uint64_t fe_dof_layout_revision = 0;
    std::uint64_t fe_constraint_layout_revision = 0;
    std::uint64_t fe_block_layout_revision = 0;
    std::uint64_t restart_layout_revision = 0;
    std::size_t pair_count = 0;
    std::size_t active_pair_count = 0;

    [[nodiscard]] static ContactOperatorRevisionSnapshot capture(
        const svmp::search::ContactProximityMap& contact_map,
        svmp::search::ContactExternalRevisions external = {});
};

struct ContactOperatorRefreshDecision {
    bool structural_rebuild = false;
    bool value_update = false;
    bool matrix_rebuild = false;
    bool matrix_free_rebuild = false;
    bool preconditioner_refresh = false;
    bool restart_metadata_update = false;
    std::string reason;
};

[[nodiscard]] ContactOperatorRefreshDecision classifyContactOperatorRefresh(
    const ContactOperatorRevisionSnapshot& cached,
    const ContactOperatorRevisionSnapshot& current) noexcept;

[[nodiscard]] bool contactOperatorStructuralRebuildRequired(
    const ContactOperatorRevisionSnapshot& cached,
    const ContactOperatorRevisionSnapshot& current) noexcept;

} // namespace systems
} // namespace FE
} // namespace svmp

#endif // defined(SVMP_FE_WITH_MESH) && SVMP_FE_WITH_MESH

#endif // SVMP_FE_SYSTEMS_CONTACTOPERATORINVALIDATION_H
