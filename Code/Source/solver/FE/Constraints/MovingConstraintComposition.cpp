/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "Constraints/MovingConstraintComposition.h"

#include <algorithm>
#include <utility>
#include <unordered_map>

namespace svmp {
namespace FE {
namespace constraints {
namespace {

bool compatibleDuplicate(const MovingConstraintEntry& a,
                         const MovingConstraintEntry& b) noexcept
{
    return a.kind == b.kind &&
           a.logical_region_id == b.logical_region_id &&
           a.conflict_policy == MovingConstraintConflictPolicy::AllowCompatibleDuplicate &&
           b.conflict_policy == MovingConstraintConflictPolicy::AllowCompatibleDuplicate;
}

} // namespace

MovingConstraintCompositionResult composeMovingConstraints(
    std::vector<MovingConstraintEntry> entries)
{
    entries.erase(std::remove_if(entries.begin(),
                                 entries.end(),
                                 [](const MovingConstraintEntry& entry) {
                                     return !entry.active;
                                 }),
                  entries.end());

    std::stable_sort(entries.begin(),
                     entries.end(),
                     [](const MovingConstraintEntry& a, const MovingConstraintEntry& b) {
                         if (a.priority != b.priority) {
                             return a.priority < b.priority;
                         }
                         return a.id < b.id;
                     });

    MovingConstraintCompositionResult result;
    std::unordered_map<GlobalIndex, std::size_t> owner_by_dof;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& entry = entries[i];
        for (const GlobalIndex dof : entry.constrained_dofs) {
            if (dof == INVALID_GLOBAL_INDEX) {
                continue;
            }
            const auto it = owner_by_dof.find(dof);
            if (it == owner_by_dof.end()) {
                owner_by_dof.emplace(dof, i);
                continue;
            }

            const auto& first = entries[it->second];
            if (compatibleDuplicate(first, entry)) {
                continue;
            }
            if (entry.conflict_policy == MovingConstraintConflictPolicy::HigherPriorityWins ||
                first.conflict_policy == MovingConstraintConflictPolicy::HigherPriorityWins) {
                continue;
            }

            MovingConstraintConflict conflict;
            conflict.constrained_dof = dof;
            conflict.first_id = first.id;
            conflict.second_id = entry.id;
            conflict.first_kind = first.kind;
            conflict.second_kind = entry.kind;
            result.conflicts.push_back(std::move(conflict));
        }
    }

    result.ok = result.conflicts.empty();
    result.ordered_entries = std::move(entries);
    return result;
}

bool movingConstraintRequiresRebuild(
    const MovingConstraintRevisionSnapshot& cached,
    const MovingConstraintRevisionSnapshot& current) noexcept
{
    return cached.geometry_revision != current.geometry_revision ||
           cached.topology_revision != current.topology_revision ||
           cached.ownership_revision != current.ownership_revision ||
           cached.numbering_revision != current.numbering_revision ||
           cached.label_revision != current.label_revision ||
           cached.fe_dof_layout_revision != current.fe_dof_layout_revision;
}

const char* movingConstraintKindName(MovingConstraintKind kind) noexcept
{
    switch (kind) {
        case MovingConstraintKind::CyclicPeriodic: return "cyclic_periodic";
        case MovingConstraintKind::GeometryDependentPeriodic: return "geometry_dependent_periodic";
        case MovingConstraintKind::SlidingInterface: return "sliding_interface";
        case MovingConstraintKind::TiedInterface: return "tied_interface";
        case MovingConstraintKind::RigidRegion: return "rigid_region";
        case MovingConstraintKind::AffineRegion: return "affine_region";
        case MovingConstraintKind::ContactCandidate: return "contact_candidate";
        case MovingConstraintKind::UserDefined: return "user_defined";
    }
    return "unknown";
}

} // namespace constraints
} // namespace FE
} // namespace svmp
