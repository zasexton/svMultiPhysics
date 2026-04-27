/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#ifndef SVMP_FE_CONSTRAINTS_MOVINGCONSTRAINTCOMPOSITION_H
#define SVMP_FE_CONSTRAINTS_MOVINGCONSTRAINTCOMPOSITION_H

/**
 * @file MovingConstraintComposition.h
 * @brief Deterministic composition rules for geometry-dependent moving constraints.
 */

#include "Core/Types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace svmp {
namespace FE {
namespace constraints {

enum class MovingConstraintKind : std::uint8_t {
    CyclicPeriodic,
    GeometryDependentPeriodic,
    SlidingInterface,
    TiedInterface,
    RigidRegion,
    AffineRegion,
    ContactCandidate,
    UserDefined
};

enum class MovingConstraintTimeLevel : std::uint8_t {
    TrialIterate,
    AcceptedNonlinearState,
    AcceptedTimeStep,
    AcceptedRemeshOrRezoneState
};

enum class MovingConstraintConflictPolicy : std::uint8_t {
    Error,
    HigherPriorityWins,
    AllowCompatibleDuplicate
};

struct MovingConstraintRevisionSnapshot {
    std::uint64_t geometry_revision{0};
    std::uint64_t topology_revision{0};
    std::uint64_t ownership_revision{0};
    std::uint64_t numbering_revision{0};
    std::uint64_t label_revision{0};
    std::uint64_t fe_dof_layout_revision{0};
};

struct MovingConstraintEntry {
    std::string id{};
    MovingConstraintKind kind{MovingConstraintKind::UserDefined};
    MovingConstraintTimeLevel time_level{MovingConstraintTimeLevel::TrialIterate};
    MovingConstraintConflictPolicy conflict_policy{MovingConstraintConflictPolicy::Error};
    std::uint32_t priority{100u};
    std::vector<GlobalIndex> constrained_dofs{};
    std::vector<GlobalIndex> master_dofs{};
    std::string logical_region_id{};
    MovingConstraintRevisionSnapshot revisions{};
    bool geometry_dependent{true};
    bool active{true};
};

struct MovingConstraintConflict {
    GlobalIndex constrained_dof{INVALID_GLOBAL_INDEX};
    std::string first_id{};
    std::string second_id{};
    MovingConstraintKind first_kind{MovingConstraintKind::UserDefined};
    MovingConstraintKind second_kind{MovingConstraintKind::UserDefined};
};

struct MovingConstraintCompositionResult {
    std::vector<MovingConstraintEntry> ordered_entries{};
    std::vector<MovingConstraintConflict> conflicts{};
    bool ok{true};
};

[[nodiscard]] MovingConstraintCompositionResult composeMovingConstraints(
    std::vector<MovingConstraintEntry> entries);

[[nodiscard]] bool movingConstraintRequiresRebuild(
    const MovingConstraintRevisionSnapshot& cached,
    const MovingConstraintRevisionSnapshot& current) noexcept;

[[nodiscard]] const char* movingConstraintKindName(MovingConstraintKind kind) noexcept;

} // namespace constraints
} // namespace FE
} // namespace svmp

#endif // SVMP_FE_CONSTRAINTS_MOVINGCONSTRAINTCOMPOSITION_H
