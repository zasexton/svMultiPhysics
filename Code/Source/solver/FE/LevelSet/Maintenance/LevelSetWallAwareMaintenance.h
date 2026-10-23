/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#pragma once

/**
 * @file LevelSetWallAwareMaintenance.h
 * @ingroup fe_level_set
 * @brief The accepted wall-contact cells of a maintained level-set field.
 *
 * Maintenance that must not move the contact line (the conservative-phase
 * reconciliation) protects the accepted contact cells of the field's walls.
 * A wall takes them either from the retained contact rules of its interface's
 * current geometry snapshot, or from the constraints captured at the
 * interface's accepted contact stage.  Which walls a field has, and where
 * each takes its cells from, is physics: the host supplies it as plain data
 * (LevelSetWallAwareMaintenanceInput) after validating its own contact
 * declarations and accepted stages.  The geometry (the snapshots, their rules
 * and the canonical constraints) is resolved here.
 *
 * resolveLevelSetWallContactConstraints holds the geometry part of the
 * Application driver's former wall-aware maintenance context;
 * makeAcceptedSnapshotWallConstraint and canonicalizeAcceptedWallConstraints
 * come from the Navier-Stokes FreeSurfaceContactStages.  The messages are
 * unchanged.
 */

#include "Core/Types.h"
#include "LevelSet/LevelSetWallContactConstraint.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace svmp {
namespace FE {

namespace interfaces {
struct FreeSurfaceGeometryRuleRecord;
} // namespace interfaces

namespace systems {
class CollectiveCommunicator;
class FESystem;
} // namespace systems

namespace level_set {

/// Where the accepted contact cells of a wall come from.
enum class LevelSetWallContactSource : std::uint8_t {
    /// The retained contact rules of the wall in the interface's current
    /// geometry snapshot (today: a Young wall).
    EndpointSnapshot,
    /// The constraints captured at the interface's accepted contact stage
    /// (today: a DynamicRenE wall; it supersedes a Young datum on the same
    /// wall).
    AcceptedStage,
};

struct LevelSetWallAwareWall {
    int boundary_marker{-1};
    LevelSetWallContactSource source{LevelSetWallContactSource::EndpointSnapshot};
    /// AcceptedStage walls only, rank-local: whether this rank's accepted
    /// contact state owns contact on the wall (owned quadrature points or a
    /// positive owned contact measure).
    bool accepted_stage_has_owned_contact{false};
};

struct LevelSetWallAwareInterface {
    int interface_marker{-1};
    /// Ascending, unique boundary markers.  Empty when the wall law supplies
    /// no contact cells; such an interface is skipped.
    std::vector<LevelSetWallAwareWall> walls{};
    /// With AcceptedStage walls: the snapshot revision key of the interface's
    /// accepted contact stage.  The captured constraints must carry this key.
    std::uint64_t accepted_stage_geometry_revision{0u};
};

/// The walls of one maintained level-set field, supplied by the host from its
/// contact declarations.  Every member is equal on every rank except
/// accepted_stage_has_owned_contact.
struct LevelSetWallAwareMaintenanceInput {
    /// The field's interfaces that have a wall law, in the host's order.
    /// Empty when the field has none.
    std::vector<LevelSetWallAwareInterface> interfaces{};
};

/**
 * @brief The accepted wall-contact cells of a maintained level-set field.
 *
 * First agrees on @p input across ranks.  AcceptedStage walls take their
 * cells from @p accepted_stage_constraints (those of the interface, on an
 * AcceptedStage wall, carrying the interface's
 * accepted_stage_geometry_revision; any other constraint of the interface
 * fails), and a wall has a captured constraint on some rank exactly where
 * the rank reports owned contact.  EndpointSnapshot walls take theirs from
 * the retained, locally owned contact rules of the interface's current
 * geometry snapshot in the cut integration context of @p system; the
 * snapshot checks run for every interface with a wall.
 *
 * Returns the rank-local constraints, one per parent cell, in canonical
 * order; empty when @p input has no interface.  Throws std::runtime_error
 * when @p input differs across ranks or the geometry is missing, stale or
 * inconsistent.  Collective over the active communicator of @p system
 * (@p system supplies only the cut integration context, the mesh access and
 * the communicator).
 */
[[nodiscard]] std::vector<LevelSetWallContactConstraint>
resolveLevelSetWallContactConstraints(
    const systems::FESystem& system,
    const LevelSetWallAwareMaintenanceInput& input,
    std::span<const LevelSetWallContactConstraint> accepted_stage_constraints);

/**
 * @brief The wall-contact constraint of a retained contact rule of the
 *        geometry snapshot with revision key @p snapshot_revision.
 *
 * Decision D4: the constraint names the accepted contact cell; no angle or
 * physical frame is read.  Throws std::runtime_error when the rule is not a
 * retained contact rule, has no parent cell, or carries another snapshot
 * revision.  Rank-local.
 */
[[nodiscard]] LevelSetWallContactConstraint makeAcceptedSnapshotWallConstraint(
    const interfaces::FreeSurfaceGeometryRuleRecord& record,
    int interface_marker,
    std::uint64_t snapshot_revision);

/**
 * @brief @p constraints sorted by parent cell, interface, wall and revision,
 *        with duplicates removed.
 *
 * Throws std::runtime_error (naming @p source) on every rank when a rank
 * holds two different constraints for one parent cell.  Collective over
 * @p comm.
 */
[[nodiscard]] std::vector<LevelSetWallContactConstraint>
canonicalizeAcceptedWallConstraints(
    std::vector<LevelSetWallContactConstraint> constraints,
    const systems::CollectiveCommunicator& comm,
    std::string_view source);

} // namespace level_set
} // namespace FE
} // namespace svmp
