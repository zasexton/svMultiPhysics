/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "LevelSet/Maintenance/LevelSetWallAwareMaintenance.h"

#include "Assembly/CutIntegrationContext.h"
#include "Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "Systems/FESystem.h"
#include "Systems/SystemCollectives.h"

#include <algorithm>
#include <cstddef>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace svmp {
namespace FE {
namespace level_set {

namespace {

// The moved bodies keep the names of the callers' communicator helpers; each
// forwards to the FE/Systems collective that those helpers call.
svmp::FE::systems::CollectiveCommunicator activeFESystemCommunicator(
    const svmp::FE::systems::FESystem& system)
{
#if FE_HAS_MPI
  return svmp::FE::systems::CollectiveCommunicator(
      system.activeMpiCommunicator());
#else
  (void)system;
  return {};
#endif
}

double globalMinDouble(
    double local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMinDouble(local, comm);
}

std::pair<std::uint64_t, std::uint64_t> globalMinMaxUint64(
    std::uint64_t local,
    const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveMinMaxUint64(local, comm);
}

std::size_t globalSumSize(
    std::size_t local, const svmp::FE::systems::CollectiveCommunicator& comm)
{
  return svmp::FE::systems::collectiveSumSize(local, comm);
}

[[nodiscard]] bool sameAcceptedWallConstraint(
    const svmp::FE::level_set::LevelSetWallContactConstraint& left,
    const svmp::FE::level_set::LevelSetWallContactConstraint& right) noexcept
{
  return left.interface_marker == right.interface_marker &&
         left.boundary_marker == right.boundary_marker &&
         left.parent_cell_global_id == right.parent_cell_global_id &&
         left.geometry_revision == right.geometry_revision;
}

/// FNV-1a signature of the rank-independent members of @p input.
std::uint64_t wallAwareInputSignature(
    const LevelSetWallAwareMaintenanceInput& input) noexcept
{
  std::uint64_t signature = 1469598103934665603ull;
  const auto mix = [&signature](std::uint64_t value) {
    signature ^= value;
    signature *= 1099511628211ull;
  };
  mix(static_cast<std::uint64_t>(input.interfaces.size()));
  for (const auto& field_interface : input.interfaces) {
    mix(static_cast<std::uint64_t>(
        static_cast<std::int64_t>(field_interface.interface_marker)));
    mix(field_interface.accepted_stage_geometry_revision);
    mix(static_cast<std::uint64_t>(field_interface.walls.size()));
    for (const auto& wall : field_interface.walls) {
      mix(static_cast<std::uint64_t>(
          static_cast<std::int64_t>(wall.boundary_marker)));
      mix(static_cast<std::uint64_t>(wall.source));
    }
  }
  return signature;
}

} // namespace

svmp::FE::level_set::LevelSetWallContactConstraint
makeAcceptedSnapshotWallConstraint(
    const svmp::FE::interfaces::FreeSurfaceGeometryRuleRecord& record,
    int interface_marker,
    std::uint64_t snapshot_revision)
{
  // Decision D4: the constraint names the accepted contact cell; no angle or
  // physical frame is read.
  const auto parent =
      record.reference_rule.provenance.parent_entity_global_id;
  if (record.role !=
          svmp::FE::interfaces::FreeSurfaceGeometryRuleRole::Contact ||
      record.retention !=
          svmp::FE::interfaces::FreeSurfaceGeometryRetention::Retained ||
      parent == svmp::FE::INVALID_GLOBAL_INDEX || snapshot_revision == 0u ||
      record.reference_rule.provenance.free_surface_snapshot_revision_key !=
          snapshot_revision ||
      record.physical_rule.free_surface_snapshot_revision_key !=
          snapshot_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Accepted contact geometry cannot "
        "provide a wall-contact constraint for interface marker " +
        std::to_string(interface_marker) + ", boundary marker " +
        std::to_string(record.physical_boundary_marker) +
        ": the retained rule has incomplete or stale provenance.");
  }
  return svmp::FE::level_set::LevelSetWallContactConstraint{
      .interface_marker = interface_marker,
      .boundary_marker = record.physical_boundary_marker,
      .parent_cell_global_id = parent,
      .geometry_revision = snapshot_revision,
  };
}

std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
canonicalizeAcceptedWallConstraints(
    std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
        constraints,
    const svmp::FE::systems::CollectiveCommunicator& comm,
    std::string_view source)
{
  const auto order = [](const auto& left, const auto& right) {
    return std::tie(left.parent_cell_global_id,
                    left.interface_marker,
                    left.boundary_marker,
                    left.geometry_revision) <
           std::tie(right.parent_cell_global_id,
                    right.interface_marker,
                    right.boundary_marker,
                    right.geometry_revision);
  };
  std::sort(constraints.begin(), constraints.end(), order);
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint> unique;
  unique.reserve(constraints.size());
  bool local_conflict = false;
  for (const auto& constraint : constraints) {
    if (!unique.empty() &&
        unique.back().parent_cell_global_id ==
            constraint.parent_cell_global_id) {
      if (sameAcceptedWallConstraint(unique.back(), constraint)) {
        continue;
      }
      local_conflict = true;
      continue;
    }
    unique.push_back(constraint);
  }
  if (globalMinDouble(local_conflict ? 0.0 : 1.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] " + std::string(source) +
        " assigns distinct accepted wall-contact constraints to one parent "
        "cell.");
  }
  return unique;
}

std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
resolveLevelSetWallContactConstraints(
    const svmp::FE::systems::FESystem& system,
    const LevelSetWallAwareMaintenanceInput& input,
    std::span<const svmp::FE::level_set::LevelSetWallContactConstraint>
        accepted_stage_constraints)
{
  const auto comm = activeFESystemCommunicator(system);
  const auto [minimum_input_signature, maximum_input_signature] =
      globalMinMaxUint64(wallAwareInputSignature(input), comm);
  if (minimum_input_signature != maximum_input_signature) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance input differs across the FE communicator.");
  }
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      local_constraints;
  if (input.interfaces.empty()) {
    return local_constraints;
  }

  const auto* context = system.cutIntegrationContext();
  if (globalMinDouble(context != nullptr ? 1.0 : 0.0, comm) != 1.0) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Wall-aware level-set maintenance requires an authoritative geometry snapshot on every rank.");
  }

  for (const auto& field_interface : input.interfaces) {
    if (field_interface.walls.empty()) {
      continue;
    }
    const auto find_wall = [&field_interface](int boundary_marker)
        -> const LevelSetWallAwareWall* {
      const auto found = std::find_if(
          field_interface.walls.begin(),
          field_interface.walls.end(),
          [boundary_marker](const auto& wall) {
            return wall.boundary_marker == boundary_marker;
          });
      return found != field_interface.walls.end() ? &*found : nullptr;
    };
    const bool has_accepted_stage_wall = std::any_of(
        field_interface.walls.begin(),
        field_interface.walls.end(),
        [](const auto& wall) {
          return wall.source == LevelSetWallContactSource::AcceptedStage;
        });

    if (has_accepted_stage_wall) {
      std::map<int, std::size_t> local_rule_count_by_wall;
      std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
          declaration_constraints;
      bool local_constraints_valid = true;
      for (const auto& constraint : accepted_stage_constraints) {
        if (constraint.interface_marker !=
            field_interface.interface_marker) {
          continue;
        }
        const auto* wall = find_wall(constraint.boundary_marker);
        const bool accepted_stage_wall =
            wall != nullptr &&
            wall->source == LevelSetWallContactSource::AcceptedStage;
        local_constraints_valid =
            accepted_stage_wall &&
            constraint.parent_cell_global_id !=
                svmp::FE::INVALID_GLOBAL_INDEX &&
            constraint.geometry_revision ==
                field_interface.accepted_stage_geometry_revision &&
            local_constraints_valid;
        if (!accepted_stage_wall ||
            constraint.parent_cell_global_id ==
                svmp::FE::INVALID_GLOBAL_INDEX ||
            constraint.geometry_revision !=
                field_interface.accepted_stage_geometry_revision) {
          continue;
        }
        declaration_constraints.push_back(constraint);
      }
      if (globalMinDouble(local_constraints_valid ? 1.0 : 0.0, comm) !=
          1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected a captured contact constraint that does not match its accepted stage.");
      }
      declaration_constraints = canonicalizeAcceptedWallConstraints(
          std::move(declaration_constraints),
          comm,
          "Accepted contact-stage wall-aware maintenance");
      for (const auto& constraint : declaration_constraints) {
        local_constraints.push_back(constraint);
        ++local_rule_count_by_wall[constraint.boundary_marker];
      }
      for (const auto& wall : field_interface.walls) {
        if (wall.source != LevelSetWallContactSource::AcceptedStage) {
          continue;
        }
        const auto global_rule_count = globalSumSize(
            local_rule_count_by_wall[wall.boundary_marker], comm);
        if (wall.accepted_stage_has_owned_contact !=
            (global_rule_count > 0u)) {
          throw std::runtime_error(
              "[svMultiPhysics::Application] Wall-aware level-set maintenance contact-rule coverage disagrees with the accepted dynamic wall state for boundary marker " +
              std::to_string(wall.boundary_marker) + ".");
        }
      }
    }

    const bool local_marker_available =
        context->hasFreeSurfaceGeometrySnapshotForMarker(
            field_interface.interface_marker);
    if (globalMinDouble(local_marker_available ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance is missing the authoritative snapshot for interface marker " +
          std::to_string(field_interface.interface_marker) + ".");
    }
    const auto snapshot_revision =
        context->freeSurfaceGeometrySnapshotRevisionForMarker(
            field_interface.interface_marker);
    const auto [minimum_revision, maximum_revision] =
        globalMinMaxUint64(snapshot_revision, comm);
    if (minimum_revision != maximum_revision) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance found different geometry revisions across ranks for interface marker " +
          std::to_string(field_interface.interface_marker) + ".");
    }
    const auto& snapshots = context->freeSurfaceGeometrySnapshots();
    const auto snapshot_it = std::find_if(
        snapshots.begin(), snapshots.end(), [&](const auto& candidate) {
          return candidate &&
                 candidate->revision().snapshot_revision_key ==
                     snapshot_revision;
        });
    if (globalMinDouble(snapshot_it != snapshots.end() ? 1.0 : 0.0, comm) !=
        1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance found incomplete snapshot storage for interface marker " +
          std::to_string(field_interface.interface_marker) + ".");
    }
    bool local_snapshot_current = true;
    try {
      context->assertAllFreeSurfaceGeometrySnapshotsCurrent(
          system.meshAccess());
    } catch (const std::exception&) {
      local_snapshot_current = false;
    }
    if (globalMinDouble(local_snapshot_current ? 1.0 : 0.0, comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected a stale geometry snapshot for interface marker " +
          std::to_string(field_interface.interface_marker) + ".");
    }
    const auto& snapshot = **snapshot_it;
    const bool local_snapshot_matches_declaration =
        snapshot.revision().complete() &&
        snapshot.revision().interface_marker ==
            field_interface.interface_marker &&
        snapshot.revision().snapshot_revision_key == snapshot_revision;
    if (globalMinDouble(local_snapshot_matches_declaration ? 1.0 : 0.0,
                        comm) != 1.0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Wall-aware level-set maintenance snapshot provenance does not match interface marker " +
          std::to_string(field_interface.interface_marker) + ".");
    }
    for (const auto& wall : field_interface.walls) {
      if (wall.source != LevelSetWallContactSource::EndpointSnapshot) {
        continue;
      }
      const int boundary_marker = wall.boundary_marker;
      std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
          wall_constraints;
      bool local_parent_ids_valid = true;
      bool local_rules_valid = true;
      std::string local_rule_diagnostic;
      for (const auto& record : snapshot.rules()) {
        if (record.role != svmp::FE::interfaces::
                               FreeSurfaceGeometryRuleRole::Contact ||
            record.retention != svmp::FE::interfaces::
                                    FreeSurfaceGeometryRetention::Retained ||
            record.physical_boundary_marker != boundary_marker ||
            !record.locally_owned) {
          continue;
        }
        const auto parent =
            record.reference_rule.provenance.parent_entity_global_id;
        local_parent_ids_valid =
            parent != svmp::FE::INVALID_GLOBAL_INDEX &&
            local_parent_ids_valid;
        if (parent == svmp::FE::INVALID_GLOBAL_INDEX) {
          continue;
        }
        try {
          wall_constraints.push_back(makeAcceptedSnapshotWallConstraint(
              record,
              field_interface.interface_marker,
              snapshot_revision));
        } catch (const std::exception& error) {
          local_rules_valid = false;
          local_rule_diagnostic = error.what();
        }
      }
      if (globalMinDouble(local_parent_ids_valid ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance found a contact rule without a partition-independent parent identity.");
      }
      if (globalMinDouble(local_rules_valid ? 1.0 : 0.0, comm) != 1.0) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Wall-aware level-set maintenance rejected a prescribed-angle contact rule with incomplete or stale provenance" +
            (local_rule_diagnostic.empty()
                 ? std::string{"."}
                 : std::string{": "} + local_rule_diagnostic));
      }
      wall_constraints = canonicalizeAcceptedWallConstraints(
          std::move(wall_constraints),
          comm,
          "Prescribed wall-aware level-set maintenance");
      local_constraints.insert(
          local_constraints.end(),
          wall_constraints.begin(),
          wall_constraints.end());
    }
  }
  return canonicalizeAcceptedWallConstraints(
      std::move(local_constraints),
      comm,
      "Wall-aware level-set maintenance");
}

} // namespace level_set
} // namespace FE
} // namespace svmp
