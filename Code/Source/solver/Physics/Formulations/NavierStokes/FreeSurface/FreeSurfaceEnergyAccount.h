/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#pragma once

/**
 * @file
 * @brief Free-surface energy account: the physical endpoint channels of the
 *        accepted-step maintenance account and the residual-work channels.
 *
 * The level-set maintenance work ledger and the accepted-step maintenance
 * account are FE services
 * (FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h).
 */

#include "FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

struct LevelSetMaintenancePhysicalEndpointChannels {
  std::optional<double> surface_wall_energy_change{};
  std::optional<double> surface_transport_coupling_work{};
  std::optional<double> gravitational_energy_change{};
  std::optional<double>
      gravitational_transport_coupling_work{};
  std::optional<double> bulk_viscous_dissipation_rate{};
  std::optional<double> external_pressure_work{};
  std::optional<double> convection_work{};
  std::optional<double> pressure_continuity_work{};
  std::optional<double> nonconservative_body_force_work{};
  std::optional<double> weak_boundary_work{};
  std::optional<double> vms_pspg_work{};
  std::optional<double> cut_stabilization_work{};
  std::optional<double> ghost_penalty_work{};
};

/** One exact assembled residual/state pairing at the converged operator stage. */
struct FreeSurfaceResidualWorkPairing {
  std::string channel{};
  bool produced{false};
  std::string owner_component{};
  std::string operator_tag{};
  double residual_state_pairing{0.0};
};

/**
 * Signed step work from the currently connected Navier--Stokes residual
 * groups.  Positive values add modeled stored energy.
 */
struct FreeSurfaceResidualWorkChannels {
  std::optional<double> convection{};
  std::optional<double> pressure_continuity{};
  std::optional<double> nonconservative_body_force{};
  std::optional<double> weak_boundary{};
  std::optional<double> vms_pspg{};
  std::optional<double> cut_stabilization{};
  std::optional<double> ghost_penalty{};
};

/**
 * Validate one explicit owner/applicability row for every connected channel
 * and convert residual pairings to signed step work as -dt R_i(x)[x].
 */
[[nodiscard]] FreeSurfaceResidualWorkChannels
evaluateFreeSurfaceResidualWorkChannels(
    std::span<const FreeSurfaceResidualWorkPairing> pairings,
    double dt);

/**
 * Pair the post-Transport endpoint with the preceding accepted
 * post-maintenance stored state. Coupling work uses
 *   endpoint stored-energy change - dt * endpoint potential power.
 * Exterior-pressure power is converted to signed step work with dt.
 * Missing producers remain unavailable.
 */
[[nodiscard]] LevelSetMaintenancePhysicalEndpointChannels
evaluateLevelSetMaintenancePhysicalEndpointChannels(
    const svmp::FE::level_set::LevelSetMaintenanceAcceptedStepEnergyAccount&
        account,
    std::optional<double>
        preceding_gravitational_energy,
    std::optional<double>
        preceding_surface_wall_energy);

/** Log the physical-endpoint channels of an accepted-step account. */
void logLevelSetMaintenancePhysicalEndpointChannels(
    const svmp::FE::level_set::LevelSetMaintenanceAcceptedStepEnergyAccount&
        account,
    const LevelSetMaintenancePhysicalEndpointChannels& channels,
    std::uint64_t preceding_accepted_step,
    double preceding_accepted_time,
    std::uint64_t preceding_state_revision,
    std::uint64_t preceding_snapshot_set_revision,
    std::uint64_t preceding_mesh_topology_set_revision,
    std::uint64_t preceding_cut_topology_set_revision,
    std::optional<std::uint64_t>
        preceding_extension_map_revision,
    std::size_t preceding_functional_count,
    double preceding_surface_wall_energy,
    std::optional<double> preceding_gravitational_energy,
    std::optional<int> active_volume_owner_interface_marker);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
