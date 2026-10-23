/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyAccount.h"

#include "FE/Core/RankLog.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

LevelSetMaintenancePhysicalEndpointChannels
evaluateLevelSetMaintenancePhysicalEndpointChannels(
    const svmp::FE::level_set::LevelSetMaintenanceAcceptedStepEnergyAccount&
        account,
    std::optional<double> preceding_gravitational_energy,
    std::optional<double> preceding_surface_wall_energy)
{
  if (!account.post_transport.has_value() ||
      !std::isfinite(account.dt) || !(account.dt > 0.0) ||
      (preceding_gravitational_energy.has_value() &&
       !std::isfinite(*preceding_gravitational_energy)) ||
      (preceding_surface_wall_energy.has_value() &&
       !std::isfinite(*preceding_surface_wall_energy))) {
    throw std::invalid_argument(
        "Physical endpoint energy channels require a complete post-Transport endpoint, positive duration, and finite preceding energies.");
  }
  const auto& endpoint = *account.post_transport;
  LevelSetMaintenancePhysicalEndpointChannels channels;
  if (preceding_surface_wall_energy.has_value()) {
    channels.surface_wall_energy_change =
        endpoint.liquid_gas_surface_energy +
        endpoint.solid_liquid_wall_energy -
        *preceding_surface_wall_energy;
    if (endpoint.surface_wall_potential_power.has_value()) {
      channels.surface_transport_coupling_work =
          *channels.surface_wall_energy_change -
          account.dt *
              *endpoint.surface_wall_potential_power;
    }
  }
  if (preceding_gravitational_energy.has_value() &&
      endpoint.gravitational_energy.has_value()) {
    channels.gravitational_energy_change =
        *endpoint.gravitational_energy -
        *preceding_gravitational_energy;
    if (endpoint.gravitational_potential_power.has_value()) {
      channels.gravitational_transport_coupling_work =
          *channels.gravitational_energy_change -
          account.dt *
              *endpoint.gravitational_potential_power;
    }
  }
  channels.bulk_viscous_dissipation_rate =
      endpoint.bulk_viscous_dissipation_rate;
  if (endpoint.external_pressure_power.has_value()) {
    channels.external_pressure_work =
        account.dt * *endpoint.external_pressure_power;
  }
  const std::array<std::optional<double>, 6> values{{
      channels.surface_wall_energy_change,
      channels.surface_transport_coupling_work,
      channels.gravitational_energy_change,
      channels.gravitational_transport_coupling_work,
      channels.bulk_viscous_dissipation_rate,
      channels.external_pressure_work,
  }};
  if (std::any_of(
          values.begin(), values.end(), [](const auto& value) {
            return value.has_value() &&
                !std::isfinite(*value);
          }) ||
      (channels.bulk_viscous_dissipation_rate.has_value() &&
       *channels.bulk_viscous_dissipation_rate < 0.0)) {
    throw std::overflow_error(
        "Physical endpoint energy-channel evaluation is not finite or has negative dissipation.");
  }
  return channels;
}

FreeSurfaceResidualWorkChannels
evaluateFreeSurfaceResidualWorkChannels(
    std::span<const FreeSurfaceResidualWorkPairing> pairings,
    double dt)
{
  if (!std::isfinite(dt) || !(dt > 0.0)) {
    throw std::invalid_argument(
        "Free-surface residual work requires a finite positive duration.");
  }
  if (pairings.size() != 7u) {
    throw std::invalid_argument(
        "Free-surface residual work requires exactly seven connected channel rows.");
  }
  FreeSurfaceResidualWorkChannels channels;
  for (const auto& pairing : pairings) {
    if (pairing.owner_component.find_first_not_of(" \t\r\n") ==
        std::string::npos) {
      throw std::invalid_argument(
          "Free-surface residual work requires one named channel owner.");
    }
    if (!std::isfinite(pairing.residual_state_pairing)) {
      throw std::invalid_argument(
          "Free-surface residual/state pairing must be finite.");
    }
    if (pairing.produced && pairing.operator_tag.empty()) {
      throw std::invalid_argument(
          "A produced free-surface residual-work channel requires an operator tag.");
    }
    if (!pairing.produced &&
        (!pairing.operator_tag.empty() ||
         pairing.residual_state_pairing != 0.0)) {
      throw std::invalid_argument(
          "An inapplicable free-surface residual-work channel requires an empty operator and exact-zero pairing.");
    }
    const double signed_work =
        pairing.produced
            ? -dt * pairing.residual_state_pairing
            : 0.0;
    if (!std::isfinite(signed_work)) {
      throw std::overflow_error(
          "Free-surface residual step work is not finite.");
    }
    std::optional<double>* destination = nullptr;
    if (pairing.channel == "convection") {
      destination = &channels.convection;
    } else if (pairing.channel == "pressure_continuity") {
      destination = &channels.pressure_continuity;
    } else if (pairing.channel == "nonconservative_body_force") {
      destination = &channels.nonconservative_body_force;
    } else if (pairing.channel == "weak_boundary") {
      destination = &channels.weak_boundary;
    } else if (pairing.channel == "vms_pspg") {
      destination = &channels.vms_pspg;
    } else if (pairing.channel == "cut_stabilization") {
      destination = &channels.cut_stabilization;
    } else if (pairing.channel == "ghost_penalty") {
      destination = &channels.ghost_penalty;
    } else {
      throw std::invalid_argument(
          "Free-surface residual work contains an unknown connected channel '" +
          pairing.channel + "'.");
    }
    if (destination->has_value()) {
      throw std::invalid_argument(
          "Free-surface residual work contains a duplicate channel '" +
          pairing.channel + "'.");
    }
    *destination = signed_work;
  }
  if (!channels.convection.has_value() ||
      !channels.pressure_continuity.has_value() ||
      !channels.nonconservative_body_force.has_value() ||
      !channels.weak_boundary.has_value() ||
      !channels.vms_pspg.has_value() ||
      !channels.cut_stabilization.has_value() ||
      !channels.ghost_penalty.has_value()) {
    throw std::invalid_argument(
        "Free-surface residual work is missing a connected channel.");
  }
  return channels;
}

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
    std::optional<int> active_volume_owner_interface_marker)
{
  if (!account.post_transport.has_value()) {
    throw std::logic_error(
        "[svMultiPhysics::Application] Physical endpoint channel logging requires a post-Transport endpoint.");
  }
  const auto append_channel =
      [](std::ostringstream& log,
         std::string_view name,
         std::optional<double> value) {
        log << " " << name << "_available="
            << (value.has_value() ? "true" : "false")
            << " " << name << "=" << value.value_or(0.0);
      };
  std::ostringstream log;
  log << std::setprecision(17)
      << "[svMultiPhysics::Application] Level-set maintenance physical endpoint channels"
      << " diagnostic=level_set_maintenance_physical_endpoint_channels"
      << " step=" << account.step
      << " attempt=" << account.attempt
      << " time=" << account.time
      << " dt=" << account.dt
      << " preceding_accepted_step=" << preceding_accepted_step
      << " preceding_accepted_time=" << preceding_accepted_time
      << " preceding_state_revision=" << preceding_state_revision
      << " preceding_snapshot_set_revision="
      << preceding_snapshot_set_revision
      << " preceding_mesh_topology_set_revision="
      << preceding_mesh_topology_set_revision
      << " preceding_cut_topology_set_revision="
      << preceding_cut_topology_set_revision
      << " preceding_extension_map_revision_available="
      << (preceding_extension_map_revision.has_value()
              ? "true"
              : "false")
      << " preceding_extension_map_revision="
      << preceding_extension_map_revision.value_or(0u)
      << " preceding_functional_count="
      << preceding_functional_count
      << " preceding_surface_wall_energy="
      << preceding_surface_wall_energy
      << " post_transport_algebraic_state_revision="
      << account.post_transport->algebraic_state_revision
      << " post_transport_snapshot_set_revision="
      << account.post_transport->snapshot_set_revision
      << " post_transport_mesh_topology_set_revision="
      << account.post_transport->mesh_topology_set_revision
      << " post_transport_cut_topology_set_revision="
      << account.post_transport->cut_topology_set_revision
      << " post_transport_extension_map_revision="
      << account.post_transport->extension_map_revision.value_or(0u)
      << " active_volume_owner_available="
      << (active_volume_owner_interface_marker.has_value()
              ? "true"
              : "false")
      << " active_volume_owner_interface_marker="
      << active_volume_owner_interface_marker.value_or(-1);
  append_channel(
      log,
      "preceding_gravitational_energy",
      preceding_gravitational_energy);
  append_channel(
      log,
      "surface_wall_energy_change",
      channels.surface_wall_energy_change);
  append_channel(
      log,
      "surface_transport_coupling_work",
      channels.surface_transport_coupling_work);
  append_channel(
      log,
      "gravitational_energy_change",
      channels.gravitational_energy_change);
  append_channel(
      log,
      "gravitational_transport_coupling_work",
      channels.gravitational_transport_coupling_work);
  append_channel(
      log,
      "bulk_viscous_dissipation_rate",
      channels.bulk_viscous_dissipation_rate);
  append_channel(
      log,
      "external_pressure_work",
      channels.external_pressure_work);
  append_channel(
      log,
      "convection_work",
      channels.convection_work);
  append_channel(
      log,
      "pressure_continuity_work",
      channels.pressure_continuity_work);
  append_channel(
      log,
      "nonconservative_body_force_work",
      channels.nonconservative_body_force_work);
  append_channel(
      log,
      "weak_boundary_work",
      channels.weak_boundary_work);
  append_channel(
      log,
      "vms_pspg_work",
      channels.vms_pspg_work);
  append_channel(
      log,
      "cut_stabilization_work",
      channels.cut_stabilization_work);
  append_channel(
      log,
      "ghost_penalty_work",
      channels.ghost_penalty_work);
  log << " complete_energy_record_connected=false";
  svmp::FE::rank_log::oopCout() << log.str() << std::endl;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
