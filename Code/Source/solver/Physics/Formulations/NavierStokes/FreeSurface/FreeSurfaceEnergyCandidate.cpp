/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyCandidate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using svmp::FE::level_set::mixMaintenanceRevision;

namespace {

const char* freeSurfaceEnergyChannelApplicabilityName(
    FreeSurfaceEnergyChannelApplicability applicability) noexcept
{
  switch (applicability) {
    case FreeSurfaceEnergyChannelApplicability::Produced:
      return "produced";
    case FreeSurfaceEnergyChannelApplicability::NotApplicable:
      return "not_applicable";
    case FreeSurfaceEnergyChannelApplicability::Unspecified:
      return "unspecified";
  }
  return "invalid";
}

} // namespace

FreeSurfaceAcceptedEnergyGroup summarizeAcceptedFreeSurfaceEnergyGroup(
    std::span<const svmp::FE::systems::
                  FreeSurfaceDiscreteFunctionalHistoryRecord> records)
{
  if (records.empty()) {
    throw std::invalid_argument(
        "A free-surface energy endpoint requires one nonempty accepted functional group.");
  }
  FreeSurfaceAcceptedEnergyGroup group{
      .accepted_step = records.front().accepted_step,
      .accepted_time = static_cast<double>(
          records.front().accepted_time),
      .dt = static_cast<double>(records.front().dt),
      .pre_maintenance_endpoint_state_revision =
          records.front().pre_maintenance_endpoint_state_revision,
      .state_revision = records.front().state_revision,
      .extension_map_revision =
          records.front().extension_map_revision,
  };
  group.snapshot_set_revision = 1469598103934665603ull;
  group.mesh_topology_set_revision = 1469598103934665603ull;
  group.cut_topology_set_revision = 1469598103934665603ull;
  bool surface_power_complete = true;
  bool pressure_power_complete = true;
  bool contact_dissipation_complete = true;
  double surface_power = 0.0;
  double pressure_power = 0.0;
  double navier_slip_dissipation = 0.0;
  double line_friction_dissipation = 0.0;
  std::size_t active_volume_owner_count = 0u;
  for (const auto& record : records) {
    if (record.accepted_step != group.accepted_step ||
        static_cast<double>(record.accepted_time) !=
            group.accepted_time ||
        static_cast<double>(record.dt) != group.dt ||
        record.pre_maintenance_endpoint_state_revision !=
            group.pre_maintenance_endpoint_state_revision ||
        record.state_revision != group.state_revision ||
        record.extension_map_revision !=
            group.extension_map_revision ||
        record.geometry_revision.snapshot_revision_key == 0u ||
        record.cut_topology_revision == 0u ||
        !std::isfinite(record.state.liquid_gas_surface_energy) ||
        !std::isfinite(record.state.young_wall_energy)) {
      throw std::invalid_argument(
          "Accepted free-surface energy endpoint records have inconsistent metadata, provenance, or stored energy.");
    }
    group.liquid_gas_surface_energy += static_cast<double>(
        record.state.liquid_gas_surface_energy);
    group.solid_liquid_wall_energy += static_cast<double>(
        record.state.young_wall_energy);
    group.geometry_pruning_occurred =
        group.geometry_pruning_occurred ||
        record.geometry_pruning_occurred;
    mixMaintenanceRevision(
        group.snapshot_set_revision,
        static_cast<std::uint64_t>(
            record.declaration.interface_marker));
    mixMaintenanceRevision(
        group.snapshot_set_revision,
        record.geometry_revision.snapshot_revision_key);
    mixMaintenanceRevision(
        group.mesh_topology_set_revision,
        static_cast<std::uint64_t>(
            record.declaration.interface_marker));
    mixMaintenanceRevision(
        group.mesh_topology_set_revision,
        record.geometry_revision.mesh_topology_revision);
    mixMaintenanceRevision(
        group.cut_topology_set_revision,
        static_cast<std::uint64_t>(
            record.declaration.interface_marker));
    mixMaintenanceRevision(
        group.cut_topology_set_revision,
        record.cut_topology_revision);

    if (record.declaration.endpoint_functional_power_enabled) {
      if (!record.endpoint_functional_power.has_value()) {
        surface_power_complete = false;
      } else {
        surface_power += static_cast<double>(
            record.endpoint_functional_power
                ->liquid_gas_surface_energy_variation +
            record.endpoint_functional_power
                ->young_wall_energy_variation);
      }
    } else {
      surface_power_complete = false;
    }
    if (record.declaration.external_pressure_power_parameters
            .has_value()) {
      if (!record.external_pressure_power.has_value()) {
        pressure_power_complete = false;
      } else {
        pressure_power += static_cast<double>(
            record.external_pressure_power
                ->external_pressure_power);
      }
    } else {
      pressure_power_complete = false;
    }

    const bool contact_applicable =
        !record.declaration.parameters.dynamic_contact_coefficients
             .empty();
    group.contact_dissipation_applicable =
        group.contact_dissipation_applicable || contact_applicable;
    if (contact_applicable) {
      if (!record.contact_stage.has_value()) {
        contact_dissipation_complete = false;
      } else {
        navier_slip_dissipation += static_cast<double>(
            record.contact_stage->state.wall_slip_dissipation);
        line_friction_dissipation += static_cast<double>(
            record.contact_stage->state.line_friction_dissipation);
      }
    }

    if (record.declaration.active_volume_energy_parameters
            .has_value()) {
      ++active_volume_owner_count;
      if (!record.active_volume_energy.has_value()) {
        throw std::invalid_argument(
            "The accepted free-surface active-volume owner did not publish stored energy.");
      }
      group.kinetic_energy = static_cast<double>(
          record.active_volume_energy->kinetic_energy);
      group.gravitational_energy = static_cast<double>(
          record.active_volume_energy->gravitational_energy);
      group.gravitational_potential_power = static_cast<double>(
          record.active_volume_energy
              ->gravitational_potential_power);
      if (record.active_volume_dissipation.has_value()) {
        group.bulk_viscous_dissipation_rate =
            static_cast<double>(record.active_volume_dissipation
                                    ->bulk_viscous_dissipation_rate);
      }
      group.backward_euler_kinetic_work =
          record.backward_euler_kinetic_work;
    }
  }
  if (active_volume_owner_count != 1u) {
    throw std::invalid_argument(
        "A complete one-phase free-surface energy endpoint requires exactly one active-volume owner.");
  }
  const std::array<double, 9> finite_sums{{
      group.kinetic_energy,
      group.gravitational_energy,
      group.gravitational_potential_power,
      group.liquid_gas_surface_energy,
      group.solid_liquid_wall_energy,
      surface_power,
      pressure_power,
      navier_slip_dissipation,
      line_friction_dissipation,
  }};
  if (!std::all_of(
          finite_sums.begin(), finite_sums.end(), [](double value) {
            return std::isfinite(value);
          }) ||
      group.kinetic_energy < 0.0 ||
      group.liquid_gas_surface_energy < 0.0 ||
      navier_slip_dissipation < 0.0 ||
      line_friction_dissipation < 0.0 ||
      (group.bulk_viscous_dissipation_rate.has_value() &&
       (!std::isfinite(*group.bulk_viscous_dissipation_rate) ||
        *group.bulk_viscous_dissipation_rate < 0.0))) {
    throw std::invalid_argument(
        "The accepted free-surface energy endpoint contains a nonfinite or inadmissible value.");
  }
  group.snapshot_set_revision =
      group.snapshot_set_revision == 0u
          ? 1u
          : group.snapshot_set_revision;
  group.mesh_topology_set_revision =
      group.mesh_topology_set_revision == 0u
          ? 1u
          : group.mesh_topology_set_revision;
  group.cut_topology_set_revision =
      group.cut_topology_set_revision == 0u
          ? 1u
          : group.cut_topology_set_revision;
  if (surface_power_complete) {
    group.surface_wall_potential_power = surface_power;
  }
  if (pressure_power_complete) {
    group.external_pressure_power = pressure_power;
  }
  if (!group.contact_dissipation_applicable) {
    group.navier_slip_dissipation_rate = 0.0;
    group.line_friction_dissipation_rate = 0.0;
  } else if (contact_dissipation_complete) {
    group.navier_slip_dissipation_rate =
        navier_slip_dissipation;
    group.line_friction_dissipation_rate =
        line_friction_dissipation;
  }
  return group;
}

void appendFreeSurfaceEnergyChannelSource(
    std::ostream& stream,
    std::string_view channel,
    const FreeSurfaceEnergyChannelSource& source)
{
  stream
      << ' ' << channel << "_applicability="
      << freeSurfaceEnergyChannelApplicabilityName(
             source.applicability)
      << ' ' << channel << "_owner=" << std::quoted(source.owner);
}

FreeSurfaceCompleteEnergyBuildResult
buildFixedTopologyOnePhaseEnergyCandidate(
    std::uint64_t transaction_id,
    std::uint64_t attempt,
    bool preceding_history_complete,
    const FreeSurfaceAcceptedEnergyGroup& before,
    const FreeSurfaceAcceptedEnergyGroup& after,
    std::uint64_t residual_work_state_revision,
    const FreeSurfaceResidualWorkChannels& residual_work,
    std::span<const svmp::FE::systems::
                  FreeSurfaceResidualWorkDeclaration>
        residual_declarations,
    const std::optional<svmp::FE::systems::
              FreeSurfaceExternalBoundaryEnergyDeclaration>&
        boundary_energy,
    bool numerical_maintenance_absent,
    bool aggregation_present,
    bool aggregation_neutral,
    bool aggregation_rootless)
{
  FreeSurfaceCompleteEnergyBuildResult result;
  const auto require = [&](bool condition, std::string requirement) {
    if (!condition) {
      result.missing_requirements.push_back(
          std::move(requirement));
    }
  };
  const double expected_time_before = after.accepted_time - after.dt;
  const double time_scale = std::max(
      {1.0,
       std::abs(expected_time_before),
       std::abs(before.accepted_time),
       std::abs(after.accepted_time),
       std::abs(after.dt)});
  require(transaction_id != 0u, "transaction_id");
  require(attempt != 0u, "attempt_index");
  require(preceding_history_complete,
          "preceding_complete_energy_history");
  require(after.accepted_step == before.accepted_step + 1u,
          "consecutive_accepted_steps");
  require(std::isfinite(after.dt) && after.dt > 0.0,
          "positive_backward_euler_duration");
  require(
      std::abs(before.accepted_time - expected_time_before) <=
          64.0 * std::numeric_limits<double>::epsilon() *
              time_scale,
      "consecutive_endpoint_times");
  require(before.state_revision != 0u && after.state_revision != 0u,
          "nonzero_algebraic_revisions");
  require(residual_work_state_revision != 0u,
          "accepted_operator_residual_work_stage");
  require(
      after.pre_maintenance_endpoint_state_revision ==
          after.state_revision &&
          residual_work_state_revision == after.state_revision,
      "common_unmodified_operator_and_stored_energy_endpoint");
  require(
      before.mesh_topology_set_revision ==
              after.mesh_topology_set_revision &&
          before.cut_topology_set_revision ==
              after.cut_topology_set_revision,
      "fixed_mesh_and_cut_topology");
  require(
      !before.extension_map_revision.has_value() &&
          !after.extension_map_revision.has_value(),
      "extension_work_classification");
  require(
      !before.geometry_pruning_occurred &&
          !after.geometry_pruning_occurred,
      "pruning_work_classification");
  require(numerical_maintenance_absent,
          "zero_numerical_maintenance_substage_coverage");
  require(!aggregation_rootless,
          "rootless_aggregation_event_classification");
  require(!aggregation_present || aggregation_neutral,
          "accepted_aggregation_projection_neutrality");
  require(after.surface_wall_potential_power.has_value(),
          "surface_wall_endpoint_power");
  require(after.bulk_viscous_dissipation_rate.has_value(),
          "bulk_viscous_endpoint_dissipation");
  require(after.external_pressure_power.has_value(),
          "prescribed_exterior_pressure_power");
  require(after.backward_euler_kinetic_work.has_value(),
          "backward_euler_kinetic_identity");
  require(after.navier_slip_dissipation_rate.has_value(),
          "navier_slip_dissipation_classification");
  require(after.line_friction_dissipation_rate.has_value(),
          "line_friction_dissipation_classification");
  require(boundary_energy.has_value(),
          "external_boundary_energy_owner");
  if (boundary_energy.has_value()) {
    require(
        boundary_energy->imposed_traction ==
            svmp::FE::systems::
                FreeSurfaceExternalBoundaryEnergyApplicability::
                    NotApplicable,
        "imposed_traction_work_producer");
    require(
        boundary_energy->open_boundary_flux ==
            svmp::FE::systems::
                FreeSurfaceExternalBoundaryEnergyApplicability::
                    NotApplicable,
        "open_boundary_energy_flux_producer");
    require(boundary_energy->homogeneous_velocity_data,
            "prescribed_boundary_motion_work");
    require(boundary_energy->pressure_dirichlet_absent,
            "pressure_dirichlet_boundary_flux");
  }
  require(residual_declarations.size() == 7u,
          "seven_residual_work_declarations");
  const auto residual_value_present =
      [](const std::optional<double>& value) {
        return value.has_value() && std::isfinite(*value);
      };
  require(residual_value_present(residual_work.convection),
          "convection_work");
  require(
      residual_value_present(residual_work.pressure_continuity),
      "pressure_continuity_work");
  require(
      residual_value_present(
          residual_work.nonconservative_body_force),
      "nonconservative_body_force_work");
  require(residual_value_present(residual_work.weak_boundary),
          "weak_boundary_work");
  require(residual_value_present(residual_work.vms_pspg),
          "vms_pspg_work");
  require(
      residual_value_present(residual_work.cut_stabilization),
      "cut_stabilization_work");
  require(residual_value_present(residual_work.ghost_penalty),
          "ghost_penalty_work");

  const auto find_residual = [&](auto channel) {
    return std::find_if(
        residual_declarations.begin(),
        residual_declarations.end(),
        [&](const auto& declaration) {
          return declaration.channel == channel;
        });
  };
  for (const auto channel : {
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               Convection,
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               PressureContinuity,
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               NonconservativeBodyForce,
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               WeakBoundary,
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               VmsPspg,
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               CutStabilization,
           svmp::FE::systems::FreeSurfaceResidualWorkChannel::
               GhostPenalty}) {
    require(find_residual(channel) != residual_declarations.end(),
            std::string("residual_owner_") +
                svmp::FE::systems::
                    freeSurfaceResidualWorkChannelName(channel));
  }
  if (!result.missing_requirements.empty()) {
    std::sort(
        result.missing_requirements.begin(),
        result.missing_requirements.end());
    result.missing_requirements.erase(
        std::unique(
            result.missing_requirements.begin(),
            result.missing_requirements.end()),
        result.missing_requirements.end());
    return result;
  }

  const auto source = [](FreeSurfaceEnergyChannelApplicability applicability,
                         std::string owner) {
    return FreeSurfaceEnergyChannelSource{
        .applicability = applicability,
        .owner = std::move(owner),
    };
  };
  const auto produced = [&](std::string owner) {
    return source(
        FreeSurfaceEnergyChannelApplicability::Produced,
        std::move(owner));
  };
  const auto inapplicable = [&](std::string owner) {
    return source(
        FreeSurfaceEnergyChannelApplicability::NotApplicable,
        std::move(owner));
  };
  const auto residual_source = [&](auto channel) {
    const auto declaration = find_residual(channel);
    return source(
        declaration->applicability ==
                svmp::FE::systems::
                    FreeSurfaceResidualWorkApplicability::Produced
            ? FreeSurfaceEnergyChannelApplicability::Produced
            : FreeSurfaceEnergyChannelApplicability::NotApplicable,
        declaration->owner_component);
  };
  const auto& kinetic_work =
      *after.backward_euler_kinetic_work;
  const double surface_energy_change =
      after.liquid_gas_surface_energy +
      after.solid_liquid_wall_energy -
      before.liquid_gas_surface_energy -
      before.solid_liquid_wall_energy;
  const double gravitational_energy_change =
      after.gravitational_energy - before.gravitational_energy;
  const auto stored = [](const FreeSurfaceAcceptedEnergyGroup& group) {
    return FreeSurfaceStoredEnergy{
        .kinetic = group.kinetic_energy,
        .gravitational = group.gravitational_energy,
        .liquid_gas_surface =
            group.liquid_gas_surface_energy,
        .solid_liquid_wall =
            group.solid_liquid_wall_energy,
        .gas_applicability =
            FreeSurfaceGasEnergyApplicability::NotApplicable,
        .gas_or_compressibility = 0.0,
    };
  };
  const auto maintenance_source = [&](std::string owner) {
    return inapplicable(std::move(owner));
  };
  FreeSurfaceCompleteEnergyCandidate candidate{
      .metadata =
          FreeSurfaceEnergyAttemptMetadata{
              .transaction_id = transaction_id,
              .step = after.accepted_step,
              .attempt = attempt,
              .time_before = expected_time_before,
              .time_after = after.accepted_time,
              .dt = after.dt,
              .temporal_scheme =
                  FreeSurfaceEnergyTemporalScheme::BackwardEuler,
              .physical_evaluation_time = after.accepted_time,
              .physical_evaluation_stage_fraction = 1.0,
              .algebraic_state_revision_before =
                  before.state_revision,
              .physical_endpoint_algebraic_state_revision =
                  after.state_revision,
              .algebraic_state_revision_after =
                  after.state_revision,
              .snapshot_set_revision_before =
                  before.snapshot_set_revision,
              .physical_endpoint_snapshot_set_revision =
                  after.snapshot_set_revision,
              .snapshot_set_revision_after =
                  after.snapshot_set_revision,
              .mesh_topology_set_revision_before =
                  before.mesh_topology_set_revision,
              .physical_endpoint_mesh_topology_set_revision =
                  after.mesh_topology_set_revision,
              .mesh_topology_set_revision_after =
                  after.mesh_topology_set_revision,
              .cut_topology_set_revision_before =
                  before.cut_topology_set_revision,
              .physical_endpoint_cut_topology_set_revision =
                  after.cut_topology_set_revision,
              .cut_topology_set_revision_after =
                  after.cut_topology_set_revision,
              .extension_map_revision_before = std::nullopt,
              .physical_endpoint_extension_map_revision =
                  std::nullopt,
              .extension_map_revision_after = std::nullopt,
          },
      .before = stored(before),
      .physical_endpoint = stored(after),
      .after = stored(after),
      .dissipation =
          FreeSurfacePhysicalDissipationRate{
              .bulk_viscous =
                  *after.bulk_viscous_dissipation_rate,
              .navier_slip =
                  *after.navier_slip_dissipation_rate,
              .line_friction =
                  *after.line_friction_dissipation_rate,
          },
      .external =
          FreeSurfaceExternalWork{
              .pressure =
                  after.dt * *after.external_pressure_power,
              .body_force =
                  *residual_work.nonconservative_body_force,
              .imposed_traction = 0.0,
              .open_boundary_flux = 0.0,
          },
      .numerical =
          FreeSurfaceNumericalWork{
              .time_discretization =
                  -static_cast<double>(
                      kinetic_work.time_discretization_loss),
              .kinetic_domain_transport =
                  static_cast<double>(
                      kinetic_work
                          .kinetic_energy_before_on_endpoint_domain) -
                  before.kinetic_energy,
              .gravitational_transport_coupling =
                  gravitational_energy_change -
                  after.dt *
                      after.gravitational_potential_power,
              .convection = *residual_work.convection,
              .pressure_continuity =
                  *residual_work.pressure_continuity,
              .surface_transport_coupling =
                  surface_energy_change -
                  after.dt *
                      *after.surface_wall_potential_power,
              .weak_boundary = *residual_work.weak_boundary,
              .vms_pspg = *residual_work.vms_pspg,
              .cut_stabilization =
                  *residual_work.cut_stabilization,
              .ghost_penalty = *residual_work.ghost_penalty,
              .aggregation = 0.0,
              .extension = 0.0,
              .pruning = 0.0,
              .limiting = 0.0,
              .redistancing = 0.0,
              .local_reconciliation = 0.0,
              .global_correction = 0.0,
          },
      .sources =
          FreeSurfaceEnergyChannelSources{
              .stored =
                  FreeSurfaceStoredEnergySources{
                      .kinetic = produced(
                          "FreeSurfaceGeometrySnapshot.active_volume_energy"),
                      .gravitational = produced(
                          "FreeSurfaceGeometrySnapshot.active_volume_energy"),
                      .liquid_gas_surface = produced(
                          "FreeSurfaceGeometrySnapshot.discrete_functional"),
                      .solid_liquid_wall = produced(
                          "FreeSurfaceGeometrySnapshot.discrete_functional"),
                      .gas_or_compressibility = inapplicable(
                          "OnePhaseLiquidPrescribedExteriorPressure.gas_energy"),
                  },
              .dissipation =
                  FreeSurfacePhysicalDissipationSources{
                      .bulk_viscous = produced(
                          "FreeSurfaceGeometrySnapshot.active_volume_dissipation"),
                      .navier_slip =
                          after.contact_dissipation_applicable
                              ? produced(
                                    "FreeSurfaceGeometrySnapshot.contact_stage_navier_slip")
                              : inapplicable(
                                    "FreeSurfaceGeometrySnapshot.contact_stage_navier_slip"),
                      .line_friction =
                          after.contact_dissipation_applicable
                              ? produced(
                                    "FreeSurfaceGeometrySnapshot.contact_stage_line_friction")
                              : inapplicable(
                                    "FreeSurfaceGeometrySnapshot.contact_stage_line_friction"),
                  },
              .external =
                  FreeSurfaceExternalWorkSources{
                      .pressure = produced(
                          "FreeSurfaceGeometrySnapshot.external_pressure_power"),
                      .body_force = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::
                                  NonconservativeBodyForce),
                      .imposed_traction = inapplicable(
                          boundary_energy->owner_component +
                          ".imposed_traction"),
                      .open_boundary_flux = inapplicable(
                          boundary_energy->owner_component +
                          ".open_boundary_flux"),
                  },
              .numerical =
                  FreeSurfaceNumericalWorkSources{
                      .time_discretization = produced(
                          "FreeSurfaceGeometrySnapshot.backward_euler_kinetic_work"),
                      .kinetic_domain_transport = produced(
                          "FreeSurfaceGeometrySnapshot.backward_euler_kinetic_work"),
                      .gravitational_transport_coupling = produced(
                          "FreeSurfaceGeometrySnapshot.active_volume_energy"),
                      .convection = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::Convection),
                      .pressure_continuity = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::
                                  PressureContinuity),
                      .surface_transport_coupling = produced(
                          "FreeSurfaceGeometrySnapshot.endpoint_functional_power"),
                      .weak_boundary = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::
                                  WeakBoundary),
                      .vms_pspg = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::VmsPspg),
                      .cut_stabilization = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::
                                  CutStabilization),
                      .ghost_penalty = residual_source(
                          svmp::FE::systems::
                              FreeSurfaceResidualWorkChannel::
                                  GhostPenalty),
                      .aggregation = aggregation_present
                          ? produced(
                                "SmallCutAggregationConstraint.accepted_projection_neutrality")
                          : inapplicable(
                                "SmallCutAggregationConstraint.accepted_projection"),
                      .extension = inapplicable(
                          "LevelSetVelocityExtensionMap.direct_energy_work"),
                      .pruning = inapplicable(
                          "FreeSurfaceGeometrySnapshot.pruning"),
                      .limiting = maintenance_source(
                          "LevelSetMaintenanceWorkLedger.limiting"),
                      .redistancing = maintenance_source(
                          "LevelSetMaintenanceWorkLedger.redistancing"),
                      .local_reconciliation = maintenance_source(
                          "LevelSetMaintenanceWorkLedger.local_reconciliation"),
                      .global_correction = maintenance_source(
                          "LevelSetMaintenanceWorkLedger.global_correction"),
                  },
          },
  };
  result.candidate = std::move(candidate);
  return result;
}

FreeSurfaceEnergyAttemptMetadata
buildUnstagedFreeSurfaceEnergyAttemptMetadata(
    std::uint64_t transaction_id,
    std::uint64_t prospective_step,
    std::uint64_t attempt,
    double time_before,
    double dt,
    const FreeSurfaceAcceptedEnergyGroup& before)
{
  const double time_scale = std::max(
      {1.0, std::abs(time_before), std::abs(before.accepted_time)});
  const double time_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * time_scale;
  if (transaction_id == 0u || prospective_step == 0u ||
      attempt == 0u || !std::isfinite(time_before) ||
      !std::isfinite(dt) || dt <= 0.0 ||
      before.accepted_step ==
          std::numeric_limits<std::uint64_t>::max() ||
      prospective_step != before.accepted_step + 1u ||
      !std::isfinite(before.accepted_time) ||
      std::abs(before.accepted_time - time_before) >
          time_tolerance ||
      before.state_revision == 0u ||
      before.snapshot_set_revision == 0u ||
      before.mesh_topology_set_revision == 0u ||
      before.cut_topology_set_revision == 0u ||
      (before.extension_map_revision.has_value() &&
       *before.extension_map_revision == 0u)) {
    throw std::invalid_argument(
        "An unstaged free-surface energy attempt requires one complete matching accepted starting endpoint.");
  }
  const double time_after = time_before + dt;
  if (!std::isfinite(time_after) || time_after <= time_before) {
    throw std::invalid_argument(
        "An unstaged free-surface energy attempt requires one finite positive time interval.");
  }
  return FreeSurfaceEnergyAttemptMetadata{
      .transaction_id = transaction_id,
      .step = prospective_step,
      .attempt = attempt,
      .time_before = time_before,
      .time_after = time_after,
      .dt = dt,
      .temporal_scheme =
          FreeSurfaceEnergyTemporalScheme::BackwardEuler,
      .algebraic_state_revision_before = before.state_revision,
      .snapshot_set_revision_before = before.snapshot_set_revision,
      .mesh_topology_set_revision_before =
          before.mesh_topology_set_revision,
      .cut_topology_set_revision_before =
          before.cut_topology_set_revision,
      .extension_map_revision_before =
          before.extension_map_revision,
  };
}

FreeSurfaceEnergyRejectionReason
freeSurfaceEnergyRejectionReasonForStepReject(
    svmp::FE::timestepping::StepRejectReason reason)
{
  using StepRejectReason =
      svmp::FE::timestepping::StepRejectReason;
  switch (reason) {
    case StepRejectReason::NonlinearSolveFailed:
      return FreeSurfaceEnergyRejectionReason::
          NonlinearSolveFailure;
    case StepRejectReason::ErrorTooLarge:
      return FreeSurfaceEnergyRejectionReason::
          StepControllerRejection;
    case StepRejectReason::CutTopologyChanged:
      // The restored callback state has no rejected physical endpoint
      // revision tuple, so this is a preaccept rejection rather than a
      // claimed topology-jump energy record.
      return FreeSurfaceEnergyRejectionReason::
          PreacceptRejection;
  }
  throw std::invalid_argument(
      "Free-surface energy rejection received an unknown time-loop reason.");
}

const char* freeSurfaceEnergyRejectionReasonName(
    FreeSurfaceEnergyRejectionReason reason) noexcept
{
  switch (reason) {
    case FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure:
      return "nonlinear_solve_failure";
    case FreeSurfaceEnergyRejectionReason::StepControllerRejection:
      return "step_controller_rejection";
    case FreeSurfaceEnergyRejectionReason::PreacceptRejection:
      return "preaccept_rejection";
    case FreeSurfaceEnergyRejectionReason::TopologyChange:
      return "topology_change";
    case FreeSurfaceEnergyRejectionReason::MaintenanceRollback:
      return "maintenance_rollback";
    case FreeSurfaceEnergyRejectionReason::PublicationFailure:
      return "publication_failure";
    case FreeSurfaceEnergyRejectionReason::None:
      return "none";
  }
  return "invalid";
}

FreeSurfaceAggregationEnergyClassification
classifyAcceptedAggregationEnergy(
    std::span<const std::shared_ptr<const svmp::FE::constraints::
                        SmallCutAggregationProlongationReport>> reports,
    std::span<const svmp::FE::constraints::
                  SmallCutAggregationRefreshReport> refresh_reports,
    std::span<const svmp::FE::Real> accepted_solution)
{
  FreeSurfaceAggregationEnergyClassification classification;
  classification.rootless = std::any_of(
      refresh_reports.begin(),
      refresh_reports.end(),
      [](const auto& report) {
        return report.canonical_rootless_active_feature_count != 0u ||
            report.canonical_owned_pinned_dofs != 0u;
      });
  for (const auto& report : reports) {
    if (!report) {
      throw std::invalid_argument(
          "Accepted aggregation energy classification received a null prolongation report.");
    }
    for (const auto& row : report->rows) {
      if (row.provisional_kind ==
          svmp::FE::constraints::
              SmallCutAggregationProvisionalRowKind::
                  RootlessHomogeneousPin) {
        classification.rootless = true;
      }
      if (row.provisional_kind !=
          svmp::FE::constraints::
              SmallCutAggregationProvisionalRowKind::
                  RootedExtension) {
        continue;
      }
      classification.present = true;
      ++classification.rooted_row_count;
      if (row.slave_dof >= accepted_solution.size()) {
        throw std::out_of_range(
            "Accepted aggregation energy classification has a slave outside the FE-ordered state.");
      }
      double expected = static_cast<double>(
          row.final_inhomogeneity);
      for (const auto& entry : row.final_entries) {
        if (entry.master_dof >= accepted_solution.size()) {
          throw std::out_of_range(
              "Accepted aggregation energy classification has a master outside the FE-ordered state.");
        }
        expected += entry.weight * static_cast<double>(
            accepted_solution[entry.master_dof]);
      }
      const double actual = static_cast<double>(
          accepted_solution[row.slave_dof]);
      const double residual = actual - expected;
      const double scale =
          std::max({1.0, std::abs(actual), std::abs(expected)});
      if (!std::isfinite(residual)) {
        throw std::overflow_error(
            "Accepted aggregation constraint residual is not finite.");
      }
      classification.maximum_constraint_residual = std::max(
          classification.maximum_constraint_residual,
          std::abs(residual));
      classification.neutral =
          classification.neutral &&
          std::abs(residual) <=
              64.0 * std::numeric_limits<double>::epsilon() *
                  scale;
    }
  }
  return classification;
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
