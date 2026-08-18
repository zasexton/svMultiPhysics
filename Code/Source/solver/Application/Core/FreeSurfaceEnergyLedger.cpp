/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Application/Core/FreeSurfaceEnergyLedger.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace application::core {
namespace {

static_assert(
    std::is_nothrow_move_constructible_v<FreeSurfaceEnergyAttempt>);
static_assert(
    std::is_nothrow_copy_constructible_v<
        FreeSurfaceEnergyAttemptMetadata>);
static_assert(
    std::is_nothrow_copy_assignable_v<
        FreeSurfaceEnergyAttemptMetadata>);

bool finite(double value) noexcept
{
  return std::isfinite(value);
}

void requireFiniteChannel(
    double value,
    std::string_view channel)
{
  if (!finite(value)) {
    throw std::invalid_argument(
        "Free-surface energy channel '" + std::string(channel) +
        "' must be finite.");
  }
}

void requireNonnegativeChannel(
    double value,
    std::string_view channel)
{
  requireFiniteChannel(value, channel);
  if (value < 0.0) {
    throw std::invalid_argument(
        "Free-surface energy channel '" + std::string(channel) +
        "' must be nonnegative.");
  }
}

void validateStoredEnergy(const FreeSurfaceStoredEnergy& energy)
{
  requireNonnegativeChannel(energy.kinetic, "stored.kinetic");
  requireFiniteChannel(
      energy.gravitational, "stored.gravitational");
  requireNonnegativeChannel(
      energy.liquid_gas_surface, "stored.liquid_gas_surface");
  requireFiniteChannel(
      energy.solid_liquid_wall, "stored.solid_liquid_wall");
  requireFiniteChannel(
      energy.gas_or_compressibility,
      "stored.gas_or_compressibility");
  const bool gas_active =
      energy.gas_applicability ==
      FreeSurfaceGasEnergyApplicability::Active;
  const bool gas_inapplicable =
      energy.gas_applicability ==
      FreeSurfaceGasEnergyApplicability::NotApplicable;
  if (!gas_active && !gas_inapplicable) {
    throw std::invalid_argument(
        "Free-surface gas/compressibility energy applicability must be explicit.");
  }
  if (gas_inapplicable &&
      energy.gas_or_compressibility != 0.0) {
    throw std::invalid_argument(
        "An inapplicable gas/compressibility energy channel must be exactly zero.");
  }
}

void validateDissipation(
    const FreeSurfacePhysicalDissipationRate& dissipation)
{
  requireNonnegativeChannel(
      dissipation.bulk_viscous, "dissipation.bulk_viscous");
  requireNonnegativeChannel(
      dissipation.navier_slip, "dissipation.navier_slip");
  requireNonnegativeChannel(
      dissipation.line_friction, "dissipation.line_friction");
}

void validateExternalWork(const FreeSurfaceExternalWork& work)
{
  requireFiniteChannel(work.pressure, "external.pressure");
  requireFiniteChannel(work.body_force, "external.body_force");
  requireFiniteChannel(
      work.imposed_traction, "external.imposed_traction");
  requireFiniteChannel(
      work.open_boundary_flux, "external.open_boundary_flux");
}

void validateNumericalWork(const FreeSurfaceNumericalWork& work)
{
  const std::array<std::pair<double, std::string_view>, 17> channels{{
      {work.time_discretization, "numerical.time_discretization"},
      {work.kinetic_domain_transport,
       "numerical.kinetic_domain_transport"},
      {work.gravitational_transport_coupling,
       "numerical.gravitational_transport_coupling"},
      {work.convection, "numerical.convection"},
      {work.pressure_continuity, "numerical.pressure_continuity"},
      {work.surface_transport_coupling,
       "numerical.surface_transport_coupling"},
      {work.weak_boundary, "numerical.weak_boundary"},
      {work.vms_pspg, "numerical.vms_pspg"},
      {work.cut_stabilization, "numerical.cut_stabilization"},
      {work.ghost_penalty, "numerical.ghost_penalty"},
      {work.aggregation, "numerical.aggregation"},
      {work.extension, "numerical.extension"},
      {work.pruning, "numerical.pruning"},
      {work.limiting, "numerical.limiting"},
      {work.redistancing, "numerical.redistancing"},
      {work.local_reconciliation,
       "numerical.local_reconciliation"},
      {work.global_correction, "numerical.global_correction"},
  }};
  for (const auto& [value, channel] : channels) {
    requireFiniteChannel(value, channel);
  }
}

void validateChannelSource(
    double value,
    const FreeSurfaceEnergyChannelSource& source,
    std::string_view channel)
{
  const bool named_owner =
      source.owner.find_first_not_of(" \t\r\n") != std::string::npos;
  const bool produced =
      source.applicability ==
      FreeSurfaceEnergyChannelApplicability::Produced;
  const bool inapplicable =
      source.applicability ==
      FreeSurfaceEnergyChannelApplicability::NotApplicable;
  if ((!produced && !inapplicable) || !named_owner) {
    throw std::invalid_argument(
        "Free-surface energy channel '" + std::string(channel) +
        "' requires one named owner and an explicit applicability decision.");
  }
  if (inapplicable && value != 0.0) {
    throw std::invalid_argument(
        "Inapplicable free-surface energy channel '" +
        std::string(channel) + "' must be exactly zero.");
  }
}

void validateStoredEnergySources(
    const FreeSurfaceStoredEnergy& before,
    const FreeSurfaceStoredEnergy& physical_endpoint,
    const FreeSurfaceStoredEnergy& after,
    const FreeSurfaceStoredEnergySources& sources)
{
  const auto validate_endpoint_channel =
      [&](double before_value,
          double physical_endpoint_value,
          double after_value,
          const FreeSurfaceEnergyChannelSource& source,
          std::string_view channel) {
        validateChannelSource(before_value, source, channel);
        validateChannelSource(
            physical_endpoint_value, source, channel);
        validateChannelSource(after_value, source, channel);
      };
  validate_endpoint_channel(
      before.kinetic,
      physical_endpoint.kinetic,
      after.kinetic,
      sources.kinetic,
      "stored.kinetic");
  validate_endpoint_channel(
      before.gravitational,
      physical_endpoint.gravitational,
      after.gravitational,
      sources.gravitational,
      "stored.gravitational");
  validate_endpoint_channel(
      before.liquid_gas_surface,
      physical_endpoint.liquid_gas_surface,
      after.liquid_gas_surface,
      sources.liquid_gas_surface,
      "stored.liquid_gas_surface");
  validate_endpoint_channel(
      before.solid_liquid_wall,
      physical_endpoint.solid_liquid_wall,
      after.solid_liquid_wall,
      sources.solid_liquid_wall,
      "stored.solid_liquid_wall");
  validate_endpoint_channel(
      before.gas_or_compressibility,
      physical_endpoint.gas_or_compressibility,
      after.gas_or_compressibility,
      sources.gas_or_compressibility,
      "stored.gas_or_compressibility");

  const auto expected_gas_source =
      before.gas_applicability ==
              FreeSurfaceGasEnergyApplicability::Active
          ? FreeSurfaceEnergyChannelApplicability::Produced
          : FreeSurfaceEnergyChannelApplicability::NotApplicable;
  if (sources.gas_or_compressibility.applicability !=
      expected_gas_source) {
    throw std::invalid_argument(
        "Gas/compressibility energy applicability and its named channel source disagree.");
  }
}

void validateDissipationSources(
    const FreeSurfacePhysicalDissipationRate& values,
    const FreeSurfacePhysicalDissipationSources& sources)
{
  validateChannelSource(
      values.bulk_viscous,
      sources.bulk_viscous,
      "dissipation.bulk_viscous");
  validateChannelSource(
      values.navier_slip,
      sources.navier_slip,
      "dissipation.navier_slip");
  validateChannelSource(
      values.line_friction,
      sources.line_friction,
      "dissipation.line_friction");
}

void validateExternalWorkSources(
    const FreeSurfaceExternalWork& values,
    const FreeSurfaceExternalWorkSources& sources)
{
  validateChannelSource(
      values.pressure, sources.pressure, "external.pressure");
  validateChannelSource(
      values.body_force, sources.body_force, "external.body_force");
  validateChannelSource(
      values.imposed_traction,
      sources.imposed_traction,
      "external.imposed_traction");
  validateChannelSource(
      values.open_boundary_flux,
      sources.open_boundary_flux,
      "external.open_boundary_flux");
}

void validateNumericalWorkSources(
    const FreeSurfaceNumericalWork& values,
    const FreeSurfaceNumericalWorkSources& sources)
{
  validateChannelSource(
      values.time_discretization,
      sources.time_discretization,
      "numerical.time_discretization");
  validateChannelSource(
      values.kinetic_domain_transport,
      sources.kinetic_domain_transport,
      "numerical.kinetic_domain_transport");
  validateChannelSource(
      values.gravitational_transport_coupling,
      sources.gravitational_transport_coupling,
      "numerical.gravitational_transport_coupling");
  validateChannelSource(
      values.convection,
      sources.convection,
      "numerical.convection");
  validateChannelSource(
      values.pressure_continuity,
      sources.pressure_continuity,
      "numerical.pressure_continuity");
  validateChannelSource(
      values.surface_transport_coupling,
      sources.surface_transport_coupling,
      "numerical.surface_transport_coupling");
  validateChannelSource(
      values.weak_boundary,
      sources.weak_boundary,
      "numerical.weak_boundary");
  validateChannelSource(
      values.vms_pspg, sources.vms_pspg, "numerical.vms_pspg");
  validateChannelSource(
      values.cut_stabilization,
      sources.cut_stabilization,
      "numerical.cut_stabilization");
  validateChannelSource(
      values.ghost_penalty,
      sources.ghost_penalty,
      "numerical.ghost_penalty");
  validateChannelSource(
      values.aggregation,
      sources.aggregation,
      "numerical.aggregation");
  validateChannelSource(
      values.extension,
      sources.extension,
      "numerical.extension");
  validateChannelSource(
      values.pruning, sources.pruning, "numerical.pruning");
  validateChannelSource(
      values.limiting, sources.limiting, "numerical.limiting");
  validateChannelSource(
      values.redistancing,
      sources.redistancing,
      "numerical.redistancing");
  validateChannelSource(
      values.local_reconciliation,
      sources.local_reconciliation,
      "numerical.local_reconciliation");
  validateChannelSource(
      values.global_correction,
      sources.global_correction,
      "numerical.global_correction");
}

double storedEnergy(const FreeSurfaceStoredEnergy& energy) noexcept
{
  return energy.kinetic + energy.gravitational +
      energy.liquid_gas_surface + energy.solid_liquid_wall +
      energy.gas_or_compressibility;
}

double physicalDissipationRate(
    const FreeSurfacePhysicalDissipationRate& dissipation) noexcept
{
  return dissipation.bulk_viscous + dissipation.navier_slip +
      dissipation.line_friction;
}

double externalWork(const FreeSurfaceExternalWork& work) noexcept
{
  return work.pressure + work.body_force + work.imposed_traction +
      work.open_boundary_flux;
}

double numericalWork(const FreeSurfaceNumericalWork& work) noexcept
{
  return work.time_discretization +
      work.kinetic_domain_transport +
      work.gravitational_transport_coupling + work.convection +
      work.pressure_continuity +
      work.surface_transport_coupling + work.weak_boundary +
      work.vms_pspg + work.cut_stabilization +
      work.ghost_penalty + work.aggregation + work.extension +
      work.pruning + work.limiting + work.redistancing +
      work.local_reconciliation + work.global_correction;
}

template <typename Value>
void reserveForOneMore(std::vector<Value>& values)
{
  if (values.size() < values.capacity()) {
    return;
  }
  const auto maximum = values.max_size();
  if (values.size() == maximum) {
    throw std::length_error(
        "Free-surface energy history reached its maximum size.");
  }
  const auto doubled =
      values.capacity() > maximum / 2u
          ? maximum
          : std::max<std::size_t>(1u, values.capacity() * 2u);
  values.reserve(std::max(values.size() + 1u, doubled));
}

bool sameStoredEnergy(
    const FreeSurfaceStoredEnergy& left,
    const FreeSurfaceStoredEnergy& right) noexcept
{
  return left.gas_applicability == right.gas_applicability &&
      left.kinetic == right.kinetic &&
      left.gravitational == right.gravitational &&
      left.liquid_gas_surface == right.liquid_gas_surface &&
      left.solid_liquid_wall == right.solid_liquid_wall &&
      left.gas_or_compressibility == right.gas_or_compressibility;
}

void validateAttemptEnvelope(
    const FreeSurfaceEnergyAttemptMetadata& metadata)
{
  const bool extension_revisions_valid =
      (!metadata.extension_map_revision_before.has_value() ||
       *metadata.extension_map_revision_before != 0u) &&
      (!metadata.physical_endpoint_extension_map_revision.has_value() ||
       *metadata.physical_endpoint_extension_map_revision != 0u) &&
      (!metadata.extension_map_revision_after.has_value() ||
       *metadata.extension_map_revision_after != 0u);
  const double time_span = metadata.time_after - metadata.time_before;
  const double time_scale = std::max(
      {1.0,
       std::abs(metadata.time_before),
       std::abs(metadata.time_after),
       std::abs(metadata.dt)});
  const double time_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * time_scale;
  if (metadata.transaction_id == 0u || metadata.step == 0u ||
      metadata.attempt == 0u ||
      !finite(metadata.time_before) || !finite(metadata.time_after) ||
      !finite(metadata.dt) || metadata.dt <= 0.0 ||
      metadata.time_after <= metadata.time_before ||
      std::abs(time_span - metadata.dt) > time_tolerance ||
      metadata.algebraic_state_revision_before == 0u ||
      metadata.snapshot_set_revision_before == 0u ||
      metadata.mesh_topology_set_revision_before == 0u ||
      metadata.cut_topology_set_revision_before == 0u ||
      !extension_revisions_valid) {
    throw std::invalid_argument(
        "Free-surface energy-attempt envelope is incomplete or is not one positive endpoint interval.");
  }
}

bool endpointMetadataComplete(
    const FreeSurfaceEnergyAttemptMetadata& metadata) noexcept
{
  const bool extension_coverage_complete =
      metadata.extension_map_revision_before.has_value() ==
          metadata.physical_endpoint_extension_map_revision.has_value() &&
      metadata.extension_map_revision_before.has_value() ==
          metadata.extension_map_revision_after.has_value();
  return extension_coverage_complete &&
      metadata.physical_endpoint_algebraic_state_revision != 0u &&
      metadata.algebraic_state_revision_after != 0u &&
      metadata.physical_endpoint_snapshot_set_revision != 0u &&
      metadata.snapshot_set_revision_after != 0u &&
      metadata.physical_endpoint_mesh_topology_set_revision != 0u &&
      metadata.mesh_topology_set_revision_after != 0u &&
      metadata.physical_endpoint_cut_topology_set_revision != 0u &&
      metadata.cut_topology_set_revision_after != 0u;
}

void requireCompleteEndpointMetadata(
    const FreeSurfaceEnergyAttemptMetadata& metadata)
{
  if (!endpointMetadataComplete(metadata)) {
    throw std::invalid_argument(
        "A staged free-surface energy balance requires complete physical and post-maintenance endpoint provenance.");
  }
}

bool topologyChanged(
    const FreeSurfaceEnergyAttemptMetadata& metadata) noexcept
{
  return metadata.mesh_topology_set_revision_before !=
          metadata.physical_endpoint_mesh_topology_set_revision ||
      metadata.physical_endpoint_mesh_topology_set_revision !=
          metadata.mesh_topology_set_revision_after ||
      metadata.cut_topology_set_revision_before !=
          metadata.physical_endpoint_cut_topology_set_revision ||
      metadata.physical_endpoint_cut_topology_set_revision !=
          metadata.cut_topology_set_revision_after;
}

void requireStageableBackwardEulerEndpoint(
    const FreeSurfaceEnergyAttemptMetadata& metadata)
{
  requireCompleteEndpointMetadata(metadata);
  if (topologyChanged(metadata)) {
    throw std::invalid_argument(
        "A free-surface energy balance cannot be staged across a mesh or cut topology change.");
  }
  if (metadata.temporal_scheme !=
          FreeSurfaceEnergyTemporalScheme::BackwardEuler ||
      !finite(metadata.physical_evaluation_time) ||
      !finite(metadata.physical_evaluation_stage_fraction) ||
      metadata.physical_evaluation_time != metadata.time_after ||
      metadata.physical_evaluation_stage_fraction != 1.0) {
    throw std::invalid_argument(
        "A staged free-surface energy balance requires explicit backward-Euler physical evaluation at the accepted endpoint.");
  }
}

void validateRejectionReason(
    const FreeSurfaceEnergyAttemptMetadata& metadata,
    FreeSurfaceEnergyRejectionReason reason)
{
  const bool recognized_reason =
      reason == FreeSurfaceEnergyRejectionReason::NonlinearSolveFailure ||
      reason == FreeSurfaceEnergyRejectionReason::StepControllerRejection ||
      reason == FreeSurfaceEnergyRejectionReason::PreacceptRejection ||
      reason == FreeSurfaceEnergyRejectionReason::TopologyChange ||
      reason == FreeSurfaceEnergyRejectionReason::MaintenanceRollback ||
      reason == FreeSurfaceEnergyRejectionReason::PublicationFailure;
  if (!recognized_reason) {
    throw std::invalid_argument(
        "A rejected free-surface energy attempt requires an explicit reason.");
  }
  const bool complete_endpoint = endpointMetadataComplete(metadata);
  const bool topology_changed =
      complete_endpoint && topologyChanged(metadata);
  if ((reason == FreeSurfaceEnergyRejectionReason::TopologyChange &&
       (!complete_endpoint || !topology_changed)) ||
      (reason != FreeSurfaceEnergyRejectionReason::TopologyChange &&
       topology_changed)) {
    throw std::invalid_argument(
        "The topology-change rejection reason must agree with the recorded mesh and cut topology revisions.");
  }
}

} // namespace

void FreeSurfaceEnergyLedger::beginAttempt(
    FreeSurfaceEnergyAttemptMetadata metadata)
{
  if (active_attempt_.has_value()) {
    throw std::logic_error(
        "A free-surface energy attempt is already active.");
  }
  if (trial_balance_.has_value()) {
    throw std::logic_error(
        "A free-surface trial energy balance remains unpublished.");
  }
  validateAttemptEnvelope(metadata);
  if (metadata.transaction_id <= last_transaction_id_) {
    throw std::invalid_argument(
        "Free-surface energy transaction identifiers must increase.");
  }
  if (last_published_status_.has_value() !=
      last_published_metadata_.has_value()) {
    throw std::logic_error(
        "Free-surface energy published-attempt sequencing state is incomplete.");
  }
  if (!last_published_metadata_.has_value()) {
    if (metadata.attempt != 1u) {
      throw std::invalid_argument(
          "The first published free-surface energy attempt for a step must use attempt one.");
    }
  } else {
    const auto& previous = *last_published_metadata_;
    if (*last_published_status_ ==
        FreeSurfaceEnergyAttemptStatus::Accepted) {
      if (previous.step ==
              std::numeric_limits<std::uint64_t>::max() ||
          metadata.step != previous.step + 1u ||
          metadata.attempt != 1u) {
        throw std::invalid_argument(
            "A free-surface energy attempt after acceptance must start the next step at attempt one.");
      }
    } else if (*last_published_status_ ==
               FreeSurfaceEnergyAttemptStatus::Rejected) {
      const bool retry_valid =
          previous.attempt !=
              std::numeric_limits<std::uint64_t>::max() &&
          metadata.step == previous.step &&
          metadata.attempt == previous.attempt + 1u &&
          metadata.time_before == previous.time_before &&
          metadata.algebraic_state_revision_before ==
              previous.algebraic_state_revision_before &&
          metadata.snapshot_set_revision_before ==
              previous.snapshot_set_revision_before &&
          metadata.mesh_topology_set_revision_before ==
              previous.mesh_topology_set_revision_before &&
          metadata.cut_topology_set_revision_before ==
              previous.cut_topology_set_revision_before &&
          metadata.extension_map_revision_before ==
              previous.extension_map_revision_before;
      if (!retry_valid) {
        throw std::invalid_argument(
            "A rejected free-surface energy attempt must be followed by the next attempt on the same accepted starting endpoint.");
      }
    } else {
      throw std::logic_error(
          "A trial free-surface energy attempt cannot be a published sequencing predecessor.");
    }
  }
  if (!accepted_attempts_.empty()) {
    const auto& previous = accepted_attempts_.back().metadata;
    const bool endpoint_chain_valid =
        previous.step != std::numeric_limits<std::uint64_t>::max() &&
        metadata.step == previous.step + 1u &&
        metadata.time_before == previous.time_after &&
        metadata.algebraic_state_revision_before ==
            previous.algebraic_state_revision_after &&
        metadata.snapshot_set_revision_before ==
            previous.snapshot_set_revision_after &&
        metadata.mesh_topology_set_revision_before ==
            previous.mesh_topology_set_revision_after &&
        metadata.cut_topology_set_revision_before ==
            previous.cut_topology_set_revision_after &&
        metadata.extension_map_revision_before ==
            previous.extension_map_revision_after;
    if (!endpoint_chain_valid) {
      throw std::invalid_argument(
          "A free-surface energy attempt must continue the latest accepted step, time, algebraic state, geometry, topology, and extension endpoint.");
    }
  }

  last_transaction_id_ = metadata.transaction_id;
  active_attempt_ = std::move(metadata);
}

void FreeSurfaceEnergyLedger::stageBalance(
    FreeSurfaceStoredEnergy before,
    FreeSurfaceStoredEnergy physical_endpoint_before_maintenance,
    FreeSurfaceStoredEnergy after,
    FreeSurfacePhysicalDissipationRate dissipation_rate,
    FreeSurfaceExternalWork external_work,
    FreeSurfaceNumericalWork numerical_work,
    FreeSurfaceEnergyChannelSources channel_sources)
{
  if (!active_attempt_.has_value()) {
    throw std::logic_error(
        "A free-surface energy balance requires an active attempt.");
  }
  if (trial_balance_.has_value()) {
    throw std::logic_error(
        "A free-surface energy attempt accepts exactly one complete balance.");
  }
  requireStageableBackwardEulerEndpoint(*active_attempt_);
  validateStoredEnergy(before);
  validateStoredEnergy(physical_endpoint_before_maintenance);
  validateStoredEnergy(after);
  if (before.gas_applicability !=
          physical_endpoint_before_maintenance.gas_applicability ||
      before.gas_applicability != after.gas_applicability) {
    throw std::invalid_argument(
        "Gas/compressibility energy applicability cannot change within one step.");
  }
  validateDissipation(dissipation_rate);
  validateExternalWork(external_work);
  validateNumericalWork(numerical_work);
  validateStoredEnergySources(
      before,
      physical_endpoint_before_maintenance,
      after,
      channel_sources.stored);
  validateDissipationSources(
      dissipation_rate, channel_sources.dissipation);
  validateExternalWorkSources(
      external_work, channel_sources.external);
  validateNumericalWorkSources(
      numerical_work, channel_sources.numerical);
  const FreeSurfaceEnergyAttempt* latest_staged_attempt = nullptr;
  if (!accepted_attempts_.empty()) {
    latest_staged_attempt = &accepted_attempts_.back();
  }
  const auto latest_staged_rejection = std::find_if(
      rejected_attempts_.rbegin(),
      rejected_attempts_.rend(),
      [](const auto& attempt) {
        return attempt.balance_staged;
      });
  if (latest_staged_rejection != rejected_attempts_.rend() &&
      (latest_staged_attempt == nullptr ||
       latest_staged_rejection->metadata.transaction_id >
           latest_staged_attempt->metadata.transaction_id)) {
    latest_staged_attempt = &*latest_staged_rejection;
  }
  if (latest_staged_attempt != nullptr &&
      latest_staged_attempt->channel_sources != channel_sources) {
    throw std::invalid_argument(
        "Free-surface energy channel ownership and applicability cannot change within one ledger history.");
  }
  if (last_published_status_ ==
          FreeSurfaceEnergyAttemptStatus::Rejected &&
      latest_staged_rejection != rejected_attempts_.rend() &&
      latest_staged_rejection->metadata.step ==
          active_attempt_->step &&
      !sameStoredEnergy(before, latest_staged_rejection->before)) {
    throw std::invalid_argument(
        "A retried free-surface energy attempt must preserve the latest staged rejected starting energy.");
  }
  if (!accepted_attempts_.empty() &&
      !sameStoredEnergy(
          before, accepted_attempts_.back().after)) {
    throw std::invalid_argument(
        "Free-surface stored energy before an attempt must match the latest accepted endpoint.");
  }

  const double energy_before = storedEnergy(before);
  const double physical_endpoint_energy =
      storedEnergy(physical_endpoint_before_maintenance);
  const double energy_after = storedEnergy(after);
  const double physical_energy_change =
      physical_endpoint_energy - energy_before;
  const double maintenance_energy_change =
      energy_after - physical_endpoint_energy;
  const double energy_change = energy_after - energy_before;
  const double integrated_dissipation =
      active_attempt_->dt * physicalDissipationRate(dissipation_rate);
  const double total_external_work = externalWork(external_work);
  const double total_numerical_work = numericalWork(numerical_work);
  const double residual = energy_change + integrated_dissipation -
      total_external_work - total_numerical_work;
  const std::array<double, 10> totals{
      energy_before,
      physical_endpoint_energy,
      energy_after,
      physical_energy_change,
      maintenance_energy_change,
      energy_change,
      integrated_dissipation,
      total_external_work,
      total_numerical_work,
      residual};
  if (!std::all_of(totals.begin(), totals.end(), finite)) {
    throw std::invalid_argument(
        "The assembled free-surface energy balance is not finite.");
  }

  trial_balance_ = FreeSurfaceEnergyAttempt{
      .status = FreeSurfaceEnergyAttemptStatus::Trial,
      .rejection_reason = FreeSurfaceEnergyRejectionReason::None,
      .metadata = *active_attempt_,
      .balance_staged = true,
      .before = std::move(before),
      .physical_endpoint_before_maintenance =
          std::move(physical_endpoint_before_maintenance),
      .after = std::move(after),
      .dissipation_rate = std::move(dissipation_rate),
      .external_work = std::move(external_work),
      .numerical_work = std::move(numerical_work),
      .channel_sources = std::move(channel_sources),
      .stored_energy_before = energy_before,
      .stored_energy_physical_endpoint_before_maintenance =
          physical_endpoint_energy,
      .stored_energy_after = energy_after,
      .physical_stored_energy_change = physical_energy_change,
      .maintenance_stored_energy_change =
          maintenance_energy_change,
      .stored_energy_change = energy_change,
      .integrated_physical_dissipation = integrated_dissipation,
      .total_external_work = total_external_work,
      .total_numerical_work = total_numerical_work,
      .trial_balance_residual = residual,
      .accepted_stored_energy_change = 0.0,
      .accepted_physical_stored_energy_change = 0.0,
      .accepted_maintenance_stored_energy_change = 0.0,
      .accepted_integrated_physical_dissipation = 0.0,
      .accepted_external_work = 0.0,
      .accepted_numerical_work = 0.0,
      .accepted_balance_residual = 0.0,
  };
}

void FreeSurfaceEnergyLedger::commitAttempt()
{
  if (!active_attempt_.has_value() || !trial_balance_.has_value()) {
    throw std::logic_error(
        "A complete staged free-surface energy balance is required before commit.");
  }
  if (topologyChanged(*active_attempt_)) {
    throw std::logic_error(
        "A fixed-topology free-surface energy balance cannot be accepted across a topology change.");
  }
  reserveForOneMore(accepted_attempts_);

  trial_balance_->status = FreeSurfaceEnergyAttemptStatus::Accepted;
  trial_balance_->accepted_stored_energy_change =
      trial_balance_->stored_energy_change;
  trial_balance_->accepted_physical_stored_energy_change =
      trial_balance_->physical_stored_energy_change;
  trial_balance_->accepted_maintenance_stored_energy_change =
      trial_balance_->maintenance_stored_energy_change;
  trial_balance_->accepted_integrated_physical_dissipation =
      trial_balance_->integrated_physical_dissipation;
  trial_balance_->accepted_external_work =
      trial_balance_->total_external_work;
  trial_balance_->accepted_numerical_work =
      trial_balance_->total_numerical_work;
  trial_balance_->accepted_balance_residual =
      trial_balance_->trial_balance_residual;
  accepted_attempts_.push_back(std::move(*trial_balance_));
  last_published_status_ = FreeSurfaceEnergyAttemptStatus::Accepted;
  last_published_metadata_ = accepted_attempts_.back().metadata;
  trial_balance_.reset();
  active_attempt_.reset();
}

void FreeSurfaceEnergyLedger::rejectAttempt(
    FreeSurfaceEnergyRejectionReason reason)
{
  if (!active_attempt_.has_value() || !trial_balance_.has_value()) {
    throw std::logic_error(
        "A complete staged free-surface energy balance is required before rejection.");
  }
  validateRejectionReason(*active_attempt_, reason);
  reserveForOneMore(rejected_attempts_);

  trial_balance_->status = FreeSurfaceEnergyAttemptStatus::Rejected;
  trial_balance_->rejection_reason = reason;
  trial_balance_->accepted_stored_energy_change = 0.0;
  trial_balance_->accepted_physical_stored_energy_change = 0.0;
  trial_balance_->accepted_maintenance_stored_energy_change = 0.0;
  trial_balance_->accepted_integrated_physical_dissipation = 0.0;
  trial_balance_->accepted_external_work = 0.0;
  trial_balance_->accepted_numerical_work = 0.0;
  trial_balance_->accepted_balance_residual = 0.0;
  rejected_attempts_.push_back(std::move(*trial_balance_));
  last_published_status_ = FreeSurfaceEnergyAttemptStatus::Rejected;
  last_published_metadata_ = rejected_attempts_.back().metadata;
  trial_balance_.reset();
  active_attempt_.reset();
}

void FreeSurfaceEnergyLedger::rejectUnstagedAttempt(
    FreeSurfaceEnergyRejectionReason reason)
{
  if (!active_attempt_.has_value()) {
    throw std::logic_error(
        "An unstaged free-surface energy rejection requires an active attempt.");
  }
  if (trial_balance_.has_value()) {
    throw std::logic_error(
        "A staged free-surface energy balance must use the staged rejection path.");
  }
  validateRejectionReason(*active_attempt_, reason);
  reserveForOneMore(rejected_attempts_);

  const double unavailable =
      std::numeric_limits<double>::quiet_NaN();
  rejected_attempts_.push_back(FreeSurfaceEnergyAttempt{
      .status = FreeSurfaceEnergyAttemptStatus::Rejected,
      .rejection_reason = reason,
      .metadata = *active_attempt_,
      .balance_staged = false,
      .stored_energy_before = unavailable,
      .stored_energy_physical_endpoint_before_maintenance =
          unavailable,
      .stored_energy_after = unavailable,
      .physical_stored_energy_change = unavailable,
      .maintenance_stored_energy_change = unavailable,
      .stored_energy_change = unavailable,
      .integrated_physical_dissipation = unavailable,
      .total_external_work = unavailable,
      .total_numerical_work = unavailable,
      .trial_balance_residual = unavailable,
      .accepted_stored_energy_change = 0.0,
      .accepted_physical_stored_energy_change = 0.0,
      .accepted_maintenance_stored_energy_change = 0.0,
      .accepted_integrated_physical_dissipation = 0.0,
      .accepted_external_work = 0.0,
      .accepted_numerical_work = 0.0,
      .accepted_balance_residual = 0.0,
  });
  last_published_status_ = FreeSurfaceEnergyAttemptStatus::Rejected;
  last_published_metadata_ = rejected_attempts_.back().metadata;
  active_attempt_.reset();
}

bool FreeSurfaceEnergyLedger::attemptActive() const noexcept
{
  return active_attempt_.has_value();
}

const FreeSurfaceEnergyAttemptMetadata*
FreeSurfaceEnergyLedger::activeAttempt() const noexcept
{
  return active_attempt_.has_value() ? &*active_attempt_ : nullptr;
}

const FreeSurfaceEnergyAttempt*
FreeSurfaceEnergyLedger::trialBalance() const noexcept
{
  return trial_balance_.has_value() ? &*trial_balance_ : nullptr;
}

const std::vector<FreeSurfaceEnergyAttempt>&
FreeSurfaceEnergyLedger::acceptedAttempts() const noexcept
{
  return accepted_attempts_;
}

const std::vector<FreeSurfaceEnergyAttempt>&
FreeSurfaceEnergyLedger::rejectedAttempts() const noexcept
{
  return rejected_attempts_;
}

} // namespace application::core
