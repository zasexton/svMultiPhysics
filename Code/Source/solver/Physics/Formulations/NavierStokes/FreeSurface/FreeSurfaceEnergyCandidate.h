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
 * @brief Complete fixed-topology one-phase free-surface energy record:
 *        the channels of the free-surface energy balance (data for the FE
 *        energy-balance ledger, FE/Systems/EnergyBalanceLedger.h), accepted
 *        functional groups, the energy candidate and its ledger metadata,
 *        rejection reasons, aggregation classification, the residual-work
 *        channels and the physical endpoint channels of the accepted-step
 *        maintenance account.
 *
 * The level-set maintenance work ledger and the accepted-step maintenance
 * account are FE services
 * (FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h).
 */

#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Core/Types.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceWorkLedger.h"
#include "FE/Systems/EnergyBalanceLedger.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/StepController.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/**
 * The channels of the free-surface energy balance, in the order in which
 * they are summed and logged: 5 stored energies, 3 dissipation rates, 4
 * external works and 17 numerical works.  Declared identities: the external
 * pressure work (dt times the exterior-pressure power), the kinetic domain
 * transport (kinetic energy before on the endpoint domain minus the kinetic
 * energy before) and the gravitational and surface/wall transport couplings
 * (stored-energy change minus dt times the endpoint potential power).
 */
[[nodiscard]] const svmp::FE::systems::EnergyBalanceCatalogue&
freeSurfaceEnergyBalanceCatalogue();

/// The seven connected residual-work channels, in their order.
[[nodiscard]] std::span<const std::string> freeSurfaceResidualWorkChannelNames();

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
 * and convert residual pairings to signed step work as -dt R_i(x)[x]
 * (svmp::FE::systems::evaluateResidualStepWork).
 */
[[nodiscard]] FreeSurfaceResidualWorkChannels
evaluateFreeSurfaceResidualWorkChannels(
    std::span<const svmp::FE::systems::ResidualWorkPairing> pairings,
    double dt);

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

struct FreeSurfaceAcceptedEnergyGroup {
  std::uint64_t accepted_step{0u};
  double accepted_time{0.0};
  double dt{0.0};
  std::uint64_t pre_maintenance_endpoint_state_revision{0u};
  std::uint64_t state_revision{0u};
  std::uint64_t snapshot_set_revision{0u};
  std::uint64_t mesh_topology_set_revision{0u};
  std::uint64_t cut_topology_set_revision{0u};
  std::optional<std::uint64_t> extension_map_revision{};
  double kinetic_energy{0.0};
  double gravitational_energy{0.0};
  double gravitational_potential_power{0.0};
  double liquid_gas_surface_energy{0.0};
  double solid_liquid_wall_energy{0.0};
  std::optional<double> surface_wall_potential_power{};
  std::optional<double> bulk_viscous_dissipation_rate{};
  std::optional<double> external_pressure_power{};
  std::optional<svmp::FE::interfaces::
                    FreeSurfaceBackwardEulerKineticWorkState>
      backward_euler_kinetic_work{};
  bool contact_dissipation_applicable{false};
  std::optional<double> navier_slip_dissipation_rate{};
  std::optional<double> line_friction_dissipation_rate{};
  bool geometry_pruning_occurred{false};
};

/**
 * One complete record for the free-surface energy-balance ledger: values
 * and sources per channel of freeSurfaceEnergyBalanceCatalogue() (identity
 * channels NaN) and the inputs of its declared identities.
 */
struct FreeSurfaceCompleteEnergyCandidate {
  svmp::FE::systems::EnergyBalanceAttemptMetadata metadata{};
  svmp::FE::systems::EnergyBalanceValues values{};
  svmp::FE::systems::EnergyBalanceSources sources{};
  std::vector<svmp::FE::systems::EnergyBalanceInput> inputs{};
};

struct FreeSurfaceCompleteEnergyBuildResult {
  std::optional<FreeSurfaceCompleteEnergyCandidate> candidate{};
  std::vector<std::string> missing_requirements{};
};

struct FreeSurfaceAggregationEnergyClassification {
  bool present{false};
  bool neutral{true};
  bool rootless{false};
  std::size_t rooted_row_count{0u};
  double maximum_constraint_residual{0.0};
};

FreeSurfaceAcceptedEnergyGroup summarizeAcceptedFreeSurfaceEnergyGroup(
    std::span<const svmp::FE::systems::
                  FreeSurfaceDiscreteFunctionalHistoryRecord> records);

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
    bool aggregation_rootless);

svmp::FE::systems::EnergyBalanceAttemptMetadata
buildUnstagedFreeSurfaceEnergyAttemptMetadata(
    std::uint64_t transaction_id,
    std::uint64_t prospective_step,
    std::uint64_t attempt,
    double time_before,
    double dt,
    const FreeSurfaceAcceptedEnergyGroup& before);

svmp::FE::systems::EnergyBalanceRejectionReason
freeSurfaceEnergyRejectionReasonForStepReject(
    svmp::FE::timestepping::StepRejectReason reason);

FreeSurfaceAggregationEnergyClassification
classifyAcceptedAggregationEnergy(
    std::span<const std::shared_ptr<const svmp::FE::constraints::
                        SmallCutAggregationProlongationReport>> reports,
    std::span<const svmp::FE::constraints::
                  SmallCutAggregationRefreshReport> refresh_reports,
    std::span<const svmp::FE::Real> accepted_solution);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
