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
 *        accepted functional groups, the energy candidate and its ledger
 *        metadata, rejection reasons and aggregation classification.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyAccount.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyLedger.h"

#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Core/Types.h"
#include "FE/Interfaces/FreeSurfaceGeometrySnapshot.h"
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

struct FreeSurfaceCompleteEnergyCandidate {
  FreeSurfaceEnergyAttemptMetadata metadata{};
  FreeSurfaceStoredEnergy before{};
  FreeSurfaceStoredEnergy physical_endpoint{};
  FreeSurfaceStoredEnergy after{};
  FreeSurfacePhysicalDissipationRate dissipation{};
  FreeSurfaceExternalWork external{};
  FreeSurfaceNumericalWork numerical{};
  FreeSurfaceEnergyChannelSources sources{};
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

void appendFreeSurfaceEnergyChannelSource(
    std::ostream& stream,
    std::string_view channel,
    const FreeSurfaceEnergyChannelSource& source);

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

FreeSurfaceEnergyAttemptMetadata
buildUnstagedFreeSurfaceEnergyAttemptMetadata(
    std::uint64_t transaction_id,
    std::uint64_t prospective_step,
    std::uint64_t attempt,
    double time_before,
    double dt,
    const FreeSurfaceAcceptedEnergyGroup& before);

FreeSurfaceEnergyRejectionReason
freeSurfaceEnergyRejectionReasonForStepReject(
    svmp::FE::timestepping::StepRejectReason reason);

const char* freeSurfaceEnergyRejectionReasonName(
    FreeSurfaceEnergyRejectionReason reason) noexcept;

FreeSurfaceAggregationEnergyClassification
classifyAcceptedAggregationEnergy(
    std::span<const std::shared_ptr<const svmp::FE::constraints::
                        SmallCutAggregationProlongationReport>> reports,
    std::span<const svmp::FE::constraints::
                  SmallCutAggregationRefreshReport> refresh_reports,
    std::span<const svmp::FE::Real> accepted_solution);

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
