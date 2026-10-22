/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/TwoFluid/TwoFluidStageParticipant.h"

#include <algorithm>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::two_fluid {

using namespace free_surface::detail;

StageParticipant::StageParticipant(
    const free_surface::FreeSurfaceStepContext& context)
    : free_surface::FreeSurfaceStepContext(context)
{
  two_fluid_stage_declarations =
      sim.fe_system->twoFluidAcceptedStageDiagnosticDeclarations();
  const auto [minimum_two_fluid_stage_count,
              maximum_two_fluid_stage_count] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(two_fluid_stage_declarations.size()),
          velocity_extension_artifact_comm);
  if (minimum_two_fluid_stage_count != maximum_two_fluid_stage_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Two-fluid accepted-stage declaration coverage differs across the active FE communicator.");
  }
  const bool local_two_fluid_stage_scheme_supported =
      sim.fe_system->temporalOrder() == 1 &&
      (opts.scheme ==
           svmp::FE::timestepping::SchemeKind::BackwardEuler ||
       opts.scheme == svmp::FE::timestepping::SchemeKind::DG0 ||
       opts.scheme ==
           svmp::FE::timestepping::SchemeKind::GeneralizedAlpha);
  const auto [minimum_two_fluid_stage_scheme_supported,
              maximum_two_fluid_stage_scheme_supported] =
      globalMinMaxUint64(
          local_two_fluid_stage_scheme_supported ? 1u : 0u,
          velocity_extension_artifact_comm);
  if (minimum_two_fluid_stage_scheme_supported !=
      maximum_two_fluid_stage_scheme_supported) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Two-fluid accepted-stage scheme support differs across the active FE communicator.");
  }
  if (maximum_two_fluid_stage_count != 0u &&
      maximum_two_fluid_stage_scheme_supported == 0u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Two-fluid accepted-stage diagnostics require temporal-order-one Backward Euler, DG0, or first-order generalized-alpha.");
  }
  evaluate_two_fluid_stage =
      maximum_two_fluid_stage_count != 0u;
  reconcile_two_fluid_momentum = std::any_of(
      two_fluid_stage_declarations.begin(),
      two_fluid_stage_declarations.end(),
      [](const auto& declaration) {
        return declaration
            .require_conservative_phase_momentum_reconciliation;
      });
  two_fluid_stage_state_size =
      sim.fe_system->dofHandler().getNumDofs();
}

svmp::FE::timestepping::StepHook StageParticipant::hooks() const noexcept
{
  return evaluate_two_fluid_stage
             ? svmp::FE::timestepping::StepHook::CandidateStage
             : svmp::FE::timestepping::StepHook::None;
}

void StageParticipant::onCandidateStage(
    const svmp::FE::timestepping::CandidateStageObservation& observation)
{
  const bool scheme_matches =
      observation.scheme == opts.scheme &&
      (observation.scheme ==
               svmp::FE::timestepping::SchemeKind::BackwardEuler ||
       observation.scheme ==
               svmp::FE::timestepping::SchemeKind::DG0 ||
       observation.scheme ==
               svmp::FE::timestepping::SchemeKind::GeneralizedAlpha);
  const bool generalized_alpha_shape_matches =
      observation.scheme ==
              svmp::FE::timestepping::SchemeKind::GeneralizedAlpha
          ? observation.generalized_alpha.has_value()
          : !observation.generalized_alpha.has_value();
  const bool local_stage_ready =
      scheme_matches && generalized_alpha_shape_matches &&
      observation.temporal_order == 1 &&
      observation.step_index > 0 && observation.attempt_index >= 0 &&
      observation.state_vector != nullptr &&
      observation.rate_vector != nullptr &&
      observation.state_vector != observation.rate_vector &&
      observation.state_vector->size() ==
          two_fluid_stage_state_size &&
      observation.rate_vector->size() ==
          two_fluid_stage_state_size &&
      observation.mesh_revision.has_value();
  if (globalAnyBool(
          !local_stage_ready,
          velocity_extension_artifact_comm)) {
    if (!local_stage_ready) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Two-fluid accepted-stage diagnostics require matching temporal-order-one state/rate vectors, scheme coordinates, and a complete mesh revision.");
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE communicator rank rejected two-fluid stage readiness.");
  }

  const auto stage_state_revision =
      collectiveMeshBoundaryStateFingerprint(
          *observation.state_vector,
          velocity_extension_artifact_comm);
  const auto stage_rate_revision =
      collectiveMeshBoundaryStateFingerprint(
          *observation.rate_vector,
          velocity_extension_artifact_comm);
  std::vector<svmp::FE::Real> stage_solution;
  std::optional<
      svmp::FE::systems::OperatorStageMeasurementMetadata>
      stage_metadata;
  std::exception_ptr local_preparation_failure;
  try {
    svmp::FE::systems::SystemStateView stage_view;
    stage_view.u_vector = observation.state_vector;
    stage_solution = gatherFeOrderedSolution(
        stage_view,
        velocity_extension_artifact_comm);
    const auto& revision = *observation.mesh_revision;
    svmp::FE::systems::OperatorStageMeasurementMetadata metadata{
        .scheme_name =
            observation.scheme ==
                    svmp::FE::timestepping::SchemeKind::BackwardEuler
                ? "BackwardEuler"
                : observation.scheme ==
                          svmp::FE::timestepping::SchemeKind::DG0
                      ? "DG0"
                      : "GeneralizedAlphaFirstOrder",
        .temporal_order = observation.temporal_order,
        .prospective_accepted_step =
            static_cast<std::uint64_t>(observation.step_index),
        .prospective_attempt =
            static_cast<std::uint64_t>(observation.attempt_index),
        .step_start_time = observation.step_start_time,
        .step_end_time = observation.step_end_time,
        .state_time = observation.state_time,
        .rate_time = observation.rate_time,
        .dt = observation.dt,
        .expected_stage_geometry =
            svmp::FE::systems::OperatorStageGeometryMetadata{
                .geometry_revision = revision.geometry_revision,
                .topology_revision = revision.topology_revision,
                .ownership_revision = revision.ownership_revision,
                .numbering_revision = revision.numbering_revision,
                .field_layout_revision =
                    revision.field_layout_revision,
                .label_revision = revision.label_revision,
                .active_configuration_epoch =
                    revision.active_configuration_epoch,
                .coordinate_configuration_key =
                    revision.coordinate_configuration_key,
            },
        .state_revision = stage_state_revision,
        .rate_revision = stage_rate_revision,
        .derivative_fields = observation.time_derivative_fields,
    };
    if (observation.generalized_alpha.has_value()) {
      metadata.generalized_alpha =
          svmp::FE::systems::
              OperatorStageGeneralizedAlphaMetadata{
                  .alpha_f =
                      observation.generalized_alpha->alpha_f,
                  .alpha_m =
                      observation.generalized_alpha->alpha_m,
              };
    }
    stage_metadata.emplace(std::move(metadata));
  } catch (...) {
    local_preparation_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_preparation_failure,
      velocity_extension_artifact_comm,
      "two_fluid_operator_stage_preparation");
  auto states = evaluateCurrentTwoFluidStageDiagnostics(
      sim,
      two_fluid_stage_declarations,
      stage_solution,
      observation.state_time - observation.step_start_time,
      velocity_extension_artifact_comm);
  sim.fe_system->stageTwoFluidAcceptedStageDiagnostics(
      std::move(*stage_metadata), states);
  std::vector<svmp::FE::constraints::
                  SmallCutAggregationRefreshReport>
      captured_aggregation_reports;
  std::exception_ptr local_capture_failure;
  try {
    if (pending_two_fluid_operator_stage_aggregation_captured) {
      throw std::logic_error(
          "Two-fluid operator-stage aggregation evidence is not one-shot for the active candidate.");
    }
    const auto reports = sim.fe_system
                             ->completedSmallCutAggregationRefreshReports();
    captured_aggregation_reports.reserve(
        4u * two_fluid_stage_declarations.size());
    for (const auto& report : reports) {
      const bool required = std::any_of(
          two_fluid_stage_declarations.begin(),
          two_fluid_stage_declarations.end(),
          [&](const auto& declaration) {
            if (report.interface_marker !=
                declaration.interface_marker) {
              return false;
            }
            return report.field ==
                       declaration.negative_velocity_field ||
                   report.field ==
                       declaration.negative_pressure_field ||
                   report.field ==
                       declaration.positive_velocity_field ||
                   report.field ==
                       declaration.positive_pressure_field;
          });
      if (!required) {
        continue;
      }
      auto retained = report;
      retained.local_lineage = {};
      retained.canonical_active_features = {};
      if (retained.canonical_topology_transition.has_value()) {
        retained.canonical_topology_transition
            ->local_lineage_before = {};
        retained.canonical_topology_transition
            ->local_lineage_after = {};
        retained.canonical_topology_transition
            ->canonical_feature_transitions = {};
      }
      captured_aggregation_reports.push_back(
          std::move(retained));
    }
  } catch (...) {
    local_capture_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_capture_failure,
      velocity_extension_artifact_comm,
      "two_fluid_operator_stage_aggregation_capture");
  pending_two_fluid_operator_stage_aggregation_reports =
      std::move(captured_aggregation_reports);
  pending_two_fluid_operator_stage_aggregation_captured = true;
}

void StageParticipant::bindCandidateNumerics(
    const svmp::FE::timestepping::NewtonReport& newton_report)
{
  if (evaluate_two_fluid_stage) {
    if (!pending_two_fluid_operator_stage_aggregation_captured) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Two-fluid candidate reached its final gate without solved operator-stage aggregation evidence.");
    }
    sim.fe_system->bindPendingTwoFluidAcceptedStageNumerics(
        makeTwoFluidAcceptedStageSolveTelemetry(newton_report),
        pending_two_fluid_operator_stage_aggregation_reports);
    pending_two_fluid_operator_stage_aggregation_reports.clear();
    pending_two_fluid_operator_stage_aggregation_captured = false;
  }
}

void StageParticipant::resetAggregationEvidence() noexcept
{
  pending_two_fluid_operator_stage_aggregation_reports.clear();
  pending_two_fluid_operator_stage_aggregation_captured = false;
}

void StageParticipant::discardCandidate() noexcept
{
  sim.fe_system->discardPendingTwoFluidAcceptedStageDiagnostics();
}

void StageParticipant::commitAccepted(
    const svmp::FE::timestepping::TimeHistory& h)
{
  if (evaluate_two_fluid_stage) {
    sim.fe_system->commitPendingTwoFluidAcceptedStageDiagnostics(
        static_cast<std::uint64_t>(h.stepIndex()),
        static_cast<svmp::FE::Real>(h.time()),
        static_cast<svmp::FE::Real>(h.dt()));
  }
}

} // namespace svmp::Physics::formulations::navier_stokes::two_fluid
