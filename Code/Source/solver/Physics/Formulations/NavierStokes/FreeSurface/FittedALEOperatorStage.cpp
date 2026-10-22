/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FittedALEOperatorStage.h"

#include "FE/Systems/FESystem.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FittedALEStageRecords.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using namespace detail;

void requireFittedALEOperatorStageSchemeCoverage(
    std::uint64_t declaration_count,
    bool scheme_supported,
    std::string_view scheme_name,
    int temporal_order)
{
  if (declaration_count == 0u || scheme_supported) {
    return;
  }
  throw std::runtime_error(
      "[svMultiPhysics::Application] Registered fitted-ALE operator-stage "
      "normal measurements require a supported temporal-order-one scheme "
      "before the transient TimeLoop starts; scheme='" +
      std::string(scheme_name) + "' temporal_order=" +
      std::to_string(temporal_order) + " declaration_count=" +
      std::to_string(declaration_count) +
      ". Supported schemes are BackwardEuler, DG0, and first-order "
      "GeneralizedAlpha.");
}

FittedALEStageParticipant::FittedALEStageParticipant(
    const FreeSurfaceStepContext& context)
    : FreeSurfaceStepContext(context)
{
  const auto fitted_ale_operator_stage_declarations =
      free_surface::fittedALERecords(*sim.fe_system)
          .fittedALENormalOperatorStageMeasurementDeclarations();
  const auto [minimum_fitted_ale_measurement_count,
              maximum_fitted_ale_measurement_count] =
      globalMinMaxUint64(
          static_cast<std::uint64_t>(
              fitted_ale_operator_stage_declarations.size()),
          velocity_extension_artifact_comm);
  if (minimum_fitted_ale_measurement_count !=
      maximum_fitted_ale_measurement_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Fitted-ALE operator-stage normal "
        "measurement declaration count differs across the active FE "
        "communicator.");
  }
  const bool local_fitted_ale_operator_stage_scheme_supported =
      sim.fe_system->temporalOrder() == 1 &&
      (opts.scheme ==
           svmp::FE::timestepping::SchemeKind::BackwardEuler ||
       opts.scheme == svmp::FE::timestepping::SchemeKind::DG0 ||
       opts.scheme ==
           svmp::FE::timestepping::SchemeKind::GeneralizedAlpha);
  const auto [minimum_fitted_ale_scheme_supported,
              maximum_fitted_ale_scheme_supported] =
      globalMinMaxUint64(
          local_fitted_ale_operator_stage_scheme_supported ? 1u : 0u,
          velocity_extension_artifact_comm);
  if (minimum_fitted_ale_scheme_supported !=
      maximum_fitted_ale_scheme_supported) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Fitted-ALE operator-stage "
        "measurement scheme support differs across the active FE "
        "communicator.");
  }
  const bool fitted_ale_operator_stage_scheme_supported =
      maximum_fitted_ale_scheme_supported == 1u;
  evaluate_fitted_ale_operator_stage =
      maximum_fitted_ale_measurement_count != 0u &&
      fitted_ale_operator_stage_scheme_supported;
  fitted_ale_operator_stage_state_size =
      sim.fe_system->dofHandler().getNumDofs();
  requireFittedALEOperatorStageSchemeCoverage(
      maximum_fitted_ale_measurement_count,
      fitted_ale_operator_stage_scheme_supported,
      transient_scheme.canonical_name,
      sim.fe_system->temporalOrder());
}

svmp::FE::timestepping::StepHook FittedALEStageParticipant::hooks()
    const noexcept
{
  return evaluate_fitted_ale_operator_stage
             ? svmp::FE::timestepping::StepHook::CandidateStage
             : svmp::FE::timestepping::StepHook::None;
}

void FittedALEStageParticipant::onCandidateStage(
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
      observation.step_index > 0 &&
      observation.attempt_index >= 0 &&
      observation.state_vector != nullptr &&
      observation.rate_vector != nullptr &&
      observation.state_vector != observation.rate_vector &&
      observation.state_vector->size() ==
          fitted_ale_operator_stage_state_size &&
      observation.rate_vector->size() ==
          fitted_ale_operator_stage_state_size &&
      observation.mesh_revision.has_value();
  if (globalAnyBool(
          !local_stage_ready,
          velocity_extension_artifact_comm)) {
    if (!local_stage_ready) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Fitted-ALE operator-stage "
          "measurement requires matching temporal-order-one stage "
          "state/rate vectors, scheme coordinates, and a complete "
          "rank-local mesh revision stamp.");
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE "
        "communicator rank rejected fitted-ALE operator-stage "
        "measurement readiness.");
  }

  const auto stage_state_revision =
      collectiveMeshBoundaryStateFingerprint(
          *observation.state_vector,
          velocity_extension_artifact_comm);
  const auto stage_rate_revision =
      collectiveMeshBoundaryStateFingerprint(
          *observation.rate_vector,
          velocity_extension_artifact_comm);

  std::optional<svmp::FE::systems::SystemStateView> stage_state;
  std::optional<
      svmp::FE::systems::OperatorStageMeasurementMetadata>
      stage_metadata;
  std::exception_ptr local_stage_construction_failure;
  try {
    stage_state.emplace();
    stage_state->time = observation.state_time;
    stage_state->dt = observation.dt;
    stage_state->effective_dt = observation.dt;
    stage_state->dt_prev = observation.dt;
    stage_state->u_vector = observation.state_vector;
    stage_state->u_prev_vector = observation.rate_vector;
    stage_state->u_prev2 = {};
    stage_state->u_prev2_vector = nullptr;
    stage_state->u_history = {};
    stage_state->dt_history = {};

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
                .geometry_revision =
                    revision.geometry_revision,
                .topology_revision =
                    revision.topology_revision,
                .ownership_revision =
                    revision.ownership_revision,
                .numbering_revision =
                    revision.numbering_revision,
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
        .derivative_fields =
            observation.time_derivative_fields,
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
    local_stage_construction_failure =
        std::current_exception();
  }
  if (globalAnyBool(
          local_stage_construction_failure != nullptr,
          velocity_extension_artifact_comm)) {
    if (local_stage_construction_failure != nullptr) {
      std::rethrow_exception(
          local_stage_construction_failure);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE "
        "communicator rank could not construct fitted-ALE "
        "operator-stage measurement metadata.");
  }
  free_surface::fittedALERecords(*sim.fe_system)
      .stageFittedALENormalOperatorStageMeasurements(
          *stage_state, std::move(*stage_metadata));
}

void FittedALEStageParticipant::discardCandidate() noexcept
{
  free_surface::fittedALERecords(*sim.fe_system)
      .discardPendingFittedALENormalOperatorStageMeasurements();
}

void FittedALEStageParticipant::commitAccepted(
    const svmp::FE::timestepping::TimeHistory& h)
{
  if (evaluate_fitted_ale_operator_stage) {
    free_surface::fittedALERecords(*sim.fe_system)
        .commitPendingFittedALENormalOperatorStageMeasurements(
            static_cast<std::uint64_t>(h.stepIndex()),
            static_cast<svmp::FE::Real>(h.time()),
            static_cast<svmp::FE::Real>(h.dt()));
  }
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
