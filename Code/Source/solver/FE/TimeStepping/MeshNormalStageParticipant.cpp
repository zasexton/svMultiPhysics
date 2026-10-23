/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/MeshNormalStageParticipant.h"

#include "Backends/Interfaces/GenericVector.h"
#include "Systems/FESystem.h"
#include "Systems/MeshNormalConstraintStageMeasurements.h"

#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {

// The participant was the fitted-ALE stage participant of the Navier-Stokes
// free-surface physics (FreeSurface/FittedALEOperatorStage.cpp); its checks,
// collectives and messages are unchanged.

void requireMeshNormalStageSchemeCoverage(
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

MeshNormalStageParticipant::MeshNormalStageParticipant(
    const MeshNormalStageContext& context)
    : context_(context)
{
  const auto fitted_ale_operator_stage_declarations =
      systems::meshNormalConstraintStageMeasurements(*context_.system)
          .fittedALENormalOperatorStageMeasurementDeclarations();
  const auto [minimum_fitted_ale_measurement_count,
              maximum_fitted_ale_measurement_count] =
      systems::collectiveMinMaxUint64(
          static_cast<std::uint64_t>(
              fitted_ale_operator_stage_declarations.size()),
          context_.comm);
  if (minimum_fitted_ale_measurement_count !=
      maximum_fitted_ale_measurement_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Fitted-ALE operator-stage normal "
        "measurement declaration count differs across the active FE "
        "communicator.");
  }
  const bool local_fitted_ale_operator_stage_scheme_supported =
      context_.system->temporalOrder() == 1 &&
      (context_.scheme == SchemeKind::BackwardEuler ||
       context_.scheme == SchemeKind::DG0 ||
       context_.scheme == SchemeKind::GeneralizedAlpha);
  const auto [minimum_fitted_ale_scheme_supported,
              maximum_fitted_ale_scheme_supported] =
      systems::collectiveMinMaxUint64(
          local_fitted_ale_operator_stage_scheme_supported ? 1u : 0u,
          context_.comm);
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
      context_.system->dofHandler().getNumDofs();
  requireMeshNormalStageSchemeCoverage(
      maximum_fitted_ale_measurement_count,
      fitted_ale_operator_stage_scheme_supported,
      context_.scheme_name,
      context_.system->temporalOrder());
}

StepHook MeshNormalStageParticipant::hooks() const noexcept
{
  return evaluate_fitted_ale_operator_stage ? StepHook::CandidateStage
                                            : StepHook::None;
}

void MeshNormalStageParticipant::onCandidateStage(
    const CandidateStageObservation& observation)
{
  const bool scheme_matches =
      observation.scheme == context_.scheme &&
      (observation.scheme == SchemeKind::BackwardEuler ||
       observation.scheme == SchemeKind::DG0 ||
       observation.scheme == SchemeKind::GeneralizedAlpha);
  const bool generalized_alpha_shape_matches =
      observation.scheme == SchemeKind::GeneralizedAlpha
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
  if (systems::collectiveAnyTrue(!local_stage_ready, context_.comm)) {
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
      context_.state_fingerprint(*observation.state_vector);
  const auto stage_rate_revision =
      context_.state_fingerprint(*observation.rate_vector);

  std::optional<systems::SystemStateView> stage_state;
  std::optional<systems::OperatorStageMeasurementMetadata> stage_metadata;
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
    systems::OperatorStageMeasurementMetadata metadata{
        .scheme_name =
            observation.scheme == SchemeKind::BackwardEuler
                ? "BackwardEuler"
                : observation.scheme == SchemeKind::DG0
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
            systems::OperatorStageGeometryMetadata{
                .geometry_revision = revision.geometry_revision,
                .topology_revision = revision.topology_revision,
                .ownership_revision = revision.ownership_revision,
                .numbering_revision = revision.numbering_revision,
                .field_layout_revision = revision.field_layout_revision,
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
          systems::OperatorStageGeneralizedAlphaMetadata{
              .alpha_f = observation.generalized_alpha->alpha_f,
              .alpha_m = observation.generalized_alpha->alpha_m,
          };
    }
    stage_metadata.emplace(std::move(metadata));
  } catch (...) {
    local_stage_construction_failure = std::current_exception();
  }
  if (systems::collectiveAnyTrue(
          local_stage_construction_failure != nullptr, context_.comm)) {
    if (local_stage_construction_failure != nullptr) {
      std::rethrow_exception(local_stage_construction_failure);
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE "
        "communicator rank could not construct fitted-ALE "
        "operator-stage measurement metadata.");
  }
  systems::meshNormalConstraintStageMeasurements(*context_.system)
      .stageFittedALENormalOperatorStageMeasurements(
          *stage_state, std::move(*stage_metadata));
}

void MeshNormalStageParticipant::discardCandidate() noexcept
{
  systems::meshNormalConstraintStageMeasurements(*context_.system)
      .discardPendingFittedALENormalOperatorStageMeasurements();
}

void MeshNormalStageParticipant::commitAccepted(const TimeHistory& h)
{
  if (evaluate_fitted_ale_operator_stage) {
    systems::meshNormalConstraintStageMeasurements(*context_.system)
        .commitPendingFittedALENormalOperatorStageMeasurements(
            static_cast<std::uint64_t>(h.stepIndex()),
            static_cast<Real>(h.time()),
            static_cast<Real>(h.dt()));
  }
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
