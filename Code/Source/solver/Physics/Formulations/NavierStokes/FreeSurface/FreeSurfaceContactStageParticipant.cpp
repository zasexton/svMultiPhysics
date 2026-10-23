/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactStageParticipant.h"

#include "FE/Systems/SystemCollectives.h"

#include <exception>
#include <stdexcept>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using svmp::FE::systems::collectiveMaxDouble;
using svmp::FE::systems::collectiveMinDouble;
using svmp::FE::systems::collectiveMinMaxUint64;
using svmp::FE::systems::requireCollectivePhasePreparation;

ContactStageParticipant::ContactStageParticipant(
    const svmp::FE::timestepping::StepContext& context,
    bool has_dynamic_contact_stage,
    std::span<const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
        free_surface_declarations,
    svmp::FE::GlobalIndex fitted_ale_operator_stage_state_size)
    : svmp::FE::timestepping::StepContext(context),
      has_dynamic_contact_stage(has_dynamic_contact_stage),
      free_surface_declarations(free_surface_declarations),
      fitted_ale_operator_stage_state_size(
          fitted_ale_operator_stage_state_size)
{
  capture_dynamic_contact_generalized_alpha =
      has_dynamic_contact_stage &&
      opts.scheme ==
          svmp::FE::timestepping::SchemeKind::GeneralizedAlpha &&
      sim.fe_system->temporalOrder() == 1;
  if (has_dynamic_contact_stage &&
      opts.scheme ==
          svmp::FE::timestepping::SchemeKind::GeneralizedAlpha &&
      !capture_dynamic_contact_generalized_alpha) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Dynamic-contact generalized-alpha "
        "provenance currently requires a temporal-order-one FE system.");
  }
}

svmp::FE::timestepping::StepHook ContactStageParticipant::hooks()
    const noexcept
{
  return capture_dynamic_contact_generalized_alpha
             ? svmp::FE::timestepping::StepHook::CandidateStage
             : svmp::FE::timestepping::StepHook::None;
}

void ContactStageParticipant::onCandidateStage(
    const svmp::FE::timestepping::CandidateStageObservation& observation)
{
  std::optional<
      DynamicContactFirstOrderGeneralizedAlphaObservation>
      captured;
  std::exception_ptr local_capture_failure;
  try {
    if (pending_dynamic_contact_generalized_alpha_observation
            .has_value()) {
      throw std::logic_error(
          "Dynamic-contact generalized-alpha candidate-stage "
          "provenance is not one-shot.");
    }
    captured =
        makeDynamicContactFirstOrderGeneralizedAlphaObservation(
            observation);
    if (!dynamicContactStageMeshRevisionMatches(
            captured->mesh_revision,
            sim.fe_system->meshAccess())) {
      throw std::invalid_argument(
          "Dynamic-contact generalized-alpha provenance has a stale "
          "candidate-stage mesh revision.");
    }
    captured->contact_wall_boundary_fingerprint =
        dynamicContactWallBoundaryFingerprint(
            sim.fe_system->meshAccess(),
            free_surface_declarations);
    if (dynamic_contact_solve_attempt_index < 0 ||
        captured->attempt_index !=
            dynamic_contact_solve_attempt_index) {
      throw std::invalid_argument(
          "Dynamic-contact generalized-alpha provenance does not "
          "belong to the active solve/retry attempt.");
    }
    if (observation.state_vector->size() !=
            fitted_ale_operator_stage_state_size ||
        observation.rate_vector->size() !=
            fitted_ale_operator_stage_state_size) {
      throw std::invalid_argument(
          "Dynamic-contact generalized-alpha provenance has an "
          "incompatible state/rate layout.");
    }
    // Contact geometry/functionals consume only operator-stage state
    // values. Retaining the exact phi and velocity slices makes
    // algebraic fields authentic here; this path neither consumes nor
    // infers a field-rate value from time_derivative_fields.
  } catch (...) {
    local_capture_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_capture_failure,
      active_comm,
      "dynamic_contact_generalized_alpha_observation_capture");
  if (!captured.has_value()) {
    throw std::logic_error(
        "[svMultiPhysics::Application] Dynamic-contact generalized-"
        "alpha provenance remained unavailable after collective "
        "capture validation.");
  }
  captured->operator_stage_state =
      captureDynamicContactOperatorStageState(
          *sim.fe_system,
          free_surface_declarations,
          *observation.state_vector,
          active_comm);
  requireDynamicContactOperatorStageObservationConsensus(
      *captured, active_comm);
  std::exception_ptr local_storage_failure;
  try {
    pending_dynamic_contact_generalized_alpha_observation.emplace(
        std::move(*captured));
  } catch (...) {
    local_storage_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_storage_failure,
      active_comm,
      "dynamic_contact_generalized_alpha_observation_storage");
}

void ContactStageParticipant::beginAttempt(
    const svmp::FE::timestepping::TimeHistory& h)
{
  if (capture_dynamic_contact_generalized_alpha) {
    if (!dynamic_contact_solve_attempt_step.has_value() ||
        *dynamic_contact_solve_attempt_step != h.stepIndex()) {
      dynamic_contact_solve_attempt_step = h.stepIndex();
      dynamic_contact_solve_attempt_index = 0;
    } else {
      ++dynamic_contact_solve_attempt_index;
    }
    const auto [minimum_pending_stage, maximum_pending_stage] =
        collectiveMinMaxUint64(
            pending_dynamic_contact_generalized_alpha_observation.has_value()
                ? 1u
                : 0u,
            active_comm);
    if (minimum_pending_stage != maximum_pending_stage ||
        maximum_pending_stage != 0u) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Dynamic-contact generalized-alpha "
          "provenance survived into a new solve/retry attempt.");
    }
  }
}

FreeSurfaceContactStageTemporalCoordinates
ContactStageParticipant::resolveTemporalCoordinatesCollectively(
    const svmp::FE::timestepping::TimeHistory& history,
    int expected_attempt_index,
    std::string_view phase)
{
  std::optional<FreeSurfaceContactStageTemporalCoordinates> resolved;
  std::exception_ptr local_failure;
  try {
    if (pending_dynamic_contact_generalized_alpha_observation.has_value() &&
        !dynamicContactStageProvenanceMatches(
            *pending_dynamic_contact_generalized_alpha_observation,
            sim.fe_system->meshAccess(),
            free_surface_declarations)) {
      throw std::logic_error(
          "Dynamic-contact generalized-alpha candidate-stage "
          "base-mesh or contact-wall provenance is stale for the "
          "live FE mesh.");
    }
    resolved = resolveFreeSurfaceContactStageTemporalCoordinates(
        transient_scheme.scheme,
        history,
        expected_attempt_index,
        pending_dynamic_contact_generalized_alpha_observation);
  } catch (...) {
    local_failure = std::current_exception();
  }
  requireCollectivePhasePreparation(
      local_failure, active_comm, phase);
  if (!resolved.has_value()) {
    throw std::logic_error(
        "[svMultiPhysics::Application] Contact-stage temporal "
        "coordinates remained unavailable after collective validation.");
  }
  const auto [minimum_generalized_alpha_present,
              maximum_generalized_alpha_present] =
      collectiveMinMaxUint64(
          resolved->first_order_generalized_alpha.has_value() ? 1u : 0u,
          active_comm);
  bool parameters_agree =
      minimum_generalized_alpha_present ==
      maximum_generalized_alpha_present;
  const auto require_real_agreement = [&](svmp::FE::Real value) {
    parameters_agree =
        collectiveMinDouble(
            static_cast<double>(value),
            active_comm) ==
            collectiveMaxDouble(
                static_cast<double>(value),
                active_comm) &&
        parameters_agree;
  };
  require_real_agreement(resolved->stage_time);
  require_real_agreement(resolved->stage_alpha_f);
  if (parameters_agree &&
      maximum_generalized_alpha_present != 0u) {
    const auto& provenance =
        *resolved->first_order_generalized_alpha;
    require_real_agreement(provenance.alpha_m);
    require_real_agreement(provenance.alpha_f);
    require_real_agreement(provenance.gamma);
    require_real_agreement(provenance.dt);
  }
  if (!parameters_agree) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Contact-stage temporal "
        "provenance differs across the active FE communicator.");
  }
  return *resolved;
}

void ContactStageParticipant::clearPendingProvenance() noexcept
{
  pending_contact_stages.clear();
  pending_contact_stage_constraints.clear();
  pending_contact_stage_solution.clear();
  pending_contact_stage_endpoint_algebraic_revision = 0u;
}

void ContactStageParticipant::bindAcceptedStagesToEndpoint(
    std::span<const svmp::FE::Real> endpoint_solution,
    std::uint64_t pending_pre_maintenance_endpoint_state_revision)
{
  if (has_dynamic_contact_stage) {
    assertAcceptedFreeSurfaceContactStageEndpointUnchanged(
        pending_contact_stage_endpoint_algebraic_revision,
        endpoint_solution,
        active_comm);
    bindAcceptedFreeSurfaceContactStagesToEndpointRevision(
        pending_contact_stages,
        pending_pre_maintenance_endpoint_state_revision,
        pending_contact_stage_solution,
        active_comm);
  }
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
