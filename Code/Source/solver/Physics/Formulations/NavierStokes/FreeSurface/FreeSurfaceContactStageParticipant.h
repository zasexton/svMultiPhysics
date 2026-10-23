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
 * @brief Step participant of the accepted free-surface contact stages: the
 *        dynamic-contact generalized-alpha operator-stage observation of
 *        each solve attempt, the collective contact-stage temporal
 *        coordinates, the pending accepted contact stages of a candidate and
 *        their binding to the accepted endpoint.
 *
 * The contact-stage candidate is built by the driver
 * (buildAcceptedFreeSurfaceContactStageCandidate) inside its level-set
 * geometry transaction and stored in the pending members below.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceContactStages.h"

#include "FE/Core/Types.h"
#include "FE/LevelSet/LevelSetWallContactConstraint.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/StepContext.h"
#include "FE/TimeStepping/StepParticipants.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

class ContactStageParticipant final
    : public svmp::FE::timestepping::IStepParticipant,
      private svmp::FE::timestepping::StepContext {
public:
  /// @p has_dynamic_contact_stage: a dynamic contact law is declared on
  /// every rank; @p free_surface_declarations: the FE system's free-surface
  /// functional declarations; @p fitted_ale_operator_stage_state_size: the
  /// FE state size.  Throws when generalized-alpha contact provenance needs
  /// an FE system that is not of temporal order one.
  ContactStageParticipant(
      const svmp::FE::timestepping::StepContext& context,
      bool has_dynamic_contact_stage,
      std::span<const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
          free_surface_declarations,
      svmp::FE::GlobalIndex fitted_ale_operator_stage_state_size);

  /// CandidateStage when generalized-alpha contact provenance is captured.
  [[nodiscard]] svmp::FE::timestepping::StepHook hooks() const noexcept override;

  /// Captures the exact operator-stage observation of a generalized-alpha
  /// solve attempt (collective).
  void onCandidateStage(
      const svmp::FE::timestepping::CandidateStageObservation& observation)
      override;

  /// Counts the solve attempts of the step of @p h and requires that no
  /// observation survived the previous attempt (collective).
  void beginAttempt(const svmp::FE::timestepping::TimeHistory& h);

  /// Contact-stage temporal coordinates of @p history, validated and agreed
  /// on the active FE communicator (collective).
  [[nodiscard]] FreeSurfaceContactStageTemporalCoordinates
  resolveTemporalCoordinatesCollectively(
      const svmp::FE::timestepping::TimeHistory& history,
      int expected_attempt_index,
      std::string_view phase);

  /// Forgets the pending accepted contact stages.
  void clearPendingProvenance() noexcept;

  /// Requires the pending stages' endpoint to be @p endpoint_solution and
  /// binds them to @p pending_pre_maintenance_endpoint_state_revision
  /// (collective).
  void bindAcceptedStagesToEndpoint(
      std::span<const svmp::FE::Real> endpoint_solution,
      std::uint64_t pending_pre_maintenance_endpoint_state_revision);

  bool has_dynamic_contact_stage{false};
  std::span<const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
      free_surface_declarations{};
  svmp::FE::GlobalIndex fitted_ale_operator_stage_state_size{0};
  bool capture_dynamic_contact_generalized_alpha{false};
  std::optional<DynamicContactFirstOrderGeneralizedAlphaObservation>
      pending_dynamic_contact_generalized_alpha_observation;
  std::optional<int> dynamic_contact_solve_attempt_step;
  int dynamic_contact_solve_attempt_index{-1};
  std::vector<svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
      pending_contact_stages;
  std::vector<svmp::FE::level_set::LevelSetWallContactConstraint>
      pending_contact_stage_constraints;
  std::vector<svmp::FE::Real> pending_contact_stage_solution;
  std::uint64_t
      pending_contact_stage_endpoint_algebraic_revision{0u};
};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
