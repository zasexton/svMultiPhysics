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
 * @brief Step participant of the two-fluid accepted-stage diagnostics: it
 *        checks the declarations and the scheme before the transient loop,
 *        stages the phase diagnostics and the aggregation evidence at the
 *        converged operator stage, binds the solve numerics at the final
 *        candidate gate, discards them with a discarded candidate and
 *        publishes them after acceptance.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceStepContext.h"
#include "Physics/Formulations/NavierStokes/TwoFluid/TwoFluidAcceptedStage.h"

#include "FE/Constraints/SmallCutAggregationConstraint.h"
#include "FE/Core/Types.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/NewtonSolver.h"
#include "FE/TimeStepping/StepParticipants.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"

#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::two_fluid {

class StageParticipant final
    : public svmp::FE::timestepping::IStepParticipant,
      private free_surface::FreeSurfaceStepContext {
public:
  /// Checks the two-fluid declarations and scheme support on the active FE
  /// communicator (collective); throws when they differ across ranks or the
  /// scheme is unsupported.
  explicit StageParticipant(const free_surface::FreeSurfaceStepContext& context);

  /// CandidateStage when two-fluid accepted-stage diagnostics are declared.
  [[nodiscard]] svmp::FE::timestepping::StepHook hooks() const noexcept override;

  /// Stages the phase diagnostics and the aggregation evidence of a
  /// converged operator stage.
  void onCandidateStage(
      const svmp::FE::timestepping::CandidateStageObservation& observation)
      override;

  /// Binds the solve telemetry and aggregation evidence to the pending
  /// diagnostics at the final candidate gate.
  void bindCandidateNumerics(
      const svmp::FE::timestepping::NewtonReport& newton_report);

  /// Forgets the captured aggregation evidence.
  void resetAggregationEvidence() noexcept;

  /// Drops the pending diagnostics of a discarded candidate.
  void discardCandidate() noexcept;

  /// Publishes the pending diagnostics of the accepted step @p h.
  void commitAccepted(const svmp::FE::timestepping::TimeHistory& h);

  std::span<const svmp::FE::systems::TwoFluidAcceptedStageDiagnosticDeclaration>
      two_fluid_stage_declarations{};
  bool evaluate_two_fluid_stage{false};
  bool reconcile_two_fluid_momentum{false};
  svmp::FE::GlobalIndex two_fluid_stage_state_size{0};
  std::vector<svmp::FE::constraints::SmallCutAggregationRefreshReport>
      pending_two_fluid_operator_stage_aggregation_reports;
  bool pending_two_fluid_operator_stage_aggregation_captured{false};
};

} // namespace svmp::Physics::formulations::navier_stokes::two_fluid
