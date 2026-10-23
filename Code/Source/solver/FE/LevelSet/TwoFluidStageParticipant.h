/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#ifndef SVMP_FE_LEVELSET_TWO_FLUID_STAGE_PARTICIPANT_H
#define SVMP_FE_LEVELSET_TWO_FLUID_STAGE_PARTICIPANT_H

/**
 * @file
 * @brief Step participant of the two-fluid accepted-stage diagnostics: it
 *        checks the declarations and the scheme before the transient loop,
 *        stages the phase diagnostics and the aggregation evidence at the
 *        converged operator stage, binds the solve numerics at the final
 *        candidate gate, discards them with a discarded candidate and
 *        publishes them after acceptance.
 */

#include "LevelSet/TwoFluidAcceptedStage.h"

#include "Constraints/SmallCutAggregationConstraint.h"
#include "Core/Types.h"
#include "Systems/FESystem.h"
#include "TimeStepping/NewtonSolver.h"
#include "TimeStepping/StepContext.h"
#include "TimeStepping/StepParticipants.h"
#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeLoop.h"

#include <span>
#include <vector>

namespace svmp::FE::level_set {

class TwoFluidStageParticipant final
    : public svmp::FE::timestepping::IStepParticipant,
      private svmp::FE::timestepping::StepContext {
public:
  /// Checks the two-fluid declarations and scheme support on the active FE
  /// communicator (collective); throws when they differ across ranks or the
  /// scheme is unsupported.
  explicit TwoFluidStageParticipant(const svmp::FE::timestepping::StepContext& context);

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

} // namespace svmp::FE::level_set

#endif // SVMP_FE_LEVELSET_TWO_FLUID_STAGE_PARTICIPANT_H
