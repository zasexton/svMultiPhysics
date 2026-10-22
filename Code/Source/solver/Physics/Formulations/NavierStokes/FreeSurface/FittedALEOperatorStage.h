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
 * @brief Step participant of the fitted-ALE operator-stage normal
 *        measurements: it checks the declarations and the scheme before the
 *        transient loop, stages the measurement at the converged operator
 *        stage, discards it with a discarded candidate and publishes it after
 *        acceptance.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceStepContext.h"

#include "FE/Core/Types.h"
#include "FE/TimeStepping/StepParticipants.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"

#include <cstdint>
#include <string_view>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

/// Throws when fitted-ALE operator-stage measurements are declared
/// (@p declaration_count > 0) and the scheme is not supported.
void requireFittedALEOperatorStageSchemeCoverage(
    std::uint64_t declaration_count,
    bool scheme_supported,
    std::string_view scheme_name,
    int temporal_order);

class FittedALEStageParticipant final
    : public svmp::FE::timestepping::IStepParticipant,
      private FreeSurfaceStepContext {
public:
  /// Checks the fitted-ALE declarations and scheme support on the active FE
  /// communicator (collective); throws when they differ across ranks or the
  /// scheme is unsupported.
  explicit FittedALEStageParticipant(const FreeSurfaceStepContext& context);

  /// CandidateStage when there are measurements to evaluate.
  [[nodiscard]] svmp::FE::timestepping::StepHook hooks() const noexcept override;

  /// Stages the operator-stage measurement of a converged candidate.
  void onCandidateStage(
      const svmp::FE::timestepping::CandidateStageObservation& observation)
      override;

  /// Drops the staged measurement of a discarded candidate.
  void discardCandidate() noexcept;

  /// Publishes the staged measurement of the accepted step @p h.
  void commitAccepted(const svmp::FE::timestepping::TimeHistory& h);

  bool evaluate_fitted_ale_operator_stage{false};
  svmp::FE::GlobalIndex fitted_ale_operator_stage_state_size{0};
};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
