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
 * @brief Step participant of the free-surface energy account: the
 *        functional and residual-work declarations of the run, the
 *        backward-Euler residual-work stage of each candidate, the initial
 *        functional baseline, the accepted-step maintenance energy account
 *        and physical-endpoint channels, the accepted functional record and
 *        the complete energy ledger with its accepted and rejected records.
 */

#include "FE/LevelSet/AcceptedFunctionalRecorder.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyAccount.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyCandidate.h"
#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyLedger.h"

#include "FE/Backends/Interfaces/GenericVector.h"
#include "FE/LevelSet/LevelSetContactStageParticipant.h"
#include "FE/Systems/FESystem.h"
#include "FE/TimeStepping/StepContext.h"
#include "FE/TimeStepping/StepController.h"
#include "FE/TimeStepping/StepParticipants.h"
#include "FE/TimeStepping/TimeHistory.h"
#include "FE/TimeStepping/TimeLoop.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

class EnergyParticipant final
    : public svmp::FE::timestepping::IStepParticipant,
      private svmp::FE::timestepping::StepContext {
public:
  struct FreeSurfaceResidualWorkStage {
    std::uint64_t step{0u};
    int attempt_index{-1};
    double time{0.0};
    double dt{0.0};
    std::uint64_t algebraic_state_revision{0u};
    std::vector<FreeSurfaceResidualWorkPairing>
        pairings{};
    FreeSurfaceResidualWorkChannels channels{};
  };
  /// Reads the free-surface functional, residual-work and exterior-boundary
  /// energy declarations and checks them on the active FE communicator
  /// (collective); allocates the residual-work vector when backward-Euler
  /// residual work is evaluated.
  explicit EnergyParticipant(const svmp::FE::timestepping::StepContext& context);

  /// CandidateStage when backward-Euler residual work is evaluated.
  [[nodiscard]] svmp::FE::timestepping::StepHook hooks() const noexcept override;

  /// Assembles and stages the residual work of a converged backward-Euler
  /// operator stage (collective).
  void onCandidateStage(
      const svmp::FE::timestepping::CandidateStageObservation& observation)
      override;

  /// Requires that no residual-work stage survived into a new attempt.
  void requireNoResidualWorkStageAtStepStart() const;

  /// Counts the energy attempts of the step of @p h.
  void beginAttempt(const svmp::FE::timestepping::TimeHistory& h);

  /// Records the initial functional baseline once, before the first solve.
  void recordInitialBaseline(svmp::FE::timestepping::TimeHistory& h);

  /// Requires one staged residual-work record of the prospective accepted
  /// step on every rank (collective).
  void requireStagedResidualWorkForAcceptance(
      const svmp::FE::timestepping::TimeHistory& h,
      std::uint64_t step_acceptance_attempt);

  /// Requires the staged residual-work record at commit-ready (collective).
  void requireStagedResidualWorkAtCommitReady(
      const svmp::FE::timestepping::TimeHistory& h,
      std::uint64_t step_acceptance_attempt);

  /// Logs and drops the residual-work stage of a discarded candidate.
  void discardCandidate();

  /// Requires the accepted step to match its staged residual-work record.
  void requireAcceptedResidualWorkStage(
      const svmp::FE::timestepping::TimeHistory& h);

  /// Records the pre-maintenance endpoint revision of the accepted step and
  /// binds the pending contact stages to it (collective).
  void bindAcceptedEndpoint(
      svmp::FE::timestepping::TimeHistory& h,
      svmp::FE::level_set::ContactStageParticipant& contact_stage);

  /// Aggregates the accepted-step maintenance energy account from the
  /// maintenance work ledger and evaluates its physical-endpoint channels.
  void recordAcceptedStepMaintenanceAccount(
      const svmp::FE::timestepping::TimeHistory& h,
      const svmp::FE::level_set::LevelSetMaintenanceWorkLedger&
          level_set_maintenance_work,
      std::size_t pending_phase_accepted_attempt_begin,
      std::size_t pending_phase_accepted_row_begin,
      std::uint64_t step_acceptance_attempt);

  /// Records the accepted functionals of @p h and publishes the complete
  /// energy record and the accepted residual work (collective).
  void publishAcceptedEnergy(
      svmp::FE::timestepping::TimeHistory& h,
      std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
          pending_contact_stages);

  /// Records a rejected energy attempt of the step after @p h.
  void recordRejectedAttempt(
      const svmp::FE::timestepping::TimeHistory& h,
      svmp::FE::timestepping::StepRejectReason reason);

  /// Forgets the pending accepted-step account.
  void clearPendingAccount() noexcept;

  FreeSurfaceEnergyLedger free_surface_energy_ledger;
  std::uint64_t free_surface_energy_transaction_sequence{0u};
  bool free_surface_energy_history_contiguous{true};
  std::uint64_t free_surface_energy_attempt{0u};
  std::optional<int> free_surface_energy_attempt_step{};

  std::span<const svmp::FE::systems::FreeSurfaceDiscreteFunctionalDeclaration>
      free_surface_declarations{};
  svmp::FE::level_set::AcceptedFunctionalCoverage
      free_surface_acceptance_coverage{};
  bool has_free_surface_functional{false};
  bool has_dynamic_contact_stage{false};
  std::span<const svmp::FE::systems::FreeSurfaceResidualWorkDeclaration>
      free_surface_residual_work_declarations{};
  const std::optional<
      svmp::FE::systems::FreeSurfaceExternalBoundaryEnergyDeclaration>&
      free_surface_boundary_energy_declaration;
  bool evaluate_free_surface_residual_work{false};
  bool complete_free_surface_energy_connector_available{false};
  std::optional<FreeSurfaceResidualWorkStage>
      pending_free_surface_residual_work_stage;
  std::unique_ptr<svmp::FE::backends::GenericVector>
      free_surface_residual_work_scratch;
  bool initial_free_surface_functional_baseline_recorded{true};

  std::uint64_t pending_pre_maintenance_endpoint_state_revision{0u};
  std::optional<svmp::FE::level_set::LevelSetMaintenanceAcceptedStepEnergyAccount>
      pending_accepted_step_maintenance_energy_account;
  std::optional<LevelSetMaintenancePhysicalEndpointChannels>
      pending_accepted_step_physical_endpoint_channels;
};

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
