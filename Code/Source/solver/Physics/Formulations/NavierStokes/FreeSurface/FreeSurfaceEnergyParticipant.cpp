/* Copyright (c) Stanford University, The Regents of the
 * University of California, and others.
 *
 * All Rights Reserved.
 *
 * See License file.
 */

#include "Physics/Formulations/NavierStokes/FreeSurface/FreeSurfaceEnergyParticipant.h"

#include "FE/Core/RankLog.h"
#include "FE/LevelSet/LevelSetConservativePhaseWorkflow.h"
#include "FE/LevelSet/Maintenance/LevelSetMaintenanceRequestSchedule.h"
#include "FE/Systems/SystemCollectives.h"
#include "FE/Systems/SystemState.h"
#include "FE/TimeStepping/OperatorStageCapture.h"
#include "FE/TimeStepping/StepController.h"
#include "FE/Systems/FunctionalRecords.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp::Physics::formulations::navier_stokes::free_surface {

using svmp::FE::level_set::acceptedFunctionalDeclarationContentKey;
using svmp::FE::level_set::aggregateLevelSetMaintenanceAcceptedStepEnergy;
using svmp::FE::level_set::collectiveLevelSetMaintenanceAlgebraicRevision;
using svmp::FE::level_set::LevelSetMaintenanceWorkAttempt;
using svmp::FE::level_set::LevelSetMaintenanceWorkLedger;
using svmp::FE::level_set::LevelSetMaintenanceWorkRow;
using svmp::FE::level_set::logLevelSetMaintenanceAcceptedStepEnergyAccount;
using svmp::FE::level_set::mixConservativePhaseArtifactFingerprint;
using svmp::FE::level_set::mixMaintenanceRevision;
using svmp::FE::level_set::recordAcceptedFunctionals;
using svmp::FE::level_set::recordInitialAcceptedFunctionalBaseline;
using svmp::FE::level_set::requireCommunicatorConsistentAcceptedFunctionalCoverage;
using svmp::FE::rank_log::oopCout;
using svmp::FE::systems::captureFeOrderedVectorCollectively;
using svmp::FE::timestepping::dynamicContactStageMeshRevisionMatches;
using svmp::FE::systems::collectiveAnyTrue;
using svmp::FE::systems::collectiveMaxDouble;
using svmp::FE::systems::collectiveMinDouble;
using svmp::FE::systems::collectiveMinMaxUint64;
using svmp::FE::systems::requireCollectivePhasePreparation;
using svmp::FE::timestepping::stepRejectReasonName;

namespace {

// The channel words of the complete energy record, in catalogue order:
// stored energies at the three endpoints, then the dissipation rates and the
// external and numerical works.
void appendCompleteEnergyChannelValues(
    std::ostream& log,
    const svmp::FE::systems::EnergyBalanceCatalogue& catalogue,
    const svmp::FE::systems::EnergyBalanceAttempt& record)
{
  using svmp::FE::systems::EnergyChannelKind;
  const auto stored = catalogue.channels(EnergyChannelKind::Stored);
  for (std::size_t i = 0u; i < stored.size(); ++i) {
    log << " " << stored[i].name << "_energy_before="
        << record.values.stored_before[i]
        << " " << stored[i].name << "_energy_physical_endpoint="
        << record.values.stored_physical_endpoint[i]
        << " " << stored[i].name << "_energy_after="
        << record.values.stored_after[i];
  }
  log << " gas_energy_applicability=not_applicable";
  const auto rates = catalogue.channels(EnergyChannelKind::DissipationRate);
  for (std::size_t i = 0u; i < rates.size(); ++i) {
    log << " " << rates[i].name << "_dissipation_rate="
        << record.values.dissipation_rate[i];
  }
  const auto external = catalogue.channels(EnergyChannelKind::ExternalWork);
  for (std::size_t i = 0u; i < external.size(); ++i) {
    log << " external_" << external[i].name << "_work="
        << record.values.external_work[i];
  }
  const auto numerical = catalogue.channels(EnergyChannelKind::NumericalWork);
  for (std::size_t i = 0u; i < numerical.size(); ++i) {
    log << " numerical_" << numerical[i].name << "_work="
        << record.values.numerical_work[i];
  }
}

// " <kind>_<name>_applicability=... <kind>_<name>_owner=..." per channel,
// in catalogue order.
void appendCompleteEnergyChannelSources(
    std::ostream& log,
    const svmp::FE::systems::EnergyBalanceCatalogue& catalogue,
    const svmp::FE::systems::EnergyBalanceAttempt& record)
{
  for (const auto kind : svmp::FE::systems::kEnergyChannelKinds) {
    const auto channels = catalogue.channels(kind);
    const auto& sources =
        record.sources[svmp::FE::systems::energyChannelKindIndex(kind)];
    for (std::size_t i = 0u; i < channels.size(); ++i) {
      svmp::FE::systems::appendEnergyChannelSource(
          log,
          std::string(svmp::FE::systems::energyChannelKindPrefix(kind)) +
              "_" + channels[i].name,
          sources[i]);
    }
  }
}

} // namespace

EnergyParticipant::EnergyParticipant(const svmp::FE::timestepping::StepContext& context)
    : svmp::FE::timestepping::StepContext(context),
      free_surface_boundary_energy_declaration(
          svmp::FE::systems::functionalRecords(*context.sim.fe_system)
              .externalBoundaryEnergyDeclaration())
{
  free_surface_declarations =
      svmp::FE::systems::functionalRecords(*sim.fe_system).functionalDeclarations();
  free_surface_acceptance_coverage =
      requireCommunicatorConsistentAcceptedFunctionalCoverage(
          free_surface_declarations,
          active_comm);
  has_free_surface_functional =
      free_surface_acceptance_coverage.has_functional;
  has_dynamic_contact_stage =
      free_surface_acceptance_coverage.has_dynamic_contact;
  free_surface_residual_work_declarations =
      svmp::FE::systems::functionalRecords(*sim.fe_system).workChannelDeclarations();
  const auto [minimum_residual_work_declaration_count,
              maximum_residual_work_declaration_count] =
      collectiveMinMaxUint64(
          static_cast<std::uint64_t>(
              free_surface_residual_work_declarations.size()),
          active_comm);
  if (minimum_residual_work_declaration_count !=
      maximum_residual_work_declaration_count) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface residual-work declaration count differs across the active FE communicator.");
  }
  if (maximum_residual_work_declaration_count != 0u &&
      maximum_residual_work_declaration_count != 7u) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] The connected free-surface residual-work set must contain exactly seven channel declarations.");
  }
  std::uint64_t residual_work_declaration_fingerprint =
      14695981039346656037ull;
  for (const auto& declaration :
       free_surface_residual_work_declarations) {
    mixConservativePhaseArtifactFingerprint(
        residual_work_declaration_fingerprint,
        static_cast<std::uint8_t>(declaration.channel));
    mixConservativePhaseArtifactFingerprint(
        residual_work_declaration_fingerprint,
        static_cast<std::uint8_t>(declaration.applicability));
    mixConservativePhaseArtifactFingerprint(
        residual_work_declaration_fingerprint,
        declaration.operator_tag);
    mixConservativePhaseArtifactFingerprint(
        residual_work_declaration_fingerprint,
        declaration.owner_component);
  }
  const auto [minimum_residual_work_declaration_fingerprint,
              maximum_residual_work_declaration_fingerprint] =
      collectiveMinMaxUint64(
          residual_work_declaration_fingerprint,
          active_comm);
  if (minimum_residual_work_declaration_fingerprint !=
      maximum_residual_work_declaration_fingerprint) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface residual-work declarations differ across the active FE communicator.");
  }
  const auto [minimum_boundary_energy_declaration_present,
              maximum_boundary_energy_declaration_present] =
      collectiveMinMaxUint64(
          free_surface_boundary_energy_declaration.has_value()
              ? 1u
              : 0u,
          active_comm);
  if (minimum_boundary_energy_declaration_present !=
      maximum_boundary_energy_declaration_present) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface external-boundary energy declaration presence differs across the active FE communicator.");
  }
  std::uint64_t boundary_energy_declaration_fingerprint =
      14695981039346656037ull;
  if (free_surface_boundary_energy_declaration.has_value()) {
    mixConservativePhaseArtifactFingerprint(
        boundary_energy_declaration_fingerprint,
        static_cast<std::uint8_t>(
            free_surface_boundary_energy_declaration
                ->imposed_traction));
    mixConservativePhaseArtifactFingerprint(
        boundary_energy_declaration_fingerprint,
        static_cast<std::uint8_t>(
            free_surface_boundary_energy_declaration
                ->open_boundary_flux));
    mixConservativePhaseArtifactFingerprint(
        boundary_energy_declaration_fingerprint,
        free_surface_boundary_energy_declaration
            ->homogeneous_velocity_data);
    mixConservativePhaseArtifactFingerprint(
        boundary_energy_declaration_fingerprint,
        free_surface_boundary_energy_declaration
            ->pressure_dirichlet_absent);
    mixConservativePhaseArtifactFingerprint(
        boundary_energy_declaration_fingerprint,
        free_surface_boundary_energy_declaration
            ->owner_component);
  }
  const auto [minimum_boundary_energy_declaration_fingerprint,
              maximum_boundary_energy_declaration_fingerprint] =
      collectiveMinMaxUint64(
          boundary_energy_declaration_fingerprint,
          active_comm);
  if (minimum_boundary_energy_declaration_fingerprint !=
      maximum_boundary_energy_declaration_fingerprint) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface external-boundary energy declaration differs across the active FE communicator.");
  }
  evaluate_free_surface_residual_work =
      maximum_residual_work_declaration_count != 0u &&
      opts.scheme ==
          svmp::FE::timestepping::SchemeKind::BackwardEuler &&
      sim.fe_system->temporalOrder() == 1;
  complete_free_surface_energy_connector_available =
      has_free_surface_functional &&
      evaluate_free_surface_residual_work &&
      free_surface_boundary_energy_declaration.has_value();
  if (evaluate_free_surface_residual_work) {
    free_surface_residual_work_scratch =
        sim.backend->createVector(
            sim.fe_system->dofHandler().getNumDofs());
    if (!free_surface_residual_work_scratch) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Could not allocate the free-surface residual-work vector.");
    }
  } else if (maximum_residual_work_declaration_count != 0u) {
    oopCout()
        << "[svMultiPhysics::Application] Free-surface residual work"
        << " diagnostic=free_surface_residual_work_stage"
        << " status=not_evaluated"
        << " scheme=" << transient_scheme.canonical_name
        << " temporal_order=" << sim.fe_system->temporalOrder()
        << " reason=backward_euler_temporal_order_one_required"
        << std::endl;
  }
  initial_free_surface_functional_baseline_recorded =
      !has_free_surface_functional;
}

svmp::FE::timestepping::StepHook EnergyParticipant::hooks() const noexcept
{
  return evaluate_free_surface_residual_work
             ? svmp::FE::timestepping::StepHook::CandidateStage
             : svmp::FE::timestepping::StepHook::None;
}

void EnergyParticipant::onCandidateStage(
    const svmp::FE::timestepping::CandidateStageObservation& observation)
{
  const auto expected_size =
      sim.fe_system->dofHandler().getNumDofs();
  const bool local_stage_ready =
      observation.scheme ==
          svmp::FE::timestepping::SchemeKind::BackwardEuler &&
      observation.temporal_order == 1 &&
      observation.step_index > 0 &&
      observation.attempt_index >= 0 &&
      std::isfinite(observation.state_time) &&
      std::isfinite(observation.dt) &&
      observation.dt > 0.0 &&
      observation.state_time == observation.step_end_time &&
      !observation.generalized_alpha.has_value() &&
      observation.state_vector != nullptr &&
      observation.state_vector->size() == expected_size &&
      observation.mesh_revision.has_value() &&
      dynamicContactStageMeshRevisionMatches(
          *observation.mesh_revision,
          sim.fe_system->meshAccess()) &&
      free_surface_residual_work_scratch != nullptr &&
      free_surface_residual_work_scratch->size() ==
          expected_size &&
      !pending_free_surface_residual_work_stage.has_value();
  if (collectiveAnyTrue(
          !local_stage_ready,
          active_comm)) {
    if (!local_stage_ready) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface residual work requires one fresh backward-Euler operator-stage state with matching live mesh revisions.");
    }
    throw std::runtime_error(
        "[svMultiPhysics::Application] Another active FE communicator rank rejected free-surface residual-work stage readiness.");
  }

  auto& history = *sim.time_history;
  svmp::FE::systems::SystemStateView stage_state;
  stage_state.time = observation.state_time;
  stage_state.dt = observation.dt;
  stage_state.effective_dt = observation.dt;
  stage_state.dt_prev = history.dtPrev();
  stage_state.u = observation.state_vector->localSpan();
  stage_state.u_prev = history.uPrevSpan();
  stage_state.u_prev2 = history.uPrev2Span();
  stage_state.u_vector = observation.state_vector;
  stage_state.u_prev_vector = &history.uPrev();
  stage_state.u_prev2_vector = &history.uPrev2();
  stage_state.u_history = history.uHistorySpans();
  stage_state.dt_history = history.dtHistory();

  std::vector<svmp::FE::systems::ResidualWorkPairing>
      pairings;
  pairings.reserve(
      free_surface_residual_work_declarations.size());
  const auto constrained_dofs =
      sim.fe_system->constraints().getConstrainedDofs();
  for (const auto& declaration :
       free_surface_residual_work_declarations) {
    const bool produced =
        declaration.applicability ==
        svmp::FE::systems::
            FreeSurfaceResidualWorkApplicability::Produced;
    double residual_state_pairing = 0.0;
    if (produced) {
      std::exception_ptr local_assembly_failure;
      try {
        free_surface_residual_work_scratch->zero();
        auto residual_view =
            free_surface_residual_work_scratch
                ->createAssemblyView();
        if (!residual_view) {
          throw std::runtime_error(
              "Free-surface residual-work assembly could not create a vector view.");
        }
        svmp::FE::systems::AssemblyRequest request;
        request.op = declaration.operator_tag;
        request.want_vector = true;
        request.suppress_constraint_inhomogeneity = true;
        request.suppress_auxiliary_coupling_assembly = true;
        request.is_nonlinear_iteration = true;
        const auto assembled = transient.assemble(
            request,
            stage_state,
            nullptr,
            residual_view.get());
        if (!assembled.success) {
          throw std::runtime_error(
              "Free-surface residual-work operator '" +
              declaration.operator_tag +
              "' failed assembly: " +
              assembled.error_message);
        }
        residual_view =
            free_surface_residual_work_scratch
                ->createAssemblyView();
        if (!residual_view) {
          throw std::runtime_error(
              "Free-surface residual-work constraint projection could not create a vector view.");
        }
        residual_view->beginAssemblyPhase();
        if (!constrained_dofs.empty()) {
          residual_view->zeroVectorEntries(
              constrained_dofs);
        }
        residual_view->finalizeAssembly();
      } catch (...) {
        local_assembly_failure =
            std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_assembly_failure,
          active_comm,
          "free_surface_residual_work_operator_assembly");

      std::exception_ptr local_pairing_failure;
      try {
        residual_state_pairing = static_cast<double>(
            free_surface_residual_work_scratch->dot(
                *observation.state_vector));
        if (!std::isfinite(residual_state_pairing)) {
          throw std::overflow_error(
              "Free-surface residual/state pairing is not finite.");
        }
      } catch (...) {
        local_pairing_failure =
            std::current_exception();
      }
      requireCollectivePhasePreparation(
          local_pairing_failure,
          active_comm,
          "free_surface_residual_work_operator_pairing");
      if (collectiveMinDouble(
              residual_state_pairing,
              active_comm) !=
          collectiveMaxDouble(
              residual_state_pairing,
              active_comm)) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Free-surface residual/state pairing differs across the active FE communicator.");
      }
    }
    pairings.push_back(
        svmp::FE::systems::ResidualWorkPairing{
                .channel = svmp::FE::systems::
                    freeSurfaceResidualWorkChannelName(
                        declaration.channel),
                .produced = produced,
                .owner_component =
                    declaration.owner_component,
                .operator_tag = declaration.operator_tag,
                .residual_state_pairing =
                    residual_state_pairing,
            });
  }

  const auto candidate_solution =
      captureFeOrderedVectorCollectively(
          stage_state,
          active_comm);
  const auto live_solution = captureFeOrderedVectorCollectively(
      history.u(), active_comm);
  const auto candidate_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          candidate_solution,
          active_comm);
  const auto live_state_revision =
      collectiveLevelSetMaintenanceAlgebraicRevision(
          live_solution,
          active_comm);
  if (candidate_state_revision == 0u ||
      candidate_state_revision != live_state_revision) {
    throw std::runtime_error(
        "[svMultiPhysics::Application] Free-surface residual work was not evaluated on the live converged candidate state.");
  }
  auto channels = evaluateFreeSurfaceResidualWorkChannels(
          pairings, observation.dt);
  pending_free_surface_residual_work_stage.emplace(
      FreeSurfaceResidualWorkStage{
          .step = static_cast<std::uint64_t>(
              observation.step_index),
          .attempt_index = observation.attempt_index,
          .time = observation.state_time,
          .dt = observation.dt,
          .algebraic_state_revision =
              candidate_state_revision,
          .pairings = std::move(pairings),
          .channels = std::move(channels),
      });
  const auto& staged =
      *pending_free_surface_residual_work_stage;
  oopCout()
      << std::setprecision(17)
      << "[svMultiPhysics::Application] Free-surface residual work"
      << " diagnostic=free_surface_residual_work_stage"
      << " status=staged"
      << " step=" << staged.step
      << " attempt_index=" << staged.attempt_index
      << " time=" << staged.time
      << " dt=" << staged.dt
      << " algebraic_state_revision="
      << staged.algebraic_state_revision
      << " convection_work="
      << *staged.channels.convection
      << " pressure_continuity_work="
      << *staged.channels.pressure_continuity
      << " nonconservative_body_force_work="
      << *staged.channels.nonconservative_body_force
      << " weak_boundary_work="
      << *staged.channels.weak_boundary
      << " vms_pspg_work="
      << *staged.channels.vms_pspg
      << " cut_stabilization_work="
      << *staged.channels.cut_stabilization
      << " ghost_penalty_work="
      << *staged.channels.ghost_penalty
      << " accepted_state_contribution=staged_not_published"
      << std::endl;
}

void EnergyParticipant::requireNoResidualWorkStageAtStepStart() const
{
  if (evaluate_free_surface_residual_work &&
      pending_free_surface_residual_work_stage.has_value()) {
    throw std::logic_error(
        "[svMultiPhysics::Application] A free-surface residual-work stage survived into a new solve/retry attempt.");
  }
}

void EnergyParticipant::beginAttempt(
    const svmp::FE::timestepping::TimeHistory& h)
{
  if (!free_surface_energy_attempt_step.has_value() ||
      *free_surface_energy_attempt_step != h.stepIndex()) {
    free_surface_energy_attempt_step = h.stepIndex();
    free_surface_energy_attempt = 1u;
  } else {
    if (free_surface_energy_attempt ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error(
          "[svMultiPhysics::Application] Free-surface energy attempt index overflowed.");
    }
    ++free_surface_energy_attempt;
  }
}

void EnergyParticipant::recordInitialBaseline(
    svmp::FE::timestepping::TimeHistory& h)
{
  if (!initial_free_surface_functional_baseline_recorded) {
    if (h.stepIndex() < 0) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] An initial free-surface functional baseline requires a nonnegative step index.");
    }
    const auto initial_solution =
        captureFeOrderedVectorCollectively(
            h.u(), active_comm);
    recordInitialAcceptedFunctionalBaseline(
        sim,
        static_cast<std::uint64_t>(h.stepIndex()),
        static_cast<svmp::FE::Real>(h.time()),
        std::span<const svmp::FE::Real>(
            initial_solution.data(), initial_solution.size()),
        transient_scheme.scheme ==
            svmp::FE::timestepping::SchemeKind::BackwardEuler,
        current_velocity_extension_revision());
    initial_free_surface_functional_baseline_recorded = true;
  }
}

void EnergyParticipant::requireStagedResidualWorkForAcceptance(
    const svmp::FE::timestepping::TimeHistory& h,
    std::uint64_t step_acceptance_attempt)
{
  if (evaluate_free_surface_residual_work) {
    const auto [minimum_stage_present,
                maximum_stage_present] =
        collectiveMinMaxUint64(
            pending_free_surface_residual_work_stage
                    .has_value()
                ? 1u
                : 0u,
            active_comm);
    if (minimum_stage_present != 1u ||
        maximum_stage_present != 1u) {
      throw std::runtime_error(
          "[svMultiPhysics::Application] Free-surface acceptance requires one staged residual-work record on every active FE rank.");
    }
    const auto& residual_work_stage =
        *pending_free_surface_residual_work_stage;
    if (residual_work_stage.step !=
            static_cast<std::uint64_t>(
                h.stepIndex() + 1) ||
        residual_work_stage.attempt_index < 0 ||
        static_cast<std::uint64_t>(
            residual_work_stage.attempt_index + 1) !=
            step_acceptance_attempt ||
        residual_work_stage.time != h.time() + h.dt() ||
        residual_work_stage.dt != h.dt() ||
        residual_work_stage.algebraic_state_revision == 0u) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Staged free-surface residual work does not match the prospective accepted-step transaction.");
    }
  }
}

void EnergyParticipant::requireStagedResidualWorkAtCommitReady(
    const svmp::FE::timestepping::TimeHistory& h,
    std::uint64_t step_acceptance_attempt)
{
  if (evaluate_free_surface_residual_work) {
    const auto [minimum_stage_present,
                maximum_stage_present] =
        collectiveMinMaxUint64(
            pending_free_surface_residual_work_stage
                    .has_value()
                ? 1u
                : 0u,
            active_comm);
    if (minimum_stage_present != 1u ||
        maximum_stage_present != 1u ||
        pending_free_surface_residual_work_stage->step !=
            static_cast<std::uint64_t>(
                h.stepIndex() + 1) ||
        static_cast<std::uint64_t>(
            pending_free_surface_residual_work_stage
                    ->attempt_index +
                1) != step_acceptance_attempt) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Free-surface residual work reached commit-ready without one matching staged record on every active FE rank.");
    }
  }
}

void EnergyParticipant::discardCandidate()
{
  if (pending_free_surface_residual_work_stage.has_value()) {
    oopCout()
        << std::setprecision(17)
        << "[svMultiPhysics::Application] Free-surface residual work"
        << " diagnostic=free_surface_residual_work_stage"
        << " status=discarded"
        << " step="
        << pending_free_surface_residual_work_stage->step
        << " attempt_index="
        << pending_free_surface_residual_work_stage
               ->attempt_index
        << " accepted_state_convection_work=0"
        << " accepted_state_pressure_continuity_work=0"
        << " accepted_state_nonconservative_body_force_work=0"
        << " accepted_state_weak_boundary_work=0"
        << " accepted_state_vms_pspg_work=0"
        << " accepted_state_cut_stabilization_work=0"
        << " accepted_state_ghost_penalty_work=0"
        << std::endl;
    pending_free_surface_residual_work_stage.reset();
  }
}

void EnergyParticipant::requireAcceptedResidualWorkStage(
    const svmp::FE::timestepping::TimeHistory& h)
{
  if (evaluate_free_surface_residual_work) {
    const auto [minimum_stage_present,
                maximum_stage_present] =
        collectiveMinMaxUint64(
            pending_free_surface_residual_work_stage.has_value()
                ? 1u
                : 0u,
            active_comm);
    if (minimum_stage_present != 1u ||
        maximum_stage_present != 1u ||
        pending_free_surface_residual_work_stage->step !=
            static_cast<std::uint64_t>(h.stepIndex()) ||
        pending_free_surface_residual_work_stage->time !=
            h.time() ||
        pending_free_surface_residual_work_stage->dt != h.dt()) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Accepted free-surface step does not match its staged residual-work record.");
    }
  }
}

void EnergyParticipant::bindAcceptedEndpoint(
    svmp::FE::timestepping::TimeHistory& h,
    svmp::FE::level_set::ContactStageParticipant& contact_stage)
{
  if (has_free_surface_functional) {
    const auto endpoint_solution =
        captureFeOrderedVectorCollectively(
            h.u(), active_comm);
    pending_pre_maintenance_endpoint_state_revision =
        collectiveLevelSetMaintenanceAlgebraicRevision(
            endpoint_solution,
            active_comm);
    contact_stage.bindAcceptedStagesToEndpoint(
        endpoint_solution, pending_pre_maintenance_endpoint_state_revision);
  }
}

void EnergyParticipant::recordAcceptedStepMaintenanceAccount(
    const svmp::FE::timestepping::TimeHistory& h,
    const LevelSetMaintenanceWorkLedger& level_set_maintenance_work,
    std::size_t pending_phase_accepted_attempt_begin,
    std::size_t pending_phase_accepted_row_begin,
    std::uint64_t step_acceptance_attempt)
{
  if (has_free_surface_functional) {
    const auto& accepted_step_maintenance_attempts =
        level_set_maintenance_work.acceptedAttempts();
    if (pending_phase_accepted_attempt_begin >
        accepted_step_maintenance_attempts.size()) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Accepted-step maintenance account start is outside the accepted-attempt history.");
    }
    const auto& accepted_step_maintenance_rows =
        level_set_maintenance_work.acceptedRows();
    if (pending_phase_accepted_row_begin >
        accepted_step_maintenance_rows.size()) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Accepted-step maintenance account start is outside the accepted-row history.");
    }
    pending_accepted_step_maintenance_energy_account =
        aggregateLevelSetMaintenanceAcceptedStepEnergy(
                std::span<const LevelSetMaintenanceWorkAttempt>(
                    accepted_step_maintenance_attempts)
                    .subspan(
                        pending_phase_accepted_attempt_begin),
                std::span<const LevelSetMaintenanceWorkRow>(
                    accepted_step_maintenance_rows)
                    .subspan(pending_phase_accepted_row_begin));
    if (pending_accepted_step_maintenance_energy_account
            .has_value()) {
      const auto& account =
          *pending_accepted_step_maintenance_energy_account;
      if (account.step !=
              static_cast<std::uint64_t>(h.stepIndex()) ||
          account.attempt != std::max<std::uint64_t>(
                                 1u, step_acceptance_attempt) ||
          account.time != h.time() || account.dt != h.dt()) {
        throw std::logic_error(
            "[svMultiPhysics::Application] Accepted-step maintenance energy account does not match the accepted time-step transaction.");
      }
      logLevelSetMaintenanceAcceptedStepEnergyAccount(
          account);
      if (account.post_transport.has_value()) {
        const auto history =
            svmp::FE::systems::functionalRecords(*sim.fe_system)
                .acceptedFunctionalHistory();
        const auto declaration_count =
            free_surface_declarations.size();
        if (declaration_count == 0u ||
            history.size() < declaration_count ||
            h.stepIndex() <= 0) {
          throw std::logic_error(
              "[svMultiPhysics::Application] Physical maintenance endpoint channels require a complete preceding accepted functional group.");
        }
        const auto preceding_group =
            history.last(declaration_count);
        const auto preceding_step =
            preceding_group.front().accepted_step;
        const auto preceding_time =
            static_cast<double>(
                preceding_group.front().accepted_time);
        const auto preceding_state_revision =
            preceding_group.front().state_revision;
        const auto preceding_extension_map_revision =
            preceding_group.front().extension_map_revision;
        const auto expected_preceding_step =
            static_cast<std::uint64_t>(h.stepIndex() - 1);
        const double expected_preceding_time =
            account.time - account.dt;
        const double preceding_time_scale = std::max(
            {1.0,
             std::abs(preceding_time),
             std::abs(expected_preceding_time)});
        if (preceding_step != expected_preceding_step ||
            preceding_state_revision == 0u ||
            !std::isfinite(preceding_time) ||
            !std::isfinite(expected_preceding_time) ||
            std::abs(
                preceding_time - expected_preceding_time) >
                64.0 *
                    std::numeric_limits<double>::epsilon() *
                    preceding_time_scale) {
          throw std::logic_error(
              "[svMultiPhysics::Application] The preceding accepted functional endpoint does not match the physical maintenance interval.");
        }
        double preceding_surface_wall_energy = 0.0;
        std::optional<double>
            preceding_gravitational_energy;
        std::optional<int>
            active_volume_owner_interface_marker;
        std::uint64_t preceding_snapshot_set_revision =
            1469598103934665603ull;
        std::uint64_t
            preceding_mesh_topology_set_revision =
                1469598103934665603ull;
        std::uint64_t preceding_cut_topology_set_revision =
            1469598103934665603ull;
        for (std::size_t index = 0u;
             index < declaration_count; ++index) {
          const auto& record = preceding_group[index];
          const auto& declaration =
              free_surface_declarations[index];
          if (record.accepted_step != preceding_step ||
              record.accepted_time !=
                  preceding_group.front().accepted_time ||
              record.state_revision !=
                  preceding_state_revision ||
              record.extension_map_revision !=
                  preceding_extension_map_revision ||
              record.declaration.interface_marker !=
                  declaration.interface_marker ||
              acceptedFunctionalDeclarationContentKey(
                  record.declaration) !=
                  acceptedFunctionalDeclarationContentKey(
                  declaration) ||
              !std::isfinite(
                  record.state.liquid_gas_surface_energy) ||
              !std::isfinite(record.state.young_wall_energy) ||
              record.geometry_revision.snapshot_revision_key ==
                  0u ||
              record.cut_topology_revision == 0u ||
              record.active_volume_energy.has_value() !=
                  declaration.active_volume_energy_parameters
                      .has_value()) {
            throw std::logic_error(
                "[svMultiPhysics::Application] The preceding accepted functional group has inconsistent ownership, order, provenance, or energy values.");
          }
          preceding_surface_wall_energy +=
              static_cast<double>(
                  record.state.liquid_gas_surface_energy) +
              static_cast<double>(
                  record.state.young_wall_energy);
          mixMaintenanceRevision(
              preceding_snapshot_set_revision,
              static_cast<std::uint64_t>(
                  record.declaration.interface_marker));
          mixMaintenanceRevision(
              preceding_snapshot_set_revision,
              record.geometry_revision.snapshot_revision_key);
          mixMaintenanceRevision(
              preceding_mesh_topology_set_revision,
              static_cast<std::uint64_t>(
                  record.declaration.interface_marker));
          mixMaintenanceRevision(
              preceding_mesh_topology_set_revision,
              record.geometry_revision.mesh_topology_revision);
          mixMaintenanceRevision(
              preceding_cut_topology_set_revision,
              static_cast<std::uint64_t>(
                  record.declaration.interface_marker));
          mixMaintenanceRevision(
              preceding_cut_topology_set_revision,
              record.cut_topology_revision);
          if (record.active_volume_energy.has_value()) {
            if (preceding_gravitational_energy.has_value() ||
                !std::isfinite(
                    record.active_volume_energy
                        ->gravitational_energy)) {
              throw std::logic_error(
                  "[svMultiPhysics::Application] Physical maintenance endpoint channels require at most one finite active-volume energy owner.");
            }
            preceding_gravitational_energy =
                static_cast<double>(
                    record.active_volume_energy
                        ->gravitational_energy);
            active_volume_owner_interface_marker =
                record.declaration.interface_marker;
          }
        }
        if (!std::isfinite(
                preceding_surface_wall_energy)) {
          throw std::overflow_error(
              "[svMultiPhysics::Application] The preceding accepted surface and wall energy sum is not finite.");
        }
        preceding_snapshot_set_revision =
            preceding_snapshot_set_revision == 0u
                ? 1u
                : preceding_snapshot_set_revision;
        preceding_mesh_topology_set_revision =
            preceding_mesh_topology_set_revision == 0u
                ? 1u
                : preceding_mesh_topology_set_revision;
        preceding_cut_topology_set_revision =
            preceding_cut_topology_set_revision == 0u
                ? 1u
                : preceding_cut_topology_set_revision;
        pending_accepted_step_physical_endpoint_channels =
            evaluateLevelSetMaintenancePhysicalEndpointChannels(
                    account,
                    preceding_gravitational_energy,
                    preceding_surface_wall_energy);
        if (evaluate_free_surface_residual_work) {
          if (!pending_free_surface_residual_work_stage
                   .has_value() ||
              !account.maintenance_start.has_value()) {
            throw std::logic_error(
                "[svMultiPhysics::Application] Accepted residual work requires the staged operator endpoint and maintenance-chain start.");
          }
          const auto& residual_work_stage =
              *pending_free_surface_residual_work_stage;
          if (residual_work_stage.step != account.step ||
              residual_work_stage.attempt_index < 0 ||
              static_cast<std::uint64_t>(
                  residual_work_stage.attempt_index + 1) !=
                  account.attempt ||
              residual_work_stage.time != account.time ||
              residual_work_stage.dt != account.dt ||
              residual_work_stage.algebraic_state_revision !=
                  account.maintenance_start
                      ->algebraic_state_revision) {
            throw std::logic_error(
                "[svMultiPhysics::Application] Accepted residual-work stage does not match the physical maintenance-chain start.");
          }
          auto& physical_channels =
              *pending_accepted_step_physical_endpoint_channels;
          physical_channels.convection_work =
              residual_work_stage.channels.convection;
          physical_channels.pressure_continuity_work =
              residual_work_stage.channels.pressure_continuity;
          physical_channels.nonconservative_body_force_work =
              residual_work_stage.channels
                  .nonconservative_body_force;
          physical_channels.weak_boundary_work =
              residual_work_stage.channels.weak_boundary;
          physical_channels.vms_pspg_work =
              residual_work_stage.channels.vms_pspg;
          physical_channels.cut_stabilization_work =
              residual_work_stage.channels.cut_stabilization;
          physical_channels.ghost_penalty_work =
              residual_work_stage.channels.ghost_penalty;
        }
        logLevelSetMaintenancePhysicalEndpointChannels(
            account,
            *pending_accepted_step_physical_endpoint_channels,
            preceding_step,
            preceding_time,
            preceding_state_revision,
            preceding_snapshot_set_revision,
            preceding_mesh_topology_set_revision,
            preceding_cut_topology_set_revision,
            preceding_extension_map_revision,
            declaration_count,
            preceding_surface_wall_energy,
            preceding_gravitational_energy,
            active_volume_owner_interface_marker);
      }
    }
  }
}

void EnergyParticipant::publishAcceptedEnergy(
    svmp::FE::timestepping::TimeHistory& h,
    std::span<const svmp::FE::systems::FreeSurfaceAcceptedContactStageState>
        pending_contact_stages)
{
  bool accepted_complete_energy_record_connected = false;
  if (has_free_surface_functional) {
    const auto accepted_functional_extension_map_revision =
        current_velocity_extension_revision();
    const auto accepted_state_solution =
        captureFeOrderedVectorCollectively(
            h.u(), active_comm);
    std::vector<svmp::FE::Real>
        backward_euler_previous_solution;
    if (transient_scheme.scheme ==
        svmp::FE::timestepping::SchemeKind::BackwardEuler) {
      if (h.historyDepth() < 2) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Backward-Euler free-surface kinetic work requires two accepted-history slots.");
      }
      backward_euler_previous_solution =
          captureFeOrderedVectorCollectively(
              h.uPrev2(), active_comm);
    }
    const auto accepted_state_revision =
        collectiveLevelSetMaintenanceAlgebraicRevision(
            accepted_state_solution,
            active_comm);
    if (pending_accepted_step_maintenance_energy_account
            .has_value()) {
      const auto& maintenance_account =
          *pending_accepted_step_maintenance_energy_account;
      const bool maintenance_endpoint_available =
          maintenance_account.maintenance_end.has_value();
      const auto maintenance_endpoint_revision =
          maintenance_endpoint_available
              ? maintenance_account.maintenance_end
                    ->algebraic_state_revision
              : 0u;
      const bool accepted_endpoint_matches_maintenance_end =
          maintenance_endpoint_available &&
          maintenance_endpoint_revision ==
              accepted_state_revision;
      oopCout()
          << std::setprecision(17)
          << "[svMultiPhysics::Application] Level-set maintenance-to-accepted endpoint connection"
          << " diagnostic=level_set_maintenance_accepted_endpoint_connection"
          << " step=" << h.stepIndex()
          << " attempt=" << maintenance_account.attempt
          << " maintenance_endpoint_available="
          << (maintenance_endpoint_available ? "true" : "false")
          << " maintenance_endpoint_algebraic_state_revision="
          << maintenance_endpoint_revision
          << " accepted_endpoint_algebraic_state_revision="
          << accepted_state_revision
          << " accepted_endpoint_matches_maintenance_end="
          << (accepted_endpoint_matches_maintenance_end
                  ? "true"
                  : "false")
          << " accepted_endpoint_match_scope=algebraic_state_revision_only"
          << " subsequent_state_change_requires_separate_work="
          << (maintenance_endpoint_available &&
                      !accepted_endpoint_matches_maintenance_end
                  ? "true"
                  : "false")
          << " complete_energy_record_connected=false"
          << std::endl;
    }
    bool backward_euler_kinetic_work_bound = true;
    const auto accepted_functional_states =
        recordAcceptedFunctionals(
            sim,
            static_cast<std::uint64_t>(h.stepIndex()),
            h.time(),
            h.dt(),
            pending_pre_maintenance_endpoint_state_revision,
            accepted_state_revision,
            pending_contact_stages,
            std::span<const svmp::FE::Real>(
                accepted_state_solution.data(),
                accepted_state_solution.size()),
            std::span<const svmp::FE::Real>(
                backward_euler_previous_solution.data(),
                backward_euler_previous_solution.size()),
            accepted_functional_extension_map_revision,
            &backward_euler_kinetic_work_bound);
    if (!backward_euler_kinetic_work_bound) {
      // The complete energy ledger cannot telescope across an unbound
      // kinetic-work record; it reports every later record as lacking a
      // complete preceding history.
      free_surface_energy_history_contiguous = false;
    }
    if (pending_accepted_step_maintenance_energy_account
            .has_value() &&
        pending_accepted_step_maintenance_energy_account
            ->maintenance_end.has_value()) {
      const auto declaration_count =
          free_surface_declarations.size();
      if (declaration_count == 0u ||
          accepted_functional_states.size() !=
              declaration_count) {
        throw std::logic_error(
            "[svMultiPhysics::Application] The accepted maintenance endpoint comparison requires a complete current functional group.");
      }
      const auto& maintenance_endpoint =
          *pending_accepted_step_maintenance_energy_account
               ->maintenance_end;
      double current_liquid_gas_surface_energy = 0.0;
      double current_solid_liquid_wall_energy = 0.0;
      std::optional<double> current_kinetic_energy;
      std::optional<double> current_gravitational_energy;
      std::uint64_t current_snapshot_set_revision =
          1469598103934665603ull;
      std::uint64_t current_mesh_topology_set_revision =
          1469598103934665603ull;
      std::uint64_t current_cut_topology_set_revision =
          1469598103934665603ull;
      for (std::size_t index = 0u;
           index < declaration_count; ++index) {
        const auto& state =
            accepted_functional_states[index];
        const auto& declaration =
            free_surface_declarations[index];
        if (state.interface_marker !=
                declaration.interface_marker ||
            !std::isfinite(
                state.state.liquid_gas_surface_energy) ||
            !std::isfinite(state.state.young_wall_energy) ||
            state.cut_topology_revision == 0u ||
            state.active_volume_energy.has_value() !=
                declaration.active_volume_energy_parameters
                    .has_value()) {
          throw std::logic_error(
              "[svMultiPhysics::Application] The current accepted functional group cannot be paired with the maintenance endpoint.");
        }
        current_liquid_gas_surface_energy +=
            static_cast<double>(
                state.state.liquid_gas_surface_energy);
        current_solid_liquid_wall_energy +=
            static_cast<double>(
                state.state.young_wall_energy);
        mixMaintenanceRevision(
            current_snapshot_set_revision,
            static_cast<std::uint64_t>(
                state.interface_marker));
        mixMaintenanceRevision(
            current_snapshot_set_revision,
            state.geometry_revision.snapshot_revision_key);
        mixMaintenanceRevision(
            current_mesh_topology_set_revision,
            static_cast<std::uint64_t>(
                state.interface_marker));
        mixMaintenanceRevision(
            current_mesh_topology_set_revision,
            state.geometry_revision.mesh_topology_revision);
        mixMaintenanceRevision(
            current_cut_topology_set_revision,
            static_cast<std::uint64_t>(
                state.interface_marker));
        mixMaintenanceRevision(
            current_cut_topology_set_revision,
            state.cut_topology_revision);
        if (state.active_volume_energy.has_value()) {
          if (current_kinetic_energy.has_value() ||
              !std::isfinite(
                  state.active_volume_energy->kinetic_energy) ||
              !std::isfinite(
                  state.active_volume_energy
                      ->gravitational_energy)) {
            throw std::logic_error(
                "[svMultiPhysics::Application] The current accepted endpoint requires at most one finite active-volume energy owner.");
          }
          current_kinetic_energy =
              static_cast<double>(
                  state.active_volume_energy->kinetic_energy);
          current_gravitational_energy =
              static_cast<double>(
                  state.active_volume_energy
                      ->gravitational_energy);
        }
      }
      if (!std::isfinite(
              current_liquid_gas_surface_energy) ||
          !std::isfinite(
              current_solid_liquid_wall_energy) ||
          current_kinetic_energy.has_value() !=
              maintenance_endpoint.kinetic_energy.has_value() ||
          current_gravitational_energy.has_value() !=
              maintenance_endpoint.gravitational_energy
                  .has_value() ||
          current_kinetic_energy.has_value() !=
              maintenance_endpoint.modeled_stored_energy
                  .has_value()) {
        throw std::logic_error(
            "[svMultiPhysics::Application] The current accepted endpoint and maintenance endpoint have inconsistent stored-energy coverage.");
      }
      current_snapshot_set_revision =
          current_snapshot_set_revision == 0u
              ? 1u
              : current_snapshot_set_revision;
      current_mesh_topology_set_revision =
          current_mesh_topology_set_revision == 0u
              ? 1u
              : current_mesh_topology_set_revision;
      current_cut_topology_set_revision =
          current_cut_topology_set_revision == 0u
              ? 1u
              : current_cut_topology_set_revision;
      const auto current_extension_map_revision =
          accepted_functional_extension_map_revision;
      const double kinetic_energy_change =
          current_kinetic_energy.has_value()
              ? *current_kinetic_energy -
                    *maintenance_endpoint.kinetic_energy
              : 0.0;
      const double gravitational_energy_change =
          current_gravitational_energy.has_value()
              ? *current_gravitational_energy -
                    *maintenance_endpoint.gravitational_energy
              : 0.0;
      const double liquid_gas_surface_energy_change =
          current_liquid_gas_surface_energy -
          maintenance_endpoint.liquid_gas_surface_energy;
      const double solid_liquid_wall_energy_change =
          current_solid_liquid_wall_energy -
          maintenance_endpoint.solid_liquid_wall_energy;
      const bool modeled_energy_available =
          current_kinetic_energy.has_value() &&
          maintenance_endpoint.modeled_stored_energy.has_value();
      const double current_modeled_stored_energy =
          modeled_energy_available
              ? *current_kinetic_energy +
                    *current_gravitational_energy +
                    current_liquid_gas_surface_energy +
                    current_solid_liquid_wall_energy
              : 0.0;
      const double modeled_stored_energy_change =
          modeled_energy_available
              ? current_modeled_stored_energy -
                    *maintenance_endpoint.modeled_stored_energy
              : 0.0;
      const std::array<double, 5> endpoint_energy_changes{{
          kinetic_energy_change,
          gravitational_energy_change,
          liquid_gas_surface_energy_change,
          solid_liquid_wall_energy_change,
          modeled_stored_energy_change,
      }};
      if (!std::all_of(
              endpoint_energy_changes.begin(),
              endpoint_energy_changes.end(),
              [](double value) {
                return std::isfinite(value);
              })) {
        throw std::overflow_error(
            "[svMultiPhysics::Application] The maintenance-to-current endpoint energy change is not finite.");
      }
      const bool algebraic_revision_matches =
          maintenance_endpoint.algebraic_state_revision ==
          accepted_state_revision;
      const bool snapshot_revision_matches =
          maintenance_endpoint.snapshot_set_revision ==
          current_snapshot_set_revision;
      const bool mesh_topology_revision_matches =
          maintenance_endpoint.mesh_topology_set_revision ==
          current_mesh_topology_set_revision;
      const bool cut_topology_revision_matches =
          maintenance_endpoint.cut_topology_set_revision ==
          current_cut_topology_set_revision;
      const bool extension_revision_matches =
          maintenance_endpoint.extension_map_revision ==
          current_extension_map_revision;
      const bool stored_energy_endpoint_match_available =
          modeled_energy_available;
      const bool component_energy_matches =
          stored_energy_endpoint_match_available &&
          kinetic_energy_change == 0.0 &&
          gravitational_energy_change == 0.0 &&
          liquid_gas_surface_energy_change == 0.0 &&
          solid_liquid_wall_energy_change == 0.0;
      const bool stored_energy_endpoint_match =
          stored_energy_endpoint_match_available &&
          algebraic_revision_matches &&
          snapshot_revision_matches &&
          mesh_topology_revision_matches &&
          cut_topology_revision_matches &&
          extension_revision_matches &&
          component_energy_matches;
      const bool unclassified_post_maintenance_work_required =
          !stored_energy_endpoint_match;
      oopCout()
          << std::setprecision(17)
          << "[svMultiPhysics::Application] Level-set maintenance-to-current accepted functional endpoint"
          << " diagnostic=level_set_maintenance_current_functional_endpoint"
          << " step=" << h.stepIndex()
          << " attempt="
          << pending_accepted_step_maintenance_energy_account
                 ->attempt
          << " maintenance_algebraic_state_revision="
          << maintenance_endpoint.algebraic_state_revision
          << " current_algebraic_state_revision="
          << accepted_state_revision
          << " algebraic_revision_matches="
          << (algebraic_revision_matches ? "true" : "false")
          << " maintenance_snapshot_set_revision="
          << maintenance_endpoint.snapshot_set_revision
          << " current_snapshot_set_revision="
          << current_snapshot_set_revision
          << " snapshot_revision_matches="
          << (snapshot_revision_matches ? "true" : "false")
          << " maintenance_mesh_topology_set_revision="
          << maintenance_endpoint.mesh_topology_set_revision
          << " current_mesh_topology_set_revision="
          << current_mesh_topology_set_revision
          << " mesh_topology_revision_matches="
          << (mesh_topology_revision_matches
                  ? "true"
                  : "false")
          << " maintenance_cut_topology_set_revision="
          << maintenance_endpoint.cut_topology_set_revision
          << " current_cut_topology_set_revision="
          << current_cut_topology_set_revision
          << " cut_topology_revision_matches="
          << (cut_topology_revision_matches
                  ? "true"
                  : "false")
          << " cut_topology_source=accepted_functional_state"
          << " maintenance_extension_map_revision_available="
          << (maintenance_endpoint.extension_map_revision
                      .has_value()
                  ? "true"
                  : "false")
          << " current_extension_map_revision_available="
          << (current_extension_map_revision.has_value()
                  ? "true"
                  : "false")
          << " maintenance_extension_map_revision="
          << maintenance_endpoint.extension_map_revision
                 .value_or(0u)
          << " current_extension_map_revision="
          << current_extension_map_revision.value_or(0u)
          << " extension_revision_matches="
          << (extension_revision_matches ? "true" : "false")
          << " kinetic_energy_change_available="
          << (current_kinetic_energy.has_value()
                  ? "true"
                  : "false")
          << " kinetic_energy_change="
          << kinetic_energy_change
          << " gravitational_energy_change_available="
          << (current_gravitational_energy.has_value()
                  ? "true"
                  : "false")
          << " gravitational_energy_change="
          << gravitational_energy_change
          << " liquid_gas_surface_energy_change="
          << liquid_gas_surface_energy_change
          << " solid_liquid_wall_energy_change="
          << solid_liquid_wall_energy_change
          << " modeled_stored_energy_change_available="
          << (modeled_energy_available ? "true" : "false")
          << " modeled_stored_energy_change="
          << modeled_stored_energy_change
          << " unclassified_post_maintenance_work_required="
          << (unclassified_post_maintenance_work_required
                  ? "true"
                  : "false")
          << " stored_energy_endpoint_match_available="
          << (stored_energy_endpoint_match_available
                  ? "true"
                  : "false")
          << " stored_energy_endpoint_match="
          << (stored_energy_endpoint_match ? "true" : "false")
          << " complete_energy_record_connected=false"
          << std::endl;
    }

    FreeSurfaceCompleteEnergyBuildResult complete_energy_build;
    FreeSurfaceAggregationEnergyClassification
        aggregation_energy;
    const auto functional_history =
        svmp::FE::systems::functionalRecords(*sim.fe_system).acceptedFunctionalHistory();
    const auto declaration_count =
        free_surface_declarations.size();
    const bool functional_history_aligned =
        declaration_count != 0u &&
        functional_history.size() % declaration_count == 0u &&
        functional_history.size() / declaration_count >= 2u;
    if (!functional_history_aligned) {
      complete_energy_build.missing_requirements.push_back(
          "two_complete_accepted_functional_endpoints");
    } else {
      const auto previous_group_begin =
          functional_history.size() - 2u * declaration_count;
      const auto current_group_begin =
          functional_history.size() - declaration_count;
      const auto before_energy =
          summarizeAcceptedFreeSurfaceEnergyGroup(
              functional_history.subspan(
                  previous_group_begin, declaration_count));
      const auto after_energy =
          summarizeAcceptedFreeSurfaceEnergyGroup(
              functional_history.subspan(
                  current_group_begin, declaration_count));

      const auto aggregation_prolongations =
          sim.fe_system
              ->finalizedSmallCutAggregationProlongations();
      const auto aggregation_refresh_reports =
          sim.fe_system
              ->completedSmallCutAggregationRefreshReports();
      aggregation_energy = classifyAcceptedAggregationEnergy(
          aggregation_prolongations,
          aggregation_refresh_reports,
          std::span<const svmp::FE::Real>(
              accepted_state_solution.data(),
              accepted_state_solution.size()));
      const auto [minimum_aggregation_present,
                  maximum_aggregation_present] =
          collectiveMinMaxUint64(
              aggregation_energy.present ? 1u : 0u,
              active_comm);
      const auto [minimum_aggregation_neutral,
                  maximum_aggregation_neutral] =
          collectiveMinMaxUint64(
              aggregation_energy.neutral ? 1u : 0u,
              active_comm);
      const auto [minimum_aggregation_rootless,
                  maximum_aggregation_rootless] =
          collectiveMinMaxUint64(
              aggregation_energy.rootless ? 1u : 0u,
              active_comm);
      const auto [minimum_aggregation_rooted_rows,
                  maximum_aggregation_rooted_rows] =
          collectiveMinMaxUint64(
              static_cast<std::uint64_t>(
                  aggregation_energy.rooted_row_count),
              active_comm);
      const auto minimum_aggregation_residual =
          collectiveMinDouble(
              aggregation_energy.maximum_constraint_residual,
              active_comm);
      const auto maximum_aggregation_residual =
          collectiveMaxDouble(
              aggregation_energy.maximum_constraint_residual,
              active_comm);
      if (minimum_aggregation_present !=
              maximum_aggregation_present ||
          minimum_aggregation_neutral !=
              maximum_aggregation_neutral ||
          minimum_aggregation_rootless !=
              maximum_aggregation_rootless ||
          minimum_aggregation_rooted_rows !=
              maximum_aggregation_rooted_rows ||
          minimum_aggregation_residual !=
              maximum_aggregation_residual) {
        throw std::runtime_error(
            "[svMultiPhysics::Application] Accepted aggregation energy classification differs across the active FE communicator.");
      }

      FreeSurfaceResidualWorkChannels residual_work;
      std::uint64_t residual_work_state_revision = 0u;
      if (pending_free_surface_residual_work_stage.has_value()) {
        residual_work =
            pending_free_surface_residual_work_stage->channels;
        residual_work_state_revision =
            pending_free_surface_residual_work_stage
                ->algebraic_state_revision;
      }
      const bool numerical_maintenance_absent =
          !pending_accepted_step_maintenance_energy_account
               .has_value() ||
          pending_accepted_step_maintenance_energy_account
                  ->modeled_energy_breakdown
                  .numerical_maintenance_total.row_count == 0u;
      complete_energy_build =
          buildFixedTopologyOnePhaseEnergyCandidate(
              free_surface_energy_transaction_sequence + 1u,
              free_surface_energy_attempt,
              free_surface_energy_history_contiguous,
              before_energy,
              after_energy,
              residual_work_state_revision,
              residual_work,
              free_surface_residual_work_declarations,
              free_surface_boundary_energy_declaration,
              numerical_maintenance_absent,
              aggregation_energy.present,
              aggregation_energy.neutral,
              aggregation_energy.rootless);
    }

    if (!complete_energy_build.candidate.has_value()) {
      const bool preceding_history_complete =
          free_surface_energy_history_contiguous;
      free_surface_energy_history_contiguous = false;
      std::sort(
          complete_energy_build.missing_requirements.begin(),
          complete_energy_build.missing_requirements.end());
      complete_energy_build.missing_requirements.erase(
          std::unique(
              complete_energy_build.missing_requirements.begin(),
              complete_energy_build.missing_requirements.end()),
          complete_energy_build.missing_requirements.end());
      std::ostringstream missing_requirements;
      for (std::size_t index = 0u;
           index <
               complete_energy_build.missing_requirements.size();
           ++index) {
        if (index != 0u) {
          missing_requirements << ',';
        }
        missing_requirements
            << complete_energy_build.missing_requirements[index];
      }
      oopCout()
          << std::setprecision(17)
          << "[svMultiPhysics::Application] Free-surface complete energy"
          << " diagnostic=free_surface_complete_energy_record"
          << " status=not_published"
          << " step=" << h.stepIndex()
          << " attempt="
          << free_surface_energy_attempt
          << " preceding_history_complete="
          << (preceding_history_complete ? "true" : "false")
          << " accepted_history_contiguous=false"
          << " missing_requirement_count="
          << complete_energy_build.missing_requirements.size()
          << " missing_requirements="
          << missing_requirements.str()
          << " aggregation_present="
          << (aggregation_energy.present ? "true" : "false")
          << " aggregation_neutral="
          << (aggregation_energy.neutral ? "true" : "false")
          << " aggregation_rootless="
          << (aggregation_energy.rootless ? "true" : "false")
          << " aggregation_rooted_row_count="
          << aggregation_energy.rooted_row_count
          << " aggregation_maximum_constraint_residual="
          << aggregation_energy.maximum_constraint_residual
          << " complete_energy_record_connected=false"
          << std::endl;
    } else {
      auto candidate =
          std::move(*complete_energy_build.candidate);
      const auto transaction_id =
          candidate.metadata.transaction_id;
      free_surface_energy_ledger.beginAttempt(candidate.metadata);
      free_surface_energy_ledger.stageBalance(
          std::move(candidate.values),
          std::move(candidate.sources),
          candidate.inputs);
      free_surface_energy_ledger.commitAttempt();
      free_surface_energy_transaction_sequence = transaction_id;
      accepted_complete_energy_record_connected = true;
      const auto& record =
          free_surface_energy_ledger.acceptedAttempts().back();
      auto& record_log = oopCout();
      record_log
          << std::setprecision(17)
          << "[svMultiPhysics::Application] Free-surface complete energy"
          << " diagnostic=free_surface_complete_energy_record"
          << " status=accepted"
          << " transaction_id="
          << record.metadata.transaction_id
          << " step=" << record.metadata.step
          << " attempt=" << record.metadata.attempt
          << " time_before=" << record.metadata.time_before
          << " time_after=" << record.metadata.time_after
          << " dt=" << record.metadata.dt
          << " temporal_scheme=backward_euler"
          << " physical_evaluation_time="
          << record.metadata.physical_evaluation_time
          << " physical_evaluation_stage_fraction="
          << record.metadata.physical_evaluation_stage_fraction
          << " algebraic_state_revision_before="
          << record.metadata.algebraic_state_revision_before
          << " physical_endpoint_algebraic_state_revision="
          << record.metadata
                 .physical_endpoint_algebraic_state_revision
          << " algebraic_state_revision_after="
          << record.metadata.algebraic_state_revision_after
          << " snapshot_set_revision_before="
          << record.metadata.snapshot_set_revision_before
          << " physical_endpoint_snapshot_set_revision="
          << record.metadata
                 .physical_endpoint_snapshot_set_revision
          << " snapshot_set_revision_after="
          << record.metadata.snapshot_set_revision_after
          << " mesh_topology_set_revision_before="
          << record.metadata.mesh_topology_set_revision_before
          << " physical_endpoint_mesh_topology_set_revision="
          << record.metadata
                 .physical_endpoint_mesh_topology_set_revision
          << " mesh_topology_set_revision_after="
          << record.metadata.mesh_topology_set_revision_after
          << " cut_topology_set_revision_before="
          << record.metadata.cut_topology_set_revision_before
          << " physical_endpoint_cut_topology_set_revision="
          << record.metadata
                 .physical_endpoint_cut_topology_set_revision
          << " cut_topology_set_revision_after="
          << record.metadata.cut_topology_set_revision_after
          << " extension_map_revision_present=false"
          << " geometry_pruning_occurred=false";
      appendCompleteEnergyChannelValues(
          record_log,
          free_surface_energy_ledger.catalogue(),
          record);
      record_log
          << " stored_energy_change="
          << record.accepted_stored_energy_change
          << " physical_stored_energy_change="
          << record.accepted_physical_stored_energy_change
          << " maintenance_stored_energy_change="
          << record.accepted_maintenance_stored_energy_change
          << " integrated_physical_dissipation="
          << record.accepted_integrated_physical_dissipation
          << " external_work="
          << record.accepted_external_work
          << " numerical_work="
          << record.accepted_numerical_work
          << " accepted_balance_residual="
          << record.accepted_balance_residual
          << " aggregation_present="
          << (aggregation_energy.present ? "true" : "false")
          << " aggregation_neutral="
          << (aggregation_energy.neutral ? "true" : "false")
          << " aggregation_rooted_row_count="
          << aggregation_energy.rooted_row_count
          << " aggregation_maximum_constraint_residual="
          << aggregation_energy.maximum_constraint_residual
          << " accepted_history_contiguous=true"
          << " complete_energy_record_connected=true"
          << std::endl;

      auto& source_log = oopCout();
      source_log
          << "[svMultiPhysics::Application] Free-surface complete energy"
          << " diagnostic=free_surface_complete_energy_channel_sources"
          << " status=accepted"
          << " transaction_id=" << record.metadata.transaction_id
          << " step=" << record.metadata.step
          << " attempt=" << record.metadata.attempt;
      appendCompleteEnergyChannelSources(
          source_log,
          free_surface_energy_ledger.catalogue(),
          record);
      source_log << " complete_energy_record_connected=true"
                 << std::endl;
    }
  }
  if (pending_free_surface_residual_work_stage.has_value()) {
    const auto& residual_work_stage =
        *pending_free_surface_residual_work_stage;
    const bool maintenance_chain_start_available =
        pending_accepted_step_maintenance_energy_account
                .has_value() &&
        pending_accepted_step_maintenance_energy_account
            ->maintenance_start.has_value();
    const bool maintenance_chain_start_matches =
        maintenance_chain_start_available &&
        pending_accepted_step_maintenance_energy_account
                ->maintenance_start
                ->algebraic_state_revision ==
            residual_work_stage.algebraic_state_revision;
    const bool physical_channel_connection_complete =
        pending_accepted_step_physical_endpoint_channels
                .has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->convection_work.has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->pressure_continuity_work.has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->nonconservative_body_force_work.has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->weak_boundary_work.has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->vms_pspg_work.has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->cut_stabilization_work.has_value() &&
        pending_accepted_step_physical_endpoint_channels
            ->ghost_penalty_work.has_value();
    oopCout()
        << std::setprecision(17)
        << "[svMultiPhysics::Application] Free-surface residual work"
        << " diagnostic=free_surface_residual_work_stage"
        << " status=accepted"
        << " step=" << residual_work_stage.step
        << " attempt_index="
        << residual_work_stage.attempt_index
        << " time=" << residual_work_stage.time
        << " dt=" << residual_work_stage.dt
        << " algebraic_state_revision="
        << residual_work_stage.algebraic_state_revision
        << " maintenance_chain_start_available="
        << (maintenance_chain_start_available ? "true" : "false")
        << " maintenance_chain_start_matches="
        << (maintenance_chain_start_matches ? "true" : "false")
        << " physical_channel_connection_complete="
        << (physical_channel_connection_complete ? "true" : "false")
        << " accepted_state_convection_work="
        << *residual_work_stage.channels.convection
        << " accepted_state_pressure_continuity_work="
        << *residual_work_stage.channels.pressure_continuity
        << " accepted_state_nonconservative_body_force_work="
        << *residual_work_stage.channels
                .nonconservative_body_force
        << " accepted_state_weak_boundary_work="
        << *residual_work_stage.channels.weak_boundary
        << " accepted_state_vms_pspg_work="
        << *residual_work_stage.channels.vms_pspg
        << " accepted_state_cut_stabilization_work="
        << *residual_work_stage.channels.cut_stabilization
        << " accepted_state_ghost_penalty_work="
        << *residual_work_stage.channels.ghost_penalty
        << " complete_energy_record_connected="
        << (accepted_complete_energy_record_connected
                ? "true"
                : "false")
        << std::endl;
    pending_free_surface_residual_work_stage.reset();
  }
}

void EnergyParticipant::recordRejectedAttempt(
    const svmp::FE::timestepping::TimeHistory& h,
    svmp::FE::timestepping::StepRejectReason reason)
{
  if (complete_free_surface_energy_connector_available &&
      free_surface_energy_history_contiguous) {
    const auto functional_history =
        svmp::FE::systems::functionalRecords(*sim.fe_system).acceptedFunctionalHistory();
    const auto declaration_count =
        free_surface_declarations.size();
    if (h.stepIndex() < 0 || declaration_count == 0u ||
        functional_history.size() < declaration_count ||
        functional_history.size() % declaration_count != 0u) {
      throw std::logic_error(
          "[svMultiPhysics::Application] A rejected free-surface energy attempt requires one complete accepted functional endpoint.");
    }
    const auto before =
        summarizeAcceptedFreeSurfaceEnergyGroup(
            functional_history.last(declaration_count));
    const auto accepted_step =
        static_cast<std::uint64_t>(h.stepIndex());
    const double accepted_time_scale = std::max(
        {1.0,
         std::abs(before.accepted_time),
         std::abs(h.time())});
    const double accepted_time_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        accepted_time_scale;
    if (before.accepted_step != accepted_step ||
        std::abs(before.accepted_time - h.time()) >
            accepted_time_tolerance ||
        !free_surface_energy_attempt_step.has_value() ||
        *free_surface_energy_attempt_step != h.stepIndex() ||
        free_surface_energy_attempt == 0u) {
      throw std::logic_error(
          "[svMultiPhysics::Application] Rejected free-surface energy provenance does not match the restored accepted endpoint and solve attempt.");
    }
    if (free_surface_energy_transaction_sequence ==
        std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error(
          "[svMultiPhysics::Application] Free-surface energy transaction identifier overflowed.");
    }
    const auto ledger_reason =
        freeSurfaceEnergyRejectionReasonForStepReject(reason);
    auto metadata =
        buildUnstagedFreeSurfaceEnergyAttemptMetadata(
            free_surface_energy_transaction_sequence + 1u,
            accepted_step + 1u,
            free_surface_energy_attempt,
            h.time(),
            h.dt(),
            before);
    const auto transaction_id = metadata.transaction_id;
    free_surface_energy_ledger.beginAttempt(
        std::move(metadata));
    free_surface_energy_ledger.rejectUnstagedAttempt(
        ledger_reason);
    free_surface_energy_transaction_sequence = transaction_id;
    const auto& record =
        free_surface_energy_ledger.rejectedAttempts().back();
    oopCout()
        << std::setprecision(17)
        << "[svMultiPhysics::Application] Free-surface complete energy"
        << " diagnostic=free_surface_complete_energy_record"
        << " status=rejected"
        << " transaction_id="
        << record.metadata.transaction_id
        << " step=" << record.metadata.step
        << " attempt=" << record.metadata.attempt
        << " time_before=" << record.metadata.time_before
        << " time_after=" << record.metadata.time_after
        << " dt=" << record.metadata.dt
        << " temporal_scheme=backward_euler"
        << " time_loop_rejection_reason="
        << stepRejectReasonName(reason)
        << " ledger_rejection_reason="
        << svmp::FE::systems::energyBalanceRejectionReasonName(
               record.rejection_reason)
        << " algebraic_state_revision_before="
        << record.metadata.algebraic_state_revision_before
        << " snapshot_set_revision_before="
        << record.metadata.snapshot_set_revision_before
        << " mesh_topology_set_revision_before="
        << record.metadata.mesh_topology_set_revision_before
        << " cut_topology_set_revision_before="
        << record.metadata.cut_topology_set_revision_before
        << " extension_map_revision_before_available="
        << (record.metadata.extension_map_revision_before
                    .has_value()
                ? "true"
                : "false")
        << " extension_map_revision_before="
        << record.metadata.extension_map_revision_before
               .value_or(0u)
        << " physical_endpoint_metadata_available=false"
        << " topology_event_endpoint_metadata_available=false"
        << " topology_change_mapped_to_preaccept="
        << (reason == svmp::FE::timestepping::
                          StepRejectReason::CutTopologyChanged
                ? "true"
                : "false")
        << " balance_staged=false"
        << " trial_balance_values_available=false"
        << " accepted_stored_energy_change="
        << record.accepted_stored_energy_change
        << " accepted_physical_stored_energy_change="
        << record.accepted_physical_stored_energy_change
        << " accepted_maintenance_stored_energy_change="
        << record.accepted_maintenance_stored_energy_change
        << " accepted_integrated_physical_dissipation="
        << record.accepted_integrated_physical_dissipation
        << " accepted_external_work="
        << record.accepted_external_work
        << " accepted_numerical_work="
        << record.accepted_numerical_work
        << " accepted_balance_residual="
        << record.accepted_balance_residual
        << " accepted_history_contiguous=true"
        << " complete_energy_attempt_connected=true"
        << std::endl;
  } else if (complete_free_surface_energy_connector_available) {
    oopCout()
        << "[svMultiPhysics::Application] Free-surface complete energy"
        << " diagnostic=free_surface_complete_energy_record"
        << " status=not_published"
        << " step=" << h.stepIndex() + 1
        << " attempt=" << free_surface_energy_attempt
        << " time_loop_rejection_reason="
        << stepRejectReasonName(reason)
        << " accepted_history_contiguous=false"
        << " missing_requirements=preceding_complete_energy_history"
        << " complete_energy_attempt_connected=false"
        << std::endl;
  }
}

void EnergyParticipant::clearPendingAccount() noexcept
{
  pending_pre_maintenance_endpoint_state_revision = 0u;
  pending_accepted_step_maintenance_energy_account.reset();
  pending_accepted_step_physical_endpoint_channels.reset();
}

} // namespace svmp::Physics::formulations::navier_stokes::free_surface
