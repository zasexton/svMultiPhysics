/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include "TimeStepping/GeneratedState.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

namespace svmp {
namespace FE {
namespace timestepping {

namespace {

// The driver's environment flag reader, unchanged.
bool parseBoolEnv(const char* name, bool default_value)
{
  const char* env = std::getenv(name);
  if (!env) {
    return default_value;
  }
  std::string v(env);
  std::transform(v.begin(), v.end(), v.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  if (v == "1" || v == "true" || v == "on" || v == "yes") {
    return true;
  }
  if (v == "0" || v == "false" || v == "off" || v == "no") {
    return false;
  }
  return default_value;
}

} // namespace

// Moved from Application/Core/ApplicationDriver.cpp with bodies unchanged
// (stateSyncPointName there).

const char* generatedStateSyncPointName(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point) noexcept
{
  using Point = svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  switch (point) {
  case Point::OuterFixedPointState:
    return "outer_fixed_point";
  case Point::ProjectedOuterFixedPointState:
    return "projected_outer_fixed_point";
  case Point::EndpointCandidateState:
    return "endpoint_candidate";
  case Point::ProjectedEndpointCandidateState:
    return "projected_endpoint_candidate";
  case Point::AcceptedNonlinearState:
    return "accepted";
  case Point::ResidualAssembly:
    return "residual";
  case Point::JacobianAssembly:
    return "jacobian";
  case Point::JacobianAndResidualAssembly:
    return "jacobian_and_residual";
  case Point::LineSearchTrialResidual:
    return "line_search_trial";
  case Point::RestoredNonlinearState:
    return "restored";
  case Point::RestoredOuterFixedPointState:
    return "restored_outer_fixed_point";
  case Point::RestoredProjectedOuterFixedPointState:
    return "restored_projected_outer_fixed_point";
  case Point::RestoredTimeStepState:
    return "restored_time_step";
  case Point::RestoredProjectedTimeStepState:
    return "restored_projected_time_step";
  case Point::FinalResidualAssembly:
    return "final_residual";
  }
  return "unknown";
}

bool restoresAcceptedTimeStepGeneratedState(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
    noexcept
{
  using Point =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  return point == Point::RestoredTimeStepState ||
         point == Point::RestoredProjectedTimeStepState;
}

bool refreshesFrozenLevelSetExtensionAtStateSync(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point,
    bool use_external_state_fixed_point) noexcept
{
  using Point =
      svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  return use_external_state_fixed_point ||
         point == Point::RestoredProjectedTimeStepState;
}

bool allowCachedCurvatureAfterProjectionFailure(
    svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint point)
{
  using Point = svmp::FE::timestepping::NewtonOptions::StateSynchronizationPoint;
  return point == Point::LineSearchTrialResidual &&
         parseBoolEnv("SVMP_CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE", false);
}

bool synchronizeTransientLineSearchTrials(
    bool residual_defining_state_changes)
{
  // R(u, G(u)) is the nonlinear residual when generated cuts, projected
  // curvature, velocity extension, or state-dependent constraints are part of
  // the problem definition.  Backtracking must evaluate that deterministic
  // residual at each trial.  Freezing G at the previous iterate instead
  // globalizes a different Picard surrogate and can accept a step that
  // increases the actual residual.  Keep an explicit opt-out for controlled
  // legacy comparisons.
  return parseBoolEnv("SVMP_SYNC_LINE_SEARCH_TRIALS",
                      residual_defining_state_changes);
}

// ---------------------------------------------------------------------------
// Generated-state registry
// ---------------------------------------------------------------------------

const char* generatedStateRoleName(GeneratedStateRole role) noexcept
{
  switch (role) {
  case GeneratedStateRole::Geometry:
    return "geometry";
  case GeneratedStateRole::DerivedField:
    return "derived_field";
  case GeneratedStateRole::Extension:
    return "extension";
  case GeneratedStateRole::Reference:
    return "reference";
  }
  return "unknown";
}

FunctionGeneratedStateParticipant::FunctionGeneratedStateParticipant(
    std::string name,
    GeneratedStateRole role,
    Callbacks callbacks)
    : name_(std::move(name)), role_(role), callbacks_(std::move(callbacks))
{
}

void FunctionGeneratedStateParticipant::prepare(const GeneratedStateSync& sync)
{
  if (callbacks_.prepare) {
    callbacks_.prepare(sync);
  }
}

void FunctionGeneratedStateParticipant::synchronize(
    const systems::SystemStateView& state,
    GeneratedStateSync& sync)
{
  if (callbacks_.synchronize) {
    callbacks_.synchronize(state, sync);
  }
}

void FunctionGeneratedStateParticipant::invalidate()
{
  if (callbacks_.invalidate) {
    callbacks_.invalidate();
  }
}

GeneratedStateRegistry::GeneratedStateRegistry(GeneratedStatePolicy policy)
    : policy_(policy)
{
}

void GeneratedStateRegistry::add(
    std::shared_ptr<IGeneratedStateParticipant> participant)
{
  if (!participant) {
    throw std::invalid_argument(
        "GeneratedStateRegistry::add: null participant");
  }
  const auto role = participant->role();
  const auto position = std::find_if(
      participants_.begin(), participants_.end(), [role](const auto& entry) {
        return static_cast<int>(entry->role()) > static_cast<int>(role);
      });
  participants_.insert(position, std::move(participant));
}

bool GeneratedStateRegistry::synchronizes(
    NewtonOptions::StateSynchronizationPoint point) const noexcept
{
  using Point = NewtonOptions::StateSynchronizationPoint;
  if (!policy_.outer_fixed_point) {
    return true;
  }
  const bool outer = point == Point::OuterFixedPointState ||
                     point == Point::ProjectedOuterFixedPointState ||
                     point == Point::RestoredOuterFixedPointState ||
                     point == Point::RestoredProjectedOuterFixedPointState;
  if (policy_.solve == GeneratedStateSolve::Steady) {
    return outer;
  }
  return outer || point == Point::EndpointCandidateState ||
         point == Point::ProjectedEndpointCandidateState ||
         point == Point::RestoredTimeStepState ||
         point == Point::RestoredProjectedTimeStepState;
}

bool GeneratedStateRegistry::constraintConstructionPass(
    NewtonOptions::StateSynchronizationPoint point) const noexcept
{
  using Point = NewtonOptions::StateSynchronizationPoint;
  const bool outer_pass =
      policy_.outer_fixed_point &&
      (point == Point::OuterFixedPointState ||
       point == Point::RestoredOuterFixedPointState);
  if (policy_.solve == GeneratedStateSolve::Steady) {
    return outer_pass;
  }
  return outer_pass || point == Point::EndpointCandidateState ||
         point == Point::RestoredTimeStepState;
}

void GeneratedStateRegistry::synchronize(
    const systems::SystemStateView& state,
    NewtonOptions::StateSynchronizationPoint point) const
{
  using Point = NewtonOptions::StateSynchronizationPoint;
  // 1. point filter
  if (!synchronizes(point)) {
    return;
  }
  GeneratedStateSync sync;
  sync.point = point;
  sync.restoring_generated_state =
      point == Point::RestoredOuterFixedPointState ||
      point == Point::RestoredTimeStepState;
  sync.restoring_time_step_state =
      restoresAcceptedTimeStepGeneratedState(point);
  // 2. rank-local preparation
  for (const auto& participant : participants_) {
    participant->prepare(sync);
  }
  // 3. solve gates of a transient solve without the outer fixed point
  if (policy_.solve == GeneratedStateSolve::Transient &&
      !policy_.outer_fixed_point) {
    if (point == Point::LineSearchTrialResidual &&
        !policy_.synchronize_line_search_trials) {
      return;
    }
    if (!sync.restoring_time_step_state && !policy_.refresh_within_solve) {
      return;
    }
    if (!sync.restoring_time_step_state && policy_.per_step_only) {
      return;
    }
  }
  // 4. geometry
  for (const auto& participant : participants_) {
    if (participant->role() == GeneratedStateRole::Geometry) {
      participant->synchronize(state, sync);
    }
  }
  // 5. constraint-construction pass
  if (constraintConstructionPass(point)) {
    return;
  }
  // 6. derived fields, extensions and references
  for (const auto& participant : participants_) {
    if (participant->role() != GeneratedStateRole::Geometry) {
      participant->synchronize(state, sync);
    }
  }
}

void GeneratedStateRegistry::invalidate() const
{
  for (const auto& participant : participants_) {
    participant->invalidate();
  }
}

void GeneratedStateRegistry::bind(NewtonOptions& options) const
{
  auto self = shared_from_this();
  options.synchronize_state =
      [self](const systems::SystemStateView& state,
             NewtonOptions::StateSynchronizationPoint point) {
        self->synchronize(state, point);
      };
  std::shared_ptr<IGeneratedStateParticipant> provider;
  for (const auto& participant : participants_) {
    if (!participant->providesEpochs()) {
      continue;
    }
    if (provider) {
      throw std::logic_error(
          "GeneratedStateRegistry::bind: more than one participant provides "
          "generated-state epochs");
    }
    provider = participant;
  }
  if (!provider || policy_.solve != GeneratedStateSolve::Transient ||
      !policy_.outer_fixed_point) {
    return;
  }
  options.external_state_discontinuity =
      [provider](NewtonOptions::StateSynchronizationPoint point) {
        return provider->epochDiscontinuity(point);
      };
  options.acknowledge_external_state_discontinuity =
      [provider](NewtonOptions::StateSynchronizationPoint point) {
        provider->acknowledgeEpochDiscontinuity(point);
      };
  options.external_state_epoch_revisited =
      [provider](NewtonOptions::StateSynchronizationPoint point) {
        return provider->epochRevisited(point);
      };
  options.acknowledge_external_state_frozen_epoch = [provider]() {
    provider->acknowledgeFrozenEpoch();
  };
}

} // namespace timestepping
} // namespace FE
} // namespace svmp
