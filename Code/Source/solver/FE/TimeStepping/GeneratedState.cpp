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
#include <string>

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

} // namespace timestepping
} // namespace FE
} // namespace svmp
