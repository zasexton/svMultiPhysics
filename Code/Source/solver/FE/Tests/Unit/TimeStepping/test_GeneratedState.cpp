/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "TimeStepping/GeneratedState.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace ts = svmp::FE::timestepping;
using Point = ts::NewtonOptions::StateSynchronizationPoint;

const std::vector<std::pair<Point, std::string>>& allPoints()
{
  static const std::vector<std::pair<Point, std::string>> points{
      {Point::OuterFixedPointState, "outer_fixed_point"},
      {Point::ProjectedOuterFixedPointState, "projected_outer_fixed_point"},
      {Point::EndpointCandidateState, "endpoint_candidate"},
      {Point::ProjectedEndpointCandidateState, "projected_endpoint_candidate"},
      {Point::AcceptedNonlinearState, "accepted"},
      {Point::ResidualAssembly, "residual"},
      {Point::JacobianAssembly, "jacobian"},
      {Point::JacobianAndResidualAssembly, "jacobian_and_residual"},
      {Point::LineSearchTrialResidual, "line_search_trial"},
      {Point::RestoredNonlinearState, "restored"},
      {Point::RestoredOuterFixedPointState, "restored_outer_fixed_point"},
      {Point::RestoredProjectedOuterFixedPointState,
       "restored_projected_outer_fixed_point"},
      {Point::RestoredTimeStepState, "restored_time_step"},
      {Point::RestoredProjectedTimeStepState, "restored_projected_time_step"},
      {Point::FinalResidualAssembly, "final_residual"},
  };
  return points;
}

// Sets an environment variable for the lifetime of the guard and restores
// the previous value afterwards.
class ScopedEnvironment {
public:
  ScopedEnvironment(const char* name, const char* value) : name_(name)
  {
    if (const char* previous = std::getenv(name)) {
      previous_ = std::string(previous);
    }
    if (value != nullptr) {
      ::setenv(name, value, 1);
    } else {
      ::unsetenv(name);
    }
  }
  ScopedEnvironment(const ScopedEnvironment&) = delete;
  ScopedEnvironment& operator=(const ScopedEnvironment&) = delete;
  ~ScopedEnvironment()
  {
    if (previous_.has_value()) {
      ::setenv(name_, previous_->c_str(), 1);
    } else {
      ::unsetenv(name_);
    }
  }

private:
  const char* name_;
  std::optional<std::string> previous_{};
};

} // namespace

// The tokens are provenance strings of the generated-state refreshes; the
// cut-topology attempt protocol matches on some of them, and log parsers on
// all of them.
TEST(GeneratedStateSyncPoints, NamesAreTheApplicationProvenanceTokens)
{
  ASSERT_EQ(allPoints().size(), 15u);
  for (const auto& [point, name] : allPoints()) {
    EXPECT_EQ(std::string(ts::generatedStateSyncPointName(point)), name);
  }
}

TEST(GeneratedStateSyncPoints, OnlyTheTimeStepRestoresRestoreTheAcceptedState)
{
  for (const auto& [point, name] : allPoints()) {
    const bool expected = point == Point::RestoredTimeStepState ||
                          point == Point::RestoredProjectedTimeStepState;
    EXPECT_EQ(ts::restoresAcceptedTimeStepGeneratedState(point), expected)
        << name;
  }
}

TEST(GeneratedStateSyncPoints, FrozenExtensionRefreshesWithTheOuterFixedPoint)
{
  for (const auto& [point, name] : allPoints()) {
    EXPECT_TRUE(ts::refreshesFrozenLevelSetExtensionAtStateSync(point, true))
        << name;
    EXPECT_EQ(ts::refreshesFrozenLevelSetExtensionAtStateSync(point, false),
              point == Point::RestoredProjectedTimeStepState)
        << name;
  }
}

TEST(GeneratedStateSyncPoints, CachedCurvatureReuseIsAnOptInForLineSearchTrials)
{
  {
    ScopedEnvironment unset("SVMP_CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE",
                            nullptr);
    for (const auto& [point, name] : allPoints()) {
      EXPECT_FALSE(ts::allowCachedCurvatureAfterProjectionFailure(point))
          << name;
    }
  }
  ScopedEnvironment enabled("SVMP_CURVATURE_REUSE_CACHE_ON_TRIAL_FAILURE",
                            "1");
  for (const auto& [point, name] : allPoints()) {
    EXPECT_EQ(ts::allowCachedCurvatureAfterProjectionFailure(point),
              point == Point::LineSearchTrialResidual)
        << name;
  }
}

TEST(GeneratedStateSyncPoints, LineSearchTrialSynchronizationFollowsTheResidual)
{
  {
    ScopedEnvironment unset("SVMP_SYNC_LINE_SEARCH_TRIALS", nullptr);
    EXPECT_TRUE(ts::synchronizeTransientLineSearchTrials(true));
    EXPECT_FALSE(ts::synchronizeTransientLineSearchTrials(false));
  }
  {
    ScopedEnvironment off("SVMP_SYNC_LINE_SEARCH_TRIALS", "off");
    EXPECT_FALSE(ts::synchronizeTransientLineSearchTrials(true));
  }
  {
    ScopedEnvironment on("SVMP_SYNC_LINE_SEARCH_TRIALS", "YES");
    EXPECT_TRUE(ts::synchronizeTransientLineSearchTrials(false));
  }
  ScopedEnvironment unknown("SVMP_SYNC_LINE_SEARCH_TRIALS", "maybe");
  EXPECT_TRUE(ts::synchronizeTransientLineSearchTrials(true));
  EXPECT_FALSE(ts::synchronizeTransientLineSearchTrials(false));
}
