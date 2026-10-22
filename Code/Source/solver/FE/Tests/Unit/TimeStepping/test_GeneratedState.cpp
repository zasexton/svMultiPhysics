/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "TimeStepping/GeneratedState.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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

// ---------------------------------------------------------------------------
// Generated-state registry
// ---------------------------------------------------------------------------

namespace {

using Role = ts::GeneratedStateRole;

class RecordingParticipant final : public ts::IGeneratedStateParticipant {
public:
  RecordingParticipant(std::string name, Role role, std::vector<std::string>* trace)
      : name_(std::move(name)), role_(role), trace_(trace)
  {
  }

  std::string_view name() const noexcept override { return name_; }
  Role role() const noexcept override { return role_; }
  void prepare(const ts::GeneratedStateSync&) override
  {
    trace_->push_back("prepare:" + name_);
  }
  void synchronize(const svmp::FE::systems::SystemStateView&,
                   ts::GeneratedStateSync& sync) override
  {
    trace_->push_back("sync:" + name_);
    if (role_ == Role::Geometry) {
      sync.evaluated_state_source_revisions[7] = 99u;
    } else {
      seen_revision_ = sync.evaluated_state_source_revisions.count(7) != 0u
                           ? sync.evaluated_state_source_revisions.at(7)
                           : 0u;
    }
  }
  void invalidate() override { trace_->push_back("invalidate:" + name_); }

  std::uint64_t seen_revision_{0u};

private:
  std::string name_;
  Role role_;
  std::vector<std::string>* trace_;
};

// The generated-state synchronization of the Application's transient solve
// (ApplicationDriver::runTransient, opts.newton.synchronize_state before the
// registry), transcribed as a trace: the resets at restore points and the
// reinstall window are "prepare", the cut refresh is the geometry
// synchronization, curvature, extension and the semi-implicit reference
// follow it.
std::vector<std::string> transientReference(const ts::GeneratedStatePolicy& p,
                                            Point point)
{
  std::vector<std::string> trace;
  const bool fp = p.outer_fixed_point;
  const bool restoring_time_step_state =
      ts::restoresAcceptedTimeStepGeneratedState(point);
  if (fp && point != Point::OuterFixedPointState &&
      point != Point::ProjectedOuterFixedPointState &&
      point != Point::EndpointCandidateState &&
      point != Point::ProjectedEndpointCandidateState &&
      point != Point::RestoredOuterFixedPointState &&
      point != Point::RestoredProjectedOuterFixedPointState &&
      point != Point::RestoredTimeStepState &&
      point != Point::RestoredProjectedTimeStepState) {
    return trace;
  }
  for (const char* name : {"geometry", "curvature", "extension", "reference"}) {
    trace.push_back(std::string("prepare:") + name);
  }
  if (!fp && point == Point::LineSearchTrialResidual &&
      !p.synchronize_line_search_trials) {
    return trace;
  }
  if (!restoring_time_step_state && !fp && !p.refresh_within_solve) {
    return trace;
  }
  if (!restoring_time_step_state && !fp && p.per_step_only) {
    return trace;
  }
  trace.push_back("sync:geometry");
  const bool outer_constraint_construction_pass =
      (fp && (point == Point::OuterFixedPointState ||
              point == Point::RestoredOuterFixedPointState)) ||
      point == Point::EndpointCandidateState ||
      point == Point::RestoredTimeStepState;
  if (outer_constraint_construction_pass) {
    return trace;
  }
  trace.push_back("sync:curvature");
  trace.push_back("sync:extension");
  trace.push_back("sync:reference");
  return trace;
}

// The steady solve (ApplicationDriver::runSteadyState, newton_opts.
// synchronize_state before the registry); it has no reference participant.
std::vector<std::string> steadyReference(const ts::GeneratedStatePolicy& p,
                                         Point point)
{
  std::vector<std::string> trace;
  const bool fp = p.outer_fixed_point;
  if (fp && point != Point::OuterFixedPointState &&
      point != Point::ProjectedOuterFixedPointState &&
      point != Point::RestoredOuterFixedPointState &&
      point != Point::RestoredProjectedOuterFixedPointState) {
    return trace;
  }
  for (const char* name : {"geometry", "curvature", "extension"}) {
    trace.push_back(std::string("prepare:") + name);
  }
  trace.push_back("sync:geometry");
  const bool outer_constraint_construction_pass =
      fp && (point == Point::OuterFixedPointState ||
             point == Point::RestoredOuterFixedPointState);
  if (outer_constraint_construction_pass) {
    return trace;
  }
  trace.push_back("sync:curvature");
  trace.push_back("sync:extension");
  return trace;
}

std::shared_ptr<ts::GeneratedStateRegistry> makeRegistry(
    const ts::GeneratedStatePolicy& policy,
    std::vector<std::string>* trace,
    bool with_reference)
{
  auto registry = std::make_shared<ts::GeneratedStateRegistry>(policy);
  // Registered out of role order on purpose: the registry orders by role.
  if (with_reference) {
    registry->add(std::make_shared<RecordingParticipant>("reference", Role::Reference, trace));
  }
  registry->add(std::make_shared<RecordingParticipant>("extension", Role::Extension, trace));
  registry->add(std::make_shared<RecordingParticipant>("geometry", Role::Geometry, trace));
  registry->add(std::make_shared<RecordingParticipant>("curvature", Role::DerivedField, trace));
  return registry;
}

} // namespace

TEST(GeneratedStateRegistry, TransientOrderMatchesTheApplicationSynchronizationForEveryPoint)
{
  for (const bool fp : {false, true}) {
    for (const bool within : {false, true}) {
      for (const bool trials : {false, true}) {
        for (const bool per_step : {false, true}) {
          ts::GeneratedStatePolicy policy;
          policy.solve = ts::GeneratedStateSolve::Transient;
          policy.outer_fixed_point = fp;
          policy.refresh_within_solve = within;
          policy.synchronize_line_search_trials = trials;
          policy.per_step_only = per_step;
          for (const auto& [point, name] : allPoints()) {
            std::vector<std::string> trace;
            const auto registry = makeRegistry(policy, &trace, true);
            registry->synchronize(svmp::FE::systems::SystemStateView{}, point);
            EXPECT_EQ(trace, transientReference(policy, point))
                << name << " fp=" << fp << " within=" << within
                << " trials=" << trials << " per_step=" << per_step;
          }
        }
      }
    }
  }
}

TEST(GeneratedStateRegistry, SteadyOrderMatchesTheApplicationSynchronizationForEveryPoint)
{
  for (const bool fp : {false, true}) {
    ts::GeneratedStatePolicy policy;
    policy.solve = ts::GeneratedStateSolve::Steady;
    policy.outer_fixed_point = fp;
    for (const auto& [point, name] : allPoints()) {
      std::vector<std::string> trace;
      const auto registry = makeRegistry(policy, &trace, false);
      registry->synchronize(svmp::FE::systems::SystemStateView{}, point);
      EXPECT_EQ(trace, steadyReference(policy, point)) << name << " fp=" << fp;
    }
  }
}

TEST(GeneratedStateRegistry, GeometryRevisionsReachTheLaterRolesAndInvalidateRunsInOrder)
{
  std::vector<std::string> trace;
  ts::GeneratedStatePolicy policy;
  policy.outer_fixed_point = true;
  auto registry = std::make_shared<ts::GeneratedStateRegistry>(policy);
  auto geometry = std::make_shared<RecordingParticipant>("geometry", Role::Geometry, &trace);
  auto extension_a = std::make_shared<RecordingParticipant>("extension_a", Role::Extension, &trace);
  auto extension_b = std::make_shared<RecordingParticipant>("extension_b", Role::Extension, &trace);
  registry->add(extension_a);
  registry->add(geometry);
  registry->add(extension_b);
  registry->synchronize(svmp::FE::systems::SystemStateView{},
                        Point::ProjectedOuterFixedPointState);
  EXPECT_EQ(extension_a->seen_revision_, 99u);
  EXPECT_EQ(extension_b->seen_revision_, 99u);
  trace.clear();
  registry->invalidate();
  EXPECT_EQ(trace, (std::vector<std::string>{
                       "invalidate:geometry", "invalidate:extension_a",
                       "invalidate:extension_b"}));
  EXPECT_THROW(registry->add(nullptr), std::invalid_argument);
}

namespace {

class EpochParticipant final : public ts::IGeneratedStateParticipant {
public:
  std::string_view name() const noexcept override { return "epochs"; }
  Role role() const noexcept override { return Role::Geometry; }
  void synchronize(const svmp::FE::systems::SystemStateView&,
                   ts::GeneratedStateSync&) override
  {
    ++synchronizations;
  }
  bool providesEpochs() const noexcept override { return provides; }
  bool epochDiscontinuity(Point point) const override
  {
    return point == Point::OuterFixedPointState;
  }
  void acknowledgeEpochDiscontinuity(Point) override { ++acknowledged; }
  bool epochRevisited(Point) const override { return true; }
  void acknowledgeFrozenEpoch() override { ++frozen; }

  bool provides{true};
  int synchronizations{0};
  int acknowledged{0};
  int frozen{0};
};

} // namespace

TEST(GeneratedStateRegistry, BindInstallsEpochHooksOnlyForATransientOuterFixedPoint)
{
  for (const auto solve : {ts::GeneratedStateSolve::Steady, ts::GeneratedStateSolve::Transient}) {
    for (const bool fp : {false, true}) {
      for (const bool provides : {false, true}) {
        ts::GeneratedStatePolicy policy;
        policy.solve = solve;
        policy.outer_fixed_point = fp;
        policy.refresh_within_solve = true;
        auto registry = std::make_shared<ts::GeneratedStateRegistry>(policy);
        auto participant = std::make_shared<EpochParticipant>();
        participant->provides = provides;
        registry->add(participant);
        ts::NewtonOptions options;
        registry->bind(options);
        ASSERT_TRUE(static_cast<bool>(options.synchronize_state));
        options.synchronize_state(svmp::FE::systems::SystemStateView{},
                                  Point::OuterFixedPointState);
        EXPECT_EQ(participant->synchronizations, 1);
        const bool hooks =
            solve == ts::GeneratedStateSolve::Transient && fp && provides;
        EXPECT_EQ(static_cast<bool>(options.external_state_discontinuity), hooks);
        EXPECT_EQ(static_cast<bool>(options.acknowledge_external_state_discontinuity), hooks);
        EXPECT_EQ(static_cast<bool>(options.external_state_epoch_revisited), hooks);
        EXPECT_EQ(static_cast<bool>(options.acknowledge_external_state_frozen_epoch), hooks);
        if (hooks) {
          EXPECT_TRUE(options.external_state_discontinuity(Point::OuterFixedPointState));
          EXPECT_FALSE(options.external_state_discontinuity(Point::EndpointCandidateState));
          options.acknowledge_external_state_discontinuity(Point::OuterFixedPointState);
          EXPECT_EQ(participant->acknowledged, 1);
          EXPECT_TRUE(options.external_state_epoch_revisited(Point::OuterFixedPointState));
          options.acknowledge_external_state_frozen_epoch();
          EXPECT_EQ(participant->frozen, 1);
        }
      }
    }
  }
  // At most one participant provides epochs.
  ts::GeneratedStatePolicy policy;
  policy.outer_fixed_point = true;
  auto registry = std::make_shared<ts::GeneratedStateRegistry>(policy);
  registry->add(std::make_shared<EpochParticipant>());
  registry->add(std::make_shared<EpochParticipant>());
  ts::NewtonOptions options;
  EXPECT_THROW(registry->bind(options), std::logic_error);
}

TEST(GeneratedStateRegistry, FunctionParticipantForwardsItsCallbacks)
{
  int prepared = 0;
  int synchronized = 0;
  int invalidated = 0;
  ts::FunctionGeneratedStateParticipant participant(
      "callbacks", Role::Extension,
      {.prepare = [&](const ts::GeneratedStateSync&) { ++prepared; },
       .synchronize =
           [&](const svmp::FE::systems::SystemStateView&, ts::GeneratedStateSync&) {
             ++synchronized;
           },
       .invalidate = [&] { ++invalidated; }});
  EXPECT_EQ(participant.name(), "callbacks");
  EXPECT_EQ(participant.role(), Role::Extension);
  ts::GeneratedStateSync sync;
  participant.prepare(sync);
  participant.synchronize(svmp::FE::systems::SystemStateView{}, sync);
  participant.invalidate();
  EXPECT_EQ(prepared, 1);
  EXPECT_EQ(synchronized, 1);
  EXPECT_EQ(invalidated, 1);
  ts::FunctionGeneratedStateParticipant empty("empty", Role::Reference, {});
  EXPECT_NO_THROW(empty.prepare(sync));
  EXPECT_NO_THROW(empty.synchronize(svmp::FE::systems::SystemStateView{}, sync));
  EXPECT_NO_THROW(empty.invalidate());
  EXPECT_STREQ(ts::generatedStateRoleName(Role::DerivedField), "derived_field");
}
