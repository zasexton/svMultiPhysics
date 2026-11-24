/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "TimeStepping/GeneratedState.h"

#include <algorithm>
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

TEST(GeneratedStateSelection, PerStepOnlyIsOffUnlessTheEnvironmentSetsIt)
{
  {
    ScopedEnvironment unset("SVMP_CUT_REFRESH_PER_STEP_ONLY", nullptr);
    EXPECT_FALSE(ts::perStepOnlyGeneratedState());
  }
  {
    ScopedEnvironment on("SVMP_CUT_REFRESH_PER_STEP_ONLY", "1");
    EXPECT_TRUE(ts::perStepOnlyGeneratedState());
  }
  ScopedEnvironment unknown("SVMP_CUT_REFRESH_PER_STEP_ONLY", "maybe");
  EXPECT_FALSE(ts::perStepOnlyGeneratedState());
}

namespace {

// The selection lines of the Application's steady and transient solves
// before they moved to FE, transcribed with their variable names.
ts::GeneratedStateSelection transcribedSelection(const ts::GeneratedStateRequests& r,
                                                 ts::GeneratedStateSolve solve,
                                                 bool per_step_only_generated_state)
{
  ts::GeneratedStateSelection s;
  if (solve == ts::GeneratedStateSolve::Steady) {
    const bool steady_has_generated_state = r.cut_geometry || r.maintenance || r.extension;
    const bool use_steady_external_state_fixed_point = steady_has_generated_state;
    s.generated_state = steady_has_generated_state;
    s.accepted_state_sync_invalidates_residual =
        steady_has_generated_state && !use_steady_external_state_fixed_point;
    s.jacobian_check_refreshes_geometry_within_solve = !use_steady_external_state_fixed_point;
    s.jacobian_check_frozen_algebraic_extension = false;
    s.policy = ts::GeneratedStatePolicy{.solve = ts::GeneratedStateSolve::Steady,
                                        .outer_fixed_point = use_steady_external_state_fixed_point};
    s.track_cut_topology = false;
    return s;
  }
  const bool has_frozen_algebraic_level_set_extension = r.monolithic_extension;
  const bool has_transient_generated_state = r.cut_geometry || r.maintenance || r.extension;
  const bool use_transient_external_state_fixed_point =
      has_transient_generated_state && !per_step_only_generated_state;
  const bool refresh_generated_geometry_within_solve =
      has_transient_generated_state && !use_transient_external_state_fixed_point &&
      !has_frozen_algebraic_level_set_extension;
  const bool accepted_state_sync_invalidates_residual =
      refresh_generated_geometry_within_solve && has_transient_generated_state;
  const bool synchronize_line_search_trials =
      ts::synchronizeTransientLineSearchTrials(accepted_state_sync_invalidates_residual);
  s.generated_state = has_transient_generated_state;
  s.accepted_state_sync_invalidates_residual = accepted_state_sync_invalidates_residual;
  s.policy = ts::GeneratedStatePolicy{.solve = ts::GeneratedStateSolve::Transient,
                                      .outer_fixed_point = use_transient_external_state_fixed_point,
                                      .refresh_within_solve = refresh_generated_geometry_within_solve,
                                      .synchronize_line_search_trials = synchronize_line_search_trials,
                                      .per_step_only = per_step_only_generated_state};
  s.track_cut_topology = r.cut_geometry;
  s.jacobian_check_refreshes_geometry_within_solve = refresh_generated_geometry_within_solve;
  s.jacobian_check_frozen_algebraic_extension = has_frozen_algebraic_level_set_extension;
  return s;
}

void expectSameSelection(const ts::GeneratedStateSelection& a,
                         const ts::GeneratedStateSelection& b,
                         const std::string& where)
{
  EXPECT_EQ(a.policy.solve, b.policy.solve) << where;
  EXPECT_EQ(a.policy.outer_fixed_point, b.policy.outer_fixed_point) << where;
  EXPECT_EQ(a.policy.refresh_within_solve, b.policy.refresh_within_solve) << where;
  EXPECT_EQ(a.policy.synchronize_line_search_trials, b.policy.synchronize_line_search_trials) << where;
  EXPECT_EQ(a.policy.per_step_only, b.policy.per_step_only) << where;
  EXPECT_EQ(a.generated_state, b.generated_state) << where;
  EXPECT_EQ(a.accepted_state_sync_invalidates_residual, b.accepted_state_sync_invalidates_residual)
      << where;
  EXPECT_EQ(a.track_cut_topology, b.track_cut_topology) << where;
  EXPECT_EQ(a.jacobian_check_refreshes_geometry_within_solve,
            b.jacobian_check_refreshes_geometry_within_solve)
      << where;
  EXPECT_EQ(a.jacobian_check_frozen_algebraic_extension,
            b.jacobian_check_frozen_algebraic_extension)
      << where;
}

} // namespace

TEST(GeneratedStateSelection, MatchesTheApplicationSelectionForEveryRequestCombination)
{
  for (const char* trials : {static_cast<const char*>(nullptr), "0", "1"}) {
    ScopedEnvironment environment("SVMP_SYNC_LINE_SEARCH_TRIALS", trials);
    for (const auto solve : {ts::GeneratedStateSolve::Steady, ts::GeneratedStateSolve::Transient}) {
      for (int mask = 0; mask < 32; ++mask) {
        const ts::GeneratedStateRequests requests{.cut_geometry = (mask & 1) != 0,
                                                  .maintenance = (mask & 2) != 0,
                                                  .extension = (mask & 4) != 0,
                                                  .monolithic_extension = (mask & 8) != 0};
        const bool per_step_only = (mask & 16) != 0;
        const std::string where = std::string("trials=") + (trials ? trials : "unset") +
                                  " solve=" + std::to_string(static_cast<int>(solve)) +
                                  " mask=" + std::to_string(mask);
        expectSameSelection(ts::legacyGeneratedStateSelection(requests, solve, per_step_only),
                            transcribedSelection(requests, solve, per_step_only), where);
      }
    }
  }
  // Cases no reference deck reaches: a monolithic extension refreshed once
  // per step, and a steady solve, which ignores per-step-only.
  const auto lone_monolithic = ts::legacyGeneratedStateSelection(
      {.extension = true, .monolithic_extension = true}, ts::GeneratedStateSolve::Transient,
      /*per_step_only=*/true);
  EXPECT_FALSE(lone_monolithic.policy.outer_fixed_point);
  EXPECT_FALSE(lone_monolithic.policy.refresh_within_solve);
  EXPECT_TRUE(lone_monolithic.jacobian_check_frozen_algebraic_extension);
  const auto steady = ts::legacyGeneratedStateSelection(
      {.maintenance = true}, ts::GeneratedStateSolve::Steady, /*per_step_only=*/true);
  EXPECT_TRUE(steady.policy.outer_fixed_point);
  EXPECT_FALSE(steady.policy.per_step_only);
}

// ---------------------------------------------------------------------------
// Outer fixed-point preset
// ---------------------------------------------------------------------------

namespace {

using FixedPointOptions = ts::NewtonOptions::ExternalStateFixedPointOptions;

void expectSameFixedPointOptions(const FixedPointOptions& a, const FixedPointOptions& b,
                                 const std::string& where)
{
  EXPECT_EQ(a.enabled, b.enabled) << where;
  EXPECT_EQ(a.max_iterations, b.max_iterations) << where;
  EXPECT_EQ(a.max_discontinuity_restarts, b.max_discontinuity_restarts) << where;
  EXPECT_EQ(a.constraint_cycle_exit_start_iteration, b.constraint_cycle_exit_start_iteration)
      << where;
  EXPECT_EQ(a.dynamic_relaxation.enabled, b.dynamic_relaxation.enabled) << where;
  EXPECT_EQ(a.dynamic_relaxation.start_iteration, b.dynamic_relaxation.start_iteration) << where;
  EXPECT_EQ(a.dynamic_relaxation.initial_factor, b.dynamic_relaxation.initial_factor) << where;
  EXPECT_EQ(a.dynamic_relaxation.minimum_factor, b.dynamic_relaxation.minimum_factor) << where;
  EXPECT_EQ(a.dynamic_relaxation.maximum_factor, b.dynamic_relaxation.maximum_factor) << where;
  EXPECT_EQ(a.dynamic_relaxation.denominator_relative_tolerance,
            b.dynamic_relaxation.denominator_relative_tolerance)
      << where;
  EXPECT_EQ(a.residual_first_on_predicted_certificate, b.residual_first_on_predicted_certificate)
      << where;
}

// The outer fixed-point block of the Application's transient solve before
// it moved to FE, transcribed with an override standing for each deck key
// (std::nullopt: the key is not stated).
FixedPointOptions transcribedTransientPreset(std::optional<int> max_passes,
                                             std::optional<int> restarts,
                                             std::optional<int> start_pass,
                                             std::optional<int> exit_pass_key)
{
  FixedPointOptions o;
  constexpr int kOuterFixedPointPlainPasses = 12;
  constexpr int kOuterFixedPointDefaultMaxPasses = 30;
  o.max_iterations = kOuterFixedPointDefaultMaxPasses;
  o.dynamic_relaxation.enabled = true;
  o.dynamic_relaxation.start_iteration = kOuterFixedPointPlainPasses - 1;
  if (max_passes) {
    o.max_iterations = std::max(1, *max_passes);
  }
  int cut_topology_restart_limit = o.max_iterations;
  if (restarts && *restarts >= 0) {
    cut_topology_restart_limit = *restarts;
  }
  o.max_discontinuity_restarts = cut_topology_restart_limit;
  if (start_pass) {
    o.dynamic_relaxation.enabled = *start_pass > 0;
    o.dynamic_relaxation.start_iteration = std::max(0, *start_pass - 1);
  }
  int exit_pass = kOuterFixedPointPlainPasses + 1;
  if (exit_pass_key) {
    exit_pass = *exit_pass_key;
  }
  o.constraint_cycle_exit_start_iteration = exit_pass > 0 ? exit_pass - 1 : -1;
  return o;
}

} // namespace

TEST(OuterFixedPointPreset, PinsTheSteadyAndTransientValues)
{
  const auto steady = ts::outerFixedPointPreset(ts::GeneratedStateSolve::Steady);
  EXPECT_FALSE(steady.enabled);
  EXPECT_EQ(steady.max_iterations, 12);
  EXPECT_FALSE(steady.dynamic_relaxation.enabled);
  EXPECT_EQ(steady.dynamic_relaxation.start_iteration, 0);
  EXPECT_EQ(steady.max_discontinuity_restarts, 0);
  EXPECT_EQ(steady.constraint_cycle_exit_start_iteration, -1);

  const auto transient = ts::outerFixedPointPreset(ts::GeneratedStateSolve::Transient);
  EXPECT_FALSE(transient.enabled);
  EXPECT_EQ(transient.max_iterations, 30);
  EXPECT_TRUE(transient.dynamic_relaxation.enabled);
  EXPECT_EQ(transient.dynamic_relaxation.start_iteration, 11);  // pass 12
  EXPECT_EQ(transient.max_discontinuity_restarts, 30);
  EXPECT_EQ(transient.constraint_cycle_exit_start_iteration, 12);  // pass 13
  for (const auto& preset : {steady, transient}) {
    EXPECT_EQ(preset.dynamic_relaxation.initial_factor, 1.0);
    EXPECT_EQ(preset.dynamic_relaxation.minimum_factor, 0.05);
    EXPECT_EQ(preset.dynamic_relaxation.maximum_factor, 25.0);
    EXPECT_EQ(preset.dynamic_relaxation.denominator_relative_tolerance, 1.0e-12);
    EXPECT_TRUE(preset.residual_first_on_predicted_certificate);
  }
}

TEST(OuterFixedPointPreset, AppliesTheOverridesAsTheApplicationDid)
{
  const std::vector<std::optional<int>> values{std::nullopt, -1, 0, 1, 5, 40};
  for (const auto& max_passes : values) {
    for (const auto& restarts : values) {
      for (const auto& start : values) {
        for (const auto& exit : values) {
          const ts::OuterFixedPointOverrides overrides{.max_passes = max_passes,
                                                       .relaxation_start_pass = start,
                                                       .cycle_exit_pass = exit,
                                                       .topology_restarts = restarts};
          const auto name = [](const std::optional<int>& v) {
            return v ? std::to_string(*v) : std::string("unset");
          };
          const std::string where = "passes=" + name(max_passes) + " restarts=" + name(restarts) +
                                    " start=" + name(start) + " exit=" + name(exit);
          expectSameFixedPointOptions(
              ts::outerFixedPointPreset(ts::GeneratedStateSolve::Transient, overrides),
              transcribedTransientPreset(max_passes, restarts, start, exit), where);
          // A steady solve ignores the overrides.
          expectSameFixedPointOptions(
              ts::outerFixedPointPreset(ts::GeneratedStateSolve::Steady, overrides),
              ts::outerFixedPointPreset(ts::GeneratedStateSolve::Steady), where);
        }
      }
    }
  }
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
