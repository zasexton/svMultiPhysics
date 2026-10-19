/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "TimeStepping/FieldRepairPublication.h"
#include "TimeStepping/TimeHistory.h"

#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

namespace ts = svmp::FE::timestepping;
namespace ts_test = svmp::FE::timestepping::test;
using svmp::FE::Real;
using Closure = ts::FirstOrderGeneralizedAlphaMaintenanceClosure;
using Status = ts::FirstOrderGeneralizedAlphaMaintenancePlanStatus;

// alpha_m = 3/4, alpha_f = 1/2, gamma = 1/2 + alpha_m - alpha_f, dt = 1/10.
ts::FirstOrderGeneralizedAlphaMaintenanceScheme midpointScheme()
{
    return ts::makeFirstOrderGeneralizedAlphaMaintenanceScheme(0.75, 0.5, 0.75, 0.1);
}

} // namespace

TEST(FieldRepairPublication, SchemeRequiresConsistentFiniteParameters)
{
    EXPECT_NO_THROW((void)midpointScheme());
    EXPECT_THROW((void)ts::makeFirstOrderGeneralizedAlphaMaintenanceScheme(0.75, 0.5, 0.7, 0.1),
                 std::invalid_argument);  // gamma != 1/2 + alpha_m - alpha_f
    EXPECT_THROW((void)ts::makeFirstOrderGeneralizedAlphaMaintenanceScheme(0.75, 1.5, -0.25, 0.1),
                 std::invalid_argument);  // alpha_f > 1
    EXPECT_THROW((void)ts::makeFirstOrderGeneralizedAlphaMaintenanceScheme(0.75, 0.5, 0.75, 0.0),
                 std::invalid_argument);  // dt = 0
    EXPECT_THROW((void)ts::makeFirstOrderGeneralizedAlphaMaintenanceScheme(
                     std::numeric_limits<Real>::quiet_NaN(), 0.5, 0.75, 0.1),
                 std::invalid_argument);
}

TEST(FieldRepairPublication, SameRepresentationDeltaShiftsEveryStoredState)
{
    const std::vector<Real> delta = {0.25, -1.5, 2.0};
    const auto plan = ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
        midpointScheme(), Closure::SameRepresentationDelta, delta, delta);
    ASSERT_EQ(plan.status, Status::AlgebraicallyComplete) << plan.diagnostic;
    ASSERT_TRUE(plan.post_accept.has_value());
    EXPECT_EQ(plan.post_accept->u_delta, delta);
    EXPECT_EQ(plan.post_accept->u_prev_delta, delta);
    EXPECT_EQ(plan.post_accept->u_prev2_and_deeper_delta, delta);
    EXPECT_EQ(plan.post_accept->u_dot_delta, std::vector<Real>(3, 0.0));
    EXPECT_EQ(plan.post_accept->accepted_stage_rate_delta, std::vector<Real>(3, 0.0));
    ASSERT_TRUE(plan.implied_prior_state_delta.has_value());
    EXPECT_EQ(*plan.implied_prior_state_delta, delta);
    EXPECT_FALSE(plan.requires_separate_geometric_motion_account);
}

TEST(FieldRepairPublication, PreservePriorStateAndRateUpdatesTheEndpointRate)
{
    const auto scheme = midpointScheme();
    const std::vector<Real> endpoint = {1.0, -2.0};
    const std::vector<Real> stage = {0.5 * endpoint[0], 0.5 * endpoint[1]};
    const auto plan = ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
        scheme, Closure::PreservePriorStateAndRate, stage, endpoint);
    ASSERT_EQ(plan.status, Status::AlgebraicallyComplete) << plan.diagnostic;
    ASSERT_TRUE(plan.post_accept.has_value());
    const auto& publication = *plan.post_accept;
    EXPECT_EQ(publication.u_delta, endpoint);
    EXPECT_EQ(publication.u_prev_delta, endpoint);
    EXPECT_EQ(publication.u_prev2_and_deeper_delta, std::vector<Real>(2, 0.0));
    EXPECT_EQ(publication.prior_rate_delta, std::vector<Real>(2, 0.0));
    const Real rate_scale = *scheme.gamma * *scheme.dt;
    for (std::size_t i = 0; i < endpoint.size(); ++i) {
        EXPECT_EQ(publication.u_dot_delta[i], endpoint[i] / rate_scale);
        EXPECT_EQ(publication.accepted_stage_rate_delta[i],
                  *scheme.alpha_m * publication.u_dot_delta[i]);
    }
    EXPECT_TRUE(plan.requires_separate_geometric_motion_account);
    EXPECT_LE(plan.max_endpoint_update_identity_residual, plan.identity_tolerance);
}

TEST(FieldRepairPublication, MismatchedClosureReportsTheImpliedPriorState)
{
    const std::vector<Real> endpoint = {1.0};
    const std::vector<Real> stage = {0.9};
    const auto plan = ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
        midpointScheme(), Closure::SameRepresentationDelta, stage, endpoint);
    EXPECT_EQ(plan.status, Status::ClosureMismatch);
    EXPECT_FALSE(plan.post_accept.has_value());
    ASSERT_TRUE(plan.implied_prior_state_delta.has_value());
    EXPECT_EQ((*plan.implied_prior_state_delta)[0], (0.9 - 0.5 * 1.0) / 0.5);
    EXPECT_TRUE(plan.requires_separate_geometric_motion_account);
}

TEST(FieldRepairPublication, ClosureFailuresAreReportedNotPublished)
{
    const std::vector<Real> one = {1.0};
    const std::vector<Real> zero = {0.0};

    ts::FirstOrderGeneralizedAlphaMaintenanceScheme endpoint_scheme{};
    endpoint_scheme.alpha_f = 1.0;
    EXPECT_EQ(ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                  endpoint_scheme, Closure::PreservePriorStateAndRate, one, one)
                  .status,
              Status::EndpointPolicyMismatch);
    EXPECT_EQ(ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                  endpoint_scheme, Closure::SameRepresentationDelta, zero, one)
                  .status,
              Status::EndpointPolicyMismatch);
    EXPECT_EQ(ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                  endpoint_scheme, Closure::SameRepresentationDelta, one, one)
                  .status,
              Status::AlgebraicallyComplete);

    ts::FirstOrderGeneralizedAlphaMaintenanceScheme stage_only{};
    stage_only.alpha_f = 0.5;
    const std::vector<Real> half = {0.5};
    EXPECT_EQ(ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                  stage_only, Closure::PreservePriorStateAndRate, half, one)
                  .status,
              Status::MissingRateParameters);

    ts::FirstOrderGeneralizedAlphaMaintenanceScheme near_endpoint{};
    near_endpoint.alpha_f = 1.0 - 1.0e-9;
    EXPECT_EQ(ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                  near_endpoint, Closure::SameRepresentationDelta, zero, one)
                  .status,
              Status::NearSingularStageInversion);
}

TEST(FieldRepairPublication, PlannerRejectsInvalidInputs)
{
    const std::vector<Real> one = {1.0};
    const std::vector<Real> two = {1.0, 2.0};
    const std::vector<Real> empty{};
    const std::vector<Real> nan = {std::numeric_limits<Real>::quiet_NaN()};
    EXPECT_THROW((void)ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                     midpointScheme(), Closure::SameRepresentationDelta, one, two),
                 std::invalid_argument);
    EXPECT_THROW((void)ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                     midpointScheme(), Closure::SameRepresentationDelta, empty, empty),
                 std::invalid_argument);
    EXPECT_THROW((void)ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                     midpointScheme(), Closure::SameRepresentationDelta, nan, one),
                 std::invalid_argument);
    auto partial = midpointScheme();
    partial.dt.reset();  // rate parameters must be all present or all absent
    EXPECT_THROW((void)ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
                     partial, Closure::SameRepresentationDelta, one, one),
                 std::invalid_argument);
}

TEST(FieldRepairPublication, CommonDeltaAddsToEveryStoredState)
{
    const std::vector<Real> delta = {1.0, 2.0};
    const auto plan = ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
        midpointScheme(), Closure::SameRepresentationDelta, delta, delta);
    const std::vector<Real> common = {0.5, -0.5};
    const auto combined =
        ts::combineFirstOrderGeneralizedAlphaPublicationWithCommonDelta(plan, common);
    EXPECT_EQ(combined.u_delta, (std::vector<Real>{1.5, 1.5}));
    EXPECT_EQ(combined.u_prev_delta, (std::vector<Real>{1.5, 1.5}));
    EXPECT_EQ(combined.u_prev2_and_deeper_delta, (std::vector<Real>{1.5, 1.5}));

    const std::vector<Real> one = {1.0};
    const auto incomplete = ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
        midpointScheme(), Closure::SameRepresentationDelta, std::vector<Real>{0.9}, one);
    EXPECT_THROW((void)ts::combineFirstOrderGeneralizedAlphaPublicationWithCommonDelta(incomplete, one),
                 std::invalid_argument);
}

TEST(FieldRepairPublication, StagingAppliesFieldSlicesInFieldOrder)
{
    auto staged = ts::makeStagedPostacceptMaintenanceHistoryPublication(
        std::vector<Real>(4, 0.0),
        {std::vector<Real>(4, 10.0), std::vector<Real>(4, 20.0)});
    const std::vector<Real> delta = {1.0, 2.0};
    ts::stageMaintainedFieldPostAcceptPublication(
        ts::makeCommonRepresentationMaintainedFieldPublication(1, 1, delta), staged);
    EXPECT_EQ(staged.current, (std::vector<Real>{0.0, 1.0, 2.0, 0.0}));
    EXPECT_EQ(staged.history[0], (std::vector<Real>{10.0, 11.0, 12.0, 10.0}));
    EXPECT_EQ(staged.history[1], (std::vector<Real>{20.0, 21.0, 22.0, 20.0}));
    ASSERT_EQ(staged.schedule.size(), 1u);
    EXPECT_EQ(staged.schedule[0].field, 1);
    EXPECT_EQ(staged.schedule[0].field_offset, 1u);
    EXPECT_EQ(staged.schedule[0].field_count, 2u);
    EXPECT_FALSE(staged.schedule[0].publishes_u_dot);

    // A field staged out of order, or a slice past the end, changes nothing.
    const auto before = staged.current;
    EXPECT_THROW(ts::stageMaintainedFieldPostAcceptPublication(
                     ts::makeCommonRepresentationMaintainedFieldPublication(0, 0, delta), staged),
                 std::invalid_argument);
    EXPECT_THROW(ts::stageMaintainedFieldPostAcceptPublication(
                     ts::makeCommonRepresentationMaintainedFieldPublication(2, 3, delta), staged),
                 std::invalid_argument);
    EXPECT_EQ(staged.current, before);
    EXPECT_EQ(staged.schedule.size(), 1u);

    EXPECT_THROW((void)ts::makeStagedPostacceptMaintenanceHistoryPublication(
                     std::vector<Real>(4, 0.0), {std::vector<Real>(3, 0.0)}),
                 std::invalid_argument);
}

TEST(FieldRepairPublication, PublishesStagedStateHistoryAndRateOnOneRank)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);
    auto history = ts::TimeHistory::allocate(
        *factory, 4, /*history_depth=*/2, /*allocate_second_order_state=*/true);
    ASSERT_TRUE(history.hasUDotState());
    ts_test::setVectorByDof(history.u(), {1.0, 2.0, 3.0, 4.0});
    ts_test::setVectorByDof(history.uPrevK(1), {1.0, 2.0, 3.0, 4.0});
    ts_test::setVectorByDof(history.uPrevK(2), {0.5, 1.5, 2.5, 3.5});

    const auto scheme = midpointScheme();
    const std::vector<Real> endpoint = {0.2, -0.4};
    const std::vector<Real> stage = {0.5 * endpoint[0], 0.5 * endpoint[1]};
    const auto plan = ts::planFirstOrderGeneralizedAlphaMaintenancePublication(
        scheme, Closure::PreservePriorStateAndRate, stage, endpoint);
    ASSERT_EQ(plan.status, Status::AlgebraicallyComplete) << plan.diagnostic;

    const svmp::FE::systems::CollectiveCommunicator one_rank{};
    auto staged = ts::makeStagedPostacceptMaintenanceHistoryPublication(
        svmp::FE::systems::captureFeOrderedVectorCollectively(history.u(), one_rank),
        {svmp::FE::systems::captureFeOrderedVectorCollectively(history.uPrevK(1), one_rank),
         svmp::FE::systems::captureFeOrderedVectorCollectively(history.uPrevK(2), one_rank)},
        svmp::FE::systems::captureFeOrderedVectorCollectively(history.uDot(), one_rank));
    ts::stageMaintainedFieldPostAcceptPublication(
        ts::makeFirstOrderGeneralizedAlphaMaintainedFieldPublication(
            /*field=*/0, /*field_offset=*/2, plan, /*publish_u_dot=*/true,
            /*maintained_field_has_time_derivative=*/true),
        staged);

    bool rate_publication_started = false;
    ts::publishStagedPostacceptMaintenanceHistoryCollectively(
        history, staged, one_rank, &rate_publication_started);
    EXPECT_TRUE(rate_publication_started);
    EXPECT_EQ(ts_test::getVectorByDof(history.u()),
              (std::vector<Real>{1.0, 2.0, 3.0 + 0.2, 4.0 - 0.4}));
    EXPECT_EQ(ts_test::getVectorByDof(history.uPrevK(1)),
              (std::vector<Real>{1.0, 2.0, 3.0 + 0.2, 4.0 - 0.4}));
    EXPECT_EQ(ts_test::getVectorByDof(history.uPrevK(2)),
              (std::vector<Real>{0.5, 1.5, 2.5, 3.5}));
    const Real rate_scale = *scheme.gamma * *scheme.dt;
    EXPECT_EQ(ts_test::getVectorByDof(history.uDot()),
              (std::vector<Real>{0.0, 0.0, 0.2 / rate_scale, -0.4 / rate_scale}));
}
