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

#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace ts = svmp::FE::timestepping;
namespace ts_test = svmp::FE::timestepping::test;
using svmp::FE::Real;

} // namespace

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

    // An explicit field mapping: the two endpoint copies and the endpoint
    // rate change, the older state does not.
    const std::vector<Real> endpoint = {0.2, -0.4};
    const std::vector<Real> unchanged = {0.0, 0.0};
    const std::vector<Real> rate = {2.0, -4.0};

    const svmp::FE::systems::CollectiveCommunicator one_rank{};
    auto staged = ts::makeStagedPostacceptMaintenanceHistoryPublication(
        svmp::FE::systems::captureFeOrderedVectorCollectively(history.u(), one_rank),
        {svmp::FE::systems::captureFeOrderedVectorCollectively(history.uPrevK(1), one_rank),
         svmp::FE::systems::captureFeOrderedVectorCollectively(history.uPrevK(2), one_rank)},
        svmp::FE::systems::captureFeOrderedVectorCollectively(history.uDot(), one_rank));
    ts::stageMaintainedFieldPostAcceptPublication(
        ts::MaintainedFieldPostAcceptPublicationView{
            .field = 0,
            .field_offset = 2,
            .u_delta = endpoint,
            .u_prev_delta = endpoint,
            .u_prev2_and_deeper_delta = unchanged,
            .u_dot_delta = std::span<const Real>(rate),
        },
        staged);
    ASSERT_EQ(staged.schedule.size(), 1u);
    EXPECT_TRUE(staged.schedule.front().publishes_u_dot);

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
    EXPECT_EQ(ts_test::getVectorByDof(history.uDot()),
              (std::vector<Real>{0.0, 0.0, 2.0, -4.0}));
}
