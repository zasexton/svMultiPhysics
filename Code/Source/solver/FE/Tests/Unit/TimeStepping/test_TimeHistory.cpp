/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"
#include "TimeStepping/TimeHistory.h"

#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <cmath>
#include <limits>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

TEST(TimeHistory, AllocateCreatesVectorsAndDtHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    const svmp::FE::GlobalIndex n = 4;
    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, n, /*history_depth=*/3, /*allocate_second_order_state=*/false);
    EXPECT_EQ(h.historyDepth(), 3);
    EXPECT_FALSE(h.hasSecondOrderState());

    EXPECT_EQ(h.u().size(), n);
    EXPECT_EQ(h.uPrev().size(), n);
    EXPECT_EQ(h.uPrev2().size(), n);
    EXPECT_EQ(h.uPrevK(3).size(), n);
    EXPECT_EQ(h.dtHistory().size(), static_cast<std::size_t>(h.historyDepth()));
}

TEST(TimeHistory, EnsureSecondOrderStateAllocatesZeroedVectors)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/2, /*allocate_second_order_state=*/false);
    EXPECT_FALSE(h.hasSecondOrderState());
    h.ensureSecondOrderState(*factory);
    EXPECT_TRUE(h.hasSecondOrderState());
    EXPECT_EQ(h.uDot().size(), 4);
    EXPECT_EQ(h.uDDot().size(), 4);
    EXPECT_DOUBLE_EQ(h.uDot().norm(), 0.0);
    EXPECT_DOUBLE_EQ(h.uDDot().norm(), 0.0);
}

TEST(TimeHistory, PrevKValidatesBounds)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/2);
    EXPECT_THROW((void)h.uPrevK(0), svmp::FE::InvalidArgumentException);
    EXPECT_THROW((void)h.uPrevK(3), svmp::FE::InvalidArgumentException);
}

TEST(TimeHistory, DtHistoryValidationAndPriming)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/3);
    EXPECT_FALSE(h.dtHistoryIsValid(1));
    EXPECT_FALSE(h.dtHistoryIsValid(3));
    EXPECT_TRUE(h.dtHistoryIsValid(0));

    h.setDt(0.2);
    h.setPrevDt(0.2);
    h.primeDtHistory(/*dt_default=*/0.0);
    EXPECT_TRUE(h.dtHistoryIsValid(3));
    EXPECT_NEAR(h.dtHistory()[0], 0.2, 1e-15);
    EXPECT_NEAR(h.dtHistory()[1], 0.2, 1e-15);
    EXPECT_NEAR(h.dtHistory()[2], 0.2, 1e-15);

    EXPECT_THROW(h.setDtHistory(std::vector<double>{0.1, 0.1}), svmp::FE::InvalidArgumentException);
}

TEST(TimeHistory, AcceptStepShiftsHistoryAndUpdatesTime)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 3, /*history_depth=*/3);
    h.setTime(1.0);
    h.setDt(0.1);
    h.setPrevDt(0.1);
    h.setStepIndex(7);
    h.primeDtHistory(0.1);

    const std::vector<svmp::FE::Real> u_prev = {1.0, 2.0, 3.0};
    const std::vector<svmp::FE::Real> u_prev2 = {4.0, 5.0, 6.0};
    const std::vector<svmp::FE::Real> u_prev3 = {7.0, 8.0, 9.0};
    const std::vector<svmp::FE::Real> u_cur = {10.0, 20.0, 30.0};

    ts_test::setVectorByDof(h.uPrev(), u_prev);
    ts_test::setVectorByDof(h.uPrev2(), u_prev2);
    ts_test::setVectorByDof(h.uPrevK(3), u_prev3);
    ts_test::setVectorByDof(h.u(), u_cur);

    h.acceptStep(/*accepted_dt=*/0.25);

    EXPECT_EQ(h.stepIndex(), 8);
    EXPECT_NEAR(h.time(), 1.25, 1e-15);
    EXPECT_NEAR(h.dt(), 0.25, 1e-15);
    EXPECT_NEAR(h.dtPrev(), 0.25, 1e-15);
    ASSERT_EQ(h.dtHistory().size(), 3u);
    EXPECT_NEAR(h.dtHistory()[0], 0.25, 1e-15);

    EXPECT_EQ(ts_test::getVectorByDof(h.uPrev()), u_cur);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrev2()), u_prev);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrevK(3)), u_prev2);
}

TEST(TimeHistory, ResetCurrentToPreviousCopiesMostRecentHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/2);

    const std::vector<svmp::FE::Real> u_prev = {1.0, -2.0, 3.0, -4.0};
    const std::vector<svmp::FE::Real> u_cur = {0.1, 0.2, 0.3, 0.4};
    ts_test::setVectorByDof(h.uPrev(), u_prev);
    ts_test::setVectorByDof(h.u(), u_cur);

    h.resetCurrentToPrevious();
    EXPECT_EQ(ts_test::getVectorByDof(h.u()), u_prev);
}

TEST(TimeHistory, DtHistoryIsValidDetectsMixedValidAndInvalidEntries)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/3);
    h.setDtHistory(std::vector<double>{0.1, 0.0, 0.1});

    EXPECT_TRUE(h.dtHistoryIsValid(1));
    EXPECT_FALSE(h.dtHistoryIsValid(2));
    EXPECT_FALSE(h.dtHistoryIsValid(3));
}

TEST(TimeHistory, UpdateGhostsDoesNotChangeVectorValuesInSerial)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/3, /*allocate_second_order_state=*/true);

    const std::vector<svmp::FE::Real> u = {0.1, 0.2, 0.3, 0.4};
    const std::vector<svmp::FE::Real> u_prev = {1.0, 2.0, 3.0, 4.0};
    const std::vector<svmp::FE::Real> u_prev2 = {-1.0, -2.0, -3.0, -4.0};
    const std::vector<svmp::FE::Real> u_prev3 = {9.0, 8.0, 7.0, 6.0};
    const std::vector<svmp::FE::Real> v = {5.0, 6.0, 7.0, 8.0};
    const std::vector<svmp::FE::Real> a = {-5.0, -6.0, -7.0, -8.0};

    ts_test::setVectorByDof(h.u(), u);
    ts_test::setVectorByDof(h.uPrev(), u_prev);
    ts_test::setVectorByDof(h.uPrev2(), u_prev2);
    ts_test::setVectorByDof(h.uPrevK(3), u_prev3);
    ts_test::setVectorByDof(h.uDot(), v);
    ts_test::setVectorByDof(h.uDDot(), a);

    h.updateGhosts();

    EXPECT_EQ(ts_test::getVectorByDof(h.u()), u);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrev()), u_prev);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrev2()), u_prev2);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrevK(3)), u_prev3);
    EXPECT_EQ(ts_test::getVectorByDof(h.uDot()), v);
    EXPECT_EQ(ts_test::getVectorByDof(h.uDDot()), a);
}

TEST(TimeHistory, RepackPreservesValues)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeHistory tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    auto h = svmp::FE::timestepping::TimeHistory::allocate(*factory, 4, /*history_depth=*/3, /*allocate_second_order_state=*/true);

    const std::vector<svmp::FE::Real> u = {0.1, 0.2, 0.3, 0.4};
    const std::vector<svmp::FE::Real> u_prev = {1.0, 2.0, 3.0, 4.0};
    const std::vector<svmp::FE::Real> u_prev2 = {-1.0, -2.0, -3.0, -4.0};
    const std::vector<svmp::FE::Real> u_prev3 = {9.0, 8.0, 7.0, 6.0};
    const std::vector<svmp::FE::Real> v = {5.0, 6.0, 7.0, 8.0};
    const std::vector<svmp::FE::Real> a = {-5.0, -6.0, -7.0, -8.0};

    ts_test::setVectorByDof(h.u(), u);
    ts_test::setVectorByDof(h.uPrev(), u_prev);
    ts_test::setVectorByDof(h.uPrev2(), u_prev2);
    ts_test::setVectorByDof(h.uPrevK(3), u_prev3);
    ts_test::setVectorByDof(h.uDot(), v);
    ts_test::setVectorByDof(h.uDDot(), a);

    h.repack(*factory);

    EXPECT_EQ(ts_test::getVectorByDof(h.u()), u);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrev()), u_prev);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrev2()), u_prev2);
    EXPECT_EQ(ts_test::getVectorByDof(h.uPrevK(3)), u_prev3);
    EXPECT_EQ(ts_test::getVectorByDof(h.uDot()), v);
    EXPECT_EQ(ts_test::getVectorByDof(h.uDDot()), a);
}
