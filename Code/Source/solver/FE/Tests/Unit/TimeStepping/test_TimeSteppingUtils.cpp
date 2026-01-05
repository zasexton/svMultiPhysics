/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Core/FEException.h"

#include "TimeStepping/TimeHistory.h"
#include "TimeStepping/TimeSteppingUtils.h"

#include "Tests/Unit/TimeStepping/TimeSteppingTestHelpers.h"

#include <cmath>
#include <limits>
#include <vector>

namespace ts_test = svmp::FE::timestepping::test;

TEST(TimeSteppingUtils, ClampDtRespectsBounds)
{
    using svmp::FE::timestepping::utils::clampDt;
    EXPECT_DOUBLE_EQ(clampDt(1.0, 0.0, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(clampDt(1.0, 0.5, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(clampDt(0.25, 0.5, 0.0), 0.5);
    EXPECT_DOUBLE_EQ(clampDt(2.0, 0.0, 1.5), 1.5);
    EXPECT_DOUBLE_EQ(clampDt(2.0, 0.5, 1.5), 1.5);
    EXPECT_DOUBLE_EQ(clampDt(1.0, 0.5, 1.5), 1.0);
}

TEST(TimeSteppingUtils, GeneralizedAlphaFirstOrderFromRhoInfMatchesReferenceEndpoints)
{
    using svmp::FE::timestepping::utils::generalizedAlphaFirstOrderFromRhoInf;

    {
        const auto p = generalizedAlphaFirstOrderFromRhoInf(1.0);
        EXPECT_NEAR(p.alpha_m, 0.5, 1e-15);
        EXPECT_NEAR(p.alpha_f, 0.5, 1e-15);
        EXPECT_NEAR(p.gamma, 0.5, 1e-15);
    }
    {
        const auto p = generalizedAlphaFirstOrderFromRhoInf(0.0);
        EXPECT_NEAR(p.alpha_m, 1.5, 1e-15);
        EXPECT_NEAR(p.alpha_f, 1.0, 1e-15);
        EXPECT_NEAR(p.gamma, 1.0, 1e-15);
    }

    EXPECT_THROW(generalizedAlphaFirstOrderFromRhoInf(-0.1), svmp::FE::InvalidArgumentException);
    EXPECT_THROW(generalizedAlphaFirstOrderFromRhoInf(1.1), svmp::FE::InvalidArgumentException);
    EXPECT_THROW(generalizedAlphaFirstOrderFromRhoInf(std::numeric_limits<double>::infinity()), svmp::FE::InvalidArgumentException);
    EXPECT_THROW(generalizedAlphaFirstOrderFromRhoInf(std::numeric_limits<double>::quiet_NaN()), svmp::FE::InvalidArgumentException);
}

TEST(TimeSteppingUtils, GeneralizedAlphaSecondOrderFromRhoInfMatchesReferenceEndpoints)
{
    using svmp::FE::timestepping::utils::generalizedAlphaSecondOrderFromRhoInf;

    {
        const auto p = generalizedAlphaSecondOrderFromRhoInf(1.0);
        EXPECT_NEAR(p.alpha_m, 0.5, 1e-15);
        EXPECT_NEAR(p.alpha_f, 0.5, 1e-15);
        EXPECT_NEAR(p.gamma, 0.5, 1e-15);
        EXPECT_NEAR(p.beta, 0.25, 1e-15);
    }
    {
        const auto p = generalizedAlphaSecondOrderFromRhoInf(0.0);
        EXPECT_NEAR(p.alpha_m, 2.0, 1e-15);
        EXPECT_NEAR(p.alpha_f, 1.0, 1e-15);
        EXPECT_NEAR(p.gamma, 1.5, 1e-15);
        EXPECT_NEAR(p.beta, 1.0, 1e-15);
    }

    EXPECT_THROW(generalizedAlphaSecondOrderFromRhoInf(-0.1), svmp::FE::InvalidArgumentException);
    EXPECT_THROW(generalizedAlphaSecondOrderFromRhoInf(1.1), svmp::FE::InvalidArgumentException);
    EXPECT_THROW(generalizedAlphaSecondOrderFromRhoInf(std::numeric_limits<double>::infinity()), svmp::FE::InvalidArgumentException);
    EXPECT_THROW(generalizedAlphaSecondOrderFromRhoInf(std::numeric_limits<double>::quiet_NaN()), svmp::FE::InvalidArgumentException);
}

TEST(TimeSteppingUtils, InitializeSecondOrderStateFromDisplacementHistoryUsesAvailableHistory)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeSteppingUtils tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using svmp::FE::timestepping::utils::initializeSecondOrderStateFromDisplacementHistory;

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    constexpr double dt = 0.1;
    const std::vector<svmp::FE::Real> coeff = {1.0, -0.5, 0.25, 2.0};

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 static_cast<svmp::FE::GlobalIndex>(coeff.size()),
                                                                 /*history_depth=*/4,
                                                                 /*allocate_second_order_state=*/true);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setDtHistory(std::vector<double>{dt, dt, dt, dt});

    auto uAt = [&](double t) {
        std::vector<svmp::FE::Real> u(coeff.size(), 0.0);
        for (std::size_t i = 0; i < u.size(); ++i) {
            u[i] = coeff[i] * static_cast<svmp::FE::Real>(t * t);
        }
        return u;
    };

    ts_test::setVectorByDof(history.uPrevK(1), uAt(0.0));
    ts_test::setVectorByDof(history.uPrevK(2), uAt(-dt));
    ts_test::setVectorByDof(history.uPrevK(3), uAt(-2.0 * dt));
    ts_test::setVectorByDof(history.uPrevK(4), uAt(-3.0 * dt));

    history.uDot().zero();
    history.uDDot().zero();

    const auto rep = initializeSecondOrderStateFromDisplacementHistory(
        history,
        history.uDot().localSpan(),
        history.uDDot().localSpan(),
        /*overwrite_u_dot=*/true,
        /*overwrite_u_ddot=*/true,
        /*max_points=*/6);

    EXPECT_TRUE(rep.initialized_velocity);
    EXPECT_TRUE(rep.initialized_acceleration);
    EXPECT_GE(rep.velocity_points, 3);
    EXPECT_GE(rep.acceleration_points, 3);

    // u(t) = c*t^2 => u'(0)=0, u''(0)=2c
    const auto v = history.uDot().localSpan();
    const auto a = history.uDDot().localSpan();
    ASSERT_EQ(v.size(), coeff.size());
    ASSERT_EQ(a.size(), coeff.size());
    for (std::size_t i = 0; i < coeff.size(); ++i) {
        EXPECT_NEAR(v[i], 0.0, 1e-12);
        EXPECT_NEAR(a[i], static_cast<svmp::FE::Real>(2.0) * coeff[i], 1e-10);
    }
}

TEST(TimeSteppingUtils, InitializeSecondOrderStateFromDisplacementHistoryRespectsMaxPoints)
{
#if !defined(FE_HAS_EIGEN) || !FE_HAS_EIGEN
    GTEST_SKIP() << "TimeSteppingUtils tests require the Eigen backend (enable FE_ENABLE_EIGEN)";
#endif
    using svmp::FE::timestepping::utils::initializeSecondOrderStateFromDisplacementHistory;

    auto factory = ts_test::createTestFactory();
    ASSERT_NE(factory.get(), nullptr);

    constexpr double dt = 0.1;
    const std::vector<svmp::FE::Real> coeff = {1.0, -0.5, 0.25, 2.0};

    auto history = svmp::FE::timestepping::TimeHistory::allocate(*factory,
                                                                 static_cast<svmp::FE::GlobalIndex>(coeff.size()),
                                                                 /*history_depth=*/7,
                                                                 /*allocate_second_order_state=*/true);
    history.setDt(dt);
    history.setPrevDt(dt);
    history.setDtHistory(std::vector<double>(static_cast<std::size_t>(history.historyDepth()), dt));

    auto uAt = [&](double t) {
        std::vector<svmp::FE::Real> u(coeff.size(), 0.0);
        for (std::size_t i = 0; i < u.size(); ++i) {
            u[i] = coeff[i] * static_cast<svmp::FE::Real>(t * t);
        }
        return u;
    };

    for (int k = 1; k <= history.historyDepth(); ++k) {
        const double t = -static_cast<double>(k - 1) * dt;
        ts_test::setVectorByDof(history.uPrevK(k), uAt(t));
    }

    history.uDot().zero();
    history.uDDot().zero();

    const auto rep = initializeSecondOrderStateFromDisplacementHistory(
        history,
        history.uDot().localSpan(),
        history.uDDot().localSpan(),
        /*overwrite_u_dot=*/true,
        /*overwrite_u_ddot=*/true,
        /*max_points=*/3);

    EXPECT_TRUE(rep.initialized_velocity);
    EXPECT_TRUE(rep.initialized_acceleration);
    EXPECT_EQ(rep.velocity_points, 3);
    EXPECT_EQ(rep.acceleration_points, 3);
}
