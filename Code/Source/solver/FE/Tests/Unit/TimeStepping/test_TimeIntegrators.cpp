/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/TimeIntegrationContext.h"
#include "Core/Types.h"
#include "Systems/SystemState.h"

#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/NewmarkBeta.h"

#include <vector>

TEST(TimeIntegrators, NewmarkBetaBuildContextMatchesDocumentation)
{
    using svmp::FE::Real;

    svmp::FE::timestepping::NewmarkBetaIntegrator integ({.beta = 0.25, .gamma = 0.5});

    const double dt = 0.1;
    const std::vector<Real> u(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<Real> u_prev2(4, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev, u_prev2};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;

    const auto ctx = integ.buildContext(/*max_time_derivative_order=*/2, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    ASSERT_TRUE(ctx.dt2.has_value());

    EXPECT_EQ(ctx.dt1->order, 1);
    ASSERT_EQ(ctx.dt1->a.size(), 3u);
    EXPECT_NEAR(ctx.dt1->a[0], 0.5 / (0.25 * dt), 1e-15);
    EXPECT_NEAR(ctx.dt1->a[1], 0.0, 1e-15);
    EXPECT_NEAR(ctx.dt1->a[2], 1.0, 1e-15);

    EXPECT_EQ(ctx.dt2->order, 2);
    ASSERT_EQ(ctx.dt2->a.size(), 2u);
    EXPECT_NEAR(ctx.dt2->a[0], 1.0 / (0.25 * dt * dt), 1e-12);
    EXPECT_NEAR(ctx.dt2->a[1], 1.0, 1e-15);
}

TEST(TimeIntegrators, GeneralizedAlphaSecondOrderBuildContextMatchesDocumentation)
{
    using svmp::FE::Real;

    svmp::FE::timestepping::GeneralizedAlphaSecondOrderIntegrator integ({
        .alpha_m = 0.5,
        .alpha_f = 0.5,
        .beta = 0.25,
        .gamma = 0.5,
    });

    const double dt = 0.1;
    const std::vector<Real> u(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<Real> u_prev2(4, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev, u_prev2};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;

    const auto ctx = integ.buildContext(/*max_time_derivative_order=*/2, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    ASSERT_TRUE(ctx.dt2.has_value());

    EXPECT_EQ(ctx.dt1->order, 1);
    ASSERT_EQ(ctx.dt1->a.size(), 3u);
    EXPECT_NEAR(ctx.dt1->a[0], 0.5 / (0.25 * dt), 1e-15);
    EXPECT_NEAR(ctx.dt1->a[2], 1.0, 1e-15);

    EXPECT_EQ(ctx.dt2->order, 2);
    ASSERT_EQ(ctx.dt2->a.size(), 2u);
    EXPECT_NEAR(ctx.dt2->a[0], 0.5 / (0.5 * 0.25 * dt * dt), 1e-12);
    EXPECT_NEAR(ctx.dt2->a[1], 1.0, 1e-15);
}

TEST(TimeIntegrators, GeneralizedAlphaFirstOrderBuildContextProducesExpectedStencilSizeAndCoeffs)
{
    using svmp::FE::Real;

    svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator integ({
        .alpha_m = 0.5,
        .alpha_f = 0.5,
        .gamma = 0.5,
        .history_rate_order = 2,
    });

    const double dt = 0.1;
    const std::vector<Real> u(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<Real> u_prev2(4, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev, u_prev2};
    const std::vector<double> dt_hist = {dt, dt};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    const auto ctx = integ.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    EXPECT_EQ(ctx.dt1->order, 1);

    // With u_history depth 2, the integrator clamps q=1 (1-step history rate reconstruction).
    ASSERT_EQ(ctx.dt1->a.size(), 3u);
    EXPECT_NEAR(ctx.dt1->a[0], 20.0, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[1], -20.0, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[2], 0.0, 1e-12);
}

