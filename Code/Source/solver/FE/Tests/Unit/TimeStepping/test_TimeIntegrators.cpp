/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "Assembly/TimeIntegrationContext.h"
#include "Core/FEException.h"
#include "Core/Types.h"
#include "Systems/SystemState.h"
#include "Systems/TimeIntegrator.h"

#include "TimeStepping/GeneralizedAlpha.h"
#include "TimeStepping/NewmarkBeta.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

std::vector<double> solveLinearSystem(std::vector<double> A, std::vector<double> b, int n)
{
    // Naive Gaussian elimination with partial pivoting (n <= 6 in these tests).
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double piv_abs = std::abs(A[static_cast<std::size_t>(k * n + k)]);
        for (int r = k + 1; r < n; ++r) {
            const double a = std::abs(A[static_cast<std::size_t>(r * n + k)]);
            if (a > piv_abs) {
                piv_abs = a;
                piv = r;
            }
        }
        if (!(piv_abs > 0.0)) {
            throw std::runtime_error("solveLinearSystem: singular matrix");
        }
        if (piv != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(A[static_cast<std::size_t>(k * n + j)], A[static_cast<std::size_t>(piv * n + j)]);
            }
            std::swap(b[static_cast<std::size_t>(k)], b[static_cast<std::size_t>(piv)]);
        }

        const double diag = A[static_cast<std::size_t>(k * n + k)];
        for (int j = k; j < n; ++j) {
            A[static_cast<std::size_t>(k * n + j)] /= diag;
        }
        b[static_cast<std::size_t>(k)] /= diag;

        for (int r = k + 1; r < n; ++r) {
            const double fac = A[static_cast<std::size_t>(r * n + k)];
            if (fac == 0.0) continue;
            for (int j = k; j < n; ++j) {
                A[static_cast<std::size_t>(r * n + j)] -= fac * A[static_cast<std::size_t>(k * n + j)];
            }
            b[static_cast<std::size_t>(r)] -= fac * b[static_cast<std::size_t>(k)];
        }
    }

    // Back substitution.
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = b[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= A[static_cast<std::size_t>(i * n + j)] * x[static_cast<std::size_t>(j)];
        }
        x[static_cast<std::size_t>(i)] = sum;
    }
    return x;
}

std::vector<double> derivativeWeightsAtZero(const std::vector<double>& nodes)
{
    const int n = static_cast<int>(nodes.size());
    EXPECT_GT(n, 0);

    // Solve for weights w such that:
    //   sum_j w_j * x_j^k = d/dx x^k |_{x=0}
    // for k = 0..n-1.
    std::vector<double> A(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            double pow = 1.0;
            for (int p = 0; p < k; ++p) {
                pow *= nodes[static_cast<std::size_t>(j)];
            }
            A[static_cast<std::size_t>(k * n + j)] = pow;
        }
    }

    std::vector<double> rhs(static_cast<std::size_t>(n), 0.0);
    if (n >= 2) {
        rhs[1] = 1.0;
    } else {
        // With a single node, exact derivative recovery is impossible; still return {0}.
        rhs[0] = 0.0;
    }
    return solveLinearSystem(std::move(A), std::move(rhs), n);
}

} // namespace

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

TEST(TimeIntegrators, GeneralizedAlphaFirstOrderValidatesParameters)
{
    using svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator;
    using svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions;

    {
        GeneralizedAlphaFirstOrderIntegratorOptions o;
        o.alpha_m = 0.5;
        o.alpha_f = 0.0;
        o.gamma = 0.5;
        EXPECT_THROW((void)GeneralizedAlphaFirstOrderIntegrator{o}, svmp::FE::InvalidArgumentException);
    }
    {
        GeneralizedAlphaFirstOrderIntegratorOptions o;
        o.alpha_m = 0.5;
        o.alpha_f = 0.5;
        o.gamma = 0.0;
        EXPECT_THROW((void)GeneralizedAlphaFirstOrderIntegrator{o}, svmp::FE::InvalidArgumentException);
    }
    {
        GeneralizedAlphaFirstOrderIntegratorOptions o;
        o.alpha_m = 0.5;
        o.alpha_f = 0.5;
        o.gamma = 0.5;
        o.history_rate_order = -1;
        EXPECT_THROW((void)GeneralizedAlphaFirstOrderIntegrator{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(TimeIntegrators, GeneralizedAlphaFirstOrderHistoryRateOrderZeroUsesInjectedRateSlot)
{
    using svmp::FE::Real;
    using svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator;
    using svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegratorOptions;

    GeneralizedAlphaFirstOrderIntegrator integ(GeneralizedAlphaFirstOrderIntegratorOptions{
        .alpha_m = 0.6,
        .alpha_f = 0.5,
        .gamma = 0.5,
        .history_rate_order = 0,
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

    const auto ctx = integ.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    EXPECT_EQ(ctx.dt1->order, 1);
    ASSERT_EQ(ctx.dt1->a.size(), 3u);

    const double alpha_m = 0.6;
    const double alpha_f = 0.5;
    const double gamma = 0.5;
    const double c = alpha_m / (gamma * dt * alpha_f);
    const double c0 = 1.0 - alpha_m / gamma;

    EXPECT_NEAR(ctx.dt1->a[0], c, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[1], -c, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[2], c0, 1e-12);
}

TEST(TimeIntegrators, GeneralizedAlphaFirstOrderThrowsOnInsufficientHistory)
{
    using svmp::FE::Real;
    using svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator;

    GeneralizedAlphaFirstOrderIntegrator integ({
        .alpha_m = 0.6,
        .alpha_f = 0.5,
        .gamma = 0.5,
        .history_rate_order = 1,
    });

    const double dt = 0.1;
    const std::vector<Real> u(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev;
    state.u_history = u_hist;

    EXPECT_THROW((void)integ.buildContext(/*max_time_derivative_order=*/1, state), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrators, GeneralizedAlphaFirstOrderVariableDtHistoryAffectsWeights)
{
    using svmp::FE::Real;
    using svmp::FE::timestepping::GeneralizedAlphaFirstOrderIntegrator;

    GeneralizedAlphaFirstOrderIntegrator integ({
        .alpha_m = 0.6,
        .alpha_f = 0.5,
        .gamma = 0.5,
        .history_rate_order = 2,
    });

    const double dt = 0.1;
    const std::vector<Real> u(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<Real> u_prev2(4, 0.0);
    const std::vector<Real> u_prev3(4, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev, u_prev2, u_prev3};
    const std::vector<double> dt_hist = {0.1, 0.2};

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
    ASSERT_EQ(ctx.dt1->a.size(), 4u);

    // Derivative reconstruction uses nodes at {0, -h1, -(h1+h2)}, with h1=dt_prev, h2=dt_{n-1}.
    const double h1 = dt_hist[0];
    const double h2 = dt_hist[1];
    const double H = h1 + h2;
    const double w0 = (2.0 * h1 + h2) / (h1 * H);
    const double w1 = -H / (h1 * h2);
    const double w2 = h1 / (H * h2);

    const double alpha_m = 0.6;
    const double alpha_f = 0.5;
    const double gamma = 0.5;
    const double c = alpha_m / (gamma * dt * alpha_f);
    const double c0 = 1.0 - alpha_m / gamma;

    EXPECT_NEAR(ctx.dt1->a[0], c, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[1], -c + c0 * w0, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[2], c0 * w1, 1e-12);
    EXPECT_NEAR(ctx.dt1->a[3], c0 * w2, 1e-12);
}

TEST(TimeIntegrators, GeneralizedAlphaSecondOrderValidatesParameters)
{
    using svmp::FE::timestepping::GeneralizedAlphaSecondOrderIntegrator;
    using svmp::FE::timestepping::GeneralizedAlphaSecondOrderIntegratorOptions;

    {
        GeneralizedAlphaSecondOrderIntegratorOptions o;
        o.alpha_m = 0.5;
        o.alpha_f = 0.0;
        o.beta = 0.25;
        o.gamma = 0.5;
        EXPECT_THROW((void)GeneralizedAlphaSecondOrderIntegrator{o}, svmp::FE::InvalidArgumentException);
    }
    {
        GeneralizedAlphaSecondOrderIntegratorOptions o;
        o.alpha_m = 0.5;
        o.alpha_f = 0.5;
        o.beta = 0.0;
        o.gamma = 0.5;
        EXPECT_THROW((void)GeneralizedAlphaSecondOrderIntegrator{o}, svmp::FE::InvalidArgumentException);
    }
    {
        GeneralizedAlphaSecondOrderIntegratorOptions o;
        o.alpha_m = 0.5;
        o.alpha_f = 0.5;
        o.beta = 0.25;
        o.gamma = 0.0;
        EXPECT_THROW((void)GeneralizedAlphaSecondOrderIntegrator{o}, svmp::FE::InvalidArgumentException);
    }
}

TEST(TimeIntegrators, GeneralizedAlphaSecondOrderThrowsOnInsufficientHistory)
{
    using svmp::FE::Real;
    using svmp::FE::timestepping::GeneralizedAlphaSecondOrderIntegrator;

    GeneralizedAlphaSecondOrderIntegrator integ({
        .alpha_m = 0.5,
        .alpha_f = 0.5,
        .beta = 0.25,
        .gamma = 0.5,
    });

    const double dt = 0.1;
    const std::vector<Real> u(4, 0.0);
    const std::vector<Real> u_prev(4, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev;
    state.u_history = u_hist;

    EXPECT_THROW((void)integ.buildContext(/*max_time_derivative_order=*/2, state), svmp::FE::InvalidArgumentException);
}

TEST(TimeIntegrators, BDFIntegratorOrder1MatchesBackwardDifferenceEvenWithDtHistory)
{
    using svmp::FE::Real;

    svmp::FE::systems::BDFIntegrator integ(/*order=*/1);

    const double dt = 0.1;
    const std::vector<Real> u(1, 0.0);
    const std::vector<Real> u_prev(1, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev};
    const std::vector<double> dt_hist = {0.2};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = 0.2;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    const auto ctx = integ.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    ASSERT_EQ(ctx.dt1->a.size(), 2u);
    EXPECT_NEAR(ctx.dt1->a[0], 1.0 / dt, 1e-15);
    EXPECT_NEAR(ctx.dt1->a[1], -1.0 / dt, 1e-15);
}

TEST(TimeIntegrators, BDFIntegratorOrder2MatchesBDF2IntegratorOnVariableDt)
{
    using svmp::FE::Real;

    svmp::FE::systems::BDFIntegrator integ(/*order=*/2);

    const double dt = 0.1;
    const double dt_prev = 0.2;
    const std::vector<Real> u(1, 0.0);
    const std::vector<Real> u_prev(1, 0.0);
    const std::vector<Real> u_prev2(1, 0.0);
    const std::vector<std::span<const Real>> u_hist = {u_prev, u_prev2};
    const std::vector<double> dt_hist = {dt_prev};

    svmp::FE::systems::SystemStateView state;
    state.dt = dt;
    state.dt_prev = dt_prev;
    state.u = u;
    state.u_prev = u_prev;
    state.u_prev2 = u_prev2;
    state.u_history = u_hist;
    state.dt_history = dt_hist;

    const auto ctx = integ.buildContext(/*max_time_derivative_order=*/1, state);
    ASSERT_TRUE(ctx.dt1.has_value());
    ASSERT_EQ(ctx.dt1->a.size(), 3u);

    const double r = dt / dt_prev;
    const double inv_dt = 1.0 / dt;
    const std::vector<double> expected = {
        ((1.0 + 2.0 * r) / (1.0 + r)) * inv_dt,
        (-(1.0 + r)) * inv_dt,
        ((r * r) / (1.0 + r)) * inv_dt};

    EXPECT_NEAR(ctx.dt1->a[0], expected[0], 1e-14);
    EXPECT_NEAR(ctx.dt1->a[1], expected[1], 1e-14);
    EXPECT_NEAR(ctx.dt1->a[2], expected[2], 1e-14);
}

TEST(TimeIntegrators, BDFIntegratorVariableStepWeightsMatchIndependentVandermondeForOrders3To5)
{
    using svmp::FE::Real;

    struct OwnedState {
        std::vector<std::vector<Real>> vecs{};
        std::vector<std::span<const Real>> u_hist{};
        std::vector<double> dt_hist{};
        svmp::FE::systems::SystemStateView view{};
    };

    auto makeState = [](double dt, double dt_prev, const std::vector<double>& dt_hist, int history_states) {
        OwnedState out;
        out.vecs.assign(static_cast<std::size_t>(history_states + 1), std::vector<Real>(1, 0.0));
        out.u_hist.reserve(static_cast<std::size_t>(history_states));
        for (int k = 1; k <= history_states; ++k) {
            out.u_hist.push_back(out.vecs[static_cast<std::size_t>(k)]);
        }
        out.dt_hist = dt_hist;

        out.view.dt = dt;
        out.view.dt_prev = dt_prev;
        out.view.u = out.vecs[0];
        out.view.u_prev = out.vecs[1];
        out.view.u_prev2 = out.vecs[std::min(2, history_states)];
        out.view.u_history = out.u_hist;
        out.view.dt_history = out.dt_hist;
        return out;
    };

    struct Case {
        int order{0};
        double dt{0.0};
        double dt_prev{0.0};
        std::vector<double> dt_hist{};
    };

    const std::vector<Case> cases = {
        {.order = 3, .dt = 0.1, .dt_prev = 0.2, .dt_hist = {0.2, 0.15}},
        {.order = 4, .dt = 0.1, .dt_prev = 0.2, .dt_hist = {0.2, 0.15, 0.05}},
        {.order = 5, .dt = 0.1, .dt_prev = 0.2, .dt_hist = {0.2, 0.15, 0.05, 0.25}},
    };

    for (const auto& c : cases) {
        svmp::FE::systems::BDFIntegrator integ(c.order);
        auto owned = makeState(c.dt, c.dt_prev, c.dt_hist, /*history_states=*/c.order);
        const auto ctx = integ.buildContext(/*max_time_derivative_order=*/1, owned.view);
        ASSERT_TRUE(ctx.dt1.has_value());
        const int stencil_points = c.order + 1;
        ASSERT_EQ(static_cast<int>(ctx.dt1->a.size()), stencil_points);

        // Nodes used by BDFIntegrator (relative to t_{n+1}): 0, -dt, -(dt + dt_{n}), ...
        std::vector<double> nodes;
        nodes.reserve(static_cast<std::size_t>(stencil_points));
        nodes.push_back(0.0);
        double accum = 0.0;
        for (int j = 1; j < stencil_points; ++j) {
            if (j == 1) {
                accum += c.dt;
            } else {
                accum += c.dt_hist[static_cast<std::size_t>(j - 2)];
            }
            nodes.push_back(-accum);
        }

        const auto w = derivativeWeightsAtZero(nodes);
        ASSERT_EQ(static_cast<int>(w.size()), stencil_points);
        for (int j = 0; j < stencil_points; ++j) {
            EXPECT_NEAR(ctx.dt1->a[static_cast<std::size_t>(j)], w[static_cast<std::size_t>(j)], 1e-10)
                << "order=" << c.order << " j=" << j;
        }
    }
}
