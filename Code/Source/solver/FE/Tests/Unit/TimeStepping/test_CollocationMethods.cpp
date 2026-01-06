/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 */

#include <gtest/gtest.h>

#include "TimeStepping/CollocationMethods.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using svmp::FE::timestepping::collocation::CollocationFamily;
using svmp::FE::timestepping::collocation::CollocationMethod;
using svmp::FE::timestepping::collocation::SecondOrderCollocationData;

void expectVectorNear(const std::vector<double>& a, const std::vector<double>& b, double tol)
{
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "i=" << i;
    }
}

std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B, int n)
{
    std::vector<double> C(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const double aik = A[static_cast<std::size_t>(i * n + k)];
            for (int j = 0; j < n; ++j) {
                C[static_cast<std::size_t>(i * n + j)] += aik * B[static_cast<std::size_t>(k * n + j)];
            }
        }
    }
    return C;
}

} // namespace

TEST(CollocationMethods, GaussNodesMatchReference)
{
    constexpr double tol = 1e-14;

    {
        const auto c = svmp::FE::timestepping::collocation::gaussNodesUnit(1);
        expectVectorNear(c, {0.5}, tol);
    }

    {
        const auto c = svmp::FE::timestepping::collocation::gaussNodesUnit(2);
        expectVectorNear(c,
                         {0.2113248654051871177454256097490212721762,
                          0.7886751345948128822545743902509787278238},
                         tol);
    }

    {
        const auto c = svmp::FE::timestepping::collocation::gaussNodesUnit(3);
        expectVectorNear(c,
                         {0.1127016653792583114820734600217600389167,
                          0.5,
                          0.8872983346207416885179265399782399610833},
                         tol);
    }

    {
        const auto c = svmp::FE::timestepping::collocation::gaussNodesUnit(4);
        expectVectorNear(c,
                         {0.06943184420297371238802675555359524745214,
                          0.3300094782075718675986671204483776563997,
                          0.6699905217924281324013328795516223436003,
                          0.9305681557970262876119732444464047525479},
                         tol);
    }
}

TEST(CollocationMethods, RadauIIANodesMatchReference)
{
    constexpr double tol = 1e-14;

    {
        const auto c = svmp::FE::timestepping::collocation::radauIIANodesUnit(1);
        expectVectorNear(c, {1.0}, tol);
    }

    {
        const auto c = svmp::FE::timestepping::collocation::radauIIANodesUnit(2);
        expectVectorNear(c,
                         {1.0 / 3.0,
                          1.0},
                         tol);
    }

    {
        const auto c = svmp::FE::timestepping::collocation::radauIIANodesUnit(3);
        expectVectorNear(c,
                         {0.1550510257216821901802715925294108608034,
                          0.6449489742783178098197284074705891391966,
                          1.0},
                         tol);
    }

    {
        const auto c = svmp::FE::timestepping::collocation::radauIIANodesUnit(4);
        expectVectorNear(c,
                         {0.08858795951270394739554614376945619688573,
                          0.40946686444073471086492625206882989405190,
                          0.78765946176084705602524188987599962334808,
                          1.0},
                         tol);
    }
}

TEST(CollocationMethods, InvertDenseMatrixMatchesKnownInverse)
{
    const std::vector<double> A = {
        4.0, 7.0,
        2.0, 6.0};
    const auto inv = svmp::FE::timestepping::collocation::invertDenseMatrix(A, /*n=*/2);
    ASSERT_EQ(inv.size(), 4u);
    expectVectorNear(inv,
                     {0.6, -0.7,
                      -0.2, 0.4},
                     1e-15);
}

TEST(CollocationMethods, BuildCollocationMethodGauss2MatchesKnownButcherMatrix)
{
    const CollocationMethod method = svmp::FE::timestepping::collocation::buildCollocationMethod(CollocationFamily::Gauss, 2);

    EXPECT_EQ(method.stages, 2);
    EXPECT_EQ(method.order, 4);
    EXPECT_FALSE(method.stiffly_accurate);
    EXPECT_EQ(method.final_stage, 0);

    ASSERT_EQ(method.c.size(), 2u);
    ASSERT_EQ(method.ainv.size(), 4u);
    ASSERT_EQ(method.row_sums.size(), 2u);
    ASSERT_EQ(method.final_w.size(), 2u);

    // Gauss(2) Butcher tableau on [0,1].
    const double s3 = std::sqrt(3.0);
    const std::vector<double> A = {
        0.25, 0.25 - s3 / 6.0,
        0.25 + s3 / 6.0, 0.25};
    const auto I = matmul(A, method.ainv, /*n=*/2);
    expectVectorNear(I, {1.0, 0.0, 0.0, 1.0}, 1e-12);

    // row_sums is a cached post-process of A^{-1}.
    for (int i = 0; i < method.stages; ++i) {
        const double sum = method.ainv[static_cast<std::size_t>(i * method.stages + 0)]
            + method.ainv[static_cast<std::size_t>(i * method.stages + 1)];
        EXPECT_NEAR(method.row_sums[static_cast<std::size_t>(i)], sum, 1e-15);
    }

    // final_w = b^T * A^{-1} (for Gauss methods).
    const std::vector<double> b = {0.5, 0.5};
    std::vector<double> expected_final_w(2, 0.0);
    for (int j = 0; j < method.stages; ++j) {
        for (int i = 0; i < method.stages; ++i) {
            expected_final_w[static_cast<std::size_t>(j)] += b[static_cast<std::size_t>(i)]
                * method.ainv[static_cast<std::size_t>(i * method.stages + j)];
        }
    }
    expectVectorNear(method.final_w, expected_final_w, 1e-14);

    // Linear exactness for the end-of-step reconstruction: u_{n+1} - u_n = dt*u'
    // when U_j - u_n = c_j*dt*u'.
    const double sum_cw = method.final_w[0] * method.c[0] + method.final_w[1] * method.c[1];
    EXPECT_NEAR(sum_cw, 1.0, 1e-12);
}

TEST(CollocationMethods, BuildCollocationMethodRadau2MatchesKnownButcherMatrix)
{
    const CollocationMethod method = svmp::FE::timestepping::collocation::buildCollocationMethod(CollocationFamily::RadauIIA, 2);

    EXPECT_EQ(method.stages, 2);
    EXPECT_EQ(method.order, 3);
    EXPECT_TRUE(method.stiffly_accurate);
    EXPECT_EQ(method.final_stage, 1);

    ASSERT_EQ(method.c.size(), 2u);
    ASSERT_EQ(method.ainv.size(), 4u);
    ASSERT_EQ(method.row_sums.size(), 2u);
    EXPECT_TRUE(method.final_w.empty());

    // Radau IIA(2) Butcher tableau on [0,1].
    const std::vector<double> A = {
        5.0 / 12.0, -1.0 / 12.0,
        3.0 / 4.0, 1.0 / 4.0};
    const auto I = matmul(A, method.ainv, /*n=*/2);
    expectVectorNear(I, {1.0, 0.0, 0.0, 1.0}, 1e-12);
}

TEST(CollocationSecondOrderData, ReconstructsDerivativesForCubicGauss2)
{
    const CollocationMethod method = svmp::FE::timestepping::collocation::buildCollocationMethod(CollocationFamily::Gauss, 2);
    const SecondOrderCollocationData data = svmp::FE::timestepping::collocation::buildSecondOrderCollocationData(method);

    ASSERT_EQ(data.stages, 2);
    ASSERT_EQ(data.n_constraints, 4);
    ASSERT_EQ(data.d1.size(), 4u);
    ASSERT_EQ(data.d2.size(), 4u);

    const double a0 = 1.2;
    const double a1 = -0.3;
    const double a2 = 0.7;
    const double a3 = 2.1;
    auto p = [&](double t) { return a0 + a1 * t + a2 * t * t + a3 * t * t * t; };
    auto dp = [&](double t) { return a1 + 2.0 * a2 * t + 3.0 * a3 * t * t; };
    auto ddp = [&](double t) { return 2.0 * a2 + 6.0 * a3 * t; };

    const double u0 = p(0.0);
    const double dv0 = dp(0.0);

    std::vector<double> U(method.stages, 0.0);
    for (int j = 0; j < method.stages; ++j) {
        U[static_cast<std::size_t>(j)] = p(method.c[static_cast<std::size_t>(j)]);
    }

    for (int i = 0; i < method.stages; ++i) {
        double p1 = data.d1_u0[static_cast<std::size_t>(i)] * u0 + data.d1_dv0[static_cast<std::size_t>(i)] * dv0;
        double p2 = data.d2_u0[static_cast<std::size_t>(i)] * u0 + data.d2_dv0[static_cast<std::size_t>(i)] * dv0;
        for (int j = 0; j < method.stages; ++j) {
            p1 += data.d1[static_cast<std::size_t>(i * method.stages + j)] * U[static_cast<std::size_t>(j)];
            p2 += data.d2[static_cast<std::size_t>(i * method.stages + j)] * U[static_cast<std::size_t>(j)];
        }
        const double ci = method.c[static_cast<std::size_t>(i)];
        EXPECT_NEAR(p1, dp(ci), 1e-12);
        EXPECT_NEAR(p2, ddp(ci), 1e-12);
    }

    double u1 = data.u1_u0 * u0 + data.u1_dv0 * dv0;
    double du1 = data.du1_u0 * u0 + data.du1_dv0 * dv0;
    double ddu1 = data.ddu1_u0 * u0 + data.ddu1_dv0 * dv0;
    for (int j = 0; j < method.stages; ++j) {
        u1 += data.u1[static_cast<std::size_t>(j)] * U[static_cast<std::size_t>(j)];
        du1 += data.du1[static_cast<std::size_t>(j)] * U[static_cast<std::size_t>(j)];
        ddu1 += data.ddu1[static_cast<std::size_t>(j)] * U[static_cast<std::size_t>(j)];
    }

    EXPECT_NEAR(u1, p(1.0), 1e-12);
    EXPECT_NEAR(du1, dp(1.0), 1e-12);
    EXPECT_NEAR(ddu1, ddp(1.0), 1e-12);
}

TEST(CollocationSecondOrderData, ReconstructsDerivativesForCubicRadau2)
{
    const CollocationMethod method = svmp::FE::timestepping::collocation::buildCollocationMethod(CollocationFamily::RadauIIA, 2);
    const SecondOrderCollocationData data = svmp::FE::timestepping::collocation::buildSecondOrderCollocationData(method);

    const double a0 = -0.8;
    const double a1 = 2.4;
    const double a2 = -0.1;
    const double a3 = 0.9;
    auto p = [&](double t) { return a0 + a1 * t + a2 * t * t + a3 * t * t * t; };
    auto dp = [&](double t) { return a1 + 2.0 * a2 * t + 3.0 * a3 * t * t; };
    auto ddp = [&](double t) { return 2.0 * a2 + 6.0 * a3 * t; };

    const double u0 = p(0.0);
    const double dv0 = dp(0.0);

    std::vector<double> U(method.stages, 0.0);
    for (int j = 0; j < method.stages; ++j) {
        U[static_cast<std::size_t>(j)] = p(method.c[static_cast<std::size_t>(j)]);
    }

    for (int i = 0; i < method.stages; ++i) {
        double p1 = data.d1_u0[static_cast<std::size_t>(i)] * u0 + data.d1_dv0[static_cast<std::size_t>(i)] * dv0;
        double p2 = data.d2_u0[static_cast<std::size_t>(i)] * u0 + data.d2_dv0[static_cast<std::size_t>(i)] * dv0;
        for (int j = 0; j < method.stages; ++j) {
            p1 += data.d1[static_cast<std::size_t>(i * method.stages + j)] * U[static_cast<std::size_t>(j)];
            p2 += data.d2[static_cast<std::size_t>(i * method.stages + j)] * U[static_cast<std::size_t>(j)];
        }
        const double ci = method.c[static_cast<std::size_t>(i)];
        EXPECT_NEAR(p1, dp(ci), 1e-12);
        EXPECT_NEAR(p2, ddp(ci), 1e-12);
    }

    double u1 = data.u1_u0 * u0 + data.u1_dv0 * dv0;
    double du1 = data.du1_u0 * u0 + data.du1_dv0 * dv0;
    double ddu1 = data.ddu1_u0 * u0 + data.ddu1_dv0 * dv0;
    for (int j = 0; j < method.stages; ++j) {
        u1 += data.u1[static_cast<std::size_t>(j)] * U[static_cast<std::size_t>(j)];
        du1 += data.du1[static_cast<std::size_t>(j)] * U[static_cast<std::size_t>(j)];
        ddu1 += data.ddu1[static_cast<std::size_t>(j)] * U[static_cast<std::size_t>(j)];
    }

    EXPECT_NEAR(u1, p(1.0), 1e-12);
    EXPECT_NEAR(du1, dp(1.0), 1e-12);
    EXPECT_NEAR(ddu1, ddp(1.0), 1e-12);
}

