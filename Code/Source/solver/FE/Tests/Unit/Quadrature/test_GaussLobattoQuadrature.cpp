/**
 * @file test_GaussLobattoQuadrature.cpp
 * @brief Unit tests for Gauss-Lobatto quadrature rules
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/GaussLobattoQuadrature.h"
#include <numeric>
#include <cmath>

using namespace svmp::FE::quadrature;

TEST(GaussLobattoQuadrature1D, WeightNormalization) {
    GaussLobattoQuadrature1D quad(5);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 2.0, 1e-13);
}

TEST(GaussLobattoQuadrature1D, EndpointInclusion) {
    GaussLobattoQuadrature1D quad(4);
    EXPECT_NEAR(quad.point(0)[0], -1.0, 1e-14);
    EXPECT_NEAR(quad.point(quad.num_points() - 1)[0], 1.0, 1e-14);
}

TEST(GaussLobattoQuadrature1D, IntegratesCubicExactly) {
    // 4-point Gauss-Lobatto is exact to degree 5
    GaussLobattoQuadrature1D quad(4);
    auto f = [](const QuadPoint& p) { return std::pow(p[0], 3); };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    EXPECT_NEAR(acc, 0.0, 1e-13);
}

TEST(GaussLobattoQuadrature1D, IntegratesQuarticExactlyWhenOrderSufficient) {
    GaussLobattoQuadrature1D quad(5); // order 7
    auto f = [](const QuadPoint& p) { return std::pow(p[0], 4); };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    EXPECT_NEAR(acc, 2.0 / 5.0, 1e-12);
}
