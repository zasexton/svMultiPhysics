/**
 * @file test_GaussQuadrature.cpp
 * @brief Unit tests for 1D Gauss and Gauss-Lobatto quadrature rules
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/GaussLobattoQuadrature.h"
#include <cmath>
#include <numeric>

using namespace svmp::FE::quadrature;

TEST(GaussQuadrature1D, WeightNormalization) {
    GaussQuadrature1D q3(3);
    double sum = std::accumulate(q3.weights().begin(), q3.weights().end(), 0.0);
    EXPECT_NEAR(sum, 2.0, 1e-14);
}

TEST(GaussQuadrature1D, IntegratesQuarticExactly) {
    const int points = 3; // order = 5
    GaussQuadrature1D quad(points);
    auto integrand = [](const QuadPoint& p) { return std::pow(p[0], 4); };

    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        val += quad.weight(i) * integrand(quad.point(i));
    }

    EXPECT_NEAR(val, 2.0 / 5.0, 1e-12);
}

TEST(GaussLobattoQuadrature1D, IncludesEndpoints) {
    GaussLobattoQuadrature1D quad(4);
    EXPECT_NEAR(quad.point(0)[0], -1.0, 1e-14);
    EXPECT_NEAR(quad.point(quad.num_points() - 1)[0], 1.0, 1e-14);

    auto integrand = [](const QuadPoint& p) { return std::pow(p[0], 3); };
    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        val += quad.weight(i) * integrand(quad.point(i));
    }

    // Odd polynomial should integrate to zero
    EXPECT_NEAR(val, 0.0, 1e-13);
}
