/**
 * @file test_HexahedronQuadrature.cpp
 * @brief Unit tests for hexahedron tensor-product quadrature
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/HexahedronQuadrature.h"
#include "FE/Core/Types.h"
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

TEST(HexahedronQuadrature, WeightSumMatchesVolume) {
    HexahedronQuadrature quad(3);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 8.0, 1e-12);
}

TEST(HexahedronQuadrature, IntegratesQuadraticExactly) {
    HexahedronQuadrature quad(4);
    auto f = [](const QuadPoint& p) { return p[0] * p[0]; };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    // Integral over [-1,1]^3 of x^2 = (2/3)*2*2 = 8/3
    EXPECT_NEAR(acc, 8.0 / 3.0, 1e-12);
}

TEST(HexahedronQuadrature, OddIntegrandVanishes) {
    HexahedronQuadrature quad(3);
    auto f = [](const QuadPoint& p) { return p[0] * p[1] * p[2]; };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    EXPECT_NEAR(acc, 0.0, 1e-14);
}

TEST(HexahedronQuadrature, PointsInsideReferenceElement) {
    HexahedronQuadrature quad(2);
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        EXPECT_GE(p[0], -1.0 - 1e-12);
        EXPECT_LE(p[0], 1.0 + 1e-12);
        EXPECT_GE(p[1], -1.0 - 1e-12);
        EXPECT_LE(p[1], 1.0 + 1e-12);
        EXPECT_GE(p[2], -1.0 - 1e-12);
        EXPECT_LE(p[2], 1.0 + 1e-12);
    }
}

TEST(HexahedronQuadrature, ReducedIntegrationConstant) {
    HexahedronQuadrature quad(3, QuadratureType::Reduced);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 8.0, 1e-12);
}
