/**
 * @file test_QuadrilateralQuadrature.cpp
 * @brief Unit tests for quadrilateral tensor-product quadrature
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Core/Types.h"
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

TEST(QuadrilateralQuadrature, WeightSumMatchesArea) {
    QuadrilateralQuadrature quad(3);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 4.0, 1e-12);
}

TEST(QuadrilateralQuadrature, IntegratesEvenPolynomial) {
    QuadrilateralQuadrature quad(3);
    auto f = [](const QuadPoint& p) { return p[0] * p[0] * p[1] * p[1]; };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    // \int_{-1}^1 x^2 dx = 2/3, so product = (2/3)^2 = 4/9
    EXPECT_NEAR(acc, 4.0 / 9.0, 1e-12);
}

TEST(QuadrilateralQuadrature, IntegratesOddPolynomialToZero) {
    QuadrilateralQuadrature quad(4);
    auto f = [](const QuadPoint& p) { return p[0] * p[1]; };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    EXPECT_NEAR(acc, 0.0, 1e-14);
}

TEST(QuadrilateralQuadrature, PointsInsideReferenceElement) {
    QuadrilateralQuadrature quad(2);
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        EXPECT_GE(p[0], -1.0 - 1e-12);
        EXPECT_LE(p[0], 1.0 + 1e-12);
        EXPECT_GE(p[1], -1.0 - 1e-12);
        EXPECT_LE(p[1], 1.0 + 1e-12);
    }
}

TEST(QuadrilateralQuadrature, ReducedIntegrationConstant) {
    QuadrilateralQuadrature quad(3, QuadratureType::Reduced);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 4.0, 1e-12);
}

TEST(QuadrilateralQuadrature, AnisotropicOrders) {
    // Different orders per direction should still integrate separable monomials within each order
    QuadrilateralQuadrature quad(4, 2);
    auto f = [](const QuadPoint& p) { return p[0] * p[0] * p[1] * p[1]; };
    double acc = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        acc += quad.weight(i) * f(quad.point(i));
    }
    // Integral = (2/3)*(2/3) = 4/9 over [-1,1]^2
    EXPECT_NEAR(acc, 4.0 / 9.0, 1e-12);
}
