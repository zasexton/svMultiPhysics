/**
 * @file test_TriangleQuadrature.cpp
 * @brief Unit tests for triangle quadrature
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/TriangleQuadrature.h"
#include <numeric>
#include <cmath>

using namespace svmp::FE::quadrature;

namespace {
double factorial_int(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) v *= static_cast<double>(i);
    return v;
}
}

TEST(TriangleQuadrature, WeightSumMatchesArea) {
    TriangleQuadrature quad(3);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST(TriangleQuadrature, IntegratesLinearFunctions) {
    TriangleQuadrature quad(3);

    auto integrand = [](const QuadPoint& p) { return p[0] + p[1]; };
    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        val += quad.weight(i) * integrand(quad.point(i));
    }

    // ∫_T (x+y) dA over reference triangle = 1/3
    EXPECT_NEAR(val, 1.0 / 3.0, 5e-12);
}

TEST(TriangleQuadrature, PolynomialExactnessDegreeThree) {
    TriangleQuadrature quad(5); // order 5 integrates degree up to 5
    double ix2 = 0.0;
    double ixy = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        double w = quad.weight(i);
        ix2 += w * p[0] * p[0];
        ixy += w * p[0] * p[1];
    }
    // Exact integrals on reference triangle (0,0)-(1,0)-(0,1): a! b! / (a+b+2)!
    const double ix2_exact = factorial_int(2) * factorial_int(0) / factorial_int(4); // 2!/0!/4! = 1/12
    const double ixy_exact = factorial_int(1) * factorial_int(1) / factorial_int(4);  // 1/24
    EXPECT_NEAR(ix2, ix2_exact, 1e-12);
    EXPECT_NEAR(ixy, ixy_exact, 1e-12);
}

TEST(TriangleQuadrature, RequestedOrderThreeIntegratesCubics) {
    // Historically, odd requested orders could under-integrate due to n rounding.
    // Request order=3 and verify exactness for total-degree-3 monomials.
    TriangleQuadrature quad(3);

    auto integrate_monomial = [&](int a, int b) {
        double acc = 0.0;
        for (std::size_t i = 0; i < quad.num_points(); ++i) {
            const auto& p = quad.point(i);
            acc += quad.weight(i) * std::pow(p[0], a) * std::pow(p[1], b);
        }
        return acc;
    };

    auto exact_monomial = [&](int a, int b) {
        return factorial_int(a) * factorial_int(b) / factorial_int(a + b + 2);
    };

    EXPECT_NEAR(integrate_monomial(3, 0), exact_monomial(3, 0), 1e-12);
    EXPECT_NEAR(integrate_monomial(2, 1), exact_monomial(2, 1), 1e-12);
    EXPECT_NEAR(integrate_monomial(1, 2), exact_monomial(1, 2), 1e-12);
    EXPECT_NEAR(integrate_monomial(0, 3), exact_monomial(0, 3), 1e-12);
}
