/**
 * @file test_TetrahedronQuadrature.cpp
 * @brief Unit tests for tetrahedron quadrature
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include <numeric>

using namespace svmp::FE::quadrature;

namespace {
double factorial_int(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) v *= static_cast<double>(i);
    return v;
}
}

TEST(TetrahedronQuadrature, WeightSumMatchesVolume) {
    TetrahedronQuadrature quad(3);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-12);
}

TEST(TetrahedronQuadrature, IntegratesLinearMonomials) {
    TetrahedronQuadrature quad(4);
    double ix = 0.0, iy = 0.0, iz = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        double w = quad.weight(i);
        ix += w * p[0];
        iy += w * p[1];
        iz += w * p[2];
    }
    EXPECT_NEAR(ix, 1.0 / 24.0, 1e-12);
    EXPECT_NEAR(iy, 1.0 / 24.0, 1e-12);
    EXPECT_NEAR(iz, 1.0 / 24.0, 1e-12);
}

TEST(TetrahedronQuadrature, PolynomialExactnessDegreeThree) {
    TetrahedronQuadrature quad(5); // order 5 integrates degree up to 5
    double ix2 = 0.0;
    double ixyz = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        double w = quad.weight(i);
        ix2 += w * p[0] * p[0];
        ixyz += w * p[0] * p[1] * p[2];
    }
    // Exact integrals on reference tetrahedron: a! b! c! / (a+b+c+3)!
    const double ix2_exact = factorial_int(2) * factorial_int(0) * factorial_int(0) / factorial_int(5);   // 2!/5! = 1/60
    const double ixyz_exact = factorial_int(1) * factorial_int(1) * factorial_int(1) / factorial_int(6); // 1/120
    EXPECT_NEAR(ix2, ix2_exact, 1e-12);
    EXPECT_NEAR(ixyz, ixyz_exact, 1e-12);
}

TEST(TetrahedronQuadrature, RequestedOrderTwoIntegratesQuadratics) {
    // Historically, low even requested orders could under-integrate due to n rounding.
    // Request order=2 and verify exactness for total-degree-2 monomials.
    TetrahedronQuadrature quad(2);

    double ix2 = 0.0, iy2 = 0.0, iz2 = 0.0;
    double ixy = 0.0, ixz = 0.0, iyz = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        const double w = quad.weight(i);
        ix2 += w * p[0] * p[0];
        iy2 += w * p[1] * p[1];
        iz2 += w * p[2] * p[2];
        ixy += w * p[0] * p[1];
        ixz += w * p[0] * p[2];
        iyz += w * p[1] * p[2];
    }

    const double ix2_exact = factorial_int(2) * factorial_int(0) * factorial_int(0) / factorial_int(5); // 1/60
    const double iy2_exact = ix2_exact;
    const double iz2_exact = ix2_exact;
    const double ixy_exact = factorial_int(1) * factorial_int(1) * factorial_int(0) / factorial_int(5); // 1/120
    const double ixz_exact = ixy_exact;
    const double iyz_exact = ixy_exact;

    EXPECT_NEAR(ix2, ix2_exact, 1e-12);
    EXPECT_NEAR(iy2, iy2_exact, 1e-12);
    EXPECT_NEAR(iz2, iz2_exact, 1e-12);
    EXPECT_NEAR(ixy, ixy_exact, 1e-12);
    EXPECT_NEAR(ixz, ixz_exact, 1e-12);
    EXPECT_NEAR(iyz, iyz_exact, 1e-12);
}
