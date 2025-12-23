/**
 * @file test_PyramidQuadrature.cpp
 * @brief Unit tests for pyramid quadrature
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/PyramidQuadrature.h"
#include <numeric>
#include <cmath>

using namespace svmp::FE::quadrature;

namespace {
bool point_inside_pyramid(const QuadPoint& p) {
    // Reference pyramid: z in [0,1], x,y in [-1+z, 1-z]
    if (p[2] < -1e-12 || p[2] > 1.0 + 1e-12) return false;
    double bound = 1.0 - p[2];
    return (p[0] >= -bound - 1e-12) && (p[0] <= bound + 1e-12) &&
	           (p[1] >= -bound - 1e-12) && (p[1] <= bound + 1e-12);
}

double factorial_int(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) v *= static_cast<double>(i);
    return v;
}

double pyramid_monomial_integral(int a, int b, int c) {
    // Reference pyramid: base square [-1,1]^2 at z=0, apex at (0,0,1).
    // Using mapping x=(1-t)a0, y=(1-t)b0, z=t with (a0,b0) in [-1,1]^2, t in [0,1]
    // and Jacobian (1-t)^2:
    // ∫ x^a y^b z^c dV = (∫ a0^a da0)(∫ b0^b db0) * ∫_0^1 t^c (1-t)^{a+b+2} dt.
    if ((a % 2) != 0 || (b % 2) != 0) {
        return 0.0;
    }

    const double ia = 2.0 / static_cast<double>(a + 1);
    const double ib = 2.0 / static_cast<double>(b + 1);
    const int m = a + b + 2;
    const double beta = factorial_int(c) * factorial_int(m) / factorial_int(c + m + 1);
    return ia * ib * beta;
}
}

TEST(PyramidQuadrature, WeightSumMatchesVolume) {
    PyramidQuadrature quad(3);
    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 4.0 / 3.0, 1e-12);
}

TEST(PyramidQuadrature, IntegratesHeightLinearly) {
    PyramidQuadrature quad(4);
    double iz = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        iz += quad.weight(i) * quad.point(i)[2];
    }
    // Exact integral of z over reference pyramid = 1/3
    EXPECT_NEAR(iz, 1.0 / 3.0, 1e-12);
}

TEST(PyramidQuadrature, PointsInsideReferenceElement) {
    PyramidQuadrature quad(3);
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        EXPECT_TRUE(point_inside_pyramid(quad.point(i)));
    }
}

TEST(PyramidQuadrature, PolynomialExactnessUpToOrder) {
    for (int order = 1; order <= 6; ++order) {
        PyramidQuadrature quad(order);

        for (int total_degree = 0; total_degree <= order; ++total_degree) {
            for (int a = 0; a <= total_degree; ++a) {
                for (int b = 0; b <= total_degree - a; ++b) {
                    const int c = total_degree - a - b;

                    double acc = 0.0;
                    for (std::size_t i = 0; i < quad.num_points(); ++i) {
                        const auto& p = quad.point(i);
                        acc += quad.weight(i) *
                               std::pow(p[0], a) *
                               std::pow(p[1], b) *
                               std::pow(p[2], c);
                    }

                    const double exact = pyramid_monomial_integral(a, b, c);
                    const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
                    EXPECT_NEAR(acc, exact, tol)
                        << "Order " << order << " failed for x^" << a
                        << " y^" << b << " z^" << c;
                }
            }
        }
    }
}

TEST(PyramidQuadrature, PolynomialExactnessAnisotropicOrders) {
    // Cover the (order_ab, order_t) constructor and ensure it is exact up to its reported order.
    PyramidQuadrature quad(/*order_ab=*/5, /*order_t=*/2);
    const int degree_max = quad.order();

    for (int total_degree = 0; total_degree <= degree_max; ++total_degree) {
        for (int a = 0; a <= total_degree; ++a) {
            for (int b = 0; b <= total_degree - a; ++b) {
                const int c = total_degree - a - b;

                double acc = 0.0;
                for (std::size_t i = 0; i < quad.num_points(); ++i) {
                    const auto& p = quad.point(i);
                    acc += quad.weight(i) *
                           std::pow(p[0], a) *
                           std::pow(p[1], b) *
                           std::pow(p[2], c);
                }

                const double exact = pyramid_monomial_integral(a, b, c);
                const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
                EXPECT_NEAR(acc, exact, tol)
                    << "Anisotropic rule failed for x^" << a
                    << " y^" << b << " z^" << c;
            }
        }
    }
}
