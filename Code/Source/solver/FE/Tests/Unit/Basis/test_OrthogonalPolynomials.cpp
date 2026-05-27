/**
 * @file test_OrthogonalPolynomials.cpp
 * @brief Unit tests for orthogonal polynomial utilities used in basis functions
 */

#include <gtest/gtest.h>
#include "FE/Basis/BasisTolerance.h"
#include "FE/Basis/OrthogonalPolynomials.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::basis::orthopoly;
using namespace svmp::FE::quadrature;

namespace {

long double binomial_reference(long double top, int k) {
    long double value = 1.0L;
    for (int i = 1; i <= k; ++i) {
        value *= (top - static_cast<long double>(k - i)) / static_cast<long double>(i);
    }
    return value;
}

long double jacobi_reference(int n, long double alpha, long double beta, long double x) {
    long double sum = 0.0L;
    const long double half_x_minus_one = (x - 1.0L) * 0.5L;
    const long double half_x_plus_one = (x + 1.0L) * 0.5L;
    for (int m = 0; m <= n; ++m) {
        const long double left = binomial_reference(static_cast<long double>(n) + alpha, n - m);
        const long double right = binomial_reference(static_cast<long double>(n) + beta, m);
        sum += left * right *
               std::pow(half_x_minus_one, m) *
               std::pow(half_x_plus_one, n - m);
    }
    return sum;
}

Real scaled_tolerance(long double expected, Real multiplier) {
    return multiplier * std::max(Real(1), static_cast<Real>(std::abs(expected)));
}

} // namespace

TEST(OrthogonalPolynomials, LegendreValues) {
    const Real x = Real(0.3);
    EXPECT_NEAR(legendre(0, x), 1.0, 1e-14);
    EXPECT_NEAR(legendre(1, x), x, 1e-14);
    const Real p2_expected = Real(0.5) * (Real(3) * x * x - Real(1));
    EXPECT_NEAR(legendre(2, x), p2_expected, 1e-13);
}

TEST(OrthogonalPolynomials, LegendreDerivativeIsFiniteAtEndpoints) {
    for (int n = 0; n <= 8; ++n) {
        const auto minus = legendre_derivative(n, Real(-1));
        const auto zero = legendre_derivative(n, Real(0));
        const auto plus = legendre_derivative(n, Real(1));

        EXPECT_TRUE(std::isfinite(minus.value));
        EXPECT_TRUE(std::isfinite(minus.derivative));
        EXPECT_TRUE(std::isfinite(zero.value));
        EXPECT_TRUE(std::isfinite(zero.derivative));
        EXPECT_TRUE(std::isfinite(plus.value));
        EXPECT_TRUE(std::isfinite(plus.derivative));

        const Real expected_minus =
            ((n % 2) == 0 ? Real(1) : Real(-1));
        const Real expected_dp_minus =
            (((n + 1) % 2) == 0 ? Real(1) : Real(-1)) *
            Real(n * (n + 1)) * Real(0.5);
        const Real expected_dp_plus = Real(n * (n + 1)) * Real(0.5);

        EXPECT_NEAR(minus.value, expected_minus, 1e-13);
        EXPECT_NEAR(plus.value, Real(1), 1e-13);
        EXPECT_NEAR(minus.derivative, expected_dp_minus, 1e-12);
        EXPECT_NEAR(plus.derivative, expected_dp_plus, 1e-12);
    }
}

TEST(OrthogonalPolynomials, NamedDerivativeApisReturnConsistentData) {
    const Real x = Real(0.27);
    const auto legendre_result = legendre_derivative(5, x);
    EXPECT_NEAR(legendre_result.value, legendre(5, x), Real(1e-14));

    const auto sequence = legendre_sequence_derivatives(5, x);
    ASSERT_EQ(sequence.values.size(), 6u);
    ASSERT_EQ(sequence.derivatives.size(), 6u);
    EXPECT_EQ(sequence.values.back(), legendre_result.value);
    EXPECT_EQ(sequence.derivatives.back(), legendre_result.derivative);

    const auto second_sequence = legendre_sequence_second_derivatives(5, x);
    EXPECT_EQ(second_sequence.values, sequence.values);
    EXPECT_EQ(second_sequence.derivatives, sequence.derivatives);
    ASSERT_EQ(second_sequence.second_derivatives.size(), 6u);

    const auto dubiner = dubiner_derivatives(2, 1, Real(0.2), Real(0.3));
    const auto dubiner_second = dubiner_with_second_derivatives(2, 1, Real(0.2), Real(0.3));
    EXPECT_EQ(dubiner.value, dubiner_second.value);
    EXPECT_NEAR(dubiner.dxi, dubiner_second.dxi, Real(1e-14));
    EXPECT_NEAR(dubiner.deta, dubiner_second.deta, Real(1e-14));

    const auto proriol = proriol_derivatives(1, 1, 1, Real(0.1), Real(0.2), Real(0.15));
    const auto proriol_second =
        proriol_with_second_derivatives(1, 1, 1, Real(0.1), Real(0.2), Real(0.15));
    EXPECT_EQ(proriol.value, proriol_second.value);
    EXPECT_EQ(proriol.gradient[0], proriol_second.gradient[0]);
    EXPECT_EQ(proriol.gradient[1], proriol_second.gradient[1]);
    EXPECT_NEAR(proriol.gradient[2], proriol_second.gradient[2], Real(1e-14));
}

TEST(OrthogonalPolynomials, LegendreOrthogonality) {
    const int max_n = 3;
    GaussQuadrature1D quad(8); // sufficiently high order

    for (int m = 0; m <= max_n; ++m) {
        for (int n = 0; n <= max_n; ++n) {
            double inner = 0.0;
            for (std::size_t i = 0; i < quad.num_points(); ++i) {
                const Real x = quad.point(i)[0];
                inner += quad.weight(i) *
                         legendre(m, x) *
                         legendre(n, x);
            }
            if (m == n) {
                const double expected = 2.0 / static_cast<double>(2 * m + 1);
                EXPECT_NEAR(inner, expected, 1e-10);
            } else {
                EXPECT_NEAR(inner, 0.0, 1e-10);
            }
        }
    }
}

TEST(OrthogonalPolynomials, JacobiMatchesLegendreWithinLooseTolerance) {
    const Real xs[] = {Real(-0.9), Real(-0.2), Real(0.0), Real(0.3), Real(0.8)};
    for (int n = 0; n <= 8; ++n) {
        for (Real x : xs) {
            EXPECT_NEAR(jacobi(n, Real(0), Real(0), x), legendre(n, x), 1e-12);
        }
    }
}

TEST(OrthogonalPolynomials, JacobiDerivativeRoughlyMatchesFiniteDifference) {
    const int n = 6;
    const Real alpha = Real(0.5);
    const Real beta = Real(-0.25);
    const Real x = Real(0.1);
    const Real eps = Real(1e-6);

    const Real exact = jacobi_derivative(n, alpha, beta, x);
    const Real fd = (jacobi(n, alpha, beta, x + eps) -
                     jacobi(n, alpha, beta, x - eps)) / (Real(2) * eps);

    const Real scale = std::max({std::abs(exact), std::abs(fd), Real(1)});
    EXPECT_NEAR(exact, fd, Real(1e-5) * scale);
}

TEST(OrthogonalPolynomials, JacobiFusedSecondDerivativesMatchScalarApis) {
    struct Case {
        Real alpha;
        Real beta;
        Real x;
    };

    const std::array<Case, 8> cases{{
        {Real(0), Real(0), Real(-0.9)},
        {Real(0), Real(0), Real(0.42)},
        {Real(0), Real(0), Real(0.9)},
        {Real(0.5), Real(-0.25), Real(0.1)},
        {Real(1.25), Real(0.75), Real(-0.35)},
        {Real(3), Real(0), Real(0.65)},
        {Real(7), Real(0), Real(-0.2)},
        {Real(9), Real(0), Real(0.98)},
    }};

    for (const auto& c : cases) {
        for (int n = 0; n <= 10; ++n) {
            const auto fused = jacobi_with_second_derivative(n, c.alpha, c.beta, c.x);
            const Real value = jacobi(n, c.alpha, c.beta, c.x);
            const Real first = jacobi_derivative(n, c.alpha, c.beta, c.x);
            const Real second = jacobi_second_derivative(n, c.alpha, c.beta, c.x);

            const Real value_scale = std::max({std::abs(fused.value), std::abs(value), Real(1)});
            const Real first_scale = std::max({std::abs(fused.derivative), std::abs(first), Real(1)});
            const Real second_scale =
                std::max({std::abs(fused.second_derivative), std::abs(second), Real(1)});

            EXPECT_NEAR(fused.value, value, Real(1e-12) * value_scale)
                << "n=" << n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
            EXPECT_NEAR(fused.derivative, first, Real(1e-11) * first_scale)
                << "n=" << n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
            EXPECT_NEAR(fused.second_derivative, second, Real(1e-10) * second_scale)
                << "n=" << n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
        }
    }
}

TEST(OrthogonalPolynomials, JacobiFusedSequenceMatchesScalarApis) {
    struct Case {
        Real alpha;
        Real beta;
        Real x;
    };

    const std::array<Case, 4> cases{{
        {Real(0), Real(0), Real(-0.6)},
        {Real(0.5), Real(-0.25), Real(0.2)},
        {Real(5), Real(0), Real(0.7)},
        {Real(9), Real(0), Real(-0.95)},
    }};

    constexpr int max_order = 12;
    std::vector<Real> values(static_cast<std::size_t>(max_order + 1));
    std::vector<Real> first(static_cast<std::size_t>(max_order + 1));
    std::vector<Real> second(static_cast<std::size_t>(max_order + 1));

    for (const auto& c : cases) {
        jacobi_sequence_with_second_derivatives_to(max_order,
                                                   c.alpha,
                                                   c.beta,
                                                   c.x,
                                                   values,
                                                   first,
                                                   second);
        for (int n = 0; n <= max_order; ++n) {
            const auto idx = static_cast<std::size_t>(n);
            const Real value_ref = jacobi(n, c.alpha, c.beta, c.x);
            const Real first_ref = jacobi_derivative(n, c.alpha, c.beta, c.x);
            const Real second_ref = jacobi_second_derivative(n, c.alpha, c.beta, c.x);

            const Real value_scale = std::max({std::abs(values[idx]), std::abs(value_ref), Real(1)});
            const Real first_scale = std::max({std::abs(first[idx]), std::abs(first_ref), Real(1)});
            const Real second_scale =
                std::max({std::abs(second[idx]), std::abs(second_ref), Real(1)});

            EXPECT_NEAR(values[idx], value_ref, Real(1e-12) * value_scale)
                << "n=" << n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
            EXPECT_NEAR(first[idx], first_ref, Real(1e-11) * first_scale)
                << "n=" << n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
            EXPECT_NEAR(second[idx], second_ref, Real(1e-10) * second_scale)
                << "n=" << n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
        }
    }
}

TEST(OrthogonalPolynomials, JacobiValidatedEnvelopeMatchesLongDoubleReference) {
    struct Case {
        int n;
        Real alpha;
        Real beta;
        Real x;
    };

    const std::array<Case, 12> cases{{
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(-1)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(-0.999999999999)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(-0.98)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(0.35)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(0.98)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(0.999999999999)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(20), Real(0), Real(1)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(24), Real(16), Real(-0.85)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(24), Real(16), Real(0.0)},
        {kMaxValidatedJacobiRecurrenceOrder, Real(24), Real(16), Real(0.85)},
        {4, Real(20), Real(0), Real(-0.999999999999)},
        {4, Real(20), Real(0), Real(0.999999999999)},
    }};

    for (const auto& c : cases) {
        ASSERT_LE(c.n, kMaxValidatedJacobiRecurrenceOrder);
        ASSERT_LE(c.alpha + c.beta, kMaxValidatedJacobiAlphaBetaSum);

        const long double reference =
            jacobi_reference(c.n,
                             static_cast<long double>(c.alpha),
                             static_cast<long double>(c.beta),
                             static_cast<long double>(c.x));
        const auto fused = jacobi_with_second_derivative(c.n, c.alpha, c.beta, c.x);
        const Real value = jacobi(c.n, c.alpha, c.beta, c.x);

        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE(std::isfinite(fused.value));
        EXPECT_TRUE(std::isfinite(fused.derivative));
        EXPECT_TRUE(std::isfinite(fused.second_derivative));
        EXPECT_NEAR(value, static_cast<Real>(reference), scaled_tolerance(reference, Real(2e-10)))
            << "n=" << c.n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
        EXPECT_NEAR(fused.value, static_cast<Real>(reference),
                    scaled_tolerance(reference, Real(2e-10)))
            << "n=" << c.n << " alpha=" << c.alpha << " beta=" << c.beta << " x=" << c.x;
    }
}

TEST(OrthogonalPolynomials, JacobiEndpointDerivativesMatchReferenceFormula) {
    constexpr Real alpha = Real(20);
    constexpr Real beta = Real(7);

    for (int n = 0; n <= kMaxValidatedJacobiRecurrenceOrder; ++n) {
        ASSERT_LE(n, kMaxValidatedJacobiRecurrenceOrder);
        ASSERT_LE(alpha + beta, kMaxValidatedJacobiAlphaBetaSum);

        for (Real x : {Real(-1), Real(1)}) {
            const long double value_ref =
                jacobi_reference(n,
                                 static_cast<long double>(alpha),
                                 static_cast<long double>(beta),
                                 static_cast<long double>(x));
            long double first_ref = 0.0L;
            if (n > 0) {
                first_ref = 0.5L *
                            (static_cast<long double>(alpha + beta) +
                             static_cast<long double>(n + 1)) *
                            jacobi_reference(n - 1,
                                             static_cast<long double>(alpha + Real(1)),
                                             static_cast<long double>(beta + Real(1)),
                                             static_cast<long double>(x));
            }
            long double second_ref = 0.0L;
            if (n > 1) {
                second_ref = 0.25L *
                             (static_cast<long double>(alpha + beta) +
                              static_cast<long double>(n + 1)) *
                             (static_cast<long double>(alpha + beta) +
                              static_cast<long double>(n + 2)) *
                             jacobi_reference(n - 2,
                                              static_cast<long double>(alpha + Real(2)),
                                              static_cast<long double>(beta + Real(2)),
                                              static_cast<long double>(x));
            }

            const auto fused = jacobi_with_second_derivative(n, alpha, beta, x);
            EXPECT_NEAR(fused.value, static_cast<Real>(value_ref),
                        scaled_tolerance(value_ref, Real(2e-10)))
                << "n=" << n << " x=" << x;
            EXPECT_NEAR(fused.derivative, static_cast<Real>(first_ref),
                        scaled_tolerance(first_ref, Real(5e-10)))
                << "n=" << n << " x=" << x;
            EXPECT_NEAR(fused.second_derivative, static_cast<Real>(second_ref),
                        scaled_tolerance(second_ref, Real(1e-9)))
                << "n=" << n << " x=" << x;
        }
    }
}

TEST(OrthogonalPolynomials, ProriolHighAlphaNearDegenerateSimplexIsFinite) {
    const int p = 4;
    const int q = 4;
    const int r = 4;
    static_assert(p + q + r <= kMaxValidatedSimplexModalTotalOrder);

    struct Point {
        Real xi;
        Real eta;
        Real zeta;
    };
    const std::array<Point, 3> points{{
        {Real(0.02), Real(0.03), Real(0.91)},
        {Real(1e-9), Real(2e-9), Real(1) - Real(4e-9)},
        {Real(0.11), Real(0.17), Real(0.21)},
    }};

    for (const auto& point : points) {
        const Real value = proriol(p, q, r, point.xi, point.eta, point.zeta);
        const auto first = proriol_derivatives(p, q, r, point.xi, point.eta, point.zeta);
        const auto second = proriol_with_second_derivatives(p, q, r, point.xi, point.eta, point.zeta);

        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE(std::isfinite(first.value));
        EXPECT_TRUE(std::isfinite(first.gradient[0]));
        EXPECT_TRUE(std::isfinite(first.gradient[1]));
        EXPECT_TRUE(std::isfinite(first.gradient[2]));
        EXPECT_TRUE(std::isfinite(second.value));
        EXPECT_NEAR(first.value, value, Real(1e-12) * std::max(std::abs(value), Real(1)));
        EXPECT_NEAR(second.value, value, Real(1e-12) * std::max(std::abs(value), Real(1)));
    }
}

TEST(OrthogonalPolynomials, LegendreSecondDerivativeSequenceMatchesScalarJacobi) {
    const Real xs[] = {Real(-0.9), Real(-0.35), Real(0.0), Real(0.42), Real(0.9)};
    for (Real x : xs) {
        const auto seq = legendre_sequence_second_derivatives(10, x);
        ASSERT_EQ(seq.values.size(), 11u);
        ASSERT_EQ(seq.derivatives.size(), 11u);
        ASSERT_EQ(seq.second_derivatives.size(), 11u);
        for (int n = 0; n <= 10; ++n) {
            const auto first = legendre_derivative(n, x);
            EXPECT_NEAR(seq.values[static_cast<std::size_t>(n)], legendre(n, x), Real(1e-12))
                << "n=" << n << " x=" << x;
            EXPECT_NEAR(seq.derivatives[static_cast<std::size_t>(n)],
                        first.derivative,
                        Real(1e-12))
                << "n=" << n << " x=" << x;
            EXPECT_NEAR(seq.second_derivatives[static_cast<std::size_t>(n)],
                        jacobi_second_derivative(n, Real(0), Real(0), x),
                        Real(1e-11))
                << "n=" << n << " x=" << x;
        }
    }
}

TEST(OrthogonalPolynomials, IntegratedLegendreZeroMean) {
    for (int n = 1; n <= 4; ++n) {
        const Real val = integrated_legendre(n, Real(1)); // integral from -1 to 1
        EXPECT_NEAR(val, 0.0, 1e-12);
    }
}

TEST(OrthogonalPolynomials, DubinerDerivativesMatchFiniteDifference) {
    const int p = 2;
    const int q = 1;

    // Interior point in the reference triangle (xi>=0, eta>=0, xi+eta<=1)
    const Real xi = Real(0.2);
    const Real eta = Real(0.3);

    const Real eps = Real(1e-6);
    const auto first = dubiner_derivatives(p, q, xi, eta);

    const Real fd_xi =
        (dubiner(p, q, xi + eps, eta) - dubiner(p, q, xi - eps, eta)) / (Real(2) * eps);
    const Real fd_eta =
        (dubiner(p, q, xi, eta + eps) - dubiner(p, q, xi, eta - eps)) / (Real(2) * eps);

    const Real scale_xi = std::max({std::abs(first.dxi), std::abs(fd_xi), Real(1)});
    const Real scale_eta = std::max({std::abs(first.deta), std::abs(fd_eta), Real(1)});

    EXPECT_NEAR(first.dxi, fd_xi, Real(1e-6) * scale_xi);
    EXPECT_NEAR(first.deta, fd_eta, Real(1e-6) * scale_eta);
    (void)first.value;
}

TEST(OrthogonalPolynomials, ProriolDerivativesMatchFiniteDifference) {
    const int p = 1;
    const int q = 1;
    const int r = 1;

    // Interior point in the reference tetrahedron (xi,eta,zeta>=0, sum<=1)
    const Real xi = Real(0.1);
    const Real eta = Real(0.2);
    const Real zeta = Real(0.15);

    const Real eps = Real(1e-6);
    const auto first = proriol_derivatives(p, q, r, xi, eta, zeta);

    const Real fd_xi =
        (proriol(p, q, r, xi + eps, eta, zeta) - proriol(p, q, r, xi - eps, eta, zeta)) /
        (Real(2) * eps);
    const Real fd_eta =
        (proriol(p, q, r, xi, eta + eps, zeta) - proriol(p, q, r, xi, eta - eps, zeta)) /
        (Real(2) * eps);
    const Real fd_zeta =
        (proriol(p, q, r, xi, eta, zeta + eps) - proriol(p, q, r, xi, eta, zeta - eps)) /
        (Real(2) * eps);

    const Real scale_xi = std::max({std::abs(first.gradient[0]), std::abs(fd_xi), Real(1)});
    const Real scale_eta = std::max({std::abs(first.gradient[1]), std::abs(fd_eta), Real(1)});
    const Real scale_zeta = std::max({std::abs(first.gradient[2]), std::abs(fd_zeta), Real(1)});

    EXPECT_NEAR(first.gradient[0], fd_xi, Real(1e-6) * scale_xi);
    EXPECT_NEAR(first.gradient[1], fd_eta, Real(1e-6) * scale_eta);
    EXPECT_NEAR(first.gradient[2], fd_zeta, Real(1e-6) * scale_zeta);
    (void)first.value;
}

TEST(OrthogonalPolynomials, CollapsedCoordinateThresholdsAreConsistent) {
    const Real collapsed_eps =
        svmp::FE::basis::detail::basis_scaled_tolerance(Real(1), Real(10)) * Real(0.5);

    {
        const int p = 2;
        const int q = 1;
        const Real xi = Real(0);
        const Real eta = Real(1) - collapsed_eps;

        const Real value = dubiner(p, q, xi, eta);
        const auto first = dubiner_derivatives(p, q, xi, eta);
        const auto second = dubiner_with_second_derivatives(p, q, xi, eta);

        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE(std::isfinite(first.value));
        EXPECT_TRUE(std::isfinite(first.dxi));
        EXPECT_TRUE(std::isfinite(first.deta));
        EXPECT_TRUE(std::isfinite(second.value));
        EXPECT_NEAR(value, first.value, Real(1e-14));
        EXPECT_NEAR(value, second.value, Real(1e-14));
    }

    {
        const int p = 1;
        const int q = 1;
        const int r = 1;
        const Real xi = Real(0);
        const Real eta = Real(0);
        const Real zeta = Real(1) - collapsed_eps;

        const Real value = proriol(p, q, r, xi, eta, zeta);
        const auto first = proriol_derivatives(p, q, r, xi, eta, zeta);
        const auto second = proriol_with_second_derivatives(p, q, r, xi, eta, zeta);

        EXPECT_TRUE(std::isfinite(value));
        EXPECT_TRUE(std::isfinite(first.value));
        EXPECT_TRUE(std::isfinite(first.gradient[0]));
        EXPECT_TRUE(std::isfinite(first.gradient[1]));
        EXPECT_TRUE(std::isfinite(first.gradient[2]));
        EXPECT_TRUE(std::isfinite(second.value));
        EXPECT_NEAR(value, first.value, Real(1e-14));
        EXPECT_NEAR(value, second.value, Real(1e-14));
    }
}

TEST(OrthogonalPolynomials, DubinerOrthogonality) {
    // Dubiner polynomials should be orthogonal on the reference triangle
    // under the unit weight: integral of D(p1,q1) * D(p2,q2) dA = 0 for (p1,q1) != (p2,q2)
    const int max_order = 3;
    TriangleQuadrature quad(2 * max_order + 2);

    // Collect all (p,q) multi-indices up to total order max_order
    struct PQ { int p; int q; };
    std::vector<PQ> indices;
    for (int total = 0; total <= max_order; ++total) {
        for (int p = 0; p <= total; ++p) {
            indices.push_back({p, total - p});
        }
    }

    const std::size_t n = indices.size();
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
            double inner = 0.0;
            for (std::size_t q = 0; q < quad.num_points(); ++q) {
                const Real xi = quad.point(q)[0];
                const Real eta = quad.point(q)[1];
                inner += quad.weight(q) *
                         dubiner(indices[a].p, indices[a].q, xi, eta) *
                         dubiner(indices[b].p, indices[b].q, xi, eta);
            }
            if (a == b) {
                // Diagonal: should be positive
                EXPECT_GT(inner, 0.0)
                    << "D(" << indices[a].p << "," << indices[a].q << ") self-inner-product";
            } else {
                // Off-diagonal: should be zero
                EXPECT_NEAR(inner, 0.0, 1e-10)
                    << "D(" << indices[a].p << "," << indices[a].q
                    << ") x D(" << indices[b].p << "," << indices[b].q << ")";
            }
        }
    }
}

TEST(OrthogonalPolynomials, ProriolOrthogonality) {
    // Proriol polynomials should be orthogonal on the reference tetrahedron
    const int max_order = 2;
    TetrahedronQuadrature quad(2 * max_order + 2);

    struct PQR { int p; int q; int r; };
    std::vector<PQR> indices;
    for (int total = 0; total <= max_order; ++total) {
        for (int p = 0; p <= total; ++p) {
            for (int q = 0; q <= total - p; ++q) {
                indices.push_back({p, q, total - p - q});
            }
        }
    }

    const std::size_t n = indices.size();
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a; b < n; ++b) {
            double inner = 0.0;
            for (std::size_t q = 0; q < quad.num_points(); ++q) {
                const Real xi = quad.point(q)[0];
                const Real eta = quad.point(q)[1];
                const Real zeta = quad.point(q)[2];
                inner += quad.weight(q) *
                         proriol(indices[a].p, indices[a].q, indices[a].r, xi, eta, zeta) *
                         proriol(indices[b].p, indices[b].q, indices[b].r, xi, eta, zeta);
            }
            if (a == b) {
                EXPECT_GT(inner, 0.0)
                    << "P(" << indices[a].p << "," << indices[a].q << "," << indices[a].r
                    << ") self-inner-product";
            } else {
                EXPECT_NEAR(inner, 0.0, 1e-10)
                    << "P(" << indices[a].p << "," << indices[a].q << "," << indices[a].r
                    << ") x P(" << indices[b].p << "," << indices[b].q << "," << indices[b].r << ")";
            }
        }
    }
}
