/**
 * @file test_OrthogonalPolynomials.cpp
 * @brief Unit tests for orthogonal polynomial utilities used in basis functions
 */

#include <gtest/gtest.h>
#include "FE/Basis/OrthogonalPolynomials.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::basis::orthopoly;
using namespace svmp::FE::quadrature;

TEST(OrthogonalPolynomials, LegendreValues) {
    const Real x = Real(0.3);
    EXPECT_NEAR(legendre(0, x), 1.0, 1e-14);
    EXPECT_NEAR(legendre(1, x), x, 1e-14);
    const Real p2_expected = Real(0.5) * (Real(3) * x * x - Real(1));
    EXPECT_NEAR(legendre(2, x), p2_expected, 1e-13);
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
    auto [val, dxi, deta] = dubiner_with_derivatives(p, q, xi, eta);

    const Real fd_xi =
        (dubiner(p, q, xi + eps, eta) - dubiner(p, q, xi - eps, eta)) / (Real(2) * eps);
    const Real fd_eta =
        (dubiner(p, q, xi, eta + eps) - dubiner(p, q, xi, eta - eps)) / (Real(2) * eps);

    const Real scale_xi = std::max({std::abs(dxi), std::abs(fd_xi), Real(1)});
    const Real scale_eta = std::max({std::abs(deta), std::abs(fd_eta), Real(1)});

    EXPECT_NEAR(dxi, fd_xi, Real(1e-6) * scale_xi);
    EXPECT_NEAR(deta, fd_eta, Real(1e-6) * scale_eta);
    (void)val;
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
    auto [val, dxi, deta, dzeta] = proriol_with_derivatives(p, q, r, xi, eta, zeta);

    const Real fd_xi =
        (proriol(p, q, r, xi + eps, eta, zeta) - proriol(p, q, r, xi - eps, eta, zeta)) /
        (Real(2) * eps);
    const Real fd_eta =
        (proriol(p, q, r, xi, eta + eps, zeta) - proriol(p, q, r, xi, eta - eps, zeta)) /
        (Real(2) * eps);
    const Real fd_zeta =
        (proriol(p, q, r, xi, eta, zeta + eps) - proriol(p, q, r, xi, eta, zeta - eps)) /
        (Real(2) * eps);

    const Real scale_xi = std::max({std::abs(dxi), std::abs(fd_xi), Real(1)});
    const Real scale_eta = std::max({std::abs(deta), std::abs(fd_eta), Real(1)});
    const Real scale_zeta = std::max({std::abs(dzeta), std::abs(fd_zeta), Real(1)});

    EXPECT_NEAR(dxi, fd_xi, Real(1e-6) * scale_xi);
    EXPECT_NEAR(deta, fd_eta, Real(1e-6) * scale_eta);
    EXPECT_NEAR(dzeta, fd_zeta, Real(1e-6) * scale_zeta);
    (void)val;
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
