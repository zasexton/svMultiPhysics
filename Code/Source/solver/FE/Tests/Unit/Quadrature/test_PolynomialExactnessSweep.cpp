/**
 * @file test_PolynomialExactnessSweep.cpp
 * @brief Parameterized polynomial exactness sweeps for quadrature rules
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/GaussQuadrature.h"
#include "FE/Quadrature/GaussLobattoQuadrature.h"
#include "FE/Quadrature/TriangleQuadrature.h"
#include "FE/Quadrature/TetrahedronQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/HexahedronQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"
#include "FE/Quadrature/ReferenceMonomialIntegrals.h"
#include <limits>
#include <numeric>
#include <cmath>
#include <vector>

using namespace svmp::FE::quadrature;

TEST(PolynomialSweep, Gauss1DOrders) {
    for (int n = 1; n <= 5; ++n) {
        GaussQuadrature1D quad(n);
        const int max_deg = 2 * n - 1;
        for (int k = 0; k <= max_deg; ++k) {
            double acc = 0.0;
            for (std::size_t i = 0; i < quad.num_points(); ++i) {
                acc += quad.weight(i) * std::pow(quad.point(i)[0], k);
            }
            // Odd powers integrate to zero, even powers: 2/(k+1) when k even
            if (k % 2 == 1) {
                EXPECT_NEAR(acc, 0.0, 1e-13);
            } else {
                double exact = 2.0 / static_cast<double>(k + 1);
                EXPECT_NEAR(acc, exact, 1e-12);
            }
        }
    }
}

TEST(PolynomialSweep, GaussLobattoOrders) {
    for (int n = 2; n <= 5; ++n) {
        GaussLobattoQuadrature1D quad(n);
        const int max_deg = 2 * n - 3;
        for (int k = 0; k <= max_deg; ++k) {
            double acc = 0.0;
            for (std::size_t i = 0; i < quad.num_points(); ++i) {
                acc += quad.weight(i) * std::pow(quad.point(i)[0], k);
            }
            if (k % 2 == 1) {
                EXPECT_NEAR(acc, 0.0, 1e-12);
            } else {
                double exact = 2.0 / static_cast<double>(k + 1);
                EXPECT_NEAR(acc, exact, 1e-11);
            }
        }
    }
}

TEST(PolynomialSweep, TriangleMonomials) {
    for (int order = 1; order <= 6; ++order) {
        TriangleQuadrature quad(order);
        ASSERT_GE(quad.order(), order);
        for (int a = 0; a <= quad.order(); ++a) {
            for (int b = 0; a + b <= quad.order(); ++b) {
                long double sum = 0.0L;
                for (std::size_t i = 0; i < quad.num_points(); ++i) {
                    const auto& p = quad.point(i);
                    sum += quad.weight(i) * std::pow(static_cast<long double>(p[0]), a) *
                           std::pow(static_cast<long double>(p[1]), b);
                }
                const double exact = reference_integrals::integral_triangle_monomial(a, b);
                EXPECT_NEAR(static_cast<double>(sum), exact, 64 * std::numeric_limits<double>::epsilon() * exact)
                    << "order " << order << ", exponents " << a << "," << b;
            }
        }
    }
}

TEST(PolynomialSweep, TetrahedronMonomials) {
    for (int order = 1; order <= 6; ++order) {
        TetrahedronQuadrature quad(order);
        ASSERT_GE(quad.order(), order);
        for (int a = 0; a <= quad.order(); ++a) {
            for (int b = 0; a + b <= quad.order(); ++b) {
                for (int c = 0; a + b + c <= quad.order(); ++c) {
                    long double sum = 0.0L;
                    for (std::size_t i = 0; i < quad.num_points(); ++i) {
                        const auto& p = quad.point(i);
                        sum += quad.weight(i) * std::pow(static_cast<long double>(p[0]), a) *
                               std::pow(static_cast<long double>(p[1]), b) *
                               std::pow(static_cast<long double>(p[2]), c);
                    }
                    const double exact = reference_integrals::integral_tetra_monomial(a, b, c);
                    EXPECT_NEAR(static_cast<double>(sum), exact, 64 * std::numeric_limits<double>::epsilon() * exact)
                        << "order " << order << ", exponents " << a << "," << b << "," << c;
                }
            }
        }
    }
}

TEST(PolynomialSweep, TensorMonomialsQuadrilateral) {
    for (int order = 2; order <= 4; ++order) {
        QuadrilateralQuadrature quad(order);
        for (int ax = 0; ax <= order; ++ax) {
            for (int ay = 0; ay <= order; ++ay) {
                double acc = 0.0;
                for (std::size_t i = 0; i < quad.num_points(); ++i) {
                    const auto& p = quad.point(i);
                    acc += quad.weight(i) * std::pow(p[0], ax) * std::pow(p[1], ay);
                }
                double exact = ((ax % 2) ? 0.0 : 2.0 / static_cast<double>(ax + 1)) *
                               ((ay % 2) ? 0.0 : 2.0 / static_cast<double>(ay + 1));
                EXPECT_NEAR(acc, exact, 1e-11);
            }
        }
    }
}

TEST(PolynomialSweep, TensorMonomialsHexahedron) {
    for (int order = 2; order <= 4; ++order) {
        HexahedronQuadrature quad(order);
        for (int ax = 0; ax <= order; ++ax) {
            for (int ay = 0; ay <= order; ++ay) {
                for (int az = 0; az <= order; ++az) {
                    double acc = 0.0;
                    for (std::size_t i = 0; i < quad.num_points(); ++i) {
                        const auto& p = quad.point(i);
                        acc += quad.weight(i) * std::pow(p[0], ax) * std::pow(p[1], ay) * std::pow(p[2], az);
                    }
                    double exact = ((ax % 2) ? 0.0 : 2.0 / static_cast<double>(ax + 1)) *
                                   ((ay % 2) ? 0.0 : 2.0 / static_cast<double>(ay + 1)) *
                                   ((az % 2) ? 0.0 : 2.0 / static_cast<double>(az + 1));
                    EXPECT_NEAR(acc, exact, 1e-11);
                }
            }
        }
    }
}

TEST(PolynomialSweep, WedgeMonomialsIsotropic) {
    for (int order = 2; order <= 6; ++order) {
        WedgeQuadrature quad(order);
        const int max_deg = std::min(order, quad.order());

        for (int total_degree = 0; total_degree <= max_deg; ++total_degree) {
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

                    const double exact = reference_integrals::integral_wedge_monomial(a, b, c);
                    const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
                    EXPECT_NEAR(acc, exact, tol)
                        << "Order " << order << " failed for x^" << a
                        << " y^" << b << " z^" << c;
                }
            }
        }
    }
}

TEST(PolynomialSweep, WedgeMonomialsAnisotropic) {
    struct Case {
        int tri_order;
        int line_order;
    };
    const std::vector<Case> cases = {
        {6, 2},
        {2, 6},
        {5, 4},
    };

    for (const auto& c : cases) {
        WedgeQuadrature quad(c.tri_order, c.line_order);
        const int max_deg = quad.order();

        for (int total_degree = 0; total_degree <= max_deg; ++total_degree) {
            for (int a = 0; a <= total_degree; ++a) {
                for (int b = 0; b <= total_degree - a; ++b) {
                    const int pz = total_degree - a - b;

                    double acc = 0.0;
                    for (std::size_t i = 0; i < quad.num_points(); ++i) {
                        const auto& p = quad.point(i);
                        acc += quad.weight(i) *
                               std::pow(p[0], a) *
                               std::pow(p[1], b) *
                               std::pow(p[2], pz);
                    }

                    const double exact = reference_integrals::integral_triangle_monomial(a, b) * reference_integrals::integral_monomial_1d(pz);
                    const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
                    EXPECT_NEAR(acc, exact, tol)
                        << "Anisotropic (" << c.tri_order << "," << c.line_order
                        << ") failed for x^" << a << " y^" << b << " z^" << pz;
                }
            }
        }
    }
}
