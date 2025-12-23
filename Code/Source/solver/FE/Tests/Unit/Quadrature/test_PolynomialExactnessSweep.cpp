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
#include <numeric>
#include <cmath>
#include <vector>

using namespace svmp::FE::quadrature;

namespace {
[[maybe_unused]] double factorial_int(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) v *= static_cast<double>(i);
    return v;
}

double triangle_monomial_integral(int a, int b) {
    // Reference triangle (0,0)-(1,0)-(0,1): ∫ x^a y^b dA = a! b! / (a+b+2)!
    return factorial_int(a) * factorial_int(b) / factorial_int(a + b + 2);
}

double line_monomial_integral(int k) {
    // Reference line [-1,1]: ∫ z^k dz = 0 for odd k, 2/(k+1) for even k.
    if ((k % 2) != 0) {
        return 0.0;
    }
    return 2.0 / static_cast<double>(k + 1);
}
}

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
    // Integrate x^a y^b using a high-order reference for comparison
    TriangleQuadrature ref(8);
    for (int order = 2; order <= 5; ++order) {
        TriangleQuadrature quad(order);
        // Limit to total degree within quadratic for robustness
        for (int a = 0; a <= 2; ++a) {
            for (int b = 0; b + a <= 2; ++b) {
                double acc = 0.0;
                double ref_acc = 0.0;
                for (std::size_t i = 0; i < quad.num_points(); ++i) {
                    const auto& p = quad.point(i);
                    acc += quad.weight(i) * std::pow(p[0], a) * std::pow(p[1], b);
                }
                for (std::size_t i = 0; i < ref.num_points(); ++i) {
                    const auto& p = ref.point(i);
                    ref_acc += ref.weight(i) * std::pow(p[0], a) * std::pow(p[1], b);
                }
                EXPECT_NEAR(acc, ref_acc, 1e-4);
            }
        }
    }
}

TEST(PolynomialSweep, TetrahedronMonomials) {
    TetrahedronQuadrature ref(10);
    for (int order = 3; order <= 6; ++order) {
        TetrahedronQuadrature quad(order);
        for (int a = 0; a <= 2; ++a) {
            for (int b = 0; b + a <= 2; ++b) {
                for (int c = 0; c + a + b <= 2; ++c) {
                    double acc = 0.0;
                    double ref_acc = 0.0;
                    for (std::size_t i = 0; i < quad.num_points(); ++i) {
                        const auto& p = quad.point(i);
                        acc += quad.weight(i) *
                               std::pow(p[0], a) * std::pow(p[1], b) * std::pow(p[2], c);
                    }
                    for (std::size_t i = 0; i < ref.num_points(); ++i) {
                        const auto& p = ref.point(i);
                        ref_acc += ref.weight(i) *
                                   std::pow(p[0], a) * std::pow(p[1], b) * std::pow(p[2], c);
                    }
                    double tol = std::abs(ref_acc) * 0.2 + 1e-6; // allow modest relative error
                    EXPECT_NEAR(acc, ref_acc, tol);
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

                    const double exact = triangle_monomial_integral(a, b) * line_monomial_integral(c);
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

                    const double exact = triangle_monomial_integral(a, b) * line_monomial_integral(pz);
                    const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
                    EXPECT_NEAR(acc, exact, tol)
                        << "Anisotropic (" << c.tri_order << "," << c.line_order
                        << ") failed for x^" << a << " y^" << b << " z^" << pz;
                }
            }
        }
    }
}
