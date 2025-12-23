/**
 * @file test_CompositeAdaptiveSingularSurface.cpp
 * @brief Tests for composite, adaptive, singular, and surface quadrature utilities
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/CompositeQuadrature.h"
#include "FE/Quadrature/AdaptiveQuadrature.h"
#include "FE/Quadrature/SingularQuadrature.h"
#include "FE/Quadrature/SurfaceQuadrature.h"
#include "FE/Quadrature/QuadrilateralQuadrature.h"
#include "FE/Quadrature/WedgeQuadrature.h"
#include "FE/Quadrature/GaussQuadrature.h"
#include <algorithm>
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

namespace {
bool point_in_reference_triangle(const QuadPoint& p, double tol = 1e-12) {
    return p[0] >= -tol && p[1] >= -tol && (p[0] + p[1]) <= 1.0 + tol;
}

bool point_in_reference_tetrahedron(const QuadPoint& p, double tol = 1e-12) {
    return p[0] >= -tol && p[1] >= -tol && p[2] >= -tol &&
           (p[0] + p[1] + p[2]) <= 1.0 + tol;
}

double factorial_int(int n) {
    double v = 1.0;
    for (int i = 2; i <= n; ++i) v *= static_cast<double>(i);
    return v;
}

double triangle_monomial_integral(int a, int b) {
    // Reference triangle (0,0)-(1,0)-(0,1): a! b! / (a+b+2)!
    return factorial_int(a) * factorial_int(b) / factorial_int(a + b + 2);
}

double tetrahedron_monomial_integral(int a, int b, int c) {
    // Reference tetrahedron with vertices (0,0,0),(1,0,0),(0,1,0),(0,0,1):
    // a! b! c! / (a+b+c+3)!
    return factorial_int(a) * factorial_int(b) * factorial_int(c) /
           factorial_int(a + b + c + 3);
}
} // namespace

TEST(CompositeQuadrature, PreservesConstantIntegral) {
    QuadrilateralQuadrature base(3);
    CompositeQuadrature composite(base, 2);

    auto integrate = [](const QuadratureRule& q) {
        double sum = 0.0;
        for (std::size_t i = 0; i < q.num_points(); ++i) sum += q.weight(i);
        return sum;
    };

    EXPECT_NEAR(integrate(base), integrate(composite), 1e-12);
}

TEST(CompositeQuadrature, WedgeSubdivisionsPreserveConstantIntegral) {
    WedgeQuadrature base(3);
    for (int subdivisions : {1, 2}) {
        CompositeQuadrature composite(base, subdivisions);

        double base_sum = 0.0;
        for (std::size_t i = 0; i < base.num_points(); ++i) base_sum += base.weight(i);

        double comp_sum = 0.0;
        for (std::size_t i = 0; i < composite.num_points(); ++i) comp_sum += composite.weight(i);

        EXPECT_NEAR(comp_sum, base_sum, 1e-12) << "subdivisions=" << subdivisions;
    }
}

TEST(AdaptiveQuadrature, ConvergesOnPolynomial) {
    // Start with a rule already exact for quartic to validate convergence detection
    GaussQuadrature1D base(3); // order 5
    AdaptiveQuadrature adaptive(1e-8, 4);
    auto f = [](const QuadPoint& p) { return p[0] * p[0] * p[0] * p[0]; };

    auto result = adaptive.integrate(base, f);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.value, 2.0 / 5.0, 1e-10);
}

TEST(AdaptiveQuadrature, OddIntegrandConvergesToZero) {
    GaussQuadrature1D base(2); // symmetric
    AdaptiveQuadrature adaptive(1e-8, 4);
    auto f = [](const QuadPoint& p) { return p[0]; };
    auto result = adaptive.integrate(base, f);
    EXPECT_TRUE(result.converged);
    EXPECT_NEAR(result.value, 0.0, 1e-12);
}

TEST(AdaptiveQuadrature, TwoDimensionalSmoothFunction) {
    QuadrilateralQuadrature base(3);
    AdaptiveQuadrature adaptive(1e-6, 4);
    auto f = [](const QuadPoint& p) { return std::exp(p[0] * p[1]); };
    auto result = adaptive.integrate(base, f);
    EXPECT_TRUE(result.converged);
    // Reference via higher-order quad
    QuadrilateralQuadrature ref(6);
    double ref_val = 0.0;
    for (std::size_t i = 0; i < ref.num_points(); ++i) {
        ref_val += ref.weight(i) * f(ref.point(i));
    }
    EXPECT_NEAR(result.value, ref_val, std::abs(ref_val) * 5e-2);
}

TEST(SingularQuadrature, DuffyTriangleWeightSum) {
    auto rule = SingularQuadrature::duffy_triangle(4);
    double sum = 0.0;
    for (std::size_t i = 0; i < rule->num_points(); ++i) sum += rule->weight(i);
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST(SingularQuadrature, DuffyTrianglePointsInsideReferenceTriangle) {
    auto rule = SingularQuadrature::duffy_triangle(4);
    for (std::size_t i = 0; i < rule->num_points(); ++i) {
        EXPECT_TRUE(point_in_reference_triangle(rule->point(i)))
            << "Point outside reference triangle at index " << i;
    }
}

TEST(SingularQuadrature, DuffyTriangleIntegratesLinearMonomials) {
    auto rule = SingularQuadrature::duffy_triangle(4);
    double ix = 0.0;
    double iy = 0.0;
    for (std::size_t i = 0; i < rule->num_points(); ++i) {
        const auto& p = rule->point(i);
        const double w = rule->weight(i);
        ix += w * p[0];
        iy += w * p[1];
    }
    EXPECT_NEAR(ix, 1.0 / 6.0, 1e-12);
    EXPECT_NEAR(iy, 1.0 / 6.0, 1e-12);
}

TEST(SingularQuadrature, DuffyTriangleMomentSymmetry) {
    auto rule = SingularQuadrature::duffy_triangle(4);

    double ix = 0.0;
    double iy = 0.0;
    double ix2 = 0.0;
    double iy2 = 0.0;
    double ixy = 0.0;
    for (std::size_t i = 0; i < rule->num_points(); ++i) {
        const auto& p = rule->point(i);
        const double w = rule->weight(i);
        ix += w * p[0];
        iy += w * p[1];
        ix2 += w * p[0] * p[0];
        iy2 += w * p[1] * p[1];
        ixy += w * p[0] * p[1];
    }

    EXPECT_NEAR(ix, iy, 1e-13);
    EXPECT_NEAR(ix2, iy2, 1e-13);

    EXPECT_NEAR(ix, triangle_monomial_integral(1, 0), 1e-12);
    EXPECT_NEAR(iy, triangle_monomial_integral(0, 1), 1e-12);
    EXPECT_NEAR(ix2, triangle_monomial_integral(2, 0), 1e-12);
    EXPECT_NEAR(iy2, triangle_monomial_integral(0, 2), 1e-12);
    EXPECT_NEAR(ixy, triangle_monomial_integral(1, 1), 1e-12);
}

TEST(SingularQuadrature, DuffyTrianglePolynomialExactnessDegreeThree) {
    // Ensure the Duffy triangle rule integrates all monomials up to total degree 3.
    auto rule = SingularQuadrature::duffy_triangle(3);
    const int max_deg = 3;

    for (int total_degree = 0; total_degree <= max_deg; ++total_degree) {
        for (int a = 0; a <= total_degree; ++a) {
            const int b = total_degree - a;
            double acc = 0.0;
            for (std::size_t i = 0; i < rule->num_points(); ++i) {
                const auto& p = rule->point(i);
                acc += rule->weight(i) * std::pow(p[0], a) * std::pow(p[1], b);
            }
            const double exact = triangle_monomial_integral(a, b);
            const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
            EXPECT_NEAR(acc, exact, tol)
                << "x^" << a << " y^" << b << " mismatch";
        }
    }
}

TEST(SingularQuadrature, DuffyTriangleIntegratesMildSingularity) {
    auto rule = SingularQuadrature::duffy_triangle(8);
    auto baseline = TriangleQuadrature(16); // higher-order reference
    auto f = [](const QuadPoint& p) { return 1.0 / std::sqrt(p[0] + p[1] + 0.2); };

    auto integrate = [&](const auto& q) {
        double sum = 0.0;
        for (std::size_t i = 0; i < q.num_points(); ++i) {
            sum += q.weight(i) * f(q.point(i));
        }
        return sum;
    };

    double duffy = integrate(*rule);
    double ref = integrate(baseline);
    double tol = std::abs(ref) * 0.15 + 1e-6; // allow modest relative error
    EXPECT_NEAR(duffy, ref, tol);
}

TEST(SingularQuadrature, DuffyTetraWeightSum) {
    auto rule = SingularQuadrature::duffy_tetrahedron(4);
    double sum = 0.0;
    for (std::size_t i = 0; i < rule->num_points(); ++i) sum += rule->weight(i);
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-12);
}

TEST(SingularQuadrature, DuffyTetraPointsInsideReferenceTetrahedron) {
    auto rule = SingularQuadrature::duffy_tetrahedron(4);
    for (std::size_t i = 0; i < rule->num_points(); ++i) {
        EXPECT_TRUE(point_in_reference_tetrahedron(rule->point(i)))
            << "Point outside reference tetrahedron at index " << i;
    }
}

TEST(SingularQuadrature, DuffyTetraIntegratesLinearMonomials) {
    auto rule = SingularQuadrature::duffy_tetrahedron(4);
    double ix = 0.0;
    double iy = 0.0;
    double iz = 0.0;
    for (std::size_t i = 0; i < rule->num_points(); ++i) {
        const auto& p = rule->point(i);
        const double w = rule->weight(i);
        ix += w * p[0];
        iy += w * p[1];
        iz += w * p[2];
    }
    EXPECT_NEAR(ix, 1.0 / 24.0, 1e-12);
    EXPECT_NEAR(iy, 1.0 / 24.0, 1e-12);
    EXPECT_NEAR(iz, 1.0 / 24.0, 1e-12);
}

TEST(SingularQuadrature, DuffyTetraMomentSymmetry) {
    auto rule = SingularQuadrature::duffy_tetrahedron(4);

    double ix = 0.0, iy = 0.0, iz = 0.0;
    double ix2 = 0.0, iy2 = 0.0, iz2 = 0.0;
    double ixy = 0.0, ixz = 0.0, iyz = 0.0;
    for (std::size_t i = 0; i < rule->num_points(); ++i) {
        const auto& p = rule->point(i);
        const double w = rule->weight(i);
        ix += w * p[0];
        iy += w * p[1];
        iz += w * p[2];
        ix2 += w * p[0] * p[0];
        iy2 += w * p[1] * p[1];
        iz2 += w * p[2] * p[2];
        ixy += w * p[0] * p[1];
        ixz += w * p[0] * p[2];
        iyz += w * p[1] * p[2];
    }

    EXPECT_NEAR(ix, iy, 1e-13);
    EXPECT_NEAR(ix, iz, 1e-13);
    EXPECT_NEAR(ix2, iy2, 1e-13);
    EXPECT_NEAR(ix2, iz2, 1e-13);
    EXPECT_NEAR(ixy, ixz, 1e-13);
    EXPECT_NEAR(ixy, iyz, 1e-13);

    EXPECT_NEAR(ix, tetrahedron_monomial_integral(1, 0, 0), 1e-12);
    EXPECT_NEAR(iy, tetrahedron_monomial_integral(0, 1, 0), 1e-12);
    EXPECT_NEAR(iz, tetrahedron_monomial_integral(0, 0, 1), 1e-12);
    EXPECT_NEAR(ix2, tetrahedron_monomial_integral(2, 0, 0), 1e-12);
    EXPECT_NEAR(iy2, tetrahedron_monomial_integral(0, 2, 0), 1e-12);
    EXPECT_NEAR(iz2, tetrahedron_monomial_integral(0, 0, 2), 1e-12);
    EXPECT_NEAR(ixy, tetrahedron_monomial_integral(1, 1, 0), 1e-12);
    EXPECT_NEAR(ixz, tetrahedron_monomial_integral(1, 0, 1), 1e-12);
    EXPECT_NEAR(iyz, tetrahedron_monomial_integral(0, 1, 1), 1e-12);
}

TEST(SingularQuadrature, DuffyTetraPolynomialExactnessDegreeTwo) {
    // Ensure the Duffy tetra rule integrates all monomials up to total degree 2.
    auto rule = SingularQuadrature::duffy_tetrahedron(2);
    const int max_deg = 2;

    for (int total_degree = 0; total_degree <= max_deg; ++total_degree) {
        for (int a = 0; a <= total_degree; ++a) {
            for (int b = 0; b <= total_degree - a; ++b) {
                const int c = total_degree - a - b;

                double acc = 0.0;
                for (std::size_t i = 0; i < rule->num_points(); ++i) {
                    const auto& p = rule->point(i);
                    acc += rule->weight(i) *
                           std::pow(p[0], a) * std::pow(p[1], b) * std::pow(p[2], c);
                }

                const double exact = tetrahedron_monomial_integral(a, b, c);
                const double tol = std::max(1e-12, std::abs(exact) * 1e-11);
                EXPECT_NEAR(acc, exact, tol)
                    << "x^" << a << " y^" << b << " z^" << c << " mismatch";
            }
        }
    }
}

TEST(SingularQuadrature, DuffyTetraIntegratesMildSingularity) {
    auto rule = SingularQuadrature::duffy_tetrahedron(12);
    auto baseline = TetrahedronQuadrature(20); // higher-order reference
    auto f = [](const QuadPoint& p) { return 1.0 / std::sqrt(p[0] + p[1] + p[2] + 0.3); };

    auto integrate = [&](const auto& q) {
        double sum = 0.0;
        for (std::size_t i = 0; i < q.num_points(); ++i) {
            sum += q.weight(i) * f(q.point(i));
        }
        return sum;
    };

    double duffy = integrate(*rule);
    double ref = integrate(baseline);
    double tol = std::abs(ref) * 0.25 + 1e-6; // modest relative error allowance
    EXPECT_NEAR(duffy, ref, tol);
}

TEST(SurfaceQuadrature, QuadFaceUsesQuadRule) {
    auto face_rule = SurfaceQuadrature::face_rule(ElementType::Hex8, 0, 3);
    double sum = 0.0;
    for (std::size_t i = 0; i < face_rule->num_points(); ++i) sum += face_rule->weight(i);
    EXPECT_NEAR(sum, 4.0, 1e-12);
}

TEST(SurfaceQuadrature, WedgeTriFaceWeightSum) {
    auto face_rule = SurfaceQuadrature::face_rule(ElementType::Wedge6, 0, 3);
    double sum = 0.0;
    for (std::size_t i = 0; i < face_rule->num_points(); ++i) sum += face_rule->weight(i);
    // Triangular face area should be 0.5
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST(SurfaceQuadrature, EdgeRuleWeightSum) {
    auto edge_rule = SurfaceQuadrature::edge_rule(svmp::CellFamily::Line, 0, 3);
    double sum = 0.0;
    for (std::size_t i = 0; i < edge_rule->num_points(); ++i) sum += edge_rule->weight(i);
    EXPECT_NEAR(sum, 2.0, 1e-12);
}
