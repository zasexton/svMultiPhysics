/**
 * @file test_SpacesIntegration.cpp
 * @brief Small integration examples using Spaces + Elements + Geometry
 *
 * These tests exercise the interaction between FunctionSpace, Elements,
 * Quadrature, and Geometry modules on a single element without any Mesh
 * dependency. They verify that interpolation and evaluation via Spaces
 * behave consistently when mapped to physical coordinates.
 */

#include <gtest/gtest.h>

#include "FE/Spaces/H1Space.h"
#include "FE/Geometry/IsoparametricMapping.h"

using namespace svmp::FE;
using namespace svmp::FE::spaces;

/**
 * @brief Interpolate a polynomial on a reference triangle and verify L2 accuracy
 *
 * We consider the triangle with vertices (0,0), (1,0), (0,1) and a linear
 * function f(x, y) = x + 2y. We interpolate f into H1(Triangle3, p=1) using
 * H1Space::interpolate (which operates in reference space) and then measure
 * the L2 error over the physical element using IsoparametricMapping.
 *
 * Since the space contains all linear polynomials, the interpolation should
 * be exact up to numerical quadrature tolerance.
 */
TEST(SpacesIntegration, TriangleH1InterpolationMatchesAnalyticalFunction) {
    H1Space space(ElementType::Triangle3, 1);

    // Interpolate f(x̂) = x̂ + 2 ŷ on the reference triangle
    std::vector<Real> coeffs;
    FunctionSpace::ValueFunction f_ref = [](const FunctionSpace::Value& xi) {
        FunctionSpace::Value out{};
        out[0] = xi[0] + Real(2) * xi[1];
        return out;
    };
    space.interpolate(f_ref, coeffs);

    // Build an isoparametric mapping using the same basis nodes as geometry.
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 1);
    const auto& ref_nodes = basis->nodes();
    std::vector<math::Vector<Real, 3>> nodes(ref_nodes.begin(), ref_nodes.end());
    geometry::IsoparametricMapping mapping(basis, nodes);

    const auto& elem = space.element();
    auto quad = elem.quadrature();
    ASSERT_TRUE(quad);

    Real l2_error_sq = Real(0);

    for (std::size_t q = 0; q < quad->num_points(); ++q) {
        const auto xi = quad->point(q);
        const Real w  = quad->weight(q);

        // Reference -> physical
        const auto x_phys = mapping.map_to_physical(xi);
        const Real f_exact = x_phys[0] + Real(2) * x_phys[1];

        // Evaluate interpolant in reference coordinates
        const Real f_h = space.evaluate_scalar(xi, coeffs);

        const Real diff = f_h - f_exact;
        const Real detJ = std::abs(mapping.jacobian_determinant(xi));
        l2_error_sq += w * detJ * diff * diff;
    }

    EXPECT_NEAR(l2_error_sq, 0.0, 1e-12);
}
