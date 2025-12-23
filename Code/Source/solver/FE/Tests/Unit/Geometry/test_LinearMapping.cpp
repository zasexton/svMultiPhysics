/**
 * @file test_LinearMapping.cpp
 * @brief Tests for affine linear geometry mappings
 */

#include <gtest/gtest.h>
#include "FE/Geometry/LinearMapping.h"
#include "FE/Quadrature/GaussQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(LinearMapping, LineMapsEndpointsAndMidpoint) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);

    math::Vector<Real, 3> x0 = map.map_to_physical({Real(-1), Real(0), Real(0)});
    math::Vector<Real, 3> x1 = map.map_to_physical({Real(1), Real(0), Real(0)});
    math::Vector<Real, 3> xm = map.map_to_physical({Real(0), Real(0), Real(0)});

    EXPECT_NEAR(x0[0], 0.0, 1e-14);
    EXPECT_NEAR(x1[0], 1.0, 1e-14);
    EXPECT_NEAR(xm[0], 0.5, 1e-14);

    auto J = map.jacobian({});
    EXPECT_NEAR(J(0,0), 0.5, 1e-14);
    EXPECT_NEAR(map.jacobian_determinant({}), 0.5, 1e-14);
}

TEST(LinearMapping, TriangleMapsBarycentricCenter) {
    // Node ordering must match LagrangeBasis/NodeOrderingConventions for Triangle3:
    // (0,0), (1,0), (0,1) on the reference unit right triangle.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)}, // ref node (0,0)
        {Real(1), Real(0), Real(0)}, // ref node (1,0)
        {Real(0), Real(1), Real(0)}  // ref node (0,1)
    };
    LinearMapping map(ElementType::Triangle3, nodes);
    math::Vector<Real, 3> xi{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0/3.0, 1e-12);

    auto det = map.jacobian_determinant({});
    EXPECT_NEAR(det, 1.0, 1e-12); // matches reference triangle measure
}

TEST(LinearMapping, TetraMapsBarycenterAndDeterminant) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)},
        {Real(0), Real(0), Real(1)}
    };
    LinearMapping map(ElementType::Tetra4, nodes);
    math::Vector<Real, 3> xi{Real(0.25), Real(0.25), Real(0.25)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], 0.25, 1e-12);
    EXPECT_NEAR(x[1], 0.25, 1e-12);
    EXPECT_NEAR(x[2], 0.25, 1e-12);

    auto det = map.jacobian_determinant(xi);
    EXPECT_NEAR(det, 1.0, 1e-12);

    // Roundtrip reference mapping
    auto xi_back = map.map_to_reference(x);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
    EXPECT_NEAR(xi_back[2], xi[2], 1e-8);
}

TEST(LinearMapping, LineJacobianInverseAndTransformGradient) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(2), Real(0), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);
    auto J = map.jacobian({});
    EXPECT_NEAR(J(0,0), 1.0, 1e-12);
    auto Jinv = map.jacobian_inverse({});
    EXPECT_NEAR(Jinv(0,0), 1.0, 1e-12);

    math::Vector<Real,3> grad_ref{Real(2), Real(0), Real(0)};
    auto grad_phys = map.transform_gradient(grad_ref, math::Vector<Real,3>{});
    EXPECT_NEAR(grad_phys[0], 2.0, 1e-12);
}

TEST(LinearMapping, LineAlongYAxisHasValidJacobianFrame) {
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(0), Real(2), Real(0)}
    };
    LinearMapping map(ElementType::Line2, nodes);

    auto J = map.jacobian({});
    EXPECT_NEAR(J(0,0), 0.0, 1e-12);
    EXPECT_NEAR(J(1,0), 1.0, 1e-12); // dy/dxi = length/2
    EXPECT_NEAR(map.jacobian_determinant({}), 1.0, 1e-12);

    // Roundtrip reference mapping at the midpoint.
    const auto x_mid = map.map_to_physical(math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    const auto xi_back = map.map_to_reference(x_mid);
    EXPECT_NEAR(xi_back[0], 0.0, 1e-8);
}
