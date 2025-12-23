/**
 * @file test_HigherOrderMappings.cpp
 * @brief Tests for higher-order isoparametric mappings across element types
 */

#include <gtest/gtest.h>
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/SerendipityBasis.h"
#include "FE/Basis/NodeOrderingConventions.h"

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(HigherOrderMapping, IdentityLineOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Line2, 2); // Line3
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.3), Real(0), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-12);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityTriangleOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Triangle3, 2); // Triangle6
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(0.1), Real(0)}; // inside reference triangle
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityTetraOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Tetra4, 2); // Tetra10
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(0.2), Real(0.1)}; // sum < 1
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityQuadOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2); // Quad9
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(-0.3), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-12);
    EXPECT_NEAR(x[1], xi[1], 1e-12);

    auto det = map.jacobian_determinant(xi);
    EXPECT_NEAR(det, 1.0, 1e-10);
}

TEST(HigherOrderMapping, IdentityQuadSerendipityOrder2) {
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, 2);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(1),  Real(1),  Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(0),  Real(-1), Real(0)}, // mid-bottom
        {Real(1),  Real(0),  Real(0)}, // mid-right
        {Real(0),  Real(1),  Real(0)}, // mid-top
        {Real(-1), Real(0),  Real(0)}, // mid-left
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(0.2), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, IdentityHexOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 2); // Hex27
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(-0.2), Real(0.25)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    auto det = map.jacobian_determinant(xi);
    EXPECT_NEAR(det, 1.0, 1e-8);
}

TEST(HigherOrderMapping, IdentityHexSerendipityOrder2) {
    // Geometry-mode serendipity basis uses robust Hex8-style mapping
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, true);
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-1), Real(-1), Real(-1)},
        {Real(1),  Real(-1), Real(-1)},
        {Real(1),  Real(1),  Real(-1)},
        {Real(-1), Real(1),  Real(-1)},
        {Real(-1), Real(-1), Real(1)},
        {Real(1),  Real(-1), Real(1)},
        {Real(1),  Real(1),  Real(1)},
        {Real(-1), Real(1),  Real(1)},
        // x-edges (y,z fixed)
        {Real(0), Real(-1), Real(-1)},
        {Real(0), Real(-1), Real(1)},
        {Real(0), Real(1),  Real(-1)},
        {Real(0), Real(1),  Real(1)},
        // y-edges
        {Real(-1), Real(0), Real(-1)},
        {Real(1),  Real(0), Real(-1)},
        {Real(-1), Real(0), Real(1)},
        {Real(1),  Real(0), Real(1)},
        // z-edges
        {Real(-1), Real(-1), Real(0)},
        {Real(1),  Real(-1), Real(0)},
        {Real(-1), Real(1),  Real(0)},
        {Real(1),  Real(1),  Real(0)},
    };
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(-0.1), Real(0.2)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(HigherOrderMapping, Quad8SerendipityAffineDistortion) {
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Quad8, 2);
    // Affine skew/scale: x' = 2x + y, y' = x + 3y
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-2-1), Real(-1-3), Real(0)},   // (-1,-1)
        {Real(2-1),  Real(1-3),  Real(0)},   // (1,-1)
        {Real(2+1),  Real(1+3),  Real(0)},   // (1,1)
        {Real(-2+1), Real(-1+3), Real(0)},   // (-1,1)
        {Real(0-1),  Real(-3),   Real(0)},   // (0,-1)
        {Real(2),    Real(1),    Real(0)},   // (1,0)
        {Real(0+1),  Real(3),    Real(0)},   // (0,1)
        {Real(-2),   Real(-1),   Real(0)},   // (-1,0)
    };
    IsoparametricMapping map(basis, nodes);
    math::Vector<Real,3> xi{Real(0.2), Real(-0.1), Real(0)};
    auto x = map.map_to_physical(xi);
    // Analytic map: x' = 2x + y, y' = x + 3y
    Real xp = 2*xi[0] + xi[1];
    Real yp = xi[0] + 3*xi[1];
    EXPECT_NEAR(x[0], xp, 1e-10);
    EXPECT_NEAR(x[1], yp, 1e-10);
    auto det = map.jacobian_determinant(xi);
    // Jacobian of affine map: [[2,1],[1,3]] det = 5
    EXPECT_NEAR(det, 5.0, 1e-10);
}

TEST(HigherOrderMapping, Hex20SerendipityAffineDistortion) {
    // Geometry-mode Hex20 mapping; edge DOFs inert for geometry
    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, true);
    // Affine map: x' = 2x + y, y' = x + 3y, z' = 4z
    std::vector<math::Vector<Real,3>> nodes = {
        {Real(-2-1), Real(-1-3), Real(-4)},   // (-1,-1,-1)
        {Real(2-1),  Real(1-3),  Real(-4)},   // (1,-1,-1)
        {Real(2+1),  Real(1+3),  Real(-4)},   // (1,1,-1)
        {Real(-2+1), Real(-1+3), Real(-4)},   // (-1,1,-1)
        {Real(-2-1), Real(-1-3), Real(4)},    // (-1,-1,1)
        {Real(2-1),  Real(1-3),  Real(4)},    // (1,-1,1)
        {Real(2+1),  Real(1+3),  Real(4)},    // (1,1,1)
        {Real(-2+1), Real(-1+3), Real(4)},    // (-1,1,1)
        {Real(0-1),  Real(-3),   Real(-4)},   // (0,-1,-1)
        {Real(0+1),  Real(3),    Real(-4)},   // (0,1,-1)
        {Real(0-1),  Real(-3),   Real(4)},    // (0,-1,1)
        {Real(0+1),  Real(3),    Real(4)},    // (0,1,1)
        {Real(-2),   Real(-1),   Real(-4)},   // (-1,0,-1)
        {Real(2),    Real(1),    Real(-4)},   // (1,0,-1)
        {Real(-2),   Real(-1),   Real(4)},    // (-1,0,1)
        {Real(2),    Real(1),    Real(4)},    // (1,0,1)
        {Real(-2-1), Real(-1-3), Real(0)},    // (-1,-1,0)
        {Real(2-1),  Real(1-3),  Real(0)},    // (1,-1,0)
        {Real(2+1),  Real(1+3),  Real(0)},    // (1,1,0)
        {Real(-2+1), Real(-1+3), Real(0)},    // (-1,1,0)
    };
    IsoparametricMapping map(basis, nodes);
    math::Vector<Real,3> xi{Real(0.2), Real(-0.1), Real(0.3)};
    auto x = map.map_to_physical(xi);
    Real xp = 2*xi[0] + xi[1];
    Real yp = xi[0] + 3*xi[1];
    Real zp = 4*xi[2];
    EXPECT_NEAR(x[0], xp, 1e-10);
    EXPECT_NEAR(x[1], yp, 1e-10);
    EXPECT_NEAR(x[2], zp, 1e-10);
    auto det = map.jacobian_determinant(xi);
    // Jacobian [[2,1,0],[1,3,0],[0,0,4]] det = 2*3*4 - 1*1*4 = 24 - 4 = 20
    EXPECT_NEAR(det, 20.0, 1e-8);
}

TEST(HigherOrderMapping, Hex20SerendipityNodalIdentityNoGeometryMode) {
    using svmp::FE::basis::NodeOrdering;

    auto basis = std::make_shared<basis::SerendipityBasis>(ElementType::Hex20, 2, false);

    std::vector<math::Vector<Real,3>> nodes;
    const std::size_t nn = NodeOrdering::num_nodes(ElementType::Hex20);
    nodes.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Hex20, i));
    }

    IsoparametricMapping map(basis, nodes);

    // Nodal identity: x(xi_i) = xi_i for all reference nodes.
    for (std::size_t i = 0; i < nn; ++i) {
        const auto xi = nodes[i];
        const auto x = map.map_to_physical(xi);
        EXPECT_NEAR(x[0], xi[0], 1e-10);
        EXPECT_NEAR(x[1], xi[1], 1e-10);
        EXPECT_NEAR(x[2], xi[2], 1e-10);
    }

    const auto det = map.jacobian_determinant(math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    EXPECT_NEAR(det, 1.0, 1e-8);
}
