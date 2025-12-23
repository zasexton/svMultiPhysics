/**
 * @file test_IsoparametricMapping.cpp
 * @brief Tests for isoparametric and inverse mappings
 */

#include <gtest/gtest.h>
#include "FE/Geometry/IsoparametricMapping.h"
#include "FE/Geometry/InverseMapping.h"
#include "FE/Geometry/MappingFactory.h"
#include "FE/Geometry/GeometryValidator.h"
#include "FE/Basis/LagrangeBasis.h"
#include "FE/Basis/HierarchicalBasis.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(IsoparametricMapping, IdentityQuadOrder2) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 2);
    const auto& ref_nodes = basis->nodes();
    std::vector<math::Vector<Real, 3>> nodes(ref_nodes.begin(), ref_nodes.end());

    IsoparametricMapping map(basis, nodes);
    math::Vector<Real, 3> xi{Real(0.2), Real(-0.3), Real(0)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-12);
    EXPECT_NEAR(x[1], xi[1], 1e-12);

    auto J = map.jacobian(xi);
    EXPECT_NEAR(J(0,0), 1.0, 1e-12);
    EXPECT_NEAR(J(1,1), 1.0, 1e-12);
    EXPECT_NEAR(map.jacobian_determinant(xi), 1.0, 1e-12);

    auto xi_back = map.map_to_reference(x);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
}

TEST(IsoparametricMapping, FactoryCreatesLinearTriangle) {
    MappingRequest req{ElementType::Triangle3, 1, true, nullptr};
    // Node ordering must match LagrangeBasis/NodeOrderingConventions for Triangle3:
    // (0,0), (1,0), (0,1) on the reference unit right triangle.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0), Real(0), Real(0)}, // ref node (0,0)
        {Real(1), Real(0), Real(0)}, // ref node (1,0)
        {Real(0), Real(1), Real(0)}  // ref node (0,1)
    };

    auto mapping = MappingFactory::create(req, nodes);
    math::Vector<Real, 3> center{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    auto x = mapping->map_to_physical(center);
    EXPECT_NEAR(x[0], 1.0/3.0, 1e-12);
    EXPECT_NEAR(x[1], 1.0/3.0, 1e-12);

    auto q = GeometryValidator::evaluate(*mapping, center);
    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_LT(q.condition_number, 10.0);
}

TEST(IsoparametricMapping, IdentityHexJacobianAndInverse) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real, 3> xi{Real(0.1), Real(-0.2), Real(0.3)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-12);
    EXPECT_NEAR(x[1], xi[1], 1e-12);
    EXPECT_NEAR(x[2], xi[2], 1e-12);

    auto J = map.jacobian(xi);
    EXPECT_NEAR(J(0,0), 1.0, 1e-12);
    EXPECT_NEAR(J(1,1), 1.0, 1e-12);
    EXPECT_NEAR(J(2,2), 1.0, 1e-12);
    EXPECT_NEAR(map.jacobian_determinant(xi), 1.0, 1e-12);

    auto xi_back = map.map_to_reference(x);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
    EXPECT_NEAR(xi_back[2], xi[2], 1e-8);
}

TEST(IsoparametricMapping, GeometryMappingHelpers3DIdentity) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Hex8, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);
    math::Vector<Real,3> xi{Real(0.3), Real(-0.1), Real(0.2)};

    auto Jinv = map.jacobian_inverse(xi);
    EXPECT_NEAR(Jinv(0,0), 1.0, 1e-12);
    EXPECT_NEAR(Jinv(1,1), 1.0, 1e-12);
    EXPECT_NEAR(Jinv(2,2), 1.0, 1e-12);
    EXPECT_NEAR(map.jacobian_determinant(xi), 1.0, 1e-12);

    math::Vector<Real,3> g{Real(1), Real(2), Real(3)};
    auto gt = map.transform_gradient(g, xi);
    EXPECT_NEAR(gt[0], g[0], 1e-12);
    EXPECT_NEAR(gt[1], g[1], 1e-12);
    EXPECT_NEAR(gt[2], g[2], 1e-12);
}

TEST(MappingFactory, CreatesSubparametricMapping) {
    MappingRequest req{ElementType::Quad4, 0, false, nullptr};
    auto basis_geo = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 0);
    auto nodes = basis_geo->nodes();
    auto mapping = MappingFactory::create(req, nodes);
    auto* sub = dynamic_cast<SubparametricMapping*>(mapping.get());
    ASSERT_NE(sub, nullptr);
    EXPECT_EQ(sub->geometry_order(), 0);
    // With order 0 geometry the map is constant at the single node
    auto x = mapping->map_to_physical(math::Vector<Real,3>{Real(0.1), Real(-0.2), Real(0)});
    EXPECT_NEAR(x[0], nodes[0][0], 1e-12);
    EXPECT_NEAR(x[1], nodes[0][1], 1e-12);
}

TEST(MappingFactory, CreatesSuperparametricMapping) {
    MappingRequest req{ElementType::Quad4, 3, false, nullptr};
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 3);
    auto nodes = basis->nodes();
    auto mapping = MappingFactory::create(req, nodes);
    auto* sup = dynamic_cast<SuperparametricMapping*>(mapping.get());
    ASSERT_NE(sup, nullptr);
    EXPECT_EQ(sup->geometry_order(), 3);
    auto x = mapping->map_to_physical(math::Vector<Real,3>{Real(0.2), Real(0.3), Real(0)});
    EXPECT_NEAR(x[0], 0.2, 1e-12);
    EXPECT_NEAR(x[1], 0.3, 1e-12);
}

TEST(MappingFactory, CreatesWedge15GeometryUsingLinearWedge6) {
    // Geometry should be constructed using the 6 vertex nodes of Wedge15
    MappingRequest req{ElementType::Wedge15, 2, false, nullptr};
    // Start from canonical Wedge6 vertex nodes, then append dummy mid-edge nodes
    auto w6_basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
    const auto& w6_nodes = w6_basis->nodes();
    std::vector<math::Vector<Real,3>> nodes = {
        w6_nodes[0],
        w6_nodes[1],
        w6_nodes[2],
        w6_nodes[3],
        w6_nodes[4],
        w6_nodes[5],
        // remaining 9 nodes are higher-order DOFs, unused by linear geometry
        w6_nodes[0],
        w6_nodes[1],
        w6_nodes[2],
        w6_nodes[3],
        w6_nodes[4],
        w6_nodes[5],
        w6_nodes[0],
        w6_nodes[1],
        w6_nodes[2],
    };
    auto mapping = MappingFactory::create(req, nodes);
    EXPECT_EQ(mapping->dimension(), 3);
    // Check that mapping behaves like a linear wedge (affine mapping)
    math::Vector<Real,3> xi{Real(0.2), Real(0.1), Real(0.3)};
    auto x = mapping->map_to_physical(xi);
    EXPECT_GT(mapping->jacobian_determinant(xi), 0.0);
    // xi are barycentric (x,y) and z; with identity wedge geometry we expect:
    EXPECT_NEAR(x[0], xi[0], 1e-8);
    EXPECT_NEAR(x[1], xi[1], 1e-8);
    EXPECT_NEAR(x[2], xi[2], 1e-8);
}

TEST(MappingFactory, CreatesQuadraticPyramidGeometryUsingLinearPyramid5) {
    // Geometry should be constructed using the 5 vertex nodes of Pyramid13
    MappingRequest req{ElementType::Pyramid13, 2, false, nullptr};
    // Reference pyramid vertices plus quadratic mid-edge/face nodes (unused by geometry)
    std::vector<math::Vector<Real,3>> nodes = {
        // base z=0
        {Real(-1), Real(-1), Real(0)},  // v0
        {Real(1),  Real(-1), Real(0)},  // v1
        {Real(1),  Real(1),  Real(0)},  // v2
        {Real(-1), Real(1),  Real(0)},  // v3
        // apex
        {Real(0),  Real(0),  Real(1)},  // v4
        // edge midpoints and face nodes (schematic, unused by geometry)
        {Real(0),  Real(-1), Real(0)},
        {Real(1),  Real(0),  Real(0)},
        {Real(0),  Real(1),  Real(0)},
        {Real(-1), Real(0),  Real(0)},
        {Real(-0.5), Real(-0.5), Real(0.5)},
        {Real(0.5),  Real(-0.5), Real(0.5)},
        {Real(0.5),  Real(0.5),  Real(0.5)},
        {Real(-0.5), Real(0.5),  Real(0.5)},
    };
    auto mapping = MappingFactory::create(req, nodes);
    EXPECT_EQ(mapping->dimension(), 3);
    math::Vector<Real,3> xi{Real(0.1), Real(-0.2), Real(0.4)};
    auto x = mapping->map_to_physical(xi);
    EXPECT_GT(mapping->jacobian_determinant(xi), 0.0);
    EXPECT_NEAR(x[0], xi[0], 1e-8);
    EXPECT_NEAR(x[1], xi[1], 1e-8);
    EXPECT_NEAR(x[2], xi[2], 1e-8);
}

TEST(MappingFactory, Quad8SerendipityGeometry) {
    // Identity geometry mapping for Quad8 using serendipity basis
    MappingRequest req{ElementType::Quad8, 2, false, nullptr};
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
    auto mapping = MappingFactory::create(req, nodes);
    math::Vector<Real,3> xi{Real(0.1), Real(0.2), Real(0)};
    auto x = mapping->map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    auto q = GeometryValidator::evaluate(*mapping, xi);
    EXPECT_TRUE(q.positive_jacobian);
}

TEST(MappingFactory, Hex20SerendipityGeometry) {
    // Identity geometry mapping for Hex20 using serendipity basis
    MappingRequest req{ElementType::Hex20, 2, false, nullptr};
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
    auto mapping = MappingFactory::create(req, nodes);
    math::Vector<Real,3> xi{Real(0.1), Real(-0.1), Real(0.2)};
    auto x = mapping->map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);
    auto q = GeometryValidator::evaluate(*mapping, xi);
    EXPECT_TRUE(q.positive_jacobian);
}

TEST(MappingFactory, RespectsUserProvidedBasis) {
    auto hier = std::make_shared<basis::HierarchicalBasis>(ElementType::Quad4, 1);
    MappingRequest req{ElementType::Quad4, 1, false, hier};
    // Nodes sized to match hierarchical basis size
    std::vector<math::Vector<Real,3>> nodes(hier->size(), math::Vector<Real,3>{});
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        nodes[i][0] = static_cast<Real>(i) * Real(0.1);
        nodes[i][1] = static_cast<Real>(i) * Real(0.05);
    }
    auto mapping = MappingFactory::create(req, nodes);
    EXPECT_EQ(mapping->element_type(), ElementType::Quad4);
    EXPECT_EQ(mapping->dimension(), 2);
}

TEST(IsoparametricMapping, FactoryThrowsOnMismatchedNodes) {
    MappingRequest req{ElementType::Quad4, 1, false, nullptr};
    std::vector<math::Vector<Real,3>> bad_nodes = {
        {Real(0), Real(0), Real(0)},
        {Real(1), Real(0), Real(0)} // missing nodes
    };
    EXPECT_THROW(auto m = MappingFactory::create(req, bad_nodes), FEException);
}


TEST(IsoparametricMapping, IdentityWedgeLinear) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge6, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.2), Real(0.1), Real(-0.3)}; // xi+eta<=1 for tri, z in [-1,1]
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(IsoparametricMapping, IdentityWedgeQuadratic) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Wedge18, 2);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(0.2), Real(0.3)};
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(IsoparametricMapping, IdentityPyramidLinear) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Pyramid5, 1);
    auto nodes = basis->nodes();
    IsoparametricMapping map(basis, nodes);

    math::Vector<Real,3> xi{Real(0.1), Real(-0.2), Real(0.4)}; // z in [0,1]
    auto x = map.map_to_physical(xi);
    EXPECT_NEAR(x[0], xi[0], 1e-10);
    EXPECT_NEAR(x[1], xi[1], 1e-10);
    EXPECT_NEAR(x[2], xi[2], 1e-10);

    EXPECT_GT(map.jacobian_determinant(xi), 0.0);
}

TEST(GeometryValidator, HighAspectRatioQuadHasLargeConditionNumber) {
    auto basis = std::make_shared<basis::LagrangeBasis>(ElementType::Quad4, 1);
    // Physical rectangle with aspect ratio 1000:1 (2000 x 2).
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0),    Real(0), Real(0)},
        {Real(2000), Real(0), Real(0)},
        {Real(2000), Real(2), Real(0)},
        {Real(0),    Real(2), Real(0)}
    };
    IsoparametricMapping map(basis, nodes);
    const auto q = GeometryValidator::evaluate(map, math::Vector<Real,3>{Real(0), Real(0), Real(0)});
    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_NEAR(q.condition_number, 1000.0, 1e-6);
}

TEST(GeometryValidator, NearSingularTriangleReportsHugeConditionNumber) {
    // Skinny triangle: base 1000, height 1e-6 -> condition number ~ 1e9.
    std::vector<math::Vector<Real, 3>> nodes = {
        {Real(0),    Real(0),    Real(0)},
        {Real(1000), Real(0),    Real(0)},
        {Real(0),    Real(1e-6), Real(0)}
    };
    LinearMapping map(ElementType::Triangle3, nodes);
    const math::Vector<Real,3> xi{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    const auto q = GeometryValidator::evaluate(map, xi);
    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_TRUE(std::isfinite(q.condition_number));
    EXPECT_GT(q.condition_number, 1e8);
}
