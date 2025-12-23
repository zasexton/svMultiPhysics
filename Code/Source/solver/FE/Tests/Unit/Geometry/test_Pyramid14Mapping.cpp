/**
 * @file test_Pyramid14Mapping.cpp
 * @brief Tests for quadratic rational geometry mapping on Pyramid14.
 */

#include <gtest/gtest.h>

#include "FE/Geometry/MappingFactory.h"
#include "FE/Geometry/GeometryValidator.h"
#include "FE/Geometry/InverseMapping.h"
#include "FE/Basis/NodeOrderingConventions.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::geometry;

TEST(Pyramid14Mapping, IdentityRationalGeometry) {
    using svmp::FE::basis::NodeOrdering;

    // Build identity geometry for reference Pyramid14 using its canonical nodes
    std::vector<math::Vector<Real,3>> nodes;
    const std::size_t nn = NodeOrdering::num_nodes(ElementType::Pyramid14);
    nodes.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Pyramid14, i));
    }

    MappingRequest req{ElementType::Pyramid14, 2, false, nullptr};
    auto mapping = MappingFactory::create(req, nodes);
    EXPECT_EQ(mapping->dimension(), 3);

    // Map a few interior points and verify near-identity
    const math::Vector<Real,3> xis[] = {
        {Real(0.1), Real(-0.2), Real(0.3)},
        {Real(-0.25), Real(0.15), Real(0.6)},
        {Real(0.0), Real(0.0), Real(0.5)}
    };

    for (const auto& xi : xis) {
        auto x = mapping->map_to_physical(xi);
        auto q = GeometryValidator::evaluate(*mapping, xi);
        EXPECT_TRUE(q.positive_jacobian);
        EXPECT_TRUE(std::isfinite(q.condition_number));

        EXPECT_NEAR(x[0], xi[0], 1e-8);
        EXPECT_NEAR(x[1], xi[1], 1e-8);
        EXPECT_NEAR(x[2], xi[2], 1e-8);
    }
}

TEST(Pyramid14Mapping, IdentityRationalGeometryBoundaryAndNearApex) {
    using svmp::FE::basis::NodeOrdering;

    std::vector<math::Vector<Real,3>> nodes;
    const std::size_t nn = NodeOrdering::num_nodes(ElementType::Pyramid14);
    nodes.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Pyramid14, i));
    }

    MappingRequest req{ElementType::Pyramid14, 2, false, nullptr};
    auto mapping = MappingFactory::create(req, nodes);

    const math::Vector<Real,3> xis[] = {
        {Real(1.0),  Real(0.0),  Real(0.0)},   // base boundary
        {Real(0.0),  Real(-1.0), Real(0.0)},   // base boundary
        {Real(0.0),  Real(0.0),  Real(0.95)},  // near apex
        {Real(0.1),  Real(0.0),  Real(0.9)},   // near apex (within |x|<=1-z)
        {Real(-0.2), Real(0.15), Real(0.6)}    // interior
    };

    for (const auto& xi : xis) {
        auto x = mapping->map_to_physical(xi);
        auto q = GeometryValidator::evaluate(*mapping, xi);
        EXPECT_TRUE(q.positive_jacobian);
        EXPECT_TRUE(std::isfinite(q.condition_number));

        EXPECT_NEAR(x[0], xi[0], 1e-8);
        EXPECT_NEAR(x[1], xi[1], 1e-8);
        EXPECT_NEAR(x[2], xi[2], 1e-8);

        auto xi_back = mapping->map_to_reference(x);
        EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
        EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
        EXPECT_NEAR(xi_back[2], xi[2], 1e-8);
    }
}

TEST(Pyramid14Mapping, AffineDistortionIsReproducedAndInverted) {
    using svmp::FE::basis::NodeOrdering;

    std::vector<math::Vector<Real,3>> nodes;
    const std::size_t nn = NodeOrdering::num_nodes(ElementType::Pyramid14);
    nodes.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Pyramid14, i));
    }

    auto affine = [](const math::Vector<Real,3>& p) {
        return math::Vector<Real,3>{
            Real(2) * p[0] + Real(0.1) * p[1] + Real(0.2) * p[2] + Real(0.5),
            Real(-0.3) * p[0] + Real(1.5) * p[1] + Real(0.0) * p[2] + Real(-0.25),
            Real(0.1) * p[0] + Real(-0.2) * p[1] + Real(0.75) * p[2] + Real(1.0)
        };
    };

    for (auto& n : nodes) {
        n = affine(n);
    }

    MappingRequest req{ElementType::Pyramid14, 2, false, nullptr};
    auto mapping = MappingFactory::create(req, nodes);

    const math::Vector<Real,3> xis[] = {
        {Real(0.1), Real(-0.2), Real(0.3)},
        {Real(-0.05), Real(0.05), Real(0.8)},
        {Real(0.0), Real(0.0), Real(0.5)}
    };

    for (const auto& xi : xis) {
        const auto x = mapping->map_to_physical(xi);
        const auto x_ref = affine(xi);
        EXPECT_NEAR(x[0], x_ref[0], 1e-8);
        EXPECT_NEAR(x[1], x_ref[1], 1e-8);
        EXPECT_NEAR(x[2], x_ref[2], 1e-8);

        const auto xi_back = mapping->map_to_reference(x);
        EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
        EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
        EXPECT_NEAR(xi_back[2], xi[2], 1e-8);
    }
}

TEST(Pyramid14Mapping, CurvedGeometryInverseMappingRobust) {
    using svmp::FE::basis::NodeOrdering;

    std::vector<math::Vector<Real,3>> nodes;
    const std::size_t nn = NodeOrdering::num_nodes(ElementType::Pyramid14);
    nodes.reserve(nn);
    for (std::size_t i = 0; i < nn; ++i) {
        nodes.push_back(NodeOrdering::get_node_coords(ElementType::Pyramid14, i));
    }

    // Keep the 5 vertices fixed; perturb higher-order nodes mildly to induce curvature.
    for (std::size_t i = 5; i < nodes.size(); ++i) {
        nodes[i][2] += Real(0.02);
    }

    MappingRequest req{ElementType::Pyramid14, 2, false, nullptr};
    auto mapping = MappingFactory::create(req, nodes);

    const math::Vector<Real,3> xi{Real(0.05), Real(-0.1), Real(0.4)};
    const auto x = mapping->map_to_physical(xi);

    InverseMappingOptions opts;
    opts.use_line_search = true;
    const auto xi_back = InverseMapping::solve_with_options(*mapping, x, math::Vector<Real,3>{}, opts);
    EXPECT_NEAR(xi_back[0], xi[0], 1e-8);
    EXPECT_NEAR(xi_back[1], xi[1], 1e-8);
    EXPECT_NEAR(xi_back[2], xi[2], 1e-8);

    const auto q = GeometryValidator::evaluate(*mapping, xi);
    EXPECT_TRUE(q.positive_jacobian);
    EXPECT_TRUE(std::isfinite(q.condition_number));
}
