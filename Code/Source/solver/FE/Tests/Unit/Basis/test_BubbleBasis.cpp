/**
 * @file test_BubbleBasis.cpp
 * @brief Tests for scalar interior bubble basis functions
 */

#include <gtest/gtest.h>

#include "FE/Basis/BubbleBasis.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/NodeOrderingConventions.h"

using namespace svmp::FE;
using namespace svmp::FE::basis;

TEST(BubbleBasis, TriangleMetadata) {
    BubbleBasis basis(ElementType::Triangle3);
    EXPECT_EQ(basis.basis_type(), BasisType::Bubble);
    EXPECT_EQ(basis.element_type(), ElementType::Triangle3);
    EXPECT_EQ(basis.dimension(), 2);
    EXPECT_EQ(basis.order(), 3);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, TetraMetadata) {
    BubbleBasis basis(ElementType::Tetra4);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.order(), 4);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, QuadMetadata) {
    BubbleBasis basis(ElementType::Quad4);
    EXPECT_EQ(basis.dimension(), 2);
    EXPECT_EQ(basis.order(), 2);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, HexMetadata) {
    BubbleBasis basis(ElementType::Hex8);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.order(), 2);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, TriangleZeroOnEdgesPositiveInside) {
    BubbleBasis basis(ElementType::Triangle3);
    std::vector<Real> vals;

    // Vertices: bubble = 0
    for (std::size_t i = 0; i < 3; ++i) {
        auto xi = NodeOrdering::get_node_coords(ElementType::Triangle3, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    // Edge midpoints: bubble = 0
    const math::Vector<Real, 3> edge_mids[] = {
        {Real(0.5), Real(0), Real(0)},    // edge 0-1
        {Real(0.5), Real(0.5), Real(0)},  // edge 1-2
        {Real(0), Real(0.5), Real(0)},    // edge 2-0
    };
    for (const auto& xi : edge_mids) {
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14);
    }

    // Centroid: bubble > 0
    math::Vector<Real, 3> centroid{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
    basis.evaluate_values(centroid, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-12);  // 27 * (1/3)^3 = 1
}

TEST(BubbleBasis, TetraZeroOnFacesPositiveInside) {
    BubbleBasis basis(ElementType::Tetra4);
    std::vector<Real> vals;

    // All 4 vertices: bubble = 0
    for (std::size_t i = 0; i < 4; ++i) {
        auto xi = NodeOrdering::get_node_coords(ElementType::Tetra4, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    // Centroid: bubble > 0
    math::Vector<Real, 3> centroid{Real(0.25), Real(0.25), Real(0.25)};
    basis.evaluate_values(centroid, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-12);  // 256 * (1/4)^4 = 1
}

TEST(BubbleBasis, QuadZeroOnEdgesPositiveInside) {
    BubbleBasis basis(ElementType::Quad4);
    std::vector<Real> vals;

    // All 4 vertices: bubble = 0
    for (std::size_t i = 0; i < 4; ++i) {
        auto xi = NodeOrdering::get_node_coords(ElementType::Quad4, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    // Edge midpoints: bubble = 0
    const math::Vector<Real, 3> edge_mids[] = {
        {Real(0), Real(-1), Real(0)},
        {Real(1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)},
        {Real(-1), Real(0), Real(0)},
    };
    for (const auto& xi : edge_mids) {
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14);
    }

    // Center: bubble = 1
    math::Vector<Real, 3> center{Real(0), Real(0), Real(0)};
    basis.evaluate_values(center, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-14);
}

TEST(BubbleBasis, HexZeroOnFacesPositiveInside) {
    BubbleBasis basis(ElementType::Hex8);
    std::vector<Real> vals;

    for (std::size_t i = 0; i < 8; ++i) {
        auto xi = NodeOrdering::get_node_coords(ElementType::Hex8, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    math::Vector<Real, 3> center{Real(0), Real(0), Real(0)};
    basis.evaluate_values(center, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-14);
}

TEST(BubbleBasis, GradientMatchesFiniteDifference) {
    const struct Case {
        ElementType type;
        math::Vector<Real, 3> xi;
        int dim;
    } cases[] = {
        {ElementType::Triangle3, {Real(0.2), Real(0.3), Real(0)}, 2},
        {ElementType::Tetra4, {Real(0.1), Real(0.2), Real(0.15)}, 3},
        {ElementType::Quad4, {Real(0.3), Real(-0.2), Real(0)}, 2},
        {ElementType::Hex8, {Real(0.2), Real(-0.3), Real(0.1)}, 3},
    };

    const Real eps = Real(1e-6);
    for (const auto& c : cases) {
        BubbleBasis basis(c.type);
        std::vector<Gradient> grads;
        basis.evaluate_gradients(c.xi, grads);
        ASSERT_EQ(grads.size(), 1u);

        for (int d = 0; d < c.dim; ++d) {
            auto xi_p = c.xi, xi_m = c.xi;
            xi_p[static_cast<std::size_t>(d)] += eps;
            xi_m[static_cast<std::size_t>(d)] -= eps;

            std::vector<Real> vp, vm;
            basis.evaluate_values(xi_p, vp);
            basis.evaluate_values(xi_m, vm);

            const Real fd = (vp[0] - vm[0]) / (Real(2) * eps);
            EXPECT_NEAR(grads[0][static_cast<std::size_t>(d)], fd, 1e-5)
                << "Element " << static_cast<int>(c.type) << ", dim " << d;
        }
    }
}

TEST(BubbleBasis, GradientZeroAtSymmetricCentroid) {
    // By symmetry, the gradient of the bubble at the centroid should be zero
    // for symmetric reference elements (Tri centroid, Quad center, Hex center).
    {
        BubbleBasis basis(ElementType::Triangle3);
        math::Vector<Real, 3> centroid{Real(1.0/3.0), Real(1.0/3.0), Real(0)};
        std::vector<Gradient> grads;
        basis.evaluate_gradients(centroid, grads);
        EXPECT_NEAR(grads[0][0], 0.0, 1e-12);
        EXPECT_NEAR(grads[0][1], 0.0, 1e-12);
    }
    {
        BubbleBasis basis(ElementType::Quad4);
        math::Vector<Real, 3> center{Real(0), Real(0), Real(0)};
        std::vector<Gradient> grads;
        basis.evaluate_gradients(center, grads);
        EXPECT_NEAR(grads[0][0], 0.0, 1e-14);
        EXPECT_NEAR(grads[0][1], 0.0, 1e-14);
    }
}

TEST(BubbleBasis, UnsupportedElementThrows) {
    EXPECT_THROW(BubbleBasis(ElementType::Wedge6), FEException);
    EXPECT_THROW(BubbleBasis(ElementType::Pyramid5), FEException);
    EXPECT_THROW(BubbleBasis(ElementType::Unknown), FEException);
}

TEST(BubbleBasis, FactoryCreatesBubble) {
    BasisRequest req{ElementType::Triangle3, BasisType::Bubble, 3, Continuity::C0, FieldType::Scalar};
    auto basis = BasisFactory::create(req);
    EXPECT_EQ(basis->basis_type(), BasisType::Bubble);
    EXPECT_EQ(basis->size(), 1u);
}
