/**
 * @file test_BubbleBasis.cpp
 * @brief Tests for scalar interior bubble basis functions
 */

#include <gtest/gtest.h>

#include "FE/Basis/BubbleBasis.h"
#include "FE/Basis/BasisFactory.h"
#include "FE/Basis/NodeOrderingConventions.h"

#include <array>

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

TEST(BubbleBasis, LineMetadata) {
    BubbleBasis basis(ElementType::Line2);
    EXPECT_EQ(basis.dimension(), 1);
    EXPECT_EQ(basis.order(), 2);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, HexMetadata) {
    BubbleBasis basis(ElementType::Hex8);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.order(), 2);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, WedgeMetadata) {
    BubbleBasis basis(ElementType::Wedge6);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.order(), 5);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, PyramidMetadata) {
    BubbleBasis basis(ElementType::Pyramid5);
    EXPECT_EQ(basis.dimension(), 3);
    EXPECT_EQ(basis.order(), 5);
    EXPECT_EQ(basis.size(), 1u);
}

TEST(BubbleBasis, AliasVariantsConstructWithTopologyIntrinsicOrders) {
    struct Case {
        ElementType alias_type;
        int expected_dimension;
        int expected_order;
    };

    const Case cases[] = {
        {ElementType::Line3, 1, 2},
        {ElementType::Triangle6, 2, 3},
        {ElementType::Tetra10, 3, 4},
        {ElementType::Quad8, 2, 2},
        {ElementType::Quad9, 2, 2},
        {ElementType::Hex20, 3, 2},
        {ElementType::Hex27, 3, 2},
        {ElementType::Wedge15, 3, 5},
        {ElementType::Wedge18, 3, 5},
        {ElementType::Pyramid13, 3, 5},
        {ElementType::Pyramid14, 3, 5},
    };

    for (const auto& c : cases) {
        BubbleBasis basis(c.alias_type);
        EXPECT_EQ(basis.element_type(), c.alias_type);
        EXPECT_EQ(basis.dimension(), c.expected_dimension)
            << "alias element type " << static_cast<int>(c.alias_type);
        EXPECT_EQ(basis.order(), c.expected_order)
            << "alias element type " << static_cast<int>(c.alias_type);
        EXPECT_EQ(basis.size(), 1u);
    }
}

TEST(BubbleBasis, AliasVariantsMatchCanonicalEvaluations) {
    struct Case {
        ElementType alias_type;
        ElementType canonical_type;
        math::Vector<Real, 3> xi;
        int dimension;
    };

    const Case cases[] = {
        {ElementType::Line3, ElementType::Line2, {Real(0.23), Real(0), Real(0)}, 1},
        {ElementType::Triangle6, ElementType::Triangle3, {Real(0.2), Real(0.3), Real(0)}, 2},
        {ElementType::Tetra10, ElementType::Tetra4, {Real(0.12), Real(0.18), Real(0.15)}, 3},
        {ElementType::Quad8, ElementType::Quad4, {Real(0.3), Real(-0.25), Real(0)}, 2},
        {ElementType::Hex20, ElementType::Hex8, {Real(0.1), Real(-0.2), Real(0.15)}, 3},
        {ElementType::Wedge15, ElementType::Wedge6, {Real(0.2), Real(0.25), Real(-0.1)}, 3},
        {ElementType::Pyramid13, ElementType::Pyramid5, {Real(0.08), Real(-0.06), Real(0.2)}, 3},
    };

    for (const auto& c : cases) {
        BubbleBasis alias_basis(c.alias_type);
        BubbleBasis canonical_basis(c.canonical_type);

        std::vector<Real> alias_values;
        std::vector<Real> canonical_values;
        alias_basis.evaluate_values(c.xi, alias_values);
        canonical_basis.evaluate_values(c.xi, canonical_values);

        ASSERT_EQ(alias_values.size(), canonical_values.size());
        EXPECT_NEAR(alias_values[0], canonical_values[0], 1e-14)
            << "alias element type " << static_cast<int>(c.alias_type);

        std::vector<Gradient> alias_gradients;
        std::vector<Gradient> canonical_gradients;
        alias_basis.evaluate_gradients(c.xi, alias_gradients);
        canonical_basis.evaluate_gradients(c.xi, canonical_gradients);

        ASSERT_EQ(alias_gradients.size(), canonical_gradients.size());
        for (int d = 0; d < c.dimension; ++d) {
            const auto sd = static_cast<std::size_t>(d);
            EXPECT_NEAR(alias_gradients[0][sd], canonical_gradients[0][sd], 1e-14)
                << "alias element type " << static_cast<int>(c.alias_type)
                << ", dim " << d;
        }
    }
}

TEST(BubbleBasis, LineZeroAtBoundaryPositiveInside) {
    BubbleBasis basis(ElementType::Line2);
    std::vector<Real> vals;

    basis.evaluate_values({Real(-1), Real(0), Real(0)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(1), Real(0), Real(0)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0), Real(0), Real(0)}, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-14);
}

TEST(BubbleBasis, TriangleZeroOnEdgesPositiveInside) {
    BubbleBasis basis(ElementType::Triangle3);
    std::vector<Real> vals;

    // Vertices: bubble = 0
    for (std::size_t i = 0; i < 3; ++i) {
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Triangle3, i);
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
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Tetra4, i);
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
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Quad4, i);
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
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Hex8, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    math::Vector<Real, 3> center{Real(0), Real(0), Real(0)};
    basis.evaluate_values(center, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-14);
}

TEST(BubbleBasis, WedgeZeroOnBoundaryPositiveInside) {
    BubbleBasis basis(ElementType::Wedge6);
    std::vector<Real> vals;

    for (std::size_t i = 0; i < 6; ++i) {
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Wedge6, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    basis.evaluate_values({Real(0.2), Real(0.3), Real(-1)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0.2), Real(0.3), Real(1)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0.0), Real(0.3), Real(0.1)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0.2), Real(0.0), Real(-0.4)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0.4), Real(0.6), Real(0.2)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);

    basis.evaluate_values({Real(1.0/3.0), Real(1.0/3.0), Real(0)}, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-12);
}

TEST(BubbleBasis, PyramidZeroOnBoundaryPositiveInside) {
    BubbleBasis basis(ElementType::Pyramid5);
    std::vector<Real> vals;

    for (std::size_t i = 0; i < 5; ++i) {
        auto xi = ReferenceNodeLayout::get_node_coords(ElementType::Pyramid5, i);
        basis.evaluate_values(xi, vals);
        EXPECT_NEAR(vals[0], 0.0, 1e-14) << "Vertex " << i;
    }

    basis.evaluate_values({Real(0), Real(0), Real(0)}, vals);
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0.75), Real(0), Real(0.25)}, vals); // x = 1-z
    EXPECT_NEAR(vals[0], 0.0, 1e-14);
    basis.evaluate_values({Real(0), Real(-0.6), Real(0.4)}, vals); // y = -(1-z)
    EXPECT_NEAR(vals[0], 0.0, 1e-14);

    basis.evaluate_values({Real(0), Real(0), Real(0.2)}, vals);
    EXPECT_NEAR(vals[0], 1.0, 1e-12);
}

TEST(BubbleBasis, GradientMatchesFiniteDifference) {
    const struct Case {
        ElementType type;
        math::Vector<Real, 3> xi;
        int dim;
    } cases[] = {
        {ElementType::Line2, {Real(0.2), Real(0), Real(0)}, 1},
        {ElementType::Triangle3, {Real(0.2), Real(0.3), Real(0)}, 2},
        {ElementType::Tetra4, {Real(0.1), Real(0.2), Real(0.15)}, 3},
        {ElementType::Quad4, {Real(0.3), Real(-0.2), Real(0)}, 2},
        {ElementType::Hex8, {Real(0.2), Real(-0.3), Real(0.1)}, 3},
        {ElementType::Wedge6, {Real(0.2), Real(0.15), Real(0.1)}, 3},
        {ElementType::Pyramid5, {Real(0.1), Real(-0.08), Real(0.25)}, 3},
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

TEST(BubbleBasis, PyramidHessianMatchesGradientFiniteDifferenceAtNonSymmetricPoints) {
    BubbleBasis basis(ElementType::Pyramid5);
    const std::array<math::Vector<Real, 3>, 3> points = {{
        {Real(0.11), Real(-0.07), Real(0.24)},
        {Real(-0.19), Real(0.08), Real(0.42)},
        {Real(0.05), Real(0.21), Real(0.31)}
    }};

    const Real eps = Real(1e-6);
    for (const auto& xi : points) {
        std::vector<Hessian> hessians;
        basis.evaluate_hessians(xi, hessians);
        ASSERT_EQ(hessians.size(), 1u);

        for (int col = 0; col < 3; ++col) {
            auto xi_p = xi;
            auto xi_m = xi;
            xi_p[static_cast<std::size_t>(col)] += eps;
            xi_m[static_cast<std::size_t>(col)] -= eps;

            std::vector<Gradient> grad_p;
            std::vector<Gradient> grad_m;
            basis.evaluate_gradients(xi_p, grad_p);
            basis.evaluate_gradients(xi_m, grad_m);
            ASSERT_EQ(grad_p.size(), 1u);
            ASSERT_EQ(grad_m.size(), 1u);

            for (int row = 0; row < 3; ++row) {
                const auto row_index = static_cast<std::size_t>(row);
                const auto col_index = static_cast<std::size_t>(col);
                const Real fd = (grad_p[0][static_cast<std::size_t>(row)] -
                                 grad_m[0][static_cast<std::size_t>(row)]) /
                                (Real(2) * eps);
                EXPECT_NEAR(hessians[0](row_index, col_index), fd, Real(2e-5))
                    << "xi=(" << xi[0] << ", " << xi[1] << ", " << xi[2]
                    << "), row=" << row << ", col=" << col;
            }
        }
    }
}

TEST(BubbleBasis, GradientZeroAtSymmetricCentroid) {
    // By symmetry, the gradient of the bubble at the centroid should be zero
    // for symmetric reference elements (Tri centroid, Quad center, Hex center).
    {
        BubbleBasis basis(ElementType::Line2);
        std::vector<Gradient> grads;
        basis.evaluate_gradients({Real(0), Real(0), Real(0)}, grads);
        EXPECT_NEAR(grads[0][0], 0.0, 1e-14);
    }
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
    {
        BubbleBasis basis(ElementType::Wedge6);
        std::vector<Gradient> grads;
        basis.evaluate_gradients({Real(1.0/3.0), Real(1.0/3.0), Real(0)}, grads);
        EXPECT_NEAR(grads[0][0], 0.0, 1e-12);
        EXPECT_NEAR(grads[0][1], 0.0, 1e-12);
        EXPECT_NEAR(grads[0][2], 0.0, 1e-12);
    }
    {
        BubbleBasis basis(ElementType::Pyramid5);
        std::vector<Gradient> grads;
        basis.evaluate_gradients({Real(0), Real(0), Real(0.2)}, grads);
        EXPECT_NEAR(grads[0][0], 0.0, 1e-12);
        EXPECT_NEAR(grads[0][1], 0.0, 1e-12);
        EXPECT_NEAR(grads[0][2], 0.0, 1e-12);
    }
}

TEST(BubbleBasis, UnsupportedElementThrows) {
    EXPECT_THROW({ BubbleBasis basis(ElementType::Unknown); }, BasisElementCompatibilityException);
}

TEST(BubbleBasis, FactoryCreatesBubble) {
    BasisRequest req;
    req.element_type = ElementType::Triangle3;
    req.basis_type = BasisType::Bubble;
    req.continuity = Continuity::C0;
    req.field_type = FieldType::Scalar;
    auto basis = basis_factory::create(req);
    EXPECT_EQ(basis->basis_type(), BasisType::Bubble);
    EXPECT_EQ(basis->size(), 1u);
}

TEST(BubbleBasis, FactoryRejectsExplicitOrder) {
    BasisRequest req;
    req.element_type = ElementType::Triangle3;
    req.basis_type = BasisType::Bubble;
    req.continuity = Continuity::C0;
    req.field_type = FieldType::Scalar;
    req.order = 3;

    EXPECT_THROW((void)basis_factory::create(req), BasisConfigurationException);
}
