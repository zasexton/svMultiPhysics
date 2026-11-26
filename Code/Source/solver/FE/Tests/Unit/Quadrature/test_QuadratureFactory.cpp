/**
 * @file test_QuadratureFactory.cpp
 * @brief Unit tests for quadrature factory and cache
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/GaussLobattoQuadrature.h"
#include "FE/Core/FEException.h"
#include "FE/Core/Types.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

TEST(QuadratureFactory, ProducesCachedRules) {
    auto q1 = QuadratureFactory::create(ElementType::Quad4, 3);
    auto q2 = QuadratureFactory::create(ElementType::Quad4, 3);

    ASSERT_TRUE(q1);
    ASSERT_TRUE(q2);
    EXPECT_EQ(q1.get(), q2.get());  // cache hit
}

TEST(QuadratureFactory, DifferentiatesRuleTypes) {
    auto gauss = QuadratureFactory::create(ElementType::Line2, 3, QuadratureType::GaussLegendre);
    auto lobatto = QuadratureFactory::create(ElementType::Line2, 3, QuadratureType::GaussLobatto);

    ASSERT_TRUE(gauss);
    ASSERT_TRUE(lobatto);
    EXPECT_NE(gauss->num_points(), lobatto->num_points());
}

TEST(QuadratureFactory, RecommendedOrderScaling) {
    EXPECT_GE(QuadratureFactory::recommended_order(2, true), 4);
    EXPECT_GE(QuadratureFactory::recommended_order(3, false), 5);
}

TEST(QuadratureFactory, CreatesAllCanonicalElementTypes) {
    std::vector<ElementType> elems = {
        ElementType::Line2, ElementType::Triangle3, ElementType::Quad4,
        ElementType::Tetra4, ElementType::Hex8, ElementType::Wedge6, ElementType::Pyramid5
    };
    for (auto e : elems) {
        auto rule = QuadratureFactory::create(e, 3);
        ASSERT_TRUE(rule) << "Failed to create rule for element " << static_cast<int>(e);
        EXPECT_GT(rule->num_points(), 0u);
        // Constant integration should match reference measure
        double sum = 0.0;
        for (std::size_t i = 0; i < rule->num_points(); ++i) sum += rule->weight(i);
        EXPECT_NEAR(sum, rule->reference_measure(), 1e-12);
    }
}

TEST(QuadratureFactory, ReducedIntegrationStillIntegratesConstant) {
    auto hex_reduced = QuadratureFactory::create(ElementType::Hex8, 3, QuadratureType::Reduced, false);
    ASSERT_TRUE(hex_reduced);
    double sum = 0.0;
    for (std::size_t i = 0; i < hex_reduced->num_points(); ++i) sum += hex_reduced->weight(i);
    EXPECT_NEAR(sum, hex_reduced->reference_measure(), 1e-12);
}

TEST(QuadratureFactory, ReducedIntegrationSupportsSimplexAndWedge) {
    struct Case {
        ElementType element;
        int order;
    };
    // Choose orders that ensure (order-1) changes the underlying point count.
    const std::vector<Case> cases = {
        {ElementType::Line2, 4},
        {ElementType::Triangle3, 5},
        {ElementType::Tetra4, 3},
        {ElementType::Wedge6, 6},
    };

    for (const auto& c : cases) {
        auto full = QuadratureFactory::create(c.element, c.order, QuadratureType::GaussLegendre, false);
        auto reduced = QuadratureFactory::create(c.element, c.order, QuadratureType::Reduced, false);
        ASSERT_TRUE(full);
        ASSERT_TRUE(reduced);

        EXPECT_LT(reduced->num_points(), full->num_points())
            << "Reduced integration did not reduce point count for element "
            << static_cast<int>(c.element) << " at order " << c.order;

        double sum = 0.0;
        for (std::size_t i = 0; i < reduced->num_points(); ++i) sum += reduced->weight(i);
        EXPECT_NEAR(sum, reduced->reference_measure(), 1e-12);
    }
}

TEST(QuadratureFactory, GaussLobattoUsesCeilForEvenOrders) {
    // Order 4 requires at least 4 Gauss-Lobatto points (2n-3 >= 4).
    auto quad = QuadratureFactory::create(ElementType::Line2, 4, QuadratureType::GaussLobatto, false);
    ASSERT_TRUE(quad);
    EXPECT_EQ(quad->num_points(), 4u);

    double acc = 0.0;
    for (std::size_t i = 0; i < quad->num_points(); ++i) {
        const double x = quad->point(i)[0];
        acc += quad->weight(i) * (x * x * x * x);
    }
    EXPECT_NEAR(acc, 2.0 / 5.0, 1e-12);
}

TEST(QuadratureFactory, GaussLobattoRawWeightsMatchKnownFivePointRule) {
    const auto [nodes, weights] = GaussLobattoQuadrature1D::generate_raw(5);
    ASSERT_EQ(nodes.size(), 5u);
    ASSERT_EQ(weights.size(), 5u);

    EXPECT_NEAR(nodes[0], -1.0, 1e-14);
    EXPECT_NEAR(nodes[1], -std::sqrt(3.0 / 7.0), 1e-12);
    EXPECT_NEAR(nodes[2], 0.0, 1e-12);
    EXPECT_NEAR(nodes[3], std::sqrt(3.0 / 7.0), 1e-12);
    EXPECT_NEAR(nodes[4], 1.0, 1e-14);

    EXPECT_NEAR(weights[0], 1.0 / 10.0, 1e-14);
    EXPECT_NEAR(weights[1], 49.0 / 90.0, 1e-14);
    EXPECT_NEAR(weights[2], 32.0 / 45.0, 1e-14);
    EXPECT_NEAR(weights[3], 49.0 / 90.0, 1e-14);
    EXPECT_NEAR(weights[4], 1.0 / 10.0, 1e-14);
}

TEST(QuadratureFactory, PointElementCreates0DQuadrature) {
    auto rule = QuadratureFactory::create(ElementType::Point1, 1, QuadratureType::GaussLegendre, false);
    ASSERT_TRUE(rule);
    EXPECT_EQ(rule->cell_family(), svmp::CellFamily::Point);
    EXPECT_EQ(rule->dimension(), 0);
    ASSERT_EQ(rule->num_points(), 1u);
    EXPECT_NEAR(rule->weight(0), 1.0, 1e-15);
    EXPECT_NEAR(rule->reference_measure(), 1.0, 1e-15);
}

TEST(QuadratureFactory, UnsupportedQuadratureTypesThrow) {
    for (QuadratureType t : {QuadratureType::Newton, QuadratureType::Composite, QuadratureType::Custom}) {
        try {
            (void)QuadratureFactory::create(ElementType::Hex8, 3, t, false);
            FAIL() << "Expected FEException for QuadratureType=" << static_cast<int>(t);
        } catch (const FEException& e) {
            EXPECT_EQ(e.status(), FEStatus::NotImplemented);
        }
    }
}

TEST(QuadratureFactory, PyramidReducedIntegrationReducesPoints) {
    auto full = QuadratureFactory::create(ElementType::Pyramid5, 4, QuadratureType::GaussLegendre, false);
    auto reduced = QuadratureFactory::create(ElementType::Pyramid5, 4, QuadratureType::Reduced, false);
    ASSERT_TRUE(full);
    ASSERT_TRUE(reduced);

    EXPECT_LT(reduced->num_points(), full->num_points());

    double sum = 0.0;
    for (std::size_t i = 0; i < reduced->num_points(); ++i) sum += reduced->weight(i);
    EXPECT_NEAR(sum, reduced->reference_measure(), 1e-12);
}

TEST(QuadratureFactory, PyramidGaussLobattoThrows) {
    try {
        (void)QuadratureFactory::create(ElementType::Pyramid5, 3, QuadratureType::GaussLobatto, false);
        FAIL() << "Expected FEException for Pyramid5 with GaussLobatto";
    } catch (const FEException& e) {
        EXPECT_EQ(e.status(), FEStatus::NotImplemented);
    }
}

namespace {
// Whether every point of a has a point of b at the same place and weight.
bool sameWeightedPointSet(const QuadratureRule& a, const QuadratureRule& b) {
    if (a.num_points() != b.num_points()) return false;
    for (std::size_t i = 0; i < a.num_points(); ++i) {
        bool found = false;
        for (std::size_t j = 0; j < b.num_points() && !found; ++j) {
            Real distance = std::abs(a.weight(i) - b.weight(j));
            for (int d = 0; d < 3; ++d) distance += std::abs(a.point(i)[d] - b.point(j)[d]);
            found = distance < 1e-14;
        }
        if (!found) return false;
    }
    return true;
}
} // namespace

TEST(QuadratureFactory, CreateAcceptsPositionBasedType) {
    auto tri_pos = QuadratureFactory::create(ElementType::Triangle3, 2, QuadratureType::PositionBased, false);
    auto tri_gauss = QuadratureFactory::create(ElementType::Triangle3, 2, QuadratureType::GaussLegendre, false);
    ASSERT_TRUE(tri_pos);
    ASSERT_TRUE(tri_gauss);
    EXPECT_EQ(tri_pos->cell_family(), svmp::CellFamily::Triangle);
    EXPECT_EQ(tri_pos->dimension(), 2);
    EXPECT_EQ(tri_pos->num_points(), 3u);
    // The legacy Gaussian placement is the symmetric degree-2 default rule.
    EXPECT_TRUE(sameWeightedPointSet(*tri_pos, *tri_gauss));

    auto tet_pos = QuadratureFactory::create(ElementType::Tetra4, 2, QuadratureType::PositionBased, false);
    auto tet_gauss = QuadratureFactory::create(ElementType::Tetra4, 2, QuadratureType::GaussLegendre, false);
    ASSERT_TRUE(tet_pos);
    ASSERT_TRUE(tet_gauss);
    EXPECT_EQ(tet_pos->cell_family(), svmp::CellFamily::Tetra);
    EXPECT_EQ(tet_pos->dimension(), 3);
    EXPECT_EQ(tet_pos->num_points(), 4u);
    EXPECT_TRUE(sameWeightedPointSet(*tet_pos, *tet_gauss));
}

TEST(QuadratureFactory, InvalidInputsThrow) {
    EXPECT_THROW(QuadratureFactory::create(ElementType::Hex8, 0), FEException);
    EXPECT_THROW(QuadratureFactory::create(ElementType::Unknown, 2), FEException);
}

TEST(QuadratureFactory, DefaultSimplexRulesDoNotDependOnVertexOrdering) {
    // A streamline-diffusion integrand tau (a.grad phi)^2 on the unit
    // triangle, a = (-x, y), phi = x + y - 1/2, tau = (a^T G a)^(-1/2) with
    // the regular-simplex metric G = [[2, 1], [1, 2]] / sqrt(3), is not a
    // polynomial. Integrated through each cyclic vertex relabeling of the
    // same triangle it must not depend on the numbering.
    const auto supg = [](Real x, Real y) {
        const Real streamline = y - x;
        const Real directional =
            (Real(2) * x * x - Real(2) * x * y + Real(2) * y * y) / std::sqrt(Real(3));
        return directional > Real(0) ? streamline * streamline / std::sqrt(directional) : Real(0);
    };
    const std::array<std::array<Real, 2>, 3> vertices{{{0, 0}, {1, 0}, {0, 1}}};
    for (int order = 1; order <= 20; ++order) {
        const auto rule = QuadratureFactory::create(ElementType::Triangle3, order);
        ASSERT_TRUE(rule);
        EXPECT_GE(rule->order(), order);
        std::array<Real, 3> integrals{};
        for (int shift = 0; shift < 3; ++shift) {
            const auto& v0 = vertices[static_cast<std::size_t>(shift)];
            const auto& v1 = vertices[static_cast<std::size_t>((shift + 1) % 3)];
            const auto& v2 = vertices[static_cast<std::size_t>((shift + 2) % 3)];
            for (std::size_t i = 0; i < rule->num_points(); ++i) {
                const auto& xi = rule->point(i);
                const Real l0 = Real(1) - xi[0] - xi[1];
                const Real x = l0 * v0[0] + xi[0] * v1[0] + xi[1] * v2[0];
                const Real y = l0 * v0[1] + xi[0] * v1[1] + xi[1] * v2[1];
                integrals[static_cast<std::size_t>(shift)] += rule->weight(i) * supg(x, y);
            }
        }
        const auto [low, high] = std::minmax_element(integrals.begin(), integrals.end());
        if (order < 20) {
            EXPECT_NEAR(*low, *high, 1e-14) << "order " << order;
        }
    }

    // On the unit tetrahedron the even vertex permutations move every point
    // of a symmetric rule onto another point of the same weight.
    const auto tet_integrand = [](Real x, Real y, Real z) {
        return std::sqrt(Real(1) + x * x + Real(2) * y * y + Real(3) * z * z) * (x - z);
    };
    const std::array<std::array<Real, 3>, 4> tet{{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    const std::array<std::array<int, 4>, 3> orderings{{{0, 1, 2, 3}, {1, 2, 0, 3}, {2, 3, 0, 1}}};
    for (int order = 1; order <= 6; ++order) {
        const auto rule = QuadratureFactory::create(ElementType::Tetra4, order);
        ASSERT_TRUE(rule);
        EXPECT_GE(rule->order(), order);
        std::vector<Real> integrals;
        for (const auto& ordering : orderings) {
            Real integral = 0;
            for (std::size_t i = 0; i < rule->num_points(); ++i) {
                const auto& xi = rule->point(i);
                const std::array<Real, 4> lambda{Real(1) - xi[0] - xi[1] - xi[2], xi[0], xi[1], xi[2]};
                std::array<Real, 3> x{};
                for (int a = 0; a < 4; ++a) {
                    for (int d = 0; d < 3; ++d) {
                        x[static_cast<std::size_t>(d)] +=
                            lambda[static_cast<std::size_t>(a)] *
                            tet[static_cast<std::size_t>(ordering[static_cast<std::size_t>(a)])]
                               [static_cast<std::size_t>(d)];
                    }
                }
                integral += rule->weight(i) * tet_integrand(x[0], x[1], x[2]);
            }
            integrals.push_back(integral);
        }
        const auto [low, high] = std::minmax_element(integrals.begin(), integrals.end());
        EXPECT_NEAR(*low, *high, 1e-14) << "order " << order;
    }
}
