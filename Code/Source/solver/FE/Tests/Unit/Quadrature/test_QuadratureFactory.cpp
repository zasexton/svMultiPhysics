/**
 * @file test_QuadratureFactory.cpp
 * @brief Unit tests for quadrature factory and cache
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Core/FEException.h"
#include "FE/Core/Types.h"
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
        {ElementType::Tetra4, 6},
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

TEST(QuadratureFactory, CreateAcceptsPositionBasedType) {
    auto tri_pos = QuadratureFactory::create(ElementType::Triangle3, 2, QuadratureType::PositionBased, false);
    auto tri_gauss = QuadratureFactory::create(ElementType::Triangle3, 2, QuadratureType::GaussLegendre, false);
    ASSERT_TRUE(tri_pos);
    ASSERT_TRUE(tri_gauss);
    EXPECT_EQ(tri_pos->cell_family(), svmp::CellFamily::Triangle);
    EXPECT_EQ(tri_pos->dimension(), 2);
    EXPECT_EQ(tri_pos->num_points(), 3u);
    EXPECT_NE(tri_pos->num_points(), tri_gauss->num_points());

    auto tet_pos = QuadratureFactory::create(ElementType::Tetra4, 2, QuadratureType::PositionBased, false);
    auto tet_gauss = QuadratureFactory::create(ElementType::Tetra4, 2, QuadratureType::GaussLegendre, false);
    ASSERT_TRUE(tet_pos);
    ASSERT_TRUE(tet_gauss);
    EXPECT_EQ(tet_pos->cell_family(), svmp::CellFamily::Tetra);
    EXPECT_EQ(tet_pos->dimension(), 3);
    EXPECT_EQ(tet_pos->num_points(), 4u);
    EXPECT_NE(tet_pos->num_points(), tet_gauss->num_points());
}

TEST(QuadratureFactory, InvalidInputsThrow) {
    EXPECT_THROW(QuadratureFactory::create(ElementType::Hex8, 0), FEException);
    EXPECT_THROW(QuadratureFactory::create(ElementType::Unknown, 2), FEException);
}
