/**
 * @file test_PositionBasedQuadrature.cpp
 * @brief Unit tests for position-based reduced integration quadrature
 *
 * Tests the position-based quadrature rules for triangles and tetrahedra,
 * including legacy compatibility with qmTRI3 and qmTET4 parameters.
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/PositionBasedParams.h"
#include "FE/Quadrature/PositionBasedTriangleQuadrature.h"
#include "FE/Quadrature/PositionBasedTetrahedronQuadrature.h"
#include "FE/Quadrature/QuadratureFactory.h"
#include "FE/Quadrature/QuadratureCache.h"
#include <cmath>
#include <numeric>

using namespace svmp::FE::quadrature;
using namespace svmp::FE;

// =============================================================================
// PositionBasedParams Tests
// =============================================================================

TEST(PositionBasedParams, DefaultConstructor) {
    PositionBasedParams params;
    EXPECT_DOUBLE_EQ(params.modifier, 0.0);
}

TEST(PositionBasedParams, ExplicitConstructor) {
    PositionBasedParams params{0.5};
    EXPECT_DOUBLE_EQ(params.modifier, 0.5);
}

TEST(PositionBasedParams, TET4CentralIsValid) {
    auto params = PositionBasedParams::tet4_central();
    EXPECT_DOUBLE_EQ(params.modifier, 0.25);
    EXPECT_TRUE(params.is_valid_for_tet4());
}

TEST(PositionBasedParams, TET4GaussianIsValid) {
    auto params = PositionBasedParams::tet4_gaussian();
    const double expected = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    EXPECT_NEAR(params.modifier, expected, 1e-14);
    EXPECT_TRUE(params.is_valid_for_tet4());
}

TEST(PositionBasedParams, TET4NodalIsValid) {
    auto params = PositionBasedParams::tet4_nodal();
    EXPECT_DOUBLE_EQ(params.modifier, 1.0);
    EXPECT_TRUE(params.is_valid_for_tet4());
}

TEST(PositionBasedParams, TRI3CentralIsValid) {
    auto params = PositionBasedParams::tri3_central();
    EXPECT_NEAR(params.modifier, 1.0 / 3.0, 1e-14);
    EXPECT_TRUE(params.is_valid_for_tri3());
}

TEST(PositionBasedParams, TRI3GaussianIsValid) {
    auto params = PositionBasedParams::tri3_gaussian();
    EXPECT_NEAR(params.modifier, 2.0 / 3.0, 1e-14);
    EXPECT_TRUE(params.is_valid_for_tri3());
}

TEST(PositionBasedParams, TRI3NodalIsValid) {
    auto params = PositionBasedParams::tri3_nodal();
    EXPECT_DOUBLE_EQ(params.modifier, 1.0);
    EXPECT_TRUE(params.is_valid_for_tri3());
}

TEST(PositionBasedParams, InvalidTET4Range) {
    PositionBasedParams params{0.2};  // Below minimum 0.25
    EXPECT_FALSE(params.is_valid_for_tet4());

    PositionBasedParams params2{1.1};  // Above maximum 1.0
    EXPECT_FALSE(params2.is_valid_for_tet4());
}

TEST(PositionBasedParams, InvalidTRI3Range) {
    PositionBasedParams params{0.2};  // Below minimum 1/3
    EXPECT_FALSE(params.is_valid_for_tri3());

    PositionBasedParams params2{1.1};  // Above maximum 1.0
    EXPECT_FALSE(params2.is_valid_for_tri3());
}

TEST(PositionBasedParams, LegacyCompatibility) {
    // Test that from_qmTET4 and from_qmTRI3 work correctly
    const double qmTET4_default = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    const double qmTRI3_default = 2.0 / 3.0;

    auto tet4_params = PositionBasedParams::from_qmTET4(qmTET4_default);
    auto tri3_params = PositionBasedParams::from_qmTRI3(qmTRI3_default);

    EXPECT_NEAR(tet4_params.modifier, qmTET4_default, 1e-14);
    EXPECT_NEAR(tri3_params.modifier, qmTRI3_default, 1e-14);
}

// =============================================================================
// PositionBasedTriangleQuadrature Tests
// =============================================================================

TEST(PositionBasedTriangleQuadrature, GaussianWeightSum) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_gaussian());

    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 0.5, 1e-12);  // Reference triangle area
}

TEST(PositionBasedTriangleQuadrature, CentralWeightSum) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_central());

    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST(PositionBasedTriangleQuadrature, NodalWeightSum) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_nodal());

    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 0.5, 1e-12);
}

TEST(PositionBasedTriangleQuadrature, NumPointsIsThree) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_gaussian());
    EXPECT_EQ(quad.num_points(), 3u);
}

TEST(PositionBasedTriangleQuadrature, DimensionIsTwo) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_gaussian());
    EXPECT_EQ(quad.dimension(), 2);
}

TEST(PositionBasedTriangleQuadrature, GaussianMatchesLegacy) {
    // Verify that Gaussian position matches legacy qmTRI3 = 2/3 implementation
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_gaussian());

    const double s = 2.0 / 3.0;
    const double t = 1.0 / 6.0;

    // Expected points from legacy implementation
    std::vector<std::pair<double, double>> expected = {
        {t, t},
        {s, t},
        {t, s}
    };

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(quad.point(i)[0], expected[i].first, 1e-12);
        EXPECT_NEAR(quad.point(i)[1], expected[i].second, 1e-12);
    }
}

TEST(PositionBasedTriangleQuadrature, CentralPointsAtCentroid) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_central());

    // Central rule: s = 1/3, t = 1/3, all points at centroid
    const double centroid = 1.0 / 3.0;

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(quad.point(i)[0], centroid, 1e-12);
        EXPECT_NEAR(quad.point(i)[1], centroid, 1e-12);
    }
}

TEST(PositionBasedTriangleQuadrature, NodalPointsAtVertices) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_nodal());

    // Nodal rule: s = 1.0, t = 0.0, points at vertices
    std::vector<std::pair<double, double>> vertices = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    };

    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(quad.point(i)[0], vertices[i].first, 1e-12);
        EXPECT_NEAR(quad.point(i)[1], vertices[i].second, 1e-12);
    }
}

TEST(PositionBasedTriangleQuadrature, GaussianIntegratesConstant) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_gaussian());

    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        val += quad.weight(i) * 1.0;  // Integrate constant 1
    }
    EXPECT_NEAR(val, 0.5, 1e-12);  // Area of reference triangle
}

TEST(PositionBasedTriangleQuadrature, GaussianIntegratesLinear) {
    PositionBasedTriangleQuadrature quad(PositionBasedParams::tri3_gaussian());

    // Integrate x + y over reference triangle
    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        val += quad.weight(i) * (p[0] + p[1]);
    }
    // Exact: ∫∫(x+y) dA = 1/3 for reference triangle
    EXPECT_NEAR(val, 1.0 / 3.0, 1e-12);
}

TEST(PositionBasedTriangleQuadrature, IsGaussianMethod) {
    PositionBasedTriangleQuadrature gaussian(PositionBasedParams::tri3_gaussian());
    PositionBasedTriangleQuadrature central(PositionBasedParams::tri3_central());
    PositionBasedTriangleQuadrature nodal(PositionBasedParams::tri3_nodal());

    EXPECT_TRUE(gaussian.is_gaussian());
    EXPECT_FALSE(central.is_gaussian());
    EXPECT_FALSE(nodal.is_gaussian());
}

TEST(PositionBasedTriangleQuadrature, IsCentralMethod) {
    PositionBasedTriangleQuadrature gaussian(PositionBasedParams::tri3_gaussian());
    PositionBasedTriangleQuadrature central(PositionBasedParams::tri3_central());

    EXPECT_FALSE(gaussian.is_central());
    EXPECT_TRUE(central.is_central());
}

TEST(PositionBasedTriangleQuadrature, IsNodalMethod) {
    PositionBasedTriangleQuadrature gaussian(PositionBasedParams::tri3_gaussian());
    PositionBasedTriangleQuadrature nodal(PositionBasedParams::tri3_nodal());

    EXPECT_FALSE(gaussian.is_nodal());
    EXPECT_TRUE(nodal.is_nodal());
}

TEST(PositionBasedTriangleQuadrature, InvalidModifierThrows) {
    EXPECT_THROW(
        PositionBasedTriangleQuadrature(PositionBasedParams{0.2}),
        svmp::FE::FEException);

    EXPECT_THROW(
        PositionBasedTriangleQuadrature(PositionBasedParams{1.1}),
        svmp::FE::FEException);
}

// =============================================================================
// PositionBasedTetrahedronQuadrature Tests
// =============================================================================

TEST(PositionBasedTetrahedronQuadrature, GaussianWeightSum) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_gaussian());

    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-12);  // Reference tet volume
}

TEST(PositionBasedTetrahedronQuadrature, CentralWeightSum) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_central());

    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-12);
}

TEST(PositionBasedTetrahedronQuadrature, NodalWeightSum) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_nodal());

    double sum = std::accumulate(quad.weights().begin(), quad.weights().end(), 0.0);
    EXPECT_NEAR(sum, 1.0 / 6.0, 1e-12);
}

TEST(PositionBasedTetrahedronQuadrature, NumPointsIsFour) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_gaussian());
    EXPECT_EQ(quad.num_points(), 4u);
}

TEST(PositionBasedTetrahedronQuadrature, DimensionIsThree) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_gaussian());
    EXPECT_EQ(quad.dimension(), 3);
}

TEST(PositionBasedTetrahedronQuadrature, GaussianMatchesLegacy) {
    // Verify that Gaussian position matches legacy qmTET4 implementation
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_gaussian());

    const double s = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    const double t = (5.0 - std::sqrt(5.0)) / 20.0;  // = (1-s)/3

    // Expected points from legacy implementation
    std::vector<std::array<double, 3>> expected = {
        {s, t, t},
        {t, s, t},
        {t, t, s},
        {t, t, t}
    };

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(quad.point(i)[0], expected[i][0], 1e-12);
        EXPECT_NEAR(quad.point(i)[1], expected[i][1], 1e-12);
        EXPECT_NEAR(quad.point(i)[2], expected[i][2], 1e-12);
    }
}

TEST(PositionBasedTetrahedronQuadrature, CentralPointsAtCentroid) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_central());

    // Central rule: s = 0.25, t = 0.25, all points at centroid
    const double centroid = 0.25;

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(quad.point(i)[0], centroid, 1e-12);
        EXPECT_NEAR(quad.point(i)[1], centroid, 1e-12);
        EXPECT_NEAR(quad.point(i)[2], centroid, 1e-12);
    }
}

TEST(PositionBasedTetrahedronQuadrature, NodalPointsAtVertices) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_nodal());

    // Nodal rule: s = 1.0, t = 0.0, points at vertices
    std::vector<std::array<double, 3>> vertices = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 0.0}
    };

    for (std::size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(quad.point(i)[0], vertices[i][0], 1e-12);
        EXPECT_NEAR(quad.point(i)[1], vertices[i][1], 1e-12);
        EXPECT_NEAR(quad.point(i)[2], vertices[i][2], 1e-12);
    }
}

TEST(PositionBasedTetrahedronQuadrature, GaussianIntegratesConstant) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_gaussian());

    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        val += quad.weight(i) * 1.0;  // Integrate constant 1
    }
    EXPECT_NEAR(val, 1.0 / 6.0, 1e-12);  // Volume of reference tet
}

TEST(PositionBasedTetrahedronQuadrature, GaussianIntegratesLinear) {
    PositionBasedTetrahedronQuadrature quad(PositionBasedParams::tet4_gaussian());

    // Integrate x + y + z over reference tetrahedron
    double val = 0.0;
    for (std::size_t i = 0; i < quad.num_points(); ++i) {
        const auto& p = quad.point(i);
        val += quad.weight(i) * (p[0] + p[1] + p[2]);
    }
    // Exact: ∫∫∫(x+y+z) dV = 1/8 for reference tetrahedron
    EXPECT_NEAR(val, 1.0 / 8.0, 1e-12);
}

TEST(PositionBasedTetrahedronQuadrature, InvalidModifierThrows) {
    EXPECT_THROW(
        PositionBasedTetrahedronQuadrature(PositionBasedParams{0.2}),
        svmp::FE::FEException);

    EXPECT_THROW(
        PositionBasedTetrahedronQuadrature(PositionBasedParams{1.1}),
        svmp::FE::FEException);
}

// =============================================================================
// QuadratureFactory Position-Based Tests
// =============================================================================

TEST(QuadratureFactoryPositionBased, CreatePositionBasedTriangle) {
    auto rule = QuadratureFactory::create_position_based(
        ElementType::Triangle3,
        PositionBasedParams::tri3_gaussian(),
        false);  // No cache

    EXPECT_EQ(rule->num_points(), 3u);
    EXPECT_EQ(rule->dimension(), 2);
}

TEST(QuadratureFactoryPositionBased, CreatePositionBasedTetra) {
    auto rule = QuadratureFactory::create_position_based(
        ElementType::Tetra4,
        PositionBasedParams::tet4_gaussian(),
        false);  // No cache

    EXPECT_EQ(rule->num_points(), 4u);
    EXPECT_EQ(rule->dimension(), 3);
}

TEST(QuadratureFactoryPositionBased, CreateLegacyCompatibleTriangle) {
    const double qmTRI3 = 2.0 / 3.0;
    auto rule = QuadratureFactory::create_legacy_compatible(
        ElementType::Triangle3, qmTRI3, false);

    EXPECT_EQ(rule->num_points(), 3u);
}

TEST(QuadratureFactoryPositionBased, CreateLegacyCompatibleTetra) {
    const double qmTET4 = (5.0 + 3.0 * std::sqrt(5.0)) / 20.0;
    auto rule = QuadratureFactory::create_legacy_compatible(
        ElementType::Tetra4, qmTET4, false);

    EXPECT_EQ(rule->num_points(), 4u);
}

TEST(QuadratureFactoryPositionBased, CreateCentral) {
    auto tri_rule = QuadratureFactory::create_central(ElementType::Triangle3, false);
    auto tet_rule = QuadratureFactory::create_central(ElementType::Tetra4, false);

    EXPECT_EQ(tri_rule->num_points(), 3u);
    EXPECT_EQ(tet_rule->num_points(), 4u);
}

TEST(QuadratureFactoryPositionBased, CreateNodal) {
    auto tri_rule = QuadratureFactory::create_nodal(ElementType::Triangle3, false);
    auto tet_rule = QuadratureFactory::create_nodal(ElementType::Tetra4, false);

    EXPECT_EQ(tri_rule->num_points(), 3u);
    EXPECT_EQ(tet_rule->num_points(), 4u);
}

TEST(QuadratureFactoryPositionBased, SupportsPositionBased) {
    EXPECT_TRUE(QuadratureFactory::supports_position_based(ElementType::Triangle3));
    EXPECT_TRUE(QuadratureFactory::supports_position_based(ElementType::Triangle6));
    EXPECT_TRUE(QuadratureFactory::supports_position_based(ElementType::Tetra4));
    EXPECT_TRUE(QuadratureFactory::supports_position_based(ElementType::Tetra10));

    EXPECT_FALSE(QuadratureFactory::supports_position_based(ElementType::Quad4));
    EXPECT_FALSE(QuadratureFactory::supports_position_based(ElementType::Hex8));
    EXPECT_FALSE(QuadratureFactory::supports_position_based(ElementType::Line2));
}

TEST(QuadratureFactoryPositionBased, DefaultLegacyModifier) {
    const double tri_default = QuadratureFactory::default_legacy_modifier(ElementType::Triangle3);
    const double tet_default = QuadratureFactory::default_legacy_modifier(ElementType::Tetra4);

    EXPECT_NEAR(tri_default, 2.0 / 3.0, 1e-14);
    EXPECT_NEAR(tet_default, (5.0 + 3.0 * std::sqrt(5.0)) / 20.0, 1e-14);
}

TEST(QuadratureFactoryPositionBased, UnsupportedElementThrows) {
    EXPECT_THROW(
        QuadratureFactory::create_position_based(
            ElementType::Quad4,
            PositionBasedParams{0.5},
            false),
        svmp::FE::FEException);

    EXPECT_THROW(
        QuadratureFactory::create_central(ElementType::Hex8, false),
        svmp::FE::FEException);
}

// =============================================================================
// Caching Tests
// =============================================================================

TEST(QuadratureFactoryPositionBased, CachingWorks) {
    QuadratureCache::instance().clear();

    auto rule1 = QuadratureFactory::create_position_based(
        ElementType::Tetra4,
        PositionBasedParams::tet4_gaussian(),
        true);  // Use cache

    auto rule2 = QuadratureFactory::create_position_based(
        ElementType::Tetra4,
        PositionBasedParams::tet4_gaussian(),
        true);  // Use cache

    // Should be the same pointer (cached)
    EXPECT_EQ(rule1.get(), rule2.get());
}

TEST(QuadratureFactoryPositionBased, DifferentModifiersNotCached) {
    QuadratureCache::instance().clear();

    auto rule1 = QuadratureFactory::create_position_based(
        ElementType::Tetra4,
        PositionBasedParams::tet4_gaussian(),
        true);

    auto rule2 = QuadratureFactory::create_position_based(
        ElementType::Tetra4,
        PositionBasedParams::tet4_central(),
        true);

    // Different modifiers should be different pointers
    EXPECT_NE(rule1.get(), rule2.get());
}

TEST(QuadratureFactoryPositionBased, HigherOrderElementsMapped) {
    // Triangle6 should map to Triangle3 for position-based quadrature
    auto rule = QuadratureFactory::create_position_based(
        ElementType::Triangle6,  // Higher-order
        PositionBasedParams::tri3_gaussian(),
        false);

    EXPECT_EQ(rule->num_points(), 3u);
    EXPECT_EQ(rule->cell_family(), svmp::CellFamily::Triangle);
}

// =============================================================================
// Validation Tests
// =============================================================================

TEST(PositionBasedQuadrature, RuleIsValid) {
    PositionBasedTriangleQuadrature tri(PositionBasedParams::tri3_gaussian());
    PositionBasedTetrahedronQuadrature tet(PositionBasedParams::tet4_gaussian());

    EXPECT_TRUE(tri.is_valid());
    EXPECT_TRUE(tet.is_valid());
}

TEST(PositionBasedQuadrature, AllPositionsAreValid) {
    // Test a range of valid position modifiers
    std::vector<double> tri_modifiers = {1.0/3.0, 0.4, 0.5, 2.0/3.0, 0.8, 1.0};
    std::vector<double> tet_modifiers = {0.25, 0.3, 0.4, 0.585, 0.7, 1.0};

    for (double mod : tri_modifiers) {
        PositionBasedTriangleQuadrature quad(PositionBasedParams{mod});
        EXPECT_TRUE(quad.is_valid()) << "Failed for TRI3 modifier " << mod;
    }

    for (double mod : tet_modifiers) {
        PositionBasedTetrahedronQuadrature quad(PositionBasedParams{mod});
        EXPECT_TRUE(quad.is_valid()) << "Failed for TET4 modifier " << mod;
    }
}
