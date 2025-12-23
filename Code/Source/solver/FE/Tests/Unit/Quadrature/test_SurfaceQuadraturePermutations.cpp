/**
 * @file test_SurfaceQuadraturePermutations.cpp
 * @brief Surface and edge quadrature permutations across element types
 */

#include <gtest/gtest.h>
#include "FE/Quadrature/SurfaceQuadrature.h"

using namespace svmp::FE;
using namespace svmp::FE::quadrature;

namespace {
double sum_weights(const QuadratureRule& q) {
    double sum = 0.0;
    for (std::size_t i = 0; i < q.num_points(); ++i) sum += q.weight(i);
    return sum;
}
}

TEST(SurfaceQuadraturePermutations, TetraFacesAllHaveTriangleMeasure) {
    for (int f = 0; f < 4; ++f) {
        auto rule = SurfaceQuadrature::face_rule(ElementType::Tetra4, f, 3);
        ASSERT_TRUE(rule);
        EXPECT_NEAR(sum_weights(*rule), 0.5, 1e-12);
    }
}

TEST(SurfaceQuadraturePermutations, HexFacesAllHaveQuadMeasure) {
    for (int f = 0; f < 6; ++f) {
        auto rule = SurfaceQuadrature::face_rule(ElementType::Hex8, f, 3);
        ASSERT_TRUE(rule);
        EXPECT_NEAR(sum_weights(*rule), 4.0, 1e-12);
    }
}

TEST(SurfaceQuadraturePermutations, WedgeFacesTriangleAndQuad) {
    // Faces 0-1 triangle, 2-4 quad per implementation
    for (int f = 0; f < 5; ++f) {
        auto rule = SurfaceQuadrature::face_rule(ElementType::Wedge6, f, 3);
        ASSERT_TRUE(rule);
        if (f < 2) {
            EXPECT_NEAR(sum_weights(*rule), 0.5, 1e-12);
        } else {
            EXPECT_NEAR(sum_weights(*rule), 4.0, 1e-12);
        }
    }
}

TEST(SurfaceQuadraturePermutations, PyramidFacesTriangleAndQuad) {
    for (int f = 0; f < 5; ++f) {
        auto rule = SurfaceQuadrature::face_rule(ElementType::Pyramid5, f, 3);
        ASSERT_TRUE(rule);
        if (f == 0) {
            EXPECT_NEAR(sum_weights(*rule), 4.0, 1e-12);
        } else {
            EXPECT_NEAR(sum_weights(*rule), 0.5, 1e-12);
        }
    }
}

TEST(SurfaceQuadraturePermutations, InvalidFaceIdThrows) {
    EXPECT_THROW(SurfaceQuadrature::face_rule(ElementType::Hex8, 6, 2), FEException);
    EXPECT_THROW(SurfaceQuadrature::face_rule(ElementType::Tetra4, 5, 2), FEException);
}

TEST(SurfaceQuadraturePermutations, EdgeRuleLineFamily) {
    for (int e = 0; e < 3; ++e) {
        auto line_rule = SurfaceQuadrature::edge_rule(svmp::CellFamily::Line, e, 3);
        ASSERT_TRUE(line_rule);
        EXPECT_NEAR(sum_weights(*line_rule), 2.0, 1e-12);
    }
}

TEST(SurfaceQuadraturePermutations, EdgeRuleFromTriangleFace) {
    auto tri_face = SurfaceQuadrature::face_rule(ElementType::Tetra4, 0, 3);
    ASSERT_TRUE(tri_face);
    auto edge_rule = SurfaceQuadrature::edge_rule(svmp::CellFamily::Triangle, 0, 3);
    ASSERT_TRUE(edge_rule);
    EXPECT_NEAR(sum_weights(*edge_rule), 2.0, 1e-12);
}

TEST(SurfaceQuadraturePermutations, ReducedTriangleFaceUsesFewerPoints) {
    auto full = SurfaceQuadrature::face_rule(ElementType::Tetra4, 0, 5, QuadratureType::GaussLegendre);
    auto reduced = SurfaceQuadrature::face_rule(ElementType::Tetra4, 0, 5, QuadratureType::Reduced);
    ASSERT_TRUE(full);
    ASSERT_TRUE(reduced);

    EXPECT_LT(reduced->num_points(), full->num_points());
    EXPECT_NEAR(sum_weights(*reduced), 0.5, 1e-12);
}

TEST(SurfaceQuadraturePermutations, ReducedEdgeRuleUsesFewerPoints) {
    auto full = SurfaceQuadrature::edge_rule(svmp::CellFamily::Triangle, 0, 4, QuadratureType::GaussLegendre);
    auto reduced = SurfaceQuadrature::edge_rule(svmp::CellFamily::Triangle, 0, 4, QuadratureType::Reduced);
    ASSERT_TRUE(full);
    ASSERT_TRUE(reduced);

    EXPECT_LT(reduced->num_points(), full->num_points());
    EXPECT_NEAR(sum_weights(*reduced), 2.0, 1e-12);
}

TEST(SurfaceQuadraturePermutations, ReducedLineFamilyEdgeRuleUsesFewerPoints) {
    auto full = SurfaceQuadrature::edge_rule(svmp::CellFamily::Line, 0, 4, QuadratureType::GaussLegendre);
    auto reduced = SurfaceQuadrature::edge_rule(svmp::CellFamily::Line, 0, 4, QuadratureType::Reduced);
    ASSERT_TRUE(full);
    ASSERT_TRUE(reduced);

    EXPECT_LT(reduced->num_points(), full->num_points());
    EXPECT_NEAR(sum_weights(*reduced), 2.0, 1e-12);
}
