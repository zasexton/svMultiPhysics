/**
 * @file test_VectorComponentExtractor.cpp
 * @brief Unit tests for VectorComponentExtractor
 */

#include <gtest/gtest.h>
#include "FE/Spaces/VectorComponentExtractor.h"
#include <cmath>

using namespace svmp::FE;
using namespace svmp::FE::spaces;
using Vec3 = VectorComponentExtractor::Vec3;
using Vec2 = VectorComponentExtractor::Vec2;

class VectorComponentExtractorTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

TEST_F(VectorComponentExtractorTest, NormalComponentSingleNormal) {
    std::vector<Vec3> vectors = {
        Vec3{1.0, 0.0, 0.0},
        Vec3{0.0, 1.0, 0.0},
        Vec3{1.0, 1.0, 1.0}
    };
    Vec3 normal{0.0, 0.0, 1.0};

    auto result = VectorComponentExtractor::normal_component(vectors, normal);

    ASSERT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 0.0, tol);
    EXPECT_NEAR(result[1], 0.0, tol);
    EXPECT_NEAR(result[2], 1.0, tol);
}

TEST_F(VectorComponentExtractorTest, NormalComponentVaryingNormals) {
    std::vector<Vec3> vectors = {
        Vec3{1.0, 0.0, 0.0},
        Vec3{0.0, 1.0, 0.0}
    };
    std::vector<Vec3> normals = {
        Vec3{1.0, 0.0, 0.0},
        Vec3{0.0, 1.0, 0.0}
    };

    auto result = VectorComponentExtractor::normal_component(vectors, normals);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 1.0, tol);  // v·n = 1·1 = 1
    EXPECT_NEAR(result[1], 1.0, tol);  // v·n = 1·1 = 1
}

TEST_F(VectorComponentExtractorTest, TangentialComponentOrthogonal) {
    std::vector<Vec3> vectors = {
        Vec3{1.0, 2.0, 3.0}
    };
    Vec3 normal{0.0, 0.0, 1.0};

    auto result = VectorComponentExtractor::tangential_component(vectors, normal);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0][0], 1.0, tol);
    EXPECT_NEAR(result[0][1], 2.0, tol);
    EXPECT_NEAR(result[0][2], 0.0, tol);  // z component removed
}

TEST_F(VectorComponentExtractorTest, DecompositionPythagorean) {
    Vec3 v{3.0, 4.0, 5.0};
    Vec3 n = Vec3{1.0, 1.0, 1.0}.normalized();

    bool valid = VectorComponentExtractor::verify_decomposition(v, n, 1e-10);
    EXPECT_TRUE(valid);
}

TEST_F(VectorComponentExtractorTest, TangentialComponent2D) {
    std::vector<Vec2> vectors = {
        Vec2{1.0, 1.0},
        Vec2{2.0, 0.0}
    };
    Vec2 normal{0.0, 1.0};  // Tangent is {-1, 0}

    auto result = VectorComponentExtractor::tangential_component_2d(vectors, normal);

    ASSERT_EQ(result.size(), 2u);
    // v·t = v_y * n_x - v_x * n_y = 1*0 - 1*1 = -1
    EXPECT_NEAR(result[0], -1.0, tol);
    // v·t = 0*0 - 2*1 = -2
    EXPECT_NEAR(result[1], -2.0, tol);
}

TEST_F(VectorComponentExtractorTest, ComputeTangentBasis) {
    Vec3 normal{0.0, 0.0, 1.0};
    Vec3 t1, t2;

    VectorComponentExtractor::compute_tangent_basis(normal, t1, t2);

    // Verify orthogonality
    EXPECT_NEAR(t1.dot(normal), 0.0, tol);
    EXPECT_NEAR(t2.dot(normal), 0.0, tol);
    EXPECT_NEAR(t1.dot(t2), 0.0, tol);

    // Verify unit length
    EXPECT_NEAR(t1.norm(), 1.0, tol);
    EXPECT_NEAR(t2.norm(), 1.0, tol);
}

TEST_F(VectorComponentExtractorTest, ComputeTangentBasisArbitraryNormal) {
    Vec3 normal = Vec3{1.0, 2.0, 3.0}.normalized();
    Vec3 t1, t2;

    VectorComponentExtractor::compute_tangent_basis(normal, t1, t2);

    EXPECT_NEAR(t1.dot(normal), 0.0, tol);
    EXPECT_NEAR(t2.dot(normal), 0.0, tol);
    EXPECT_NEAR(t1.dot(t2), 0.0, tol);
    EXPECT_NEAR(t1.norm(), 1.0, tol);
    EXPECT_NEAR(t2.norm(), 1.0, tol);
}

TEST_F(VectorComponentExtractorTest, ProjectToTangentPlane) {
    std::vector<Vec3> vectors = {
        Vec3{1.0, 2.0, 3.0}
    };
    Vec3 t1{1.0, 0.0, 0.0};
    Vec3 t2{0.0, 1.0, 0.0};

    auto result = VectorComponentExtractor::project_to_tangent_plane(vectors, t1, t2);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0][0], 1.0, tol);  // Component along t1
    EXPECT_NEAR(result[0][1], 2.0, tol);  // Component along t2
}

TEST_F(VectorComponentExtractorTest, HdivNormalTrace) {
    std::vector<Vec3> field = {
        Vec3{1.0, 0.0, 2.0},
        Vec3{0.0, 3.0, 1.0}
    };
    Vec3 normal{0.0, 0.0, 1.0};

    auto result = VectorComponentExtractor::hdiv_normal_trace(field, normal);

    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 2.0, tol);
    EXPECT_NEAR(result[1], 1.0, tol);
}

TEST_F(VectorComponentExtractorTest, HcurlTangentialTrace) {
    std::vector<Vec3> field = {
        Vec3{1.0, 2.0, 3.0}
    };
    Vec3 normal{0.0, 0.0, 1.0};

    auto result = VectorComponentExtractor::hcurl_tangential_trace(field, normal);

    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0][0], 1.0, tol);
    EXPECT_NEAR(result[0][1], 2.0, tol);
    EXPECT_NEAR(result[0][2], 0.0, tol);
}

TEST_F(VectorComponentExtractorTest, ComputeTangent2D) {
    Vec2 normal{1.0, 0.0};
    Vec2 tangent = VectorComponentExtractor::compute_tangent_2d(normal);

    // Tangent should be rotated 90 degrees CCW
    EXPECT_NEAR(tangent[0], 0.0, tol);
    EXPECT_NEAR(tangent[1], 1.0, tol);

    // Check orthogonality
    EXPECT_NEAR(normal[0] * tangent[0] + normal[1] * tangent[1], 0.0, tol);
}
