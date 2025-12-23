/**
 * @file test_OrientationManager.cpp
 * @brief Unit tests for OrientationManager
 */

#include <gtest/gtest.h>
#include "FE/Spaces/OrientationManager.h"
#include <algorithm>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::spaces;

class OrientationManagerTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

// =============================================================================
// Edge Orientation Tests
// =============================================================================

TEST_F(OrientationManagerTest, EdgeOrientationAligned) {
    // Local and reference/global direction use the same endpoint ordering.
    auto sign = OrientationManager::edge_orientation(5, 10, 5, 10);
    EXPECT_EQ(sign, 1);
}

TEST_F(OrientationManagerTest, EdgeOrientationOpposite) {
    // Local edge direction is the reverse of the reference direction.
    auto sign = OrientationManager::edge_orientation(10, 5, 5, 10);
    EXPECT_EQ(sign, -1);
}

TEST_F(OrientationManagerTest, EdgeOrientationSimple) {
    EXPECT_EQ(OrientationManager::edge_orientation(1, 5), 1);
    EXPECT_EQ(OrientationManager::edge_orientation(5, 1), -1);
    EXPECT_EQ(OrientationManager::edge_orientation(100, 200), 1);
}

TEST_F(OrientationManagerTest, OrientEdgeDofsIdentity) {
    std::vector<Real> edge_dofs = {1.0, 2.0, 3.0};  // 2 vertex + 1 interior

    auto result = OrientationManager::orient_edge_dofs(edge_dofs, 1, true);

    EXPECT_EQ(result.size(), 3u);
    EXPECT_NEAR(result[0], 1.0, tol);
    EXPECT_NEAR(result[1], 2.0, tol);
    EXPECT_NEAR(result[2], 3.0, tol);
}

TEST_F(OrientationManagerTest, OrientEdgeDofsFlipped) {
    std::vector<Real> edge_dofs = {1.0, 2.0};  // Just vertex DOFs

    auto result = OrientationManager::orient_edge_dofs(edge_dofs, -1, true);

    EXPECT_EQ(result.size(), 2u);
    // Vertex DOFs should be swapped
    EXPECT_NEAR(result[0], 2.0, tol);
    EXPECT_NEAR(result[1], 1.0, tol);
}

TEST_F(OrientationManagerTest, OrientHCurlEdgeDofsReversesAndFlipsSign) {
    std::vector<Real> edge_dofs = {1.0, 2.0, 3.0, 4.0};
    auto result = OrientationManager::orient_hcurl_edge_dofs(edge_dofs, -1);
    ASSERT_EQ(result.size(), edge_dofs.size());
    EXPECT_NEAR(result[0], -4.0, tol);
    EXPECT_NEAR(result[1], -3.0, tol);
    EXPECT_NEAR(result[2], -2.0, tol);
    EXPECT_NEAR(result[3], -1.0, tol);
}

// =============================================================================
// Triangle Face Orientation Tests
// =============================================================================

TEST_F(OrientationManagerTest, TriangleFaceOrientationIdentity) {
    // Local and global vertices have same ordering
    std::array<int, 3> local = {10, 20, 30};
    std::array<int, 3> global = {10, 20, 30};

    auto orient = OrientationManager::triangle_face_orientation(local, global);

    EXPECT_EQ(orient.rotation, 0);
    EXPECT_FALSE(orient.reflection);
    EXPECT_EQ(orient.sign, 1);
}

TEST_F(OrientationManagerTest, TriangleFaceOrientationRotated) {
    std::array<int, 3> local = {20, 30, 10};
    std::array<int, 3> global = {10, 20, 30};

    auto orient = OrientationManager::triangle_face_orientation(local, global);

    EXPECT_EQ(orient.rotation, 2);
    EXPECT_FALSE(orient.reflection);
    EXPECT_EQ(orient.sign, 1);
}

TEST_F(OrientationManagerTest, TriangleFaceOrientationReflected) {
    // Vertices in opposite winding order
    std::array<int, 3> local = {10, 30, 20};
    std::array<int, 3> global = {10, 20, 30};

    auto orient = OrientationManager::triangle_face_orientation(local, global);

    // Should detect reflection
    EXPECT_TRUE(orient.reflection);
    EXPECT_EQ(orient.sign, -1);
}

// =============================================================================
// Quad Face Orientation Tests
// =============================================================================

TEST_F(OrientationManagerTest, QuadFaceOrientationIdentity) {
    std::array<int, 4> local = {10, 20, 30, 40};
    std::array<int, 4> global = {10, 20, 30, 40};

    auto orient = OrientationManager::quad_face_orientation(local, global);

    EXPECT_EQ(orient.rotation, 0);
    EXPECT_FALSE(orient.reflection);
    EXPECT_EQ(orient.sign, 1);
}

TEST_F(OrientationManagerTest, QuadFaceOrientationRotated90) {
    std::array<int, 4> local = {20, 30, 40, 10};  // Rotated
    std::array<int, 4> global = {10, 20, 30, 40};

    auto orient = OrientationManager::quad_face_orientation(local, global);

    EXPECT_EQ(orient.rotation, 3);  // Rotate back to match canonical
}

// =============================================================================
// Permutation Utilities
// =============================================================================

TEST_F(OrientationManagerTest, PermutationSignEven) {
    std::vector<int> identity = {0, 1, 2, 3};
    EXPECT_EQ(OrientationManager::permutation_sign(identity), 1);

    std::vector<int> two_swaps = {1, 0, 3, 2};  // Two transpositions = even
    EXPECT_EQ(OrientationManager::permutation_sign(two_swaps), 1);
}

TEST_F(OrientationManagerTest, PermutationSignOdd) {
    std::vector<int> one_swap = {1, 0, 2, 3};  // One transposition = odd
    EXPECT_EQ(OrientationManager::permutation_sign(one_swap), -1);

    std::vector<int> three_cycle = {1, 2, 0, 3};  // Cycle (0 1 2) = 2 swaps = even
    // Actually this is 2 transpositions, so even
    EXPECT_EQ(OrientationManager::permutation_sign(three_cycle), 1);
}

TEST_F(OrientationManagerTest, ApplyPermutation) {
    std::vector<Real> values = {10.0, 20.0, 30.0, 40.0};
    std::vector<int> perm = {2, 0, 3, 1};  // new[i] = old[perm[i]]

    auto result = OrientationManager::apply_permutation(values, perm);

    EXPECT_EQ(result.size(), 4u);
    EXPECT_NEAR(result[0], 30.0, tol);  // values[2]
    EXPECT_NEAR(result[1], 10.0, tol);  // values[0]
    EXPECT_NEAR(result[2], 40.0, tol);  // values[3]
    EXPECT_NEAR(result[3], 20.0, tol);  // values[1]
}

TEST_F(OrientationManagerTest, ComposePermutations) {
    std::vector<int> p1 = {1, 2, 0};  // Cycle left
    std::vector<int> p2 = {2, 0, 1};  // Cycle right

    auto composed = OrientationManager::compose_permutations(p1, p2);

    EXPECT_EQ(composed.size(), 3u);
    // p1 ∘ p2 [i] = p1[p2[i]]
    EXPECT_EQ(composed[0], p1[p2[0]]);
    EXPECT_EQ(composed[1], p1[p2[1]]);
    EXPECT_EQ(composed[2], p1[p2[2]]);
}

TEST_F(OrientationManagerTest, InvertPermutation) {
    std::vector<int> perm = {2, 0, 1};

    auto inv = OrientationManager::invert_permutation(perm);

    // Applying perm then inv should give identity
    auto composed = OrientationManager::compose_permutations(inv, perm);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(composed[i], i);
    }
}

TEST_F(OrientationManagerTest, CanonicalOrderingTriangle) {
    std::array<int, 3> vertices = {30, 10, 20};

    auto order = OrientationManager::canonical_ordering(vertices);

    // Should sort by global index: 10, 20, 30
    EXPECT_EQ(vertices[order[0]], 10);
    EXPECT_EQ(vertices[order[1]], 20);
    EXPECT_EQ(vertices[order[2]], 30);
}

TEST_F(OrientationManagerTest, CanonicalOrderingQuad) {
    std::array<int, 4> vertices = {40, 10, 30, 20};

    auto order = OrientationManager::canonical_ordering(vertices);

    EXPECT_EQ(vertices[order[0]], 10);
    EXPECT_EQ(vertices[order[1]], 20);
    EXPECT_EQ(vertices[order[2]], 30);
    EXPECT_EQ(vertices[order[3]], 40);
}

// =============================================================================
// Face DOF Orientation Tests
// =============================================================================

TEST_F(OrientationManagerTest, OrientTriangleFaceDofsIdentity) {
    std::vector<Real> face_dofs = {1.0, 2.0, 3.0};
    OrientationManager::FaceOrientation orient;
    orient.rotation = 0;
    orient.reflection = false;
    orient.sign = 1;

    auto result = OrientationManager::orient_triangle_face_dofs(face_dofs, orient, 4);

    EXPECT_EQ(result.size(), face_dofs.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        EXPECT_NEAR(result[i], face_dofs[i], tol);
    }
}

TEST_F(OrientationManagerTest, OrientQuadFaceDofsIdentity) {
    std::vector<Real> face_dofs = {1.0, 2.0, 3.0, 4.0};
    OrientationManager::FaceOrientation orient;
    orient.rotation = 0;
    orient.reflection = false;
    orient.sign = 1;

    auto result = OrientationManager::orient_quad_face_dofs(face_dofs, orient, 3);

    EXPECT_EQ(result.size(), face_dofs.size());
}

TEST_F(OrientationManagerTest, IsEvenPermutation) {
    EXPECT_TRUE(OrientationManager::is_even_permutation({0, 1, 2}));  // Identity
    EXPECT_FALSE(OrientationManager::is_even_permutation({1, 0, 2})); // One swap
    EXPECT_TRUE(OrientationManager::is_even_permutation({1, 2, 0}));  // Cycle = 2 swaps
}

TEST_F(OrientationManagerTest, OrientHCurlTriangleFaceDofsIdentityAndInverse) {
    constexpr int k = 2;
    const std::size_t block = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1) / 2u;
    const std::size_t ndofs = 2u * block;
    std::vector<Real> face_dofs(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        face_dofs[i] = Real(0.05) * Real(i + 1);
    }

    const std::array<int, 3> global = {10, 20, 30};
    const std::array<int, 3> local = global;
    const auto orient = OrientationManager::triangle_face_orientation(local, global);
    const auto inv = OrientationManager::triangle_face_orientation(global, local);

    const auto mapped = OrientationManager::orient_hcurl_triangle_face_dofs(face_dofs, orient, k);
    ASSERT_EQ(mapped.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(mapped[i], face_dofs[i], tol);
    }

    const auto back = OrientationManager::orient_hcurl_triangle_face_dofs(mapped, inv, k);
    ASSERT_EQ(back.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(back[i], face_dofs[i], tol);
    }
}

TEST_F(OrientationManagerTest, OrientHCurlTriangleFaceDofsRotationIsInvertible) {
    constexpr int k = 2;
    const std::size_t block = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1) / 2u;
    const std::size_t ndofs = 2u * block;
    std::vector<Real> face_dofs(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        face_dofs[i] = Real(1.0) + Real(0.11) * Real(i);
    }

    const std::array<int, 3> global = {10, 20, 30};
    const std::array<int, 3> local = {20, 30, 10}; // rotated
    const auto orient = OrientationManager::triangle_face_orientation(local, global);
    const auto inv = OrientationManager::triangle_face_orientation(global, local);

    const auto mapped = OrientationManager::orient_hcurl_triangle_face_dofs(face_dofs, orient, k);
    const auto back = OrientationManager::orient_hcurl_triangle_face_dofs(mapped, inv, k);
    ASSERT_EQ(back.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(back[i], face_dofs[i], tol);
    }
}

TEST_F(OrientationManagerTest, OrientHCurlTriangleFaceDofsReflectionIsInvertible) {
    constexpr int k = 2;
    const std::size_t block = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1) / 2u;
    const std::size_t ndofs = 2u * block;
    std::vector<Real> face_dofs(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        face_dofs[i] = Real(-0.2) + Real(0.09) * Real(i);
    }

    const std::array<int, 3> global = {10, 20, 30};
    const std::array<int, 3> local = {10, 30, 20}; // reflected
    const auto orient = OrientationManager::triangle_face_orientation(local, global);
    const auto inv = OrientationManager::triangle_face_orientation(global, local);

    ASSERT_TRUE(orient.reflection);
    ASSERT_EQ(orient.sign, -1);

    const auto mapped = OrientationManager::orient_hcurl_triangle_face_dofs(face_dofs, orient, k);
    const auto back = OrientationManager::orient_hcurl_triangle_face_dofs(mapped, inv, k);
    ASSERT_EQ(back.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(back[i], face_dofs[i], tol);
    }
}

TEST_F(OrientationManagerTest, OrientHCurlQuadFaceDofsIdentityAndInverse) {
    constexpr int k = 2;
    const std::size_t block = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1);
    const std::size_t ndofs = 2u * block;
    std::vector<Real> face_dofs(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        face_dofs[i] = Real(0.1) * Real(i + 1);
    }

    const std::array<int, 4> global = {10, 20, 30, 40};
    const std::array<int, 4> local = global;
    const auto orient = OrientationManager::quad_face_orientation(local, global);
    const auto inv = OrientationManager::quad_face_orientation(global, local);

    const auto mapped = OrientationManager::orient_hcurl_quad_face_dofs(face_dofs, orient, k);
    EXPECT_EQ(mapped.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(mapped[i], face_dofs[i], tol);
    }

    const auto back = OrientationManager::orient_hcurl_quad_face_dofs(mapped, inv, k);
    ASSERT_EQ(back.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(back[i], face_dofs[i], tol);
    }
}

TEST_F(OrientationManagerTest, OrientHCurlQuadFaceDofsRotationIsInvertible) {
    constexpr int k = 2;
    const std::size_t block = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1);
    const std::size_t ndofs = 2u * block;
    std::vector<Real> face_dofs(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        face_dofs[i] = Real(1.0) + Real(0.05) * Real(i);
    }

    const std::array<int, 4> global = {10, 20, 30, 40};
    const std::array<int, 4> local = {20, 30, 40, 10}; // rotated
    const auto orient = OrientationManager::quad_face_orientation(local, global);
    const auto inv = OrientationManager::quad_face_orientation(global, local);

    const auto mapped = OrientationManager::orient_hcurl_quad_face_dofs(face_dofs, orient, k);
    const auto back = OrientationManager::orient_hcurl_quad_face_dofs(mapped, inv, k);
    ASSERT_EQ(back.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(back[i], face_dofs[i], tol);
    }
}

TEST_F(OrientationManagerTest, OrientHCurlQuadFaceDofsReflectionIsInvertible) {
    constexpr int k = 2;
    const std::size_t block = static_cast<std::size_t>(k) * static_cast<std::size_t>(k + 1);
    const std::size_t ndofs = 2u * block;
    std::vector<Real> face_dofs(ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        face_dofs[i] = Real(-0.3) + Real(0.07) * Real(i);
    }

    const std::array<int, 4> global = {10, 20, 30, 40};
    const std::array<int, 4> local = {10, 40, 30, 20}; // reflected
    const auto orient = OrientationManager::quad_face_orientation(local, global);
    const auto inv = OrientationManager::quad_face_orientation(global, local);

    ASSERT_TRUE(orient.reflection);
    ASSERT_EQ(orient.sign, -1);

    const auto mapped = OrientationManager::orient_hcurl_quad_face_dofs(face_dofs, orient, k);
    const auto back = OrientationManager::orient_hcurl_quad_face_dofs(mapped, inv, k);
    ASSERT_EQ(back.size(), ndofs);
    for (std::size_t i = 0; i < ndofs; ++i) {
        EXPECT_NEAR(back[i], face_dofs[i], tol);
    }
}
