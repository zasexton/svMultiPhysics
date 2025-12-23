/**
 * @file test_FaceRestriction.cpp
 * @brief Unit tests for FaceRestriction
 */

#include <gtest/gtest.h>
#include "FE/Spaces/FaceRestriction.h"
#include "FE/Spaces/H1Space.h"
#include "FE/Spaces/L2Space.h"
#include "FE/Basis/LagrangeBasis.h"
#include <algorithm>
#include <numeric>

using namespace svmp::FE;
using namespace svmp::FE::spaces;

class FaceRestrictionTest : public ::testing::Test {
protected:
    static constexpr Real tol = 1e-12;
};

// =============================================================================
// Topology Tests
// =============================================================================

TEST_F(FaceRestrictionTest, TriangleTopology) {
    FaceRestriction fr(ElementType::Triangle3, 1);

    const auto& topo = fr.topology();
    EXPECT_EQ(topo.dimension, 2);
    EXPECT_EQ(topo.num_vertices, 3);
    EXPECT_EQ(topo.num_edges, 3);
    EXPECT_EQ(topo.num_faces, 3);  // In 2D, faces = edges
}

TEST_F(FaceRestrictionTest, QuadTopology) {
    FaceRestriction fr(ElementType::Quad4, 1);

    const auto& topo = fr.topology();
    EXPECT_EQ(topo.dimension, 2);
    EXPECT_EQ(topo.num_vertices, 4);
    EXPECT_EQ(topo.num_edges, 4);
    EXPECT_EQ(topo.num_faces, 4);
}

TEST_F(FaceRestrictionTest, TetrahedronTopology) {
    FaceRestriction fr(ElementType::Tetra4, 1);

    const auto& topo = fr.topology();
    EXPECT_EQ(topo.dimension, 3);
    EXPECT_EQ(topo.num_vertices, 4);
    EXPECT_EQ(topo.num_edges, 6);
    EXPECT_EQ(topo.num_faces, 4);
}

TEST_F(FaceRestrictionTest, HexahedronTopology) {
    FaceRestriction fr(ElementType::Hex8, 1);

    const auto& topo = fr.topology();
    EXPECT_EQ(topo.dimension, 3);
    EXPECT_EQ(topo.num_vertices, 8);
    EXPECT_EQ(topo.num_edges, 12);
    EXPECT_EQ(topo.num_faces, 6);
}

TEST_F(FaceRestrictionTest, WedgeTopology) {
    FaceRestriction fr(ElementType::Wedge6, 1);

    const auto& topo = fr.topology();
    EXPECT_EQ(topo.dimension, 3);
    EXPECT_EQ(topo.num_vertices, 6);
    EXPECT_EQ(topo.num_edges, 9);
    EXPECT_EQ(topo.num_faces, 5);  // 2 triangles + 3 quads
}

TEST_F(FaceRestrictionTest, PyramidTopology) {
    FaceRestriction fr(ElementType::Pyramid5, 1);

    const auto& topo = fr.topology();
    EXPECT_EQ(topo.dimension, 3);
    EXPECT_EQ(topo.num_vertices, 5);
    EXPECT_EQ(topo.num_edges, 8);
    EXPECT_EQ(topo.num_faces, 5);  // 1 quad + 4 triangles
}

// =============================================================================
// DOF Count Tests (H1)
// =============================================================================

TEST_F(FaceRestrictionTest, TriangleP1FaceDofs) {
    FaceRestriction fr(ElementType::Triangle3, 1, Continuity::C0);

    // P1 triangle: 3 vertex DOFs, 0 edge interior DOFs
    EXPECT_EQ(fr.num_element_dofs(), 3u);

    // Each edge (face in 2D) has 2 vertex DOFs
    for (int f = 0; f < 3; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 2u);
    }
}

TEST_F(FaceRestrictionTest, TriangleP2FaceDofs) {
    FaceRestriction fr(ElementType::Triangle6, 2, Continuity::C0);

    // P2 triangle: 3 vertex + 3 edge interior = 6 DOFs
    EXPECT_EQ(fr.num_element_dofs(), 6u);

    // Each edge has 2 vertex + 1 interior = 3 DOFs
    for (int f = 0; f < 3; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 3u);
    }
}

TEST_F(FaceRestrictionTest, QuadP1FaceDofs) {
    FaceRestriction fr(ElementType::Quad4, 1, Continuity::C0);

    // Q1 quad: 4 vertex DOFs
    EXPECT_EQ(fr.num_element_dofs(), 4u);

    // Each edge has 2 vertex DOFs
    for (int f = 0; f < 4; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 2u);
    }
}

TEST_F(FaceRestrictionTest, TetraP1FaceDofs) {
    FaceRestriction fr(ElementType::Tetra4, 1, Continuity::C0);

    // P1 tet: 4 vertex DOFs
    EXPECT_EQ(fr.num_element_dofs(), 4u);

    // Each triangular face has 3 vertex DOFs
    for (int f = 0; f < 4; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 3u);
    }
}

TEST_F(FaceRestrictionTest, HexP1FaceDofs) {
    FaceRestriction fr(ElementType::Hex8, 1, Continuity::C0);

    // Q1 hex: 8 vertex DOFs
    EXPECT_EQ(fr.num_element_dofs(), 8u);

    // Each quad face has 4 vertex DOFs
    for (int f = 0; f < 6; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 4u);
    }
}

// =============================================================================
// L2 Space Tests (all DOFs interior)
// =============================================================================

TEST_F(FaceRestrictionTest, L2SpaceAllInterior) {
    FaceRestriction fr(ElementType::Triangle3, 2, Continuity::L2);

    // L2 P2 triangle: (p+1)(p+2)/2 = 6 DOFs
    EXPECT_EQ(fr.num_element_dofs(), 6u);

    // Face restriction is geometric: boundary nodes still appear on faces.
    for (int f = 0; f < 3; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 3u);
    }

    auto interior = fr.interior_dofs();
    EXPECT_EQ(interior.size(), 0u);
}

// =============================================================================
// Restrict/Scatter Tests
// =============================================================================

TEST_F(FaceRestrictionTest, RestrictToFaceTriangle) {
    FaceRestriction fr(ElementType::Triangle3, 1, Continuity::C0);

    // Element DOFs: v0, v1, v2
    std::vector<Real> elem_vals = {1.0, 2.0, 3.0};

    // Face 0 connects v0-v1
    auto face0 = fr.restrict_to_face(elem_vals, 0);
    EXPECT_EQ(face0.size(), 2u);
    // Values should be from v0 and v1 (order may vary)
    EXPECT_TRUE((face0[0] == 1.0 && face0[1] == 2.0) ||
                (face0[0] == 2.0 && face0[1] == 1.0));
}

TEST_F(FaceRestrictionTest, ScatterFromFaceTriangle) {
    FaceRestriction fr(ElementType::Triangle3, 1, Continuity::C0);

    std::vector<Real> elem_vals(3, 0.0);
    std::vector<Real> face_vals = {10.0, 20.0};

    fr.scatter_from_face(face_vals, 0, elem_vals);

    // Values should be added to appropriate positions
    // (exact positions depend on DOF ordering)
    Real sum = elem_vals[0] + elem_vals[1] + elem_vals[2];
    EXPECT_NEAR(sum, 30.0, tol);
}

TEST_F(FaceRestrictionTest, RoundTripRestrictScatter) {
    FaceRestriction fr(ElementType::Quad4, 1, Continuity::C0);

    std::vector<Real> original = {1.0, 2.0, 3.0, 4.0};
    std::vector<Real> reconstructed(4, 0.0);

    // Scatter from each face
    for (int f = 0; f < 4; ++f) {
        auto face_vals = fr.restrict_to_face(original, f);
        fr.scatter_from_face(face_vals, f, reconstructed);
    }

    // Each vertex should be hit by two faces
    for (std::size_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR(reconstructed[i], 2.0 * original[i], tol);
    }
}

// =============================================================================
// Factory Cache Tests
// =============================================================================

TEST_F(FaceRestrictionTest, FactoryCaching) {
    auto fr1 = FaceRestrictionFactory::get(ElementType::Triangle3, 2, Continuity::C0);
    auto fr2 = FaceRestrictionFactory::get(ElementType::Triangle3, 2, Continuity::C0);

    // Should return same pointer (cached)
    EXPECT_EQ(fr1.get(), fr2.get());
}

TEST_F(FaceRestrictionTest, FactoryDifferentConfigs) {
    auto fr1 = FaceRestrictionFactory::get(ElementType::Triangle3, 1, Continuity::C0);
    auto fr2 = FaceRestrictionFactory::get(ElementType::Triangle3, 2, Continuity::C0);

    // Different configurations should return different objects
    EXPECT_NE(fr1.get(), fr2.get());
}

TEST_F(FaceRestrictionTest, FactoryFromSpace) {
    H1Space space(ElementType::Quad4, 2);
    auto fr = FaceRestrictionFactory::get(space);

    EXPECT_EQ(fr->element_type(), ElementType::Quad4);
    EXPECT_EQ(fr->polynomial_order(), 2);
}

// =============================================================================
// Edge DOF Tests
// =============================================================================

TEST_F(FaceRestrictionTest, EdgeDofsTetrahedron) {
    FaceRestriction fr(ElementType::Tetra10, 2, Continuity::C0);

    // P2 tet: 4 vertex + 6 edge DOFs = 10 total
    EXPECT_EQ(fr.num_element_dofs(), 10u);

    // Each edge should have 3 DOFs (2 vertex + 1 interior)
    for (int e = 0; e < 6; ++e) {
        EXPECT_EQ(fr.num_edge_dofs(e), 3u);
    }
}

TEST_F(FaceRestrictionTest, VertexDofsLinearElements) {
    FaceRestriction fr(ElementType::Tetra4, 1, Continuity::C0);

    // Each vertex should have exactly 1 DOF
    for (int v = 0; v < 4; ++v) {
        auto vdofs = fr.vertex_dofs(v);
        EXPECT_EQ(vdofs.size(), 1u);
    }
}

TEST_F(FaceRestrictionTest, InteriorDofsHighOrder) {
    FaceRestriction fr(ElementType::Triangle3, 3, Continuity::C0);

    // P3 triangle: (p+1)(p+2)/2 = 10 DOFs total
    EXPECT_EQ(fr.num_element_dofs(), 10u);

    // Interior DOFs: (p-1)(p-2)/2 = 1
    auto interior = fr.interior_dofs();
    ASSERT_EQ(interior.size(), 1u);
    for (int f = 0; f < fr.topology().num_faces; ++f) {
        const auto fdofs = fr.face_dofs(f);
        EXPECT_EQ(std::find(fdofs.begin(), fdofs.end(), interior[0]), fdofs.end());
    }

    // Each face (edge) has 2 vertex + (p-1) edge DOFs = 4 DOFs.
    for (int f = 0; f < fr.topology().num_faces; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 4u);
    }
}

TEST_F(FaceRestrictionTest, WedgeFaceTypesAreMixedTriangleAndQuad) {
    FaceRestriction fr(ElementType::Wedge6, 1, Continuity::C0);
    const auto& topo = fr.topology();
    ASSERT_EQ(topo.face_types.size(), 5u);
    ASSERT_EQ(topo.face_vertices.size(), 5u);

    // Faces 0/1 are triangles, remaining are quads.
    EXPECT_EQ(topo.face_types[0], ElementType::Triangle3);
    EXPECT_EQ(topo.face_types[1], ElementType::Triangle3);
    EXPECT_EQ(topo.face_types[2], ElementType::Quad4);
    EXPECT_EQ(topo.face_types[3], ElementType::Quad4);
    EXPECT_EQ(topo.face_types[4], ElementType::Quad4);

    EXPECT_EQ(topo.face_vertices[0].size(), 3u);
    EXPECT_EQ(topo.face_vertices[1].size(), 3u);
    EXPECT_EQ(topo.face_vertices[2].size(), 4u);
    EXPECT_EQ(topo.face_vertices[3].size(), 4u);
    EXPECT_EQ(topo.face_vertices[4].size(), 4u);
}

TEST_F(FaceRestrictionTest, QuadP3InteriorDofsAreContiguousAtEnd) {
    const int p = 3;
    FaceRestriction fr(ElementType::Quad4, p, Continuity::C0);

    // Qp quad: (p+1)^2 DOFs
    EXPECT_EQ(fr.num_element_dofs(), static_cast<std::size_t>((p + 1) * (p + 1)));

    // Interior DOFs: (p-1)^2 = 4
    auto interior = fr.interior_dofs();
    ASSERT_EQ(interior.size(), 4u);

    svmp::FE::basis::LagrangeBasis basis(ElementType::Quad4, p);
    const auto& nodes = basis.nodes();
    ASSERT_EQ(nodes.size(), fr.num_element_dofs());
    for (int dof : interior) {
        const auto& x = nodes[static_cast<std::size_t>(dof)];
        EXPECT_LT(std::abs(x[0]), 1.0 - tol);
        EXPECT_LT(std::abs(x[1]), 1.0 - tol);
    }
}

// =============================================================================
// Supported Element Tests
// =============================================================================

TEST_F(FaceRestrictionTest, IsSupported) {
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Triangle3));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Quad4));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Tetra4));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Hex8));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Wedge6));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Pyramid5));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Wedge18));
    EXPECT_TRUE(FaceRestriction::is_supported(ElementType::Pyramid14));

    EXPECT_FALSE(FaceRestriction::is_supported(ElementType::Point1));
    EXPECT_FALSE(FaceRestriction::is_supported(ElementType::Unknown));
}

// =============================================================================
// Serendipity / Special Element Coverage
// =============================================================================

TEST_F(FaceRestrictionTest, Quad8Order2HasThreeDofsPerEdge) {
    FaceRestriction fr(ElementType::Quad8, 2, Continuity::C0);
    EXPECT_EQ(fr.num_element_dofs(), 8u);
    for (int f = 0; f < fr.topology().num_faces; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 3u);
    }
}

TEST_F(FaceRestrictionTest, Hex20Order2HasEightDofsPerFace) {
    FaceRestriction fr(ElementType::Hex20, 2, Continuity::C0);
    EXPECT_EQ(fr.num_element_dofs(), 20u);
    for (int f = 0; f < fr.topology().num_faces; ++f) {
        EXPECT_EQ(fr.num_face_dofs(f), 8u);
    }
}

TEST_F(FaceRestrictionTest, Wedge15Order2HasTriangle6AndQuad8Faces) {
    FaceRestriction fr(ElementType::Wedge15, 2, Continuity::C0);
    EXPECT_EQ(fr.num_element_dofs(), 15u);
    const auto& topo = fr.topology();
    ASSERT_EQ(topo.face_types.size(), 5u);
    for (int f = 0; f < topo.num_faces; ++f) {
        const auto type = topo.face_types[static_cast<std::size_t>(f)];
        if (type == ElementType::Triangle3) {
            EXPECT_EQ(fr.num_face_dofs(f), 6u);
        } else if (type == ElementType::Quad4) {
            EXPECT_EQ(fr.num_face_dofs(f), 8u);
        } else {
            FAIL() << "Unexpected wedge face type";
        }
    }
}

TEST_F(FaceRestrictionTest, Wedge18Order2HasTriangle6AndQuad9Faces) {
    FaceRestriction fr(ElementType::Wedge18, 2, Continuity::C0);
    EXPECT_EQ(fr.num_element_dofs(), 18u);
    const auto& topo = fr.topology();
    ASSERT_EQ(topo.face_types.size(), 5u);
    for (int f = 0; f < topo.num_faces; ++f) {
        const auto type = topo.face_types[static_cast<std::size_t>(f)];
        if (type == ElementType::Triangle3) {
            EXPECT_EQ(fr.num_face_dofs(f), 6u);
        } else if (type == ElementType::Quad4) {
            EXPECT_EQ(fr.num_face_dofs(f), 9u);
        } else {
            FAIL() << "Unexpected wedge face type";
        }
    }
}

TEST_F(FaceRestrictionTest, Pyramid13Order2HasQuad8AndTriangle6Faces) {
    FaceRestriction fr(ElementType::Pyramid13, 2, Continuity::C0);
    EXPECT_EQ(fr.num_element_dofs(), 13u);
    const auto& topo = fr.topology();
    ASSERT_EQ(topo.face_types.size(), 5u);
    for (int f = 0; f < topo.num_faces; ++f) {
        const auto type = topo.face_types[static_cast<std::size_t>(f)];
        if (type == ElementType::Quad4) {
            EXPECT_EQ(fr.num_face_dofs(f), 8u);
        } else if (type == ElementType::Triangle3) {
            EXPECT_EQ(fr.num_face_dofs(f), 6u);
        } else {
            FAIL() << "Unexpected pyramid face type";
        }
    }
}

TEST_F(FaceRestrictionTest, Pyramid14Order2HasQuad9AndTriangle6Faces) {
    FaceRestriction fr(ElementType::Pyramid14, 2, Continuity::C0);
    EXPECT_EQ(fr.num_element_dofs(), 14u);
    const auto& topo = fr.topology();
    ASSERT_EQ(topo.face_types.size(), 5u);
    for (int f = 0; f < topo.num_faces; ++f) {
        const auto type = topo.face_types[static_cast<std::size_t>(f)];
        if (type == ElementType::Quad4) {
            EXPECT_EQ(fr.num_face_dofs(f), 9u);
        } else if (type == ElementType::Triangle3) {
            EXPECT_EQ(fr.num_face_dofs(f), 6u);
        } else {
            FAIL() << "Unexpected pyramid face type";
        }
    }
}
