/* Copyright (c) Stanford University, The Regents of the University of California, and others.
 *
 * All Rights Reserved.
 *
 * See Copyright-SimVascular.txt for additional details.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "gtest/gtest.h"
#include "Boundary/BoundaryDetector.h"
#include "Boundary/BoundaryKey.h"
#include "Core/MeshBase.h"
#include "Topology/CellShape.h"

namespace svmp {
namespace test {

/**
 * @brief Test fixture for BoundaryDetector tests
 *
 * Provides helper methods to create simple test meshes for boundary detection.
 */
class BoundaryDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Cleanup if needed
    }

    /**
     * @brief Create a simple single tetrahedron mesh
     *
     * Creates a tet with 4 vertices and 1 cell. All 4 faces should be on the boundary.
     */
    MeshBase create_single_tet_mesh() {
        // Define vertices
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // Vertex 0
            1.0, 0.0, 0.0,  // Vertex 1
            0.0, 1.0, 0.0,  // Vertex 2
            0.0, 0.0, 1.0   // Vertex 3
        };

        // Define cell connectivity (tetrahedron)
        std::vector<offset_t> offs = {0, 4};
        std::vector<index_t> connectivity = {0, 1, 2, 3};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Tetra;
        shapes[0].order = 1;
        shapes[0].num_corners = 4;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a mesh with two tetrahedra sharing a face
     *
     * The shared face should be interior, other faces should be boundary.
     */
    MeshBase create_two_tet_mesh() {
        // Define vertices
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // Vertex 0
            1.0, 0.0, 0.0,  // Vertex 1
            0.0, 1.0, 0.0,  // Vertex 2
            0.0, 0.0, 1.0,  // Vertex 3
            0.0, 0.0, 2.0   // Vertex 4 (for second tet)
        };

        // Two cells with shared face (0,1,2)
        std::vector<offset_t> offs = {0, 4, 8};
        std::vector<index_t> connectivity = {
            0, 1, 2, 3,   // First tet
            0, 1, 2, 4    // Second tet
        };
        std::vector<CellShape> shapes(2);
        for (int i = 0; i < 2; ++i) {
            shapes[i].family = CellFamily::Tetra;
            shapes[i].order = 1;
            shapes[i].num_corners = 4;
        }

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a simple triangle mesh (2D)
     */
    MeshBase create_single_triangle_mesh() {
        // Define vertices
        std::vector<real_t> X_ref = {
            0.0, 0.0,  // Vertex 0
            1.0, 0.0,  // Vertex 1
            0.0, 1.0   // Vertex 2
        };

        // Single triangle
        std::vector<offset_t> offs = {0, 3};
        std::vector<index_t> connectivity = {0, 1, 2};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Triangle;
        shapes[0].order = 1;
        shapes[0].num_corners = 3;

        MeshBase mesh;
        mesh.build_from_arrays(2, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a simple quad mesh (2D)
     */
    MeshBase create_single_quad_mesh() {
        std::vector<real_t> X_ref = {
            0.0, 0.0,  // Vertex 0
            1.0, 0.0,  // Vertex 1
            1.0, 1.0,  // Vertex 2
            0.0, 1.0   // Vertex 3
        };

        std::vector<offset_t> offs = {0, 4};
        std::vector<index_t> connectivity = {0, 1, 2, 3};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Quad;
        shapes[0].order = 1;
        shapes[0].num_corners = 4;

        MeshBase mesh;
        mesh.build_from_arrays(2, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a single hexahedron mesh (3D)
     */
    MeshBase create_single_hex_mesh() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.0, 0.0, 1.0,  // 4
            1.0, 0.0, 1.0,  // 5
            1.0, 1.0, 1.0,  // 6
            0.0, 1.0, 1.0   // 7
        };

        std::vector<offset_t> offs = {0, 8};
        std::vector<index_t> connectivity = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Hex;
        shapes[0].order = 1;
        shapes[0].num_corners = 8;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a 1D line mesh embedded in 3D coordinates
     */
    MeshBase create_single_line_mesh_in_3d() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0   // 1
        };

        std::vector<offset_t> offs = {0, 2};
        std::vector<index_t> connectivity = {0, 1};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Line;
        shapes[0].order = 1;
        shapes[0].num_corners = 2;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a 2D surface triangle embedded in 3D coordinates
     */
    MeshBase create_surface_triangle_mesh_in_3d() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0   // 2
        };

        std::vector<offset_t> offs = {0, 3};
        std::vector<index_t> connectivity = {0, 1, 2};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Triangle;
        shapes[0].order = 1;
        shapes[0].num_corners = 3;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create two disconnected triangles (2D)
     */
    MeshBase create_two_disconnected_triangles_mesh() {
        std::vector<real_t> X_ref = {
            0.0, 0.0,   // 0
            1.0, 0.0,   // 1
            0.0, 1.0,   // 2
            10.0, 0.0,  // 3
            11.0, 0.0,  // 4
            10.0, 1.0   // 5
        };

        std::vector<offset_t> offs = {0, 3, 6};
        std::vector<index_t> connectivity = {0, 1, 2, 3, 4, 5};
        std::vector<CellShape> shapes(2);
        for (int i = 0; i < 2; ++i) {
            shapes[i].family = CellFamily::Triangle;
            shapes[i].order = 1;
            shapes[i].num_corners = 3;
        }

        MeshBase mesh;
        mesh.build_from_arrays(2, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    /**
     * @brief Create a mixed-dimensional mesh (tetra + triangle).
     *
     * Boundary detection should operate on the maximal cell dimension only.
     */
    MeshBase create_mixed_dim_mesh_tet_plus_triangle() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0   // 3
        };

        std::vector<offset_t> offs = {0, 4, 7};
        std::vector<index_t> connectivity = {
            0, 1, 2, 3,  // tet
            0, 1, 2      // triangle
        };

        std::vector<CellShape> shapes(2);
        shapes[0].family = CellFamily::Tetra;
        shapes[0].order = 1;
        shapes[0].num_corners = 4;
        shapes[1].family = CellFamily::Triangle;
        shapes[1].order = 1;
        shapes[1].num_corners = 3;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }
};

// ==========================================
// Tests: Basic Boundary Detection
// ==========================================

TEST_F(BoundaryDetectorTest, SingleTetHasBoundary) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_TRUE(info.has_boundary());
    EXPECT_FALSE(info.is_closed());
}

TEST_F(BoundaryDetectorTest, SingleTetHasFourBoundaryFaces) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // A single tet has 4 faces, all should be on boundary
    EXPECT_EQ(info.boundary_entities.size(), 4);
    EXPECT_EQ(info.interior_entities.size(), 0);
}

TEST_F(BoundaryDetectorTest, SingleTetHasFourBoundaryVertices) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // All 4 vertices should be on boundary
    EXPECT_EQ(info.boundary_vertices.size(), 4);
}

TEST_F(BoundaryDetectorTest, TwoTetsShareInteriorFace) {
    MeshBase mesh = create_two_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // 2 tets have 8 faces total, but share 1 face
    // So: 7 unique faces, 6 boundary + 1 interior
    EXPECT_EQ(info.boundary_entities.size(), 6);
    EXPECT_EQ(info.interior_entities.size(), 1);
}

// ==========================================
// Tests: Boundary Orientation (Right-Hand Rule)
// ==========================================

TEST_F(BoundaryDetectorTest, BoundaryFacesHaveOrientation) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // Each boundary face should have oriented vertices
    EXPECT_EQ(info.oriented_boundary_entities.size(), info.boundary_entities.size());

    // Each face should have 3 vertices (triangular)
    for (const auto& oriented_verts : info.oriented_boundary_entities) {
        EXPECT_EQ(oriented_verts.size(), 3);
    }
}

// ==========================================
// Tests: Chain Complex Approach
// ==========================================

TEST_F(BoundaryDetectorTest, ChainComplexMatchesIncidenceCounting) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();
    auto chain_boundary = detector.detect_boundary_chain_complex();

    // Both methods should find the same number of boundary faces
    EXPECT_EQ(chain_boundary.size(), info.boundary_entities.size());
}

// ==========================================
// Tests: Closed Mesh Detection
// ==========================================

TEST_F(BoundaryDetectorTest, SingleTetIsNotClosed) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    EXPECT_FALSE(detector.is_closed_mesh());
}

// ==========================================
// Tests: Non-Manifold Detection
// ==========================================

TEST_F(BoundaryDetectorTest, ManifoldMeshHasNoNonManifoldFaces) {
    MeshBase mesh = create_two_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_FALSE(info.has_nonmanifold());
    EXPECT_EQ(info.nonmanifold_entities.size(), 0);
}

// ==========================================
// Tests: 2D Mesh (Triangle)
// ==========================================

TEST_F(BoundaryDetectorTest, SingleTriangleHasBoundary) {
    MeshBase mesh = create_single_triangle_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_TRUE(info.has_boundary());
    // A single triangle has 3 edges, all should be on boundary
    EXPECT_EQ(info.boundary_entities.size(), 3);
}

TEST_F(BoundaryDetectorTest, SingleTriangleHasSingleBoundaryComponent) {
    MeshBase mesh = create_single_triangle_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 3u);
    EXPECT_EQ(info.components[0].n_vertices(), 3u);
}

TEST_F(BoundaryDetectorTest, TwoDisconnectedTrianglesHaveTwoBoundaryComponents) {
    MeshBase mesh = create_two_disconnected_triangles_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 6u);
    ASSERT_EQ(info.components.size(), 2u);
    for (const auto& comp : info.components) {
        EXPECT_EQ(comp.n_entities(), 3u);
        EXPECT_EQ(comp.n_vertices(), 3u);
    }
}

TEST_F(BoundaryDetectorTest, SingleQuadHasSingleBoundaryComponent) {
    MeshBase mesh = create_single_quad_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 4u);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 4u);
    EXPECT_EQ(info.components[0].n_vertices(), 4u);
}

TEST_F(BoundaryDetectorTest, SingleHexHasSingleBoundaryComponent) {
    MeshBase mesh = create_single_hex_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 6u);
    EXPECT_EQ(info.boundary_vertices.size(), 8u);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 6u);
    EXPECT_EQ(info.components[0].n_vertices(), 8u);
}

TEST_F(BoundaryDetectorTest, LineIn3DHasTwoBoundaryVertices) {
    MeshBase mesh = create_single_line_mesh_in_3d();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 2u);
    EXPECT_EQ(info.boundary_vertices.size(), 2u);
    ASSERT_EQ(info.boundary_types.size(), info.boundary_entities.size());
    for (auto kind : info.boundary_types) {
        EXPECT_EQ(kind, EntityKind::Vertex);
    }
    EXPECT_EQ(info.components.size(), 2u);
}

TEST_F(BoundaryDetectorTest, SurfaceTriangleIn3DHasEdgeBoundaryTypes) {
    MeshBase mesh = create_surface_triangle_mesh_in_3d();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 3u);
    ASSERT_EQ(info.boundary_types.size(), info.boundary_entities.size());
    for (auto kind : info.boundary_types) {
        EXPECT_EQ(kind, EntityKind::Edge);
    }
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 3u);
}

TEST_F(BoundaryDetectorTest, MixedDimMeshUsesMaximalCellsOnly) {
    MeshBase mesh = create_mixed_dim_mesh_tet_plus_triangle();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // Only tet faces should contribute (4 boundary faces); the triangle is lower-dimensional.
    EXPECT_EQ(info.boundary_entities.size(), 4u);
    ASSERT_EQ(info.boundary_types.size(), info.boundary_entities.size());
    for (auto kind : info.boundary_types) {
        EXPECT_EQ(kind, EntityKind::Face);
    }

    // Tet boundary faces are triangles.
    ASSERT_EQ(info.oriented_boundary_entities.size(), info.boundary_entities.size());
    for (const auto& verts : info.oriented_boundary_entities) {
        EXPECT_EQ(verts.size(), 3u);
    }
}

TEST_F(BoundaryDetectorTest, TriangleBoundaryEdgesHaveTwoVertices) {
    MeshBase mesh = create_single_triangle_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // Each boundary edge should have 2 vertices
    for (const auto& oriented_verts : info.oriented_boundary_entities) {
        EXPECT_EQ(oriented_verts.size(), 2);
    }
}

// ==========================================
// Tests: BoundaryKey Functionality
// ==========================================

TEST_F(BoundaryDetectorTest, BoundaryKeyEqualityIgnoresOrder) {
    std::vector<index_t> v1 = {0, 1, 2};
    std::vector<index_t> v2 = {2, 0, 1};
    std::vector<index_t> v3 = {1, 2, 0};

    BoundaryKey key1(v1);
    BoundaryKey key2(v2);
    BoundaryKey key3(v3);

    // All should be equal (same vertices, different order)
    EXPECT_EQ(key1, key2);
    EXPECT_EQ(key2, key3);
    EXPECT_EQ(key1, key3);
}

TEST_F(BoundaryDetectorTest, BoundaryKeyHashConsistent) {
    std::vector<index_t> v1 = {0, 1, 2};
    std::vector<index_t> v2 = {2, 0, 1};

    BoundaryKey key1(v1);
    BoundaryKey key2(v2);

    BoundaryKey::Hash hasher;

    // Same keys should produce same hash
    EXPECT_EQ(hasher(key1), hasher(key2));
}

// ==========================================
// Tests: Connected Components
// ==========================================

TEST_F(BoundaryDetectorTest, SingleTetHasOneComponent) {
    MeshBase mesh = create_single_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // Single tet should have 1 connected boundary component
    EXPECT_EQ(info.n_components(), 1);
}

TEST_F(BoundaryDetectorTest, TwoTetsHaveOneComponent) {
    MeshBase mesh = create_two_tet_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // Two tets sharing a face should still form 1 connected boundary
    EXPECT_EQ(info.n_components(), 1);
}

} // namespace test
} // namespace svmp
