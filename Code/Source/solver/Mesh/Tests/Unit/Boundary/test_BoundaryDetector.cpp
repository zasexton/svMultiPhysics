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
