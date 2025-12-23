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

#include <algorithm>
#include <cmath>

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

    MeshBase create_single_wedge_mesh() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0,  // 3
            1.0, 0.0, 1.0,  // 4
            0.0, 1.0, 1.0   // 5
        };

        std::vector<offset_t> offs = {0, 6};
        std::vector<index_t> connectivity = {0, 1, 2, 3, 4, 5};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Wedge;
        shapes[0].order = 1;
        shapes[0].num_corners = 6;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    MeshBase create_single_pyramid_mesh() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            1.0, 1.0, 0.0,  // 2
            0.0, 1.0, 0.0,  // 3
            0.5, 0.5, 1.0   // 4 apex
        };

        std::vector<offset_t> offs = {0, 5};
        std::vector<index_t> connectivity = {0, 1, 2, 3, 4};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Pyramid;
        shapes[0].order = 1;
        shapes[0].num_corners = 5;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    MeshBase create_single_polygon_mesh(int n) {
        std::vector<real_t> X_ref;
        X_ref.reserve(static_cast<size_t>(2 * n));
        for (int i = 0; i < n; ++i) {
            const real_t a = 2.0 * 3.14159265358979323846 * static_cast<real_t>(i) / static_cast<real_t>(n);
            X_ref.push_back(std::cos(a));
            X_ref.push_back(std::sin(a));
        }

        std::vector<offset_t> offs = {0, static_cast<offset_t>(n)};
        std::vector<index_t> connectivity;
        connectivity.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) connectivity.push_back(i);

        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Polygon;
        shapes[0].order = 1;
        shapes[0].num_corners = n;

        MeshBase mesh;
        mesh.build_from_arrays(2, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    MeshBase create_three_tets_share_face_nonmanifold_mesh() {
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0,  // 3
            0.0, 0.0, 2.0,  // 4
            0.0, 0.0, 3.0   // 5
        };

        std::vector<offset_t> offs = {0, 4, 8, 12};
        std::vector<index_t> connectivity = {
            0, 1, 2, 3,
            0, 1, 2, 4,
            0, 1, 2, 5
        };

        std::vector<CellShape> shapes(3);
        for (int i = 0; i < 3; ++i) {
            shapes[i].family = CellFamily::Tetra;
            shapes[i].order = 1;
            shapes[i].num_corners = 4;
        }

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    MeshBase create_quadratic_tet_mesh_p2() {
        // VTK-style quadratic tetra ordering: corners then edge nodes in edge-view order.
        std::vector<real_t> X_ref = {
            0.0, 0.0, 0.0,  // 0
            1.0, 0.0, 0.0,  // 1
            0.0, 1.0, 0.0,  // 2
            0.0, 0.0, 1.0,  // 3
            0.5, 0.0, 0.0,  // 4 (0-1)
            0.0, 0.5, 0.0,  // 5 (0-2)
            0.0, 0.0, 0.5,  // 6 (0-3)
            0.5, 0.5, 0.0,  // 7 (1-2)
            0.5, 0.0, 0.5,  // 8 (1-3)
            0.0, 0.5, 0.5   // 9 (2-3)
        };

        std::vector<offset_t> offs = {0, 10};
        std::vector<index_t> connectivity = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Tetra;
        shapes[0].order = 2;
        shapes[0].num_corners = 4;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    MeshBase create_cubic_tet_mesh_p3() {
        // VTK-style cubic tetra ordering:
        // corners (4) + edge nodes (6 edges * 2 nodes) + face interior nodes (4 faces * 1 node) = 20.
        std::vector<real_t> X_ref;
        X_ref.reserve(20 * 3);

        auto push = [&](real_t x, real_t y, real_t z) {
            X_ref.push_back(x);
            X_ref.push_back(y);
            X_ref.push_back(z);
        };

        // corners
        push(0.0, 0.0, 0.0); // 0
        push(1.0, 0.0, 0.0); // 1
        push(0.0, 1.0, 0.0); // 2
        push(0.0, 0.0, 1.0); // 3

        // edge nodes (two per edge)
        push(1.0 / 3.0, 0.0, 0.0); // 4  (0-1)
        push(2.0 / 3.0, 0.0, 0.0); // 5  (0-1)

        push(0.0, 1.0 / 3.0, 0.0); // 6  (0-2)
        push(0.0, 2.0 / 3.0, 0.0); // 7  (0-2)

        push(0.0, 0.0, 1.0 / 3.0); // 8  (0-3)
        push(0.0, 0.0, 2.0 / 3.0); // 9  (0-3)

        push(2.0 / 3.0, 1.0 / 3.0, 0.0); // 10 (1-2)
        push(1.0 / 3.0, 2.0 / 3.0, 0.0); // 11 (1-2)

        push(2.0 / 3.0, 0.0, 1.0 / 3.0); // 12 (1-3)
        push(1.0 / 3.0, 0.0, 2.0 / 3.0); // 13 (1-3)

        push(0.0, 2.0 / 3.0, 1.0 / 3.0); // 14 (2-3)
        push(0.0, 1.0 / 3.0, 2.0 / 3.0); // 15 (2-3)

        // face interior nodes (one per face in the oriented-face table order)
        push(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0); // 16 face {1,2,3}
        push(0.0, 1.0 / 3.0, 1.0 / 3.0);       // 17 face {0,3,2}
        push(1.0 / 3.0, 0.0, 1.0 / 3.0);       // 18 face {0,1,3}
        push(1.0 / 3.0, 1.0 / 3.0, 0.0);       // 19 face {0,2,1}

        std::vector<offset_t> offs = {0, 20};
        std::vector<index_t> connectivity;
        connectivity.reserve(20);
        for (int i = 0; i < 20; ++i) connectivity.push_back(i);

        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Tetra;
        shapes[0].order = 3;
        shapes[0].num_corners = 4;

        MeshBase mesh;
        mesh.build_from_arrays(3, X_ref, offs, connectivity, shapes);
        return mesh;
    }

    MeshBase create_cubic_triangle_mesh_p3() {
        // VTK-style cubic triangle ordering: corners (3) + edge nodes (3 edges * 2 nodes) + 1 interior = 10.
        std::vector<real_t> X_ref;
        X_ref.reserve(10 * 2);

        auto push = [&](real_t x, real_t y) {
            X_ref.push_back(x);
            X_ref.push_back(y);
        };

        // corners
        push(0.0, 0.0); // 0
        push(1.0, 0.0); // 1
        push(0.0, 1.0); // 2

        // edge nodes (two per edge, in edge-view order)
        push(1.0 / 3.0, 0.0); // 3 (0-1)
        push(2.0 / 3.0, 0.0); // 4 (0-1)

        push(2.0 / 3.0, 1.0 / 3.0); // 5 (1-2)
        push(1.0 / 3.0, 2.0 / 3.0); // 6 (1-2)

        push(0.0, 2.0 / 3.0); // 7 (2-0)
        push(0.0, 1.0 / 3.0); // 8 (2-0)

        // interior node
        push(1.0 / 3.0, 1.0 / 3.0); // 9

        std::vector<offset_t> offs = {0, 10};
        std::vector<index_t> connectivity;
        connectivity.reserve(10);
        for (int i = 0; i < 10; ++i) connectivity.push_back(i);

        std::vector<CellShape> shapes(1);
        shapes[0].family = CellFamily::Triangle;
        shapes[0].order = 3;
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

// ==========================================
// Additional Tests: Wedge / Pyramid / Polygon
// ==========================================

TEST_F(BoundaryDetectorTest, SingleWedgeHasFiveBoundaryFacesAndOneComponent) {
    MeshBase mesh = create_single_wedge_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 5u);
    EXPECT_EQ(info.interior_entities.size(), 0u);
    EXPECT_FALSE(info.has_nonmanifold());

    // Two triangles and three quads.
    size_t tri = 0, quad = 0;
    for (const auto& verts : info.oriented_boundary_entities) {
        if (verts.size() == 3) tri++;
        if (verts.size() == 4) quad++;
    }
    EXPECT_EQ(tri, 2u);
    EXPECT_EQ(quad, 3u);

    EXPECT_EQ(info.boundary_vertices.size(), 6u);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 5u);
    EXPECT_EQ(info.components[0].n_vertices(), 6u);
}

TEST_F(BoundaryDetectorTest, SinglePyramidHasFiveBoundaryFacesAndOneComponent) {
    MeshBase mesh = create_single_pyramid_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 5u);
    EXPECT_FALSE(info.has_nonmanifold());

    size_t tri = 0, quad = 0;
    for (const auto& verts : info.oriented_boundary_entities) {
        if (verts.size() == 3) tri++;
        if (verts.size() == 4) quad++;
    }
    EXPECT_EQ(tri, 4u);
    EXPECT_EQ(quad, 1u);

    EXPECT_EQ(info.boundary_vertices.size(), 5u);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 5u);
    EXPECT_EQ(info.components[0].n_vertices(), 5u);
}

TEST_F(BoundaryDetectorTest, Polygon2DHasBoundaryEdgesAndOneComponent) {
    MeshBase mesh = create_single_polygon_mesh(5);
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 5u);
    EXPECT_EQ(info.boundary_vertices.size(), 5u);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_entities(), 5u);
    EXPECT_EQ(info.components[0].n_vertices(), 5u);
}

// ==========================================
// Additional Tests: Non-manifold
// ==========================================

TEST_F(BoundaryDetectorTest, NonManifoldFaceDetected) {
    MeshBase mesh = create_three_tets_share_face_nonmanifold_mesh();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_TRUE(info.has_nonmanifold());
    EXPECT_EQ(info.nonmanifold_entities.size(), 1u);
    EXPECT_EQ(info.interior_entities.size(), 0u);
    EXPECT_EQ(info.boundary_entities.size(), 9u);
    EXPECT_EQ(info.boundary_vertices.size(), 6u);

    const auto nonmanifold = detector.detect_nonmanifold_codim1();
    EXPECT_EQ(nonmanifold.size(), 1u);
}

// ==========================================
// Additional Tests: High-order elements
// ==========================================

TEST_F(BoundaryDetectorTest, QuadraticTetraIncludesAllBoundaryNodes) {
    MeshBase mesh = create_quadratic_tet_mesh_p2();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    // Same number of topological faces, but boundary nodes include edge midpoints.
    EXPECT_EQ(info.boundary_entities.size(), 4u);
    EXPECT_EQ(info.boundary_vertices.size(), 10u);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_vertices(), 10u);

    ASSERT_EQ(info.oriented_boundary_entities.size(), info.boundary_entities.size());
    for (const auto& verts : info.oriented_boundary_entities) {
        // Quadratic triangle boundary ring: 3 corners + 3 edge nodes.
        EXPECT_EQ(verts.size(), 6u);
    }
}

TEST_F(BoundaryDetectorTest, CubicTetraIncludesFaceInteriorNodesInBoundaryVertices) {
    MeshBase mesh = create_cubic_tet_mesh_p3();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 4u);
    EXPECT_EQ(info.boundary_vertices.size(), 20u);
    EXPECT_TRUE(info.boundary_vertices.count(16) > 0);
    EXPECT_TRUE(info.boundary_vertices.count(19) > 0);

    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_vertices(), 20u);

    for (const auto& verts : info.oriented_boundary_entities) {
        // Cubic triangle boundary ring: 3 corners + 3 edges*(2 nodes) = 9 nodes.
        EXPECT_EQ(verts.size(), 9u);
        EXPECT_TRUE(std::find(verts.begin(), verts.end(), 16) == verts.end());
    }
}

TEST_F(BoundaryDetectorTest, CubicTriangleExcludesInteriorNodeFromBoundaryVertices) {
    MeshBase mesh = create_cubic_triangle_mesh_p3();
    BoundaryDetector detector(mesh);

    auto info = detector.detect_boundary();

    EXPECT_EQ(info.boundary_entities.size(), 3u);
    EXPECT_EQ(info.boundary_vertices.size(), 9u);
    EXPECT_TRUE(info.boundary_vertices.count(9) == 0);
    ASSERT_EQ(info.components.size(), 1u);
    EXPECT_EQ(info.components[0].n_vertices(), 9u);

    auto expected_edge_nodes = [](index_t a, index_t b) -> std::vector<index_t> {
        if (a > b) std::swap(a, b);
        if (a == 0 && b == 1) return {3, 4};
        if (a == 1 && b == 2) return {5, 6};
        if (a == 0 && b == 2) return {7, 8};
        return {};
    };

    ASSERT_EQ(info.oriented_boundary_entities.size(), info.boundary_entities.size());
    for (size_t i = 0; i < info.boundary_entities.size(); ++i) {
        const index_t ent_id = info.boundary_entities[i];
        const auto& key = info.entity_keys[ent_id];
        const auto& endpoints = key.vertices();
        ASSERT_EQ(endpoints.size(), 2u);
        auto expected = expected_edge_nodes(endpoints[0], endpoints[1]);
        ASSERT_EQ(expected.size(), 2u);

        const auto& verts = info.oriented_boundary_entities[i];
        // Cubic edge polyline: 2 corners + 2 edge nodes = 4.
        EXPECT_EQ(verts.size(), 4u);
        EXPECT_TRUE(std::find(verts.begin(), verts.end(), 9) == verts.end());

        EXPECT_TRUE((verts.front() == endpoints[0] && verts.back() == endpoints[1]) ||
                    (verts.front() == endpoints[1] && verts.back() == endpoints[0]));

        std::vector<index_t> middle = {verts[1], verts[2]};
        std::sort(middle.begin(), middle.end());
        std::sort(expected.begin(), expected.end());
        EXPECT_EQ(middle, expected);
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
