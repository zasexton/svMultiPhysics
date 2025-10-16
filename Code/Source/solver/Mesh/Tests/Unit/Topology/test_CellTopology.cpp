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
#include "Topology/CellTopology.h"
#include "Topology/CellShape.h"

namespace svmp {
namespace test {

/**
 * @brief Test fixture for CellTopology tests
 *
 * Tests the canonical topology definitions for all cell types.
 */
class CellTopologyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup if needed
    }

    void TearDown() override {
        // Cleanup if needed
    }
};

// ==========================================
// Tests: Tetrahedron Topology
// ==========================================

TEST_F(CellTopologyTest, TetHasFourFaces) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Tetra);

    EXPECT_EQ(faces.size(), 4);
}

TEST_F(CellTopologyTest, TetFacesAreTriangular) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Tetra);

    for (const auto& face : faces) {
        EXPECT_EQ(face.size(), 3);  // Each face has 3 vertices
    }
}

TEST_F(CellTopologyTest, TetOrientedFacesMatchCanonicalCount) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Tetra);
    auto oriented_faces = CellTopology::get_oriented_boundary_faces(CellFamily::Tetra);

    EXPECT_EQ(faces.size(), oriented_faces.size());
}

TEST_F(CellTopologyTest, TetHasSixEdges) {
    auto edges = CellTopology::get_edges(CellFamily::Tetra);

    EXPECT_EQ(edges.size(), 6);
}

TEST_F(CellTopologyTest, TetEdgesHaveTwoVertices) {
    auto edges = CellTopology::get_edges(CellFamily::Tetra);

    for (const auto& edge : edges) {
        EXPECT_EQ(edge.size(), 2);  // Each edge has 2 vertices
    }
}

// ==========================================
// Tests: Hexahedron Topology
// ==========================================

TEST_F(CellTopologyTest, HexHasSixFaces) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Hex);

    EXPECT_EQ(faces.size(), 6);
}

TEST_F(CellTopologyTest, HexFacesAreQuadrilateral) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Hex);

    for (const auto& face : faces) {
        EXPECT_EQ(face.size(), 4);  // Each face has 4 vertices
    }
}

TEST_F(CellTopologyTest, HexHasTwelveEdges) {
    auto edges = CellTopology::get_edges(CellFamily::Hex);

    EXPECT_EQ(edges.size(), 12);
}

// ==========================================
// Tests: Wedge (Prism) Topology
// ==========================================

TEST_F(CellTopologyTest, WedgeHasFiveFaces) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Wedge);

    EXPECT_EQ(faces.size(), 5);  // 2 triangular + 3 quadrilateral
}

TEST_F(CellTopologyTest, WedgeHasNineEdges) {
    auto edges = CellTopology::get_edges(CellFamily::Wedge);

    EXPECT_EQ(edges.size(), 9);
}

// ==========================================
// Tests: Pyramid Topology
// ==========================================

TEST_F(CellTopologyTest, PyramidHasFiveFaces) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Pyramid);

    EXPECT_EQ(faces.size(), 5);  // 1 quad base + 4 triangular sides
}

TEST_F(CellTopologyTest, PyramidHasEightEdges) {
    auto edges = CellTopology::get_edges(CellFamily::Pyramid);

    EXPECT_EQ(edges.size(), 8);
}

// ==========================================
// Tests: Triangle Topology (2D)
// ==========================================

TEST_F(CellTopologyTest, TriangleHasThreeEdges) {
    auto edges = CellTopology::get_boundary_faces(CellFamily::Triangle);

    // In 2D, boundary faces are edges
    EXPECT_EQ(edges.size(), 3);
}

TEST_F(CellTopologyTest, TriangleEdgesHaveTwoVertices) {
    auto edges = CellTopology::get_boundary_faces(CellFamily::Triangle);

    for (const auto& edge : edges) {
        EXPECT_EQ(edge.size(), 2);
    }
}

// ==========================================
// Tests: Quadrilateral Topology (2D)
// ==========================================

TEST_F(CellTopologyTest, QuadHasFourEdges) {
    auto edges = CellTopology::get_boundary_faces(CellFamily::Quad);

    // In 2D, boundary faces are edges
    EXPECT_EQ(edges.size(), 4);
}

TEST_F(CellTopologyTest, QuadEdgesHaveTwoVertices) {
    auto edges = CellTopology::get_boundary_faces(CellFamily::Quad);

    for (const auto& edge : edges) {
        EXPECT_EQ(edge.size(), 2);
    }
}

// ==========================================
// Tests: Canonical vs Oriented Faces
// ==========================================

TEST_F(CellTopologyTest, CanonicalAndOrientedHaveSameVertices) {
    auto canonical = CellTopology::get_boundary_faces(CellFamily::Tetra);
    auto oriented = CellTopology::get_oriented_boundary_faces(CellFamily::Tetra);

    ASSERT_EQ(canonical.size(), oriented.size());

    for (size_t i = 0; i < canonical.size(); ++i) {
        EXPECT_EQ(canonical[i].size(), oriented[i].size());

        // Vertices should be the same, just potentially in different order
        std::vector<index_t> canon_sorted = canonical[i];
        std::vector<index_t> orient_sorted = oriented[i];

        std::sort(canon_sorted.begin(), canon_sorted.end());
        std::sort(orient_sorted.begin(), orient_sorted.end());

        EXPECT_EQ(canon_sorted, orient_sorted);
    }
}

// ==========================================
// Tests: Face Vertex Indices Are Valid
// ==========================================

TEST_F(CellTopologyTest, TetFaceIndicesAreValid) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Tetra);

    // Tet has 4 vertices (indices 0-3)
    for (const auto& face : faces) {
        for (index_t idx : face) {
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, 4);
        }
    }
}

TEST_F(CellTopologyTest, HexFaceIndicesAreValid) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Hex);

    // Hex has 8 vertices (indices 0-7)
    for (const auto& face : faces) {
        for (index_t idx : face) {
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, 8);
        }
    }
}

// ==========================================
// Tests: Edge Vertex Indices Are Valid
// ==========================================

TEST_F(CellTopologyTest, TetEdgeIndicesAreValid) {
    auto edges = CellTopology::get_edges(CellFamily::Tetra);

    // Tet has 4 vertices (indices 0-3)
    for (const auto& edge : edges) {
        for (index_t idx : edge) {
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, 4);
        }
    }
}

TEST_F(CellTopologyTest, HexEdgeIndicesAreValid) {
    auto edges = CellTopology::get_edges(CellFamily::Hex);

    // Hex has 8 vertices (indices 0-7)
    for (const auto& edge : edges) {
        for (index_t idx : edge) {
            EXPECT_GE(idx, 0);
            EXPECT_LT(idx, 8);
        }
    }
}

// ==========================================
// Tests: No Duplicate Faces
// ==========================================

TEST_F(CellTopologyTest, TetHasNoDuplicateFaces) {
    auto faces = CellTopology::get_boundary_faces(CellFamily::Tetra);

    // Create sorted versions for comparison
    std::vector<std::vector<index_t>> sorted_faces;
    for (auto face : faces) {
        std::sort(face.begin(), face.end());
        sorted_faces.push_back(face);
    }

    // Check for duplicates
    for (size_t i = 0; i < sorted_faces.size(); ++i) {
        for (size_t j = i + 1; j < sorted_faces.size(); ++j) {
            EXPECT_NE(sorted_faces[i], sorted_faces[j]);
        }
    }
}

// ==========================================
// Tests: No Duplicate Edges
// ==========================================

TEST_F(CellTopologyTest, TetHasNoDuplicateEdges) {
    auto edges = CellTopology::get_edges(CellFamily::Tetra);

    // Create sorted versions for comparison
    std::vector<std::array<index_t, 2>> sorted_edges;
    for (const auto& edge : edges) {
        std::array<index_t, 2> sorted_edge = {{edge[0], edge[1]}};
        if (sorted_edge[0] > sorted_edge[1]) {
            std::swap(sorted_edge[0], sorted_edge[1]);
        }
        sorted_edges.push_back(sorted_edge);
    }

    // Check for duplicates
    for (size_t i = 0; i < sorted_edges.size(); ++i) {
        for (size_t j = i + 1; j < sorted_edges.size(); ++j) {
            EXPECT_NE(sorted_edges[i], sorted_edges[j]);
        }
    }
}

// ==========================================
// Tests: All Cell Types Supported
// ==========================================

TEST_F(CellTopologyTest, AllCellFamiliesHaveFaceDefinitions) {
    // Test that all cell families return non-empty face definitions
    std::vector<CellFamily> families = {
        CellFamily::Tetra,
        CellFamily::Hex,
        CellFamily::Wedge,
        CellFamily::Pyramid,
        CellFamily::Triangle,
        CellFamily::Quad
    };

    for (auto family : families) {
        auto faces = CellTopology::get_boundary_faces(family);
        EXPECT_GT(faces.size(), 0) << "Cell family should have face definitions";
    }
}

TEST_F(CellTopologyTest, AllCellFamiliesHaveOrientedFaceDefinitions) {
    std::vector<CellFamily> families = {
        CellFamily::Tetra,
        CellFamily::Hex,
        CellFamily::Wedge,
        CellFamily::Pyramid,
        CellFamily::Triangle,
        CellFamily::Quad
    };

    for (auto family : families) {
        auto oriented_faces = CellTopology::get_oriented_boundary_faces(family);
        EXPECT_GT(oriented_faces.size(), 0) << "Cell family should have oriented face definitions";
    }
}

} // namespace test
} // namespace svmp
