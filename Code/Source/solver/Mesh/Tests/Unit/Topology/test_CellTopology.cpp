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
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

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
    auto oriented  = CellTopology::get_oriented_boundary_faces(CellFamily::Tetra);
    ASSERT_EQ(canonical.size(), oriented.size());
    // Compare as multisets of faces (ignore face ordering and within-face order)
    auto normalize_faces = [](const std::vector<std::vector<index_t>>& faces) {
        std::vector<std::vector<index_t>> norm = faces;
        for (auto& f : norm) std::sort(f.begin(), f.end());
        std::sort(norm.begin(), norm.end());
        return norm;
    };
    auto cset = normalize_faces(canonical);
    auto oset = normalize_faces(oriented);
    EXPECT_EQ(cset, oset);
}

// Check that canonical vs oriented faces match (as sets) across families
TEST_F(CellTopologyTest, CanonicalVsOrientedMatchForAllFixedFamilies) {
    std::vector<CellFamily> families = {
        CellFamily::Triangle,
        CellFamily::Quad,
        CellFamily::Tetra,
        CellFamily::Hex,
        CellFamily::Wedge,
        CellFamily::Pyramid
    };

    for (auto family : families) {
        auto canonical = CellTopology::get_boundary_faces(family);
        auto oriented  = CellTopology::get_oriented_boundary_faces(family);
        ASSERT_EQ(canonical.size(), oriented.size());
        auto norm = [](const std::vector<std::vector<index_t>>& faces) {
            std::vector<std::vector<index_t>> v = faces;
            for (auto& f : v) std::sort(f.begin(), f.end());
            std::sort(v.begin(), v.end());
            return v;
        };
        EXPECT_EQ(norm(canonical), norm(oriented));
    }
}

// ==========================================
// Tests: Canonical views are sorted within-face
// ==========================================

TEST_F(CellTopologyTest, CanonicalViewFacesAreSorted) {
    std::vector<CellFamily> families = {
        CellFamily::Triangle, CellFamily::Quad,
        CellFamily::Tetra,    CellFamily::Hex,
        CellFamily::Wedge,    CellFamily::Pyramid
    };
    for (auto family : families) {
        auto view = CellTopology::get_boundary_faces_canonical_view(family);
        ASSERT_TRUE(view.indices != nullptr);
        ASSERT_TRUE(view.offsets != nullptr);
        for (int f = 0; f < view.face_count; ++f) {
            int b = view.offsets[f], e = view.offsets[f+1];
            for (int i = b + 1; i < e; ++i) {
                EXPECT_LE(view.indices[i-1], view.indices[i]);
            }
        }
    }
}

// ==========================================
// Tests: Oriented faces traverse shared edges in opposite directions (3D)
// ==========================================

static void expect_edge_orientations_cancel(const std::vector<std::vector<index_t>>& faces) {
    // For each undirected edge {a,b}, the signed count across oriented faces should be zero.
    std::unordered_map<long long, int> sum_counts;
    auto ukey = [](index_t a, index_t b) {
        index_t lo = std::min(a,b);
        index_t hi = std::max(a,b);
        return (static_cast<long long>(lo) << 32) | static_cast<unsigned long long>(static_cast<uint32_t>(hi));
    };
    for (const auto& f : faces) {
        const size_t k = f.size();
        for (size_t i = 0; i < k; ++i) {
            index_t u = f[i];
            index_t v = f[(i+1) % k];
            long long key = ukey(u,v);
            // add +1 if direction matches (lo->hi), -1 otherwise
            sum_counts[key] += (u < v) ? +1 : -1;
        }
    }
    for (const auto& p : sum_counts) {
        EXPECT_EQ(p.second, 0) << "edge sum nonzero for key=" << p.first;
    }
}

TEST_F(CellTopologyTest, OrientedFacesOppositeEdgeTraversal_TetHexWedgePyr) {
    // Only 3D families
    std::vector<CellFamily> families = {
        CellFamily::Tetra, CellFamily::Hex, CellFamily::Wedge, CellFamily::Pyramid
    };
    for (auto family : families) {
        auto faces = CellTopology::get_oriented_boundary_faces(family);
        expect_edge_orientations_cancel(faces);
    }
}

// ==========================================
// Tests: View integrity (sizes and offsets)
// ==========================================

TEST_F(CellTopologyTest, ViewsHaveConsistentOffsetsAndSizes) {
    auto check_faces = [](const CellTopology::FaceListView& v) {
        if (!v.indices) return; // skip empties
        ASSERT_TRUE(v.offsets != nullptr);
        ASSERT_GE(v.face_count, 1);
        // last offset equals total indices length
        int total = v.offsets[v.face_count];
        ASSERT_GT(total, 0);
        // offsets must be non-decreasing
        for (int i = 1; i <= v.face_count; ++i) {
            EXPECT_GE(v.offsets[i], v.offsets[i-1]);
        }
    };
    auto check_edges = [](const CellTopology::EdgeListView& v) {
        if (!v.pairs_flat) return;
        ASSERT_GE(v.edge_count, 1);
    };

    // Fixed families
    std::vector<CellFamily> families = {
        CellFamily::Triangle, CellFamily::Quad,
        CellFamily::Tetra,    CellFamily::Hex,
        CellFamily::Wedge,    CellFamily::Pyramid
    };
    for (auto f : families) {
        check_faces(CellTopology::get_oriented_boundary_faces_view(f));
        check_faces(CellTopology::get_boundary_faces_canonical_view(f));
        check_edges(CellTopology::get_edges_view(f));
    }
}

// ==========================================
// Tests: Variable-arity families (polygon, prism(m), pyramid(m))
// ==========================================

TEST_F(CellTopologyTest, PolygonViews_M5) {
    int m = 5;
    auto fv = CellTopology::get_polygon_faces_view(m);
    auto fcv= CellTopology::get_polygon_faces_canonical_view(m);
    auto ev = CellTopology::get_polygon_edges_view(m);
    ASSERT_EQ(fv.face_count, m);
    ASSERT_EQ(ev.edge_count, m);
    // canonical pairs are sorted within each edge (a <= b)
    for (int i = 0; i < m; ++i) {
        int b = fcv.offsets[i], e = fcv.offsets[i+1];
        ASSERT_EQ(e - b, 2);
        EXPECT_LE(fcv.indices[b+0], fcv.indices[b+1]);
    }
}

TEST_F(CellTopologyTest, PrismViews_M5_CountsAndOrientation) {
    int m = 5;
    auto fv = CellTopology::get_prism_faces_view(m);
    auto ev = CellTopology::get_prism_edges_view(m);
    ASSERT_EQ(fv.face_count, 2 + m);
    ASSERT_EQ(ev.edge_count, 3 * m);
    // Build materialized faces and check oriented edge cancellation
    std::vector<std::vector<index_t>> faces;
    faces.reserve(fv.face_count);
    for (int f = 0; f < fv.face_count; ++f) {
        int b = fv.offsets[f], e = fv.offsets[f+1];
        faces.emplace_back(fv.indices + b, fv.indices + e);
    }
    expect_edge_orientations_cancel(faces);
}

TEST_F(CellTopologyTest, PyramidViews_M5_CountsAndOrientation) {
    int m = 5;
    auto fv = CellTopology::get_pyramid_faces_view(m);
    auto ev = CellTopology::get_pyramid_edges_view(m);
    ASSERT_EQ(fv.face_count, 1 + m);
    ASSERT_EQ(ev.edge_count, m + m);
    std::vector<std::vector<index_t>> faces;
    for (int f = 0; f < fv.face_count; ++f) {
        int b = fv.offsets[f], e = fv.offsets[f+1];
        faces.emplace_back(fv.indices + b, fv.indices + e);
    }
    expect_edge_orientations_cancel(faces);
}

// ==========================================
// Tests: Higher-order assembly helpers
// ==========================================

TEST_F(CellTopologyTest, HighOrderAssembly_Tri6_Tet10) {
    // Tri6: 3 corners + 3 edge mids
    std::vector<index_t> tri_c = {0,1,2};
    std::vector<index_t> tri_e = {3,4,5};
    auto tri_order = CellTopology::HighOrderOrdering::assemble_quadratic_polygon(3, tri_c, tri_e);
    std::vector<index_t> tri_expected = {0,1,2,3,4,5};
    EXPECT_EQ(tri_order, tri_expected);

    // Tet10: 4 corners + 6 edge mids
    std::vector<index_t> tet_c = {0,1,2,3};
    std::vector<index_t> tet_e = {4,5,6,7,8,9};
    auto tet_order = CellTopology::HighOrderOrdering::assemble_quadratic(CellFamily::Tetra, tet_c, tet_e);
    std::vector<index_t> tet_expected = {0,1,2,3,4,5,6,7,8,9};
    EXPECT_EQ(tet_order, tet_expected);
}

TEST_F(CellTopologyTest, HighOrderAssembly_SizeValidation) {
    // Mismatched counts should throw
    std::vector<index_t> c = {0,1,2,3};
    std::vector<index_t> e_bad = {4,5}; // wrong size for tet
    EXPECT_THROW(
        (void)CellTopology::HighOrderOrdering::assemble_quadratic(CellFamily::Tetra, c, e_bad),
        std::invalid_argument);
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
