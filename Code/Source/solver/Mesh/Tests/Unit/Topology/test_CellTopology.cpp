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

// ==========================================
// Tests: High-order inference (Lagrange/Serendipity)
// ==========================================

TEST_F(CellTopologyTest, InferLagrangeOrder_Wedge) {
    // n = (p+1)*(p+1)*(p+2)/2
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Wedge, 6), 1);
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Wedge, 18), 2);
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Wedge, 40), 3);
}

TEST_F(CellTopologyTest, InferLagrangeOrder_Pyramid) {
    // Quadratic classic (13) and Lagrange (14)
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Pyramid, 13), 2);
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Pyramid, 14), 2);
    // Higher orders via Lagrange formula: N = sum_{m=1}^{p+1} m^2
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Pyramid, 30), 3); // p=3
    EXPECT_EQ(CellTopology::infer_lagrange_order(CellFamily::Pyramid, 55), 4); // p=4
}

TEST_F(CellTopologyTest, InferSerendipityOrder_Quad_CommonSequence) {
    EXPECT_EQ(CellTopology::infer_serendipity_order(CellFamily::Quad, 8), 2);
    EXPECT_EQ(CellTopology::infer_serendipity_order(CellFamily::Quad, 12), 3);
    EXPECT_EQ(CellTopology::infer_serendipity_order(CellFamily::Quad, 16), 4);
    EXPECT_EQ(CellTopology::infer_serendipity_order(CellFamily::Quad, 20), 5);
    EXPECT_EQ(CellTopology::infer_serendipity_order(CellFamily::Quad, 10), -1);
}

// ==========================================
// Tests: Pyramid high-order pattern counts (p>=3)
// ==========================================

TEST_F(CellTopologyTest, PyramidHighOrderPattern_VolumeNodeCounts) {
    for (int p : {3,4,5}) {
        auto pat = CellTopology::high_order_pattern(CellFamily::Pyramid, p, CellTopology::HighOrderKind::Lagrange);
        size_t corners = 5;
        size_t edges = 8 * (p - 1);
        size_t faces_tri = 4 * ((p - 1) * (p - 2) / 2);
        size_t face_quad = (p - 1) * (p - 1);
        size_t vol = (p >= 3) ? (size_t)((p - 2) * (p - 1) * (2*p - 3) / 6) : 0;
        size_t expected = corners + edges + faces_tri + face_quad + vol;
        EXPECT_EQ(pat.sequence.size(), expected);
    }
}

// ==========================================
// Tests: Prism(m=3) oriented faces match Wedge oriented faces
// ==========================================

TEST_F(CellTopologyTest, PrismM3_OrientedFacesMatchWedge) {
    int m = 3;
    auto pfv = CellTopology::get_prism_faces_view(m);
    ASSERT_EQ(pfv.face_count, 5);
    std::vector<std::vector<index_t>> prism_faces;
    for (int f=0; f<pfv.face_count; ++f) {
        int b = pfv.offsets[f], e = pfv.offsets[f+1];
        prism_faces.emplace_back(pfv.indices + b, pfv.indices + e);
    }
    auto wedge_faces = CellTopology::get_oriented_boundary_faces(CellFamily::Wedge);
    EXPECT_EQ(prism_faces, wedge_faces);
}

// ==========================================
// Tests: Face->Edge mapping matches derived mapping from faces+edges
// ==========================================

static std::vector<std::vector<int>> derive_face_edges(CellFamily family) {
    auto fview = CellTopology::get_oriented_boundary_faces_view(family);
    if (!(fview.indices && fview.offsets && fview.face_count > 0)) {
        fview = CellTopology::get_boundary_faces_canonical_view(family);
    }
    auto eview = CellTopology::get_edges_view(family);
    auto pack = [](index_t a, index_t b) -> long long {
        index_t lo = std::min(a,b);
        index_t hi = std::max(a,b);
        return (static_cast<long long>(lo) << 32) | static_cast<unsigned long long>(static_cast<uint32_t>(hi));
    };
    std::unordered_map<long long,int> edge_map;
    for (int ei=0; ei<eview.edge_count; ++ei) {
        index_t a = eview.pairs_flat[2*ei+0];
        index_t b = eview.pairs_flat[2*ei+1];
        edge_map.emplace(pack(a,b), ei);
    }
    std::vector<std::vector<int>> out(static_cast<size_t>(fview.face_count));
    for (int fi=0; fi<fview.face_count; ++fi) {
        int b = fview.offsets[fi], e = fview.offsets[fi+1];
        int fv = e-b;
        if (fv == 2) {
            index_t u = fview.indices[b+0];
            index_t v = fview.indices[b+1];
            out[fi] = { edge_map.at(pack(u,v)) };
        } else {
            std::vector<int> edges; edges.reserve(fv);
            for (int k=0;k<fv;++k) {
                index_t u = fview.indices[b+k];
                index_t v = fview.indices[b+((k+1)%fv)];
                edges.push_back(edge_map.at(pack(u,v)));
            }
            out[fi] = std::move(edges);
        }
    }
    return out;
}

TEST_F(CellTopologyTest, FaceEdges_MatchDerivedMapping_AllFixedFamilies) {
    std::vector<CellFamily> families = {
        CellFamily::Triangle,
        CellFamily::Quad,
        CellFamily::Tetra,
        CellFamily::Hex,
        CellFamily::Wedge,
        CellFamily::Pyramid
    };
    for (auto fam : families) {
        auto derived = derive_face_edges(fam);
        auto mapped  = CellTopology::get_face_edges(fam);
        ASSERT_EQ(derived.size(), mapped.size());
        EXPECT_EQ(derived, mapped);
    }
}

// ==========================================
// Tests: Canonical views sorted for prism/pyramid
// ==========================================

TEST_F(CellTopologyTest, PrismCanonicalViewFacesAreSorted) {
    int m = 6;
    auto view = CellTopology::get_prism_faces_canonical_view(m);
    ASSERT_TRUE(view.indices != nullptr);
    ASSERT_TRUE(view.offsets != nullptr);
    for (int f=0; f<view.face_count; ++f) {
        int b = view.offsets[f], e = view.offsets[f+1];
        for (int i=b+1;i<e;++i) EXPECT_LE(view.indices[i-1], view.indices[i]);
    }
}

TEST_F(CellTopologyTest, PyramidCanonicalViewFacesAreSorted) {
    int m = 5;
    auto view = CellTopology::get_pyramid_faces_canonical_view(m);
    ASSERT_TRUE(view.indices != nullptr);
    ASSERT_TRUE(view.offsets != nullptr);
    for (int f=0; f<view.face_count; ++f) {
        int b = view.offsets[f], e = view.offsets[f+1];
        for (int i=b+1;i<e;++i) EXPECT_LE(view.indices[i-1], view.indices[i]);
    }
}

// ==========================================
// Tests: Invalid m for variable-arity views returns empty
// ==========================================

TEST_F(CellTopologyTest, VariableArity_InvalidM_ReturnsEmptyViews) {
    auto pfv = CellTopology::get_polygon_faces_view(2);
    auto pfe = CellTopology::get_polygon_edges_view(2);
    EXPECT_EQ(pfv.indices, nullptr);
    EXPECT_EQ(pfv.face_count, 0);
    EXPECT_EQ(pfe.pairs_flat, nullptr);
    EXPECT_EQ(pfe.edge_count, 0);

    auto qfv = CellTopology::get_prism_faces_view(2);
    auto qfe = CellTopology::get_prism_edges_view(2);
    EXPECT_EQ(qfv.indices, nullptr);
    EXPECT_EQ(qfv.face_count, 0);
    EXPECT_EQ(qfe.pairs_flat, nullptr);
    EXPECT_EQ(qfe.edge_count, 0);

    auto yfv = CellTopology::get_pyramid_faces_view(2);
    auto yfe = CellTopology::get_pyramid_edges_view(2);
    EXPECT_EQ(yfv.indices, nullptr);
    EXPECT_EQ(yfv.face_count, 0);
    EXPECT_EQ(yfe.pairs_flat, nullptr);
    EXPECT_EQ(yfe.edge_count, 0);
}


// ==========================================
// Tests: High-order pattern counts for Tetra and Wedge (VTK Lagrange)
// ==========================================

TEST_F(CellTopologyTest, TetraHighOrderPattern_VolumeNodeCounts) {
    for (int p : {3,4,5}) {
        auto pat = CellTopology::high_order_pattern(CellFamily::Tetra, p, CellTopology::HighOrderKind::Lagrange);
        size_t corners = 4;
        size_t edges = 6 * (p - 1);
        size_t faces = 4 * ((p - 1) * (p - 2) / 2);
        size_t vol = (size_t)((p - 1) * (p - 2) * (p - 3) / 6);
        size_t expected = corners + edges + faces + vol;
        EXPECT_EQ(pat.sequence.size(), expected);
    }
}

TEST_F(CellTopologyTest, WedgeHighOrderPattern_VolumeNodeCounts) {
    for (int p : {3,4,5}) {
        auto pat = CellTopology::high_order_pattern(CellFamily::Wedge, p, CellTopology::HighOrderKind::Lagrange);
        size_t corners = 6;
        size_t edges = 9 * (p - 1);
        size_t tri_face = (p - 1) * (p - 2) / 2;  // per triangular face
        size_t quad_face = (p - 1) * (p - 1);     // per quadrilateral face
        size_t faces = 2 * tri_face + 3 * quad_face;
        size_t vol = (size_t)((p - 1) * tri_face); // (p-1) layers of triangular interior
        size_t expected = corners + edges + faces + vol;
        EXPECT_EQ(pat.sequence.size(), expected);
    }
}
} // namespace test
} // namespace svmp

namespace svmp { namespace test {

// ==========================================
// Tests: Oriented faces and edges views (exact sequences)
// ==========================================

static void expect_oriented_faces_exact(CellFamily fam,
                                        const std::vector<int>& expected_off,
                                        const std::vector<index_t>& expected_idx) {
    auto view = CellTopology::get_oriented_boundary_faces_view(fam);
    ASSERT_TRUE(view.indices != nullptr);
    ASSERT_EQ(view.face_count + 1, (int)expected_off.size());
    for (int i=0;i<=view.face_count;++i) ASSERT_EQ(view.offsets[i], expected_off[i]);
    int total = view.offsets[view.face_count];
    ASSERT_EQ(total, (int)expected_idx.size());
    for (int i=0;i<total;++i) ASSERT_EQ(view.indices[i], expected_idx[i]);
}

static void expect_edges_exact(CellFamily fam, const std::vector<index_t>& expected_pairs_flat) {
    auto v = CellTopology::get_edges_view(fam);
    ASSERT_TRUE(v.pairs_flat != nullptr);
    ASSERT_EQ(2*v.edge_count, (int)expected_pairs_flat.size());
    for (int i=0;i<2*v.edge_count;++i) ASSERT_EQ(v.pairs_flat[i], expected_pairs_flat[i]);
}

TEST_F(CellTopologyTest, OrientedFacesExact_FixedFamilies) {
    // Tetra
    expect_oriented_faces_exact(CellFamily::Tetra,
        {0,3,6,9,12},
        {1,2,3, 0,3,2, 0,1,3, 0,2,1});
    // Hex
    expect_oriented_faces_exact(CellFamily::Hex,
        {0,4,8,12,16,20,24},
        {0,3,2,1,  4,5,6,7,  0,1,5,4,  1,2,6,5,  2,3,7,6,  3,0,4,7});
    // Triangle (faces are edges)
    expect_oriented_faces_exact(CellFamily::Triangle,
        {0,2,4,6},
        {0,1, 1,2, 2,0});
    // Quad (faces are edges)
    expect_oriented_faces_exact(CellFamily::Quad,
        {0,2,4,6,8},
        {0,1, 1,2, 2,3, 3,0});
    // Wedge
    expect_oriented_faces_exact(CellFamily::Wedge,
        {0,3,6,10,14,18},
        {0,2,1, 3,4,5, 0,1,4,3, 1,2,5,4, 2,0,3,5});
    // Pyramid
    expect_oriented_faces_exact(CellFamily::Pyramid,
        {0,4,7,10,13,16},
        {0,1,2,3, 0,4,1, 1,4,2, 2,4,3, 3,4,0});
}

TEST_F(CellTopologyTest, EdgesViewExact_FixedFamilies) {
    // Tetra
    expect_edges_exact(CellFamily::Tetra, {0,1, 0,2, 0,3, 1,2, 1,3, 2,3});
    // Hex
    expect_edges_exact(CellFamily::Hex, {0,1, 1,2, 2,3, 3,0, 4,5, 5,6, 6,7, 7,4, 0,4, 1,5, 2,6, 3,7});
    // Triangle
    expect_edges_exact(CellFamily::Triangle, {0,1, 1,2, 2,0});
    // Quad
    expect_edges_exact(CellFamily::Quad, {0,1, 1,2, 2,3, 3,0});
    // Wedge
    expect_edges_exact(CellFamily::Wedge, {0,1, 1,2, 2,0, 3,4, 4,5, 5,3, 0,3, 1,4, 2,5});
    // Pyramid
    expect_edges_exact(CellFamily::Pyramid, {0,1, 1,2, 2,3, 3,0, 0,4, 1,4, 2,4, 3,4});
}

// ==========================================
// Tests: High-order role sequence basics (p=3) for Tetra/Hex
// ==========================================

static void expect_role_sequence_basics(CellFamily fam, int p, int corners, int edges, int tri_faces, int quad_faces, int vols) {
    auto pat = CellTopology::high_order_pattern(fam, p, CellTopology::HighOrderKind::Lagrange);
    // Check grouping: corners first
    ASSERT_GE((int)pat.sequence.size(), corners);
    for (int i=0;i<corners;++i) ASSERT_EQ(pat.sequence[i].role, CellTopology::HONodeRole::Corner);
    int pos = corners;
    // Edges
    for (int i=0;i<edges;++i) ASSERT_EQ(pat.sequence[pos++].role, CellTopology::HONodeRole::Edge);
    // Faces (tri + quad)
    for (int i=0;i<tri_faces+quad_faces;++i) ASSERT_EQ(pat.sequence[pos++].role, CellTopology::HONodeRole::Face);
    // Volumes
    for (int i=0;i<vols;++i) ASSERT_EQ(pat.sequence[pos++].role, CellTopology::HONodeRole::Volume);
    ASSERT_EQ(pos, (int)pat.sequence.size());
}

TEST_F(CellTopologyTest, HighOrderPattern_RoleSequence_p3_Tetra_Hex) {
    // Tetra p=3: corners=4, edges=6*(p-1)=12, faces=4*((p-1)(p-2)/2)=4, vols=0
    expect_role_sequence_basics(CellFamily::Tetra, 3, 4, 12, 4, 0, 0);
    // Hex p=3: corners=8, edges=12*(p-1)=24, faces=6*((p-1)^2)=24, vols=(p-1)^3=8
    expect_role_sequence_basics(CellFamily::Hex, 3, 8, 24, 0, 24, 8);
}


// ==========================================
// Tests: Face list order for Hex/Wedge/Pyramid
// ==========================================

TEST_F(CellTopologyTest, FaceListOrder_Hex_Wedge_Pyramid) {
    auto hex = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
    ASSERT_EQ(hex.face_count, 6);
    // bottom
    ASSERT_EQ(hex.indices[hex.offsets[0]+0], 0);
    ASSERT_EQ(hex.indices[hex.offsets[0]+1], 3);
    ASSERT_EQ(hex.indices[hex.offsets[0]+2], 2);
    ASSERT_EQ(hex.indices[hex.offsets[0]+3], 1);
    // top
    ASSERT_EQ(hex.indices[hex.offsets[1]+0], 4);
    ASSERT_EQ(hex.indices[hex.offsets[1]+1], 5);
    ASSERT_EQ(hex.indices[hex.offsets[1]+2], 6);
    ASSERT_EQ(hex.indices[hex.offsets[1]+3], 7);

    auto wed = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
    ASSERT_EQ(wed.face_count, 5);
    // bottom tri then top tri
    ASSERT_EQ(wed.indices[wed.offsets[0]+0], 0);
    ASSERT_EQ(wed.indices[wed.offsets[0]+1], 2);
    ASSERT_EQ(wed.indices[wed.offsets[0]+2], 1);
    ASSERT_EQ(wed.indices[wed.offsets[1]+0], 3);
    ASSERT_EQ(wed.indices[wed.offsets[1]+1], 4);
    ASSERT_EQ(wed.indices[wed.offsets[1]+2], 5);

    auto pyr = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
    ASSERT_EQ(pyr.face_count, 5);
    // base quad first
    ASSERT_EQ(pyr.indices[pyr.offsets[0]+0], 0);
    ASSERT_EQ(pyr.indices[pyr.offsets[0]+1], 1);
    ASSERT_EQ(pyr.indices[pyr.offsets[0]+2], 2);
    ASSERT_EQ(pyr.indices[pyr.offsets[0]+3], 3);
}

// ==========================================
// Tests: 2D CCW orientation explicit checks
// ==========================================

TEST_F(CellTopologyTest, Orientation2D_CCW_Tri_Quad) {
    auto tri = CellTopology::get_oriented_boundary_faces_view(CellFamily::Triangle);
    ASSERT_EQ(tri.face_count, 3);
    // Expect edges (0,1), (1,2), (2,0)
    ASSERT_EQ(tri.indices[tri.offsets[0]+0], 0);
    ASSERT_EQ(tri.indices[tri.offsets[0]+1], 1);
    ASSERT_EQ(tri.indices[tri.offsets[1]+0], 1);
    ASSERT_EQ(tri.indices[tri.offsets[1]+1], 2);
    ASSERT_EQ(tri.indices[tri.offsets[2]+0], 2);
    ASSERT_EQ(tri.indices[tri.offsets[2]+1], 0);

    auto quad = CellTopology::get_oriented_boundary_faces_view(CellFamily::Quad);
    ASSERT_EQ(quad.face_count, 4);
    // Expect edges (0,1), (1,2), (2,3), (3,0)
    for (int e=0;e<4;++e) {
        int a = quad.indices[quad.offsets[e]+0];
        int b = quad.indices[quad.offsets[e]+1];
        if (e == 0) { ASSERT_EQ(a,0); ASSERT_EQ(b,1); }
        if (e == 1) { ASSERT_EQ(a,1); ASSERT_EQ(b,2); }
        if (e == 2) { ASSERT_EQ(a,2); ASSERT_EQ(b,3); }
        if (e == 3) { ASSERT_EQ(a,3); ASSERT_EQ(b,0); }
    }
}

// ==========================================
// Tests: Face→edge loop order matches face vertex loops (fixed families)
// ==========================================

TEST_F(CellTopologyTest, FaceEdges_LoopOrder_MatchesFaceVertices) {
    auto families = {CellFamily::Triangle, CellFamily::Quad, CellFamily::Tetra, CellFamily::Hex, CellFamily::Wedge, CellFamily::Pyramid};
    for (auto fam : families) {
        auto faces = CellTopology::get_oriented_boundary_faces_view(fam);
        auto edges = CellTopology::get_edges_view(fam);
        auto f2e = CellTopology::get_face_edges(fam);
        // build a lookup from edge index to undirected pair
        auto pack = [](index_t a, index_t b){index_t lo=std::min(a,b), hi=std::max(a,b); return (static_cast<long long>(lo)<<32) | static_cast<unsigned long long>((uint32_t)hi);};
        std::vector<long long> idx2pair( (size_t)edges.edge_count );
        for (int ei=0; ei<edges.edge_count; ++ei) idx2pair[ei] = pack(edges.pairs_flat[2*ei], edges.pairs_flat[2*ei+1]);
        for (int fi=0; fi<faces.face_count; ++fi) {
            int b = faces.offsets[fi], e = faces.offsets[fi+1];
            int kcnt = e-b;
            // 2D faces are edges: mapping has a single edge per face
            if (kcnt == 2) {
                ASSERT_EQ(f2e[fi].size(), 1u);
                index_t u = faces.indices[b+0];
                index_t v = faces.indices[b+1];
                long long want = pack(u,v);
                ASSERT_EQ(idx2pair[f2e[fi][0]], want);
                continue;
            }
            // 3D faces: loop over sides
            for (int k=0;k<kcnt;++k) {
                index_t u = faces.indices[b+k];
                index_t v = faces.indices[b+((k+1)%kcnt)];
                long long want = pack(u,v);
                ASSERT_EQ(idx2pair[ f2e[fi][k] ], want);
            }
        }
    }
}

// ==========================================
// Tests: High‑order per‑face interior ordering
// ==========================================

// Helper to extract face roles in sequence for a given face id
static std::vector<std::pair<int,int>> face_interior_roles(const CellTopology::HighOrderPattern& pat, int face_id) {
    std::vector<std::pair<int,int>> out;
    for (const auto& r : pat.sequence) if (r.role == CellTopology::HONodeRole::Face && r.idx0 == face_id) out.emplace_back(r.idx1, r.idx2);
    return out;
}

TEST_F(CellTopologyTest, HighOrder_PerFaceOrdering_Tetra_p4) {
    int p = 4; // tri faces
    auto pat = CellTopology::high_order_pattern(CellFamily::Tetra, p, CellTopology::HighOrderKind::Lagrange);
    auto faces = CellTopology::get_oriented_boundary_faces_view(CellFamily::Tetra);
    // Expected lexicographic (i,j): (1,1),(1,2),(2,1)
    std::vector<std::pair<int,int>> expected = {{1,1},{1,2},{2,1}};
    for (int fi=0; fi<faces.face_count; ++fi) {
        auto got = face_interior_roles(pat, fi);
        ASSERT_EQ(got, expected);
    }
}

TEST_F(CellTopologyTest, HighOrder_PerFaceOrdering_Hex_p3) {
    int p = 3; // quad faces row‑major i=1..2, j=1..2
    auto pat = CellTopology::high_order_pattern(CellFamily::Hex, p, CellTopology::HighOrderKind::Lagrange);
    auto faces = CellTopology::get_oriented_boundary_faces_view(CellFamily::Hex);
    std::vector<std::pair<int,int>> expected;
    for (int i=1;i<=p-1;++i) for (int j=1;j<=p-1;++j) expected.emplace_back(i,j);
    for (int fi=0; fi<faces.face_count; ++fi) {
        auto got = face_interior_roles(pat, fi);
        ASSERT_EQ(got, expected);
    }
}

TEST_F(CellTopologyTest, HighOrder_PerFaceOrdering_Wedge_p3) {
    int p = 3;
    auto pat = CellTopology::high_order_pattern(CellFamily::Wedge, p, CellTopology::HighOrderKind::Lagrange);
    auto faces = CellTopology::get_oriented_boundary_faces_view(CellFamily::Wedge);
    for (int fi=0; fi<faces.face_count; ++fi) {
        int b=faces.offsets[fi], e=faces.offsets[fi+1];
        int fv = e-b;
        auto got = face_interior_roles(pat, fi);
        if (fv == 3) { // tri face
            std::vector<std::pair<int,int>> expected;
            for (int i=1;i<=p-2;++i) for (int j=1;j<=p-1-i;++j) expected.emplace_back(i,j);
            ASSERT_EQ(got, expected);
        } else if (fv == 4) { // quad face
            std::vector<std::pair<int,int>> expected;
            for (int i=1;i<=p-1;++i) for (int j=1;j<=p-1;++j) expected.emplace_back(i,j);
            ASSERT_EQ(got, expected);
        }
    }
}

TEST_F(CellTopologyTest, HighOrder_PerFaceOrdering_Pyramid_p3) {
    int p = 3;
    auto pat = CellTopology::high_order_pattern(CellFamily::Pyramid, p, CellTopology::HighOrderKind::Lagrange);
    auto faces = CellTopology::get_oriented_boundary_faces_view(CellFamily::Pyramid);
    for (int fi=0; fi<faces.face_count; ++fi) {
        int b=faces.offsets[fi], e=faces.offsets[fi+1];
        int fv = e-b;
        auto got = face_interior_roles(pat, fi);
        if (fv == 3) { // tri face
            std::vector<std::pair<int,int>> expected;
            for (int i=1;i<=p-2;++i) for (int j=1;j<=p-1-i;++j) expected.emplace_back(i,j);
            ASSERT_EQ(got, expected);
        } else if (fv == 4) { // quad face (base)
            std::vector<std::pair<int,int>> expected;
            for (int i=1;i<=p-1;++i) for (int j=1;j<=p-1;++j) expected.emplace_back(i,j);
            ASSERT_EQ(got, expected);
        }
    }
}

}} // namespace svmp::test
