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

#include "Core/MeshBase.h"
#include "Topology/CellShape.h"
#include "Topology/CellTopology.h"
#include "Topology/MeshTopology.h"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace svmp {
namespace test {

static std::set<std::pair<index_t, index_t>> as_edge_set(
    const std::vector<std::array<index_t, 2>>& edges) {
  std::set<std::pair<index_t, index_t>> out;
  for (const auto& e : edges) {
    index_t a = e[0];
    index_t b = e[1];
    if (a > b) std::swap(a, b);
    out.insert({a, b});
  }
  return out;
}

static std::set<std::pair<index_t, index_t>> expected_edges_from_view(
    const index_t* vertices_ptr, size_t n_vertices, const CellTopology::EdgeListView& eview) {
  std::set<std::pair<index_t, index_t>> out;
  for (int i = 0; i < eview.edge_count; ++i) {
    const index_t li0 = eview.pairs_flat[2 * i + 0];
    const index_t li1 = eview.pairs_flat[2 * i + 1];
    EXPECT_GE(li0, 0);
    EXPECT_GE(li1, 0);
    EXPECT_LT(static_cast<size_t>(li0), n_vertices);
    EXPECT_LT(static_cast<size_t>(li1), n_vertices);
    index_t a = vertices_ptr[li0];
    index_t b = vertices_ptr[li1];
    if (a > b) std::swap(a, b);
    out.insert({a, b});
  }
  return out;
}

static std::map<std::pair<index_t, index_t>, std::set<index_t>> edge2cell_map(
    const std::vector<std::array<index_t, 2>>& edges,
    const std::vector<offset_t>& offsets,
    const std::vector<index_t>& edge2cell) {
  std::map<std::pair<index_t, index_t>, std::set<index_t>> out;
  EXPECT_EQ(offsets.size(), edges.size() + 1);
  for (size_t e = 0; e < edges.size(); ++e) {
    index_t a = edges[e][0];
    index_t b = edges[e][1];
    if (a > b) std::swap(a, b);
    offset_t b0 = offsets[e];
    offset_t b1 = offsets[e + 1];
    out[{a, b}] = std::set<index_t>(edge2cell.begin() + b0, edge2cell.begin() + b1);
  }
  return out;
}

TEST(MeshTopologyTest, ExtractEdges_TetraMatchesCellTopology) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});

  auto edges = MeshTopology::extract_edges(mesh);
  auto edge_set = as_edge_set(edges);

  auto [vptr, nv] = mesh.cell_vertices_span(0);
  auto expected = expected_edges_from_view(vptr, nv, CellTopology::get_edges_view(CellFamily::Tetra));

  EXPECT_EQ(edge_set, expected);
  EXPECT_EQ(edge_set.size(), 6u);
}

TEST(MeshTopologyTest, ExtractEdges_WedgeMatchesCellTopology) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 6; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Wedge, {0, 1, 2, 3, 4, 5});

  auto edges = MeshTopology::extract_edges(mesh);
  auto edge_set = as_edge_set(edges);

  auto [vptr, nv] = mesh.cell_vertices_span(0);
  auto expected = expected_edges_from_view(vptr, nv, CellTopology::get_edges_view(CellFamily::Wedge));

  EXPECT_EQ(edge_set, expected);
  EXPECT_EQ(edge_set.size(), 9u);
}

TEST(MeshTopologyTest, ExtractEdges_PolygonUsesCornerCount) {
  MeshBase mesh(2);
  for (index_t i = 0; i < 5; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Polygon, {0, 1, 2, 3, 4});

  auto edges = MeshTopology::extract_edges(mesh);
  auto edge_set = as_edge_set(edges);

  auto [vptr, nv] = mesh.cell_vertices_span(0);
  auto expected = expected_edges_from_view(vptr, nv, CellTopology::get_polygon_edges_view(5));

  EXPECT_EQ(edge_set, expected);
  EXPECT_EQ(edge_set.size(), 5u);
}

TEST(MeshTopologyTest, ExtractEdges_PolyhedronRequiresFaces) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Polyhedron, {0, 1, 2, 3});

  EXPECT_THROW((void)MeshTopology::extract_edges(mesh), std::runtime_error);
}

TEST(MeshTopologyTest, ExtractEdges_PolyhedronUsesFaces) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Polyhedron, {0, 1, 2, 3});

  // Represent a tetrahedron as a polyhedron with 4 triangular faces.
  std::vector<CellShape> face_shapes(4);
  for (auto& s : face_shapes) {
    s.family = CellFamily::Triangle;
    s.num_corners = 3;
    s.order = 1;
    s.is_mixed_order = false;
  }
  std::vector<offset_t> face2v_off = {0, 3, 6, 9, 12};
  std::vector<index_t> face2v = {
      0, 1, 2,  // base
      0, 1, 3,  // side
      1, 2, 3,  // side
      0, 2, 3   // side
  };
  std::vector<std::array<index_t, 2>> face2c = {
      {{0, INVALID_INDEX}},
      {{0, INVALID_INDEX}},
      {{0, INVALID_INDEX}},
      {{0, INVALID_INDEX}},
  };
  mesh.set_faces_from_arrays(face_shapes, face2v_off, face2v, face2c);

  auto edges = MeshTopology::extract_edges(mesh);
  auto edge_set = as_edge_set(edges);

  std::set<std::pair<index_t, index_t>> expected = {
      {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  EXPECT_EQ(edge_set, expected);
}

TEST(MeshTopologyTest, BuildVertex2Codim1ReturnsEmptyCSRWhenNoFaces) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  // No finalize() -> no faces.

  std::vector<offset_t> off;
  std::vector<index_t> ent;
  MeshTopology::build_vertex2codim1(mesh, off, ent);

  ASSERT_EQ(off.size(), mesh.n_vertices() + 1);
  EXPECT_TRUE(ent.empty());
  for (size_t i = 0; i < off.size(); ++i) {
    EXPECT_EQ(off[i], 0);
  }
}

TEST(MeshTopologyTest, VertexCodim1_TetraHasThreeIncidentFacesPerVertex) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.finalize();

  std::vector<offset_t> off;
  std::vector<index_t> ent;
  MeshTopology::build_vertex2codim1(mesh, off, ent);
  ASSERT_EQ(off.size(), mesh.n_vertices() + 1);
  ASSERT_EQ(mesh.n_faces(), 4u);
  ASSERT_EQ(ent.size(), 12u);

  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    auto incident = MeshTopology::vertex_codim1(mesh, v, off, ent);
    EXPECT_EQ(incident.size(), 3u);
  }
}

TEST(MeshTopologyTest, BuildCodim1ToCodim1_TetraFacesHaveThreeNeighbors) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.finalize();

  std::vector<offset_t> off;
  std::vector<index_t> adj;
  MeshTopology::build_codim1_to_codim1(mesh, off, adj);

  ASSERT_EQ(mesh.n_faces(), 4u);
  ASSERT_EQ(off.size(), mesh.n_faces() + 1);
  for (size_t f = 0; f < mesh.n_faces(); ++f) {
    EXPECT_EQ(static_cast<size_t>(off[f + 1] - off[f]), 3u);
  }
}

TEST(MeshTopologyTest, FaceCountValenceAndEuler_Tetra) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.finalize();

  auto stats = MeshTopology::face_count_by_cell_type(mesh);
  ASSERT_TRUE(stats.find(CellFamily::Tetra) != stats.end());
  EXPECT_EQ(stats[CellFamily::Tetra].first, 4);
  EXPECT_EQ(stats[CellFamily::Tetra].second, 4);

  auto valence = MeshTopology::vertex_valence(mesh);
  ASSERT_EQ(valence.size(), 4u);
  for (auto v : valence) EXPECT_EQ(v, 3);

  EXPECT_TRUE(MeshTopology::irregular_vertices(mesh, 3).empty());
  EXPECT_EQ(MeshTopology::euler_characteristic(mesh), 1);
}

TEST(MeshTopologyTest, BoundaryEdges_TetraEqualsAllEdges) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 4; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.finalize();

  auto all = as_edge_set(MeshTopology::extract_edges(mesh));
  auto bnd = as_edge_set(MeshTopology::boundary_edges(mesh));
  EXPECT_EQ(all, bnd);
}

TEST(MeshTopologyTest, BuildEdge2Cell_TwoTetsShareFace) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 5; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.add_cell(1, CellFamily::Tetra, {0, 1, 2, 4});
  mesh.finalize();

  auto edges = MeshTopology::extract_edges(mesh);
  std::vector<offset_t> off;
  std::vector<index_t> e2c;
  MeshTopology::build_edge2cell(mesh, edges, off, e2c);

  auto m = edge2cell_map(edges, off, e2c);
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(0, 1)), (std::set<index_t>{0, 1}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(0, 2)), (std::set<index_t>{0, 1}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(1, 2)), (std::set<index_t>{0, 1}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(0, 3)), (std::set<index_t>{0}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(1, 3)), (std::set<index_t>{0}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(2, 3)), (std::set<index_t>{0}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(0, 4)), (std::set<index_t>{1}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(1, 4)), (std::set<index_t>{1}));
  EXPECT_EQ(m.at(std::make_pair<index_t, index_t>(2, 4)), (std::set<index_t>{1}));
}

TEST(MeshTopologyTest, NeighborAndBoundaryQueries_TwoTetsShareFace) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 5; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.add_cell(1, CellFamily::Tetra, {0, 1, 2, 4});
  mesh.finalize();

  // cell neighbors (via shared face)
  const auto n0_vec = MeshTopology::cell_neighbors(mesh, 0);
  const auto n1_vec = MeshTopology::cell_neighbors(mesh, 1);
  EXPECT_EQ(std::set<index_t>(n0_vec.begin(), n0_vec.end()), (std::set<index_t>{1}));
  EXPECT_EQ(std::set<index_t>(n1_vec.begin(), n1_vec.end()), (std::set<index_t>{0}));

  // vertex->cell adjacency
  const auto v0_cells = MeshTopology::vertex_cells(mesh, 0);
  const auto v3_cells = MeshTopology::vertex_cells(mesh, 3);
  const auto v4_cells = MeshTopology::vertex_cells(mesh, 4);
  EXPECT_EQ(std::set<index_t>(v0_cells.begin(), v0_cells.end()), (std::set<index_t>{0, 1}));
  EXPECT_EQ(std::set<index_t>(v3_cells.begin(), v3_cells.end()), (std::set<index_t>{0}));
  EXPECT_EQ(std::set<index_t>(v4_cells.begin(), v4_cells.end()), (std::set<index_t>{1}));

  // Find the shared internal face by vertex set {0,1,2}
  index_t shared_face = INVALID_INDEX;
  for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
    auto verts = mesh.face_vertices(f);
    std::sort(verts.begin(), verts.end());
    if (verts == std::vector<index_t>({0, 1, 2})) {
      shared_face = f;
      break;
    }
  }
  ASSERT_NE(shared_face, INVALID_INDEX);
  const auto fc = MeshTopology::codim1_cells(mesh, shared_face);
  EXPECT_EQ(std::set<index_t>(fc.begin(), fc.end()), (std::set<index_t>{0, 1}));
  EXPECT_FALSE(MeshTopology::is_boundary_codim1(mesh, shared_face));

  // Boundary identification
  auto bfaces = MeshTopology::boundary_codim1(mesh);
  EXPECT_EQ(bfaces.size(), 6u); // total 7 faces, 1 internal
  EXPECT_TRUE(MeshTopology::is_boundary_cell(mesh, 0));
  EXPECT_TRUE(MeshTopology::is_boundary_cell(mesh, 1));

  auto bverts = MeshTopology::boundary_vertices(mesh);
  EXPECT_EQ(std::set<index_t>(bverts.begin(), bverts.end()),
            (std::set<index_t>{0, 1, 2, 3, 4}));
}

TEST(MeshTopologyTest, NonManifoldEdge_TwoTetsShareOnlyAnEdge) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 6; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.add_cell(1, CellFamily::Tetra, {0, 1, 4, 5});
  mesh.finalize();

  EXPECT_FALSE(MeshTopology::is_manifold(mesh));
  auto nm_edges = as_edge_set(MeshTopology::non_manifold_edges(mesh));
  EXPECT_EQ(nm_edges, (std::set<std::pair<index_t, index_t>>{{0, 1}}));
  auto nm_verts = MeshTopology::non_manifold_vertices(mesh);
  EXPECT_EQ(std::set<index_t>(nm_verts.begin(), nm_verts.end()), (std::set<index_t>{0, 1}));
}

TEST(MeshTopologyTest, BuildCell2CellUsesFacesWhenAvailable) {
  MeshBase mesh(3);
  // Two tets sharing face (0,1,2), plus one disconnected tet.
  for (index_t i = 0; i < 9; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.add_cell(1, CellFamily::Tetra, {0, 1, 2, 4});
  mesh.add_cell(2, CellFamily::Tetra, {5, 6, 7, 8});
  mesh.finalize(); // builds faces and edges

  std::vector<offset_t> offsets;
  std::vector<index_t> cell2cell;
  MeshTopology::build_cell2cell(mesh, offsets, cell2cell);

  ASSERT_EQ(offsets.size(), 4u);
  const std::set<index_t> n0(cell2cell.begin() + offsets[0], cell2cell.begin() + offsets[1]);
  const std::set<index_t> n1(cell2cell.begin() + offsets[1], cell2cell.begin() + offsets[2]);
  const std::set<index_t> n2(cell2cell.begin() + offsets[2], cell2cell.begin() + offsets[3]);

  EXPECT_EQ(n0, (std::set<index_t>{1}));
  EXPECT_EQ(n1, (std::set<index_t>{0}));
  EXPECT_TRUE(n2.empty());
}

TEST(MeshTopologyTest, FindComponentsDetectsDisconnectedSubmeshes) {
  MeshBase mesh(3);
  for (index_t i = 0; i < 9; ++i) {
    mesh.add_vertex(i, {0.0, 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Tetra, {0, 1, 2, 3});
  mesh.add_cell(1, CellFamily::Tetra, {0, 1, 2, 4});
  mesh.add_cell(2, CellFamily::Tetra, {5, 6, 7, 8});
  mesh.finalize();

  const auto comps = MeshTopology::find_components(mesh);
  ASSERT_EQ(comps.size(), 3u);

  EXPECT_EQ(MeshTopology::count_components(mesh), 2);
  EXPECT_FALSE(MeshTopology::is_connected(mesh));

  // Cells 0 and 1 must be in the same component; cell 2 in a different one.
  EXPECT_EQ(comps[0], comps[1]);
  EXPECT_NE(comps[0], comps[2]);
}

} // namespace test
} // namespace svmp
