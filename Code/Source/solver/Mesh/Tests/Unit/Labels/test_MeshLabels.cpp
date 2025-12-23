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
#include "Labels/MeshLabels.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace svmp {
namespace test {

namespace {

MeshBase make_two_triangles_square_2d() {
  // 2D square split into two triangles:
  // (0,1)--(1,1)
  //   |   /  |
  //   |  /   |
  // (0,0)--(1,0)
  MeshBase mesh;
  std::vector<real_t> X = {
      0.0, 0.0,  // 0
      1.0, 0.0,  // 1
      1.0, 1.0,  // 2
      0.0, 1.0   // 3
  };
  std::vector<offset_t> offsets = {0, 3, 6};
  std::vector<index_t> conn = {0, 1, 2, 0, 2, 3};
  std::vector<CellShape> shapes(2);
  for (auto& cs : shapes) {
    cs.family = CellFamily::Triangle;
    cs.order = 1;
    cs.num_corners = 3;
  }
  mesh.build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

MeshBase make_two_disjoint_triangles_2d() {
  MeshBase mesh;
  std::vector<real_t> X = {
      0.0, 0.0,  // 0
      1.0, 0.0,  // 1
      0.0, 1.0,  // 2
      10.0, 0.0, // 3
      11.0, 0.0, // 4
      10.0, 1.0  // 5
  };
  std::vector<offset_t> offsets = {0, 3, 6};
  std::vector<index_t> conn = {0, 1, 2, 3, 4, 5};
  std::vector<CellShape> shapes(2);
  for (auto& cs : shapes) {
    cs.family = CellFamily::Triangle;
    cs.order = 1;
    cs.num_corners = 3;
  }
  mesh.build_from_arrays(/*spatial_dim=*/2, X, offsets, conn, shapes);
  mesh.finalize();
  return mesh;
}

index_t find_edge_face(const MeshBase& mesh, index_t a, index_t b) {
  for (index_t f = 0; f < static_cast<index_t>(mesh.n_faces()); ++f) {
    auto [vptr, nv] = mesh.face_vertices_span(f);
    if (nv != 2) continue;
    const index_t v0 = vptr[0];
    const index_t v1 = vptr[1];
    if ((v0 == a && v1 == b) || (v0 == b && v1 == a)) {
      return f;
    }
  }
  return INVALID_INDEX;
}

index_t find_edge_index(const MeshBase& mesh, index_t a, index_t b) {
  for (index_t e = 0; e < static_cast<index_t>(mesh.n_edges()); ++e) {
    const auto ev = mesh.edge_vertices(e);
    if ((ev[0] == a && ev[1] == b) || (ev[0] == b && ev[1] == a)) {
      return e;
    }
  }
  return INVALID_INDEX;
}

} // namespace

TEST(MeshLabelsTest, RegionLabelBasics) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_EQ(mesh.n_cells(), 2u);

  MeshLabels::set_region_label(mesh, 0, 10);
  MeshLabels::set_region_label(mesh, 1, 20);

  EXPECT_EQ(MeshLabels::region_label(mesh, 0), 10);
  EXPECT_EQ(MeshLabels::region_label(mesh, 1), 20);

  auto c10 = MeshLabels::cells_with_region(mesh, 10);
  ASSERT_EQ(c10.size(), 1u);
  EXPECT_EQ(c10[0], 0);

  auto uniq = MeshLabels::unique_region_labels(mesh);
  EXPECT_TRUE(uniq.count(10) > 0);
  EXPECT_TRUE(uniq.count(20) > 0);

  auto counts = MeshLabels::count_by_region(mesh);
  EXPECT_EQ(counts[10], 1u);
  EXPECT_EQ(counts[20], 1u);
}

TEST(MeshLabelsTest, BoundaryLabelBasics) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_EQ(mesh.n_faces(), 5u); // 4 boundary edges + 1 interior edge

  // Label two opposite boundary edges with label 5
  const auto e01 = find_edge_face(mesh, 0, 1);
  const auto e23 = find_edge_face(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_boundary_label(mesh, e01, 5);
  MeshLabels::set_boundary_label(mesh, e23, 5);

  EXPECT_EQ(MeshLabels::boundary_label(mesh, e01), 5);
  EXPECT_EQ(MeshLabels::boundary_label(mesh, e23), 5);

  auto f5 = MeshLabels::faces_with_boundary(mesh, 5);
  ASSERT_EQ(f5.size(), 2u);
  std::sort(f5.begin(), f5.end());
  EXPECT_EQ(f5[0], std::min(e01, e23));
  EXPECT_EQ(f5[1], std::max(e01, e23));

  auto uniq = MeshLabels::unique_boundary_labels(mesh);
  EXPECT_EQ(uniq.size(), 1u);
  EXPECT_TRUE(uniq.count(5) > 0);

  auto counts = MeshLabels::count_by_boundary(mesh);
  EXPECT_EQ(counts[5], 2u);
}

TEST(MeshLabelsTest, EdgeLabelBasics) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_GT(mesh.n_edges(), 0u);

  const auto e01 = find_edge_index(mesh, 0, 1);
  const auto e23 = find_edge_index(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_edge_label(mesh, e01, 7);
  MeshLabels::set_edge_label(mesh, e23, 7);

  EXPECT_EQ(MeshLabels::edge_label(mesh, e01), 7);
  EXPECT_EQ(MeshLabels::edge_label(mesh, e23), 7);

  auto edges = MeshLabels::edges_with_label(mesh, 7);
  std::sort(edges.begin(), edges.end());
  ASSERT_EQ(edges.size(), 2u);
  EXPECT_EQ(edges[0], std::min(e01, e23));
  EXPECT_EQ(edges[1], std::max(e01, e23));

  auto uniq = MeshLabels::unique_edge_labels(mesh);
  EXPECT_EQ(uniq, (std::unordered_set<label_t>{7}));

  auto counts = MeshLabels::count_by_edge(mesh);
  EXPECT_EQ(counts[7], 2u);
}

TEST(MeshLabelsTest, VertexLabelBasics) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_EQ(mesh.n_vertices(), 4u);

  MeshLabels::set_vertex_label(mesh, 0, 4);
  MeshLabels::set_vertex_label(mesh, 2, 4);
  MeshLabels::set_vertex_labels(mesh, {1, 3}, 5);

  EXPECT_EQ(MeshLabels::vertex_label(mesh, 0), 4);
  EXPECT_EQ(MeshLabels::vertex_label(mesh, 2), 4);
  EXPECT_EQ(MeshLabels::vertex_label(mesh, 1), 5);
  EXPECT_EQ(MeshLabels::vertex_label(mesh, 3), 5);

  auto v4 = MeshLabels::vertices_with_label(mesh, 4);
  std::sort(v4.begin(), v4.end());
  EXPECT_EQ(v4, (std::vector<index_t>{0, 2}));

  auto uniq = MeshLabels::unique_vertex_labels(mesh);
  EXPECT_EQ(uniq, (std::unordered_set<label_t>{4, 5}));

  auto counts = MeshLabels::count_by_vertex(mesh);
  EXPECT_EQ(counts[4], 2u);
  EXPECT_EQ(counts[5], 2u);
}

TEST(MeshLabelsTest, NamedSets_AddRemoveList) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_EQ(mesh.n_cells(), 2u);

  MeshLabels::add_to_set(mesh, EntityKind::Volume, "cells", std::vector<index_t>{0, 1});
  EXPECT_TRUE(MeshLabels::has_set(mesh, EntityKind::Volume, "cells"));

  auto cells = MeshLabels::get_set(mesh, EntityKind::Volume, "cells");
  EXPECT_EQ(cells.size(), 2u);

  MeshLabels::remove_from_set(mesh, EntityKind::Volume, "cells", 0);
  cells = MeshLabels::get_set(mesh, EntityKind::Volume, "cells");
  EXPECT_EQ(cells.size(), 1u);
  EXPECT_EQ(cells[0], 1);

  auto names = MeshLabels::list_sets(mesh, EntityKind::Volume);
  EXPECT_TRUE(std::find(names.begin(), names.end(), "cells") != names.end());

  MeshLabels::remove_set(mesh, EntityKind::Volume, "cells");
  EXPECT_FALSE(MeshLabels::has_set(mesh, EntityKind::Volume, "cells"));
}

TEST(MeshLabelsTest, NamedSets_NoDuplicates) {
  auto mesh = make_two_triangles_square_2d();

  MeshLabels::add_to_set(mesh, EntityKind::Volume, "cells", std::vector<index_t>{0, 0, 1, 1});
  auto cells = MeshLabels::get_set(mesh, EntityKind::Volume, "cells");
  std::sort(cells.begin(), cells.end());
  EXPECT_EQ(cells, (std::vector<index_t>{0, 1}));
}

TEST(MeshLabelsTest, CreateSetFromLabel_OverwritesAndSupportsKinds) {
  auto mesh = make_two_triangles_square_2d();

  MeshLabels::set_region_label(mesh, 0, 1);
  MeshLabels::set_region_label(mesh, 1, 2);

  MeshLabels::create_set_from_label(mesh, EntityKind::Volume, "region", 1);
  EXPECT_EQ(MeshLabels::get_set(mesh, EntityKind::Volume, "region"), (std::vector<index_t>{0}));

  MeshLabels::create_set_from_label(mesh, EntityKind::Volume, "region", 2);
  EXPECT_EQ(MeshLabels::get_set(mesh, EntityKind::Volume, "region"), (std::vector<index_t>{1}));

  MeshLabels::set_vertex_labels(mesh, {0, 1}, 9);
  MeshLabels::create_set_from_label(mesh, EntityKind::Vertex, "vtx", 9);
  auto vset = MeshLabels::get_set(mesh, EntityKind::Vertex, "vtx");
  std::sort(vset.begin(), vset.end());
  EXPECT_EQ(vset, (std::vector<index_t>{0, 1}));

  const auto e01 = find_edge_index(mesh, 0, 1);
  const auto e23 = find_edge_index(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);
  MeshLabels::set_edge_labels(mesh, {e01, e23}, 11);
  MeshLabels::create_set_from_label(mesh, EntityKind::Edge, "edge", 11);
  auto eset = MeshLabels::get_set(mesh, EntityKind::Edge, "edge");
  std::sort(eset.begin(), eset.end());
  EXPECT_EQ(eset, (std::vector<index_t>{std::min(e01, e23), std::max(e01, e23)}));

  const auto f01 = find_edge_face(mesh, 0, 1);
  const auto f23 = find_edge_face(mesh, 2, 3);
  ASSERT_NE(f01, INVALID_INDEX);
  ASSERT_NE(f23, INVALID_INDEX);
  MeshLabels::set_boundary_labels(mesh, {f01, f23}, 5);
  MeshLabels::create_set_from_label(mesh, EntityKind::Face, "face", 5);
  auto fset = MeshLabels::get_set(mesh, EntityKind::Face, "face");
  std::sort(fset.begin(), fset.end());
  EXPECT_EQ(fset, (std::vector<index_t>{std::min(f01, f23), std::max(f01, f23)}));
}

TEST(MeshLabelsTest, LabelRegistry_BijectionAndClear) {
  MeshBase mesh(2);

  MeshLabels::register_label(mesh, "fluid", 1);
  MeshLabels::register_label(mesh, "solid", 2);

  EXPECT_EQ(MeshLabels::label_name(mesh, 1), "fluid");
  EXPECT_EQ(MeshLabels::label_from_name(mesh, "solid"), 2);

  auto registry = MeshLabels::list_label_names(mesh);
  EXPECT_EQ(registry[1], "fluid");
  EXPECT_EQ(registry[2], "solid");

  // Overwrite label 1 with a new name; old name should no longer resolve.
  MeshLabels::register_label(mesh, "new_fluid", 1);
  EXPECT_EQ(MeshLabels::label_name(mesh, 1), "new_fluid");
  EXPECT_EQ(MeshLabels::label_from_name(mesh, "fluid"), INVALID_LABEL);

  MeshLabels::clear_label_registry(mesh);
  EXPECT_EQ(MeshLabels::label_name(mesh, 1), "");
  EXPECT_EQ(MeshLabels::label_from_name(mesh, "new_fluid"), INVALID_LABEL);
}

TEST(MeshLabelsTest, LabelRegistry_RejectsNegativeLabels) {
  MeshBase mesh(2);
  EXPECT_THROW(MeshLabels::register_label(mesh, "bad", INVALID_LABEL), std::invalid_argument);
  EXPECT_THROW(MeshLabels::register_label(mesh, "bad2", -7), std::invalid_argument);
}

TEST(MeshLabelsTest, RenumberLabels_UpdatesRegistry) {
  auto mesh = make_two_disjoint_triangles_2d();
  MeshLabels::set_region_label(mesh, 0, 10);
  MeshLabels::set_region_label(mesh, 1, 20);

  MeshLabels::register_label(mesh, "ten", 10);
  MeshLabels::register_label(mesh, "twenty", 20);
  MeshLabels::register_label(mesh, "keep", 99);

  MeshLabels::renumber_labels(mesh, EntityKind::Volume);

  EXPECT_EQ(MeshLabels::label_from_name(mesh, "ten"), 0);
  EXPECT_EQ(MeshLabels::label_from_name(mesh, "twenty"), 1);
  EXPECT_EQ(MeshLabels::label_name(mesh, 0), "ten");
  EXPECT_EQ(MeshLabels::label_name(mesh, 1), "twenty");
  EXPECT_EQ(MeshLabels::label_name(mesh, 10), "");
  EXPECT_EQ(MeshLabels::label_from_name(mesh, "keep"), 99);
}

TEST(MeshLabelsTest, MergeLabels_UpdatesRegistry) {
  {
    auto mesh = make_two_triangles_square_2d();
    MeshLabels::set_region_label(mesh, 0, 1);
    MeshLabels::set_region_label(mesh, 1, 2);

    MeshLabels::register_label(mesh, "a", 1);
    MeshLabels::register_label(mesh, "b", 2);
    MeshLabels::register_label(mesh, "target", 9);

    MeshLabels::merge_labels(mesh, EntityKind::Volume, {1, 2}, 9);
    EXPECT_EQ(MeshLabels::label_name(mesh, 9), "target");
    EXPECT_EQ(MeshLabels::label_from_name(mesh, "target"), 9);
    EXPECT_EQ(MeshLabels::label_from_name(mesh, "a"), INVALID_LABEL);
    EXPECT_EQ(MeshLabels::label_from_name(mesh, "b"), INVALID_LABEL);
  }

  {
    auto mesh = make_two_triangles_square_2d();
    MeshLabels::set_region_label(mesh, 0, 1);
    MeshLabels::set_region_label(mesh, 1, 2);

    MeshLabels::register_label(mesh, "a", 1);
    MeshLabels::register_label(mesh, "b", 2);

    MeshLabels::merge_labels(mesh, EntityKind::Volume, {1, 2}, 9);
    EXPECT_EQ(MeshLabels::label_name(mesh, 9), "a");
    EXPECT_EQ(MeshLabels::label_from_name(mesh, "a"), 9);
    EXPECT_EQ(MeshLabels::label_from_name(mesh, "b"), INVALID_LABEL);
  }
}

TEST(MeshLabelsTest, SplitByConnectivity_VolumeTwoComponents) {
  auto mesh = make_two_disjoint_triangles_2d();
  ASSERT_EQ(mesh.n_cells(), 2u);

  MeshLabels::set_region_label(mesh, 0, 7);
  MeshLabels::set_region_label(mesh, 1, 7);

  auto m = MeshLabels::split_by_connectivity(mesh, EntityKind::Volume, 7);
  ASSERT_EQ(m.size(), 2u);

  EXPECT_EQ(mesh.region_label(0), 7);
  EXPECT_EQ(mesh.region_label(1), 8);
  EXPECT_EQ(m.at(0), 7);
  EXPECT_EQ(m.at(1), 8);
}

TEST(MeshLabelsTest, SplitByConnectivity_Volume_UpdatesRegistry) {
  auto mesh = make_two_disjoint_triangles_2d();
  MeshLabels::set_region_label(mesh, 0, 7);
  MeshLabels::set_region_label(mesh, 1, 7);
  MeshLabels::register_label(mesh, "region", 7);

  MeshLabels::split_by_connectivity(mesh, EntityKind::Volume, 7);

  EXPECT_EQ(mesh.region_label(0), 7);
  EXPECT_EQ(mesh.region_label(1), 8);
  EXPECT_EQ(MeshLabels::label_name(mesh, 7), "region");
  EXPECT_EQ(MeshLabels::label_name(mesh, 8), "region_component_1");
  EXPECT_EQ(MeshLabels::label_from_name(mesh, "region_component_1"), 8);
}

TEST(MeshLabelsTest, SplitByConnectivity_Volume_NoFacesFallback) {
  MeshBase mesh(3);

  // Two disjoint "polyhedron" cells; MeshBase::finalize() can't derive faces for Polyhedron.
  // Connectivity should fall back to shared-vertex adjacency (none here), producing 2 components.
  for (index_t i = 0; i < 8; ++i) {
    mesh.add_vertex(i, {static_cast<real_t>(i), 0.0, 0.0});
  }
  mesh.add_cell(0, CellFamily::Polyhedron, {0, 1, 2, 3});
  mesh.add_cell(1, CellFamily::Polyhedron, {4, 5, 6, 7});

  MeshLabels::set_region_label(mesh, 0, 7);
  MeshLabels::set_region_label(mesh, 1, 7);

  auto m = MeshLabels::split_by_connectivity(mesh, EntityKind::Volume, 7);
  ASSERT_EQ(m.size(), 2u);
  EXPECT_EQ(mesh.region_label(0), 7);
  EXPECT_EQ(mesh.region_label(1), 8);
}

TEST(MeshLabelsTest, SplitByConnectivity_FaceTwoComponents) {
  auto mesh = make_two_triangles_square_2d();

  const auto e01 = find_edge_face(mesh, 0, 1);
  const auto e23 = find_edge_face(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_boundary_label(mesh, e01, 5);
  MeshLabels::set_boundary_label(mesh, e23, 5);

  auto m = MeshLabels::split_by_connectivity(mesh, EntityKind::Face, 5);
  ASSERT_EQ(m.size(), 2u);

  std::unordered_set<label_t> new_labels = {mesh.boundary_label(e01), mesh.boundary_label(e23)};
  EXPECT_EQ(new_labels.size(), 2u);
  EXPECT_TRUE(new_labels.count(5) > 0);
  EXPECT_TRUE(new_labels.count(6) > 0);
}

TEST(MeshLabelsTest, SplitByConnectivity_FaceLineSegmentsShareVertex) {
  auto mesh = make_two_triangles_square_2d();

  // Adjacent boundary edges (0,1) and (1,2) share vertex 1 and should remain connected.
  const auto e01 = find_edge_face(mesh, 0, 1);
  const auto e12 = find_edge_face(mesh, 1, 2);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e12, INVALID_INDEX);

  MeshLabels::set_boundary_label(mesh, e01, 5);
  MeshLabels::set_boundary_label(mesh, e12, 5);

  auto m = MeshLabels::split_by_connectivity(mesh, EntityKind::Face, 5);
  ASSERT_EQ(m.size(), 2u);

  EXPECT_EQ(mesh.boundary_label(e01), 5);
  EXPECT_EQ(mesh.boundary_label(e12), 5);
  EXPECT_EQ(m.at(e01), 5);
  EXPECT_EQ(m.at(e12), 5);
}

TEST(MeshLabelsTest, SplitByConnectivity_EdgeTwoComponents) {
  auto mesh = make_two_triangles_square_2d();

  const auto e01 = find_edge_index(mesh, 0, 1);
  const auto e23 = find_edge_index(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_edge_label(mesh, e01, 5);
  MeshLabels::set_edge_label(mesh, e23, 5);

  auto m = MeshLabels::split_by_connectivity(mesh, EntityKind::Edge, 5);
  ASSERT_EQ(m.size(), 2u);

  std::unordered_set<label_t> new_labels = {mesh.edge_label(e01), mesh.edge_label(e23)};
  EXPECT_EQ(new_labels.size(), 2u);
  EXPECT_TRUE(new_labels.count(5) > 0);
  EXPECT_TRUE(new_labels.count(6) > 0);
  EXPECT_EQ(m.at(e01), mesh.edge_label(e01));
  EXPECT_EQ(m.at(e23), mesh.edge_label(e23));
}

TEST(MeshLabelsTest, SplitByConnectivity_VertexTwoComponents) {
  auto mesh = make_two_disjoint_triangles_2d();
  ASSERT_EQ(mesh.n_vertices(), 6u);

  MeshLabels::set_vertex_label(mesh, 0, 7);
  MeshLabels::set_vertex_label(mesh, 3, 7);

  auto m = MeshLabels::split_by_connectivity(mesh, EntityKind::Vertex, 7);
  ASSERT_EQ(m.size(), 2u);
  EXPECT_EQ(mesh.vertex_label(0), 7);
  EXPECT_EQ(mesh.vertex_label(3), 8);
  EXPECT_EQ(m.at(0), 7);
  EXPECT_EQ(m.at(3), 8);
}

TEST(MeshLabelsTest, RenumberLabels_Volume) {
  auto mesh = make_two_disjoint_triangles_2d();
  MeshLabels::set_region_label(mesh, 0, 10);
  MeshLabels::set_region_label(mesh, 1, 20);

  auto map = MeshLabels::renumber_labels(mesh, EntityKind::Volume);
  EXPECT_EQ(map.at(10), 0);
  EXPECT_EQ(map.at(20), 1);
  EXPECT_EQ(mesh.region_label(0), 0);
  EXPECT_EQ(mesh.region_label(1), 1);
}

TEST(MeshLabelsTest, RenumberLabels_Face) {
  auto mesh = make_two_triangles_square_2d();

  const auto e01 = find_edge_face(mesh, 0, 1);
  const auto e23 = find_edge_face(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_boundary_label(mesh, e01, 10);
  MeshLabels::set_boundary_label(mesh, e23, 20);

  auto map = MeshLabels::renumber_labels(mesh, EntityKind::Face);
  EXPECT_EQ(map.at(10), 0);
  EXPECT_EQ(map.at(20), 1);

  std::unordered_set<label_t> labels = {mesh.boundary_label(e01), mesh.boundary_label(e23)};
  EXPECT_EQ(labels, (std::unordered_set<label_t>{0, 1}));
}

TEST(MeshLabelsTest, RenumberLabels_Edge) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_GT(mesh.n_edges(), 0u);

  const auto e01 = find_edge_index(mesh, 0, 1);
  const auto e23 = find_edge_index(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_edge_label(mesh, e01, 10);
  MeshLabels::set_edge_label(mesh, e23, 20);

  auto map = MeshLabels::renumber_labels(mesh, EntityKind::Edge);
  EXPECT_EQ(map.at(10), 0);
  EXPECT_EQ(map.at(20), 1);
  EXPECT_EQ(mesh.edge_label(e01), 0);
  EXPECT_EQ(mesh.edge_label(e23), 1);
}

TEST(MeshLabelsTest, RenumberLabels_Vertex) {
  auto mesh = make_two_triangles_square_2d();
  ASSERT_EQ(mesh.n_vertices(), 4u);

  MeshLabels::set_vertex_label(mesh, 0, 10);
  MeshLabels::set_vertex_label(mesh, 3, 20);

  auto map = MeshLabels::renumber_labels(mesh, EntityKind::Vertex);
  EXPECT_EQ(map.at(10), 0);
  EXPECT_EQ(map.at(20), 1);
  EXPECT_EQ(mesh.vertex_label(0), 0);
  EXPECT_EQ(mesh.vertex_label(3), 1);
  EXPECT_EQ(mesh.vertex_label(1), INVALID_LABEL);
  EXPECT_EQ(mesh.vertex_label(2), INVALID_LABEL);
}

TEST(MeshLabelsTest, MergeLabels_VolumeAndFace) {
  auto mesh = make_two_triangles_square_2d();
  MeshLabels::set_region_label(mesh, 0, 1);
  MeshLabels::set_region_label(mesh, 1, 2);
  MeshLabels::merge_labels(mesh, EntityKind::Volume, {1, 2}, 9);
  EXPECT_EQ(mesh.region_label(0), 9);
  EXPECT_EQ(mesh.region_label(1), 9);

  const auto e01 = find_edge_face(mesh, 0, 1);
  const auto e23 = find_edge_face(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);
  MeshLabels::set_boundary_label(mesh, e01, 1);
  MeshLabels::set_boundary_label(mesh, e23, 2);
  MeshLabels::merge_labels(mesh, EntityKind::Face, {1, 2}, 9);
  EXPECT_EQ(mesh.boundary_label(e01), 9);
  EXPECT_EQ(mesh.boundary_label(e23), 9);
}

TEST(MeshLabelsTest, MergeLabels_EdgeAndVertex) {
  auto mesh = make_two_triangles_square_2d();

  const auto e01 = find_edge_index(mesh, 0, 1);
  const auto e23 = find_edge_index(mesh, 2, 3);
  ASSERT_NE(e01, INVALID_INDEX);
  ASSERT_NE(e23, INVALID_INDEX);

  MeshLabels::set_edge_label(mesh, e01, 1);
  MeshLabels::set_edge_label(mesh, e23, 2);
  MeshLabels::merge_labels(mesh, EntityKind::Edge, {1, 2}, 9);
  EXPECT_EQ(mesh.edge_label(e01), 9);
  EXPECT_EQ(mesh.edge_label(e23), 9);

  MeshLabels::set_vertex_label(mesh, 0, 1);
  MeshLabels::set_vertex_label(mesh, 2, 2);
  MeshLabels::merge_labels(mesh, EntityKind::Vertex, {1, 2}, 9);
  EXPECT_EQ(mesh.vertex_label(0), 9);
  EXPECT_EQ(mesh.vertex_label(2), 9);
}

TEST(MeshLabelsTest, ExportImportAndCopy) {
  auto source = make_two_disjoint_triangles_2d();
  auto target = make_two_disjoint_triangles_2d();

  MeshLabels::set_region_label(source, 0, 3);
  MeshLabels::set_region_label(source, 1, 4);
  MeshLabels::copy_labels(source, target, EntityKind::Volume);
  EXPECT_EQ(target.region_label(0), 3);
  EXPECT_EQ(target.region_label(1), 4);

  auto exported = MeshLabels::export_labels(source, EntityKind::Volume);
  ASSERT_EQ(exported.size(), source.n_cells());

  MeshLabels::set_region_label(target, 0, 0);
  MeshLabels::set_region_label(target, 1, 0);
  MeshLabels::import_labels(target, EntityKind::Volume, exported);
  EXPECT_EQ(target.region_label(0), 3);
  EXPECT_EQ(target.region_label(1), 4);
}

TEST(MeshLabelsTest, ExportImportAndCopy_EdgeVertex) {
  auto source = make_two_triangles_square_2d();
  auto target = make_two_triangles_square_2d();

  // Ensure edge indexing matches between meshes (edge extraction order is not guaranteed).
  target.set_edges_from_arrays(source.edge2vertex());

  const auto e01_src = find_edge_index(source, 0, 1);
  const auto e23_src = find_edge_index(source, 2, 3);
  const auto e01_tgt = find_edge_index(target, 0, 1);
  const auto e23_tgt = find_edge_index(target, 2, 3);
  ASSERT_NE(e01_src, INVALID_INDEX);
  ASSERT_NE(e23_src, INVALID_INDEX);
  ASSERT_NE(e01_tgt, INVALID_INDEX);
  ASSERT_NE(e23_tgt, INVALID_INDEX);

  MeshLabels::set_edge_labels(source, {e01_src, e23_src}, 7);
  MeshLabels::set_vertex_labels(source, {0, 2}, 5);

  MeshLabels::copy_labels(source, target, EntityKind::Edge);
  MeshLabels::copy_labels(source, target, EntityKind::Vertex);
  EXPECT_EQ(target.edge_label(e01_tgt), 7);
  EXPECT_EQ(target.edge_label(e23_tgt), 7);
  EXPECT_EQ(target.vertex_label(0), 5);
  EXPECT_EQ(target.vertex_label(2), 5);

  auto exported_edges = MeshLabels::export_labels(source, EntityKind::Edge);
  auto exported_vertices = MeshLabels::export_labels(source, EntityKind::Vertex);
  ASSERT_EQ(exported_edges.size(), source.n_edges());
  ASSERT_EQ(exported_vertices.size(), source.n_vertices());

  // Clear labels then import them back.
  for (index_t e = 0; e < static_cast<index_t>(target.n_edges()); ++e) {
    target.set_edge_label(e, INVALID_LABEL);
  }
  for (index_t v = 0; v < static_cast<index_t>(target.n_vertices()); ++v) {
    target.set_vertex_label(v, INVALID_LABEL);
  }

  MeshLabels::import_labels(target, EntityKind::Edge, exported_edges);
  MeshLabels::import_labels(target, EntityKind::Vertex, exported_vertices);
  EXPECT_EQ(target.edge_label(e01_tgt), 7);
  EXPECT_EQ(target.edge_label(e23_tgt), 7);
  EXPECT_EQ(target.vertex_label(0), 5);
  EXPECT_EQ(target.vertex_label(2), 5);
}

} // namespace test
} // namespace svmp
