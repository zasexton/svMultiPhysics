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

/**
 * @file test_HierarchicalRefinementTree.cpp
 * @brief Tests for hierarchical refinement provenance (cell tree + vertex provenance flattening)
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/Options.h"
#include "../../../Adaptivity/Conformity.h"
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/QualityGuards.h"
#include "../../../Core/MeshBase.h"
#include "../../../Labels/MeshLabels.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace svmp {
namespace test {

namespace {

std::unique_ptr<MeshBase> make_single_quad() {
  auto mesh = std::make_unique<MeshBase>();
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {1.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 1.0, 0.0});
  mesh->add_cell(0, CellFamily::Quad, {0, 1, 2, 3});
  return mesh;
}

std::unique_ptr<MeshBase> make_two_triangle_edge_mesh() {
  auto mesh = std::make_unique<MeshBase>();

  // Two triangles sharing edge (0,1) on y=0.
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {0.0, 1.0, 0.0});   // above
  mesh->add_vertex(3, {0.0, -1.0, 0.0});  // below

  mesh->add_cell(0, CellFamily::Triangle, {0, 1, 2});  // upper
  mesh->add_cell(1, CellFamily::Triangle, {1, 0, 3});  // lower (shares edge)

  return mesh;
}

index_t find_vertex_by_xy(const MeshBase& mesh, double x, double y, double tol = 1e-12) {
  const double t2 = tol * tol;
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    auto p = mesh.get_vertex_coords(v);
    const double dx = static_cast<double>(p[0]) - x;
    const double dy = static_cast<double>(p[1]) - y;
    const double dz = static_cast<double>(p[2]);
    if (dx * dx + dy * dy + dz * dz < t2) return v;
  }
  return INVALID_INDEX;
}

AdaptivityOptions hierarchical_base_options() {
  AdaptivityOptions options;
  options.enable_refinement = true;
  options.enable_coarsening = false;
  options.check_quality = false;
  options.verbosity = 0;
  options.max_refinement_level = 10;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::HIERARCHICAL;
  return options;
}

} // namespace

TEST(HierarchicalRefinementTree, CellGidTreePersistsAcrossRefinementSteps) {
  auto mesh = make_single_quad();

  AdaptivityOptions options = hierarchical_base_options();
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;

  AdaptivityManager manager(options);

  const gid_t root_gid = mesh->cell_gids().at(0);

  // Refine once.
  {
    std::vector<bool> marks(mesh->n_cells(), true);
    auto r = manager.refine(*mesh, marks, nullptr);
    ASSERT_TRUE(r.success);
  }

  ASSERT_EQ(mesh->n_cells(), 4u);
  const auto level1 = mesh->cell_gids();
  ASSERT_EQ(level1.size(), 4u);

  // All first-level children should point to the original root.
  for (gid_t cg : level1) {
    EXPECT_EQ(MeshLabels::get_parent_cell_gid(*mesh, cg), root_gid);
  }

  // Root should list the first-level children.
  {
    auto children = MeshLabels::get_children_cells_gid(*mesh, root_gid);
    std::sort(children.begin(), children.end());
    auto sorted_level1 = level1;
    std::sort(sorted_level1.begin(), sorted_level1.end());
    EXPECT_EQ(children, sorted_level1);
  }

  // Refine second time (refine all level-1 cells).
  {
    std::vector<bool> marks(mesh->n_cells(), true);
    auto r = manager.refine(*mesh, marks, nullptr);
    ASSERT_TRUE(r.success);
  }

  ASSERT_EQ(mesh->n_cells(), 16u);
  const auto level2 = mesh->cell_gids();
  ASSERT_EQ(level2.size(), 16u);

  // Second-level children should point to one of the level-1 parents, and grandparent is root.
  const std::set<gid_t> level1_set(level1.begin(), level1.end());
  for (gid_t cg : level2) {
    gid_t parent = MeshLabels::get_parent_cell_gid(*mesh, cg);
    ASSERT_TRUE(level1_set.count(parent) > 0);
    EXPECT_EQ(MeshLabels::get_parent_cell_gid(*mesh, parent), root_gid);
  }

  // Each level-1 parent should list exactly 4 children.
  for (gid_t pg : level1) {
    auto children = MeshLabels::get_children_cells_gid(*mesh, pg);
    EXPECT_EQ(children.size(), 4u);
  }
}

TEST(HierarchicalRefinementTree, VertexProvenanceFlattensToRootCornerWeights) {
  auto mesh = make_two_triangle_edge_mesh();

  AdaptivityOptions options = hierarchical_base_options();
  options.conformity_mode = AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;
  options.max_hanging_level = 2;  // allow multi-level imbalance

  AdaptivityManager manager(options);

  // Level 1: refine only the upper triangle (cell 0).
  {
    std::vector<bool> marks(mesh->n_cells(), false);
    marks[0] = true;
    auto r = manager.refine(*mesh, marks, nullptr);
    ASSERT_TRUE(r.success);
  }

  // Level 2: refine only level-1 cells (children of the refined region).
  {
    std::vector<bool> marks(mesh->n_cells(), false);
    for (index_t c = 0; c < static_cast<index_t>(mesh->n_cells()); ++c) {
      if (mesh->refinement_level(c) == 1) {
        marks[static_cast<size_t>(c)] = true;
      }
    }
    auto r = manager.refine(*mesh, marks, nullptr);
    ASSERT_TRUE(r.success);
  }

  // Root edge endpoints (masters) are the original vertices at x=0 and x=1 on y=0.
  const index_t v0 = find_vertex_by_xy(*mesh, 0.0, 0.0);
  const index_t v1 = find_vertex_by_xy(*mesh, 1.0, 0.0);
  ASSERT_NE(v0, INVALID_INDEX);
  ASSERT_NE(v1, INVALID_INDEX);

  const gid_t g0 = mesh->vertex_gids().at(static_cast<size_t>(v0));
  const gid_t g1 = mesh->vertex_gids().at(static_cast<size_t>(v1));

  // Quarter points along the shared edge.
  const index_t v025 = find_vertex_by_xy(*mesh, 0.25, 0.0);
  const index_t v075 = find_vertex_by_xy(*mesh, 0.75, 0.0);
  ASSERT_NE(v025, INVALID_INDEX);
  ASSERT_NE(v075, INVALID_INDEX);

  auto check_flat = [&](index_t v, double w0, double w1) {
    const gid_t vg = mesh->vertex_gids().at(static_cast<size_t>(v));
    auto flat = MeshLabels::flatten_vertex_provenance_gid(*mesh, vg);
    std::map<gid_t, double> m;
    for (const auto& kv : flat) m[kv.first] += kv.second;
    ASSERT_EQ(m.size(), 2u);
    ASSERT_TRUE(m.count(g0) > 0);
    ASSERT_TRUE(m.count(g1) > 0);
    EXPECT_NEAR(m[g0], w0, 1e-12);
    EXPECT_NEAR(m[g1], w1, 1e-12);
    EXPECT_NEAR(m[g0] + m[g1], 1.0, 1e-12);
  };

  // v(0.25) = 0.75*v0 + 0.25*v1; v(0.75) = 0.25*v0 + 0.75*v1.
  check_flat(v025, 0.75, 0.25);
  check_flat(v075, 0.25, 0.75);
}

} // namespace test
} // namespace svmp
