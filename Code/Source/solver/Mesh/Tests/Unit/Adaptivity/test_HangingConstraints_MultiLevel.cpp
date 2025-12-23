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
 * @file test_HangingConstraints_MultiLevel.cpp
 * @brief Tests for solver-ready hanging-vertex constraints with multi-level imbalance
 *
 * This test constructs a two-triangle mesh, refines only one side twice (level diff = 2),
 * and checks that ConformityUtils::build_hanging_vertex_constraints produces fully-flattened
 * constraints (in terms of coarse edge endpoints) for multiple hanging points on the edge.
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/Conformity.h"
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/Options.h"
#include "../../../Adaptivity/QualityGuards.h"
#include "../../../Core/MeshBase.h"

#include <cmath>
#include <memory>
#include <vector>

namespace svmp {
namespace test {

namespace {

std::unique_ptr<MeshBase> make_two_triangle_edge_mesh() {
  auto mesh = std::make_unique<MeshBase>();

  // Two triangles sharing edge (0,1) on y=0.
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {0.0, 1.0, 0.0});   // above
  mesh->add_vertex(3, {0.0, -1.0, 0.0});  // below

  mesh->add_cell(0, CellFamily::Triangle, {0, 1, 2});  // upper
  mesh->add_cell(1, CellFamily::Triangle, {1, 0, 3});  // lower

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

AdaptivityOptions options_for_multilevel_hanging() {
  AdaptivityOptions options;
  options.enable_refinement = true;
  options.enable_coarsening = false;
  options.check_quality = false;
  options.verbosity = 0;
  options.max_refinement_level = 10;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::HIERARCHICAL; // stable IDs for multi-level stencils
  options.conformity_mode = AdaptivityOptions::ConformityMode::ALLOW_HANGING_NODES;
  options.max_hanging_level = 2; // allow level difference of 2
  return options;
}

} // namespace

TEST(HangingConstraints, MultiLevelEdgeProducesQuarterPointStencils) {
  auto mesh = make_two_triangle_edge_mesh();

  AdaptivityManager manager(options_for_multilevel_hanging());

  // Level 1: refine only the upper triangle (cell 0).
  {
    std::vector<bool> marks(mesh->n_cells(), false);
    marks[0] = true;
    auto r = manager.refine(*mesh, marks, nullptr);
    ASSERT_TRUE(r.success);
  }

  // Level 2: refine only level-1 cells.
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

  const index_t v0 = find_vertex_by_xy(*mesh, 0.0, 0.0);
  const index_t v1 = find_vertex_by_xy(*mesh, 1.0, 0.0);
  const index_t v025 = find_vertex_by_xy(*mesh, 0.25, 0.0);
  const index_t v05 = find_vertex_by_xy(*mesh, 0.50, 0.0);
  const index_t v075 = find_vertex_by_xy(*mesh, 0.75, 0.0);

  ASSERT_NE(v0, INVALID_INDEX);
  ASSERT_NE(v1, INVALID_INDEX);
  ASSERT_NE(v025, INVALID_INDEX);
  ASSERT_NE(v05, INVALID_INDEX);
  ASSERT_NE(v075, INVALID_INDEX);

  auto constraints = ConformityUtils::build_hanging_vertex_constraints(*mesh);

  // Expect stencils for the three interior points on the shared edge.
  ASSERT_EQ(constraints.size(), 3u);
  ASSERT_TRUE(constraints.count(static_cast<size_t>(v025)) > 0);
  ASSERT_TRUE(constraints.count(static_cast<size_t>(v05)) > 0);
  ASSERT_TRUE(constraints.count(static_cast<size_t>(v075)) > 0);

  auto check = [&](index_t hv, double w0, double w1) {
    const auto& eq = constraints.at(static_cast<size_t>(hv));
    ASSERT_EQ(eq.size(), 2u);
    ASSERT_TRUE(eq.count(static_cast<size_t>(v0)) > 0);
    ASSERT_TRUE(eq.count(static_cast<size_t>(v1)) > 0);
    EXPECT_NEAR(eq.at(static_cast<size_t>(v0)), w0, 1e-12);
    EXPECT_NEAR(eq.at(static_cast<size_t>(v1)), w1, 1e-12);
    EXPECT_NEAR(eq.at(static_cast<size_t>(v0)) + eq.at(static_cast<size_t>(v1)), 1.0, 1e-12);
  };

  check(v05, 0.5, 0.5);
  check(v025, 0.75, 0.25);
  check(v075, 0.25, 0.75);

  // GID-keyed constraints (for FE/solver layers that use stable GIDs).
  const gid_t g0 = mesh->vertex_gids().at(static_cast<size_t>(v0));
  const gid_t g1 = mesh->vertex_gids().at(static_cast<size_t>(v1));
  const gid_t g025 = mesh->vertex_gids().at(static_cast<size_t>(v025));
  const gid_t g05 = mesh->vertex_gids().at(static_cast<size_t>(v05));
  const gid_t g075 = mesh->vertex_gids().at(static_cast<size_t>(v075));

  auto gid_constraints = ConformityUtils::build_hanging_vertex_constraints_gid(*mesh);
  ASSERT_EQ(gid_constraints.size(), 3u);
  ASSERT_TRUE(gid_constraints.count(g025) > 0);
  ASSERT_TRUE(gid_constraints.count(g05) > 0);
  ASSERT_TRUE(gid_constraints.count(g075) > 0);

  auto check_gid = [&](gid_t hv, double w0, double w1) {
    const auto& eq = gid_constraints.at(hv);
    ASSERT_EQ(eq.size(), 2u);
    ASSERT_TRUE(eq.count(g0) > 0);
    ASSERT_TRUE(eq.count(g1) > 0);
    EXPECT_NEAR(eq.at(g0), w0, 1e-12);
    EXPECT_NEAR(eq.at(g1), w1, 1e-12);
    EXPECT_NEAR(eq.at(g0) + eq.at(g1), 1.0, 1e-12);
  };

  check_gid(g05, 0.5, 0.5);
  check_gid(g025, 0.75, 0.25);
  check_gid(g075, 0.25, 0.75);

  // Ensure we did not generate constraints on true domain boundaries in this mesh.
  for (const auto& kv : constraints) {
    const index_t v = static_cast<index_t>(kv.first);
    auto p = mesh->get_vertex_coords(v);
    EXPECT_NEAR(static_cast<double>(p[1]), 0.0, 1e-12) << "Only shared-edge vertices should be constrained";
  }
}

} // namespace test
} // namespace svmp
