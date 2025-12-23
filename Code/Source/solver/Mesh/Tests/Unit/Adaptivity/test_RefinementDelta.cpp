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
 * @file test_RefinementDelta.cpp
 * @brief Tests for GID stability + RefinementDelta emission from Mesh adaptivity
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/Options.h"
#include "../../../Adaptivity/RefinementDelta.h"
#include "../../../Core/MeshBase.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

namespace svmp {
namespace test {

namespace {

AdaptivityOptions base_options() {
  AdaptivityOptions options;
  options.enable_refinement = true;
  options.enable_coarsening = false;
  options.check_quality = false;
  options.verbosity = 0;
  options.max_refinement_level = 5;
  options.max_level_difference = 1;
  options.use_green_closure = true;
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED_GREEN;
  // Keep per-refinement quality gates enabled; the unit tetra should pass.
  options.enforce_quality_after_refinement = true;
  options.min_refined_quality = 0.0;
  return options;
}

std::unique_ptr<MeshBase> make_two_disjoint_tets_with_custom_gids() {
  auto mesh = std::make_unique<MeshBase>();

  // Tet 0 (unit tetra at origin)
  mesh->add_vertex(100, {0.0, 0.0, 0.0});  // v0
  mesh->add_vertex(101, {1.0, 0.0, 0.0});  // v1
  mesh->add_vertex(102, {0.0, 1.0, 0.0});  // v2
  mesh->add_vertex(103, {0.0, 0.0, 1.0});  // v3

  // Tet 1 (translated)
  mesh->add_vertex(200, {10.0, 0.0, 0.0}); // v4
  mesh->add_vertex(201, {11.0, 0.0, 0.0}); // v5
  mesh->add_vertex(202, {10.0, 1.0, 0.0}); // v6
  mesh->add_vertex(203, {10.0, 0.0, 1.0}); // v7

  mesh->add_cell(1000, CellFamily::Tetra, {0, 1, 2, 3});
  mesh->add_cell(2000, CellFamily::Tetra, {4, 5, 6, 7});

  mesh->finalize();
  return mesh;
}

} // namespace

TEST(RefinementDelta, RefinementPreservesExistingGIDsAndEmitsDelta) {
  auto mesh = make_two_disjoint_tets_with_custom_gids();
  const auto old_vgids = mesh->vertex_gids();
  const auto old_cgids = mesh->cell_gids();

  AdaptivityManager manager(base_options());
  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);

  // One tet refined into 8 children; one untouched tet remains.
  EXPECT_EQ(mesh->n_cells(), 9u);
  EXPECT_EQ(mesh->n_vertices(), 14u);

  // Original vertex GIDs preserved (by value).
  for (gid_t g : old_vgids) {
    EXPECT_NE(std::find(mesh->vertex_gids().begin(), mesh->vertex_gids().end(), g), mesh->vertex_gids().end());
  }

  // Untouched cell keeps its original GID.
  EXPECT_NE(std::find(mesh->cell_gids().begin(), mesh->cell_gids().end(), old_cgids[1]), mesh->cell_gids().end());

  // New cell GIDs are unique and greater than any old GID.
  const gid_t max_old_cell_gid = *std::max_element(old_cgids.begin(), old_cgids.end());
  std::set<gid_t> unique_cell_gids(mesh->cell_gids().begin(), mesh->cell_gids().end());
  EXPECT_EQ(unique_cell_gids.size(), mesh->cell_gids().size());
  for (gid_t g : mesh->cell_gids()) {
    if (g != old_cgids[1]) {
      EXPECT_GE(g, max_old_cell_gid + 1);
    }
  }

  // Delta emitted with one refined parent and 6 edge-midpoint vertices.
  ASSERT_TRUE(result.refinement_delta != nullptr);
  ASSERT_EQ(result.refinement_delta->refined_cells.size(), 1u);
  ASSERT_EQ(result.refinement_delta->new_vertices.size(), 6u);

  const auto& crec = result.refinement_delta->refined_cells[0];
  EXPECT_EQ(crec.parent_cell_gid, old_cgids[0]);
  EXPECT_EQ(crec.family, CellFamily::Tetra);
  EXPECT_EQ(crec.spec.pattern, RefinementPattern::RED);
  EXPECT_EQ(crec.child_cell_gids.size(), 8u);

  // Provenance for the 6 tet edge midpoints: each depends on exactly 2 parents with weight 0.5.
  std::set<std::pair<gid_t, gid_t>> expected_edges = {
      {100, 101}, {100, 102}, {100, 103},
      {101, 102}, {101, 103}, {102, 103},
  };
  std::set<std::pair<gid_t, gid_t>> observed_edges;
  for (const auto& vrec : result.refinement_delta->new_vertices) {
    ASSERT_EQ(vrec.parent_vertex_weights.size(), 2u);
    gid_t a = vrec.parent_vertex_weights[0].first;
    gid_t b = vrec.parent_vertex_weights[1].first;
    double wa = vrec.parent_vertex_weights[0].second;
    double wb = vrec.parent_vertex_weights[1].second;
    if (a > b) std::swap(a, b);
    observed_edges.insert({a, b});
    EXPECT_NEAR(wa, 0.5, 1e-14);
    EXPECT_NEAR(wb, 0.5, 1e-14);
  }
  EXPECT_EQ(observed_edges, expected_edges);
}

TEST(RefinementDelta, CoarseningRestoresParentCellGID) {
  auto mesh = std::make_unique<MeshBase>();

  // Single tetra with nontrivial GIDs.
  mesh->add_vertex(10, {0.0, 0.0, 0.0});
  mesh->add_vertex(11, {1.0, 0.0, 0.0});
  mesh->add_vertex(12, {0.0, 1.0, 0.0});
  mesh->add_vertex(13, {0.0, 0.0, 1.0});
  mesh->add_cell(42, CellFamily::Tetra, {0, 1, 2, 3});
  mesh->finalize();

  AdaptivityOptions opts = base_options();
  AdaptivityManager manager(opts);

  // Refine.
  std::vector<bool> refine_marks(mesh->n_cells(), false);
  refine_marks[0] = true;
  auto refined = manager.refine(*mesh, refine_marks, nullptr);
  ASSERT_TRUE(refined.success);
  ASSERT_EQ(mesh->n_cells(), 8u);

  // Coarsen all children back to the parent.
  std::vector<bool> coarsen_marks(mesh->n_cells(), true);
  auto coarsened = manager.coarsen(*mesh, coarsen_marks, nullptr);
  ASSERT_TRUE(coarsened.success);

  ASSERT_EQ(mesh->n_cells(), 1u);
  ASSERT_EQ(mesh->cell_gids().size(), 1u);
  EXPECT_EQ(mesh->cell_gids()[0], 42) << "Parent cell GID should be restored after coarsening";

  // Corner vertex GIDs preserved.
  std::set<gid_t> vg(mesh->vertex_gids().begin(), mesh->vertex_gids().end());
  EXPECT_TRUE(vg.count(10));
  EXPECT_TRUE(vg.count(11));
  EXPECT_TRUE(vg.count(12));
  EXPECT_TRUE(vg.count(13));
}

} // namespace test
} // namespace svmp

