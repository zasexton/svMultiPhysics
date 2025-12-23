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
 * @file test_RedGreenClosure.cpp
 * @brief Tests for true RED–GREEN closure (selector-aware GREEN/BLUE specs)
 *
 * The ClosureConformityEnforcer now computes per-cell RefinementSpec for closure:
 * - Triangles: 1 edge split -> GREEN(edge), 2 edges split -> BLUE(shared vertex), 3 -> RED.
 *
 * These tests validate both:
 * - Spec selection (GREEN/BLUE selector correctness)
 * - End-to-end AdaptivityManager refinement results (cell/vertex counts, no cracks)
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

std::unique_ptr<MeshBase> make_two_triangle_mesh() {
  auto mesh = std::make_unique<MeshBase>();

  // Square split by diagonal (0,2):
  // 3----2
  // |  / |
  // | /  |
  // 0----1
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {1.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 1.0, 0.0});

  mesh->add_cell(0, CellFamily::Triangle, {0, 1, 2});
  mesh->add_cell(1, CellFamily::Triangle, {0, 2, 3});

  return mesh;
}

std::unique_ptr<MeshBase> make_three_triangle_fan() {
  auto mesh = std::make_unique<MeshBase>();

  // Three triangles around vertex 0:
  //   2
  //  / \
  // 3---1
  //   0
  // Place vertex 0 below the (3,1) edge to avoid a degenerate (collinear) triangle.
  mesh->add_vertex(0, {0.0, -1.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {0.0, 2.0, 0.0});
  mesh->add_vertex(3, {-1.0, 0.0, 0.0});

  mesh->add_cell(0, CellFamily::Triangle, {0, 1, 2});
  mesh->add_cell(1, CellFamily::Triangle, {0, 2, 3});
  // Keep consistent (counter-clockwise) orientation so quality gates do not
  // interpret the element as inverted.
  mesh->add_cell(2, CellFamily::Triangle, {0, 1, 3});

  return mesh;
}

size_t count_vertices_near(const MeshBase& mesh, const std::array<double, 3>& p, double tol = 1e-12) {
  size_t count = 0;
  for (index_t v = 0; v < static_cast<index_t>(mesh.n_vertices()); ++v) {
    auto x = mesh.get_vertex_coords(v);
    const double dx = static_cast<double>(x[0]) - p[0];
    const double dy = static_cast<double>(x[1]) - p[1];
    const double dz = static_cast<double>(x[2]) - p[2];
    const double d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < tol * tol) count++;
  }
  return count;
}

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
  return options;
}

} // namespace

TEST(RedGreenClosure, ClosureComputesGreenSpecForNeighbor) {
  auto mesh = make_two_triangle_mesh();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[0] = MarkType::REFINE;

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;

  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(0u) > 0);
  ASSERT_TRUE(specs.count(1u) > 0);

  EXPECT_EQ(specs.at(0).pattern, RefinementPattern::RED);
  EXPECT_EQ(specs.at(1).pattern, RefinementPattern::GREEN);
  EXPECT_EQ(specs.at(1).selector, 0u) << "Cell (0,2,3) should split local edge (0,1) == global edge (0,2)";
  EXPECT_EQ(marks[1], MarkType::REFINE);
}

TEST(RedGreenClosure, AdaptivityManagerRefinesGreenNeighborAndProducesConformingMesh) {
  auto mesh = make_two_triangle_mesh();

  AdaptivityOptions options = base_options();
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);

  // One RED triangle (4) + one GREEN triangle (2) = 6.
  EXPECT_EQ(mesh->n_cells(), 6u);
  // Unique split edges: (0,1), (1,2), (0,2) => +3 vertices from 4 => 7.
  EXPECT_EQ(mesh->n_vertices(), 7u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));

  // Shared-edge midpoint should exist exactly once (no duplicate vertices).
  EXPECT_EQ(count_vertices_near(*mesh, {0.5, 0.5, 0.0}), 1u);
}

TEST(RedGreenClosure, ClosureComputesBlueSpecForTwoEdgeSplitRequirement) {
  auto mesh = make_three_triangle_fan();

  std::vector<MarkType> marks(mesh->n_cells(), MarkType::NONE);
  marks[0] = MarkType::REFINE;
  marks[1] = MarkType::REFINE;

  AdaptivityOptions options = base_options();
  ClosureConformityEnforcer enforcer;

  (void)enforcer.enforce_conformity(*mesh, marks, options);

  const auto specs = enforcer.get_cell_refinement_specs();
  ASSERT_TRUE(specs.count(2u) > 0);
  EXPECT_EQ(marks[2], MarkType::REFINE);

  EXPECT_EQ(specs.at(2).pattern, RefinementPattern::BLUE);
  EXPECT_EQ(specs.at(2).selector, 0u) << "Cell (0,1,3) should split edges meeting at local vertex 0";
}

TEST(RedGreenClosure, AdaptivityManagerUsesBlueClosureAndProducesConformingMesh) {
  auto mesh = make_three_triangle_fan();

  AdaptivityOptions options = base_options();
  // Keep inversion/Jacobian checks but relax the aggregate quality threshold so the
  // BLUE closure pattern is actually committed (this test focuses on topology).
  options.min_refined_quality = 0.0;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;
  marks[1] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(manager.get_last_marks().size(), 3u);
  EXPECT_EQ(manager.get_last_marks()[2], MarkType::REFINE) << "Closure should refine the third triangle";
  EXPECT_EQ(result.num_refined, 3u);

  // 2 RED triangles -> 8 children, 1 BLUE -> 3 children => 11 total.
  EXPECT_EQ(mesh->n_cells(), 11u);
  // Unique split edges: (0,1), (0,2), (0,3), (1,2), (2,3) => +5 vertices from 4 => 9.
  EXPECT_EQ(mesh->n_vertices(), 9u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));
}

TEST(RedGreenClosure, AdaptivityManagerUpgradesBlueClosureToRedWhenQualityGateTrips) {
  auto mesh = make_three_triangle_fan();

  AdaptivityOptions options = base_options();
  // Default min_refined_quality can reject BLUE closure for some geometries; the
  // manager must still enforce conformity by upgrading to RED.
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;
  marks[1] = true;

  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);
  ASSERT_EQ(manager.get_last_marks().size(), 3u);
  EXPECT_EQ(manager.get_last_marks()[2], MarkType::REFINE);

  // 2 RED triangles -> 8 children, closure upgraded to RED -> 4 children => 12 total.
  EXPECT_EQ(mesh->n_cells(), 12u);
  // Upgrading to RED also splits the previously-unsplit edge (1,3) => +1 vertex.
  EXPECT_EQ(mesh->n_vertices(), 10u);
  EXPECT_TRUE(ConformityUtils::is_mesh_conforming(*mesh));
}

} // namespace test
} // namespace svmp
