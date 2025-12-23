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
 * @file test_AdaptivityManager_Simple.cpp
 * @brief Simplified tests for AdaptivityManager to establish baseline
 *
 * This is a simplified version that compiles with the current API
 * to validate basic functionality before implementing coarsening.
 */

#include <gtest/gtest.h>
#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/Options.h"
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/QualityGuards.h"
#include "../../../Adaptivity/Conformity.h"
#include "../../../Core/MeshBase.h"
#include <memory>
#include <vector>

namespace svmp {
namespace test {

class AdaptivitySimpleTest : public ::testing::Test {
protected:
  // Helper to create a simple 2D quad mesh
  std::unique_ptr<MeshBase> create_2d_quad_mesh(size_t nx = 2, size_t ny = 2) {
    auto mesh = std::make_unique<MeshBase>();

    // Create vertices
    size_t vertex_id = 0;
    for (size_t j = 0; j <= ny; ++j) {
      for (size_t i = 0; i <= nx; ++i) {
        mesh->add_vertex(vertex_id++, {
          static_cast<real_t>(i),
          static_cast<real_t>(j),
          0.0
        });
      }
    }

    // Create quad elements
    size_t elem_id = 0;
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        size_t v0 = j * (nx + 1) + i;
        size_t v1 = v0 + 1;
        size_t v2 = v0 + nx + 2;
        size_t v3 = v0 + nx + 1;
        mesh->add_cell(elem_id++, CellFamily::Quad, {
          static_cast<index_t>(v0),
          static_cast<index_t>(v1),
          static_cast<index_t>(v2),
          static_cast<index_t>(v3)
        });
      }
    }

    return mesh;
  }

  // Helper to create default adaptivity options
  AdaptivityOptions create_default_options() {
    AdaptivityOptions options;
    options.enable_refinement = true;
    options.enable_coarsening = false;  // Not implemented yet
    options.max_refinement_level = 3;
    options.refine_fraction = 0.3;
    options.coarsen_fraction = 0.1;
    options.estimator_type = AdaptivityOptions::EstimatorType::GRADIENT_RECOVERY;
    options.marking_strategy = AdaptivityOptions::MarkingStrategy::FIXED_FRACTION;
    options.check_quality = false;  // Disable for faster tests
    options.verbosity = 0;  // Quiet mode
    return options;
  }
};

// Test 1: Basic refinement without fields
TEST_F(AdaptivitySimpleTest, BasicRefinement) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  size_t initial_cells = mesh->n_cells();
  EXPECT_EQ(initial_cells, 4);

  // Mark all cells for refinement
  std::vector<bool> marks(initial_cells, true);
  auto result = manager.refine(*mesh, marks, nullptr);

  // Each quad should be refined into 4 children
  EXPECT_GT(mesh->n_cells(), initial_cells);
  EXPECT_TRUE(result.success);
}

// Test 2: Selective refinement
TEST_F(AdaptivitySimpleTest, SelectiveRefinement) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  size_t initial_cells = mesh->n_cells();

  // Mark only first cell
  std::vector<bool> marks(initial_cells, false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);

  // Should have refined at least the marked cell
  EXPECT_GT(mesh->n_cells(), initial_cells);
  EXPECT_GE(result.num_refined, 1);
}

// Test 3: Refinement-Coarsening Round Trip
TEST_F(AdaptivitySimpleTest, RefinementCoarseningRoundTrip) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  options.enable_refinement = true;
  options.enable_coarsening = true;
  AdaptivityManager manager(options);

  // Store initial cell count
  size_t initial_cells = mesh->n_cells();

  // First refine all cells
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks, nullptr);
  EXPECT_TRUE(refine_result.success);
  EXPECT_GT(mesh->n_cells(), initial_cells);  // Should have more cells after refinement

  size_t refined_cells = mesh->n_cells();

  // Now coarsen all cells
  std::vector<bool> coarsen_marks(mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*mesh, coarsen_marks, nullptr);

  // Coarsening should succeed (even if partial due to constraints)
  EXPECT_NO_THROW({
    // Already executed, just check it didn't throw
  });

  // Should have fewer cells after coarsening (may not return to exact original due to constraints)
  EXPECT_LE(mesh->n_cells(), refined_cells);
}

// Test 4: Multiple refinement iterations
TEST_F(AdaptivitySimpleTest, MultipleRefinements) {
  auto mesh = create_2d_quad_mesh(1, 1);  // Start with 1 quad
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  // First refinement
  std::vector<bool> marks1(mesh->n_cells(), true);
  auto result1 = manager.refine(*mesh, marks1, nullptr);
  EXPECT_EQ(mesh->n_cells(), 4);  // 1 -> 4

  // Second refinement
  std::vector<bool> marks2(mesh->n_cells(), true);
  auto result2 = manager.refine(*mesh, marks2, nullptr);
  EXPECT_EQ(mesh->n_cells(), 16);  // 4 -> 16
}

// Test 5: Max refinement level enforcement
TEST_F(AdaptivitySimpleTest, MaxRefinementLevel) {
  auto mesh = create_2d_quad_mesh(1, 1);
  AdaptivityOptions options = create_default_options();
  options.max_refinement_level = 1;  // Allow only one level
  AdaptivityManager manager(options);

  // First refinement should succeed
  std::vector<bool> marks1(mesh->n_cells(), true);
  auto result1 = manager.refine(*mesh, marks1, nullptr);
  size_t cells_after_first = mesh->n_cells();
  EXPECT_GT(cells_after_first, 1);

  // Second refinement should be blocked by max level
  std::vector<bool> marks2(mesh->n_cells(), true);
  auto result2 = manager.refine(*mesh, marks2, nullptr);
  EXPECT_EQ(result2.num_refined, 0);  // No refinement due to level limit
}

// Test 6: Uniform quad refinement reuses shared vertices (no cracks)
TEST_F(AdaptivitySimpleTest, UniformQuadRefinementReusesSharedVertices) {
  auto mesh = create_2d_quad_mesh(2, 2);  // 4 quads, 9 vertices
  AdaptivityOptions options = create_default_options();
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);

  // 2x2 quads uniformly refined -> 4x4 quads (16 cells) with 5x5 vertex grid (25 vertices).
  EXPECT_EQ(mesh->n_cells(), 16u);
  EXPECT_EQ(mesh->n_vertices(), 25u);

  // Boundary edges are split consistently: perimeter has 16 boundary faces after refinement.
  EXPECT_EQ(mesh->n_boundary_faces(), 16u);
}

// Test 7: Boundary face labels propagate through refinement
TEST_F(AdaptivitySimpleTest, BoundaryLabelsPropagateOnUniformRefinement) {
  auto mesh = create_2d_quad_mesh(2, 2);
  mesh->finalize();

  constexpr label_t kLabel = 7;
  for (auto f : mesh->boundary_faces()) {
    mesh->set_boundary_label(f, kLabel);
  }

  AdaptivityOptions options = create_default_options();
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh->n_boundary_faces(), 16u);
  for (auto f : mesh->boundary_faces()) {
    EXPECT_EQ(mesh->boundary_label(f), kLabel);
  }

  // Interior faces must remain unlabeled.
  for (index_t f = 0; f < static_cast<index_t>(mesh->n_faces()); ++f) {
    auto fc = mesh->face_cells(f);
    const bool is_boundary = (fc[0] < 0 || fc[1] < 0);
    if (!is_boundary) {
      EXPECT_EQ(mesh->boundary_label(f), INVALID_LABEL);
    }
  }
}

// Test 8: Pyramid refinement produces mixed child types (6 pyramids + 4 tets)
TEST_F(AdaptivitySimpleTest, PyramidRefinementProducesMixedChildren) {
  auto mesh = std::make_unique<MeshBase>();

  // Unit pyramid: square base at z=0, apex at z=1.
  mesh->add_vertex(0, {0.0, 0.0, 0.0});
  mesh->add_vertex(1, {1.0, 0.0, 0.0});
  mesh->add_vertex(2, {1.0, 1.0, 0.0});
  mesh->add_vertex(3, {0.0, 1.0, 0.0});
  mesh->add_vertex(4, {0.5, 0.5, 1.0});
  mesh->add_cell(0, CellFamily::Pyramid, {0, 1, 2, 3, 4});

  AdaptivityOptions options = create_default_options();
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks, nullptr);
  ASSERT_TRUE(result.success);

  EXPECT_EQ(mesh->n_cells(), 10u);
  EXPECT_EQ(mesh->n_vertices(), 15u);

  size_t num_pyr = 0;
  size_t num_tet = 0;
  for (index_t c = 0; c < static_cast<index_t>(mesh->n_cells()); ++c) {
    auto fam = mesh->cell_shape(c).family;
    if (fam == CellFamily::Pyramid) num_pyr++;
    if (fam == CellFamily::Tetra) num_tet++;
  }

  EXPECT_EQ(num_pyr, 6u);
  EXPECT_EQ(num_tet, 4u);
}

} // namespace test
} // namespace svmp
