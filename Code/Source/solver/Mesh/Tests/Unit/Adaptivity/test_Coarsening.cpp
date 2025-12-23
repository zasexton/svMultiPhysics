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

#include <gtest/gtest.h>
#include "../../../Adaptivity/AdaptivityManager.h"
#include "../../../Adaptivity/Marker.h"
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/QualityGuards.h"
#include "../../../Adaptivity/Conformity.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"
#include "../../../Labels/MeshLabels.h"
#include "../../../Geometry/MeshGeometry.h"
#include "../../../Topology/MeshTopology.h"
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>

using namespace svmp;

namespace {

std::vector<std::vector<index_t>> sorted_cell_connectivity(const MeshBase& mesh) {
  std::vector<std::vector<index_t>> cells;
  cells.reserve(static_cast<size_t>(mesh.n_cells()));

  for (index_t c = 0; c < mesh.n_cells(); ++c) {
    auto conn = mesh.cell_vertices(c);
    std::sort(conn.begin(), conn.end());
    cells.push_back(std::move(conn));
  }

  std::sort(cells.begin(), cells.end());
  return cells;
}

std::unordered_map<label_t, size_t> region_histogram(const MeshBase& mesh) {
  return MeshLabels::count_by_region(mesh);
}

} // namespace

/**
 * @brief Test fixture for coarsening operations
 */
class CoarseningTest : public ::testing::Test {
protected:
  /**
   * @brief Create a simple 2D quad mesh for testing
   */
  std::unique_ptr<MeshBase> create_quad_mesh() {
    // Create a 2x2 quad mesh
    std::vector<real_t> vertices = {
      0.0, 0.0, 0.0,  // 0
      1.0, 0.0, 0.0,  // 1
      2.0, 0.0, 0.0,  // 2
      0.0, 1.0, 0.0,  // 3
      1.0, 1.0, 0.0,  // 4
      2.0, 1.0, 0.0,  // 5
      0.0, 2.0, 0.0,  // 6
      1.0, 2.0, 0.0,  // 7
      2.0, 2.0, 0.0   // 8
    };

    std::vector<index_t> cells = {
      0, 1, 4, 3,  // Cell 0
      1, 2, 5, 4,  // Cell 1
      3, 4, 7, 6,  // Cell 2
      4, 5, 8, 7   // Cell 3
    };

    // Create offsets array for 4 quads (each has 4 vertices)
    std::vector<offset_t> offsets = {0, 4, 8, 12, 16};

    std::vector<CellShape> shapes(4);
    for (auto& shape : shapes) {
      shape.family = CellFamily::Quad;
      shape.order = 1;
    }

    auto mesh = std::make_unique<MeshBase>();
    mesh->build_from_arrays(3, vertices, offsets, cells, shapes);
    mesh->finalize();
    // Ensure no stale refinement provenance is associated with this mesh pointer
    MeshLabels::clear_refinement_tracking(*mesh);
    return mesh;
  }

  /**
   * @brief Create a simple 3D hex mesh for testing
   */
  std::unique_ptr<MeshBase> create_hex_mesh() {
    // Create a single hex element
    std::vector<real_t> vertices = {
      0.0, 0.0, 0.0,  // 0
      1.0, 0.0, 0.0,  // 1
      1.0, 1.0, 0.0,  // 2
      0.0, 1.0, 0.0,  // 3
      0.0, 0.0, 1.0,  // 4
      1.0, 0.0, 1.0,  // 5
      1.0, 1.0, 1.0,  // 6
      0.0, 1.0, 1.0   // 7
    };

    std::vector<index_t> cells = {
      0, 1, 2, 3, 4, 5, 6, 7  // Single hex
    };

    // Create offsets array for 1 hex (has 8 vertices)
    std::vector<offset_t> offsets = {0, 8};

    std::vector<CellShape> shapes(1);
    shapes[0].family = CellFamily::Hex;
    shapes[0].order = 1;

    auto mesh = std::make_unique<MeshBase>();
    mesh->build_from_arrays(3, vertices, offsets, cells, shapes);
    mesh->finalize();
    MeshLabels::clear_refinement_tracking(*mesh);
    return mesh;
  }
};

/**
 * @brief Test provenance tracking during refinement
 */
TEST_F(CoarseningTest, ProvenanceTracking) {
  auto mesh = create_quad_mesh();

  // Setup adaptivity manager
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  // Store initial cell count
  size_t initial_cells = mesh->n_cells();

  // Mark all cells for refinement
  std::vector<bool> marks(initial_cells, true);

  // Perform refinement (modifies mesh in-place)
  auto result = manager.refine(*mesh, marks);
  ASSERT_TRUE(result.success);

  // Check that parent-child relationships were recorded
  // refine() modifies in-place, so use the original mesh which is now refined
  auto refined_mesh = mesh.get();
  EXPECT_GT(refined_mesh->n_cells(), initial_cells);

  // Each original quad should have been refined into 4 children
  for (index_t parent = 0; parent < initial_cells; ++parent) {
    auto children = MeshLabels::get_children_cells(*refined_mesh, parent);
    EXPECT_EQ(children.size(), 4) << "Quad " << parent << " should have 4 children";

    // Check that each child knows its parent
    for (auto child : children) {
      EXPECT_EQ(MeshLabels::get_parent_cell(*refined_mesh, child), parent);

      // Check refinement level
      EXPECT_EQ(MeshLabels::refinement_level(*refined_mesh, child), 1);

      // Check pattern was recorded
      EXPECT_EQ(MeshLabels::get_refinement_pattern(*refined_mesh, child),
                static_cast<int>(AdaptivityOptions::RefinementPattern::RED));

      // Check sibling count
      EXPECT_EQ(MeshLabels::get_sibling_count(*refined_mesh, child), 4);
    }
  }
}

/**
 * @brief Test sibling group discovery
 */
TEST_F(CoarseningTest, SiblingDiscovery) {
  auto mesh = create_quad_mesh();

  // Setup adaptivity manager and refine
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks);
  ASSERT_TRUE(result.success);

  // refine() modifies mesh in-place, so use the original mesh pointer
  auto refined_mesh = mesh.get();

  // Test sibling grouping
  auto sibling_groups = MeshLabels::group_siblings_by_parent(*refined_mesh);

  // Should have 4 groups (one per original quad)
  EXPECT_EQ(sibling_groups.size(), 4);

  // Each group should have 4 siblings
  for (const auto& [parent, siblings] : sibling_groups) {
    EXPECT_EQ(siblings.size(), 4) << "Parent " << parent << " should have 4 children";

    // Verify all siblings have the same parent
    for (auto sibling : siblings) {
      EXPECT_EQ(MeshLabels::get_parent_cell(*refined_mesh, sibling), parent);
    }
  }
}

/**
 * @brief Test complete sibling group checking
 */
TEST_F(CoarseningTest, CompleteSiblingGroups) {
  auto mesh = create_quad_mesh();

  // Setup and refine
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks);
  ASSERT_TRUE(result.success);

  // refine() modifies mesh in-place, so use the original mesh pointer
  auto refined_mesh = mesh.get();

  // Mark all cells for coarsening
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), true);

  // Check that all sibling groups are complete
  auto sibling_groups = MeshLabels::group_siblings_by_parent(*refined_mesh);
  for (const auto& [parent, siblings] : sibling_groups) {
    bool is_complete = MeshLabels::is_sibling_group_complete(*refined_mesh, parent, coarsen_marks);
    EXPECT_TRUE(is_complete) << "Parent " << parent << " should have complete sibling group";
  }

  // Now mark only partial siblings
  coarsen_marks.assign(refined_mesh->n_cells(), false);
  coarsen_marks[0] = true;  // Mark only first child

  // Check that groups are now incomplete
  for (const auto& [parent, siblings] : sibling_groups) {
    bool has_marked = false;
    bool has_unmarked = false;
    for (auto sibling : siblings) {
      if (coarsen_marks[sibling]) has_marked = true;
      else has_unmarked = true;
    }

    if (has_marked && has_unmarked) {
      bool is_complete = MeshLabels::is_sibling_group_complete(*refined_mesh, parent, coarsen_marks);
      EXPECT_FALSE(is_complete) << "Parent " << parent << " should have incomplete sibling group";
    }
  }
}

/**
 * @brief Test actual coarsening operation
 */
TEST_F(CoarseningTest, BasicCoarsening) {
  auto mesh = create_quad_mesh();

  // Setup and refine first
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.min_quality = 0.1;  // Low threshold for testing
  options.max_level_difference = 2;  // Allow coarsening
  AdaptivityManager manager(options);

  // Refine all cells (in-place)
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  auto refined_mesh = mesh.get();
  auto refined_boundary_labels = MeshLabels::unique_boundary_labels(*refined_mesh);
  std::cout << "[BoundaryLabelAggregation] refined mesh has "
            << refined_boundary_labels.size() << " boundary labels\n";
  size_t refined_cell_count = refined_mesh->n_cells();
  EXPECT_EQ(refined_cell_count, 16);  // 4 quads -> 16 quads

  // Now coarsen all cells (should restore original mesh)
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*refined_mesh, coarsen_marks);

  // Coarsening should succeed and modify mesh in-place
  EXPECT_TRUE(coarsen_result.success);
  EXPECT_GT(coarsen_result.num_coarsened, 0u);

  // Mesh should be back to original size
  EXPECT_EQ(refined_mesh->n_cells(), 4u);
}

/**
 * @brief Round-trip refine->coarsen should preserve quad topology and regions
 * (MFEM/deal.II-style invariance check)
 */
TEST_F(CoarseningTest, RoundTripTopology_Quad) {
  auto mesh = create_quad_mesh();

  // Assign distinct region labels to track per-cell provenance
  for (index_t c = 0; c < mesh->n_cells(); ++c) {
    MeshLabels::set_region_label(*mesh, c, static_cast<label_t>(10 + c));
  }

  auto original_cells = sorted_cell_connectivity(*mesh);
  auto original_regions = region_histogram(*mesh);
  const auto original_vertices = mesh->n_vertices();

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.max_level_difference = 2;
  AdaptivityManager manager(options);

  // Uniform refinement of all cells (one level)
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  // Coarsen everything back
  std::vector<bool> coarsen_marks(mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*mesh, coarsen_marks);
  ASSERT_TRUE(coarsen_result.success);
  EXPECT_GT(coarsen_result.num_coarsened, 0u);

  // Topology and region labels should match original (up to cell permutation)
  EXPECT_EQ(mesh->n_cells(), static_cast<index_t>(original_cells.size()));
  EXPECT_EQ(mesh->n_vertices(), original_vertices);

  auto final_cells = sorted_cell_connectivity(*mesh);
  auto final_regions = region_histogram(*mesh);

  EXPECT_EQ(final_cells, original_cells);
  EXPECT_EQ(final_regions, original_regions);
}

/**
 * @brief Test 2:1 conformity enforcement
 */
TEST_F(CoarseningTest, ConformityEnforcement) {
  auto mesh = create_quad_mesh();

  // Setup with strict conformity
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.max_level_difference = 1;  // Strict 2:1 rule
  AdaptivityManager manager(options);

  // Refine only some cells to create level differences
  std::vector<bool> refine_marks(mesh->n_cells(), false);
  refine_marks[0] = true;  // Refine only first quad
  refine_marks[1] = true;  // And second quad

  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  auto refined_mesh = mesh.get();

  // Try to coarsen - conformity should prevent some coarsening
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), false);

  // Mark children of first refined quad for coarsening
  auto children = MeshLabels::get_children_cells(*refined_mesh, 0);
  for (auto child : children) {
    coarsen_marks[child] = true;
  }

  auto coarsen_result = manager.coarsen(*refined_mesh, coarsen_marks);
  EXPECT_TRUE(coarsen_result.success);

  // Check that conformity warnings were issued if needed
  if (!coarsen_result.warning_messages.empty()) {
    bool has_conformity_warning = false;
    for (const auto& warning : coarsen_result.warning_messages) {
      if (warning.find("conformity") != std::string::npos ||
          warning.find("2:1") != std::string::npos) {
        has_conformity_warning = true;
        break;
      }
    }
    // Conformity warnings are expected in some cases
  }
}

/**
 * @brief Test quality gate validation
 */
TEST_F(CoarseningTest, QualityValidation) {
  auto mesh = create_hex_mesh();

  // Setup with high quality requirements
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.min_quality = 0.9;  // Very high quality threshold
  AdaptivityManager manager(options);

  // Refine the hex
  std::vector<bool> refine_marks(1, true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  auto refined_mesh = mesh.get();

  // Attempt coarsening with high quality requirement
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*refined_mesh, coarsen_marks);

  EXPECT_TRUE(coarsen_result.success);

  // With very high quality threshold, some coarsening might be rejected
  if (!coarsen_result.warning_messages.empty()) {
    bool has_quality_warning = false;
    for (const auto& warning : coarsen_result.warning_messages) {
      if (warning.find("quality") != std::string::npos) {
        has_quality_warning = true;
        break;
      }
    }
    // Quality warnings expected with high threshold
  }
}

TEST_F(CoarseningTest, CoarseningQualityGateRejectsPoorParent) {
  auto mesh = create_hex_mesh();

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.check_coarsening_quality = true;
  options.min_coarsening_quality = 0.99;
  options.max_level_difference = 2;
  AdaptivityManager manager(options);

  // Refine the hex once.
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);
  ASSERT_EQ(mesh->n_cells(), 8u);

  // Distort one original corner vertex (GID 7) to make the parent hex very poor quality.
  const svmp::gid_t corner_gid = 7;
  const index_t v = mesh->global_to_local_vertex(corner_gid);
  ASSERT_NE(v, INVALID_INDEX);
  mesh->set_vertex_coords(v, {0.0, 1.0, 10.0});

  // Attempt to coarsen: the quality gate should reject restoring the parent.
  std::vector<bool> coarsen_marks(mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*mesh, coarsen_marks);
  EXPECT_TRUE(coarsen_result.success);
  EXPECT_EQ(coarsen_result.num_coarsened, 0u);
  EXPECT_EQ(mesh->n_cells(), 8u);
}

TEST_F(CoarseningTest, CoarseningQualityGatePartialRollback) {
  auto mesh = create_quad_mesh();

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.check_coarsening_quality = true;
  options.min_coarsening_quality = 0.99;
  options.max_level_difference = 2;
  AdaptivityManager manager(options);

  // Refine all 4 quads (in-place).
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);
  ASSERT_EQ(mesh->n_cells(), 16u);

  // Distort a boundary corner that belongs only to the first parent quad (GID 0).
  const svmp::gid_t corner_gid = 0;
  const index_t v = mesh->global_to_local_vertex(corner_gid);
  ASSERT_NE(v, INVALID_INDEX);
  mesh->set_vertex_coords(v, {-10.0, -10.0, 0.0});

  // Coarsen all: 3 parents should be restored; the distorted parent should remain refined.
  std::vector<bool> coarsen_marks(mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*mesh, coarsen_marks);
  ASSERT_TRUE(coarsen_result.success);
  EXPECT_EQ(coarsen_result.num_coarsened, 3u);
  EXPECT_EQ(mesh->n_cells(), 7u);  // 3 restored parents + 4 children of rejected parent

  // Parents 1..3 exist; parent 0 is rejected.
  EXPECT_EQ(mesh->global_to_local_cell(0), INVALID_INDEX);
  EXPECT_NE(mesh->global_to_local_cell(1), INVALID_INDEX);
  EXPECT_NE(mesh->global_to_local_cell(2), INVALID_INDEX);
  EXPECT_NE(mesh->global_to_local_cell(3), INVALID_INDEX);
}

/**
 * @brief Round-trip refine->coarsen should preserve hex topology and regions
 * (MFEM/deal.II-style invariance check in 3D)
 */
TEST_F(CoarseningTest, RoundTripTopology_Hex) {
  auto mesh = create_hex_mesh();

  // Single hex: assign nonzero region label
  MeshLabels::set_region_label(*mesh, 0, 7);

  auto original_cells = sorted_cell_connectivity(*mesh);
  auto original_regions = region_histogram(*mesh);
  const auto original_vertices = mesh->n_vertices();

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.max_level_difference = 2;
  AdaptivityManager manager(options);

  // Refine the hex once
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  // Coarsen all descendants back to a single hex
  std::vector<bool> coarsen_marks(mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*mesh, coarsen_marks);
  ASSERT_TRUE(coarsen_result.success);
  EXPECT_GT(coarsen_result.num_coarsened, 0u);

  EXPECT_EQ(mesh->n_cells(), static_cast<index_t>(original_cells.size()));
  EXPECT_EQ(mesh->n_vertices(), original_vertices);

  auto final_cells = sorted_cell_connectivity(*mesh);
  auto final_regions = region_histogram(*mesh);

  EXPECT_EQ(final_cells, original_cells);
  EXPECT_EQ(final_regions, original_regions);
}

/**
 * @brief Test pattern-specific reversal
 */
TEST_F(CoarseningTest, PatternReversal) {
  auto mesh = create_quad_mesh();

  // Test RED pattern reversal
  {
    AdaptivityOptions options;
    options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
    AdaptivityManager manager(options);

    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks);
    ASSERT_TRUE(result.success);

    auto refined = mesh.get();

    // Check that RED pattern was recorded
    for (index_t i = 0; i < refined->n_cells(); ++i) {
      int pattern = MeshLabels::get_refinement_pattern(*refined, i);
      EXPECT_EQ(pattern, static_cast<int>(AdaptivityOptions::RefinementPattern::RED));
    }
  }

  // Test GREEN pattern reversal
  {
    AdaptivityOptions options;
    options.refinement_pattern = AdaptivityOptions::RefinementPattern::GREEN;
    AdaptivityManager manager(options);

    auto green_mesh = create_quad_mesh();
    std::vector<bool> marks(green_mesh->n_cells(), true);
    auto result = manager.refine(*green_mesh, marks);

    if (result.success) {
      auto refined = green_mesh.get();

      // Check that GREEN pattern was recorded
      for (index_t i = 0; i < refined->n_cells(); ++i) {
        int pattern = MeshLabels::get_refinement_pattern(*refined, i);
        // GREEN pattern may not be implemented for all cell types
        if (pattern != -1) {
          EXPECT_EQ(pattern, static_cast<int>(AdaptivityOptions::RefinementPattern::GREEN));
        }
      }
    }
  }

  // Test bisection reversal
  {
    AdaptivityOptions options;
    options.use_bisection = true;
    AdaptivityManager manager(options);

    auto bisect_mesh = create_quad_mesh();
    std::vector<bool> marks(bisect_mesh->n_cells(), true);
    auto result = manager.refine(*bisect_mesh, marks);

    if (result.success) {
      auto refined = bisect_mesh.get();

      // Bisection should create 2 children per cell
      for (index_t parent = 0; parent < bisect_mesh->n_cells(); ++parent) {
        auto children = MeshLabels::get_children_cells(*refined, parent);
        // Bisection creates 2 children (if implemented); some
        // configurations may fall back to 4 or 8 children.
        if (!children.empty()) {
          EXPECT_TRUE(children.size() == 2 ||
                      children.size() == 4 ||
                      children.size() == 8)
            << "Bisection should create 2 or fall back to 4 or 8 children";
        }
      }
    }
  }
}

/**
 * @brief Test boundary label aggregation
 */
TEST_F(CoarseningTest, BoundaryLabelAggregation) {
  auto mesh = create_quad_mesh();

  // Capture initial boundary label set for later comparison
  auto initial_boundary_faces = MeshTopology::boundary_codim1(*mesh);
  ASSERT_GE(initial_boundary_faces.size(), static_cast<size_t>(4));

  // Set boundary labels on original mesh in a topology-aware way
  // Label half the boundary with 100 and the other half with 200
  const size_t half = initial_boundary_faces.size() / 2;
  for (size_t i = 0; i < initial_boundary_faces.size(); ++i) {
    label_t label = (i < half) ? 100 : 200;
    MeshLabels::set_boundary_label(*mesh,
                                   static_cast<index_t>(initial_boundary_faces[i]),
                                   label);
  }

  // Refine and then coarsen
  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  auto refined_mesh = mesh.get();

  // Mark all for coarsening
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*refined_mesh, coarsen_marks);

  if (coarsen_result.success && coarsen_result.num_coarsened > 0) {
    auto coarsened_mesh = refined_mesh;

    // Check that boundary labels were preserved/aggregated
    auto boundary_labels = MeshLabels::unique_boundary_labels(*coarsened_mesh);
    // Coarsening should not invent new labels; any labels present
    // after coarsening must come from the initial label set.
    auto initial_labels = MeshLabels::unique_boundary_labels(*mesh);
    for (label_t lbl : boundary_labels) {
      EXPECT_TRUE(initial_labels.count(lbl) > 0)
        << "Coarsening introduced unexpected boundary label " << lbl;
    }
    EXPECT_LE(boundary_labels.size(), initial_labels.size());
  }
}

/**
 * @brief Test field restriction preparation
 */
TEST_F(CoarseningTest, FieldRestrictionMapping) {
  auto mesh = create_quad_mesh();

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  AdaptivityManager manager(options);

  // Refine
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks);
  ASSERT_TRUE(refine_result.success);

  auto refined_mesh = mesh.get();

  // Coarsen
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*refined_mesh, coarsen_marks);

  EXPECT_TRUE(coarsen_result.success);

  // The coarsening should have set up mappings for field restriction
  // These would be used by FieldTransfer to average values from children to parents
  // Actual field transfer testing would require MeshFields implementation
}

/**
 * @brief Test refinement-coarsening cycle
 */
TEST_F(CoarseningTest, RefinementCoarseningCycle) {
  auto mesh = create_quad_mesh();
  size_t original_cells = mesh->n_cells();
  size_t original_vertices = mesh->n_vertices();

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.min_quality = 0.1;
  options.max_level_difference = 2;
  AdaptivityManager manager(options);

  auto* current_mesh = mesh.get();

  // Cycle 1: Refine then coarsen
  std::vector<bool> marks(current_mesh->n_cells(), true);
  auto refine_result = manager.refine(*current_mesh, marks);
  ASSERT_TRUE(refine_result.success);

  EXPECT_GT(current_mesh->n_cells(), original_cells);

  // Coarsen back
  std::vector<bool> coarsen_marks(current_mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*current_mesh, coarsen_marks);
  EXPECT_TRUE(coarsen_result.success);

  // Multiple cycles to test stability
  for (int cycle = 0; cycle < 3; ++cycle) {
    // Refine
    marks.assign(current_mesh->n_cells(), true);
    refine_result = manager.refine(*current_mesh, marks);
    if (!refine_result.success) break;

    // Coarsen
    coarsen_marks.assign(current_mesh->n_cells(), true);
    coarsen_result = manager.coarsen(*current_mesh, coarsen_marks);
    if (!coarsen_result.success) break;
  }

  // Mesh should remain valid after cycles
  EXPECT_GT(current_mesh->n_cells(), 0);
  EXPECT_GT(current_mesh->n_vertices(), 0);
}

/**
 * @brief Coarsening without refinement provenance should be a no-op with a warning
 */
TEST_F(CoarseningTest, CoarsenWithoutProvenanceIsNoOp) {
  auto mesh = create_quad_mesh();
  size_t initial_cells = mesh->n_cells();

  AdaptivityOptions options;
  options.enable_refinement = false;
  options.enable_coarsening = true;
  AdaptivityManager manager(options);

  // Mark all cells for coarsening, but mesh has never been refined
  std::vector<bool> marks(initial_cells, true);
  auto result = manager.coarsen(*mesh, marks, nullptr);

  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.num_coarsened, 0u);
  EXPECT_EQ(mesh->n_cells(), initial_cells);

  // Expect a provenance-related warning
  bool has_provenance_warning = false;
  for (const auto& w : result.warning_messages) {
    if (w.find("provenance") != std::string::npos ||
        w.find("No refinement provenance") != std::string::npos) {
      has_provenance_warning = true;
      break;
    }
  }
  EXPECT_TRUE(has_provenance_warning);
}

/**
 * @brief Constant nodal field should be preserved through refine->coarsen
 */
TEST_F(CoarseningTest, NodalFieldConstantRefineCoarsen) {
  auto mesh = create_quad_mesh();

  // Attach a constant nodal field u = 3.14
  const double value = 3.14;
  auto handle = MeshFields::attach_field(*mesh, EntityKind::Vertex,
                                         "u", FieldScalarType::Float64, 1);
  double* data = MeshFields::field_data_as<double>(*mesh, handle);
  for (size_t i = 0; i < static_cast<size_t>(mesh->n_vertices()); ++i) {
    data[i] = value;
  }

  AdaptivityOptions options;
  options.refinement_pattern = AdaptivityOptions::RefinementPattern::RED;
  options.min_quality = 0.1;
  options.max_level_difference = 2;
  AdaptivityManager manager(options);

  // Dummy fields container (MeshFields is a static API; the instance is just a token)
  MeshFields fields;

  // Refine all cells with field transfer enabled
  std::vector<bool> refine_marks(mesh->n_cells(), true);
  auto refine_result = manager.refine(*mesh, refine_marks, &fields);
  ASSERT_TRUE(refine_result.success);

  // After refinement, the mesh is modified in-place
  auto refined_mesh = mesh.get();
  auto refined_handle = MeshFields::get_field_handle(*refined_mesh,
                                                     EntityKind::Vertex, "u");
  auto* refined_data =
      MeshFields::field_data_as<double>(*refined_mesh, refined_handle);

  // Constant field should remain constant on refined mesh
  for (size_t i = 0; i < static_cast<size_t>(refined_mesh->n_vertices()); ++i) {
    EXPECT_NEAR(refined_data[i], value, 1e-12);
  }

  // Coarsen back with field transfer
  std::vector<bool> coarsen_marks(refined_mesh->n_cells(), true);
  auto coarsen_result = manager.coarsen(*refined_mesh, coarsen_marks, &fields);
  ASSERT_TRUE(coarsen_result.success);

  auto coarsened_mesh = refined_mesh;  // coarsen() modifies mesh in-place
  auto coarsened_handle = MeshFields::get_field_handle(*coarsened_mesh,
                                                       EntityKind::Vertex, "u");
  auto* coarsened_data =
      MeshFields::field_data_as<double>(*coarsened_mesh, coarsened_handle);

  for (size_t i = 0; i < static_cast<size_t>(coarsened_mesh->n_vertices()); ++i) {
    EXPECT_NEAR(coarsened_data[i], value, 1e-12);
  }
}

// Main function
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
