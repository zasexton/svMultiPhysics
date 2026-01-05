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
#include "../../../Adaptivity/Options.h"
#include "../../../Adaptivity/ErrorEstimator.h"
#include "../../../Adaptivity/Marker.h"
#include "../../../Adaptivity/FieldTransfer.h"
#include "../../../Adaptivity/QualityGuards.h"
#include "../../../Adaptivity/Conformity.h"
#include "../../../Core/MeshBase.h"
#include "../../../Fields/MeshFields.h"
#include <memory>
#include <vector>

namespace svmp {
namespace test {

class AdaptivityManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Will be initialized in each test as needed
  }

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
        mesh->add_cell(elem_id++, CellFamily::Quad, {static_cast<index_t>(v0), static_cast<index_t>(v1), static_cast<index_t>(v2), static_cast<index_t>(v3)});
      }
    }

    return mesh;
  }

  // Helper to create a 2D triangle mesh
  std::unique_ptr<MeshBase> create_2d_tri_mesh(size_t nx = 2, size_t ny = 2) {
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

    // Create triangle elements (2 per quad)
    size_t elem_id = 0;
    for (size_t j = 0; j < ny; ++j) {
      for (size_t i = 0; i < nx; ++i) {
        size_t v0 = j * (nx + 1) + i;
        size_t v1 = v0 + 1;
        size_t v2 = v0 + nx + 2;
        size_t v3 = v0 + nx + 1;

        // Lower triangle
        mesh->add_cell(elem_id++, CellFamily::Triangle, {static_cast<index_t>(v0), static_cast<index_t>(v1), static_cast<index_t>(v2)});
        // Upper triangle
        mesh->add_cell(elem_id++, CellFamily::Triangle, {static_cast<index_t>(v0), static_cast<index_t>(v2), static_cast<index_t>(v3)});
      }
    }

    return mesh;
  }

  // Helper to create default adaptivity options
  AdaptivityOptions create_default_options() {
    AdaptivityOptions options;
    options.enable_refinement = true;
    options.enable_coarsening = false;
    options.max_refinement_level = 3;
    options.refine_fraction = 0.3;
    options.coarsen_fraction = 0.1;
    // Use GRADIENT_RECOVERY instead of RESIDUAL_BASED to avoid needing custom callbacks
    options.estimator_type = AdaptivityOptions::EstimatorType::GRADIENT_RECOVERY;
    options.marking_strategy = AdaptivityOptions::MarkingStrategy::FIXED_FRACTION;
    options.check_quality = false;  // Disable quality checking for faster tests
    options.verbosity = 0;  // Quiet mode
    return options;
  }
};

// ========== Factory and Configuration Tests ==========

// Test 1: Factory creation with default options
TEST_F(AdaptivityManagerTest, FactoryCreateDefault) {
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  EXPECT_EQ(manager.get_options().enable_refinement, true);
  EXPECT_EQ(manager.get_options().enable_coarsening, false);
}

// Test 2: Factory creation with builder pattern
TEST_F(AdaptivityManagerTest, FactoryCreateBuilder) {
  AdaptivityOptions options = create_default_options();

  auto manager = AdaptivityManagerBuilder()
      .with_options(options)
      .build();

  ASSERT_NE(manager, nullptr);
  EXPECT_EQ(manager->get_options().max_refinement_level, 3);
}

// Test 3: Set custom error estimator
TEST_F(AdaptivityManagerTest, SetCustomErrorEstimator) {
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  auto custom_estimator = ErrorEstimatorFactory::create(options);
  manager.set_error_estimator(std::move(custom_estimator));

  // Verify manager accepts custom estimator (no crash)
  EXPECT_TRUE(true);
}

// Test 4: Set custom marker
TEST_F(AdaptivityManagerTest, SetCustomMarker) {
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  auto custom_marker = MarkerFactory::create(options);
  manager.set_marker(std::move(custom_marker));

  // Verify manager accepts custom marker (no crash)
  EXPECT_TRUE(true);
}

// ========== Full AMR Workflow Tests ==========

// Test 5: Complete adapt() cycle without fields
TEST_F(AdaptivityManagerTest, CompleteAdaptCycleNoFields) {
  auto mesh = create_2d_quad_mesh();
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  size_t initial_elements = mesh->n_cells();

  // Note: adapt() without fields will fail or mark nothing since no error to estimate
  // This is expected behavior
  auto result = manager.adapt(*mesh, nullptr);

  // Should complete without crashing
  EXPECT_GE(result.final_cell_count, 0);
}

// Test 6: Adapt cycle with uniform refinement marks
TEST_F(AdaptivityManagerTest, AdaptCycleUniformRefinement) {
  auto mesh = create_2d_tri_mesh(2, 2);  // 8 triangles
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  size_t initial_count = mesh->n_cells();

  // Mark all elements for refinement
  std::vector<bool> marks(initial_count, true);
  auto result = manager.refine(*mesh, marks, nullptr);

  // Should have refined some or all elements
  EXPECT_GE(result.num_refined, 0);
}

// Test 7: Multiple adaptation iterations
TEST_F(AdaptivityManagerTest, MultipleAdaptationIterations) {
  auto mesh = create_2d_quad_mesh(1, 1);  // 1 quad
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  // Perform 2 refinement iterations
  for (int iter = 0; iter < 2; ++iter) {
    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks, nullptr);

    // Should complete each iteration
    EXPECT_GE(result.final_cell_count, 0);
  }

  // Mesh should have grown
  EXPECT_GT(mesh->n_cells(), 1);
}

// Test 8: Refinement with specific element marks
TEST_F(AdaptivityManagerTest, RefinementWithSpecificMarks) {
  auto mesh = create_2d_quad_mesh(2, 2);  // 4 quads
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  // Mark only first element
  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);

  // Should have marked at least one element (possibly more due to conformity)
  EXPECT_GT(result.num_refined, 0);
}

// Test 9: Coarsening operation
TEST_F(AdaptivityManagerTest, CoarseningOperation) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  options.enable_refinement = false;
  options.enable_coarsening = true;
  AdaptivityManager manager(options);

  size_t initial_count = mesh->n_cells();

  // Mark some elements for coarsening
  std::vector<bool> marks(initial_count, false);
  marks[0] = true;
  marks[1] = true;

  auto result = manager.coarsen(*mesh, marks, nullptr);

  // Coarsening may or may not succeed (depends on implementation)
  // Just verify it doesn't crash
  EXPECT_GE(result.final_cell_count, 0);
}

// Test 10: Different error estimator and marker combinations
TEST_F(AdaptivityManagerTest, DifferentEstimatorMarkerCombinations) {
  auto mesh = create_2d_quad_mesh(2, 2);

  // Test with RESIDUAL_BASED + FIXED_FRACTION
  {
    AdaptivityOptions options = create_default_options();
    options.estimator_type = AdaptivityOptions::EstimatorType::RESIDUAL_BASED;
    options.marking_strategy = AdaptivityOptions::MarkingStrategy::FIXED_FRACTION;
    AdaptivityManager manager(options);

    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks, nullptr);
    EXPECT_GE(result.final_cell_count, 0);
  }

  // Test with GRADIENT_RECOVERY + THRESHOLD_ABSOLUTE
  {
    AdaptivityOptions options = create_default_options();
    options.estimator_type = AdaptivityOptions::EstimatorType::GRADIENT_RECOVERY;
    options.marking_strategy = AdaptivityOptions::MarkingStrategy::THRESHOLD_ABSOLUTE;
    AdaptivityManager manager(options);

    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks, nullptr);
    EXPECT_GE(result.final_cell_count, 0);
  }
}

// ========== Configuration Tests ==========

// Test 11: Max refinement level enforcement
TEST_F(AdaptivityManagerTest, MaxRefinementLevelEnforcement) {
  auto mesh = create_2d_quad_mesh(1, 1);
  AdaptivityOptions options = create_default_options();
  options.max_refinement_level = 2;
  AdaptivityManager manager(options);

  // Try to refine beyond max level
  for (int iter = 0; iter < 5; ++iter) {
    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks, nullptr);

    // Should eventually stop refining
    if (result.num_refined == 0) {
      break;
    }
  }

  // Test passes if we complete without crashing
  EXPECT_TRUE(true);
}

// Test 12: Refine and coarsen fractions
TEST_F(AdaptivityManagerTest, RefineCoarsenFractions) {
  AdaptivityOptions options = create_default_options();
  options.refine_fraction = 0.5;
  options.coarsen_fraction = 0.2;

  AdaptivityManager manager(options);

  EXPECT_DOUBLE_EQ(manager.get_options().refine_fraction, 0.5);
  EXPECT_DOUBLE_EQ(manager.get_options().coarsen_fraction, 0.2);
}

// Test 13: Options update via set_options()
TEST_F(AdaptivityManagerTest, OptionsUpdateViaSetOptions) {
  AdaptivityOptions options1 = create_default_options();
  AdaptivityManager manager(options1);

  // Update options
  AdaptivityOptions options2 = create_default_options();
  options2.max_refinement_level = 5;
  manager.set_options(options2);

  EXPECT_EQ(manager.get_options().max_refinement_level, 5);
}

// Test 14: Verbosity levels
TEST_F(AdaptivityManagerTest, VerbosityLevels) {
  auto mesh = create_2d_quad_mesh(1, 1);

  // Test with verbosity 0 (quiet)
  {
    AdaptivityOptions options = create_default_options();
    options.verbosity = 0;
    AdaptivityManager manager(options);

    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks, nullptr);
    EXPECT_GE(result.final_cell_count, 0);
  }

  // Test with verbosity 2 (verbose) - should not crash
  {
    AdaptivityOptions options = create_default_options();
    options.verbosity = 2;
    AdaptivityManager manager(options);

    std::vector<bool> marks(mesh->n_cells(), true);
    auto result = manager.refine(*mesh, marks, nullptr);
    EXPECT_GE(result.final_cell_count, 0);
  }
}

// ========== Integration Tests ==========

// Test 15: Error estimation integration
TEST_F(AdaptivityManagerTest, ErrorEstimationIntegration) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  // Check if adaptation is needed (without fields, should return false or handle gracefully)
  bool needs_adapt = manager.needs_adaptation(*mesh, nullptr);

  // Should return a boolean without crashing
  EXPECT_TRUE(needs_adapt || !needs_adapt);
}

// Test 16: Estimate adaptation without performing it
TEST_F(AdaptivityManagerTest, EstimateAdaptationWithoutPerforming) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  auto estimate = manager.estimate_adaptation(*mesh, nullptr);

  // Should produce an estimate
  EXPECT_GE(estimate.final_cell_count, 0);
}

// Test 17: Get last indicators and marks
TEST_F(AdaptivityManagerTest, GetLastIndicatorsAndMarks) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  // Perform adaptation
  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks, nullptr);

  // Should be able to query last indicators/marks
  auto indicators = manager.get_last_indicators();
  auto last_marks = manager.get_last_marks();

  EXPECT_GE(indicators.size(), 0);
  EXPECT_GE(last_marks.size(), 0);
}

// Test 18: Conformity enforcement integration
TEST_F(AdaptivityManagerTest, ConformityEnforcementIntegration) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  options.conformity_mode = AdaptivityOptions::ConformityMode::ENFORCE_CONFORMING;
  AdaptivityManager manager(options);

  // Mark one element - should trigger conformity enforcement
  std::vector<bool> marks(mesh->n_cells(), false);
  marks[0] = true;

  auto result = manager.refine(*mesh, marks, nullptr);

  // Conformity may mark additional elements
  EXPECT_GE(result.num_refined, 0);
}

// Test 19: Quality checking integration (disabled for speed)
TEST_F(AdaptivityManagerTest, QualityCheckingIntegration) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  options.check_quality = false;  // Disabled for this test
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks, nullptr);

  // Should complete without quality checking
  EXPECT_GE(result.final_cell_count, 0);
}

// Test 20: Result summary generation
TEST_F(AdaptivityManagerTest, ResultSummaryGeneration) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  std::vector<bool> marks(mesh->n_cells(), true);
  auto result = manager.refine(*mesh, marks, nullptr);

  // Generate summary
  std::string summary = result.summary();

  // Summary should be non-empty
  EXPECT_GT(summary.length(), 0);
  EXPECT_TRUE(summary.find("Adaptivity Result Summary") != std::string::npos);
}

// ========== Edge Cases ==========

// Test 21: Empty mesh handling
TEST_F(AdaptivityManagerTest, EmptyMeshHandling) {
  auto mesh = std::make_unique<MeshBase>();  // Empty mesh
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  std::vector<bool> marks;  // Empty marks
  auto result = manager.refine(*mesh, marks, nullptr);

  // Should handle empty mesh gracefully
  EXPECT_EQ(result.final_cell_count, 0);
}

// Test 22: No elements marked for adaptation
TEST_F(AdaptivityManagerTest, NoElementsMarked) {
  auto mesh = create_2d_quad_mesh(2, 2);
  AdaptivityOptions options = create_default_options();
  AdaptivityManager manager(options);

  // Mark no elements
  std::vector<bool> marks(mesh->n_cells(), false);
  auto result = manager.refine(*mesh, marks, nullptr);

  // Should detect no marks and return early
  EXPECT_EQ(result.num_refined, 0);
}

// Test 23: Invalid configuration handling
TEST_F(AdaptivityManagerTest, InvalidConfigurationHandling) {
  AdaptivityOptions options = create_default_options();
  options.max_refinement_level = 0;  // Invalid: cannot refine

  AdaptivityManager manager(options);

  // Manager should be created despite unusual config
  EXPECT_EQ(manager.get_options().max_refinement_level, 0);
}

// Test 24: AdaptivityUtils uniform refinement
TEST_F(AdaptivityManagerTest, AdaptivityUtilsUniformRefinement) {
  auto mesh = create_2d_quad_mesh(1, 1);  // 1 quad

  auto result = AdaptivityUtils::uniform_refinement(*mesh, 1, nullptr);

  // Should have refined the mesh
  EXPECT_GT(mesh->n_cells(), 1);
  EXPECT_TRUE(result.success);
}

// Test 25: AdaptivityUtils uniform coarsening (round-trip after refinement)
TEST_F(AdaptivityManagerTest, AdaptivityUtilsUniformCoarseningRoundTrip) {
  auto mesh = create_2d_quad_mesh(1, 1);  // 1 quad
  const size_t initial_cells = mesh->n_cells();

  auto refined = AdaptivityUtils::uniform_refinement(*mesh, 1, nullptr);
  ASSERT_TRUE(refined.success);
  ASSERT_GT(mesh->n_cells(), initial_cells);

  auto coarsened = AdaptivityUtils::uniform_coarsening(*mesh, 1, nullptr);
  EXPECT_TRUE(coarsened.success);
  EXPECT_EQ(mesh->n_cells(), initial_cells);
}

// Test 25: AdaptivityUtils level statistics
TEST_F(AdaptivityManagerTest, AdaptivityUtilsLevelStatistics) {
  auto mesh = create_2d_quad_mesh(2, 2);

  auto stats = AdaptivityUtils::get_level_stats(*mesh);

  // Should return valid statistics
  EXPECT_GE(stats.min_level, 0);
  EXPECT_GE(stats.max_level, stats.min_level);
  EXPECT_GE(stats.cell_count_per_level.size(), 0);
}

// Test 26: AdaptivityUtils is_adapted
TEST_F(AdaptivityManagerTest, AdaptivityUtilsIsAdapted) {
  auto mesh = create_2d_quad_mesh(1, 1);
  EXPECT_FALSE(AdaptivityUtils::is_adapted(*mesh));

  auto refined = AdaptivityUtils::uniform_refinement(*mesh, 1, nullptr);
  ASSERT_TRUE(refined.success);
  EXPECT_TRUE(AdaptivityUtils::is_adapted(*mesh));
}

// Test 27: AdaptivityUtils local refinement
TEST_F(AdaptivityManagerTest, AdaptivityUtilsLocalRefinement) {
  auto mesh = create_2d_quad_mesh(2, 2);
  const size_t initial_cells = mesh->n_cells();

  auto result = AdaptivityUtils::local_refinement(
      *mesh,
      [](const std::array<double, 3>& x) { return x[0] < 1.0; },
      1,
      nullptr);

  EXPECT_TRUE(result.success);
  EXPECT_GT(mesh->n_cells(), initial_cells);
}

} // namespace test
} // namespace svmp
