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
 * @file test_WedgeRefinement.cpp
 * @brief Comprehensive unit tests for wedge (prism) h-refinement rules
 *
 * This test suite validates:
 * - Basic refinement correctness (child count, vertex count)
 * - Geometric verification (volume conservation, edge midpoints)
 * - Topology validation (distinct vertices, proper connectivity)
 * - Edge cases (degenerate geometries, various orientations)
 * - Regression tests for previous degenerate connectivity bug
 */

#include <gtest/gtest.h>
#include "../../../Adaptivity/RefinementRules.h"
#include <cmath>
#include <set>
#include <algorithm>
#include <numeric>

namespace svmp {
namespace test {

// ==============================================================================
// Helper Functions
// ==============================================================================

/**
 * @brief Compute volume of a wedge (triangular prism)
 *
 * Volume = base_triangle_area * height
 * For wedge with vertices {v0,v1,v2} (bottom) and {v3,v4,v5} (top)
 */
double compute_wedge_volume(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 6) {
    return 0.0;
  }

  // Bottom triangle vertices
  const auto& v0 = verts[0];
  const auto& v1 = verts[1];
  const auto& v2 = verts[2];

  // Compute base triangle area using cross product
  std::array<double, 3> edge01 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> edge02 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

  std::array<double, 3> cross = {
    edge01[1] * edge02[2] - edge01[2] * edge02[1],
    edge01[2] * edge02[0] - edge01[0] * edge02[2],
    edge01[0] * edge02[1] - edge01[1] * edge02[0]
  };

  double base_area = 0.5 * std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);

  // Compute average height (vertical distance from bottom to top)
  // For a proper wedge, all vertical edges should have the same length
  const auto& v3 = verts[3];
  double dx = v3[0] - v0[0];
  double dy = v3[1] - v0[1];
  double dz = v3[2] - v0[2];
  double height = std::sqrt(dx*dx + dy*dy + dz*dz);

  return base_area * height;
}

/**
 * @brief Extract child vertices from parent vertices and new vertices
 */
std::vector<std::array<double, 3>> get_child_vertices(
    const std::vector<std::array<double, 3>>& parent_verts,
    const std::vector<std::array<double, 3>>& new_verts,
    const std::vector<size_t>& child_connectivity) {

  std::vector<std::array<double, 3>> child_verts;
  child_verts.reserve(child_connectivity.size());

  for (size_t idx : child_connectivity) {
    if (idx < parent_verts.size()) {
      child_verts.push_back(parent_verts[idx]);
    } else {
      size_t new_idx = idx - parent_verts.size();
      if (new_idx < new_verts.size()) {
        child_verts.push_back(new_verts[new_idx]);
      }
    }
  }

  return child_verts;
}

/**
 * @brief Check if all vertices in connectivity are distinct
 */
bool has_distinct_vertices(const std::vector<size_t>& connectivity) {
  std::set<size_t> unique_verts(connectivity.begin(), connectivity.end());
  return unique_verts.size() == connectivity.size();
}

/**
 * @brief Compute distance between two points
 */
double distance(const std::array<double, 3>& p1, const std::array<double, 3>& p2) {
  double dx = p2[0] - p1[0];
  double dy = p2[1] - p1[1];
  double dz = p2[2] - p1[2];
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}

/**
 * @brief Check if point is at midpoint of edge
 */
bool is_midpoint(const std::array<double, 3>& p,
                 const std::array<double, 3>& v1,
                 const std::array<double, 3>& v2,
                 double tol = 1e-10) {
  std::array<double, 3> expected = {
    0.5 * (v1[0] + v2[0]),
    0.5 * (v1[1] + v2[1]),
    0.5 * (v1[2] + v2[2])
  };
  return distance(p, expected) < tol;
}

// ==============================================================================
// Test Fixture
// ==============================================================================

class WedgeRefinementTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create unit wedge: right triangle base at z=0, height=1
    unit_wedge_verts = {
      {0.0, 0.0, 0.0},  // 0: bottom corner
      {1.0, 0.0, 0.0},  // 1: bottom corner
      {0.0, 1.0, 0.0},  // 2: bottom corner
      {0.0, 0.0, 1.0},  // 3: top corner
      {1.0, 0.0, 1.0},  // 4: top corner
      {0.0, 1.0, 1.0}   // 5: top corner
    };

    unit_wedge_volume = compute_wedge_volume(unit_wedge_verts);
  }

  std::vector<std::array<double, 3>> unit_wedge_verts;
  double unit_wedge_volume;
  WedgeRefinementRule rule;
};

// ==============================================================================
// Basic Correctness Tests
// ==============================================================================

TEST_F(WedgeRefinementTest, ProducesEightChildren) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8)
      << "Wedge RED refinement should produce 8 children";
}

TEST_F(WedgeRefinementTest, CreatesTwelveNewVertices) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.new_vertices.size(), 12)
      << "Wedge refinement should create 12 new vertices "
      << "(9 edge midpoints + 3 mid-layer vertices)";
}

TEST_F(WedgeRefinementTest, IncrementsRefinementLevel) {
  size_t initial_level = 0;
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, initial_level);

  EXPECT_EQ(refined.child_level, initial_level + 1)
      << "Child refinement level should be parent level + 1";
}

TEST_F(WedgeRefinementTest, StoresCorrectRefinementPattern) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::RED)
      << "Wedge isotropic refinement uses RED pattern";
}

TEST_F(WedgeRefinementTest, CanRefineReturnsTrue) {
  EXPECT_TRUE(rule.can_refine(CellType::Wedge, 0));
  EXPECT_TRUE(rule.can_refine(CellType::Wedge, 5));
}

TEST_F(WedgeRefinementTest, NumChildrenReturnsEight) {
  size_t num_children = rule.num_children(CellType::Wedge, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 8);
}

// ==============================================================================
// Topology Validation Tests
// ==============================================================================

TEST_F(WedgeRefinementTest, AllChildrenHaveSixVertices) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i].size(), 6)
        << "Child " << i << " should have 6 vertices (wedge topology)";
  }
}

TEST_F(WedgeRefinementTest, AllChildrenHaveDistinctVertices) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    EXPECT_TRUE(has_distinct_vertices(child))
        << "Child " << i << " has repeated vertices: ["
        << child[0] << ", " << child[1] << ", " << child[2] << ", "
        << child[3] << ", " << child[4] << ", " << child[5] << "]";
  }
}

TEST_F(WedgeRefinementTest, ChildVertexIndicesAreValid) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_verts = unit_wedge_verts.size() + refined.new_vertices.size();

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    for (size_t j = 0; j < child.size(); ++j) {
      EXPECT_LT(child[j], total_verts)
          << "Child " << i << " vertex " << j << " has invalid index " << child[j];
    }
  }
}

TEST_F(WedgeRefinementTest, VerifyExpectedConnectivity) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  // Expected connectivity from WEDGE_REFINEMENT_VERIFICATION.md
  std::vector<std::vector<size_t>> expected_connectivity = {
    // Bottom layer
    {0, 6, 8, 12, 15, 17},    // Child 0: Corner at v0
    {6, 1, 7, 15, 13, 16},    // Child 1: Corner at v1
    {8, 7, 2, 17, 16, 14},    // Child 2: Corner at v2
    {6, 7, 8, 15, 16, 17},    // Child 3: Center (bottom)

    // Top layer
    {12, 15, 17, 3, 9, 11},   // Child 4: Corner at v3
    {15, 13, 16, 9, 4, 10},   // Child 5: Corner at v4
    {17, 16, 14, 11, 10, 5},  // Child 6: Corner at v5
    {15, 16, 17, 9, 10, 11}   // Child 7: Center (top)
  };

  ASSERT_EQ(refined.child_connectivity.size(), expected_connectivity.size());

  for (size_t i = 0; i < expected_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i], expected_connectivity[i])
        << "Child " << i << " has incorrect connectivity";
  }
}

// ==============================================================================
// Geometric Verification Tests
// ==============================================================================

TEST_F(WedgeRefinementTest, VolumeConservation) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_wedge_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_vol = compute_wedge_volume(child_verts);

    EXPECT_GT(child_vol, 0.0) << "Child " << i << " has non-positive volume";
    children_volume_sum += child_vol;
  }

  EXPECT_NEAR(unit_wedge_volume, children_volume_sum, 1e-10)
      << "Volume conservation violated: parent volume = " << unit_wedge_volume
      << ", children sum = " << children_volume_sum;
}

TEST_F(WedgeRefinementTest, EdgeMidpointsAreCorrect) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_GE(refined.new_vertices.size(), 9);

  // Bottom triangle edges (indices 6, 7, 8)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], unit_wedge_verts[0], unit_wedge_verts[1]))
      << "Vertex 6 should be midpoint of edge 0-1";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], unit_wedge_verts[1], unit_wedge_verts[2]))
      << "Vertex 7 should be midpoint of edge 1-2";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[2], unit_wedge_verts[2], unit_wedge_verts[0]))
      << "Vertex 8 should be midpoint of edge 2-0";

  // Top triangle edges (indices 9, 10, 11)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[3], unit_wedge_verts[3], unit_wedge_verts[4]))
      << "Vertex 9 should be midpoint of edge 3-4";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[4], unit_wedge_verts[4], unit_wedge_verts[5]))
      << "Vertex 10 should be midpoint of edge 4-5";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[5], unit_wedge_verts[5], unit_wedge_verts[3]))
      << "Vertex 11 should be midpoint of edge 5-3";

  // Vertical edges (indices 12, 13, 14)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[6], unit_wedge_verts[0], unit_wedge_verts[3]))
      << "Vertex 12 should be midpoint of edge 0-3";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[7], unit_wedge_verts[1], unit_wedge_verts[4]))
      << "Vertex 13 should be midpoint of edge 1-4";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[8], unit_wedge_verts[2], unit_wedge_verts[5]))
      << "Vertex 14 should be midpoint of edge 2-5";
}

TEST_F(WedgeRefinementTest, MidLayerVerticesAreCorrect) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_EQ(refined.new_vertices.size(), 12);

  // Mid-layer vertices are midpoints of vertical edges connecting
  // bottom edge midpoints to top edge midpoints

  // Vertex 15: mid(6, 9) = mid(mid(0,1), mid(3,4))
  EXPECT_TRUE(is_midpoint(refined.new_vertices[9], refined.new_vertices[0], refined.new_vertices[3]))
      << "Vertex 15 should be midpoint between vertices 6 and 9";

  // Vertex 16: mid(7, 10) = mid(mid(1,2), mid(4,5))
  EXPECT_TRUE(is_midpoint(refined.new_vertices[10], refined.new_vertices[1], refined.new_vertices[4]))
      << "Vertex 16 should be midpoint between vertices 7 and 10";

  // Vertex 17: mid(8, 11) = mid(mid(2,0), mid(5,3))
  EXPECT_TRUE(is_midpoint(refined.new_vertices[11], refined.new_vertices[2], refined.new_vertices[5]))
      << "Vertex 17 should be midpoint between vertices 8 and 11";
}

TEST_F(WedgeRefinementTest, AllChildrenHavePositiveVolume) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_wedge_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_vol = compute_wedge_volume(child_verts);

    EXPECT_GT(child_vol, 1e-12) << "Child " << i << " has near-zero or negative volume";
  }
}

TEST_F(WedgeRefinementTest, ChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  std::vector<double> child_volumes;
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_wedge_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    child_volumes.push_back(compute_wedge_volume(child_verts));
  }

  double expected_child_vol = unit_wedge_volume / 8.0;

  for (size_t i = 0; i < child_volumes.size(); ++i) {
    EXPECT_NEAR(child_volumes[i], expected_child_vol, 1e-10)
        << "Child " << i << " volume differs from expected equal subdivision";
  }
}

// ==============================================================================
// Edge Cases and Stress Tests
// ==============================================================================

TEST_F(WedgeRefinementTest, ScaledWedgeVolumeConservation) {
  // Test with a scaled wedge
  std::vector<std::array<double, 3>> scaled_wedge = {
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0},
    {0.0, 2.0, 0.0},
    {0.0, 0.0, 3.0},
    {2.0, 0.0, 3.0},
    {0.0, 2.0, 3.0}
  };

  double parent_volume = compute_wedge_volume(scaled_wedge);
  auto refined = rule.refine(scaled_wedge, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(scaled_wedge, refined.new_vertices, child_conn);
    children_volume_sum += compute_wedge_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-9)
      << "Volume conservation fails for scaled wedge";
}

TEST_F(WedgeRefinementTest, TranslatedWedgeVolumeConservation) {
  // Test with a translated wedge
  std::vector<std::array<double, 3>> translated_wedge = {
    {10.0, 20.0, 30.0},
    {11.0, 20.0, 30.0},
    {10.0, 21.0, 30.0},
    {10.0, 20.0, 31.0},
    {11.0, 20.0, 31.0},
    {10.0, 21.0, 31.0}
  };

  double parent_volume = compute_wedge_volume(translated_wedge);
  auto refined = rule.refine(translated_wedge, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(translated_wedge, refined.new_vertices, child_conn);
    children_volume_sum += compute_wedge_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for translated wedge";
}

TEST_F(WedgeRefinementTest, RotatedWedgeProducesEightChildren) {
  // Test with a rotated wedge (45 degrees about z-axis)
  double angle = M_PI / 4.0;
  double cos_a = std::cos(angle);
  double sin_a = std::sin(angle);

  std::vector<std::array<double, 3>> rotated_wedge = {
    {0.0, 0.0, 0.0},
    {cos_a, sin_a, 0.0},
    {-sin_a, cos_a, 0.0},
    {0.0, 0.0, 1.0},
    {cos_a, sin_a, 1.0},
    {-sin_a, cos_a, 1.0}
  };

  auto refined = rule.refine(rotated_wedge, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8);
  EXPECT_EQ(refined.new_vertices.size(), 12);
}

TEST_F(WedgeRefinementTest, SmallWedgeVolumeConservation) {
  // Test with a very small wedge
  std::vector<std::array<double, 3>> small_wedge = {
    {0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0},
    {0.0, 0.001, 0.0},
    {0.0, 0.0, 0.001},
    {0.001, 0.0, 0.001},
    {0.0, 0.001, 0.001}
  };

  double parent_volume = compute_wedge_volume(small_wedge);
  EXPECT_GT(parent_volume, 0.0) << "Parent small wedge should have positive volume";

  auto refined = rule.refine(small_wedge, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(small_wedge, refined.new_vertices, child_conn);
    children_volume_sum += compute_wedge_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-15)
      << "Volume conservation fails for small wedge";
}

// ==============================================================================
// Regression Tests
// ==============================================================================

TEST_F(WedgeRefinementTest, RegressionNoDegenerateChildren) {
  // Regression test: Previous implementation had degenerate children 5-7
  // with repeated vertex indices. This test ensures the bug is fixed.
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  // Specifically check the previously problematic children
  const auto& child5 = refined.child_connectivity[5];
  const auto& child6 = refined.child_connectivity[6];
  const auto& child7 = refined.child_connectivity[7];

  EXPECT_TRUE(has_distinct_vertices(child5))
      << "Child 5 has repeated vertices (regression from previous bug)";
  EXPECT_TRUE(has_distinct_vertices(child6))
      << "Child 6 has repeated vertices (regression from previous bug)";
  EXPECT_TRUE(has_distinct_vertices(child7))
      << "Child 7 has repeated vertices (regression from previous bug)";
}

TEST_F(WedgeRefinementTest, RegressionCorrectChild5Connectivity) {
  // Child 5 was: {9, 13, 10, 4, 13, 10} (vertex 13 and 10 repeated)
  // Should be: {15, 13, 16, 9, 4, 10}
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  std::vector<size_t> expected_child5 = {15, 13, 16, 9, 4, 10};
  EXPECT_EQ(refined.child_connectivity[5], expected_child5)
      << "Child 5 connectivity does not match expected (regression check)";
}

TEST_F(WedgeRefinementTest, RegressionCorrectChild6Connectivity) {
  // Child 6 was: {11, 10, 14, 5, 10, 14} (vertices 10 and 14 repeated)
  // Should be: {17, 16, 14, 11, 10, 5}
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  std::vector<size_t> expected_child6 = {17, 16, 14, 11, 10, 5};
  EXPECT_EQ(refined.child_connectivity[6], expected_child6)
      << "Child 6 connectivity does not match expected (regression check)";
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(WedgeRefinementTest, ThrowsOnInvalidCellType) {
  EXPECT_THROW(
    rule.refine(unit_wedge_verts, CellType::Tetra, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when CellType is not WEDGE";
}

TEST_F(WedgeRefinementTest, ThrowsOnInvalidVertexCount) {
  std::vector<std::array<double, 3>> invalid_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0}
    // Only 3 vertices instead of 6
  };

  EXPECT_THROW(
    rule.refine(invalid_verts, CellType::Wedge, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when vertex count is not 6";
}

// ==============================================================================
// Parameterized Tests for Multiple Refinement Levels
// ==============================================================================

class WedgeMultiLevelRefinementTest : public ::testing::TestWithParam<size_t> {
protected:
  void SetUp() override {
    unit_wedge_verts = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {1.0, 0.0, 1.0},
      {0.0, 1.0, 1.0}
    };
  }

  std::vector<std::array<double, 3>> unit_wedge_verts;
  WedgeRefinementRule rule;
};

TEST_P(WedgeMultiLevelRefinementTest, CorrectChildLevelAtEachLevel) {
  size_t level = GetParam();

  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, level);

  EXPECT_EQ(refined.child_level, level + 1)
      << "Child level incorrect at refinement level " << level;
}

TEST_P(WedgeMultiLevelRefinementTest, VolumeConservationAtEachLevel) {
  size_t level = GetParam();

  double parent_volume = compute_wedge_volume(unit_wedge_verts);
  auto refined = rule.refine(unit_wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, level);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(unit_wedge_verts, refined.new_vertices, child_conn);
    children_volume_sum += compute_wedge_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails at refinement level " << level;
}

INSTANTIATE_TEST_SUITE_P(
  MultiLevel,
  WedgeMultiLevelRefinementTest,
  ::testing::Values(0, 1, 2, 3, 5, 9)
);

} // namespace test
} // namespace svmp
