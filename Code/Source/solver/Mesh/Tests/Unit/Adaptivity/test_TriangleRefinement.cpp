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
 * @file test_TriangleRefinement.cpp
 * @brief Comprehensive unit tests for triangle h-refinement rules
 *
 * This test suite validates:
 * - Basic refinement correctness (child count, vertex count)
 * - Geometric verification (area conservation, edge midpoints)
 * - Topology validation (distinct vertices, proper connectivity)
 * - Edge cases (degenerate geometries, various orientations)
 * - 2D-specific properties (planarity, edge alignment)
 *
 * Triangle RED refinement:
 * - 3 parent vertices -> 3 edge midpoints -> 6 total vertices
 * - 4 child triangles: 3 corner triangles + 1 center triangle
 * - Area conservation: A_parent = 4 * A_child
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
 * @brief Compute area of a triangle using cross product
 *
 * Area = 0.5 * ||(v1-v0) x (v2-v0)||
 */
double compute_triangle_area(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 3) {
    return 0.0;
  }

  const auto& v0 = verts[0];
  const auto& v1 = verts[1];
  const auto& v2 = verts[2];

  // Edge vectors from v0
  std::array<double, 3> edge01 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> edge02 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

  // Cross product
  std::array<double, 3> cross = {
    edge01[1] * edge02[2] - edge01[2] * edge02[1],
    edge01[2] * edge02[0] - edge01[0] * edge02[2],
    edge01[0] * edge02[1] - edge01[1] * edge02[0]
  };

  double magnitude = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
  return 0.5 * magnitude;
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

/**
 * @brief Check if a triangle has positive area (non-degenerate)
 */
bool has_positive_area(const std::vector<std::array<double, 3>>& verts, double tol = 1e-12) {
  return compute_triangle_area(verts) > tol;
}

/**
 * @brief Check if three points are coplanar with a reference plane
 */
bool are_coplanar(const std::vector<std::array<double, 3>>& points,
                  const std::array<double, 3>& normal,
                  const std::array<double, 3>& point_on_plane,
                  double tol = 1e-10) {
  for (const auto& p : points) {
    double dist = std::abs((p[0] - point_on_plane[0]) * normal[0] +
                          (p[1] - point_on_plane[1]) * normal[1] +
                          (p[2] - point_on_plane[2]) * normal[2]);
    if (dist > tol) {
      return false;
    }
  }
  return true;
}

// ==============================================================================
// Test Fixture
// ==============================================================================

class TriangleRefinementTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create unit right triangle in xy-plane
    unit_triangle_verts = {
      {0.0, 0.0, 0.0},  // 0: origin corner
      {1.0, 0.0, 0.0},  // 1: x-axis corner
      {0.0, 1.0, 0.0}   // 2: y-axis corner
    };

    unit_triangle_area = compute_triangle_area(unit_triangle_verts);
  }

  std::vector<std::array<double, 3>> unit_triangle_verts;
  double unit_triangle_area;
  TriangleRefinementRule rule;
};

// ==============================================================================
// Basic Correctness Tests
// ==============================================================================

TEST_F(TriangleRefinementTest, ProducesFourChildren) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 4)
      << "Triangle RED refinement should produce 4 children";
}

TEST_F(TriangleRefinementTest, CreatesThreeNewVertices) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.new_vertices.size(), 3)
      << "Triangle refinement should create 3 new vertices (3 edge midpoints)";
}

TEST_F(TriangleRefinementTest, IncrementsRefinementLevel) {
  size_t initial_level = 0;
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, initial_level);

  EXPECT_EQ(refined.child_level, initial_level + 1)
      << "Child refinement level should be parent level + 1";
}

TEST_F(TriangleRefinementTest, StoresCorrectRefinementPattern) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::RED)
      << "Triangle isotropic refinement uses RED pattern";
}

TEST_F(TriangleRefinementTest, CanRefineReturnsTrue) {
  EXPECT_TRUE(rule.can_refine(CellType::Triangle, 0));
  EXPECT_TRUE(rule.can_refine(CellType::Triangle, 5));
}

TEST_F(TriangleRefinementTest, NumChildrenReturnsFour) {
  size_t num_children = rule.num_children(CellType::Triangle, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 4);
}

// ==============================================================================
// Topology Validation Tests
// ==============================================================================

TEST_F(TriangleRefinementTest, AllChildrenHaveThreeVertices) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i].size(), 3)
        << "Child " << i << " should have 3 vertices (triangle topology)";
  }
}

TEST_F(TriangleRefinementTest, AllChildrenHaveDistinctVertices) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    EXPECT_TRUE(has_distinct_vertices(child))
        << "Child " << i << " has repeated vertices: ["
        << child[0] << ", " << child[1] << ", " << child[2] << "]";
  }
}

TEST_F(TriangleRefinementTest, ChildVertexIndicesAreValid) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_verts = unit_triangle_verts.size() + refined.new_vertices.size();

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    for (size_t j = 0; j < child.size(); ++j) {
      EXPECT_LT(child[j], total_verts)
          << "Child " << i << " vertex " << j << " has invalid index " << child[j];
    }
  }
}

TEST_F(TriangleRefinementTest, VerifyExpectedConnectivity) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // Expected connectivity from implementation
  // Original vertices: 0, 1, 2
  // New vertices: 3 (m01), 4 (m12), 5 (m20)
  std::vector<std::vector<size_t>> expected_connectivity = {
    {0, 3, 5},  // Child 0: Corner triangle at vertex 0
    {3, 1, 4},  // Child 1: Corner triangle at vertex 1
    {5, 4, 2},  // Child 2: Corner triangle at vertex 2
    {3, 4, 5}   // Child 3: Center triangle
  };

  ASSERT_EQ(refined.child_connectivity.size(), expected_connectivity.size());

  for (size_t i = 0; i < expected_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i], expected_connectivity[i])
        << "Child " << i << " has incorrect connectivity";
  }
}

TEST_F(TriangleRefinementTest, CornerChildrenUseOriginalVertices) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // First 3 children are corner triangles, each should use one original vertex
  for (size_t i = 0; i < 3; ++i) {
    const auto& child = refined.child_connectivity[i];
    bool has_original = false;
    for (size_t v : child) {
      if (v < 3) {  // Original vertex indices are 0-2
        has_original = true;
        break;
      }
    }
    EXPECT_TRUE(has_original)
        << "Corner child " << i << " should contain at least one original vertex";
  }
}

TEST_F(TriangleRefinementTest, CenterChildUsesOnlyNewVertices) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // Last child is center triangle, should only use new vertices
  const auto& center_child = refined.child_connectivity[3];
  for (size_t v : center_child) {
    EXPECT_GE(v, 3)
        << "Center child should only use new vertices (indices >= 3)";
  }
}

// ==============================================================================
// Geometric Verification Tests
// ==============================================================================

TEST_F(TriangleRefinementTest, AreaConservation) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_area = compute_triangle_area(child_verts);

    EXPECT_GT(child_area, 0.0) << "Child " << i << " has non-positive area";
    children_area_sum += child_area;
  }

  EXPECT_NEAR(unit_triangle_area, children_area_sum, 1e-10)
      << "Area conservation violated: parent area = " << unit_triangle_area
      << ", children sum = " << children_area_sum;
}

TEST_F(TriangleRefinementTest, EdgeMidpointsAreCorrect) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_EQ(refined.new_vertices.size(), 3);

  // Verify all 3 edge midpoints
  // m01 (edge 0-1)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], unit_triangle_verts[0], unit_triangle_verts[1]))
      << "Vertex 3 should be midpoint of edge 0-1";

  // m12 (edge 1-2)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], unit_triangle_verts[1], unit_triangle_verts[2]))
      << "Vertex 4 should be midpoint of edge 1-2";

  // m20 (edge 2-0)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[2], unit_triangle_verts[2], unit_triangle_verts[0]))
      << "Vertex 5 should be midpoint of edge 2-0";
}

TEST_F(TriangleRefinementTest, AllChildrenHavePositiveArea) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_area = compute_triangle_area(child_verts);

    EXPECT_GT(child_area, 1e-12) << "Child " << i << " has near-zero or negative area";
  }
}

TEST_F(TriangleRefinementTest, ChildrenHaveEqualAreas) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  std::vector<double> child_areas;
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    child_areas.push_back(compute_triangle_area(child_verts));
  }

  double expected_child_area = unit_triangle_area / 4.0;

  for (size_t i = 0; i < child_areas.size(); ++i) {
    EXPECT_NEAR(child_areas[i], expected_child_area, 1e-10)
        << "Child " << i << " area differs from expected equal subdivision";
  }
}

TEST_F(TriangleRefinementTest, ChildrenArePlanar) {
  // All child triangles should lie in the same plane as the parent
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // Compute parent plane normal (should be z-axis for unit triangle)
  std::array<double, 3> normal = {0.0, 0.0, 1.0};
  std::array<double, 3> point = unit_triangle_verts[0];

  // Check all new vertices are coplanar with parent
  EXPECT_TRUE(are_coplanar(refined.new_vertices, normal, point))
      << "New vertices should be coplanar with parent triangle";

  // Check each child triangle is coplanar
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    EXPECT_TRUE(are_coplanar(child_verts, normal, point))
        << "Child " << i << " should be coplanar with parent triangle";
  }
}

// ==============================================================================
// Edge Cases and Stress Tests
// ==============================================================================

TEST_F(TriangleRefinementTest, ScaledTriangleAreaConservation) {
  // Test with a scaled triangle
  std::vector<std::array<double, 3>> scaled_triangle = {
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0},
    {0.0, 2.0, 0.0}
  };

  double parent_area = compute_triangle_area(scaled_triangle);
  auto refined = rule.refine(scaled_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(scaled_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-9)
      << "Area conservation fails for scaled triangle";
}

TEST_F(TriangleRefinementTest, TranslatedTriangleAreaConservation) {
  // Test with a translated triangle
  std::vector<std::array<double, 3>> translated_triangle = {
    {10.0, 20.0, 30.0},
    {11.0, 20.0, 30.0},
    {10.0, 21.0, 30.0}
  };

  double parent_area = compute_triangle_area(translated_triangle);
  auto refined = rule.refine(translated_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(translated_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-10)
      << "Area conservation fails for translated triangle";
}

TEST_F(TriangleRefinementTest, RotatedTriangleProducesFourChildren) {
  // Test with a rotated triangle (45 degrees about z-axis)
  double angle = M_PI / 4.0;
  double cos_a = std::cos(angle);
  double sin_a = std::sin(angle);

  std::vector<std::array<double, 3>> rotated_triangle = {
    {0.0, 0.0, 0.0},
    {cos_a, sin_a, 0.0},
    {-sin_a, cos_a, 0.0}
  };

  auto refined = rule.refine(rotated_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 4);
  EXPECT_EQ(refined.new_vertices.size(), 3);
}

TEST_F(TriangleRefinementTest, SmallTriangleAreaConservation) {
  // Test with a very small triangle
  std::vector<std::array<double, 3>> small_triangle = {
    {0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0},
    {0.0, 0.001, 0.0}
  };

  double parent_area = compute_triangle_area(small_triangle);
  EXPECT_GT(parent_area, 0.0) << "Parent small triangle should have positive area";

  auto refined = rule.refine(small_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(small_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-15)
      << "Area conservation fails for small triangle";
}

TEST_F(TriangleRefinementTest, EquilateralTriangleAreaConservation) {
  // Test with an equilateral triangle
  double h = std::sqrt(3.0) / 2.0;
  std::vector<std::array<double, 3>> equilateral_triangle = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.5, h, 0.0}
  };

  double parent_area = compute_triangle_area(equilateral_triangle);
  EXPECT_GT(parent_area, 0.0) << "Equilateral triangle should have positive area";

  auto refined = rule.refine(equilateral_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(equilateral_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-10)
      << "Area conservation fails for equilateral triangle";
}

TEST_F(TriangleRefinementTest, ObtuseTriangleAreaConservation) {
  // Test with an obtuse triangle
  std::vector<std::array<double, 3>> obtuse_triangle = {
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0},
    {0.1, 0.1, 0.0}
  };

  double parent_area = compute_triangle_area(obtuse_triangle);
  EXPECT_GT(parent_area, 0.0) << "Obtuse triangle should have positive area";

  auto refined = rule.refine(obtuse_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(obtuse_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-10)
      << "Area conservation fails for obtuse triangle";
}

TEST_F(TriangleRefinementTest, SkinnyTriangleAreaConservation) {
  // Test with a skinny triangle (high aspect ratio)
  std::vector<std::array<double, 3>> skinny_triangle = {
    {0.0, 0.0, 0.0},
    {10.0, 0.0, 0.0},
    {5.0, 0.1, 0.0}
  };

  double parent_area = compute_triangle_area(skinny_triangle);
  EXPECT_GT(parent_area, 0.0) << "Skinny triangle should have positive area";

  auto refined = rule.refine(skinny_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(skinny_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-10)
      << "Area conservation fails for skinny triangle";
}

TEST_F(TriangleRefinementTest, TriangleIn3DSpaceAreaConservation) {
  // Test with a triangle not in xy-plane
  std::vector<std::array<double, 3>> tilted_triangle = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.5},
    {0.0, 1.0, 0.5}
  };

  double parent_area = compute_triangle_area(tilted_triangle);
  EXPECT_GT(parent_area, 0.0) << "Tilted triangle should have positive area";

  auto refined = rule.refine(tilted_triangle, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(tilted_triangle, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-10)
      << "Area conservation fails for triangle in 3D space";
}

// ==============================================================================
// Symmetry Tests
// ==============================================================================

TEST_F(TriangleRefinementTest, CornerTrianglesHaveEqualAreas) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // First 3 children are corner triangles and should have equal areas
  std::vector<double> corner_areas;
  for (size_t i = 0; i < 3; ++i) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    corner_areas.push_back(compute_triangle_area(child_verts));
  }

  for (size_t i = 1; i < corner_areas.size(); ++i) {
    EXPECT_NEAR(corner_areas[0], corner_areas[i], 1e-10)
        << "Corner triangle " << i << " area differs from corner triangle 0";
  }
}

TEST_F(TriangleRefinementTest, CenterTriangleHasSameAreaAsCorners) {
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // Center triangle should have the same area as corner triangles
  auto corner_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                         refined.child_connectivity[0]);
  double corner_area = compute_triangle_area(corner_verts);

  auto center_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                         refined.child_connectivity[3]);
  double center_area = compute_triangle_area(center_verts);

  EXPECT_NEAR(corner_area, center_area, 1e-10)
      << "Center triangle area should equal corner triangle areas";
}

TEST_F(TriangleRefinementTest, RefinedTrianglePreservesOrientation) {
  // Check that child triangles preserve the orientation of the parent
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, 0);

  // Compute parent normal
  const auto& v0 = unit_triangle_verts[0];
  const auto& v1 = unit_triangle_verts[1];
  const auto& v2 = unit_triangle_verts[2];

  std::array<double, 3> edge01 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> edge02 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

  std::array<double, 3> parent_normal = {
    edge01[1] * edge02[2] - edge01[2] * edge02[1],
    edge01[2] * edge02[0] - edge01[0] * edge02[2],
    edge01[0] * edge02[1] - edge01[1] * edge02[0]
  };

  // Check each child has the same orientation (positive dot product with parent normal)
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);

    const auto& c0 = child_verts[0];
    const auto& c1 = child_verts[1];
    const auto& c2 = child_verts[2];

    std::array<double, 3> c_edge01 = {c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]};
    std::array<double, 3> c_edge02 = {c2[0] - c0[0], c2[1] - c0[1], c2[2] - c0[2]};

    std::array<double, 3> child_normal = {
      c_edge01[1] * c_edge02[2] - c_edge01[2] * c_edge02[1],
      c_edge01[2] * c_edge02[0] - c_edge01[0] * c_edge02[2],
      c_edge01[0] * c_edge02[1] - c_edge01[1] * c_edge02[0]
    };

    double dot_product = parent_normal[0] * child_normal[0] +
                        parent_normal[1] * child_normal[1] +
                        parent_normal[2] * child_normal[2];

    EXPECT_GT(dot_product, 0.0)
        << "Child " << i << " has opposite orientation from parent";
  }
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(TriangleRefinementTest, ThrowsOnInvalidCellType) {
  EXPECT_THROW(
    rule.refine(unit_triangle_verts, CellType::Tetra, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when CellType is not TRIANGLE";
}

TEST_F(TriangleRefinementTest, ThrowsOnInvalidVertexCount) {
  std::vector<std::array<double, 3>> invalid_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0}
    // Only 2 vertices instead of 3
  };

  EXPECT_THROW(
    rule.refine(invalid_verts, CellType::Triangle, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when vertex count is not 3";
}

// ==============================================================================
// Parameterized Tests for Multiple Refinement Levels
// ==============================================================================

class TriangleMultiLevelRefinementTest : public ::testing::TestWithParam<size_t> {
protected:
  void SetUp() override {
    unit_triangle_verts = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0}
    };
  }

  std::vector<std::array<double, 3>> unit_triangle_verts;
  TriangleRefinementRule rule;
};

TEST_P(TriangleMultiLevelRefinementTest, CorrectChildLevelAtEachLevel) {
  size_t level = GetParam();

  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, level);

  EXPECT_EQ(refined.child_level, level + 1)
      << "Child level incorrect at refinement level " << level;
}

TEST_P(TriangleMultiLevelRefinementTest, AreaConservationAtEachLevel) {
  size_t level = GetParam();

  double parent_area = compute_triangle_area(unit_triangle_verts);
  auto refined = rule.refine(unit_triangle_verts, CellType::Triangle,
                             RefinementPattern::ISOTROPIC, level);

  double children_area_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(unit_triangle_verts, refined.new_vertices, child_conn);
    children_area_sum += compute_triangle_area(child_verts);
  }

  EXPECT_NEAR(parent_area, children_area_sum, 1e-10)
      << "Area conservation fails at refinement level " << level;
}

INSTANTIATE_TEST_SUITE_P(
  MultiLevel,
  TriangleMultiLevelRefinementTest,
  ::testing::Values(0, 1, 2, 3, 5, 9)
);

} // namespace test
} // namespace svmp
