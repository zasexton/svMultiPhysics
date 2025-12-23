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
 * @file test_PyramidRefinement.cpp
 * @brief Comprehensive unit tests for pyramid h-refinement rules
 *
 * This test suite validates:
 * - Basic refinement correctness (child count, vertex count)
 * - Geometric verification (volume conservation, edge midpoints)
 * - Topology validation (distinct vertices, proper connectivity)
 * - Special pyramid properties (inverted central pyramid, apex pyramid)
 * - Edge cases (degenerate geometries, various orientations)
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
 * @brief Compute volume of a tetrahedron using scalar triple product
 *
 * Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
 */
double compute_tet_volume(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 4) {
    return 0.0;
  }

  const auto& v0 = verts[0];
  const auto& v1 = verts[1];
  const auto& v2 = verts[2];
  const auto& v3 = verts[3];

  // Vectors from v0 to other vertices
  std::array<double, 3> a = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> b = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  std::array<double, 3> c = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};

  // Scalar triple product: a · (b × c)
  double scalar_triple = a[0] * (b[1] * c[2] - b[2] * c[1]) +
                         a[1] * (b[2] * c[0] - b[0] * c[2]) +
                         a[2] * (b[0] * c[1] - b[1] * c[0]);

  return std::abs(scalar_triple) / 6.0;
}

/**
 * @brief Compute volume of a pyramid
 *
 * Volume = (1/3) * base_area * height
 * For pyramid with square base {v0,v1,v2,v3} and apex v4
 */
double compute_pyramid_volume(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 5) {
    return 0.0;
  }

  // Base quadrilateral vertices
  const auto& v0 = verts[0];
  const auto& v1 = verts[1];
  const auto& v2 = verts[2];
  const auto& v3 = verts[3];
  const auto& apex = verts[4];

  // Compute base area (for quadrilateral, split into two triangles)
  // Triangle 1: v0, v1, v2
  std::array<double, 3> edge01 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> edge02 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

  std::array<double, 3> cross1 = {
    edge01[1] * edge02[2] - edge01[2] * edge02[1],
    edge01[2] * edge02[0] - edge01[0] * edge02[2],
    edge01[0] * edge02[1] - edge01[1] * edge02[0]
  };

  double area1 = 0.5 * std::sqrt(cross1[0]*cross1[0] + cross1[1]*cross1[1] + cross1[2]*cross1[2]);

  // Triangle 2: v0, v2, v3
  std::array<double, 3> edge03 = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};

  std::array<double, 3> cross2 = {
    edge02[1] * edge03[2] - edge02[2] * edge03[1],
    edge02[2] * edge03[0] - edge02[0] * edge03[2],
    edge02[0] * edge03[1] - edge02[1] * edge03[0]
  };

  double area2 = 0.5 * std::sqrt(cross2[0]*cross2[0] + cross2[1]*cross2[1] + cross2[2]*cross2[2]);

  double base_area = area1 + area2;

  // Compute height (perpendicular distance from apex to base plane)
  // Use the normal to the base plane (from cross product)
  std::array<double, 3> normal = {cross1[0] + cross2[0], cross1[1] + cross2[1], cross1[2] + cross2[2]};
  double normal_mag = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);

  if (normal_mag < 1e-15) {
    return 0.0; // Degenerate base
  }

  // Normalize the normal
  normal[0] /= normal_mag;
  normal[1] /= normal_mag;
  normal[2] /= normal_mag;

  // Vector from v0 to apex
  std::array<double, 3> v0_to_apex = {apex[0] - v0[0], apex[1] - v0[1], apex[2] - v0[2]};

  // Height is the absolute value of the dot product
  double height = std::abs(v0_to_apex[0] * normal[0] + v0_to_apex[1] * normal[1] + v0_to_apex[2] * normal[2]);

  return (1.0 / 3.0) * base_area * height;
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
 * @brief Check if point is at center of quadrilateral face
 */
bool is_face_center(const std::array<double, 3>& p,
                    const std::array<double, 3>& v0,
                    const std::array<double, 3>& v1,
                    const std::array<double, 3>& v2,
                    const std::array<double, 3>& v3,
                    double tol = 1e-10) {
  std::array<double, 3> expected = {
    0.25 * (v0[0] + v1[0] + v2[0] + v3[0]),
    0.25 * (v0[1] + v1[1] + v2[1] + v3[1]),
    0.25 * (v0[2] + v1[2] + v2[2] + v3[2])
  };
  return distance(p, expected) < tol;
}

/**
 * @brief Compute volume of a child element (pyramid or tetrahedron)
 *
 * Determines element type based on vertex count:
 * - 5 vertices: pyramid
 * - 4 vertices: tetrahedron
 */
double compute_child_volume(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() == 5) {
    return compute_pyramid_volume(verts);
  } else if (verts.size() == 4) {
    return compute_tet_volume(verts);
  } else {
    return 0.0;
  }
}

// ==============================================================================
// Test Fixture
// ==============================================================================

class PyramidRefinementTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create unit pyramid: square base at z=0, apex at (0.5, 0.5, 1)
    unit_pyramid_verts = {
      {0.0, 0.0, 0.0},   // 0: base corner
      {1.0, 0.0, 0.0},   // 1: base corner
      {1.0, 1.0, 0.0},   // 2: base corner
      {0.0, 1.0, 0.0},   // 3: base corner
      {0.5, 0.5, 1.0}    // 4: apex
    };

    unit_pyramid_volume = compute_pyramid_volume(unit_pyramid_verts);
  }

  std::vector<std::array<double, 3>> unit_pyramid_verts;
  double unit_pyramid_volume;
  PyramidRefinementRule rule;
};

// ==============================================================================
// Basic Correctness Tests
// ==============================================================================

TEST_F(PyramidRefinementTest, ProducesTenChildren) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 10)
      << "Pyramid RED refinement should produce 10 children (6 pyramids + 4 tetrahedra)";
}

TEST_F(PyramidRefinementTest, CreatesTenNewVertices) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.new_vertices.size(), 10)
      << "Pyramid refinement should create 10 new vertices "
      << "(8 edge midpoints + 1 base center + 1 mid-height center)";
}

TEST_F(PyramidRefinementTest, IncrementsRefinementLevel) {
  size_t initial_level = 0;
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, initial_level);

  EXPECT_EQ(refined.child_level, initial_level + 1)
      << "Child refinement level should be parent level + 1";
}

TEST_F(PyramidRefinementTest, StoresCorrectRefinementPattern) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::RED)
      << "Pyramid isotropic refinement uses RED pattern";
}

TEST_F(PyramidRefinementTest, CanRefineReturnsTrue) {
  EXPECT_TRUE(rule.can_refine(CellType::Pyramid, 0));
  EXPECT_TRUE(rule.can_refine(CellType::Pyramid, 5));
}

TEST_F(PyramidRefinementTest, NumChildrenReturnsTen) {
  size_t num_children = rule.num_children(CellType::Pyramid, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 10);
}

// ==============================================================================
// Topology Validation Tests
// ==============================================================================

TEST_F(PyramidRefinementTest, ChildrenHaveCorrectVertexCounts) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  // Children 0-5: pyramids (5 vertices each)
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(refined.child_connectivity[i].size(), 5)
        << "Child " << i << " should have 5 vertices (pyramid topology)";
  }

  // Children 6-9: tetrahedra (4 vertices each)
  for (size_t i = 6; i < 10; ++i) {
    EXPECT_EQ(refined.child_connectivity[i].size(), 4)
        << "Child " << i << " should have 4 vertices (tetrahedron topology)";
  }
}

TEST_F(PyramidRefinementTest, AllChildrenHaveDistinctVertices) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    EXPECT_TRUE(has_distinct_vertices(child))
        << "Child " << i << " has repeated vertices: ["
        << child[0] << ", " << child[1] << ", " << child[2] << ", "
        << child[3] << ", " << child[4] << "]";
  }
}

TEST_F(PyramidRefinementTest, ChildVertexIndicesAreValid) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_verts = unit_pyramid_verts.size() + refined.new_vertices.size();

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    for (size_t j = 0; j < child.size(); ++j) {
      EXPECT_LT(child[j], total_verts)
          << "Child " << i << " vertex " << j << " has invalid index " << child[j];
    }
  }
}

TEST_F(PyramidRefinementTest, VerifyExpectedConnectivity) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  // Expected connectivity from implementation (10 children: 6 pyramids + 4 tetrahedra)
  std::vector<std::vector<size_t>> expected_connectivity = {
    // Pyramids (children 0-5)
    {0, 5, 13, 8, 9},      // Child 0: Corner pyramid at v0
    {5, 1, 6, 13, 10},     // Child 1: Corner pyramid at v1
    {13, 6, 2, 7, 11},     // Child 2: Corner pyramid at v2
    {8, 13, 7, 3, 12},     // Child 3: Corner pyramid at v3
    {5, 6, 7, 8, 14},      // Child 4: Central pyramid
    {9, 10, 11, 12, 4},    // Child 5: Apex pyramid
    // Tetrahedra (children 6-9)
    {5, 10, 14, 9},        // Child 6: Tet connector 0-1
    {6, 11, 14, 10},       // Child 7: Tet connector 1-2
    {7, 12, 14, 11},       // Child 8: Tet connector 2-3
    {8, 9, 14, 12}         // Child 9: Tet connector 3-0
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

TEST_F(PyramidRefinementTest, VolumeConservation) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_pyramid_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);

    // Children 0-5 are pyramids, children 6-9 are tetrahedra
    double child_vol = (i < 6) ? compute_pyramid_volume(child_verts)
                               : compute_tet_volume(child_verts);

    EXPECT_GT(child_vol, 0.0) << "Child " << i << " has non-positive volume";
    children_volume_sum += child_vol;
  }

  EXPECT_NEAR(unit_pyramid_volume, children_volume_sum, 1e-10)
      << "Volume conservation violated: parent volume = " << unit_pyramid_volume
      << ", children sum = " << children_volume_sum;
}

TEST_F(PyramidRefinementTest, BaseEdgeMidpointsAreCorrect) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_GE(refined.new_vertices.size(), 4);

  // Base edge midpoints (indices 5, 6, 7, 8)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], unit_pyramid_verts[0], unit_pyramid_verts[1]))
      << "Vertex 5 should be midpoint of edge 0-1";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], unit_pyramid_verts[1], unit_pyramid_verts[2]))
      << "Vertex 6 should be midpoint of edge 1-2";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[2], unit_pyramid_verts[2], unit_pyramid_verts[3]))
      << "Vertex 7 should be midpoint of edge 2-3";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[3], unit_pyramid_verts[3], unit_pyramid_verts[0]))
      << "Vertex 8 should be midpoint of edge 3-0";
}

TEST_F(PyramidRefinementTest, LateralEdgeMidpointsAreCorrect) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_GE(refined.new_vertices.size(), 8);

  // Lateral edge midpoints (indices 9, 10, 11, 12)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[4], unit_pyramid_verts[0], unit_pyramid_verts[4]))
      << "Vertex 9 should be midpoint of edge 0-4";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[5], unit_pyramid_verts[1], unit_pyramid_verts[4]))
      << "Vertex 10 should be midpoint of edge 1-4";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[6], unit_pyramid_verts[2], unit_pyramid_verts[4]))
      << "Vertex 11 should be midpoint of edge 2-4";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[7], unit_pyramid_verts[3], unit_pyramid_verts[4]))
      << "Vertex 12 should be midpoint of edge 3-4";
}

TEST_F(PyramidRefinementTest, BaseCenterIsCorrect) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_EQ(refined.new_vertices.size(), 10);

  // Base center (index 13 = new_vertices[8])
  EXPECT_TRUE(is_face_center(refined.new_vertices[8],
                             unit_pyramid_verts[0], unit_pyramid_verts[1],
                             unit_pyramid_verts[2], unit_pyramid_verts[3]))
      << "Vertex 13 should be at the center of the base quadrilateral";
}

TEST_F(PyramidRefinementTest, AllChildrenHavePositiveVolume) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_pyramid_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);

    // Children 0-5 are pyramids, children 6-9 are tetrahedra
    double child_vol = (i < 6) ? compute_pyramid_volume(child_verts)
                               : compute_tet_volume(child_verts);

    EXPECT_GT(child_vol, 1e-12) << "Child " << i << " has near-zero or negative volume";
  }
}

TEST_F(PyramidRefinementTest, CornerPyramidsHaveEqualVolumes) {
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  // The 4 corner pyramids (children 0-3) should have equal volumes
  std::vector<double> corner_volumes;
  for (size_t i = 0; i < 4; ++i) {
    auto child_verts = get_child_vertices(unit_pyramid_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    corner_volumes.push_back(compute_pyramid_volume(child_verts));
  }

  for (size_t i = 1; i < corner_volumes.size(); ++i) {
    EXPECT_NEAR(corner_volumes[0], corner_volumes[i], 1e-10)
        << "Corner pyramid " << i << " volume differs from corner pyramid 0";
  }
}

// ==============================================================================
// Special Pyramid Properties Tests
// ==============================================================================

TEST_F(PyramidRefinementTest, CentralPyramidHasMidHeightCenterAsApex) {
  // The central pyramid (child 4) should have mid-height center as apex
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  const auto& central_pyramid = refined.child_connectivity[4];
  ASSERT_EQ(central_pyramid.size(), 5);

  // The apex (last vertex) should be vertex 14 (mid-height center)
  EXPECT_EQ(central_pyramid[4], 14)
      << "Central pyramid should have mid-height center (vertex 14) as apex";
}

TEST_F(PyramidRefinementTest, ApexPyramidHasOriginalApex) {
  // The apex pyramid (child 5) should have the original apex (vertex 4)
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  const auto& apex_pyramid = refined.child_connectivity[5];
  ASSERT_EQ(apex_pyramid.size(), 5);

  // The apex (last vertex) should be vertex 4 (original apex)
  EXPECT_EQ(apex_pyramid[4], 4)
      << "Apex pyramid should have original apex (vertex 4)";
}

TEST_F(PyramidRefinementTest, ApexPyramidBaseLiesOnLateralMidpoints) {
  // The apex pyramid base should be formed by the 4 lateral edge midpoints
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  const auto& apex_pyramid = refined.child_connectivity[5];
  std::vector<size_t> expected_base = {9, 10, 11, 12};

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(apex_pyramid[i], expected_base[i])
        << "Apex pyramid base vertex " << i << " is incorrect";
  }
}

// ==============================================================================
// Edge Cases and Stress Tests
// ==============================================================================

TEST_F(PyramidRefinementTest, ScaledPyramidVolumeConservation) {
  // Test with a scaled pyramid
  std::vector<std::array<double, 3>> scaled_pyramid = {
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0},
    {2.0, 2.0, 0.0},
    {0.0, 2.0, 0.0},
    {1.0, 1.0, 3.0}
  };

  double parent_volume = compute_pyramid_volume(scaled_pyramid);
  auto refined = rule.refine(scaled_pyramid, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(scaled_pyramid, refined.new_vertices, child_conn);
    children_volume_sum += compute_child_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-9)
      << "Volume conservation fails for scaled pyramid";
}

TEST_F(PyramidRefinementTest, TranslatedPyramidVolumeConservation) {
  // Test with a translated pyramid
  std::vector<std::array<double, 3>> translated_pyramid = {
    {10.0, 20.0, 30.0},
    {11.0, 20.0, 30.0},
    {11.0, 21.0, 30.0},
    {10.0, 21.0, 30.0},
    {10.5, 20.5, 31.0}
  };

  double parent_volume = compute_pyramid_volume(translated_pyramid);
  auto refined = rule.refine(translated_pyramid, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(translated_pyramid, refined.new_vertices, child_conn);
    children_volume_sum += compute_child_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for translated pyramid";
}

TEST_F(PyramidRefinementTest, TallPyramidVolumeConservation) {
  // Test with a tall pyramid (high aspect ratio)
  std::vector<std::array<double, 3>> tall_pyramid = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.5, 0.5, 10.0}
  };

  double parent_volume = compute_pyramid_volume(tall_pyramid);
  auto refined = rule.refine(tall_pyramid, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(tall_pyramid, refined.new_vertices, child_conn);
    children_volume_sum += compute_child_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-9)
      << "Volume conservation fails for tall pyramid";
}

TEST_F(PyramidRefinementTest, FlatPyramidVolumeConservation) {
  // Test with a flat pyramid (low aspect ratio)
  std::vector<std::array<double, 3>> flat_pyramid = {
    {0.0, 0.0, 0.0},
    {10.0, 0.0, 0.0},
    {10.0, 10.0, 0.0},
    {0.0, 10.0, 0.0},
    {5.0, 5.0, 0.1}
  };

  double parent_volume = compute_pyramid_volume(flat_pyramid);
  EXPECT_GT(parent_volume, 0.0) << "Parent flat pyramid should have positive volume";

  auto refined = rule.refine(flat_pyramid, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(flat_pyramid, refined.new_vertices, child_conn);
    children_volume_sum += compute_child_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for flat pyramid";
}

TEST_F(PyramidRefinementTest, SmallPyramidVolumeConservation) {
  // Test with a very small pyramid
  std::vector<std::array<double, 3>> small_pyramid = {
    {0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0},
    {0.001, 0.001, 0.0},
    {0.0, 0.001, 0.0},
    {0.0005, 0.0005, 0.001}
  };

  double parent_volume = compute_pyramid_volume(small_pyramid);
  EXPECT_GT(parent_volume, 0.0) << "Parent small pyramid should have positive volume";

  auto refined = rule.refine(small_pyramid, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(small_pyramid, refined.new_vertices, child_conn);
    children_volume_sum += compute_child_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-15)
      << "Volume conservation fails for small pyramid";
}

TEST_F(PyramidRefinementTest, RotatedPyramidProducesTenChildren) {
  // Test with a rotated pyramid (45 degrees about z-axis)
  double angle = M_PI / 4.0;
  double cos_a = std::cos(angle);
  double sin_a = std::sin(angle);

  std::vector<std::array<double, 3>> rotated_pyramid = {
    {0.0, 0.0, 0.0},
    {cos_a, sin_a, 0.0},
    {cos_a - sin_a, sin_a + cos_a, 0.0},
    {-sin_a, cos_a, 0.0},
    {(cos_a - sin_a)/2.0, (sin_a + cos_a)/2.0, 1.0}
  };

  auto refined = rule.refine(rotated_pyramid, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 10);
  EXPECT_EQ(refined.new_vertices.size(), 10);
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(PyramidRefinementTest, ThrowsOnInvalidCellType) {
  EXPECT_THROW(
    rule.refine(unit_pyramid_verts, CellType::Tetra, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when CellType is not PYRAMID";
}

TEST_F(PyramidRefinementTest, ThrowsOnInvalidVertexCount) {
  std::vector<std::array<double, 3>> invalid_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {1.0, 1.0, 0.0}
    // Only 3 vertices instead of 5
  };

  EXPECT_THROW(
    rule.refine(invalid_verts, CellType::Pyramid, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when vertex count is not 5";
}

// ==============================================================================
// Parameterized Tests for Multiple Refinement Levels
// ==============================================================================

class PyramidMultiLevelRefinementTest : public ::testing::TestWithParam<size_t> {
protected:
  void SetUp() override {
    unit_pyramid_verts = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.5, 0.5, 1.0}
    };
  }

  std::vector<std::array<double, 3>> unit_pyramid_verts;
  PyramidRefinementRule rule;
};

TEST_P(PyramidMultiLevelRefinementTest, CorrectChildLevelAtEachLevel) {
  size_t level = GetParam();

  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, level);

  EXPECT_EQ(refined.child_level, level + 1)
      << "Child level incorrect at refinement level " << level;
}

TEST_P(PyramidMultiLevelRefinementTest, VolumeConservationAtEachLevel) {
  size_t level = GetParam();

  double parent_volume = compute_pyramid_volume(unit_pyramid_verts);
  auto refined = rule.refine(unit_pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, level);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(unit_pyramid_verts, refined.new_vertices, child_conn);
    children_volume_sum += compute_child_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails at refinement level " << level;
}

INSTANTIATE_TEST_SUITE_P(
  MultiLevel,
  PyramidMultiLevelRefinementTest,
  ::testing::Values(0, 1, 2, 3, 5, 9)
);

} // namespace test
} // namespace svmp
