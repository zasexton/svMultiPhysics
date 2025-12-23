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
 * @file test_TetrahedronRefinement.cpp
 * @brief Comprehensive unit tests for tetrahedron h-refinement rules
 *
 * This test suite validates:
 * - Basic refinement correctness (child count, vertex count)
 * - Geometric verification (volume conservation, edge midpoints)
 * - Topology validation (distinct vertices, proper connectivity)
 * - Edge cases (degenerate geometries, various orientations)
 * - Octahedron subdivision correctness (4 interior tetrahedra)
 *
 * Tetrahedron RED refinement:
 * - 4 parent vertices -> 6 edge midpoints -> 10 total vertices
 * - 8 child tetrahedra: 4 corner tets + 4 interior tets (octahedron subdivision)
 * - Volume conservation: V_parent = 8 * V_child
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
 * Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
 */
double compute_tetrahedron_volume(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 4) {
    return 0.0;
  }

  const auto& v0 = verts[0];
  const auto& v1 = verts[1];
  const auto& v2 = verts[2];
  const auto& v3 = verts[3];

  // Edge vectors from v0
  std::array<double, 3> e1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> e2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  std::array<double, 3> e3 = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};

  // Compute determinant (scalar triple product)
  double det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
             - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
             + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);

  return std::abs(det) / 6.0;
}

/**
 * @brief Compute signed Jacobian determinant (6x volume) for a tetrahedron
 *
 * det = det(v1-v0, v2-v0, v3-v0)
 */
double compute_tetrahedron_signed_det(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 4) {
    return 0.0;
  }

  const auto& v0 = verts[0];
  const auto& v1 = verts[1];
  const auto& v2 = verts[2];
  const auto& v3 = verts[3];

  std::array<double, 3> e1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  std::array<double, 3> e2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  std::array<double, 3> e3 = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};

  return e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
       - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
       + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
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
 * @brief Check if a tetrahedron has positive volume (non-degenerate)
 */
bool has_positive_volume(const std::vector<std::array<double, 3>>& verts, double tol = 1e-12) {
  return compute_tetrahedron_volume(verts) > tol;
}

// ==============================================================================
// Test Fixture
// ==============================================================================

class TetrahedronRefinementTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create unit tetrahedron: right-angle corner at origin
    unit_tet_verts = {
      {0.0, 0.0, 0.0},  // 0: origin corner
      {1.0, 0.0, 0.0},  // 1: x-axis corner
      {0.0, 1.0, 0.0},  // 2: y-axis corner
      {0.0, 0.0, 1.0}   // 3: z-axis corner
    };

    unit_tet_volume = compute_tetrahedron_volume(unit_tet_verts);
  }

  std::vector<std::array<double, 3>> unit_tet_verts;
  double unit_tet_volume;
  TetrahedronRefinementRule rule;
};

// ==============================================================================
// Basic Correctness Tests
// ==============================================================================

TEST_F(TetrahedronRefinementTest, ProducesEightChildren) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8)
      << "Tetrahedron RED refinement should produce 8 children";
}

TEST_F(TetrahedronRefinementTest, CreatesSixNewVertices) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.new_vertices.size(), 6)
      << "Tetrahedron refinement should create 6 new vertices (6 edge midpoints)";
}

TEST_F(TetrahedronRefinementTest, IncrementsRefinementLevel) {
  size_t initial_level = 0;
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, initial_level);

  EXPECT_EQ(refined.child_level, initial_level + 1)
      << "Child refinement level should be parent level + 1";
}

TEST_F(TetrahedronRefinementTest, StoresCorrectRefinementPattern) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::RED)
      << "Tetrahedron isotropic refinement uses RED pattern";
}

TEST_F(TetrahedronRefinementTest, CanRefineReturnsTrue) {
  EXPECT_TRUE(rule.can_refine(CellType::Tetra, 0));
  EXPECT_TRUE(rule.can_refine(CellType::Tetra, 5));
}

TEST_F(TetrahedronRefinementTest, NumChildrenReturnsEight) {
  size_t num_children = rule.num_children(CellType::Tetra, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 8);
}

TEST_F(TetrahedronRefinementTest, FaceGreenPreservesParentOrientation) {
  // A tetra with apex below the base face requires a different vertex ordering
  // to maintain positive orientation (similar to RedGreenClosure3D fixtures).
  std::vector<std::array<double, 3>> tet = {
      {0.0, 0.0, 0.0},   // 0
      {0.0, 1.0, 0.0},   // 1  (swapped vs canonical)
      {1.0, 0.0, 0.0},   // 2
      {0.0, 0.0, -1.0}   // 3 (apex)
  };

  const double det_parent = compute_tetrahedron_signed_det(tet);
  ASSERT_GT(det_parent, 0.0);

  // Refine the face opposite the apex (local vertex 3).
  auto refined = rule.refine(tet, CellType::Tetra, RefinementSpec{RefinementPattern::GREEN, 3u}, 0);
  ASSERT_EQ(refined.child_connectivity.size(), 4u);
  ASSERT_EQ(refined.new_vertices.size(), 3u);

  for (const auto& child_conn : refined.child_connectivity) {
    const auto child_verts = get_child_vertices(tet, refined.new_vertices, child_conn);
    const double det_child = compute_tetrahedron_signed_det(child_verts);
    EXPECT_GT(det_child * det_parent, 0.0) << "Child tet must preserve parent orientation";
  }
}

// ==============================================================================
// Topology Validation Tests
// ==============================================================================

TEST_F(TetrahedronRefinementTest, AllChildrenHaveFourVertices) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i].size(), 4)
        << "Child " << i << " should have 4 vertices (tetrahedron topology)";
  }
}

TEST_F(TetrahedronRefinementTest, AllChildrenHaveDistinctVertices) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    EXPECT_TRUE(has_distinct_vertices(child))
        << "Child " << i << " has repeated vertices: ["
        << child[0] << ", " << child[1] << ", " << child[2] << ", " << child[3] << "]";
  }
}

TEST_F(TetrahedronRefinementTest, ChildVertexIndicesAreValid) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_verts = unit_tet_verts.size() + refined.new_vertices.size();

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    for (size_t j = 0; j < child.size(); ++j) {
      EXPECT_LT(child[j], total_verts)
          << "Child " << i << " vertex " << j << " has invalid index " << child[j];
    }
  }
}

TEST_F(TetrahedronRefinementTest, VerifyExpectedConnectivity) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  // Expected connectivity from implementation
  // Original vertices: 0, 1, 2, 3
  // New vertices: 4 (m01), 5 (m02), 6 (m03), 7 (m12), 8 (m13), 9 (m23)
  std::vector<std::vector<size_t>> expected_connectivity = {
    // Corner tetrahedra (at original vertices)
    {0, 4, 5, 6},  // Child 0: Corner at vertex 0
    {4, 1, 7, 8},  // Child 1: Corner at vertex 1
    {5, 7, 2, 9},  // Child 2: Corner at vertex 2
    {6, 8, 9, 3},  // Child 3: Corner at vertex 3

    // Octahedron subdivision (4 interior tetrahedra)
    {4, 5, 6, 8},  // Child 4: Interior octahedron tet
    {4, 5, 8, 7},  // Child 5: Interior octahedron tet
    {5, 6, 8, 9},  // Child 6: Interior octahedron tet
    {5, 7, 9, 8}   // Child 7: Interior octahedron tet (oriented)
  };

  ASSERT_EQ(refined.child_connectivity.size(), expected_connectivity.size());

  for (size_t i = 0; i < expected_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i], expected_connectivity[i])
        << "Child " << i << " has incorrect connectivity";
  }
}

TEST_F(TetrahedronRefinementTest, AllChildrenHavePositiveSignedDeterminant) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts =
        get_child_vertices(unit_tet_verts, refined.new_vertices, refined.child_connectivity[i]);
    EXPECT_GT(compute_tetrahedron_signed_det(child_verts), 0.0)
        << "Child " << i << " is inverted (negative determinant)";
  }
}

TEST_F(TetrahedronRefinementTest, CornerChildrenUseOriginalVertices) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  // First 4 children are corner tets, each should use one original vertex
  for (size_t i = 0; i < 4; ++i) {
    const auto& child = refined.child_connectivity[i];
    bool has_original = false;
    for (size_t v : child) {
      if (v < 4) {  // Original vertex indices are 0-3
        has_original = true;
        break;
      }
    }
    EXPECT_TRUE(has_original)
        << "Corner child " << i << " should contain at least one original vertex";
  }
}

TEST_F(TetrahedronRefinementTest, InteriorChildrenUseOnlyNewVertices) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  // Last 4 children are interior tets (octahedron), should only use new vertices
  for (size_t i = 4; i < 8; ++i) {
    const auto& child = refined.child_connectivity[i];
    for (size_t v : child) {
      EXPECT_GE(v, 4)
          << "Interior child " << i << " should only use new vertices (indices >= 4)";
    }
  }
}

// ==============================================================================
// Geometric Verification Tests
// ==============================================================================

TEST_F(TetrahedronRefinementTest, VolumeConservation) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_vol = compute_tetrahedron_volume(child_verts);

    EXPECT_GT(child_vol, 0.0) << "Child " << i << " has non-positive volume";
    children_volume_sum += child_vol;
  }

  EXPECT_NEAR(unit_tet_volume, children_volume_sum, 1e-10)
      << "Volume conservation violated: parent volume = " << unit_tet_volume
      << ", children sum = " << children_volume_sum;
}

TEST_F(TetrahedronRefinementTest, EdgeMidpointsAreCorrect) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_EQ(refined.new_vertices.size(), 6);

  // Verify all 6 edge midpoints
  // m01 (edge 0-1)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], unit_tet_verts[0], unit_tet_verts[1]))
      << "Vertex 4 should be midpoint of edge 0-1";

  // m02 (edge 0-2)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], unit_tet_verts[0], unit_tet_verts[2]))
      << "Vertex 5 should be midpoint of edge 0-2";

  // m03 (edge 0-3)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[2], unit_tet_verts[0], unit_tet_verts[3]))
      << "Vertex 6 should be midpoint of edge 0-3";

  // m12 (edge 1-2)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[3], unit_tet_verts[1], unit_tet_verts[2]))
      << "Vertex 7 should be midpoint of edge 1-2";

  // m13 (edge 1-3)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[4], unit_tet_verts[1], unit_tet_verts[3]))
      << "Vertex 8 should be midpoint of edge 1-3";

  // m23 (edge 2-3)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[5], unit_tet_verts[2], unit_tet_verts[3]))
      << "Vertex 9 should be midpoint of edge 2-3";
}

TEST_F(TetrahedronRefinementTest, AllChildrenHavePositiveVolume) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_vol = compute_tetrahedron_volume(child_verts);

    EXPECT_GT(child_vol, 1e-12) << "Child " << i << " has near-zero or negative volume";
  }
}

TEST_F(TetrahedronRefinementTest, ChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  std::vector<double> child_volumes;
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    child_volumes.push_back(compute_tetrahedron_volume(child_verts));
  }

  double expected_child_vol = unit_tet_volume / 8.0;

  for (size_t i = 0; i < child_volumes.size(); ++i) {
    EXPECT_NEAR(child_volumes[i], expected_child_vol, 1e-10)
        << "Child " << i << " volume differs from expected equal subdivision";
  }
}

TEST_F(TetrahedronRefinementTest, CornerChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  // First 4 children are corner tets and should have equal volumes
  std::vector<double> corner_volumes;
  for (size_t i = 0; i < 4; ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    corner_volumes.push_back(compute_tetrahedron_volume(child_verts));
  }

  for (size_t i = 1; i < corner_volumes.size(); ++i) {
    EXPECT_NEAR(corner_volumes[0], corner_volumes[i], 1e-10)
        << "Corner child " << i << " volume differs from corner child 0";
  }
}

TEST_F(TetrahedronRefinementTest, InteriorChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  // Last 4 children are interior octahedron tets and should have equal volumes
  std::vector<double> interior_volumes;
  for (size_t i = 4; i < 8; ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    interior_volumes.push_back(compute_tetrahedron_volume(child_verts));
  }

  for (size_t i = 1; i < interior_volumes.size(); ++i) {
    EXPECT_NEAR(interior_volumes[0], interior_volumes[i], 1e-10)
        << "Interior child " << (i+4) << " volume differs from interior child 4";
  }
}

// ==============================================================================
// Edge Cases and Stress Tests
// ==============================================================================

TEST_F(TetrahedronRefinementTest, ScaledTetrahedronVolumeConservation) {
  // Test with a scaled tetrahedron
  std::vector<std::array<double, 3>> scaled_tet = {
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0},
    {0.0, 2.0, 0.0},
    {0.0, 0.0, 2.0}
  };

  double parent_volume = compute_tetrahedron_volume(scaled_tet);
  auto refined = rule.refine(scaled_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(scaled_tet, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-9)
      << "Volume conservation fails for scaled tetrahedron";
}

TEST_F(TetrahedronRefinementTest, TranslatedTetrahedronVolumeConservation) {
  // Test with a translated tetrahedron
  std::vector<std::array<double, 3>> translated_tet = {
    {10.0, 20.0, 30.0},
    {11.0, 20.0, 30.0},
    {10.0, 21.0, 30.0},
    {10.0, 20.0, 31.0}
  };

  double parent_volume = compute_tetrahedron_volume(translated_tet);
  auto refined = rule.refine(translated_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(translated_tet, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for translated tetrahedron";
}

TEST_F(TetrahedronRefinementTest, RotatedTetrahedronProducesEightChildren) {
  // Test with a rotated tetrahedron (45 degrees about z-axis)
  double angle = M_PI / 4.0;
  double cos_a = std::cos(angle);
  double sin_a = std::sin(angle);

  std::vector<std::array<double, 3>> rotated_tet = {
    {0.0, 0.0, 0.0},
    {cos_a, sin_a, 0.0},
    {-sin_a, cos_a, 0.0},
    {0.0, 0.0, 1.0}
  };

  auto refined = rule.refine(rotated_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8);
  EXPECT_EQ(refined.new_vertices.size(), 6);
}

TEST_F(TetrahedronRefinementTest, SmallTetrahedronVolumeConservation) {
  // Test with a very small tetrahedron
  std::vector<std::array<double, 3>> small_tet = {
    {0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0},
    {0.0, 0.001, 0.0},
    {0.0, 0.0, 0.001}
  };

  double parent_volume = compute_tetrahedron_volume(small_tet);
  EXPECT_GT(parent_volume, 0.0) << "Parent small tetrahedron should have positive volume";

  auto refined = rule.refine(small_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(small_tet, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-15)
      << "Volume conservation fails for small tetrahedron";
}

TEST_F(TetrahedronRefinementTest, RegularTetrahedronVolumeConservation) {
  // Test with a regular tetrahedron (all edges equal length)
  double s = 1.0 / std::sqrt(2.0);
  std::vector<std::array<double, 3>> regular_tet = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.5, std::sqrt(3.0)/2.0, 0.0},
    {0.5, std::sqrt(3.0)/6.0, std::sqrt(2.0/3.0)}
  };

  double parent_volume = compute_tetrahedron_volume(regular_tet);
  EXPECT_GT(parent_volume, 0.0) << "Regular tetrahedron should have positive volume";

  auto refined = rule.refine(regular_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(regular_tet, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for regular tetrahedron";
}

TEST_F(TetrahedronRefinementTest, FlatTetrahedronVolumeConservation) {
  // Test with a flat tetrahedron (low aspect ratio)
  std::vector<std::array<double, 3>> flat_tet = {
    {0.0, 0.0, 0.0},
    {10.0, 0.0, 0.0},
    {0.0, 10.0, 0.0},
    {5.0, 5.0, 0.1}
  };

  double parent_volume = compute_tetrahedron_volume(flat_tet);
  EXPECT_GT(parent_volume, 0.0) << "Flat tetrahedron should have positive volume";

  auto refined = rule.refine(flat_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(flat_tet, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for flat tetrahedron";
}

TEST_F(TetrahedronRefinementTest, ElongatedTetrahedronVolumeConservation) {
  // Test with an elongated tetrahedron (high aspect ratio)
  std::vector<std::array<double, 3>> elongated_tet = {
    {0.0, 0.0, 0.0},
    {0.1, 0.0, 0.0},
    {0.0, 0.1, 0.0},
    {0.0, 0.0, 10.0}
  };

  double parent_volume = compute_tetrahedron_volume(elongated_tet);
  EXPECT_GT(parent_volume, 0.0) << "Elongated tetrahedron should have positive volume";

  auto refined = rule.refine(elongated_tet, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(elongated_tet, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for elongated tetrahedron";
}

// ==============================================================================
// Octahedron Subdivision Tests
// ==============================================================================

TEST_F(TetrahedronRefinementTest, OctahedronSubdivisionIsCorrect) {
  // The 6 edge midpoints form an octahedron in the center
  // This octahedron is subdivided into 4 tetrahedra (children 4-7)
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  // Verify that children 4-7 only use edge midpoint vertices (indices 4-9)
  for (size_t i = 4; i < 8; ++i) {
    const auto& child = refined.child_connectivity[i];
    for (size_t v : child) {
      EXPECT_GE(v, 4) << "Octahedron child " << i << " should only use edge midpoints";
      EXPECT_LE(v, 9) << "Octahedron child " << i << " should only use edge midpoints";
    }
  }
}

TEST_F(TetrahedronRefinementTest, OctahedronVolumeSumIsCorrect) {
  // The 4 octahedron children should sum to half the parent volume
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double octahedron_volume = 0.0;
  for (size_t i = 4; i < 8; ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    octahedron_volume += compute_tetrahedron_volume(child_verts);
  }

  // The octahedron subdivision should account for exactly half the parent volume
  EXPECT_NEAR(octahedron_volume, unit_tet_volume / 2.0, 1e-10)
      << "Octahedron subdivision volume should be half of parent volume";
}

TEST_F(TetrahedronRefinementTest, CornerTetsVolumeSumIsCorrect) {
  // The 4 corner tets should sum to half the parent volume
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, 0);

  double corner_volume = 0.0;
  for (size_t i = 0; i < 4; ++i) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    corner_volume += compute_tetrahedron_volume(child_verts);
  }

  // The corner tets should account for exactly half the parent volume
  EXPECT_NEAR(corner_volume, unit_tet_volume / 2.0, 1e-10)
      << "Corner tets volume should be half of parent volume";
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(TetrahedronRefinementTest, ThrowsOnInvalidCellType) {
  EXPECT_THROW(
    rule.refine(unit_tet_verts, CellType::Hex, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when CellType is not TETRA";
}

TEST_F(TetrahedronRefinementTest, ThrowsOnInvalidVertexCount) {
  std::vector<std::array<double, 3>> invalid_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0}
    // Only 3 vertices instead of 4
  };

  EXPECT_THROW(
    rule.refine(invalid_verts, CellType::Tetra, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when vertex count is not 4";
}

// ==============================================================================
// Parameterized Tests for Multiple Refinement Levels
// ==============================================================================

class TetrahedronMultiLevelRefinementTest : public ::testing::TestWithParam<size_t> {
protected:
  void SetUp() override {
    unit_tet_verts = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0}
    };
  }

  std::vector<std::array<double, 3>> unit_tet_verts;
  TetrahedronRefinementRule rule;
};

TEST_P(TetrahedronMultiLevelRefinementTest, CorrectChildLevelAtEachLevel) {
  size_t level = GetParam();

  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, level);

  EXPECT_EQ(refined.child_level, level + 1)
      << "Child level incorrect at refinement level " << level;
}

TEST_P(TetrahedronMultiLevelRefinementTest, VolumeConservationAtEachLevel) {
  size_t level = GetParam();

  double parent_volume = compute_tetrahedron_volume(unit_tet_verts);
  auto refined = rule.refine(unit_tet_verts, CellType::Tetra,
                             RefinementPattern::ISOTROPIC, level);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(unit_tet_verts, refined.new_vertices, child_conn);
    children_volume_sum += compute_tetrahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails at refinement level " << level;
}

INSTANTIATE_TEST_SUITE_P(
  MultiLevel,
  TetrahedronMultiLevelRefinementTest,
  ::testing::Values(0, 1, 2, 3, 5, 9)
);

} // namespace test
} // namespace svmp
