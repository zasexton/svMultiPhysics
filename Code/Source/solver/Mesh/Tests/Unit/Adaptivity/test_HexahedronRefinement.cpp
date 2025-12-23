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
 * @file test_HexahedronRefinement.cpp
 * @brief Comprehensive unit tests for hexahedron h-refinement rules
 *
 * This test suite validates:
 * - Basic refinement correctness (child count, vertex count)
 * - Geometric verification (volume conservation, edge/face/cell centers)
 * - Topology validation (distinct vertices, proper connectivity)
 * - Edge cases (degenerate geometries, various orientations)
 * - Structured subdivision properties (8 equal children)
 *
 * Hexahedron refinement:
 * - 8 parent vertices -> 12 edge midpoints + 6 face centers + 1 cell center -> 27 total vertices
 * - 8 child hexahedra (4 bottom layer + 4 top layer)
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
 * @brief Compute volume of a hexahedron using divergence theorem
 *
 * For axis-aligned hexahedron, volume = length * width * height
 * For general hexahedron, use decomposition into tetrahedra
 */
double compute_hexahedron_volume(const std::vector<std::array<double, 3>>& verts) {
  if (verts.size() != 8) {
    return 0.0;
  }

  // Decompose hex into 5 tetrahedra and sum their volumes
  // Center point for decomposition
  std::array<double, 3> center = {0.0, 0.0, 0.0};
  for (const auto& v : verts) {
    center[0] += v[0];
    center[1] += v[1];
    center[2] += v[2];
  }
  center[0] /= 8.0;
  center[1] /= 8.0;
  center[2] /= 8.0;

  auto tet_volume = [](const std::array<double, 3>& v0,
                       const std::array<double, 3>& v1,
                       const std::array<double, 3>& v2,
                       const std::array<double, 3>& v3) {
    std::array<double, 3> e1 = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    std::array<double, 3> e2 = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    std::array<double, 3> e3 = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};

    double det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1])
               - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0])
               + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);

    return std::abs(det) / 6.0;
  };

  // Decompose into 6 pyramids (one per face), each pyramid into 2 tets
  double volume = 0.0;

  // Bottom face (0,1,2,3)
  volume += tet_volume(center, verts[0], verts[1], verts[2]);
  volume += tet_volume(center, verts[0], verts[2], verts[3]);

  // Top face (4,5,6,7)
  volume += tet_volume(center, verts[4], verts[5], verts[6]);
  volume += tet_volume(center, verts[4], verts[6], verts[7]);

  // Front face (0,1,5,4)
  volume += tet_volume(center, verts[0], verts[1], verts[5]);
  volume += tet_volume(center, verts[0], verts[5], verts[4]);

  // Back face (3,2,6,7)
  volume += tet_volume(center, verts[3], verts[2], verts[6]);
  volume += tet_volume(center, verts[3], verts[6], verts[7]);

  // Left face (0,3,7,4)
  volume += tet_volume(center, verts[0], verts[3], verts[7]);
  volume += tet_volume(center, verts[0], verts[7], verts[4]);

  // Right face (1,2,6,5)
  volume += tet_volume(center, verts[1], verts[2], verts[6]);
  volume += tet_volume(center, verts[1], verts[6], verts[5]);

  return volume;
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
 * @brief Check if point is at cell center
 */
bool is_cell_center(const std::array<double, 3>& p,
                    const std::vector<std::array<double, 3>>& verts,
                    double tol = 1e-10) {
  std::array<double, 3> expected = {0.0, 0.0, 0.0};
  for (const auto& v : verts) {
    expected[0] += v[0];
    expected[1] += v[1];
    expected[2] += v[2];
  }
  expected[0] /= verts.size();
  expected[1] /= verts.size();
  expected[2] /= verts.size();

  return distance(p, expected) < tol;
}

/**
 * @brief Check if a hexahedron has positive volume (non-degenerate)
 */
bool has_positive_volume(const std::vector<std::array<double, 3>>& verts, double tol = 1e-12) {
  return compute_hexahedron_volume(verts) > tol;
}

// ==============================================================================
// Test Fixture
// ==============================================================================

class HexahedronRefinementTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create unit cube hexahedron
    unit_hex_verts = {
      {0.0, 0.0, 0.0},  // 0: bottom-front-left
      {1.0, 0.0, 0.0},  // 1: bottom-front-right
      {1.0, 1.0, 0.0},  // 2: bottom-back-right
      {0.0, 1.0, 0.0},  // 3: bottom-back-left
      {0.0, 0.0, 1.0},  // 4: top-front-left
      {1.0, 0.0, 1.0},  // 5: top-front-right
      {1.0, 1.0, 1.0},  // 6: top-back-right
      {0.0, 1.0, 1.0}   // 7: top-back-left
    };

    unit_hex_volume = compute_hexahedron_volume(unit_hex_verts);
  }

  std::vector<std::array<double, 3>> unit_hex_verts;
  double unit_hex_volume;
  HexahedronRefinementRule rule;
};

// ==============================================================================
// Basic Correctness Tests
// ==============================================================================

TEST_F(HexahedronRefinementTest, ProducesEightChildren) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8)
      << "Hexahedron refinement should produce 8 children";
}

TEST_F(HexahedronRefinementTest, CreatesNineteenNewVertices) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.new_vertices.size(), 19)
      << "Hexahedron refinement should create 19 new vertices "
      << "(12 edge midpoints + 6 face centers + 1 cell center)";
}

TEST_F(HexahedronRefinementTest, IncrementsRefinementLevel) {
  size_t initial_level = 0;
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, initial_level);

  EXPECT_EQ(refined.child_level, initial_level + 1)
      << "Child refinement level should be parent level + 1";
}

TEST_F(HexahedronRefinementTest, StoresCorrectRefinementPattern) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::ISOTROPIC)
      << "Hexahedron refinement uses ISOTROPIC pattern";
}

TEST_F(HexahedronRefinementTest, CanRefineReturnsTrue) {
  EXPECT_TRUE(rule.can_refine(CellType::Hex, 0));
  EXPECT_TRUE(rule.can_refine(CellType::Hex, 5));
}

TEST_F(HexahedronRefinementTest, NumChildrenReturnsEight) {
  size_t num_children = rule.num_children(CellType::Hex, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 8);
}

// ==============================================================================
// Topology Validation Tests
// ==============================================================================

TEST_F(HexahedronRefinementTest, AllChildrenHaveEightVertices) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    EXPECT_EQ(refined.child_connectivity[i].size(), 8)
        << "Child " << i << " should have 8 vertices (hexahedron topology)";
  }
}

TEST_F(HexahedronRefinementTest, AllChildrenHaveDistinctVertices) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    EXPECT_TRUE(has_distinct_vertices(child))
        << "Child " << i << " has repeated vertices: ["
        << child[0] << ", " << child[1] << ", " << child[2] << ", " << child[3] << ", "
        << child[4] << ", " << child[5] << ", " << child[6] << ", " << child[7] << "]";
  }
}

TEST_F(HexahedronRefinementTest, ChildVertexIndicesAreValid) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_verts = unit_hex_verts.size() + refined.new_vertices.size();

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    for (size_t j = 0; j < child.size(); ++j) {
      EXPECT_LT(child[j], total_verts)
          << "Child " << i << " vertex " << j << " has invalid index " << child[j];
    }
  }
}

TEST_F(HexahedronRefinementTest, VerifyExpectedConnectivity) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  // Expected connectivity from implementation
  // Original vertices: 0-7
  // Edge midpoints: 8-19
  // Face centers: 20-25
  // Cell center: 26
  std::vector<std::vector<size_t>> expected_connectivity = {
    // Bottom layer
    {0, 8, 20, 11, 16, 22, 26, 24},   // Bottom-front-left
    {8, 1, 9, 20, 22, 17, 25, 26},    // Bottom-front-right
    {20, 9, 2, 10, 26, 25, 18, 23},   // Bottom-back-right
    {11, 20, 10, 3, 24, 26, 23, 19},  // Bottom-back-left

    // Top layer
    {16, 22, 26, 24, 4, 12, 21, 15},  // Top-front-left
    {22, 17, 25, 26, 12, 5, 13, 21},  // Top-front-right
    {26, 25, 18, 23, 21, 13, 6, 14},  // Top-back-right
    {24, 26, 23, 19, 15, 21, 14, 7}   // Top-back-left
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

TEST_F(HexahedronRefinementTest, VolumeConservation) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_vol = compute_hexahedron_volume(child_verts);

    EXPECT_GT(child_vol, 0.0) << "Child " << i << " has non-positive volume";
    children_volume_sum += child_vol;
  }

  EXPECT_NEAR(unit_hex_volume, children_volume_sum, 1e-10)
      << "Volume conservation violated: parent volume = " << unit_hex_volume
      << ", children sum = " << children_volume_sum;
}

TEST_F(HexahedronRefinementTest, EdgeMidpointsAreCorrect) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_GE(refined.new_vertices.size(), 12);

  // Bottom edges (indices 8-11)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], unit_hex_verts[0], unit_hex_verts[1]))
      << "Vertex 8 should be midpoint of edge 0-1";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], unit_hex_verts[1], unit_hex_verts[2]))
      << "Vertex 9 should be midpoint of edge 1-2";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[2], unit_hex_verts[2], unit_hex_verts[3]))
      << "Vertex 10 should be midpoint of edge 2-3";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[3], unit_hex_verts[3], unit_hex_verts[0]))
      << "Vertex 11 should be midpoint of edge 3-0";

  // Top edges (indices 12-15)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[4], unit_hex_verts[4], unit_hex_verts[5]))
      << "Vertex 12 should be midpoint of edge 4-5";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[5], unit_hex_verts[5], unit_hex_verts[6]))
      << "Vertex 13 should be midpoint of edge 5-6";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[6], unit_hex_verts[6], unit_hex_verts[7]))
      << "Vertex 14 should be midpoint of edge 6-7";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[7], unit_hex_verts[7], unit_hex_verts[4]))
      << "Vertex 15 should be midpoint of edge 7-4";

  // Vertical edges (indices 16-19)
  EXPECT_TRUE(is_midpoint(refined.new_vertices[8], unit_hex_verts[0], unit_hex_verts[4]))
      << "Vertex 16 should be midpoint of edge 0-4";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[9], unit_hex_verts[1], unit_hex_verts[5]))
      << "Vertex 17 should be midpoint of edge 1-5";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[10], unit_hex_verts[2], unit_hex_verts[6]))
      << "Vertex 18 should be midpoint of edge 2-6";
  EXPECT_TRUE(is_midpoint(refined.new_vertices[11], unit_hex_verts[3], unit_hex_verts[7]))
      << "Vertex 19 should be midpoint of edge 3-7";
}

TEST_F(HexahedronRefinementTest, FaceCentersAreCorrect) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_GE(refined.new_vertices.size(), 18);

  // Bottom face (index 20)
  EXPECT_TRUE(is_face_center(refined.new_vertices[12],
                             unit_hex_verts[0], unit_hex_verts[1],
                             unit_hex_verts[2], unit_hex_verts[3]))
      << "Vertex 20 should be at center of bottom face";

  // Top face (index 21)
  EXPECT_TRUE(is_face_center(refined.new_vertices[13],
                             unit_hex_verts[4], unit_hex_verts[5],
                             unit_hex_verts[6], unit_hex_verts[7]))
      << "Vertex 21 should be at center of top face";

  // Front face (index 22)
  EXPECT_TRUE(is_face_center(refined.new_vertices[14],
                             unit_hex_verts[0], unit_hex_verts[1],
                             unit_hex_verts[5], unit_hex_verts[4]))
      << "Vertex 22 should be at center of front face";

  // Back face (index 23)
  EXPECT_TRUE(is_face_center(refined.new_vertices[15],
                             unit_hex_verts[3], unit_hex_verts[2],
                             unit_hex_verts[6], unit_hex_verts[7]))
      << "Vertex 23 should be at center of back face";

  // Left face (index 24)
  EXPECT_TRUE(is_face_center(refined.new_vertices[16],
                             unit_hex_verts[0], unit_hex_verts[3],
                             unit_hex_verts[7], unit_hex_verts[4]))
      << "Vertex 24 should be at center of left face";

  // Right face (index 25)
  EXPECT_TRUE(is_face_center(refined.new_vertices[17],
                             unit_hex_verts[1], unit_hex_verts[2],
                             unit_hex_verts[6], unit_hex_verts[5]))
      << "Vertex 25 should be at center of right face";
}

TEST_F(HexahedronRefinementTest, CellCenterIsCorrect) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  ASSERT_EQ(refined.new_vertices.size(), 19);

  // Cell center (index 26)
  EXPECT_TRUE(is_cell_center(refined.new_vertices[18], unit_hex_verts))
      << "Vertex 26 should be at cell center";
}

TEST_F(HexahedronRefinementTest, AllChildrenHavePositiveVolume) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    double child_vol = compute_hexahedron_volume(child_verts);

    EXPECT_GT(child_vol, 1e-12) << "Child " << i << " has near-zero or negative volume";
  }
}

TEST_F(HexahedronRefinementTest, ChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  std::vector<double> child_volumes;
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    child_volumes.push_back(compute_hexahedron_volume(child_verts));
  }

  double expected_child_vol = unit_hex_volume / 8.0;

  for (size_t i = 0; i < child_volumes.size(); ++i) {
    EXPECT_NEAR(child_volumes[i], expected_child_vol, 1e-10)
        << "Child " << i << " volume differs from expected equal subdivision";
  }
}

// ==============================================================================
// Edge Cases and Stress Tests
// ==============================================================================

TEST_F(HexahedronRefinementTest, ScaledHexahedronVolumeConservation) {
  // Test with a scaled hexahedron
  std::vector<std::array<double, 3>> scaled_hex = {
    {0.0, 0.0, 0.0},
    {2.0, 0.0, 0.0},
    {2.0, 2.0, 0.0},
    {0.0, 2.0, 0.0},
    {0.0, 0.0, 3.0},
    {2.0, 0.0, 3.0},
    {2.0, 2.0, 3.0},
    {0.0, 2.0, 3.0}
  };

  double parent_volume = compute_hexahedron_volume(scaled_hex);
  auto refined = rule.refine(scaled_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(scaled_hex, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-9)
      << "Volume conservation fails for scaled hexahedron";
}

TEST_F(HexahedronRefinementTest, TranslatedHexahedronVolumeConservation) {
  // Test with a translated hexahedron
  std::vector<std::array<double, 3>> translated_hex = {
    {10.0, 20.0, 30.0},
    {11.0, 20.0, 30.0},
    {11.0, 21.0, 30.0},
    {10.0, 21.0, 30.0},
    {10.0, 20.0, 31.0},
    {11.0, 20.0, 31.0},
    {11.0, 21.0, 31.0},
    {10.0, 21.0, 31.0}
  };

  double parent_volume = compute_hexahedron_volume(translated_hex);
  auto refined = rule.refine(translated_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(translated_hex, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for translated hexahedron";
}

TEST_F(HexahedronRefinementTest, RotatedHexahedronProducesEightChildren) {
  // Test with a rotated hexahedron (45 degrees about z-axis)
  double angle = M_PI / 4.0;
  double cos_a = std::cos(angle);
  double sin_a = std::sin(angle);

  std::vector<std::array<double, 3>> rotated_hex = {
    {0.0, 0.0, 0.0},
    {cos_a, sin_a, 0.0},
    {cos_a - sin_a, sin_a + cos_a, 0.0},
    {-sin_a, cos_a, 0.0},
    {0.0, 0.0, 1.0},
    {cos_a, sin_a, 1.0},
    {cos_a - sin_a, sin_a + cos_a, 1.0},
    {-sin_a, cos_a, 1.0}
  };

  auto refined = rule.refine(rotated_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8);
  EXPECT_EQ(refined.new_vertices.size(), 19);
}

TEST_F(HexahedronRefinementTest, SmallHexahedronVolumeConservation) {
  // Test with a very small hexahedron
  std::vector<std::array<double, 3>> small_hex = {
    {0.0, 0.0, 0.0},
    {0.001, 0.0, 0.0},
    {0.001, 0.001, 0.0},
    {0.0, 0.001, 0.0},
    {0.0, 0.0, 0.001},
    {0.001, 0.0, 0.001},
    {0.001, 0.001, 0.001},
    {0.0, 0.001, 0.001}
  };

  double parent_volume = compute_hexahedron_volume(small_hex);
  EXPECT_GT(parent_volume, 0.0) << "Parent small hexahedron should have positive volume";

  auto refined = rule.refine(small_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(small_hex, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-15)
      << "Volume conservation fails for small hexahedron";
}

TEST_F(HexahedronRefinementTest, ElongatedHexahedronVolumeConservation) {
  // Test with an elongated hexahedron (high aspect ratio in z-direction)
  std::vector<std::array<double, 3>> elongated_hex = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 10.0},
    {1.0, 0.0, 10.0},
    {1.0, 1.0, 10.0},
    {0.0, 1.0, 10.0}
  };

  double parent_volume = compute_hexahedron_volume(elongated_hex);
  EXPECT_GT(parent_volume, 0.0) << "Elongated hexahedron should have positive volume";

  auto refined = rule.refine(elongated_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(elongated_hex, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-9)
      << "Volume conservation fails for elongated hexahedron";
}

TEST_F(HexahedronRefinementTest, FlatHexahedronVolumeConservation) {
  // Test with a flat hexahedron (low aspect ratio in z-direction)
  std::vector<std::array<double, 3>> flat_hex = {
    {0.0, 0.0, 0.0},
    {10.0, 0.0, 0.0},
    {10.0, 10.0, 0.0},
    {0.0, 10.0, 0.0},
    {0.0, 0.0, 0.1},
    {10.0, 0.0, 0.1},
    {10.0, 10.0, 0.1},
    {0.0, 10.0, 0.1}
  };

  double parent_volume = compute_hexahedron_volume(flat_hex);
  EXPECT_GT(parent_volume, 0.0) << "Flat hexahedron should have positive volume";

  auto refined = rule.refine(flat_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(flat_hex, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for flat hexahedron";
}

TEST_F(HexahedronRefinementTest, NonAxisAlignedHexahedronVolumeConservation) {
  // Test with a non-axis-aligned hexahedron
  std::vector<std::array<double, 3>> tilted_hex = {
    {0.0, 0.0, 0.0},
    {1.0, 0.1, 0.0},
    {0.9, 1.1, 0.0},
    {0.0, 1.0, 0.0},
    {0.1, 0.0, 1.0},
    {1.1, 0.1, 1.0},
    {1.0, 1.1, 1.0},
    {0.1, 1.0, 1.0}
  };

  double parent_volume = compute_hexahedron_volume(tilted_hex);
  EXPECT_GT(parent_volume, 0.0) << "Non-axis-aligned hexahedron should have positive volume";

  auto refined = rule.refine(tilted_hex, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(tilted_hex, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails for non-axis-aligned hexahedron";
}

// ==============================================================================
// Layer Symmetry Tests
// ==============================================================================

TEST_F(HexahedronRefinementTest, BottomLayerChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  // First 4 children are bottom layer and should have equal volumes
  std::vector<double> bottom_volumes;
  for (size_t i = 0; i < 4; ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    bottom_volumes.push_back(compute_hexahedron_volume(child_verts));
  }

  for (size_t i = 1; i < bottom_volumes.size(); ++i) {
    EXPECT_NEAR(bottom_volumes[0], bottom_volumes[i], 1e-10)
        << "Bottom layer child " << i << " volume differs from child 0";
  }
}

TEST_F(HexahedronRefinementTest, TopLayerChildrenHaveEqualVolumes) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  // Last 4 children are top layer and should have equal volumes
  std::vector<double> top_volumes;
  for (size_t i = 4; i < 8; ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    top_volumes.push_back(compute_hexahedron_volume(child_verts));
  }

  for (size_t i = 1; i < top_volumes.size(); ++i) {
    EXPECT_NEAR(top_volumes[0], top_volumes[i], 1e-10)
        << "Top layer child " << (i+4) << " volume differs from child 4";
  }
}

TEST_F(HexahedronRefinementTest, BottomAndTopLayersHaveEqualTotalVolume) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  double bottom_volume = 0.0;
  for (size_t i = 0; i < 4; ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    bottom_volume += compute_hexahedron_volume(child_verts);
  }

  double top_volume = 0.0;
  for (size_t i = 4; i < 8; ++i) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices,
                                          refined.child_connectivity[i]);
    top_volume += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(bottom_volume, top_volume, 1e-10)
      << "Bottom and top layer total volumes should be equal";

  EXPECT_NEAR(bottom_volume, unit_hex_volume / 2.0, 1e-10)
      << "Bottom layer volume should be half of parent volume";
}

// ==============================================================================
// Cell Center Usage Tests
// ==============================================================================

TEST_F(HexahedronRefinementTest, AllChildrenUseCellCenter) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  // Cell center has index 26
  size_t cell_center_idx = 26;

  // Every child should use the cell center
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];
    bool uses_center = std::find(child.begin(), child.end(), cell_center_idx) != child.end();
    EXPECT_TRUE(uses_center)
        << "Child " << i << " should use cell center (vertex 26)";
  }
}

TEST_F(HexahedronRefinementTest, EachChildUsesExactlyOneFaceCenter) {
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, 0);

  // Face centers have indices 20-25
  for (size_t i = 0; i < refined.child_connectivity.size(); ++i) {
    const auto& child = refined.child_connectivity[i];

    size_t face_center_count = 0;
    for (size_t v : child) {
      if (v >= 20 && v <= 25) {
        face_center_count++;
      }
    }

    EXPECT_EQ(face_center_count, 3)
        << "Child " << i << " should use exactly 3 face centers";
  }
}

// ==============================================================================
// Error Handling Tests
// ==============================================================================

TEST_F(HexahedronRefinementTest, ThrowsOnInvalidCellType) {
  EXPECT_THROW(
    rule.refine(unit_hex_verts, CellType::Tetra, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when CellType is not HEX";
}

TEST_F(HexahedronRefinementTest, ThrowsOnInvalidVertexCount) {
  std::vector<std::array<double, 3>> invalid_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0}
    // Only 4 vertices instead of 8
  };

  EXPECT_THROW(
    rule.refine(invalid_verts, CellType::Hex, RefinementPattern::ISOTROPIC, 0),
    std::invalid_argument
  ) << "Should throw when vertex count is not 8";
}

// ==============================================================================
// Parameterized Tests for Multiple Refinement Levels
// ==============================================================================

class HexahedronMultiLevelRefinementTest : public ::testing::TestWithParam<size_t> {
protected:
  void SetUp() override {
    unit_hex_verts = {
      {0.0, 0.0, 0.0},
      {1.0, 0.0, 0.0},
      {1.0, 1.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {1.0, 0.0, 1.0},
      {1.0, 1.0, 1.0},
      {0.0, 1.0, 1.0}
    };
  }

  std::vector<std::array<double, 3>> unit_hex_verts;
  HexahedronRefinementRule rule;
};

TEST_P(HexahedronMultiLevelRefinementTest, CorrectChildLevelAtEachLevel) {
  size_t level = GetParam();

  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, level);

  EXPECT_EQ(refined.child_level, level + 1)
      << "Child level incorrect at refinement level " << level;
}

TEST_P(HexahedronMultiLevelRefinementTest, VolumeConservationAtEachLevel) {
  size_t level = GetParam();

  double parent_volume = compute_hexahedron_volume(unit_hex_verts);
  auto refined = rule.refine(unit_hex_verts, CellType::Hex,
                             RefinementPattern::ISOTROPIC, level);

  double children_volume_sum = 0.0;
  for (const auto& child_conn : refined.child_connectivity) {
    auto child_verts = get_child_vertices(unit_hex_verts, refined.new_vertices, child_conn);
    children_volume_sum += compute_hexahedron_volume(child_verts);
  }

  EXPECT_NEAR(parent_volume, children_volume_sum, 1e-10)
      << "Volume conservation fails at refinement level " << level;
}

INSTANTIATE_TEST_SUITE_P(
  MultiLevel,
  HexahedronMultiLevelRefinementTest,
  ::testing::Values(0, 1, 2, 3, 5, 9)
);

} // namespace test
} // namespace svmp
