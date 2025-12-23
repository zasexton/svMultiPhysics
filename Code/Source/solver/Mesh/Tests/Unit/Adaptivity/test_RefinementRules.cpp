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
 * @file test_RefinementRules.cpp
 * @brief Integration tests for refinement rules and RefinementRulesManager
 *
 * This test suite validates:
 * - RefinementRulesManager singleton and rule registration
 * - Cross-element refinement patterns
 * - Rule compatibility and consistency
 * - Default pattern selection
 * - Manager interface correctness
 */

#include <gtest/gtest.h>
#include "../../../Adaptivity/RefinementRules.h"
#include <cmath>
#include <set>
#include <algorithm>

namespace svmp {
namespace test {

// ==============================================================================
// Test Fixture for RefinementRulesManager
// ==============================================================================

class RefinementRulesManagerTest : public ::testing::Test {
protected:
  void SetUp() override {
    manager = &RefinementRulesManager::instance();
  }

  RefinementRulesManager* manager;
};

// ==============================================================================
// Manager Singleton Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, SingletonInstanceIsConsistent) {
  auto& instance1 = RefinementRulesManager::instance();
  auto& instance2 = RefinementRulesManager::instance();

  EXPECT_EQ(&instance1, &instance2)
      << "RefinementRulesManager should return same singleton instance";
}

// ==============================================================================
// Rule Retrieval Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, GetRuleForWedge) {
  auto* rule = manager->get_rule(CellType::Wedge);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for WEDGE";
}

TEST_F(RefinementRulesManagerTest, GetRuleForPyramid) {
  auto* rule = manager->get_rule(CellType::Pyramid);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for PYRAMID";
}

TEST_F(RefinementRulesManagerTest, GetRuleForTetrahedron) {
  auto* rule = manager->get_rule(CellType::Tetra);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for TETRA";
}

TEST_F(RefinementRulesManagerTest, GetRuleForHexahedron) {
  auto* rule = manager->get_rule(CellType::Hex);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for HEX";
}

TEST_F(RefinementRulesManagerTest, GetRuleForTriangle) {
  auto* rule = manager->get_rule(CellType::Triangle);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for TRIANGLE";
}

TEST_F(RefinementRulesManagerTest, GetRuleForQuad) {
  auto* rule = manager->get_rule(CellType::Quad);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for QUAD";
}

TEST_F(RefinementRulesManagerTest, GetRuleForLine) {
  auto* rule = manager->get_rule(CellType::Line);
  ASSERT_NE(rule, nullptr) << "Should return a valid rule for LINE";
}

// ==============================================================================
// Manager CanRefine Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, CanRefineWedge) {
  EXPECT_TRUE(manager->can_refine(CellType::Wedge, 0));
  EXPECT_TRUE(manager->can_refine(CellType::Wedge, 5));
}

TEST_F(RefinementRulesManagerTest, CanRefinePyramid) {
  EXPECT_TRUE(manager->can_refine(CellType::Pyramid, 0));
  EXPECT_TRUE(manager->can_refine(CellType::Pyramid, 5));
}

TEST_F(RefinementRulesManagerTest, CanRefineAtHighLevel) {
  // Test that refinement is allowed at high levels (within limits)
  EXPECT_TRUE(manager->can_refine(CellType::Wedge, 9));
  EXPECT_TRUE(manager->can_refine(CellType::Pyramid, 9));
}

// ==============================================================================
// Manager NumChildren Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, WedgeProducesEightChildren) {
  size_t num_children = manager->num_children(CellType::Wedge, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 8);
}

TEST_F(RefinementRulesManagerTest, PyramidProducesSixChildren) {
  size_t num_children = manager->num_children(CellType::Pyramid, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 10)  // 6 pyramids + 4 tetrahedra
      << "Pyramid isotropic refinement produces 10 children";
}

TEST_F(RefinementRulesManagerTest, TetrahedronProducesEightChildren) {
  size_t num_children = manager->num_children(CellType::Tetra, RefinementPattern::RED);
  EXPECT_EQ(num_children, 8);
}

TEST_F(RefinementRulesManagerTest, HexahedronProducesEightChildren) {
  size_t num_children = manager->num_children(CellType::Hex, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 8);
}

TEST_F(RefinementRulesManagerTest, TriangleProducesFourChildren) {
  size_t num_children = manager->num_children(CellType::Triangle, RefinementPattern::RED);
  EXPECT_EQ(num_children, 4);
}

TEST_F(RefinementRulesManagerTest, QuadProducesFourChildren) {
  size_t num_children = manager->num_children(CellType::Quad, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 4);
}

TEST_F(RefinementRulesManagerTest, LineProducesTwoChildren) {
  size_t num_children = manager->num_children(CellType::Line, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(num_children, 2);
}

// ==============================================================================
// Manager Refine Interface Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, RefineWedgeThroughManager) {
  std::vector<std::array<double, 3>> wedge_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
    {1.0, 0.0, 1.0},
    {0.0, 1.0, 1.0}
  };

  auto refined = manager->refine(wedge_verts, CellType::Wedge,
                                 RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 8);
  EXPECT_EQ(refined.new_vertices.size(), 12);
  EXPECT_EQ(refined.pattern, RefinementPattern::RED);
}

TEST_F(RefinementRulesManagerTest, RefinePyramidThroughManager) {
  std::vector<std::array<double, 3>> pyramid_verts = {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.5, 0.5, 1.0}
  };

  auto refined = manager->refine(pyramid_verts, CellType::Pyramid,
                                 RefinementPattern::ISOTROPIC, 0);

  EXPECT_EQ(refined.child_connectivity.size(), 10)  // 6 pyramids + 4 tetrahedra
      << "Pyramid refinement produces 10 children";
  EXPECT_EQ(refined.new_vertices.size(), 10)  // 8 edge mids + 1 base center + 1 mid-height
      << "Pyramid refinement creates 10 new vertices";
  EXPECT_EQ(refined.pattern, RefinementPattern::RED);
}

// ==============================================================================
// Rule Compatibility Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, WedgeCompatiblePatterns) {
  auto* rule = manager->get_rule(CellType::Wedge);
  ASSERT_NE(rule, nullptr);

  auto patterns = rule->compatible_patterns(CellType::Wedge);

  EXPECT_FALSE(patterns.empty())
      << "Wedge should have at least one compatible pattern";

  EXPECT_NE(std::find(patterns.begin(), patterns.end(), RefinementPattern::ISOTROPIC),
            patterns.end())
      << "Wedge should support ISOTROPIC refinement";
}

TEST_F(RefinementRulesManagerTest, PyramidCompatiblePatterns) {
  auto* rule = manager->get_rule(CellType::Pyramid);
  ASSERT_NE(rule, nullptr);

  auto patterns = rule->compatible_patterns(CellType::Pyramid);

  EXPECT_FALSE(patterns.empty())
      << "Pyramid should have at least one compatible pattern";

  EXPECT_NE(std::find(patterns.begin(), patterns.end(), RefinementPattern::ISOTROPIC),
            patterns.end())
      << "Pyramid should support ISOTROPIC refinement";
}

TEST_F(RefinementRulesManagerTest, WedgeDefaultPattern) {
  auto* rule = manager->get_rule(CellType::Wedge);
  ASSERT_NE(rule, nullptr);

  auto default_pattern = rule->default_pattern(CellType::Wedge);

  EXPECT_EQ(default_pattern, RefinementPattern::ISOTROPIC)
      << "Wedge default pattern should be ISOTROPIC";
}

TEST_F(RefinementRulesManagerTest, PyramidDefaultPattern) {
  auto* rule = manager->get_rule(CellType::Pyramid);
  ASSERT_NE(rule, nullptr);

  auto default_pattern = rule->default_pattern(CellType::Pyramid);

  EXPECT_EQ(default_pattern, RefinementPattern::ISOTROPIC)
      << "Pyramid default pattern should be ISOTROPIC";
}

// ==============================================================================
// Cross-Element Consistency Tests
// ==============================================================================

TEST_F(RefinementRulesManagerTest, AllElementTypesHaveRules) {
  std::vector<CellType> element_types = {
    CellType::Line,
    CellType::Triangle,
    CellType::Quad,
    CellType::Tetra,
    CellType::Hex,
    CellType::Wedge,
    CellType::Pyramid
  };

  for (auto cell_type : element_types) {
    auto* rule = manager->get_rule(cell_type);
    EXPECT_NE(rule, nullptr)
        << "Missing rule for element type " << static_cast<int>(cell_type);
  }
}

TEST_F(RefinementRulesManagerTest, AllRulesProducePositiveChildCount) {
  std::vector<std::pair<CellType, RefinementPattern>> test_cases = {
    {CellType::Line, RefinementPattern::ISOTROPIC},
    {CellType::Triangle, RefinementPattern::RED},
    {CellType::Quad, RefinementPattern::ISOTROPIC},
    {CellType::Tetra, RefinementPattern::RED},
    {CellType::Hex, RefinementPattern::ISOTROPIC},
    {CellType::Wedge, RefinementPattern::ISOTROPIC},
    {CellType::Pyramid, RefinementPattern::ISOTROPIC}
  };

  for (const auto& [cell_type, pattern] : test_cases) {
    size_t num_children = manager->num_children(cell_type, pattern);
    EXPECT_GT(num_children, 0)
        << "Element type " << static_cast<int>(cell_type)
        << " produces zero children";
  }
}

// ==============================================================================
// RefinementUtils Tests
// ==============================================================================

TEST(RefinementUtilsTest, EdgeMidpointIsCorrect) {
  std::array<double, 3> v1 = {0.0, 0.0, 0.0};
  std::array<double, 3> v2 = {2.0, 4.0, 6.0};

  auto midpoint = RefinementUtils::edge_midpoint(v1, v2);

  EXPECT_DOUBLE_EQ(midpoint[0], 1.0);
  EXPECT_DOUBLE_EQ(midpoint[1], 2.0);
  EXPECT_DOUBLE_EQ(midpoint[2], 3.0);
}

TEST(RefinementUtilsTest, FaceCenterForTriangle) {
  std::vector<std::array<double, 3>> face_verts = {
    {0.0, 0.0, 0.0},
    {3.0, 0.0, 0.0},
    {0.0, 3.0, 0.0}
  };

  auto center = RefinementUtils::face_center(face_verts);

  EXPECT_DOUBLE_EQ(center[0], 1.0);
  EXPECT_DOUBLE_EQ(center[1], 1.0);
  EXPECT_DOUBLE_EQ(center[2], 0.0);
}

TEST(RefinementUtilsTest, FaceCenterForQuad) {
  std::vector<std::array<double, 3>> face_verts = {
    {0.0, 0.0, 0.0},
    {4.0, 0.0, 0.0},
    {4.0, 4.0, 0.0},
    {0.0, 4.0, 0.0}
  };

  auto center = RefinementUtils::face_center(face_verts);

  EXPECT_DOUBLE_EQ(center[0], 2.0);
  EXPECT_DOUBLE_EQ(center[1], 2.0);
  EXPECT_DOUBLE_EQ(center[2], 0.0);
}

TEST(RefinementUtilsTest, CellCenterForTetrahedron) {
  std::vector<std::array<double, 3>> cell_verts = {
    {0.0, 0.0, 0.0},
    {4.0, 0.0, 0.0},
    {0.0, 4.0, 0.0},
    {0.0, 0.0, 4.0}
  };

  auto center = RefinementUtils::cell_center(cell_verts);

  EXPECT_DOUBLE_EQ(center[0], 1.0);
  EXPECT_DOUBLE_EQ(center[1], 1.0);
  EXPECT_DOUBLE_EQ(center[2], 1.0);
}

// ==============================================================================
// Pattern Consistency Tests
// ==============================================================================

TEST(RefinementPatternTest, WedgeAndPyramidUseSamePatternEnum) {
  // Both wedge and pyramid use ISOTROPIC pattern, but produce different numbers
  // This test ensures the pattern enum is consistent across element types

  WedgeRefinementRule wedge_rule;
  PyramidRefinementRule pyramid_rule;

  auto wedge_pattern = wedge_rule.default_pattern(CellType::Wedge);
  auto pyramid_pattern = pyramid_rule.default_pattern(CellType::Pyramid);

  EXPECT_EQ(wedge_pattern, RefinementPattern::ISOTROPIC);
  EXPECT_EQ(pyramid_pattern, RefinementPattern::ISOTROPIC);

  // But they produce different numbers of children
  EXPECT_EQ(wedge_rule.num_children(CellType::Wedge, wedge_pattern), 8)
      << "Wedge produces 8 children";
  EXPECT_EQ(pyramid_rule.num_children(CellType::Pyramid, pyramid_pattern), 10)
      << "Pyramid produces 10 children (6 pyramids + 4 tetrahedra)";
}

// ==============================================================================
// Refinement Level Consistency Tests
// ==============================================================================

TEST(RefinementLevelTest, WedgeAndPyramidIncrementLevelConsistently) {
  std::vector<std::array<double, 3>> wedge_verts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}
  };

  std::vector<std::array<double, 3>> pyramid_verts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0}, {0.5, 0.5, 1.0}
  };

  WedgeRefinementRule wedge_rule;
  PyramidRefinementRule pyramid_rule;

  for (size_t level = 0; level < 5; ++level) {
    auto wedge_refined = wedge_rule.refine(wedge_verts, CellType::Wedge,
                                           RefinementPattern::ISOTROPIC, level);
    auto pyramid_refined = pyramid_rule.refine(pyramid_verts, CellType::Pyramid,
                                               RefinementPattern::ISOTROPIC, level);

    EXPECT_EQ(wedge_refined.child_level, level + 1)
        << "Wedge child level incorrect at level " << level;
    EXPECT_EQ(pyramid_refined.child_level, level + 1)
        << "Pyramid child level incorrect at level " << level;
  }
}

// ==============================================================================
// Vertex Count Consistency Tests
// ==============================================================================

TEST(VertexCountTest, WedgeCreatesExpectedNewVertices) {
  // Wedge: 9 edge midpoints + 3 mid-layer = 12 new vertices
  std::vector<std::array<double, 3>> wedge_verts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}
  };

  WedgeRefinementRule rule;
  auto refined = rule.refine(wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  size_t expected_new_vertices = 12;
  EXPECT_EQ(refined.new_vertices.size(), expected_new_vertices)
      << "Wedge should create exactly 12 new vertices";
}

TEST(VertexCountTest, PyramidCreatesExpectedNewVertices) {
  // Pyramid: 8 edge midpoints + 1 base center + 1 mid-height = 10 new vertices
  std::vector<std::array<double, 3>> pyramid_verts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0}, {0.5, 0.5, 1.0}
  };

  PyramidRefinementRule rule;
  auto refined = rule.refine(pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  size_t expected_new_vertices = 10;
  EXPECT_EQ(refined.new_vertices.size(), expected_new_vertices)
      << "Pyramid should create exactly 10 new vertices";
}

// ==============================================================================
// Total Vertex Count Tests
// ==============================================================================

TEST(TotalVertexTest, WedgeRefinementTotalVertices) {
  std::vector<std::array<double, 3>> wedge_verts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {0.0, 1.0, 1.0}
  };

  WedgeRefinementRule rule;
  auto refined = rule.refine(wedge_verts, CellType::Wedge,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_vertices = wedge_verts.size() + refined.new_vertices.size();
  EXPECT_EQ(total_vertices, 18)  // 6 original + 12 new
      << "Total vertex count for refined wedge should be 18";
}

TEST(TotalVertexTest, PyramidRefinementTotalVertices) {
  std::vector<std::array<double, 3>> pyramid_verts = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0},
    {0.0, 1.0, 0.0}, {0.5, 0.5, 1.0}
  };

  PyramidRefinementRule rule;
  auto refined = rule.refine(pyramid_verts, CellType::Pyramid,
                             RefinementPattern::ISOTROPIC, 0);

  size_t total_vertices = pyramid_verts.size() + refined.new_vertices.size();
  EXPECT_EQ(total_vertices, 15)  // 5 original + 10 new
      << "Total vertex count for refined pyramid should be 15";
}

} // namespace test
} // namespace svmp
