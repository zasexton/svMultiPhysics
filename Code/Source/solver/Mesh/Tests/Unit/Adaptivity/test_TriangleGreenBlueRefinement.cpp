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
 * @file test_TriangleGreenBlueRefinement.cpp
 * @brief Unit tests for selector-aware triangle GREEN/BLUE refinement variants
 *
 * These tests validate:
 * - Correct edge/vertex selector interpretation (GREEN edge_to_split, BLUE shared_vertex)
 * - Correct new-vertex placement and weights
 * - Child connectivity and area conservation
 */

#include <gtest/gtest.h>

#include "../../../Adaptivity/RefinementRules.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace svmp {
namespace test {

namespace {

double tri_area(const std::vector<std::array<double, 3>>& v) {
  EXPECT_EQ(v.size(), 3u);
  const auto& a = v[0];
  const auto& b = v[1];
  const auto& c = v[2];

  const std::array<double, 3> ab{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const std::array<double, 3> ac{c[0] - a[0], c[1] - a[1], c[2] - a[2]};

  const std::array<double, 3> cross{
      ab[1] * ac[2] - ab[2] * ac[1],
      ab[2] * ac[0] - ab[0] * ac[2],
      ab[0] * ac[1] - ab[1] * ac[0],
  };
  const double n = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
  return 0.5 * n;
}

std::vector<std::array<double, 3>> child_vertices(const std::vector<std::array<double, 3>>& parent,
                                                  const std::vector<std::array<double, 3>>& new_verts,
                                                  const std::vector<size_t>& conn) {
  std::vector<std::array<double, 3>> out;
  out.reserve(conn.size());
  for (size_t i : conn) {
    if (i < parent.size()) {
      out.push_back(parent[i]);
    } else {
      out.push_back(new_verts[i - parent.size()]);
    }
  }
  return out;
}

double dist(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  const double dx = a[0] - b[0];
  const double dy = a[1] - b[1];
  const double dz = a[2] - b[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool is_midpoint(const std::array<double, 3>& p,
                 const std::array<double, 3>& a,
                 const std::array<double, 3>& b,
                 double tol = 1e-12) {
  const std::array<double, 3> mid{0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])};
  return dist(p, mid) < tol;
}

} // namespace

class TriangleGreenBlueRefinementTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Unit right triangle in the xy-plane.
    tri_ = {
        {0.0, 0.0, 0.0},  // 0
        {1.0, 0.0, 0.0},  // 1
        {0.0, 1.0, 0.0},  // 2
    };
    area_ = tri_area(tri_);
  }

  std::vector<std::array<double, 3>> tri_;
  double area_ = 0.0;
  TriangleRefinementRule rule_;
};

TEST_F(TriangleGreenBlueRefinementTest, GreenEdge0SplitsCorrectEdgeAndConservesArea) {
  RefinementSpec spec{RefinementPattern::GREEN, 0};
  auto refined = rule_.refine(tri_, CellType::Triangle, spec, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::GREEN);
  ASSERT_EQ(refined.new_vertices.size(), 1u);
  ASSERT_EQ(refined.new_vertex_weights.size(), 1u);
  ASSERT_EQ(refined.child_connectivity.size(), 2u);

  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], tri_[0], tri_[1]));
  EXPECT_EQ(refined.new_vertex_weights[0].size(), 2u);
  EXPECT_EQ(refined.new_vertex_weights[0][0].first, 0u);
  EXPECT_DOUBLE_EQ(refined.new_vertex_weights[0][0].second, 0.5);
  EXPECT_EQ(refined.new_vertex_weights[0][1].first, 1u);
  EXPECT_DOUBLE_EQ(refined.new_vertex_weights[0][1].second, 0.5);

  // Expected connectivity for edge 0 split: v0=0, v1=1, v2=2, midpoint=3.
  EXPECT_EQ(refined.child_connectivity[0], (std::vector<size_t>{0, 3, 2}));
  EXPECT_EQ(refined.child_connectivity[1], (std::vector<size_t>{3, 1, 2}));

  double sum = 0.0;
  for (const auto& conn : refined.child_connectivity) {
    auto cv = child_vertices(tri_, refined.new_vertices, conn);
    const double a = tri_area(cv);
    EXPECT_GT(a, 1e-14);
    sum += a;
  }
  EXPECT_NEAR(sum, area_, 1e-12);
}

TEST_F(TriangleGreenBlueRefinementTest, GreenEdge1SplitsCorrectEdgeAndConservesArea) {
  RefinementSpec spec{RefinementPattern::GREEN, 1};
  auto refined = rule_.refine(tri_, CellType::Triangle, spec, 0);

  ASSERT_EQ(refined.new_vertices.size(), 1u);
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], tri_[1], tri_[2]));
  EXPECT_EQ(refined.new_vertex_weights[0].size(), 2u);
  EXPECT_EQ(refined.new_vertex_weights[0][0].first, 1u);
  EXPECT_EQ(refined.new_vertex_weights[0][1].first, 2u);

  // Expected connectivity for edge 1 split: v0=1, v1=2, v2=0, midpoint=3.
  EXPECT_EQ(refined.child_connectivity[0], (std::vector<size_t>{1, 3, 0}));
  EXPECT_EQ(refined.child_connectivity[1], (std::vector<size_t>{3, 2, 0}));

  double sum = 0.0;
  for (const auto& conn : refined.child_connectivity) {
    sum += tri_area(child_vertices(tri_, refined.new_vertices, conn));
  }
  EXPECT_NEAR(sum, area_, 1e-12);
}

TEST_F(TriangleGreenBlueRefinementTest, GreenEdge2SplitsCorrectEdgeAndConservesArea) {
  RefinementSpec spec{RefinementPattern::GREEN, 2};
  auto refined = rule_.refine(tri_, CellType::Triangle, spec, 0);

  ASSERT_EQ(refined.new_vertices.size(), 1u);
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], tri_[2], tri_[0]));
  EXPECT_EQ(refined.new_vertex_weights[0].size(), 2u);
  EXPECT_EQ(refined.new_vertex_weights[0][0].first, 2u);
  EXPECT_EQ(refined.new_vertex_weights[0][1].first, 0u);

  // Expected connectivity for edge 2 split: v0=2, v1=0, v2=1, midpoint=3.
  EXPECT_EQ(refined.child_connectivity[0], (std::vector<size_t>{2, 3, 1}));
  EXPECT_EQ(refined.child_connectivity[1], (std::vector<size_t>{3, 0, 1}));

  double sum = 0.0;
  for (const auto& conn : refined.child_connectivity) {
    sum += tri_area(child_vertices(tri_, refined.new_vertices, conn));
  }
  EXPECT_NEAR(sum, area_, 1e-12);
}

TEST_F(TriangleGreenBlueRefinementTest, BlueSharedVertex0CreatesTwoMidpointsAndThreeChildren) {
  RefinementSpec spec{RefinementPattern::BLUE, 0};
  auto refined = rule_.refine(tri_, CellType::Triangle, spec, 0);

  EXPECT_EQ(refined.pattern, RefinementPattern::BLUE);
  ASSERT_EQ(refined.new_vertices.size(), 2u);
  ASSERT_EQ(refined.new_vertex_weights.size(), 2u);
  ASSERT_EQ(refined.child_connectivity.size(), 3u);

  // shared vertex s=0 => split edges (0,1) and (0,2).
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], tri_[0], tri_[1]));
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], tri_[0], tri_[2]));

  EXPECT_EQ(refined.child_connectivity[0], (std::vector<size_t>{0, 3, 4}));
  EXPECT_EQ(refined.child_connectivity[1], (std::vector<size_t>{3, 1, 2}));
  EXPECT_EQ(refined.child_connectivity[2], (std::vector<size_t>{4, 3, 2}));

  double sum = 0.0;
  for (const auto& conn : refined.child_connectivity) {
    const double a = tri_area(child_vertices(tri_, refined.new_vertices, conn));
    EXPECT_GT(a, 1e-14);
    sum += a;
  }
  EXPECT_NEAR(sum, area_, 1e-12);
}

TEST_F(TriangleGreenBlueRefinementTest, BlueSharedVertex1CreatesCorrectMidpointsAndConservesArea) {
  RefinementSpec spec{RefinementPattern::BLUE, 1};
  auto refined = rule_.refine(tri_, CellType::Triangle, spec, 0);

  ASSERT_EQ(refined.new_vertices.size(), 2u);
  // shared vertex s=1 => split edges (1,2) and (1,0).
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], tri_[1], tri_[2]));
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], tri_[1], tri_[0]));

  double sum = 0.0;
  for (const auto& conn : refined.child_connectivity) {
    sum += tri_area(child_vertices(tri_, refined.new_vertices, conn));
  }
  EXPECT_NEAR(sum, area_, 1e-12);
}

TEST_F(TriangleGreenBlueRefinementTest, BlueSharedVertex2CreatesCorrectMidpointsAndConservesArea) {
  RefinementSpec spec{RefinementPattern::BLUE, 2};
  auto refined = rule_.refine(tri_, CellType::Triangle, spec, 0);

  ASSERT_EQ(refined.new_vertices.size(), 2u);
  // shared vertex s=2 => split edges (2,0) and (2,1).
  EXPECT_TRUE(is_midpoint(refined.new_vertices[0], tri_[2], tri_[0]));
  EXPECT_TRUE(is_midpoint(refined.new_vertices[1], tri_[2], tri_[1]));

  double sum = 0.0;
  for (const auto& conn : refined.child_connectivity) {
    sum += tri_area(child_vertices(tri_, refined.new_vertices, conn));
  }
  EXPECT_NEAR(sum, area_, 1e-12);
}

TEST_F(TriangleGreenBlueRefinementTest, NumChildrenReportsBlueAsThree) {
  EXPECT_EQ(rule_.num_children(CellType::Triangle, RefinementPattern::BLUE), 3u);
}

} // namespace test
} // namespace svmp

